import os
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.dialects import mysql
from sqlalchemy.exc import OperationalError
from sqlalchemy.schema import CreateTable

from app.api import create_app
from app.config import Settings
from app.forecasting import generate_forecast
from app.persistence import (
    ForecastDatabase,
    ForecastRequestRow,
    ForecastResponseRow,
)


def test_app_uses_resolved_mysql_url(monkeypatch):
    import app.api

    captured = {}

    class DatabaseSpy:
        def __init__(self, url):
            captured["url"] = url

    monkeypatch.setattr(app.api, "ForecastDatabase", DatabaseSpy)
    monkeypatch.setattr(
        app.api.settings,
        "mysql_url",
        "sqlite:///./storage/resolved.sqlite3",
    )

    app.api.create_app()

    assert captured["url"] == "sqlite:///./storage/resolved.sqlite3"


def test_forecast_endpoint_validates_and_persists_request_and_response():
    db = ForecastDatabase("sqlite:///:memory:")

    def mock_runner(request_id, query, quarters):
        return {
            "request_id": request_id,
            "forecast_period": "Upcoming quarter (period not established)",
            "generation_mode": "mock",
            "estimate": {"method": "No source data."},
            "forecast": {
                "revenue_outlook": "Not estimated.",
                "net_profit_outlook": "Not estimated.",
                "operating_margin_outlook": "Not estimated.",
            },
        }

    app = create_app(db, mock_runner)
    with TestClient(app) as client:
        invalid = client.post("/forecast", json={"query": "short"})
        assert invalid.status_code == 422

        response = client.post(
            "/forecast",
            json={"query": "Forecast from actual TCS quarterly reports", "quarters": 3},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["generation_mode"] == "mock"
        assert payload["request_id"]

        log_response = client.get(f"/logs/{payload['request_id']}")
        assert log_response.status_code == 200
        log = log_response.json()
        assert log["status"] == "completed"
        assert log["response"]["request_id"] == payload["request_id"]


def test_forecast_runs_with_initialized_persistence_database():
    db = ForecastDatabase("sqlite:///:memory:")
    app = create_app(db)
    with TestClient(app) as client:
        response = client.post(
            "/forecast",
            json={"query": "Forecast from actual TCS quarterly reports"},
        )
        assert response.status_code == 200


def test_documented_sqlite_storage_path_creates_parent_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = ForecastDatabase("sqlite:///./storage/tcs_forecasting_v2.sqlite3")

    db.initialize()
    db.create_request("sqlite-storage-test", "Verify SQLite storage initialization")

    assert (tmp_path / "storage" / "tcs_forecasting_v2.sqlite3").is_file()
    assert db.get_request("sqlite-storage-test")["status"] == "processing"


def test_forecast_endpoint_returns_source_grounded_json_and_persists_it():
    data_dir = Path(__file__).resolve().parents[1] / "data"

    def mock_runner(request_id, query, quarters):
        return generate_forecast(
            request_id,
            query,
            quarters,
            Settings(
                _env_file=None,
                mock_mode=True,
                data_dir=data_dir,
                database_url="sqlite:///:memory:",
            ),
        )

    app = create_app(ForecastDatabase("sqlite:///:memory:"), mock_runner)
    with TestClient(app) as client:
        response = client.post(
            "/forecast",
            json={
                "query": (
                    "Forecast the next TCS quarter using verified financial "
                    "reports and earnings call transcripts."
                ),
                "quarters": 3,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["generation_mode"] == "mock"
        assert payload["forecast_period"] == "Q3 FY27"
        assert any(
            item["period"] == "Q2 FY27"
            and item["metric"] == "revenue"
            and item["value"] == 73188.0
            for item in payload["historical_metrics"]
        )
        assert payload["source_evidence"]

        log_response = client.get(f"/logs/{payload['request_id']}")
        assert log_response.status_code == 200
        assert log_response.json()["status"] == "completed"
        assert log_response.json()["response"] == payload


def test_forecast_returns_503_when_database_initialization_fails():
    class UnavailableDatabase(ForecastDatabase):
        def initialize(self):
            raise OperationalError("connect", {}, RuntimeError("unavailable"))

    app = create_app(UnavailableDatabase("sqlite:///:memory:"))
    with TestClient(app) as client:
        assert client.get("/health").json()["database_ready"] is False
        response = client.post(
            "/forecast",
            json={"query": "Forecast from actual TCS quarterly reports"},
        )
        assert response.status_code == 503


def test_health_and_unknown_log_id():
    app = create_app(ForecastDatabase("sqlite:///:memory:"))
    with TestClient(app) as client:
        assert client.get("/health").json()["database_ready"] is True
        assert client.get("/logs/missing").status_code == 404


def test_mysql_response_schema_uses_native_json_column():
    statement = str(
        CreateTable(ForecastResponseRow.__table__).compile(dialect=mysql.dialect())
    )
    assert "response_json JSON NOT NULL" in statement


def test_forecast_failure_does_not_expose_exception_in_saved_log():
    db = ForecastDatabase("sqlite:///:memory:")
    app = create_app(
        db,
        lambda *_: (_ for _ in ()).throw(RuntimeError("secret=do-not-return")),
    )
    with TestClient(app) as client:
        response = client.post(
            "/forecast",
            json={"query": "Forecast from actual TCS quarterly reports"},
        )
        assert response.status_code == 500
        assert "do-not-return" not in response.text
        request_id = response.json()["detail"].rsplit(" ", 1)[-1]
        log = client.get(f"/logs/{request_id}").json()
        assert "do-not-return" not in log["error_message"]


MYSQL_TEST_URL = os.getenv("TCS_V2_TEST_DATABASE_URL")


@pytest.mark.skipif(
    not MYSQL_TEST_URL,
    reason="Set TCS_V2_TEST_DATABASE_URL to a disposable MySQL test schema.",
)
def test_mysql_persists_request_and_response_when_configured():
    if not MYSQL_TEST_URL.startswith("mysql"):
        pytest.fail("TCS_V2_TEST_DATABASE_URL must use a MySQL SQLAlchemy URL.")
    engine = create_engine(MYSQL_TEST_URL, pool_pre_ping=True)
    database = ForecastDatabase(MYSQL_TEST_URL, engine=engine)
    request_id = str(uuid4())
    request_created = False
    try:
        with engine.connect() as connection:
            assert connection.exec_driver_sql("SELECT 1").scalar_one() == 1
        database.initialize()
        database.create_request(request_id, "temporary MySQL persistence test")
        request_created = True
        payload = {"request_id": request_id, "generation_mode": "mock"}
        database.save_response(request_id, payload)
        stored = database.get_request(request_id)
        assert stored is not None
        assert stored["status"] == "completed"
        assert stored["response"] == payload
    finally:
        if request_created:
            with engine.begin() as connection:
                connection.execute(
                    delete(ForecastResponseRow).where(
                        ForecastResponseRow.request_id == request_id
                    )
                )
                connection.execute(
                    delete(ForecastRequestRow).where(
                        ForecastRequestRow.request_id == request_id
                    )
                )
        engine.dispose()
