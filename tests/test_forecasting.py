import pytest

from app.config import Settings
from app.forecasting import _estimate, generate_forecast


def test_forecast_does_not_invent_data_in_mock_mode(tmp_path):
    result = generate_forecast(
        "request-1",
        "Forecast TCS revenue and margin from verified sources",
        3,
        Settings(
            _env_file=None,
            mock_mode=True,
            data_dir=tmp_path,
            database_url="sqlite:///:memory:",
        ),
    )
    assert result["generation_mode"] == "mock"
    assert result["historical_metrics"] == []
    assert result["estimate"]["revenue_next_quarter_inr_crore"] is None
    assert "Not estimated" in result["forecast"]["revenue_outlook"]
    assert any("MOCK_MODE=true" in item for item in result["limitations"])
    assert any("No verified financial report" in item for item in result["limitations"])


def test_trend_projection_uses_only_confirmed_observations():
    actuals = []
    for period, metric, value in [
        ("Q1 FY25", "revenue", 100.0),
        ("Q2 FY25", "revenue", 110.0),
        ("Q3 FY25", "revenue", 121.0),
        ("Q1 FY25", "net_profit", 10.0),
        ("Q2 FY25", "net_profit", 11.0),
        ("Q3 FY25", "net_profit", 12.1),
        ("Q1 FY25", "operating_margin", 20.0),
    ]:
        actuals.append(
            {
                "period": period,
                "metric": metric,
                "value": value,
                "review_required": False,
            }
        )
    actuals.append(
        {
            "period": "Q4 FY25",
            "metric": "revenue",
            "value": 999.0,
            "review_required": True,
        }
    )

    estimate = _estimate(actuals)
    assert estimate.revenue_next_quarter_inr_crore == 133.1
    assert estimate.net_profit_next_quarter_inr_crore == 13.31
    assert estimate.operating_margin_next_quarter_pct == 20.0


def test_real_mode_requires_api_key(tmp_path):
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        generate_forecast(
            "request-2",
            "Forecast using actual results and transcripts",
            3,
            Settings(
                _env_file=None,
                mock_mode=False,
                openai_api_key="",
                data_dir=tmp_path,
                database_url="sqlite:///:memory:",
            ),
        )


def test_real_mode_builds_langchain_agent_and_returns_validated_response(
    tmp_path, monkeypatch
):
    import langchain.agents
    import langchain_openai
    from app.schemas import ForecastResponse

    invoked = {}

    class FakeAgent:
        def invoke(self, payload):
            invoked["payload"] = payload
            response = generate_forecast(
                "nested",
                "Forecast from source docs",
                3,
                Settings(
                    _env_file=None,
                    mock_mode=True,
                    data_dir=tmp_path,
                    database_url="sqlite:///:memory:",
                ),
            )
            response["generation_mode"] = "real_ai"
            response["request_id"] = "model-request"
            return {"structured_response": ForecastResponse.model_validate(response)}

    def fake_create_agent(**kwargs):
        invoked["agent_options"] = kwargs
        return FakeAgent()

    monkeypatch.setattr(langchain.agents, "create_agent", fake_create_agent)
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: kwargs)

    result = generate_forecast(
        "request-3",
        "Forecast TCS with supplied sources",
        3,
        Settings(
            _env_file=None,
            mock_mode=False,
            openai_api_key="test-key",
            data_dir=tmp_path,
            database_url="sqlite:///:memory:",
        ),
    )
    names = {tool.name for tool in invoked["agent_options"]["tools"]}
    assert names == {
        "FinancialDataExtractorTool",
        "QualitativeAnalysisTool",
        "financial_growth_calculator",
    }
    assert invoked["payload"]["messages"]
    assert result["generation_mode"] == "real_ai"
    assert result["request_id"] == "request-3"


def test_real_mode_retries_one_transient_llm_failure(tmp_path, monkeypatch):
    import langchain.agents
    import langchain_openai
    from app.schemas import ForecastResponse

    calls = {"count": 0}

    class RateLimitError(Exception):
        pass

    class FakeAgent:
        def invoke(self, _payload):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RateLimitError("temporary")
            response = generate_forecast(
                "nested",
                "Forecast from source docs",
                3,
                Settings(
                    _env_file=None,
                    mock_mode=True,
                    data_dir=tmp_path,
                    database_url="sqlite:///:memory:",
                ),
            )
            response["generation_mode"] = "real_ai"
            response["request_id"] = "model-request"
            return {"structured_response": ForecastResponse.model_validate(response)}

    monkeypatch.setattr(langchain.agents, "create_agent", lambda **_: FakeAgent())
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: kwargs)

    result = generate_forecast(
        "request-retry",
        "Forecast TCS using verified sources and outlook",
        3,
        Settings(
            _env_file=None,
            mock_mode=False,
            openai_api_key="test-key",
            data_dir=tmp_path,
            database_url="sqlite:///:memory:",
        ),
    )
    assert calls["count"] == 2
    assert result["request_id"] == "request-retry"


def test_real_mode_bounds_retries_for_invalid_structured_output(tmp_path, monkeypatch):
    import langchain.agents
    import langchain_openai

    calls = {"count": 0}

    class FakeAgent:
        def invoke(self, _payload):
            calls["count"] += 1
            return {"structured_response": {"request_id": "incomplete"}}

    monkeypatch.setattr(langchain.agents, "create_agent", lambda **_: FakeAgent())
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: kwargs)

    with pytest.raises(RuntimeError, match="invalid structured forecast after one retry"):
        generate_forecast(
            "request-invalid",
            "Forecast TCS using verified sources and outlook",
            3,
            Settings(
                _env_file=None,
                mock_mode=False,
                openai_api_key="test-key",
                data_dir=tmp_path,
                database_url="sqlite:///:memory:",
            ),
        )
    assert calls["count"] == 2
