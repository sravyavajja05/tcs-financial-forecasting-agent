import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, create_engine, select
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class ForecastRequestRow(Base):
    __tablename__ = "forecast_requests"

    request_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)


class ForecastResponseRow(Base):
    __tablename__ = "forecast_responses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("forecast_requests.request_id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    response_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ForecastDatabase:
    def __init__(self, url: str, engine: Engine | None = None):
        if engine is None:
            parsed_url = make_url(url)
            options = {"pool_pre_ping": True, "future": True}
            if parsed_url.drivername.startswith("sqlite"):
                options["connect_args"] = {"check_same_thread": False}
                database = parsed_url.database
                if database and database != ":memory:":
                    Path(database).parent.mkdir(parents=True, exist_ok=True)
                if database in {None, "", ":memory:"}:
                    options["poolclass"] = StaticPool
            engine = create_engine(url, **options)
        self.engine = engine
        self.sessions = sessionmaker(bind=self.engine, expire_on_commit=False)

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)

    def create_request(self, request_id: str, query: str) -> None:
        with self.sessions.begin() as session:
            session.add(
                ForecastRequestRow(
                    request_id=request_id,
                    query_text=query,
                    status="processing",
                    created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                )
            )

    def save_response(self, request_id: str, payload: dict) -> None:
        with self.sessions.begin() as session:
            row = session.get(ForecastRequestRow, request_id)
            if row is None:
                raise ValueError(f"Forecast request {request_id} was not persisted.")
            session.add(
                ForecastResponseRow(
                    request_id=request_id,
                    response_json=json.loads(json.dumps(payload)),
                    created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                )
            )
            row.status = "completed"
            row.error_message = None

    def fail_request(self, request_id: str, message: str) -> None:
        try:
            with self.sessions.begin() as session:
                row = session.get(ForecastRequestRow, request_id)
                if row is not None:
                    row.status = "failed"
                    row.error_message = message[:4000]
        except SQLAlchemyError:
            logger.exception("Could not record failed forecast %s", request_id)
            raise

    def get_request(self, request_id: str) -> dict | None:
        with self.sessions() as session:
            row = session.get(ForecastRequestRow, request_id)
            if row is None:
                return None
            response = session.scalar(
                select(ForecastResponseRow)
                .where(ForecastResponseRow.request_id == request_id)
            )
            return {
                "request_id": row.request_id,
                "query_text": row.query_text,
                "status": row.status,
                "created_at": row.created_at.isoformat(),
                "error_message": row.error_message,
                "response": response.response_json if response else None,
            }
