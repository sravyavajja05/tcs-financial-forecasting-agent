import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.forecasting import generate_forecast
from app.persistence import ForecastDatabase
from app.schemas import ForecastRequest, ForecastResponse

logger = logging.getLogger(__name__)


def create_app(
    database: ForecastDatabase | None = None,
    forecast_runner=generate_forecast,
) -> FastAPI:
    db = database or ForecastDatabase(settings.mysql_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            db.initialize()
            app.state.database_ready = True
            app.state.database_error = None
        except SQLAlchemyError as exc:
            logger.exception("Database initialization failed")
            app.state.database_ready = False
            app.state.database_error = str(exc)
        yield

    application = FastAPI(
        title="TCS Financial Forecasting Agent",
        version="2.0.0",
        description="Source-grounded financial extraction, transcript RAG, and forecasting.",
        lifespan=lifespan,
    )
    application.state.database_ready = False
    application.state.database_error = None

    @application.get("/health")
    def health() -> dict:
        return {
            "status": "ok" if application.state.database_ready else "degraded",
            "database_ready": application.state.database_ready,
            "database_error": (
                "Database initialization failed; inspect server logs."
                if application.state.database_error
                else None
            ),
            "mock_mode": settings.mock_mode,
        }

    @application.post("/forecast", response_model=ForecastResponse)
    def forecast(request: ForecastRequest) -> ForecastResponse:
        if not application.state.database_ready:
            raise HTTPException(
                status_code=503,
                detail="Persistence database is unavailable; forecast was not run.",
            )
        request_id = str(uuid4())
        try:
            db.create_request(request_id, request.query)
        except SQLAlchemyError as exc:
            logger.exception("Could not persist incoming request %s", request_id)
            raise HTTPException(status_code=503, detail="Could not persist forecast request.") from exc
        try:
            response = forecast_runner(
                request_id,
                request.query,
                request.quarters,
            )
            validated = ForecastResponse.model_validate(response)
            db.save_response(request_id, validated.model_dump(mode="json"))
            return validated
        except Exception as exc:
            logger.exception("Forecast generation failed for request %s", request_id)
            try:
                db.fail_request(
                    request_id,
                    "Forecast generation failed; inspect server logs for details.",
                )
            except SQLAlchemyError:
                logger.exception("Failed to persist forecast error for %s", request_id)
            raise HTTPException(
                status_code=500,
                detail=f"Forecast generation failed. Request ID: {request_id}",
            ) from exc

    @application.get("/logs/{request_id}")
    def logs(request_id: str) -> dict:
        try:
            result = db.get_request(request_id)
        except SQLAlchemyError as exc:
            logger.exception("Could not read request log %s", request_id)
            raise HTTPException(status_code=503, detail="Persistence database is unavailable.") from exc
        if result is None:
            raise HTTPException(status_code=404, detail="Request ID not found.")
        return result

    return application


app = create_app()
