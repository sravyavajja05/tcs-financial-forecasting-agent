from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForecastRequest(StrictModel):
    query: str = Field(min_length=10, max_length=4000)
    quarters: int = Field(default=3, ge=2, le=4)


class MetricEvidence(StrictModel):
    period: str
    metric: Literal["revenue", "net_profit", "operating_margin"]
    value: float | None
    unit: str | None = None
    currency: str | None = None
    basis: str | None = None
    source_file: str
    source_url: str | None = None
    page: int | None = None
    excerpt: str
    review_required: bool = False


class SourceEvidence(StrictModel):
    category: Literal[
        "management_outlook", "recurring_theme", "risk", "opportunity", "financial"
    ]
    source_file: str
    source_url: str | None = None
    period: str | None = None
    page: int | None = None
    excerpt: str


class ForecastEstimate(StrictModel):
    revenue_next_quarter_inr_crore: float | None = None
    net_profit_next_quarter_inr_crore: float | None = None
    operating_margin_next_quarter_pct: float | None = None
    method: str


class ForecastNarrative(StrictModel):
    revenue_outlook: str
    net_profit_outlook: str
    operating_margin_outlook: str
    management_outlook: list[str] = Field(default_factory=list)
    recurring_themes: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    confidence: Literal["Low", "Medium", "High"] = "Low"
    assumptions: list[str] = Field(default_factory=list)


class ForecastResponse(StrictModel):
    request_id: str
    company: Literal["Tata Consultancy Services"] = "Tata Consultancy Services"
    forecast_period: str
    generation_mode: Literal["mock", "real_ai"]
    historical_metrics: list[MetricEvidence] = Field(default_factory=list)
    estimate: ForecastEstimate
    forecast: ForecastNarrative
    source_evidence: list[SourceEvidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "This is an analytical estimate, not investment advice or a guarantee "
        "of future performance."
    )
