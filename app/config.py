from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: SecretStr = SecretStr("")
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="OPENAI_EMBEDDING_MODEL",
    )
    mock_mode: bool = True
    data_dir: Path = Path("data")
    mysql_url: str = Field(
        default=(
            "mysql+pymysql://forecast_user:change_me@localhost:3306/"
            "tcs_forecasting_v2?charset=utf8mb4"
        ),
        validation_alias=AliasChoices(
            "MYSQL_URL", "DATABASE_URL", "mysql_url", "database_url"
        ),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    @property
    def database_url(self) -> str:
        """Backward-compatible attribute alias for the resolved MYSQL_URL."""
        return self.mysql_url


settings = Settings()
