from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_title: str = "API Diabetes"
    api_version: str = "1.1.0"
    model_path: Path = Field(default=BASE_DIR / "model" / "model.pkl")
    metadata_path: Path = Field(default=BASE_DIR / "model" / "model_metadata.json")
    metrics_path: Path = Field(default=BASE_DIR / "model" / "metrics.csv")
    database_path: Path = Field(default=BASE_DIR / "data" / "predictions.db")
    cors_origins: str = "*"
    history_default_limit: int = 20
    history_max_limit: int = 100

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
