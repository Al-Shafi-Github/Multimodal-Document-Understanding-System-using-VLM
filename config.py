from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # AI
    AI_ENGINE: str = "lmstudio"
    AI_HOST: str = "http://127.0.0.1:1234"
    AI_TIMEOUT: int = 120
    AI_API_KEY: str | None = None
    MODEL_NAME: str = "qwen2.5-vl-3b"

    # Pipeline
    CONFIDENCE_THRESHOLD: float = Field(default=0.30, ge=0.0, le=1.0)


    LOG_LEVEL: str = "INFO"

    # Computed paths
    DATA_DIR: Path = PROJECT_ROOT / "data"
    INCOMING_DIR: Path = DATA_DIR / "incoming"
    RENAMED_DIR: Path = DATA_DIR / "renamed"
    FAILED_DIR: Path = DATA_DIR / "failed"
    DATASET_DIR: Path = DATA_DIR / "golden_dataset"


settings = Settings()