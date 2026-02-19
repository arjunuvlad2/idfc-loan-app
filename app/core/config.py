"""
CONFIGURATION
Pydantic Settings reads environment variables from .env file.
All required keys will raise a clear validation error if missing.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Required API keys ────────────────────────────────────────────────────
    OPENAI_API_KEY:  str
    WEAVIATE_URL:    str
    WEAVIATE_API_KEY: str

    # ── App identity ─────────────────────────────────────────────────────────
    APP_NAME:    str = "AI  Copilot"
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL:   str = "INFO"

    # ── Model + collection names ──────────────────────────────────────────────
    LOAN_MODEL_PATH:   str = "models/risk_model.pkl"
    POLICY_COLLECTION: str = "Random Policy Docs"

    class Config:
        env_file = ".env"


settings = Settings()
