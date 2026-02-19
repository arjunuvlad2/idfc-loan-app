from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging, get_logger


setup_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: generate synthetic data + auto-train model if no .pkl exists.
    This means zero manual setup — just run uvicorn and it works.
    """
    from app.services.data_service import generate_training_data
    from app.services.risk_service import train as train_risk

    generate_training_data()

    if not Path(settings.LOAN_MODEL_PATH).exists():
        logger.info("Training ML Risk Model (first run)...")
        metrics = train_risk()
        logger.info("Model ready | AUC=%.3f | Accuracy=%.3f",
                    metrics["auc_roc"], metrics["accuracy"])
    else:
        logger.info("ML model found — skipping training.")

    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
)

# Allow all origins for workshop demos (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health():
    """Quick health check. Verify server is up before running demos."""
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "risk_model_ready":    Path(settings.LOAN_MODEL_PATH).exists(),
        "openai_configured":   bool(settings.OPENAI_API_KEY),
        "weaviate_configured": bool(settings.WEAVIATE_API_KEY),
    }



