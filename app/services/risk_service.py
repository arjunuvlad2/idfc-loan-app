"""
RISK SERVICE — Supervised Machine Learning
Random Forest Classifier trained on 2,000 synthetic historical loan records.
Predicts default probability (0–1) and maps it to a risk band for each application.
"""
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
MODEL_PATH = Path(settings.LOAN_MODEL_PATH)

# Features used by the model
NUMERIC_FEATURES     = ["cibil_score", "monthly_income", "loan_amount", "foir"]
CATEGORICAL_FEATURES = ["loan_type", "employment_type"]


def _get_risk_band(prob: float) -> str:
    """Map default probability to a risk band label."""
    if prob < 0.10: return "Low"
    if prob < 0.25: return "Medium"
    if prob < 0.45: return "High"
    return "Decline"


def _build_pipeline() -> Pipeline:
    """
    Build the scikit-learn ML pipeline.
    Pipeline = ColumnTransformer (preprocessing) → RandomForestClassifier.
    Ensures preprocessing is fit only on training data (no data leakage).
    """
    preprocessor = ColumnTransformer([
        ("numeric",      StandardScaler(),                       NUMERIC_FEATURES),
        ("categorical",  OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            class_weight="balanced",  # handles class imbalance (fewer defaults)
        )),
    ])


def train() -> Dict[str, Any]:
    """
    Train the Random Forest on synthetic data. 80/20 stratified train/test split.
    Saves trained pipeline as .pkl file — loaded on subsequent score() calls.
    Returns: dict with accuracy, AUC-ROC, and sample counts.
    """
    from app.services.data_service import generate_training_data
    df = generate_training_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["default_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = _build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metrics = {
        "accuracy":      round(float(report["accuracy"]), 4),
        "auc_roc":       round(float(roc_auc_score(y_test, y_prob)), 4),
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
        "model_path":    str(MODEL_PATH),
    }
    logger.info("Risk model trained | AUC=%.3f | Accuracy=%.3f", metrics["auc_roc"], metrics["accuracy"])
    return metrics


def _load_model() -> Pipeline:
    """Load trained model from disk. Auto-trains if .pkl file is missing."""
    if not MODEL_PATH.exists():
        logger.warning("Model not found — auto-training...")
        train()
    return joblib.load(MODEL_PATH)


def score(cibil_score: int, monthly_income: float, loan_amount: float,
          loan_type: str, employment_type: str, foir: float = 0.35) -> Dict[str, Any]:
    """
    Predict default probability for a new loan applicant.
    Runs through StandardScaler + OneHotEncoder + Random Forest pipeline.
    Returns default_probability (0-1), risk_band (Low/Medium/High/Decline),
    and a human-readable interpretation string.
    """
    model = _load_model()

    input_df = pd.DataFrame([{
        "cibil_score": cibil_score, "monthly_income": monthly_income,
        "loan_amount": loan_amount, "foir": foir,
        "loan_type": loan_type, "employment_type": employment_type,
    }])

    prob = float(model.predict_proba(input_df)[0][1])
    band = _get_risk_band(prob)

    colors = {"Low":"#0A7F50","Medium":"#D97706","High":"#EA580C","Decline":"#B91C1C"}
    interp = {
        "Low":     "Strong credit profile. Meets all standard lending criteria.",
        "Medium":  "Borderline profile. Refer to senior underwriter for manual review.",
        "High":    "Elevated default risk. Decline or require significant collateral.",
        "Decline": "High probability of default. Application should be declined.",
    }

    return {
        "default_probability": round(prob, 3),
        "risk_band":           band,
        "risk_color":          colors[band],
        "risk_score":          round(prob, 3),
        "interpretation":      interp[band],
    }
