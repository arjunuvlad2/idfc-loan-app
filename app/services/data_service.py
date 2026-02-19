"""
DATA SERVICE
Generates 2,000 synthetic historical loan records for ML training.
Reflects realistic Indian banking patterns: COVID NPA spike 2019-2021,
regional distribution across 7 cities, income ranges by employment type.
No real customer data used — fully synthetic, reproducible with seed=42.
"""
from typing import Any, Dict
import numpy as np
import pandas as pd
from app.core.logging import get_logger

logger = get_logger(__name__)
_cached_df = None  # In-memory cache — generate once, reuse on every call


def generate_training_data(n: int = 2000) -> pd.DataFrame:
    """
    Generate n synthetic loan application records with realistic default behavior.
    Default probability is derived from: CIBIL score, FOIR, loan/employment mismatch,
    and a COVID-19 uplift for years 2019-2021. These are the patterns the
    Random Forest model learns to predict from labeled historical examples.
    Returns: DataFrame with features + default_flag (0=repaid, 1=defaulted).
    """
    global _cached_df
    if _cached_df is not None and len(_cached_df) >= n:
        return _cached_df

    np.random.seed(42)

    loan_types = np.random.choice(["home_loan","personal_loan","business_loan","auto_loan"], n, p=[0.35,0.30,0.20,0.15])
    emp_types  = np.random.choice(["salaried","self_employed","business_owner","pensioner"], n, p=[0.50,0.25,0.15,0.10])
    regions    = np.random.choice(["Mumbai","Delhi","Bangalore","Chennai","Hyderabad","Pune","Kolkata"], n)
    dates      = pd.date_range("2015-01-01", "2024-12-31", periods=n)

    records = []
    for i in range(n):
        et, lt = emp_types[i], loan_types[i]

        # Income range by employment type
        if et == "salaried":     income = int(np.random.randint(25_000, 250_000))
        elif et == "pensioner":  income = int(np.random.randint(15_000, 80_000))
        else:                    income = int(np.random.randint(30_000, 400_000))

        cibil  = int(np.clip(np.random.normal(680, 80), 300, 900))
        mult   = {"home_loan":60,"personal_loan":18,"business_loan":36,"auto_loan":24}[lt]
        amount = round(income * np.random.uniform(3, mult), -3)
        foir   = round(np.random.uniform(0.10, 0.70), 2)

        # Realistic default probability
        p = 0.05
        if cibil < 600: p += 0.30
        elif cibil < 650: p += 0.15
        elif cibil < 700: p += 0.07
        if foir > 0.55:  p += 0.12
        if lt == "business_loan" and et != "business_owner": p += 0.08
        if et == "self_employed": p += 0.04
        yr = dates[i].year
        if 2019 <= yr <= 2021: p += 0.06   # COVID-19 NPA spike
        elif yr >= 2022:       p -= 0.02   # Post-COVID recovery

        records.append({
            "cibil_score": cibil, "monthly_income": income, "loan_amount": int(amount),
            "loan_type": lt, "employment_type": et, "region": regions[i],
            "foir": foir, "default_flag": int(np.random.random() < min(p, 0.70)),
            "year": int(dates[i].year),
        })

    _cached_df = pd.DataFrame(records)
    logger.info("Generated %d synthetic training records.", len(_cached_df))
    return _cached_df


def get_portfolio_summary() -> Dict[str, Any]:
    """
    Return aggregate portfolio statistics for the Portfolio Intelligence tab.
    Used by GET /portfolio to populate charts and summary cards.
    """
    df = generate_training_data()
    by_year = (df.groupby("year")
               .agg(total=("cibil_score","count"), defaults=("default_flag","sum"))
               .reset_index())
    by_year["npa_rate"] = (by_year["defaults"] / by_year["total"] * 100).round(2)

    return {
        "total_applications": int(len(df)),
        "total_defaults":     int(df["default_flag"].sum()),
        "overall_npa_rate":   round(df["default_flag"].mean() * 100, 2),
        "avg_cibil":          round(df["cibil_score"].mean(), 1),
        "by_year": by_year.to_dict(orient="records"),
        "by_loan_type": (df.groupby("loan_type")["default_flag"]
                         .agg(["count","mean"])
                         .reset_index()
                         .rename(columns={"count":"total","mean":"default_rate"})
                         .to_dict(orient="records")),
    }
