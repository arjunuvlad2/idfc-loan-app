"""
AUDIT SERVICE
In-memory circular buffer storing last 200 underwriting decisions.
Every field logged supports RBI's requirement for replayable, explainable decisions.
In production: write to PostgreSQL or DynamoDB instead.
"""
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

_audit_log: deque = deque(maxlen=200)  # Circular buffer — oldest auto-discarded
_counter: int = 0                       # Monotonic decision ID counter


def record(applicant_name: str, cibil_score: int, monthly_income: float,
           loan_amount: float, loan_type: str, default_probability: Optional[float],
           risk_band: Optional[str], decision: str, policy_id: Optional[str],
           faithfulness_score: Optional[float], faithfulness_verdict: Optional[str],
           latency_ms: float, guardrail_flagged: bool, guardrail_severity: str,
           requires_human_review: bool, kyc_verified: bool,
           explanation_preview: Optional[str] = None) -> Dict[str, Any]:
    """
    Record a completed underwriting decision to the audit log.
    Captures all pipeline fields required for RBI compliance audit trail.
    """
    global _counter
    _counter += 1
    entry = {
        "id": _counter,
        "timestamp":         datetime.now().isoformat(),
        "timestamp_display": datetime.now().strftime("%H:%M:%S"),
        "applicant_name":    applicant_name,
        "cibil_score":       cibil_score,
        "monthly_income":    monthly_income,
        "loan_amount":       loan_amount,
        "loan_type":         loan_type,
        "default_probability": default_probability,
        "risk_band":         risk_band,
        "decision":          decision,
        "policy_id":         policy_id,
        "faithfulness_score":   faithfulness_score,
        "faithfulness_verdict": faithfulness_verdict,
        "latency_ms":           round(latency_ms, 1),
        "guardrail_flagged":    guardrail_flagged,
        "guardrail_severity":   guardrail_severity,
        "requires_human_review": requires_human_review,
        "kyc_verified":          kyc_verified,
        "explanation_preview": (
            explanation_preview[:120] + "…"
            if explanation_preview and len(explanation_preview) > 120
            else explanation_preview
        ),
    }
    _audit_log.appendleft(entry)
    logger.info("AUDIT #%d | %s | CIBIL:%d | %s | %s | %.0fms",
                _counter, applicant_name, cibil_score, risk_band, decision, latency_ms)
    return entry


def recent(limit: int = 10) -> List[Dict[str, Any]]:
    """Return the most recent N audit entries (most recent first)."""
    return list(_audit_log)[:limit]


def stats() -> Dict[str, Any]:
    """
    Compute aggregate statistics for the monitoring dashboard.
    Called every 5 seconds by the frontend polling.
    """
    all_e = list(_audit_log)
    total = len(all_e)
    if total == 0:
        return {
            "total_processed": 0, "approved_count": 0, "declined_count": 0,
            "human_review_count": 0, "guardrail_blocks": 0, "kyc_verified_count": 0,
            "avg_latency_ms": 0, "avg_faithfulness_score": 0,
            "hallucination_rate_pct": 0, "escalation_rate_pct": 0,
            "guardrail_block_rate_pct": 0,
            "risk_band_distribution": {"Low":0,"Medium":0,"High":0,"Decline":0},
        }

    approved      = sum(1 for e in all_e if e["decision"] == "APPROVED")
    declined      = sum(1 for e in all_e if e["decision"] == "DECLINED")
    human_review  = sum(1 for e in all_e if e["requires_human_review"])
    g_blocks      = sum(1 for e in all_e if e["guardrail_flagged"])
    kyc_verified  = sum(1 for e in all_e if e.get("kyc_verified"))
    latencies     = [e["latency_ms"] for e in all_e if e["latency_ms"]]
    faith_scores  = [e["faithfulness_score"] for e in all_e if e["faithfulness_score"] is not None]
    hallucinations = sum(1 for e in all_e if e.get("faithfulness_verdict") in ("PARTIALLY_FAITHFUL","UNFAITHFUL"))

    band_dist = {"Low":0,"Medium":0,"High":0,"Decline":0}
    for e in all_e:
        if e.get("risk_band") in band_dist:
            band_dist[e["risk_band"]] += 1

    return {
        "total_processed": total, "approved_count": approved, "declined_count": declined,
        "human_review_count": human_review, "guardrail_blocks": g_blocks,
        "kyc_verified_count": kyc_verified,
        "avg_latency_ms":           round(sum(latencies)/len(latencies), 1) if latencies else 0,
        "avg_faithfulness_score":   round(sum(faith_scores)/len(faith_scores), 3) if faith_scores else 0,
        "hallucination_rate_pct":   round(hallucinations/total*100, 1),
        "escalation_rate_pct":      round(human_review/total*100, 1),
        "guardrail_block_rate_pct": round(g_blocks/total*100, 1),
        "risk_band_distribution":   band_dist,
    }
