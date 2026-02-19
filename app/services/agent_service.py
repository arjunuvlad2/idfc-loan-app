"""
AGENT SERVICE — GPT-4 Orchestrator
Routes queries to the correct tool using GPT-4 to understand intent.
This is the Agent Pattern: AI decides which capability to invoke
instead of hardcoded if/else routing.
"""
import json
from typing import Any, Dict, Optional

from openai import OpenAI

from app.core.config import settings
from app.core.logging import get_logger
from app.services.guardrail_service import locked_system_prompt

logger = get_logger(__name__)
_client = OpenAI(api_key=settings.OPENAI_API_KEY)


def route(query: str, data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Route a query to the appropriate tool using GPT-4.
    Routes: rag (policy questions), ml (numeric risk), fraud_review (anomaly),
    human_review (ambiguous), full_pipeline (complete application).
    Try: "What CIBIL do I need?" → rag. "CIBIL 750 income 80K?" → ml.
    """
    prompt = f"""Route this query to one of: rag, ml, fraud_review, human_review, full_pipeline.
Query: {query}
Data: {json.dumps(data or {})}
Respond ONLY in valid JSON: {{"route":"<route>","reason":"<one sentence>","confidence":<0.0-1.0>}}"""

    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": locked_system_prompt("You are a routing agent. Output only JSON.")},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        return json.loads(raw)
    except Exception as e:
        return {"route": "human_review", "reason": f"Router error: {e}", "confidence": 0.0}


def determine_decision(risk_band: str, default_probability: float,
                       has_critical_flag: bool) -> str:
    """
    Map ML risk band + anomaly flags to a final credit decision.
    CRITICAL flag → always refer to human regardless of score.
    Low → APPROVED, Medium/High → REFERRED, Decline → DECLINED.
    """
    if has_critical_flag:
        return "REFERRED_TO_UNDERWRITER"
    return {
        "Low":     "APPROVED",
        "Medium":  "REFERRED_TO_UNDERWRITER",
        "High":    "REFERRED_TO_UNDERWRITER",
        "Decline": "DECLINED",
    }.get(risk_band, "REFERRED_TO_UNDERWRITER")
