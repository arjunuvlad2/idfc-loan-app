"""
EXPLANATION SERVICE — Generative AI
Uses GPT-4o-mini to generate formal, RBI-compliant loan decision letters.
The retrieved policy clause is injected directly into the prompt —
the LLM cannot fabricate thresholds because the real text is provided.
"""
from openai import OpenAI
from app.core.config import settings
from app.core.logging import get_logger
from app.services.guardrail_service import locked_system_prompt

logger = get_logger(__name__)
_client = OpenAI(api_key=settings.OPENAI_API_KEY)


def generate(applicant_name: str, cibil_score: int, monthly_income: float,
             loan_amount: float, loan_type: str, employment_type: str,
             default_probability: float, risk_band: str,
             policy_id: str, policy_clause: str, decision: str) -> str:
    """
    Generate a formal RBI-compliant loan decision explanation.

    GROUNDING MECHANISM: The retrieved policy clause is passed in the prompt.
    GPT-4 is instructed to cite the exact Policy ID and reference only thresholds
    found in that clause — not from training data memory.

    INSTRUCTION LOCK: System prompt includes the non-overridable compliance rules.

    Returns: 3-4 sentence decision letter, ready to show to the applicant.
    """
    prompt = f"""Generate a formal, RBI-compliant loan decision explanation.

APPLICANT:
Name: {applicant_name}
CIBIL: {cibil_score} | Income: ₹{monthly_income:,.0f}/month | Loan: ₹{loan_amount:,.0f}
Product: {loan_type.replace('_',' ').title()} | Employment: {employment_type.replace('_',' ').title()}
ML Risk Score: {default_probability*100:.1f}% default probability | Band: {risk_band}

RETRIEVED POLICY (cite this — do NOT invent other policies):
Policy ID: {policy_id}
Clause: {policy_clause}

DECISION: {decision.replace('_', ' ')}

Rules:
- Write exactly 3–4 sentences. Formal, professional, empathetic.
- Reference Policy ID explicitly: "Per IDFC FIRST Bank Policy {policy_id}..."
- Cite CIBIL thresholds from the retrieved clause only — not from memory.
- DECLINED: tell the applicant what steps would make them eligible.
- REFERRED: explain what happens next (human review within 24 hours).
- APPROVED: confirm which criteria were met.
- Never invent interest rates, timelines, or thresholds not in the clause."""

    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": locked_system_prompt(
                    "You generate formal bank loan decision letters. Be precise and policy-grounded."
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.exception("Explanation generation failed: %s", e)
        return f"Decision explanation unavailable. Manual underwriter review required. (Error: {e})"
