"""
POLICY RAG SERVICE
Retrieval-Augmented Generation using Weaviate vector database.
Stores 20 RBI-compliant loan policy clauses as embeddings.
For each application, semantically retrieves the most relevant clause
and feeds it to GPT-4 — preventing hallucination through grounded generation.

BUG FIX from v1: Original code used list_all() membership check which fails in
weaviate-client v4 because list_all() returns a DICT not a list.
Fixed to use client.collections.exists() — the correct explicit API.
"""
import json
from typing import Any, Dict, List, Optional

import weaviate
from openai import OpenAI
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery

from app.core.config import settings
from app.core.logging import get_logger
from app.services.embedding_service import embed_text, embed_texts

logger = get_logger(__name__)
_openai = OpenAI(api_key=settings.OPENAI_API_KEY)

# 20 RBI-realistic policy clauses — ground truth for all LLM explanations.
# In production, loaded from a live policy management system.
POLICY_CORPUS = [
    {"policy_id":"RBI-HL-001","loan_type":"home_loan","cibil_min":700,"cibil_max":900,"customer_segment":"salaried","policy_year":"2024","clause":"Home loan applicants with CIBIL 700 or above and salaried employment are eligible for up to 80% LTV financing. Monthly EMI must not exceed 40% of net monthly income. Co-applicant may be included to enhance eligibility."},
    {"policy_id":"RBI-HL-002","loan_type":"home_loan","cibil_min":650,"cibil_max":699,"customer_segment":"salaried","policy_year":"2024","clause":"Home loan applicants with CIBIL 650–699 may be considered for up to 70% LTV with additional income verification. A co-applicant with CIBIL above 750 is strongly recommended. FOIR must not exceed 45% of gross monthly income."},
    {"policy_id":"RBI-HL-003","loan_type":"home_loan","cibil_min":300,"cibil_max":649,"customer_segment":"salaried","policy_year":"2024","clause":"Home loan applications with CIBIL below 650 are classified as sub-standard risk. Applications are declined unless the applicant provides additional collateral equal to or exceeding 150% of loan value. Mandatory senior underwriter review required."},
    {"policy_id":"RBI-PL-001","loan_type":"personal_loan","cibil_min":720,"cibil_max":900,"customer_segment":"salaried","policy_year":"2024","clause":"Personal loan applicants with CIBIL above 720 and minimum 2 years stable employment qualify for loans up to 18x monthly net salary. No collateral required. Disbursal within 48 hours subject to e-KYC verification."},
    {"policy_id":"RBI-PL-002","loan_type":"personal_loan","cibil_min":650,"cibil_max":719,"customer_segment":"salaried","policy_year":"2024","clause":"Personal loan applicants with CIBIL 650–719 are eligible for up to 12x monthly salary. Interest rate premium of 1.5–3% over base rate applies. Employment verification with last 6 months salary slips is mandatory."},
    {"policy_id":"RBI-PL-003","loan_type":"personal_loan","cibil_min":300,"cibil_max":649,"customer_segment":"salaried","policy_year":"2024","clause":"Personal loan applications with CIBIL below 650 are declined per RBI Fair Lending Guidelines. Applicants are advised to improve credit score through timely repayment before reapplying. No exceptions without senior credit committee approval."},
    {"policy_id":"RBI-BL-001","loan_type":"business_loan","cibil_min":700,"cibil_max":900,"customer_segment":"business_owner","policy_year":"2024","clause":"Business loan applicants with CIBIL above 700 and minimum 3 years business vintage may qualify for working capital loans up to ₹2 crore. GST returns for 12 months and audited financials for 2 years mandatory. DSCR must exceed 1.25 for approval."},
    {"policy_id":"RBI-BL-002","loan_type":"business_loan","cibil_min":650,"cibil_max":699,"customer_segment":"business_owner","policy_year":"2024","clause":"Business loan applicants with CIBIL 650–699 require enhanced due diligence. 24 months bank statements and minimum 5 years business vintage required. DSCR must exceed 1.40. Senior credit manager sign-off mandatory."},
    {"policy_id":"RBI-BL-003","loan_type":"business_loan","cibil_min":300,"cibil_max":649,"customer_segment":"business_owner","policy_year":"2024","clause":"Business loan applications with CIBIL below 650 are declined. RBI Master Directions 2023 require maintaining credit risk standards for MSME lending. Applicants referred to SIDBI or CGTMSE guarantee-backed products as alternatives."},
    {"policy_id":"RBI-AL-001","loan_type":"auto_loan","cibil_min":680,"cibil_max":900,"customer_segment":"salaried","policy_year":"2024","clause":"Auto loan applicants with CIBIL above 680 are eligible for up to 85% on-road price financing. Loan tenure up to 7 years. EMI not to exceed 35% of net monthly income. Comprehensive insurance mandatory for loan tenure."},
    {"policy_id":"RBI-AL-002","loan_type":"auto_loan","cibil_min":620,"cibil_max":679,"customer_segment":"salaried","policy_year":"2024","clause":"Auto loan applicants with CIBIL 620–679 are eligible for up to 75% on-road price. Tenure restricted to 5 years. Stable employment for minimum 18 months required. Rate premium of 1.75% over standard rate applies."},
    {"policy_id":"RBI-AL-003","loan_type":"auto_loan","cibil_min":300,"cibil_max":619,"customer_segment":"salaried","policy_year":"2024","clause":"Auto loan applicants with CIBIL below 620 are not eligible for standard auto loan products. Applicants may be offered secured loan against FD deposit as alternative."},
    {"policy_id":"RBI-FOIR-001","loan_type":"personal_loan","cibil_min":700,"cibil_max":900,"customer_segment":"salaried","policy_year":"2024","clause":"Applicants with FOIR exceeding 55% are ineligible for additional unsecured credit regardless of CIBIL score. This aligns with RBI Circular RBI/2023-24/73 on household debt management and consumer credit risk governance."},
    {"policy_id":"RBI-FOIR-002","loan_type":"home_loan","cibil_min":650,"cibil_max":900,"customer_segment":"salaried","policy_year":"2024","clause":"For secured loans including home loans, FOIR limit is 50% of gross monthly income. Applicants exceeding this must provide a co-applicant. Income from part-time or variable sources capped at 50% for FOIR calculation."},
    {"policy_id":"RBI-SE-001","loan_type":"personal_loan","cibil_min":700,"cibil_max":900,"customer_segment":"self_employed","policy_year":"2024","clause":"Self-employed applicants with CIBIL above 700 require ITR for 2 years with minimum net annual income of ₹5 lakh. Business continuity of 3 years minimum required. Professional tax clearance certificate mandatory."},
    {"policy_id":"RBI-SE-002","loan_type":"business_loan","cibil_min":680,"cibil_max":900,"customer_segment":"self_employed","policy_year":"2024","clause":"Self-employed professionals (doctors, CAs, architects) with CIBIL above 680 qualify for professional loans up to ₹75 lakh. Practice vintage of 3 years minimum and MCI/ICAI registration required."},
    {"policy_id":"RBI-FRAUD-001","loan_type":"personal_loan","cibil_min":300,"cibil_max":900,"customer_segment":"salaried","policy_year":"2024","clause":"Applications where declared income deviates more than 30% from ITR-reported income trigger mandatory fraud review under RBI KYC Master Direction 2016. Material misrepresentation leads to permanent blacklisting from IDFC credit facilities."},
    {"policy_id":"RBI-NPA-001","loan_type":"home_loan","cibil_min":300,"cibil_max":900,"customer_segment":"salaried","policy_year":"2024","clause":"Applicants with any active NPA classification on existing credit facilities are ineligible for new credit products. Settlement must be completed minimum 24 months prior to reapplication. NOC from previous lender mandatory."},
    {"policy_id":"RBI-BIAS-001","loan_type":"personal_loan","cibil_min":300,"cibil_max":900,"customer_segment":"salaried","policy_year":"2024","clause":"IDFC Bank Fair Lending Policy prohibits credit decisions based on religion, caste, gender, marital status, or geographic origin. All AI-assisted underwriting decisions must be explainable, auditable, and bias-tested quarterly."},
    {"policy_id":"RBI-HUMAN-001","loan_type":"home_loan","cibil_min":600,"cibil_max":680,"customer_segment":"salaried","policy_year":"2024","clause":"Borderline applications where AI risk score falls between Medium and High bands must be referred to a human underwriter within 24 hours. Human-in-the-loop is mandatory per RBI AI in Financial Services Draft Framework 2024."},
]


def _get_client() -> weaviate.WeaviateClient:
    """
    Create and return a new authenticated Weaviate Cloud client.
    Always close after use via client.close() in a finally block.
    Fresh connection per operation avoids stale connection timeouts.
    """
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=settings.WEAVIATE_URL,
        auth_credentials=weaviate.auth.AuthApiKey(api_key=settings.WEAVIATE_API_KEY),
    )


def _ensure_schema() -> None:
    """
    Create the LoanPolicy collection in Weaviate if it does not yet exist.
    BUG FIX: Uses client.collections.exists() instead of checking list_all() membership.
    In weaviate-client v4, list_all() returns a dict — the 'in' operator checks dict KEYS,
    not collection names. exists() is the correct, explicit API.
    Idempotent — safe to call multiple times.
    """
    client = _get_client()
    try:
        if not client.collections.exists(settings.POLICY_COLLECTION):
            client.collections.create(
                name=settings.POLICY_COLLECTION,
                properties=[
                    Property(name="policyId",        data_type=DataType.TEXT),
                    Property(name="loanType",         data_type=DataType.TEXT),
                    Property(name="cibilMin",         data_type=DataType.INT),
                    Property(name="cibilMax",         data_type=DataType.INT),
                    Property(name="customerSegment",  data_type=DataType.TEXT),
                    Property(name="policyYear",       data_type=DataType.TEXT),
                    Property(name="clause",           data_type=DataType.TEXT),
                ],
                vectorizer_config=Configure.Vectorizer.none(),  # We supply own OpenAI vectors
            )
            logger.info("LoanPolicy collection created in Weaviate.")
    finally:
        client.close()


def seed() -> int:
    """
    Embed all 20 RBI policy clauses and insert them into Weaviate.
    Step 1: Batch embed all clause texts in a single OpenAI API call.
    Step 2: Insert each clause with its pre-computed 1536-d vector.
    Safe to call multiple times — skips if already seeded.
    Returns: int — total clauses now stored in Weaviate.
    """
    _ensure_schema()
    client = _get_client()
    try:
        collection = client.collections.get(settings.POLICY_COLLECTION)

        # Check if already seeded
        existing = collection.aggregate.over_all(total_count=True).total_count
        if existing >= len(POLICY_CORPUS):
            logger.info("Already seeded %d clauses — skipping.", existing)
            return existing

        # Batch embed all clauses in one OpenAI API call
        vectors = embed_texts([p["clause"] for p in POLICY_CORPUS])

        # Insert each clause with its vector
        for policy, vector in zip(POLICY_CORPUS, vectors):
            collection.data.insert(
                properties={
                    "policyId":        policy["policy_id"],
                    "loanType":        policy["loan_type"],
                    "cibilMin":        policy["cibil_min"],
                    "cibilMax":        policy["cibil_max"],
                    "customerSegment": policy["customer_segment"],
                    "policyYear":      policy["policy_year"],
                    "clause":          policy["clause"],
                },
                vector=vector,
            )
        logger.info("Seeded %d RBI policy clauses into Weaviate.", len(POLICY_CORPUS))
        return len(POLICY_CORPUS)
    except Exception as e:
        logger.exception("Seeding failed: %s", e)
        raise
    finally:
        client.close()


def retrieve(query: str, loan_type: Optional[str] = None,
             cibil_score: Optional[int] = None, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Semantically retrieve the most relevant RBI policy clauses.
    Combines vector similarity (cosine distance) with metadata filters.
    Without filters, a home loan query might retrieve a business loan policy —
    metadata filters ensure we only search the relevant policy subset.
    Returns: list of policy dicts with clause text and relevance_score.
    Gracefully returns [] on any Weaviate error so pipeline keeps running.
    """
    _ensure_schema()
    client = _get_client()
    try:
        collection = client.collections.get(settings.POLICY_COLLECTION)

        # Build AND-chained metadata filters
        filters = None
        filter_list = []
        if loan_type:
            filter_list.append(Filter.by_property("loanType").equal(loan_type))
        if cibil_score is not None:
            filter_list.append(Filter.by_property("cibilMin").less_or_equal(cibil_score))
            filter_list.append(Filter.by_property("cibilMax").greater_or_equal(cibil_score))
        if filter_list:
            filters = filter_list[0]
            for f in filter_list[1:]:
                filters = filters & f

        result = collection.query.near_vector(
            near_vector=embed_text(query),
            limit=top_k,
            filters=filters,
            return_properties=["policyId","loanType","cibilMin","cibilMax","customerSegment","policyYear","clause"],
            return_metadata=MetadataQuery(distance=True),
        )

        hits = []
        if result and getattr(result, "objects", None):
            for obj in result.objects:
                p = obj.properties or {}
                dist = getattr(obj.metadata, "distance", None) if obj.metadata else None
                hits.append({
                    "policy_id":        p.get("policyId"),
                    "loan_type":        p.get("loanType"),
                    "clause":           p.get("clause"),
                    "relevance_score":  round(1 - float(dist), 3) if dist is not None else None,
                })
        return hits
    except Exception as e:
        logger.exception("Retrieve failed: %s", e)
        return []
    finally:
        client.close()


def check_faithfulness(explanation: str, policy_clause: str) -> Dict[str, Any]:
    """
    Validate that the AI explanation is grounded in the retrieved policy (no hallucinations).
    Second GPT-4 call acts as a compliance auditor — compares explanation vs source policy.
    Temperature=0 for deterministic, consistent scoring.
    Score 1.0=FAITHFUL, 0.5-0.9=PARTIALLY_FAITHFUL, <0.5=UNFAITHFUL.
    """
    prompt = f"""You are a compliance auditor for an Indian bank AI system.

RETRIEVED POLICY CLAUSE (ground truth):
{policy_clause}

AI-GENERATED EXPLANATION (to verify):
{explanation}

Check: Does the explanation fabricate policy clauses, invent interest rates, or contradict the policy?
Respond ONLY in valid JSON (no markdown, no code fences):
{{"faithfulness_score":<0.0-1.0>,"verdict":"<FAITHFUL|PARTIALLY_FAITHFUL|UNFAITHFUL>","hallucinated_claims":[],"reasoning":"<one sentence>"}}"""

    try:
        resp = _openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning("Faithfulness check failed: %s", e)
        return {"faithfulness_score": 0.5, "verdict": "UNKNOWN", "hallucinated_claims": [], "reasoning": str(e)}
