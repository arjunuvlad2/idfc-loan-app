"""
EMBEDDING SERVICE
Wraps OpenAI text-embedding-3-small to convert text into 1536-dimensional vectors.
Used by policy_rag_service to embed policy clauses (at seed time) and queries (at retrieval time).
"""
from openai import OpenAI
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
_client = OpenAI(api_key=settings.OPENAI_API_KEY)


def embed_text(text: str) -> list:
    """
    Embed a single string into a 1536-d vector.
    Used during RAG retrieval to embed the user query before searching Weaviate.
    Falls back to a single space if text is empty (OpenAI requires non-empty input).
    """
    text = text.strip() if text and text.strip() else " "
    response = _client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def embed_texts(texts: list) -> list:
    """
    Batch embed multiple strings in a single OpenAI API call.
    Used at seed time to embed all 20 policy clauses efficiently.
    One API call instead of 20 = ~20x cheaper and faster.
    """
    if not texts:
        return []
    cleaned = [t.strip() if t and t.strip() else " " for t in texts]
    response = _client.embeddings.create(model="text-embedding-3-small", input=cleaned)
    return [item.embedding for item in response.data]
