"""
Query router — directs classified queries to the correct data source.

Routing rules:
    - legal_query         → Qdrant with type=law
    - document_query      → Qdrant with owner_id / session filter
    - general_knowledge   → direct LLM (no retrieval)
    - web (Exa fallback)  → triggered when confidence < 0.50 for legal intents
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Awaitable

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.classifier.engine import QueryClassification
from app.services.retrieval import search_legal_documents, search_user_documents
from app.services.web.confidence import (
    compute_retrieval_confidence,
    should_trigger_exa,
    should_use_web_only_context,
)
from app.services.web.exa_client import search_exa
from app.services.web.cache import cache_get, cache_set
from app.services.web.policy import get_exa_policy
from app.services.llm.client import get_chat_provider_label
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
LLM_SOURCE_LABEL = get_chat_provider_label()

# ---------------------------------------------------------------------------
# Phase 3: Explicit intent → route mapping (documentation only).
# The actual routing logic is in QueryRouter.route(), but this table
# provides a single reference for understanding the intent→source mapping.
# ---------------------------------------------------------------------------
ROUTING_TABLE = {
    "general_knowledge":   {"source": LLM_SOURCE_LABEL, "retrieval": None,             "note": "Direct LLM, no retrieval"},
    "legal_query":         {"source": "qdrant",      "retrieval": "legal_search",   "note": "Dense + BM25 → legal documents"},
    "document_query":      {"source": "qdrant",      "retrieval": "user_doc_search","note": "User uploaded docs"},
    "web_fallback":        {"source": "web_exa",     "retrieval": "exa_search",     "note": "Exa web fallback (low conf legal)"},
}



# ---------------------------------------------------------------------------
# Routing result
# ---------------------------------------------------------------------------


@dataclass
class RoutingResult:
    """Aggregated retrieval result returned by the router."""

    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    primary_source: str = "none"
    skip_retrieval: bool = False  # True when LLM-direct is sufficient


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class QueryRouter:
    """Route queries to PostgreSQL, Qdrant, or direct-LLM based on intent."""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def route(
        self,
        question: str,
        classification: QueryClassification,
        db: AsyncSession,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        document_id: Optional[int] = None,
        on_exa_fallback: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> RoutingResult:
        """Execute the retrieval strategy dictated by *classification*."""

        intent = classification.intent
        lang = classification.language
        result = RoutingResult()

        logger.info(
            "Routing: intent=%s lang=%s confidence=%.2f",
            intent,
            lang,
            classification.confidence,
        )

        if classification.use_llm_direct:
            result.skip_retrieval = True
            result.primary_source = LLM_SOURCE_LABEL
            return result

        # ----- legal_query → Qdrant (law filter) -----
        if intent == "legal_query":
            docs = await search_legal_documents(
                query=question,
                db=db,
                top_k=settings.TOP_K_RESULTS,
            )
            result.retrieved_docs = docs
            result.primary_source = "legal" if docs else "none"

            # ── Exa fallback: augment when retrieval confidence is low ──
            exa_docs = await self._maybe_exa_fallback(
                question, docs, intent, lang,
                session_id=session_id, user_id=user_id,
                on_exa_fallback=on_exa_fallback,
            )
            if exa_docs:
                if should_use_web_only_context(docs, question, intent):
                    result.retrieved_docs = list(exa_docs)[: settings.TOP_K_RESULTS]
                else:
                    combined_docs = list(exa_docs) + list(docs or [])
                    combined_docs.sort(
                        key=lambda d: float(d.get("similarity", 0.0)),
                        reverse=True,
                    )
                    result.retrieved_docs = combined_docs[: settings.TOP_K_RESULTS]
                result.primary_source = "web_exa"

            return result

        # ----- document_query → Qdrant (user doc chunks, owner_id filter) -----
        if intent == "document_query":
            if session_id:
                docs = await search_user_documents(
                    query=question,
                    db=db,
                    session_id=session_id,
                    document_id=document_id,
                    owner_id=user_id,
                )
                result.retrieved_docs = docs
                result.primary_source = "user_document" if docs else "none"
            else:
                result.skip_retrieval = True
                result.primary_source = LLM_SOURCE_LABEL
            return result
        result.skip_retrieval = True
        result.primary_source = LLM_SOURCE_LABEL
        return result

    # ------------------------------------------------------------------
    # Exa web fallback (Phase: Web Search Upgrade)
    # ------------------------------------------------------------------

    async def _maybe_exa_fallback(
        self,
        question: str,
        retrieved_docs: List[Dict],
        intent: str,
        language: str,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        on_exa_fallback: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> List[Dict]:
        """Check retrieval confidence and call Exa if below threshold.

        Returns extra docs from Exa (empty list if not triggered).
        """
        if not settings.EXA_ENABLED:
            return []

        confidence = compute_retrieval_confidence(retrieved_docs)
        if not should_trigger_exa(confidence, intent, retrieved_docs, question):
            logger.info(
                "Exa fallback NOT triggered: confidence=%.3f intent=%s",
                confidence, intent,
            )
            return []

        # Check cache first
        cached = cache_get(question, intent, language)
        if cached is not None:
            if on_exa_fallback:
                try:
                    await on_exa_fallback()
                except Exception as e:
                    logger.error("Error in on_exa_fallback (cache hit): %s", e)
            logger.info(
                "Exa fallback: using %d cached results (intent=%s)",
                len(cached), intent,
            )
            return cached

        # Check budget
        policy = get_exa_policy()
        allowed, reason = policy.can_call(session_id=session_id, user_id=user_id)
        if not allowed:
            logger.info(
                "Exa fallback SKIPPED (budget): %s confidence=%.3f",
                reason, confidence,
            )
            return []

        # Call Exa
        logger.info(
            "Exa fallback TRIGGERED: confidence=%.3f intent=%s query='%s'",
            confidence, intent, question[:60],
        )
        if on_exa_fallback:
            try:
                await on_exa_fallback()
            except Exception as e:
                logger.error("Error in on_exa_fallback: %s", e)
                
        start = time.monotonic()
        exa_docs = await search_exa(question, intent=intent, language=language)
        latency = time.monotonic() - start

        # Record call and cache results
        policy.record_call(session_id=session_id, user_id=user_id)
        if exa_docs:
            cache_set(question, intent, language, exa_docs)

        logger.info(
            "Exa fallback complete: docs=%d latency=%.2fs cache_hit=false "
            "intent=%s confidence=%.3f",
            len(exa_docs), latency, intent, confidence,
        )
        return exa_docs


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_router: QueryRouter | None = None


def get_query_router() -> QueryRouter:
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router
