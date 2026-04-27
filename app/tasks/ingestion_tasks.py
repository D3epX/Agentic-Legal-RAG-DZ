"""
Ingestion tasks — legal documents + web page crawling.
"""
import logging
from app.celery_app import celery

logger = logging.getLogger(__name__)


@celery.task(bind=True, name="app.tasks.ingest_legal_batch", max_retries=1)
def ingest_legal_batch(self, documents: list):
    """Ingest a batch of legal documents with embeddings into PG + Qdrant."""
    import asyncio
    asyncio.run(_ingest_legal_async(documents))


async def _ingest_legal_async(documents: list):
    from app.db import AsyncSessionLocal
    from app.models import LegalDocument
    from app.services.documents.embeddings import get_embedding_service
    from app.services.qdrant import get_qdrant_service, COLLECTION_LEGAL_DOCUMENTS
    from qdrant_client.models import PointStruct

    embedding_service = get_embedding_service()
    qdrant = get_qdrant_service()

    async with AsyncSessionLocal() as db:
        points: list[PointStruct] = []
        for doc_data in documents:
            text = f"{doc_data['title']} {doc_data['content']}"
            emb = embedding_service.encode_single(text)
            legal_doc = LegalDocument(
                title=doc_data["title"],
                jurisdiction=doc_data.get("jurisdiction"),
                category=doc_data.get("category"),
                content=doc_data["content"],
                language=doc_data.get("language", "en"),
                source_reference=doc_data.get("source_reference"),
                keywords=doc_data.get("keywords"),
            )
            db.add(legal_doc)
            await db.flush()

            points.append(
                PointStruct(
                    id=legal_doc.id,
                    vector=emb,
                    payload={
                        "type": "law",
                        "language": legal_doc.language or "en",
                        "jurisdiction": legal_doc.jurisdiction or "",
                        "category": legal_doc.category or "",
                    },
                )
            )

        await db.commit()
        qdrant.upsert_batch(COLLECTION_LEGAL_DOCUMENTS, points)
        logger.info("Ingested %d legal documents via Celery", len(documents))


