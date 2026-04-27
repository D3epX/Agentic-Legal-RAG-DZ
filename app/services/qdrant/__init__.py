"""Qdrant sub-package — vector database access layer."""

from app.services.qdrant.client import QdrantService, get_qdrant_service
from app.services.qdrant.collections import (
    COLLECTION_LEGAL_DOCUMENTS,
    COLLECTION_DOCUMENT_CHUNKS,
    ALL_COLLECTIONS,
)

__all__ = [
    "QdrantService",
    "get_qdrant_service",
    "COLLECTION_LEGAL_DOCUMENTS",
    "COLLECTION_DOCUMENT_CHUNKS",
    "ALL_COLLECTIONS",
]
