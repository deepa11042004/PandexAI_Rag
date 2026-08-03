"""ChromaDB-backed vector store: one persistent collection per chat session."""

from __future__ import annotations

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import get_settings


class VectorStoreManager:
    """Wraps a per-session Chroma collection plus bookkeeping of source filenames/content hashes."""

    def __init__(self, session_id: str, embeddings: Embeddings) -> None:
        settings = get_settings()
        self.session_id = session_id
        self.embeddings = embeddings
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.store = Chroma(
            client=self._client,
            collection_name=f"session_{session_id}",
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: list[Document], source_id: str, content_hash: str) -> None:
        """Embed and add chunks, stamping each with source_id (deletion) and content_hash (dedup)."""
        if not documents:
            return
        for doc in documents:
            doc.metadata["source_id"] = source_id
            doc.metadata["content_hash"] = content_hash
        self.store.add_documents(documents)

    def similarity_search_with_score(
        self, query: str, k: int = 12, source_id: str | None = None
    ) -> list[tuple[Document, float]]:
        if self.count() == 0:
            return []
        filter_ = {"source_id": source_id} if source_id else None
        return self.store.similarity_search_with_score(query, k=k, filter=filter_)

    def delete_source(self, source_id: str) -> None:
        self.store.delete(where={"source_id": source_id})

    def clear(self) -> None:
        try:
            self._client.delete_collection(f"session_{self.session_id}")
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
        self.store = Chroma(
            client=self._client,
            collection_name=f"session_{self.session_id}",
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        try:
            return self.store._collection.count()  # noqa: SLF001 - langchain-chroma exposes no public count()
        except Exception:  # noqa: BLE001
            return 0

    def existing_content_hashes(self) -> set[str]:
        """All content_hash metadata values currently stored, for duplicate-source detection."""
        if self.count() == 0:
            return set()
        result = self.store._collection.get(include=["metadatas"])  # noqa: SLF001
        return {m.get("content_hash") for m in result.get("metadatas", []) if m.get("content_hash")}


def delete_session_collection(session_id: str) -> None:
    """Permanently drop a session's Chroma collection (no recreation) - used when the whole session
    is being deleted, as opposed to `VectorStoreManager.clear()` which empties it but keeps it alive
    for further use. Doesn't need an embeddings client since dropping a collection by name requires
    no embedding function, unlike the read/write operations `VectorStoreManager` wraps.
    """
    client = chromadb.PersistentClient(path=get_settings().chroma_persist_dir)
    try:
        client.delete_collection(f"session_{session_id}")
    except Exception:  # noqa: BLE001 - collection may not exist
        pass
