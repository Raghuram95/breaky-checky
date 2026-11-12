"""The searchable knowledge base: passages embedded as vectors in a persistent Chroma collection.

Embeddings come from a small sentence-transformer model (all-MiniLM-L6-v2) run on the CPU, and
closeness is cosine distance (0 means the same meaning, larger means further apart). The model is
loaded once per process through an LRU cache of size one: indexing, the workflow and the service
all share the same loaded model instead of reading it from disk again.
"""

import os
import re
from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import config

ADD_BATCH_SIZE = 500
CHUNK_NUMBER_PATTERN = r"-(\d+)$"


@lru_cache(maxsize=1)
def load_embedding_function():
    """Load the embedding model once and keep it for every later call in this process."""
    return SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL_NAME, device=config.EMBEDDING_DEVICE
    )


def build_vector_index(chunk_records: list[dict], store_path: str) -> int:
    """Build the store from scratch, replacing any collection already there, and fill it."""
    client = chromadb.PersistentClient(path=store_path)
    existing = [collection.name for collection in client.list_collections()]
    if config.VECTOR_COLLECTION_NAME in existing:
        client.delete_collection(config.VECTOR_COLLECTION_NAME)
    collection = client.create_collection(
        name=config.VECTOR_COLLECTION_NAME,
        embedding_function=load_embedding_function(),
        configuration={"hnsw": {"space": "cosine"}},
    )
    for start in range(0, len(chunk_records), ADD_BATCH_SIZE):
        batch = chunk_records[start : start + ADD_BATCH_SIZE]
        collection.add(
            ids=[record["chunk_id"] for record in batch],
            documents=[record["chunk_text"] for record in batch],
            metadatas=[{"document_name": record["document_name"]} for record in batch],
        )
    return collection.count()


def open_vector_index(store_path: str) -> Collection:
    """Open the store indexing built, with the same embedding model, without building anything."""
    if not os.path.isdir(store_path):
        raise ValueError(f"No knowledge base at {store_path}; run run_knowledge_indexing.py first")
    client = chromadb.PersistentClient(path=store_path)
    return client.get_collection(
        name=config.VECTOR_COLLECTION_NAME, embedding_function=load_embedding_function()
    )


def _chunk_number(chunk_id: str) -> int:
    """The passage's position within its document, read from the end of its identifier."""
    found = re.search(CHUNK_NUMBER_PATTERN, chunk_id)
    return int(found.group(1)) if found else 0


def _document_passages(collection: Collection, document_name: str):
    """Every stored passage of one document as (identifier, text), in document order."""
    stored = collection.get(where={"document_name": document_name}, include=["documents"])
    if not stored["ids"]:
        raise ValueError(f"The knowledge base holds no passages of {document_name}")
    return sorted(zip(stored["ids"], stored["documents"]), key=lambda pair: _chunk_number(pair[0]))


def read_document_chunks(collection: Collection, document_name: str) -> list[str]:
    """Read one document whole: every passage stored for it and no other document's."""
    return [text for _, text in _document_passages(collection, document_name)]


def find_nearest_passage(collection: Collection, question: str, document_name: str) -> str:
    """The one passage of one document closest in meaning to the question."""
    passage_count = len(_document_passages(collection, document_name))
    found = collection.query(
        query_texts=[question],
        n_results=passage_count,
        where={"document_name": document_name},
        include=["documents"],
    )
    return found["documents"][0][0]


def search_documents(collection: Collection, query_texts: list[str]) -> list[list[dict]]:
    """Search every document the chat may read; config.CHAT_RETRIEVAL_RESULT_COUNT passages each.

    The search over-fetches by the number of passages the excluded documents hold, then filters
    those documents out after retrieval. The casebook of past clearances is never handed on: it
    records what happened to earlier campaigns, and letting it into an answer would leak them.
    """
    excluded = collection.get(
        where={"document_name": {"$in": config.CHAT_EXCLUDED_DOCUMENTS}}, include=[]
    )
    candidate_count = min(
        collection.count(), config.CHAT_RETRIEVAL_RESULT_COUNT + len(excluded["ids"])
    )
    found = collection.query(
        query_texts=list(query_texts),
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )
    results = []
    for ids, texts, metadatas, distances in zip(
        found["ids"], found["documents"], found["metadatas"], found["distances"]
    ):
        passages = [
            {
                "chunk_id": chunk_id,
                "chunk_text": text,
                "document_name": metadata["document_name"],
                "distance": float(distance),
            }
            for chunk_id, text, metadata, distance in zip(ids, texts, metadatas, distances)
            if metadata["document_name"] not in config.CHAT_EXCLUDED_DOCUMENTS
        ]
        results.append(passages[: config.CHAT_RETRIEVAL_RESULT_COUNT])
    return results
