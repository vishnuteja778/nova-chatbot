"""Step 2 of RAG: the index (save/load) and the three ways to search it.

  vector  - cosine similarity between embeddings (meaning)
  bm25    - keyword matching (exact words)
  hybrid  - both, merged with Reciprocal Rank Fusion
  rerank  - hybrid candidates, re-ordered by a cross-encoder (the default)
"""
import json

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

import config
from bm25 import BM25
from documents import Chunk

_model = None


def embedder() -> TextEmbedding:
    # Loading the model takes a second, so do it once and reuse it.
    global _model
    if _model is None:
        _model = TextEmbedding(config.EMBED_MODEL)
    return _model


_reranker = None


def reranker() -> TextCrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(config.RERANK_MODEL)
    return _reranker


def embed_passages(texts: list[str]) -> np.ndarray:
    return np.array(list(embedder().passage_embed(texts)), dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    # BGE models expect queries to carry a short instruction prefix;
    # query_embed adds it for us. Documents and queries are embedded slightly
    # differently on purpose ("asymmetric" retrieval).
    return np.array(next(iter(embedder().query_embed([text]))), dtype=np.float32)


def save_index(chunks: list[Chunk], vectors: np.ndarray) -> None:
    config.INDEX_DIR.mkdir(exist_ok=True)
    np.save(config.INDEX_DIR / "vectors.npy", vectors)
    (config.INDEX_DIR / "chunks.json").write_text(
        json.dumps([c.to_dict() for c in chunks], indent=1, ensure_ascii=False), encoding="utf-8")
    (config.INDEX_DIR / "meta.json").write_text(json.dumps({
        "embed_model": config.EMBED_MODEL,
        "chunk_max_chars": config.CHUNK_MAX_CHARS,
        "chunk_overlap_chars": config.CHUNK_OVERLAP_CHARS,
    }, indent=1), encoding="utf-8")


class Retriever:
    def __init__(self):
        if not (config.INDEX_DIR / "vectors.npy").exists():
            raise SystemExit("No index found. Run:  python ingest.py")
        meta = json.loads((config.INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
        if meta["embed_model"] != config.EMBED_MODEL:
            raise SystemExit("The index was built with a different embedding model. Re-run: python ingest.py")
        self.vectors = np.load(config.INDEX_DIR / "vectors.npy")
        raw = json.loads((config.INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        self.chunks = [Chunk(**c) for c in raw]
        # BM25 is cheap to rebuild, so we don't store it.
        self.bm25 = BM25([c.embedding_text() for c in self.chunks])

    def vector_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = embed_query(query)
        # The vectors are normalised to length 1, so a dot product IS the
        # cosine similarity: 1 = same meaning, ~0 = unrelated.
        sims = self.vectors @ q
        top = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in top]

    def bm25_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        scores = self.bm25.scores(query)
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i], scores[i]) for i in top if scores[i] > 0]

    def hybrid_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Reciprocal Rank Fusion: score = sum of 1 / (RRF_K + rank) over both lists.

        Cosine scores (0..1) and BM25 scores (0..20+) are on different scales,
        so we can't just add them. RRF ignores the raw scores and uses only the
        RANK in each list. A chunk ranked high by both wins.
        """
        pool = max(k * 5, 20)   # look deeper than k in each list before merging
        fused: dict[int, float] = {}
        for results in (self.vector_search(query, pool), self.bm25_search(query, pool)):
            for rank, (chunk, _) in enumerate(results, start=1):
                fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (config.RRF_K + rank)
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [(self.chunks[i], fused[i]) for i in top]

    def rerank_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Two-stage retrieval: a fast, wide search, then a slow, precise re-ordering.

        Stage 1 (hybrid) fetches ~20 candidates. It's fast because each chunk's
        vector was computed in advance, but it never sees the query and the chunk
        TOGETHER. Stage 2, a cross-encoder, reads "query + chunk" as one input
        and outputs a relevance score. Much more accurate, but too slow to run
        on every chunk in a big collection, so we only run it on the candidates.
        """
        candidates = self.hybrid_search(query, config.RERANK_CANDIDATES)
        scores = list(reranker().rerank(query, [c.embedding_text() for c, _ in candidates]))
        ranked = sorted(zip(candidates, scores), key=lambda pair: -pair[1])[:k]
        return [(chunk, float(score)) for (chunk, _), score in ranked]

    def search(self, query: str, k: int = config.TOP_K, method: str = "rerank"):
        return {"vector": self.vector_search, "bm25": self.bm25_search,
                "hybrid": self.hybrid_search, "rerank": self.rerank_search}[method](query, k)
