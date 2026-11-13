"""RAG for Nova: search the built-in knowledge library and the user's documents.

This is a web-friendly version of the pipeline in ../rag (see that project's
README and the learning guide for the full explanation):

  INDEXING                              SEARCHING (every search_documents call)
  text -> chunks (800 chars, overlap)   query -> BM25 keyword ranking   ┐
       -> embeddings (BGE-small)        query -> vector (cosine) ranking┘ fuse with RRF -> top k
       -> int8-compressed vectors

Two kinds of documents are searched together:
  * the built-in library  (knowledge/*.md, indexed once by build_index.py)
  * the user's documents   (uploaded in the browser, indexed by /api/library,
                            stored in the browser and sent back with each chat)

Embeddings run locally with fastembed: free, no API key. If the model can't
load (for example on a server without enough space), search still works with
BM25 keywords alone, just a little less smart.
"""
import base64
import json
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
KNOWLEDGE_DIR = HERE / "knowledge"
INDEX_FILE = HERE / "knowledge_index.json"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384 numbers per text, same model as ../rag
CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP_CHARS = 150
RRF_K = 60                                # standard Reciprocal Rank Fusion constant
TOP_K = 4

# Limits for the user's own documents (they travel with every chat request).
MAX_DOC_CHARS = 300_000                   # about 150 pages of text per document
MAX_LIBRARY_CHUNKS = 400                  # across all of a user's documents


# ---------- chunks ----------

@dataclass
class Chunk:
    source: str            # file name, used in citations
    section: str           # nearest heading above the chunk
    text: str
    vec: np.ndarray | None = field(default=None, repr=False)   # normalised embedding
    origin: str = "library"                                    # "library" or "upload"

    def header_text(self) -> str:
        # "Contextual chunk header": embedding the file and section with the text
        # makes a chunk like "Above 70 is overbought" findable for "RSI".
        return f"{self.source} > {self.section}\n{self.text}"


def split_into_blocks(text: str) -> list[tuple[str, str]]:
    """Paragraphs, each tagged with the Markdown heading it sits under."""
    section = "Introduction"
    blocks = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        heading = re.match(r"^#{1,6}\s+(.*)", para)
        if heading:
            first, _, rest = para.partition("\n")
            section = re.sub(r"^#{1,6}\s+", "", first).strip()[:120]
            para = rest.strip()
            if not para:
                continue
        blocks.append((section, para))
    return blocks


def hard_split(text: str, max_chars: int) -> list[str]:
    """Cut an over-long paragraph at sentence ends (or spaces, as a last resort)."""
    units = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        units.extend(sentence.split() if len(sentence) > max_chars else [sentence])
    pieces, current = [], ""
    for u in units:
        if current and len(current) + len(u) + 1 > max_chars:
            pieces.append(current)
            current = u
        else:
            current = f"{current} {u}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_text(source: str, text: str, origin: str = "library") -> list[Chunk]:
    """Pack paragraphs into chunks of at most CHUNK_MAX_CHARS.

    A new chunk starts when the section changes (so a chunk never mixes topics)
    or when the next paragraph won't fit. Each new chunk repeats the tail of
    the previous one (overlap) so a fact on the boundary isn't cut in half.
    """
    chunks: list[Chunk] = []
    current, current_section = "", None

    def flush():
        if current.strip():
            chunks.append(Chunk(source, current_section or "Introduction", current.strip(), origin=origin))

    for section, para in split_into_blocks(text):
        for piece in hard_split(para, CHUNK_MAX_CHARS) if len(para) > CHUNK_MAX_CHARS else [para]:
            if section != current_section:
                flush()
                current, current_section = piece, section
            elif len(current) + len(piece) + 2 > CHUNK_MAX_CHARS:
                flush()
                tail = current[-CHUNK_OVERLAP_CHARS:]
                tail = tail[tail.find(" ") + 1:] if " " in tail else tail
                fits = tail and len(tail) + len(piece) + 2 <= CHUNK_MAX_CHARS
                current = f"{tail}\n\n{piece}" if fits else piece
            else:
                current = f"{current}\n\n{piece}" if current else piece
    flush()
    return chunks


# ---------- BM25 keyword search ----------

STOPWORDS = set("a an the and or of to in on for is are be it i my me do does what how when "
                "can with at by you your we our if this that as was were from about which".split())


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*", text.lower())
    return [w for w in words if w not in STOPWORDS]


class BM25:
    """The classic search-engine formula: rare words count more, repeats count
    less and less, and matches in short chunks count more than in long ones."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / max(len(self.docs), 1)
        n = len(self.docs)
        df = Counter(w for d in self.docs for w in d)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            norm = 1 - self.b + self.b * length / (self.avg or 1)
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
            out.append(s)
        return out


# ---------- embeddings ----------

_model = None
_model_failed = False
_model_lock = threading.Lock()


def embedder():
    """Load the embedding model once. Returns None if it can't be loaded."""
    global _model, _model_failed
    if _model is not None or _model_failed:
        return _model
    if os.getenv("NOVA_EMBEDDINGS", "on").strip().lower() == "off":
        _model_failed = True
        return None
    with _model_lock:
        if _model is None and not _model_failed:
            try:
                from fastembed import TextEmbedding
                # On Vercel only /tmp is writable, so the model is cached there.
                cache = "/tmp/fastembed" if os.getenv("VERCEL") else None
                _model = TextEmbedding(EMBED_MODEL, cache_dir=cache)
            except Exception as e:  # not installed, no disk space, no network...
                print(f"[rag] embeddings unavailable, using BM25 only: {type(e).__name__}: {e}")
                _model_failed = True
    return _model


def embed_passages(texts: list[str]) -> np.ndarray | None:
    model = embedder()
    if model is None or not texts:
        return None
    return np.array(list(model.passage_embed(texts)), dtype=np.float32)


def embed_query(text: str) -> np.ndarray | None:
    model = embedder()
    if model is None:
        return None
    # BGE adds a short instruction to queries ("asymmetric" retrieval).
    return np.array(next(iter(model.query_embed([text]))), dtype=np.float32)


# Vectors are stored as int8 (one byte per number instead of four), base64 text.
# The embeddings are normalised to length 1, so every number is in [-1, 1];
# multiplying by 127 and rounding loses almost no search quality. This is
# "quantisation", the same trick vector databases use to save memory.

def pack_vec(vec: np.ndarray) -> str:
    q = np.clip(np.round(vec * 127), -127, 127).astype(np.int8)
    return base64.b64encode(q.tobytes()).decode("ascii")


def unpack_vec(text: str | None) -> np.ndarray | None:
    if not text:
        return None
    try:
        raw = np.frombuffer(base64.b64decode(text, validate=True), dtype=np.int8)
    except (ValueError, TypeError):
        return None
    if raw.size != 384:
        return None
    vec = raw.astype(np.float32) / 127
    norm = np.linalg.norm(vec)
    return vec / norm if norm else None


# ---------- the built-in library ----------

def load_documents(folder: Path) -> list[tuple[str, str]]:
    docs = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.name.lower() == "readme.md":
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader
            text = "\n\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        elif suffix in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
        else:
            continue
        if text.strip():
            docs.append((path.name, text))
    return docs


def build_library_index() -> dict:
    chunks = [c for name, text in load_documents(KNOWLEDGE_DIR) for c in chunk_text(name, text)]
    vecs = embed_passages([c.header_text() for c in chunks])
    if vecs is None:
        raise SystemExit("The embedding model could not be loaded; install fastembed first.")
    return {
        "embed_model": EMBED_MODEL,
        "chunk_max_chars": CHUNK_MAX_CHARS,
        "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
        "chunks": [{"source": c.source, "section": c.section, "text": c.text, "vec": pack_vec(v)}
                   for c, v in zip(chunks, vecs)],
    }


_library: list[Chunk] | None = None


def library() -> list[Chunk]:
    global _library
    if _library is None:
        try:
            data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            _library = [Chunk(c["source"], c["section"], c["text"], unpack_vec(c.get("vec")))
                        for c in data["chunks"]]
        except FileNotFoundError:
            print("[rag] knowledge_index.json not found; run build_index.py")
            _library = []
    return _library


# ---------- the user's documents ----------

def index_upload(name: str, text: str) -> list[dict]:
    """Chunk and embed one uploaded document. The browser stores the result."""
    text = text[:MAX_DOC_CHARS]
    chunks = chunk_text(name, text, origin="upload")[:MAX_LIBRARY_CHUNKS]
    vecs = embed_passages([c.header_text() for c in chunks])
    return [{"section": c.section, "text": c.text,
             "vec": pack_vec(vecs[i]) if vecs is not None else None}
            for i, c in enumerate(chunks)]


def uploads_to_chunks(docs: list[dict]) -> list[Chunk]:
    out = []
    for d in docs:
        for c in d.get("chunks", []):
            out.append(Chunk(d["name"], c.get("section") or "Text", c["text"],
                             unpack_vec(c.get("vec")), origin="upload"))
            if len(out) >= MAX_LIBRARY_CHUNKS:
                return out
    return out


# ---------- hybrid search ----------

def search(query: str, uploads: list[Chunk] | None = None, k: int = TOP_K,
           scope: str = "all", method: str = "hybrid") -> list[tuple[Chunk, float]]:
    """Hybrid search: BM25 and vector rankings merged with Reciprocal Rank Fusion.

    BM25 and cosine scores use different scales, so we can't add them.
    RRF uses only each chunk's RANK in each list: score = sum 1 / (60 + rank).
    A chunk ranked well by both methods wins.
    """
    pool: list[Chunk] = []
    if scope in ("all", "library"):
        pool += library()
    if scope in ("all", "uploads"):
        pool += uploads or []
    if not pool or not query.strip():
        return []

    depth = max(k * 5, 20)
    rankings: list[list[int]] = []

    if method in ("hybrid", "bm25"):
        scores = BM25([c.header_text() for c in pool]).scores(query)
        order = sorted(range(len(pool)), key=lambda i: -scores[i])
        rankings.append([i for i in order[:depth] if scores[i] > 0])

    if method in ("hybrid", "vector"):
        q = embed_query(query)
        with_vec = [i for i, c in enumerate(pool) if c.vec is not None]
        if q is not None and with_vec:
            matrix = np.stack([pool[i].vec for i in with_vec])
            sims = matrix @ q           # vectors have length 1, so dot product = cosine
            order = np.argsort(-sims)[:depth]
            rankings.append([with_vec[i] for i in order])

    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, start=1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
    best = sorted(fused, key=lambda i: -fused[i])[:k]
    return [(pool[i], fused[i]) for i in best]


def status() -> dict:
    return {"library_chunks": len(library()), "embeddings": embedder() is not None}
