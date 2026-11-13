"""All the knobs of the RAG pipeline in one place.

Change a value here, re-run `python ingest.py`, then `python evaluate.py`
to see if retrieval got better or worse. That loop is how real RAG
systems are tuned.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

# --- Where things live ---
DATA_DIR = HERE / "data"          # put your .md / .txt / .pdf files here
INDEX_DIR = HERE / "index"        # ingest.py writes the searchable index here
EVAL_FILE = HERE / "eval" / "questions.json"

# --- Chunking ---
# A chunk is the unit we search over. Too big: the match is diluted and you
# waste prompt tokens. Too small: a chunk loses the context needed to answer.
CHUNK_MAX_CHARS = 800             # roughly 150-200 words
CHUNK_OVERLAP_CHARS = 150         # repeated between neighbours so no sentence is cut off alone

# --- Embeddings (runs locally on your CPU, no API key) ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384 numbers per text, ~67 MB download once

# --- Retrieval ---
TOP_K = 4                         # how many chunks go into the prompt
RRF_K = 60                        # constant in Reciprocal Rank Fusion (60 is the standard value)
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"   # cross-encoder, ~80 MB download once
RERANK_CANDIDATES = 20            # how many hybrid results the reranker re-orders

# --- Generation (Groq, same as Nova) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "").strip() or "openai/gpt-oss-20b"
