# Ask My Docs: RAG with hybrid search, reranking and evals

Ask questions about your own documents (Markdown, text, PDF) and get answers **with citations** to the exact source section. If the answer isn't in the documents, it says so instead of guessing.

```
You: When will my expense refund arrive?

Your refund will be paid with your next salary payment, which is processed
on the 28th of the month. [1]

Sources:
  [1] expenses-and-travel.md > Submitting a claim
```

📘 **Learning guide (PDF):** [docs/guides/rag-learning-guide.pdf](../docs/guides/rag-learning-guide.pdf) explains every concept and line of this project.

## How it works

```
 INDEXING (once)                          ANSWERING (every question)
 data/*.md|pdf                            question
   -> chunks (800 chars, 150 overlap)       -> BM25 keyword search  ┐
   -> embeddings (BGE-small, local)         -> vector search        ┘ fuse with RRF -> top 20
   -> index/ (vectors.npy + chunks.json)    -> cross-encoder reranker -> top 4
                                            -> Groq LLM with numbered sources -> cited answer
```

| Piece | Choice | Why |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed | Free, runs on CPU, no API key |
| Keyword search | BM25, written from scratch in [bm25.py](bm25.py) | Catches exact codes like `EXP-44` |
| Fusion | Reciprocal Rank Fusion | Combines rankings with different score scales |
| Reranker | `ms-marco-MiniLM-L-6-v2` cross-encoder | Reads question + chunk together; most accurate |
| Vector store | NumPy matrix | 27 chunks don't need a database; easy to understand |
| LLM | Groq `openai/gpt-oss-20b` | Same key as Nova |

## Results

Retrieval evaluated on 18 hand-written questions ([eval/questions.json](eval/questions.json)), top-4:

| Method | Hit@4 | MRR |
|---|---|---|
| BM25 only | 94% | 0.78 |
| Vector only | 100% | 0.96 |
| Hybrid (RRF) | 94% | 0.94 |
| **Hybrid + rerank** | **100%** | **1.00** |

Hybrid alone *lost* a question that vector search got right ("expense refund": the docs say "reimbursed", so BM25 returned noise that outvoted the correct chunk). The reranker fixed it. This is a small, self-written test set, so treat the numbers as a regression check rather than a benchmark.

## Run it

```bash
cd rag
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # then paste your Groq key into .env
python ingest.py                # build the index (downloads ~150 MB of models once)
python evaluate.py --verbose    # measure retrieval
python ask.py                   # chat with your documents
```

Use your own documents: put them in `data/`, re-run `python ingest.py`. Options: `python ask.py --method vector|bm25|hybrid|rerank --show-context -k 6 "question"`.

## Files

| File | What it does |
|---|---|
| [config.py](config.py) | Every tunable setting |
| [documents.py](documents.py) | Load files, split into chunks with section headers |
| [bm25.py](bm25.py) | Keyword ranking from scratch |
| [retriever.py](retriever.py) | Embeddings, index save/load, the 4 search methods |
| [ingest.py](ingest.py) | Build the index |
| [ask.py](ask.py) | Retrieve, prompt, stream a cited answer |
| [evaluate.py](evaluate.py) | Hit@k and MRR per method |

The sample documents in `data/` are a fictional company handbook written for this demo.
