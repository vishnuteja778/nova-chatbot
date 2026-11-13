"""Build the search index:  files in data/  ->  chunks  ->  embeddings  ->  index/

Run it again whenever you add, change or delete a document, or change a
chunking setting in config.py.

    python ingest.py
"""
import time

import config
from documents import chunk_document, load_documents
from retriever import embed_passages, save_index


def main():
    docs = load_documents(config.DATA_DIR)
    if not docs:
        raise SystemExit(f"No .md/.txt/.pdf files found in {config.DATA_DIR}")

    chunks = []
    for name, text in docs:
        new = chunk_document(name, text, config.CHUNK_MAX_CHARS,
                             config.CHUNK_OVERLAP_CHARS, start_id=len(chunks))
        print(f"  {name:<28} {len(text):>6} chars -> {len(new):>3} chunks")
        chunks.extend(new)

    print(f"\nEmbedding {len(chunks)} chunks with {config.EMBED_MODEL} ...")
    t = time.perf_counter()
    vectors = embed_passages([c.embedding_text() for c in chunks])
    print(f"Done in {time.perf_counter() - t:.1f}s. Matrix shape: {vectors.shape} "
          f"({vectors.shape[0]} chunks x {vectors.shape[1]} numbers)")

    save_index(chunks, vectors)
    print(f"Saved index to {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
