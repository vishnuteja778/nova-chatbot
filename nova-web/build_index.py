"""Build the search index for Nova's built-in knowledge library.

Run this after adding or editing files in knowledge/:

    python build_index.py

It chunks every document, embeds each chunk with the local BGE model, and
saves everything to knowledge_index.json. Commit that file: the deployed app
loads it instead of re-reading and re-embedding the documents.
"""
import json

import rag

index = rag.build_library_index()
rag.INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False, indent=0), encoding="utf-8")

sources = sorted({c["source"] for c in index["chunks"]})
print(f"Indexed {len(index['chunks'])} chunks from {len(sources)} documents:")
for s in sources:
    print(f"  {s}: {sum(c['source'] == s for c in index['chunks'])} chunks")
print(f"Saved {rag.INDEX_FILE.name} ({rag.INDEX_FILE.stat().st_size / 1024:.0f} KB)")
