# Nova's knowledge library

These files are the built-in library Nova's Researcher agent searches with RAG.

- Add your own `.md`, `.txt` or `.pdf` files here (for example a company's annual report or concall transcript).
- Then rebuild the index from the `nova-web` folder:

```bash
python build_index.py
```

This writes `knowledge_index.json`, which is committed and deployed with the app. The app never re-reads this folder at runtime; it only loads the index.

This README is skipped by the indexer.
