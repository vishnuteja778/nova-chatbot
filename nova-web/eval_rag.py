"""Measure how well Nova finds the right part of its knowledge library.

    python eval_rag.py            # summary per search method
    python eval_rag.py --verbose  # also show each miss

For each test question we know which document section holds the answer.
  Hit@4: how often that section is in the top 4 results (what the agent reads)
  MRR:   mean of 1/rank of the first correct result (1.0 = always first)
Re-run this after changing chunking, the model, or adding documents.
"""
import json
import sys
from pathlib import Path

import rag

K = rag.TOP_K
questions = json.loads((Path(__file__).parent / "knowledge_eval.json").read_text(encoding="utf-8"))
verbose = "--verbose" in sys.argv

print(f"{len(questions)} questions, top {K}\n")
print(f"{'method':<8} {'Hit@4':>6} {'MRR':>6}")
for method in ("bm25", "vector", "hybrid"):
    hits, rr, misses = 0, 0.0, []
    for item in questions:
        results = rag.search(item["q"], k=K, scope="library", method=method)
        rank = next((n for n, (c, _) in enumerate(results, 1)
                     if c.source == item["source"] and c.section == item["section"]), None)
        if rank:
            hits += 1
            rr += 1 / rank
        else:
            misses.append((item["q"], [f"{c.source} > {c.section}" for c, _ in results[:2]]))
    print(f"{method:<8} {hits / len(questions):>6.0%} {rr / len(questions):>6.2f}")
    if verbose:
        for q, got in misses:
            print(f"    miss: {q}\n          got: {got}")
