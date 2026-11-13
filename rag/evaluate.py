"""Measure retrieval quality, so changes are judged by numbers, not by feel.

For each test question we know which file holds the answer and a phrase that
must appear in the right chunk. We check whether each search method finds it.

  Hit@k  share of questions where a correct chunk is in the top k.
         If retrieval misses, the LLM cannot answer, however smart it is.
  MRR    Mean Reciprocal Rank: 1 if the correct chunk is first, 1/2 if
         second, 1/3 if third, 0 if missing. Rewards ranking it high.

    python evaluate.py            # summary table
    python evaluate.py --verbose  # also show each question's result
"""
import argparse
import json

import config
from retriever import Retriever


def first_correct_rank(results, case) -> int | None:
    for rank, (chunk, _) in enumerate(results, start=1):
        if chunk.source == case["source"] and case["expected"].lower() in chunk.text.lower():
            return rank
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=config.TOP_K)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cases = json.loads(config.EVAL_FILE.read_text(encoding="utf-8"))
    retriever = Retriever()
    methods = ["bm25", "vector", "hybrid", "rerank"]
    ranks = {m: [] for m in methods}

    for case in cases:
        row = []
        for m in methods:
            rank = first_correct_rank(retriever.search(case["question"], k=args.k, method=m), case)
            ranks[m].append(rank)
            row.append(f"{m}={rank or '-'}")
        if args.verbose:
            print(f"{case['question'][:58]:<60} {'  '.join(row)}")

    print(f"\n{len(cases)} questions, top-{args.k} retrieval\n")
    print(f"{'method':<8} {'Hit@' + str(args.k):>8} {'MRR':>7}")
    for m in methods:
        hits = sum(r is not None for r in ranks[m]) / len(cases)
        mrr = sum(1 / r for r in ranks[m] if r) / len(cases)
        print(f"{m:<8} {hits:>8.0%} {mrr:>7.2f}")


if __name__ == "__main__":
    main()
