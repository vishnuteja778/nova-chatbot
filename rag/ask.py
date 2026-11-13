"""Step 3 of RAG: retrieve chunks, put them in the prompt, let the LLM answer with citations.

    python ask.py "How many days of annual leave do I get?"
    python ask.py                       # interactive mode
    python ask.py --method vector --show-context "..."
"""
import argparse
import sys

import groq

import config
from retriever import Retriever

SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources provided.
Rules:
- After every fact, cite the source number in square brackets, like [1] or [2][3].
- If the sources do not contain the answer, say: "I couldn't find that in the documents." Do not guess or use outside knowledge.
- Be concise: a few sentences or a short list."""


def build_prompt(question: str, results) -> str:
    # Each chunk gets a number so the model can cite it and we can show
    # the user exactly where an answer came from.
    sources = "\n\n".join(
        f"[{n}] (from {chunk.source}, section \"{chunk.section}\")\n{chunk.text}"
        for n, (chunk, _) in enumerate(results, start=1)
    )
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def answer(client: groq.Groq, retriever: Retriever, question: str, method: str, k: int,
           show_context: bool) -> None:
    results = retriever.search(question, k=k, method=method)
    if not results:
        print("No matching chunks found.\n")
        return
    prompt = build_prompt(question, results)
    if show_context:
        print("----- prompt sent to the model -----\n" + prompt + "\n------------------------------------\n")

    stream = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        temperature=0.1,   # low = stick to the sources, little creativity
        stream=True,
    )
    for part in stream:
        print(part.choices[0].delta.content or "", end="", flush=True)

    print("\n\nSources:")
    for n, (chunk, score) in enumerate(results, start=1):
        print(f"  [{n}] {chunk.source} > {chunk.section}   (score {score:.3f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Ask questions about the files in data/")
    parser.add_argument("question", nargs="*", help="leave empty for interactive mode")
    parser.add_argument("--method", choices=["rerank", "hybrid", "vector", "bm25"], default="rerank")
    parser.add_argument("-k", type=int, default=config.TOP_K, help="chunks to retrieve")
    parser.add_argument("--show-context", action="store_true", help="print the full prompt")
    args = parser.parse_args()

    # The Windows console uses an old encoding (cp1252) that can't print some
    # characters models like to use (e.g. a narrow space). Switch to UTF-8.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not config.GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY is missing. Copy .env.example to .env and add your key.")
    client = groq.Groq(api_key=config.GROQ_API_KEY)
    retriever = Retriever()

    if args.question:
        answer(client, retriever, " ".join(args.question), args.method, args.k, args.show_context)
        return
    print(f"Ask about your documents ({args.method} search). Empty line to quit.\n")
    while question := input("You: ").strip():
        print()
        answer(client, retriever, question, args.method, args.k, args.show_context)


if __name__ == "__main__":
    main()
