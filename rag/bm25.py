"""Keyword search with BM25, written from scratch (no library) so you can see it.

BM25 is the classic search-engine ranking formula (Elasticsearch and Lucene
use it). It rewards a chunk for containing the query's words, especially:
  * rare words (IDF): "EXP-44" matters more than "the"
  * repeated words (TF), but with diminishing returns (parameter k1)
  * in a short chunk (length normalisation, parameter b)

Embeddings understand meaning ("puppy" ~ "dog") but are weak at exact
codes and names. BM25 is the opposite. That's why we combine them (hybrid).
"""
import math
import re
from collections import Counter

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "be", "it", "i", "my", "me", "do", "does", "what", "how", "when", "can",
    "with", "at", "by", "you", "your", "we", "our", "if", "this", "that", "as",
}


def tokenize(text: str) -> list[str]:
    # Keep hyphenated codes like "exp-44" as one token.
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    return [w for w in words if w not in STOPWORDS]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]          # word counts per chunk
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.docs), 1)
        n = len(self.docs)
        doc_freq = Counter(word for d in self.docs for word in d)  # how many chunks contain each word
        # IDF: rare words get big weights, common words small ones.
        self.idf = {w: math.log(1 + (n - df + 0.5) / (df + 0.5)) for w, df in doc_freq.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    norm = 1 - self.b + self.b * length / self.avg_len
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
            out.append(s)
        return out
