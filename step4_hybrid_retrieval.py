# Step 4: hybrid retrieval - rank candidate commits by how related they are to the slow benchmark.
#
# Query     = what the alert tells us (benchmark class + method + "slower").
# Documents = the candidate commits (message + changed files + added code).
# Two retrievers:
#   1. BM25       - keyword match (exact names like encodeIntoCompressedByteBuffer)
#   2. Embeddings - meaning match (e.g. "checksum loop" is related to "slower")
# Fusion: Reciprocal Rank Fusion (RRF) - a commit ranked high by either retriever moves up.
# The ranker never sees culprit_id, culprit_range, meta or measurements.

import csv
import glob
import json
import math
import random
import re
from collections import Counter

import numpy as np

random.seed(0)


# ---------------------------------------------------------------- text
def tokenize(text):
    """'encodeIntoCompressedByteBuffer' -> ['encode', 'into', 'compressed', 'byte', 'buffer']"""
    words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", text)
    return [w.lower() for w in words if len(w) > 1]


def commit_text(commit):
    added = [line[1:] for line in commit["diff"].split("\n")
             if line.startswith("+") and not line.startswith("+++")]
    headers = [line for line in commit["diff"].split("\n") if line.startswith("@@")]
    return " ".join([commit["message"], " ".join(commit["files"]), " ".join(headers), " ".join(added)])


def build_query(instance):
    # benchmark id looks like: org__project#package.Class.method#params.json
    method_path = instance["benchmark_context"]["benchmark"].split("#")[1]
    cls, method = method_path.split(".")[-2:]
    cls = cls.replace("Bench", "").replace("Benchmark", "")
    return f"{cls} {method} slower performance regression extra work"


# ---------------------------------------------------------------- retriever 1: BM25
def bm25_scores(query, docs, k1=1.5, b=0.75):
    q = tokenize(query)
    docs = [tokenize(d) for d in docs]
    avg_len = sum(len(d) for d in docs) / len(docs)
    n = len(docs)
    scores = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for term in set(q):
            df = sum(term in other for other in docs)          # how many commits contain the word
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))    # rare words count more
            f = tf[term]
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avg_len))
        scores.append(s)
    return scores


# ---------------------------------------------------------------- retriever 2: embeddings
_model = None


def dense_scores(query, docs):
    global _model
    try:
        from sentence_transformers import SentenceTransformer
        if _model is None:
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        vectors = _model.encode([query] + docs, normalize_embeddings=True)
        return list(vectors[1:] @ vectors[0])                  # cosine similarity
    except (ImportError, OSError):
        # fallback if the model is not installed: TF-IDF + LSA (a simple "semantic" space)
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [" ".join(tokenize(t)) for t in [query] + docs]
        tfidf = TfidfVectorizer().fit_transform(texts)
        vectors = TruncatedSVD(n_components=min(10, len(texts) - 1), random_state=0).fit_transform(tfidf)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return list(vectors[1:] @ vectors[0])


# ---------------------------------------------------------------- fusion
def ranks(scores):
    """Rank 1 = highest score."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    r = [0] * len(scores)
    for position, i in enumerate(order):
        r[i] = position + 1
    return r


def rrf(rank_lists, k=60):
    return [sum(1 / (k + r[i]) for r in rank_lists) for i in range(len(rank_lists[0]))]


def hybrid_rank(instance):
    """Commit ids, most suspicious first (used by step3_run.py)."""
    docs = [commit_text(c) for c in instance["commits"]]
    query = build_query(instance)
    fused = rrf([ranks(bm25_scores(query, docs)), ranks(dense_scores(query, docs))])
    order = sorted(range(len(docs)), key=lambda i: -fused[i])
    return [instance["commits"][i]["id"] for i in order]


# ---------------------------------------------------------------- evaluation
if __name__ == "__main__":
    rows = []
    for path in sorted(glob.glob("instances/*.json")):
        instance = json.load(open(path))
        docs = [commit_text(c) for c in instance["commits"]]
        query = build_query(instance)
        culprit = [c["id"] for c in instance["commits"]].index(instance["culprit_id"])
        bm25 = ranks(bm25_scores(query, docs))
        dense = ranks(dense_scores(query, docs))
        hybrid = ranks(rrf([bm25, dense]))
        rand = ranks([random.random() for _ in docs])
        row = {"instance": instance["instance_id"], "random": rand[culprit], "bm25": bm25[culprit],
               "dense": dense[culprit], "hybrid": hybrid[culprit]}
        rows.append(row)
        print(f"{row['instance']:15} culprit rank -> random {row['random']:2}  bm25 {row['bm25']:2}  "
              f"dense {row['dense']:2}  hybrid {row['hybrid']:2}")

    print("\nRetriever   top-1   top-3   MRR")
    for name in ["random", "bm25", "dense", "hybrid"]:
        r = [row[name] for row in rows]
        print(f"{name:10} {sum(x == 1 for x in r) / len(r):6.0%} {sum(x <= 3 for x in r) / len(r):7.0%}"
              f"   {sum(1 / x for x in r) / len(r):.2f}")

    with open("retrieval_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved retrieval_results.csv")
