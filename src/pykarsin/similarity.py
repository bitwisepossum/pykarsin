"""Near-duplicate and relevance detection over codebook text.

Char n-gram TF-IDF + cosine similarity: catches Finnish inflection/compounding
without a lemmatizer, at the cost of being purely lexical (see check_polarity).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# clusters this size or larger are more likely to be chained through a shared
# stem than to actually be one group - not auto-filtered, just flagged
CLUSTER_WARNING_SIZE = 4

NEGATION_MARKERS = frozenset({
    "ei", "eikä", "eivät", "eivätkä", "en", "et", "emme", "ette",
    "älä", "älkää",
})


def parent_label(text: str) -> str:
    return text.split(":", 1)[0].strip()


def is_parent_child(a: str, b: str) -> bool:
    pa, pb = parent_label(a), parent_label(b)
    return pa == pb or a.startswith(pb) or b.startswith(pa)


def has_negation(text: str) -> bool:
    words = text.lower().replace(":", " ").split()
    return any(w in NEGATION_MARKERS for w in words)


def check_polarity(a: str, b: str) -> bool:
    """True if exactly one of the pair carries a negation - similar wording,
    opposite meaning. Word-list based, so it won't catch every case."""
    return has_negation(a) != has_negation(b)


def build_tfidf_matrix(texts):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    return vec.fit_transform(texts)


def cosine_sim_matrix(X):
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0)  # a code is not its own duplicate
    return sim


@dataclass
class PairCandidate:
    score: float
    i: int
    j: int
    polarity_flag: bool


@dataclass
class ClusterCandidate:
    members: list[int]
    chaining_warning: bool


def find_pairs(codes: list[str], sim, pair_threshold: float) -> list[PairCandidate]:
    n = len(codes)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] <= pair_threshold or is_parent_child(codes[i], codes[j]):
                continue
            pairs.append(PairCandidate(sim[i, j], i, j, check_polarity(codes[i], codes[j])))
    pairs.sort(key=lambda p: p.score, reverse=True)
    return pairs


def find_clusters(codes: list[str], sim, cluster_threshold: float) -> list[ClusterCandidate]:
    n = len(codes)
    uf = list(range(n))

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            uf[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > cluster_threshold and not is_parent_child(codes[i], codes[j]):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    clusters = [
        ClusterCandidate(members, len(members) >= CLUSTER_WARNING_SIZE)
        for members in groups.values()
        if len(members) >= 2
    ]
    clusters.sort(key=lambda c: c.members[0])
    return clusters


@dataclass
class RelevanceFlag:
    index: int
    reason: str


def scan_relevance(
    codes: list[str],
    rq_texts: list[str],
    threshold: float = 0.1,
    min_length: int = 12,
) -> list[RelevanceFlag]:
    """Flag codes for human review instead of auto-deciding relevance - a
    plain keyword-absence filter was tried before this and threw too many
    false positives on real codebooks."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    X = vec.fit_transform(codes)
    rq_X = vec.transform(rq_texts)
    max_sim = cosine_similarity(X, rq_X).max(axis=1)

    flags = []
    for i, code in enumerate(codes):
        if len(code) < min_length:
            flags.append(RelevanceFlag(i, "too short for a reliable n-gram signal"))
        elif max_sim[i] < threshold:
            flags.append(RelevanceFlag(i, "low similarity to the stated research questions"))
    return flags
