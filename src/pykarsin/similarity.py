"""Near-duplicate and relevance detection over codebook text.

Char n-gram TF-IDF + cosine similarity: catches Finnish inflection/compounding
without a lemmatizer, at the cost of being purely lexical (see check_polarity).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# clusters this size or larger are more likely to be chained through a shared
# stem than to actually be one group - not auto-filtered, just flagged
CLUSTER_WARNING_SIZE = 4

NEGATION_MARKERS = frozenset({
    "ei", "eikä", "eivät", "eivätkä", "en", "et", "emme", "ette",
    "älä", "älkää",
})

# single source of truth for values that matter for reproducing a run - also
# quoted verbatim in the CLI's run-parameters report
TFIDF_ANALYZER = "char_wb"
TFIDF_NGRAM_RANGE = (3, 5)
DBSCAN_MIN_SAMPLES = 2
KMEANS_RANDOM_STATE = 0
KMEANS_N_INIT = 10
KMEANS_MAX_K = 20
RELEVANCE_MIN_LENGTH = 12


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
    vec = TfidfVectorizer(analyzer=TFIDF_ANALYZER, ngram_range=TFIDF_NGRAM_RANGE)
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
    # which algorithm(s) produced this cluster - only populated by
    # find_clusters_all, empty for a single-algorithm run
    algos: tuple[str, ...] = ()
    # weakest link in the cluster - lets a reader verify every member
    # actually clears cluster_threshold without recomputing anything
    min_similarity: float = 0.0


def _min_pairwise_sim(members: list[int], sim) -> float:
    return min(sim[i, j] for i, j in itertools.combinations(members, 2))


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


def _cluster_indices(indices: list[int], codes: list[str], sim, cluster_threshold: float) -> list[ClusterCandidate]:
    uf = {i: i for i in indices}

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            uf[ra] = rb

    for pos, i in enumerate(indices):
        for j in indices[pos + 1:]:
            if sim[i, j] > cluster_threshold and not is_parent_child(codes[i], codes[j]):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in indices:
        groups.setdefault(find(i), []).append(i)

    clusters = [
        ClusterCandidate(members, len(members) >= CLUSTER_WARNING_SIZE, min_similarity=_min_pairwise_sim(members, sim))
        for members in groups.values()
        if len(members) >= 2
    ]
    clusters.sort(key=lambda c: c.members[0])
    return clusters


def find_clusters(codes: list[str], sim, cluster_threshold: float) -> list[ClusterCandidate]:
    return _cluster_indices(list(range(len(codes))), codes, sim, cluster_threshold)


def find_clusters_dbscan(codes: list[str], sim, cluster_threshold: float) -> list[ClusterCandidate]:
    """DBSCAN over a precomputed cosine-distance matrix - eps is derived from
    cluster_threshold so a pair counts as neighbors under the same rule
    union-find uses (sim > cluster_threshold). Parent/child pairs get their
    distance forced to max so they're never direct neighbors, same role
    is_parent_child plays in find_clusters - including the same transitive-
    chaining exception (two parent/child codes can still land in one cluster
    through a third code both are close to)."""
    n = len(codes)
    distance = np.clip(1 - sim, 0, None)
    np.fill_diagonal(distance, 0)
    for i in range(n):
        for j in range(i + 1, n):
            if is_parent_child(codes[i], codes[j]):
                distance[i, j] = distance[j, i] = 1.0

    labels = DBSCAN(eps=1 - cluster_threshold, min_samples=DBSCAN_MIN_SAMPLES, metric="precomputed").fit_predict(distance)

    groups: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        if label != -1:
            groups.setdefault(label, []).append(i)

    clusters = [
        ClusterCandidate(members, len(members) >= CLUSTER_WARNING_SIZE, min_similarity=_min_pairwise_sim(members, sim))
        for members in groups.values()
    ]
    clusters.sort(key=lambda c: c.members[0])
    return clusters


def find_clusters_kmeans(
    codes: list[str], sim, X, cluster_threshold: float, info: dict | None = None
) -> list[ClusterCandidate]:
    """K-means is a flat partitioner - it has no notion of "not a duplicate"
    and forces every code into some cluster, which doesn't match this tool's
    "only report actual near-duplicates" semantics. So it's used only to
    bucket codes into candidate neighborhoods (k picked by best silhouette
    score); the existing threshold/parent-child union-find rule then decides
    what's actually reported within each bucket, same as find_clusters.

    k and its silhouette score are data-dependent - not otherwise visible
    anywhere - so callers that need to report exactly how a run was produced
    can pass a dict via `info` to get them back."""
    n = len(codes)
    if n < 4:
        if info is not None:
            info["k"] = None
            info["silhouette_score"] = None
            info["k_search_range"] = None
        return []

    distance = np.clip(1 - sim, 0, None)
    np.fill_diagonal(distance, 0)

    max_k = min(n - 1, KMEANS_MAX_K)
    best_k, best_score = 2, -1.0
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT).fit_predict(X)
        score = silhouette_score(distance, labels, metric="precomputed")
        if score > best_score:
            best_k, best_score = k, score

    if info is not None:
        info["k"] = best_k
        info["silhouette_score"] = best_score
        info["k_search_range"] = (2, max_k)

    labels = KMeans(n_clusters=best_k, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT).fit_predict(X)
    partitions: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        partitions.setdefault(label, []).append(i)

    clusters = []
    for indices in partitions.values():
        clusters.extend(_cluster_indices(indices, codes, sim, cluster_threshold))
    clusters.sort(key=lambda c: c.members[0])
    return clusters


# single source of truth for the algorithm names used both in CLI choices
# and as find_clusters_all's per-cluster provenance tags
CLUSTER_ALGORITHMS = ("unionfind", "dbscan", "kmeans")


def find_clusters_all(
    codes: list[str], sim, X, cluster_threshold: float, info: dict | None = None
) -> list[ClusterCandidate]:
    """Run every clustering algorithm and merge the results into one list.
    Clusters with an identical member set (common - unionfind and dbscan
    frequently agree) are reported once, tagged with every algorithm that
    found them; clusters unique to one algorithm keep their single tag."""
    kmeans_info: dict = {} if info is not None else None
    by_algo = {
        "unionfind": find_clusters(codes, sim, cluster_threshold),
        "dbscan": find_clusters_dbscan(codes, sim, cluster_threshold),
        "kmeans": find_clusters_kmeans(codes, sim, X, cluster_threshold, info=kmeans_info),
    }
    if info is not None:
        info["kmeans"] = kmeans_info

    merged: dict[frozenset[int], ClusterCandidate] = {}
    for algo, clusters in by_algo.items():
        for c in clusters:
            key = frozenset(c.members)
            if key in merged:
                merged[key].algos += (algo,)
            else:
                merged[key] = ClusterCandidate(c.members, c.chaining_warning, (algo,), min_similarity=c.min_similarity)

    result = list(merged.values())
    result.sort(key=lambda c: c.members[0])
    return result


@dataclass
class RelevanceFlag:
    index: int
    reason: str


def scan_relevance(
    codes: list[str],
    rq_texts: list[str],
    threshold: float = 0.1,
    min_length: int = RELEVANCE_MIN_LENGTH,
) -> list[RelevanceFlag]:
    """Flag codes for human review instead of auto-deciding relevance - a
    plain keyword-absence filter was tried before this and threw too many
    false positives on real codebooks."""
    vec = TfidfVectorizer(analyzer=TFIDF_ANALYZER, ngram_range=TFIDF_NGRAM_RANGE)
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
