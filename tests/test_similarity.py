import pytest

from pykarsin.similarity import (
    build_tfidf_matrix,
    cosine_sim_matrix,
    find_pairs,
    find_clusters,
    find_clusters_all,
    find_clusters_dbscan,
    find_clusters_kmeans,
    is_parent_child,
    has_negation,
    check_polarity,
    scan_relevance,
)

CODES = [
    "Asiakas kokee jonotusajan liian pitkäksi",                     # 0
    "Asiakas kokee, että jonotusaika on liian pitkä",                # 1  near-dup of 0
    "Asiakaspalvelu: Jonotusaika koetaan kohtuuttomaksi",            # 2  child of 3
    "Asiakaspalvelu",                                                # 3  parent
    "Asiakaspalvelu: Työntekijä oli epäystävällinen",                # 4  child of 3
    "Asiakaspalvelu: Työntekijä ei osannut vastata kysymykseen",     # 5  child of 3
    "Pelkää että valitus johtaa huonompaan kohteluun",               # 6  feared harm
    "Kokee joutuneensa huonompaan kohteluun valituksen jälkeen",     # 7  experienced harm, not the same as 6
    "Verkkosivun ulkoasu koettiin sekavaksi",                        # 8  near-dup of 9
    "Verkkosivuston käyttöliittymä koettiin sekavaksi",              # 9
]


def test_parent_child_excludes_hierarchy_not_near_duplicates():
    assert is_parent_child(CODES[2], CODES[3])
    assert is_parent_child(CODES[3], CODES[4])
    assert not is_parent_child(CODES[0], CODES[1])


def test_negation_marker_detection():
    assert has_negation("Työntekijä ei osannut vastata")
    assert not has_negation("Työntekijä osasi vastata")
    assert check_polarity("Työntekijä ei osannut vastata", "Työntekijä osasi vastata")
    assert not check_polarity("Työntekijä ei osannut vastata", "Työntekijä ei osaa mitään")


def test_pairs_and_clusters_at_production_defaults():
    sim = cosine_sim_matrix(build_tfidf_matrix(CODES))

    pairs = find_pairs(CODES, sim, 0.6)
    assert pairs == []  # nothing in this small set clears the high-precision bar

    clusters = find_clusters(CODES, sim, 0.5)
    member_sets = [set(c.members) for c in clusters]
    assert {0, 1} in member_sets
    assert {8, 9} in member_sets
    assert not any({2, 3}.issubset(m) for m in member_sets)  # parent/child never clusters


def test_dbscan_clusters_at_production_defaults():
    sim = cosine_sim_matrix(build_tfidf_matrix(CODES))

    clusters = find_clusters_dbscan(CODES, sim, 0.5)
    member_sets = [set(c.members) for c in clusters]
    assert {0, 1} in member_sets
    assert {8, 9} in member_sets
    assert not any({2, 3}.issubset(m) for m in member_sets)  # parent/child never clusters


def test_kmeans_clusters_at_production_defaults():
    sim = cosine_sim_matrix(build_tfidf_matrix(CODES))
    X = build_tfidf_matrix(CODES)

    clusters = find_clusters_kmeans(CODES, sim, X, 0.5)
    member_sets = [set(c.members) for c in clusters]
    assert {0, 1} in member_sets
    assert {8, 9} in member_sets
    assert not any({2, 3}.issubset(m) for m in member_sets)  # parent/child never clusters


def test_find_clusters_all_merges_matching_clusters_and_tags_algos():
    sim = cosine_sim_matrix(build_tfidf_matrix(CODES))
    X = build_tfidf_matrix(CODES)

    clusters = find_clusters_all(CODES, sim, X, 0.5)
    by_members = {frozenset(c.members): c for c in clusters}

    assert frozenset({0, 1}) in by_members
    assert frozenset({8, 9}) in by_members
    assert not any({2, 3}.issubset(m) for m in by_members)  # parent/child never clusters

    # unionfind and dbscan use identical threshold semantics, so on this
    # fixture they should agree and get merged into one tagged entry
    dup_cluster = by_members[frozenset({0, 1})]
    assert "unionfind" in dup_cluster.algos
    assert "dbscan" in dup_cluster.algos


def test_min_similarity_reflects_the_weakest_link_in_a_cluster():
    sim = cosine_sim_matrix(build_tfidf_matrix(CODES))

    for clusters in (
        find_clusters(CODES, sim, 0.5),
        find_clusters_dbscan(CODES, sim, 0.5),
    ):
        by_members = {frozenset(c.members): c for c in clusters}
        dup_cluster = by_members[frozenset({0, 1})]
        assert dup_cluster.min_similarity == pytest.approx(sim[0, 1])


def test_find_clusters_kmeans_reports_k_and_silhouette_via_info():
    sim = cosine_sim_matrix(build_tfidf_matrix(CODES))
    X = build_tfidf_matrix(CODES)

    info = {}
    find_clusters_kmeans(CODES, sim, X, 0.5, info=info)
    assert 2 <= info["k"] <= min(len(CODES) - 1, 20)
    assert -1.0 <= info["silhouette_score"] <= 1.0
    assert info["k_search_range"] == (2, min(len(CODES) - 1, 20))


def test_find_clusters_kmeans_info_is_none_for_too_few_codes():
    codes = CODES[:3]
    sim = cosine_sim_matrix(build_tfidf_matrix(codes))
    X = build_tfidf_matrix(codes)

    info = {}
    clusters = find_clusters_kmeans(codes, sim, X, 0.5, info=info)
    assert clusters == []
    assert info["k"] is None
    assert info["silhouette_score"] is None
    assert info["k_search_range"] is None


def test_find_clusters_all_forwards_kmeans_info():
    sim = cosine_sim_matrix(build_tfidf_matrix(CODES))
    X = build_tfidf_matrix(CODES)

    info = {}
    find_clusters_all(CODES, sim, X, 0.5, info=info)
    assert "kmeans" in info
    assert info["kmeans"]["k"] is not None


def test_feared_vs_experienced_harm_not_flagged_at_production_thresholds():
    # 6 and 7 share surface wording but differ in tense/aspect (anticipated vs
    # experienced); at production thresholds they just miss the similarity
    # bar, so no duplicate is reported and no special-casing is needed
    sim = cosine_sim_matrix(build_tfidf_matrix(CODES))
    assert sim[6, 7] < 0.5


def test_polarity_flag_on_a_negated_pair():
    a, b = "Asiakas ei koe saavansa apua", "Asiakas kokee saavansa apua"
    sim = cosine_sim_matrix(build_tfidf_matrix([a, b]))
    pairs = find_pairs([a, b], sim, 0.3)
    assert len(pairs) == 1
    assert pairs[0].polarity_flag


def test_large_cluster_gets_chaining_warning():
    codes = [f"Yhteinen kantasana {i}" for i in range(5)]
    sim = cosine_sim_matrix(build_tfidf_matrix(codes))
    clusters = find_clusters(codes, sim, 0.1)
    assert len(clusters) == 1
    assert clusters[0].chaining_warning


def test_large_cluster_gets_chaining_warning_dbscan():
    codes = [f"Yhteinen kantasana {i}" for i in range(5)]
    sim = cosine_sim_matrix(build_tfidf_matrix(codes))
    clusters = find_clusters_dbscan(codes, sim, 0.1)
    assert len(clusters) == 1
    assert clusters[0].chaining_warning


def test_large_cluster_gets_chaining_warning_kmeans():
    codes = [f"Yhteinen kantasana {i}" for i in range(5)]
    sim = cosine_sim_matrix(build_tfidf_matrix(codes))
    X = build_tfidf_matrix(codes)
    clusters = find_clusters_kmeans(codes, sim, X, 0.1)
    assert len(clusters) == 1
    assert clusters[0].chaining_warning


def test_scan_relevance_flags_short_and_off_topic_codes():
    codes = [
        "Asiakas kokee jonotusajan liian pitkäksi",
        "x",
        "Täysin erillinen ja epäolennainen huomio säästä",
    ]
    rqs = ["Miten asiakaspalvelun jonotusaika koetaan?"]
    flags = {f.index: f.reason for f in scan_relevance(codes, rqs, threshold=0.2, min_length=5)}
    assert 1 in flags  # too short
    assert 2 in flags  # low similarity to the RQ
    assert 0 not in flags
