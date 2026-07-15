import io

from rich.console import Console

from pykarsin.io_csv import CodeRecord
from pykarsin.similarity import PairCandidate, ClusterCandidate, RelevanceFlag
from pykarsin.report import (
    build_report,
    build_run_parameters,
    render_report,
    render_dry_run_stats,
    export_csv,
    export_markdown,
    export_html,
)

RECORDS = [
    CodeRecord(row=2, code="Foo A", comment=""),
    CodeRecord(row=3, code="Foo B", comment=""),
    CodeRecord(row=4, code="Bar A", comment=""),
    CodeRecord(row=5, code="Bar B", comment=""),
    CodeRecord(row=6, code="Bar C", comment=""),
]


def test_build_report_drops_two_member_cluster_already_covered_by_a_pair():
    pairs = [PairCandidate(score=0.9, i=0, j=1, polarity_flag=False)]
    clusters = [
        ClusterCandidate(members=[0, 1], chaining_warning=False),   # redundant with the pair above
        ClusterCandidate(members=[2, 3, 4], chaining_warning=True),  # kept, not reducible to one pair
    ]
    report = build_report(pairs, clusters)
    assert report.high == pairs
    assert report.check_polarity == []
    assert len(report.clusters) == 1
    assert report.clusters[0].members == [2, 3, 4]


def test_build_report_splits_polarity_flagged_pairs():
    pairs = [
        PairCandidate(score=0.9, i=0, j=1, polarity_flag=False),
        PairCandidate(score=0.8, i=2, j=3, polarity_flag=True),
    ]
    report = build_report(pairs, [])
    assert len(report.high) == 1
    assert len(report.check_polarity) == 1


def test_render_dry_run_stats_included_count_reflects_include_merged():
    merged = [RECORDS[0]]
    active = RECORDS[1:]

    console = Console(file=io.StringIO(), width=200)
    render_dry_run_stats(console, RECORDS, active, merged, active)
    line = next(l for l in console.file.getvalue().splitlines() if "Included in this scan" in l)
    assert str(len(active)) in line

    console = Console(file=io.StringIO(), width=200)
    render_dry_run_stats(console, RECORDS, active, merged, RECORDS)
    line = next(l for l in console.file.getvalue().splitlines() if "Included in this scan" in l)
    assert str(len(RECORDS)) in line


def test_render_report_does_not_raise():
    console = Console(file=io.StringIO())
    report = build_report(
        [PairCandidate(score=0.9, i=0, j=1, polarity_flag=False)],
        [ClusterCandidate(members=[2, 3, 4], chaining_warning=True)],
    )
    render_report(console, RECORDS, report)


def test_export_csv_contains_expected_rows(tmp_path):
    report = build_report(
        [PairCandidate(score=0.9, i=0, j=1, polarity_flag=False)],
        [ClusterCandidate(members=[2, 3, 4], chaining_warning=True)],
    )
    path = tmp_path / "out.csv"
    export_csv(str(path), RECORDS, report)
    text = path.read_text(encoding="utf-8")
    assert "Foo A" in text and "Foo B" in text
    assert "Bar A" in text and "verify chaining" in text


def test_export_markdown_contains_expected_sections(tmp_path):
    report = build_report(
        [PairCandidate(score=0.9, i=0, j=1, polarity_flag=False)],
        [ClusterCandidate(members=[2, 3, 4], chaining_warning=True)],
    )
    path = tmp_path / "out.md"
    export_markdown(str(path), RECORDS, report)
    text = path.read_text(encoding="utf-8")
    assert "## High confidence" in text
    assert "## Related groups" in text
    assert "chaining artifact" in text


def test_export_html_contains_expected_sections(tmp_path):
    report = build_report(
        [PairCandidate(score=0.9, i=0, j=1, polarity_flag=False)],
        [ClusterCandidate(members=[2, 3, 4], chaining_warning=True)],
    )
    path = tmp_path / "out.html"
    export_html(str(path), RECORDS, report)
    text = path.read_text(encoding="utf-8")
    assert text.lstrip().lower().startswith("<!doctype html>")
    assert "Foo A" in text and "Foo B" in text
    assert "Bar A" in text
    assert "High confidence" in text
    assert "Related group" in text
    assert "chaining artifact" in text
    assert 'class="meter' in text
    assert 'class="table-wrap"' in text


def test_export_html_escapes_special_characters(tmp_path):
    records = [
        CodeRecord(row=2, code="A & B < C", comment=""),
        CodeRecord(row=3, code="Plain code", comment=""),
    ]
    report = build_report([PairCandidate(score=0.9, i=0, j=1, polarity_flag=False)], [])
    path = tmp_path / "out.html"
    export_html(str(path), records, report)
    text = path.read_text(encoding="utf-8")
    assert "A & B < C" not in text
    assert "A &amp; B &lt; C" in text


def test_export_html_empty_report_shows_no_items_message(tmp_path):
    report = build_report([], [])
    path = tmp_path / "out.html"
    export_html(str(path), RECORDS, report)
    text = path.read_text(encoding="utf-8")
    assert "No high-confidence duplicates found." in text
    assert "No polarity-conflict pairs found." in text
    assert "No related groups found." in text


def _sample_relevance_flags():
    return [RelevanceFlag(index=2, reason="too short for a reliable n-gram signal")]


def test_export_csv_includes_relevance_rows(tmp_path):
    report = build_report([], [])
    path = tmp_path / "out.csv"
    export_csv(str(path), RECORDS, report, relevance_flags=_sample_relevance_flags())
    text = path.read_text(encoding="utf-8")
    assert "relevance" in text
    assert "Bar A" in text and "too short for a reliable n-gram signal" in text


def test_export_markdown_includes_relevance_section(tmp_path):
    report = build_report([], [])
    path = tmp_path / "out.md"
    export_markdown(str(path), RECORDS, report, relevance_flags=_sample_relevance_flags())
    text = path.read_text(encoding="utf-8")
    assert "## Needs human read (relevance)" in text
    assert "Bar A" in text and "too short for a reliable n-gram signal" in text


def test_export_markdown_omits_relevance_section_when_not_scanned(tmp_path):
    report = build_report([], [])
    path = tmp_path / "out.md"
    export_markdown(str(path), RECORDS, report, relevance_flags=None)
    text = path.read_text(encoding="utf-8")
    assert "Needs human read" not in text


def test_export_html_includes_relevance_section(tmp_path):
    report = build_report([], [])
    path = tmp_path / "out.html"
    export_html(str(path), RECORDS, report, relevance_flags=_sample_relevance_flags())
    text = path.read_text(encoding="utf-8")
    assert "Needs human read" in text
    assert "Bar A" in text and "too short for a reliable n-gram signal" in text


def test_export_html_omits_relevance_section_when_not_scanned(tmp_path):
    report = build_report([], [])
    path = tmp_path / "out.html"
    export_html(str(path), RECORDS, report, relevance_flags=None)
    text = path.read_text(encoding="utf-8")
    assert 'class="relevance"' not in text  # legend still mentions relevance in general terms, but no section


def _base_param_kwargs(**overrides):
    kwargs = dict(
        pykarsin_version="0.1.0",
        python_version="3.13.0",
        numpy_version="2.0.0",
        sklearn_version="1.5.0",
        generated_at="2026-01-01T00:00:00+00:00",
        csv_path="book.csv",
        code_col="Code",
        comment_col="Comment",
        group_cols=["Code Group 1"],
        include_merged=False,
        total_codes=10,
        active_codes=9,
        merged_codes=1,
        included_codes=9,
        tfidf_analyzer="char_wb",
        tfidf_ngram_range=(3, 5),
        pair_threshold=0.6,
        cluster_threshold=0.5,
        cluster_algo="unionfind",
    )
    kwargs.update(overrides)
    return kwargs


def test_build_run_parameters_omits_algo_specific_rows_for_unionfind():
    params = dict(build_run_parameters(**_base_param_kwargs()))
    assert "clustering algorithm" in params
    assert "dbscan eps" not in params
    assert "kmeans chosen k" not in params
    assert "relevance percentile (bottom N% flagged)" not in params


def test_build_run_parameters_includes_dbscan_rows():
    params = dict(build_run_parameters(**_base_param_kwargs(cluster_algo="dbscan", dbscan_min_samples=2)))
    assert params["dbscan eps"] == "0.5"
    assert params["dbscan min_samples"] == "2"


def test_build_run_parameters_includes_kmeans_rows():
    kmeans_info = {"k": 4, "silhouette_score": 0.42, "k_search_range": (2, 8)}
    params = dict(build_run_parameters(**_base_param_kwargs(cluster_algo="kmeans", kmeans_info=kmeans_info)))
    assert params["kmeans chosen k"] == "4"
    assert params["kmeans silhouette score"] == "0.420"
    assert params["kmeans k search range"] == "2-8"


def test_build_run_parameters_includes_relevance_rows_when_scanned():
    params = dict(build_run_parameters(
        **_base_param_kwargs(relevance_percentile=10.0, relevance_min_length=12, relevance_path="rq.txt")
    ))
    assert params["relevance percentile (bottom N% flagged)"] == "10.0"
    assert params["relevance question file"] == "rq.txt"


def test_render_and_export_include_run_parameters_section(tmp_path):
    report = build_report([], [ClusterCandidate(members=[2, 3, 4], chaining_warning=True)])
    params = build_run_parameters(**_base_param_kwargs())

    console = Console(file=io.StringIO(), width=200)
    render_report(console, RECORDS, report, params=params)
    assert "Run parameters" in console.file.getvalue()

    md_path = tmp_path / "out.md"
    export_markdown(str(md_path), RECORDS, report, params=params)
    assert "## Run parameters" in md_path.read_text(encoding="utf-8")

    csv_path = tmp_path / "out.csv"
    export_csv(str(csv_path), RECORDS, report, params=params)
    assert "parameter" in csv_path.read_text(encoding="utf-8")

    html_path = tmp_path / "out.html"
    export_html(str(html_path), RECORDS, report, params=params)
    assert "Run parameters" in html_path.read_text(encoding="utf-8")


def _sample_by_algo():
    return {
        "unionfind": [ClusterCandidate(members=[2, 3], chaining_warning=False, min_similarity=0.7)],
        "dbscan": [ClusterCandidate(members=[2, 3], chaining_warning=False, min_similarity=0.7)],
        "kmeans": [],
    }


def test_render_report_includes_algo_comparison_when_by_algo_given():
    report = build_report([], [])
    console = Console(file=io.StringIO(), width=200)
    render_report(console, RECORDS, report, by_algo=_sample_by_algo())
    out = console.file.getvalue()
    # rich wraps long table titles across lines at this width, so check for
    # the words rather than the exact phrase (see test_scan_all_cluster_algos_reports_known_near_duplicates)
    assert "Algorithm" in out and "comparison" in out
    assert "unionfind" in out and "dbscan" in out


def test_render_report_omits_algo_comparison_when_by_algo_not_given():
    report = build_report([], [])
    console = Console(file=io.StringIO(), width=200)
    render_report(console, RECORDS, report)
    assert "Algorithm comparison" not in console.file.getvalue()


def test_export_csv_includes_algo_comparison_rows(tmp_path):
    report = build_report([], [])
    path = tmp_path / "out.csv"
    export_csv(str(path), RECORDS, report, by_algo=_sample_by_algo())
    text = path.read_text(encoding="utf-8")
    assert "algo_comparison" in text
    assert "Bar A" in text and "Bar B" in text


def test_export_csv_omits_algo_comparison_rows_when_not_given(tmp_path):
    report = build_report([], [])
    path = tmp_path / "out.csv"
    export_csv(str(path), RECORDS, report)
    assert "algo_comparison" not in path.read_text(encoding="utf-8")


def test_export_markdown_includes_algo_comparison_section(tmp_path):
    report = build_report([], [])
    path = tmp_path / "out.md"
    export_markdown(str(path), RECORDS, report, by_algo=_sample_by_algo())
    text = path.read_text(encoding="utf-8")
    assert "## Algorithm comparison" in text
    assert "### unionfind" in text and "### dbscan" in text


def test_export_markdown_omits_algo_comparison_section_when_not_given(tmp_path):
    report = build_report([], [])
    path = tmp_path / "out.md"
    export_markdown(str(path), RECORDS, report)
    assert "Algorithm comparison" not in path.read_text(encoding="utf-8")


def test_export_html_includes_algo_comparison_section(tmp_path):
    report = build_report([], [])
    path = tmp_path / "out.html"
    export_html(str(path), RECORDS, report, by_algo=_sample_by_algo())
    text = path.read_text(encoding="utf-8")
    assert 'class="comparison"' in text
    assert "Algorithm comparison" in text


def test_export_html_omits_algo_comparison_section_when_not_given(tmp_path):
    report = build_report([], [])
    path = tmp_path / "out.html"
    export_html(str(path), RECORDS, report)
    assert 'class="comparison"' not in path.read_text(encoding="utf-8")
