import io

from rich.console import Console

from pykarsin.io_csv import CodeRecord
from pykarsin.similarity import PairCandidate, ClusterCandidate, RelevanceFlag
from pykarsin.report import build_report, render_report, render_dry_run_stats, export_csv, export_markdown, export_html

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
