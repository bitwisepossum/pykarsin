import csv
import io
from pathlib import Path

import pytest
from rich.console import Console

from pykarsin.cli import build_parser, run_scan

FIXTURE = str(Path(__file__).parent / "fixtures" / "example_codebook.csv")


def _run(args_list):
    args = build_parser().parse_args(args_list)
    buf = io.StringIO()
    console = Console(file=buf, width=200)
    code = run_scan(args, console)
    return code, buf.getvalue()


def _line_with(output, label):
    return next(line for line in output.splitlines() if label in line)


def _fixture_counts():
    with open(FIXTURE, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    data_rows = [r for r in rows[1:] if r[0].strip()]
    merged = [r for r in data_rows if "merged with" in r[1].lower()]
    return len(data_rows), len(merged)


def test_scan_yes_reports_known_near_duplicates():
    code, output = _run(["scan", FIXTURE, "--yes"])
    assert code == 0
    assert "jonotusaika" in output.lower()
    assert "verkkosivu" in output.lower()


def test_scan_dbscan_reports_known_near_duplicates():
    code, output = _run(["scan", FIXTURE, "--yes", "--cluster-algo", "dbscan"])
    assert code == 0
    assert "jonotusaika" in output.lower()
    assert "verkkosivu" in output.lower()


def test_scan_kmeans_reports_known_near_duplicates():
    code, output = _run(["scan", FIXTURE, "--yes", "--cluster-algo", "kmeans"])
    assert code == 0
    assert "jonotusaika" in output.lower()
    assert "verkkosivu" in output.lower()


def test_scan_all_cluster_algos_reports_known_near_duplicates():
    # console table titles get width-truncated by rich, so provenance text
    # isn't reliably checkable here - see the markdown export test below
    code, output = _run(["scan", FIXTURE, "--yes", "--cluster-algo", "all"])
    assert code == 0
    assert "jonotusaika" in output.lower()
    assert "verkkosivu" in output.lower()


def test_scan_all_cluster_algos_exports_provenance(tmp_path):
    out = tmp_path / "review.md"
    code, _ = _run(["scan", FIXTURE, "--yes", "--cluster-algo", "all", "--export", "markdown", "--export-path", str(out)])
    assert code == 0
    assert "found by:" in out.read_text(encoding="utf-8").lower()


def test_scan_all_cluster_algos_exports_run_parameters(tmp_path):
    out = tmp_path / "review.md"
    code, _ = _run(["scan", FIXTURE, "--yes", "--cluster-algo", "all", "--export", "markdown", "--export-path", str(out)])
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "## Run parameters" in text
    assert "kmeans silhouette score" in text
    assert "dbscan eps" in text
    assert "min pairwise similarity" in text


def test_scan_unionfind_run_parameters_omit_dbscan_and_kmeans_rows(tmp_path):
    out = tmp_path / "review.csv"
    code, _ = _run(["scan", FIXTURE, "--yes", "--cluster-algo", "unionfind", "--export", "csv", "--export-path", str(out)])
    assert code == 0
    text = out.read_text(encoding="utf-8").lower()
    assert "clustering algorithm" in text
    assert "dbscan eps" not in text
    assert "kmeans chosen k" not in text


def test_scan_all_cluster_algos_exports_raw_algorithm_comparison(tmp_path):
    md_path = tmp_path / "review.md"
    code, _ = _run(["scan", FIXTURE, "--yes", "--cluster-algo", "all", "--export", "markdown", "--export-path", str(md_path)])
    assert code == 0
    md_text = md_path.read_text(encoding="utf-8").lower()
    assert "## algorithm comparison" in md_text
    assert "### unionfind" in md_text
    assert "### dbscan" in md_text

    csv_path = tmp_path / "review.csv"
    code, _ = _run(["scan", FIXTURE, "--yes", "--cluster-algo", "all", "--export", "csv", "--export-path", str(csv_path)])
    assert code == 0
    assert "algo_comparison" in csv_path.read_text(encoding="utf-8").lower()


def test_scan_single_algo_export_omits_algorithm_comparison(tmp_path):
    out = tmp_path / "review.md"
    code, _ = _run(["scan", FIXTURE, "--yes", "--cluster-algo", "unionfind", "--export", "markdown", "--export-path", str(out)])
    assert code == 0
    assert "algorithm comparison" not in out.read_text(encoding="utf-8").lower()


def test_scan_dry_run_excludes_merged_rows_by_default():
    total, merged_count = _fixture_counts()
    active = total - merged_count
    code, output = _run(["scan", FIXTURE, "--yes", "--dry-run"])
    assert code == 0
    assert str(active) in _line_with(output, "Included in this scan")


def test_scan_include_merged_widens_the_included_count():
    total, _ = _fixture_counts()
    code, output = _run(["scan", FIXTURE, "--yes", "--include-merged", "--dry-run"])
    assert code == 0
    assert str(total) in _line_with(output, "Included in this scan")


def test_scan_export_markdown(tmp_path):
    out = tmp_path / "review.md"
    code, _ = _run(["scan", FIXTURE, "--yes", "--export", "markdown", "--export-path", str(out)])
    assert code == 0
    assert out.exists()
    assert "High confidence" in out.read_text(encoding="utf-8")


def test_scan_export_html(tmp_path):
    out = tmp_path / "review.html"
    code, _ = _run(["scan", FIXTURE, "--yes", "--export", "html", "--export-path", str(out)])
    assert code == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert text.lstrip().lower().startswith("<!doctype html>")
    assert "High confidence" in text


def test_scan_export_html_includes_relevance_results(tmp_path):
    rq_path = tmp_path / "rq.txt"
    rq_path.write_text("Miten asiakaspalvelun jonotusaika koetaan?\n", encoding="utf-8")
    out = tmp_path / "review.html"
    code, _ = _run([
        "scan", FIXTURE, "--yes",
        "--relevance", str(rq_path), "--relevance-threshold", "0.15",
        "--export", "html", "--export-path", str(out),
    ])
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "Needs human read" in text


def test_scan_export_all_writes_three_files(tmp_path):
    src = tmp_path / "book.csv"
    src.write_text(Path(FIXTURE).read_text(encoding="utf-8"), encoding="utf-8")
    code, _ = _run(["scan", str(src), "--yes", "--export", "all"])
    assert code == 0
    assert (tmp_path / "book_pykarsin_review.csv").exists()
    assert (tmp_path / "book_pykarsin_review.md").exists()
    assert (tmp_path / "book_pykarsin_review.html").exists()


def test_scan_export_dir_overrides_directory_for_single_format(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    code, _ = _run(["scan", FIXTURE, "--yes", "--export", "markdown", "--export-dir", str(out_dir)])
    assert code == 0
    expected = out_dir / "example_codebook_pykarsin_review.md"
    assert expected.exists()


def test_scan_group_cols_range_matches_prefix_auto_detect():
    prefix_code, prefix_output = _run(["scan", FIXTURE, "--yes", "--dry-run"])
    range_code, range_output = _run(["scan", FIXTURE, "--yes", "--dry-run", "--group-cols", "3-4"])
    assert prefix_code == range_code == 0
    assert _line_with(prefix_output, "Ungrouped") == _line_with(range_output, "Ungrouped")


def test_scan_export_path_with_all_raises():
    args = build_parser().parse_args(["scan", FIXTURE, "--yes", "--export", "all", "--export-path", "review.csv"])
    console = Console(file=io.StringIO(), width=200)
    with pytest.raises(ValueError, match="--export-path"):
        run_scan(args, console)


def _dispatch_by_substring(answers):
    """rich.prompt.Prompt/Confirm are the *same* class object imported into both
    cli.py and columns.py, so patching one module's `.ask` patches both - dispatch
    on the prompt text itself instead of call order."""

    def f(prompt_text, *a, **k):
        for key, value in answers.items():
            if key in prompt_text:
                return value
        raise AssertionError(f"unexpected prompt: {prompt_text!r}")

    return f


def test_scan_interactive_prompts_accept_blank_defaults(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("pykarsin.cli.Confirm.ask", _dispatch_by_substring({
        "look correct": True, "merged": False,
    }))
    monkeypatch.setattr("pykarsin.cli.FloatPrompt.ask", lambda prompt, default=None, **k: default)
    monkeypatch.setattr("pykarsin.cli.Prompt.ask", _dispatch_by_substring({
        "Header prefix": "code group", "Research question": "", "Export report": "none",
        "Clustering algorithm": "unionfind",
    }))

    code, output = _run(["scan", FIXTURE])
    assert code == 0
    assert "jonotusaika" in output.lower()


def test_scan_interactive_export_all_choice(monkeypatch, tmp_path):
    src = tmp_path / "book.csv"
    src.write_text(Path(FIXTURE).read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("pykarsin.cli.Confirm.ask", _dispatch_by_substring({
        "look correct": True, "merged": False,
    }))
    monkeypatch.setattr("pykarsin.cli.FloatPrompt.ask", lambda prompt, default=None, **k: default)
    monkeypatch.setattr("pykarsin.cli.Prompt.ask", _dispatch_by_substring({
        "Header prefix": "code group", "Research question": "", "Export report": "all", "Output directory": "",
        "Clustering algorithm": "unionfind",
    }))

    code, _ = _run(["scan", str(src)])
    assert code == 0
    assert (tmp_path / "book_pykarsin_review.csv").exists()
    assert (tmp_path / "book_pykarsin_review.md").exists()
    assert (tmp_path / "book_pykarsin_review.html").exists()


def test_scan_interactive_does_not_reprompt_explicit_flags(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    # "look correct" (columns) still needs answering; "merged"/thresholds were passed
    # explicitly on the command line and must never be prompted for
    monkeypatch.setattr("pykarsin.cli.Confirm.ask", _dispatch_by_substring({"look correct": True}))
    monkeypatch.setattr(
        "pykarsin.cli.FloatPrompt.ask",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not prompt for an explicitly-passed threshold")),
    )
    monkeypatch.setattr("pykarsin.cli.Prompt.ask", _dispatch_by_substring({
        "Header prefix": "code group", "Research question": "", "Export report": "none",
        "Clustering algorithm": "unionfind",
    }))

    code, _ = _run([
        "scan", FIXTURE, "--include-merged", "--pair-threshold", "0.6", "--cluster-threshold", "0.5",
    ])
    assert code == 0
