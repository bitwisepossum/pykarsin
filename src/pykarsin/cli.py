"""CLI entry point: parse args, wire the modules together, print/export the report."""
from __future__ import annotations

import argparse
import os
import platform
import sys
from datetime import datetime, timezone

import numpy
import sklearn
from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, Prompt

from pykarsin import __version__
from pykarsin.columns import confirm_columns, guess_columns, parse_column_range, resolve_column
from pykarsin.io_csv import build_records, read_csv_rows, split_merged
from pykarsin.report import (
    build_report,
    build_run_parameters,
    export_csv,
    export_html,
    export_markdown,
    render_dry_run_stats,
    render_report,
)
from pykarsin.similarity import (
    CLUSTER_ALGORITHMS,
    DBSCAN_MIN_SAMPLES,
    RELEVANCE_MIN_LENGTH,
    TFIDF_ANALYZER,
    TFIDF_NGRAM_RANGE,
    build_tfidf_matrix,
    cosine_sim_matrix,
    find_clusters,
    find_clusters_all,
    find_clusters_dbscan,
    find_clusters_kmeans,
    find_pairs,
    scan_relevance,
)

EXPORT_SUFFIX = {"csv": "csv", "markdown": "md", "html": "html"}
EXPORTERS = {"csv": export_csv, "markdown": export_markdown, "html": export_html}
CLUSTER_ALGOS = [*CLUSTER_ALGORITHMS, "all"]

# hardcoded fallbacks applied after the interactive-prompt decision - see the
# sentinel-default argparse args below
DEFAULT_PAIR_THRESHOLD = 0.6
DEFAULT_CLUSTER_THRESHOLD = 0.5
DEFAULT_RELEVANCE_THRESHOLD = 0.1
DEFAULT_GROUP_COL_PREFIX = "code group"
DEFAULT_CLUSTER_ALGO = "unionfind"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pykarsin")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="scan a codebook CSV for duplicates and relevance issues")
    scan.add_argument("csv_path")
    # these default to None (rather than their real default) so run_scan can tell
    # "not passed on the command line" apart from "passed the default value" -
    # only the former gets prompted for interactively
    scan.add_argument("--pair-threshold", type=float, default=None)
    scan.add_argument("--cluster-threshold", type=float, default=None)
    scan.add_argument("--cluster-algo", choices=CLUSTER_ALGOS, default=None)
    scan.add_argument(
        "--include-merged", action="store_true", default=None,
        help="include rows marked 'merged with ...' in the scan",
    )
    scan.add_argument("--relevance", metavar="PATH", help="text file with 1-3 research questions, one per line")
    scan.add_argument("--relevance-threshold", type=float, default=None)
    scan.add_argument("--export", choices=["csv", "markdown", "html", "all"])
    scan.add_argument("--export-path", metavar="PATH", help="explicit output file (single export format only)")
    scan.add_argument("--export-dir", metavar="PATH", help="output directory (default: same folder as csv_path)")
    scan.add_argument("--code-col", metavar="NAME")
    scan.add_argument("--comment-col", metavar="NAME")
    scan.add_argument("--group-col-prefix", default=None)
    scan.add_argument(
        "--group-cols", metavar="RANGE",
        help="explicit group columns, e.g. 3-8 or 3,5,9 (overrides --group-col-prefix auto-detect)",
    )
    scan.add_argument("-y", "--yes", action="store_true", help="skip interactive column confirmation")
    scan.add_argument("--dry-run", action="store_true", help="print corpus stats and exit before running the scan")
    return parser


def _resolve_float(console: Console, prompt_text: str, current: float | None, default: float, *, interactive: bool) -> float:
    if current is not None:
        return current
    if interactive:
        return FloatPrompt.ask(prompt_text, default=default, console=console)
    return default


def _resolve_choice(
    console: Console, prompt_text: str, current: str | None, choices: list[str], default: str, *, interactive: bool
) -> str:
    if current is not None:
        return current
    if interactive:
        return Prompt.ask(prompt_text, choices=choices, default=default, console=console)
    return default


def _export_target(csv_path: str, export_dir: str | None, fmt: str) -> str:
    stem = os.path.basename(csv_path).rsplit(".", 1)[0]
    directory = export_dir or os.path.dirname(csv_path) or "."
    return os.path.join(directory, f"{stem}_pykarsin_review.{EXPORT_SUFFIX[fmt]}")


def run_scan(args: argparse.Namespace, console: Console) -> int:
    headers, data_rows = read_csv_rows(args.csv_path)

    interactive = not args.yes and sys.stdin.isatty()

    if interactive and args.group_col_prefix is None:
        args.group_col_prefix = Prompt.ask(
            "Header prefix for group columns", default=DEFAULT_GROUP_COL_PREFIX, console=console
        )
    group_col_prefix = args.group_col_prefix or DEFAULT_GROUP_COL_PREFIX

    guess = guess_columns(headers, group_col_prefix)
    if args.code_col:
        guess.code_idx = resolve_column(headers, args.code_col)
    if args.comment_col:
        guess.comment_idx = resolve_column(headers, args.comment_col)
    if args.group_cols:
        guess.group_idxs = parse_column_range(args.group_cols, headers)

    sample_row = data_rows[0] if data_rows else []
    mapping = confirm_columns(console, headers, sample_row, guess, interactive=interactive)

    if interactive and args.include_merged is None:
        args.include_merged = Confirm.ask(
            "Include merged/superseded rows in this scan?", default=False, console=console
        )
    include_merged = bool(args.include_merged)

    all_records = build_records(data_rows, mapping.code_idx, mapping.comment_idx, mapping.group_idxs)
    active, merged = split_merged(all_records)
    records = all_records if include_merged else active

    render_dry_run_stats(console, all_records, active, merged, records)
    if args.dry_run:
        return 0

    cluster_algo = _resolve_choice(
        console, "Clustering algorithm", args.cluster_algo, CLUSTER_ALGOS, DEFAULT_CLUSTER_ALGO, interactive=interactive
    )
    pair_threshold = _resolve_float(
        console, "Pairwise similarity threshold", args.pair_threshold, DEFAULT_PAIR_THRESHOLD, interactive=interactive
    )
    cluster_threshold = _resolve_float(
        console, "Cluster similarity threshold", args.cluster_threshold, DEFAULT_CLUSTER_THRESHOLD, interactive=interactive
    )
    console.print(f"Clustering algorithm: {cluster_algo}")

    codes = [r.code for r in records]
    X = build_tfidf_matrix(codes)
    sim = cosine_sim_matrix(X)
    pairs = find_pairs(codes, sim, pair_threshold)
    kmeans_info: dict | None = {} if cluster_algo in ("kmeans", "all") else None
    by_algo = None
    if cluster_algo == "all":
        clusters = find_clusters_all(codes, sim, X, cluster_threshold, info=kmeans_info)
        by_algo = kmeans_info.get("by_algo")
        kmeans_info = kmeans_info.get("kmeans")
    elif cluster_algo == "dbscan":
        clusters = find_clusters_dbscan(codes, sim, cluster_threshold)
    elif cluster_algo == "kmeans":
        clusters = find_clusters_kmeans(codes, sim, X, cluster_threshold, info=kmeans_info)
    else:
        clusters = find_clusters(codes, sim, cluster_threshold)
    report = build_report(pairs, clusters)

    relevance_path = args.relevance
    if interactive and relevance_path is None:
        relevance_path = Prompt.ask(
            "Research question file to check relevance against (blank to skip)", default="", console=console
        ) or None

    relevance_flags = None
    relevance_threshold = None
    if relevance_path:
        relevance_threshold = _resolve_float(
            console, "Relevance similarity threshold", args.relevance_threshold,
            DEFAULT_RELEVANCE_THRESHOLD, interactive=interactive,
        )
        with open(relevance_path, encoding="utf-8") as f:
            rq_texts = [line.strip() for line in f if line.strip()]
        relevance_flags = scan_relevance(codes, rq_texts, relevance_threshold)

    params = build_run_parameters(
        pykarsin_version=__version__,
        python_version=platform.python_version(),
        numpy_version=numpy.__version__,
        sklearn_version=sklearn.__version__,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        csv_path=args.csv_path,
        code_col=headers[mapping.code_idx],
        comment_col=headers[mapping.comment_idx] if mapping.comment_idx is not None else None,
        group_cols=[headers[i] for i in mapping.group_idxs],
        include_merged=include_merged,
        total_codes=len(all_records),
        active_codes=len(active),
        merged_codes=len(merged),
        included_codes=len(records),
        tfidf_analyzer=TFIDF_ANALYZER,
        tfidf_ngram_range=TFIDF_NGRAM_RANGE,
        pair_threshold=pair_threshold,
        cluster_threshold=cluster_threshold,
        cluster_algo=cluster_algo,
        dbscan_min_samples=DBSCAN_MIN_SAMPLES,
        kmeans_info=kmeans_info,
        relevance_threshold=relevance_threshold,
        relevance_min_length=RELEVANCE_MIN_LENGTH if relevance_threshold is not None else None,
        relevance_path=relevance_path,
    )

    render_report(console, records, report, relevance_flags, params, by_algo)

    export_format = args.export
    if interactive and export_format is None:
        choice = Prompt.ask(
            "Export report as", choices=["none", "csv", "markdown", "html", "all"], default="none", console=console
        )
        export_format = None if choice == "none" else choice

    if export_format == "all" and args.export_path:
        raise ValueError("--export-path can't be used with --export all (one path, three files) - use --export-dir instead.")

    if export_format:
        export_dir = args.export_dir
        if interactive and export_dir is None and not args.export_path:
            export_dir = Prompt.ask(
                "Output directory (blank = same folder as input CSV)", default="", console=console
            ) or None

        if export_format == "all":
            for fmt in ("csv", "markdown", "html"):
                path = _export_target(args.csv_path, export_dir, fmt)
                EXPORTERS[fmt](path, records, report, relevance_flags, params, by_algo)
                console.print(f"Exported to {path}")
        else:
            path = args.export_path or _export_target(args.csv_path, export_dir, export_format)
            EXPORTERS[export_format](path, records, report, relevance_flags, params, by_algo)
            console.print(f"Exported to {path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    try:
        return run_scan(args, console)
    except (ValueError, FileNotFoundError, OSError) as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
