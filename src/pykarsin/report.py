"""Rich console rendering and csv/markdown/html export of a scan report.

Output is a review queue, not a deletion list - nothing here writes back to
the source CSV.
"""
from __future__ import annotations

import csv as csv_mod
import html as html_mod
from dataclasses import dataclass

from rich.table import Table

from pykarsin.similarity import KMEANS_N_INIT, KMEANS_RANDOM_STATE, ClusterCandidate, PairCandidate, RelevanceFlag


@dataclass
class Report:
    high: list[PairCandidate]
    check_polarity: list[PairCandidate]
    clusters: list[ClusterCandidate]


def build_report(pairs: list[PairCandidate], clusters: list[ClusterCandidate]) -> Report:
    high = [p for p in pairs if not p.polarity_flag]
    check_polarity = [p for p in pairs if p.polarity_flag]

    # a 2-member cluster that already cleared the pairwise threshold is the
    # same finding as a pair - don't show it twice
    pair_member_sets = {frozenset((p.i, p.j)) for p in pairs}
    clusters = [c for c in clusters if not (len(c.members) == 2 and frozenset(c.members) in pair_member_sets)]

    return Report(high, check_polarity, clusters)


def build_run_parameters(
    *,
    pykarsin_version: str,
    python_version: str,
    numpy_version: str,
    sklearn_version: str,
    generated_at: str,
    csv_path: str,
    code_col: str,
    comment_col: str | None,
    group_cols: list[str],
    include_merged: bool,
    total_codes: int,
    active_codes: int,
    merged_codes: int,
    included_codes: int,
    tfidf_analyzer: str,
    tfidf_ngram_range: tuple[int, int],
    pair_threshold: float,
    cluster_threshold: float,
    cluster_algo: str,
    dbscan_min_samples: int | None = None,
    kmeans_info: dict | None = None,
    relevance_threshold: float | None = None,
    relevance_min_length: int | None = None,
    relevance_path: str | None = None,
) -> list[tuple[str, str]]:
    """Flat (label, value) pairs, in report order - a run's exported file is
    only scientifically replicable if it records exactly what produced it,
    so this is meant to cover every setting that can change the result."""
    params = [
        ("pykarsin version", pykarsin_version),
        ("python version", python_version),
        ("numpy version", numpy_version),
        ("scikit-learn version", sklearn_version),
        ("generated at (UTC)", generated_at),
        ("input CSV", csv_path),
        ("code column", code_col),
        ("comment column", comment_col or "(none)"),
        ("group columns", ", ".join(group_cols) if group_cols else "(none)"),
        ("include merged rows", str(include_merged)),
        ("total codes", str(total_codes)),
        ("active codes", str(active_codes)),
        ("merged / superseded codes", str(merged_codes)),
        ("codes included in this scan", str(included_codes)),
        ("TF-IDF analyzer", tfidf_analyzer),
        ("TF-IDF n-gram range", f"{tfidf_ngram_range[0]}-{tfidf_ngram_range[1]}"),
        ("pairwise similarity threshold", str(pair_threshold)),
        ("cluster similarity threshold", str(cluster_threshold)),
        ("clustering algorithm", cluster_algo),
    ]

    if cluster_algo in ("dbscan", "all") and dbscan_min_samples is not None:
        params.append(("dbscan eps", str(round(1 - cluster_threshold, 6))))
        params.append(("dbscan min_samples", str(dbscan_min_samples)))

    if cluster_algo in ("kmeans", "all") and kmeans_info is not None:
        k = kmeans_info.get("k")
        score = kmeans_info.get("silhouette_score")
        k_range = kmeans_info.get("k_search_range")
        params.append(("kmeans random_state", str(KMEANS_RANDOM_STATE)))
        params.append(("kmeans n_init", str(KMEANS_N_INIT)))
        params.append(("kmeans k search range", f"{k_range[0]}-{k_range[1]}" if k_range else "(too few codes to search)"))
        params.append(("kmeans chosen k", str(k) if k is not None else "(too few codes to cluster)"))
        params.append(("kmeans silhouette score", f"{score:.3f}" if score is not None else "(n/a)"))

    if relevance_threshold is not None:
        params.append(("relevance threshold", str(relevance_threshold)))
        params.append(("relevance min length", str(relevance_min_length)))
        params.append(("relevance question file", relevance_path or "(none)"))

    return params


def render_run_parameters(console, params: list[tuple[str, str]]) -> None:
    table = Table(title="Run parameters")
    table.add_column("Parameter")
    table.add_column("Value")
    for label, value in params:
        table.add_row(label, value)
    console.print(table)


def render_dry_run_stats(console, all_records, active_records, merged_records, included_records) -> None:
    table = Table(title="Corpus stats")
    table.add_column("Metric")
    table.add_column("Count")
    table.add_row("Total codes", str(len(all_records)))
    table.add_row("Active codes", str(len(active_records)))
    table.add_row("Merged / superseded", str(len(merged_records)))
    # differs from "Active codes" only when --include-merged pulls merged rows back in
    table.add_row("Included in this scan", str(len(included_records)))
    table.add_row("Ungrouped (active)", str(sum(1 for r in active_records if not r.groups)))
    console.print(table)


def _render_pair_table(console, title, records, pairs) -> None:
    if not pairs:
        return
    table = Table(title=title)
    table.add_column("Score")
    table.add_column("Row A")
    table.add_column("Code A")
    table.add_column("Row B")
    table.add_column("Code B")
    for p in pairs:
        a, b = records[p.i], records[p.j]
        table.add_row(f"{p.score:.3f}", str(a.row), a.code, str(b.row), b.code)
    console.print(table)


def _algo_label(c: ClusterCandidate) -> str:
    return f" [found by: {', '.join(sorted(c.algos))}]" if c.algos else ""


def _render_clusters(console, records, clusters) -> None:
    for idx, c in enumerate(clusters, start=1):
        title = f"Related group {idx} ({len(c.members)} codes, min pairwise similarity {c.min_similarity:.3f}){_algo_label(c)}"
        if c.chaining_warning:
            title += " - verify this isn't a chaining artifact from a shared stem"
        table = Table(title=title)
        table.add_column("Row")
        table.add_column("Code")
        for m in c.members:
            table.add_row(str(records[m].row), records[m].code)
        console.print(table)


def render_relevance_flags(console, records, flags: list[RelevanceFlag]) -> None:
    if not flags:
        return
    table = Table(title="Needs human read (relevance)")
    table.add_column("Row")
    table.add_column("Code")
    table.add_column("Reason")
    for f in flags:
        r = records[f.index]
        table.add_row(str(r.row), r.code, f.reason)
    console.print(table)


def render_report(
    console, records, report: Report,
    relevance_flags: list[RelevanceFlag] | None = None,
    params: list[tuple[str, str]] | None = None,
) -> None:
    if params:
        render_run_parameters(console, params)
    _render_pair_table(console, "High confidence duplicates", records, report.high)
    _render_pair_table(console, "Check polarity", records, report.check_polarity)
    _render_clusters(console, records, report.clusters)
    if relevance_flags:
        render_relevance_flags(console, records, relevance_flags)


# --- csv export ---

def _write_pair_row(writer, records, p: PairCandidate, tier: str) -> None:
    a, b = records[p.i], records[p.j]
    writer.writerow([tier, a.row, a.code, b.row, b.code, f"{p.score:.3f}", ""])


def export_csv(
    path: str, records, report: Report,
    relevance_flags: list[RelevanceFlag] | None = None,
    params: list[tuple[str, str]] | None = None,
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.writer(f)
        writer.writerow(["tier", "row", "code", "paired_row", "paired_code", "score", "note"])
        if params:
            for label, value in params:
                writer.writerow(["parameter", "", label, "", "", "", value])
        for p in report.high:
            _write_pair_row(writer, records, p, "high")
        for p in report.check_polarity:
            _write_pair_row(writer, records, p, "check_polarity")
        for idx, c in enumerate(report.clusters, start=1):
            note = f"related group {idx}" + (" - verify chaining" if c.chaining_warning else "")
            if c.algos:
                note += f" (found by: {', '.join(sorted(c.algos))})"
            note += f" (min pairwise similarity: {c.min_similarity:.3f})"
            for m in c.members:
                writer.writerow(["medium", records[m].row, records[m].code, "", "", "", note])
        if relevance_flags:
            for fl in relevance_flags:
                r = records[fl.index]
                writer.writerow(["relevance", r.row, r.code, "", "", "", fl.reason])


# --- markdown export ---

def _markdown_pair_section(title: str, records, pairs) -> list[str]:
    lines = [f"## {title}", ""]
    if not pairs:
        lines += ["(none)", ""]
        return lines
    lines += ["| Score | Row A | Code A | Row B | Code B |", "|---|---|---|---|---|"]
    for p in pairs:
        a, b = records[p.i], records[p.j]
        lines.append(f"| {p.score:.3f} | {a.row} | {a.code} | {b.row} | {b.code} |")
    lines.append("")
    return lines


def _markdown_relevance_section(records, flags: list[RelevanceFlag]) -> list[str]:
    lines = ["## Needs human read (relevance)", ""]
    if not flags:
        lines += ["(none)", ""]
        return lines
    lines += ["| Row | Code | Reason |", "|---|---|---|"]
    for f in flags:
        r = records[f.index]
        lines.append(f"| {r.row} | {r.code} | {f.reason} |")
    lines.append("")
    return lines


def _markdown_parameters_section(params: list[tuple[str, str]]) -> list[str]:
    lines = ["## Run parameters", "", "| Parameter | Value |", "|---|---|"]
    for label, value in params:
        lines.append(f"| {label} | {value} |")
    lines.append("")
    return lines


def export_markdown(
    path: str, records, report: Report,
    relevance_flags: list[RelevanceFlag] | None = None,
    params: list[tuple[str, str]] | None = None,
) -> None:
    lines = ["# Duplicate review", ""]
    if params:
        lines += _markdown_parameters_section(params)
    lines += _markdown_pair_section("High confidence", records, report.high)
    lines += _markdown_pair_section("Check polarity", records, report.check_polarity)
    lines.append("## Related groups")
    lines.append("")
    for idx, c in enumerate(report.clusters, start=1):
        warn = " (verify this isn't a chaining artifact)" if c.chaining_warning else ""
        algo_note = f" _(found by: {', '.join(sorted(c.algos))})_" if c.algos else ""
        lines.append(f"### Group {idx}{warn}{algo_note}")
        lines.append(f"min pairwise similarity: {c.min_similarity:.3f}")
        lines.append("")
        for m in c.members:
            lines.append(f"- row {records[m].row}: {records[m].code}")
        lines.append("")
    if relevance_flags is not None:
        lines += _markdown_relevance_section(records, relevance_flags)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --- html export ---
# plain string-building, same approach as markdown above - no templating
# engine, so every piece of user-supplied text must go through _esc()

def _esc(value) -> str:
    return html_mod.escape(str(value))


_HTML_STYLE = """
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; max-width: 900px;
       margin: 2rem auto; padding: 0 1rem; line-height: 1.5; color: #222; font-size: 16px; }
h1 { margin-bottom: 0.2rem; }
.subtitle { color: #666; margin-top: 0; }
.legend { background: #f5f5f5; border: 1px solid #ddd; border-radius: 6px; padding: 1rem 1.25rem; margin: 1.5rem 0; }
.legend p { margin: 0.35rem 0; }
section { margin: 2rem 0; padding: 1rem 1.25rem; border-radius: 6px; border-left: 4px solid #999; }
section.high { background: #fdecea; border-left-color: #c0392b; }
section.polarity { background: #fff8e6; border-left-color: #b7950b; }
section.groups { background: #eaf2fb; border-left-color: #2471a3; }
section.relevance { background: #f0f0f0; border-left-color: #666; }
section.meta { background: #f5f5f5; border-left-color: #888; }
section h2 { margin-top: 0; }
section h3 { margin-bottom: 0.25rem; }
table { width: 100%; border-collapse: collapse; margin-top: 0.75rem; }
th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #ddd; }
.empty { color: #666; font-style: italic; }
.warning { color: #7a4b00; font-size: 0.9em; }
.meta { color: #666; font-size: 0.9em; }
footer { color: #888; font-size: 0.85em; margin-top: 3rem; border-top: 1px solid #ddd; padding-top: 1rem; }
"""


def _html_pair_section(title: str, css_class: str, records, pairs, empty_msg: str) -> str:
    lines = [f'<section class="{css_class}">', f"<h2>{_esc(title)}</h2>"]
    if not pairs:
        lines.append(f'<p class="empty">{_esc(empty_msg)}</p>')
    else:
        lines.append("<table><tr><th>Score</th><th>Row A</th><th>Code A</th><th>Row B</th><th>Code B</th></tr>")
        for p in pairs:
            a, b = records[p.i], records[p.j]
            lines.append(
                f"<tr><td>{p.score:.3f}</td><td>{a.row}</td><td>{_esc(a.code)}</td>"
                f"<td>{b.row}</td><td>{_esc(b.code)}</td></tr>"
            )
        lines.append("</table>")
    lines.append("</section>")
    return "\n".join(lines)


def _html_parameters_section(params: list[tuple[str, str]]) -> str:
    lines = ['<section class="meta">', "<h2>Run parameters</h2>", "<table><tr><th>Parameter</th><th>Value</th></tr>"]
    for label, value in params:
        lines.append(f"<tr><td>{_esc(label)}</td><td>{_esc(value)}</td></tr>")
    lines.append("</table>")
    lines.append("</section>")
    return "\n".join(lines)


def _html_clusters_section(records, clusters) -> str:
    lines = ['<section class="groups">', "<h2>Related groups</h2>"]
    if not clusters:
        lines.append('<p class="empty">No related groups found.</p>')
    for idx, c in enumerate(clusters, start=1):
        lines.append(f"<h3>Related group {idx} ({len(c.members)} codes)</h3>")
        lines.append(f'<p class="meta">Min pairwise similarity: {c.min_similarity:.3f}</p>')
        if c.algos:
            lines.append(f'<p class="meta">Found by: {_esc(", ".join(sorted(c.algos)))}</p>')
        if c.chaining_warning:
            lines.append('<p class="warning">Verify this isn&#8217;t a chaining artifact from a shared word stem.</p>')
        lines.append("<ul>")
        for m in c.members:
            lines.append(f"<li>row {records[m].row}: {_esc(records[m].code)}</li>")
        lines.append("</ul>")
    lines.append("</section>")
    return "\n".join(lines)


def _html_relevance_section(records, flags: list[RelevanceFlag]) -> str:
    lines = ['<section class="relevance">', "<h2>Needs human read (relevance)</h2>"]
    if not flags:
        lines.append('<p class="empty">No relevance flags - every code matched the research questions reasonably well.</p>')
    else:
        lines.append("<table><tr><th>Row</th><th>Code</th><th>Reason</th></tr>")
        for f in flags:
            r = records[f.index]
            lines.append(f"<tr><td>{r.row}</td><td>{_esc(r.code)}</td><td>{_esc(f.reason)}</td></tr>")
        lines.append("</table>")
    lines.append("</section>")
    return "\n".join(lines)


def export_html(
    path: str, records, report: Report,
    relevance_flags: list[RelevanceFlag] | None = None,
    params: list[tuple[str, str]] | None = None,
) -> None:
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>pykarsin duplicate review</title>",
        f"<style>{_HTML_STYLE}</style>",
        "</head>",
        "<body>",
        "<h1>Duplicate review</h1>",
        f'<p class="subtitle">{len(records)} codes scanned.</p>',
    ]
    if params:
        parts.append(_html_parameters_section(params))
    parts += [
        '<div class="legend">',
        "<p><strong>High confidence:</strong> these code pairs look like near-duplicates. Consider merging.</p>",
        "<p><strong>Check polarity:</strong> worded almost the same, but one side may have the opposite "
        "meaning (e.g. a negation). Read both before deciding.</p>",
        "<p><strong>Related groups:</strong> three or more codes that loosely resemble each other. Not every "
        "member is necessarily a duplicate of every other - read each one.</p>",
        "<p><strong>Needs human read:</strong> codes that don't closely match your research questions, or are "
        "very short.</p>",
        "<p>This tool compares wording, not meaning. Always check the original excerpts before merging or "
        "deleting anything.</p>",
        "</div>",
        _html_pair_section("High confidence", "high", records, report.high, "No high-confidence duplicates found."),
        _html_pair_section("Check polarity", "polarity", records, report.check_polarity, "No polarity-conflict pairs found."),
        _html_clusters_section(records, report.clusters),
    ]
    if relevance_flags is not None:
        parts.append(_html_relevance_section(records, relevance_flags))
    parts += [
        "<footer>Generated by pykarsin - review manually before merging or deleting codebook entries.</footer>",
        "</body>",
        "</html>",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")
