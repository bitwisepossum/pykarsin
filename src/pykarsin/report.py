"""Rich console rendering and csv/markdown/html export of a scan report.

Output is a review queue, not a deletion list - nothing here writes back to
the source CSV.
"""
from __future__ import annotations

import csv as csv_mod
import html as html_mod
from dataclasses import dataclass

from rich.table import Table

from pykarsin.similarity import ClusterCandidate, PairCandidate, RelevanceFlag


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


def _render_clusters(console, records, clusters) -> None:
    for idx, c in enumerate(clusters, start=1):
        title = f"Related group {idx} ({len(c.members)} codes)"
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


def render_report(console, records, report: Report, relevance_flags: list[RelevanceFlag] | None = None) -> None:
    _render_pair_table(console, "High confidence duplicates", records, report.high)
    _render_pair_table(console, "Check polarity", records, report.check_polarity)
    _render_clusters(console, records, report.clusters)
    if relevance_flags:
        render_relevance_flags(console, records, relevance_flags)


# --- csv export ---

def _write_pair_row(writer, records, p: PairCandidate, tier: str) -> None:
    a, b = records[p.i], records[p.j]
    writer.writerow([tier, a.row, a.code, b.row, b.code, f"{p.score:.3f}", ""])


def export_csv(path: str, records, report: Report, relevance_flags: list[RelevanceFlag] | None = None) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.writer(f)
        writer.writerow(["tier", "row", "code", "paired_row", "paired_code", "score", "note"])
        for p in report.high:
            _write_pair_row(writer, records, p, "high")
        for p in report.check_polarity:
            _write_pair_row(writer, records, p, "check_polarity")
        for idx, c in enumerate(report.clusters, start=1):
            note = f"related group {idx}" + (" - verify chaining" if c.chaining_warning else "")
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


def export_markdown(path: str, records, report: Report, relevance_flags: list[RelevanceFlag] | None = None) -> None:
    lines = ["# Duplicate review", ""]
    lines += _markdown_pair_section("High confidence", records, report.high)
    lines += _markdown_pair_section("Check polarity", records, report.check_polarity)
    lines.append("## Related groups")
    lines.append("")
    for idx, c in enumerate(report.clusters, start=1):
        warn = " (verify this isn't a chaining artifact)" if c.chaining_warning else ""
        lines.append(f"### Group {idx}{warn}")
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
section h2 { margin-top: 0; }
section h3 { margin-bottom: 0.25rem; }
table { width: 100%; border-collapse: collapse; margin-top: 0.75rem; }
th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #ddd; }
.empty { color: #666; font-style: italic; }
.warning { color: #7a4b00; font-size: 0.9em; }
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


def _html_clusters_section(records, clusters) -> str:
    lines = ['<section class="groups">', "<h2>Related groups</h2>"]
    if not clusters:
        lines.append('<p class="empty">No related groups found.</p>')
    for idx, c in enumerate(clusters, start=1):
        lines.append(f"<h3>Related group {idx} ({len(c.members)} codes)</h3>")
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


def export_html(path: str, records, report: Report, relevance_flags: list[RelevanceFlag] | None = None) -> None:
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
