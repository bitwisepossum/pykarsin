"""CSV reading: delimiter sniffing, encoding, merged-row split."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field

MERGED_MARKER = "merged with"


@dataclass
class CodeRecord:
    row: int  # 1-based CSV row number, header row is 1
    code: str
    comment: str
    groups: list[str] = field(default_factory=list)


def read_csv_rows(path: str) -> tuple[list[str], list[list[str]]]:
    """Sniff the delimiter - Finnish-locale Excel commonly exports
    semicolon-separated CSV, not comma - and strip a UTF-8 BOM if present."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(f, dialect))
    if not rows:
        raise ValueError(f"{path}: file has no rows")
    return rows[0], rows[1:]


def build_records(
    data_rows: list[list[str]],
    code_idx: int,
    comment_idx: int | None,
    group_idxs: list[int],
) -> list[CodeRecord]:
    records = []
    for offset, row in enumerate(data_rows):
        code = row[code_idx].strip() if code_idx < len(row) else ""
        if not code:
            continue  # blank trailing rows are common in exports
        comment = row[comment_idx].strip() if comment_idx is not None and comment_idx < len(row) else ""
        groups = [row[i].strip() for i in group_idxs if i < len(row) and row[i].strip()]
        records.append(CodeRecord(row=offset + 2, code=code, comment=comment, groups=groups))
    return records


def split_merged(records: list[CodeRecord]) -> tuple[list[CodeRecord], list[CodeRecord]]:
    """Return (active, merged) - a merged row is an already-superseded code."""
    active, merged = [], []
    for r in records:
        (merged if MERGED_MARKER in r.comment.lower() else active).append(r)
    return active, merged
