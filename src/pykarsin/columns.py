"""Column auto-detection and interactive confirmation for a codebook CSV.

No assumption about which tool the CSV was exported from - just the generic
Code/Comment/Group-prefix header convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rich.prompt import Confirm, Prompt
from rich.table import Table


@dataclass
class ColumnGuess:
    code_idx: int | None
    comment_idx: int | None
    group_idxs: list[int] = field(default_factory=list)


@dataclass
class ColumnMapping:
    code_idx: int
    comment_idx: int | None
    group_idxs: list[int] = field(default_factory=list)


def guess_columns(headers: list[str], group_prefix: str = "code group") -> ColumnGuess:
    normalized = [h.strip().lower() for h in headers]
    prefix = group_prefix.strip().lower()

    # group-prefix check runs first so e.g. "Code Group 1" isn't taken for "Code"
    group_idxs = [i for i, h in enumerate(normalized) if h.startswith(prefix)]
    remaining = [i for i in range(len(headers)) if i not in group_idxs]

    code_idx = next((i for i in remaining if normalized[i] == "code"), None)
    comment_idx = next((i for i in remaining if normalized[i] in ("comment", "comments")), None)
    return ColumnGuess(code_idx, comment_idx, group_idxs)


def resolve_column(headers: list[str], name: str) -> int:
    normalized = [h.strip().lower() for h in headers]
    try:
        return normalized.index(name.strip().lower())
    except ValueError:
        raise ValueError(f"column {name!r} not found in headers: {headers}") from None


def parse_column_range(spec: str, headers: list[str]) -> list[int]:
    """Parse comma-separated 1-based column tokens into sorted 0-based
    indices. A token is a single column ("3"), a bounded range ("3-8"), or
    an open range to the last column ("3-") - lets wide per-slot group
    exports (e.g. Atlas.ti's "Code Group 1".."Code Group 50") be selected
    without typing every column number."""
    idxs: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            end_s = end_s.strip()
            try:
                start = int(start_s)
                end = int(end_s) if end_s else len(headers)
            except ValueError:
                raise ValueError(f"invalid column range {token!r}") from None
        else:
            try:
                start = end = int(token)
            except ValueError:
                raise ValueError(f"invalid column number {token!r}") from None
        if not (1 <= start <= end <= len(headers)):
            raise ValueError(f"column range {token!r} out of bounds for {len(headers)} columns")
        idxs.update(range(start - 1, end))
    return sorted(idxs)


def _role_of(i: int, guess: ColumnGuess) -> str:
    if i == guess.code_idx:
        return "Code"
    if i == guess.comment_idx:
        return "Comment"
    if i in guess.group_idxs:
        return "Group"
    return "(ignored)"


def render_preview(console, headers: list[str], sample_row: list[str], guess: ColumnGuess) -> None:
    table = Table(title="Column mapping preview")
    table.add_column("#")
    table.add_column("Header")
    table.add_column("Detected as")
    table.add_column("Sample")
    for i, header in enumerate(headers):
        sample = sample_row[i] if i < len(sample_row) else ""
        table.add_row(str(i + 1), header, _role_of(i, guess), sample)
    console.print(table)


def _parse_index(raw: str, headers: list[str]) -> int | None:
    """Return a 0-based index for a 1-based user-typed column number, or
    None if raw doesn't parse/isn't in range - never raises."""
    if not raw.strip():
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    idx = n - 1
    return idx if 0 <= idx < len(headers) else None


def _ask_column_index(console, prompt_text: str, headers: list[str], *, allow_blank: bool) -> int | None:
    while True:
        raw = Prompt.ask(prompt_text, default="")
        if not raw.strip():
            if allow_blank:
                return None
            console.print("Code column is required - please enter a number.")
            continue
        idx = _parse_index(raw, headers)
        if idx is None:
            console.print(f"Please enter a number between 1 and {len(headers)}.")
            continue
        return idx


def _ask_group_indices(console, prompt_text: str, headers: list[str]) -> list[int]:
    while True:
        raw = Prompt.ask(prompt_text, default="")
        if not raw.strip():
            return []
        try:
            return parse_column_range(raw, headers)
        except ValueError as e:
            console.print(f"{e} - please enter numbers or ranges like 3-8, separated by commas.")


def confirm_columns(
    console,
    headers: list[str],
    sample_row: list[str],
    guess: ColumnGuess,
    *,
    interactive: bool,
) -> ColumnMapping:
    while True:
        render_preview(console, headers, sample_row, guess)

        if guess.code_idx is None and not interactive:
            raise ValueError("Can't determine the Code column. Use --code-col to specify it.")
        if not interactive:
            return ColumnMapping(guess.code_idx, guess.comment_idx, guess.group_idxs)
        if guess.code_idx is not None and Confirm.ask("Does this look correct?", default=True):
            return ColumnMapping(guess.code_idx, guess.comment_idx, guess.group_idxs)

        if guess.code_idx is None:
            console.print("Couldn't find a Code column automatically - please tell me which column numbers to use.")

        code_idx = _ask_column_index(console, "Column number for Code", headers, allow_blank=False)
        comment_idx = _ask_column_index(console, "Column number for Comment (blank for none)", headers, allow_blank=True)
        group_idxs = _ask_group_indices(
            console, "Column numbers for Groups, e.g. 3-8 or 3,5,9 (blank for none)", headers
        )
        guess = ColumnGuess(code_idx, comment_idx, group_idxs)
