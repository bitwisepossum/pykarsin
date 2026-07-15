import io

import pytest
from rich.console import Console

from pykarsin.columns import guess_columns, parse_column_range, resolve_column, confirm_columns, ColumnGuess


def test_guess_columns_exact_match():
    g = guess_columns(["Code", "Comment", "Code Group 1", "Code Group 2"])
    assert g.code_idx == 0
    assert g.comment_idx == 1
    assert g.group_idxs == [2, 3]


def test_guess_columns_case_insensitive():
    g = guess_columns(["code", "COMMENT"])
    assert g.code_idx == 0
    assert g.comment_idx == 1


def test_guess_columns_missing_comment():
    g = guess_columns(["Code", "Notes"])
    assert g.code_idx == 0
    assert g.comment_idx is None


def test_guess_columns_unrelated_extra_column_ignored_not_misclassified():
    g = guess_columns(["Code", "Comment", "Notes"])
    assert g.group_idxs == []  # "Notes" is neither code, comment, nor a group column


def test_guess_columns_group_prefix_checked_before_code():
    g = guess_columns(["Code Group 1", "Code"])
    assert g.group_idxs == [0]
    assert g.code_idx == 1


def test_parse_column_range_single_token():
    assert parse_column_range("3", ["a", "b", "c", "d"]) == [2]


def test_parse_column_range_bounded_range():
    assert parse_column_range("2-4", ["a", "b", "c", "d", "e"]) == [1, 2, 3]


def test_parse_column_range_open_ended():
    assert parse_column_range("3-", ["a", "b", "c", "d", "e"]) == [2, 3, 4]


def test_parse_column_range_comma_mixed_and_dedup():
    assert parse_column_range("1, 3-5, 5", ["a", "b", "c", "d", "e", "f"]) == [0, 2, 3, 4]


def test_parse_column_range_blank_returns_empty():
    assert parse_column_range("", ["a", "b"]) == []


def test_parse_column_range_out_of_range_raises():
    with pytest.raises(ValueError):
        parse_column_range("5", ["a", "b"])


def test_parse_column_range_non_numeric_raises():
    with pytest.raises(ValueError):
        parse_column_range("x-y", ["a", "b"])


def test_parse_column_range_start_greater_than_end_raises():
    with pytest.raises(ValueError):
        parse_column_range("4-2", ["a", "b", "c", "d", "e"])


def test_resolve_column_found_and_missing():
    assert resolve_column(["Code", "Comment"], "comment") == 1
    with pytest.raises(ValueError):
        resolve_column(["Code", "Comment"], "nope")


def test_confirm_columns_noninteractive_returns_mapping_when_resolved():
    console = Console(file=io.StringIO())
    guess = ColumnGuess(code_idx=0, comment_idx=1, group_idxs=[2])
    mapping = confirm_columns(console, ["Code", "Comment", "Code Group 1"], ["Foo", "", "G1"], guess, interactive=False)
    assert (mapping.code_idx, mapping.comment_idx, mapping.group_idxs) == (0, 1, [2])


def test_confirm_columns_noninteractive_raises_when_code_unresolved():
    console = Console(file=io.StringIO())
    guess = ColumnGuess(code_idx=None, comment_idx=None, group_idxs=[])
    with pytest.raises(ValueError):
        confirm_columns(console, ["A", "B"], ["x", "y"], guess, interactive=False)


def test_confirm_columns_non_numeric_input_is_rejected_then_retried(monkeypatch):
    console = Console(file=io.StringIO())
    headers = ["A", "B", "Code", "D"]
    guess = ColumnGuess(code_idx=None, comment_idx=None, group_idxs=[])

    prompts = iter(["Code", "3", "", ""])  # Code: bad, then valid; Comment/Group: blank
    monkeypatch.setattr("pykarsin.columns.Prompt.ask", lambda *a, **k: next(prompts))
    monkeypatch.setattr("pykarsin.columns.Confirm.ask", lambda *a, **k: True)  # accept the remap once valid

    mapping = confirm_columns(console, headers, ["x", "y", "z", "w"], guess, interactive=True)
    assert (mapping.code_idx, mapping.comment_idx, mapping.group_idxs) == (2, None, [])


def test_confirm_columns_out_of_range_number_is_rejected_then_retried(monkeypatch):
    console = Console(file=io.StringIO())
    headers = ["A", "B", "Code", "D"]
    guess = ColumnGuess(code_idx=None, comment_idx=None, group_idxs=[])

    prompts = iter(["99", "3", "", ""])
    monkeypatch.setattr("pykarsin.columns.Prompt.ask", lambda *a, **k: next(prompts))
    monkeypatch.setattr("pykarsin.columns.Confirm.ask", lambda *a, **k: True)

    mapping = confirm_columns(console, headers, ["x", "y", "z", "w"], guess, interactive=True)
    assert mapping.code_idx == 2


def test_confirm_columns_blank_comment_and_multiple_groups(monkeypatch):
    console = Console(file=io.StringIO())
    headers = ["Code", "B", "C", "D"]
    guess = ColumnGuess(code_idx=None, comment_idx=None, group_idxs=[])

    prompts = iter(["1", "", "2,3"])
    monkeypatch.setattr("pykarsin.columns.Prompt.ask", lambda *a, **k: next(prompts))
    monkeypatch.setattr("pykarsin.columns.Confirm.ask", lambda *a, **k: True)

    mapping = confirm_columns(console, headers, ["x", "y", "z", "w"], guess, interactive=True)
    assert (mapping.code_idx, mapping.comment_idx, mapping.group_idxs) == (0, None, [1, 2])


def test_confirm_columns_group_range_input_expands_to_indices(monkeypatch):
    console = Console(file=io.StringIO())
    headers = ["Code", "B", "C", "D", "E", "F"]
    guess = ColumnGuess(code_idx=None, comment_idx=None, group_idxs=[])

    prompts = iter(["1", "", "2-4"])
    monkeypatch.setattr("pykarsin.columns.Prompt.ask", lambda *a, **k: next(prompts))
    monkeypatch.setattr("pykarsin.columns.Confirm.ask", lambda *a, **k: True)

    mapping = confirm_columns(console, headers, ["x"] * 6, guess, interactive=True)
    assert (mapping.code_idx, mapping.comment_idx, mapping.group_idxs) == (0, None, [1, 2, 3])


def test_confirm_columns_group_bad_range_input_is_rejected_then_retried(monkeypatch):
    console = Console(file=io.StringIO())
    headers = ["Code", "B", "C"]
    guess = ColumnGuess(code_idx=None, comment_idx=None, group_idxs=[])

    prompts = iter(["1", "", "9-10", "2"])  # Group: out of range, then valid
    monkeypatch.setattr("pykarsin.columns.Prompt.ask", lambda *a, **k: next(prompts))
    monkeypatch.setattr("pykarsin.columns.Confirm.ask", lambda *a, **k: True)

    mapping = confirm_columns(console, headers, ["x", "y", "z"], guess, interactive=True)
    assert mapping.group_idxs == [1]


def test_confirm_columns_code_idx_none_shows_explanation_and_skips_confirm_first(monkeypatch):
    buf = io.StringIO()
    console = Console(file=buf, width=200)
    headers = ["Code", "B"]
    guess = ColumnGuess(code_idx=None, comment_idx=None, group_idxs=[])

    calls = []
    prompts = iter(["1", "", ""])

    def record_confirm(*a, **k):
        calls.append("confirm")
        return True

    def record_prompt(*a, **k):
        calls.append("prompt")
        return next(prompts)

    monkeypatch.setattr("pykarsin.columns.Confirm.ask", record_confirm)
    monkeypatch.setattr("pykarsin.columns.Prompt.ask", record_prompt)

    confirm_columns(console, headers, ["x", "y"], guess, interactive=True)
    assert "Couldn't find a Code column automatically" in buf.getvalue()
    # code_idx starts None, so the yes/no confirm is skipped until a guess exists
    assert calls == ["prompt", "prompt", "prompt", "confirm"]


def test_confirm_columns_rejected_confirm_then_valid_remap(monkeypatch):
    console = Console(file=io.StringIO())
    headers = ["Wrong", "Code"]
    guess = ColumnGuess(code_idx=0, comment_idx=None, group_idxs=[])  # wrong guess, but resolved -> Confirm is asked

    confirms = iter([False, True])  # reject the wrong guess, accept the corrected one
    monkeypatch.setattr("pykarsin.columns.Confirm.ask", lambda *a, **k: next(confirms))
    prompts = iter(["2", "", ""])
    monkeypatch.setattr("pykarsin.columns.Prompt.ask", lambda *a, **k: next(prompts))

    mapping = confirm_columns(console, headers, ["x", "y"], guess, interactive=True)
    assert mapping.code_idx == 1
