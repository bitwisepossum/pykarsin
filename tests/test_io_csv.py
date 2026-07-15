from pykarsin.io_csv import read_csv_rows, build_records, split_merged


def test_read_csv_rows_comma(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("Code,Comment\nFoo,\nBar,merged with C01\n", encoding="utf-8")
    headers, rows = read_csv_rows(str(p))
    assert headers == ["Code", "Comment"]
    assert rows == [["Foo", ""], ["Bar", "merged with C01"]]


def test_read_csv_rows_semicolon_and_bom(tmp_path):
    p = tmp_path / "a.csv"
    p.write_bytes("﻿Code;Comment\nFoo;\n".encode("utf-8"))
    headers, rows = read_csv_rows(str(p))
    assert headers == ["Code", "Comment"]  # BOM stripped, not glued to "Code"
    assert rows == [["Foo", ""]]


def test_build_records_skips_blank_rows_and_numbers_from_two():
    rows = [["Foo", "c1"], ["", "c2"], ["Bar", "c3"]]
    records = build_records(rows, code_idx=0, comment_idx=1, group_idxs=[])
    assert [r.row for r in records] == [2, 4]  # row 3 (blank code) skipped
    assert [r.code for r in records] == ["Foo", "Bar"]


def test_build_records_collects_nonempty_group_columns():
    rows = [["Foo", "", "G1", ""]]
    records = build_records(rows, code_idx=0, comment_idx=1, group_idxs=[2, 3])
    assert records[0].groups == ["G1"]


def test_split_merged_excludes_by_default():
    rows = [["Foo", ""], ["Bar", "Merged With C01"]]
    records = build_records(rows, code_idx=0, comment_idx=1, group_idxs=[])
    active, merged = split_merged(records)
    assert [r.code for r in active] == ["Foo"]
    assert [r.code for r in merged] == ["Bar"]
