import csv
from types import SimpleNamespace

import pytest

import compare_tokenizers as cmp
import claude_tokens as ct


class FakeClient:
    def __init__(self):
        self.messages = self

    def count_tokens(self, **kwargs):
        return SimpleNamespace(input_tokens=len(kwargs["messages"][0]["content"]) // 4 + 7)


def test_texts_cover_the_required_kinds():
    assert list(cmp.TEXTS) == ["English", "Russian", "Code", "Mixed", "Long"]
    assert len(cmp.TEXTS["Long"]) > 10_000
    assert any(ord(c) > 0x400 for c in cmp.TEXTS["Mixed"]) and "tiktoken" in cmp.TEXTS["Mixed"]


def test_local_counts_for_openai_tokenizers():
    counts = cmp.count_local(names=["cl100k_base", "o200k_base"])
    assert set(counts) == {cmp.LABELS["cl100k_base"], cmp.LABELS["o200k_base"]}
    for column in counts.values():
        assert set(column) == set(cmp.TEXTS) and all(n > 0 for n in column.values())
    old, new = counts[cmp.LABELS["cl100k_base"]], counts[cmp.LABELS["o200k_base"]]
    assert new["Russian"] < old["Russian"]
    assert old["Long"] > 100 * old["English"] // 4  # the long text really is long


def test_claude_without_key_is_skipped_with_a_note(monkeypatch):
    monkeypatch.delenv(ct.KEY_VARIABLE, raising=False)
    counts, notes = cmp.count_claude()
    assert counts == {} and ct.KEY_VARIABLE in notes[0]


def test_claude_counts_use_the_client():
    counts, notes = cmp.count_claude(client=FakeClient())
    assert notes == [] and set(counts) == set(cmp.claude_column_labels())
    assert all(v > 0 for col in counts.values() for v in col.values())


def test_rows_leave_missing_columns_empty():
    columns = ["a", "claude-x"]
    rows = cmp.build_rows({"a": {k: 1 for k in cmp.TEXTS}}, columns)
    assert rows[0][:3] == ["English", len(cmp.TEXTS["English"]), 1] and rows[0][3] is None


def test_markdown_table_shape_and_missing_marker():
    rows = [["English", 1234, 10, None]]
    md = cmp.render_markdown(["a", "b"], rows, notes=["note"])
    assert "| Текст | Символов | a | b |" in md
    assert "| English | 1 234 | 10 | — |" in md
    assert "- note" in md


def test_main_writes_both_files_without_a_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(ct.KEY_VARIABLE, raising=False)
    monkeypatch.setattr(ct, "load_dotenv_if_available", lambda *a, **k: False)  # ignore a real .env
    md, csv_file = tmp_path / "r.md", tmp_path / "r.csv"
    assert cmp.main(md_file=md, csv_file=csv_file) == 0
    text = md.read_text(encoding="utf-8")
    assert "cl100k_base" in text and "claude-opus-5-5" in text and "—" in text
    rows = list(csv.reader(csv_file.open(encoding="utf-8-sig")))
    assert rows[0][:2] == ["text", "chars"] and len(rows) == 1 + len(cmp.TEXTS)
    assert len(rows[1]) == len(rows[0])
