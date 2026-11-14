import json
from pathlib import Path

from school_lab.cli import main

DATA = Path(__file__).resolve().parents[1] / "data" / "lab2"


def test_analyze_end_to_end_and_top_k(tmp_path, capsys):
    assert main(["analyze", str(DATA / "submissions.jsonl"), "--assignments", str(DATA / "assignments.json"),
                 "--submission-id", "family-01-0", "--top-k", "3", "--output", str(tmp_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["reports"] == 6
    rows = [json.loads(line) for line in (tmp_path / "pairs.jsonl").read_text().splitlines()]
    assert all(row["left_id"] != row["right_id"] and row["assignment_id"] == "squares" for row in rows)
    assert all(row["right_id"] != "family-01-v2" for row in rows)


def test_evaluate_end_to_end(tmp_path, capsys):
    assert main(["evaluate", str(DATA / "submissions.jsonl"), "--assignments", str(DATA / "assignments.json"),
                 "--provenance", str(DATA / "provenance.jsonl"), "--pairs", str(DATA / "pairs.jsonl"),
                 "--output", str(tmp_path)]) == 0
    assert "token" in json.loads(capsys.readouterr().out)["methods"]
    assert (tmp_path / "quality.json").exists()


def test_unavailable_input(tmp_path, capsys):
    assert main(["import", str(tmp_path / "absent.jsonl")]) == 2
    assert json.loads(capsys.readouterr().err)["code"] == "source_unavailable"


def test_unknown_candidate(tmp_path, capsys):
    assert main(["analyze", str(DATA / "submissions.jsonl"), "--assignments", str(DATA / "assignments.json"),
                 "--submission-id", "absent", "--output", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().err)["code"] == "invalid_input"
