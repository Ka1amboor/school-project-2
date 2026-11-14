import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from school_lab.assignments import load_assignments
from school_lab.comparison import compare
from school_lab.errors import AnalysisFailure
from school_lab.evaluation import (
    build_index,
    evaluate,
    metrics,
    read_jsonl,
    select_threshold,
    validate_corpus,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "lab2"


@pytest.fixture
def corpus():
    index = build_index(DATA / "submissions.jsonl", load_assignments(DATA / "assignments.json"))
    pairs, audit = validate_corpus(index, DATA / "provenance.jsonl", DATA / "pairs.jsonl")
    return index, pairs, audit


def test_corpus_requirements_and_split(corpus):
    index, pairs, audit = corpus
    assert len(index.records) >= 30 and len(pairs) >= 40
    assert sum(pair.template_only for pair in pairs) >= 5
    assert audit["family_leakage"] is False
    assert audit["student_versions"]
    provenance = read_jsonl(DATA / "provenance.jsonl")
    tune = {row["family_id"] for row in provenance if row["split"] == "tune"}
    evaluation = {row["family_id"] for row in provenance if row["split"] == "evaluation"}
    assert not tune & evaluation


def test_family_leakage_is_rejected(corpus, tmp_path):
    index, _, _ = corpus
    rows = read_jsonl(DATA / "provenance.jsonl")
    rows[1]["split"] = "evaluation"
    path = tmp_path / "leak.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(AnalysisFailure, match="Утечка семейства"):
        validate_corpus(index, path, DATA / "pairs.jsonl")


def test_metrics_known_matrix_and_abstentions():
    result = metrics([(True, 1), (False, 1), (True, 0), (False, 0), (False, None)], 0.5)
    assert result["confusion_matrix"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert result["precision"] == result["recall"] == result["f1"] == 0.5
    assert result["undetermined_fraction"] == 0.2


def test_threshold_uses_only_tune(corpus, tmp_path):
    index, pairs, audit = corpus
    result = evaluate(index, pairs, audit, tmp_path)
    for method in ("token", "ast"):
        tune = [(pair.derivation_relation, compare(index.records[pair.left_id], index.records[pair.right_id], method).score)
                for pair in pairs if pair.split == "tune"]
        assert result["methods"][method]["threshold"] == select_threshold(tune)
    assert result == evaluate(index, pairs, audit, tmp_path)


def test_optimization_same_scores_and_evidence_on_all_labeled_pairs(corpus):
    index, pairs, _ = corpus
    for pair in pairs:
        for method in ("token", "ast"):
            a, b = index.records[pair.left_id], index.records[pair.right_id]
            before = compare(a, b, method, reference=True)
            after = compare(a, b, method)
            for field in ("report_id", "created_at"):
                before = replace(before, **{field: getattr(after, field)})
            assert asdict(before) == asdict(after)


def test_all_corpus_evidence_has_existing_lines(corpus):
    index, pairs, _ = corpus
    sources = {row["submission_id"]: row["code"] for row in read_jsonl(DATA / "submissions.jsonl")}
    for pair in pairs:
        for method in ("token", "ast"):
            report = compare(index.records[pair.left_id], index.records[pair.right_id], method)
            for evidence in report.evidence:
                assert 1 <= evidence.start_line <= evidence.end_line <= len(sources[evidence.submission_id].splitlines())
