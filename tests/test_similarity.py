import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from school_lab import features
from school_lab.assignments import load_assignments
from school_lab.comparison import FeatureIndex, compare, jaccard, jaccard_reference
from school_lab.configuration import load_similarity_config
from school_lab.errors import AnalysisFailure, SubmissionError
from school_lab.features import SimilarityConfig, extract_features, feature_pipeline
from school_lab.models import Assignment
from school_lab.parsing import parse_submission

DATA = Path(__file__).resolve().parents[1] / "data" / "lab2"
ASSIGNMENT = Assignment("assignment-01", "from typing import Iterable\n", ("solve",))


@pytest.fixture
def record(raw_submission):
    return extract_features(parse_submission(raw_submission), ASSIGNMENT)


def other(record, **kwargs):
    return replace(record, submission_id="other", student_id="student-other", **kwargs)


@pytest.mark.parametrize("method", ["token", "ast"])
def test_identical_maximum(record, method):
    report = compare(record, other(record), method)
    assert report.score == 1.0
    assert report.review_required
    assert report.evidence


def test_formatting_and_comments_do_not_change_features(raw_submission, record):
    raw_submission["code"] = raw_submission["code"].replace("    ", "  ") + "# комментарий\n\n"
    updated = extract_features(parse_submission(raw_submission), ASSIGNMENT)
    assert record.token.locations.keys() == updated.token.locations.keys()
    assert record.ast.locations.keys() == updated.ast.locations.keys()


def test_renames_keep_ast_structure(raw_submission, record):
    raw_submission["code"] = raw_submission["code"].replace("values", "data").replace("x", "item")
    updated = other(extract_features(parse_submission(raw_submission), ASSIGNMENT))
    assert compare(record, updated, "ast").score == 1.0
    assert compare(record, updated, "token").score < 1.0


def test_independent_solution(raw_submission, record):
    raw_submission["code"] = "def solve(data):\n    total = 0\n    for number in data:\n        total += number ** 2\n    return total\n"
    updated = other(extract_features(parse_submission(raw_submission), ASSIGNMENT))
    assert compare(record, updated, "ast").score < 1.0
    assert compare(record, updated, "token").score < 1.0


@pytest.mark.parametrize("method", ["token", "ast"])
def test_template_only_is_insufficient(raw_submission, method):
    raw_submission["code"] = ASSIGNMENT.allowed_template
    a = extract_features(parse_submission(raw_submission), ASSIGNMENT)
    report = compare(a, other(a), method)
    assert report.score is None
    assert report.status == "insufficient_data"
    assert report.evidence == ()
    assert report.left_template_ranges == ((1, 1),)
    assert report.review_required


@pytest.mark.parametrize("source", ["", "\n", "# comment only\n"])
@pytest.mark.parametrize("method", ["token", "ast"])
def test_empty_code(raw_submission, source, method):
    raw_submission["code"] = source
    record = extract_features(parse_submission(raw_submission), ASSIGNMENT)
    assert compare(record, other(record), method).score is None


def test_empty_union_only():
    assert jaccard({}, {}) is None
    assert jaccard({("a",): (1, 1)}, {}) == 0.0


def test_syntax_error_is_explicit(raw_submission):
    raw_submission["code"] = "def solve(:\n    return 1\n"
    record = extract_features(parse_submission(raw_submission), ASSIGNMENT)
    report = compare(record, other(record), "ast")
    assert report.score is None
    assert "AST:" in report.errors[0]


def test_valid_tokens_but_invalid_ast(raw_submission):
    raw_submission["code"] = "return 1 +\n"
    record = extract_features(parse_submission(raw_submission), ASSIGNMENT)
    assert record.ast.error
    assert not record.token.error
    assert compare(record, other(record), "token").score == 1.0
    assert compare(record, other(record), "ast").score is None


@pytest.mark.parametrize("changes", [{"submission_id": "sub-001"},
                                    {"student_id": "student-001"},
                                    {"assignment_id": "other"}, {"config_hash": "stale"}])
def test_ineligible_pair_rejected(record, changes):
    candidate = replace(other(record), **changes)
    with pytest.raises(AnalysisFailure):
        compare(record, candidate)


@pytest.mark.parametrize("threshold", [-1, 2, float("nan"), float("inf"), True])
def test_invalid_threshold(record, threshold):
    with pytest.raises(AnalysisFailure):
        compare(record, other(record), threshold=threshold)


def test_top_k_excludes_self_assignment_and_versions(record):
    index = FeatureIndex()
    for item in (record, other(record), replace(record, submission_id="version-2"),
                 replace(other(record), submission_id="wrong", assignment_id="different")):
        index.add(item)
    result = index.top_k(record.submission_id, "ast", 0.5)
    assert [row.right_id for row in result] == ["other"]
    assert list(index.student_versions().values()) == [["sub-001", "version-2"]]
    assert len(list(index.pairs("ast", 0.5))) == 2


def test_index_conflict_and_duplicate(record):
    index = FeatureIndex()
    index.add(record)
    index.add(record)
    assert len(index.records) == 1
    with pytest.raises(AnalysisFailure):
        index.add(replace(record, submission_hash="changed"))


@pytest.mark.parametrize("method", ["token", "ast"])
def test_evidence_source_lines_and_template_exclusion(raw_submission, method):
    a = extract_features(parse_submission(raw_submission), ASSIGNMENT)
    b = other(a)
    report = compare(a, b, method)
    for evidence in report.evidence:
        assert evidence.submission_id in {a.submission_id, b.submission_id}
        assert 2 <= evidence.start_line <= evidence.end_line <= a.line_count


def test_new_template_changes_hash_and_scores(raw_submission):
    value = parse_submission(raw_submission)
    a = extract_features(value, ASSIGNMENT)
    b = extract_features(value, ASSIGNMENT, SimilarityConfig("2", (value.code,)))
    assert a.config_hash != b.config_hash
    assert compare(a, other(a), "ast").score == 1.0
    assert compare(b, other(b), "ast").score is None
    assert compare(b, other(b), "token").score is None
    assert compare(b, other(b)).analyzer_version == "2.0"


def test_pipeline_is_lazy_and_closable(raw_submission, tmp_path, monkeypatch):
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps(raw_submission) + "\n{broken\n", encoding="utf-8")
    calls = []
    original = features.parse_submission

    def spy(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(features, "parse_submission", spy)
    stream = feature_pipeline(path, {ASSIGNMENT.assignment_id: ASSIGNMENT})
    assert not calls
    assert next(stream).record
    assert len(calls) == 1
    assert next(stream).error.code == "invalid_json"
    stream.close()


def test_pipeline_continues_after_invalid_and_unknown(raw_submission, tmp_path):
    path = tmp_path / "input.jsonl"
    bad = dict(raw_submission, assignment_id="unknown")
    path.write_text("{}\n" + json.dumps(bad) + "\n" + json.dumps(raw_submission) + "\n", encoding="utf-8")
    rows = list(feature_pipeline(path, {ASSIGNMENT.assignment_id: ASSIGNMENT}))
    assert [item.error.code if item.error else None for item in rows] == ["schema_error", "unknown_assignment", None]


def test_jaccard_optimization_equivalence_symmetry_and_range():
    rng = random.Random(20261111)
    for _ in range(100):
        a = {(str(i),): (1, 1) for i in range(50) if rng.random() < 0.5}
        b = {(str(i),): (1, 1) for i in range(50) if rng.random() < 0.5}
        score = jaccard(a, b)
        assert score == jaccard_reference(a, b) == jaccard(b, a)
        assert score is None or 0 <= score <= 1


@pytest.mark.parametrize("method", ["token", "ast"])
def test_contract_unchanged_after_optimization(record, method):
    a = compare(record, other(record), method)
    b = compare(record, other(record), method, reference=True)
    assert a.score == b.score
    assert a.evidence == b.evidence
    assert a.explanation == b.explanation
    assert a.config_hash == b.config_hash
    assert a.review_required == b.review_required


def test_config_validation(tmp_path):
    config = load_similarity_config(DATA / "config.json")
    assert config == SimilarityConfig()
    path = tmp_path / "config.json"
    path.write_text('{"version":"1","extra_templates":42}', encoding="utf-8")
    with pytest.raises(SubmissionError):
        load_similarity_config(path)


def test_assignment_duplicate_rejected(tmp_path):
    path = tmp_path / "assignments.json"
    row = {"assignment_id": "x", "allowed_template": "", "requirements": []}
    path.write_text(json.dumps({"assignments": [row, row]}), encoding="utf-8")
    with pytest.raises(SubmissionError):
        load_assignments(path)
