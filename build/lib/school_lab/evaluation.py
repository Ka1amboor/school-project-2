"""Family-disjoint threshold selection and held-out educational evaluation."""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from .comparison import FeatureIndex, PairReport, compare
from .errors import AnalysisFailure
from .features import DEFAULT_CONFIG, METHODS, SimilarityConfig, feature_pipeline
from .hashing import stable_hash
from .models import Assignment
from .parsing import decode_json, require_bool, require_fields, require_string


@dataclass(frozen=True, slots=True)
class LabeledPair:
    left_id: str
    right_id: str
    split: str
    derivation_relation: bool
    template_only: bool
    reason: str


def read_jsonl(path: Path) -> list[dict[str, object]]:
    result = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            value = decode_json(line)
            if not isinstance(value, dict):
                raise AnalysisFailure(f"{path.name}:{line_no}: ожидается объект")
            result.append(value)
    return result


def validate_corpus(index: FeatureIndex, provenance_path: Path,
                    pairs_path: Path) -> tuple[list[LabeledPair], dict[str, object]]:
    provenance = read_jsonl(provenance_path)
    by_id: dict[str, dict[str, object]] = {}
    family_splits: dict[str, str] = {}
    for row in provenance:
        sid, family, split = row.get("submission_id"), row.get("family_id"), row.get("split")
        if not isinstance(sid, str) or sid not in index.records or sid in by_id:
            raise AnalysisFailure("Некорректный или повторный submission_id в происхождении")
        if not isinstance(family, str) or not family or split not in ("tune", "evaluation"):
            raise AnalysisFailure("Для каждого примера нужны family_id и корректный split")
        assert isinstance(split, str)
        if family in family_splits and family_splits[family] != split:
            raise AnalysisFailure(f"Утечка семейства {family} между tune и evaluation")
        if row.get("assignment_id") != index.records[sid].assignment_id:
            raise AnalysisFailure("Происхождение ссылается на другое задание")
        if not row.get("source") or not row.get("transformation"):
            raise AnalysisFailure("Источник и преобразование примера обязательны")
        family_splits[family] = split
        by_id[sid] = row
    if by_id.keys() != index.records.keys():
        raise AnalysisFailure("Не у всех сдач сохранено происхождение")
    for row in provenance:
        parent = row.get("parent_id")
        if parent is not None and (
            parent not in by_id or by_id[parent]["family_id"] != row["family_id"]
        ):
            raise AnalysisFailure("Родитель производного примера должен быть в том же семействе")
    pairs, seen = [], set()
    for row in read_jsonl(pairs_path):
        try:
            require_fields(row, frozenset({"left_id", "right_id", "split", "derivation_relation",
                                          "template_only", "reason"}))
            pair = LabeledPair(
                require_string(row["left_id"], "left_id"),
                require_string(row["right_id"], "right_id"),
                require_string(row["split"], "split"),
                require_bool(row["derivation_relation"], "derivation_relation"),
                require_bool(row["template_only"], "template_only"),
                require_string(row["reason"], "reason"),
            )
        except (TypeError, ValueError) as exc:
            raise AnalysisFailure(f"Некорректная схема пары: {exc}") from exc
        if type(pair.derivation_relation) is not bool or type(pair.template_only) is not bool:
            raise AnalysisFailure("Метки пары должны быть bool")
        if pair.template_only and pair.derivation_relation:
            raise AnalysisFailure("template_only несовместимо с derivation_relation=true")
        key = tuple(sorted((pair.left_id, pair.right_id)))
        if key in seen or pair.left_id == pair.right_id:
            raise AnalysisFailure("Повторная пара или self-match в разметке")
        seen.add(key)
        if pair.left_id not in by_id or pair.right_id not in by_id:
            raise AnalysisFailure("Неизвестная сдача в размеченной паре")
        a, b = index.records[pair.left_id], index.records[pair.right_id]
        if a.assignment_id != b.assignment_id or a.student_id == b.student_id:
            raise AnalysisFailure("Пара должна включать разных студентов одного задания")
        if by_id[a.submission_id]["split"] != pair.split or by_id[b.submission_id]["split"] != pair.split:
            raise AnalysisFailure("Пара пересекает границу tune/evaluation")
        if pair.derivation_relation and (
            by_id[a.submission_id]["family_id"] != by_id[b.submission_id]["family_id"]
        ):
            raise AnalysisFailure("Производные одного исходника должны быть в одном семействе")
        pairs.append(pair)
    if len(index.records) < 30 or len(pairs) < 40 or sum(p.template_only for p in pairs) < 5:
        raise AnalysisFailure("Нужно >=30 сдач, >=40 пар и >=5 пар с разрешённым шаблоном")
    for split in ("tune", "evaluation"):
        if {p.derivation_relation for p in pairs if p.split == split} != {True, False}:
            raise AnalysisFailure("Каждое разбиение должно содержать оба класса")
    return pairs, {
        "submissions": len(index.records), "pairs": len(pairs),
        "template_only_pairs": sum(p.template_only for p in pairs),
        "family_splits": dict(sorted(family_splits.items())), "family_leakage": False,
        "student_versions": index.student_versions(),
        "pairs_by_split": {s: sum(p.split == s for p in pairs) for s in ("tune", "evaluation")},
        "provenance_hash": stable_hash(provenance),
        "pairs_hash": stable_hash([asdict(pair) for pair in pairs]),
    }


def metrics(values: list[tuple[bool, float | None]], threshold: float) -> dict[str, object]:
    tp = fp = tn = fn = unknown = 0
    for expected, score in values:
        if score is None:
            unknown += 1
        elif score >= threshold:
            tp += expected
            fp += not expected
        else:
            tn += not expected
            fn += expected
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "precision": precision, "recall": recall, "f1": f1,
            "total": len(values), "scored": len(values) - unknown, "undetermined": unknown,
            "undetermined_fraction": unknown / len(values) if values else 0.0,
            "matrix_policy": "Матрица и P/R/F1 только по определённым score; отказы отдельно"}


def select_threshold(values: list[tuple[bool, float | None]]) -> float:
    candidates = {0.0, 1.0, *(score for _, score in values if score is not None)}

    def objective(threshold: float) -> tuple[float, float, float]:
        result = metrics(values, threshold)
        return float(result["f1"]), float(result["precision"]), threshold  # type: ignore[arg-type]

    return max(candidates, key=objective)


def build_index(path: Path, assignments: Mapping[str, Assignment],
                config: SimilarityConfig = DEFAULT_CONFIG) -> FeatureIndex:
    index = FeatureIndex()
    for item in feature_pipeline(path, assignments, config):
        if item.error:
            raise AnalysisFailure(f"Строка {item.line_no}: {item.error.message}")
        assert item.record is not None
        index.add(item.record)
    return index


def evaluate(index: FeatureIndex, pairs: list[LabeledPair], audit: dict[str, object],
             output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "experiment_version": "2", "task": "Учебное отношение derivation_relation",
        "audit": audit, "threshold_selection": "max F1 on tune; ties: precision, larger threshold",
        "methods": {}, "feature_hash": stable_hash([
            {"id": row.submission_id, "hash": row.submission_hash, "config": row.config_hash}
            for row in sorted(index.records.values(), key=lambda record: record.submission_id)
        ]),
    }
    methods: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for method in METHODS:
        initial: list[tuple[LabeledPair, PairReport]] = [
            (pair, compare(index.records[pair.left_id], index.records[pair.right_id], method))
            for pair in pairs
        ]
        tuning = [(pair.derivation_relation, report.score) for pair, report in initial if pair.split == "tune"]
        threshold = select_threshold(tuning)
        errors = []
        for pair, report in initial:
            prediction = None if report.score is None else report.score >= threshold
            row = {**asdict(pair), "method": method, "score": report.score,
                   "threshold": threshold, "prediction": prediction,
                   "review_required": prediction is None or prediction,
                   "status": report.status, "processing_errors": report.errors,
                   "evidence": [asdict(item) for item in report.evidence],
                   "config_hash": report.config_hash}
            rows.append(row)
            if pair.split == "evaluation" and (prediction is None or prediction != pair.derivation_relation):
                errors.append(row)
        methods[method] = {
            "threshold": threshold, "tune": metrics(tuning, threshold),
            "evaluation": metrics([(p.derivation_relation, r.score) for p, r in initial
                                    if p.split == "evaluation"], threshold),
            "evaluation_errors_and_abstentions": errors,
        }
    result["methods"] = methods
    (output / "quality.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    with (output / "labeled_scores.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result
