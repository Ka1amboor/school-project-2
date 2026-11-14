"""Comparison depends on feature contracts; it does not read or store source code."""

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import combinations
from uuid import uuid4

from .errors import AnalysisFailure
from .features import FEATURE_VERSION, FeatureRecord, Gram, Span
from .models import Evidence


def jaccard_reference(left: Mapping[Gram, Span], right: Mapping[Gram, Span]) -> float | None:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else None


def jaccard(left: Mapping[Gram, Span], right: Mapping[Gram, Span]) -> float | None:
    # Intersect key views in C; avoid copying both inputs and allocating the union.
    intersection = len(left.keys() & right.keys())
    union_size = len(left) + len(right) - intersection
    return intersection / union_size if union_size else None


@dataclass(frozen=True, slots=True)
class PairReport:
    report_id: str
    report_version: str
    analyzer_id: str
    analyzer_version: str
    created_at: str
    left_id: str
    right_id: str
    left_hash: str
    right_hash: str
    assignment_id: str
    config_hash: str
    config_version: str
    threshold: float
    score: float | None
    status: str
    review_required: bool
    explanation: str
    evidence: tuple[Evidence, ...]
    left_template_ranges: tuple[Span, ...]
    right_template_ranges: tuple[Span, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compare(left: FeatureRecord, right: FeatureRecord, method: str = "token",
            threshold: float = 0.5, *, reference: bool = False) -> PairReport:
    if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
        raise AnalysisFailure("Порог должен быть конечным числом от 0 до 1")
    if left.submission_id == right.submission_id:
        raise AnalysisFailure("self-match исключён")
    if left.student_id == right.student_id:
        raise AnalysisFailure("Версии одного студента сравниваются отдельно от межстуденческих пар")
    if left.assignment_id != right.assignment_id:
        raise AnalysisFailure("Сравниваются только работы одного задания")
    if left.config_hash != right.config_hash:
        raise AnalysisFailure("Признаки вычислены с разными конфигурациями")
    a, b = left.method(method), right.method(method)
    errors = tuple(f"{record.submission_id}: {features.error}"
                   for record, features in ((left, a), (right, b)) if features.error)
    score = None if errors else (jaccard_reference if reference else jaccard)(
        a.locations, b.locations
    )
    evidence: list[Evidence] = []
    if score is not None:
        # Stable, bounded excerpts; each shared feature contributes both source locations.
        for key in sorted(a.locations.keys() & b.locations.keys())[:5]:
            for record, features in ((left, a), (right, b)):
                start, end = features.locations[key]
                if not 1 <= start <= end <= record.line_count:
                    raise AnalysisFailure("Evidence выходит за границы исходника")
                item = Evidence(record.submission_id, "code", start, end)
                if item not in evidence:
                    evidence.append(item)
    review = score is None or score >= threshold
    explanation = (
        f"Jaccard множеств {'токеновых 3-грамм' if method == 'token' else 'нормализованных AST-поддеревьев'}; "
        f"признаков: {len(a.locations)} и {len(b.locations)}. "
        "Точные блоки разрешённого шаблона исключены; их исходные диапазоны сохранены. "
        "Evidence содержит до пяти общих признаков. "
        + ("Недостаточно данных: ошибка обработки или пустое объединение. " if score is None else "")
        + "review_required означает ручной разбор; score не доказывает нарушение."
    )
    return PairReport(
        str(uuid4()), "2", method, FEATURE_VERSION, datetime.now(UTC).isoformat(),
        left.submission_id, right.submission_id, left.submission_hash, right.submission_hash,
        left.assignment_id, left.config_hash, left.config_version, threshold, score,
        "insufficient_data" if score is None else "ok", review, explanation,
        tuple(evidence), left.template_ranges, right.template_ranges, errors,
    )


class FeatureIndex:
    def __init__(self) -> None:
        self.records: dict[str, FeatureRecord] = {}

    def add(self, record: FeatureRecord) -> None:
        previous = self.records.get(record.submission_id)
        if previous and previous != record:
            raise AnalysisFailure(f"Конфликт версии сдачи {record.submission_id}")
        self.records[record.submission_id] = record

    def pairs(self, method: str, threshold: float) -> Iterator[PairReport]:
        for left, right in combinations(self.records.values(), 2):
            if left.assignment_id == right.assignment_id and left.student_id != right.student_id:
                yield compare(left, right, method, threshold)

    def top_k(self, submission_id: str, method: str, threshold: float, k: int = 5) -> list[PairReport]:
        if k < 1:
            raise AnalysisFailure("k должен быть положительным")
        if submission_id not in self.records:
            raise AnalysisFailure(f"Неизвестная сдача: {submission_id}")
        left = self.records[submission_id]
        best: list[PairReport] = []
        for right in self.records.values():
            if right.submission_id == left.submission_id or right.student_id == left.student_id:
                continue
            if left.assignment_id != right.assignment_id:
                continue
            best.append(compare(left, right, method, threshold))
            best.sort(key=lambda item: (-(item.score if item.score is not None else -1), item.right_id))
            del best[k:]
        return best

    def student_versions(self) -> dict[str, list[str]]:
        groups: dict[tuple[str, str], list[str]] = {}
        for record in self.records.values():
            groups.setdefault((record.assignment_id, record.student_id), []).append(record.submission_id)
        return {f"{assignment}/{student}": ids for (assignment, student), ids in groups.items()
                if len(ids) > 1}
