

import ast
import io
import tokenize
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .analyzers import template_ranges
from .errors import AnalysisFailure, SubmissionError
from .hashing import stable_hash, submission_hash
from .models import Assignment, Submission
from .parsing import parse_submission
from .sources import JSONLSource

FEATURE_VERSION = "2.0"
METHODS = ("token", "ast")
Span = tuple[int, int]
Gram = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SimilarityConfig:
    version: str = "1"
    extra_templates: tuple[str, ...] = ()

    def digest(self, assignment: Assignment) -> str:
        return stable_hash({
            "feature_version": FEATURE_VERSION, "config_version": self.version,
            "assignment_id": assignment.assignment_id,
            "allowed_template": assignment.allowed_template,
            "requirements": assignment.requirements,
            "extra_templates": self.extra_templates, "ngram_size": 3,
        })


DEFAULT_CONFIG = SimilarityConfig()


@dataclass(frozen=True, slots=True)
class MethodFeatures:

    locations: Mapping[Gram, Span]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    submission_id: str
    student_id: str
    assignment_id: str
    submission_hash: str
    config_hash: str
    config_version: str
    line_count: int
    template_ranges: tuple[Span, ...]
    token: MethodFeatures
    ast: MethodFeatures

    def method(self, name: str) -> MethodFeatures:
        if name not in METHODS:
            raise AnalysisFailure(f"Неизвестный метод: {name}")
        return self.token if name == "token" else self.ast


@dataclass(frozen=True, slots=True)
class PipelineItem:
    line_no: int
    record: FeatureRecord | None = None
    error: SubmissionError | None = None


def token_features(code: str, excluded: frozenset[int]) -> MethodFeatures:
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
               tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING}
    window: list[tokenize.TokenInfo] = []
    locations: dict[Gram, Span] = {}
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type in ignored:
                continue
            if token.type == tokenize.ERRORTOKEN and not token.string.isspace():
                raise tokenize.TokenError(f"Недопустимый токен {token.string!r}", token.start)
            if token.type == tokenize.ERRORTOKEN:
                continue
            if any(line in excluded for line in range(token.start[0], token.end[0] + 1)):
                window.clear()  # No artificial 3-grams across an excluded template.
                continue
            window.append(token)
            if len(window) > 3:
                window.pop(0)
            if len(window) == 3:
                gram = tuple(f"{item.type}:{item.string}" for item in window)
                locations.setdefault(gram, (window[0].start[0], window[-1].end[0]))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        return MethodFeatures({}, f"tokenize: {exc}")
    return MethodFeatures(locations)


class NormalizeNames(ast.NodeTransformer):


    def __init__(self, excluded: frozenset[int]) -> None:
        self.names: dict[str, str] = {}
        self.excluded = excluded

    def name(self, value: str) -> str:
        return self.names.setdefault(value, f"name_{len(self.names)}")

    def visit(self, node: ast.AST) -> ast.AST | None:
        if isinstance(node, ast.stmt) and all(
            line in self.excluded for line in range(node.lineno, (node.end_lineno or node.lineno) + 1)
        ):
            return None
        return cast(ast.AST, super().visit(node))

    def generic_visit(self, node: ast.AST) -> ast.AST:

        for field, value in ast.iter_fields(node):
            if isinstance(value, str) and (
                (isinstance(node, ast.Name) and field == "id")
                or (isinstance(node, ast.arg) and field == "arg")
                or (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and field == "name")
                or (isinstance(node, ast.Attribute) and field == "attr")
                or (isinstance(node, ast.keyword) and field == "arg")
                or (isinstance(node, ast.alias) and field == "asname")
            ):
                setattr(node, field, self.name(value))
            elif isinstance(value, list) and field in {"names"} and isinstance(
                node, (ast.Global, ast.Nonlocal)
            ):
                setattr(node, field, [self.name(item) for item in value])
            elif isinstance(value, list):
                result = []
                for item in value:
                    visited = self.visit(item) if isinstance(item, ast.AST) else item
                    if visited is not None:
                        result.append(visited)
                setattr(node, field, result)
            elif isinstance(value, ast.AST):
                visited = self.visit(value)
                if visited is None:
                    delattr(node, field)
                else:
                    setattr(node, field, visited)
        return node


def ast_features(code: str, excluded: frozenset[int]) -> MethodFeatures:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError) as exc:
        return MethodFeatures({}, f"AST: {exc}")
    try:
        normalized = NormalizeNames(excluded).visit(tree)
        assert normalized is not None
        locations: dict[Gram, Span] = {}
        for node in ast.walk(normalized):
            if isinstance(node, ast.stmt) and hasattr(node, "lineno"):
                # Subtree structure retains constants, operators and nesting.
                key = (ast.dump(node, annotate_fields=True, include_attributes=False),)
                locations.setdefault(key, (node.lineno, node.end_lineno or node.lineno))
        return MethodFeatures(locations)
    except RecursionError as exc:
        return MethodFeatures({}, f"AST: превышена глубина дерева ({exc})")


def extract_features(value: Submission, assignment: Assignment,
                     config: SimilarityConfig = DEFAULT_CONFIG) -> FeatureRecord:
    if value.assignment_id != assignment.assignment_id:
        raise AnalysisFailure("Задание и сдача не соответствуют друг другу")
    templates = (assignment.allowed_template, *config.extra_templates)
    ranges = tuple(sorted({span for template in templates
                           for span in template_ranges(value.code, template)}))
    excluded = frozenset(line for start, end in ranges for line in range(start, end + 1))
    return FeatureRecord(
        value.submission_id, value.student_id, value.assignment_id, submission_hash(value),
        config.digest(assignment), config.version, len(value.code.splitlines()), ranges,
        token_features(value.code, excluded), ast_features(value.code, excluded),
    )


def feature_pipeline(path: Path, assignments: Mapping[str, Assignment],
                     config: SimilarityConfig = DEFAULT_CONFIG) -> Iterator[PipelineItem]:
    with JSONLSource(path).open() as records:
        for item in records:
            if item.error is not None:
                yield PipelineItem(item.line_no, error=item.error)
                continue
            try:
                value = parse_submission(item.raw)
                assignment = assignments.get(value.assignment_id)
                if assignment is None:
                    raise SubmissionError("unknown_assignment", "assignment_id", value.assignment_id)
                result = extract_features(value, assignment, config)
            except SubmissionError as exc:
                yield PipelineItem(item.line_no, error=exc)
            else:
                yield PipelineItem(item.line_no, record=result)
