
import ast
import io
import json
import random
import tokenize
from itertools import combinations
from pathlib import Path

SEED = 20261111
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "lab2"
TEMPLATE = '''from typing import Iterable

def check_input(values):
    """Разрешённая инфраструктура задания: не часть авторского решения."""
    if values is None:
        raise ValueError("values must not be None")
    return list(values)
'''

BASES = [
    ("squares", "tune", "def solve(values):\n    return sum(x * x for x in values if x >= 0)\n"),
    ("squares", "tune", "def solve(values):\n    total = 0\n    for item in values:\n        if item >= 0:\n            total += item ** 2\n    return total\n"),
    ("squares", "tune", "def solve(values):\n    positive = filter(lambda number: number >= 0, values)\n    return sum(map(lambda number: number ** 2, positive))\n"),
    ("words", "tune", "def solve(words):\n    result = {}\n    for word in words:\n        result[word] = result.get(word, 0) + 1\n    return result\n"),
    ("words", "tune", "def solve(words):\n    from collections import Counter\n    return dict(Counter(words))\n"),
    ("squares", "evaluation", "def solve(values):\n    squares = [value ** 2 for value in values if value >= 0]\n    return sum(squares)\n"),
    ("squares", "evaluation", "def solve(values):\n    total = 0\n    for value in values:\n        if value < 0:\n            continue\n        total = total + value * value\n    return total\n"),
    ("words", "evaluation", "def solve(words):\n    return {word: words.count(word) for word in set(words)}\n"),
    ("words", "evaluation", "def solve(items):\n    return {item: items.count(item) for item in set(items)}\n"),
    ("words", "evaluation", "def solve(words):\n    result = {}\n    for word in sorted(words):\n        if word in result:\n            result[word] += 1\n        else:\n            result[word] = 1\n    return result\n"),
]


def renamed(code: str) -> str:
    names = {"solve": "calculate", "values": "data", "words": "entries", "items": "data",
             "x": "element", "item": "entry", "value": "element", "word": "entry",
             "total": "accumulator", "result": "mapping", "squares": "powers",
             "positive": "selected", "number": "element"}
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        tokens.append(token._replace(string=names.get(token.string, token.string))
                      if token.type == tokenize.NAME else token)
    result = tokenize.untokenize(tokens)
    ast.parse(result)
    return result


def submission(sid: str, student: str, assignment: str, code: str) -> dict[str, object]:
    return {
        "schema_version": 1, "submission_id": sid, "student_id": student,
        "assignment_id": assignment, "submitted_at": "2026-10-20T12:00:00+03:00",
        "text": "Синтетический учебный пример. Происхождение описано в provenance.jsonl.",
        "code": code, "template_id": "allowed-v1",
        "ai_declaration": {"used": True, "tool": "OpenAI Codex",
                           "purpose": "Подготовка синтетического учебного корпуса",
                           "verified_by_student": False},
    }


def write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
                    encoding="utf-8")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    submissions, provenance, pairs = [], [], []
    families: list[tuple[str, str, str, list[str]]] = []
    for index, (assignment, split, code) in enumerate(BASES, 1):
        family = f"family-{index:02}"
        ids = []
        variants = [("original", code), ("exact_copy", code),
                    ("formatting", "# Новый комментарий\n\n" + code.replace("    ", "  ")),
                    ("rename", renamed(code))]
        for number, (variant, body) in enumerate(variants):
            sid = f"{family}-{number}"
            ids.append(sid)
            submissions.append(submission(sid, f"student-{index:02}-{number}", assignment,
                                          TEMPLATE + "\n" + body))
            provenance.append({
                "submission_id": sid, "assignment_id": assignment, "family_id": family,
                "split": split, "variant": variant, "source": "synthetic_authored_for_lab2",
                "parent_id": None if number == 0 else f"{family}-0",
                "transformation": "independently constructed base" if number == 0 else variant,
            })
        families.append((family, assignment, split, ids))
        for left, right in combinations(ids, 2):
            pairs.append({"left_id": left, "right_id": right, "split": split,
                          "derivation_relation": True, "template_only": False,
                          "reason": "Производные одного явно заданного исходника"})
    for a, b in combinations(families, 2):
        if a[1:3] != b[1:3]:
            continue
        for i, j in ((0, 0), (2, 0), (3, 2)):
            pairs.append({"left_id": a[3][i], "right_id": b[3][j], "split": a[2],
                          "derivation_relation": False, "template_only": True,
                          "reason": "Независимо построенные решения; общий разрешённый шаблон. "
                                    "Возможны совпадающие короткие идиомы Python."})
    for split, assignment, family in (("tune", "squares", "family-01"),
                                       ("evaluation", "squares", "family-06")):
        for variant, code in (("syntax_error", "def solve(:\n    return 1\n"),
                              ("empty_code", "# Решение не представлено\n")):
            sid = f"{split}-{variant}"
            submissions.append(submission(sid, sid, assignment, code))
            provenance.append({"submission_id": sid, "assignment_id": assignment,
                               "family_id": f"{split}-{variant}", "split": split,
                               "variant": variant, "parent_id": None,
                               "source": "synthetic_authored_for_lab2",
                               "transformation": "Специально построенный граничный пример"})
            pairs.append({"left_id": f"{family}-0", "right_id": sid, "split": split,
                          "derivation_relation": False, "template_only": False,
                          "reason": "Граничный пример; ожидается неопределённый результат"})
    # A later version from the same student is not an inter-student candidate.
    version = dict(submissions[0], submission_id="family-01-v2")
    submissions.append(version)
    provenance.append(dict(provenance[0], submission_id="family-01-v2", variant="student_version",
                           parent_id="family-01-0", transformation="Повторная версия того же студента"))
    random.Random(SEED).shuffle(submissions)
    write_jsonl(DEST / "submissions.jsonl", submissions)
    write_jsonl(DEST / "provenance.jsonl", provenance)
    write_jsonl(DEST / "pairs.jsonl", pairs)
    assignments = {"assignments": [
        {"assignment_id": name, "allowed_template": TEMPLATE,
         "requirements": ["Функция solve", description]}
        for name, description in (("squares", "Сумма квадратов неотрицательных чисел"),
                                  ("words", "Частоты слов в списке"))
    ]}
    (DEST / "assignments.json").write_text(json.dumps(assignments, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(f"seed={SEED}; submissions={len(submissions)}; pairs={len(pairs)}")


if __name__ == "__main__":
    main()
