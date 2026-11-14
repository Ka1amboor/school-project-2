
import argparse
import json
import sys
from pathlib import Path

from .assignments import load_assignments
from .comparison import FeatureIndex
from .configuration import load_similarity_config
from .errors import AnalysisFailure, SubmissionError
from .evaluation import build_index, evaluate, validate_corpus
from .features import METHODS, feature_pipeline
from .repositories import InMemorySubmissionRepository
from .services import SubmissionService, import_source
from .sources import JSONLSource


def parser_for_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="school-lab", description="Учебные сдачи и анализ сходства — Python Core")
    subcommands = parser.add_subparsers(dest="command", required=True)
    import_command = subcommands.add_parser("import", help="Импорт файла сдач")
    import_command.add_argument("path", type=Path)
    for name, help_text in (("analyze", "ЛР2: сравнение пар или top-k"),
                            ("evaluate", "ЛР2: настройка порога и независимая оценка")):
        command = subcommands.add_parser(name, help=help_text)
        command.add_argument("path", type=Path, help="JSONL сдач")
        command.add_argument("--assignments", type=Path, required=True)
        command.add_argument("--config", type=Path)
        command.add_argument("--output", type=Path, required=True)
        if name == "analyze":
            command.add_argument("--method", choices=(*METHODS, "both"), default="both")
            command.add_argument("--threshold", type=float, default=0.5,
                                 help="Для оценки качества используйте пороги из evaluate")
            command.add_argument("--submission-id")
            command.add_argument("--top-k", type=int, default=5)
        else:
            command.add_argument("--provenance", type=Path, required=True)
            command.add_argument("--pairs", type=Path, required=True)
    return parser


def run_analyze(args: argparse.Namespace) -> int:
    assignments = load_assignments(args.assignments)
    config = load_similarity_config(args.config)
    index = FeatureIndex()
    errors = []
    for item in feature_pipeline(args.path, assignments, config):
        if item.error:
            errors.append({"line_no": item.line_no, **item.error.to_dict()})
        else:
            assert item.record is not None
            index.add(item.record)
    if not 0 <= args.threshold <= 1 or args.top_k < 1:
        raise AnalysisFailure("Порог должен быть в [0, 1], top-k должен быть положительным")
    args.output.mkdir(parents=True, exist_ok=True)
    count = unknown = 0
    methods = METHODS if args.method == "both" else (args.method,)
    with (args.output / "pairs.jsonl").open("w", encoding="utf-8") as stream:
        for method in methods:
            reports = (iter(index.top_k(args.submission_id, method, args.threshold, args.top_k))
                       if args.submission_id else index.pairs(method, args.threshold))
            for report in reports:
                count += 1
                unknown += report.score is None
                stream.write(json.dumps(report.to_dict(), ensure_ascii=False) + "\n")
    summary = {"records": len(index.records), "reports": count, "undetermined": unknown,
               "student_versions": index.student_versions(), "errors": errors,
               "methods": methods, "feature_version": "2.0", "threshold": args.threshold}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    args = parser_for_cli().parse_args(argv)
    try:
        if args.command == "import":
            summary = import_source(JSONLSource(args.path), SubmissionService(InMemorySubmissionRepository()))
            print(json.dumps(summary.to_dict(), ensure_ascii=False))
            return 1 if summary.errors else 0
        if args.command == "analyze":
            return run_analyze(args)
        assignments = load_assignments(args.assignments)
        index = build_index(args.path, assignments, load_similarity_config(args.config))
        pairs, audit = validate_corpus(index, args.provenance, args.pairs)
        result = evaluate(index, pairs, audit, args.output)
        methods = result["methods"]
        assert isinstance(methods, dict)
        print(json.dumps({"methods": {key: {"threshold": value["threshold"],
                                           "evaluation": value["evaluation"]}
                                      for key, value in methods.items()},
                          "output": str(args.output)}, ensure_ascii=False))
        return 0
    except (SubmissionError, AnalysisFailure, ValueError) as exc:
        print(json.dumps({"code": "invalid_input", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    except (OSError, UnicodeError) as exc:
        print(json.dumps({"code": "source_unavailable", "message": str(exc)}, ensure_ascii=False),
              file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())
