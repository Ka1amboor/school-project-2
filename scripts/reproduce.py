

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("+", sys.executable, *args, flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-benchmark", action="store_true", help="Быстрая проверка без нагрузочных замеров")
    args = parser.parse_args()
    run("scripts/generate_corpus.py")
    run("-m", "pytest", "-q")
    run("-m", "ruff", "check", ".")
    run("-m", "mypy", "src")
    run("-m", "school_lab", "evaluate", "data/lab2/submissions.jsonl",
        "--assignments", "data/lab2/assignments.json", "--config", "data/lab2/config.json",
        "--provenance", "data/lab2/provenance.jsonl", "--pairs", "data/lab2/pairs.jsonl",
        "--output", "results/lab2")
    run("-m", "school_lab", "analyze", "data/lab2/submissions.jsonl",
        "--assignments", "data/lab2/assignments.json", "--config", "data/lab2/config.json",
        "--output", "results/lab2/analysis")
    run("scripts/demo_template.py")
    if not args.skip_benchmark:
        run("scripts/benchmark_lab2.py")
        run("scripts/profile_lab2.py")
    run("scripts/check_installation.py")
    run("scripts/write_report.py")


if __name__ == "__main__":
    main()
