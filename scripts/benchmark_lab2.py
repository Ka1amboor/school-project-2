

import argparse
import cProfile
import gc
import json
import platform
import pstats
import random
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

from school_lab.assignments import load_assignments
from school_lab.comparison import jaccard, jaccard_reference
from school_lab.evaluation import build_index, read_jsonl
from school_lab.hashing import stable_hash

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "lab2"
SEED = 20261111
QUERY_REPEATS = 10


def write_load(path: Path, count: int) -> None:

    source = [row for row in read_jsonl(DATA / "submissions.jsonl")
              if str(row["submission_id"]).endswith("-0")]
    source.sort(key=lambda row: str(row["submission_id"]))
    rng = random.Random(SEED)
    with path.open("w", encoding="utf-8") as stream:
        for i in range(count):
            row = dict(source[rng.randrange(len(source))], submission_id=f"load-{i}",
                       student_id=f"load-student-{i}")
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_queries(index, implementation, repetitions=QUERY_REPEATS):
    query = next(iter(index.records.values()))
    candidates = [record for record in index.records.values()
                  if record.assignment_id == query.assignment_id and record.student_id != query.student_id]
    checksum = 0.0
    calls = 0
    for _ in range(repetitions):
        for method in ("token", "ast"):
            a = query.method(method).locations
            for candidate in candidates:
                score = implementation(a, candidate.method(method).locations)
                checksum += score or 0.0
                calls += 1
    return checksum, calls


def worker(path: Path, implementation: str) -> dict[str, object]:
    assignments = load_assignments(DATA / "assignments.json")
    compare = jaccard_reference if implementation == "before" else jaccard
    gc.collect()
    start = time.perf_counter()
    index = build_index(path, assignments)
    built = time.perf_counter()
    checksum, calls = score_queries(index, compare)
    end = time.perf_counter()
    result = {"implementation": implementation, "records": len(index.records),
              "index_wall_seconds": built - start, "query_wall_seconds": end - built,
              "wall_seconds": end - start, "score_calls": calls, "checksum": checksum}
    del index
    gc.collect()
    # A separate identical pass measures peak allocated Python memory; timing above is untraced.
    tracemalloc.start()
    index = build_index(path, assignments)
    score_queries(index, compare)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result["peak_python_bytes"] = peak
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "lab2")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--implementation", choices=("before", "after"), default="after")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.implementation)))
        return
    if args.repeats < 3:
        parser.error("По ТЗ нужно минимум три повтора")
    args.output.mkdir(parents=True, exist_ok=True)
    samples = []
    with tempfile.TemporaryDirectory(prefix="school-lab2-") as tmp:
        for size in (100, 1000, 5000):
            path = Path(tmp) / f"load-{size}.jsonl"
            write_load(path, size)
            for repeat in range(args.repeats):
                # Alternate order to reduce systematic warm-cache bias.
                order = ("before", "after") if repeat % 2 == 0 else ("after", "before")
                for implementation in order:
                    output = subprocess.check_output([
                        sys.executable, str(Path(__file__).resolve()), "--worker", str(path),
                        "--implementation", implementation,
                    ], text=True, cwd=ROOT)
                    row = dict(json.loads(output), repeat=repeat + 1)
                    samples.append(row)
                    print(f"n={size} {implementation} repeat={repeat + 1}: "
                          f"wall={row['wall_seconds']:.4f}s peak={row['peak_python_bytes'] / 1048576:.2f}MiB",
                          flush=True)
            if size == 5000:
                index = build_index(path, load_assignments(DATA / "assignments.json"))
                for implementation, function in (("before", jaccard_reference), ("after", jaccard)):
                    profile = cProfile.Profile()
                    profile.runcall(score_queries, index, function)
                    profile.dump_stats(str(args.output / f"profile_{implementation}.prof"))
                    with (args.output / f"profile_{implementation}.txt").open("w", encoding="utf-8") as stream:
                        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("tottime").print_stats(25)
    medians = []
    for size in (100, 1000, 5000):
        for implementation in ("before", "after"):
            rows = [row for row in samples if row["records"] == size and row["implementation"] == implementation]
            medians.append({"records": size, "implementation": implementation,
                            **{key: statistics.median(row[key] for row in rows)
                               for key in ("index_wall_seconds", "query_wall_seconds", "wall_seconds",
                                           "peak_python_bytes", "score_calls", "checksum")}})
        checksums = {row["checksum"] for row in samples if row["records"] == size}
        if len(checksums) != 1:
            raise RuntimeError("Оптимизация изменила результаты")
    result = {
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "machine": platform.machine(), "processor": platform.processor()},
        "seed": SEED, "repeats": args.repeats, "query_repeats": QUERY_REPEATS,
        "load": "Синтетические уникальные ID; десять небольших исходных решений; два задания",
        "workload": "read -> validate -> token+AST feature index; one-query scan per method x10; sequential",
        "memory_policy": "Отдельный идентичный прогон tracemalloc, peak Python allocations, не RSS",
        "timing_policy": "perf_counter без tracemalloc; новый процесс для каждого повтора; без конкурентности",
        "comparison_policy": "before=copy both sets and union/intersection; after=key-view intersection, union size=len(a)+len(b)-i",
        "source_hashes": {name: stable_hash((ROOT / "src" / "school_lab" / name).read_text(encoding="utf-8"))
                          for name in ("features.py", "comparison.py")},
        "samples": samples, "medians": medians,
    }
    (args.output / "performance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
