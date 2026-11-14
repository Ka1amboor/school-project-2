

import json
from pathlib import Path

from school_lab.assignments import load_assignments
from school_lab.comparison import compare
from school_lab.configuration import load_similarity_config
from school_lab.evaluation import build_index

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "lab2"


def main() -> None:
    output = ROOT / "results" / "lab2" / "template_demo"
    output.mkdir(parents=True, exist_ok=True)
    assignments = load_assignments(DATA / "assignments.json")
    demo = {}
    for name, config_file in (("before", "config.json"), ("after", "config_with_new_template.json")):
        config = load_similarity_config(DATA / config_file)
        index = build_index(DATA / "submissions.jsonl", assignments, config)
        reports = [compare(index.records["family-08-0"], index.records["family-09-0"], method,
                           1.0 if method == "ast" else 0.11764705882352941).to_dict()
                   for method in ("token", "ast")]
        demo[name] = {"configuration": {"version": config.version, "extra_templates": config.extra_templates},
                      "reports": reports}
    (output / "reports.json").write_text(json.dumps(demo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, data in demo.items():
        print(name, [(row["analyzer_id"], row["score"], row["config_version"], row["config_hash"][:12])
                     for row in data["reports"]])


if __name__ == "__main__":
    main()
