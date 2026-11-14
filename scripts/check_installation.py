

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    wheel = ROOT / "wheels" / "school_lab-0.2.0-py3-none-any.whl"
    log = []
    with tempfile.TemporaryDirectory(prefix="school-lab2-install-") as tmp:
        directory = Path(tmp)
        environment = directory / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        commands = [
            ["-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
            ["-c", ("import school_lab, sys; from pathlib import Path; from importlib.metadata import version; "
             "assert Path(school_lab.__file__).is_relative_to(sys.prefix); "
             "assert version('school-lab') == '0.2.0'; print('isolated import OK, version=0.2.0')")],
            ["-m", "school_lab", "--help"],
            ["-m", "school_lab", "import", str(ROOT / "data" / "submissions.jsonl")],
            ["-m", "school_lab", "evaluate", str(ROOT / "data/lab2/submissions.jsonl"),
             "--assignments", str(ROOT / "data/lab2/assignments.json"),
             "--provenance", str(ROOT / "data/lab2/provenance.jsonl"),
             "--pairs", str(ROOT / "data/lab2/pairs.jsonl"), "--output", str(directory / "quality")],
        ]
        for args in commands:
            result = subprocess.run([str(python), *args], cwd=directory, text=True,
                                    capture_output=True, check=True)
            log.extend(["$ python " + " ".join(args), result.stdout.strip(), result.stderr.strip()])
    path = ROOT / "results" / "lab2" / "installation_check.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(log) + "\nPASS: isolated offline installation and CLI\n", encoding="utf-8")
    print("PASS: isolated offline installation and CLI")


if __name__ == "__main__":
    main()
