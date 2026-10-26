"""Run all evaluations and write results/metrics.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from rmatch.evaluate import run_all  # noqa: E402

if __name__ == "__main__":
    res = run_all()
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(k, json.dumps(v, indent=1))
