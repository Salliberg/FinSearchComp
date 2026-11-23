"""Summarise an eval.py result file with the same rules as the Harbor verifier.

- a scored item counts its judge score (0 or 1);
- an empty answer counts as 0 (eval.py skips it; Harbor scores it 0);
- a T1 judge answer of {"score": "null"} (no ground truth) is left out;
- an unparseable judge reply or a failed judge call is left out and reported.

Example:
    python parity/score.py parity/runs/run1/eval.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_ERROR_SCORE = -100000


def main(path: str) -> int:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    scores = defaultdict(list)
    counts = defaultdict(int)
    for item in items:
        tier = item["prompt_id"][0][1:3]  # "T1", "T2", "T3"
        evals = item.get("evaluations", [])
        if not evals:
            scores[tier].append(0.0)
            counts["empty answer (scored 0)"] += 1
            continue
        score = evals[0]["score"]
        if score is None:
            counts["T1 no ground truth (excluded)"] += 1
        elif score == DEFAULT_ERROR_SCORE:
            counts["judge error (excluded)"] += 1
        else:
            scores[tier].append(float(score))

    summary = {}
    for tier in sorted(scores):
        s = scores[tier]
        summary[tier] = {"n": len(s), "accuracy": sum(s) / len(s)}
    every = [x for s in scores.values() for x in s]
    summary["overall"] = {"n": len(every), "accuracy": sum(every) / len(every) if every else 0.0}
    summary["counts"] = dict(counts)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
