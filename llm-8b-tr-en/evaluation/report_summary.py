"""Summarize existing Turkish evaluation JSON without loading a language model."""

import argparse
import json
import math
from pathlib import Path


def summarize(report):
    """Aggregate category scores; reject malformed or unmeasured results."""
    categories = report.get("results")
    if not isinstance(categories, dict) or not categories:
        raise ValueError("Expected nonempty 'results' object")
    output = {}
    all_scores = []
    for name, rows in categories.items():
        if not isinstance(name, str) or not isinstance(rows, list) or not rows:
            raise ValueError("Each category must contain nonempty result rows")
        scores = []
        for row in rows:
            if not isinstance(row, dict) or isinstance(row.get("score"), bool):
                raise ValueError("Each row must have a numeric score")
            value = row.get("score")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError("Scores must be finite numbers from 0 to 100")
            scores.append(value)
        output[name] = {"count": len(scores), "average_score": sum(scores) / len(scores)}
        all_scores.extend(scores)
    return {"total_tests": len(all_scores), "average_score": sum(all_scores) / len(all_scores), "categories": output}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize a saved Turkish model evaluation")
    parser.add_argument("input", help="Path to turkish_eval_results.json")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary")
    args = parser.parse_args(argv)
    try:
        with Path(args.input).open(encoding="utf-8") as stream:
            summary = summarize(json.load(stream))
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"Invalid evaluation report: {exc}\n")
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Overall: {summary['average_score']:.1f}% ({summary['total_tests']} prompts)")
        for category, result in summary["categories"].items():
            print(f"  {category}: {result['average_score']:.1f}% ({result['count']} prompts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
