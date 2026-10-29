"""Read-only audit of a saved training run; no model or GPU is required.

Usage:
    python results/training_audit.py [path/to/training_info.json] [--json]
        [--min-tokens-per-parameter RATIO]

Derived metrics describe the recorded training run, not held-out model quality.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


DEFAULT_REPORT = Path(__file__).with_name("training_info.json")


def _positive_integer(record: dict, key: str) -> int:
    value = record.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _finite_number(record: dict, key: str, *, positive: bool = True) -> float:
    value = record.get(key)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    if value <= 0 if positive else value < 0:
        raise ValueError(f"{key} must be {'positive' if positive else 'non-negative'}")
    return float(value)


def nonnegative_finite(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite non-negative number") from exc
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return value


def audit_record(record: dict) -> dict:
    """Validate saved metadata and calculate explicitly derived metrics."""
    if not isinstance(record, dict):
        raise ValueError("training report must be a JSON object")
    params = _positive_integer(record, "total_params")
    steps = _positive_integer(record, "training_steps")
    tokens = _positive_integer(record, "total_tokens_processed")
    minutes = _finite_number(record, "training_time_minutes")
    loss = _finite_number(record, "final_loss", positive=False)
    return {
        "total_params": params,
        "training_steps": steps,
        "total_tokens_processed": tokens,
        "training_time_minutes": minutes,
        "final_training_loss": loss,
        "derived": {
            "tokens_per_step": tokens / steps,
            "tokens_per_second": tokens / (minutes * 60),
            "tokens_per_parameter": tokens / params,
        },
    }


def audit_file(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as stream:
            record = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read training report: {exc}") from exc
    return audit_record(record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", action="store_true", help="print machine-readable metrics")
    parser.add_argument("--min-tokens-per-parameter", type=nonnegative_finite,
                        help="fail if recorded training tokens per parameter fall below this ratio")
    args = parser.parse_args(argv)
    try:
        report = audit_file(args.path)
    except ValueError as exc:
        parser.exit(1, f"Training audit failed: {exc}\n")
    passed = True
    if args.min_tokens_per_parameter is not None:
        actual = report["derived"]["tokens_per_parameter"]
        passed = actual >= args.min_tokens_per_parameter
        report["coverage_check"] = {
            "minimum_tokens_per_parameter": args.min_tokens_per_parameter,
            "passed": passed,
        }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Recorded steps: {report['training_steps']:,}")
        print(f"Recorded tokens: {report['total_tokens_processed']:,}")
        print(f"Final training loss (not validation): {report['final_training_loss']:.5f}")
        print(f"Derived tokens/s: {report['derived']['tokens_per_second']:.2f}")
        print(f"Derived tokens/parameter: {report['derived']['tokens_per_parameter']:.6f}")
    if not passed:
        print("Training coverage below --min-tokens-per-parameter threshold", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
