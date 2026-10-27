"""Offline token and GPU-hour budget estimates for *planned* training runs.

Example:
  python training/budget_planner.py --steps 400 --micro-batch 4 --grad-accum 4 \
    --seq-len 2048 --gpus 1 --tokens-per-second 3309 --gpu-hour-price 3

Throughput is an optional user-supplied cluster-wide measurement, not a
hardware prediction. Estimates assume full sequences without padding.
No models, datasets, services or credentials are accessed.
"""
from __future__ import annotations

import argparse
import json
import math


def positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite non-negative number") from exc
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return value


def plan(*, steps: int, micro_batch: int, grad_accum: int, seq_len: int,
         gpus: int = 1, tokens_per_second: float | None = None,
         gpu_hour_price: float | None = None) -> dict:
    if any(type(n) is not int or n <= 0 for n in (steps, micro_batch, grad_accum, seq_len, gpus)):
        raise ValueError("steps, micro_batch, grad_accum, seq_len and gpus must be positive integers")
    if tokens_per_second is not None and (not math.isfinite(tokens_per_second) or tokens_per_second <= 0):
        raise ValueError("tokens_per_second must be positive and finite")
    if gpu_hour_price is not None and (not math.isfinite(gpu_hour_price) or gpu_hour_price < 0):
        raise ValueError("gpu_hour_price must be finite and non-negative")
    if gpu_hour_price is not None and tokens_per_second is None:
        raise ValueError("gpu_hour_price requires measured tokens_per_second")

    per_step = micro_batch * grad_accum * seq_len * gpus
    total = per_step * steps
    hours = total / tokens_per_second / 3600 if tokens_per_second is not None else None
    return {
        "tokens_per_optimizer_step": per_step,
        "planned_tokens": total,
        "estimated_wall_hours": hours,
        "estimated_gpu_hours": hours * gpus if hours is not None else None,
        "estimated_cost": hours * gpus * gpu_hour_price
        if hours is not None and gpu_hour_price is not None else None,
        "assumptions": "Full-length unpadded sequences; measured cluster-wide throughput; no overhead adjustment.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=positive_int, required=True)
    parser.add_argument("--micro-batch", type=positive_int, required=True)
    parser.add_argument("--grad-accum", type=positive_int, required=True)
    parser.add_argument("--seq-len", type=positive_int, required=True)
    parser.add_argument("--gpus", type=positive_int, default=1)
    parser.add_argument("--tokens-per-second", type=nonnegative_float)
    parser.add_argument("--gpu-hour-price", type=nonnegative_float)
    args = parser.parse_args()
    try:
        result = plan(steps=args.steps, micro_batch=args.micro_batch,
                      grad_accum=args.grad_accum, seq_len=args.seq_len,
                      gpus=args.gpus, tokens_per_second=args.tokens_per_second,
                      gpu_hour_price=args.gpu_hour_price)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
