"""Read-only structural audit of a Llama-style model config (no GPU needed).

Usage: python llm-8b-tr-en/configs/audit_model_config.py [config.json] [--json]
Checks attention shape and grouped-query divisibility, not model quality.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).with_name("model_config.json")
REQUIRED_POSITIVE_INTS = (
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim",
    "max_position_embeddings",
)


def audit_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("model config must be a JSON object")
    if config.get("model_type") != "llama":
        raise ValueError("model_type must be 'llama'")
    for key in REQUIRED_POSITIVE_INTS:
        value = config.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["num_attention_heads"] * config["head_dim"] != config["hidden_size"]:
        raise ValueError("num_attention_heads * head_dim must equal hidden_size")
    if config["num_attention_heads"] % config["num_key_value_heads"]:
        raise ValueError("num_key_value_heads must divide num_attention_heads")
    if "rope_theta" in config:
        theta = config["rope_theta"]
        if type(theta) not in (int, float) or not math.isfinite(theta) or theta <= 0:
            raise ValueError("rope_theta must be a positive finite number")
    if "tie_word_embeddings" in config and type(config["tie_word_embeddings"]) is not bool:
        raise ValueError("tie_word_embeddings must be a boolean")
    return {
        "ok": True,
        "model_type": config["model_type"],
        "layers": config["num_hidden_layers"],
        "hidden_size": config["hidden_size"],
        "attention_heads": config["num_attention_heads"],
        "kv_heads": config["num_key_value_heads"],
        "queries_per_kv_head": config["num_attention_heads"] // config["num_key_value_heads"],
        "context_tokens": config["max_position_embeddings"],
    }


def audit_file(path: Path) -> dict:
    try:
        with Path(path).open(encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read model config: {exc}") from exc
    return audit_config(config)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = audit_file(args.path)
    except ValueError as exc:
        parser.exit(1, f"Model config audit failed: {exc}\n")
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(
            f"OK: {report['layers']} layers, {report['attention_heads']} query heads, "
            f"{report['kv_heads']} KV heads, {report['context_tokens']} context tokens"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
