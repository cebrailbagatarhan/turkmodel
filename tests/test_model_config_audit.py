"""Tests for the offline model configuration shape audit."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "llm-8b-tr-en" / "configs" / "audit_model_config.py"
CONFIG = SCRIPT.with_name("model_config.json")
spec = importlib.util.spec_from_file_location("audit_model_config", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

SAMPLE = {
    "model_type": "llama",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "max_position_embeddings": 1024,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
}


class ModelConfigAuditTests(unittest.TestCase):
    def test_valid_grouped_query_attention(self):
        report = module.audit_config(SAMPLE)
        self.assertTrue(report["ok"])
        self.assertEqual(report["queries_per_kv_head"], 4)
        self.assertEqual(report["context_tokens"], 1024)

    def test_rejects_invalid_positive_integers(self):
        for key in module.REQUIRED_POSITIVE_INTS:
            for value in (0, -1, True, 1.5, "8", None):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, key):
                        module.audit_config({**SAMPLE, key: value})

    def test_rejects_attention_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "head_dim"):
            module.audit_config({**SAMPLE, "head_dim": 16})

    def test_rejects_invalid_kv_head_divisibility(self):
        with self.assertRaisesRegex(ValueError, "divide"):
            module.audit_config({**SAMPLE, "num_key_value_heads": 3})

    def test_rejects_invalid_optional_types(self):
        for value in (float("nan"), float("inf"), 0, -1, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "rope_theta"):
                    module.audit_config({**SAMPLE, "rope_theta": value})
        with self.assertRaisesRegex(ValueError, "tie_word_embeddings"):
            module.audit_config({**SAMPLE, "tie_word_embeddings": "false"})

    def test_rejects_non_object_and_malformed_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for value in ("[]", "{broken"):
                path.write_text(value, encoding="utf-8")
                with self.assertRaises(ValueError):
                    module.audit_file(path)

    def test_checked_in_config_and_cli(self):
        report = module.audit_file(CONFIG)
        self.assertEqual(report["queries_per_kv_head"], 4)
        self.assertEqual(report["layers"], 32)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(CONFIG), "--json"],
            check=True, capture_output=True, text=True
        )
        self.assertEqual(json.loads(result.stdout), report)


if __name__ == "__main__":
    unittest.main()
