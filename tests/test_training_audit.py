"""Regression checks for the read-only training metadata audit."""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from results.training_audit import audit_file, audit_record


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = {
    "total_params": 1000,
    "training_steps": 4,
    "total_tokens_processed": 800,
    "training_time_minutes": 2.0,
    "final_loss": 2.5,
}


class TrainingAuditTests(unittest.TestCase):
    def test_derived_values_are_labeled_and_correct(self):
        report = audit_record(SAMPLE)
        self.assertEqual(report["derived"]["tokens_per_step"], 200)
        self.assertAlmostEqual(report["derived"]["tokens_per_second"], 800 / 120)
        self.assertEqual(report["derived"]["tokens_per_parameter"], 0.8)
        self.assertNotIn("eval_loss", report)

    def test_rejects_invalid_counts_and_booleans(self):
        for key in ("total_params", "training_steps", "total_tokens_processed"):
            for value in (0, -1, True, 1.5, None):
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        audit_record({**SAMPLE, key: value})

    def test_rejects_nonfinite_or_negative_measurements(self):
        for key in ("training_time_minutes", "final_loss"):
            for value in (float("nan"), float("inf"), -1, "5", True):
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        audit_record({**SAMPLE, key: value})
        self.assertEqual(audit_record({**SAMPLE, "final_loss": 0})["final_training_loss"], 0)

    def test_rejects_malformed_or_non_object_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            for raw in ('{bad', '[]', '"string"'):
                path.write_text(raw, encoding="utf-8")
                with self.assertRaises(ValueError):
                    audit_file(path)

    def test_checked_in_training_report_and_cli(self):
        path = ROOT / "results" / "training_info.json"
        report = audit_file(path)
        self.assertEqual(report["training_steps"], 400)
        self.assertEqual(report["total_tokens_processed"], 13107200)
        result = subprocess.run(
            [sys.executable, str(ROOT / "results" / "training_audit.py"), str(path), "--json"],
            check=True, capture_output=True, text=True
        )
        self.assertEqual(json.loads(result.stdout), report)

    def test_cli_coverage_gate_passes_and_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(SAMPLE), encoding="utf-8")
            command = [sys.executable, str(ROOT / "results" / "training_audit.py"),
                       str(path), "--json", "--min-tokens-per-parameter"]
            passing = subprocess.run(command + ["0.8"], capture_output=True, text=True)
            self.assertEqual(passing.returncode, 0, passing.stderr)
            self.assertEqual(json.loads(passing.stdout)["coverage_check"],
                             {"minimum_tokens_per_parameter": 0.8, "passed": True})
            failing = subprocess.run(command + ["0.9"], capture_output=True, text=True)
            self.assertEqual(failing.returncode, 1)
            self.assertFalse(json.loads(failing.stdout)["coverage_check"]["passed"])
            self.assertIn("below", failing.stderr)

    def test_cli_rejects_invalid_coverage_thresholds(self):
        script = str(ROOT / "results" / "training_audit.py")
        for threshold in ("-1", "nan", "inf", "-inf", "oops"):
            with self.subTest(threshold=threshold):
                completed = subprocess.run(
                    [sys.executable, script, f"--min-tokens-per-parameter={threshold}"],
                    capture_output=True, text=True)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("finite non-negative", completed.stderr)


if __name__ == "__main__":
    unittest.main()
