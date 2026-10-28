"""No-GPU tests for the offline training budget planner."""
import importlib.util
import math
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "llm-8b-tr-en" / "training" / "budget_planner.py"
spec = importlib.util.spec_from_file_location("budget_planner", SCRIPT)
budget_planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget_planner)


class BudgetPlannerTests(unittest.TestCase):
    def test_known_training_run_token_count(self):
        result = budget_planner.plan(steps=400, micro_batch=4, grad_accum=4, seq_len=2048)
        self.assertEqual(result["tokens_per_optimizer_step"], 32768)
        self.assertEqual(result["planned_tokens"], 13107200)
        self.assertIsNone(result["estimated_wall_hours"])

    def test_multi_gpu_scaling(self):
        result = budget_planner.plan(steps=2, micro_batch=2, grad_accum=3, seq_len=100, gpus=4)
        self.assertEqual(result["planned_tokens"], 4800)

    def test_time_and_cost_use_cluster_throughput(self):
        result = budget_planner.plan(steps=10, micro_batch=1, grad_accum=1,
                                     seq_len=360, gpus=2, tokens_per_second=2,
                                     gpu_hour_price=3)
        self.assertEqual(result["estimated_wall_hours"], 1)
        self.assertEqual(result["estimated_gpu_hours"], 2)
        self.assertEqual(result["estimated_cost"], 6)

    def test_reject_zero_and_negative_dimensions(self):
        for steps in (0, -1):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                budget_planner.plan(steps=steps, micro_batch=1, grad_accum=1, seq_len=1)

    def test_reject_nonfinite_and_zero_throughput(self):
        for value in (0, -1, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                budget_planner.plan(steps=1, micro_batch=1, grad_accum=1, seq_len=1,
                                    tokens_per_second=value)

    def test_cost_requires_throughput(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            budget_planner.plan(steps=1, micro_batch=1, grad_accum=1,
                                seq_len=1, gpu_hour_price=3)

    def test_cli_json_and_no_external_dependencies(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--steps", "2", "--micro-batch", "1",
             "--grad-accum", "1", "--seq-len", "10"],
            capture_output=True, text=True, check=True)
        self.assertIn('"planned_tokens": 20', completed.stdout)

    def test_cli_rejects_invalid_throughput(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--steps", "2", "--micro-batch", "1",
             "--grad-accum", "1", "--seq-len", "10", "--tokens-per-second", "0"],
            capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)


    def test_overhead_adjusts_time_gpu_hours_and_cost_not_tokens(self):
        result = budget_planner.plan(steps=1, micro_batch=1, grad_accum=1,
                                     seq_len=3600, gpus=2, tokens_per_second=2,
                                     gpu_hour_price=3, overhead_percent=25)
        self.assertEqual(result["planned_tokens"], 7200)
        self.assertEqual(result["estimated_wall_hours"], 1.25)
        self.assertEqual(result["estimated_gpu_hours"], 2.5)
        self.assertEqual(result["estimated_cost"], 7.5)

    def test_default_overhead_preserves_existing_estimates(self):
        result = budget_planner.plan(steps=1, micro_batch=1, grad_accum=1,
                                     seq_len=3600, tokens_per_second=1,
                                     gpu_hour_price=2)
        self.assertEqual(result["estimated_wall_hours"], 1)
        self.assertEqual(result["estimated_cost"], 2)
        self.assertEqual(result["overhead_percent"], 0)

    def test_invalid_overhead_rejected(self):
        for value in (-1, 101, math.nan, math.inf, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                budget_planner.plan(steps=1, micro_batch=1, grad_accum=1,
                                    seq_len=1, overhead_percent=value)

    def test_cli_overhead_and_invalid_limit(self):
        command = [sys.executable, str(SCRIPT), "--steps", "1",
                   "--micro-batch", "1", "--grad-accum", "1",
                   "--seq-len", "3600", "--tokens-per-second", "1"]
        valid = subprocess.run(command + ["--overhead-percent", "20"],
                               capture_output=True, text=True)
        self.assertEqual(valid.returncode, 0, valid.stderr)
        self.assertIn('"estimated_wall_hours": 1.2', valid.stdout)
        invalid = subprocess.run(command + ["--overhead-percent", "101"],
                                 capture_output=True, text=True)
        self.assertNotEqual(invalid.returncode, 0)


if __name__ == "__main__":
    unittest.main()
