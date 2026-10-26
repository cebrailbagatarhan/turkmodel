import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("report_summary.py")
spec = importlib.util.spec_from_file_location("report_summary", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SummaryTests(unittest.TestCase):
    def test_category_and_overall_scores(self):
        data = {"results": {"grammar": [{"score": 100}, {"score": 0}], "creative": [{"score": 75}]}}
        summary = module.summarize(data)
        self.assertEqual(summary["total_tests"], 3)
        self.assertAlmostEqual(summary["average_score"], 175 / 3)
        self.assertEqual(summary["categories"]["grammar"]["average_score"], 50)

    def test_invalid_scores_rejected(self):
        for score in (float("nan"), float("inf"), -1, 101, True, "50"):
            with self.subTest(score=score), self.assertRaises(ValueError):
                module.summarize({"results": {"x": [{"score": score}]}})

    def test_empty_report_rejected(self):
        with self.assertRaises(ValueError):
            module.summarize({"results": {}})

    def test_cli_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            path.write_text(json.dumps({"results": {"grammar": [{"score": 60}]}}), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(module.main([str(path), "--json"]), 0)
            self.assertEqual(json.loads(output.getvalue())["total_tests"], 1)

    def test_missing_file_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                module.main(["/missing/eval.json"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
