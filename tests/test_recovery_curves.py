"""Retained training curves must exclude the unsaved branch of an interrupted run."""
import importlib.util
from pathlib import Path
import sys
import unittest

TOOLS = Path(__file__).resolve().parents[1]/"scripts/amp_tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("collect_training_curves", TOOLS/"collect_training_curves.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RecoveryCurveTests(unittest.TestCase):
    def test_unsaved_updates_are_replaced_by_actual_recovery(self):
        old = {"reward": {"steps": [5399, 5400, 5401, 5497], "values": [1, 2, 999, 999]}}
        new = {"reward": {"steps": [5401, 5402, 6000], "values": [3, 4, 5]}}
        result = module.merge_recovered_series(old, new, 5400)["reward"]
        self.assertEqual(result["steps"], [5399, 5400, 5401, 5402, 6000])
        self.assertEqual(result["values"], [1, 2, 3, 4, 5])

    def test_zero_restart_and_duplicate_updates_are_rejected(self):
        for steps in ([1, 2], [5400, 5401], [5401, 5401]):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                module.merge_recovered_series({}, {"reward": {"steps": steps, "values": [0]*len(steps)}}, 5400)


if __name__ == "__main__":
    unittest.main()
