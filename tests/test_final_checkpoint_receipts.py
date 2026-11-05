"""Never mistake an uploaded input checkpoint for a completed training output."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts/amp_tools"))
spec = importlib.util.spec_from_file_location("fetch_final_amp", ROOT/"scripts/amp_tools/fetch_final_amp.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FinalCheckpointReceiptTests(unittest.TestCase):
    def test_concatenated_sdk_logs_and_exact_iteration(self):
        log = 'CHECKPOINT_SAVED {"iteration":5900,"bytes":10,"sha256":"'+"a"*64+'"}\n'
        log += 'prefixCHECKPOINT_SAVED {"iteration":6000,"bytes":20,"sha256":"'+"b"*64+'"}[SDK] upload\n'
        self.assertEqual(module.save_receipt(log, 6000)["bytes"], 20)
        with self.assertRaises(RuntimeError): module.save_receipt(log, 6100)

    def test_input_6000_does_not_prove_final_6000_was_uploaded(self):
        rows = [{"id": 1, "fileName": "input_exp112_model_6000.pt", "checkpoint": "6000"},
                {"id": 2, "fileName": "model_5900.pt", "checkpoint": "5900"}]
        with self.assertRaises(RuntimeError): module.final_model(rows, 6000)
        rows.append({"id": 3, "fileName": "model_6000.pt", "checkpoint": "6000"})
        self.assertEqual(module.final_model(rows, 6000)["id"], 3)

    def test_stopped_output_uses_uploaded_iteration_without_relabeling_budget(self):
        rows = [{"fileName": "input_exp113t_model_6000.pt", "checkpoint": "6000"},
                {"fileName": "model_5300.pt", "checkpoint": "5300"},
                {"fileName": "model_5200.pt", "checkpoint": "5200"}]
        self.assertEqual(module.stopped_model_iteration(rows, "Learning iteration 5322/6000", 6000), (5300, 5322))
        with self.assertRaises(ValueError):
            module.stopped_model_iteration(rows[:1], "Learning iteration 5322/6000", 6000)
        with self.assertRaises(ValueError):
            module.stopped_model_iteration(rows, "", 6000)


if __name__ == "__main__":
    unittest.main()
