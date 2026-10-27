"""Acceptance must reject missing conditions, resets and undefined gait cycles."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/amp_tools"))
from summarize_amp_acceptance import auxiliary_gates, braking_report
from analyze_gait import metrics


class AcceptanceTests(unittest.TestCase):
    def data(self, count=18):
        fields = ["command_requested", "command_applied", "time_s", "episode_id", "done",
                  "vx", "vy", "wz", "active_stage"]
        data = np.zeros(count, dtype=[(k, float) for k in fields])
        data["time_s"] = (np.arange(count)+1)*.02
        data["command_requested"][:4] = .4
        data["command_applied"][:4] = .4
        return data

    def test_stop_uses_confirmation_time_and_planar_norm(self):
        data = self.data()
        result = braking_report(data, .02)["stops"][0]
        self.assertAlmostEqual(result["confirmation_time_s"], .2)
        self.assertAlmostEqual(result["entry_time_for_02s_hold_s"], .02)
        data["vx"][4:] = data["vy"][4:] = .019
        self.assertIsNone(braking_report(data, .02)["stops"][0]["confirmation_time_s"])

    def test_quiet_samples_cannot_be_joined_across_episode_reset(self):
        data = self.data()
        data["done"][10] = 1
        data["episode_id"][11:] = 1
        self.assertIsNone(braking_report(data, .02)["stops"][0]["confirmation_time_s"])
        data["command_requested"] = .4
        data["command_applied"] = 0
        self.assertEqual(braking_report(data, .02)["stops"], [])

    def test_missing_rsi_speeds_and_resting_backward_policy_cannot_pass(self):
        cases = []
        for mode, vx in [("rsi",None),("standing",-.2),("standing",.1),("standing",.65)]:
            cases.append(dict(seed=5, reset_mode=mode, vx=vx,
                              evaluation=dict(rows=3000 if vx is None else 1000,
                                              non_timeout_terminations=0,
                                              walking=dict(samples=825, mean_vx=vx, rmse_vx=0))))
        self.assertTrue(all(c["passed"] for c in auxiliary_gates(cases, "exp1.5")))
        self.assertEqual(sum(not c["collected"] for c in auxiliary_gates(cases, "exp1.6")), 2)
        cases[1]["evaluation"]["walking"].update(mean_vx=0, rmse_vx=.2)
        self.assertFalse(auxiliary_gates(cases, "exp1.5")[1]["passed"])

    def test_constant_pose_has_no_measured_gait_period(self):
        names = [s+"_"+j+"_joint" for s in ("left","right")
                 for j in ("hip_pitch","knee_pitch","ankle_pitch")]
        result = metrics(np.zeros((200,6)), np.zeros((200,2,3)), np.zeros((200,2)),
                         np.zeros(200), names, .02)
        self.assertIsNone(result["same_leg_cycle_period_s"])
        self.assertIsNone(result["trunk_forward_distance_per_same_leg_cycle_m"])


if __name__ == "__main__":
    unittest.main()
