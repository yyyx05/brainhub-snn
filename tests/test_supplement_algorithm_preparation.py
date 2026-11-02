"""Standard-library-only preparation checks; no torch, data, or GPU load."""

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.supplement_algorithm_residual import (
    SEEDS, check_input_paths, commands, historical_path, parse_args, reproduction_error,
)


class PreparationTests(unittest.TestCase):
    def historical(self, topology, seed):
        return json.loads(historical_path(topology, seed).read_text(encoding="utf-8"))

    def test_comparison_identity_on_six_historical_fixtures(self):
        # This tests the comparator, not a new episode reproduction.
        for topology in ("fc", "srnn"):
            for seed in SEEDS[topology]:
                with self.subTest(topology=topology, seed=seed):
                    result = reproduction_error(topology, seed, self.historical(topology, seed))
                    self.assertEqual(result["status"], "exact_match")
                    self.assertEqual(result["max_accuracy_error_points"], 0)
                    self.assertGreater(result["compared_scalar_fields"], 50)

    def test_fc_accuracy_prediction_and_code_errors(self):
        report = self.historical("fc", 38472)
        row = report["masks"][0]
        row["shifted_after"]["accuracy_percent"] += 1
        row["shifted_after"]["predictions"][0] = -1
        row["optimizer_diagnostics"]["code_writes"] += 2
        row["adaptation_seconds_wall"] = 123456
        result = reproduction_error("fc", 38472, report)
        self.assertEqual(result["status"], "differences")
        self.assertEqual(result["max_accuracy_error_points"], 1)
        self.assertEqual(result["prediction_mismatches"]["row.shifted_after.predictions"], 1)
        errors = {item["field"]: item for item in result["differing_fields"]}
        self.assertEqual(errors["row.optimizer_diagnostics.code_writes"]["absolute_error"], 2)
        self.assertNotIn("row.adaptation_seconds_wall", errors)

    def test_srnn_fraction_accuracy_converted_to_points(self):
        report = self.historical("srnn", 128474)
        row = next(row for row in report["optimizers"] if row["optimizer"] == "direct_response")
        row["evaluations"][2]["shifted"]["accuracy"] += 0.01
        result = reproduction_error("srnn", 128474, report)
        self.assertAlmostEqual(result["max_accuracy_error_points"], 1)
        self.assertIsNone(result["prediction_mismatches"])

    def test_missing_and_nonfinite_never_claim_exact_reproduction(self):
        report = self.historical("fc", 38472)
        for alteration in ("missing", "nonfinite"):
            changed = copy.deepcopy(report)
            metric = changed["masks"][0]["shifted_after"]
            if alteration == "missing":
                del metric["predictions"]
            else:
                metric["accuracy_percent"] = float("nan")
            result = reproduction_error("fc", 38472, changed)
            self.assertEqual(result["status"], "incomplete")
            self.assertTrue(result["missing_fields"])

    def test_missing_formal_srnn_inputs_fail_before_torch(self):
        args = parse_args(["--topology", "srnn", "--data-root", "D:/datasets",
                           "--handoff", "missing_formal_package", "--device", "cpu"])
        with patch.object(Path, "exists", return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, "B64_FORMAL_SEED_A.json"):
                check_input_paths(args)

    def test_fc_and_srnn_commands_preserve_three_seeds_and_device(self):
        for topology, device in (("fc", "cuda"), ("srnn", "cpu")):
            argv = ["--topology", topology, "--data-root", "D:/datasets"]
            if topology == "srnn":
                argv += ["--handoff", "declared_formal_package"]
            args = parse_args(argv)
            self.assertEqual(args.device, device)
            jobs = list(commands(args))
            self.assertFalse(args.run)
            self.assertEqual(len(jobs), 6)
            self.assertEqual([int(job[job.index("--seed") + 1]) for job in jobs],
                             [seed for seed in SEEDS[topology] for _ in range(2)])
            self.assertEqual([job[job.index("--arm") + 1] for job in jobs], ["on", "drop"] * 3)
            self.assertTrue(all(job[job.index("--device") + 1] == device for job in jobs))


if __name__ == "__main__":
    unittest.main()
