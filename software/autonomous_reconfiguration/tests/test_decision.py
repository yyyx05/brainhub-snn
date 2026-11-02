from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snn_reconfig.decision import DecisionEngine


ENGINE = DecisionEngine.from_files(
    ROOT / "catalog" / "hardware_profiles.json",
    ROOT / "catalog" / "fingerprint_prototypes.json",
)


def load_example(name: str) -> dict:
    return json.loads((ROOT / "examples" / name).read_text(encoding="utf-8"))


class DecisionEngineTest(unittest.TestCase):
    def test_nmnist_selects_fc(self) -> None:
        result = ENGINE.decide(load_example("nmnist_query.json"), "off")
        self.assertEqual(result["status"], "hardware_profile_selected")
        self.assertEqual(result["selected_profile"]["profile_code"], "0x01")
        self.assertEqual(
            result["selected_profile"]["dispatch_contract_id"],
            "fc_native_stream_v1",
        )
        self.assertEqual(
            result["hardware_plan"]["profile_dispatch_contract"],
            "fc_native_stream_v1",
        )
        self.assertFalse(result["learning_decision"]["enabled"])

    def test_shd_selects_srnn_and_explicit_learning(self) -> None:
        result = ENGINE.decide(load_example("shd_query.json"), "on")
        self.assertEqual(result["selected_profile"]["profile_code"], "0x10")
        self.assertTrue(result["learning_decision"]["enabled"])

    def test_dvsgesture_selects_scnn(self) -> None:
        result = ENGINE.decide(load_example("dvsgesture_query.json"), "auto")
        self.assertEqual(result["selected_profile"]["profile_code"], "0x20")
        self.assertFalse(result["learning_decision"]["enabled"])

    def test_auto_learning_uses_bounded_unvalidated_policy(self) -> None:
        request = load_example("nmnist_query.json")
        request["runtime_health"] = {
            "baseline_confidence": 0.91,
            "observed_confidence": 0.76,
            "drift_score": 0.30,
        }
        result = ENGINE.decide(request, "auto")
        self.assertTrue(result["learning_decision"]["enabled"])
        self.assertEqual(
            result["learning_decision"]["policy_status"],
            "engineering_policy_unvalidated",
        )

    def test_unsupported_task_emits_no_opcode(self) -> None:
        result = ENGINE.decide(load_example("smnist_unsupported_query.json"), "auto")
        self.assertEqual(result["status"], "requires_accelerator_extension")
        self.assertIsNone(result["selected_profile"])
        self.assertIsNone(result["hardware_plan"])

    def test_cifar10dvs_does_not_reuse_dvsgesture_opcode(self) -> None:
        result = ENGINE.decide(load_example("cifar10dvs_unsupported_query.json"), "auto")
        self.assertEqual(result["status"], "requires_accelerator_extension")
        self.assertIsNone(result["hardware_plan"])

    def test_spatial_shape_is_a_hard_constraint(self) -> None:
        request = load_example("nmnist_query.json")
        request["fingerprint"]["spatial_shape"] = [32, 32]
        result = ENGINE.decide(request, "off")
        self.assertEqual(result["status"], "requires_accelerator_extension")


if __name__ == "__main__":
    unittest.main()
