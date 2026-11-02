from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snn_reconfig.decision import DecisionEngine, load_json
from snn_reconfig.runtime_bridge import TraceTransport, execute_selected_profile


ENGINE = DecisionEngine.from_files(
    ROOT / "catalog" / "hardware_profiles.json",
    ROOT / "catalog" / "fingerprint_prototypes.json",
)
DISPATCH_CONTRACTS = {
    item["contract_id"]: item
    for item in load_json(ROOT / "catalog" / "profile_dispatch_contracts.json")["contracts"]
}


class RecordingTransport:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def stop_accepting_new_work(self) -> None:
        self.calls.append(("stop",))

    def wait_for_quiescent(self) -> None:
        self.calls.append(("quiescent",))

    def load_deployment_assets(self, assets: list[dict]) -> None:
        self.calls.append(("load", len(assets)))

    def request_profile(self, opcode: str, learning_enable: bool) -> None:
        self.calls.append(("profile", opcode, learning_enable))

    def wait_for_profile_alignment(self) -> None:
        self.calls.append(("aligned",))

    def submit_runtime_descriptors(
        self, profile_id: str, dispatch_contract_id: str
    ) -> None:
        self.calls.append(("descriptors", profile_id, dispatch_contract_id))

    def wait_for_learning_commit(self) -> None:
        self.calls.append(("commit",))

    def continue_without_learning_commit(self) -> None:
        self.calls.append(("continue",))

    def resume_accepting_work(self) -> None:
        self.calls.append(("resume",))


class RuntimeBridgeTest(unittest.TestCase):
    def _decision(self, example: str, learning: str) -> dict:
        return ENGINE.decide(load_json(ROOT / "examples" / example), learning)

    def test_learning_sequence_waits_for_commit_before_resume(self) -> None:
        transport = RecordingTransport()
        result = execute_selected_profile(self._decision("shd_query.json", "on"), transport)
        self.assertEqual(result["status"], "execution_sequence_completed")
        self.assertEqual(
            transport.calls,
            [
                ("stop",),
                ("quiescent",),
                ("load", 4),
                ("profile", "0x10", True),
                ("aligned",),
                ("descriptors", "srnn_shd_h450_m2", "srnn_native_stream_v1"),
                ("commit",),
                ("resume",),
            ],
        )

    def test_inference_sequence_does_not_wait_for_commit(self) -> None:
        transport = RecordingTransport()
        execute_selected_profile(self._decision("nmnist_query.json", "off"), transport)
        self.assertEqual(transport.calls[-2:], [("continue",), ("resume",)])

    def test_rejected_task_cannot_reach_transport(self) -> None:
        transport = RecordingTransport()
        decision = self._decision("smnist_unsupported_query.json", "off")
        with self.assertRaises(ValueError):
            execute_selected_profile(decision, transport)
        self.assertEqual(transport.calls, [])

    def test_trace_transport_records_profile_specific_request_family(self) -> None:
        transport = TraceTransport(DISPATCH_CONTRACTS)
        execute_selected_profile(self._decision("dvsgesture_query.json", "off"), transport)
        dispatch = next(
            event
            for event in transport.events
            if event["operation"] == "submit_runtime_descriptors"
        )
        self.assertEqual(dispatch["profile_id"], "scnn_dvsgesture_vgg9")
        self.assertEqual(
            dispatch["request_family"], "scnn_graph_descriptor_and_frame_service"
        )

    def test_all_profiles_preserve_fixed_learning_scope_and_commit_order(self) -> None:
        cases = {
            "nmnist_query.json": ("W2 only", "fc_native_stream_v1"),
            "shd_query.json": ("M2 Wout only", "srnn_native_stream_v1"),
            "dvsgesture_query.json": (
                "classifier only",
                "scnn_graph_frame_service_v1",
            ),
        }
        for example, (expected_scope, expected_contract) in cases.items():
            with self.subTest(example=example):
                decision = self._decision(example, "on")
                self.assertEqual(
                    decision["selected_profile"]["learning_scope"][
                        "trainable_parameters"
                    ],
                    expected_scope,
                )
                self.assertTrue(decision["hardware_plan"]["learning_enable"])
                self.assertEqual(
                    decision["hardware_plan"]["profile_dispatch_contract"],
                    expected_contract,
                )

                transport = TraceTransport(DISPATCH_CONTRACTS)
                result = execute_selected_profile(decision, transport)
                self.assertEqual(result["status"], "execution_sequence_completed")
                operations = [event["operation"] for event in transport.events]
                self.assertLess(
                    operations.index("wait_for_learning_commit_and_published_version"),
                    operations.index("resume_accepting_work"),
                )


if __name__ == "__main__":
    unittest.main()
