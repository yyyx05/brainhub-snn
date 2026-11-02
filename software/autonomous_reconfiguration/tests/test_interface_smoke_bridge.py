from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock


COMPONENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPONENT))

from snn_reconfig.decision import DecisionEngine, load_json
from snn_reconfig.interface_smoke_bridge import FixtureReplayTransport, check_transcript


ENGINE = DecisionEngine.from_files(
    COMPONENT / "catalog/hardware_profiles.json", COMPONENT / "catalog/fingerprint_prototypes.json"
)
EXAMPLES = ("nmnist_query.json", "shd_query.json", "dvsgesture_query.json", "nmnist_query.json")


def decision(example: str = "nmnist_query.json", learning: str = "off") -> dict:
    return ENGINE.decide(load_json(COMPONENT / "examples" / example), learning)


def planned_sequence() -> FixtureReplayTransport:
    transport = FixtureReplayTransport()
    for example in EXAMPLES:
        transport.plan(decision(example))
    return transport


def synthetic_transcript(transport: FixtureReplayTransport) -> str:
    """Parser unit fixture only, never actual DUT evidence."""
    lines = [f"IF_ACK,{index},{op},{code},{data}"
             for index, (op, code, data) in enumerate(transport.commands)]
    lines += [f"IF_PROFILE,{code},{index + 1},5" for index, code in enumerate((1, 16, 32, 1))]
    lines += ["IF_OBSERVATIONS,1,80,80,0,0", "IF_OBSERVATIONS,16,0,0,80,0",
              "IF_OBSERVATIONS,32,0,0,0,1", "IF_OBSERVATIONS,1,80,80,0,0"]
    lines += [f"IF_DRAIN,{code},5" for code in (1, 16, 32, 1)]
    return "\n".join(lines + ["IF_COUNTS,1,4,3,1,4,1,4,2", "IF_PASS"])


class InterfaceSmokeBridgeTest(unittest.TestCase):
    def test_exact_fixed_sequence_and_descriptor_families(self) -> None:
        transport = planned_sequence()
        self.assertEqual(len(transport.commands), 32)
        self.assertEqual(transport.commands[5], (6, 1, 80))
        self.assertEqual(transport.commands[13], (6, 16, 80))
        self.assertEqual(transport.commands[21], (6, 32, 0xC0801))
        self.assertEqual([s["learning_scope"] for s in transport.selections],
                         ["W2 only", "M2 Wout only", "classifier only", "W2 only"])
        self.assertTrue(all(s["execution"]["status"] == "simulation_replay_pending"
                            for s in transport.selections))

    def test_reject_in_middle_is_atomic(self) -> None:
        transport = FixtureReplayTransport()
        transport.plan(decision("shd_query.json"))
        before = copy.deepcopy(transport.commands)
        with self.assertRaisesRegex(ValueError, "rejected"):
            transport.plan(decision("smnist_unsupported_query.json"))
        self.assertEqual(transport.commands, before)
        transport.plan(decision("dvsgesture_query.json"))
        self.assertEqual(len(transport.commands), 16)

    def test_learning_on_has_no_transport_side_effect(self) -> None:
        for example in EXAMPLES[:3]:
            transport = FixtureReplayTransport()
            with self.subTest(example=example), self.assertRaisesRegex(ValueError, "learning-on"):
                transport.plan(decision(example, "on"))
            self.assertEqual(transport.commands, [])

    def test_wrong_binding_or_writable_scope_is_rejected_before_stop(self) -> None:
        for field, value in (("profile_id", "unregistered"), ("profile_code", "0x02"),
                             ("dispatch_contract_id", "srnn_native_stream_v1")):
            malformed = decision()
            malformed["selected_profile"][field] = value
            transport = FixtureReplayTransport()
            with self.subTest(field=field), self.assertRaises(ValueError):
                transport.plan(malformed)
            self.assertEqual(transport.commands, [])
        malformed = decision()
        malformed["selected_profile"]["learning_scope"]["trainable_parameters"] = "W0+W2"
        transport = FixtureReplayTransport()
        with self.assertRaisesRegex(ValueError, "mask"):
            transport.plan(malformed)
        self.assertEqual(transport.commands, [])

    def test_plan_metadata_mismatch_is_rejected_before_stop(self) -> None:
        malformed = decision()
        malformed["hardware_plan"]["profile_opcode"] = "0x10"
        transport = FixtureReplayTransport()
        with self.assertRaises(ValueError):
            transport.plan(malformed)
        self.assertEqual(transport.commands, [])

    def test_out_of_order_call_or_empty_program_is_rejected(self) -> None:
        transport = FixtureReplayTransport()
        with self.assertRaises(ValueError):
            transport.wait_for_quiescent()
        with self.assertRaises(ValueError):
            transport.write_program(Mock())

    def test_serialization_is_ascii_native_replay_not_json_or_mock(self) -> None:
        transport = planned_sequence()
        path = Mock()
        transport.write_program(path)
        text = path.write_text.call_args.args[0]
        self.assertEqual(len(text.splitlines()), 32)
        self.assertEqual(text.splitlines()[21], "6 32 788481")
        self.assertEqual(path.write_text.call_args.kwargs, {"encoding": "ascii"})

    def test_parser_accepts_synthetic_unit_fixture_only_with_all_evidence(self) -> None:
        transport = planned_sequence()
        result = check_transcript(synthetic_transcript(transport), transport.commands)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["acknowledged_commands"], 32)

    def test_parser_rejects_missing_duplicate_reordered_and_reset_evidence(self) -> None:
        transport = planned_sequence()
        log = synthetic_transcript(transport)
        for mutated in (log.replace("IF_PASS", ""), log.replace("IF_ACK,0,1,1,0", ""),
                        log + "\nIF_ACK,0,1,1,0", log.replace("IF_PROFILE,16,2,5", "IF_PROFILE,16,1,5"),
                        log.replace("IF_COUNTS,1,", "IF_COUNTS,2,"),
                        log.replace("IF_DRAIN,32,5", "IF_DRAIN,32,0"), log + "\nIF_FATAL bad DUT"):
            with self.subTest(mutated=mutated[-40:]):
                self.assertEqual(check_transcript(mutated, transport.commands)["status"], "FAIL")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            check_transcript(log + "\nIF_COUNTS,1,4,3,1,4,1,4,2", transport.commands)


if __name__ == "__main__":
    unittest.main()
