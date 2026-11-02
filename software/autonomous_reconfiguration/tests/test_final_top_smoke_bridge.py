"""Directed bridge/parser unit tests. Synthetic log fixtures are not DUT runs."""

import copy
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

COMPONENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPONENT))
from snn_reconfig.decision import DecisionEngine, load_json
from snn_reconfig.final_top_smoke_bridge import (
    CONFIG, DISPATCH, DENSE_FIELDS, DESCRIPTOR_FIELDS, FinalTopReplayTransport,
    check_transcript, scnn_descriptors,
)

ENGINE = DecisionEngine.from_files(COMPONENT / "catalog/hardware_profiles.json",
                                   COMPONENT / "catalog/fingerprint_prototypes.json")


def decision(example="nmnist_query.json", learning="off"):
    return copy.deepcopy(ENGINE.decide(load_json(COMPONENT / "examples" / example), learning))


def sequence():
    transport = FinalTopReplayTransport()
    for example in ("nmnist_query.json", "shd_query.json", "smnist_unsupported_query.json",
                    "dvsgesture_query.json", "nmnist_query.json"):
        if "unsupported" in example:
            before = copy.deepcopy(transport.commands)
            with unittest.TestCase().assertRaises(ValueError):
                transport.plan(decision(example))
            assert transport.commands == before
        else:
            transport.plan(decision(example))
    return transport


def parser_fixture(transport):
    lines = [f"FT_ACK,{index},{op},{code},{len(payload)}"
             for index, (op, code, payload) in enumerate(transport.commands)]
    lines += ["FT_CONFIG,1,1,1,0", "FT_CONFIG,16,2,2,0", "FT_CONFIG,32,4,3,0", "FT_CONFIG,1,1,1,0"]
    lines += [f"FT_DRAIN,{code},0" for code in (1,16,32,1)]
    lines += ["FT_DENSE,1,2,1,1", "FT_DENSE,16,2,1,1", "FT_DENSE,1,2,1,1",
              "FT_SCNN,32,4,68,68,4,4,144,144,144,144,20,40,0",
              "FT_COUNTS,1,3,3,6,32,0,0,0,0", "FT_PASS"]
    return "\n".join(lines)


class FinalTopSmokeBridgeTest(unittest.TestCase):
    def test_actual_decision_sequence_and_separate_parameter_owner_encoding(self):
        transport = sequence()
        self.assertEqual(len(transport.commands), 32)
        configs = [payload for op, code, payload in transport.commands if op == CONFIG]
        self.assertEqual([p[0] for p in configs], [1,2,4,1])
        self.assertEqual([p[2] for p in configs], [1,16,32,1])
        self.assertEqual([p[1] for p in configs], [0,0,1,0])
        self.assertTrue(all(s["execution"]["status"] == "simulation_replay_pending" for s in transport.selections))
        self.assertEqual([s["writable_scope"] for s in transport.selections],
                         ["W2 only", "M2 Wout only", "classifier only", "W2 only"])

    def test_full_descriptor_serialization_not_tag_proxy(self):
        records = scnn_descriptors()
        self.assertEqual(len(DESCRIPTOR_FIELDS), 20)
        self.assertEqual(len(records), 32)
        self.assertEqual([(r["sample"], r["timestep"], r["last"]) for r in records[7::8]],
                         [(0,0,1),(0,1,1),(1,0,1),(1,1,1)])
        transport = sequence()
        payload = transport.commands[21][2]
        self.assertEqual(len(payload), 740)
        self.assertEqual(payload[:4], (2,2,0,32))
        self.assertEqual(payload[4:27], tuple(records[0][k] for k in ("sample","timestep","last",*DESCRIPTOR_FIELDS)))
        for index in range(32):
            self.assertEqual(payload[4+index*23:4+(index+1)*23],
                             tuple(records[index][k] for k in ("sample","timestep","last",*DESCRIPTOR_FIELDS)))

    def test_dense_synthetic_native_fields(self):
        commands = sequence().commands
        for index, mode, base in ((5,0,64),(13,1,0),(29,0,64)):
            op, code, payload = commands[index]
            self.assertEqual(op, DISPATCH)
            self.assertEqual(len(payload), len(DENSE_FIELDS))
            fields = dict(zip(DENSE_FIELDS, payload))
            self.assertEqual((fields["mode"],fields["parameter_base"],fields["neuron_count"]), (mode,base,2))
            self.assertEqual((fields["expected_pre_state"], fields["expected_spike"]), (20,1))
            self.assertEqual(fields["recurrent_enable"], 0)
            self.assertEqual(payload[15:19], (0,0,0,0))

    def test_rejection_and_invalid_metadata_are_atomic(self):
        transport = sequence()
        before = copy.deepcopy(transport.commands)
        with self.assertRaises(ValueError):
            transport.plan(decision("smnist_unsupported_query.json"))
        self.assertEqual(transport.commands, before)
        for field, value in (("profile_code","0x02"), ("dispatch_contract_id","wrong"),
                             ("profile_id","unknown")):
            malformed = decision(); malformed["selected_profile"][field] = value
            fresh = FinalTopReplayTransport()
            with self.assertRaises(ValueError): fresh.plan(malformed)
            self.assertEqual(fresh.commands, [])
        malformed = decision(); malformed["hardware_plan"]["profile_opcode"] = "0x10"
        fresh = FinalTopReplayTransport()
        with self.assertRaises(ValueError): fresh.plan(malformed)
        self.assertEqual(fresh.commands, [])

    def test_learning_and_scope_changes_rejected(self):
        for example in ("nmnist_query.json","shd_query.json","dvsgesture_query.json"):
            fresh = FinalTopReplayTransport()
            with self.assertRaises(ValueError): fresh.plan(decision(example,"on"))
            self.assertEqual(fresh.commands, [])
        malformed = decision()
        malformed["selected_profile"]["learning_scope"]["trainable_parameters"] = "W0+W2"
        fresh = FinalTopReplayTransport()
        with self.assertRaisesRegex(ValueError, "scope"): fresh.plan(malformed)
        self.assertEqual(fresh.commands, [])

    def test_order_and_serialized_program(self):
        fresh = FinalTopReplayTransport()
        with self.assertRaises(ValueError): fresh.wait_for_quiescent()
        with self.assertRaises(ValueError): fresh.write_program(Mock())
        transport = sequence(); path = Mock(); transport.write_program(path)
        lines = path.write_text.call_args.args[0].splitlines()
        self.assertEqual(len(lines),32)
        for command, line in zip(transport.commands,lines):
            op, code, payload = command
            self.assertEqual(list(map(int,line.split())), [op,code,len(payload),*payload])

    def test_parser_requires_all_real_evidence(self):
        transport = sequence(); log = parser_fixture(transport)
        self.assertEqual(check_transcript(log,transport.commands)["status"], "PASS")
        for bad in (log.replace("FT_PASS",""), log + "\nFT_PASS", log.replace("FT_ACK,0,1,1,0",""),
                    log.replace("FT_COUNTS,1,","FT_COUNTS,2,"),
                    log.replace("FT_CONFIG,32,4,3,0","FT_CONFIG,32,4,2,0"),
                    log.replace("144,144,20,40,0","144,144,0,40,0"),
                    log.replace("FT_COUNTS,1,3,3,6,32,0,0,0,0","FT_COUNTS,1,3,3,6,32,1,0,0,0"),
                    log + "\nFT_FATAL failure"):
            self.assertEqual(check_transcript(bad,transport.commands)["status"], "FAIL")

    def test_existing_generator_and_resource_monitor_are_readonly(self):
        sys.path.insert(0,str(COMPONENT.parents[1] / "scripts"))
        from run_frontend_final_top_smoke import OVERRIDES, process_tree_rss
        from run_fc_final_counts import ansi_header, build_settings
        module, parameters, ports = ansi_header(COMPONENT.parents[1] / "rtl/top/unified_h40_final_composition_top_v1.sv")
        self.assertEqual(module,"unified_h40_final_composition_top_v1")
        self.assertTrue(set(OVERRIDES) <= {name for _,name,_ in parameters})
        self.assertIn("i_graph_descriptor_valid",{name for _,_,name in ports})
        build, _, _ = build_settings(Path("obj"),Path("sources.f"),jobs=2,top="tb_frontend_final_top_smoke")
        self.assertEqual(build[build.index("-j")+1],"2")
        import os
        self.assertGreater(process_tree_rss(os.getpid()),0)

    def test_runner_rejects_removed_executable_option_before_side_effects(self):
        sys.path.insert(0,str(COMPONENT.parents[1] / "scripts"))
        import run_frontend_final_top_smoke as runner
        error = io.StringIO()
        with patch.object(sys, "argv", ["runner", "--run-name", "unused",
                                        "--reuse-executable", "old.exe"]), \
                patch.object(runner, "prepare") as prepare, \
                patch.object(runner, "run_bounded") as run, \
                patch.object(Path, "mkdir") as mkdir, redirect_stderr(error):
            with self.assertRaises(SystemExit) as stopped:
                runner.main()
        self.assertEqual(stopped.exception.code, 2)
        self.assertIn("unrecognized arguments: --reuse-executable", error.getvalue())
        prepare.assert_not_called()
        run.assert_not_called()
        mkdir.assert_not_called()

    def test_verification_store_preference_fallback_and_missing(self):
        sys.path.insert(0,str(COMPONENT.parents[1] / "scripts"))
        from run_fc_final_counts import resolve_verification_store
        root = Path("compact-project")
        primary = root / "rtl/learning/unified_tp_shared_forward_rmw_store.sv"
        support = root / "verification/support/unified_tp_shared_forward_rmw_store.sv"
        for present, expected in (({primary, support}, primary), ({support}, support)):
            with self.subTest(present=present), patch.object(
                    Path, "is_file", autospec=True, side_effect=lambda path: path in present):
                self.assertEqual(resolve_verification_store(root), expected)
        with patch.object(Path, "is_file", return_value=False):
            with self.assertRaises(FileNotFoundError) as missing:
                resolve_verification_store(root)
        self.assertIn(str(primary), str(missing.exception))
        self.assertIn(str(support), str(missing.exception))

    def test_generated_tb_reads_resolved_store_header(self):
        sys.path.insert(0,str(COMPONENT.parents[1] / "scripts"))
        import run_fc_final_counts as helper
        support = Path("compact-project/verification/support/unified_tp_shared_forward_rmw_store.sv")
        # Parse the existing store header read-only; no SV copy or compile.
        store_header = helper.ansi_header(helper.resolve_verification_store())
        original_header = helper.ansi_header
        path = Mock()
        with patch.object(helper, "resolve_verification_store", return_value=support) as resolve, \
                patch.object(helper, "ansi_header", side_effect=lambda source: (
                    store_header if source == support else original_header(source))) as header:
            helper.generate_tb(path, top="tb_frontend_final_top_smoke",
                               body=COMPONENT.parents[1] / "tb/tb_frontend_final_top_smoke.sv")
        resolve.assert_called_once_with()
        self.assertEqual(header.call_args_list[1].args, (support,))
        content = path.write_text.call_args.args[0]
        self.assertIn("unified_h40_final_composition_top_v1 #(", content)
        self.assertIn("unified_tp_shared_forward_rmw_store #(", content)

    def test_runner_sources_include_store_once_in_both_layouts(self):
        sys.path.insert(0,str(COMPONENT.parents[1] / "scripts"))
        import run_frontend_final_top_smoke as runner
        root = Path("compact-project")
        primary = root / "rtl/learning/unified_tp_shared_forward_rmw_store.sv"
        support = root / "verification/support/unified_tp_shared_forward_rmw_store.sv"
        top = root / "rtl/top/unified_h40_final_composition_top_v1.sv"
        tb = root / "generated_tb.sv"
        for existing, present, store in (([top, primary], {primary, support}, primary),
                                         ([top], {support}, support)):
            with self.subTest(store=store), patch.object(runner, "ROOT", root), \
                    patch.object(Path, "glob", return_value=iter(existing)), \
                    patch.object(Path, "is_file", autospec=True,
                                 side_effect=lambda path: path in present):
                sources = runner.simulation_sources(tb)
            self.assertEqual(sources.count(store), 1)
            self.assertEqual(sources, [*sorted(existing), *([] if store in existing else [store]), tb])
            self.assertEqual(len(sources), len(set(sources)))

    def test_runner_always_builds_before_simulation(self):
        sys.path.insert(0,str(COMPONENT.parents[1] / "scripts"))
        import run_frontend_final_top_smoke as runner
        # Exercise main's control flow with all output and subprocess I/O mocked.
        # This is a runner unit test, not another RTL build or DUT simulation.
        output = MagicMock(spec=Path)
        output.__truediv__.return_value = output
        output.as_posix.return_value = "mock-output"
        root = MagicMock(spec=Path)
        root.__truediv__.return_value = output
        transport = sequence()
        with patch.object(sys, "argv", ["runner", "--run-name", "fresh", "--jobs", "2"]), \
                patch.object(runner, "OUTPUT", root), \
                patch.object(runner, "prepare", return_value=(transport, [{}, {}, {"commands_added":0}])), \
                patch.object(runner, "generate_tb"), \
                patch.object(runner, "build_settings", return_value=(
                    ["new-verilator", "--output-split", "3000"], Path("newly-built.exe"), {})), \
                patch.object(runner, "run_bounded", side_effect=[
                    {"exit_code":0}, {"exit_code":0}, {"exit_code":0}]) as run, \
                patch.object(runner, "check_transcript", return_value={"status":"PASS"}), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 0)
        self.assertEqual([call.args[2] for call in run.call_args_list],
                         ["unit_tests", "build", "simulation"])
        self.assertEqual(run.call_args_list[1].args[0][0], "new-verilator")
        self.assertEqual(run.call_args_list[2].args[0][0], "newly-built.exe")


if __name__ == "__main__":
    unittest.main()
