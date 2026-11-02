import sys
import contextlib
import io
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_m2_compact_core as m
from run_m2_compact_synthesis import selected_tcl
from audit_m2_capacity_bounds import parse


class CompactCoreTests(unittest.TestCase):
    def source(self, relative):
        return Path(m.local_path(m.SOURCE/'rtl'/relative)).read_text()

    def test_preparation_copies_original_rom_and_keeps_rtl_unchanged(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result=m.prepare()
        relative='rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem'
        self.assertEqual((result/relative).read_bytes(),
                         Path(m.local_path(m.SOURCE/relative)).read_bytes())
        original=Path(m.local_path(m.OUTPUT))
        rtls=list((result/'rtl').rglob('*.sv'))
        self.assertEqual(len(rtls),108)
        for path in rtls:
            self.assertEqual(path.read_bytes(),(original/path.relative_to(result)).read_bytes())

    def test_parameter_chain(self):
        for relative, children in m.CHAIN.items():
            result = m.transform(relative, self.source(relative))
            self.assertIn('parameter int M2_CACHE_DEPTH = 578', result)
            self.assertEqual(result.count('.M2_CACHE_DEPTH(M2_CACHE_DEPTH)'), len(children))

    def test_both_address_consumers_use_capacity(self):
        adapter = 'learning/unified_srnn_m2_h40_forward_observation_adapter_v1.sv'
        scheduler = 'memory/unified_srnn_m2_h40_q14_cache_refill_scheduler_v1.sv'
        self.assertEqual(m.transform(adapter, self.source(adapter)).count('                M2_CACHE_DEPTH &&'), 2)
        self.assertIn("cache_last_row_ext < 11'(M2_CACHE_DEPTH)",
                      m.transform(scheduler, self.source(scheduler)))

    def test_context0_original_store_preserved(self):
        relative = 'memory/experimental_dual_context_weight_store_p4all_v1.sv'
        original = self.source(relative)
        block = original[original.index('            unified_dual_role_weight_store_p4all_candidate_v1 #('):
                         original.index('            end else begin : g_unused')]
        result = m.transform(relative, original)
        self.assertIn(block, result)
        self.assertIn('m2_compact_store #(.DEPTH(M2_CACHE_DEPTH))', result)
        self.assertEqual(result.count('assign request_slot_ready'), 1)

    def test_version_policy_unchanged(self):
        relative = 'learning/unified_srnn_m2_timestep_train_infer_coordinator.sv'
        original = self.source(relative)
        self.assertEqual(m.transform(relative, original), original)
        self.assertIn('assign o_h40_prepare_version = o_forward_adapt_version;', original)

    def test_selected_capacity_rejects_unsupported_maximum(self):
        relative = 'top/unified_h40_final_composition_top_v1.sv'
        result = m.transform(relative, self.source(relative))
        self.assertIn('SRNN_M2_BACKBONE_MAX > M2_CACHE_DEPTH', result)
        self.assertIn('SRNN_M2_HIDDEN_MAX > M2_CACHE_DEPTH', result)

    def test_synthesis_selection_only(self):
        tcl='-generic [list "ENABLE_ONLINE_LEARNING=$enable_online_learning"]\nputs $status_handle "enable_online_learning=$enable_online_learning"\n'
        changed=selected_tcl(tcl)
        self.assertIn('"M2_CACHE_DEPTH=512"',changed)
        self.assertIn('"m2_cache_depth=512"',changed)
        with self.assertRaises(ValueError):
            selected_tcl(tcl+tcl)

    def test_boundary_parser_rejects_incomplete_and_early_pass(self):
        cases='CAPACITY_PUBLIC_PASS depth=512 cases=16\nCAPACITY_PUBLIC_PASS depth=578 cases=16\n'
        terminal='CAPACITY_BOUNDS_PASS depths=2 cases=32\n'
        self.assertEqual(parse(cases+terminal)['cases'],32)
        for log in (terminal+cases,cases, cases+terminal+'Fatal: mismatch'):
            with self.assertRaises(ValueError):parse(log)


if __name__ == '__main__':
    unittest.main()
