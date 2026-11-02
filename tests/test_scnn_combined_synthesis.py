import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/experiments"))
from run_srnn_resident_remote import synthesis_inputs, synthesis_scanner_inputs


class CombinedSynthesisTests(unittest.TestCase):
    def state(self, status):
        return dict(status=status,build=dict(exit=0),run_off=dict(exit=0),run_on=dict(exit=0))

    def test_combined_uses_actual_rtl_wrapper_and_both_diffs(self):
        wrapper,evidence=synthesis_inputs(self.state("PASS_SCNN_real_sample0_B1T4_pair_candidate_compatibility"))
        self.assertTrue(wrapper.startswith("rtl/"))
        self.assertEqual(evidence,["scnn_optimization_overlay.json","fc_scanner_changes.diff","scnn_publisher_changes.diff"])

    def test_srnn_original_source_layout_preserved(self):
        wrapper,evidence=synthesis_inputs(self.state("PASS_candidate_fixed_sample_pair_not_physical_QoR"))
        self.assertTrue(wrapper.startswith("verification/"))
        self.assertEqual(evidence,["source_changes.diff"])

    def test_unfinished_or_missing_exit_not_accepted(self):
        state=self.state("run_on_running")
        with self.assertRaises(ValueError): synthesis_inputs(state)
        state=self.state("PASS_SCNN_real_sample0_B1T4_pair_candidate_compatibility")
        state["run_on"]["exit"]=None
        with self.assertRaises(ValueError): synthesis_inputs(state)

    def test_scanner_overlay_requires_its_own_full_fc_pair(self):
        state=self.state('PASS_FC_B16T99_pair_scanner_pipeline_not_physical_QoR')
        self.assertEqual(synthesis_scanner_inputs(state),'rtl/learning/unified_fc_tp_w2_contribution_scheduler.sv')
        state['run_on']['exit']=None
        with self.assertRaises(ValueError): synthesis_scanner_inputs(state)
        state['status']='PASS_natural_shared_learner_publication_component_not_final_top_or_RTL_query'
        with self.assertRaises(ValueError): synthesis_scanner_inputs(state)


if __name__ == "__main__": unittest.main()
