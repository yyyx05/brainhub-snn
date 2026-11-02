import importlib.util
import os
from pathlib import Path
import unittest

MODULE = Path(__file__).resolve().parents[1]/"run_m2_xpm_context.py"
spec = importlib.util.spec_from_file_location("xpm_context",MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def log(variant, bad=0):
    depth = 512 if variant == 2 else 578
    return (f"XPM_CONTEXT_CAPACITY variant={variant} context0=578 context1={depth}\n"
            f"XPM_CONTEXT_COMMON_PASS variant={variant}\n"
            f"XPM_CONTEXT_RESULT variant={variant} depth={depth} reads={578+depth+77} "
            f"fills={(578+depth)*4+6} stalls=20 conflicts=5 replacements=100 "
            f"held_read=7 held_fill=5 hazard_bad_cycles={bad} hazard_bad_terms={bad}\n"
            f"XPM_CONTEXT_COMPLETE variant={variant}\n")


class XpmContextTests(unittest.TestCase):
    def test_baseline_failure_is_not_pass(self):
        self.assertEqual(m.parse_run(log(0,10),0)["contract_status"],"FAIL_pending_fill_payload")

    def test_candidate_both_depths(self):
        for variant in (1,2):
            self.assertEqual(m.parse_run(log(variant),variant)["contract_status"],"PASS_exercised_scope")

    def test_candidate_corruption_rejected(self):
        with self.assertRaises(ValueError): m.parse_run(log(1,1),1)

    def test_diagnostics_and_missing_data_rejected(self):
        for text in (log(1)+"Fatal: payload mismatch\n", log(1)*2,
                     log(1).replace("held_read=7","held_read=0"),
                     log(1).replace("reads=1233","reads=1232")):
            with self.assertRaises(ValueError): m.parse_run(text,1)

    def test_wrapper_retains_original_context0(self):
        root = MODULE.parents[2]
        path = root/"results/paper_evidence/supplement_20261015/performance_optimization/scnn_issue_banked_synth_r1/rtl/memory/experimental_dual_context_weight_store_p4all_v1.sv"
        source = Path("\\\\?\\"+str(path) if os.name == "nt" else path).read_text()
        transformed = m.transform_wrapper(source)
        original_store = source[source.index("            unified_dual_role_weight_store_p4all_candidate_v1 #("):source.index("            end else begin : g_unused")]
        self.assertIn(original_store,transformed)
        self.assertIn("parameter int M2_DEPTH = 578",transformed)
        self.assertEqual(transformed.count("assign request_slot_ready"),1)

    def test_archived_actual_xpm_logs(self):
        root = MODULE.parents[2]/"results/paper_evidence/supplement_20261015/performance_optimization/m2_xpm_context_r1"
        root = Path("\\\\?\\"+str(root) if os.name == "nt" else root)
        for variant in (0,1,2):
            result = m.parse_run((root/f"variant{variant}/run_stdout.log").read_text(),variant)
            self.assertEqual(result["hazard_bad_cycles"],12 if variant==0 else 0)

    def test_early_completion_rejected(self):
        text = log(1).replace("XPM_CONTEXT_COMPLETE variant=1\n","")
        with self.assertRaises(ValueError):
            m.parse_run("XPM_CONTEXT_COMPLETE variant=1\n"+text,1)


if __name__ == "__main__": unittest.main()
