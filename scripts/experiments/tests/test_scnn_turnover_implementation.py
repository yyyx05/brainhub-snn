from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_scnn_turnover_implementation import implementation_status,implementation_tcl,run_paths


class TurnoverImplementationTest(unittest.TestCase):
    def test_retry_paths_are_isolated_and_validated(self):
        first=run_paths('scnn_turnover_core_impl_r1')
        retry=run_paths('scnn_turnover_core_impl_r2')
        self.assertTrue(all(a!=b for a,b in zip(first,retry)))
        self.assertEqual(retry[1],'/tmp/snn_scnn_turnover_core_impl_r2')
        self.assertEqual(retry[2].as_posix(),'verification/experiments/scnn_forward_turnover/core_implementation_r2')
        for name in ('../scnn_turnover_core_impl_r2','scnn_turnover_core_impl_r0','scnn_turnover_core_impl_r2/extra'):
            with self.assertRaises(ValueError):run_paths(name)

    def test_recipe_commands_unchanged(self):
        from run_srnn_resident_remote import ROOT,RESULT,local_path
        p=Path(local_path(ROOT/RESULT/'m2_compact_impl_r1/vivado/run_unified_h40_final_composition_v1_impl_from_dcp.tcl'))
        old=p.read_text();new=implementation_tcl(old)
        for line in ('opt_design','place_design -directive ExtraNetDelay_high','phys_opt_design -directive ExploreWithHoldFix',
                     'route_design -directive AggressiveExplore','phys_opt_design -directive Explore'):
            self.assertEqual([x for x in new.splitlines() if x==line],[line])
        self.assertIn('close_design\nopen_checkpoint [file join $out_dir post_route_physopt.dcp]',new)
        self.assertIn('routed checkpoint reopen mismatch',new)
        self.assertEqual(new[:old.index('write_checkpoint -force [file join $out_dir post_route_physopt.dcp]')],old[:old.index('write_checkpoint -force [file join $out_dir post_route_physopt.dcp]')])

    def test_pass_requires_exit_reopen_crc_and_no_diagnostic(self):
        args=[0,dict(status='pass'),dict(status='pass'),True,[],[]]
        self.assertEqual(implementation_status(*args),'PASS_candidate_internal15ns_checkpoint_reopened')
        for i,value in ((0,1),(2,{}),(3,False),(4,['ERROR']),(5,['critical'])):
            changed=args.copy();changed[i]=value
            self.assertTrue(implementation_status(*changed).startswith('FAIL'))

    def test_diagnostic_is_not_timing_pass(self):
        self.assertTrue(implementation_status(0,dict(status='diagnostic_complete'),dict(status='pass'),True,[],[]).startswith('DIAGNOSTIC'))
        with self.assertRaises(ValueError):implementation_tcl('no checkpoint command')


if __name__=='__main__':unittest.main()
