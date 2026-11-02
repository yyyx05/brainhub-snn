from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_final_raw_source_implementation as runner


class PairedImplementationTest(unittest.TestCase):
    def test_recipe_preserves_optimization_and_old_protections(self):
        from run_srnn_resident_remote import ROOT, RESULT, local_path
        original = Path(local_path(ROOT / RESULT / 'm2_compact_impl_r1' / runner.ORIGINAL)).read_text()
        generated = runner.implementation_tcl(original)
        commands = ('opt_design', 'place_design -directive ExtraNetDelay_high',
            'phys_opt_design -directive ExploreWithHoldFix',
            'route_design -directive AggressiveExplore', 'phys_opt_design -directive Explore')
        self.assertEqual([line for line in generated.splitlines() if line in commands], list(commands))
        self.assertIn('u_core/g_srnn_phase_bridge.u_srnn_phase_bridge', generated)
        self.assertIn('u_core/u_system', generated)
        self.assertIn('u_core/u_parameter_plane', generated)
        self.assertIn('unexpected inherited 15ns clock', generated)
        for line in original.splitlines():
            if any(token in line for token in ('post_place_level6', 'original_protected_names',
                    'trace_payload_nonconstant_control_count', 'timing_closed')):
                self.assertIn(line, generated)
        with self.assertRaises(ValueError):
            runner.implementation_tcl(original.replace('    u_system\n', ''))

    def make_report(self, root, tcl_status='pass', critical=False):
        out = root / runner.OUT
        out.mkdir(parents=True)
        (out / 'vivado.log').write_text('Vivado implementation complete\n' + (
            'CRITICAL WARNING: retained diagnostic\n' if critical else ''))
        (out / 'vivado_status.txt').write_text(
            'status=' + tcl_status + '\ntop=unified_h40_raw_source_top_v1\npart=xc7vx690tffg1761-3\n')
        (out / 'checkpoint_reopen_status.txt').write_text('status=pass\n')
        with zipfile.ZipFile(out / 'post_route_physopt.dcp', 'w') as archive:
            archive.writestr('synthetic_test_member', 'not a Vivado checkpoint')
        return out

    def test_negative_sta_is_diagnostic_and_keeps_pair_measurable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_report(root, 'diagnostic_complete')
            parsed = runner.classify(root, 0)
            self.assertEqual(parsed['status'], 'DIAGNOSTIC_implementation')
            self.assertTrue(parsed['measurement_complete'])

    def test_failed_exit_or_missing_checkpoint_stops_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.make_report(root)
            self.assertFalse(runner.classify(root, 1)['measurement_complete'])
            (report / 'post_route_physopt.dcp').unlink()
            self.assertFalse(runner.classify(root, 0)['measurement_complete'])

    def test_critical_warning_cannot_become_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_report(root, critical=True)
            parsed = runner.classify(root, 0)
            self.assertEqual(parsed['status'], 'DIAGNOSTIC_implementation')
            self.assertEqual(len(parsed['critical_warnings']), 1)

    def test_wrong_top_and_error_log_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.make_report(root)
            status = report / 'vivado_status.txt'
            status.write_text(status.read_text().replace('unified_h40_raw_source_top_v1', 'old_top'))
            self.assertFalse(runner.classify(root, 0)['measurement_complete'])
            status.write_text(status.read_text().replace('old_top', 'unified_h40_raw_source_top_v1'))
            (report / 'vivado.log').write_text('ERROR: implementation failed\n')
            self.assertFalse(runner.classify(root, 0)['measurement_complete'])


if __name__ == '__main__':
    unittest.main()
