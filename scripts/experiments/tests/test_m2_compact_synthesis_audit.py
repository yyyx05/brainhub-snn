import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_m2_compact_synthesis as m


class SynthesisAuditTests(unittest.TestCase):
    def report(self, name):
        return Path(m.local_path(m.ROOT / m.RESULT / m.NAME / m.OUT / name)).read_text()

    def test_actual_resources(self):
        self.assertEqual(m.resources(self.report('utilization_post_synth.rpt')),
                         dict(lut=207966, ff=80371, bram36eq=594.5, dsp=1196))

    def test_missing_resource_rejected(self):
        text = self.report('utilization_post_synth.rpt').replace('Block RAM Tile', 'Unknown memory')
        with self.assertRaises(ValueError):
            m.resources(text)

    def test_actual_timing_keeps_failed_endpoints(self):
        result = m.timing(self.report('timing_summary_post_synth.rpt'))
        self.assertEqual(result['wns_ns'], -0.529)
        self.assertEqual(result['tns_ns'], -132.055)
        self.assertEqual(result['setup_failing'], 286)
        self.assertEqual(result['hold_failing'], 0)
        self.assertEqual(result['pulse_failing'], 0)

    def test_ambiguous_timing_rejected(self):
        text = '| Design Timing Summary\n  -0.1 -1.0 1 2 0.1 0.0 0 2 6.8 0.0 0 3\n'
        text += '  -0.2 -2.0 2 2 0.1 0.0 0 2 6.8 0.0 0 3\n| Clock Summary'
        with self.assertRaises(ValueError):
            m.timing(text)

    def test_context_banks_do_not_double_count_self(self):
        rows = m.hierarchy(self.report('utilization_hierarchical_post_synth.rpt'))
        self.assertFalse(any(row['name'].startswith('(') for row in rows))
        banks = [row for row in rows
                 if 'g_context[1].g_enabled.g_compact.u_store/' in row['path']
                 and row['name'].startswith('g_bank[') and row['name'].endswith('.u_memory')]
        self.assertEqual(len(banks), 32)
        self.assertEqual(len({row['path'] for row in banks}), 32)
        self.assertEqual(sum(row['ramb36'] for row in banks), 80)
        self.assertEqual(sum(row['ramb18'] for row in banks), 0)


if __name__ == '__main__':
    unittest.main()
