import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
from run_srnn_equal_storage import MODULE, OLD, NEW, transform
from srnn_equal_storage_worker import residency, expected_loads
from summarize_srnn_resident_storage import REPORT, RESULT, account, banks_from_report


class EqualStorageTests(unittest.TestCase):
    def test_one_assignment_only(self):
        original = (RESULT / 'scnn_issue_banked_synth_r1' / MODULE).read_text(encoding='utf-8')
        changed = transform(original)
        self.assertEqual(changed.replace(NEW, OLD), original)
        self.assertEqual(sum(a!=b for a,b in zip(original.splitlines(),changed.splitlines())), 1)

    def test_transform_rejects_ambiguous_input(self):
        for text in ('', OLD+'\n'+OLD):
            with self.assertRaises(ValueError):
                transform(text)

    def test_original_resident_baseline_count(self):
        previous = json.loads((RESULT / 'srnn_specialized_r1/status.json').read_text())
        reference = previous['sample_definition']['reference_counters']
        for mode in ('off','on'):
            counts = residency((RESULT / f'srnn_specialized_r1/run_{mode}.log').read_text())
            self.assertEqual(expected_loads(reference,mode,'baseline'),counts['wout_loads'])
        self.assertLess(expected_loads(reference,'on','candidate'),expected_loads(reference,'on','baseline'))

    def test_final_commit_does_not_add_refill(self):
        reference = dict(on=[dict(code_writes=0,residual_writes=1),dict(code_writes=1,residual_writes=1)])
        self.assertEqual(expected_loads(reference,'on','baseline'),2)
        self.assertEqual(expected_loads(reference,'on','candidate'),1)

    def test_residency_requires_unique_terminal(self):
        good = 'RESIDENCY_PASS backbone_loads=2 wh1_loads=1 wout_loads=1 wout_reloads=0\n'
        self.assertEqual(residency(good)['wh1_loads'],1)
        for bad in ('',good+good):
            with self.assertRaises(ValueError):
                residency(bad)

    def test_actual_mapping_and_payload_accounting(self):
        result = account(banks_from_report(REPORT.read_text()))
        total = result['totals']
        self.assertEqual(total['ramb36'],192)
        self.assertEqual(total['ramb18'],0)
        self.assertEqual(total['payload_bits'],(450*256+256*20)*16)
        self.assertEqual(total['declared_bits'],8*4*578*200)
        self.assertEqual(total['published_rows'],450*(6*4+2)+256*2)
        self.assertEqual(sum(row['published_rows']==0 for row in result['banks']),4)

    def test_archived_pair_preserves_functional_terminal(self):
        folder = RESULT / 'srnn_equal_storage_r1'
        if not (folder/'candidate/on/measurements.json').is_file():
            self.skipTest('comparison has not been downloaded')
        for mode in ('off','on'):
            a = json.loads((folder/f'baseline/{mode}/measurements.json').read_text())
            b = json.loads((folder/f'candidate/{mode}/measurements.json').read_text())
            self.assertEqual({k:v for k,v in a['terminal'].items() if k!='end'},
                             {k:v for k,v in b['terminal'].items() if k!='end'})
            self.assertEqual(a['terminal']['end']-b['terminal']['end'],
                a['sample_service_cycles_including_refill_excluding_tb_checks']-b['sample_service_cycles_including_refill_excluding_tb_checks'])


if __name__ == '__main__':
    unittest.main()
