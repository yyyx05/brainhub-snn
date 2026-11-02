from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_scnn_turnover_recovery import transform,parse_log,load_parser
from run_srnn_resident_remote import ROOT,RESULT,local_path


class TurnoverRecoveryPreparationTest(unittest.TestCase):
    def original(self):return Path(local_path(ROOT/RESULT/'final_combination_recovery_r1'))

    def test_only_capacity_parameter_added(self):
        old=(self.original()/'tb_srnn_profile_measurement.sv').read_text();new=transform(old)
        self.assertEqual(new.replace('\n        .M2_CACHE_DEPTH(512),',''),old)
        self.assertEqual(new.count('.M2_CACHE_DEPTH(512)'),1)

    def test_both_archived_arms_parse_without_rerun(self):
        root=self.original();ref=json.loads((root/'recovery/reference.json').read_text())
        parser=load_parser(root/'scripts/experiments/srnn_resident_server_worker.py')
        for arm in ('off','on'):
            checks=parse_log((root/('run_'+arm+'.log')).read_text(),arm,ref['expected'][arm],parser)
            self.assertEqual(checks['recovered']['resets'],1)
            self.assertEqual(checks['recovered']['parameter_loads'],120320)
            self.assertEqual(len(checks['tiles']),12)

    def test_wrong_counts_order_or_missing_tile_rejected(self):
        root=self.original();ref=json.loads((root/'recovery/reference.json').read_text())['expected']['on'];log=(root/'run_on.log').read_text()
        lines=log.splitlines();accepted=next(i for i,s in enumerate(lines) if s.startswith('RESIDENT_ABORT_ACCEPTED '))
        final=next(i for i,s in enumerate(lines) if s.startswith('RESIDENT_ABORT_RECOVERY_PASS '))
        lines[accepted],lines[final]=lines[final],lines[accepted]
        changes=[log.replace('resets=1','resets=2'),log.replace('unchanged_parameters=120320','unchanged_parameters=120319'),
                 log.replace('H40_ACTIVITY tile=','MISSING_ACTIVITY tile=',1),'\n'.join(lines),log+'\n%Error: failure']
        for changed in changes:
            with self.assertRaises(ValueError):parse_log(changed,'on',ref)


if __name__=='__main__':unittest.main()
