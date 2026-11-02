from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_scnn_turnover_tail import transform,parse_log,BANK_READS
from run_srnn_resident_remote import ROOT,RESULT,local_path


class TurnoverTailPreparationTest(unittest.TestCase):
    def original(self):return Path(local_path(ROOT/RESULT/'scnn_tail_visibility_r1'))

    def footer(self,cycles):
        lines=[f'SCNN_TURNOVER_TAIL_BANK bank={i} reads={n} responses={n}' for i,n in enumerate(BANK_READS)]
        lines += [f'SCNN_TURNOVER_TAIL_PUBLICATION window={w} replacements=100' for w in (1,2,3)]
        lines += [f'SCNN_TURNOVER_TAIL_SERVICE_PASS reads=122773504 responses=122773504 replacements=10000 query_replacements=1000 slots=1 code_bits=16 address_bits=24 tag_bits=16 minimum_latency=1 replaceable=1 m2_cache_depth=512 end_cycle={cycles}']
        return '\n'.join(lines)+'\n'

    def valid_log(self):
        original=(self.original()/'run.log').read_text()
        from prepare_scnn_tail_testbench import parse_log as previous
        return original+'\n'+self.footer(previous(original)['final']['cycles'])

    def test_overlay_retains_original_assertions(self):
        old=(self.original()/'tb_scnn_tail_visibility.sv').read_text();new=transform(old)
        for line in old.splitlines():
            if '$fatal(' in line:self.assertIn(line,new)
        self.assertIn('tail_code_writes != 662',new)
        self.assertIn('tail_forward_responses != 122773504',new)
        self.assertEqual(new.count('.M2_CACHE_DEPTH(512)'),1)
        self.assertEqual(new.count('.g_context[0].g_enabled.g_original.u_store.g_bank'),2)
        self.assertIn('fg_finish();\n        st_report();\n        $finish;',new)

    def test_old_numeric_log_with_synthetic_native_footer(self):
        checks=parse_log(self.valid_log())
        self.assertEqual(checks['final']['code_changes'],662)
        self.assertEqual(len(checks['native_publications']),3)
        self.assertIn('not_full_graph_or_abort',checks['status'])

    def test_missing_or_bad_native_coverage_rejected(self):
        log=self.valid_log()
        changes=[log.replace('query_replacements=1000','query_replacements=0'),
                 log.replace('window=3 replacements=100','window=3 replacements=0'),
                 log.replace('tag_bits=16','tag_bits=4'),
                 log.replace('reads=122683392 responses=122683392','reads=122683392 responses=122683391'),
                 log+'\n'+log.splitlines()[-1],
                 (self.original()/'run.log').read_text()]
        for changed in changes:
            with self.assertRaises(ValueError):parse_log(changed)


if __name__=='__main__':unittest.main()
