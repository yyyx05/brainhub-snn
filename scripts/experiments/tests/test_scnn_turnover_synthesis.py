from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_scnn_turnover_synthesis import result_status


class SynthesisStatusTest(unittest.TestCase):
    def state(self):
        return dict(status='pass',black_boxes='0',period_ns='15.000',source_count='108',enable_online_learning='1',m2_cache_depth='512')

    def test_synthesis_pass_is_not_sta_or_release(self):
        self.assertEqual(result_status(0,self.state(),True,[],[]),'PASS_synthesis_candidate_only_not_STA_or_final_functional_release')

    def test_bad_exit_or_checkpoint_cannot_pass(self):
        for code,crc in ((1,True),(139,None),(0,False),(0,None)):
            self.assertTrue(result_status(code,self.state(),crc,[],[]).startswith('FAIL'))

    def test_wrong_configuration_or_diagnostic_cannot_pass(self):
        for key,value in (('m2_cache_depth','578'),('source_count','107'),('black_boxes','1'),('period_ns','20.000')):
            state=self.state();state[key]=value
            self.assertTrue(result_status(0,state,True,[],[]).startswith('FAIL'))
        self.assertEqual(result_status(0,self.state(),True,[],['missing ROM']),'FAIL_tool_diagnostic')


if __name__=='__main__':unittest.main()
