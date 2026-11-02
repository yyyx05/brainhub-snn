import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from summarize_scnn_turnover_frame import compare


class ComparisonTest(unittest.TestCase):
    def fixture(self):
        service=dict(reads=1000,responses=1000,replacements=0,slots=1,code_bits=16,address_bits=24,
            tag_bits=16,minimum_latency=1,replaceable=1,m2_cache_depth=512,end_cycle=1103)
        bucket=dict(layer=1,cycles=1000,idle=1,context_load=500,feature_request=99,
            feature_response=100,command_request=100,command_wait=100,context_release=99,classifier=1)
        before=dict(service=service,frame_cycles=1000,initialization_cycles=100,banks=[dict(reads=1000)],
            tables=dict(bucket=[bucket],handshake=[dict(reads=1000)],publisher=[dict(reads=1000)]))
        after=copy.deepcopy(before)
        after.update(frame_cycles=800,initialization_cycles=90)
        after['service'].update(end_cycle=893,replacements=210)
        after['tables']['bucket'][0].update(cycles=800,context_load=300)
        after['tables']['publisher'][0]['responses']=1000
        budget=dict(layers=[dict(max_turnovers=200)],max_turnover_savings_per_frame=200,classifier_savings_per_publish=10)
        return before,after,['SCNN_PROFILE_EVENT kind=initialize_start cycle=1\n']*2,budget

    def test_frame_and_startup_inclusive_are_distinct(self):
        result=compare(*self.fixture())
        self.assertEqual(result['frame_saved_cycles'],200)
        self.assertEqual(result['startup_inclusive_saved_cycles'],210)
        self.assertEqual(result['metrics'][2]['baseline_cycles'],1102)
        self.assertEqual(result['boundary_cycles_outside_frame_and_initialization_interval'],[2,2])
        self.assertEqual(result['context_load_saved_cycles'],200)
        self.assertTrue(all(v for k,v in result['budget_reconciliation'].items() if k.startswith('actual_') or k.startswith('startup_')))

    def test_different_service_rejected(self):
        args=self.fixture();args[1]['service']['minimum_latency']=4
        with self.assertRaises(ValueError):compare(*args)

    def test_different_work_rejected(self):
        args=self.fixture();args[1]['tables']['handshake'][0]['reads']=999
        with self.assertRaises(ValueError):compare(*args)

    def test_unexplained_budget_difference_remains_visible(self):
        args=self.fixture();args[1]['tables']['bucket'][0].update(cycles=801,context_load=301)
        result=compare(*args)
        self.assertFalse(result['layers'][0]['budget_matches'])
        self.assertEqual(result['descriptor_saved_cycles'],199)


if __name__=='__main__':unittest.main()
