import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('natural',ROOT/'scripts/experiments/prepare_scnn_natural_component.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class NaturalComponentTests(unittest.TestCase):
    def test_actual_public_interface_and_stimulus_boundary(self):
        source=module.testbench_source((ROOT/module.SERVICE).read_text(),(ROOT/module.EXPERIMENT/'body.svh').read_text())
        self.assertEqual(source.count('.i_scnn_cfg_batch_count(i_scnn_cfg_batch_count)'),1)
        self.assertIn('i_scnn_cfg_batch_count=5\'d16',source)
        self.assertIn('for(int a=0;a<P;a++) residuals[a]=0;',source)
        self.assertIn('NOT a hardware query engine',source)
        self.assertIn('dut.g_output_learning_core.u_shared_output_learning_path.u_learning_slice',source)
        self.assertNotIn('warm_residual',source)
        self.assertNotIn("i_read_ready = '0",source)
        self.assertNotIn("i_result_ready = '0",source)
        self.assertIn("i_scnn_cfg_source_stride=24'd2048; i_scnn_cfg_output_stride=0",source)
    def test_missing_and_fatal_are_rejected(self):
        for log in ('','NATURAL_COMPONENT_PASS windows=3','%Fatal old failure'):
            with self.assertRaises(ValueError):module.parse_log(log)
    def test_complete_counts_and_duplicate(self):
        windows='\n'.join(f'NATURAL_COMPONENT_WINDOW window={i+1} code={c} residual={r} rmw={r}' for i,(c,r) in enumerate(((0,20041),(0,20011),(662,20221))))
        pubs='\n'.join(f'NATURAL_COMPONENT_PUBLICATION publication={i} words=22528 rows=1892' for i in range(4))
        end='NATURAL_COMPONENT_PASS windows=3 trace_inputs=98304 logit_inputs=528 rmw=60273 code_writes=662 residual_writes=60273 publications=4 published_words=90112 published_rows=7568 query_current_checks=44 query_logit_checks=44 query_current_changes=4 query_logit_changes=4 reset=1'
        log=windows+'\n'+pubs+'\n'+end
        self.assertTrue(module.parse_log(log)['status'].startswith('PASS_'))
        for bad in (log+'\n'+end,log.replace('code_writes=662','code_writes=0')):
            with self.assertRaises(ValueError):module.parse_log(bad)
