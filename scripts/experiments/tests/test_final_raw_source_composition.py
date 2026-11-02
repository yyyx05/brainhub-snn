import importlib.util
from pathlib import Path
import re
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'prepare_final_raw_source_composition.py'
spec = importlib.util.spec_from_file_location('raw_composition', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (module.RESULT / 'srnn_gradient_retire_core_r1/rtl/top' /
                      (module.CORE + '.sv')).read_text()

    def test_every_existing_port_and_parameter_is_forwarded(self):
        for policy in (0, 1):
            text = module.wrapper(self.source, policy)
            checks = module.check_wrapper(self.source, text, policy)
            self.assertGreater(checks['core_ports'], 400)
            self.assertEqual(checks['intercepted_group_inputs'], 8)
            self.assertNotIn('.i_fc_srnn_streamed_feedforward(raw_', text)
            self.assertIn('.i_fc_srnn_streamed_feedforward(i_fc_srnn_streamed_feedforward)', text)

    def test_bridge_named_ports_match_real_header(self):
        bridge = (module.ROOT / 'verification/experiments/raw_source_binding_r1/raw_source_binding.sv').read_text()
        header = bridge.split(') (', 1)[1].split(');', 1)[0]
        names = re.findall(r'\b[io]_\w+\b', header)
        text = module.wrapper(self.source, 1)
        connections = text.split('u_raw_source (', 1)[1].split(');', 1)[0]
        actual = re.findall(r'\.(\w+)\(', connections)
        self.assertEqual(len(actual), len(set(actual)))
        self.assertCountEqual(actual, names)

    def test_missing_port_cannot_pass(self):
        text = module.wrapper(self.source, 1)
        text = text.replace('.i_fc_srnn_group_valid(raw_group_valid)', '.i_fc_srnn_group_valid(i_fc_srnn_group_valid)')
        with self.assertRaisesRegex(ValueError, 'forwarding'):
            module.check_wrapper(self.source, text, 1)

    def test_policy_is_static_and_explicit(self):
        with self.assertRaises(ValueError):
            module.wrapper(self.source, 2)


if __name__ == '__main__':
    unittest.main()
