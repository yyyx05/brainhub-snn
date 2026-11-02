"""Validate the packaging adapter without modifying or simulating the core."""

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ip/unified_snn_native_ip"))
from generate_native_wrapper import ansi_header, render_wrapper


class NativeIpWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.top = ROOT / "rtl/top/unified_h40_final_composition_top_v1.sv"
        cls.module, cls.parameters, cls.ports = ansi_header(cls.top)
        cls.text = render_wrapper(cls.top)

    def test_every_port_keeps_direction_signedness_and_range(self):
        header = self.text.split(");", 1)[0]
        actual = re.findall(r"^\s+(input|output)\s+(wire.*?)\s+(\w+),?$", header, re.M)
        expected = [(direction, kind.replace("logic", "wire", 1), name)
                    for direction, kind, name in self.ports]
        self.assertEqual(actual, expected)
        self.assertNotRegex(header, r"\blogic\b")

    def test_all_parameters_and_ports_are_forwarded_once(self):
        body = self.text.split(");", 1)[1]
        mappings = re.findall(r"\.(\w+)\((\w+)\)", body)
        expected = [(name, name) for _, name, _ in self.parameters]
        expected += [(name, name) for _, _, name in self.ports]
        self.assertEqual(mappings, expected)
        self.assertEqual(body.count(self.module + " #("), 1)
        self.assertNotRegex(body, r"\b(always|initial|assign|generate)\b")

    def test_defaults_are_preserved(self):
        for kind, name, value in self.parameters:
            declaration = "integer " if kind == "int" else ""
            self.assertIn(f"parameter {declaration}{name} = {value}", self.text)


if __name__ == "__main__":
    unittest.main()
