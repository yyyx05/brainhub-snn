"""Source-IP packaging dependencies, without Vivado or RTL simulation."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ip/unified_snn_native_ip"))
from validate_source_package import validate

ROM = "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem"


class NativeIpAuxiliaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = "src/rtl/top/final.sv"
        files = {
            source: "module final(input logic i_clk, input logic i_rst_n); endmodule\n",
            "src/filelist.f": source + "\n",
            "constraints/unified_h40_final_composition_v1_ooc.xdc": "# fixture\n",
            "package_ip.tcl": "# fixture\n",
            ROM: "100\n" * 1302,
            "source_closure_report.json": json.dumps({
                "status": "pass", "closure_lint": "pass", "reachable_source_count": 1}),
            "ip_manifest.json": json.dumps({
                "source_count": 1, "top_source": source, "top_module": "final",
                "input_source_count": 3, "excluded_unreachable_source_count": 2,
                "source_selection": "fixture", "part": "fixture", "clock_period_ns": 15,
                "interface_kind": "native", "auxiliary_source_files": [ROM]}),
        }
        for name, content in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="ascii")

    def test_original_rom_dependency_is_required_and_reported(self):
        self.assertEqual(validate(self.root)["auxiliary_source_files"], [ROM])
        (self.root / ROM).unlink()
        with self.assertRaises(OSError):
            validate(self.root)

    def test_omitted_manifest_dependency_is_not_accepted(self):
        path = self.root / "ip_manifest.json"
        manifest = json.loads(path.read_text())
        del manifest["auxiliary_source_files"]
        path.write_text(json.dumps(manifest), encoding="ascii")
        with self.assertRaises(RuntimeError):
            validate(self.root)

    def test_truncated_or_out_of_range_rom_is_not_accepted(self):
        for content in ("100\n" * 1301, "200\n" + "100\n" * 1301,
                        "xyz\n" + "100\n" * 1301):
            with self.subTest(content=content[:8]):
                (self.root / ROM).write_text(content, encoding="ascii")
                with self.assertRaises(RuntimeError):
                    validate(self.root)


if __name__ == "__main__":
    unittest.main()
