"""Platform-only checks: do not compile or load a dataset."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_fc_final_counts import build_settings, generate_tb


class RunnerTests(unittest.TestCase):
    def test_windows_defaults_preserve_msys_abi(self):
        command, exe, env = build_settings(Path("obj"), Path("sources.f"),
                                           platform="nt", environ={"PATH": "original"})
        self.assertEqual(command[0], "C:/msys64/mingw64/bin/verilator_bin.exe")
        self.assertEqual(exe.name, "Vtb_fc_final_counts.exe")
        self.assertIn("-D_GLIBCXX_USE_CXX11_ABI=0", command[command.index("-CFLAGS") + 1])
        self.assertEqual(env["VERILATOR_ROOT"], "C:/msys64/mingw64/share/verilator")
        self.assertTrue(env["PATH"].endswith(";original"))

    def test_linux_explicit_tool_preserves_environment(self):
        original = {"PATH": "/usr/bin", "VERILATOR_ROOT": "/tools/verilator/share"}
        command, exe, env = build_settings(Path("obj"), Path("sources.f"),
            platform="posix", environ=original, verilator="/tools/verilator/bin/verilator", jobs=4)
        self.assertEqual(command[0], "/tools/verilator/bin/verilator")
        self.assertEqual(command[command.index("-j") + 1], "4")
        self.assertEqual(command[command.index("-CFLAGS") + 1], "-O1")
        self.assertEqual(exe.name, "Vtb_fc_final_counts")
        self.assertEqual(env, original)
        self.assertIsNot(env, original)

    def test_linux_path_lookup_and_missing_tool(self):
        with patch("scripts.run_fc_final_counts.shutil.which", return_value="/bin/verilator"):
            command, _, _ = build_settings(Path("obj"), Path("sources.f"), platform="posix", environ={})
            self.assertEqual(command[0], "/bin/verilator")
        with patch("scripts.run_fc_final_counts.shutil.which", return_value=None):
            with self.assertRaises(FileNotFoundError):
                build_settings(Path("obj"), Path("sources.f"), platform="posix", environ={})

    def test_invalid_parallelism(self):
        with self.assertRaises(ValueError):
            build_settings(Path("obj"), Path("sources.f"), jobs=0)

    def test_generated_tb_keeps_production_composition_and_store(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tb.sv"
            generate_tb(path)
            content = path.read_text(encoding="ascii")
        self.assertIn("unified_h40_final_composition_top_v1 #(", content)
        self.assertIn("unified_tp_shared_forward_rmw_store #(", content)
        self.assertIn("FC_FINAL_COUNTS_PASS", content)


if __name__ == "__main__":
    unittest.main()
