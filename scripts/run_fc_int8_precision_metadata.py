#!/usr/bin/env python3
"""Serial FC precision/current-scale metadata checks and optional final-top lint."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
CLIENT = "rtl/control/unified_fc_srnn_h40_p4_layer_client_v1.sv"
WALKER = "rtl/control/unified_binary_p4_destination_walker_v1.sv"
ARRAY_PATH = "rtl/tiles/unified_binary_p4_h40_array12_path_v1.sv"
CASES = [
    ("tb_fc_int8_precision_metadata", [CLIENT, WALKER],
     ("FC_INT8_PRECISION_METADATA_PASS", "FC_CURRENT_SHIFT6_METADATA_PASS",
      "METADATA_LEGACY_DEFAULT_RAW_PASS", "METADATA_SCNN_FACTOR_UNCHANGED_PASS")),
    ("tb_unified_fc_srnn_h40_p4_layer_client_v1", [CLIENT], "PASS:"),
    ("tb_unified_binary_p4_destination_walker_v1", [WALKER], "PASS:"),
]
ARRAY12_CASE = ("tb_fc_current_shift6_array12_metadata", [WALKER, ARRAY_PATH],
                "FC_CURRENT_SHIFT6_ARRAY12_METADATA_PASS")
INGRESS_LEGACY_CASE = (
    "tb_unified_fc_srnn_h40_tp_observation_ingress_v1",
    ["rtl/learning/unified_fc_srnn_h40_tp_observation_reorder_v1.sv",
     "rtl/learning/unified_fc_srnn_h40_tp_replay_router_v1.sv",
     "rtl/learning/unified_fc_srnn_h40_tp_observation_ingress_v1.sv"],
    "PASS: one H40 ingress reorders physical completions",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lint-final-top", action="store_true")
    parser.add_argument("--lint-only", action="store_true")
    parser.add_argument("--array12", action="store_true",
                        help="include the isolated array12 test; coordinate with other array builds")
    parser.add_argument("--array12-only", action="store_true",
                        help="run only the isolated array12 transaction test")
    parser.add_argument("--ingress-legacy-only", action="store_true",
                        help="run legacy ingress with all four optional logit cfg ports omitted")
    args = parser.parse_args()
    stamp = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out = ROOT / "results/paper_evidence/supplement_20261004/rtl/fc_w2_repair/current_shift6_metadata" / stamp
    # Keep generated specialized C++ filenames below the Windows path limit.
    work = ROOT / "work/fc_metadata" / stamp
    out.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["VERILATOR_ROOT"] = "C:/msys64/mingw64/share/verilator"
    env["PATH"] = "C:/msys64/mingw64/bin;C:/msys64/usr/bin;" + env.get("PATH", "")
    verilator = "C:/msys64/mingw64/bin/verilator_bin.exe"
    common = ["--timing", "--assert", "-Wall", "-Wno-fatal", "-Wno-DECLFILENAME",
              "-Wno-PINMISSING", "-Wno-PINCONNECTEMPTY", "-Wno-PROCASSINIT",
              "-Wno-UNUSEDSIGNAL", "-Wno-UNUSEDPARAM", "-Wno-BLKSEQ",
              "-Wno-SYNCASYNCNET"]
    record = {"scope": "independent FC INT8/shift6 metadata, stalls, mode rejection, abort, array12 transaction consistency and legacy compatibility; optional ingress legacy regression and final-top lint",
              "production_generation": False, "jobs": 1,
              "ingress_legacy_requested": args.ingress_legacy_only,
              "array12_requested": args.array12 or args.array12_only, "stages": []}
    print(f"Evidence: {out}", flush=True)

    def run(name, command, timeout, marker=None):
        stage = {"name": name, "command": command, "timeout_seconds": timeout}
        record["stages"].append(stage)
        started = time.monotonic()
        log_path = out / f"{name}.log"
        try:
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=timeout)
            stage["exit_code"] = result.returncode
            text = log_path.read_text(encoding="utf-8", errors="replace")
            markers = (marker,) if isinstance(marker, str) else marker or ()
            stage["pass"] = result.returncode == 0 and all(m in text for m in markers)
        except subprocess.TimeoutExpired:
            stage["timed_out"] = True
            stage["pass"] = False
        except OSError as error:
            stage["launch_error"] = str(error)
            stage["pass"] = False
        finally:
            stage["wall_seconds"] = round(time.monotonic() - started, 3)
            (out / "commands.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {'PASS' if stage['pass'] else 'FAIL'} ({stage['wall_seconds']} s)", flush=True)
        if not stage["pass"]:
            raise SystemExit(f"Inspect {log_path}")

    if not args.lint_only:
        if args.ingress_legacy_only:
            cases = [INGRESS_LEGACY_CASE]
        else:
            cases = [ARRAY12_CASE] if args.array12_only else CASES + ([ARRAY12_CASE] if args.array12 else [])
        for top, sources, marker in cases:
            obj_name = "array12" if top == ARRAY12_CASE[0] else "ingress" if top == INGRESS_LEGACY_CASE[0] else top
            obj = work / obj_name
            bench = "tb_fc_int8_precision_metadata" if top == "tb_fc_current_shift6_array12_metadata" else top
            run(top + "_build", [verilator, "--binary", "-j", "1", *common,
                "-CFLAGS", "-O0 -D_GLIBCXX_USE_CXX11_ABI=0", "--top-module", top,
                "--Mdir", obj.as_posix(),
                "-y", (ROOT / "rtl/common").as_posix(),
                "-y", (ROOT / "rtl/memory").as_posix(),
                "-y", (ROOT / "rtl/tiles").as_posix(),
                "-y", (ROOT / "rtl/control").as_posix(),
                *[(ROOT / source).as_posix() for source in sources],
                (ROOT / "tb" / f"{bench}.sv").as_posix()], 300)
            run(top, [str(obj / f"V{top}.exe")], 30, marker)
    if args.lint_final_top or args.lint_only:
        lint_obj = work / "lint"
        lint_obj.mkdir()
        source_list = lint_obj / "rtl.f"
        source_list.write_text("".join(
            f'"{p.as_posix()}"\n' for p in sorted((ROOT / "rtl").rglob("*.sv"))),
            encoding="utf-8")
        run("final_top_lint", [verilator, "--lint-only", *common,
            "-Wno-WIDTHTRUNC", "-Wno-WIDTHEXPAND", "-Wno-CMPCONST",
            "-Wno-TIMESCALEMOD", "-Wno-WIDTHCONCAT",
            "--top-module", "unified_h40_final_composition_top_v1",
            "-GTOPOLOGY_MASK=7", "--Mdir", lint_obj.as_posix(),
            "-f", source_list.as_posix()], 240)
    print("FC_INT8_METADATA_CHECKS_PASS", flush=True)


if __name__ == "__main__":
    main()
