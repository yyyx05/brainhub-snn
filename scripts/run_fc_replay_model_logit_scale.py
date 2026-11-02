#!/usr/bin/env python3
"""Build only the logit-scale router TB and its unchanged legacy regression."""

import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
ROUTER = "rtl/learning/unified_fc_srnn_h40_tp_replay_router_v1.sv"
TESTS = (
    "tb_fc_replay_model_logit_scale",
    "tb_unified_fc_srnn_h40_tp_replay_router_v1",
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    stamp = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out = ROOT / "results/paper_evidence/supplement_20261004/rtl/fc_w2_repair/logit_scale" / stamp
    work = ROOT / "work/supplement_20261004/fc_replay_logit_scale" / stamp
    out.mkdir(parents=True)
    work.mkdir(parents=True)
    print(f"Evidence: {out.relative_to(ROOT)}", flush=True)

    env = os.environ.copy()
    if os.name == "nt":
        # Match the existing small-router runner, but never use its -j0 build.
        msys = Path(env.get("MSYS2_ROOT", "C:/msys64"))
        verilator = str(msys / "mingw64/bin/verilator_bin.exe")
        env["VERILATOR_ROOT"] = (msys / "mingw64/share/verilator").as_posix()
        env["PATH"] = os.pathsep.join(
            [str(msys / "mingw64/bin"), str(msys / "usr/bin"), env.get("PATH", "")]
        )
        cflags = "-O0 -D_GLIBCXX_USE_CXX11_ABI=0"
    else:
        verilator = shutil.which("verilator") or "verilator"
        cflags = "-O0"

    records = []
    for top in TESTS:
        sources = [ROUTER, f"tb/{top}.sv"]
        obj = work / top
        command = [
            verilator, "--binary", "--timing", "--assert", "-j", "1",
            "-Wall", "-Wno-fatal", "-Wno-DECLFILENAME", "-Wno-PROCASSINIT",
            "-CFLAGS", cflags, "--top-module", top, "--Mdir", obj.as_posix(),
            *[(ROOT / source).as_posix() for source in sources],
        ]
        executable = obj / ("V" + top + (".exe" if os.name == "nt" else ""))
        record = {"top": top, "sources": sources, "cwd": str(ROOT), "jobs": 1}
        records.append(record)
        for stage, args in (("build", command), ("run", [str(executable)])):
            record[stage + "_command"] = args
            write_json(out / "commands.json", records)
            started = time.monotonic()
            with (out / f"{top}.{stage}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    check=False,
                )
            record[stage + "_exit"] = result.returncode
            record[stage + "_seconds"] = round(time.monotonic() - started, 3)
            write_json(out / "commands.json", records)
            print(f"{top} {stage}: exit={result.returncode}", flush=True)
            if result.returncode:
                raise SystemExit(f"Failure preserved in {out / (top + '.' + stage + '.log')}")
        log_text = (out / f"{top}.run.log").read_text(encoding="utf-8")
        marker = "FC_REPLAY_MODEL_LOGIT_SCALE_PASS" if top == TESTS[0] else "PASS: ordered H40 replay"
        if marker not in log_text:
            raise SystemExit(f"Missing terminal PASS for {top}; logs preserved in {out}")

    log_text = (out / f"{TESTS[0]}.run.log").read_text(encoding="utf-8")
    match = re.search(r"FC_REPLAY_MODEL_LOGIT_SCALE_PASS ([^\n]+)", log_text)
    counts = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", match[1])}
    summary = {
        "status": "PASS", "tests": len(records), "counts": counts,
        "model_scale_q16": 47303,
        "numeric_rule": "sat16(RNE_signed(raw24 * scale_q16 / 65536)); optional raw24 = wrap24(previous_sample_class + raw_step_W2_dot)",
        "legacy_rule": "signed pre_state >>> 4, unchanged with scale disabled/ports omitted",
        "legacy_test_ports_omitted": True,
        "simulator": "Verilator (two-state); X/Z behavior specified by RTL case equality",
        "scope": "Directed router arithmetic, interleaved cumulative sample/class state, config latching, streaming stalls, final drain, abort/clear and legacy FC-hidden/SRNN routing",
        "temporal_accumulation_added": True,
        "accumulator": "signed24, 64 sample IDs x 16 classes; synchronous read; first-row overwrite; full-row validity; partial abort invalidates active sample",
        "learning_semantic_reference": "results/paper_evidence/supplement_20261004/algorithm/a3_audit/a3_audit_20261006_zh.md",
        "full_online_learning_validated": False,
        "final_top_simulated": False,
        "full_FC_numeric_parity": False,
        "physical_results_apply_to_repaired_RTL": False,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
