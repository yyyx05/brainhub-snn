#!/usr/bin/env python3
"""Small real-weight cache repair test plus directly affected legacy regressions."""
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
from run_fc_w2_precision_probe import MODEL, ROOT, read_model

MEM = "rtl/memory/"
TESTS = {
    "tb_fc_w2_int8_cache_repair": [
        MEM + "unified_array16_weight_refill_dispatcher_v1.sv",
        MEM + "unified_h40_scalar_parameter_row_publisher_v1.sv",
        MEM + "unified_fc_h40_w0_cache_refill_scheduler_v1.sv",
        MEM + "unified_array16_two_client_weight_fill_arbiter_v1.sv",
        MEM + "unified_array16_shared_weight_cache_refill_plane_v1.sv",
        MEM + "unified_dual_role_weight_store_p4all_candidate_v1.sv",
        "rtl/tiles/unified_dual_role_h40_tile_p4all_cascade_accum_candidate_v1.sv",
        "rtl/tiles/unified_dual_role_dsp_lane_p4_cascade_candidate_v1.sv"],
    "tb_unified_dual_role_weight_store_p4all_candidate_v1": [
        MEM + "unified_dual_role_weight_store_p4all_candidate_v1.sv"],
    "tb_unified_fc_h40_w0_cache_refill_scheduler_v1": [
        MEM + "unified_array16_weight_refill_dispatcher_v1.sv",
        MEM + "unified_h40_scalar_parameter_row_publisher_v1.sv",
        MEM + "unified_fc_h40_w0_cache_refill_scheduler_v1.sv"],
    "tb_unified_array16_shared_weight_cache_refill_plane_v1": [
        MEM + "unified_array16_weight_refill_dispatcher_v1.sv",
        MEM + "unified_h40_scalar_parameter_row_publisher_v1.sv",
        MEM + "unified_fc_h40_w0_cache_refill_scheduler_v1.sv",
        MEM + "unified_array16_two_client_weight_fill_arbiter_v1.sv",
        MEM + "unified_array16_shared_weight_cache_refill_plane_v1.sv",
        "rtl/learning/unified_scnn_classifier_cache_refill_publisher_v1.sv",
        MEM + "unified_scnn_h40_conv_context_refill_v1.sv",
        "rtl/learning/unified_tp_two_client_forward_read_arbiter.sv",
        MEM + "unified_dual_role_weight_store_p4all_candidate_v1.sv"],
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    stamp = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out = ROOT / "results/paper_evidence/supplement_20261004/rtl/fc_w2_repair" / stamp
    work = ROOT / "work/supplement_20261004/fc_w2_repair" / stamp
    out.mkdir(parents=True)
    work.mkdir(parents=True)
    print(f"Evidence: {out.relative_to(ROOT)}", flush=True)
    write_json(out / "model.json", read_model())
    with np.load(ROOT / MODEL / "weights_h160_int5_w2_int8.npz", allow_pickle=False) as model:
        weights = model["W2"].astype(np.int16).ravel()
    fixture = out / "w2.mem"
    fixture.write_text("".join(f"{int(x) & 65535:04x}\n" for x in weights), encoding="ascii")
    env = os.environ.copy()
    env["VERILATOR_ROOT"] = "C:/msys64/mingw64/share/verilator"
    env["PATH"] = "C:/msys64/mingw64/bin;C:/msys64/usr/bin;" + env.get("PATH", "")
    records = []
    for top, sources in TESTS.items():
        obj = work / top
        cmd = ["C:/msys64/mingw64/bin/verilator_bin.exe", "--binary", "--timing", "--assert",
               "-j", "1", "-Wall", "-Wno-fatal", "-CFLAGS", "-O0 -D_GLIBCXX_USE_CXX11_ABI=0",
               "--top-module", top, "--Mdir", obj.as_posix(),
               *[(ROOT / src).as_posix() for src in sources], (ROOT / "tb" / (top + ".sv")).as_posix()]
        run = [str(obj / ("V" + top + ".exe")), "+WEIGHTS=" + fixture.relative_to(ROOT).as_posix()]
        record = {"top": top, "sources": sources, "build_command": cmd, "run_command": run,
                  "cwd": str(ROOT), "jobs": 1, "dut_changed": True, "memory_model": "behavioral"}
        records.append(record)
        for stage, command in [("build", cmd), ("run", run)]:
            started = time.monotonic()
            with (out / f"{top}.{stage}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            record[stage + "_exit"] = result.returncode
            record[stage + "_seconds"] = round(time.monotonic() - started, 3)
            write_json(out / "commands.json", records)
            print(f"{top} {stage}: exit={result.returncode}, {record[stage + '_seconds']}s", flush=True)
            if result.returncode:
                raise SystemExit(f"Preserved failure at {out}")
        text = (out / f"{top}.run.log").read_text(encoding="utf-8")
        if "PASS" not in text or "$fatal" in text:
            raise RuntimeError(f"Missing terminal PASS in {top}")
    text = (out / "tb_fc_w2_int8_cache_repair.run.log").read_text(encoding="utf-8")
    match = re.search(r"FC_W2_INT8_CACHE_REPAIR_PASS (.*)", text)
    if not match:
        raise RuntimeError("Missing repair-specific PASS")
    counts = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", match[1])}
    # Original 5 successes/5 rejections/173 fills remain asserted within the TB.
    # Destination-base cases add 7 successes, 3 rejections and 24 published rows.
    assert counts["successes"] == 12 and counts["rejected"] == 8 and counts["fills"] == 197
    assert counts["checks"] == 23040
    assert counts["base4_fills"] == 5 and counts["endpoint_fills"] == 5
    assert counts["local_dirty_checks"] == 1
    assert counts["dot_checks"] == 30 and counts["mac_commands"] == 640
    summary = {"status": "PASS", "tests": len(records), "real_W2_codes_checked": 1600,
               "counts": counts, "full_FC_numeric_parity": False, "final_top_simulated": False,
               "physical_results_apply_to_repaired_RTL": False,
               "destination_base_checked": [0, 3, 4, 10, 11],
               "invalid_destination_base_count_checked": [[12, 1], [11, 2], [15, 2]],
               "destination_base_changes_parameter_addresses": False,
               "scope": "FC scalar refill through the shared plane, physical destination-base routing, local dirty-tile addressing, P4 cache and raw shared dot-product on real W2 with directed UQ6.10 magnitudes; not output neuron semantics"}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
