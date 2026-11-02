#!/usr/bin/env python3
"""Bounded existing-DUT precision probe; all generated artifacts stay scoped."""

import csv
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/paper_evidence/supplement_20261004/rtl/fc_w2_precision_probe"
WORK = ROOT / "work/supplement_20261004/fc_w2_precision_probe"
MODEL = "imports/fc/p3a_h160/data/weights/p3a_nmnist_h160_k8_int5_w2_int8_hash8"
EXPORTED = "data/weights/p3a_nmnist_h160_h480_active4_input2312"
SOURCES = [
    "rtl/memory/unified_array16_weight_refill_dispatcher_v1.sv",
    "rtl/memory/unified_h40_scalar_parameter_row_publisher_v1.sv",
    "rtl/memory/unified_fc_h40_w0_cache_refill_scheduler_v1.sv",
    "rtl/memory/unified_dual_role_weight_store_p4all_candidate_v1.sv",
    "tb/tb_fc_w2_precision_probe.sv",
]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_model():
    with np.load(ROOT / MODEL / "weights_h160_int5_w2_int8.npz", allow_pickle=False) as data:
        weights = data["W2"].copy()
    assert weights.shape == (160, 10) and weights.dtype == np.int8
    locations = []
    for tile in range(4):
        imported = ROOT / MODEL / f"w2_cluster_{tile:02d}.hex"
        exported = ROOT / EXPORTED / imported.name
        source_rows = imported.read_text(encoding="ascii").splitlines()
        packed_rows = exported.read_text(encoding="ascii").splitlines()
        assert source_rows == packed_rows and len(packed_rows) == 40
        for row, packed in enumerate(packed_rows):
            assert len(packed) == 20
            word = int(packed, 16)
            for cls in range(10):
                code = (word >> (8 * cls)) & 255
                code = code - 256 if code >= 128 else code
                hidden = tile * 40 + row
                assert code == int(weights[hidden, cls])
                locations.append({"code": code, "hidden": hidden, "class": cls,
                                  "source": f"{EXPORTED}/{imported.name}:{row + 1} bits[{8*cls+7}:{8*cls}]",
                                  "npz_field": f"{MODEL}/weights_h160_int5_w2_int8.npz:W2[{hidden},{cls}]"})
    minimum = min(locations, key=lambda item: item["code"])
    maximum = max(locations, key=lambda item: item["code"])
    return {
        "model": f"{MODEL}/weights_h160_int5_w2_int8.npz", "field": "W2",
        "shape": list(weights.shape), "dtype": str(weights.dtype), "count": len(locations),
        "minimum": minimum, "maximum": maximum,
        "outside_int5": sum(x["code"] < -16 or x["code"] > 15 for x in locations),
        "npz_imported_hex_exported_hex_equal": True,
        "packing_source": f"{MODEL}/cluster_w2_manifest.json#/layout",
    }


def main():
    stamp = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out = EVIDENCE / stamp
    out.mkdir(parents=True)
    print(f"Evidence: {out.relative_to(ROOT).as_posix()}", flush=True)
    model = read_model()
    write_json(out / "model_extrema.json", model)
    cases = [{"kind": "int5_or_int8_boundary", "code": code, "source": "directed numeric boundary"}
             for code in [0, -16, 15, -17, 16, 31, 127, -127]]
    cases.extend([dict(model["minimum"], kind="real_W2_min"),
                  dict(model["maximum"], kind="real_W2_max")])
    for i, case in enumerate(cases):
        case["case_id"] = i
        case["expected_fc"] = "ACCEPT" if -16 <= case["code"] <= 15 else "REJECT"
    write_json(out / "cases.json", cases)
    (out / "cases.mem").write_text("".join(f"{case['code'] & 65535:04x}\n" for case in cases), encoding="ascii")
    env = os.environ.copy()
    env["VERILATOR_ROOT"] = "C:/msys64/mingw64/share/verilator"
    env["PATH"] = "C:/msys64/mingw64/bin;C:/msys64/usr/bin;" + env.get("PATH", "")
    obj = WORK / stamp / "obj_dir"
    obj.parent.mkdir(parents=True, exist_ok=False)
    cmd = ["C:/msys64/mingw64/bin/verilator_bin.exe", "--binary", "--timing", "--assert",
           "-j", "1", "-Wall", "-Wno-fatal", "-Wno-DECLFILENAME", "-Wno-PINCONNECTEMPTY",
           "-Wno-UNUSEDSIGNAL", "-Wno-BLKSEQ", "-Wno-SYNCASYNCNET",
           "-CFLAGS", "-O0 -D_GLIBCXX_USE_CXX11_ABI=0",
           "--top-module", "tb_fc_w2_precision_probe", "--Mdir", obj.as_posix(),
           *[(ROOT / source).as_posix() for source in SOURCES]]
    exe = obj / "Vtb_fc_w2_precision_probe.exe"
    run_cmd = [str(exe), "+CASES=" + (out / "cases.mem").relative_to(ROOT).as_posix(),
               f"+CASE_COUNT={len(cases)}"]
    record = {"scope": "existing scheduler/publisher/dispatcher and real weight-store decoder; not final top",
              "sources": SOURCES, "build_command": cmd, "run_command": run_cmd,
              "cwd": str(ROOT), "jobs": 1, "dut_modified": False, "top_generated": False,
              "waveforms": False, "peak_process_tree_rss_bytes": "NA"}
    write_json(out / "commands.json", record)
    for stage, command in [("build", cmd), ("run", run_cmd)]:
        started = time.monotonic()
        with (out / f"{stage}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record[stage + "_exit_code"] = result.returncode
        record[stage + "_wall_seconds"] = round(time.monotonic() - started, 3)
        write_json(out / "commands.json", record)
        print(f"{stage}: exit={result.returncode}, wall={record[stage + '_wall_seconds']} s", flush=True)
        if result.returncode:
            raise SystemExit(f"{stage} failed; preserve {out / (stage + '.log')}")
    log_text = (out / "run.log").read_text(encoding="utf-8")
    assert "FC_W2_PRECISION_PROBE_PASS" in log_text
    observed = []
    for line in log_text.splitlines():
        if line.startswith("FC_W2_PRECISION_CASE "):
            row = dict(re.findall(r"(\w+)=([^\s]+)", line))
            case = cases[int(row["case_id"])]
            assert int(row["code"]) == case["code"]
            if row["route"] == "fc_scheduler":
                assert row["result"] == case["expected_fc"]
            else:
                assert row["result"] == "INT8_PACK_SRNN_DECODE_PASS"
            row.update(kind=case["kind"], source=case["source"])
            observed.append(row)
    assert len(observed) == 2 * len(cases)
    fields = sorted({key for row in observed for key in row})
    with (out / "observations.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="NA")
        writer.writeheader()
        writer.writerows(observed)
    summary = {"probe_assertions": "PASS", "final_fc_int8_load_contract": "ISSUE_REPRODUCED",
               "model": model, "cases": len(cases), "observations": len(observed),
               "final_top_simulated": False, "complete_FC_forward_simulated": False,
               "reject_not_silent_truncation": True,
               "alternate_INT8_leaf_is_not_final_FC_support": True,
               "run_log": (out / "run.log").relative_to(ROOT).as_posix()}
    write_json(out / "summary.json", summary)
    print(log_text, end="")


if __name__ == "__main__":
    main()
