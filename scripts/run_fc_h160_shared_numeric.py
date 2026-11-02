#!/usr/bin/env python3
"""Real H160/T100 inference through the common P4 array; no learning claim."""
import argparse
import datetime as dt
from itertools import islice
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fc_w2_numeric_reference import (
    ROOT, EVENTS, load_model, lif18, trace16, signed_wrap, w2_dot,
    accumulate_logits, scale_raw_to_q6,
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mem(path, values, bits):
    mask = (1 << bits) - 1
    path.write_text("".join(f"{int(v) & mask:0{(bits + 3) // 4}x}\n" for v in values), encoding="ascii")


def parse_measurements(text):
    def records(prefix):
        return [{key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", line)}
                for line in text.splitlines() if line.startswith(prefix + " ")]

    loads, passes, terminals = (records(prefix) for prefix in
        ("FC_H160_LOAD", "FC_H160_PASS_METRICS", "FC_H160_SHARED_NUMERIC_PASS"))
    if len(loads) != 1 or len(passes) != 2 or len(terminals) != 1:
        raise ValueError("expected one load, two passes and one terminal numeric PASS")
    load, terminal = loads[0], terminals[0]
    steps = terminal["steps"]
    if not 1 <= steps <= 100 or [p["pass"] for p in passes] != [0, 1]:
        raise ValueError("invalid step count or pass order")
    if (load["fills"], load["payload_bytes"], load["parameter_reads"],
        load["parameter_responses"], load["parameter_bytes"]) != (
            9408, 235200, 371520, 371520, 743040):
        raise ValueError("incomplete real-model scalar refill")
    if (terminal["hidden_checks"], terminal["output_checks"], terminal["q6_checks"]) != (
            steps * 160 * 2, steps * 10 * 2, steps * 10 * 2):
        raise ValueError("incomplete numeric comparison")
    for p in passes:
        if p["steps"] != steps or p["cycles"] != sum(p[k] for k in (
                "w0_cycles", "w2_cycles", "adapter_cycles")):
            raise ValueError("phase cycles do not match retired steps/total")
        if p["weight_read_requests"] != p["tile_issues"] or (
                p["weight_read_bytes"] != p["weight_read_requests"] * 100):
            raise ValueError("P4 physical cache traffic mismatch")
        for field in ("state_read_words", "state_write_words", "trace_read_words", "trace_write_words"):
            if p[field] != steps * 17:
                raise ValueError("missing state/trace group transfer")
        p["state_trace_interface_bytes"] = sum(p[k] * 40 for k in (
            "state_read_words", "state_write_words", "trace_read_words", "trace_write_words"))
        p["dense_model_sops"] = steps * (2312 * 160 + 160 * 10)
        if p["effective_sops"] != p["input_events"] * 160 + p["hidden_spikes"] * 10:
            raise ValueError("effective operation count mismatch")
        p["cold_cycles_including_refill_and_init"] = load["cycles"] + p["init_cycles"] + p["cycles"]
    for field in ("commands", "tile_issues", "effective_sops", "input_events", "hidden_spikes"):
        if passes[0][field] != passes[1][field]:
            raise ValueError("backpressure changed work count")
    return {"status": "PASS", "sample_id": 0, "steps": steps, "load": load,
        "checks": terminal, "passes": passes,
        "physical_array": "12xH40/P4; W0 tiles0..3, W2 tile4",
        "final_top_simulated": False, "learning_simulated": False,
        "scope": "production scalar refill, cache, array and cumulative replay router; TB source grouping and observation serialization",
        "physical_timing_applies": False, "full_dataset_accuracy": False,
        "traffic_scope": "accepted behavioral interfaces, not board/DDR traffic; cache bytes include disabled lanes/banks",
        "cold_cycle_scope": "hypothetical fresh-refill cost for each pass, not a second observed refill",
        "operation_scope": "effective versus dense binary weight-add count, not a measured speedup or PE busy utilization"}


def validate_counts(counts):
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[1] != 2312 or not 1 <= counts.shape[0] <= 100:
        raise ValueError("counts must have shape [T,2312], 1<=T<=100; no truncation is allowed")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0) or np.any(counts > 65535):
        raise ValueError("counts must be nonnegative uint16-range integers")
    return counts.astype(np.uint16)


def count_groups(frame):
    """Bank-conflict-free commands; a repeated feature retains its multiplicity."""
    for row, lanes in enumerate(np.asarray(frame).reshape(578, 4)):
        for rep in range(int(max(lanes))):
            yield row, sum(int(int(value) > rep) << lane for lane, value in enumerate(lanes))


def make_fixture(out, steps, counts=None, sample_id=0, source=None):
    model = load_model()
    w0, w2 = model["W0"], model["W2"]
    counts_mode = counts is not None
    if counts_mode:
        counts = validate_counts(counts)
        if steps > len(counts):
            raise ValueError("requested steps exceed raw sample length")
        raw_steps = len(counts)
        counts = np.pad(counts, ((0, 100-raw_steps), (0, 0)))
        frames = [np.repeat(np.arange(2312), frame.astype(np.int64)).tolist() for frame in counts]
    else:
        with EVENTS.open(encoding="ascii") as stream:
            words = [int(line.strip(), 16) for line in islice(stream, 100 * 49)]
        if len(words) != 4900:
            raise ValueError("short sample-0 T100 input")
        frames = []
        for t in range(100):
            features = [word >> 8 for word in words[t*49:(t+1)*49] if word >> 8 != 4095]
            if len(set(features)) != len(features):
                raise ValueError("duplicate event needs multiplicity-preserving scheduling")
            if any((word & 255) != 1 or word >> 8 >= 2312
                   for word in words[t*49:(t+1)*49] if word >> 8 != 4095):
                raise ValueError("input is not binary 2312-channel events")
            frames.append(features)
        raw_steps = 100
    # Physical cache: tile 0..3 W0, tile 4 W2. Source index modulo four is bank.
    packed_w0 = [sum((int(w0[fi, tile*40+lane]) & 31) << (5*lane)
                     for lane in range(40)) for tile in range(4) for fi in range(2312)]
    packed_w2 = [sum((int(w2[h, c]) & 255) << (8*c) for c in range(10)) for h in range(160)]
    mem(out / "w0.mem", packed_w0, 200)
    mem(out / "w2.mem", packed_w2, 200)
    mem(out / "parameters.mem", np.concatenate([w0.ravel(), w2.ravel()]), 16)
    if counts_mode:
        mem(out / "input_counts.mem", counts.ravel(), 16)
    else:
        masks = []
        for features in frames:
            frame = [0] * 578
            for fi in features:
                frame[fi // 4] |= 1 << (fi % 4)
            masks.extend(frame)
        mem(out / "input.mem", masks, 4)
    pre_all, trace_all, spike_all, dot_all, q6_all, cumulative_all = [], [], [], [], [], []
    voltage, trace, logits = [0]*160, [0]*160, [0]*10
    scale = round(float(model["W2_scale"]) * 64 * 65536)
    records = []
    for t, features in enumerate(frames):
        totals = (np.asarray(counts[t], dtype=np.int64) @ w0.astype(np.int64)
                  if counts_mode else np.sum(w0[features].astype(np.int64), axis=0))
        current = [signed_wrap(int(value) << 6, 18) for value in totals]
        pre, spikes, voltage = map(list, zip(*[lif18(i, v) for i, v in zip(current, voltage)]))
        trace = [trace16(e, s, fractional_bits=10) for e, s in zip(trace, spikes)]
        dot = w2_dot(w2.tolist(), spikes)
        logits = accumulate_logits(logits, dot)
        q6 = [scale_raw_to_q6(c, scale) for c in logits]
        pre_all.extend(pre)
        trace_all.extend(trace)
        spike_all.extend(spikes)
        dot_all.extend(dot)
        q6_all.extend(q6)
        cumulative_all.extend(logits)
        records.append({"t": t, "events": len(features), "hidden_spikes": sum(spikes),
                        "dot_raw": dot, "cumulative_raw": logits})
    for name, values, bits in [("pre", pre_all, 32), ("trace", trace_all, 32),
                               ("spike", spike_all, 1), ("dot", dot_all, 32),
                               ("q6", q6_all, 16), ("cumulative", cumulative_all, 32)]:
        mem(out / (name + ".mem"), values, bits)
    reference = {
        "sample_id": sample_id, "steps": steps, "raw_steps": raw_steps,
        "input_representation": "counts with repeated P4 commands" if counts_mode else "legacy binary masks",
        "weight_shapes": [[2312, 160], [160, 10]],
        "W0": "INT5, current <<6", "W2": "INT8, binary hidden-spike dot, no LIF",
        "trace": "UQ6.10", "scale_to_q6_q16": scale,
        "records": records[:steps], "prediction": int(np.argmax(records[steps-1]["cumulative_raw"])),
        "source": source if source is not None else str(EVENTS.relative_to(ROOT)),
    }
    write_json(out / "reference.json", reference)
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, choices=range(1, 101))
    parser.add_argument("--counts-npz", type=Path, help="formal counts, not legacy 49-slot events")
    parser.add_argument("--counts-key", default="dense_counts")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--reuse-executable", type=Path)
    parser.add_argument("--summarize", type=Path, help="parse an existing run without compiling or simulating")
    args = parser.parse_args()
    if args.summarize:
        out = args.summarize.resolve()
        commands = json.loads((out / "commands.json").read_text(encoding="utf-8"))
        if commands.get("run_exit") != 0:
            raise ValueError("run did not exit successfully")
        measurements = parse_measurements((out / "run.log").read_text(encoding="utf-8"))
        reference = json.loads((out / "reference.json").read_text(encoding="utf-8"))
        measurements["sample_id"] = reference["sample_id"]
        measurements["input_representation"] = reference.get("input_representation", "legacy binary masks")
        write_json(out / "measurements.json", measurements)
        print(f"Parsed existing evidence: {out}")
        return
    stamp = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out = ROOT / "results/paper_evidence/supplement_20261004/rtl/fc_w2_repair/h160" / stamp
    work = ROOT / "work/supplement_20261004/fc_h160_shared" / stamp
    out.mkdir(parents=True)
    work.mkdir(parents=True)
    counts = None
    sample_metadata = {}
    if args.counts_npz:
        with np.load(args.counts_npz, allow_pickle=False) as data:
            counts = data[args.counts_key]
            if counts.ndim == 3:
                if not 0 <= args.sample_index < len(counts):
                    raise ValueError("sample-index is outside the exported batch")
                counts = counts[args.sample_index]
            for key in ("dataset_indices", "labels", "lengths"):
                if key in data:
                    sample_metadata[key] = int(data[key][args.sample_index])
        counts = validate_counts(counts)
    steps = args.steps if args.steps is not None else (len(counts) if counts is not None else 100)
    sample_id = args.sample_id if args.sample_id is not None else sample_metadata.get("dataset_indices", args.sample_index)
    reference = make_fixture(out, steps, counts, sample_id,
                             str(args.counts_npz.resolve()) if args.counts_npz else None)
    reference["sample_metadata"] = sample_metadata
    write_json(out / "reference.json", reference)
    # Reuse the existing array-path source list, not a second RTL implementation.
    runner = (ROOT / "scripts/run_verilator_binary_p4_h40_array12_path_v1.ps1").read_text(encoding="utf-8-sig")
    sources = [s.replace("\\", "/") for s in re.findall(r"Join-Path \$root '([^']+)'", runner) if s.startswith("rtl")]
    sources += ["rtl/learning/unified_fc_srnn_h40_tp_replay_router_v1.sv", "tb/tb_fc_h160_shared_numeric.sv"]
    sources += ["rtl/memory/unified_array16_weight_refill_dispatcher_v1.sv",
                "rtl/memory/unified_h40_scalar_parameter_row_publisher_v1.sv",
                "rtl/memory/unified_fc_h40_w0_cache_refill_scheduler_v1.sv"]
    top = "tb_fc_h160_shared_numeric"
    env = os.environ.copy()
    env["VERILATOR_ROOT"] = "C:/msys64/mingw64/share/verilator"
    env["PATH"] = "C:/msys64/mingw64/bin;C:/msys64/usr/bin;" + env.get("PATH", "")
    filelist = work / "sources.f"
    filelist.write_text("\n".join('"' + (ROOT / s).as_posix() + '"' for s in sources) + "\n", encoding="ascii")
    obj = work / "obj"
    build = ["C:/msys64/mingw64/bin/verilator_bin.exe", "--binary", "--timing", "--assert",
             "-j", "1", "-Wall", "-Wno-fatal", "--output-split", "10000",
             "-CFLAGS", "-O1 -D_GLIBCXX_USE_CXX11_ABI=0", "--top-module", top,
             "--Mdir", obj.as_posix(), "-f", filelist.as_posix()]
    exe = args.reuse_executable.resolve() if args.reuse_executable else obj / ("V" + top + ".exe")
    run = [str(exe), "+FIXTURE=" + out.relative_to(ROOT).as_posix(), f"+STEPS={steps}",
           f"+COUNTS={int(counts is not None)}"]
    record = {"sources": sources, "build": build, "run": run, "cwd": str(ROOT),
              "reuse_executable": str(exe) if args.reuse_executable else None}
    print(f"Evidence: {out}", flush=True)
    for stage, command in ([] if args.reuse_executable else [("build", build)]) + [("run", run)]:
        started = time.monotonic()
        with (out / (stage + ".log")).open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record[stage + "_exit"] = proc.returncode
        record[stage + "_seconds"] = round(time.monotonic() - started, 3)
        write_json(out / "commands.json", record)
        print(f"{stage}: exit={proc.returncode} seconds={record[stage + '_seconds']}", flush=True)
        if proc.returncode:
            raise SystemExit(f"Preserved failure at {out}")
    text = (out / "run.log").read_text(encoding="utf-8")
    measurements = parse_measurements(text)
    measurements["sample_id"] = sample_id
    measurements["input_representation"] = reference["input_representation"]
    if any(p["input_events"] != sum(r["events"] for r in reference["records"])
           for p in measurements["passes"]):
        raise RuntimeError("RTL input event count differs from counts reference")
    write_json(out / "measurements.json", measurements)
    if "FC_H160_SHARED_NUMERIC_PASS" not in text:
        raise RuntimeError("missing completion marker")
    runs = [{key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", line)}
            for line in text.splitlines() if line.startswith("FC_H160_PASS_METRICS")]
    write_json(out / "summary.json", {"status": "PASS", "sample_id": sample_id, "steps": steps,
        "passes": runs, "physical_array": "12xH40/P4; W0 tiles0..3, W2 tile4",
        "final_top_simulated": False, "learning_simulated": False,
        "scope": "scalar parameters via production refill to shared P4 cache; real input to integer hidden/pre/trace and linear W2; per-sample raw24 accumulation and Q6 via production replay router with TB observation serialization",
        "physical_timing_applies": False, "full_dataset_accuracy": False})


if __name__ == "__main__":
    main()
