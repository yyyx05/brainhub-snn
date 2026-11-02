#!/usr/bin/env python3
"""Fixed clean sample0 B1T20/T4 preparation, CPU only; never run RTL."""
from __future__ import annotations

import argparse
from itertools import islice
import json
import os
from pathlib import Path
import re
import time

import generate_scnn_real_final_graph as b1
import generate_scnn_real_final_b4t12 as temporal
from analyze_scnn_b4t4_rtl_update_contract import (
    read_signed_hex, round_nearest_even_ratio, saturate16, softmax_error_q016,
    write_signed_hex,
)

ROOT = b1.ROOT
BASE = ROOT / "results/paper_evidence/supplement_20261004/performance/current_source_measurements/scnn"
TOP = "tb_scnn_profile_measurement"
BODY = ROOT / "tb/include/scnn_profile_measurement_body.svh"
MONITOR = ROOT / "tb/include/scnn_profile_measurement_monitor.svh"
STEPS, WINDOW, FEATURES, CLASSES = 20, 4, 2048, 11
PARAMS = FEATURES * CLASSES
RAW = ROOT / "server_runs/scnn_subject_disjoint_ablation_20260919/payload/data/test_264x20_inputs_q8_8.hex"
LABELS = RAW.with_name("test_264_labels.hex")


def schedule():
    return [dict(frame=t, sample=0, timestep=t, time_base=t//WINDOW*WINDOW,
                 local_timestep=t % WINDOW, count=WINDOW, batch=1,
                 trace_reset=t < WINDOW, sample_reset=t == 0,
                 wait_for_window=t % WINDOW == WINDOW-1) for t in range(STEPS)]


def wrap(value, bits):
    return (value + (1 << (bits-1))) % (1 << bits) - (1 << (bits-1))


def replay(features, initial, label, learning, *, window=WINDOW):
    """Scalar integer head, continuous trace/logits, existing /640 update."""
    if not features or not initial or len(features) % window:
        raise ValueError("complete nonempty windows required")
    width, classes = len(features[0]), len(initial)
    if any(len(row) != width for row in [*features, *initial]) or not 0 <= label < classes:
        raise ValueError("invalid feature/code/label dimensions")
    codes = [x for row in initial for x in row]
    residuals = [0] * len(codes)
    trace, cumulative, gradient = [0]*width, [0]*classes, [0]*len(codes)
    result = dict(currents=[], centered=[], cumulative=[], traces=[], windows=[])
    for t, frame in enumerate(features):
        current = [saturate16(sum(x*w for x, w in zip(frame, codes[c*width:(c+1)*width])) >> 15)
                   for c in range(classes)]
        cumulative = [wrap(a+b, 32) for a, b in zip(cumulative, current)]
        centered = [saturate16(wrap(x-max(cumulative), 32)) for x in cumulative]
        trace = [wrap((old*250 >> 8)+x, 16) for old, x in zip(trace, frame)]
        if learning:
            if any(x < 0 or x > 32767 for x in trace):
                raise ValueError("trace outside the existing nonnegative adapter contract")
            errors = softmax_error_q016(centered, label)
            gradient = [wrap(g+errors[c]*(2*trace[f]), 48)
                        for c in range(classes) for f, g in enumerate(gradient[c*width:(c+1)*width])]
        for name, row in (("currents", current), ("centered", centered),
                          ("cumulative", cumulative), ("traces", trace)):
            result[name].extend(row)
        if t % window == window-1:
            deltas = [round_nearest_even_ratio(-g*640, 1 << 26) for g in gradient]
            if any(not -(1 << 17) <= d < (1 << 17) for d in deltas):
                raise ValueError("delta outside existing signed18 contract")
            before_codes, before_residuals = codes[:], residuals[:]
            for i, delta in enumerate(deltas):
                total = residuals[i]+delta
                sign = -1 if total < 0 else 1
                candidate = codes[i]+sign*(abs(total)//32768)
                codes[i] = saturate16(candidate)
                residuals[i] = 0 if candidate != codes[i] else sign*(abs(total) % 32768)
            result["windows"].append(dict(
                codes=codes[:], residuals=residuals[:], gradient=gradient[:], delta=deltas,
                rmw_mask=[int(d != 0) for d in deltas],
                counts=[sum(a != b for a, b in zip(codes, before_codes)),
                        sum(a != b for a, b in zip(residuals, before_residuals)),
                        sum(d != 0 for d in deltas), max(map(abs, deltas))]))
            gradient = [0]*len(codes)
    return result


def testbench_source():
    text = temporal.testbench_source()
    text = text[:text.index("    function automatic logic [PARAM_ADDR_WIDTH-1:0] fg_conv_base(")]
    for old, new in (
        ("module " + temporal.TOP, "module " + TOP),
        ("localparam int MAX_BATCH = 16;", "localparam int MAX_BATCH = 4;"),
        ("localparam int MAX_TIMESTEPS = 12;", "localparam int MAX_TIMESTEPS = 20;"),
        ("timestep_phase < 52;", "timestep_phase < 20;"),
        ("assign ft_i_batch_count = BATCH_COUNT_WIDTH'(tv_query ? 1 : 4);",
         "assign ft_i_batch_count = BATCH_COUNT_WIDTH'(1);"),
        ("        if (timestep_phase != 0 && timestep_phase % 16 == 0) fg_start_window();",
         "        if (timestep_phase != 0 && timestep_phase % 4 == 0) fg_start_window();"),
    ):
        text = b1.replace_once(text, old, new)
    text = text.replace("[0:52*", "[0:20*")
    text = text.replace("localparam int TEST_TIMESTEPS = 2;", "localparam int TEST_TIMESTEPS = 20;")
    text = text.replace("SCNN_TEMPORAL_FRAME_PASS", "SCNN_PROFILE_FRAME_NUMERIC_PASS")
    # Golden loops are zero-delay simulator work; count their cycle spans, not
    # CPU seconds. The inherited driver still provides its original clock edges.
    text = text.replace("        run_current_layer();", "        run_current_layer();\n        pm_golden_begin();")
    text = b1.replace_once(text, "        fg_check_frame();", "        pm_golden_end();\n        fg_check_frame();")
    inherited = b1.BODY.read_text(encoding="ascii")
    helpers = "".join(temporal.extract_function(inherited, name)
                      for name in ("fg_conv_base", "fg_bank", "fg_ddr_code"))
    helpers += "".join(temporal.extract_task(inherited, name)
                       for name in ("fg_initialize_top", "fg_write_row", "fg_read_row"))
    continued = temporal.BODY.read_text(encoding="ascii")
    ddr = continued[continued.index("    always @(posedge clk) begin : temporal_external_ddr"):
                    continued.index("    task automatic fg_load_expected;")]
    start = ddr.index("                if (tv_window == 2 && index == 18433) begin")
    end = ddr.index("                fg_rmw_seen[index]", start)
    ddr = ddr[:start] + ddr[end:]
    helpers += ddr
    helpers += "".join(temporal.extract_task(continued, name) for name in (
        "fg_start_window", "tv_load_frame", "restore_tile_context", "fg_classifier_restore",
        "service_release", "run_current_layer", "fg_check_store"))
    helpers = helpers.replace("SCNN_TEMPORAL_LAYER", "SCNN_PROFILE_LAYER")
    helpers = helpers.replace("tv_contributions[index] > 16", "tv_contributions[index] > 4")
    helpers = b1.replace_once(helpers, "        longint unsigned started;",
                             "        longint unsigned started;\n        pm_golden_end();")
    checks = temporal.CHECKS.read_text(encoding="ascii")
    checks = checks[:checks.index("    task automatic fg_finish;")]
    checks = checks.replace("[0:52*", "[0:20*").replace("[0:48*", "[0:20*")
    checks = checks.replace("tv_publish_seen [0:3]", "tv_publish_seen [0:5]")
    checks = checks.replace("frame = int'(tv_window)*16+(trace_row%4)*4+trace_row/4;",
                            "frame = int'(tv_window)*4+trace_row;")
    checks = checks.replace("trace_row >= 16", "trace_row >= 4").replace("publication > 3", "publication > 5")
    checks = checks.replace("tv_traces_checked != 32768*(int'(tv_window)+1)",
                            "tv_traces_checked != 8192*(int'(tv_window)+1)")
    return (text + helpers + temporal.scalar_helpers() + BODY.read_text(encoding="ascii") + checks +
            MONITOR.read_text(encoding="ascii") + "\nendmodule\n`default_nettype wire\n")


def compute_reference():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import numpy as np
    import torch
    from evaluate_scnn_full_integer import IntegerNetwork, check_prefix

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    with RAW.open(encoding="ascii") as stream:
        words = [int(line, 16) for line in islice((s for s in stream if s.strip()), STEPS*2048)]
    if len(words) != STEPS*2048 or any(not 0 <= x <= 32767 for x in words):
        raise ValueError("missing original sample0 T20 signed16 nonnegative raw input")
    raw = np.asarray(words, dtype=np.int64).reshape(1, STEPS, 2, 32, 32)
    prefix = b1.ROOT / "results/paper_evidence/rtl/scnn_full_graph_same_array_20260927/t2_reference"
    if words[:4096] != read_signed_hex(prefix / "input_sample0_t0_t1_q8_8.hex", 4096):
        raise ValueError("sample0 differs from the validated clean T0/T1 prefix")
    label = int(LABELS.read_text(encoding="ascii").splitlines()[0], 16)
    features, probes = [], []

    def capture(t, layer, values):
        if layer == 0:
            probes.append([])
        if layer < 8:
            value = values["next"][0].cpu().numpy()
            c, h, w = value.shape
            probes[t].extend([int(value[0, 0, 0]), int(value[c//2, h//2, w//2]), int(value[-1, -1, -1])])
            if layer == 7:
                features.append(values["pooled"][0].flatten().cpu().numpy().tolist())

    network = IntegerNetwork("cpu")
    prefix_checks = check_prefix(network)
    network.forward(torch.from_numpy(raw), sample_ids=[0], verify_ids=[0], capture=capture)
    initial = network.codes[-1].tolist()
    references = {mode: replay(features, initial, label, mode == "on") for mode in ("off", "on")}
    return dict(raw=words, features=[x for row in features for x in row], probes=[x for row in probes for x in row],
                initial=[x for row in initial for x in row], label=label, references=references,
                prefix_checks=prefix_checks, seconds=round(time.perf_counter()-started, 3))


def write_fixture(output, result):
    output = Path(output).resolve()
    if not output.is_relative_to(BASE.resolve()):
        raise ValueError("new fixture must stay under the dedicated SCNN measurement directory")
    output.mkdir(parents=True, exist_ok=False)
    inventory = []
    common = {"raw.hex": result["raw"], "features.hex": result["features"],
              "probes.hex": result["probes"], "initial_codes.hex": result["initial"], "label.hex": [result["label"]]}
    for name, values in common.items():
        write_signed_hex(output / name, values)
        inventory.append(dict(path=name, words=len(values), bits=16))
    for mode, ref in result["references"].items():
        directory = output / mode
        directory.mkdir()
        arrays = {k: ref[k] for k in ("currents", "centered", "traces", "cumulative")}
        arrays["counts"] = [n for w in ref["windows"] for n in w["counts"]]
        for name in ("codes", "residuals", "gradient", "delta", "rmw_mask"):
            arrays[name] = [n for w in ref["windows"] for n in w[name]]
        for name, values in arrays.items():
            bits = {"gradient": 48, "delta": 18, "cumulative": 32}.get(name, 16)
            write_signed_hex(directory / (name + ".hex"), values, bits)
            inventory.append(dict(path=mode+"/"+name+".hex", words=len(values), bits=bits))
    manifest = dict(status="prepared_reference_not_RTL_PASS", sample=0, batch=1, timesteps=20,
                    window_timesteps=4, time_bases=[0, 4, 8, 12, 16], label=result["label"],
                    corruption="none; original clean sample0", seed="no RNG used; no seed/model alteration",
                    multiplier=640, protocol="underfilled B1 of the existing T4/640 window; not B1-normalized or a formal B4 adaptation episode",
                    state="cold t0 only; continuous observed membrane/trace/logits across five windows; no fresh query",
                    frame_schedule=schedule(), files=inventory, prefix_checks=result["prefix_checks"],
                    reference_cpu_seconds=result["seconds"],
                    windows={m: [dict(window=i+1, code_changes=w["counts"][0], residual_changes=w["counts"][1],
                                     rmw=w["counts"][2]) for i, w in enumerate(r["windows"])]
                             for m, r in result["references"].items()},
                    scope="fixed-input CPU oracle only; pooled features, 24 post-LIF probes/frame and scalar integer classifier/update; no RTL throughput yet")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="ascii")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(BASE.resolve()):
        parser.error("fresh output under the dedicated SCNN measurement directory required")
    print(json.dumps(write_fixture(output, compute_reference()), indent=2))


if __name__ == "__main__":
    main()
