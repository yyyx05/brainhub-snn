#!/usr/bin/env python3
"""Parent-scheduled, raw-input same-DUT B4T4 successor. No SSH or Vivado.

Independent of the B1T2 successor files, which may be running concurrently.
Default 12 windows naturally cross codes, but require about 25 hours on the
previous simulation host. --stage files (default) never prepares or launches.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from supplement_scnn_graph_closure import GRAPH_STEM, GRAPH_TB, WARNING_FLAGS, run_logged, source_list
from supplement_scnn_same_dut_b4t4_reference import BASE, ROOT, INPUT, LABELS, generate

TOP = "tb_supplement_scnn_same_dut_b4t4"
PASS = "SCNN_SAME_DUT_B4T4_GRAPH_LEARN_NEXT_PASS"
CHECKS = ROOT / "tests/scnn_same_dut_b4t4_checks.svh"


def testbench_source(windows):
    if not 1 <= windows <= 16:
        raise ValueError("expected 1..16 windows")
    source = GRAPH_TB.read_text(encoding="utf-8-sig")

    def edit(old, new, count=1):
        nonlocal source
        if source.count(old) != count:
            raise ValueError(f"upstream TB changed: expected {count} occurrences of {old!r}")
        source = source.replace(old, new)

    edit("module tb_unified_" + GRAPH_STEM + ";", "module " + TOP + ";")
    edit("localparam int TEST_TIMESTEPS = 2;", f"""localparam int TEST_TIMESTEPS = 4;
    localparam int B4_WINDOWS = {windows};
    localparam int B4_GRAPH_FRAMES = B4_WINDOWS*16 + 4;""")
    edit("[0:TEST_TIMESTEPS*", "[0:B4_GRAPH_FRAMES*", source.count("[0:TEST_TIMESTEPS*"))
    edit("integer cycle_count = 0;", "longint unsigned cycle_count = 0;")
    edit(".MAX_BATCH(TEST_TIMESTEPS)", ".MAX_BATCH(16), .CLASSIFIER_CAPTURE_MAX_BATCH(4)")
    edit(".BATCH_COUNT_WIDTH(2)", ".BATCH_COUNT_WIDTH(5)")
    edit(".SAMPLE_INDEX_WIDTH(1), .TIMESTEP_INDEX_WIDTH(1)",
         ".SAMPLE_INDEX_WIDTH(2), .TIMESTEP_INDEX_WIDTH(2)")
    edit(".TIMESTEP_COUNT_WIDTH(2)", ".TIMESTEP_COUNT_WIDTH(3)")
    edit(".i_learning_batch_count(1'b1)", ".i_learning_batch_count(5'd4)")
    edit(".i_learning_timestep_count(2'd2)", ".i_learning_timestep_count(3'd4)")
    edit(".i_learning_profile_id(8'h31), .i_learning_tensor_id(8'h12)",
         ".i_learning_profile_id(8'h20), .i_learning_tensor_id(8'h0c)")
    edit(".i_learning_computed_trace_capture_enable(1'b0)", ".i_learning_computed_trace_capture_enable(1'b1)")
    edit(".i_learning_vote_increment(16'd1)", ".i_learning_vote_increment(16'd0)")
    edit(".i_learning_vote_threshold(16'd1)", ".i_learning_vote_threshold(16'h8000)")
    edit(".i_learning_enable(1'b0), .i_learning_release(1'b0)",
         ".i_learning_enable(b4_learning), .i_learning_release(b4_release)")
    edit(".o_learning_release_ready()", ".o_learning_release_ready(b4_release_ready)")
    edit(".i_frame_index(8'(timestep_phase))", ".i_frame_index(8'(b4_t))")
    edit(".i_sample_reset(timestep_phase == 0)", ".i_sample_reset(b4_t == 0)")
    edit(".i_sample_last(timestep_phase == TEST_TIMESTEPS-1)", ".i_sample_last(b4_t == 3)")
    edit(".i_sample_label(4'd0)", ".i_sample_label(b4_labels[b4_sample][3:0])")
    edit(".i_sample_index(1'b0), .i_timestep_index(1'(timestep_phase))",
         ".i_sample_index(2'(b4_sample)), .i_timestep_index(2'(b4_t))")
    edit("if (timestep_phase != 0) begin", "if (b4_t != 0) begin")
    edit("cfg_initial_active_pair = 1'b0;", "cfg_initial_active_pair = !classifier_pair;", 8)
    edit("context_active_pair != 1'b0", "context_active_pair != cfg_initial_active_pair", 8)
    edit("context_active_pair != 1'b1", "context_active_pair != classifier_pair")
    edit("release_active_pair != 1'b0", "release_active_pair != cfg_initial_active_pair")
    # Preserve the old read-only cache spot check across alternating code pairs.
    start = source.index("            if (integer'(context_output_channel_base) == 0) begin")
    end = source.index("    task automatic load_conv2_context;", start)
    fragment = source[start:end]
    pattern = r"dut\.u_core\.u_array\.u_array\.g_region\[2\]\.u_region\.g_tile\[0\]\s*\.u_tile\.g_p4_all_frontend\.u_weight_store\.g_bank\[([01])\]\s*\.memory\[0\]"

    def bank_pair(match):
        prefix = "dut.u_core.u_array.u_array.g_region[2].u_region.g_tile[0].u_tile.g_p4_all_frontend.u_weight_store"
        bank = int(match.group(1))
        return f"(context_active_pair ? {prefix}.g_bank[{bank+2}].memory[0] : {prefix}.g_bank[{bank}].memory[0])"

    fragment, count = re.subn(pattern, bank_pair, fragment)
    if count != 4:
        raise ValueError("unexpected tile8 spot-check shape")
    source = source[:start] + fragment + source[end:]
    edit("                else\n                    conv8_request_count <= conv8_request_count + 1;",
         "                else if (!dut.prescan_busy)\n                    conv8_request_count <= conv8_request_count + 1;")
    edit("            while (serializer_busy)\n", "            while (serializer_busy || !idle)\n")
    edit("        load_all_classifier_parameters();", "        b4_load_initial();\n        load_all_classifier_parameters();")
    edit("timestep_phase < TEST_TIMESTEPS;", "timestep_phase < B4_GRAPH_FRAMES;")
    edit("        load_reference_frame(timestep_phase);", "        b4_begin_frame();\n        load_reference_frame(b4_t);")
    edit('fixture_dir =\n                "results/paper_evidence/rtl/scnn_full_graph_same_array_20260927/t2_reference";',
         'fixture_dir = $sformatf("%s/sample%0d", b4_directory, b4_sample);')
    # Classifier references depend on the latest codes; Conv references do not.
    begin = source.index('            $readmemh($sformatf("%s/classifier_t%0d_expected_current_q8_8.hex",')
    end = source.index('            $readmemh($sformatf("%s/conv1_t%0d_expected_pre_q8_8.hex",', begin)
    source = source[:begin] + """            for (int c = 0; c < CLASSES; c++) begin
                classifier_expected_current[c] = b4_query ? b4_query_currents[b4_t*CLASSES+c] : b4_currents[b4_frame*CLASSES+c];
                classifier_expected_centered[c] = b4_query ? b4_query_logits[b4_t*CLASSES+c] : b4_logits[b4_frame*CLASSES+c];
            end
""" + source[end:]
    edit("                !classifier_cache_valid || !classifier_pair ||\n                classifier_version != CLASSIFIER_VERSION ||",
         "                !classifier_cache_valid || classifier_pair != (1 ^ (b4_commits & 1)) ||\n                classifier_version != CLASSIFIER_VERSION + 16'(b4_commits) ||")
    # Old status strings contain fixed sample, spike-count and learning claims.
    source = source.replace("learning_off=1", "b4t4_candidate=1")
    source = source.replace("sample=0 timestep=0", "bounded_b4t4_frame")
    source = source.replace("sample=0 timestep=%0d", "global_frame=%0d")
    source = re.sub(r"\bspikes=\d+(?:/\d+)?", "spikes_checked=1", source)
    edit('        $display("SCNN_Q015_REAL_CONV1_TO_CLASSIFIER_T2_SAME_ARRAY_PASS sample=0 timesteps=2 conv1_to_conv8_exact=1 state_trace_spill_restore=1 cumulative_logits_exact=1 one_shared_array=1 b4t4_candidate=1");',
         "        b4_finish();")
    edit('        $display("SUPPLEMENT_FULLGRAPH_TIMESTEP', '        b4_check_frame();\n        $display("SUPPLEMENT_FULLGRAPH_TIMESTEP')
    edit("repeat (32000000)", "repeat (64'(B4_GRAPH_FRAMES)*20000000 + 10000000)")
    edit("endmodule", CHECKS.read_text(encoding="ascii") + "\nendmodule")
    return source


def dependencies():
    paths = set(source_list(GRAPH_STEM))
    paths.update([GRAPH_TB, CHECKS, INPUT, LABELS])
    paths.add(ROOT / "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem")
    for name in ("run_scnn_same_dut_b4t4.py", "supplement_scnn_same_dut_b4t4_reference.py",
                 "supplement_scnn_graph_closure.py", "supplement_scnn_graph_reference.py",
                 "analyze_scnn_b4t4_rtl_update_contract.py", "supplement_scnn_graph_host.mk",
                 "evaluate_scnn_full_integer.py", "build_scnn_vgg9_full_frame_t2_reference.py",
                 "supplement_scnn_weight_quantization.py", f"run_verilator_{GRAPH_STEM}.ps1"):
        paths.add(ROOT / "scripts" / name)
    paths.add(ROOT / "server/run_scnn_same_dut_b4t4.sh")
    paths.add(ROOT / "scripts/assess_scnn_same_dut_b4t4_options.py")
    paths.add(ROOT / "tests/test_scnn_same_dut_b4t4.py")
    paths.add(ROOT / "tests/scnn_same_dut_b4t4_README.md")
    for bank in (*range(1, 9), 12):
        paths.add(ROOT / f"models/scnn_dvsgesture_vgg9_q0_15/bank_{bank:02d}.hex")
    for directory in ("scnn_q8_8_value_input_20260926/full_frame_t1_reference",
                      "scnn_full_graph_same_array_20260927/t2_reference"):
        paths.update((ROOT / "results/paper_evidence/rtl" / directory).glob("*.hex"))
    return sorted(paths)


def prepare(output, windows):
    deps = dependencies()
    for path in deps:
        if not path.is_file():
            raise FileNotFoundError(path)
    source = testbench_source(windows)
    output.mkdir(parents=True, exist_ok=False)
    tb = output / (TOP + ".sv")
    tb.write_text(source, encoding="ascii")
    predictions = generate(windows, output / "fixture")
    (output / "sources.f").write_text("".join('"' + path.as_posix() + '"\n' for path in [*source_list(GRAPH_STEM), tb]), encoding="utf-8")
    (output / "dependency_paths.txt").write_text("".join(p.relative_to(ROOT).as_posix() + "\n" for p in deps), encoding="utf-8")
    record = dict(status="prepared_not_elaborated_or_simulated", windows=windows, input_files=len(deps), predictions=predictions)
    (output / "preparation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="ascii")
    return record


def build_command(output, verilator, jobs):
    return [verilator, "--binary", "-j", str(jobs), "--timing", "--assert", "--prefix", "Vscgraph",
            "--top-module", TOP, "--Mdir", str(output / "obj"), *WARNING_FLAGS,
            "--output-groups", "4", "--output-split", "5000", "--output-split-cfuncs", "2000",
            "-MAKEFLAGS", "OPT_FAST=-O1 OPT_GLOBAL=-O1 -f " + str(ROOT / "scripts/supplement_scnn_graph_host.mk"),
            "-f", str(output / "sources.f")]


def summarize(output):
    run = json.loads((output / "run.json").read_text(encoding="ascii"))
    if run.get("status") != "pass" or run.get("exit_code") != 0:
        raise ValueError("RTL simulation did not pass")
    expected = json.loads((output / "fixture/expectations.json").read_text(encoding="ascii"))
    keys = ("windows", "frames", "features", "currents", "logits", "traces", "rmw", "code_writes",
            "residual_writes", "commits", "publisher_words", "query_current_changes", "query_logit_changes", "cycles")
    pattern = PASS + " " + " ".join(key + r"=(\d+)" for key in keys)
    matches = re.findall(pattern, (output / "run.log").read_text(encoding="utf-8", errors="replace"))
    if len(matches) != 1:
        raise ValueError("missing/duplicate terminal B4T4 metrics")
    metrics = dict(zip(keys, map(int, matches[0])))
    n, frames = len(expected["windows"]), expected["graph_frames"]
    checks = dict(windows=n, frames=frames, features=frames*2048, currents=frames*11,
                  logits=frames*11, traces=n*32768, commits=n, publisher_words=(n+1)*1892,
                  query_current_changes=expected["query_current_changes"], query_logit_changes=expected["query_logit_changes"])
    for key in ("rmw", "code_writes", "residual_writes"):
        checks[key] = sum(row[key] for row in expected["windows"])
    if any(metrics[k] != v for k, v in checks.items()) or metrics["cycles"] <= 0:
        raise ValueError(f"terminal metrics mismatch {metrics}, expected {checks}")
    visible = checks["code_writes"] > 0 and checks["query_current_changes"] > 0 and checks["query_logit_changes"] > 0
    summary = dict(expected, status="same_DUT_raw_graph_code_crossing_next_forward_PASS" if visible else "same_DUT_raw_graph_B4T4_residual_only_PASS",
                   same_DUT=True, code_crossing_visible=visible, metrics=metrics,
                   simulation_wall_seconds=run["wall_seconds"], final_composition_top=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("files", "prepare", "build", "run", "summarize", "all"), default="files")
    parser.add_argument("--output", type=Path, default=BASE / "server_run1")
    parser.add_argument("--windows", type=int, choices=range(1, 17), default=12)
    parser.add_argument("--verilator", default="verilator")
    parser.add_argument("--jobs", type=int, choices=range(1, 13), default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(BASE):
        parser.error("output must be inside scnn_same_dut_b4t4")
    if args.stage == "files":
        print("\n".join(path.relative_to(ROOT).as_posix() for path in dependencies()))
        return
    for stage in (("prepare", "build", "run") if args.stage == "all" else (args.stage,)):
        if stage == "prepare":
            print(json.dumps(prepare(output, args.windows), indent=2), flush=True)
        elif stage == "build":
            if sys.platform == "win32":
                parser.error("large compilation is server-only; parent owns scheduling")
            if (output / "obj").exists():
                raise FileExistsError("choose a new output for a new build")
            executable = shutil.which(args.verilator)
            if not executable:
                raise FileNotFoundError(args.verilator)
            run_logged(build_command(output, executable, args.jobs), output, "build")
        else:
            if stage == "run":
                if sys.platform == "win32":
                    parser.error("long simulation is server-only; parent owns scheduling")
                run_logged([str(output / "obj/Vscgraph"), "+SCNN_B4T4_FIXTURE=" + str(output / "fixture")], output, "run", PASS)
            print(json.dumps(summarize(output), indent=2), flush=True)


if __name__ == "__main__":
    main()
