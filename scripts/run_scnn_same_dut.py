#!/usr/bin/env python3
"""Prepare a same-DUT SCNN numeric test; compilation requires --stage all/build.

Run on the parent's scheduled Linux Verilator host, never starts SSH or Vivado.
Reuses the existing real fullgraph stimulus and every per-layer assertion in a
NEW generated TB. No production source or historical test/evidence is edited.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from supplement_scnn_graph_closure import GRAPH_STEM, GRAPH_TB, WARNING_FLAGS, run_logged, source_list
from supplement_scnn_same_dut_reference import ROOT, write_fixture

BASE = ROOT / "results/paper_evidence/supplement_20261004/rtl/scnn_same_dut"
TOP = "tb_supplement_scnn_same_dut"
PASS = "SCNN_SAME_DUT_GRAPH_LEARN_NEXT_PASS"
CHECKS = ROOT / "tests/scnn_same_dut_checks.svh"


def testbench_source() -> str:
    source = GRAPH_TB.read_text(encoding="utf-8-sig")

    def edit(old, new, count=1):
        nonlocal source
        if source.count(old) != count:
            raise ValueError(f"upstream TB changed: expected {count} occurrences of {old!r}")
        source = source.replace(old, new)

    edit("module tb_unified_" + GRAPH_STEM + ";", "module " + TOP + ";")
    # Global frame indices 0,1 are the learning window; 2 is a fresh t0 sample.
    # Keep hardware learner capacity at B1T2. Enlarge only scoreboard seen bits.
    edit("[0:TEST_TIMESTEPS*", "[0:3*", source.count("[0:TEST_TIMESTEPS*"))
    edit("timestep_phase < TEST_TIMESTEPS;", "timestep_phase < 3;")
    edit("load_reference_frame(timestep_phase);", """load_reference_frame(timestep_phase == 2 ? 0 : timestep_phase);
        for (int c = 0; c < CLASSES; c++) begin
            classifier_expected_current[c] = sd_current[timestep_phase*CLASSES+c];
            classifier_expected_centered[c] = sd_logits[timestep_phase*CLASSES+c];
        end
        if (timestep_phase == 2) begin
            // Public maintenance ports reset only the fresh sample's state.
            // Parameters, residuals, publication version and DUT reset persist.
            for (int tile = 0; tile < 12; tile++)
                clear_tile_state_at_work_base(tile, CLASSIFIER_WORK_BASE);
        end""")
    edit("if (timestep_phase != 0) begin", "if (timestep_phase == 1) begin")
    edit(".i_frame_index(8'(timestep_phase))", ".i_frame_index(8'(timestep_phase == 2 ? 0 : timestep_phase))")
    edit(".i_sample_reset(timestep_phase == 0)", ".i_sample_reset(timestep_phase != 1)")
    edit(".i_sample_last(timestep_phase == TEST_TIMESTEPS-1)", ".i_sample_last(timestep_phase != 0)")
    edit(".i_sample_label(4'd0)", ".i_sample_label(sd_label[0][3:0])")
    edit(".i_timestep_index(1'(timestep_phase))", ".i_timestep_index(1'(timestep_phase == 2 ? 0 : timestep_phase))")
    edit(".i_learning_enable(1'b0), .i_learning_release(1'b0)",
         ".i_learning_enable(sd_learning), .i_learning_release(sd_release)")
    edit(".o_learning_release_ready()", ".o_learning_release_ready(sd_release_ready)")
    edit(".i_learning_vote_increment(16'd1)", ".i_learning_vote_increment(16'd0)")
    edit(".i_learning_vote_threshold(16'd1)", ".i_learning_vote_threshold(16'h8000)")
    edit(".i_learning_computed_trace_capture_enable(1'b0)", ".i_learning_computed_trace_capture_enable(1'b1)")
    edit(".i_learning_profile_id(8'h31), .i_learning_tensor_id(8'h12)",
         ".i_learning_profile_id(8'h20), .i_learning_tensor_id(8'h0c)")
    edit("cfg_initial_active_pair = 1'b0;", "cfg_initial_active_pair = !classifier_pair;", 8)
    edit("context_active_pair != 1'b0", "context_active_pair != cfg_initial_active_pair", 8)
    edit("context_active_pair != 1'b1", "context_active_pair != classifier_pair")
    edit("release_active_pair != 1'b0", "release_active_pair != cfg_initial_active_pair")
    # The old tile8 spot check uses constant bank indices, so select the actual
    # bank pair in its read-only expression. Weight-fill stimulus is unchanged.
    start = source.index("            if (integer'(context_output_channel_base) == 0) begin")
    end = source.index("    task automatic load_conv2_context;", start)
    fragment = source[start:end]
    pattern = r"dut\.u_core\.u_array\.u_array\.g_region\[2\]\.u_region\.g_tile\[0\]\s*\.u_tile\.g_p4_all_frontend\.u_weight_store\.g_bank\[([01])\]\s*\.memory\[0\]"
    def bank_pair(match):
        prefix = "dut.u_core.u_array.u_array.g_region[2].u_region.g_tile[0].u_tile.g_p4_all_frontend.u_weight_store"
        bank = int(match.group(1))
        return f"(context_active_pair ? {prefix}.g_bank[{bank+2}].memory[0] : {prefix}.g_bank[{bank}].memory[0])"
    fragment, replacements = re.subn(pattern, bank_pair, fragment)
    if replacements != 4:
        raise ValueError("cannot adapt tile8 cache-pair check")
    source = source[:start] + fragment + source[end:]
    # Prescan requests use the same external feature service but are not MAC
    # requests. Preserve all original exact convolution request checks.
    edit("                else\n                    conv8_request_count <= conv8_request_count + 1;",
         "                else if (!dut.prescan_busy)\n                    conv8_request_count <= conv8_request_count + 1;")
    edit("        load_all_classifier_parameters();", "        sd_load_expected();\n        load_all_classifier_parameters();")
    edit("                !classifier_cache_valid || !classifier_pair ||\n                classifier_version != CLASSIFIER_VERSION ||",
         "                !classifier_cache_valid ||\n                ((timestep_phase != 1 || !sd_learn_on) &&\n                 (classifier_pair != !(sd_learn_on && timestep_phase >= 2) ||\n                  classifier_version != CLASSIFIER_VERSION + 16'(sd_learn_on && timestep_phase >= 2))) ||")
    # Do not inherit old log labels claiming learning_off=1 or t0 for every frame.
    source = source.replace("learning_off=1", "same_dut_candidate=1")
    source = source.replace("sample=0 timestep=0", "sample=0 bounded_frame")
    source = re.sub(r"\bspikes=\d+(?:/\d+)?", "spikes_checked=1", source)
    edit('        $display("SCNN_Q015_REAL_CONV1_TO_CLASSIFIER_T2_SAME_ARRAY_PASS sample=0 timesteps=2 conv1_to_conv8_exact=1 state_trace_spill_restore=1 cumulative_logits_exact=1 one_shared_array=1 same_dut_candidate=1");',
         '        sd_finish();')
    edit("        $display(\"SUPPLEMENT_FULLGRAPH_TIMESTEP", "        sd_check_frame();\n        $display(\"SUPPLEMENT_FULLGRAPH_TIMESTEP")
    edit("repeat (32000000)", "repeat (52000000)")
    edit("            while (serializer_busy)\n", "            while (serializer_busy || !idle)\n")
    edit("endmodule", CHECKS.read_text(encoding="ascii") + "\nendmodule")
    return source


def dependencies() -> list[Path]:
    paths = set(source_list(GRAPH_STEM))
    paths.update([GRAPH_TB, CHECKS])
    paths.add(ROOT / "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem")
    for filename in ("run_scnn_same_dut.py", "supplement_scnn_same_dut_reference.py",
                     "supplement_scnn_graph_closure.py", "supplement_scnn_graph_reference.py",
                     "analyze_scnn_b4t4_rtl_update_contract.py", "supplement_scnn_graph_host.mk",
                     f"run_verilator_{GRAPH_STEM}.ps1"):
        paths.add(ROOT / "scripts" / filename)
    paths.add(ROOT / "server/run_scnn_same_dut.sh")
    for bank in (*range(1, 9), 12):
        paths.add(ROOT / f"models/scnn_dvsgesture_vgg9_q0_15/bank_{bank:02d}.hex")
    for dirname in ("scnn_q8_8_value_input_20260926/full_frame_t1_reference",
                    "scnn_full_graph_same_array_20260927/t2_reference"):
        paths.update((ROOT / "results/paper_evidence/rtl" / dirname).glob("*.hex"))
    paths.add(ROOT / "imports/scnn/datasets/DVSGesture/hardware/test_10_labels.hex")
    return sorted(paths)


def build_command(output: Path, verilator: str, jobs: int) -> list[str]:
    return [verilator, "--binary", "-j", str(jobs), "--timing", "--assert",
            "--prefix", "Vscgraph", "--top-module", TOP, "--Mdir", str(output / "obj"),
            *WARNING_FLAGS, "--output-groups", "4", "--output-split", "5000",
            "--output-split-cfuncs", "2000", "-MAKEFLAGS",
            "OPT_FAST=-O1 OPT_GLOBAL=-O1 -f " + str(ROOT / "scripts/supplement_scnn_graph_host.mk"),
            "-f", str(output / "sources.f")]


def prepare(output: Path) -> dict:
    deps = dependencies()
    for path in deps:
        if not path.is_file():
            raise FileNotFoundError(path)
    source = testbench_source()
    output.mkdir(parents=True, exist_ok=False)
    tb = output / (TOP + ".sv")
    tb.write_text(source, encoding="ascii")
    predictions = {arm: write_fixture(output / ("fixture_" + arm), arm == "on") for arm in ("on", "off")}
    (output / "sources.f").write_text("".join('"' + path.as_posix() + '"\n' for path in [*source_list(GRAPH_STEM), tb]), encoding="utf-8")
    (output / "dependency_paths.txt").write_text("".join(path.relative_to(ROOT).as_posix() + "\n" for path in deps), encoding="utf-8")
    record = dict(status="prepared_not_compiled_or_simulated", input_files=len(deps), predictions=predictions)
    (output / "preparation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="ascii")
    return record


def summarize(output: Path, arm: str) -> dict:
    result = json.loads((output / f"run_{arm}.json").read_text(encoding="utf-8"))
    if result.get("status") != "pass" or result.get("exit_code") != 0:
        raise ValueError("simulation has not successfully completed")
    log = (output / f"run_{arm}.log").read_text(encoding="utf-8", errors="replace")
    pattern = (PASS + r" learning=(\d+) frames=3 features=(\d+) currents=(\d+) logits=(\d+) "
               r"traces=(\d+) rmw=(\d+) code_writes=(\d+) residual_writes=(\d+) commits=(\d+) "
               r"publisher_words=(\d+) cycles=(\d+)")
    matches = re.findall(pattern, log)
    if len(matches) != 1:
        raise ValueError("missing/duplicate terminal same-DUT metrics")
    keys = ("learning", "features", "currents", "logits", "traces", "rmw", "code_writes",
            "residual_writes", "commits", "publisher_words", "cycles")
    metrics = dict(zip(keys, map(int, matches[0])))
    expected = json.loads((output / f"fixture_{arm}/expectations.json").read_text(encoding="ascii"))
    on = arm == "on"
    checks = dict(learning=int(on), features=6144, currents=33, logits=33,
                  traces=4096 if on else 0, rmw=expected["expected_rmw_transactions"],
                  code_writes=expected["expected_code_changes"],
                  residual_writes=expected["expected_residual_changes"], commits=int(on),
                  publisher_words=1892 * (2 if on else 1))
    if any(metrics[key] != value for key, value in checks.items()) or metrics["cycles"] <= 0:
        raise ValueError(f"terminal metrics mismatch: {metrics}, expected {checks}")
    summary = dict(expected, status="same_DUT_bounded_graph_numeric_PASS", metrics=metrics,
                   same_DUT_graph_learning=on, full_graph_after_commit=on,
                   final_composition_top=False, simulation_wall_seconds=result["wall_seconds"])
    (output / f"summary_{arm}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "build", "run", "summarize", "all", "files"), default="files")
    parser.add_argument("--output", type=Path, default=BASE / "server_run1")
    parser.add_argument("--learning", choices=("on", "off", "both"), default="both")
    parser.add_argument("--verilator", default="verilator")
    parser.add_argument("--jobs", type=int, choices=range(1, 13), default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(BASE):
        parser.error("output must be under the new scnn_same_dut evidence directory")
    if args.stage == "files":
        print("\n".join(path.relative_to(ROOT).as_posix() for path in dependencies()))
        return
    for stage in (("prepare", "build", "run") if args.stage == "all" else (args.stage,)):
        if stage == "prepare":
            print(json.dumps(prepare(output), indent=2), flush=True)
        elif stage == "build":
            if sys.platform == "win32":
                parser.error("large graph compilation is server-only; parent owns scheduling")
            if (output / "obj").exists():
                raise FileExistsError("choose a new output for a new source build")
            verilator = shutil.which(args.verilator)
            if not verilator:
                raise FileNotFoundError(args.verilator)
            run_logged(build_command(output, verilator, args.jobs), output, "build")
        else:
            for arm in (("on", "off") if args.learning == "both" else (args.learning,)):
                if stage == "run":
                    run_logged([str(output / "obj/Vscgraph"), "+SCNN_SAME_DUT_FIXTURE=" + str(output / ("fixture_" + arm)),
                                "+SCNN_LEARNING=" + str(int(arm == "on"))], output, "run_" + arm, PASS)
                print(json.dumps(summarize(output, arm), indent=2), flush=True)


if __name__ == "__main__":
    main()
