#!/usr/bin/env python3
"""Independent B4T12/query4 extension; never edits the active B1T2 sources."""
from pathlib import Path
import re

import generate_scnn_real_final_graph as b1

ROOT = b1.ROOT
TOP = "tb_scnn_real_final_b4t12"
BODY = ROOT / "tb/include/scnn_real_final_b4t12_body.svh"
CHECKS = ROOT / "tests/scnn_real_final_b4t12_checks.svh"


def extract_task(text, name):
    match = re.search(r"    task automatic " + name + r"\b.*?    endtask\n", text, re.S)
    if not match:
        raise ValueError(f"missing inherited task {name}")
    return match.group()


def extract_function(text, name):
    match = re.search(r"    function automatic [^\n]*\b" + name + r"\(.*?    endfunction\n", text, re.S)
    if not match:
        raise ValueError(f"missing inherited function {name}")
    return match.group()


def scalar_helpers():
    lines = ["    task automatic tv_record_scalar(input int layer, input int index,",
             "        input logic signed [15:0] pre, input logic spike);",
             "        int area, size, channels, post, probe_index;",
             "        case (layer)"]
    for layer in range(1, 9):
        lines.append(f"            {layer}: begin size = CONV{layer}_SPATIAL; channels = CONV{layer}_CHANNELS; end")
    lines += ["            default: $fatal(1, \"invalid scalar layer\");", "        endcase",
              "        area = size*size;",
              "        if (index < 0 || index >= channels*area) $fatal(1, \"invalid scalar index\");",
              "        if (spike != (pre >= 16'sd256)) $fatal(1, \"scalar threshold mismatch\");",
              "        post = int'(pre) - (spike ? 256 : 0);",
              "        case (layer)"]
    for layer in range(1, 9):
        lines.append(f"            {layer}: begin conv{layer}_observed_pre[index] = pre; conv{layer}_observed_spike[index] = spike; conv{layer}_observed_next[index] = 16'(post); end")
    lines += ["        endcase", "        for (int probe = 0; probe < 3; probe++) begin",
              "            case (probe)", "                0: probe_index = 0;",
              "                1: probe_index = (channels/2)*area+(size/2)*size+size/2;",
              "                2: probe_index = channels*area-1;", "            endcase",
              "            if (index == probe_index) begin",
              "                if (tv_probe_seen[timestep_phase*24+(layer-1)*3+probe] ||",
              "                    post != int'(tv_membrane_probes[timestep_phase*24+(layer-1)*3+probe]))",
              "                    $fatal(1, \"independent membrane probe frame=%0d layer=%0d probe=%0d got=%0d expected=%0d\",",
              "                        timestep_phase, layer, probe, post, tv_membrane_probes[timestep_phase*24+(layer-1)*3+probe]);",
              "                tv_probe_seen[timestep_phase*24+(layer-1)*3+probe] = 1; tv_probes_checked++;",
              "            end", "        end", "    endtask",
              "    function automatic logic tv_pooled_occupancy(input int layer, input int index);",
              "        int size, pooled, channel, row, col, source_index; logic value;",
              "        case (layer)", "            2: size = CONV2_SPATIAL;",
              "            4: size = CONV4_SPATIAL;", "            6: size = CONV6_SPATIAL;",
              "            default: size = CONV8_SPATIAL;", "        endcase",
              "        pooled = size/2; channel = index/(pooled*pooled); row = (index/pooled)%pooled; col = index%pooled;",
              "        value = 0;",
              "        for (int y = 0; y < 2; y++) for (int x = 0; x < 2; x++) begin",
              "            source_index = channel*size*size+(row*2+y)*size+col*2+x;",
              "            case (layer)", "                2: value |= conv2_observed_spike[source_index];",
              "                4: value |= conv4_observed_spike[source_index];",
              "                6: value |= conv6_observed_spike[source_index];",
              "                8: value |= conv8_observed_spike[source_index];",
              "            endcase", "        end", "        return value;", "    endfunction"]
    return "\n".join(lines) + "\n"


def testbench_source():
    source = b1.testbench_source()
    # Keep the raw graph tasks/scoreboards and public final-top wiring. Replace
    # the appended B1 schedule/models, not the active B1 source files.
    source = source[:source.index("    integer fg_learn_on = 1;")]
    source = b1.replace_once(source, "module " + b1.TOP, "module " + TOP)
    source = b1.replace_once(source, "localparam int MAX_BATCH = 4;", "localparam int MAX_BATCH = 16;")
    source = b1.replace_once(source, "localparam int MAX_TIMESTEPS = 2;", "localparam int MAX_TIMESTEPS = 12;")
    source = source.replace("[0:3*", "[0:52*")
    source = b1.replace_once(source, "timestep_phase < 3;", "timestep_phase < 52;")
    # This simulator lowers repeat counters to signed 32 bits, even for a
    # sized 64-bit literal. Nested positive counts retain the 3e9-cycle limit.
    source = b1.replace_once(source, "repeat (160000000)", "repeat (3) repeat (1000000000)")
    replacements = {
        "assign ft_i_batch_count = BATCH_COUNT_WIDTH'(1);": "assign ft_i_batch_count = BATCH_COUNT_WIDTH'(tv_query ? 1 : 4);",
        "assign ft_i_timestep = '0;": "assign ft_i_timestep = TIMESTEP_WIDTH'(tv_query ? 0 : tv_window*4);",
        "assign ft_i_timestep_count = TIMESTEP_COUNT_WIDTH'(timestep_phase == 2 ? 1 : 2);": "assign ft_i_timestep_count = TIMESTEP_COUNT_WIDTH'(4);",
        "assign ft_i_labels_flat = { {(MAX_BATCH-1)*CLASS_WIDTH{1'b0}}, fg_label[0][CLASS_WIDTH-1:0] };":
            "assign ft_i_labels_flat = {{(MAX_BATCH-4)*CLASS_WIDTH{1'b0}}, fg_label[3][CLASS_WIDTH-1:0], fg_label[2][CLASS_WIDTH-1:0], fg_label[1][CLASS_WIDTH-1:0], fg_label[0][CLASS_WIDTH-1:0]};",
        "assign ft_i_frame_index = FRAME_WIDTH'(timestep_phase == 2 ? 0 : timestep_phase);": "assign ft_i_frame_index = FRAME_WIDTH'(tv_time);",
        "assign ft_i_target_trace_state_reset = 1'b1;": "assign ft_i_target_trace_state_reset = !tv_query && tv_window == 0;",
        "        if (timestep_phase == 2) fg_start_window();": "        if (timestep_phase != 0 && timestep_phase % 16 == 0) fg_start_window();",
        "        load_reference_frame(timestep_phase == 2 ? 0 : timestep_phase);": "        tv_load_frame();",
        "(timestep_phase != 1 && classifier_version != fg_window_version)":
            "(classifier_version != fg_window_version && !(tv_window_end && !tv_query && classifier_version == fg_window_version+1))",
    }
    for old, new in replacements.items():
        source = b1.replace_once(source, old, new)
    source = b1.replace_task(source, "load_reference_frame")
    # Remove old sample0 oracle reads. Keep all nine actual weight bank reads.
    source = re.sub(r'        \$readmemh\(\s*"results/[^"\n]+"\s*,\s*\w+\);\n', "", source)
    if "full_frame_t1_reference" in source or "t2_reference" in source:
        raise ValueError("old B1 oracle remains in temporal stimulus")
    for layer in range(1, 9):
        # Full observed state/trace backing is sample-banked; golden probes never
        # populate it. Keep per-frame scalar duplicate/count assertions.
        for kind in ("state", "trace"):
            source = b1.replace_once(source,
                f"conv{layer}_{kind}_backing [0:CONV{layer}_SCALARS-1]",
                f"conv{layer}_{kind}_backing [0:4*CONV{layer}_SCALARS-1]")
            source = source.replace(f"conv{layer}_{kind}_backing[index]",
                                    f"conv{layer}_{kind}_backing[tv_sample*CONV{layer}_SCALARS+index]")
        numeric = f"scalar_pre_state !==\n                                conv{layer}_expected_pre[expected_index] ||\n                            scalar_spike !==\n                                conv{layer}_expected_spike[expected_index]"
        source = b1.replace_once(source, numeric, "scalar_spike !== (scalar_pre_state >= 16'sd256)")
        marker = f"                    if (scalar_channel >= CONV{layer}_CHANNELS ||"
        source = b1.replace_once(source, marker,
            f"                    tv_record_scalar({layer}, expected_index, scalar_pre_state, scalar_spike);\n" + marker)
        for kind in ("pre", "next", "spike"):
            source = source.replace(f"conv{layer}_expected_{kind}", f"conv{layer}_observed_{kind}")
    for layer in (2, 4, 6):
        source = re.sub(rf"conv{layer}_expected_pooled_spike\[(\w+)\]", rf"tv_pooled_occupancy({layer}, \1)", source)
    source = source.replace("SCNN_REAL_FINAL_FRAME_PASS sample=0 timestep=%0d conv1_to_conv8_exact=1",
        "SCNN_TEMPORAL_FRAME_PASS frame=%0d conv_membrane_probes_exact=1 full_transfer_consistency=1")
    source = source.replace("SCNN_REAL_FINAL_LAYER", "SCNN_TEMPORAL_LAYER")
    # Use the already reviewed public DDR mapping and maintenance transport.
    inherited = b1.BODY.read_text(encoding="ascii")
    helpers = "".join(extract_function(inherited, name) for name in ("fg_conv_base", "fg_bank", "fg_ddr_code"))
    helpers += "".join(extract_task(inherited, name) for name in ("fg_initialize_top", "fg_write_row", "fg_read_row"))
    return source + helpers + scalar_helpers() + BODY.read_text(encoding="ascii") + "\n" + CHECKS.read_text(encoding="ascii") + "\nendmodule\n`default_nettype wire\n"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x", encoding="ascii") as stream:
        stream.write(testbench_source())
