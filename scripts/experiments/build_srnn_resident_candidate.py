#!/usr/bin/env python3
"""Generate a narrowly scoped residency overlay; accepted production is read-only."""
import argparse
import difflib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if os.name == "nt" and not str(ROOT).startswith("\\\\?\\"):
    ROOT = Path("\\\\?\\" + str(ROOT))
EXPERIMENT = ROOT / "verification/experiments/srnn_resident_weight_context"


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"expected one anchor, got {text.count(old)}: {old[:100]}")
    return text.replace(old, new, 1)


def transform(relative, text):
    name = Path(relative).name
    if name == "unified_scnn_h40_array16_selected_layers_train_infer_system_v1.sv":
        text = once(text, "    logic m2_cache_evicted;", "    logic m2_cache_evicted;\n    logic weight_fill_context;\n    logic [1:0] topology_weight_fill_grant;")
        text = once(text, "    assign m2_cache_evicted = |(weight_fill_valid & weight_fill_ready) &&\n        !(|(m2_weight_fill_valid & m2_weight_fill_ready));",
                    "    assign weight_fill_context = o_cache_fill_grant == 2'd2 && topology_weight_fill_grant == 2'd2;\n"
                    "    assign m2_cache_evicted = 1'b0;")
        text = once(text, ".o_grant(), .o_busy(topology_weight_fill_busy),", ".o_grant(topology_weight_fill_grant), .o_busy(topology_weight_fill_busy),")
        anchor = next(line for line in text.splitlines() if "input  wire logic" in line and "i_srnn_h40_cache_invalidate," in line)
        text = once(text, anchor, anchor + "\n    input  wire logic                         i_srnn_h40_wh1_cache_invalidate,")
        anchor = "            .i_cache_invalidate(i_srnn_h40_cache_invalidate || m2_cache_evicted),"
        text = once(text, anchor, anchor + "\n            .i_wh1_cache_invalidate(i_srnn_h40_wh1_cache_invalidate),")
        text = once(text, "        .i_weight_fill_valid(weight_fill_valid),",
                    "        .i_weight_fill_context(weight_fill_context),\n        .i_weight_fill_valid(weight_fill_valid),")
    elif name in (
        "unified_h40_array16_three_topology_top_v1.sv",
        "unified_binary_p4_h40_array12_path_v1.sv",
        "unified_dual_role_h40_stationary_array12_candidate_v1.sv",
        "unified_dual_role_h40_stationary_region4_candidate_v1.sv",
        "unified_dual_role_h40_stationary_tile_candidate_v1.sv",
    ):
        anchor = next(line for line in text.splitlines() if "input  wire logic" in line and "i_weight_fill_valid" in line)
        text = once(text, anchor, "    input  wire logic                          i_weight_fill_context,\n" + anchor)
        if name == "unified_h40_array16_three_topology_top_v1.sv":
            anchor = "        .i_weight_fill_valid(maintenance_weight_fill_valid),"
            text = once(text, anchor, "        .i_weight_fill_context(i_weight_fill_context),\n" + anchor)
        elif name == "unified_binary_p4_h40_array12_path_v1.sv":
            anchor = "        .i_weight_fill_valid(i_weight_fill_valid[2:0]),"
            text = once(text, anchor, "        .i_weight_fill_context(i_weight_fill_context),\n" + anchor)
        elif name == "unified_dual_role_h40_stationary_array12_candidate_v1.sv":
            text = once(text, ".P4_ALL_CASCADE(1'b1),", ".P4_ALL_CASCADE(1'b1),\n                .M2_WEIGHT_CONTEXT(region < 2),")
            anchor = "                .i_weight_fill_valid(i_weight_fill_valid[region]),"
            text = once(text, anchor, "                .i_weight_fill_context(i_weight_fill_context),\n" + anchor)
        elif name == "unified_dual_role_h40_stationary_region4_candidate_v1.sv":
            text = once(text, "    parameter bit P4_ALL_CASCADE = 1'b0,", "    parameter bit P4_ALL_CASCADE = 1'b0,\n    parameter bit M2_WEIGHT_CONTEXT = 1'b0,")
            text = once(text, ".P4_ALL_CASCADE(P4_ALL_CASCADE)\n", ".P4_ALL_CASCADE(P4_ALL_CASCADE),\n                .M2_WEIGHT_CONTEXT(M2_WEIGHT_CONTEXT)\n")
            anchor = "                .i_weight_fill_valid(i_weight_fill_valid &&"
            text = once(text, anchor, "                .i_weight_fill_context(i_weight_fill_context),\n" + anchor)
        else:
            text = once(text, "    parameter bit P4_ALL_CASCADE = 1'b0\n", "    parameter bit P4_ALL_CASCADE = 1'b0,\n    parameter bit M2_WEIGHT_CONTEXT = 1'b0\n")
            text = once(text, "            unified_dual_role_weight_store_p4all_candidate_v1\n            u_weight_store (",
                        "            experimental_dual_context_weight_store_p4all_v1 #(\n"
                        "                .ENABLE_M2_CONTEXT(M2_WEIGHT_CONTEXT)\n            ) u_weight_store (")
            anchor = "                .ENABLE_M2_CONTEXT(M2_WEIGHT_CONTEXT)\n            ) u_weight_store (\n                .i_clk(i_clk), .i_rst_n(i_rst_n),"
            text = once(text, anchor, anchor + "\n                .i_fill_context(i_weight_fill_context),")
    elif name == "unified_srnn_m2_h40_forward_engine_v1.sv":
        anchor = next(line for line in text.splitlines() if "input  logic" in line and "i_cache_invalidate," in line)
        text = once(text, anchor, anchor + "\n    input  logic                                      i_wh1_cache_invalidate,")
        text = once(text, "    logic configured_q;", "    logic configured_q;\n    logic wh1_resident_q;")
        text = once(text, "            configured_q <= 1'b0;", "            configured_q <= 1'b0;\n            wh1_resident_q <= 1'b0;")
        text = once(text, "            if (cfg_fire) begin", "            if (cfg_fire) begin\n                wh1_resident_q <= 1'b0;")
        text = once(text, "            if (i_cache_invalidate) begin", "            if (i_wh1_cache_invalidate)\n                wh1_resident_q <= 1'b0;\n            if (i_cache_invalidate) begin")
        text = once(text, "                if (prep_state_q != PREP_IDLE)\n                    o_cache_valid <= 1'b0;",
                    "                if (prep_state_q != PREP_IDLE) begin\n                    o_cache_valid <= 1'b0;\n                    wh1_resident_q <= 1'b0;\n                end")
        text = once(text, "                                prep_state_q <= PREP_WH1_ISSUE;",
                    "                                prep_state_q <= wh1_resident_q ? PREP_WOUT_ISSUE : PREP_WH1_ISSUE;")
        text = once(text, "                            prep_state_q <= PREP_WOUT_ISSUE;",
                    "                            wh1_resident_q <= 1'b1;\n                            prep_state_q <= PREP_WOUT_ISSUE;")
    elif name == "unified_srnn_m2_timestep_train_infer_coordinator.sv":
        anchor = next(line for line in text.splitlines() if "output logic" in line and "o_h40_cache_invalidate," in line)
        text = once(text, anchor, anchor + "\n    output logic                                      o_h40_wh1_cache_invalidate,")
        anchor = "    assign o_h40_cache_invalidate = 1'b0;"
        text = once(text, anchor, anchor + "\n    assign o_h40_wh1_cache_invalidate = 1'b0;")
        anchor = "        assign o_h40_cache_invalidate = shared_cfg_fire ||"
        text = once(text, anchor, "        assign o_h40_wh1_cache_invalidate = shared_cfg_fire ||\n"
                    "            (i_load_valid && o_load_ready) ||\n"
                    "            (ENABLE_WH1_LEARNING && o_learning_commit_pulse && o_commit_code_dirty);\n" + anchor)
    elif name == "unified_tp_c17_learning_parameter_plane_v1.sv":
        anchor = "    output logic o_srnn_h40_cache_invalidate,"
        text = once(text, anchor, anchor + "\n    output logic o_srnn_h40_wh1_cache_invalidate,")
        anchor = "        .o_h40_cache_invalidate(o_srnn_h40_cache_invalidate),"
        text = once(text, anchor, anchor + "\n        .o_h40_wh1_cache_invalidate(o_srnn_h40_wh1_cache_invalidate),")
        anchor = "        assign o_srnn_h40_cache_invalidate = 1'b0;"
        text = once(text, anchor, anchor + "\n        assign o_srnn_h40_wh1_cache_invalidate = 1'b0;")
    elif name == "unified_h40_final_composition_top_v1.sv":
        anchor = "    logic link_plane_to_system_srnn_h40_cache_invalidate;"
        text = once(text, anchor, anchor + "\n    logic link_plane_to_system_srnn_h40_wh1_cache_invalidate;")
        anchor = "        .i_srnn_h40_cache_invalidate(link_plane_to_system_srnn_h40_cache_invalidate),"
        text = once(text, anchor, anchor + "\n        .i_srnn_h40_wh1_cache_invalidate(link_plane_to_system_srnn_h40_wh1_cache_invalidate),")
        anchor = "        .o_srnn_h40_cache_invalidate(link_plane_to_system_srnn_h40_cache_invalidate),"
        text = once(text, anchor, anchor + "\n        .o_srnn_h40_wh1_cache_invalidate(link_plane_to_system_srnn_h40_wh1_cache_invalidate),")
    else:
        raise ValueError(name)
    return text


FILES = (
    "rtl/top/unified_scnn_h40_array16_selected_layers_train_infer_system_v1.sv",
    "rtl/top/unified_h40_array16_three_topology_top_v1.sv",
    "rtl/tiles/unified_binary_p4_h40_array12_path_v1.sv",
    "rtl/tiles/unified_dual_role_h40_stationary_array12_candidate_v1.sv",
    "rtl/tiles/unified_dual_role_h40_stationary_region4_candidate_v1.sv",
    "rtl/tiles/unified_dual_role_h40_stationary_tile_candidate_v1.sv",
    "rtl/learning/unified_srnn_m2_h40_forward_engine_v1.sv",
    "rtl/learning/unified_srnn_m2_timestep_train_infer_coordinator.sv",
    "rtl/top/unified_tp_c17_learning_parameter_plane_v1.sv",
    "rtl/top/unified_h40_final_composition_top_v1.sv",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="candidate_r1")
    parser.add_argument("--specialize-m2-mode", action="store_true")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        parser.error("simple name required")
    output = EXPERIMENT / args.name
    if (output / "overlay.json").exists():
        raise FileExistsError(output)
    output.mkdir(exist_ok=True)
    diff = []
    for relative in FILES:
        original = (ROOT / relative).read_text(encoding="utf-8")
        changed = transform(relative, original)
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(changed, encoding="utf-8", newline="\n")
        diff.extend(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                                         fromfile="accepted/" + relative, tofile="candidate/" + relative))
    if args.specialize_m2_mode:
        relative = "verification/experiments/srnn_resident_weight_context/rtl/experimental_dual_context_weight_store_p4all_v1.sv"
        original = (ROOT / relative).read_text(encoding="utf-8")
        changed = once(original, ".o_read_ready(read_ready[ctx_idx]), .i_mode(i_mode),",
                       ".o_read_ready(read_ready[ctx_idx]),\n                .i_mode(ctx_idx == 1 ? MODE_M2_Q14 : i_mode),")
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(changed, encoding="utf-8", newline="\n")
        diff.extend(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
            fromfile="prototype/"+relative, tofile="specialized/"+relative))
    (output / "changes.diff").write_text("".join(diff), encoding="utf-8")
    (output / "overlay.json").write_text(json.dumps(dict(files=FILES, specialize_m2_mode=args.specialize_m2_mode, status="generated_not_measured",
        scope="SRNN bounded resident contexts; explicit M2 invalidation/config still clears WH1; Wout refilled on version change"), indent=2)+"\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
