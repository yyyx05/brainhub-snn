#!/usr/bin/env python3
"""Generate a public-port final-composition B1T2 + fresh T0 numeric test.

Reuse the established full-shape raw graph scoreboard, not its old cache-fill
driver or graph-online DUT. The appended models live outside the DUT.
"""
from pathlib import Path
import re

from run_fc_final_counts import ROOT, ansi_header

TOP = "tb_scnn_real_final_graph"
SOURCE = ROOT / "tb/tb_unified_scnn_q015_real_conv1_to_classifier_t2_same_array_v1.sv"
FINAL = ROOT / "rtl/top/unified_h40_final_composition_top_v1.sv"
BODY = ROOT / "tb/include/scnn_real_final_graph_body.svh"
CHECKS = ROOT / "tests/scnn_real_final_graph_checks.svh"
CONV_WORDS = (1152, 73728, 294912, 589824, 1179648, 2359296, 2359296, 2359296)
CONV_BASES = tuple(sum(CONV_WORDS[:i]) for i in range(8))
CLASSIFIER_BASE = sum(CONV_WORDS)
PARAM_DEPTH = 16777216


def replace_once(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError(f"source edit expected {count} occurrences: {old[:100]!r}")
    return text.replace(old, new)


def replace_task(text, name, replacement=""):
    text, count = re.subn(r"    task automatic " + name + r"\b.*?    endtask\n", replacement,
                         text, flags=re.S)
    if count != 1:
        raise ValueError(f"expected one task {name}, found {count}")
    return text


def connections(text):
    """Read balanced named instance connections (casts contain parentheses)."""
    result = {}
    offset = 0
    while match := re.search(r"\.(\w+)\s*\(", text[offset:]):
        start = offset + match.end()
        end, depth = start, 1
        while depth:
            if text[end] == "(":
                depth += 1
            elif text[end] == ")":
                depth -= 1
            end += 1
        result[match.group(1)] = text[start:end-1].strip()
        offset = end
    return result


def final_instance(original, existing_parameters):
    module, parameters, ports = ansi_header(FINAL)
    old = connections(original[original.index(") dut (") + len(") dut ("):])
    overrides = {"PARAM_DEPTH": str(PARAM_DEPTH), "MAX_BATCH": "4", "MAX_TIMESTEPS": "2",
                 "ENABLE_SCNN_FRAME_SERVICE": "1'b1", "ENABLE_SCNN_CONV8_LEARNING": "1'b0"}
    lines = [f"    localparam {kind} {name} = {overrides.get(name, value)};"
             for kind, name, value in parameters if name not in existing_parameters]
    lines += [f"    {kind} ft_{name};" for _, kind, name in ports]
    inputs = {name: old[name] for direction, _, name in ports
              if direction == "input" and name in old}
    inputs.update({
        "i_initialize_valid": "fg_initialize", "i_classifier_pair": "1'b1",
        "i_context_model_version": "CLASSIFIER_VERSION", "i_context_adapt_version": "16'd0",
        "i_system_parameter_owner": "1'b1", "i_system_learning_enable": "fg_learning",
        "i_system_profile_aligned": "1'b1", "i_system_profile_id": "8'h20",
        "i_profile_id": "8'h20", "i_conv_tensor_id": "8'h08", "i_classifier_tensor_id": "8'h12",
        "i_parameter_owner": "3'b100", "i_topology_learning_enable": "3'b100",
        "i_request_valid": "fg_request", "i_learning_enable": "fg_learning",
        "i_batch_count": "BATCH_COUNT_WIDTH'(1)", "i_timestep": "'0",
        "i_timestep_count": "TIMESTEP_COUNT_WIDTH'(timestep_phase == 2 ? 1 : 2)",
        "i_sample_version": "fg_window_version", "i_labels_flat": "{ {(MAX_BATCH-1)*CLASS_WIDTH{1'b0}}, fg_label[0][CLASS_WIDTH-1:0] }",
        "i_total_input_channels": "CHANNEL_COUNT_WIDTH'(CONV8_CHANNELS)",
        "i_total_output_channels": "CHANNEL_COUNT_WIDTH'(CONV8_CHANNELS)",
        "i_conv_parameter_base": "fg_conv_base(layer_phase)", "i_conv_weight_q16_base": "'0",
        "i_classifier_parameter_base": "PARAM_ADDR_WIDTH'(FG_CLASSIFIER_BASE)",
        "i_classifier_feature_count": "FEATURE_COUNT_WIDTH'(CLASSIFIER_FEATURES)",
        "i_classifier_class_count": "CLASS_COUNT_WIDTH'(CLASSES)",
        "i_classifier_trace_leak_valid": "1'b1", "i_classifier_trace_leak_q8_8": "16'sd250",
        "i_classifier_vote_increment": "16'd0", "i_classifier_vote_threshold": "16'h8000",
        "i_code_min": "-16'sd32768", "i_code_max": "16'sd32767",
        "i_target_feature_count": "FEATURE_COUNT_WIDTH'(CLASSIFIER_FEATURES)",
        "i_target_trace_leak_q8_8": "16'sd250", "i_target_trace_state_reset": "1'b1",
        "i_frame_index": "FRAME_WIDTH'(timestep_phase == 2 ? 0 : timestep_phase)",
        "i_graph_descriptor_valid": "start_valid", "i_graph_descriptor_last": "layer_phase == 8",
        "i_state_service_ready": "context_load_ready || context_release_ready",
        "i_store_load_ready": "1'b0", "i_store_protocol_error": "1'b0",
        "i_store_forward_read_ready": "!fg_forward_valid",
        "i_store_forward_response_valid": "fg_forward_valid",
        "i_store_forward_response_tag": "fg_forward_tag", "i_store_forward_response_code": "fg_forward_code",
        "i_store_forward_response_error": "1'b0",
        "i_store_learning_read_ready": "!fg_learning_valid",
        "i_store_learning_response_valid": "fg_learning_valid",
        "i_store_learning_response_tag": "fg_learning_tag", "i_store_learning_response_code": "fg_learning_code",
        "i_store_learning_response_residual": "fg_learning_residual", "i_store_learning_response_error": "1'b0",
        "i_store_learning_write_ready": "!fg_learning_valid",
        "i_store_quiescent": "!fg_forward_valid && !fg_learning_valid && !ft_o_store_forward_read_valid && !ft_o_store_learning_read_valid && !ft_o_store_learning_write_valid && !ft_o_store_load_valid",
    })
    # Unused topology command inputs remain inactive. All DUT inputs are public.
    for direction, _, name in ports:
        if direction == "input":
            lines.append(f"    assign ft_{name} = {inputs.get(name, chr(39)+'0')};")
    lines += [f"    {module} #(",
              ",\n".join(f"        .{name}({name})" for _, name, _ in parameters),
              "    ) dut (", ",\n".join(f"        .{name}(ft_{name})" for _, _, name in ports), "    );"]
    # Keep the old raw-graph scoreboard's signal names without bypassing the DUT.
    aliases = {old[name]: f"ft_{name}" for direction, _, name in ports
               if direction == "output" and name in old and re.fullmatch(r"\w+", old[name])}
    aliases.update({
        "start_ready": "ft_o_graph_descriptor_ready", "done": "ft_o_graph_descriptor_done",
        "busy": "dut.u_system.u_array.scnn_client_busy", "idle": "dut.u_system.u_array.scnn_client_idle",
        "active_owner": "ft_o_array_active_owner", "tile_idle": "ft_o_array_tile_idle",
        "tile_error": "ft_o_array_tile_sticky_error", "classifier_pair": "ft_o_active_classifier_pair",
        "classifier_version": "ft_o_parameter_version", "cache_prime_done": "ft_o_initialize_done",
        "context_load_valid": "ft_o_state_service_valid && !ft_o_state_service_release",
        "context_release_valid": "ft_o_state_service_valid && ft_o_state_service_release",
        "context_load_weights": "1'b0", # Cache handshake has already completed before external service.
        "logit_wave_fire": "dut.u_system.logit_wave_valid && dut.u_system.logit_wave_ready",
        "logit_wave_mask": "dut.u_system.logit_wave_mask", "logit_wave_base": "dut.u_system.logit_wave_base",
        "logit_wave_count": "dut.u_system.logit_wave_count", "logit_wave_flat": "dut.u_system.logit_wave_flat",
    })
    for dest in ("context", "release"):
        for field in ("output_channel_base", "output_channel_count", "spatial_base", "spatial_count", "active_pair", "work_base_addr"):
            aliases[f"{dest}_{field}"] = f"ft_o_state_service_{field}"
    for name in ("valid", "ready", "tile_flat", "bank", "addr_flat", "data_flat"):
        aliases[f"weight_fill_{name}"] = f"dut.u_system.scnn_weight_fill_{name}"
    lines += [f"    assign {name} = {value};" for name, value in aliases.items()]
    return "\n".join(lines) + "\n"


def testbench_source():
    text = SOURCE.read_text(encoding="utf-8")
    text = replace_once(text, SOURCE.stem, TOP)
    text = text.replace("[0:TEST_TIMESTEPS*", "[0:3*")
    begin = text.index("    unified_scnn_h40_array16_graph_online_learning_system_v1 #(")
    end = text.index("    task automatic load_all_classifier_parameters;", begin)
    parameter_names = set(re.findall(r"localparam\s+.*?\b(\w+)\s*=", text[:begin]))
    text = text[:begin] + final_instance(text[begin:end], parameter_names) + text[end:]
    # Remove the obsolete driver, not the numerical scoreboard.
    for task in ("load_all_classifier_parameters", "prime_classifier_cache", "configure_learning_client", "fill_weight", "clear_tile_state_at_work_base"):
        text = replace_task(text, task)
    for layer in range(1, 9):
        text = replace_task(text, f"load_conv{layer}_context")
        text = replace_once(text, f"                if (context_load_weights)\n                    load_conv{layer}_context();\n", "")
    text = replace_task(text, "restore_tile_context")
    text = replace_task(text, "service_release")
    text = replace_task(text, "run_current_layer")
    text = replace_once(text, "                classifier_context_count = classifier_context_count + 1;",
        "                if (!ft_o_state_service_is_classifier) $fatal(1, \"classifier service kind\");\n"
        "                fg_classifier_restore();\n"
        "                classifier_context_count = classifier_context_count + 1;")
    text = replace_once(text, "            if (context_work_base_addr != 0)",
        "            if (ft_o_state_service_is_classifier) $fatal(1, \"Conv service kind\");\n"
        "            if (context_work_base_addr != 0)")
    # Weight fill aliases are monitors only, never initialized/driven by the TB.
    text = re.sub(r"(logic \[[^\n]+\] weight_fill_\w+) = '0;", r"\1;", text)
    # Replace only diagnostics referencing the old wrapper. Core numeric assertions stay.
    start = text.index('                $display("ERROR_WRAPPER')
    finish = text.index('                $display("ERROR_CORE', start)
    text = text[:start] + '                $display("FINAL_ERROR backend=%b owner=%b hold=%b", ft_o_backend_error, ft_o_owner_encoding_error, ft_o_inference_hold);\n' + text[finish:]
    start = text.index('                $display("ERROR_INPUT')
    finish = text.index('                $fatal(1, "three-layer path error")', start)
    text = text[:start] + text[finish:]
    text = text.replace("dut.u_core", "dut.u_system.u_array")
    text = replace_once(text, "        load_all_classifier_parameters();", "        fg_load_expected();\n        fg_initialize_top();")
    text = replace_once(text, "        prime_classifier_cache();", "")
    text = replace_once(text, "        for (int tile = 0; tile < 12; tile++)\n            clear_tile_state_at_work_base(tile, CLASSIFIER_WORK_BASE);", "")
    text = replace_once(text, "        configure_learning_client();", "        fg_start_window();")
    text = replace_once(text, "timestep_phase < TEST_TIMESTEPS;", "timestep_phase < 3;")
    text = replace_once(text, "        load_reference_frame(timestep_phase);",
        "        if (timestep_phase == 2) fg_start_window();\n"
        "        load_reference_frame(timestep_phase == 2 ? 0 : timestep_phase);\n"
        "        for (int c = 0; c < CLASSES; c++) begin\n"
        "            classifier_expected_current[c] = fg_currents[timestep_phase*CLASSES+c];\n"
        "            classifier_expected_centered[c] = fg_logits[timestep_phase*CLASSES+c];\n        end")
    text = replace_once(text, "                classifier_version != CLASSIFIER_VERSION ||",
        "                (timestep_phase != 1 && classifier_version != fg_window_version) ||")
    marker = '        $display("SUPPLEMENT_FULLGRAPH_TIMESTEP'
    text = replace_once(text, marker, "        fg_check_frame();\n" + marker)
    text = text.replace("SCNN_Q015_REAL_CONV1_TO_CLASSIFIER_TIMESTEP_PASS", "SCNN_REAL_FINAL_FRAME_PASS")
    text = text.replace("one_shared_array=1 learning_off=1 requests=", "one_shared_array=1 requests=")
    text = re.sub(r'        \$display\("SCNN_Q015_REAL_CONV1_TO_CLASSIFIER_T2_SAME_ARRAY_PASS[^\n]+\n',
                  "        fg_finish();\n", text)
    text = replace_once(text, "repeat (32000000)", "repeat (160000000)")
    constants = [f"    localparam int FG_CLASSIFIER_BASE = {CLASSIFIER_BASE};"]
    constants += [f"    localparam int FG_CONV{i+1}_BASE = {base};" for i, base in enumerate(CONV_BASES)]
    text = replace_once(text, "endmodule", "\n".join(constants) + "\n" + BODY.read_text(encoding="ascii") +
                        "\n" + CHECKS.read_text(encoding="ascii") + "\nendmodule")
    return text


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x", encoding="ascii") as stream:
        stream.write(testbench_source())
