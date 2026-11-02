from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_TB = (
    ROOT / "tb/tb_unified_scnn_h40_array16_selected_layers_train_infer_system_v1.sv"
)
OUTPUT_TB = ROOT / "tb/tb_unified_h40_final_composition_scnn_v1.sv"
WRAPPER = ROOT / "rtl/top/unified_h40_final_composition_top_v1.sv"
SYSTEM_MODULE = "unified_scnn_h40_array16_selected_layers_train_infer_system_v1"


def balanced_end(text: str, start: int) -> int:
    opening = text[start]
    closing = {"(": ")", "[": "]", "{": "}"}[opening]
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unterminated delimiter at {start}")


def split_top_level(blob: str) -> list[str]:
    items: list[str] = []
    start = 0
    round_depth = square_depth = brace_depth = 0
    for index, char in enumerate(blob):
        if char == "(":
            round_depth += 1
        elif char == ")":
            round_depth -= 1
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        elif char == "," and not (round_depth or square_depth or brace_depth):
            items.append(blob[start:index].strip())
            start = index + 1
    tail = blob[start:].strip()
    if tail:
        items.append(tail)
    return items


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def declaration_name(declaration: str) -> str:
    match = re.search(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=.*)?$", declaration.strip(), re.S
    )
    if not match:
        raise ValueError(f"cannot find declaration name: {declaration!r}")
    return match.group(1)


def module_interface(path: Path, module: str) -> tuple[dict[str, str], dict[str, str]]:
    text = strip_comments(path.read_text(encoding="utf-8"))
    match = re.search(rf"\bmodule\s+{re.escape(module)}\b", text)
    if not match:
        raise ValueError(f"module {module} not found in {path}")
    cursor = match.end()
    while text[cursor].isspace():
        cursor += 1
    parameters: dict[str, str] = {}
    if text[cursor] == "#":
        start = text.index("(", cursor)
        end = balanced_end(text, start)
        for declaration in split_top_level(text[start + 1:end]):
            normalized = " ".join(declaration.split())
            parameters[declaration_name(normalized.split("=", 1)[0])] = normalized
        cursor = end + 1
    start = text.index("(", cursor)
    end = balanced_end(text, start)
    ports: dict[str, str] = {}
    for declaration in split_top_level(text[start + 1:end]):
        normalized = " ".join(declaration.split())
        direction = re.match(r"(input|output|inout)\b", normalized)
        if not direction:
            raise ValueError(f"non-ANSI port declaration: {declaration!r}")
        ports[declaration_name(normalized)] = direction.group(1)
    return parameters, ports


def parse_named_connections(blob: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in split_top_level(blob):
        match = re.match(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", item, flags=re.S)
        if not match:
            raise ValueError(f"bad named connection: {item!r}")
        start = item.index("(", match.end() - 1)
        end = balanced_end(item, start)
        result[match.group(1)] = item[start + 1:end].strip()
    return result


def instance_span(
    text: str, module: str, instance: str
) -> tuple[int, int, dict[str, str], dict[str, str]]:
    start_match = re.search(rf"\b{re.escape(module)}\b\s*#\s*\(", text)
    if not start_match:
        raise ValueError(f"missing instance module {module}")
    parameter_start = text.index("(", start_match.start())
    parameter_end = balanced_end(text, parameter_start)
    instance_match = re.search(
        rf"\)\s*{re.escape(instance)}\s*\(", text[parameter_end:]
    )
    if not instance_match:
        raise ValueError(f"missing instance {instance}")
    absolute_match_start = parameter_end + instance_match.start()
    port_start = text.index("(", absolute_match_start + 1)
    port_end = balanced_end(text, port_start)
    end = port_end + 1
    while end < len(text) and text[end].isspace():
        end += 1
    if text[end] != ";":
        raise ValueError(f"missing semicolon after {instance}")
    return (
        start_match.start(),
        end + 1,
        parse_named_connections(text[parameter_start + 1:parameter_end]),
        parse_named_connections(text[port_start + 1:port_end]),
    )


wrapper_parameters, wrapper_ports = module_interface(
    WRAPPER, "unified_h40_final_composition_top_v1"
)
text = SOURCE_TB.read_text(encoding="utf-8")
start, end, system_parameters, system_ports = instance_span(
    text, SYSTEM_MODULE, "dut"
)

parameters = [
    (name, expression)
    for name, expression in system_parameters.items()
    if name in wrapper_parameters
]
parameters = [
    (name, "4194304" if name == "PARAM_DEPTH" else
           "24" if name == "PARAM_ADDR_WIDTH" else expression)
    for name, expression in parameters
]
# This retained single-descriptor regression intentionally exercises legacy
# mode. Full-frame-service coverage has a separate current-source testbench.
parameters.append(("ENABLE_SCNN_FRAME_SERVICE", "1'b0"))


def store_expression(wrapper_name: str) -> str | None:
    if wrapper_name.startswith("o_store_"):
        return system_ports.get("o_scnn_store_" + wrapper_name[len("o_store_"):])
    if wrapper_name.startswith("i_store_"):
        return system_ports.get("i_scnn_store_" + wrapper_name[len("i_store_"):])
    return None


connections: list[tuple[str, str]] = []
for name, direction in wrapper_ports.items():
    expression = system_ports.get(name)
    if expression is None:
        expression = store_expression(name)
    if name == "i_parameter_owner":
        expression = "parameter_owner"
    elif name == "i_topology_learning_enable":
        expression = "3'b100"
    elif name == "i_store_quiescent":
        expression = "shared_backend_quiescent"
    elif name == "i_store_protocol_error":
        expression = "shared_backend_error"
    if expression is None:
        expression = "'0" if direction == "input" else ""
    connections.append((name, expression))

replacement = [
    "    // Generated exact-composition adaptation of the validated selected-system",
    "    // sequence. The SCNN path uses the production shared output learner and store.",
    "    unified_h40_final_composition_top_v1 #(",
]
for index, (name, expression) in enumerate(parameters):
    replacement.append(
        f"        .{name}({expression})" + ("," if index + 1 < len(parameters) else "")
    )
replacement.append("    ) dut (")
for index, (name, expression) in enumerate(connections):
    replacement.append(
        f"        .{name}({expression})" + ("," if index + 1 < len(connections) else "")
    )
replacement.extend([
    "    );",
    "",
    "    // Signals consumed by the inherited dense-path scoreboards are internal",
    "    // production links in the composition wrapper.",
    "    assign fc_student_trace_valid = dut.link_fc_student_trace_valid;",
    "    assign fc_student_trace_sample = dut.link_fc_student_trace_sample;",
    "    assign fc_student_trace_hidden = dut.link_fc_student_trace_hidden;",
    "    assign fc_student_trace_q016 = dut.link_fc_student_trace_q016;",
    "    assign fc_logit_row_valid = dut.link_fc_logit_row_valid;",
    "    assign fc_logit_row_ready = dut.link_fc_logit_row_ready;",
    "    assign fc_logit_sample = dut.link_fc_logit_sample;",
    "    assign fc_logit_label = dut.link_fc_logit_label;",
    "    assign fc_logit_valid = dut.link_fc_logit_valid;",
    "    assign fc_logit_ready = dut.link_fc_logit_ready;",
    "    assign fc_logit_q6 = dut.link_fc_logit_q6;",
    "    assign srnn_backbone_valid = dut.link_srnn_plane_backbone_valid;",
    "    assign srnn_backbone_path = dut.link_srnn_plane_backbone_path;",
    "    assign srnn_backbone_sample_id = dut.link_srnn_plane_backbone_sample_id;",
    "    assign srnn_backbone_id = dut.link_srnn_plane_backbone_id;",
    "    assign srnn_backbone_q14 = dut.link_srnn_plane_backbone_q14;",
    "    assign srnn_backbone_last = dut.link_srnn_plane_backbone_last;",
    "    assign parameter_load_ready = shared_load_ready[0];",
])

# Preserve all existing hierarchical diagnostics while inserting one wrapper level.
prefix = text[:start].replace("dut.", "dut.u_system.")
suffix = text[end:].replace("dut.", "dut.u_system.")
text = prefix + "\n".join(replacement) + suffix
text = text.replace(
    "dut.u_system.u_train_infer.",
    "dut.u_system.g_scnn_client.u_train_infer.",
)
text = text.replace(
    "module tb_unified_scnn_h40_array16_selected_layers_train_infer_system_v1 #(",
    "module tb_unified_h40_final_composition_scnn_v1 #(",
)
text = text.replace(
    "logic signed [15:0] parameter_load_code = '0;",
    "logic signed [15:0] parameter_load_code = '0;\n"
    "    logic [2:0] parameter_owner = 3'b001;",
)
text = text.replace("logic fc_logit_row_ready = 1'b1;", "logic fc_logit_row_ready;")
text = text.replace("logic fc_logit_ready = 1'b1;", "logic fc_logit_ready;")
text = text.replace(
    ".i_parameter_owner(3'b100),",
    ".i_parameter_owner(parameter_owner),",
)
text = text.replace(
    ".i_load_valid({scnn_store_load_valid, 2'b00}),",
    ".i_load_valid({scnn_store_load_valid, 1'b0, parameter_load_valid}),",
)
text = text.replace(
    ".i_load_addr({scnn_store_load_addr, 16'b0}),",
    ".i_load_addr({scnn_store_load_addr, 8'b0, parameter_load_addr}),",
)
text = text.replace(
    ".i_load_code({scnn_store_load_code, 32'b0}),",
    ".i_load_code({scnn_store_load_code, 16'b0, parameter_load_code}),",
)
text = text.replace(
    "        progress_stage = 3;\n        initialize_system();",
    "        progress_stage = 3;\n"
    "        while (!shared_backend_quiescent) tick();\n"
    "        parameter_owner = 3'b100;\n"
    "        tick();\n"
    "        initialize_system();",
)
text = text.replace(
    "        if (dut.u_system.g_scnn_client.u_train_infer.runtime_result_valid &&\n"
    "                dut.u_system.g_scnn_client.u_train_infer.runtime_result_ready) begin\n"
    "            if (dut.u_system.g_scnn_client.u_train_infer.runtime_result_tensor == 8'h08)\n"
    "                conv_write_count <= conv_write_count + 1;\n"
    "            if (dut.u_system.g_scnn_client.u_train_infer.runtime_result_tensor == 8'h12)\n"
    "                classifier_write_count <= classifier_write_count + 1;\n"
    "        end",
    "        if (scnn_store_learning_write_valid && shared_learning_write_ready[2]) begin\n"
    "            if (scnn_store_learning_write_addr >= CLASSIFIER_BASE &&\n"
    "                    scnn_store_learning_write_addr <\n"
    "                        CLASSIFIER_BASE + FEATURES * CLASSES)\n"
    "                classifier_write_count <= classifier_write_count + 1;\n"
    "            else if (scnn_store_learning_write_addr < CLASSIFIER_BASE)\n"
    "                conv_write_count <= conv_write_count + 1;\n"
    "            else\n"
    "                $fatal(1, \"shared output learner wrote outside the selected SCNN banks: addr=%0d\",\n"
    "                    scnn_store_learning_write_addr);\n"
    "            if (!scnn_store_learning_write_code_enable &&\n"
    "                    !scnn_store_learning_write_residual_enable)\n"
    "                $fatal(1, \"shared output learner issued a no-op write\");\n"
    "        end",
)
text = re.sub(
    r"        progress_stage = 4;.*?        progress_stage = 7;",
    "        progress_stage = 7;",
    text,
    count=1,
    flags=re.S,
)
text = re.sub(
    r"        progress_stage = 9;.*?"
    r"        \$display\(\"PASS: one 12xH40/P4 array executes.*?"
    r"SCNN_WEIGHT_FRACTIONAL_BITS\);",
    "        if (!quiescent || !shared_backend_quiescent || shared_backend_error ||\n"
    "                sticky_error || array_tile_error != 0 || array_active_owner != 0 ||\n"
    "                array_tile_idle != 16'hffff)\n"
    "            $fatal(1, \"final SCNN quiescence mismatch owner=%0d quiet=%0b/%0b error=%0b/%0b tile=%h idle=%h\",\n"
    "                array_active_owner, quiescent, shared_backend_quiescent,\n"
    "                sticky_error, shared_backend_error, array_tile_error,\n"
    "                array_tile_idle);\n\n"
    "        $display(\"FINAL_COMPOSITION_SCNN_CLASSIFIER_V1_PASS exact_top=1 sequence=SCNN_FORWARD_LEARN_COMMIT_NEXT_FORWARD classifier_only=%0b target_graphs=%0d weight_frac=%0d shared_output_learner=1 shared_store_client=2 synthetic_pooled_features=1\", CLASSIFIER_ONLY, target_graph_count, SCNN_WEIGHT_FRACTIONAL_BITS);",
    text,
    count=1,
    flags=re.S,
)
text = text.replace(
    "adapt_version != 8 || parameter_version != 6 ||",
    "adapt_version != 8 || parameter_version != 6 || !versions_aligned ||",
)
text = text.replace(
    "model_version != 6 || parameter_version != 6 ||",
    "model_version != 6 || parameter_version != 6 || !versions_aligned ||",
)

OUTPUT_TB.write_text(text, encoding="utf-8")
print(f"generated {OUTPUT_TB} with {len(connections)} connected wrapper ports")
