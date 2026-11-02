from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PATH = ROOT / "rtl/top/unified_scnn_h40_array16_selected_layers_train_infer_system_v1.sv"
PLANE_PATH = ROOT / "rtl/top/unified_tp_c17_learning_parameter_plane_v1.sv"
OUTPUT_PATH = ROOT / "rtl/top/unified_h40_final_composition_top_v1.sv"

SYSTEM_MODULE = "unified_scnn_h40_array16_selected_layers_train_infer_system_v1"
PLANE_MODULE = "unified_tp_c17_learning_parameter_plane_v1"


def balanced_end(text: str, start: int) -> int:
    pairs = {"(": ")", "[": "]", "{": "}"}
    opening = text[start]
    closing = pairs[opening]
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == opening:
            depth += 1
        elif char == closing:
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


def module_blobs(path: Path, module: str) -> tuple[str, str]:
    text = strip_comments(path.read_text(encoding="utf-8"))
    match = re.search(rf"\bmodule\s+{re.escape(module)}\b", text)
    if not match:
        raise ValueError(f"module {module} not found in {path}")
    cursor = match.end()
    while text[cursor].isspace():
        cursor += 1
    parameter_blob = ""
    if text[cursor] == "#":
        parameter_start = text.index("(", cursor)
        parameter_end = balanced_end(text, parameter_start)
        parameter_blob = text[parameter_start + 1:parameter_end]
        cursor = parameter_end + 1
    port_start = text.index("(", cursor)
    port_end = balanced_end(text, port_start)
    return parameter_blob, text[port_start + 1:port_end]


def declaration_name(declaration: str) -> str:
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=.*)?$", declaration.strip(), re.S)
    if not match:
        raise ValueError(f"cannot find declaration name: {declaration!r}")
    return match.group(1)


def parse_parameters(blob: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for declaration in split_top_level(blob):
        normalized = " ".join(declaration.split())
        if not normalized.startswith("parameter "):
            raise ValueError(f"non-parameter declaration: {declaration!r}")
        result.append((declaration_name(normalized.split("=", 1)[0]), normalized))
    return result


def parse_ports(blob: str) -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []
    for declaration in split_top_level(blob):
        normalized = " ".join(declaration.split())
        match = re.match(r"(input|output|inout)\b", normalized)
        if not match:
            raise ValueError(f"non-ANSI port declaration: {declaration!r}")
        result.append((declaration_name(normalized), match.group(1), normalized))
    return result


def rename_declaration(declaration: str, old: str, new: str) -> str:
    return re.sub(rf"\b{re.escape(old)}\s*$", new, declaration)


def wire_declaration(declaration: str, old_name: str, new_name: str) -> str:
    prefix = re.sub(rf"\b{re.escape(old_name)}\s*$", "", declaration).strip()
    prefix = re.sub(r"^(input|output|inout)\s+", "", prefix)
    prefix = re.sub(r"\bwire\b\s*", "", prefix)
    if not re.search(r"\blogic\b", prefix):
        prefix = "logic " + prefix
    return f"    {prefix.strip()} {new_name};"


system_parameter_blob, system_port_blob = module_blobs(SYSTEM_PATH, SYSTEM_MODULE)
plane_parameter_blob, plane_port_blob = module_blobs(PLANE_PATH, PLANE_MODULE)
system_parameters = parse_parameters(system_parameter_blob)
plane_parameters = parse_parameters(plane_parameter_blob)
system_ports = parse_ports(system_port_blob)
plane_ports = parse_ports(plane_port_blob)
system_by_name = {name: (direction, declaration) for name, direction, declaration in system_ports}
plane_by_name = {name: (direction, declaration) for name, direction, declaration in plane_ports}

system_connections: dict[str, str] = {}
plane_connections: dict[str, str] = {}
link_sources: dict[str, tuple[str, str]] = {}


def connect(system_name: str, plane_name: str, link_name: str) -> None:
    if system_name not in system_by_name:
        raise ValueError(f"missing system port {system_name}")
    if plane_name not in plane_by_name:
        raise ValueError(f"missing plane port {plane_name}")
    system_connections[system_name] = link_name
    plane_connections[plane_name] = link_name
    source_name = system_name if system_by_name[system_name][0] == "output" else plane_name
    source_decl = system_by_name[system_name][1] if source_name == system_name else plane_by_name[plane_name][1]
    link_sources[link_name] = (source_name, source_decl)


for system_name in system_by_name:
    if system_name.startswith("o_scnn_store_"):
        candidate = "i_" + system_name[2:]
        if candidate in plane_by_name:
            connect(system_name, candidate,
                    "link_system_to_plane_" + system_name[2:])
    elif system_name.startswith("i_scnn_store_"):
        candidate = "o_" + system_name[2:]
        if candidate in plane_by_name:
            connect(system_name, candidate,
                    "link_plane_to_system_" + system_name[2:])
    elif system_name.startswith("i_srnn_h40_"):
        candidate = "o_" + system_name[2:]
        if candidate in plane_by_name:
            connect(system_name, candidate,
                    "link_plane_to_system_" + system_name[2:])
    elif system_name.startswith("o_srnn_h40_"):
        candidate = "i_" + system_name[2:]
        if candidate in plane_by_name:
            connect(system_name, candidate,
                    "link_system_to_plane_" + system_name[2:])
    elif system_name.startswith("o_external_"):
        candidate = "i_scnn_external_" + system_name[len("o_external_"):]
        if candidate in plane_by_name:
            connect(system_name, candidate,
                    "link_system_to_plane_scnn_external_" +
                    system_name[len("o_external_"):])
    elif system_name.startswith("i_external_"):
        candidate = "o_scnn_external_" + system_name[len("i_external_"):]
        if candidate in plane_by_name:
            connect(system_name, candidate,
                    "link_plane_to_system_scnn_external_" +
                    system_name[len("i_external_"):])

for system_name, plane_name in {
    "o_fc_cache_forward_read_valid": "i_fc_forward_read_valid",
    "i_fc_cache_forward_read_ready": "o_fc_forward_read_ready",
    "o_fc_cache_forward_read_addr": "i_fc_forward_read_addr",
    "o_fc_cache_forward_read_tag": "i_fc_forward_read_tag",
    "i_fc_cache_forward_response_valid": "o_fc_forward_response_valid",
    "o_fc_cache_forward_response_ready": "i_fc_forward_response_ready",
    "i_fc_cache_forward_response_tag": "o_fc_forward_response_tag",
    "i_fc_cache_forward_response_code": "o_fc_forward_response_code",
    "i_fc_cache_forward_response_error": "o_fc_forward_response_error",
    "o_fc_student_trace_valid": "i_fc_student_trace_valid",
    "i_fc_student_trace_ready": "o_fc_student_trace_ready",
    "o_fc_student_trace_sample": "i_fc_student_trace_sample",
    "o_fc_student_trace_hidden": "i_fc_student_trace_hidden",
    "o_fc_student_trace_q016": "i_fc_student_trace_q016",
    "o_fc_logit_row_valid": "i_fc_logit_row_valid",
    "i_fc_logit_row_ready": "o_fc_logit_row_ready",
    "o_fc_logit_sample": "i_fc_logit_sample",
    "o_fc_logit_label": "i_fc_logit_label",
    "o_fc_logit_valid": "i_fc_logit_valid",
    "i_fc_logit_ready": "o_fc_logit_ready",
    "o_fc_logit_q6": "i_fc_logit_q6",
}.items():
    connect(system_name, plane_name, "link_" + plane_name[2:])

# Learning-disabled forward execution must never wait on a dormant learning
# client. The physical observer remains connected, but its ready path is
# bypassed unless the FC learning owner is enabled.
system_connections["i_fc_student_trace_ready"] = (
    "(HAS_FC && ENABLE_ONLINE_LEARNING && i_topology_learning_enable[0]) ? "
    "link_fc_student_trace_ready : 1'b1"
)
system_connections["i_fc_logit_row_ready"] = (
    "(HAS_FC && ENABLE_ONLINE_LEARNING && i_topology_learning_enable[0]) ? "
    "link_fc_logit_row_ready : 1'b1"
)
system_connections["i_fc_logit_ready"] = (
    "(HAS_FC && ENABLE_ONLINE_LEARNING && i_topology_learning_enable[0]) ? "
    "link_fc_logit_ready : 1'b1"
)

system_connections["i_scnn_store_protocol_error"] = "link_router_protocol_error"
plane_connections["o_router_protocol_error"] = "link_router_protocol_error"
link_sources["link_router_protocol_error"] = (
    "o_router_protocol_error", plane_by_name["o_router_protocol_error"][1]
)

common_connections = {
    "i_clk": "i_clk",
    "i_rst_n": "i_rst_n",
    "i_clear_error": "i_clear_error",
}
system_connections.update(common_connections)
plane_connections.update(common_connections)

system_connections.update({
    "i_parameter_load_valid": "1'b0",
    "i_parameter_load_addr": "'0",
    "i_parameter_load_code": "'0",
    "i_parameter_load_residual": "'0",
    "o_parameter_load_ready": "",
    "i_fc_surrogate_ready": "1'b1",
})

actual_links = {
    "o_srnn_backbone_valid": "link_srnn_actual_valid",
    "i_srnn_backbone_ready": "link_srnn_actual_ready",
    "o_srnn_backbone_path": "link_srnn_actual_path",
    "o_srnn_backbone_sample_id": "link_srnn_actual_sample_id",
    "o_srnn_backbone_id": "link_srnn_actual_backbone_id",
    "o_srnn_backbone_q14": "link_srnn_actual_q14",
    "o_srnn_backbone_last": "link_srnn_actual_last",
}
for name, link in actual_links.items():
    system_connections[name] = link
    link_sources[link] = (name, system_by_name[name][1])

plane_backbone_links = {
    "i_srnn_backbone_valid": "link_srnn_plane_backbone_valid",
    "o_srnn_backbone_ready": "link_srnn_plane_backbone_ready",
    "i_srnn_backbone_path": "link_srnn_plane_backbone_path",
    "i_srnn_backbone_sample_id": "link_srnn_plane_backbone_sample_id",
    "i_srnn_backbone_id": "link_srnn_plane_backbone_id",
    "i_srnn_backbone_q14": "link_srnn_plane_backbone_q14",
    "i_srnn_backbone_last": "link_srnn_plane_backbone_last",
    "i_srnn_backbone_done": "link_srnn_plane_backbone_done",
    "i_srnn_backbone_error": "link_srnn_plane_backbone_error",
    "i_srnn_timestep_start": "link_srnn_plane_timestep_start",
    "o_srnn_timestep_start_ready": "link_srnn_plane_timestep_start_ready",
}
for name, link in plane_backbone_links.items():
    plane_connections[name] = link
    link_sources[link] = (name, plane_by_name[name][1])
plane_connections["i_srnn_learning_enable"] = (
    "HAS_SRNN && ENABLE_ONLINE_LEARNING && i_srnn_phase_learning_enable"
)
plane_connections["i_fc_observation_done"] = "link_fc_observation_done"

parameter_names: set[str] = {"TOPOLOGY_MASK"}
wrapper_parameters: list[str] = ["parameter int TOPOLOGY_MASK = 7"]
wrapper_parameter_overrides = {
    "ENABLE_SCNN_FRAME_SERVICE": (
        "parameter bit ENABLE_SCNN_FRAME_SERVICE = 1'b1"
    ),
    # The C17 standalone default models its legacy 32K internal parameter
    # store.  The production composition disables that store and exposes one
    # external backing image for FC W2, SRNN M2 and SCNN Conv1..8/classifier.
    # The complete SCNN image exceeds 2^22 words. Use the existing 24-bit
    # protocol's full external range; this does not infer on-chip storage.
    "PARAM_DEPTH": "parameter int PARAM_DEPTH = 16777216",
    "PARAM_ADDR_WIDTH": "parameter int PARAM_ADDR_WIDTH = 24",
    # SCNN classifier learning flattens one B4xT4 adaptation window into
    # sixteen learner samples before the single gradient reduction/RMW pass.
    "MAX_BATCH": "parameter int MAX_BATCH = 16",
    "ENABLE_SRNN_WH1_LEARNING": (
        "parameter bit ENABLE_SRNN_WH1_LEARNING = 1'b0"
    ),
    "ENABLE_ONLINE_LEARNING": (
        "parameter bit ENABLE_ONLINE_LEARNING = 1'b1"
    ),
    "ENABLE_SCNN_CONV8_LEARNING": (
        "parameter bit ENABLE_SCNN_CONV8_LEARNING = 1'b0"
    ),
    "SCNN_WEIGHT_FRACTIONAL_BITS": (
        "parameter int SCNN_WEIGHT_FRACTIONAL_BITS = 15"
    ),
    "ACCUMULATE_SCNN_SAMPLE_LOGITS": (
        "parameter bit ACCUMULATE_SCNN_SAMPLE_LOGITS = 1'b1"
    ),
}
for name, declaration in system_parameters + plane_parameters:
    if name in {"USE_INTERNAL_SCNN_PARAMETER_STORE", "ENABLE_FC_CLIENT",
                "ENABLE_SRNN_CLIENT", "ENABLE_SCNN_CLIENT",
                "ENABLE_SRNN_M2_H40_CLIENT", "ENABLE_SRNN_GROUP_STREAMER",
                "USE_SHARED_SRNN_H40_FORWARD", "USE_EXTERNAL_OUTPUT_LEARNER",
                "SHARE_FC_SCNN_OUTPUT_LEARNER"}:
        continue
    if name not in parameter_names:
        parameter_names.add(name)
        wrapper_parameters.append(
            wrapper_parameter_overrides.get(name, declaration))

manual_ports = [
    "input logic i_clk",
    "input logic i_rst_n",
    "input logic i_clear_error",
    "input logic i_srnn_phase_abort",
    "input logic i_srnn_phase_capture_cfg_valid",
    "output logic o_srnn_phase_capture_cfg_ready",
    "input logic [$clog2(SRNN_BACKBONE_MAX + 1)-1:0] i_srnn_phase_record_count",
    "input logic i_srnn_phase_start_valid",
    "output logic o_srnn_phase_start_ready",
    "input logic i_srnn_phase_learning_enable",
    "input logic i_srnn_phase_target_valid",
    "output logic o_srnn_phase_target_ready",
    "input logic [5:0] i_srnn_phase_target_sample_id",
    "input logic [9:0] i_srnn_phase_target_backbone_id",
    "input logic signed [31:0] i_srnn_phase_target_q14",
    "input logic i_srnn_phase_target_last",
    "input logic i_srnn_phase_target_done",
    "input logic i_srnn_phase_target_error",
    "output logic o_srnn_phase_capture_done",
    "output logic o_srnn_phase_capture_complete",
    "output logic o_srnn_phase_delivery_done",
    "output logic o_srnn_phase_busy",
    "output logic o_srnn_phase_idle",
    "output logic o_srnn_phase_sticky_error",
]

wrapper_ports: list[str] = []
wrapper_port_names: set[str] = set()


def append_port(declaration: str) -> None:
    name = declaration_name(declaration)
    if name in wrapper_port_names:
        raise ValueError(f"duplicate wrapper port {name}")
    wrapper_port_names.add(name)
    wrapper_ports.append(declaration)


for name, _, declaration in system_ports:
    if name not in system_connections:
        append_port(declaration)
for declaration in manual_ports:
    append_port(declaration)


def plane_external_name(name: str) -> str:
    if name in {"i_parameter_owner", "i_topology_learning_enable"}:
        return name
    if name.startswith(("i_store_", "o_store_")) or name.startswith("o_backend_"):
        return name
    if name.startswith("o_owner_"):
        return name
    if name.startswith("i_"):
        return "i_learning_" + name[2:]
    if name.startswith("o_"):
        return "o_learning_" + name[2:]
    raise ValueError(name)


for name, _, declaration in plane_ports:
    if name in plane_connections:
        continue
    external_name = plane_external_name(name)
    plane_connections[name] = external_name
    append_port(rename_declaration(declaration, name, external_name))

# Apply the B3 synthesis-time learning boundary only after the ordinary child
# ports have been exported. This preserves the exact B2 wrapper interface while
# making update-only traffic constant-dead when ENABLE_ONLINE_LEARNING is zero.
system_connections["i_system_learning_enable"] = (
    "(TOPOLOGY_MASK != 0) && ENABLE_ONLINE_LEARNING && i_system_learning_enable"
)
system_connections["i_learning_enable"] = (
    "HAS_SCNN && ENABLE_ONLINE_LEARNING && i_learning_enable"
)
system_connections["i_fc_srnn_learning_cfg_valid"] = (
    "(HAS_FC || HAS_SRNN) && ENABLE_ONLINE_LEARNING && "
    "i_fc_srnn_learning_cfg_valid"
)
plane_connections["i_topology_learning_enable"] = (
    "ENABLE_ONLINE_LEARNING ? "
    "(i_topology_learning_enable & TOPOLOGY_MASK[2:0]) : 3'b000"
)

lines: list[str] = [
    "`timescale 1ns/1ps",
    "`default_nettype none",
    "",
    "// Production composition boundary: one C17 12xH40/P4 system, one",
    "// FC/SRNN/SCNN parameter plane and one SRNN backbone-to-M2 phase bridge.",
    "// The file is generated from the child ANSI interfaces; edit the generator",
    "// when changing composition policy.",
    "module unified_h40_final_composition_top_v1 #(",
]
for index, declaration in enumerate(wrapper_parameters):
    lines.append("    " + declaration + ("," if index + 1 < len(wrapper_parameters) else ""))
lines.append(") (")
for index, declaration in enumerate(wrapper_ports):
    lines.append("    " + declaration + ("," if index + 1 < len(wrapper_ports) else ""))
lines.extend([
    ");",
    "",
    "    localparam bit HAS_FC = (TOPOLOGY_MASK & 1) != 0;",
    "    localparam bit HAS_SRNN = (TOPOLOGY_MASK & 2) != 0;",
    "    localparam bit HAS_SCNN = (TOPOLOGY_MASK & 4) != 0;",
    "",
])

for link_name in sorted(link_sources):
    source_name, declaration = link_sources[link_name]
    lines.append(wire_declaration(declaration, source_name, link_name))
lines.append("    logic link_fc_observation_done;")
lines.append("")
lines.append("    assign link_fc_observation_done =")
lines.append("        HAS_FC && o_fc_srnn_learning_capture_done &&")
lines.append("        i_fc_srnn_learning_fc_output_layer &&")
lines.append("        ({1'b0, i_fc_srnn_learning_sample_id} + 7'd1 ==")
lines.append("            {2'b00, i_learning_fc_batch_count});")
lines.append("")

lines.append("    generate if (HAS_SRNN) begin : g_srnn_phase_bridge")
lines.append("      unified_srnn_backbone_m2_phase_bridge_v1 #(")
lines.append("          .MAX_RECORDS(SRNN_BACKBONE_MAX),")
lines.append("          .SAMPLE_ID_WIDTH(6), .BACKBONE_ID_WIDTH(10)")
lines.append("      ) u_srnn_phase_bridge (")
bridge_connections = [
    ("i_clk", "i_clk"), ("i_rst_n", "i_rst_n"),
    ("i_abort", "i_srnn_phase_abort"), ("i_clear_error", "i_clear_error"),
    ("i_capture_cfg_valid", "i_srnn_phase_capture_cfg_valid"),
    ("o_capture_cfg_ready", "o_srnn_phase_capture_cfg_ready"),
    ("i_capture_record_count", "i_srnn_phase_record_count"),
    ("i_actual_valid", "link_srnn_actual_valid"),
    ("o_actual_ready", "link_srnn_actual_ready"),
    ("i_actual_path", "link_srnn_actual_path"),
    ("i_actual_sample_id", "link_srnn_actual_sample_id"),
    ("i_actual_backbone_id", "link_srnn_actual_backbone_id"),
    ("i_actual_q14", "link_srnn_actual_q14"),
    ("i_actual_last", "link_srnn_actual_last"),
    ("o_capture_done", "o_srnn_phase_capture_done"),
    ("o_capture_complete", "o_srnn_phase_capture_complete"),
    ("i_sequence_start_valid", "i_srnn_phase_start_valid"),
    ("o_sequence_start_ready", "o_srnn_phase_start_ready"),
    ("i_include_target_path",
     "ENABLE_ONLINE_LEARNING && i_srnn_phase_learning_enable && "
     "ENABLE_SRNN_WH1_LEARNING"),
    ("o_plane_timestep_start", "link_srnn_plane_timestep_start"),
    ("i_plane_timestep_start_ready", "link_srnn_plane_timestep_start_ready"),
    ("i_target_valid", "i_srnn_phase_target_valid"),
    ("o_target_ready", "o_srnn_phase_target_ready"),
    ("i_target_sample_id", "i_srnn_phase_target_sample_id"),
    ("i_target_backbone_id", "i_srnn_phase_target_backbone_id"),
    ("i_target_q14", "i_srnn_phase_target_q14"),
    ("i_target_last", "i_srnn_phase_target_last"),
    ("i_target_done", "i_srnn_phase_target_done"),
    ("i_target_error", "i_srnn_phase_target_error"),
    ("o_plane_backbone_valid", "link_srnn_plane_backbone_valid"),
    ("i_plane_backbone_ready", "link_srnn_plane_backbone_ready"),
    ("o_plane_backbone_path", "link_srnn_plane_backbone_path"),
    ("o_plane_backbone_sample_id", "link_srnn_plane_backbone_sample_id"),
    ("o_plane_backbone_id", "link_srnn_plane_backbone_id"),
    ("o_plane_backbone_q14", "link_srnn_plane_backbone_q14"),
    ("o_plane_backbone_last", "link_srnn_plane_backbone_last"),
    ("o_plane_backbone_done", "link_srnn_plane_backbone_done"),
    ("o_plane_backbone_error", "link_srnn_plane_backbone_error"),
    ("o_delivery_done", "o_srnn_phase_delivery_done"),
    ("o_busy", "o_srnn_phase_busy"), ("o_idle", "o_srnn_phase_idle"),
    ("o_sticky_error", "o_srnn_phase_sticky_error"),
]
for index, (port, signal) in enumerate(bridge_connections):
    lines.append(f"          .{port}({signal})" + ("," if index + 1 < len(bridge_connections) else ""))
lines.append("      );")
lines.append("    end else begin : g_no_srnn_phase_bridge")
for signal, value in [
        ("link_srnn_actual_ready", "1'b0"),
        ("link_srnn_plane_backbone_valid", "1'b0"),
        ("link_srnn_plane_backbone_path", "1'b0"),
        ("link_srnn_plane_backbone_sample_id", "'0"),
        ("link_srnn_plane_backbone_id", "'0"),
        ("link_srnn_plane_backbone_q14", "'0"),
        ("link_srnn_plane_backbone_last", "1'b0"),
        ("link_srnn_plane_backbone_done", "1'b0"),
        ("link_srnn_plane_backbone_error", "1'b0"),
        ("link_srnn_plane_timestep_start", "1'b0"),
        ("o_srnn_phase_capture_cfg_ready", "1'b0"),
        ("o_srnn_phase_start_ready", "1'b0"),
        ("o_srnn_phase_target_ready", "1'b0"),
        ("o_srnn_phase_capture_done", "1'b0"),
        ("o_srnn_phase_capture_complete", "1'b0"),
        ("o_srnn_phase_delivery_done", "1'b0"),
        ("o_srnn_phase_busy", "1'b0"),
        ("o_srnn_phase_idle", "1'b1"),
        ("o_srnn_phase_sticky_error", "1'b0"),
]:
    lines.append(f"      assign {signal} = {value};")
lines.append("    end endgenerate")
lines.append("")

system_parameter_names = [name for name, _ in system_parameters]
lines.append(f"    {SYSTEM_MODULE} #(")
for index, name in enumerate(system_parameter_names):
    value = {
        "USE_INTERNAL_SCNN_PARAMETER_STORE": "1'b0",
        "ENABLE_FC_CLIENT": "HAS_FC",
        "ENABLE_SRNN_CLIENT": "HAS_SRNN",
        "ENABLE_SCNN_CLIENT": "HAS_SCNN",
        "ENABLE_SRNN_M2_H40_CLIENT": "HAS_SRNN",
        "ENABLE_SRNN_GROUP_STREAMER": "HAS_SRNN",
        "USE_EXTERNAL_OUTPUT_LEARNER": "1'b1",
    }.get(name, name)
    lines.append(f"        .{name}({value})" + ("," if index + 1 < len(system_parameter_names) else ""))
lines.append("    ) u_system (")
for index, (name, _, _) in enumerate(system_ports):
    signal = system_connections.get(name, name)
    connection = f".{name}({signal})" if signal else f".{name}()"
    lines.append("        " + connection + ("," if index + 1 < len(system_ports) else ""))
lines.append("    );")
lines.append("")

plane_parameter_names = [name for name, _ in plane_parameters]
lines.append(f"    {PLANE_MODULE} #(")
for index, name in enumerate(plane_parameter_names):
    value = {
        "ENABLE_ONLINE_LEARNING": "ENABLE_ONLINE_LEARNING",
        "ENABLE_FC_CLIENT": "HAS_FC",
        "ENABLE_SRNN_CLIENT": "HAS_SRNN",
        "ENABLE_SCNN_CLIENT": "HAS_SCNN",
        "USE_SHARED_SRNN_H40_FORWARD": "1'b1",
        "SHARE_FC_SCNN_OUTPUT_LEARNER": "1'b1",
        "ENABLE_SRNN_WH1_LEARNING": "ENABLE_SRNN_WH1_LEARNING",
    }.get(name, name)
    lines.append(f"        .{name}({value})" + ("," if index + 1 < len(plane_parameter_names) else ""))
lines.append("    ) u_parameter_plane (")
for index, (name, _, _) in enumerate(plane_ports):
    signal = plane_connections[name]
    lines.append(f"        .{name}({signal})" + ("," if index + 1 < len(plane_ports) else ""))
lines.append("    );")
lines.extend(["endmodule", "", "`default_nettype wire", ""])

OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"generated {OUTPUT_PATH} with {len(wrapper_ports)} ports")
