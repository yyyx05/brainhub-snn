#!/usr/bin/env python3
"""Generate ports from the actual candidate engine for directed cache control tests."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from run_fc_final_counts import ansi_header


def main():
    root = ROOT
    if os.name == "nt":
        root = Path("\\\\?\\" + str(root))
    experiment = root / "verification/experiments/srnn_resident_weight_context"
    engine = experiment / "candidate_r3/rtl/learning/unified_srnn_m2_h40_forward_engine_v1.sv"
    module, parameters, ports = ansi_header(engine)
    driven = {"i_forward_read_ready", "i_forward_response_valid", "i_forward_response_tag",
              "i_forward_response_code", "i_forward_response_error"}
    lines = ["`timescale 1ns/1ps", "`default_nettype none", "module tb_srnn_resident_control;"]
    lines += [f"    localparam {kind} {name} = {value};" for kind, name, value in parameters]
    for direction, kind, name in ports:
        initial = " = '0" if direction == "input" and name not in driven else ""
        lines.append(f"    {kind} {name}{initial};")
    lines += [f"    {module} #(", ",\n".join(f"        .{name}({name})" for _, name, _ in parameters),
              "    ) u_dut (", ",\n".join(f"        .{name}({name})" for _, _, name in ports), "    );"]
    lines.append((experiment / "tb/srnn_resident_control_body.svh").read_text(encoding="utf-8"))
    lines += ["endmodule", "`default_nettype wire", ""]
    output = experiment / "tb/tb_srnn_resident_control.sv"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
