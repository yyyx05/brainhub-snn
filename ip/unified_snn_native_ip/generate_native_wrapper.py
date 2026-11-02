"""Generate a wire-only Verilog adapter for Vivado 2022.2 IP Packager."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from run_fc_final_counts import ansi_header

WRAPPER = "unified_snn_native_wrapper"


def render_wrapper(top_source):
    module, parameters, ports = ansi_header(top_source)
    declarations = []
    for kind, name, value in parameters:
        if kind not in ("int", "bit"):
            raise ValueError(f"unsupported wrapper parameter type: {kind}")
        verilog_kind = "integer " if kind == "int" else ""
        declarations.append(f"    parameter {verilog_kind}{name} = {value}")
    lines = ["`timescale 1ns/1ps", "`default_nettype none",
             "// Packaging adapter only: all parameters and ports pass directly to the measured core.",
             f"module {WRAPPER} #(", ",\n".join(declarations), ") ("]
    lines.extend([",\n".join(f"    {direction} {kind.replace('logic', 'wire', 1)} {name}"
                             for direction, kind, name in ports), ");",
                  f"    {module} #(",
                  ",\n".join(f"        .{name}({name})" for _, name, _ in parameters),
                  "    ) u_core (",
                  ",\n".join(f"        .{name}({name})" for _, _, name in ports),
                  "    );", "endmodule", "`default_nettype wire", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("top_source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_wrapper(args.top_source), encoding="ascii")


if __name__ == "__main__":
    main()
