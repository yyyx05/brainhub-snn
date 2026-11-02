"""Directed subsystem address separation; serial local Verilator, no array/SSH/EDA."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re

from run_fc_final_counts import ansi_header
from run_scnn_logit_sample_contexts import ROOT, execute, shell_path


TOP = "tb_scnn_subsystem_parameter_bases"
DUT = ROOT / "rtl/top/unified_scnn_selected_layers_train_infer_subsystem_v1.sv"
BODY = ROOT / "tb/include/scnn_subsystem_parameter_bases_body.svh"


def generate_tb(path: Path, split: bool) -> None:
    # The subsystem uses multiline ANSI declarations; normalize each declaration.
    source = re.sub(r"//[^\n]*", "", DUT.read_text(encoding="utf-8"))
    module = re.search(r"module\s+(\w+)\s*#\(", source)
    parameter_end = source.index(") (", module.end())
    port_end = source.index("\n);", parameter_end)
    parameters = []
    for declaration in source[module.end():parameter_end].split(","):
        match = re.fullmatch(r"\s*parameter\s+(.+?)\s+(\w+)\s*=\s*(.+?)\s*",
                             " ".join(declaration.split()))
        if not match:
            raise ValueError(f"unsupported parameter: {declaration}")
        parameters.append(match.groups())
    ports = []
    for declaration in source[parameter_end + 3:port_end].split(","):
        match = re.fullmatch(r"\s*(input|output)\s+(?:wire\s+)?(logic.*?)\s+(\w+)\s*",
                             " ".join(declaration.split()))
        if not match:
            raise ValueError(f"unsupported port: {declaration}")
        ports.append(match.groups())
    overrides = dict(USE_INTERNAL_PARAMETER_STORE="0", USE_CONTEXT_PARAMETER_BASE=str(int(split)),
        PARAM_DEPTH="256", PARAM_ADDR_WIDTH="8", TAG_WIDTH="8", MAX_BATCH="2",
        MAX_TIMESTEPS="2", MAX_INPUT_CHANNELS="2", MAX_OUTPUT_CHANNELS="2",
        OBS_INPUT_CHANNELS="2", OBS_OUTPUT_CHANNELS="2", Q16_ADDR_WIDTH="8",
        Q64_ADDR_WIDTH="6", MAX_FEATURES="8", MAX_CLASSES="3")
    lines = ["`timescale 1ns/1ps", "`default_nettype none", f"module {TOP};"]
    lines += [f"    localparam {kind} {name} = {overrides.get(name, value)};"
              for kind, name, value in parameters]
    for direction, kind, name in ports:
        initial = " = '0" if direction == "input" and not name.startswith("i_store_") else ""
        lines.append(f"    {kind} {name}{initial};")
    # Omit the new generic in the legacy arm to exercise its production default.
    connections = [f"        .{name}({name})" for _, name, _ in parameters
                   if split or name != "USE_CONTEXT_PARAMETER_BASE"]
    lines += [f"    {module.group(1)} #(\n" + ",\n".join(connections) + "\n    ) dut (.*);"]
    store, _, store_ports = ansi_header(ROOT / "rtl/learning/unified_tp_shared_forward_rmw_store.sv")
    connections = []
    names = {name for _, _, name in ports}
    for direction, _, name in store_ports:
        if name in ("i_clk", "i_rst_n", "i_clear_error"):
            signal = name
        elif name == "o_protocol_error":
            signal = "i_store_protocol_error"
        else:
            signal = ("o_store_" if direction == "input" else "i_store_") + name[2:]
        if signal not in names:
            raise ValueError(f"missing store connection: {signal}")
        connections.append(f"        .{name}({signal})")
    lines += [f"    {store} #(.DEPTH(PARAM_DEPTH), .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),",
              "        .TAG_WIDTH(TAG_WIDTH)) backing (", ",\n".join(connections), "    );",
              BODY.read_text(encoding="ascii"), "endmodule", "`default_nettype wire"]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("split", "default"), action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verilator", default="verilator")
    parser.add_argument("--bash", default="C:/msys64/usr/bin/bash.exe")
    args = parser.parse_args()
    import os
    windows = os.name == "nt"
    output = (args.output or ROOT / "sim_work/scnn_subsystem_parameter_bases" /
              datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("output must remain inside the workspace")
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status="RUNNING", jobs=1, completed=[],
                  scope="captured learning base plus real forward refills; zero Conv gradient, no Conv RMW coverage; subsystem/store, not final top")
    print(f"OUTPUT {output}", flush=True)
    try:
        for case in args.case or ["split", "default"]:
            folder = output / case
            folder.mkdir()
            tb = folder / (TOP + ".sv")
            generate_tb(tb, case == "split")
            sources = folder / "sources.f"
            sources.write_text("\n".join('"' + p.as_posix() + '"'
                for p in [*sorted(ROOT.glob("rtl/**/*.sv")), tb]) + "\n", encoding="ascii")
            command = [args.verilator, "--binary", "--timing", "--assert", "-j", "1",
                       "-Wall", "-Wno-fatal", "-Wno-UNUSEDSIGNAL", "-Wno-UNUSEDPARAM",
                       "-Wno-PROCASSINIT", "-Wno-SYNCASYNCNET", "-Wno-BLKSEQ",
                       "-Wno-PINCONNECTEMPTY", "--output-split", "3000",
                       "--top-module", TOP, "--Mdir", shell_path(folder / "obj", windows),
                       "-f", shell_path(sources, windows)]
            if windows:
                command += ["-CFLAGS", "-O1 -D_GLIBCXX_USE_CXX11_ABI=0"]
            exe = folder / "obj" / ("V" + TOP + (".exe" if windows else ""))
            run = [shell_path(exe, windows)]
            (folder / "commands.json").write_text(json.dumps(dict(build=command, run=run), indent=2),
                                                   encoding="ascii")
            build = execute(command, folder / "build.log", args.bash, windows)
            log = execute(run, folder / "run.log", args.bash, windows)
            match = re.search(r"SCNN_PARAMETER_BASE_PASS split=(\d+) conv_reads=(\d+) conv_writes=(\d+) "
                              r"classifier_writes=(\d+) refills=(\d+) negative_cases=(\d+)", log)
            expected = (int(case == "split"), 0, 0, 24, 2, 2)
            if not match or tuple(map(int, match.groups())) != expected:
                raise RuntimeError(f"missing/incomplete dynamic address checks: {folder}")
            record["completed"].append(dict(case=case, status="PASS", counts=expected,
                warnings=[line for line in build.splitlines() if line.startswith("%Warning")]))
            print(match[0], flush=True)
        record["status"] = "PASS"
    except Exception as error:
        record.update(status="FAIL", error=str(error))
        raise
    finally:
        (output / "summary.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
