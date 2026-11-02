#!/usr/bin/env python3
"""Bounded real-count FC training through the production composition top."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def resolve_verification_store(root=None):
    root = ROOT if root is None else root
    candidates = (
        root / "rtl/learning/unified_tp_shared_forward_rmw_store.sv",
        root / "verification/support/unified_tp_shared_forward_rmw_store.sv",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("verification store not found; checked: " +
                            ", ".join(str(path) for path in candidates))


def build_settings(obj, source_list, *, verilator=None, jobs=1, platform=None, environ=None,
                   top="tb_fc_final_counts"):
    windows = (os.name if platform is None else platform) == "nt"
    env = dict(os.environ if environ is None else environ)
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if windows and verilator is None:
        verilator = "C:/msys64/mingw64/bin/verilator_bin.exe"
        env["VERILATOR_ROOT"] = "C:/msys64/mingw64/share/verilator"
        env["PATH"] = "C:/msys64/mingw64/bin;C:/msys64/usr/bin;" + env.get("PATH", "")
    elif verilator is None:
        verilator = shutil.which("verilator", path=env.get("PATH"))
        if not verilator:
            raise FileNotFoundError("pass --verilator or put Verilator on PATH")
    cflags = "-O1 -D_GLIBCXX_USE_CXX11_ABI=0" if windows else "-O1"
    build = [str(verilator), "--binary", "--timing", "--assert", "-j", str(jobs),
             "-Wall", "-Wno-fatal", "--output-split", "3000", "-CFLAGS", cflags,
             "--top-module", top, "--Mdir", obj.as_posix(),
             "-f", source_list.as_posix()]
    exe = obj / ("V" + top + (".exe" if windows else ""))
    return build, exe, env


def ansi_header(path):
    text = path.read_text(encoding="utf-8")
    module = re.search(r"module\s+(\w+)\s*#\(", text)
    parameter_end = text.index(") (", module.end())
    port_end = text.index("\n);", parameter_end)
    parameters = []
    for line in text[module.end():parameter_end].splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        match = re.fullmatch(r"parameter\s+(.+?)\s+(\w+)\s*=\s*(.+)", line)
        if not match:
            raise ValueError(f"unsupported parameter declaration: {line}")
        parameters.append(match.groups())
    ports = []
    for line in text[parameter_end+3:port_end].splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        match = re.fullmatch(r"(input|output)\s+(?:wire\s+)?(logic.*?)\s+(\w+)", line)
        if not match:
            raise ValueError(f"unsupported port declaration: {line}")
        ports.append(match.groups())
    return module.group(1), parameters, ports


def generate_tb(path, *, top="tb_fc_final_counts", body=None, parameter_overrides=None):
    module, parameters, ports = ansi_header(ROOT / "rtl/top/unified_h40_final_composition_top_v1.sv")
    overrides = {"PARAM_DEPTH": "371520"}
    overrides.update(parameter_overrides or {})
    lines = ["`timescale 1ns/1ps", "`default_nettype none", f"module {top};"]
    lines += [f"    localparam {kind} {name} = {overrides.get(name, value)};"
              for kind, name, value in parameters]
    for direction, kind, name in ports:
        initial = " = '0" if direction == "input" and not name.startswith("i_store_") else ""
        lines.append(f"    {kind} {name}{initial};")
    lines += [f"    {module} #(",
              ",\n".join(f"        .{name}({name})" for _, name, _ in parameters),
              "    ) u_final (",
              ",\n".join(f"        .{name}({name})" for _, _, name in ports), "    );"]
    store, _, store_ports = ansi_header(resolve_verification_store())
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
            raise ValueError(f"store link missing: {signal}")
        connections.append(f"        .{name}({signal})")
    lines += [f"    {store} #(.DEPTH(PARAM_DEPTH), .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH)) u_store (",
              ",\n".join(connections), "    );",
              "    assign i_store_quiescent = !o_store_load_valid && !o_store_learning_read_valid &&",
              "        !o_store_learning_write_valid && !o_store_forward_read_valid &&",
              "        !i_store_learning_response_valid && !i_store_forward_response_valid;",
              (body or ROOT / "tb/include/fc_final_counts_body.svh").read_text(encoding="utf-8"),
              "endmodule", "`default_nettype wire"]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2, choices=range(1, 100))
    parser.add_argument("--learning", choices=("on", "off"), default="on")
    parser.add_argument("--batches", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--reuse-executable", type=Path)
    parser.add_argument("--fixture", type=Path,
                        help="Reuse a generated integer reference fixture inside the project")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--verilator", help="Verilator executable; defaults to the platform tool")
    parser.add_argument("--jobs", type=int, default=1, help="Bounded compiler parallelism")
    parser.add_argument("--run-name", help="Optional new evidence/work directory name (no overwrite)")
    parser.add_argument("--activity-monitor", action="store_true", help="Bind passive physical tile counters")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.run_name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("--run-name must be a simple directory name")
    stamp = args.run_name or dt.datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out = ROOT / "results/paper_evidence/supplement_20261004/rtl/fc_final_counts" / stamp
    work = ROOT / "work/supplement_20261004/fc_final_counts" / stamp
    out.mkdir(parents=True)
    work.mkdir(parents=True)
    tb = work / "tb_fc_final_counts.sv"
    generate_tb(tb)
    print(f"Evidence: {out}", flush=True)
    if args.generate_only:
        print(f"Generated: {tb}")
        return
    fixture = args.fixture.resolve() if args.fixture else (
        ROOT / "results/paper_evidence/supplement_20261004/algorithm/fc_final_counts_reference" / stamp)
    if not args.build_only and not args.fixture:
        if args.batches != 1:
            parser.error("multiple batches require a continuous-state episode fixture")
        from build_fc_final_counts_fixture import build_fixture
        build_fixture(fixture, steps=args.steps, learning=args.learning == "on")
    if not args.build_only:
        reference = json.loads((fixture / "reference.json").read_text(encoding="utf-8"))
        if (reference["steps"] != args.steps or reference["learning"] != (args.learning == "on")
                or reference["batch"] != 16 or reference.get("batches", 1) != args.batches):
            raise ValueError("fixture timestep/batch/learning settings do not match this run")
    sources = sorted(ROOT.glob("rtl/**/*.sv")) + [tb]
    if args.activity_monitor:
        sources += [ROOT / "tb/monitors/supplement_h40_activity_monitor.sv"]
    source_list = work / "sources.f"
    source_list.write_text("\n".join('"'+p.as_posix()+'"' for p in sources)+"\n", encoding="ascii")
    obj = work / "obj"
    build, exe, env = build_settings(obj, source_list, verilator=args.verilator, jobs=args.jobs)
    if args.reuse_executable:
        exe = args.reuse_executable.resolve()
    run = [str(exe), "+FIXTURE="+fixture.relative_to(ROOT).as_posix(), f"+STEPS={args.steps}",
           f"+LEARNING={int(args.learning == 'on')}", f"+BATCHES={args.batches}"]
    record = {"cwd": str(ROOT), "build": build, "run": run,
              "fixture": str(fixture),
              "activity_monitor_requested": args.activity_monitor,
              "reuse_executable": str(exe) if args.reuse_executable else None}
    (out / "commands.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    stages = ([] if args.reuse_executable else [("build", build)])
    if not args.build_only:
        stages += [("run", run)]
    for stage, command in stages:
        start = time.monotonic()
        with (out / f"{stage}.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record[stage+"_exit"] = process.returncode
        record[stage+"_seconds"] = round(time.monotonic()-start, 3)
        (out / "commands.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
        print(f"{stage}: exit={process.returncode}, seconds={record[stage+'_seconds']}", flush=True)
        if process.returncode:
            raise SystemExit(f"Preserved failure at {out}")
    if args.build_only:
        print(f"Built executable (not simulated): {exe}")
        return
    log = (out / "run.log").read_text(encoding="utf-8")
    if "FC_FINAL_COUNTS_PASS" not in log:
        raise RuntimeError("missing final completion marker")
    rows = [{k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", line)}
            for line in log.splitlines() if line.startswith("FC_FINAL_COUNTS_")]
    (out / "measurements.json").write_text(json.dumps({
        "status": "PASS", "learning": args.learning, "steps": rows[-1]["steps"], "batch": 16,
        "batches": args.batches, "fixture": str(fixture),
        "rows": rows, "final_top_simulated": True,
        "scope": "FC raw-count timestep groups through production composition/learner/store; TB drives scheduling",
        "board_measured": False, "physical_timing_applies": False,
        "full_dataset_accuracy": False}, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
