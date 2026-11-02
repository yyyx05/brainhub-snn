"""Small Verilator-only context barrier regression; default prints the plan."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/paper_evidence/supplement_20261004/rtl/scnn_context_frame_service"
RTL = "rtl/control/unified_scnn_selected_layers_context_bridge_v1.sv"
CASES = (
    ("frame_service", "tb_unified_scnn_selected_layers_context_frame_service_v1"),
    ("legacy", "tb_unified_scnn_selected_layers_context_bridge_v1"),
)


def shell_path(path: Path, windows: bool):
    value = path.resolve().as_posix()
    if windows:
        value = "/" + value[0].lower() + value[2:]
    return value


def build_command(top, output, verilator, windows):
    args = [verilator, "--binary", "--timing", "--assert", "-j", "1", "-Wall",
            "-Wno-fatal", "-Wno-UNUSEDSIGNAL", "-Wno-UNUSEDPARAM",
            "-Wno-TIMESCALEMOD", "-Wno-PROCASSINIT", "-Wno-SYNCASYNCNET",
            "-Wno-BLKSEQ", "-Wno-PINCONNECTEMPTY"]
    if top == CASES[1][1]:
        # The unchanged legacy TB intentionally omits the disabled service ports.
        args.append("-Wno-PINMISSING")
    if windows:
        args += ["-CFLAGS", "-D_GLIBCXX_USE_CXX11_ABI=0"]
    return args + ["--Mdir", shell_path(output, windows), "--top-module", top,
                   RTL, f"tb/{top}.sv"]


def check_log(case, output):
    if case == "legacy":
        if "PASS: resident classifier and refilled Conv waves share one versioned two-pair H40 cache" not in output:
            raise ValueError("legacy PASS marker missing")
        return dict(status="PASS", unchanged_legacy_test=True)
    match = re.search(r"SCNN_CONTEXT_FRAME_SERVICE_PASS cases=(\d+) cache=(\d+) service=(\d+) load=(\d+) release=(\d+)", output)
    names = re.findall(r"SCNN_CONTEXT_FRAME_CASE_PASS (\S+)", output)
    if match is None or int(match[1]) != 14 or len(names) != 14 or len(set(names)) != 14:
        raise ValueError("incomplete frame-service directed results")
    return dict(status="PASS", cases=names,
                handshakes=dict(zip(("cache", "service", "load", "release"), map(int, match.groups()[1:]))))


def execute(args, output, bash, windows):
    if windows:
        script = ("export PATH=/mingw64/bin:/usr/bin:$PATH\n"
                  "export VERILATOR_ROOT=/mingw64/share/verilator\n"
                  f"cd {shlex.quote(shell_path(ROOT, True))}\nexec {shlex.join(args)}")
        command = [str(bash), "-lc", script]
    else:
        command = args
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8",
                            errors="replace", stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=300)
    output.write_text(result.stdout, encoding="utf-8")
    row = dict(command=args, exit_code=result.returncode,
               seconds=time.perf_counter()-started,
               warnings=[line for line in result.stdout.splitlines() if line.startswith("%Warning")])
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}: {output}\n{result.stdout[-6000:]}")
    return row, result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verilator", default="verilator")
    parser.add_argument("--bash", type=Path, default=Path("C:/msys64/usr/bin/bash.exe"))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(BASE) or output == BASE or output.exists():
        raise ValueError("choose a fresh result directory under scnn_context_frame_service")
    plan = dict(scope="leaf context barrier only, not final-top or full-graph accuracy",
                sources=[RTL, *[f"tb/{top}.sv" for _, top in CASES]],
                cases=[case for case, _ in CASES], jobs=1, run=args.run,
                output=str(output), tools="Verilator only; no SSH, Vivado or XSim")
    print(json.dumps(plan, indent=2), flush=True)
    if not args.run:
        return
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    rows = []
    windows = os.name == "nt"
    try:
        for name, top in CASES:
            folder = output / name
            folder.mkdir()
            build, _ = execute(build_command(top, folder / "obj_dir", args.verilator, windows),
                               folder / "build.log", args.bash, windows)
            executable = folder / "obj_dir" / ("V" + top + (".exe" if windows else ""))
            run, log = execute([shell_path(executable, windows)], folder / "run.log", args.bash, windows)
            row = dict(name=name, build=build, run=run, checks=check_log(name, log))
            rows.append(row)
            print(f"SCNN_CONTEXT_FRAME_RUN_PASS {name}", flush=True)
    except Exception as error:
        (output / "summary.json").write_text(json.dumps(dict(status="FAIL", plan=plan,
            completed=rows, error=str(error)), indent=2) + "\n", encoding="utf-8")
        raise
    (output / "summary.json").write_text(json.dumps(dict(status="PASS", plan=plan,
        completed=rows), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
