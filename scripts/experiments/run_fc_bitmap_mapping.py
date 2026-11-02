#!/usr/bin/env python3
"""Sequential, module-only mapping comparison; no full-core physical claim."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time


def vivado_environment(vivado, parent=None):
    # Match the accepted full-core server shell's compatibility libraries.
    env = dict(os.environ if parent is None else parent)
    root = str(PurePosixPath(vivado).parent.parent)
    env["XILINX_VIVADO"] = root
    env["PATH"] = root + "/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = root + "/lib/lnx64.o/SuSE:" + env.get("LD_LIBRARY_PATH", "")
    if Path("/lib/x86_64-linux-gnu/libudev.so.1").is_file():
        env["LD_PRELOAD"] = "/lib/x86_64-linux-gnu/libudev.so.1"
    return env


def worker(vivado, inspect_replay=False):
    started = time.monotonic()
    command = [vivado, "-mode", "batch", "-nojournal", "-nolog", "-source",
               "mapping.tcl", "-tclargs", str(Path.cwd())]
    if inspect_replay:
        for name in ("srnn_specialized_synth_r1", "scnn_issue_sparse_synth_r1"):
            command.append(str(Path.cwd().parent / name /
                               "server_results/final_composition_synth/post_synth.dcp"))
    status = {"status": "running", "command": command}
    Path("status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    try:
        with Path("vivado.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=900,
                                    env=vivado_environment(vivado))
        text = Path("vivado.log").read_text(encoding="utf-8", errors="replace")
        status.update(exit_code=result.returncode, elapsed_seconds=time.monotonic()-started,
                      errors=len(re.findall(r"(?m)^ERROR:", text)),
                      critical_warnings=len(re.findall(r"(?m)^CRITICAL WARNING:", text)),
                      warnings=len(re.findall(r"(?m)^WARNING:", text)))
        if result.returncode or status["errors"] or "FC_BITMAP_MAPPING_COMPLETE\n" not in text:
            raise RuntimeError("module mapping did not complete")
        status["status"] = "PASS_module_mapping_flow_not_full_core_or_physical_closure"
    except subprocess.TimeoutExpired:
        status.update(status="FAIL", exit_code=None, timeout_seconds=900,
                      elapsed_seconds=time.monotonic()-started)
    except Exception as error:
        status.update(status="FAIL", error=str(error))
    Path("status.json").write_text(json.dumps(status, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0 if status["status"].startswith("PASS_") else 1


def launch(name, inspect_replay=False):
    from prepare_fc_contribution_pipeline import ROOT, SOURCE, EXPERIMENT
    from run_srnn_resident_remote import connect, run, upload, fetch_tree, REMOTE, RESULT
    c = connect()
    target = REMOTE + "/" + name
    try:
        ps = run(c, "ps -eo comm,args")
        if any("/lnx64.o/vivado" in line or line.lstrip().startswith("vivado ")
               for line in ps.splitlines()):
            raise RuntimeError("another Vivado is active")
        print(run(c, "df -B1 /root/autodl-tmp"))
        with c.open_sftp() as s:
            s.mkdir(target)
            for local, remote in (
                (ROOT / EXPERIMENT / "candidate_sparse_r2" / SOURCE, "baseline.sv"),
                (ROOT / EXPERIMENT / "candidate_banked_r3" / SOURCE, "candidate.sv"),
                (Path(__file__).with_name("fc_bitmap_mapping.tcl"), "mapping.tcl"),
                (Path(__file__), "worker.py"),
            ):
                upload(s, local, target+"/"+remote)
            with s.open(target+"/clock.xdc", "w") as f:
                f.write("create_clock -name core_clk -period 15.000 [get_ports i_clk]\n")
        command = "cd " + shlex.quote(target) + " && python3 worker.py --worker"
        if inspect_replay:
            command += " --inspect-replay"
        try:
            print(run(c, command))
        finally:
            with c.open_sftp() as s:
                fetch_tree(s, target, ROOT / RESULT / name)
    finally:
        c.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--name", default="fc_bitmap_mapping_r3")
    parser.add_argument("--inspect-replay", action="store_true")
    parser.add_argument("--vivado", default="/root/autodl-tmp/Xilinx_2022_2/Vivado/2022.2/bin/vivado")
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker(args.vivado, args.inspect_replay))
    launch(args.name, args.inspect_replay)
