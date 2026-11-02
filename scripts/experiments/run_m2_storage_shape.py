#!/usr/bin/env python3
"""Small M2 memory-shape experiment, not a final-core replacement or timing signoff."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time


def parse_unit(text):
    if re.search(r"%Error|\$fatal|Assertion failed|%Warning-(?:MULTIDRIVEN|LATCH|UNOPTFLAT)", text):
        raise ValueError("unit diagnostic failure")
    pattern = (r"^M2_SHAPE_PASS depth=(\d+) reads=(\d+) writes=(\d+) "
               r"stalls=(\d+) conflicts=(\d+) replacements=(\d+)$")
    rows = [dict(zip(("depth", "reads", "writes", "stalls", "conflicts", "replacements"),
                     map(int, match))) for match in re.findall(pattern, text, re.M)]
    if sorted(row["depth"] for row in rows) != [512, 578]:
        raise ValueError("both capacity domains must pass exactly once")
    if any(any(row[key] <= 0 for key in row) for row in rows):
        raise ValueError("missing exercised behavior")
    if text.count("M2_STORAGE_SHAPE_COMPLETE\n") != 1:
        raise ValueError("missing/duplicate final completion")
    if text.index("M2_STORAGE_SHAPE_COMPLETE\n") < max(text.index("M2_SHAPE_PASS depth="+str(d)) for d in (512,578)):
        raise ValueError("early completion")
    return rows


def worker():
    from run_fc_bitmap_mapping import vivado_environment
    verilator = "/root/autodl-tmp/verilator_5_048_install/bin/verilator"
    vivado = "/root/autodl-tmp/Xilinx_2022_2/Vivado/2022.2/bin/vivado"
    status = {"status": "running", "scope": "module_only_not_core_or_physical_closure"}

    def save():
        Path("status.json").write_text(json.dumps(status, indent=2)+"\n")

    def stage(name, command, timeout, env=None):
        status["status"] = name+"_running"
        save()
        start = time.monotonic()
        with open(name+".log", "w") as stream:
            p = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stream,
                               stderr=subprocess.STDOUT, timeout=timeout, env=env)
        status[name] = {"exit": p.returncode, "seconds": time.monotonic()-start,
                        "command": command}
        save()
        if p.returncode:
            raise RuntimeError(name+" exit "+str(p.returncode))
        return Path(name+".log").read_text(errors="replace")

    try:
        build = stage("build", [verilator, "--binary", "--timing", "--assert", "-Wno-fatal",
                      "--top-module", "tb_m2_storage_shape", "--Mdir", "obj", "-j", "4",
                      "baseline.sv", "m2_compact_store.sv", "m2_storage_shape.sv",
                      "tb_m2_storage_shape.sv"], 600)
        structural = re.findall(r"%Warning-(MULTIDRIVEN|LATCH|UNOPTFLAT)", build)
        if structural:
            raise RuntimeError("structural warnings: "+str(structural))
        status["build_warnings"] = re.findall(r"%Warning-([A-Z0-9_]+)", build)
        status["checks"] = parse_unit(stage("run", ["obj/Vtb_m2_storage_shape"], 60))
        mapped = stage("mapping", [vivado, "-mode", "batch", "-nojournal", "-nolog",
                       "-source", "mapping.tcl", "-tclargs", str(Path.cwd())],
                       1200, vivado_environment(vivado))
        status["mapping_diagnostics"] = {name: len(re.findall(pattern, mapped, re.M)) for name, pattern in (
            ("errors", r"^ERROR:"), ("critical_warnings", r"^CRITICAL WARNING:"), ("warnings", r"^WARNING:"))}
        if status["mapping_diagnostics"]["errors"] or status["mapping_diagnostics"]["critical_warnings"] or "M2_STORAGE_MAPPING_COMPLETE\n" not in mapped:
            raise RuntimeError("mapping incomplete/critical diagnostics")
        status["status"] = "PASS_module_equivalence_and_synthesis_not_full_core"
    except Exception as error:
        status.update(status="FAIL", error=str(error))
    save()
    return 0 if status["status"].startswith("PASS_") else 1


def launch(name):
    from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, upload, local_path
    source = ROOT / "verification/experiments/m2_storage_shape/candidate_r1"
    target = REMOTE+"/"+name
    c = connect()
    try:
        processes = run(c, "ps -eo comm,args")
        if any("/lnx64.o/vivado" in line or line.lstrip().startswith("vivado ") for line in processes.splitlines()):
            raise RuntimeError("another Vivado is active")
        disk = run(c, "df -B1 /root/autodl-tmp")
        print(disk)
        # This module-only experiment fits well below the available space;
        # it neither copies a full core nor deletes prior artifacts.
        if int(disk.splitlines()[-1].split()[3]) < 300_000_000:
            raise RuntimeError("insufficient room for module experiment")
        with c.open_sftp() as s:
            s.mkdir(target)
            original = ROOT / RESULT / "scnn_issue_banked_synth_r1/rtl/memory/unified_dual_role_weight_store_p4all_candidate_v1.sv"
            upload(s, local_path(original), target+"/baseline.sv")
            for file in source.glob("*.sv"):
                upload(s, local_path(file), target+"/"+file.name)
            for file, destination in ((Path(__file__), "worker.py"),
                    (Path(__file__).with_name("run_fc_bitmap_mapping.py"), "run_fc_bitmap_mapping.py"),
                    (Path(__file__).with_name("m2_storage_shape.tcl"), "mapping.tcl")):
                upload(s, local_path(file), target+"/"+destination)
            with s.open(target+"/clock.xdc", "w") as f:
                f.write("create_clock -name core_clk -period 15.000 [get_ports i_clk]\n")
        command = "cd "+shlex.quote(target)+" && exec python3 worker.py --worker"
        screen = "snn_"+name
        run(c, "screen -dmS "+shlex.quote(screen)+" bash -lc "+shlex.quote(command))
        print(json.dumps({"remote":target,"screen":screen,"launched":True}))
    finally:
        c.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--name", default="m2_storage_shape_r1")
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker())
    launch(args.name)
