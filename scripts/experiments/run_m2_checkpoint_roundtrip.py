"""Probe checkpoint export/reopen only; never synthesize, place, or route."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import threading
import time
import zipfile

NAME = "m2_checkpoint_roundtrip_r1"
SOURCE = "m2_compact_impl_r1/server_results/final_composition_impl/post_place_physopt.dcp"
TCL = "verification/experiments/m2_checkpoint_roundtrip/roundtrip.tcl"
WORKER = "scripts/experiments/run_m2_checkpoint_roundtrip.py"
VIVADO = "/root/autodl-tmp/Xilinx_2022_2/Vivado/2022.2/bin/vivado"


def completed(exit_code, text):
    return (exit_code == 0 and text.splitlines().count("CHECKPOINT_ROUNDTRIP_COMPLETE") == 1
            and not re.search(r"^ERROR:|^CRITICAL WARNING:", text, re.M))


def worker(source):
    root = Path.cwd()
    out = root / "output"
    out.mkdir()
    state = dict(status="running", start=dt.datetime.now().astimezone().isoformat(),
                 source=source, scope="Checkpoint export/reopen only; no physical optimization.")
    (root / "status.json").write_text(json.dumps(state, indent=2) + "\n")
    env = os.environ.copy()
    install = str(Path(VIVADO).parents[1])
    env["XILINX_VIVADO"] = install
    env["PATH"] = install + "/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = install + "/lib/lnx64.o/SuSE:" + env.get("LD_LIBRARY_PATH", "")
    if Path("/lib/x86_64-linux-gnu/libudev.so.1").is_file():
        env["LD_PRELOAD"] = "/lib/x86_64-linux-gnu/libudev.so.1"
    command = [VIVADO, "-mode", "batch", "-nojournal", "-nolog", "-source", str(root / TCL),
               "-tclargs", source, str(out)]
    samples, done = [], threading.Event()
    def sample():
        while not done.is_set():
            samples.append(dict(time=dt.datetime.now().astimezone().isoformat(),
                                data_free=shutil.disk_usage(root).free,
                                tmp_free=shutil.disk_usage("/tmp").free))
            done.wait(1)
    thread = threading.Thread(target=sample)
    thread.start()
    start = time.monotonic()
    try:
        with (out / "vivado.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=env)
        state.update(exit=result.returncode, wall_seconds=time.monotonic()-start)
    finally:
        done.set()
        thread.join()
        (root / "disk_samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    text = (out / "vivado.log").read_text(errors="replace")
    state.update(end=dt.datetime.now().astimezone().isoformat(), command=command,
                 minimum_data_free=min(x["data_free"] for x in samples),
                 disk_free_after=shutil.disk_usage(root).free,
                 errors=re.findall(r"^ERROR:.*$", text, re.M),
                 critical_warnings=re.findall(r"^CRITICAL WARNING:.*$", text, re.M),
                 ordinary_warnings=len(re.findall(r"^WARNING:", text, re.M)))
    state["status"] = "PASS_checkpoint_export_reopen_only" if completed(state["exit"], text) else "FAIL"
    output = out / "roundtrip.dcp"
    state["output_bytes"] = output.stat().st_size if output.exists() else None
    state["output_crc_pass"] = None
    if output.exists():
        try:
            with zipfile.ZipFile(output) as archive:
                state["output_crc_pass"] = archive.testzip() is None
        except (zipfile.BadZipFile, OSError) as error:
            state["checkpoint_error"] = repr(error)
        if state["output_crc_pass"] is not True:
            state["status"] = "FAIL"
    (root / "status.json").write_text(json.dumps(state, indent=2) + "\n")


def launch():
    import run_srnn_resident_remote as m
    remote_source, destination = m.REMOTE + "/" + SOURCE, m.REMOTE + "/" + NAME
    code = r'''
import datetime,json,pathlib,shutil,zipfile
source=pathlib.Path(SOURCE);destination=pathlib.Path(DESTINATION)
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','xelab','xsim') or comm.startswith('Vtb_'):active.append([p.name,comm])
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
if active: raise RuntimeError('active EDA: '+repr(active))
with zipfile.ZipFile(source) as archive:
 if archive.testzip() is not None: raise ValueError('source CRC failed')
 raw=sum(i.file_size for i in archive.infolist())
budget=2*raw+source.stat().st_size+268435456
free=shutil.disk_usage(destination.parent).free
if free < budget: raise RuntimeError('insufficient estimated transient space')
destination.mkdir(exist_ok=False)
record=dict(time=datetime.datetime.now().astimezone().isoformat(),source=str(source),source_bytes=source.stat().st_size,
 source_uncompressed_bytes=raw,source_crc_pass=True,free_bytes=free,estimated_space_budget=budget,
 budget_boundary='Estimate: two expanded checkpoint payloads plus one compressed output plus256MiB. Not a guaranteed tool bound or historical ENOSPC proof.',active_eda=active)
(destination/'preflight.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''
    client = m.connect()
    try:
        print(m.run(client, "python3 -c " + shlex.quote(
            "SOURCE=" + repr(remote_source) + "\nDESTINATION=" + repr(destination) + "\n" + code)))
        with client.open_sftp() as sftp:
            for relative in (TCL, WORKER):
                m.upload(sftp, m.local_path(m.ROOT / relative), destination + "/" + relative)
        shell = "cd " + shlex.quote(destination) + " && exec python3 " + WORKER + " worker --source " + shlex.quote(remote_source)
        print(m.run(client, "screen -dmS snn_checkpoint_probe bash -lc " + shlex.quote(shell)))
        print(json.dumps(dict(name=NAME, remote=destination, screen="snn_checkpoint_probe", launched=True)))
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "worker"))
    parser.add_argument("--source")
    args = parser.parse_args()
    if args.action == "launch":
        launch()
    elif args.source:
        worker(args.source)
    else:
        parser.error("worker requires --source")
