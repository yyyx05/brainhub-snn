"""Repeat only checkpoint export/reopen with isolated system-disk scratch space."""
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

from run_m2_checkpoint_roundtrip import SOURCE, TCL, WORKER, VIVADO, completed

NAME = "m2_checkpoint_roundtrip_r2"
SCRATCH = "/tmp/snn_m2_checkpoint_roundtrip_r2"
THIS_WORKER = "scripts/experiments/run_m2_checkpoint_roundtrip_scratch.py"


def worker(source):
    root, scratch = Path.cwd(), Path(SCRATCH)
    scratch.mkdir(exist_ok=False)
    out, temporary = root / "output", scratch / "output"
    out.mkdir()
    state = dict(status="running", start=dt.datetime.now().astimezone().isoformat(), source=source,
                 scratch=str(scratch), scope="Same placed checkpoint export/reopen only; scratch filesystem changed.")
    def save():
        (root / "status.json").write_text(json.dumps(state, indent=2) + "\n")
    save()
    env = os.environ.copy()
    install = str(Path(VIVADO).parents[1])
    env.update(XILINX_VIVADO=install, PATH=install+"/bin:"+env.get("PATH", ""),
               LD_LIBRARY_PATH=install+"/lib/lnx64.o/SuSE:"+env.get("LD_LIBRARY_PATH", ""),
               TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch))
    if Path("/lib/x86_64-linux-gnu/libudev.so.1").is_file():
        env["LD_PRELOAD"] = "/lib/x86_64-linux-gnu/libudev.so.1"
    command = [VIVADO, "-mode", "batch", "-nojournal", "-nolog", "-source", str(root / TCL),
               "-tclargs", source, str(temporary)]
    samples, done = [], threading.Event()
    def sample():
        while not done.is_set():
            samples.append(dict(time=dt.datetime.now().astimezone().isoformat(),
                                data_free=shutil.disk_usage(root).free, tmp_free=shutil.disk_usage(scratch).free))
            done.wait(1)
    thread = threading.Thread(target=sample)
    thread.start()
    started = time.monotonic()
    try:
        with (out / "vivado.log").open("w") as log:
            result = subprocess.run(command, cwd=scratch, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
        state.update(exit=result.returncode, wall_seconds=time.monotonic()-started)
    finally:
        done.set()
        thread.join()
        (root / "disk_samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    text = (out / "vivado.log").read_text(errors="replace")
    state.update(command=command, minimum_data_free=min(x["data_free"] for x in samples),
                 minimum_tmp_free=min(x["tmp_free"] for x in samples),
                 errors=re.findall(r"^ERROR:.*$", text, re.M),
                 critical_warnings=re.findall(r"^CRITICAL WARNING:.*$", text, re.M),
                 ordinary_warnings=len(re.findall(r"^WARNING:", text, re.M)))
    retained = [p for p in scratch.rglob("*") if p.is_file() and ".Xil" not in p.relative_to(scratch).parts]
    required = sum(p.stat().st_size for p in retained)
    if shutil.disk_usage(root).free < required + 268435456:
        state.update(status="FAIL_archive_space", scratch_bytes=required)
        save()
        return
    inventory = []
    for path in retained:
        if path.is_symlink():
            raise ValueError("unexpected scratch symlink: " + str(path))
        relative = path.relative_to(scratch)
        target = root / relative
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        assert target.stat().st_size == path.stat().st_size
        inventory.append(dict(source=str(path), relative=relative.as_posix(), bytes=path.stat().st_size))
    (root / "scratch_archive.json").write_text(json.dumps(inventory, indent=2) + "\n")
    output = out / "roundtrip.dcp"
    state.update(output_bytes=output.stat().st_size if output.exists() else None, output_crc_pass=None)
    if output.exists():
        try:
            with zipfile.ZipFile(output) as archive:
                state["output_crc_pass"] = archive.testzip() is None
        except (zipfile.BadZipFile, OSError) as error:
            state["checkpoint_error"] = repr(error)
    state.update(status="PASS_checkpoint_export_reopen_only" if completed(state["exit"], text)
                 and state["output_crc_pass"] is True else "FAIL",
                 end=dt.datetime.now().astimezone().isoformat(), disk_free_after=shutil.disk_usage(root).free,
                 tmp_free_after=shutil.disk_usage(scratch).free, scratch_preserved=True)
    save()


def launch():
    import run_srnn_resident_remote as m
    destination, source = m.REMOTE + "/" + NAME, m.REMOTE + "/" + SOURCE
    code = r'''
import datetime,json,pathlib,shutil,zipfile
target=pathlib.Path(TARGET);source=pathlib.Path(SOURCE);scratch=pathlib.Path(SCRATCH)
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','xelab','xsim') or comm.startswith('Vtb_'):active.append([p.name,comm])
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
if active:raise RuntimeError('active EDA: '+repr(active))
if scratch.exists() or target.exists():raise FileExistsError('probe destination exists')
data_free=shutil.disk_usage(target.parent).free;tmp_free=shutil.disk_usage(scratch.parent).free
if tmp_free<4*1024**3 or data_free<source.stat().st_size+268435456:raise RuntimeError('insufficient scratch/archive capacity')
with zipfile.ZipFile(source) as archive:
 if archive.testzip() is not None:raise ValueError('source CRC failed')
target.mkdir()
record=dict(time=datetime.datetime.now().astimezone().isoformat(),source=str(source),source_bytes=source.stat().st_size,
 source_crc_pass=True,data_free=data_free,tmp_free=tmp_free,scratch=str(scratch),active_eda=active,
 boundary='Same Tcl/source/2threads. Tool cwd, temporary environment and output are on system-disk scratch. All noncache output copied back; no originals overwritten or deleted.')
(target/'preflight.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''
    client = m.connect()
    try:
        print(m.run(client, "python3 -c " + shlex.quote(
            "TARGET="+repr(destination)+"\nSOURCE="+repr(source)+"\nSCRATCH="+repr(SCRATCH)+"\n"+code)))
        with client.open_sftp() as sftp:
            for path in (TCL, WORKER, THIS_WORKER):
                m.upload(sftp, m.local_path(m.ROOT / path), destination + "/" + path)
        command = "cd " + shlex.quote(destination) + " && exec python3 " + THIS_WORKER + " worker --source " + shlex.quote(source)
        m.run(client, "screen -dmS snn_checkpoint_probe bash -lc " + shlex.quote(command))
        print(json.dumps(dict(name=NAME, remote=destination, scratch=SCRATCH, launched=True)))
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
