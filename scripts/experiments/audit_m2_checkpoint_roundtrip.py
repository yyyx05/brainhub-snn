"""Check fetched export-probe evidence without deriving a timing conclusion."""
import argparse
import json
from pathlib import Path
import re
import shlex
import zipfile

from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, local_path
from run_m2_checkpoint_roundtrip import NAME, TCL, WORKER, completed
from audit_m2_compact_implementation import resources


def main(name=NAME):
    local = Path(local_path(ROOT / RESULT / name))
    client = connect()
    try:
        code = r'''
import datetime,json,pathlib,shutil,subprocess
r=pathlib.Path(ROOT)
state=json.loads((r/'status.json').read_text())
if state['status']=='running':raise ValueError('still running')
files={p.relative_to(r).as_posix():dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
 for p in r.rglob('*') if p.is_file() and not set(p.relative_to(r).parts)&{'obj','elab_obj','__pycache__','.Xil'}}
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','xelab','xsim') or comm.startswith('Vtb_'):active.append([p.name,comm])
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
k=subprocess.run(['dmesg','--ctime'],capture_output=True,text=True)
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),files=files,
 active_eda=active,data_free=shutil.disk_usage(r).free,tmp_free=shutil.disk_usage('/tmp').free,
 dmesg_exit=k.returncode,dmesg_error=k.stderr)))
'''
        observation = json.loads(run(client, "python3 -c " + shlex.quote(
            "ROOT=" + repr(REMOTE + "/" + name) + "\n" + code)))
    finally:
        client.close()
    assert not observation["active_eda"]
    for path, info in observation["files"].items():
        assert (local / path).is_file() and (local / path).stat().st_size == info["size"], path
    assert not list(local.rglob("*.part"))
    actual_sources = [TCL, WORKER]
    if name == "m2_checkpoint_roundtrip_r2":
        actual_sources.append("scripts/experiments/run_m2_checkpoint_roundtrip_scratch.py")
    for path in actual_sources:
        assert (local / path).read_bytes() == Path(local_path(ROOT / path)).read_bytes()
    state = json.loads((local / "status.json").read_text())
    log = (local / "output/vivado.log").read_text(errors="replace")
    assert completed(state["exit"], log) == state["status"].startswith("PASS")
    samples = json.loads((local / "disk_samples.json").read_text())
    assert state["minimum_data_free"] == min(x["data_free"] for x in samples)
    output = local / "output/roundtrip.dcp"
    crc, members = None, None
    if output.is_file():
        with zipfile.ZipFile(output) as archive:
            crc, members = archive.testzip() is None, len(archive.infolist())
        assert crc == state["output_crc_pass"]
    snapshots, utilization = {}, {}
    if state["status"].startswith("PASS"):
        assert crc is True
        for stage in ("input", "reopened"):
            text = (local / "output" / (stage + "_snapshot.txt")).read_text()
            snapshots[stage] = dict(line.split("=", 1) for line in text.splitlines())
            utilization[stage] = resources((local / "output" / (stage + "_utilization.rpt")).read_text())
            assert "No valid timing exceptions found." in (local / "output" / (stage + "_exceptions.rpt")).read_text()
        assert snapshots["input"] == snapshots["reopened"] and utilization["input"] == utilization["reopened"]
    audit = dict(status=state["status"], raw_status=state, raw_file_count=len(observation["files"]),
        raw_bytes=sum(i["size"] for i in observation["files"].values()), all_remote_sizes_equal=True,
        partial_files=[], actual_sources_byte_equal=actual_sources, output_crc_pass=crc, output_zip_members=members,
        snapshots=snapshots, utilization=utilization, disk_sample_count=len(samples),
        sampled_minimum_data_free=state["minimum_data_free"], sampling_period_seconds=1,
        sampled_minimum_tmp_free=min(x["tmp_free"] for x in samples),
        observed_peak_additional_tmp_bytes=samples[0]["tmp_free"]-min(x["tmp_free"] for x in samples),
        observed_peak_additional_data_bytes=samples[0]["data_free"]-state["minimum_data_free"],
        errors=re.findall(r"^ERROR:.*$", log, re.M), critical_warnings=re.findall(r"^CRITICAL WARNING:.*$", log, re.M),
        ordinary_warnings=len(re.findall(r"^WARNING:", log, re.M)),
        oom_or_disk_error_lines=re.findall(r"^.*(?:Out of memory|\bOOM\b|\bKilled\b|ENOSPC|No space left|Disk quota exceeded).*$", log, re.M | re.I),
        remote_observation=observation,
        boundary="This opens and exports an existing placed checkpoint only, with no synthesis/place/route/physopt. Snapshot/resource equality is not full formal equivalence. One-second disk sampling may miss shorter peaks. Success establishes export/reopen at observed free space, not the cause of the earlier final-route export failure, nor a recovered final routed DCP or new QoR/STA. Production and historical evidence unchanged.")
    (local / "terminal_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in audit.items() if k not in ("remote_observation", "snapshots", "raw_status")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=(NAME, "m2_checkpoint_roundtrip_r2"), default=NAME)
    main(parser.parse_args().name)
