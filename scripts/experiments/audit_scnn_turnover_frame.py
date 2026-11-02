"""Audit only fully fetched, naturally terminal raw-frame A/B experiments."""
import argparse
import json
from pathlib import Path
import re
import shlex

from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, local_path, run
from run_scnn_turnover_frame import SOURCES, parse_log


def audit(name):
    if name not in ("scnn_turnover_frame_r1","scnn_turnover_frame_r2"):
        raise ValueError("explicit prepared run required")
    revision=name.rsplit("_",1)[1]
    prepared=Path(local_path(ROOT/"verification/experiments/scnn_forward_turnover"/("frame_"+revision)))
    archive=Path(local_path(ROOT/RESULT/name))
    state=json.loads((archive/"status.json").read_text())
    if not state.get("end") or state["status"].endswith("_running"):
        raise ValueError("not terminal; do not audit an active run")
    client=connect()
    try:
        probe='''import datetime,json,pathlib,shutil
p=pathlib.Path(ROOT)
files={q.relative_to(p).as_posix():q.stat().st_size for q in p.rglob('*') if q.is_file() and not set(q.relative_to(p).parts).intersection({'obj','elab_obj','__pycache__','.Xil'})}
print(json.dumps(dict(observed=datetime.datetime.now().astimezone().isoformat(),files=files,data_free=shutil.disk_usage(p).free,tmp_free=shutil.disk_usage('/tmp').free)))
'''
        remote=json.loads(run(client,"python3 -c "+shlex.quote("ROOT="+repr(REMOTE+"/"+name)+"\n"+probe)))
    finally:client.close()
    files=remote.pop("files")
    if any(p.endswith(".part") for p in files) or list(archive.rglob("*.part")):
        raise ValueError("partial archive")
    for rel,size in files.items():
        if not (archive/rel).is_file() or (archive/rel).stat().st_size != size:
            raise ValueError("raw file size mismatch: "+rel)
    inputs=[]
    for path in prepared.rglob("*"):
        if not path.is_file() or path.name=="launch_worker.py":continue
        rel=path.relative_to(prepared).as_posix()
        if (archive/rel).read_bytes()!=path.read_bytes():
            raise ValueError("prepared/actual byte mismatch: "+rel)
        inputs.append(rel)
    worker=prepared/"launch_worker.py"
    if (archive/"run_scnn_turnover_frame.py").read_bytes()!=Path(local_path(worker)).read_bytes():
        raise ValueError("actual worker byte mismatch")
    for helper in ("prepare_scnn_schedule_probe.py","prepare_scnn_resident_compatibility.py"):
        if (archive/helper).read_bytes() != (ROOT/"scripts/experiments"/helper).read_bytes():
            raise ValueError("actual helper byte mismatch")
    source=Path(local_path(ROOT/RESULT/"m2_compact_synth_r2"))
    source_paths=sorted(p.relative_to(source).as_posix() for p in (source/"rtl").rglob("*") if p.is_file())
    if len(source_paths)!=109:raise ValueError("compact source set incomplete")
    changed=[]
    for rel in source_paths:
        if (archive/"baseline"/rel).read_bytes()!=(source/rel).read_bytes():raise ValueError("baseline source changed")
        if (archive/"candidate"/rel).read_bytes()!=(source/rel).read_bytes():changed.append(rel)
    if changed!=sorted(SOURCES):raise ValueError("unexpected overlay")
    logs={}
    for arm in ("baseline","candidate"):
        for stage in ("build","run"):
            path=archive/arm/(stage+".log")
            if path.exists():
                text=path.read_text(errors="replace")
                logs[arm+"_"+stage]=dict(errors=len(re.findall(r"^%Error: (?!Exiting)",text,re.M)),
                    error_lines_including_summary=len(re.findall(r"^%Error:",text,re.M)),
                    ordinary_warnings=len(re.findall(r"^%Warning-",text,re.M)),
                    structural_warnings=re.findall(r"^%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT).*",text,re.M),
                    oom_or_killed=bool(re.search(r"out of memory|\bOOM\b|\bKilled\b",text,re.I)))
    success=state["status"].startswith("PASS")
    if success:
        checks={}
        for arm in ("baseline","candidate"):
            if any(state[arm][s]["exit"]!=0 for s in ("build","run")):raise ValueError("nonzero actual exit")
            checks[arm]=parse_log((archive/arm/"run.log").read_text(),json.loads((archive/arm/"reference.json").read_text()))
            if checks[arm]!=json.loads((archive/arm/"checks.json").read_text()):raise ValueError("strict reparse mismatch")
        if checks["baseline"]["tables"]["handshake"]!=checks["candidate"]["tables"]["handshake"]:
            raise ValueError("paired transactions changed")
        if any(r["errors"] or r["structural_warnings"] for r in logs.values()):raise ValueError("build diagnostics")
    result=dict(status="PASS_raw_first_frame_pair_archive_audit" if success else "TERMINAL_FAIL_original_preserved",
        original_status=state,raw_files=len(files),raw_bytes=sum(files.values()),
        all_raw_sizes_match=True,no_part=True,prepared_input_files_byte_equal=len(inputs),
        actual_worker_and_helpers_byte_equal=True,source_files_per_arm=len(source_paths),changed_rtl=changed,
        diagnostics=logs,remote_observation=remote,
        boundary="No simulation or EDA rerun. No DCP in this Verilator experiment, CRC not applicable. Raw status/logs remain unchanged. Successful simulation is not mapped resource or STA proof.")
    (archive/"terminal_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("status","raw_files","raw_bytes","prepared_input_files_byte_equal","diagnostics")},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name",default="scnn_turnover_frame_r2")
    audit(parser.parse_args().name)
