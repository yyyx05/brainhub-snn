"""Implement experiment A's own DCP using the unchanged turnover R2 recipe."""
import argparse
import json
from pathlib import Path
import shlex
import shutil

NAME = "scnn_cfg_payload_core_impl_r1"
SCRATCH = "/tmp/snn_" + NAME
SYNTHESIS = "scnn_cfg_payload_core_synth_r1"
SOURCE = SYNTHESIS + "/server_results/final_composition_synth/post_synth.dcp"
EXPERIMENT = Path("verification/experiments/scnn_control_payload/core_implementation_r1")
RECIPE = "scnn_turnover_core_impl_r2"
INPUTS = ("original_impl.tcl", "implementation.tcl", "implementation_changes.diff",
          "preparation.json", "launch_worker.py", "run_scnn_turnover_implementation.py")


def worker(source):
    import run_scnn_turnover_implementation as original
    original.NAME = NAME
    original.SCRATCH = SCRATCH
    original.worker(source)


def prepare():
    from run_srnn_resident_remote import ROOT, RESULT, local_path
    base = Path(local_path(ROOT / RESULT))
    audit = json.loads((base / SYNTHESIS / "terminal_audit.json").read_text())
    if audit["status"] != "PASS_cfg_payload_synthesis_evidence_not_routed_STA":
        raise ValueError("experiment A synthesis evidence incomplete")
    out = Path(local_path(ROOT / EXPERIMENT))
    out.mkdir(parents=True, exist_ok=False)
    for name in INPUTS[:3]:
        shutil.copy2(base / RECIPE / name, out / name)
    shutil.copy2(Path(__file__), out / "launch_worker.py")
    shutil.copy2(base / RECIPE / "launch_worker.py", out / "run_scnn_turnover_implementation.py")
    record = dict(status="PREPARED_NOT_IMPLEMENTED", source=SOURCE, source_checkpoint=audit["dcp"],
                  original_recipe=RECIPE + "/implementation.tcl", recipe_byte_unchanged=True,
                  recipe="opt_design; place_design ExtraNetDelay_high; phys_opt_design ExploreWithHoldFix; route_design AggressiveExplore; phys_opt_design Explore",
                  rationale="A actual cycle-miter passed; own synthesis is -41LUT/+4FF with unchanged BRAM/DSP. Synthesis setup failures783->290, WNS-0.828->-0.800ns, but both top100 sets end at publisher. Routing must measure A rather than infer final margin or removal of the learner path.",
                  boundary="A only, no B/model-key. Own108SV+ROM/512/15ns/8threads DCP. Same optimization, constraints, protections, export/reopen and all-log CW rejection. Old R2 NOT_ACCEPTED remains. No new throughput, power, bitstream or production adoption. All scratch preserved.")
    (out / "preparation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(str(out))


def launch():
    from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, upload, local_path
    prepared = Path(local_path(ROOT / EXPERIMENT))
    record = json.loads((prepared / "preparation.json").read_text())
    destination = REMOTE + "/" + NAME
    source = REMOTE + "/" + SOURCE
    probe = f'''
import datetime,json,pathlib,shutil,zipfile
p=pathlib.Path({destination!r});scratch=pathlib.Path({SCRATCH!r});source=pathlib.Path({source!r})
if p.exists() or scratch.exists():raise FileExistsError('unique implementation destination exists')
active=[]
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(q/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):active.append([q.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
if active:raise RuntimeError('another EDA active: '+repr(active))
data=shutil.disk_usage(p.parent).free;temp=shutil.disk_usage('/tmp').free
if data<768*1024**2 or temp<6*1024**3:raise RuntimeError('insufficient archive/scratch margin')
with zipfile.ZipFile(source) as z:crc=z.testzip() is None
if source.stat().st_size!={record['source_checkpoint']['bytes']!r} or not crc:raise ValueError('own source DCP size/CRC mismatch')
record=dict(time=datetime.datetime.now().astimezone().isoformat(),active_eda=active,data_free=data,tmp_free=temp,
 source=str(source),source_bytes=source.stat().st_size,source_crc_pass=crc,scratch=str(scratch),
 boundary='A own DCP, unchanged turnover R2 optimization/15ns/protection/export-reopen/all-log CW rule. Existing768MiB public/6GiB scratch margins are not peak guarantees; worker checks actual archive bytes before copy. No power/bitstream/production adoption. No cleanup.')
for name in ('cpu.max','memory.max'):
 q=pathlib.Path('/sys/fs/cgroup')/name;record[name]=q.read_text().strip() if q.exists() else None
p.mkdir();(p/'preflight.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
    client = connect()
    try:
        before = json.loads(run(client, "python3 -c " + shlex.quote(probe)))
        with client.open_sftp() as sftp:
            for rel in INPUTS:
                upload(sftp, str(prepared / rel), destination + "/" + rel)
                with sftp.open(destination + "/" + rel, "rb") as stream:
                    if stream.read() != (prepared / rel).read_bytes():
                        raise ValueError("actual input mismatch: " + rel)
        command = "cd " + shlex.quote(destination) + " && exec python3 launch_worker.py worker --source " + shlex.quote(source)
        run(client, "screen -dmS snn_cfg_payload_impl bash -lc " + shlex.quote(command))
        before.update(launched=True, remote=destination, screen="snn_cfg_payload_impl", prepared_inputs_byte_equal=len(INPUTS))
        out = Path(local_path(ROOT / RESULT / NAME))
        out.mkdir(parents=True, exist_ok=True)
        (out / "launch_observation.json").write_text(json.dumps(before, indent=2) + "\n")
        print(json.dumps(before, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "worker"))
    parser.add_argument("--source")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "launch":
        launch()
    else:
        if not args.source:
            parser.error("worker requires --source")
        worker(args.source)
