"""Synthesize only the new, functionally measured SCNN row-retirement publisher candidate."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import sys

sys.dont_write_bytecode = True
from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, upload, local_path

NAME = "scnn_row_turnover_synth_r1"
CORE = "scnn_row_turnover_core_r1"
EXPERIMENT = Path("verification/experiments") / NAME
SCRATCH = "/tmp/snn_" + NAME
MODULE = Path("rtl/learning/unified_scnn_classifier_cache_refill_publisher_v1.sv")


def worker_source(original):
    replacements = {
        "NAME = 'final_raw_source_synth_r1'": "NAME = " + repr(NAME),
        "for arm in ('baseline', 'candidate'):": "for arm in ('candidate',):",
        "PASS_paired_raw_source_synthesis_not_routed_STA":
            "PASS_scnn_row_turnover_candidate_synthesis_not_routed_STA",
    }
    for old, new in replacements.items():
        # The second arm loop belongs to the unused original launcher.
        expected = 2 if old.startswith("for arm") else 1
        if original.count(old) != expected:
            raise ValueError("unexpected accepted synthesis runner: " + old)
        original = original.replace(old, new)
    return original


def load_worker(path):
    spec = importlib.util.spec_from_file_location("executed_scnn_row_turnover_synthesis", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def prepare():
    root = Path(local_path(ROOT))
    evidence = root / RESULT
    core = evidence / CORE
    accepted = json.loads((core / "terminal_audit.json").read_text())
    if accepted["status"] != "PASS_SCNN_row_turnover_archive":
        raise ValueError("new core archive has not passed")
    prior = root / "verification/experiments/final_raw_source_composition_r1/candidate"
    tested = root / "verification/experiments" / CORE
    rtl = [p for p in (prior / "rtl").rglob("*") if p.is_file()]
    if len(rtl) != 113:
        raise ValueError("actual wrapper boundary")
    changes = [p.relative_to(prior) for p in rtl
               if p.read_bytes() != (tested / p.relative_to(prior)).read_bytes()]
    if changes != [MODULE]:
        raise ValueError("not a one-module candidate")
    for p in rtl:
        rel = p.relative_to(prior)
        if (tested / rel).read_bytes() != (core / rel).read_bytes():
            raise ValueError("prepared/actually executed RTL differs: " + str(rel))
    output = root / EXPERIMENT
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(prior, output / "candidate")
    shutil.copy2(tested / MODULE, output / "candidate" / MODULE)
    original = Path(__file__).with_name("run_final_raw_source_synthesis.py")
    shutil.copy2(original, output / "original_synthesis_worker.py")
    (output / "worker.py").write_text(worker_source(original.read_text()), newline="\n")
    shutil.copy2(__file__, output / "prepare_audit_runner.py")
    shutil.copy2(core / "terminal_audit.json", output / "functional_acceptance.json")
    record = dict(status="PREPARED_NOT_SYNTHESIZED", measured_core=CORE,
        changed_rtl=[MODULE.as_posix()], rtl_items=113,
        prior_candidate="final_raw_source_synth_r1/candidate",
        boundary="New candidate only; old candidate is an already measured comparator, not rerun. "
            "Exact accepted synthesis Tcl/shell, wrapper ports, numerical widths, 15ns, M2=512, "
            "XPM, Default/rebuilt and protections retained. No route or inherited physical timing.")
    (output / "preparation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def observe(client, create, reserve):
    values = dict(BASE=REMOTE, NAME=NAME, SCRATCH=SCRATCH, CREATE=create, RESERVE=reserve)
    source = "\n".join(k + "=" + repr(v) for k, v in values.items()) + """
import datetime,json,pathlib,shutil
base=pathlib.Path(BASE); target=base/NAME; scratch=pathlib.Path(SCRATCH)
if CREATE and (target.exists() or scratch.exists()): raise FileExistsError('run exists')
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  n=(p/'comm').read_text().strip()
  if n in ('vivado','verilator_bin','xelab','xsim','Vscgraph') or n.startswith('Vtb_'):
   active.append([p.name,n])
 except (OSError,ProcessLookupError): pass
data=shutil.disk_usage(base).free; tmp=shutil.disk_usage('/tmp').free
if active or data<RESERVE or tmp<4*1024**3:
 raise RuntimeError('existing idle/synthesis reserve: '+repr((active,data,tmp)))
if CREATE: target.mkdir()
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),
 active_eda=active,data_free=data,tmp_free=tmp)))
"""
    return json.loads(run(client, "python3 -c " + shlex.quote(source)))


def launch():
    root = Path(local_path(ROOT))
    prepared = root / EXPERIMENT
    files = sorted(p for p in prepared.rglob("*") if p.is_file())
    if not files or not (prepared / "preparation.json").is_file():
        raise ValueError("missing prepared candidate")
    payload = sum(p.stat().st_size for p in files)
    client = connect()
    try:
        before = observe(client, True, payload + 512 * 1024**2)
        destination = REMOTE + "/" + NAME
        with client.open_sftp() as sftp:
            for p in files:
                target = destination + "/" + p.relative_to(prepared).as_posix()
                upload(sftp, str(p), target)
                with sftp.open(target, "rb") as stream:
                    if stream.read() != p.read_bytes():
                        raise ValueError("uploaded input differs: " + str(p))
        at_start = observe(client, False, 512 * 1024**2)
        shell = "cd " + shlex.quote(destination) + " && exec python3 worker.py worker"
        run(client, "screen -dmS snn_" + NAME + " bash -lc " + shlex.quote(shell))
        record = dict(launched=True, before_copy=before, before_start=at_start,
            prepared_files=len(files), prepared_bytes=payload, all_uploaded_bytes_equal=True,
            remote=destination, scratch=SCRATCH, screen="snn_" + NAME,
            boundary="One new candidate. Original 512MiB public/4GiB scratch plus actual "
                "archive+128MiB reserves retained; timeout7200 retains PID; no concurrent EDA.")
        output = root / RESULT / NAME
        output.mkdir(parents=True, exist_ok=False)
        (output / "launch_observation.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
    finally:
        client.close()


def audit():
    root = Path(local_path(ROOT))
    archive, prepared = root / RESULT / NAME, root / EXPERIMENT
    output = archive / "terminal_audit.json"
    if output.exists():
        raise FileExistsError("terminal audit already exists")
    state = json.loads((archive / "status.json").read_text())
    if not state.get("end") or state["status"] == "TIMEOUT_PROCESS_RETAINED":
        raise ValueError("natural terminal required")
    # Check the full remote and scratch inventories once, after complete fetch.
    source = "ROOT=" + repr(REMOTE + "/" + NAME) + "\nSCRATCH=" + repr(SCRATCH) + """
import datetime,json,pathlib,shutil
def inventory(root):
 return {p.relative_to(root).as_posix():p.stat().st_size for p in root.rglob('*')
  if p.is_file() and not set(p.relative_to(root).parts)&{'obj','elab_obj','.Xil','__pycache__'}}
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  n=(p/'comm').read_text().strip()
  if n in ('vivado','verilator_bin','xelab','xsim','Vscgraph') or n.startswith('Vtb_'):
   active.append([p.name,n])
 except (OSError,ProcessLookupError): pass
r=pathlib.Path(ROOT); s=pathlib.Path(SCRATCH)
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),
 files=inventory(r),scratch_files=inventory(s),active_eda=active,
 data_free=shutil.disk_usage(r).free,tmp_free=shutil.disk_usage('/tmp').free)))
"""
    client = connect()
    try:
        observation = json.loads(run(client, "python3 -c " + shlex.quote(source)))
    finally:
        client.close()
    if observation["active_eda"] or list(archive.rglob("*.part")):
        raise ValueError("active EDA or incomplete transfer")
    for rel, size in observation["files"].items():
        p = archive / rel
        if rel.endswith(".part") or not p.is_file() or p.stat().st_size != size:
            raise ValueError("incomplete fetched original: " + rel)
    scratch = {r["relative"]: r["bytes"] for r in
               json.loads((archive / "scratch_archive.json").read_text())}
    if scratch != observation["scratch_files"] or any(
            observation["files"].get(k) != v for k, v in scratch.items()):
        raise ValueError("unarchived scratch original")
    inputs = [p for p in prepared.rglob("*") if p.is_file()]
    if any(p.read_bytes() != (archive / p.relative_to(prepared)).read_bytes() for p in inputs):
        raise ValueError("executed input bytes differ")
    if worker_source((archive / "original_synthesis_worker.py").read_text()) != (archive / "worker.py").read_text():
        raise ValueError("worker adaptation differs")
    worker = load_worker(archive / "worker.py")
    stage = state["arms"]["candidate"]
    if stage.get("exit") is None:
        raise ValueError("missing actual exit")
    parsed = worker.classify(archive / "candidate", stage["exit"])
    if any(stage.get(k) != v for k, v in parsed.items()):
        raise ValueError("actual worker strict reparse differs")
    compatibility_checked = 0
    if parsed["status"].startswith("PASS"):
        sv = sorted((archive / "candidate/rtl").rglob("*.sv"))
        compat = archive / "candidate" / worker.REPORT / "vivado_2022_2_sources"
        if len(sv) != 112 or len(list(compat.glob("*.sv"))) != 112:
            raise ValueError("synthesis source boundary")
        for index, p in enumerate(sv):
            expected = p.read_text().replace("`default_nettype none", "`default_nettype wire")
            if expected != (compat / f"{index:03d}_{p.name}").read_text():
                raise ValueError("compatibility source differs")
        compatibility_checked = len(sv)
    result = dict(status="PASS_scnn_row_turnover_synthesis_archive" if parsed["status"].startswith("PASS")
            else "ARCHIVED_failed_scnn_row_turnover_synthesis",
        original_status=state["status"], start=state["start"], end=state["end"],
        raw_files=len(observation["files"]), raw_bytes=sum(observation["files"].values()),
        prepared_inputs_byte_equal=len(inputs), prepared_bytes=sum(p.stat().st_size for p in inputs),
        no_part=True, scratch_missing=[], actual_worker_strict_reparse=True,
        compatibility_sources_checked=compatibility_checked, candidate=parsed,
        remote_observation=observation, boundary="Synthesis only, not routed timing or Fmax.")
    shutil.copy2(__file__, archive / "terminal_audit_runner.py")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "raw_files", "raw_bytes",
        "prepared_inputs_byte_equal", "compatibility_sources_checked")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "audit"))
    globals()[parser.parse_args().action]()
