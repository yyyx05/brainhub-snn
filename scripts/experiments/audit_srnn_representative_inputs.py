"""Audit the fetched E5 run and report its measured intervals, without rerunning RTL."""
import collections
import csv
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys

sys.dont_write_bytecode = True
from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, local_path
from run_srnn_representative_inputs import NAME, EXPERIMENT


def main():
    archive = Path(local_path(ROOT / RESULT / NAME))
    prepared = Path(local_path(ROOT / EXPERIMENT))
    state = json.loads((archive / "status.json").read_text())
    if not state.get("end") or not state["status"].startswith("PASS_"):
        raise ValueError("complete natural PASS archive required; original state is not changed")
    if any(state.get(stage, {}).get("exit") != 0 for stage in ("build", "run_off", "run_on")):
        raise ValueError("actual stage exit is not zero")
    if (archive / "terminal_audit.json").exists():
        raise FileExistsError("existing audit is retained; do not repeat an accepted audit")

    client = connect()
    try:
        code = """import datetime,json,pathlib,shutil
p=pathlib.Path(ROOT);scratch=pathlib.Path(SCRATCH);active=[]
skip={'obj','elab_obj','__pycache__','.Xil'}
def inventory(root):
 return {q.relative_to(root).as_posix():q.stat().st_size for q in root.rglob('*') if q.is_file() and not set(q.relative_to(root).parts)&skip}
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(q/'comm').read_text().strip()
  cmd=(q/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):
   active.append([q.name,comm,cmd])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),files=inventory(p),
 scratch_noncache_files=inventory(scratch),active_eda=active,
 data_free=shutil.disk_usage(p).free,tmp_free=shutil.disk_usage('/tmp').free)))
"""
        observation = json.loads(run(client, "python3 -c " + shlex.quote(
            "ROOT=" + repr(REMOTE + "/" + NAME) + "\nSCRATCH=" + repr(state["scratch"]) + "\n" + code)))
    finally:
        client.close()
    if observation["active_eda"]:
        raise ValueError("EDA remains active; do not overlap a new run")
    if observation["scratch_noncache_files"]:
        raise ValueError("scratch non-cache evidence still needs fetch")
    files = observation.pop("files")
    if list(archive.rglob("*.part")) or any(p.endswith(".part") for p in files):
        raise ValueError("partial archive")
    for relative, size in files.items():
        path = archive / relative
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError("raw file size mismatch: " + relative)

    inputs = [p for p in prepared.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for path in inputs:
        relative = path.relative_to(prepared)
        if (archive / relative).read_bytes() != path.read_bytes():
            raise ValueError("prepared input mismatch: " + str(relative))
    candidate = Path(local_path(ROOT / "verification/experiments/scnn_control_payload/candidate_r4"))
    sources = [p for p in (candidate / "rtl").rglob("*") if p.is_file()]
    if len(inputs) != 353 or len(sources) != 109 or sum(p.suffix == ".sv" for p in sources) != 108:
        raise ValueError("actual input/source set differs")
    for path in sources:
        if path.read_bytes() != (archive / path.relative_to(candidate)).read_bytes():
            raise ValueError("payload-A source mismatch: " + str(path))
    if (archive / "launch_worker.py").read_bytes() != (ROOT / "scripts/experiments/run_srnn_representative_inputs.py").read_bytes():
        raise ValueError("actual worker differs from local preparation script")
    spec = importlib.util.spec_from_file_location("archived_e5_worker", archive / "launch_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    info = json.loads((archive / "multi_fixture/reference.json").read_text())
    selected = json.loads((archive / "selection.json").read_text())["selected"]
    checks = json.loads((archive / "checks.json").read_text())
    diagnostics = {}
    for stage in ("build", "run_off", "run_on"):
        text = (archive / (stage + ".log")).read_text(errors="replace")
        warnings = collections.Counter(re.findall(r"^%Warning-([^:]+):", text, re.M))
        diagnostics[stage] = dict(
            errors=len(re.findall(r"^%Error", text, re.M)),
            fatal=bool(re.search(r"%Fatal|Assertion failed|\bFATAL\b", text)),
            ordinary_warnings=sum(warnings.values()), warning_categories=dict(sorted(warnings.items())),
            structural_warnings=[k for k in warnings if k in ("LATCH", "MULTIDRIVEN", "UNOPTFLAT")],
            oom_or_killed=bool(re.search(r"out of memory|\bOOM\b|\bKilled\b", text, re.I)),
            disk_exhaustion=bool(re.search(r"no space left on device", text, re.I)))
        if stage != "build":
            arm = stage.removeprefix("run_")
            parsed = worker.parse_log(text, arm, info)
            if parsed != checks[arm] or parsed != state[arm]:
                raise ValueError("archived worker reparse differs from checks/status: " + arm)
    if any(d["errors"] or d["fatal"] or d["structural_warnings"] for d in diagnostics.values()):
        raise ValueError("tool diagnostic; original archive retained")
    if diagnostics["build"]["ordinary_warnings"] != state["build_warnings"]["ordinary"]:
        raise ValueError("build warning count mismatch")

    rows = []
    aggregate = {}
    for arm in ("off", "on"):
        result = checks[arm]
        aggregate[arm] = dict(
            total_cycles=result["total"]["cycles"],
            initialization_including_reset_cycles=result["initialization_including_reset_cycles"],
            state_setup_cycles=result["state_setup_cycles"], service_cycles=result["service_cycles"],
            all_sequences_per_second_15ns_projection=result["all_sequences_per_second_15ns_projection"],
            warm_sequences_per_second_with_setup_15ns_projection=result["warm_sequences_per_second_with_setup_15ns_projection"],
            totals={key: sum(r[key] for r in result["sequences"]) for key in
                    ("bone", "hidden", "logits", "store", "code", "residual", "commits")})
        for item, sample in zip(result["sequences"], selected):
            rows.append(dict(
                arm=arm, dataset_index=sample["dataset_index"], native_steps=item["steps"],
                input_occupied_fraction=sample["occupied_fraction"],
                cold_weight_context=int(item["sequence"] == 0),
                service_cycles=item["cycles"], state_setup_cycles=item["state_setup_cycles"],
                initialization_cycles=result["initialization_including_reset_cycles"] if item["sequence"] == 0 else 0,
                service_plus_setup_ms_15ns=(item["cycles"] + item["state_setup_cycles"]) * 15e-6,
                code_writes=item["code"], residual_writes=item["residual"],
                commits=item["commits"], cumulative_wout_loads=item["wout_loads"]))
    report = dict(
        status="PASS_representative_A_eight_sequence_archive_audit",
        raw_files=len(files), raw_bytes=sum(files.values()), all_raw_sizes_match=True, no_part=True,
        prepared_input_files_byte_equal=len(inputs), prepared_input_bytes=sum(p.stat().st_size for p in inputs),
        candidate_source_files_byte_equal=len(sources), archived_worker_strict_reparse_matches=True,
        original_status=state["status"], start=state["start"], end=state["end"],
        stages={k: state[k] for k in ("build", "run_off", "run_on")},
        diagnostics=diagnostics, remote_observation=observation, aggregate=aggregate,
        dcp_crc="not applicable: no DCP", scope=worker.SCOPE,
        boundary="No new speedup comparison. Eight purposive first-64-pool samples, not IID/full Test or stable P95. "
            "Learning state persists in ascending index order. Fixed15ns projections, not DDR/board throughput. "
            "Absence of OOM/Killed in logs is not kernel evidence. No cleanup, production merge or old table replacement.")
    with (archive / "sequence_measurements.csv").open("x", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (archive / "terminal_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
