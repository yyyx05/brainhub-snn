"""Run the accepted three FC batch pairs on the exact final physical wrapper."""
import argparse
import collections
import json
from pathlib import Path
import re
import shlex
import shutil
import sys

from run_final_raw_source_functional import load, once

sys.dont_write_bytecode = True
NAME = "final_fc_functional_r1"
PRIOR = "fc_representative_a_r1"
PRIOR_PREPARED = "verification/experiments/fc_representative_inputs/a_r1"
TB = "tb_fc_profile_measurement.sv"
SCOPE = (
    "Exact final physical candidate wrapper/core, raw source disabled for native FC "
    "group input. Same three cold B16T99 off/on pairs and independent integer references "
    "as fc_representative_a_r1. Final-version regression, not new samples, learning "
    "accuracy, SRNN producer speedup, full Test or board throughput."
)


def roots():
    from run_srnn_resident_remote import ROOT, RESULT, local_path
    root = Path(local_path(ROOT))
    return root, root / "verification/experiments" / NAME, root / RESULT / NAME


def bind_wrapper(text):
    text = text.replace("u_final.", "u_final.u_core.")
    text = once(text, "    unified_h40_final_composition_top_v1 #(",
                "    unified_h40_raw_source_top_v1 #(\n        .RAW_SOURCE_POLICY(1),")
    text = once(text, "    ) u_final (", """    ) u_final (
        .i_raw_source_active(1'b0),
        .i_raw_count_valid(1'b0),
        .i_raw_counts(32'b0),
        .i_fc_srnn_group_feedforward_end(1'b0),""")
    return once(text, "        $finish;",
                '        $display("FINAL_FC_WRAPPER_PASS raw_active=0");\n        $finish;')


def prepare():
    root, prepared, archive = roots()
    prior = root / PRIOR_PREPARED
    if prepared.exists() or archive.exists():
        raise FileExistsError("already prepared/launched")
    shutil.copytree(prior, prepared, ignore=shutil.ignore_patterns("__pycache__"))
    (prepared / TB).write_text(bind_wrapper((prior / TB).read_text()), newline="\n")
    physical = root / "verification/experiments/final_raw_source_composition_r1/candidate/rtl"
    accepted = archive.parent / "final_raw_source_synth_r1/candidate/rtl"
    changed = []
    rtl = [p for p in physical.rglob("*") if p.is_file()]
    if len(rtl) != 113:
        raise ValueError("physical input extent")
    for path in rtl:
        rel = path.relative_to(physical)
        if path.read_bytes() != (accepted / rel).read_bytes():
            raise ValueError("physical source mismatch: " + str(rel))
        dest = prepared / "rtl" / rel
        if not dest.is_file() or dest.read_bytes() != path.read_bytes():
            changed.append(rel.as_posix())
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    sources = sorted(p.relative_to(prepared).as_posix() for p in (prepared / "rtl").rglob("*.sv"))
    (prepared / "sources.f").write_text("".join('"' + p + '"\n' for p in
        sources + [TB, "tb/monitors/supplement_h40_activity_monitor.sv"]), newline="\n")
    shutil.copy2(prior / "launch_worker.py", prepared / "accepted_fc_worker.py")
    shutil.copy2(__file__, prepared / "launch_worker.py")
    shutil.copy2(Path(__file__).with_name("run_final_raw_source_functional.py"),
                 prepared / "run_final_raw_source_functional.py")
    (prepared / "final_preparation.json").write_text(json.dumps(dict(
        scope=SCOPE, physical_rtl_files_byte_equal=113, changed_or_added_rtl=changed,
        original_numerical_assertions="unchanged", prior=PRIOR), indent=2) + "\n")
    print(json.dumps(dict(prepared=str(prepared), changed_or_added_rtl=changed)))


def parser_at(directory):
    sys.path.insert(0, str(directory / "scripts"))
    parser = load("accepted_fc_worker", directory / "accepted_fc_worker.py")
    parser.NAME, parser.SCOPE = NAME, SCOPE
    original = parser.parse_log

    def parse(log, arm, config):
        if log.count("FINAL_FC_WRAPPER_PASS raw_active=0") != 1:
            raise ValueError("final FC wrapper completion marker")
        return original(log, arm, config)

    parser.parse_log = parse
    return parser


def parse_log(log, arm, config):
    return parser_at(Path(__file__).parent).parse_log(log, arm, config)


def worker():
    parser = parser_at(Path.cwd())
    parser.worker()
    path = Path("status.json")
    state = json.loads(path.read_text())
    if state["status"] == "PASS_three_representative_FC_A_cold_batch_pairs":
        state["status"] = "PASS_final_FC_wrapper_three_batch_pairs"
    path.write_text(json.dumps(state, indent=2) + "\n")


def launch():
    from run_srnn_resident_remote import REMOTE, connect, run, upload
    from run_srnn_rmw_turnover_core import probe_code
    root, prepared, archive = roots()
    files = [p for p in prepared.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    unchanged = [p.relative_to(prepared).as_posix() for p in files
                 if (root / PRIOR_PREPARED / p.relative_to(prepared)).is_file()
                 and p.read_bytes() == (root / PRIOR_PREPARED / p.relative_to(prepared)).read_bytes()]
    changed = [p for p in files if p.relative_to(prepared).as_posix() not in unchanged]
    payload = sum(p.stat().st_size for p in changed)
    destination = REMOTE + "/" + NAME
    client = connect()
    try:
        before = json.loads(run(client, "python3 -c " + shlex.quote(
            probe_code(payload + len(unchanged) * 4096 + 256 * 1024**2))))
        code = """from pathlib import Path
import json
old=Path(OLD);new=Path(NEW);links=[]
assert not new.exists() and not Path(SCRATCH).exists()
new.mkdir()
for rel in FILES:
 src=(old/rel).resolve();dst=new/rel
 assert src.is_file() and src.is_relative_to(Path(PUBLIC)),str(src)
 dst.parent.mkdir(parents=True,exist_ok=True);st=src.stat();dst.symlink_to(src)
 links.append(dict(path=rel,target=str(src),size=st.st_size,mtime_ns=st.st_mtime_ns))
print(json.dumps(links))
"""
        links = json.loads(run(client, "python3 -c " + shlex.quote(
            "OLD=" + repr(REMOTE + "/" + PRIOR) + "\nNEW=" + repr(destination)
            + "\nPUBLIC=" + repr(REMOTE) + "\nSCRATCH=" + repr("/tmp/snn_" + NAME)
            + "\nFILES=" + repr(unchanged) + "\n" + code)))
        with client.open_sftp() as sftp:
            for path in changed:
                target = destination + "/" + path.relative_to(prepared).as_posix()
                upload(sftp, str(path), target)
                with sftp.open(target, "rb") as stream:
                    if stream.read() != path.read_bytes():
                        raise ValueError("upload bytes: " + str(path))
        start = json.loads(run(client, "python3 -c " + shlex.quote(probe_code(256 * 1024**2))))
        run(client, "screen -dmS snn_" + NAME + " bash -lc " + shlex.quote(
            "cd " + shlex.quote(destination) + " && exec python3 launch_worker.py worker"))
        archive.mkdir(exist_ok=False)
        record = dict(before_copy=before, before_start=start, linked_static_inputs=links,
                      prepared_files=len(files), prepared_bytes=sum(p.stat().st_size for p in files),
                      uploaded_files=len(changed), uploaded_bytes=payload,
                      remote=destination, screen="snn_" + NAME)
        (archive / "launch_observation.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "linked_static_inputs"}))
    finally:
        client.close()


def audit():
    from run_srnn_resident_remote import REMOTE, connect, run
    _, prepared, archive = roots()
    if (archive / "terminal_audit.json").exists():
        raise FileExistsError("terminal audit exists")
    state = json.loads((archive / "status.json").read_text())
    if not state.get("end") or any(v.get("timed_out") for v in state.values() if isinstance(v, dict)):
        raise ValueError("natural terminal required; timeout PID must be examined")
    code = """import datetime,json,pathlib,shutil
p=pathlib.Path(ROOT);s=pathlib.Path(SCRATCH);skip={'obj','elab_obj','__pycache__','.Xil'}
def inv(root):
 return {q.relative_to(root).as_posix():q.stat().st_size for q in root.rglob('*') if q.is_file() and not set(q.relative_to(root).parts)&skip}
active=[]
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(q/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):active.append([q.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
links={q.relative_to(p).as_posix():dict(target=str(q.resolve()),size=q.stat().st_size,mtime_ns=q.stat().st_mtime_ns) for q in p.rglob('*') if q.is_symlink()}
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),files=inv(p),scratch_noncache_files=inv(s),links=links,active_eda=active,data_free=shutil.disk_usage(p).free,tmp_free=shutil.disk_usage('/tmp').free)))
"""
    client = connect()
    try:
        observation = json.loads(run(client, "python3 -c " + shlex.quote(
            "ROOT=" + repr(REMOTE + "/" + NAME) + "\nSCRATCH=" + repr(state["scratch"]) + "\n" + code)))
    finally:
        client.close()
    if observation["active_eda"] or observation["scratch_noncache_files"]:
        raise ValueError("active EDA or missing scratch originals")
    launch_record = json.loads((archive / "launch_observation.json").read_text())
    links = {r["path"]: {k: r[k] for k in ("target", "size", "mtime_ns")}
             for r in launch_record["linked_static_inputs"]}
    if observation["links"] != links or list(archive.rglob("*.part")):
        raise ValueError("link metadata or partial return")
    for rel, size in observation["files"].items():
        if not (archive / rel).is_file() or (archive / rel).stat().st_size != size:
            raise ValueError("raw archive: " + rel)
    inputs = [p for p in prepared.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for path in inputs:
        if path.read_bytes() != (archive / path.relative_to(prepared)).read_bytes():
            raise ValueError("input bytes: " + str(path))
    report = dict(status="ARCHIVED_NOT_ACCEPTED", original_status=state["status"], scope=SCOPE,
                  raw_files=len(observation["files"]), raw_bytes=sum(observation["files"].values()),
                  inputs_byte_equal=len(inputs), static_links_preserved=len(links), remote_observation=observation)
    if state["status"] == "PASS_final_FC_wrapper_three_batch_pairs":
        actual = load("actual_final_fc_worker", archive / "launch_worker.py")
        config = json.loads((archive / "preparation.json").read_text())
        measurements = []
        stages = ["build"]
        for case in config["configs"]:
            for arm in ("off", "on"):
                stage = f'run_{case["batch"]}_{arm}'
                stages.append(stage)
                checks = actual.parse_log((archive / (stage + ".log")).read_text(), arm, case)
                if checks != json.loads((archive / (stage + "_checks.json")).read_text()):
                    raise ValueError("actual parser/checks: " + stage)
                buckets = checks["stages"]
                cold = sum(v["cycles"] for v in buckets.values())
                setup = sum(buckets[k]["cycles"] for k in
                            ("parameter_load", "initial_refill", "initial_state", "learning_configuration"))
                if cold != setup + buckets["outside"]["cycles"] + checks["runtime_cycles"]:
                    raise ValueError("cycle conservation: " + stage)
                measurements.append(dict(batch=case["batch"], arm=arm, runtime_cycles=checks["runtime_cycles"],
                                         counted_postreset_cycles=cold, legacy=checks["legacy"]))
        if any(state.get(s, {}).get("exit") != 0 for s in stages):
            raise ValueError("real exits missing")
        diagnostics = {}
        for stage in stages:
            text = (archive / (stage + ".log")).read_text(errors="replace")
            warnings = dict(collections.Counter(re.findall(r"^%Warning-([^:]+):", text, re.M)))
            if re.search(r"%Error|%Fatal|Assertion failed|\bFATAL\b", text) or set(warnings) & {"LATCH", "MULTIDRIVEN", "UNOPTFLAT"}:
                raise ValueError("tool diagnostic: " + stage)
            diagnostics[stage] = warnings
        if diagnostics["build"] != state["build_warnings"]:
            raise ValueError("build warnings differ")
        report.update(status="PASS_final_FC_wrapper_archive", measurements=measurements, diagnostics=diagnostics,
                      physical_rtl_files_byte_equal=113, actual_worker_strict_reparse=True)
    (archive / "terminal_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "remote_observation"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "worker", "audit"))
    globals()[parser.parse_args().action]()
