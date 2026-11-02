"""Final wrapper SCNN nonzero update/publication/query regression."""
import argparse
import collections
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

from run_final_raw_source_functional import load, once

sys.dont_write_bytecode = True
NAME = "final_scnn_functional_r1"
PRIOR = "scnn_turnover_tail_r1"
PRIOR_PREPARED = "verification/experiments/scnn_forward_turnover/tail_r1/candidate_r1"
TB = "tb_scnn_tail_visibility.sv"
SCOPE = (
    "Final physical wrapper/core, raw source inactive. Original cold B4T12 plus "
    "query4, 52 Conv8 tail descriptors with explicit CPU Conv7 feature service. "
    "Exact final shared owner/config/RMW/publication and subsequent RTL classifier "
    "query; not a full raw graph, new input distribution, full T20/Test, board, "
    "SGD or new physical result."
)


def roots():
    from run_srnn_resident_remote import ROOT, RESULT, local_path
    root = Path(local_path(ROOT))
    return root, root / "verification/experiments" / NAME, root / RESULT / NAME


def bind_wrapper(text):
    text = re.sub(r"\bdut\.", "dut.u_core.", text)
    text = once(text, "    unified_h40_final_composition_top_v1 #(",
                "    unified_h40_raw_source_top_v1 #(\n        .RAW_SOURCE_POLICY(1),")
    text = once(text, "    ) dut (", """    ) dut (
        .i_raw_source_active(1'b0),
        .i_raw_count_valid(1'b0),
        .i_raw_counts(32'b0),
        .i_fc_srnn_group_feedforward_end(1'b0),""")
    return once(text, "        $finish;",
                '        $display("FINAL_SCNN_WRAPPER_PASS raw_active=0");\n        $finish;')


def transport():
    # Reuse the final-profile input upload/link implementation in a separate module.
    module = load("scnn_final_profile_transport", Path(__file__).with_name("run_final_fc_functional.py"))
    module.NAME, module.PRIOR = NAME, PRIOR
    module.PRIOR_PREPARED, module.TB, module.SCOPE = PRIOR_PREPARED, TB, SCOPE
    module.bind_wrapper = bind_wrapper
    return module


def prepare():
    transport().prepare()
    _, out, _ = roots()
    (out / "accepted_fc_worker.py").rename(out / "accepted_tail_worker.py")
    shutil.copy2(__file__, out / "launch_worker.py")
    shutil.copy2(Path(__file__).with_name("run_final_fc_functional.py"), out / "run_final_fc_functional.py")
    print(json.dumps(dict(name=NAME, prepared=str(out), build_timeout=5400, run_timeout=21600,
                         timeout_behavior="retain PID; no kill/restart")))


def parse_log(log, directory=None):
    directory = directory or Path(__file__).parent
    original = load("original_tail_numeric_parser", directory / "prepare_scnn_tail_testbench.py")
    accepted = load("accepted_tail_parser", directory / "accepted_tail_worker.py")
    accepted.SCOPE = SCOPE
    if log.count("FINAL_SCNN_WRAPPER_PASS raw_active=0") != 1:
        raise ValueError("final SCNN wrapper completion marker")
    result = accepted.parse_log(log, original.parse_log)
    result["status"] = "PASS_final_SCNN_wrapper_nonzero_tail_query"
    return result


def worker():
    root = Path.cwd()
    scratch = Path("/tmp/snn_" + NAME)
    scratch.mkdir(exist_ok=False)
    config = json.loads((root / "reference.json").read_text())
    state = dict(status="running", start=dt.datetime.now().astimezone().isoformat(),
                 scratch=str(scratch), scope=SCOPE)

    def save():
        (root / "status.json").write_text(json.dumps(state, indent=2) + "\n")

    def execute(stage, command, timeout):
        state["status"] = stage + "_running"
        start = time.monotonic()
        with (root / (stage + ".log")).open("x") as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT)
            state[stage] = dict(command=command, pid=process.pid, exit=None)
            save()
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                state[stage].update(timed_out=True, wall_seconds=time.monotonic() - start)
                save()
                raise
        state[stage].update(exit=code, wall_seconds=time.monotonic() - start)
        save()
        if code:
            raise RuntimeError(stage + " exit " + str(code))

    try:
        command = json.loads((root / "accepted_build.json").read_text())
        command[command.index("--top-module") + 1] = "tb_scnn_tail_visibility"
        command[command.index("--Mdir") + 1] = str(scratch / "obj")
        command[command.index("-f") + 1] = "sources.f"
        command[command.index("-MAKEFLAGS") + 1] = (
            "OPT_FAST=-O1 OPT_GLOBAL=-O1 -f " + str(root / "scripts/supplement_scnn_graph_host.mk"))
        execute("build", command, config["build_timeout"])
        log = (root / "build.log").read_text(errors="replace")
        warnings = dict(collections.Counter(re.findall(r"^%Warning-([^:]+):", log, re.M)))
        state["build_warnings"] = warnings
        if "%Error" in log or set(warnings) & {"LATCH", "MULTIDRIVEN", "UNOPTFLAT"}:
            raise ValueError("structural build diagnostics")
        execute("run", [str(scratch / "obj/Vscgraph"), "+SCNN_TEMPORAL_FIXTURE=fixture",
                        "+SCNN_TEMPORAL_CONTROL=control", "+TAIL_FIXTURE=tail_reference"],
                config["run_timeout"])
        checks = parse_log((root / "run.log").read_text(), root)
        (root / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        state.update(status=checks["status"], final=checks["final"])
    except Exception as error:
        state.update(status="FAIL", error=repr(error))
    state.update(end=dt.datetime.now().astimezone().isoformat(), scratch_preserved=True)
    save()


def launch():
    transport().launch()


def audit():
    from run_srnn_resident_remote import REMOTE, connect, run
    _, prepared, archive = roots()
    if (archive / "terminal_audit.json").exists():
        raise FileExistsError("terminal audit exists")
    state = json.loads((archive / "status.json").read_text())
    if not state.get("end") or any(v.get("timed_out") for v in state.values() if isinstance(v, dict)):
        raise ValueError("natural terminal required; inspect timeout PID first")
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
        raise ValueError("active EDA or unreturned scratch evidence")
    launched = json.loads((archive / "launch_observation.json").read_text())
    links = {r["path"]: {k: r[k] for k in ("target", "size", "mtime_ns")}
             for r in launched["linked_static_inputs"]}
    if links != observation["links"] or list(archive.rglob("*.part")):
        raise ValueError("link metadata or partial return")
    for rel, size in observation["files"].items():
        if not (archive / rel).is_file() or (archive / rel).stat().st_size != size:
            raise ValueError("raw size: " + rel)
    inputs = [p for p in prepared.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for path in inputs:
        if path.read_bytes() != (archive / path.relative_to(prepared)).read_bytes():
            raise ValueError("actual input bytes: " + str(path))
    report = dict(status="ARCHIVED_NOT_ACCEPTED", original_status=state["status"], scope=SCOPE,
                  raw_files=len(observation["files"]), raw_bytes=sum(observation["files"].values()),
                  inputs_byte_equal=len(inputs), static_links_preserved=len(links), remote_observation=observation)
    if state["status"] == "PASS_final_SCNN_wrapper_nonzero_tail_query":
        if any(state.get(s, {}).get("exit") != 0 for s in ("build", "run")):
            raise ValueError("real exits missing")
        actual = load("actual_final_scnn_worker", archive / "launch_worker.py")
        checks = actual.parse_log((archive / "run.log").read_text(), archive)
        if checks != json.loads((archive / "checks.json").read_text()) or checks["final"] != state["final"]:
            raise ValueError("actual worker/checks/status mismatch")
        diagnostics = {}
        for stage in ("build", "run"):
            text = (archive / (stage + ".log")).read_text(errors="replace")
            warnings = dict(collections.Counter(re.findall(r"^%Warning-([^:]+):", text, re.M)))
            if re.search(r"%Error|%Fatal|Assertion failed|\bFATAL\b", text) or set(warnings) & {"LATCH", "MULTIDRIVEN", "UNOPTFLAT"}:
                raise ValueError("tool diagnostics: " + stage)
            diagnostics[stage] = warnings
        if diagnostics["build"] != state["build_warnings"]:
            raise ValueError("build warnings differ")
        report.update(status="PASS_final_SCNN_wrapper_archive", physical_rtl_files_byte_equal=113,
                      actual_worker_strict_reparse=True, final=checks["final"], diagnostics=diagnostics,
                      native_service=checks["native_service"], native_publications=checks["native_publications"])
    (archive / "terminal_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "remote_observation"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "worker", "audit"))
    globals()[parser.parse_args().action]()
