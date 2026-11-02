"""Verify the implemented raw-source wrapper with the accepted SRNN closed loop."""
import argparse
import collections
import datetime as dt
import difflib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
NAME = "final_raw_source_functional_r1"
PRIOR = "srnn_gradient_retire_core_r1"
TB = "tb_srnn_profile_measurement.sv"
SCOPE = (
    "Exact candidate wrapper and RTL used by final_raw_source_synth_r1/impl_r1; "
    "the existing eight sequences and 588-step SRNN learning closed loop now "
    "drive the wrapper raw ports. Original numerical assertions and monitors "
    "retained. Not new inputs, FC/SCNN full regression, board, SGD or new QoR."
)
ADDED = ("source_scan.sv", "source_stream.sv", "raw_source_binding.sv",
         "unified_h40_raw_source_top_v1.sv")
GROUP = ("valid", "recurrent", "source_valid", "weight_addr0",
         "weight_addr1", "weight_addr2", "weight_addr3")


def now():
    return dt.datetime.now().astimezone().isoformat()


def roots():
    from run_srnn_resident_remote import ROOT, RESULT, local_path
    root = Path(local_path(ROOT))
    return root, root / "verification/experiments" / NAME, root / RESULT / NAME


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("nonunique connection anchor: " + old)
    return text.replace(old, new, 1)


def bind_wrapper(text):
    # Preserve observations at the post-mux core boundary after moving the mux.
    text = text.replace("u_final.", "u_final.u_core.")
    text = once(text, "    unified_h40_final_composition_top_v1 #(",
                "    unified_h40_raw_source_top_v1 #(\n        .RAW_SOURCE_POLICY(1),")
    text = once(text, "    ) u_final (", """    ) u_final (
        .i_raw_source_active(feed_active),
        .i_raw_count_valid(count_valid),
        .i_raw_counts(count_word),
        .o_raw_count_ready(count_ready),""")
    for suffix in GROUP:
        text = once(text, f".i_fc_srnn_group_{suffix}(i_fc_srnn_group_{suffix})",
                    f".i_fc_srnn_group_{suffix}(manual_group_{suffix})")
    text = once(text, ".i_fc_srnn_group_feedforward_end(feed_active && producer_last)",
                ".i_fc_srnn_group_feedforward_end(1'b0)")
    begin = text.index("    source_scan_experiment raw_source_i(")
    end = text.index("    always @(posedge i_clk) if(i_rst_n) begin", begin)
    aliases = [
        "    assign producer_valid = u_final.u_raw_source.producer_valid;",
        "    assign producer_last = u_final.u_raw_source.producer_last;",
    ]
    aliases += [f"    assign i_fc_srnn_group_{s} = u_final.raw_group_{s};" for s in GROUP]
    text = text[:begin] + "\n".join(aliases) + "\n" + text[end:]
    text = once(text, '        if (streaming!=0 && streaming!=1) $fatal(1,"source policy range");',
                '        if (streaming!=1) $fatal(1,"static wrapper requires streaming=1");')
    text = once(text, "        $finish;",
                '        $display("FINAL_RAW_WRAPPER_PASS policy=1 sequences=8 steps=588");\n        $finish;')
    return text


def prepare():
    root, out, archive = roots()
    prior = root / "verification/experiments" / PRIOR
    result = archive.parent
    physical = root / "verification/experiments/final_raw_source_composition_r1/candidate/rtl"
    accepted = result / "final_raw_source_synth_r1/candidate/rtl"
    if out.exists() or archive.exists():
        raise FileExistsError("isolated run already prepared or launched")
    shutil.copytree(prior, out, ignore=shutil.ignore_patterns("__pycache__"))
    original = (prior / TB).read_text()
    changed = bind_wrapper(original)
    (out / TB).write_text(changed, newline="\n")
    (out / "wrapper_tb.diff").write_text("".join(difflib.unified_diff(
        original.splitlines(True), changed.splitlines(True), fromfile=PRIOR + "/" + TB,
        tofile=NAME + "/" + TB)))
    old_core = {p.relative_to(prior / "rtl").as_posix(): p
                for p in (prior / "rtl").rglob("*") if p.is_file()}
    for rel, p in old_core.items():
        if p.read_bytes() != (physical / rel).read_bytes():
            raise ValueError("functional/physical core differs: " + rel)
    for p in physical.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(physical)
        if p.read_bytes() != (accepted / rel).read_bytes():
            raise ValueError("accepted synthesis source differs: " + str(rel))
        dst = out / "rtl" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    sources = (prior / "sources.f").read_text().splitlines()
    sources = [s for s in sources if s.strip('"') not in ("source_scan.sv", "source_stream.sv")]
    sources += ['"rtl/top/' + n + '"' for n in ADDED]
    (out / "sources.f").write_text("\n".join(sources) + "\n", newline="\n")
    shutil.copy2(prior / "launch_worker.py", out / "accepted_core_parser.py")
    shutil.copy2(__file__, out / "launch_worker.py")
    expected = json.loads((result / PRIOR / "checks.json").read_text())["stream_on"]
    (out / "expected_checks.json").write_text(json.dumps(expected, indent=2) + "\n")
    record = dict(status="PREPARED_NOT_SIMULATED", scope=SCOPE, prior=PRIOR,
                  original_core_files_byte_equal=len(old_core), added_rtl=list(ADDED),
                  physical_rtl_files_byte_equal=sum(p.is_file() for p in physical.rglob("*")),
                  numerical_files_unchanged=sum(p.is_file() for p in (out / "multi_fixture").rglob("*")),
                  original_assertions="Retained; only producer wiring and hierarchical monitor paths move.")
    (out / "preparation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


def parse_log(log, info, expected, directory):
    load("run_srnn_representative_inputs", directory / "run_srnn_representative_inputs.py")
    load("source_core_parser", directory / "source_core_parser.py")
    parser = load("accepted_core_parser", directory / "accepted_core_parser.py")
    parsed = parser.parse_log(log, info)
    if log.count("FINAL_RAW_WRAPPER_PASS policy=1 sequences=8 steps=588") != 1:
        raise ValueError("wrapper completion marker missing or duplicated")
    if parsed != expected:
        differences = [k for k in set(parsed) | set(expected) if parsed.get(k) != expected.get(k)]
        raise ValueError("wrapper changed accepted numerical/cycle/traffic records: " + repr(differences))
    return parsed


def compare_wrapper_hierarchy(parsed, expected, log, original_log):
    """Allow only the one wrapper hierarchy insertion, including raw tile counters."""
    mapped = json.loads(json.dumps(expected))
    for tile in mapped["tiles"]:
        if tile.count(".u_final.") != 1 or ".u_final.u_core." in tile:
            raise ValueError("unexpected original tile hierarchy")
    mapped["tiles"] = [t.replace(".u_final.", ".u_final.u_core.", 1)
                       for t in mapped["tiles"]]
    if parsed != mapped:
        raise ValueError("non-hierarchy record difference")
    original_rows = re.findall(r"^H40_ACTIVITY .*$", original_log, re.M)
    actual_rows = re.findall(r"^H40_ACTIVITY .*$", log, re.M)
    if len(original_rows) != 12 or len(actual_rows) != 12:
        raise ValueError("incomplete raw tile activity")
    expected_rows = [r.replace(".u_final.", ".u_final.u_core.", 1) for r in original_rows]
    if sorted(expected_rows) != sorted(actual_rows):
        raise ValueError("raw tile activity changed beyond hierarchy")
    return mapped


def worker():
    root = Path.cwd()
    scratch = Path("/tmp/snn_" + NAME)
    scratch.mkdir(exist_ok=False)
    state = dict(status="running", start=now(), scratch=str(scratch), scope=SCOPE)

    def save():
        (root / "status.json").write_text(json.dumps(state, indent=2) + "\n")

    def execute(stage, command, timeout):
        state["status"] = stage + "_running"
        start = time.monotonic()
        with (root / (stage + ".log")).open("x") as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT)
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
        build = json.loads((root / "build_command.json").read_text())
        build[build.index("--Mdir") + 1] = str(scratch / "obj")
        execute("build", build, 3600)
        log = (root / "build.log").read_text(errors="replace")
        categories = collections.Counter(re.findall(r"^%Warning-([^:]+):", log, re.M))
        state["build_warnings"] = dict(categories)
        if set(categories) & {"LATCH", "MULTIDRIVEN", "UNOPTFLAT"}:
            raise ValueError("structural build warning")
        execute("run_wrapper_on", [str(scratch / "obj/Vtb_srnn_profile_measurement"),
                "+RAW=multi_fixture", "+FIXTURE=multi_fixture", "+LEARNING=1", "+STREAMING=1"], 7200)
        parsed = parse_log((root / "run_wrapper_on.log").read_text(),
                           json.loads((root / "multi_fixture/reference.json").read_text()),
                           json.loads((root / "expected_checks.json").read_text()), root)
        (root / "checks.json").write_text(json.dumps(dict(wrapper_on=parsed), indent=2) + "\n")
        state["status"] = "PASS_final_raw_wrapper_closed_loop_exact"
        state["accepted_records_exact"] = True
        state["cold_cycles"] = parsed["total"]["cycles"]
    except Exception as error:
        state.update(status="FAIL", error=repr(error))
    state.update(end=now(), scratch_preserved=True)
    save()


def launch():
    from run_srnn_resident_remote import REMOTE, connect, run, upload
    from run_srnn_rmw_turnover_core import probe_code
    root, prepared, archive = roots()
    prior = root / "verification/experiments" / PRIOR
    files = sorted(p for p in prepared.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    unchanged = [p.relative_to(prepared).as_posix() for p in files
                 if (prior / p.relative_to(prepared)).is_file()
                 and p.read_bytes() == (prior / p.relative_to(prepared)).read_bytes()]
    changed = [p for p in files if p.relative_to(prepared).as_posix() not in unchanged]
    payload = sum(p.stat().st_size for p in changed)
    destination = REMOTE + "/" + NAME
    client = connect()
    try:
        before = json.loads(run(client, "python3 -c " + shlex.quote(
            probe_code(payload + len(unchanged) * 4096 + 256 * 1024**2))))
        code = """from pathlib import Path
import filecmp,json
old=Path(OLD);new=Path(NEW);observed=[]
assert not new.exists() and not Path(SCRATCH).exists()
new.mkdir()
for rel in FILES:
 src=(old/rel).resolve();dst=new/rel
 assert src.is_file() and src.is_relative_to(Path(PUBLIC)),str(src)
 dst.parent.mkdir(parents=True,exist_ok=True)
 st=src.stat();dst.symlink_to(src)
 assert filecmp.cmp(src,dst,shallow=False),rel
 observed.append(dict(path=rel,target=str(src),size=st.st_size,mtime_ns=st.st_mtime_ns))
print(json.dumps(observed))
"""
        links = json.loads(run(client, "python3 -c " + shlex.quote(
            "OLD=" + repr(REMOTE + "/" + PRIOR) + "\nNEW=" + repr(destination)
            + "\nPUBLIC=" + repr(REMOTE) + "\nSCRATCH=" + repr("/tmp/snn_" + NAME)
            + "\nFILES=" + repr(unchanged) + "\n" + code)))
        with client.open_sftp() as sftp:
            for p in changed:
                target = destination + "/" + p.relative_to(prepared).as_posix()
                upload(sftp, str(p), target)
                with sftp.open(target, "rb") as stream:
                    if stream.read() != p.read_bytes():
                        raise ValueError("uploaded input mismatch: " + str(p))
        at_start = json.loads(run(client, "python3 -c " + shlex.quote(probe_code(256 * 1024**2))))
        shell = "cd " + shlex.quote(destination) + " && exec python3 launch_worker.py worker"
        run(client, "screen -dmS snn_" + NAME + " bash -lc " + shlex.quote(shell))
        archive.mkdir(parents=True, exist_ok=True)
        record = dict(launched=True, remote=destination, screen="snn_" + NAME,
                      before_copy=before, before_start=at_start, linked_static_inputs=links,
                      prepared_input_files=len(files), prepared_input_bytes=sum(p.stat().st_size for p in files),
                      uploaded_changed_files_byte_equal=len(changed), uploaded_changed_bytes=payload)
        (archive / "launch_observation.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "linked_static_inputs"}, indent=2))
    finally:
        client.close()


def audit():
    from run_srnn_resident_remote import REMOTE, connect, run
    _, prepared, archive = roots()
    if (archive / "terminal_audit.json").exists():
        raise FileExistsError("terminal audit already performed")
    state = json.loads((archive / "status.json").read_text())
    if not state.get("end") or any(v.get("timed_out") for v in state.values() if isinstance(v, dict)):
        raise ValueError("requires natural terminal state; timeout PID must be examined first")
    code = """import datetime,json,pathlib,shutil
p=pathlib.Path(ROOT);s=pathlib.Path(SCRATCH);skip={'obj','elab_obj','__pycache__','.Xil'}
def inventory(root):
 return {q.relative_to(root).as_posix():q.stat().st_size for q in root.rglob('*')
         if q.is_file() and not set(q.relative_to(root).parts)&skip}
active=[]
for q in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(q/'comm').read_text().strip()
  if comm in ('vivado','verilator_bin','Vscgraph','xelab','xsim') or comm.startswith('Vtb_'):active.append([q.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
links={q.relative_to(p).as_posix():dict(target=str(q.resolve()),size=q.stat().st_size,mtime_ns=q.stat().st_mtime_ns)
       for q in p.rglob('*') if q.is_symlink()}
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),files=inventory(p),
 scratch_noncache_files=inventory(s),links=links,active_eda=active,
 data_free=shutil.disk_usage(p).free,tmp_free=shutil.disk_usage('/tmp').free)))
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
    expected_links = {r["path"]: {k: r[k] for k in ("target", "size", "mtime_ns")}
                      for r in launched["linked_static_inputs"]}
    if expected_links != observation["links"] or list(archive.rglob("*.part")):
        raise ValueError("source metadata changed or partial return")
    for rel, size in observation["files"].items():
        if not (archive / rel).is_file() or (archive / rel).stat().st_size != size:
            raise ValueError("archive file size: " + rel)
    inputs = [p for p in prepared.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for p in inputs:
        if p.read_bytes() != (archive / p.relative_to(prepared)).read_bytes():
            raise ValueError("prepared/actual input bytes: " + str(p))
    report = dict(status="ARCHIVED_NOT_ACCEPTED", original_status=state["status"],
                  scope=SCOPE, raw_files=len(observation["files"]),
                  raw_bytes=sum(observation["files"].values()), inputs_byte_equal=len(inputs),
                  preserved_static_links=len(expected_links), remote_observation=observation,
                  dcp_crc="not applicable: no new DCP")
    if state["status"].startswith("PASS_"):
        if any(state.get(s, {}).get("exit") != 0 for s in ("build", "run_wrapper_on")):
            raise ValueError("missing real successful build/run exits")
        actual = load("actual_wrapper_worker", archive / "launch_worker.py")
        parsed = actual.parse_log((archive / "run_wrapper_on.log").read_text(),
                                  json.loads((archive / "multi_fixture/reference.json").read_text()),
                                  json.loads((archive / "expected_checks.json").read_text()), archive)
        if parsed != json.loads((archive / "checks.json").read_text())["wrapper_on"]:
            raise ValueError("actual worker reparse mismatch")
        report.update(status="PASS_final_raw_wrapper_functional_archive",
                      records_match_accepted_core=True, cold_cycles=parsed["total"]["cycles"],
                      sequences=len(parsed["sequences"]))
    elif state.get("error") == 'ValueError("wrapper changed accepted numerical/cycle/traffic records: [\'tiles\']")':
        # The actual worker and its original FAIL remain immutable. Reproduce that
        # failure, then verify the known wrapper path change against every raw tile row.
        if any(state.get(s, {}).get("exit") != 0 for s in ("build", "run_wrapper_on")):
            raise ValueError("hierarchy diagnosis requires successful real exits")
        actual = load("actual_wrapper_worker", archive / "launch_worker.py")
        log = (archive / "run_wrapper_on.log").read_text()
        info = json.loads((archive / "multi_fixture/reference.json").read_text())
        expected = json.loads((archive / "expected_checks.json").read_text())
        try:
            actual.parse_log(log, info, expected, archive)
        except ValueError as error:
            if repr(error) != state["error"]:
                raise ValueError("original worker failure did not reproduce") from error
        else:
            raise ValueError("original worker unexpectedly accepted")
        parsed = sys.modules["accepted_core_parser"].parse_log(log, info)
        compare_wrapper_hierarchy(parsed, expected, log,
                                  (archive.parent / PRIOR / "run_stream_on.log").read_text())
        build = (archive / "build.log").read_text(errors="replace")
        warnings = dict(collections.Counter(re.findall(r"^%Warning-([^:]+):", build, re.M)))
        if warnings != state["build_warnings"] or "%Error" in build or set(warnings) & {"LATCH", "MULTIDRIVEN", "UNOPTFLAT"}:
            raise ValueError("build diagnostics changed")
        report.update(status="PASS_final_raw_wrapper_functional_archive",
                      original_worker_failure_reproduced=True,
                      offline_correction="Only u_final -> u_final.u_core tile identity; original FAIL retained.",
                      raw_tile_activity_rows_exact_except_hierarchy=12,
                      records_match_accepted_core_except_hierarchy=True,
                      cold_cycles=parsed["total"]["cycles"], sequences=len(parsed["sequences"]),
                      build_warnings=warnings)
    (archive / "terminal_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "remote_observation"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "worker", "audit"))
    args = parser.parse_args()
    globals()[args.action]()
