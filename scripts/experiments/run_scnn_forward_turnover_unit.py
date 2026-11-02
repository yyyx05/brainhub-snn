"""Run old/new three-module chains against the same two scalar services."""
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

NAME = "scnn_forward_turnover_unit_r1"


def parse_log(text):
    if re.search(r"%Error|%Fatal|Assertion failed", text):
        raise ValueError("simulation diagnostic")
    def records(prefix):
        return [dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", line))
                for line in text.splitlines() if line.startswith(prefix + " ")]
    final = records("SCNN_TURNOVER_UNIT_PASS")
    if (len(final)!=1 or final[0].get("services") not in (2,4) or
        final[0] != dict(variants=2, services=final[0]["services"], cases_per_checker=14, owner_errors_per_checker=1)):
        raise ValueError("missing or duplicated terminal marker")
    service_count = final[0]["services"]
    cases, summaries = records("SCNN_TURNOVER_CASE"), records("SCNN_TURNOVER_CHECKER_PASS")
    if len(cases) != 28*service_count or len(summaries) != 2*service_count:
        raise ValueError("incomplete crossed comparison")
    comparison = []
    for service in range(service_count):
        groups = [[r for r in cases if r["service"] == service and r["opt"] == opt] for opt in (0, 1)]
        for group in groups:
            if [r["case_id"] for r in group] != list(range(14)):
                raise ValueError("case order or duplicate")
        for before, after in zip(*groups):
            stable = lambda row: {k:v for k,v in row.items() if k not in ("opt", "cycles")}
            if stable(before) != stable(after):
                raise ValueError("different work across implementations")
            f, c = before["features"], before["classes"]
            if before["reads"] != f*c or before["rows"] != 2*c*((f+23)//24):
                raise ValueError("incomplete matrix coverage")
            savings = before["cycles"] - after["cycles"]
            if not before["stalls"]:
                nonempty_rows = c*((f+23)//24 + (f+22)//24)
                expected = (f*c - nonempty_rows) if service else 0
                if savings != expected:
                    raise ValueError("unexpected no-stall cycle saving")
            comparison.append(dict(service=service, case_id=before["case_id"], features=f, classes=c,
                stalls=before["stalls"], reads=before["reads"], rows=before["rows"],
                reference_cycles=before["cycles"], candidate_cycles=after["cycles"], saved_cycles=savings,
                fair_no_stall_ratio=before["cycles"]/after["cycles"] if not before["stalls"] else None))
    if {(r["service"],r["opt"]) for r in summaries} != {(s,o) for s in range(service_count) for o in (0,1)}:
        raise ValueError("missing checker summary")
    for row in summaries:
        if (row["cases"],row["aborts"],row["bad_responses"],row["invalid_starts"]) != (14,4,2,1):
            raise ValueError("missing recovery cases")
        if bool(row["replacements"]) != bool(row["service"] and row["opt"]):
            raise ValueError("replacement count inconsistent with service/RTL")
    return dict(comparison=comparison, summaries=summaries,
        boundary=("Publisher -> actual Conv forwarding backend -> actual shared router, owner3. Identical16-bit host service per comparison, one response slot, latency1 minimum. Stalled ratios not performance evidence. Harness flushes router/store on module abort, not final-top cancellation, DDR or whole-core throughput. No mapped area or new STA." if service_count==2 else
        "Publisher -> actual Conv forwarding backend -> actual shared router, owner3. Identical16-bit host service per comparison, one response slot. Service0 is legacy latency1; service1 allows same-cycle replacement latency1; service2/3 use the same replacement rule with latency4/12. Stalled ratios not performance evidence. Harness flushes router/store on module abort, not final-top cancellation, DDR or whole-core throughput. No mapped area or new STA."))


def worker(verilator):
    state = dict(status="running",start=dt.datetime.now().astimezone().isoformat())
    def save():
        Path("status.json").write_text(json.dumps(state,indent=2)+"\n")
    def execute(stage, command, timeout):
        state["status"] = stage + "_running"
        save()
        start = time.monotonic()
        with Path(stage+".log").open("w") as log:
            try:
                result = subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            except subprocess.TimeoutExpired:
                state[stage] = dict(command=command,exit=None,timeout_seconds=timeout,wall_seconds=time.monotonic()-start)
                raise
        state[stage] = dict(command=command,exit=result.returncode,wall_seconds=time.monotonic()-start)
        save()
        if result.returncode:
            raise RuntimeError(stage+" exit "+str(result.returncode))
    try:
        names = ("unified_scnn_classifier_cache_refill_publisher_v1",
                 "unified_scnn_selected_layers_parameter_backend_v1",
                 "unified_tp_three_client_parameter_backend_router")
        sources = [str(Path("rtl/learning")/(n+".sv")) for n in names] + ["reference_"+n+".sv" for n in names]
        sources += ["unified_array16_weight_refill_dispatcher_v1.sv","turnover_chain.sv","tb_scnn_forward_turnover.sv"]
        execute("build",[verilator,"--binary","--timing","--assert","-Wno-fatal","-j","2",
                         "--top-module","tb_scnn_forward_turnover","--Mdir","obj","-CFLAGS","-O1"]+sources,300)
        build=Path("build.log").read_text(errors="replace")
        state["ordinary_warning_count"] = len(re.findall(r"^%Warning-",build,re.M))
        state["structural_warnings"] = re.findall(r"^%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT).*",build,re.M)
        if state["structural_warnings"] or "%Error" in build:
            raise RuntimeError("structural build diagnostic")
        execute("run",["obj/Vtb_scnn_forward_turnover"],90)
        state["checks"] = parse_log(Path("run.log").read_text(errors="replace"))
        Path("checks.json").write_text(json.dumps(state["checks"],indent=2)+"\n")
        state["status"] = "PASS_turnover_chain_unit_not_full_core_or_QoR"
    except Exception as error:
        state.update(status="FAIL",error=repr(error))
    state["end"] = dt.datetime.now().astimezone().isoformat()
    save()
    print(json.dumps(state,indent=2))
    return 0 if state["status"].startswith("PASS") else 1


def launch(name, candidate="candidate_r1"):
    from run_srnn_resident_remote import ROOT,RESULT,REMOTE,VERILATOR,connect,run,upload,fetch_tree,local_path
    from prepare_scnn_forward_turnover import EXPERIMENT
    local=Path(local_path(ROOT/EXPERIMENT/candidate))
    if not (local/"preparation.json").is_file():
        raise ValueError("candidate not prepared")
    dest=REMOTE+"/"+name
    client=connect()
    try:
        probe=r'''
import datetime,json,pathlib,shutil
r=pathlib.Path(ROOT)
if r.exists():raise FileExistsError('unique run destination already exists')
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','xelab','xsim','verilator_bin') or comm.startswith('Vtb_'):active.append([p.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
if active:raise RuntimeError('active EDA: '+repr(active))
free=shutil.disk_usage(r.parent).free
if free<512*1024**2:raise RuntimeError('insufficient unit build space')
r.mkdir()
record=dict(time=datetime.datetime.now().astimezone().isoformat(),active_eda=active,data_free=free,tmp_free=shutil.disk_usage('/tmp').free)
(r/'preflight.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''
        print(run(client,"python3 -c "+shlex.quote("ROOT="+repr(dest)+"\n"+probe)))
        with client.open_sftp() as sftp:
            for path in local.rglob("*"):
                if path.is_file():
                    upload(sftp,str(path),dest+"/"+path.relative_to(local).as_posix())
            upload(sftp,str(Path(__file__).resolve()),dest+"/worker.py")
        try:
            print(run(client,"cd "+shlex.quote(dest)+" && python3 worker.py --worker --verilator "+shlex.quote(VERILATOR)))
        finally:
            with client.open_sftp() as sftp:
                fetch_tree(sftp,dest,ROOT/RESULT/name)
    finally:
        client.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--name",default=NAME)
    parser.add_argument("--candidate",default="candidate_r1")
    parser.add_argument("--verilator",default="verilator")
    args=parser.parse_args()
    if args.worker:
        raise SystemExit(worker(args.verilator))
    launch(args.name,args.candidate)
