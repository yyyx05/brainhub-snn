"""Same-service actual raw SCNN first-frame A/B; compiled caches live in /tmp."""
import argparse
import datetime as dt
import difflib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

NAME = "scnn_turnover_frame_r2"
EXPERIMENT = Path("verification/experiments/scnn_forward_turnover/frame_r2")
BANK_READS = [1152, 73728, 294912, 589824, 1179648, 2359296, 2359296, 2359296, 22528]
SOURCES = ["rtl/learning/" + name + ".sv" for name in (
    "unified_scnn_classifier_cache_refill_publisher_v1",
    "unified_scnn_selected_layers_parameter_backend_v1",
    "unified_tp_three_client_parameter_backend_router")]

MONITOR = r'''
    longint unsigned st_reads = 0, st_responses = 0, st_replacements = 0;
    longint unsigned st_bank_reads [0:8] = '{default:0};
    longint unsigned st_bank_responses [0:8] = '{default:0};
    int st_pending_bank = 0;
    bit st_request_held = 0, st_response_held = 0;
    logic [PARAM_ADDR_WIDTH-1:0] st_held_addr;
    logic [TAG_WIDTH-1:0] st_held_request_tag, st_held_response_tag;
    logic signed [15:0] st_held_code;
    always @(posedge clk) begin : turnover_service_accounting
        bit request_fire, response_fire;
        int bank;
        if (rst_n) begin
            request_fire = ft_o_store_forward_read_valid && ft_i_store_forward_read_ready;
            response_fire = fg_forward_valid && ft_o_store_forward_response_ready;
            if (st_request_held && (!ft_o_store_forward_read_valid ||
                ft_o_store_forward_read_addr !== st_held_addr ||
                ft_o_store_forward_read_tag !== st_held_request_tag))
                $fatal(1,"turnover held request changed");
            if (st_response_held && (!fg_forward_valid || fg_forward_tag !== st_held_response_tag ||
                fg_forward_code !== st_held_code)) $fatal(1,"turnover held response changed");
            if (st_reads - st_responses != 64'(fg_forward_valid))
                $fatal(1,"turnover service credit imbalance");
            if (request_fire && fg_forward_valid && !response_fire)
                $fatal(1,"turnover overwrite of outstanding response");
            // Attribute the retiring response before installing the replacement's bank.
            if (response_fire) begin
                st_responses++; st_bank_responses[st_pending_bank]++;
            end
            if (request_fire) begin
                bank = fg_bank(int'(ft_o_store_forward_read_addr));
                if (bank < 0) $fatal(1,"turnover service unmapped address");
                st_reads++; st_bank_reads[bank]++; st_pending_bank = bank;
            end
            if (request_fire && response_fire) st_replacements++;
            st_request_held = ft_o_store_forward_read_valid && !ft_i_store_forward_read_ready;
            st_response_held = fg_forward_valid && !ft_o_store_forward_response_ready;
            st_held_addr = ft_o_store_forward_read_addr;
            st_held_request_tag = ft_o_store_forward_read_tag;
            st_held_response_tag = fg_forward_tag; st_held_code = fg_forward_code;
        end
    end
    task automatic st_report;
        if (fg_forward_valid || st_reads != st_responses || st_reads != 9239680 ||
            PARAM_ADDR_WIDTH != 24 || TAG_WIDTH != 16)
            $fatal(1,"turnover service final credit/width/count mismatch");
        for (int bank=0; bank<9; bank++) begin
            if (st_bank_reads[bank] == 0 || st_bank_reads[bank] != st_bank_responses[bank])
                $fatal(1,"turnover service bank imbalance");
            $display("SCNN_TURNOVER_BANK bank=%0d reads=%0d responses=%0d",
                bank, st_bank_reads[bank], st_bank_responses[bank]);
        end
        $display("SCNN_TURNOVER_SERVICE_PASS reads=%0d responses=%0d replacements=%0d slots=1 code_bits=16 address_bits=24 tag_bits=16 minimum_latency=1 replaceable=1 m2_cache_depth=512 end_cycle=%0d",
            st_reads, st_responses, st_replacements, cycle_count);
        $fflush();
    endtask
'''


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("ambiguous accepted TB anchor: " + before)
    return text.replace(before, after, 1)


def transform(text):
    clear = "if (fg_forward_valid && ft_o_store_forward_response_ready) fg_forward_valid <= 0;"
    request = "if (ft_o_store_forward_read_valid && ft_i_store_forward_read_ready) begin"
    # Existing sequential ordering gives a new response priority over clearing the old one.
    service_start = text.index("always @(posedge clk) begin : temporal_external_ddr")
    if text.index(clear, service_start) > text.index(request, service_start):
        raise ValueError("host must preserve replacement over old response retirement")
    text = once(text, "assign ft_i_store_forward_read_ready = !fg_forward_valid;",
        "assign ft_i_store_forward_read_ready = !fg_forward_valid || ft_o_store_forward_response_ready;")
    text = once(text, ".TOPOLOGY_MASK(TOPOLOGY_MASK),",
        ".TOPOLOGY_MASK(TOPOLOGY_MASK),\n        .M2_CACHE_DEPTH(512),")
    # Compact integration added a named branch around the unchanged Context0 store.
    for bank in (0, 1):
        text = once(text, f".g_context[0].g_enabled.u_store.g_bank[{bank}].memory[0]",
            f".g_context[0].g_enabled.g_original.u_store.g_bank[{bank}].memory[0]")
    # The old passive bank monitor used a blocking next-bank update before response accounting.
    text = once(text, "pm_traffic[8]++; pm_bank_reads[bank]++; pm_response_bank = bank;",
        "pm_traffic[8]++; pm_bank_reads[bank]++; pm_response_bank <= bank;")
    text = once(text, "        sp_report();\n        $finish;",
        "        sp_report();\n        st_report();\n        $finish;")
    return once(text, "endmodule", MONITOR + "\nendmodule")


def parse_log(log, reference):
    from prepare_scnn_schedule_probe import first_frame_reference, parse_log as parse_probe, rows
    observed = first_frame_reference(log)
    if observed["numeric"] != reference["numeric"]:
        raise ValueError("numeric/reference scope changed")
    if {k:v for k,v in observed["frame"].items() if k != "cycles"} != {
            k:v for k,v in reference["frame"].items() if k != "cycles"}:
        raise ValueError("frame work changed")
    # Reuse all original bucket/ordering/numeric-completion checks, but not old cycle values.
    checks = parse_probe(log, observed)
    service = rows(log, "SCNN_TURNOVER_SERVICE_PASS")
    if len(service) != 1:
        raise ValueError("missing/duplicate native service completion")
    service = {k:int(v) for k,v in service[0].items()}
    expected = dict(reads=9239680, responses=9239680, slots=1, code_bits=16,
        address_bits=24, tag_bits=16, minimum_latency=1, replaceable=1, m2_cache_depth=512)
    if any(service.get(k) != v for k,v in expected.items()):
        raise ValueError("native service count/contract mismatch")
    if not 0 <= service["replacements"] < service["reads"] or service["end_cycle"] <= 0:
        raise ValueError("invalid service measurement")
    banks = [{k:int(v) for k,v in r.items()} for r in rows(log,"SCNN_TURNOVER_BANK")]
    if banks != [dict(bank=i,reads=n,responses=n) for i,n in enumerate(BANK_READS)]:
        raise ValueError("per-bank original raw-frame work mismatch")
    if log.index("SCNN_SCHEDULE_PROBE_PASS") > log.index("SCNN_TURNOVER_SERVICE_PASS"):
        raise ValueError("service completion precedes numeric/bucket checks")
    frame_cycles = int(observed["frame"]["cycles"])
    descriptors = observed["descriptors"]
    if frame_cycles != int(descriptors[-1]["cycle"])-int(descriptors[0]["cycle"])+3:
        raise ValueError("frame/descriptor boundary mismatch")
    for layer, n in enumerate(BANK_READS[:8]):
        pub = checks["tables"]["publisher"][layer]
        hand = checks["tables"]["handshake"][layer]
        if pub["reads"] != n or hand["forward_reads"] != n:
            raise ValueError("raw convolution read coverage changed")
    initialization = rows(log,"SCNN_PROFILE_EVENT")
    starts = [int(r["cycle"]) for r in initialization if r["kind"] == "initialize_start"]
    ends = [int(r["cycle"]) for r in initialization if r["kind"] == "initialize_done"]
    if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
        raise ValueError("initialization measurement missing")
    checks.update(status="PASS_actual_raw_first_frame_same_replaceable_service_not_full_window",
        frame_cycles=frame_cycles, initialization_cycles=ends[0]-starts[0],
        service=service, banks=banks,
        scope="Actual sample0 raw Conv1..8, pool/classifier, learning off, one frame. Identical replaceable latency1 16-bit/one-slot TB host across RTL arms; not DDR, full T20/Test, learning updates, abort, resource or STA proof.")
    return checks


def prepare():
    from run_srnn_resident_remote import ROOT, RESULT, local_path
    root = Path(local_path(ROOT))
    out = root / EXPERIMENT
    out.mkdir(parents=True, exist_ok=False)
    accepted = root / RESULT / "scnn_schedule_probe_r2"
    original = (accepted/"tb_scnn_profile_measurement.sv").read_text(encoding="utf-8")
    changed = transform(original)
    (out/"tb_scnn_profile_measurement.sv").write_text(changed,encoding="utf-8",newline="\n")
    (out/"testbench_changes.diff").write_text("".join(difflib.unified_diff(
        original.splitlines(True),changed.splitlines(True),fromfile="accepted_probe_r2/tb",tofile="same_service_frame/tb")),encoding="utf-8")
    ref = json.loads((accepted/"reference.json").read_text(encoding="utf-8"))
    (out/"reference.json").write_text(json.dumps(ref["expected"]["off"],indent=2)+"\n")
    source = root / RESULT / "m2_compact_synth_r2"
    paths = sorted(p.relative_to(source).as_posix() for p in (source/"rtl").rglob("*") if p.is_file())
    if len(paths) != 109 or sum(p.endswith(".sv") for p in paths) != 108:
        raise ValueError("compact source set incomplete")
    dependencies = [p.relative_to(accepted).as_posix() for folder in ("models","tb","tests","build","results")
        for p in (accepted/folder).rglob("*") if p.is_file()]
    dependencies += ["scripts/supplement_scnn_graph_host.mk"]
    (out/"dependencies.json").write_text(json.dumps(sorted(dependencies),indent=2)+"\n")
    for arm in ("baseline","candidate"):
        dest = out/arm
        shutil.copytree(source/"rtl",dest/"rtl")
        if arm == "candidate":
            for rel in SOURCES:
                shutil.copy2(root/"verification/experiments/scnn_forward_turnover/candidate_r1"/rel,dest/rel)
        for rel in ("tb_scnn_profile_measurement.sv","reference.json"):
            shutil.copy2(out/rel,dest/rel)
        for rel in dependencies:
            target=dest/rel; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(accepted/rel,target)
        command = json.loads((accepted/"scnn_compatibility_build.json").read_text())["command"]
        (dest/"accepted_build.json").write_text(json.dumps(command,indent=2)+"\n")
        source_list = [p for p in paths if p.endswith(".sv")] + ["tb_scnn_profile_measurement.sv"]
        (dest/"sources.f").write_text("".join('"'+p+'"\n' for p in source_list))
    differences=[p for p in paths if (out/"baseline"/p).read_bytes() != (out/"candidate"/p).read_bytes()]
    if differences != sorted(SOURCES):
        raise ValueError("unexpected RTL difference")
    config=dict(status="PREPARED_NOT_SIMULATED",fixture=ref["fixture"],changed_rtl=differences,
        source_files=len(paths),non_rtl_dependencies=len(dependencies),baseline="m2_compact_synth_r2",
        service="Identical replaceable latency1 16-bit one-response-slot host, native address24/tag16; request wins over old response clear.",
        scope="Raw first frame learning-off paired comparison only, no inherited timing or production merge.")
    (out/"preparation.json").write_text(json.dumps(config,indent=2)+"\n")
    print(str(out).removeprefix("\\\\?\\"))


def worker(scratch):
    root=Path.cwd(); scratch=Path(scratch)
    state=dict(status="running",start=dt.datetime.now().astimezone().isoformat(),scratch=str(scratch))
    def save():
        (root/"status.json").write_text(json.dumps(state,indent=2)+"\n")
    def execute(arm,stage,command,timeout):
        state["status"]=arm+"_"+stage+"_running";save()
        started=time.monotonic()
        with (root/arm/(stage+".log")).open("w") as stream:
            try:
                completed=subprocess.run(command,cwd=root/arm,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
            except subprocess.TimeoutExpired:
                state[arm][stage]=dict(command=command,exit=None,timeout_seconds=timeout,wall_seconds=time.monotonic()-started)
                save();raise
        state[arm][stage]=dict(command=command,exit=completed.returncode,wall_seconds=time.monotonic()-started)
        save()
        if completed.returncode:raise RuntimeError(arm+" "+stage+" exit "+str(completed.returncode))
    save()
    try:
        results={}
        for arm in ("baseline","candidate"):
            state[arm]={}
            (scratch/arm).mkdir(parents=True,exist_ok=True)
            command=json.loads((root/arm/"accepted_build.json").read_text())
            command[command.index("--Mdir")+1]=str(scratch/arm/"obj")
            command[command.index("-f")+1]="sources.f"
            command[command.index("-MAKEFLAGS")+1]="OPT_FAST=-O1 OPT_GLOBAL=-O1 -f "+str(root/arm/"scripts/supplement_scnn_graph_host.mk")
            execute(arm,"build",command,5400)
            text=(root/arm/"build.log").read_text(errors="replace")
            state[arm]["ordinary_warning_count"]=len(re.findall(r"^%Warning-",text,re.M))
            state[arm]["structural_warnings"]=re.findall(r"^%Warning-(?:LATCH|MULTIDRIVEN|UNOPTFLAT).*",text,re.M)
            if state[arm]["structural_warnings"] or "%Error" in text:
                raise RuntimeError(arm+" structural build diagnostic")
            config=json.loads((root/"preparation.json").read_text())
            executable=scratch/arm/"obj"/"Vscgraph"
            execute(arm,"run",[str(executable),"+SCNN_PROFILE_FIXTURE="+config["fixture"],"+SCNN_LEARNING=0"],7200)
            results[arm]=parse_log((root/arm/"run.log").read_text(errors="replace"),json.loads((root/arm/"reference.json").read_text()))
            (root/arm/"checks.json").write_text(json.dumps(results[arm],indent=2)+"\n")
            state[arm]["frame_cycles"]=results[arm]["frame_cycles"];save()
        before,after=(results[k] for k in ("baseline","candidate"))
        for kind in ("handshake",):
            if before["tables"][kind] != after["tables"][kind]:raise ValueError("paired work differs")
        if before["banks"] != after["banks"]:raise ValueError("paired banks differ")
        comparison=dict(baseline_frame_cycles=before["frame_cycles"],candidate_frame_cycles=after["frame_cycles"],
            saved_frame_cycles=before["frame_cycles"]-after["frame_cycles"],
            same_service_frame_ratio=before["frame_cycles"]/after["frame_cycles"],
            baseline_initialization_cycles=before["initialization_cycles"],candidate_initialization_cycles=after["initialization_cycles"],
            boundary=after["scope"])
        (root/"comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
        state.update(status="PASS_actual_raw_first_frame_pair_same_service_not_full_window_or_QoR",comparison=comparison)
    except Exception as error:
        state.update(status="FAIL",error=repr(error))
    state["end"]=dt.datetime.now().astimezone().isoformat();save()
    return 0 if state["status"].startswith("PASS") else 1


def launch(name):
    from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, upload, local_path
    if not name.replace("_","").isalnum():raise ValueError("simple unique name required")
    local=Path(local_path(ROOT/EXPERIMENT)); dest=REMOTE+"/"+name; scratch="/tmp/snn_"+name
    if not (local/"preparation.json").is_file():raise ValueError("prepare first")
    client=connect()
    try:
        probe='''import datetime,json,pathlib,shutil
r=pathlib.Path(ROOT);s=pathlib.Path(SCRATCH)
if r.exists() or s.exists():raise FileExistsError('run or scratch already exists')
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','xelab','xsim','verilator_bin','Vscgraph') or comm.startswith('Vtb_'):active.append([p.name,comm])
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
if active:raise RuntimeError('active EDA: '+repr(active))
record=dict(time=datetime.datetime.now().astimezone().isoformat(),active_eda=active,data_free=shutil.disk_usage(r.parent).free,tmp_free=shutil.disk_usage('/tmp').free,scratch=str(s))
if record['data_free']<256*1024**2 or record['tmp_free']<4*1024**3:raise RuntimeError('insufficient raw-evidence/cache space')
r.mkdir();s.mkdir()
(r/'preflight.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''
        record=run(client,"python3 -c "+shlex.quote("ROOT="+repr(dest)+"\nSCRATCH="+repr(scratch)+"\n"+probe))
        print(record)
        with client.open_sftp() as sftp:
            for p in local.rglob("*"):
                if p.is_file():upload(sftp,str(p),dest+"/"+p.relative_to(local).as_posix())
            for filename in ("run_scnn_turnover_frame.py","prepare_scnn_schedule_probe.py","prepare_scnn_resident_compatibility.py"):
                upload(sftp,local_path(ROOT/"scripts/experiments"/filename),dest+"/"+filename)
        command="cd "+shlex.quote(dest)+" && exec python3 run_scnn_turnover_frame.py --worker --scratch "+shlex.quote(scratch)
        run(client,"screen -dmS snn_turnover_frame bash -lc "+shlex.quote(command))
        time.sleep(2)
        observation=json.loads(record)
        observation.update(remote=dest,launched=True,screen="snn_turnover_frame")
        out=Path(local_path(ROOT/RESULT/name));out.mkdir(parents=True,exist_ok=True)
        (out/"launch_observation.json").write_text(json.dumps(observation,indent=2)+"\n")
        print(json.dumps(observation))
    finally:client.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",nargs="?",choices=("prepare","launch"))
    parser.add_argument("--name",default=NAME)
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--scratch")
    args=parser.parse_args()
    if args.worker:raise SystemExit(worker(args.scratch))
    if args.action=="prepare":prepare()
    elif args.action=="launch":launch(args.name)
    else:parser.error("action required")
