#!/usr/bin/env python3
"""Server-side bounded build and simulation with real per-stage exit records."""
import datetime as dt
import json
import difflib
import re
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
VERILATOR = "/root/autodl-tmp/verilator_5_048_install/bin/verilator"
RECORD = ROOT / "status.json"


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(record):
    RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def stage(record, name, command, timeout):
    record.update(status=name + "_running", updated_utc=stamp())
    save(record)
    started = time.monotonic()
    with (ROOT / (name + ".log")).open("x", encoding="utf-8") as log:
        try:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            record[name] = dict(command=command, exit=None, timed_out=True,
                                timeout_seconds=timeout, wall_seconds=time.monotonic()-started)
            save(record)
            raise
    record[name] = dict(command=command, exit=result.returncode, wall_seconds=time.monotonic()-started)
    save(record)
    if result.returncode:
        raise RuntimeError(f"{name} failed, exit {result.returncode}")


def profile(record):
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_srnn_profile_measurement import parse_log, write_measurements, inspect_build_log
    original = json.loads((ROOT / "input_commands.json").read_text(encoding="utf-8"))
    tb_path = ROOT / "tb_srnn_profile_measurement.sv"
    original_tb = tb_path.read_text(encoding="utf-8")
    text = original_tb
    for mode, phase in ((0, "ff"), (1, "r")):
        anchor = f'profile_enter("backbone_refill_{phase}"); refill({mode});'
        if text.count(anchor) != 1:
            raise ValueError("missing refill anchor: " + anchor)
        text = text.replace(anchor, f'profile_enter("backbone_refill_{phase}"); if (timestep == 0) refill({mode});')
    audit = '''
    integer resident_wh1_loads = 0, resident_wout_loads = 0, resident_backbone_loads = 0;
    always @(posedge i_clk) if (i_rst_n) begin
        if (i_srnn_backbone_refill_valid && o_srnn_backbone_refill_ready) resident_backbone_loads++;
        if (`ENG.refill_fire) begin
            if (`ENG.prep_state_q == 1) resident_wh1_loads++;
            if (`ENG.prep_state_q == 3) resident_wout_loads++;
        end
    end
'''
    anchor = "    `undef RED"
    if text.count(anchor) != 1:
        raise ValueError("missing macro audit anchor")
    text = text.replace(anchor, audit + "\n" + anchor)
    anchor = "        profile_finish();"
    if text.count(anchor) != 1:
        raise ValueError("missing profile completion")
    text = text.replace(anchor, '''        if (resident_backbone_loads != 2 || resident_wh1_loads != 1 ||
            resident_wout_loads < 1 || resident_wout_loads > T ||
            (!learning && resident_wout_loads != 1))
            $fatal(1,"residency count mismatch backbone=%0d wh1=%0d wout=%0d",
                resident_backbone_loads,resident_wh1_loads,resident_wout_loads);
        $display("RESIDENCY_PASS backbone_loads=%0d wh1_loads=%0d wout_loads=%0d wout_reloads=%0d",
            resident_backbone_loads,resident_wh1_loads,resident_wout_loads,resident_wout_loads-1);
''' + anchor)
    tb_path.write_text(text, encoding="utf-8")
    (ROOT / "testbench_changes.diff").write_text("".join(difflib.unified_diff(
        original_tb.splitlines(True), text.splitlines(True), fromfile="accepted/tb", tofile="candidate/tb")), encoding="utf-8")
    sources = (ROOT / "sources.f").read_text(encoding="utf-8")
    old_tb = next(line for line in sources.splitlines() if "tb_srnn_profile_measurement.sv" in line)
    sources = sources.replace(old_tb, '"tb_srnn_profile_measurement.sv"')
    sources += '"verification/experiments/srnn_resident_weight_context/rtl/experimental_dual_context_weight_store_p4all_v1.sv"\n'
    (ROOT / "sources.f").write_text(sources, encoding="utf-8")
    build = list(original["build"])
    build[build.index("--Mdir")+1] = "obj"
    build[build.index("-f")+1] = "sources.f"
    build[build.index("-j")+1] = "4"
    record.update(sample_definition=original["sample_definition"], period_ns_projection=15,
                  scope="Candidate fixed sample0 B1T73; initial profile loading included in runtime; no steady-state claim")
    lint = [argument for argument in build if argument != "--binary"]
    lint.insert(1, "--lint-only")
    stage(record, "lint", lint, 180)
    lint_warnings = inspect_build_log((ROOT / "lint.log").read_text(encoding="utf-8", errors="replace"))
    record["lint_warnings"] = lint_warnings
    if lint_warnings["blocking"]:
        raise RuntimeError("structural warnings: " + str(lint_warnings["blocking"]))
    stage(record, "build", build, 3600)
    warnings = inspect_build_log((ROOT / "build.log").read_text(encoding="utf-8", errors="replace"))
    record["build_warnings"] = warnings
    if warnings["blocking"]:
        raise RuntimeError("structural warnings: " + str(warnings["blocking"]))
    for mode in ("off", "on"):
        command = list(original["run"][mode])
        command[0] = "obj/Vtb_srnn_profile_measurement"
        stage(record, "run_"+mode, command, 10800)
        log = (ROOT / ("run_"+mode+".log")).read_text(encoding="utf-8", errors="replace")
        measurements = parse_log(log, record["sample_definition"], 15)
        measurements["simulation_wall_seconds"] = record["run_"+mode]["wall_seconds"]
        output = ROOT / mode
        output.mkdir()
        write_measurements(output, measurements)
        if not measurements["status"].startswith("PASS_") or "RESIDENCY_PASS" not in log:
            raise RuntimeError("incomplete candidate " + mode)
        record[mode+"_cycles"] = measurements["sample_service_cycles_including_refill_excluding_tb_checks"]
        record[mode+"_sample_s_15ns_projection"] = measurements["full_sample_throughput_samples_s_projection"]
        save(record)
    record.update(status="PASS_candidate_fixed_sample_pair_not_physical_QoR", finished_utc=stamp())


def steady(record):
    tb = ROOT / "tb_srnn_profile_measurement.sv"
    original = tb.read_text(encoding="utf-8")
    text = original
    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError("steady TB anchor not unique: " + old)
        text = text.replace(old, new, 1)
    replace("        initialized_cycle = cycles; profile_enter(\"boundary\");", '''        if (learning) $fatal(1,"steady test currently covers inference only");
        for (int sample_idx=0; sample_idx<2; sample_idx++) begin
            longint setup_begin;
            setup_begin = cycles;
            if (sample_idx != 0) begin
                // Independent sequences clear logical state but retain profile weights.
                for (int tile=0; tile<12; tile++) for (int row=0; row<4; row++) begin
                    @(negedge i_clk); i_maintenance_tile = 4'(tile);
                    i_state_maintenance_write_addr = 10'(BONE_BASE+row);
                    i_trace_maintenance_write_addr = 10'(512+BONE_BASE+row);
                    i_state_maintenance_write_valid = 1; i_trace_maintenance_write_valid = 1;
                    do @(posedge i_clk); while (!o_state_maintenance_write_ready || !o_trace_maintenance_write_ready);
                    @(negedge i_clk); i_state_maintenance_write_valid = 0; i_trace_maintenance_write_valid = 0;
                end
                previous_spikes = '0; next_spikes = '0;
                bone_seen=0; hidden_seen=0; logit_seen=0; layers=0; captures=0; replays=0;
                done_count=0; commits=0; detaches=0; code_writes=0; residual_writes=0;
                gradient_seen=0; store_checks=0;
                i_learning_srnn_sequence_version = sample_idx + 1;
            end
        initialized_cycle = cycles; profile_enter("boundary");''')
    replace("if (timestep == 0) refill(0);", "if (sample_idx == 0 && timestep == 0) refill(0);")
    replace("if (timestep == 0) refill(1);", "if (sample_idx == 0 && timestep == 0) refill(1);")
    replace("        profile_finish();\n        $finish;", '''        profile_finish();
        $display("RESIDENT_STEADY_SAMPLE sample=%0d cycles=%0d state_setup_cycles=%0d bone=%0d hidden=%0d logits=%0d store=%0d",
            sample_idx,cycles-initialized_cycle,initialized_cycle-setup_begin,
            bone_seen,hidden_seen,logit_seen,store_checks);
        end
        $display("RESIDENT_STEADY_PASS samples=2");
        $finish;''')
    tb.write_text(text, encoding="utf-8")
    (ROOT / "steady_tb_changes.diff").write_text("".join(difflib.unified_diff(original.splitlines(True), text.splitlines(True), fromfile="accepted/R5_tb", tofile="steady/tb")), encoding="utf-8")
    source_status = json.loads((ROOT / "accepted_source_status.json").read_text(encoding="utf-8"))
    build = source_status["build"]["command"]
    stage(record, "build", build, 3600)
    build_text = (ROOT / "build.log").read_text(encoding="utf-8", errors="replace")
    structural = [name for name in ("LATCH", "MULTIDRIVEN", "UNOPTFLAT") if "%Warning-"+name in build_text]
    record["structural_warnings"] = structural
    if structural:
        raise RuntimeError("structural warnings: " + str(structural))
    command = source_status["run_off"]["command"]
    stage(record, "run", command, 3600)
    log = (ROOT / "run.log").read_text(encoding="utf-8", errors="replace")
    rows = []
    for match in re.finditer(r"^RESIDENT_STEADY_SAMPLE (.+)$", log, re.MULTILINE):
        row = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", match.group(1))}
        if (row["bone"], row["hidden"], row["logits"], row["store"]) != (32850,18688,1460,8783360):
            raise ValueError("incomplete steady sample checks")
        row["sample_s_at_15ns_projection"] = 1e9/(15*row["cycles"])
        row["sample_s_including_state_setup_at_15ns_projection"] = 1e9/(15*(row["cycles"]+row["state_setup_cycles"]))
        rows.append(row)
    if "RESIDENT_STEADY_PASS samples=2" not in log or [row["sample"] for row in rows] != [0,1]:
        raise ValueError("incomplete two-sample sequence")
    record.update(status="PASS_two_inference_sequences_same_profile_not_Test", samples=rows,
                  boundary="Fixed sample0 repeated twice, runtime projection excludes state setup which is separately reported; not learning-on steady state or a diverse batch.", finished_utc=stamp())


def scnn_build_command(accepted_command, host_makefile):
    command = list(accepted_command)
    if command[command.index("--prefix")+1] != "Vscgraph":
        raise ValueError("SCNN host makefile targets the accepted Vscgraph prefix")
    command[command.index("--Mdir")+1] = "obj"
    command[command.index("-f")+1] = "compatibility_sources.f"
    command[command.index("-MAKEFLAGS")+1] = "OPT_FAST=-O1 OPT_GLOBAL=-O1 -f " + host_makefile
    return command


def compatibility(record, workload):
    if workload in ("scnn", "cross_profile", "scnn_probe"):
        if workload == "cross_profile":
            from prepare_srnn_scnn_cross_profile import parse_log
        elif workload == "scnn_probe":
            from prepare_scnn_schedule_probe import parse_log
        else:
            from prepare_scnn_resident_compatibility import parse_log
        config=json.loads((ROOT/"reference.json").read_text())
        accepted=json.loads((ROOT/"scnn_compatibility_build.json").read_text())
        build=scnn_build_command(accepted["command"],str(ROOT/"scripts/supplement_scnn_graph_host.mk"))
        executable="obj/"+build[build.index("--prefix")+1]
        record.update(scope=config["scope"])
        stage(record,"build",build,3600)
        log=(ROOT/"build.log").read_text(errors="replace")
        structural=[name for name in ("LATCH","MULTIDRIVEN","UNOPTFLAT") if "%Warning-"+name in log]
        record["structural_warnings"]=structural
        if structural: raise RuntimeError("structural warnings: "+str(structural))
        for arm in (("off",) if workload == "scnn_probe" else ("off","on")):
            command=[executable,"+SCNN_PROFILE_FIXTURE="+config["fixture"],"+SCNN_LEARNING="+str(int(arm=="on"))]
            if workload == "cross_profile":
                command += ["+RAW="+config["srnn_fixture"], "+FIXTURE="+config["srnn_fixture"]]
            stage(record,"run_"+arm,command,14400)
            result=parse_log((ROOT/("run_"+arm+".log")).read_text(errors="replace"),config["expected"][arm],arm=="on")
            (ROOT/(arm+"_checks.json")).write_text(json.dumps(result,indent=2)+"\n")
            record[arm+"_final"]=result["final"];save(record)
        status = ("PASS_same_DUT_M2_off_SCNN_T4_pair_M2_off_not_DMA_or_throughput"
                  if workload == "cross_profile" else "PASS_SCNN_real_sample0_B1T4_pair_candidate_compatibility")
        if workload == "scnn_probe":
            status = "PASS_SCNN_first_frame_passive_attribution_not_full_window"
        record.update(status=status,finished_utc=stamp())
        return
    top = "tb_fc_profile_measurement" if workload == "fc" else "tb_frontend_final_top_smoke"
    build = [VERILATOR, "--binary", "--timing", "--assert", "-j", "2", "-Wall", "-Wno-fatal",
             "--output-split", "12000", "--output-split-cfuncs", "0", "-CFLAGS", "-O1",
             "--top-module", top, "--Mdir", "obj", "-f", "compatibility_sources.f"]
    stage(record, "build", build, 900)
    log = (ROOT / "build.log").read_text(encoding="utf-8", errors="replace")
    structural = [name for name in ("LATCH", "MULTIDRIVEN", "UNOPTFLAT") if "%Warning-"+name in log]
    record["structural_warnings"] = structural
    if structural:
        raise RuntimeError("structural warnings: " + str(structural))
    if workload == "fc":
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_fc_profile_measurement import parse_log
        config = json.loads((ROOT / "compatibility_input.json").read_text(encoding="utf-8"))
        record.update(scope=config["scope"], source_candidate="srnn_specialized_r1")
        overlay = ROOT / "fc_scanner_overlay.json"
        if overlay.is_file():
            record["fc_scanner_overlay"] = json.loads(overlay.read_text(encoding="utf-8"))
        for arm in ("off", "on"):
            command = list(config["runs"][arm])
            command[0] = "obj/V"+top
            stage(record, "run_"+arm, command, 1800)
            log = (ROOT / ("run_"+arm+".log")).read_text(encoding="utf-8", errors="replace")
            result = parse_log(log, arm, config["scope"], config["activity_monitor"])
            result["simulation_wall_seconds"] = record["run_"+arm]["wall_seconds"]
            (ROOT / (arm+"_measurements.json")).write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
            record[arm+"_cycles"] = result["runtime_cycles"]
        record.update(status="PASS_FC_B16T99_pair_candidate_compatibility", finished_utc=stamp())
        if overlay.is_file():
            record["status"] = "PASS_FC_B16T99_pair_scanner_pipeline_not_physical_QoR"
    else:
        sys.path.insert(0, str(ROOT / "software/autonomous_reconfiguration"))
        from snn_reconfig.final_top_smoke_bridge import check_transcript
        commands = []
        for line in (ROOT / "program.txt").read_text(encoding="ascii").splitlines():
            values = list(map(int, line.split()))
            if len(values) < 3 or values[2] != len(values)-3:
                raise ValueError("malformed existing frontend program")
            commands.append((values[0], values[1], tuple(values[3:])))
        stage(record, "run", ["obj/V"+top, "+program=program.txt"], 30)
        result = check_transcript((ROOT / "run.log").read_text(encoding="utf-8"), commands)
        (ROOT / "transcript_checks.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        if result["status"] != "PASS":
            raise ValueError("candidate native-switch replay checks failed")
        record.update(status="PASS_small_final_top_profile_switch_only", commands=len(commands),
                      boundary="Learning-off synthetic FC/SRNN backbone/SCNN B2T2 with TB services; not formal M2 dispatch, real SCNN weights or DDR/asset DMA.",
                      finished_utc=stamp())


def multisequence(record):
    original = json.loads((ROOT / "accepted_source_status.json").read_text(encoding="utf-8"))
    expected = json.loads((ROOT / "multi_fixture/reference.json").read_text(encoding="utf-8"))
    build = list(original["build"]["command"])
    build[build.index("-j")+1] = "2"
    stage(record, "build", build, 900)
    log = (ROOT / "build.log").read_text(encoding="utf-8", errors="replace")
    structural = [name for name in ("LATCH", "MULTIDRIVEN", "UNOPTFLAT") if "%Warning-"+name in log]
    record["structural_warnings"] = structural
    if structural:
        raise RuntimeError("structural warnings: " + str(structural))
    for arm in ("off", "on"):
        command = ["obj/Vtb_srnn_profile_measurement", "+RAW=multi_fixture", "+FIXTURE=multi_fixture", "+LEARNING="+str(int(arm=="on"))]
        stage(record, "run_"+arm, command, 1800)
        log = (ROOT / ("run_"+arm+".log")).read_text(encoding="utf-8", errors="replace")
        rows = []
        for match in re.finditer(r"^RESIDENT_NATIVE_SEQUENCE (.+)$", log, re.MULTILINE):
            row = {k:int(v) for k,v in re.findall(r"(\w+)=(\d+)", match.group(1))}
            if row.get("sequence") != len(rows) or len(rows) >= 2:
                raise ValueError("unexpected native sequence order")
            wanted = expected["expected"][arm][len(rows)]
            if any(row.get(k) != wanted[k] for k in ("sequence", "steps", "bone", "hidden", "logits", "store", "code", "residual", "commits")):
                raise ValueError("native sequence counters differ: "+str(row))
            if row["backbone_loads"] != 2 or row["wh1_loads"] != 1:
                raise ValueError("backbone/WH1 weights were reloaded")
            row["sample_s_at_15ns_projection"] = 1e9/(15*row["cycles"])
            row["sample_s_including_state_setup_at_15ns_projection"] = 1e9/(15*(row["cycles"]+row["state_setup_cycles"]))
            rows.append(row)
        marker = f"RESIDENT_NATIVE_PAIR_PASS sequences=2 learning={int(arm=='on')} steps={sum(expected['native_steps'])}"
        if len(rows) != 2 or marker not in log:
            raise ValueError("missing native two-sequence completion")
        record[arm+"_sequences"] = rows
        save(record)
    record.update(status="PASS_two_different_native_sequences_pair_not_Test", finished_utc=stamp(),
        scope="Official samples 0/1, native T73/T78; no global reset or parameter reload between independent sequences. Learning codes/residuals persist; dynamics clear through existing interfaces. 15ns projections pending physical QoR.")


def parse_recovery_log(log, arm, expected):
    accepted = re.findall(r"^RESIDENT_ABORT_ACCEPTED (.+)$", log, re.MULTILINE)
    complete = re.findall(r"^RESIDENT_ABORT_RECOVERY_PASS (.+)$", log, re.MULTILINE)
    if len(accepted) != 1 or len(complete) != 1:
        raise ValueError("missing unique abort/recovery completions")
    before = {k:int(v) for k,v in re.findall(r"(\w+)=(\d+)", accepted[0])}
    wanted = dict(learning=int(arm == "on"), aborts=1, bone=450, hidden=0, logits=0, gradients=0,
                  code=0, residual=0, commits=0, unchanged_parameters=120320)
    if before != wanted:
        raise ValueError("unexpected traffic or parameter change at abort")
    row = {k:int(v) for k,v in re.findall(r"(\w+)=(\d+)", complete[0])}
    if row != expected:
        raise ValueError("recovered numerical/update/reset/load counts differ: "+str(row))
    return dict(aborted=before, recovered=row)


def recovery(record):
    source = json.loads((ROOT / "accepted_source_status.json").read_text(encoding="utf-8"))
    reference = json.loads((ROOT / "recovery/reference.json").read_text(encoding="utf-8"))
    build = list(source["build"]["command"])
    build[build.index("-j")+1] = "2"
    stage(record, "build", build, 1200)
    build_text = (ROOT / "build.log").read_text(encoding="utf-8", errors="replace")
    record["structural_warnings"] = [n for n in ("LATCH", "MULTIDRIVEN", "UNOPTFLAT") if "%Warning-"+n in build_text]
    if record["structural_warnings"]:
        raise RuntimeError("structural build warning")
    for arm in ("off", "on"):
        command = list(source["run_"+arm]["command"])
        command[0] = "obj/Vtb_srnn_profile_measurement"
        stage(record, "run_"+arm, command, 600)
        log = (ROOT / ("run_"+arm+".log")).read_text(encoding="utf-8", errors="replace")
        record[arm] = parse_recovery_log(log, arm, reference["expected"][arm])
        save(record)
    record.update(status="PASS_final_top_abort_before_learning_retry_first_step_pair", scope=reference["scope"], finished_utc=stamp())


def main():
    action = sys.argv[1]
    record = dict(status="starting", started_utc=stamp(), action=action)
    save(record)
    try:
        if action == "profile":
            profile(record)
            save(record)
            return
        if action == "steady":
            steady(record)
            save(record)
            return
        if action in ("fc", "frontend", "scnn", "cross_profile", "scnn_probe"):
            compatibility(record, action)
            save(record)
            return
        if action == "multisequence":
            multisequence(record)
            save(record)
            return
        if action == "recovery":
            recovery(record)
            save(record)
            return
        if action in ("control", "store_modes"):
            top = "tb_srnn_resident_control" if action == "control" else "tb_srnn_resident_store_modes"
            build = [VERILATOR, "--binary", "--timing", "--assert", "-j", "2", "-Wall", "-Wno-fatal",
                     "--top-module", top, "--Mdir", "obj", "-f", "control_sources.f", top+".sv"]
            stage(record, "build", build, 600)
            log = (ROOT / "build.log").read_text(encoding="utf-8", errors="replace")
            structural = [name for name in ("LATCH", "MULTIDRIVEN", "UNOPTFLAT") if "%Warning-"+name in log]
            record["structural_warnings"] = structural
            if structural:
                raise RuntimeError("structural warnings: " + str(structural))
            stage(record, "run", ["obj/V"+top], 30)
            marker = "RESIDENT_CONTROL_PASS scenarios=7 " if action == "control" else "RESIDENT_STORE_MODES_PASS checks=1120"
            if marker not in (ROOT / "run.log").read_text(encoding="utf-8"):
                raise RuntimeError("missing focused test completion")
            record.update(status="PASS_cache_control_only" if action == "control" else "PASS_store_modes_only", finished_utc=stamp())
            save(record)
            return
        if action != "unit":
            raise ValueError(action)
        experiment = "verification/experiments/srnn_resident_weight_context"
        top = "tb_experimental_dual_context_weight_store_p4all_v1"
        build = [VERILATOR, "--binary", "--timing", "--assert", "-j", "2", "-Wall", "-Wno-fatal",
                 "--top-module", top, "--Mdir", "obj",
                 "rtl/memory/unified_dual_role_weight_store_p4all_candidate_v1.sv",
                 experiment + "/rtl/experimental_dual_context_weight_store_p4all_v1.sv",
                 experiment + "/tb/" + top + ".sv"]
        stage(record, "build", build, 600)
        build_text = (ROOT / "build.log").read_text(encoding="utf-8", errors="replace")
        structural = [name for name in ("LATCH", "MULTIDRIVEN", "UNOPTFLAT") if "%Warning-" + name in build_text]
        record["structural_warnings"] = structural
        if structural:
            raise RuntimeError("structural warnings: " + str(structural))
        stage(record, "run", ["obj/V" + top], 30)
        if "DUAL_CONTEXT_PASS" not in (ROOT / "run.log").read_text(encoding="utf-8"):
            raise RuntimeError("missing unit completion marker")
        record.update(status="PASS_unit_only", finished_utc=stamp())
    except Exception as error:
        record.update(status="FAIL", error=str(error), finished_utc=stamp())
        save(record)
        raise
    save(record)


if __name__ == "__main__":
    main()
