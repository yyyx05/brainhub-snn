#!/usr/bin/env python3
"""Directed sample0 B1T20 performance; default files only, no automatic retry."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

import prepare_scnn_profile_measurement as prep
import run_scnn_real_final_graph as base
from run_fc_final_counts import resolve_verification_store

ROOT, BASE, TOP = prep.ROOT, prep.BASE, prep.TOP
DEFAULT_FIXTURE = BASE / "fixed_sample0_b1t20_r1"
CLOSURE = ROOT / "build/ip/unified_snn_native_ip_source_20261009_r2/reachable_sources.f"
ROM = ROOT / "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem"
PASS = "SCNN_PROFILE_PASS"


def production_sources():
    return list(dict.fromkeys(ROOT / line.strip().strip('"')
                             for line in CLOSURE.read_text(encoding="ascii").splitlines() if line.strip()))


def sources(tb, activity=False):
    result = [*production_sources(), resolve_verification_store(ROOT), tb]
    if activity:
        result.append(ROOT / "tb/monitors/supplement_h40_activity_monitor.sv")
    return list(dict.fromkeys(result))


def dependencies(fixture, activity=False):
    paths = [*production_sources(), resolve_verification_store(ROOT), CLOSURE, ROM,
             prep.b1.SOURCE, prep.b1.BODY, prep.b1.CHECKS, prep.temporal.BODY, prep.temporal.CHECKS,
             prep.BODY, prep.MONITOR, Path(__file__).resolve()]
    paths += [ROOT / "scripts" / name for name in (
        "prepare_scnn_profile_measurement.py", "run_fc_final_counts.py", "run_scnn_real_final_graph.py",
        "generate_scnn_real_final_graph.py", "generate_scnn_real_final_b4t12.py",
        "supplement_scnn_same_dut_reference.py", "supplement_scnn_graph_reference.py",
        "analyze_scnn_b4t4_rtl_update_contract.py", "supplement_scnn_graph_host.mk")]
    paths += [ROOT / f"models/scnn_dvsgesture_vgg9_q0_15/bank_{bank:02d}.hex"
              for bank in (*range(1, 9), 12)]
    paths.append(ROOT / "tests/test_scnn_profile_measurement.py")
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="ascii"))
    paths += [fixture / "manifest.json", *(fixture / entry["path"] for entry in manifest["files"])]
    if activity:
        paths.append(ROOT / "tb/monitors/supplement_h40_activity_monitor.sv")
    paths = sorted(set(paths))
    missing = [p.relative_to(ROOT).as_posix() for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing measurement dependencies: " + ", ".join(missing))
    return paths


def prepare(output, fixture, activity=False):
    paths = dependencies(fixture, activity)
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="ascii"))
    if (manifest["sample"], manifest["batch"], manifest["timesteps"], manifest["window_timesteps"],
        manifest["multiplier"], manifest["time_bases"]) != (0, 1, 20, 4, 640, [0, 4, 8, 12, 16]):
        raise ValueError("use the fixed original sample0 B1T20/T4/640 fixture")
    text = prep.testbench_source()
    output.mkdir(parents=True, exist_ok=False)
    tb = output / (TOP+".sv")
    tb.write_text(text, encoding="ascii")
    selected = sources(tb, activity)
    (output / "sources.f").write_text("".join('"'+p.as_posix()+'"\n' for p in selected), encoding="ascii")
    (output / "dependencies.txt").write_text("".join(p.relative_to(ROOT).as_posix()+"\n" for p in paths), encoding="ascii")
    record = dict(status="prepared_not_compiled_not_simulated", top="unified_h40_final_composition_top_v1",
                  fixture=fixture.relative_to(ROOT).as_posix(), sample=0, batch=1, timesteps=20,
                  windows=[dict(base=t, count=4, batch=1) for t in (0, 4, 8, 12, 16)],
                  activity_monitor=activity, production_sv_files=len(production_sources()),
                  source_files=len(selected), dependency_files=len(paths),
                  dependency_bytes=sum(p.stat().st_size for p in paths),
                  expected_host_hours_per_arm=7.5,
                  estimate_basis="20/3 times archived current-final-top B1T2+fresh-T0 host wall 4066.994/4069.337s; not a T20 measurement; new monitor/host contention overhead unknown",
                  not_measured=["RTL T20 completion", "board/DDR timing", "golden/file-read host CPU breakdown", "full dataset", "new PPA"])
    (output / "preparation.json").write_text(json.dumps(record, indent=2)+"\n", encoding="ascii")
    return record


def rows(log, marker):
    return [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in log.splitlines() if line.startswith(marker+" ")]


def integer_rows(log, marker):
    return [{k: int(v) for k, v in row.items()} for row in rows(log, marker)]


def one(log, marker):
    result = integer_rows(log, marker)
    if len(result) != 1:
        raise ValueError("missing/duplicate " + marker)
    return result[0]


def check_build_log(log):
    structural = [name for name in ("UNOPTFLAT", "LATCH", "MULTIDRIVEN") if "%Warning-"+name in log]
    if structural:
        raise RuntimeError("structural build warnings: " + ", ".join(structural) +
                           "; inspect preserved build.log before simulation")


def elab_command(output, verilator="verilator", jobs=2):
    command = base.build_command(output, verilator, jobs)
    command[command.index("--binary")] = "--lint-only"
    command[command.index(base.TOP)] = TOP
    command[command.index("--Mdir")+1] = str(output / "elab_obj")
    for option in ("--output-groups", "--output-split", "--output-split-cfuncs", "-MAKEFLAGS"):
        index = command.index(option)
        del command[index:index+2]
    return command


def parse_result(log, expected, learning, *, activity=False):
    if re.search(r"%Error|%Fatal|Assertion failed", log):
        raise ValueError("simulator error in preserved log")
    mode, enabled = ("on" if learning else "off"), int(learning)
    final = one(log, PASS)
    exact = dict(learning=enabled, samples=1, frames=20, descriptors=160, requests=5,
                 features=40960, currents=220, logits=220, traces=40960*enabled,
                 probes=480, commits=5*enabled, publications=1892*(1+5*enabled),
                 rmw=sum(w["rmw"] for w in expected["windows"][mode]))
    if any(final.get(k) != v for k, v in exact.items()):
        raise ValueError("incomplete fixed sample0 T20 numeric completion")
    frames = integer_rows(log, "SCNN_PROFILE_FRAME")
    if [r.get("frame") for r in frames] != list(range(20)) or any(r.get("sample") != 0 or r.get("timestep") != r["frame"] for r in frames):
        raise ValueError("missing/duplicate/out-of-order T20 frames")
    descriptor_rows = rows(log, "SCNN_PROFILE_DESCRIPTOR")
    descriptors = []
    for frame in range(20):
        for layer in range(1, 9):
            pair = descriptor_rows[len(descriptors)*2:len(descriptors)*2+2]
            if len(pair) != 2 or [r["edge"] for r in pair] != ["start", "end"] or any(
                    (int(r["frame"]), int(r["layer"])) != (frame, layer) for r in pair):
                raise ValueError("descriptor start/end ordering")
            start, end = (int(r["cycle"]) for r in pair)
            if end <= start:
                raise ValueError("invalid descriptor interval")
            descriptors.append(dict(frame=frame, layer=layer, start=start, end=end, cycles=end-start))
    if len(descriptor_rows) != 320:
        raise ValueError("extra descriptor intervals")
    timing, active, traffic = (one(log, marker) for marker in
                               ("SCNN_PROFILE_TIMING", "SCNN_PROFILE_ACTIVE", "SCNN_PROFILE_TRAFFIC"))
    if (timing["sample_cycles"] <= 0 or timing["sample_cycles"] != timing["sample_end"]-timing["sample_start"] or
        timing["setup_cycles"] != timing["sample_start"]-timing["reset"] or
        timing["included_setup_cycles"] != timing["sample_end"]-timing["reset"] or
        timing["sample_cycles"]+timing["setup_cycles"] != timing["included_setup_cycles"]):
        raise ValueError("inconsistent cycle boundaries")
    for read, response in (("state_spill_requests", "state_spill_rows"),
                            ("trace_spill_requests", "trace_spill_rows"), ("forward_reads", "forward_responses"),
                            ("learning_reads", "learning_writes"), ("learning_reads", "learning_responses")):
        if traffic[read] != traffic[response]:
            raise ValueError("undrained measured traffic")
    if any(traffic[k] <= 0 for k in ("state_restore_rows", "trace_restore_rows", "state_spill_rows", "trace_spill_rows")):
        raise ValueError("missing measured maintenance words")
    if traffic["learning_writes"] != exact["rmw"] or traffic["store_loads"] != 0:
        raise ValueError("unexpected parameter writes/loads")
    banks = integer_rows(log, "SCNN_PROFILE_DDR")
    if [r["bank"] for r in banks] != list(range(9)) or any(r["forward_reads"] <= 0 or r["forward_reads"] != r["forward_responses"] for r in banks):
        raise ValueError("missing actual DDR-model bank handshakes")
    if sum(r["forward_reads"] for r in banks) != traffic["forward_reads"]:
        raise ValueError("bank/aggregate traffic mismatch")
    control = integer_rows(log, "SCNN_PROFILE_CONTROL")
    if [r["frame"] for r in control] != list(range(20)) or any(
            r[name+"_first"] <= 0 or r[name+"_end"] <= r[name+"_first"]
            for r in control for name in ("conv8", "pool", "classifier")):
        raise ValueError("missing Conv8/pool/classifier control intervals")
    events, intervals = rows(log, "SCNN_PROFILE_EVENT"), rows(log, "SCNN_PROFILE_INTERVAL")
    for kind, windows in (("window_request", list(range(1, 6))), ("window_done", list(range(1, 6))),
                          ("commit", list(range(1, 6)) if learning else []),
                          ("publish_request", list(range(1, 6)) if learning else []),
                          ("publish_done", list(range(1, 6)) if learning else []),
                          ("initialize_start", [0]), ("initialize_done", [0])):
        if [int(r["window"]) for r in events if r["kind"] == kind] != windows:
            raise ValueError("missing/duplicate phase events: " + kind)
    for row in intervals:
        if int(row["end"]) < int(row["start"]):
            raise ValueError("reversed phase interval")
    if len([r for r in intervals if r["kind"] == "commit_pending_to_pulse"]) != 5*enabled:
        raise ValueError("missing commit latency intervals")
    if any(active[k] > timing["sample_cycles"] for k in active):
        raise ValueError("active cycles exceed measured sample interval")
    if active["forward_learning_overlap"] > min(active["forward"], active["learning"]) or active["learning_publication_overlap"] > min(active["learning"], active["publication"]):
        raise ValueError("invalid overlapping cycle counters")
    if not learning and any(active[k] for k in ("learning", "learning_exclusive", "publication")):
        raise ValueError("learning-off activity after initialization")
    payload = dict(state_restore_bytes=traffic["state_restore_rows"]*40,
                   trace_restore_bytes=traffic["trace_restore_rows"]*40,
                   state_spill_bytes=traffic["state_spill_rows"]*40,
                   trace_spill_bytes=traffic["trace_spill_rows"]*40,
                   state_restore_scalar16_words=traffic["state_restore_rows"]*20,
                   trace_restore_scalar16_words=traffic["trace_restore_rows"]*20,
                   state_spill_scalar16_words=traffic["state_spill_rows"]*20,
                   trace_spill_scalar16_words=traffic["trace_spill_rows"]*20,
                   forward_response_code_bytes=traffic["forward_responses"]*2,
                   learning_read_code_residual_bytes=traffic["learning_responses"]*4,
                   enabled_learning_write_bytes=(traffic["code_write_words"]+traffic["residual_write_words"])*2)
    h40 = rows(log, "H40_ACTIVITY")
    h40_modes = rows(log, "H40_ACTIVITY_MODE")
    h40_summary = None
    if activity:
        if len(h40) != 12 or len({r["tile"] for r in h40}) != 12:
            raise ValueError("missing actual twelve-tile passive monitor records")
        buckets = ("abort", "compute", "issue_wait", "region_wait", "observation_wait", "fill", "internal", "idle")
        if any(sum(int(r[k]) for k in buckets) != int(r["cycles"]) for r in h40):
            raise ValueError("incomplete exclusive priority activity buckets")
        if len(h40_modes) != 48 or len({(r["tile"], int(r["mode"])) for r in h40_modes}) != 48:
            raise ValueError("missing per-tile mode activity")
        h40_summary = {k: sum(int(r[k]) for r in h40) for k in
                       ("cycles", "issue_fires", "fill_fires", "contributions", "lane_fires", "enabled_terms", *buckets)}
        h40_summary["scope"] = "reset-deassertion to simulator finish, including setup; not sample-only"
        h40_summary["lane_occupancy"] = h40_summary["lane_fires"]/(40*h40_summary["cycles"])
        h40_summary["p4_term_occupancy"] = h40_summary["enabled_terms"]/(160*h40_summary["cycles"])
    return dict(status="PASS", learning=learning, final=final, timing=timing, active=active,
                traffic=traffic, payload=payload, banks=banks, descriptors=descriptors,
                control=control, events=events, intervals=intervals, h40=h40,
                h40_modes=h40_modes, h40_summary=h40_summary,
                throughput=dict(
                    simulated_10ns_samples_per_second_excluding_setup=1e8/timing["sample_cycles"],
                    simulated_10ns_samples_per_second_including_setup=1e8/timing["included_setup_cycles"],
                    projected_15ns_samples_per_second_excluding_setup=1e9/(15*timing["sample_cycles"]),
                    projected_15ns_samples_per_second_including_setup=1e9/(15*timing["included_setup_cycles"])),
                caveats=["one fixed original sample0 T20 only; B1/T4 underfilled, no full-dataset/adaptation benefit claim",
                         "setup is reset-to-first-request and initial classifier publication; host readmemh of model/oracles takes zero simulated cycles, not a DMA measurement",
                         "pool_control is Conv8 writeback busy including collection; Conv client includes writeback; classifier_execute excludes tail capture; intervals overlap, do not add them as exclusive stages",
                         "learning is gradient/update busy, exclusive removes forward/publication overlap; not all learning subsystem cost",
                         "golden spans count simulated cycles only; host checking/file I/O cost remains in simulator wall seconds and is not independently timed",
                         "native DDR model counts/payload only; address/tag/control bytes and real DDR latency/bandwidth not measured",
                         "15ns is a projection, not new timing closure; no RUN136/137 source/PPA change"])


def summarize(output):
    preparation = json.loads((output / "preparation.json").read_text(encoding="ascii"))
    fixture = ROOT / preparation["fixture"]
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="ascii"))
    records = {}
    for mode in ("off", "on"):
        log = output / ("run_"+mode+".log")
        status = json.loads(log.with_suffix(".status.json").read_text(encoding="ascii"))
        if status["exit_code"] != 0:
            raise ValueError("nonzero simulator exit: " + mode)
        row = parse_result(log.read_text(encoding="utf-8", errors="replace"), manifest,
                           mode == "on", activity=preparation["activity_monitor"])
        row["host_simulator_wall_seconds"] = status["seconds"]
        row["host_file_load_seconds"] = None
        row["host_golden_seconds"] = None
        records[mode] = row
    summary = dict(status="PASS", actual_final_top=True, fixed_sample_only=True, measurements=records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="ascii")
    with (output / "sample_rates.csv").open("w", newline="", encoding="ascii") as stream:
        fields = ["learning", "sample_cycles", "setup_cycles", "included_setup_cycles", *records["off"]["throughput"]]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for mode, record in records.items():
            writer.writerow(dict(learning=mode, **{k: record["timing"][k] for k in fields[1:4]}, **record["throughput"]))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("files", "prepare", "elab", "build", "run", "summarize"), default="files")
    parser.add_argument("--run-name", default="r1")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--verilator", default="/root/autodl-tmp/verilator_5_048_install/bin/verilator")
    parser.add_argument("--learning", choices=("off", "on", "both"), default="both")
    parser.add_argument("--activity-monitor", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("simple fresh run-name required")
    fixture, output = args.fixture.resolve(), BASE / args.run_name
    if args.stage == "files":
        print("\n".join(p.relative_to(ROOT).as_posix() for p in dependencies(fixture, args.activity_monitor)))
    elif args.stage == "prepare":
        print(json.dumps(prepare(output, fixture, args.activity_monitor), indent=2))
    elif args.stage == "elab":
        base.run_logged(elab_command(output, args.verilator, args.jobs), output / "elab.log")
        check_build_log((output / "elab.log").read_text(encoding="utf-8", errors="replace"))
    elif args.stage == "build":
        if sys.platform == "win32":
            parser.error("full-shape final-top compilation is remote-only; prepare locally")
        command = base.build_command(output, args.verilator, args.jobs)
        command[command.index(base.TOP)] = TOP
        base.run_logged(command, output / "build.log")
        check_build_log((output / "build.log").read_text(encoding="utf-8", errors="replace"))
    elif args.stage == "run":
        if sys.platform == "win32":
            parser.error("full T20 simulation is remote-only")
        preparation = json.loads((output / "preparation.json").read_text(encoding="ascii"))
        for mode in (("off", "on") if args.learning == "both" else (args.learning,)):
            base.run_logged([str(output / "obj/Vscgraph"),
                             "+SCNN_PROFILE_FIXTURE="+preparation["fixture"],
                             "+SCNN_LEARNING="+str(int(mode == "on"))], output / ("run_"+mode+".log"))
    else:
        print(json.dumps(summarize(output), indent=2))


if __name__ == "__main__":
    main()
