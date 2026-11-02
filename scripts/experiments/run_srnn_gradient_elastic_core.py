"""Measure one isolated elastic-reducer arm on the final raw-input wrapper."""
import argparse
import difflib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
NAME = "srnn_gradient_elastic_core_r1"
PRIOR = "final_raw_source_functional_r1"
MODULE = "rtl/learning/unified_srnn_tp_wout_gradient_reducer.sv"
SCOPE = (
    "One elastic-reducer change on the accepted final raw-source wrapper. Same eight "
    "sequences/588 closed-loop steps, independent integer assertions, producer, M2=512, "
    "single-outstanding RMW and source boundary. Accepted prior arm not rerun. "
    "No new distribution, algorithm advantage, author-system comparison or physical evidence."
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def roots():
    from run_srnn_resident_remote import ROOT, RESULT, local_path
    root = Path(local_path(ROOT))
    return root, root / "verification/experiments" / NAME, root / RESULT / NAME


def activity(log):
    rows = {}
    for line in re.findall(r"^H40_ACTIVITY (.*)$", log, re.M):
        parts = dict(re.findall(r"(\w+)=([^\s]+)", line))
        tile = parts.pop("tile")
        if tile in rows:
            raise ValueError("duplicate tile")
        rows[tile] = {k: int(v) for k, v in parts.items() if k not in ("cycles", "idle")}
    if len(rows) != 12:
        raise ValueError("tile coverage")
    return rows


def numerical_records(parsed):
    result = json.loads(json.dumps(parsed))
    for key in ("service_cycles", "all_sequences_per_second_15ns_projection",
                "warm_sequences_per_second_with_setup_15ns_projection"):
        result.pop(key)
    result["total"].pop("cycles")
    for row in result["sequences"]:
        row.pop("cycles")
    return result


def parse_log(log, info, expected, directory):
    load("run_srnn_representative_inputs", directory / "run_srnn_representative_inputs.py")
    load("source_core_parser", directory / "source_core_parser.py")
    parser = load("accepted_core_parser", directory / "accepted_core_parser.py")
    parsed = parser.parse_log(log, info)
    if log.count("FINAL_RAW_WRAPPER_PASS policy=1 sequences=8 steps=588") != 1:
        raise ValueError("wrapper completion")
    if numerical_records(parsed) != numerical_records(expected):
        raise ValueError("non-latency numeric/traffic/initialization record changed")
    if activity(log) != json.loads((directory / "accepted_tile_activity.json").read_text()):
        raise ValueError("tile activity changed beyond total/idle cycles")
    rows = re.findall(r"^GRADIENT_ELASTIC_CORE_PASS (.*)$", log, re.M)
    if len(rows) != 1:
        raise ValueError("elastic coverage marker")
    counts = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", rows[0])}
    if set(counts) != {"inputs", "outputs", "finalize_inputs", "held_inputs", "double_holds"}:
        raise ValueError("elastic coverage fields")
    if counts["inputs"] != 3010560 or counts["outputs"] != 3010560 or counts["finalize_inputs"] <= 0:
        raise ValueError("elastic traffic/overlap coverage")
    # Latency is observed, not a pass condition: regressions must remain visible.
    return parsed


def configure(directory=None):
    if directory is None:
        directory = roots()[1]
    common = load("elastic_wrapper_common", directory / "elastic_wrapper_common.py")
    common.NAME, common.PRIOR, common.SCOPE = NAME, PRIOR, SCOPE
    common.parse_log = parse_log
    return common


def prepare(resume=False):
    root, out, archive = roots()
    prior = root / "verification/experiments" / PRIOR
    unit = root / "verification/experiments/gradient_elastic_r1"
    unit_result = archive.parent / "gradient_elastic_local_r1"
    if archive.exists() or (out.exists() and not resume):
        raise FileExistsError("new run identity already used")
    if resume and (not out.exists() or (out / "elastic_wrapper_common.py").exists()):
        raise ValueError("only resume the incomplete, never-launched preparation")
    accepted = json.loads((unit_result / "status.json").read_text())
    if accepted["status"] != "PASS_two_stage_elastic_reducer_local":
        raise ValueError("unit not accepted")
    if (unit / "original" / Path(MODULE).name).read_bytes() != (prior / MODULE).read_bytes():
        raise ValueError("unit baseline differs from actual wrapper")
    for arm in ("original", "candidate"):
        if (unit / arm / Path(MODULE).name).read_bytes() != (unit_result / arm / Path(MODULE).name).read_bytes():
            raise ValueError("tested unit source differs")
    if not resume:
        shutil.copytree(prior, out, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(unit / "candidate" / Path(MODULE).name, out / MODULE)
    shutil.copy2(unit / "changes.diff", out / "gradient_elastic.diff")
    shutil.copy2(unit_result / "status.json", out / "gradient_elastic_unit_acceptance.json")
    tb = out / "tb_srnn_profile_measurement.sv"
    old = tb.read_text()
    # Monitor only; no input driver or existing assertion is altered.
    # Reuse the TB's existing hierarchy aliases, including generate branches.
    m2 = re.findall(r"^\s*`define M2 (\S+)$", old, re.M)
    red = re.findall(r"^\s*`define RED (\S+)$", old, re.M)
    if len(m2) != 1 or len(red) != 1 or not red[0].startswith("`M2."):
        raise ValueError("existing reducer monitor aliases missing")
    engine = red[0].replace("`M2", m2[0], 1) + ".u_wout_reducer"
    monitor = f"""
    longint elastic_inputs = 0, elastic_outputs = 0;
    longint elastic_finalize_inputs = 0, elastic_held_inputs = 0, elastic_double_holds = 0;
    always @(posedge i_clk) if (i_rst_n) begin
        if ({engine}.input_fire) elastic_inputs++;
        if ({engine}.output_fire) elastic_outputs++;
        if ({engine}.input_fire && {engine}.finalize_q) elastic_finalize_inputs++;
        if ({engine}.input_fire && {engine}.pending_q && !{engine}.i_gradient_ready)
            elastic_held_inputs++;
        if ({engine}.finalize_q && {engine}.pending_q && !{engine}.i_gradient_ready)
            elastic_double_holds++;
    end
"""
    if old.count("        $finish;") != 1 or old.count("endmodule") != 1:
        raise ValueError("testbench marker")
    new = old.replace("endmodule", monitor + "\nendmodule").replace(
        "        $finish;",
        '        $display("GRADIENT_ELASTIC_CORE_PASS inputs=%0d outputs=%0d finalize_inputs=%0d held_inputs=%0d double_holds=%0d",\n'
        "            elastic_inputs, elastic_outputs, elastic_finalize_inputs, elastic_held_inputs, elastic_double_holds);\n"
        "        $finish;")
    tb.write_text(new, newline="\n")
    (out / "elastic_monitor.diff").write_text("".join(difflib.unified_diff(
        old.splitlines(True), new.splitlines(True), fromfile="accepted_TB", tofile="monitored_TB")))
    expected = json.loads((prior / "expected_checks.json").read_text())
    expected["tiles"] = [t.replace(".u_final.", ".u_final.u_core.", 1) for t in expected["tiles"]]
    (out / "expected_checks.json").write_text(json.dumps(expected, indent=2) + "\n")
    original_log = (archive.parent / PRIOR / "run_wrapper_on.log").read_text()
    (out / "accepted_tile_activity.json").write_text(json.dumps(activity(original_log), indent=2) + "\n")
    helper = Path(__file__).with_name("run_final_raw_source_functional.py").read_text()
    helper = helper.replace("PASS_final_raw_wrapper_closed_loop_exact", "PASS_elastic_gradient_closed_loop")
    helper = helper.replace("accepted_records_exact", "accepted_numerical_records_preserved")
    helper = helper.replace("PASS_final_raw_wrapper_functional_archive", "PASS_elastic_gradient_core_archive")
    helper = helper.replace("records_match_accepted_core=True", "numeric_records_match_accepted_core=True")
    (out / "elastic_wrapper_common.py").write_text(helper, newline="\n")
    shutil.copy2(__file__, out / "launch_worker.py")
    changed = [p.relative_to(out).as_posix() for p in (out / "rtl").rglob("*")
               if p.is_file() and p.read_bytes() != (prior / p.relative_to(out)).read_bytes()]
    if changed != [MODULE]:
        raise ValueError("unexpected RTL changes")
    record = dict(status="PREPARED_NOT_SIMULATED", scope=SCOPE, changed_rtl=changed,
        rtl_items=113, numerical_inputs_byte_equal=227, reducer_hierarchy=engine,
        local_unit_status=accepted["status"], assertions="All original numerical assertions retained.")
    (out / "preparation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def audit():
    _, prepared, archive = roots()
    state = json.loads((archive / "status.json").read_text())
    log = (archive / "build.log").read_text(errors="replace")
    import collections
    warnings = dict(collections.Counter(re.findall(r"^%Warning-([^:]+):", log, re.M)))
    if warnings != state.get("build_warnings") or "%Error" in log or set(warnings) & {"LATCH", "MULTIDRIVEN", "UNOPTFLAT"}:
        raise ValueError("actual build diagnostics")
    configure(prepared).audit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "resume-preparation", "launch", "worker", "audit"))
    action = parser.parse_args().action
    if action == "prepare":
        prepare()
    elif action == "resume-preparation":
        prepare(resume=True)
    elif action == "worker":
        configure(Path.cwd()).worker()
    elif action == "audit":
        audit()
    else:
        configure().launch()
