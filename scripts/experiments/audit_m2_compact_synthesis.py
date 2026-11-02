"""Audit the completed compact synthesis and compare actual whole-core reports."""
import csv
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import zipfile

from run_srnn_resident_remote import ROOT, RESULT, REMOTE, connect, run, local_path
from run_m2_compact_synthesis import selected_tcl, TCL
from prepare_m2_compact_core import REPAIRED_OUTPUT

NAME = "m2_compact_synth_r2"
OUT = "server_results/final_composition_synth"
ROM = "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem"


def resources(text):
    result = {}
    for key, label in (("lut", "Slice LUTs*"), ("ff", "Slice Registers"),
                       ("bram36eq", "Block RAM Tile"), ("dsp", "DSPs")):
        match = re.search(r"^\|\s*" + re.escape(label) + r"\s*\|\s*([\d.]+)\s*\|", text, re.M)
        if not match:
            raise ValueError("missing resource: " + label)
        result[key] = float(match[1]) if "." in match[1] else int(match[1])
    return result


def timing(text):
    section = text.split("| Design Timing Summary", 1)[1].split("| Clock Summary", 1)[0]
    rows = [line.split() for line in section.splitlines()
            if re.match(r"^\s+-?\d+\.\d+\s+-?\d+\.\d+", line)]
    if len(rows) != 1 or len(rows[0]) != 12:
        raise ValueError("ambiguous timing summary")
    names = ("wns_ns", "tns_ns", "setup_failing", "setup_total", "whs_ns", "ths_ns",
             "hold_failing", "hold_total", "wpws_ns", "tpws_ns", "pulse_failing", "pulse_total")
    return {name: float(value) if name.endswith("_ns") else int(value)
            for name, value in zip(names, rows[0])}


def hierarchy(text):
    stack, rows = [], []
    for line in text.splitlines():
        cells = line.split("|")[1:-1]
        if len(cells) != 10 or not all(re.fullmatch(r"\s*\d+\s*", x) for x in cells[2:]):
            continue
        name = cells[0].strip()
        if name.startswith("("):
            continue
        indent = len(cells[0]) - len(cells[0].lstrip())
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, name))
        rows.append(dict(path="/".join(x[1] for x in stack), name=name, module=cells[1].strip(),
                         lut=int(cells[2]), ff=int(cells[6]), ramb36=int(cells[7]), ramb18=int(cells[8])))
    if not rows:
        raise ValueError("empty hierarchy")
    return rows


def main():
    archive = Path(local_path(ROOT / RESULT))
    local, banked = archive / NAME, archive / "scnn_issue_banked_synth_r1"
    prepared = Path(local_path(REPAIRED_OUTPUT))
    remote_code = r'''
import datetime,json,pathlib,shutil,subprocess
r=pathlib.Path(ROOT)
files={p.relative_to(r).as_posix():dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
       for p in r.rglob('*') if p.is_file() and not set(p.relative_to(r).parts)&{'obj','elab_obj','__pycache__','.Xil'}}
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  comm=(p/'comm').read_text().strip()
  if comm in ('vivado','Vscgraph','verilator_bin','xelab','xsim') or comm.startswith('Vtb_'):active.append([p.name,comm])
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
k=subprocess.run(['dmesg','--ctime'],capture_output=True,text=True)
print(json.dumps(dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
 free_bytes=shutil.disk_usage(r).free,active_eda=active,dmesg_exit=k.returncode,dmesg_error=k.stderr)))
'''
    client = connect()
    try:
        observation = json.loads(run(client, "python3 -c " + shlex.quote(
            "ROOT=" + repr(REMOTE + "/" + NAME) + "\n" + remote_code)))
    finally:
        client.close()
    assert not observation["active_eda"]
    for relative, info in observation["files"].items():
        assert (local / relative).is_file() and (local / relative).stat().st_size == info["size"], relative
    assert not list(local.rglob("*.part"))
    source_paths = sorted(p.relative_to(local).as_posix() for p in (local / "rtl").rglob("*") if p.is_file())
    assert len(source_paths) == 109 and sum(p.endswith(".sv") for p in source_paths) == 108
    for relative in source_paths:
        assert (local / relative).read_bytes() == (prepared / relative).read_bytes(), relative
    assert (local / ROM).read_bytes() == (banked / ROM).read_bytes()
    assert (local / ROM).stat().st_size == 6510
    sources = sorted(p for p in source_paths if p.endswith(".sv"))
    for index, relative in enumerate(sources):
        source = (local / relative).read_text()
        transformed = local / OUT / "vivado_2022_2_sources" / f"{index:03d}_{Path(relative).name}"
        assert transformed.read_text() == source.replace("`default_nettype none", "`default_nettype wire")
    assert (local / TCL).read_text() == selected_tcl((banked / TCL).read_text())
    recipe = ["server/run_unified_h40_final_composition_v1_synth.sh",
              "vivado/unified_h40_final_composition_v1_ooc.xdc"]
    assert all((local / p).read_bytes() == (banked / p).read_bytes() for p in recipe)
    report = local / OUT
    status = dict(line.split("=", 1) for line in (report / "vivado_status.txt").read_text().splitlines())
    assert status["status"] == "pass" and status["black_boxes"] == "0"
    assert status["m2_cache_depth"] == "512" and status["source_count"] == "108"
    assert status["period_ns"] == "15.000" and status["implementation"] == "not_run"
    states = {n: (report / n).read_text().strip() for n in ("wrapper_status.txt", "vivado_exit_code.txt",
        "error_count.txt", "critical_warning_count.txt", "start_time.txt", "end_time.txt")}
    assert states["wrapper_status.txt"] == "status=pass" and states["vivado_exit_code.txt"] == "0"
    log = (report / "vivado.log").read_text()
    assert not re.search(r"^ERROR:|^CRITICAL WARNING:|Abnormal program termination", log, re.M)
    assert states["error_count.txt"] == states["critical_warning_count.txt"] == "0"
    dcps = [p for p in observation["files"] if p.endswith(".dcp")]
    assert dcps == [OUT + "/post_synth.dcp"]
    with zipfile.ZipFile(local / dcps[0]) as checkpoint:
        assert checkpoint.testzip() is None
        dcp_entries = len(checkpoint.infolist())
    rows, banks = [], []
    for label, folder, depth in (("banked_full578", banked, 578), ("compact_selected512", local, 512)):
        rpt = folder / OUT
        resource = resources((rpt / "utilization_post_synth.rpt").read_text())
        metrics = timing((rpt / "timing_summary_post_synth.rpt").read_text())
        items = hierarchy((rpt / "utilization_hierarchical_post_synth.rpt").read_text())
        contexts = {i: [x for x in items if re.fullmatch(r"g_context\[" + str(i) + r"\]\.g_enabled(?:\.g_original|\.g_compact)?\.u_store", x["name"])] for i in (0, 1)}
        assert len(contexts[0]) == 12 and len(contexts[1]) == 8
        totals = {i: sum(x["ramb36"] + x["ramb18"] / 2 for x in contexts[i]) for i in (0, 1)}
        assert totals[0] == 288 and totals[1] == (192 if depth == 578 else 80)
        row = dict(variant=label, stage="post_synth", depth=depth, **resource, **metrics,
                   context0_bram36eq=totals[0], context1_bram36eq=totals[1],
                   other_bram36eq=resource["bram36eq"] - totals[0] - totals[1])
        rows.append(row)
        if depth == 512:
            banks = [x for x in items if "g_context[1].g_enabled.g_compact.u_store/" in x["path"]
                     and re.fullmatch(r"g_bank\[\d+\]\.u_memory", x["name"])]
            assert len(banks) == len({x["path"] for x in banks}) == 32
            assert sum(x["ramb36"] for x in banks) == 80 and not any(x["ramb18"] for x in banks)
            assert all(x["ramb36"] == (1 if "g_bank[3]" in x["name"] else 3) for x in banks)
    assert float(status["wns_ns"]) == rows[1]["wns_ns"]
    assert float(status["whs_ns"]) == rows[1]["whs_ns"]
    timing_text = (report / "timing_summary_post_synth.rpt").read_text()
    checks = {name: int(re.search(r"checking " + name + r" \((\d+)\)", timing_text)[1]) for name in
              ("no_clock", "unconstrained_internal_endpoints", "no_input_delay", "no_output_delay", "loops")}
    worst = (report / "timing_setup_paths_post_synth.rpt").read_text()
    path = {k: re.search(r"^\s*" + k + r":\s*(.+)$", worst, re.M)[1] for k in
            ("Source", "Destination", "Data Path Delay", "Logic Levels", "Requirement")}
    methodology = (report / "methodology_post_synth.rpt").read_text()
    rules = [dict(rule=m[1], severity=m[2], count=int(m[3])) for m in re.finditer(
        r"^\|\s*([A-Z]+-\d+)\s*\|\s*(Warning|Critical Warning|Error)\s*\|[^|]*\|\s*(\d+)\s*\|", methodology, re.M)]
    assert sum(x["count"] for x in rules) == int(re.search(r"Violations found:\s*(\d+)", methodology)[1])
    reduction = {k: rows[0][k] - rows[1][k] for k in ("lut", "ff", "bram36eq", "dsp")}
    audit = dict(status="PASS_synthesis_evidence_not_implementation_STA", raw_status=states, tcl_status=status,
        wall_seconds=(dt.datetime.fromisoformat(states["end_time.txt"]) - dt.datetime.fromisoformat(states["start_time.txt"])).total_seconds(),
        raw_file_count=len(observation["files"]), raw_bytes=sum(x["size"] for x in observation["files"].values()),
        all_remote_sizes_equal=True, partial_files=[], source_files_byte_equal_candidate=109,
        original_rom_byte_equal=True, compatibility_sources_checked=108, recipe_byte_equal=recipe,
        dcp=dict(path=dcps[0], bytes=(local / dcps[0]).stat().st_size, zip_crc_pass=True, entries=dcp_entries),
        ordinary_warning_count=len(re.findall(r"^WARNING:", log, re.M)), methodology_rules=rules,
        oom_or_killed_log_lines=re.findall(r"^.*(?:Out of memory|\bOOM\b|\bKilled\b).*$", log, re.M | re.I),
        comparison=rows, reduction=reduction, worst_setup_path=path, timing_checks=checks,
        remote_observation=observation,
        boundary="Selected512 capacity differs from578. Whole-core measured delta, not module multiplication. Synthesis estimates are not routed STA or a throughput result. Source protection/15ns recipe preserved; netlist protected-set and trace-control checks remain in this candidate's implementation entry/exit. OOC input/output delays absent; dmesg unavailable. Missing-ROM R1 excluded. Production untouched.")
    (local / "terminal_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for name, data in (("comparison.csv", rows), ("context1_banks.csv", banks)):
        with (local / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps({k: v for k, v in audit.items() if k != "remote_observation"}, indent=2))


if __name__ == "__main__":
    main()
