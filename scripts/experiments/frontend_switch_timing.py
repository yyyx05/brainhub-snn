"""Prepare a passive experimental r5 TB copy or validate supplied timing logs.

No compile, simulation, SSH or GPU execution is performed by this CLI.
"""

import argparse
import csv
import json
from pathlib import Path
import shlex
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "verification/experiments/frontend_switch_timing"
OUTPUT = EXPERIMENT / "generated"
R5 = ROOT / "results/paper_evidence/supplement_20261004/software/frontend_final_top_smoke/r5"
MARK = "// FT_TIMING_OBSERVABILITY"
SYNTHETIC = "SYNTHETIC_NOT_HDL_EVIDENCE"
TIMING = {"FT_INTERVAL": 6, "FT_SWITCH_READY": 5, "FT_SWITCH_TOTAL": 5}
sys.path.insert(0, str(ROOT / "scripts"))
import run_frontend_final_top_smoke as base


def require(condition, message):
    if not condition:
        raise ValueError(message)


def output_directory(path):
    path = path.resolve()
    require(path.is_relative_to(EXPERIMENT.resolve()), "output must stay inside experiment directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_observability(data):
    """Remove only lines tagged by this generator; preserve all other bytes."""
    marker = MARK.encode() if isinstance(data, bytes) else MARK
    empty = b"" if isinstance(data, bytes) else ""
    return empty.join(line for line in data.splitlines(keepends=True) if marker not in line)


def instrument(text):
    require(MARK not in text and "ft_timing_" not in text, "already instrumented or variable collision")
    additions = {
        "    longint cycles = 0;": [
            "    longint ft_timing_operation_begin = 0, ft_timing_switch_begin = 0;",
        ],
        '            $display("FT_STEP,%0d,%0d,%0d",program_index,operation,opcode);': [
            "            ft_timing_operation_begin = cycles;",
            "            if (operation == 1) ft_timing_switch_begin = cycles;",
        ],
        '            $display("FT_ACK,%0d,%0d,%0d,%0d",program_index,operation,opcode,payload_count);': [
            '            $display("FT_INTERVAL,%0d,%0d,%0d,%0d,%0d,%0d",program_index,operation,opcode,ft_timing_operation_begin,cycles,cycles-ft_timing_operation_begin);',
            '            if (operation == 5) $display("FT_SWITCH_READY,%0d,%0d,%0d,%0d,%0d",program_index,opcode,ft_timing_switch_begin,cycles,cycles-ft_timing_switch_begin);',
            '            if (operation == 8) $display("FT_SWITCH_TOTAL,%0d,%0d,%0d,%0d,%0d",program_index,opcode,ft_timing_switch_begin,cycles,cycles-ft_timing_switch_begin);',
        ],
    }
    for anchor, lines in additions.items():
        require(text.count(anchor) == 1, "missing/ambiguous instrumentation anchor: " + anchor)
        text = text.replace(anchor, "".join(line + " " + MARK + "\n" for line in lines) + anchor)
    return text


def read_program(path):
    commands = []
    for line in path.read_text(encoding="ascii").splitlines():
        values = list(map(int, line.split()))
        require(len(values) >= 3 and values[2] == len(values) - 3, "malformed replay program")
        commands.append((values[0], values[1], tuple(values[3:])))
    require(len(commands) == 32, "expected the existing 32-command r5 program")
    return commands


def legacy_lines(text):
    return [line for line in text.splitlines()
            if line.startswith("FT_") and line.split(",", 1)[0] not in TIMING]


def parse_timing(text, commands, golden):
    """Validate observability and old r5 behavior, not execution provenance."""
    checked = base.check_transcript(text, commands)
    require(checked["status"] == "PASS", "existing r5 checks failed: " +
            ", ".join(k for k, v in checked["checks"].items() if not v))
    require(legacy_lines(text) == legacy_lines(golden), "legacy r5 records/counts/order changed")
    require(checked["rows"].get("FT_CYCLES") == [(9040,)], "r5 terminal must remain 9040 cycles")
    rows = {name: [] for name in TIMING}
    sequence = []
    for line in text.splitlines():
        marker, _, payload = line.partition(",")
        if marker not in (*TIMING, "FT_STEP", "FT_ACK"):
            continue
        try:
            values = tuple(map(int, payload.split(",")))
        except ValueError as exc:
            raise ValueError("malformed " + marker) from exc
        if marker in TIMING:
            require(len(values) == TIMING[marker], "wrong field count: " + marker)
            rows[marker].append(values)
        require(bool(values), "missing index")
        sequence.append((marker, values[0]))
    expected_sequence = []
    for index, (operation, opcode, _) in enumerate(commands):
        require(operation == index % 8 + 1 and opcode == (1, 16, 32, 1)[index // 8], "r5 command order")
        expected_sequence.extend((("FT_STEP", index), ("FT_INTERVAL", index)))
        if operation == 5:
            expected_sequence.append(("FT_SWITCH_READY", index))
        if operation == 8:
            expected_sequence.append(("FT_SWITCH_TOTAL", index))
        expected_sequence.append(("FT_ACK", index))
    require(sequence == expected_sequence, "missing/duplicate/out-of-order timing or ack records")
    operations = []
    previous_end = 0
    for index, (row, command) in enumerate(zip(rows["FT_INTERVAL"], commands)):
        observed_index, operation, opcode, start, end, elapsed = row
        require((observed_index, operation, opcode) == (index, command[0], command[1]), "interval identity")
        require(0 <= start <= end <= 9040 and elapsed == end - start, "invalid elapsed interval")
        require(start >= previous_end, "overlapping/reversed operation order")
        operations.append(dict(index=index, operation=operation, opcode=opcode,
                               start=start, end=end, cycles=elapsed))
        previous_end = end
    switches = []
    for transaction in range(4):
        begin = operations[transaction * 8]["start"]
        phases = operations[transaction * 8:transaction * 8 + 8]
        result = dict(transaction=transaction, opcode=phases[0]["opcode"])
        for marker, position, name in (("FT_SWITCH_READY", 4, "stop_to_version_aligned"),
                                       ("FT_SWITCH_TOTAL", 7, "full_transaction_including_workload")):
            row = rows[marker][transaction]
            terminal = phases[position]
            require(row == (terminal["index"], terminal["opcode"], begin,
                            terminal["end"], terminal["end"] - begin), "switch span endpoint mismatch")
            phase_cycles = sum(p["cycles"] for p in phases[:position + 1])
            result[name] = dict(start=begin, end=terminal["end"], cycles=row[-1],
                                operation_cycles=phase_cycles,
                                unclassified_between_operations_cycles=row[-1] - phase_cycles)
        switches.append(result)
    return dict(status="VALIDATED_TRANSCRIPT", evidence_kind=(SYNTHETIC if SYNTHETIC in text else
                "SUPPLIED_LOG_EXECUTION_PROVENANCE_NOT_ESTABLISHED"), rtl_execution_verified=False,
                hardware_timing_gap_closed=False, interval_convention="[start,end); end-start; zero allowed",
                acknowledged_commands=32, terminal_cycles=9040, legacy_r5_records_unchanged=True,
                operations=operations, switches=switches,
                limits=["Full transaction includes operation-6 workload and post-workload drain.",
                        "Version alignment is not a DUT profile-ready port.",
                        "Learning-off synthetic small geometry; not catalog DMA or formal M2.",
                        "Parsing alone cannot establish actual RTL execution."])


SERVER_DRIVER = '''"""Prepared but not run; local WSL/POSIX full-top assessment is pending."""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from run_frontend_final_top_smoke import run_bounded
sys.path.insert(0, str(ROOT / "scripts/experiments"))
from frontend_switch_timing import main
OUT = Path(__file__).resolve().parent
config = json.loads((OUT / "server_commands.json").read_text())
import os
os.chdir(ROOT)
for stage in ("build", "simulation"):
    result = run_bounded(config[stage]["argv"], OUT, stage, dict(os.environ),
                         config[stage]["timeout_seconds"])
    if result["exit_code"] != 0 or result.get("stopped_reason"):
        raise SystemExit("Preserved server failure; no automatic retry: " + stage)
raise SystemExit(main(["--parse", str(OUT / "simulation.log"), "--output", str(OUT)]))
'''


def prepare(output=OUTPUT):
    output = output_directory(output)
    # Reuse preparers as functions only. Their main/build/run entrypoints are not called.
    transport, sequence = base.prepare(output)
    golden_program = read_program(R5 / "program.txt")
    require(transport.commands == golden_program, "prepared program differs from archived r5")
    golden = (R5 / "simulation.log").read_text(encoding="utf-8")
    require(base.check_transcript(golden, transport.commands)["status"] == "PASS", "invalid archived r5")
    require("FT_CYCLES,9040" in golden.splitlines(), "missing r5 terminal counter")
    plain = output / "base_generated_tb.sv"
    tb = output / (base.TOP + ".sv")
    base.generate_tb(plain, top=base.TOP, body=ROOT / "tb/tb_frontend_final_top_smoke.sv",
                     parameter_overrides=base.OVERRIDES)
    original = plain.read_bytes()
    # generate_tb writes native newlines; retain them when adding tagged lines.
    newline = "\r\n" if b"\r\n" in original else "\n"
    modified = instrument(original.decode("ascii").replace("\r\n", "\n")).replace("\n", newline).encode("ascii")
    require(remove_observability(modified) == original, "observability removal changed base TB")
    tb.write_bytes(modified)
    sources = base.simulation_sources(tb)
    require(all(p.is_file() and p.resolve().is_relative_to(ROOT) for p in sources), "missing/package-external source")
    manifest = output / "sources.f"
    manifest.write_text("".join('"' + p.relative_to(ROOT).as_posix() + '"\n' for p in sources), encoding="ascii")
    rel = output.relative_to(ROOT)
    build, exe, _ = base.build_settings(rel / "obj", rel / "sources.f", jobs=1,
                                        top=base.TOP, verilator="verilator", platform="posix")
    build[build.index("--output-split") + 1] = "12000"
    build += ["--output-split-cfuncs", "0"]
    commands = dict(cwd="package root in local WSL or on a POSIX server", execution="NOT_RUN",
                    build=dict(argv=build, timeout_seconds=570),
                    simulation=dict(argv=[exe.as_posix(), "+program=" + (rel / "program.txt").as_posix()], timeout_seconds=30),
                    resource_policy="Existing run_bounded: process-tree RSS <2,000,000,000 bytes; no automatic retry")
    (output / "server_commands.json").write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    # Derive the root from this output location, including nested test output directories.
    depth = len(output.relative_to(ROOT).parts)
    driver = SERVER_DRIVER.replace("parents[4]", f"parents[{depth}]")
    (output / "run_on_server.py").write_text(driver, encoding="ascii")
    report = dict(status="PREPARED_NOT_RUN", hdl_compile="NOT_RUN", rtl_execution="NOT_RUN",
                  hardware_timing_gap_closed=False, added_observability_lines=6,
                  static_base_equal_after_removing_added_lines=True,
                  comparison="direct byte comparison, no content hashes",
                  acknowledged_commands_expected=32, golden_terminal_cycles=9040,
                  source_count=len(sources), source_manifest="sources.f",
                  production_rtl_source_count=sum(p.is_relative_to(ROOT / "rtl") for p in sources),
                  source_size_bytes=sum(p.stat().st_size for p in sources),
                  compile_scope="Complete final-composition top with small r5 stimulus/parameters, not isolated tinyP4",
                  measured_compile_peak_rss_bytes=None, measured_compile_seconds=None,
                  source_manifest_origin="existing runner simulation_sources; relative package paths for local WSL/POSIX",
                  restored_source_files=[], command_program_matches_archived_r5=True,
                  parameters=base.OVERRIDES, sequence=sequence,
                  golden_source=(R5 / "simulation.log").relative_to(ROOT).as_posix(),
                  server_command="python " + shlex.quote((rel / "run_on_server.py").as_posix()),
                  limits=["Generation inputs present; full HDL module/include closure not compiled or verified.",
                          "Static equality proves only additive source changes, not identical simulator scheduling.",
                          "Prepared but not run; full-top resource/dependency assessment is pending.",
                          "Source bytes/counts are not estimates of compile memory or time."])
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--parse", type=Path, metavar="LOG")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    output = output_directory(args.output)
    if args.prepare:
        report = prepare(output)
    else:
        report = parse_timing(args.parse.read_text(encoding="utf-8"), read_program(R5 / "program.txt"),
                              (R5 / "simulation.log").read_text(encoding="utf-8"))
        (output / "parsed_timing.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with (output / "operation_intervals.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("index", "operation", "opcode", "start", "end", "cycles"))
            writer.writeheader()
            writer.writerows(report["operations"])
    print(json.dumps({"status": report["status"], "output": str(output), "hardware_timing_gap_closed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
