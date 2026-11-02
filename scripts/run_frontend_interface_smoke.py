"""Run fixed frontend selections through a small real-RTL interface replay.

No remote operations, full-model build, fixture generation, or EDA is used.
The final-composition integration remains explicitly outside this small DUT.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "software" / "autonomous_reconfiguration"
sys.path.insert(0, str(COMPONENT))

from snn_reconfig.decision import DecisionEngine, load_json  # noqa: E402
from snn_reconfig.interface_smoke_bridge import FixtureReplayTransport, check_transcript  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results/paper_evidence/supplement_20261004/software/frontend_interface_smoke"
TB = ROOT / "tb/tb_frontend_interface_smoke.sv"
LIMITS = [
    "Actual DUT: shared-array three-topology subtop, resident profile lookup and frame descriptor sequencer; not final-composition top.",
    "Catalog profile selection is real; example fingerprints, H80 dense and tiny Conv8/classifier worksets are synthetic, not trained catalog assets.",
    "Profile lookup's historical SRNN H480/capacity metadata is not the H450+M2 model; only opcode/slot/mode/version are consumed.",
    "SCNN sequencer uses a 32-bit synthetic layer tag, not final-composition descriptor packing or full Conv1..Conv8 graph.",
    "Feature/context responders are testbench services, not DMA/DDR/frame-state integration or a board transport.",
    "Learning is off. Fixed writable scopes are checked in software, but no learner, commit or parameter-version publication is exercised.",
    "Not full-model accuracy, M2 inference, throughput, latency, PPA, board or post-route evidence. No long experiments rerun.",
]
BLOCKER = (
    "Full frontend-to-final-composition integration is not closed by this smoke. "
    "It needs a concrete catalog-asset DMA/state service, final-top descriptor packing, "
    "SRNN backbone-to-M2 dispatch and publication-aware transport. The existing long "
    "numeric TB services cannot be substituted by this bounded fixture without expanding scope."
)


def prepare(output: Path) -> tuple[FixtureReplayTransport, list[dict]]:
    engine = DecisionEngine.from_files(
        COMPONENT / "catalog/hardware_profiles.json", COMPONENT / "catalog/fingerprint_prototypes.json"
    )
    transport = FixtureReplayTransport()
    sequence = []
    for example in (
        "nmnist_query.json", "shd_query.json", "smnist_unsupported_query.json",
        "dvsgesture_query.json", "nmnist_query.json",
    ):
        decision = engine.decide(load_json(COMPONENT / "examples" / example), "off")
        before = len(transport.commands)
        if decision["status"] == "hardware_profile_selected":
            transport.plan(decision)
        else:
            # Exercise rejection on the same transport, not only a caller skip.
            try:
                transport.plan(decision)
            except ValueError:
                pass
            else:
                raise ValueError("rejected decision unexpectedly accepted")
        after = len(transport.commands)
        sequence.append({
            "example": example, "status": decision["status"],
            "selected_profile": decision["selected_profile"],
            "commands_added": after - before,
        })
        if decision["status"] != "hardware_profile_selected" and before != after:
            raise ValueError("rejection emitted simulator commands")
    transport.write_program(output / "program.txt")
    (output / "frontend_plan.json").write_text(json.dumps({
        "status": "simulation_replay_pending", "sequence": sequence,
        "selections": transport.selections, "commands": transport.commands,
    }, indent=2) + "\n", encoding="utf-8")
    return transport, sequence


def source_paths() -> list[Path]:
    # Reuse the explicit, validated small roundtrip source list, never all RTL.
    runner = ROOT / "scripts/run_verilator_h40_array16_three_topology_graph_top_v1.ps1"
    sources = [ROOT / name.replace("\\", "/") for name in re.findall(
        r"'(rtl[\\/][^']+\.sv)'", runner.read_text(encoding="utf-8")
    )]
    if not sources:
        raise ValueError("small roundtrip source list is unavailable")
    return sources + [
        ROOT / "rtl/control/unified_tb_profile_code_lookup_v1.sv",
        ROOT / "rtl/control/unified_scnn_frame_descriptor_sequencer_v1.sv", TB,
    ]


def find_verilator(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit).resolve() if Path(explicit).is_file() else None
    for name in ("verilator_bin", "verilator"):
        found = shutil.which(name)
        if found:
            return Path(found).resolve()
    local = Path("C:/msys64/mingw64/bin/verilator_bin.exe")
    return local if os.name == "nt" and local.is_file() else None


def run_command(command: list[str], output: Path, stem: str, env: dict, timeout: int) -> dict:
    record = {"command": command, "cwd": str(ROOT), "timeout_seconds": timeout}
    (output / f"{stem}_command.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    with (output / f"{stem}.log").open("w", encoding="utf-8") as stream:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=timeout, check=False)
            record["exit_code"] = result.returncode
        except subprocess.TimeoutExpired:
            record["exit_code"] = None
            record["timeout"] = True
        except OSError as exc:
            record["exit_code"] = None
            record["error"] = str(exc)
    record["seconds"] = round(time.monotonic() - started, 3)
    (output / f"{stem}_command.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verilator", help="local Verilator binary path")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--prepare-only", action="store_true", help="write the plan only; never claim DUT PASS")
    args = parser.parse_args()
    output = args.output.resolve()
    # Preserve terminal evidence; retries use a fresh child directory.
    if any((output / name).exists() for name in ("report.json", "build.log", "simulation.log")):
        parser.error("output contains earlier evidence; choose a fresh --output child directory")
    output.mkdir(parents=True, exist_ok=True)
    transport, sequence = prepare(output)
    report = {
        "status": "NOT_RUN", "evidence_kind": "actual_rtl_subtop_replay",
        "final_composition_status": "BLOCKED", "blocker": BLOCKER, "limits": LIMITS,
        "sequence": [{"example": s["example"], "status": s["status"],
                      "commands_added": s["commands_added"]} for s in sequence],
        "rejected_task_commands": sequence[2]["commands_added"],
        "learning_enable": False,
    }
    if not args.prepare_only:
        verilator = find_verilator(args.verilator)
        if verilator is None:
            report.update(status="BLOCKED", simulator_blocker="Local Verilator unavailable; only a plan was written, no mock DUT fallback.")
        else:
            env = os.environ.copy()
            if os.name == "nt":
                env["PATH"] = os.pathsep.join((str(verilator.parent), "C:/msys64/usr/bin", env.get("PATH", "")))
                install = verilator.parent.parent / "share/verilator"
                if install.is_dir():
                    env["VERILATOR_ROOT"] = str(install)
            sources = source_paths()
            (output / "sources.txt").write_text("\n".join(str(p) for p in sources) + "\n", encoding="utf-8")
            obj = output / "obj"
            command = [str(verilator), "--binary", "-j", str(args.jobs), "--timing", "--assert",
                       "-Wall", "-Wno-fatal", "--output-split", "12000", "--output-split-cfuncs", "0",
                       "-Wno-DECLFILENAME", "-Wno-PROCASSINIT", "-Wno-PINCONNECTEMPTY", "-Wno-PINMISSING",
                       "-Wno-WIDTHCONCAT", "-Wno-WIDTHTRUNC", "-Wno-WIDTHEXPAND", "-Wno-UNUSEDSIGNAL",
                       "-CFLAGS", "-O1", "--top-module", "tb_frontend_interface_smoke", "--Mdir", str(obj),
                       *map(str, sources)]
            print(f"Building only the small shared-array DUT, jobs={args.jobs}", flush=True)
            report["build"] = run_command(command, output, "build", env, 900)
            if report["build"]["exit_code"] != 0:
                report["status"] = "FAIL"
            else:
                executable = obj / ("Vtb_frontend_interface_smoke.exe" if os.name == "nt" else "Vtb_frontend_interface_smoke")
                simulation = [str(executable), f"+program={(output / 'program.txt').as_posix()}"]
                print("Running frontend replay and two bounded DUT fault checks", flush=True)
                report["simulation"] = run_command(simulation, output, "simulation", env, 90)
                try:
                    report["dut"] = check_transcript((output / "simulation.log").read_text(encoding="utf-8"), transport.commands)
                except (ValueError, IndexError) as exc:
                    report["dut"] = {"status": "FAIL", "parse_error": str(exc)}
                report["negative_checks"] = {}
                for stem, flag, marker in (
                    ("profile_timeout", "+force_profile_timeout", "IF_FATAL PROFILE_TIMEOUT"),
                    ("illegal_opcode", "+force_illegal_opcode", "IF_FATAL PROFILE_REJECT"),
                ):
                    fault = run_command(simulation + [flag], output, stem, env, 90)
                    fault_log = (output / f"{stem}.log").read_text(encoding="utf-8")
                    fault["expected_failure_observed"] = (
                        fault["exit_code"] is not None and fault["exit_code"] != 0
                        and marker in fault_log and "IF_PASS" not in fault_log
                    )
                    report["negative_checks"][stem] = fault
                report["status"] = "PASS" if (
                    report["simulation"]["exit_code"] == 0 and report["dut"]["status"] == "PASS"
                    and all(f["expected_failure_observed"] for f in report["negative_checks"].values())
                ) else "FAIL"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "final_composition_status": "BLOCKED", "report": str(output / "report.json")}))
    return 0 if report["status"] in ("PASS", "NOT_RUN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
