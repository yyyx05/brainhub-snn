"""Reuse the accepted physical recipe for one new SCNN row-address-reuse publisher checkpoint."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
from run_srnn_resident_remote import ROOT, RESULT, local_path

NAME = "scnn_row_reuse_impl_r1"
SYNTHESIS = "scnn_row_reuse_synth_r1"
EXPERIMENT = Path("verification/experiments") / NAME
SCRATCH = "/tmp/snn_" + NAME
OUT = Path("server_results/final_composition_impl")
ARMS = ("candidate",)


def select_run(name):
    global NAME, EXPERIMENT, SCRATCH
    if re.fullmatch(r"scnn_row_reuse_impl_r[1-9][0-9]*", name) is None:
        raise ValueError("expected an isolated scnn_row_reuse_impl_rN name")
    NAME = name
    EXPERIMENT = Path("verification/experiments") / NAME
    SCRATCH = "/tmp/snn_" + NAME


def replace_once(text, replacements):
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError("unknown implementation template: " + old)
        text = text.replace(old, new)
    return text


def worker_source(original):
    return replace_once(original, {
        "NAME = 'final_raw_source_impl_r1'": "NAME = " + repr(NAME),
        "EXPERIMENT = Path('verification/experiments/final_raw_source_implementation_r1')":
            "EXPERIMENT = Path(" + repr(EXPERIMENT.as_posix()) + ")",
        "SYNTHESIS = 'final_raw_source_synth_r1'": "SYNTHESIS = " + repr(SYNTHESIS),
        "ARMS = ('baseline', 'candidate')": "ARMS = ('candidate',)",
        "COMPLETE_paired_raw_source_implementation": "COMPLETE_scnn_row_reuse_candidate_implementation",
        "Two serial internal OOC implementations from accepted DCPs.":
            "One new internal OOC implementation from its accepted synthesis DCP.",
    })


def audit_source(original):
    result = replace_once(original, {
        "from run_final_raw_source_implementation import NAME, SCRATCH, EXPERIMENT, OUT, ARMS":
            "from executed_scnn_row_reuse_implementation_worker import NAME, SCRATCH, EXPERIMENT, OUT, ARMS",
        "    if pair_complete:\n": "    if pair_complete and set(arms) == {'baseline', 'candidate'}:\n",
        "Paired producer-plus-core internal OOC implementation with unchanged 15ns and original recipe.":
            "One new producer-plus-core internal OOC implementation with unchanged 15ns and original recipe.",
    })
    # Keep all report/protection/route checks; only the single-arm labels change.
    for old, new in {
        "PASS_paired_implementation_archive": "PASS_scnn_row_reuse_implementation_archive",
        "ARCHIVED_incomplete_paired_implementation": "ARCHIVED_incomplete_scnn_row_reuse_implementation",
        "both_internal_implementations_accepted": "candidate_internal_implementation_accepted",
        "pair_measurements_complete": "candidate_measurement_complete",
    }.items():
        result = result.replace(old, new)
    return result


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare():
    root = Path(local_path(ROOT))
    evidence = root / RESULT
    accepted = json.loads((evidence / SYNTHESIS / "terminal_audit.json").read_text())
    if accepted["status"] != "PASS_scnn_row_reuse_synthesis_archive":
        raise ValueError("accepted new synthesis required")
    record = accepted["candidate"]
    relative = Path(SYNTHESIS) / "candidate/server_results/final_composition_synth/post_synth.dcp"
    source = evidence / relative
    if record["exit"] != 0 or record["checkpoint"]["crc_pass"] is not True or source.stat().st_size != record["checkpoint"]["bytes"]:
        raise ValueError("unaccepted new checkpoint")
    prior = root / "verification/experiments/final_raw_source_implementation_r1"
    original = json.loads((prior / "preparation.json").read_text())
    out = root / EXPERIMENT
    out.mkdir(parents=True, exist_ok=False)
    for name in ("original_impl.tcl", "implementation.tcl", "implementation_changes.diff"):
        shutil.copy2(prior / name, out / name)
    scripts = Path(__file__).parent
    for name in ("run_final_raw_source_synthesis.py", "run_scnn_turnover_implementation.py",
                 "run_m2_compact_route_recovery.py"):
        shutil.copy2(scripts / name, out / name)
    worker = worker_source((scripts / "run_final_raw_source_implementation.py").read_text())
    (out / "worker.py").write_text(worker, newline="\n")
    audit = audit_source((scripts / "audit_final_raw_source_implementation.py").read_text())
    (out / "audit_candidate.py").write_text(audit, newline="\n")
    shutil.copy2(__file__, out / "prepare_runner.py")
    prep = dict(status="PREPARED_NOT_LAUNCHED", name=NAME, sources={"candidate": dict(
        relative=relative.as_posix(), bytes=source.stat().st_size)},
        source_acceptance=accepted["status"], original_recipe=original["original_recipe"],
        recipe=original["recipe"], comparator="final_raw_source_impl_r1/candidate (already accepted; not rerun)",
        boundary="New SCNN publisher only; SRNN reducer unchanged. Exact prior wrapper implementation Tcl; no new clock, "
            "pin constraints, physical strategy, power or bitstream. Internal OOC timing, not full-system signoff.")
    (out / "preparation.json").write_text(json.dumps(prep, indent=2) + "\n")
    print(json.dumps(prep, indent=2))


def launch():
    prepared = Path(local_path(ROOT / EXPERIMENT))
    worker = load("executed_scnn_row_reuse_implementation_worker", prepared / "worker.py")
    worker.launch()


def audit():
    archive = Path(local_path(ROOT / RESULT / NAME))
    load("executed_scnn_row_reuse_implementation_worker", archive / "worker.py")
    load("scnn_row_reuse_implementation_auditor", archive / "audit_candidate.py").audit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "audit"))
    parser.add_argument("--name", default=NAME, help="Isolated run identifier; never reuse a launched run")
    args = parser.parse_args()
    select_run(args.name)
    globals()[args.action]()
