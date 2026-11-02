"""Verify shared row-address arithmetic with exact accepted SCNN cycle and transaction records."""
import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
from run_srnn_resident_remote import ROOT, RESULT, local_path

NAME = "scnn_row_reuse_core_r1"
REFERENCE = "scnn_row_turnover_core_r1"
PRIOR = "final_scnn_functional_r1"
MODULE = Path("rtl/learning/unified_scnn_classifier_cache_refill_publisher_v1.sv")


def numerical_records(data):
    result = json.loads(json.dumps(data))
    for key in ("status", "scope"):
        result.pop(key)
    return result


def worker_source(original):
    replacements = {
        'NAME = "final_scnn_functional_r1"': 'NAME = ' + repr(NAME),
        'PRIOR = "scnn_turnover_tail_r1"': 'PRIOR = ' + repr(PRIOR),
        'PRIOR_PREPARED = "verification/experiments/scnn_forward_turnover/tail_r1/candidate_r1"':
            'PRIOR_PREPARED = "verification/experiments/' + PRIOR + '"',
        "Final physical wrapper/core, raw source inactive.":
            "New publisher-only candidate on the accepted wrapper/core, raw source inactive.",
        "Exact final shared owner/config/RMW/publication":
            "Unchanged shared owner/config/RMW/publication",
        "physical_rtl_files_byte_equal=113": "rtl_items=113",
    }
    for old, new in replacements.items():
        if original.count(old) != 1:
            raise ValueError("unknown tail template: " + old)
        original = original.replace(old, new)
    original = original.replace("PASS_final_SCNN_wrapper_nonzero_tail_query",
                                "PASS_SCNN_row_address_reuse_tail_query")
    original = original.replace("PASS_final_SCNN_wrapper_archive",
                                "PASS_SCNN_row_address_reuse_archive")
    anchor = "def parse_log(log, directory=None):"
    original = original.replace(anchor, inspect.getsource(numerical_records) + "\n\n" + anchor, 1)
    anchor = '    result["status"] = "PASS_SCNN_row_address_reuse_tail_query"\n    return result'
    if original.count(anchor) != 1:
        raise ValueError("parser return")
    return original.replace(anchor, '''    result["status"] = "PASS_SCNN_row_address_reuse_tail_query"
    expected = json.loads((directory / "expected_checks.json").read_text())
    if numerical_records(result) != numerical_records(expected):
        raise ValueError("cycle/numerical/traffic/commit/version records changed")
    return result''')


def load_worker(path):
    spec = importlib.util.spec_from_file_location("executed_scnn_row_address_reuse", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def prepare():
    root = Path(local_path(ROOT))
    old = root / "verification/experiments" / PRIOR
    archive = root / RESULT / PRIOR
    accepted = json.loads((archive / "terminal_audit.json").read_text())
    if accepted["status"] != "PASS_final_SCNN_wrapper_archive":
        raise ValueError("prior tail not accepted")
    reference = root / RESULT / REFERENCE
    reference_accepted = json.loads((reference / "terminal_audit.json").read_text())
    if reference_accepted["status"] != "PASS_SCNN_row_turnover_archive":
        raise ValueError("accepted row-turnover core required")
    unit = root / "verification/experiments/scnn_row_reuse_r1"
    unit_status = unit / "runs/r1/status.json"
    local = json.loads(unit_status.read_text())
    if local["status"] != "PASS_LOCAL_SCNN_ROW_ADDRESS_REUSE":
        raise ValueError("publisher unit correctness missing")
    target = root / "verification/experiments" / NAME
    if target.exists():
        raise FileExistsError("candidate preparation exists")
    candidate = unit / "rtl" / MODULE.name
    original = old / MODULE
    if candidate.read_bytes() == original.read_bytes():
        raise ValueError("candidate is unchanged")
    shutil.copytree(old, target)
    shutil.copy2(candidate, target / MODULE)
    (target / "README.md").rename(target / "prior_preparation_README.md")
    (target / "final_preparation.json").rename(target / "prior_preparation.json")
    shutil.copy2(reference / "checks.json", target / "expected_checks.json")
    shutil.copy2(unit_status, target / "local_correctness_record.json")
    shutil.copy2(__file__, target / "prepare_runner.py")
    source = Path(__file__).with_name("run_final_scnn_functional.py").read_text()
    (target / "launch_worker.py").write_text(worker_source(source), newline="\n")
    changed = [p.relative_to(target).as_posix() for p in (target / "rtl").rglob("*")
               if p.is_file() and p.read_bytes() != (old / p.relative_to(target)).read_bytes()]
    if changed != [MODULE.as_posix()]:
        raise ValueError("unexpected RTL difference")
    metadata = dict(status="PREPARED_NOT_LAUNCHED", changed_rtl=changed, rtl_items=113,
        comparator=REFERENCE, static_input_source=PRIOR,
        tb_inputs_numeric_assertions="Unchanged cold B4T12+query4; all three windows and exact timing records retained",
        local_correctness="One new 99-case phase-aligned run matches all prior row-turnover records; "
            "added retirement-index identity assertions, no old arm rerun.",
        scope="Only the SCNN publisher changes; new SRNN elastic reducer is not combined here. "
            "CPU Conv7 features plus 52 Conv8 tail descriptors; no full raw graph/T20/Test or new physical evidence.",
        schedule="Do not launch while another server EDA is active. Existing build5400/run21600 "
            "timeouts retain PID; original j2/compiler flags unchanged, no new warning waiver.")
    (target / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "audit"))
    action = parser.parse_args().action
    if action == "prepare":
        prepare()
    else:
        prepared = Path(local_path(ROOT / "verification/experiments" / NAME))
        getattr(load_worker(prepared / "launch_worker.py"), action)()


if __name__ == "__main__":
    main()
