#!/usr/bin/env python3
"""Remove a scalar-read issue bubble in an isolated SCNN publisher candidate."""
import argparse
import difflib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("rtl/learning/unified_scnn_classifier_cache_refill_publisher_v1.sv")
EXPERIMENT = Path("verification/experiments/scnn_publisher_issue")


def candidate(source):
    replacements = (
        ("o_forward_read_valid = state_q == STATE_READ_REQUEST;",
         "// Issue the current valid scalar directly; address/tag hold under backpressure.\n"
         "        o_forward_read_valid = state_q == STATE_GATHER &&\n"
         "            feature_index_int < integer'(feature_count_q);"),
        ("                    end else begin\n                        state_q <= STATE_READ_REQUEST;\n                    end",
         "                    end else if (read_fire) begin\n                        state_q <= STATE_READ_RESPONSE;\n                    end"),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError("publisher source anchor changed: " + old)
        source = source.replace(old, new, 1)
    return source


def prepare(name):
    output = ROOT / EXPERIMENT / name
    output.mkdir(parents=True, exist_ok=False)
    source = (ROOT / SOURCE).read_text(encoding="utf-8")
    changed = candidate(source)
    (output / SOURCE).parent.mkdir(parents=True)
    (output / SOURCE).write_text(changed, encoding="utf-8")
    reference = source.replace("module unified_scnn_classifier_cache_refill_publisher_v1 #(",
                               "module reference_scnn_publisher #(", 1)
    (output / "reference_scnn_publisher.sv").write_text(reference, encoding="utf-8")
    (output / "source_changes.diff").write_text("".join(difflib.unified_diff(
        source.splitlines(True), changed.splitlines(True), fromfile="production/" + str(SOURCE),
        tofile="candidate/" + str(SOURCE))), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="candidate_r1")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        parser.error("simple candidate name required")
    print(prepare(args.name))
