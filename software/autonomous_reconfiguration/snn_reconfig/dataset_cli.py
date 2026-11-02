"""Create selector-ready fingerprints from project SHD or DVS-Gesture data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset_adapters import fingerprint_dvsgesture_q8_8_hex, fingerprint_shd_h5


def _parse_indices(specification: str) -> list[int]:
    result: list[int] = []
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            start_text, stop_text = item.split(":", 1)
            result.extend(range(int(start_text), int(stop_text)))
        else:
            result.append(int(item))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="adapter", required=True)

    shd = subparsers.add_parser("shd-h5", help="read native SHD HDF5 events")
    shd.add_argument("input", type=Path)
    shd.add_argument("--indices", default="0:300", help="comma-separated IDs/ranges")
    shd.add_argument("--dataset-id", default="SHD-project-slice")
    shd.add_argument("--output", type=Path)

    dvs = subparsers.add_parser("dvsgesture-q8-8", help="read dense DVS Q8.8 frames")
    dvs.add_argument("input", type=Path)
    dvs.add_argument("--sample-count", type=int, required=True)
    dvs.add_argument("--dataset-id", default="DVS-Gesture-project-export")
    dvs.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.adapter == "shd-h5":
        fingerprint = fingerprint_shd_h5(args.input, _parse_indices(args.indices))
    else:
        fingerprint = fingerprint_dvsgesture_q8_8_hex(args.input, args.sample_count)
    payload = {
        "dataset_id": args.dataset_id,
        "source_adapter": args.adapter,
        "source_path": str(args.input),
        "fingerprint": fingerprint,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
