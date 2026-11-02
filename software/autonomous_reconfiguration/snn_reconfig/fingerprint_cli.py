"""Convert normalized event samples into a profile-selector fingerprint JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fingerprint import fingerprint_event_samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON with metadata and samples")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    output = {
        "dataset_id": payload.get("dataset_id", args.input.stem),
        "fingerprint": fingerprint_event_samples(payload["samples"], payload["metadata"]),
    }
    rendered = json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
