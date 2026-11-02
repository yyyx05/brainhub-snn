"""Command-line entry point for bounded autonomous profile selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .decision import DecisionEngine, load_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="JSON fingerprint request")
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--prototypes", type=Path, required=True)
    parser.add_argument("--learning", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    engine = DecisionEngine.from_files(args.profiles, args.prototypes)
    result = engine.decide(load_json(args.request), args.learning)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
