#!/usr/bin/env python3
"""Summarize a terminal Vivado implementation result directory.

This is intentionally a read-only parser. It does not decide whether to rerun a
flow and it does not create a new workflow gate; it only turns existing Vivado
reports into a compact JSON summary for handoff and paper tables.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read_optional(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_key_values(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def parse_utilization(text: str) -> dict[str, dict[str, float]]:
    labels = {
        "slice_luts": "Slice LUTs",
        "slice_registers": "Slice Registers",
        "bram_tiles": "Block RAM Tile",
        "dsps": "DSPs",
    }
    result: dict[str, dict[str, float]] = {}
    for key, label in labels.items():
        for line in text.splitlines():
            if not line.startswith("|"):
                continue
            fields = [field.strip() for field in line.strip().strip("|").split("|")]
            if fields and fields[0] == label and len(fields) >= 6:
                used = float(fields[1].replace(",", ""))
                available = float(fields[-2].replace(",", ""))
                percent = float(fields[-1].replace(",", ""))
                result[key] = {
                    "used": used,
                    "available": available,
                    "utilization_percent": percent,
                }
                break
    return result


def parse_timing(text: str) -> dict[str, float | int] | None:
    match = re.search(
        r"WNS\(ns\).*?\n\s*-+.*?\n"
        r"\s*([-0-9.]+)\s+([-0-9.]+)\s+(\d+)\s+\d+\s+"
        r"([-0-9.]+)\s+([-0-9.]+)\s+(\d+)",
        text,
        re.S,
    )
    if match is None:
        return None
    return {
        "wns_ns": float(match.group(1)),
        "tns_ns": float(match.group(2)),
        "setup_failing_endpoints": int(match.group(3)),
        "whs_ns": float(match.group(4)),
        "ths_ns": float(match.group(5)),
        "hold_failing_endpoints": int(match.group(6)),
    }


def parse_route_status(text: str) -> dict[str, int | bool | None]:
    def metric(label: str) -> int | None:
        match = re.search(rf"{re.escape(label)}[^:]*:\s*([0-9,]+)\s*:", text)
        return None if match is None else int(match.group(1).replace(",", ""))

    routable = metric("# of routable nets")
    fully_routed = metric("# of fully routed nets")
    routing_errors = metric("# of nets with routing errors")
    unrouted = metric("# of unrouted nets")
    all_routed = (
        routable is not None
        and fully_routed is not None
        and routing_errors == 0
        and fully_routed == routable
    )
    if unrouted is None and all_routed:
        unrouted = 0
    return {
        "routable_nets": routable,
        "fully_routed_nets": fully_routed,
        "routing_error_nets": routing_errors,
        "unrouted_nets": unrouted,
        "all_routable_nets_fully_routed": all_routed,
    }


def count_patterns(text: str) -> dict[str, int]:
    return {
        "error_lines": len(re.findall(r"^ERROR:", text, re.M)),
        "critical_warning_lines": len(re.findall(r"^CRITICAL WARNING:", text, re.M)),
    }


def find_first_existing(root: Path, names: list[str]) -> tuple[str | None, str]:
    for name in names:
        text = read_optional(root / name)
        if text:
            return name, text
    return None, ""


def summarize(run_dir: Path) -> dict[str, object]:
    run_dir = run_dir.resolve()
    wrapper = parse_key_values(read_optional(run_dir / "wrapper_status.txt"))
    vivado_status = parse_key_values(read_optional(run_dir / "vivado_status.txt"))
    exit_code = read_optional(run_dir / "vivado_exit_code.txt").strip()
    vivado_log = read_optional(run_dir / "vivado.log")

    util_name, util_text = find_first_existing(
        run_dir, ["utilization_post_route.rpt", "utilization_post_place.rpt"]
    )
    timing_name, timing_text = find_first_existing(
        run_dir,
        [
            "timing_summary_post_route_physopt.rpt",
            "timing_summary_post_route.rpt",
            "timing_summary_post_place.rpt",
        ],
    )
    route_status = parse_route_status(read_optional(run_dir / "route_status_post_route.rpt"))
    timing = parse_timing(timing_text)
    log_counts = count_patterns(vivado_log)

    gates: dict[str, bool | None] = {
        "vivado_exit_zero": None if exit_code == "" else exit_code == "0",
        "zero_errors": log_counts["error_lines"] == 0,
        "zero_critical_warnings": log_counts["critical_warning_lines"] == 0,
        "routed_without_errors": route_status["all_routable_nets_fully_routed"],
        "setup_nonnegative": None if timing is None else float(timing["wns_ns"]) >= 0.0,
        "hold_nonnegative": None if timing is None else float(timing["whs_ns"]) >= 0.0,
    }
    decided = all(value is not None for value in gates.values())
    pass_basic = decided and all(bool(value) for value in gates.values())

    return {
        "run_dir": str(run_dir),
        "wrapper": wrapper,
        "vivado_status": vivado_status,
        "vivado_exit_code": exit_code or None,
        "utilization_report": util_name,
        "utilization": parse_utilization(util_text),
        "timing_report": timing_name,
        "timing": timing,
        "route_status": route_status,
        "vivado_log_counts": log_counts,
        "basic_terminal_gates": gates,
        "basic_terminal_pass": pass_basic,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = summarize(args.run_dir)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
