#!/usr/bin/env python3
"""Build a paper-facing post-synthesis fixed-shell comparison.

The parser is intentionally read-only with respect to Vivado results. It reports
comparability and missing data, but it does not act as a workflow gate.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


CONFIG_ORDER = ("Unified", "FC-only", "SRNN-only", "SCNN-only")
FIXED_CONFIGS = CONFIG_ORDER[1:]

UTILIZATION_REPORT_NAMES = (
    "utilization_post_synth.rpt",
    "post_synth_utilization.rpt",
    "utilization_synth.rpt",
    "utilization.rpt",
)
TIMING_REPORT_NAMES = (
    "timing_summary_post_synth.rpt",
    "post_synth_timing_summary.rpt",
    "timing_summary_synth.rpt",
    "timing_summary.rpt",
)
STATUS_FILE_NAMES = (
    "vivado_status.txt",
    "synth_status.txt",
    "status.txt",
    "wrapper_status.txt",
)

RESOURCE_ROWS = (
    ("Slice LUT", "slice_luts"),
    ("Registers", "registers"),
    ("RAMB36", "ramb36"),
    ("RAMB18", "ramb18"),
    ("BRAM36eq", "bram36eq"),
    ("DSP", "dsp"),
)
TIMING_ROWS = (
    ("WNS", "wns_ns"),
    ("WHS", "whs_ns"),
)

CSV_FIELDS = (
    "Section",
    "Metric",
    "Unit",
    *CONFIG_ORDER,
    "IndependentSum",
    "Avoided",
    "Saving",
    "ReuseFactor",
    "MaxFixedShell",
    "UnifiedPremiumVsMaxFixedShell",
    "Status",
)


@dataclass(frozen=True)
class RunSummary:
    label: str
    run_dir: Path
    metrics: dict[str, float | None]
    stage: str | None
    top: str | None
    part: str | None
    period_ns: float | None
    run_status: str | None
    utilization_report: str | None
    timing_report: str | None
    status_files: tuple[str, ...]


@dataclass(frozen=True)
class DerivedResource:
    independent_sum: float | None
    avoided: float | None
    saving: float | None
    reuse_factor: float | None
    max_fixed_shell: float | None
    unified_premium: float | None


@dataclass(frozen=True)
class ComparabilityCheck:
    name: str
    unit: str
    values: dict[str, str | float | None]
    status: str


@dataclass(frozen=True)
class ComparisonSummary:
    runs: dict[str, RunSummary]
    derived: dict[str, DerivedResource]
    comparability: tuple[ComparabilityCheck, ...]
    overall_comparability: str


def read_optional(path: Path) -> str:
    if not path.is_file():
        return ""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def find_first_existing(root: Path, names: tuple[str, ...]) -> tuple[Path | None, str]:
    for name in names:
        path = root / name
        if path.is_file():
            return path, read_optional(path)
    return None, ""


def parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lower()] = value.strip()
    return values


def parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value.strip().replace(",", ""))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_utilization(text: str) -> dict[str, float | None]:
    aliases = {
        "Slice LUTs": "slice_luts",
        "Slice Registers": "registers",
        "RAMB36/FIFO": "ramb36",
        "RAMB36": "ramb36",
        "RAMB18": "ramb18",
        "DSPs": "dsp",
        "DSP48E1": "dsp",
    }
    values: dict[str, float | None] = {
        "slice_luts": None,
        "registers": None,
        "ramb36": None,
        "ramb18": None,
        "dsp": None,
    }
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) < 2:
            continue
        label = fields[0].rstrip("*").strip()
        key = aliases.get(label)
        if key is None or values[key] is not None:
            continue
        values[key] = parse_number(fields[1])
    return values


def parse_timing(text: str) -> dict[str, float | None]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip().startswith("WNS(ns)") or "WHS(ns)" not in line:
            continue
        for candidate in lines[index + 1 : index + 6]:
            numbers = re.findall(
                r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?",
                candidate,
            )
            if len(numbers) >= 6:
                return {
                    "wns_ns": parse_number(numbers[0]),
                    "whs_ns": parse_number(numbers[4]),
                }
    return {"wns_ns": None, "whs_ns": None}


def parse_clock_period(text: str) -> float | None:
    periods = {
        value
        for match in re.finditer(
            r"^\s*\S+\s+\{[^}]+\}\s+"
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+"
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*$",
            text,
            re.MULTILINE,
        )
        if (value := parse_number(match.group(1))) is not None
    }
    return next(iter(periods)) if len(periods) == 1 else None


def parse_report_metadata(text: str) -> dict[str, str]:
    labels = {
        "design": "top",
        "device": "part",
        "design state": "design_state",
    }
    metadata: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\|\s*([^:|]+?)\s*:\s*(.*?)\s*$", line)
        if match is None:
            continue
        key = labels.get(match.group(1).strip().lower())
        if key is not None and match.group(2).strip():
            metadata.setdefault(key, match.group(2).strip())
    return metadata


def first_status_value(status: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = status.get(key)
        if value:
            return value
    return None


def first_status_number(status: dict[str, str], keys: tuple[str, ...]) -> float | None:
    return parse_number(first_status_value(status, keys))


def summarize_run(label: str, run_dir: Path) -> RunSummary:
    run_dir = run_dir.resolve()
    utilization_path, utilization_text = find_first_existing(
        run_dir, UTILIZATION_REPORT_NAMES
    )
    timing_path, timing_text = find_first_existing(run_dir, TIMING_REPORT_NAMES)

    status: dict[str, str] = {}
    status_files: list[str] = []
    for name in STATUS_FILE_NAMES:
        path = run_dir / name
        if not path.is_file():
            continue
        status_files.append(name)
        for key, value in parse_key_values(read_optional(path)).items():
            status.setdefault(key, value)

    metrics = parse_utilization(utilization_text)
    status_metric_keys = {
        "slice_luts": ("slice_luts", "luts"),
        "registers": ("slice_registers", "registers", "ffs"),
        "ramb36": ("ramb36", "ramb36_count", "ramb36e1_count"),
        "ramb18": ("ramb18", "ramb18_count", "ramb18e1_count"),
        "dsp": ("dsps", "dsp", "dsp48e1_count"),
    }
    for key, aliases in status_metric_keys.items():
        if metrics[key] is None:
            metrics[key] = first_status_number(status, aliases)

    if metrics["ramb36"] is not None and metrics["ramb18"] is not None:
        metrics["bram36eq"] = metrics["ramb36"] + 0.5 * metrics["ramb18"]
    else:
        metrics["bram36eq"] = None

    timing = parse_timing(timing_text)
    metrics["wns_ns"] = timing["wns_ns"]
    metrics["whs_ns"] = timing["whs_ns"]
    if metrics["wns_ns"] is None:
        metrics["wns_ns"] = first_status_number(status, ("wns_ns", "wns"))
    if metrics["whs_ns"] is None:
        metrics["whs_ns"] = first_status_number(status, ("whs_ns", "whs"))

    report_metadata: dict[str, str] = {}
    for text in (utilization_text, timing_text):
        for key, value in parse_report_metadata(text).items():
            report_metadata.setdefault(key, value)

    top = first_status_value(status, ("top", "top_module")) or report_metadata.get(
        "top"
    )
    part = first_status_value(status, ("part", "target_part")) or report_metadata.get(
        "part"
    )
    period_ns = first_status_number(
        status, ("period_ns", "clock_period_ns", "target_period_ns")
    )
    if period_ns is None:
        period_ns = parse_clock_period(timing_text)

    design_state = report_metadata.get("design_state", "").lower()
    selected_names = [
        path.name.lower()
        for path in (utilization_path, timing_path)
        if path is not None
    ]
    evidence = status.get("evidence", "").lower()
    if (
        "synthesized" in design_state
        or any("synth" in name for name in selected_names)
        or "post-synthesis" in evidence
    ):
        stage = "post-synthesis"
    else:
        stage = None

    return RunSummary(
        label=label,
        run_dir=run_dir,
        metrics=metrics,
        stage=stage,
        top=top,
        part=part,
        period_ns=period_ns,
        run_status=first_status_value(status, ("status",)),
        utilization_report=None if utilization_path is None else utilization_path.name,
        timing_report=None if timing_path is None else timing_path.name,
        status_files=tuple(status_files),
    )


def derive_resource(runs: dict[str, RunSummary], metric: str) -> DerivedResource:
    unified = runs["Unified"].metrics[metric]
    fixed_values = [runs[label].metrics[metric] for label in FIXED_CONFIGS]
    complete = unified is not None and all(value is not None for value in fixed_values)
    if not complete:
        return DerivedResource(None, None, None, None, None, None)

    assert unified is not None
    complete_fixed = [float(value) for value in fixed_values if value is not None]
    independent_sum = sum(complete_fixed)
    avoided = independent_sum - unified
    max_fixed_shell = max(complete_fixed)
    saving = None if independent_sum == 0 else 1.0 - unified / independent_sum
    reuse_factor = None if unified == 0 else independent_sum / unified
    unified_premium = (
        None if max_fixed_shell == 0 else (unified - max_fixed_shell) / max_fixed_shell
    )
    return DerivedResource(
        independent_sum=independent_sum,
        avoided=avoided,
        saving=saving,
        reuse_factor=reuse_factor,
        max_fixed_shell=max_fixed_shell,
        unified_premium=unified_premium,
    )


def compare_available(
    name: str,
    unit: str,
    values: dict[str, str | float | None],
    normalize: Callable[[str | float], object],
) -> ComparabilityCheck:
    known = [normalize(value) for value in values.values() if value is not None]
    if not known:
        status = "NA"
    elif any(value != known[0] for value in known[1:]):
        status = "FAIL"
    elif len(known) == len(CONFIG_ORDER):
        status = "PASS"
    else:
        status = "PARTIAL"
    return ComparabilityCheck(name=name, unit=unit, values=values, status=status)


def stage_check(runs: dict[str, RunSummary]) -> ComparabilityCheck:
    values: dict[str, str | float | None] = {
        label: runs[label].stage for label in CONFIG_ORDER
    }
    known = [value for value in values.values() if value is not None]
    if any(value != "post-synthesis" for value in known):
        status = "FAIL"
    elif len(known) == len(CONFIG_ORDER):
        status = "PASS"
    elif known:
        status = "PARTIAL"
    else:
        status = "NA"
    return ComparabilityCheck("Stage", "text", values, status)


def build_comparability(runs: dict[str, RunSummary]) -> tuple[ComparabilityCheck, ...]:
    checks = [stage_check(runs)]
    checks.append(
        compare_available(
            "Top",
            "text",
            {label: runs[label].top for label in CONFIG_ORDER},
            lambda value: str(value).strip(),
        )
    )
    checks.append(
        compare_available(
            "Part",
            "text",
            {label: runs[label].part for label in CONFIG_ORDER},
            lambda value: re.sub(r"\s+", "", str(value)).lower(),
        )
    )
    checks.append(
        compare_available(
            "Period",
            "ns",
            {label: runs[label].period_ns for label in CONFIG_ORDER},
            lambda value: round(float(value), 12),
        )
    )
    return tuple(checks)


def summarize_comparison(run_dirs: dict[str, Path]) -> ComparisonSummary:
    runs = {
        label: summarize_run(label, run_dirs[label]) for label in CONFIG_ORDER
    }
    derived = {
        metric: derive_resource(runs, metric) for _, metric in RESOURCE_ROWS
    }
    comparability = build_comparability(runs)
    statuses = [check.status for check in comparability]
    if "FAIL" in statuses:
        overall = "FAIL"
    elif all(status == "PASS" for status in statuses):
        overall = "PASS"
    else:
        overall = "PARTIAL"
    return ComparisonSummary(runs, derived, comparability, overall)


def format_number(value: float | None, *, grouping: bool = False) -> str:
    if value is None:
        return "NA"
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-12):
        return f"{int(round(value)):,}" if grouping else str(int(round(value)))
    rendered = f"{value:,.6f}" if grouping else f"{value:.6f}"
    return rendered.rstrip("0").rstrip(".")


def format_ratio(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6f}"


def format_percent(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}%"


def format_factor(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}x"


def format_text(value: str | float | None, *, grouping: bool = False) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return format_number(value, grouping=grouping)
    return value


def csv_rows(summary: ComparisonSummary) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for display_name, metric in RESOURCE_ROWS:
        derived = summary.derived[metric]
        row = {
            "Section": "Resource",
            "Metric": display_name,
            "Unit": "count",
            "IndependentSum": format_number(derived.independent_sum),
            "Avoided": format_number(derived.avoided),
            "Saving": format_ratio(derived.saving),
            "ReuseFactor": format_ratio(derived.reuse_factor),
            "MaxFixedShell": format_number(derived.max_fixed_shell),
            "UnifiedPremiumVsMaxFixedShell": format_ratio(
                derived.unified_premium
            ),
            "Status": "NA",
        }
        for label in CONFIG_ORDER:
            row[label] = format_number(summary.runs[label].metrics[metric])
        rows.append(row)

    for display_name, metric in TIMING_ROWS:
        row = {
            "Section": "Timing",
            "Metric": display_name,
            "Unit": "ns",
            "IndependentSum": "NA",
            "Avoided": "NA",
            "Saving": "NA",
            "ReuseFactor": "NA",
            "MaxFixedShell": "NA",
            "UnifiedPremiumVsMaxFixedShell": "NA",
            "Status": "NA",
        }
        for label in CONFIG_ORDER:
            row[label] = format_number(summary.runs[label].metrics[metric])
        rows.append(row)

    for check in summary.comparability:
        row = {
            "Section": "Comparability",
            "Metric": check.name,
            "Unit": check.unit,
            "IndependentSum": "NA",
            "Avoided": "NA",
            "Saving": "NA",
            "ReuseFactor": "NA",
            "MaxFixedShell": "NA",
            "UnifiedPremiumVsMaxFixedShell": "NA",
            "Status": check.status,
        }
        for label in CONFIG_ORDER:
            row[label] = format_text(check.values[label])
        rows.append(row)

    rows.append(
        {
            "Section": "Comparability",
            "Metric": "Overall",
            "Unit": "text",
            **{label: "NA" for label in CONFIG_ORDER},
            "IndependentSum": "NA",
            "Avoided": "NA",
            "Saving": "NA",
            "ReuseFactor": "NA",
            "MaxFixedShell": "NA",
            "UnifiedPremiumVsMaxFixedShell": "NA",
            "Status": summary.overall_comparability,
        }
    )
    return rows


def write_csv(summary: ComparisonSummary, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows(summary))


def escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_markdown(summary: ComparisonSummary, destination: Path) -> None:
    lines = [
        "# Fixed-Shell Post-Synthesis Comparison",
        "",
        f"Overall comparability: **{summary.overall_comparability}**.",
        "",
        "All resource formulas are computed independently. No weighted area score is",
        "produced. `NA` means that a source metric is unavailable or a formula cannot",
        "be computed without inventing a value.",
        "",
        "## Stage Comparability",
        "",
        "| Check | Unit | Unified | FC-only | SRNN-only | SCNN-only | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for check in summary.comparability:
        values = [
            escape_markdown(format_text(check.values[label])) for label in CONFIG_ORDER
        ]
        lines.append(
            f"| {check.name} | {check.unit} | "
            + " | ".join(values)
            + f" | {check.status} |"
        )

    lines.extend(
        [
            "",
            "`PASS` requires all four values to be available and equal. `PARTIAL`",
            "means the available values agree but one or more values are missing;",
            "`FAIL` identifies an observed mismatch.",
            "",
            "## Resource Reuse",
            "",
            "| Resource | Unified | FC-only | SRNN-only | SCNN-only | IndependentSum | Avoided | Saving | ReuseFactor | Max fixed shell | Unified premium vs max fixed shell |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for display_name, metric in RESOURCE_ROWS:
        run_values = [
            format_number(summary.runs[label].metrics[metric], grouping=True)
            for label in CONFIG_ORDER
        ]
        derived = summary.derived[metric]
        lines.append(
            f"| {display_name} | "
            + " | ".join(run_values)
            + " | "
            + " | ".join(
                (
                    format_number(derived.independent_sum, grouping=True),
                    format_number(derived.avoided, grouping=True),
                    format_percent(derived.saving),
                    format_factor(derived.reuse_factor),
                    format_number(derived.max_fixed_shell, grouping=True),
                    format_percent(derived.unified_premium),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "`BRAM36eq = RAMB36 + 0.5 * RAMB18`. `IndependentSum` is the sum",
            "of the three fixed shells. `Avoided = IndependentSum - Unified`,",
            "`Saving = 1 - Unified / IndependentSum`, `ReuseFactor =",
            "IndependentSum / Unified`, and unified premium is relative to the largest",
            "single fixed shell.",
            "",
            "## Timing",
            "",
            "| Metric | Unit | Unified | FC-only | SRNN-only | SCNN-only |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for display_name, metric in TIMING_ROWS:
        values = [
            format_number(summary.runs[label].metrics[metric])
            for label in CONFIG_ORDER
        ]
        lines.append(
            f"| {display_name} | ns | " + " | ".join(values) + " |"
        )

    lines.extend(
        [
            "",
            "## Inputs",
            "",
            "| Configuration | Result directory | Run status | Utilization report | Timing report | Status files |",
            "|---|---|---|---|---|---|",
        ]
    )
    for label in CONFIG_ORDER:
        run = summary.runs[label]
        status_files = ", ".join(run.status_files) if run.status_files else "NA"
        lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    escape_markdown(str(run.run_dir)),
                    escape_markdown(run.run_status or "NA"),
                    escape_markdown(run.utilization_report or "NA"),
                    escape_markdown(run.timing_report or "NA"),
                    escape_markdown(status_files),
                )
            )
            + " |"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize matched Unified, FC-only, SRNN-only, and SCNN-only "
            "terminal Vivado post-synthesis result directories."
        )
    )
    parser.add_argument("unified_dir", type=Path, metavar="UNIFIED_DIR")
    parser.add_argument("fc_only_dir", type=Path, metavar="FC_ONLY_DIR")
    parser.add_argument("srnn_only_dir", type=Path, metavar="SRNN_ONLY_DIR")
    parser.add_argument("scnn_only_dir", type=Path, metavar="SCNN_ONLY_DIR")
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        dest="csv_path",
        metavar="PATH",
        help="write the comparison table as CSV",
    )
    parser.add_argument(
        "--markdown",
        required=True,
        type=Path,
        dest="markdown_path",
        metavar="PATH",
        help="write the paper-facing report as Markdown",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run_dirs = {
        "Unified": args.unified_dir,
        "FC-only": args.fc_only_dir,
        "SRNN-only": args.srnn_only_dir,
        "SCNN-only": args.scnn_only_dir,
    }
    missing_dirs = [str(path) for path in run_dirs.values() if not path.is_dir()]
    if missing_dirs:
        parser.error("result directory does not exist: " + ", ".join(missing_dirs))
    if args.csv_path.resolve() == args.markdown_path.resolve():
        parser.error("--csv and --markdown must name different files")

    summary = summarize_comparison(run_dirs)
    write_csv(summary, args.csv_path)
    write_markdown(summary, args.markdown_path)
    print(
        "FIXED_SHELL_COMPARISON_WRITTEN "
        f"comparability={summary.overall_comparability} "
        f"csv={args.csv_path.resolve()} markdown={args.markdown_path.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
