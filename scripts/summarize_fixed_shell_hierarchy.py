#!/usr/bin/env python3
"""Validate fixed-shell Vivado hierarchical-utilization reports.

The canonical expected-client document groups synthesized module names by the
topology bits that require them::

    {
      "schema": "fixed_shell_expected_clients_v1",
      "clients": {
        "fc": ["fc_client_module"],
        "srnn": ["srnn_client_module"],
        "scnn": ["scnn_client_module"]
      },
      "always": ["optional_client_present_in_every_mask"]
    }

A module may appear in more than one group. It is then required whenever any
of those topology bits is enabled and forbidden when all are disabled. An
explicit ``masks`` form with ``required``/``forbidden`` lists is also accepted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


MASK_ORDER = (7, 1, 2, 4)
CONFIGURATION_NAMES = {
    7: "Unified",
    1: "FC-only",
    2: "SRNN-only",
    4: "SCNN-only",
}
TOPOLOGY_BITS = {"fc": 1, "srnn": 2, "scnn": 4}

PHYSICAL_ARRAY_MODULE = "unified_binary_p4_h40_array12_path_v1"
H40_TILE_MODULE = "unified_dual_role_h40_stationary_tile_candidate_v1"
EXPECTED_ARRAY_COUNT = 1
EXPECTED_H40_TILE_COUNT = 12

HIERARCHY_REPORT_NAMES = (
    "utilization_hierarchical_post_synth.rpt",
    "hierarchical_utilization_post_synth.rpt",
    "utilization_hierarchical_synth.rpt",
    "utilization_hierarchical.rpt",
)

# Vivado uniquifies parameter-specialized modules with these terminal forms.
# The numeric rule deliberately requires an underscore, so version tokens such
# as ``_v1`` are retained.
PARAMETER_SUFFIX_PATTERNS = (
    re.compile(r"__parameterized\d+(?:(?:__|_)\d+)?$"),
    re.compile(r"_\d+$"),
)


@dataclass(frozen=True)
class HierarchyRow:
    instance: str
    path: str
    module: str
    normalized_module: str
    resources: tuple[int, ...]


@dataclass(frozen=True)
class ParsedHierarchy:
    rows: tuple[HierarchyRow, ...]
    parenthesized_self_rows_excluded: int


@dataclass(frozen=True)
class MaskExpectation:
    required: tuple[str, ...]
    forbidden: tuple[str, ...]


def normalize_module_name(module: str) -> str:
    """Remove only recognized Vivado parameter/uniquification suffixes."""

    normalized = module.strip()
    while normalized:
        for pattern in PARAMETER_SUFFIX_PATTERNS:
            candidate = pattern.sub("", normalized)
            if candidate != normalized:
                normalized = candidate
                break
        else:
            return normalized
    return normalized


def _parse_resource_count(value: str) -> int:
    compact = value.strip().replace(",", "")
    if not re.fullmatch(r"\d+", compact):
        raise ValueError(f"invalid hierarchy resource count: {value!r}")
    return int(compact)


def parse_hierarchy_report(text: str) -> ParsedHierarchy:
    """Parse the data rows from one Vivado hierarchical utilization report."""

    raw_rows: list[tuple[int, str, str, tuple[int, ...]]] = []
    self_rows = 0
    for line in text.splitlines():
        if not line.startswith("|") or not line.rstrip().endswith("|"):
            continue
        fields = line[1 : line.rfind("|")].split("|")
        if len(fields) != 10:
            continue

        raw_instance = fields[0].rstrip()
        instance = raw_instance.strip()
        module = fields[1].strip()
        if instance == "Instance" or module == "Module" or not instance:
            continue
        if instance.startswith("(") and instance.endswith(")"):
            self_rows += 1
            continue

        try:
            resources = tuple(_parse_resource_count(value) for value in fields[2:])
        except ValueError:
            continue
        indent = len(raw_instance) - len(raw_instance.lstrip(" "))
        raw_rows.append((indent, instance, module, resources))

    if not raw_rows:
        raise ValueError("no Vivado hierarchical-utilization rows found")

    base_indent = min(row[0] for row in raw_rows)
    stack: list[str] = []
    rows: list[HierarchyRow] = []
    for indent, instance, module, resources in raw_rows:
        relative_indent = indent - base_indent
        if relative_indent < 0 or relative_indent % 2:
            raise ValueError(
                f"unsupported hierarchy indentation for instance {instance!r}"
            )
        depth = relative_indent // 2
        if depth > len(stack):
            raise ValueError(f"hierarchy depth jumps before instance {instance!r}")
        del stack[depth:]
        path = "/".join((*stack, instance))
        stack.append(instance)
        rows.append(
            HierarchyRow(
                instance=instance,
                path=path,
                module=module,
                normalized_module=normalize_module_name(module),
                resources=resources,
            )
        )

    return ParsedHierarchy(tuple(rows), self_rows)


def read_report(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


def resolve_hierarchy_report(source: Path) -> Path:
    if source.is_file():
        return source.resolve()
    if not source.is_dir():
        raise ValueError(f"hierarchy report input does not exist: {source}")
    for name in HIERARCHY_REPORT_NAMES:
        candidate = source / name
        if candidate.is_file():
            return candidate.resolve()
    candidates = sorted(source.glob("*hierarch*.rpt"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if candidates:
        raise ValueError(
            f"multiple hierarchy reports found in {source}; pass one report file"
        )
    raise ValueError(f"no hierarchical-utilization report found in {source}")


def _module_list(value: object, context: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array of module names")
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{context}[{index}] must be a non-empty string")
        module = normalize_module_name(item)
        if module in seen:
            raise ValueError(f"duplicate normalized module {module!r} in {context}")
        seen.add(module)
        result.append(module)
    return tuple(result)


def _aliased_list(
    row: Mapping[str, object], aliases: Sequence[str], context: str
) -> tuple[str, ...] | None:
    present = [alias for alias in aliases if alias in row]
    if len(present) > 1:
        raise ValueError(f"{context} uses multiple aliases: {', '.join(present)}")
    if not present:
        return None
    return _module_list(row[present[0]], f"{context}.{present[0]}")


def _expectations_from_groups(
    payload: Mapping[str, object], groups: Mapping[str, object]
) -> dict[int, MaskExpectation]:
    unknown = sorted(set(groups) - set(TOPOLOGY_BITS))
    if unknown:
        raise ValueError("unknown topology client groups: " + ", ".join(unknown))
    missing = sorted(set(TOPOLOGY_BITS) - set(groups))
    if missing:
        raise ValueError("missing topology client groups: " + ", ".join(missing))

    activation: dict[str, int] = {}
    for module in _module_list(payload.get("always"), "always"):
        activation[module] = 7
    for group, bit in TOPOLOGY_BITS.items():
        for module in _module_list(groups[group], f"clients.{group}"):
            activation[module] = activation.get(module, 0) | bit

    if not activation:
        raise ValueError("expected-client JSON contains no client modules")
    return {
        mask: MaskExpectation(
            required=tuple(
                module for module, module_mask in activation.items() if mask & module_mask
            ),
            forbidden=tuple(
                module
                for module, module_mask in activation.items()
                if not (mask & module_mask)
            ),
        )
        for mask in MASK_ORDER
    }


def _expectations_from_masks(
    masks: Mapping[str, object], context: str
) -> dict[int, MaskExpectation]:
    result: dict[int, tuple[tuple[str, ...], tuple[str, ...] | None]] = {}
    universe: list[str] = []
    universe_seen: set[str] = set()
    for mask in MASK_ORDER:
        raw = masks.get(str(mask), masks.get(mask))
        if raw is None:
            raise ValueError(f"{context} is missing mask {mask}")
        if isinstance(raw, list):
            required = _module_list(raw, f"{context}.{mask}")
            forbidden = None
        elif isinstance(raw, dict):
            required = _aliased_list(
                raw,
                ("required", "present", "required_clients", "expected_present"),
                f"{context}.{mask}",
            )
            forbidden = _aliased_list(
                raw,
                ("forbidden", "absent", "forbidden_clients", "expected_absent"),
                f"{context}.{mask}",
            )
            if required is None:
                required = ()
        else:
            raise ValueError(f"{context}.{mask} must be an array or object")
        for module in required:
            if module not in universe_seen:
                universe_seen.add(module)
                universe.append(module)
        result[mask] = (required, forbidden)

    expectations: dict[int, MaskExpectation] = {}
    for mask in MASK_ORDER:
        required, forbidden = result[mask]
        if forbidden is None:
            required_set = set(required)
            forbidden = tuple(module for module in universe if module not in required_set)
        overlap = sorted(set(required) & set(forbidden))
        if overlap:
            raise ValueError(
                f"{context}.{mask} requires and forbids: {', '.join(overlap)}"
            )
        expectations[mask] = MaskExpectation(required, forbidden)
    return expectations


def parse_expected_clients(payload: object) -> dict[int, MaskExpectation]:
    if not isinstance(payload, dict):
        raise ValueError("expected-client JSON root must be an object")

    groups = payload.get("client_groups")
    if groups is None:
        clients = payload.get("clients")
        if isinstance(clients, dict) and set(clients).issubset(TOPOLOGY_BITS):
            groups = clients
    if groups is None and any(key in payload for key in TOPOLOGY_BITS):
        groups = {key: payload.get(key) for key in TOPOLOGY_BITS}
    if groups is not None:
        if not isinstance(groups, dict):
            raise ValueError("client_groups must be a JSON object")
        return _expectations_from_groups(payload, groups)

    masks = payload.get("masks", payload.get("expected_clients"))
    if masks is None and all(str(mask) in payload for mask in MASK_ORDER):
        masks = payload
    if not isinstance(masks, dict):
        raise ValueError(
            "expected-client JSON must define topology client groups or masks"
        )
    return _expectations_from_masks(masks, "masks")


def load_expected_clients(path: Path) -> dict[int, MaskExpectation]:
    return parse_expected_clients(json.loads(path.read_text(encoding="utf-8-sig")))


def _matches(rows: Sequence[HierarchyRow], module: str) -> list[HierarchyRow]:
    return [row for row in rows if row.normalized_module == module]


def summarize_mask(
    mask: int,
    report_path: Path,
    parsed: ParsedHierarchy,
    expectation: MaskExpectation,
) -> dict[str, object]:
    module_counts = Counter(row.normalized_module for row in parsed.rows)
    array_rows = _matches(parsed.rows, PHYSICAL_ARRAY_MODULE)
    tile_rows = _matches(parsed.rows, H40_TILE_MODULE)

    required_clients = [
        {
            "module": module,
            "count": module_counts[module],
            "paths": [row.path for row in _matches(parsed.rows, module)],
            "present": module_counts[module] > 0,
        }
        for module in expectation.required
    ]
    forbidden_clients = [
        {
            "module": module,
            "count": module_counts[module],
            "paths": [row.path for row in _matches(parsed.rows, module)],
            "absent": module_counts[module] == 0,
        }
        for module in expectation.forbidden
    ]
    checks = {
        "exactly_one_physical_array": len(array_rows) == EXPECTED_ARRAY_COUNT,
        "exactly_twelve_h40_tiles": len(tile_rows) == EXPECTED_H40_TILE_COUNT,
        "all_required_clients_present": all(
            bool(client["present"]) for client in required_clients
        ),
        "all_forbidden_clients_absent": all(
            bool(client["absent"]) for client in forbidden_clients
        ),
    }
    return {
        "mask": mask,
        "configuration": CONFIGURATION_NAMES[mask],
        "status": "pass" if all(checks.values()) else "fail",
        "report": str(report_path),
        "counts": {
            "parsed_rows": len(parsed.rows),
            "parenthesized_self_rows_excluded": (
                parsed.parenthesized_self_rows_excluded
            ),
            "physical_arrays": len(array_rows),
            "h40_tiles": len(tile_rows),
        },
        "checks": checks,
        "array_paths": [row.path for row in array_rows],
        "h40_tile_paths": [row.path for row in tile_rows],
        "required_clients": required_clients,
        "forbidden_clients": forbidden_clients,
    }


def summarize_hierarchies(
    report_sources: Mapping[int, Path],
    expectations: Mapping[int, MaskExpectation],
) -> dict[str, object]:
    if set(report_sources) != set(MASK_ORDER):
        raise ValueError("reports must contain exactly masks 7, 1, 2, and 4")
    if set(expectations) != set(MASK_ORDER):
        raise ValueError("expectations must contain exactly masks 7, 1, 2, and 4")

    masks: list[dict[str, object]] = []
    for mask in MASK_ORDER:
        report_path = resolve_hierarchy_report(report_sources[mask])
        parsed = parse_hierarchy_report(read_report(report_path))
        masks.append(summarize_mask(mask, report_path, parsed, expectations[mask]))

    return {
        "schema": "fixed_shell_hierarchy_summary_v1",
        "status": "pass" if all(row["status"] == "pass" for row in masks) else "fail",
        "invariants": {
            "physical_array_module": PHYSICAL_ARRAY_MODULE,
            "expected_physical_array_count": EXPECTED_ARRAY_COUNT,
            "h40_tile_module": H40_TILE_MODULE,
            "expected_h40_tile_count": EXPECTED_H40_TILE_COUNT,
            "parenthesized_self_rows_are_excluded": True,
        },
        "masks": masks,
        "claim_boundary": (
            "Post-synthesis hierarchy evidence only. This validates the one-array, "
            "twelve-tile structure and configured client presence or absence; it "
            "does not prove function, timing, route, power, or board behavior."
        ),
    }


def write_json(summary: Mapping[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )


def _escape_markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown(summary: Mapping[str, object], destination: Path) -> None:
    status = str(summary["status"]).upper()
    lines = [
        "# Fixed-Shell Hierarchy Summary",
        "",
        f"Overall hierarchy validation: **{status}**.",
        "",
        "Parenthesized Vivado self rows are excluded before module counting, and",
        "recognized parameter-specialization suffixes are normalized.",
        "",
        "| Mask | Configuration | Arrays | H40 tiles | Required clients | "
        "Forbidden clients absent | Status |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    masks = summary["masks"]
    assert isinstance(masks, list)
    for row in masks:
        assert isinstance(row, dict)
        counts = row["counts"]
        required = row["required_clients"]
        forbidden = row["forbidden_clients"]
        assert isinstance(counts, dict)
        assert isinstance(required, list)
        assert isinstance(forbidden, list)
        required_present = sum(bool(item["present"]) for item in required)
        forbidden_absent = sum(bool(item["absent"]) for item in forbidden)
        lines.append(
            f"| {row['mask']} | {row['configuration']} | "
            f"{counts['physical_arrays']} | {counts['h40_tiles']} | "
            f"{required_present}/{len(required)} | "
            f"{forbidden_absent}/{len(forbidden)} | "
            f"{str(row['status']).upper()} |"
        )

    for row in masks:
        assert isinstance(row, dict)
        counts = row["counts"]
        assert isinstance(counts, dict)
        lines.extend(
            [
                "",
                f"## Mask {row['mask']}: {row['configuration']}",
                "",
                f"Report: `{_escape_markdown(row['report'])}`",
                "",
                "| Expectation | Module | Count | Status |",
                "|---|---|---:|---|",
            ]
        )
        required = row["required_clients"]
        forbidden = row["forbidden_clients"]
        assert isinstance(required, list)
        assert isinstance(forbidden, list)
        for client in required:
            lines.append(
                f"| Required present | `{_escape_markdown(client['module'])}` | "
                f"{client['count']} | {'PASS' if client['present'] else 'FAIL'} |"
            )
        for client in forbidden:
            lines.append(
                f"| Required absent | `{_escape_markdown(client['module'])}` | "
                f"{client['count']} | {'PASS' if client['absent'] else 'FAIL'} |"
            )
        if not required and not forbidden:
            lines.append("| None | NA | 0 | PASS |")
        lines.extend(
            [
                "",
                "Structural counts: "
                f"{counts['physical_arrays']} physical array(s), "
                f"{counts['h40_tiles']} H40 tile(s), and "
                f"{counts['parenthesized_self_rows_excluded']} excluded self row(s).",
            ]
        )

    lines.extend(["", str(summary["claim_boundary"]), ""])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Vivado hierarchy reports for TOPOLOGY_MASK values "
            "7, 1, 2, and 4. Each input may be a report or its result directory."
        )
    )
    parser.add_argument("mask7_report", type=Path, metavar="MASK7_REPORT")
    parser.add_argument("mask1_report", type=Path, metavar="MASK1_REPORT")
    parser.add_argument("mask2_report", type=Path, metavar="MASK2_REPORT")
    parser.add_argument("mask4_report", type=Path, metavar="MASK4_REPORT")
    parser.add_argument(
        "--expected-clients",
        "--expectations",
        required=True,
        type=Path,
        dest="expected_clients",
        metavar="PATH",
        help="JSON document describing required and forbidden client modules",
    )
    parser.add_argument(
        "--json",
        "--json-output",
        required=True,
        type=Path,
        dest="json_path",
        metavar="PATH",
        help="write the machine-readable hierarchy summary",
    )
    parser.add_argument(
        "--markdown",
        "--markdown-output",
        required=True,
        type=Path,
        dest="markdown_path",
        metavar="PATH",
        help="write the human-readable hierarchy summary",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.json_path.resolve() == args.markdown_path.resolve():
        parser.error("--json and --markdown must name different files")

    report_sources = {
        7: args.mask7_report,
        1: args.mask1_report,
        2: args.mask2_report,
        4: args.mask4_report,
    }
    try:
        expectations = load_expected_clients(args.expected_clients)
        summary = summarize_hierarchies(report_sources, expectations)
        write_json(summary, args.json_path)
        write_markdown(summary, args.markdown_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"fixed-shell hierarchy input error: {exc}", file=sys.stderr)
        return 2

    print(
        "FIXED_SHELL_HIERARCHY_WRITTEN "
        f"status={str(summary['status']).upper()} "
        f"json={args.json_path.resolve()} "
        f"markdown={args.markdown_path.resolve()}"
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
