#!/usr/bin/env python3
"""Prepare a count-preserving, pre-LIF scan comparison; never run EDA."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import supplement_srnn_backbone_numeric as bone
from run_srnn_resident_remote import local_path

SOURCE = Path(local_path(ROOT / "verification/experiments/srnn_representative_inputs/a_r1"))
OUTPUT = Path(local_path(ROOT / "verification/experiments/nearest_neighbor/scan_fixture_r1"))
RESULT = Path(local_path(ROOT / "results/paper_evidence/supplement_20261015/performance_optimization/nearest_neighbor_spiker_r1"))


def dense_groups(counts):
    """Four-lane adaptation of whole-vector-zero skip plus sequential scan.

    Count repetition stays within one transaction. This is our literature
    mechanism adaptation, not the author's binary-input VHDL implementation.
    """
    counts = np.asarray(counts)
    if counts.shape != (700,) or np.any((counts < 0) | (counts > 255)):
        raise ValueError("expected one 700-source uint8 count vector")
    if not np.any(counts):
        return []
    groups = []
    for start in range(0, 700, 4):
        block = counts[start:start + 4]
        addresses = [(start + lane) // 2 for lane in range(4)]
        for repeat in range(max(1, int(block.max()))):
            mask = sum(int(value > repeat) << lane for lane, value in enumerate(block))
            groups.append((mask, *addresses))
    return groups


def pack(group):
    mask, *addresses = group
    return mask | sum(address << (8 + 11 * lane) for lane, address in enumerate(addresses))


def main():
    selection = json.loads((SOURCE / "selection.json").read_text(encoding="utf-8"))
    fixture = SOURCE / "multi_fixture"
    weights_path = fixture / "sequence_0/w0_refill_dest_major.mem"
    weights = bone.read_signed(weights_path, 16, (450, 700)).T
    if np.any((weights < -128) | (weights > 127)):
        raise ValueError("original INT8 weight range changed")
    rows, all_counts, all_currents = [], [], []
    for seq, selected in enumerate(selection["selected"]):
        steps = selected["time_steps"]
        directory = fixture / f"sequence_{seq}"
        counts = bone.read_prefix(directory / "source_sample.hex", steps)
        stored = bone.read_signed(directory / "raw700_counts.mem", 8, (steps, 1, 700))
        np.testing.assert_array_equal(counts, stored[:, 0])
        with (directory / "feedforward_p4.mem").open(encoding="ascii") as stream:
            word_count = sum(1 for line in stream if line.strip())
        original_words = bone.read_signed(directory / "feedforward_p4.mem", 64, (word_count,))
        offsets = bone.read_signed(directory / "feedforward_offsets.mem", 32, (89,))
        direct = counts @ weights
        if np.any((direct < -(1 << 31)) | (direct >= (1 << 31))):
            raise ValueError("the bounded pre-LIF fixture unexpectedly needs saturation")
        for t, frame in enumerate(counts):
            dense = dense_groups(frame)
            sparse = bone.p4_groups(frame)
            np.testing.assert_array_equal(bone.group_current(dense, weights), direct[t])
            np.testing.assert_array_equal(bone.group_current(sparse, weights), direct[t])
            np.testing.assert_array_equal(
                np.asarray([pack(group) for group in sparse]),
                original_words[int(offsets[t]):int(offsets[t + 1])])
            rows.append(dict(
                sequence=seq, dataset_index=selected["dataset_index"], timestep=t,
                count_sum=int(frame.sum()), occupied_sources=int(np.count_nonzero(frame)),
                max_count=int(frame.max()), input_count_entries=700, four_count_blocks=175,
                whole_vector_zero=int(not np.any(frame)), dense_issue_groups=len(dense),
                packed_issue_groups=len(sparse), checked_current_elements=450))
        all_counts.append(counts)
        all_currents.append(direct)

    # Exercise zero-vector and count-repeat boundaries independently of the data.
    for sources in ({}, {0: 255}, {1: 2, 2: 4, 699: 3}, {i: 1 for i in range(700)}):
        frame = np.zeros(700, dtype=np.int64)
        for source, count in sources.items():
            frame[source] = count
        expected = frame @ weights
        for groups in (dense_groups(frame), bone.p4_groups(frame)):
            np.testing.assert_array_equal(bone.group_current(groups, weights), expected)

    OUTPUT.mkdir(parents=True, exist_ok=False)
    bone.write_hex(OUTPUT / "counts.mem", np.concatenate(all_counts), 8)
    bone.write_hex(OUTPUT / "expected_ff_current.mem", np.concatenate(all_currents), 32)
    bone.write_hex(OUTPUT / "weights_dest_major.mem", weights.T, 16)
    with (OUTPUT / "frames.csv").open("x", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    totals = {key: sum(row[key] for row in rows) for key in (
        "count_sum", "occupied_sources", "input_count_entries", "four_count_blocks",
        "whole_vector_zero", "dense_issue_groups", "packed_issue_groups",
        "checked_current_elements")}
    report = dict(
        status="PASS_pre_LIF_software_fixture_only_not_RTL",
        frame_count=len(rows), selected_indices=[s["dataset_index"] for s in selection["selected"]],
        totals=totals, boundary_cases=4,
        source_fixture="verification/experiments/srnn_representative_inputs/a_r1/multi_fixture",
        weight_shape=[700, 450], weight_precision="original signed INT8 in sign-extended 16-bit words",
        current_precision="signed32; all checked sums in range, no LIF or state update",
        address_mapping="source=2*row+(lane&1); four adjacent sources use rows r,r,r+1,r+1",
        mechanism="whole-vector-zero skip plus sequential four-source scan with count repetition",
        adaptation="four-lane count-preserving literature mechanism, NOT author RTL or published performance",
        packed_source="existing bone.p4_groups software producer, byte-equivalent packets to E5 prepared input",
        preparation_boundary="Both must receive all 700 count entries and inspect all 175 blocks per frame. "
            "Counts are work, not cycles. No free host packing, assumed overlap, physical port mapping, "
            "or zero-cost group discovery is included in a measured-throughput claim.",
        next_measurement="Use equal actual weight ports/capacity and the same consumer. Include ingress, "
            "group preparation, backpressure, and drain in RTL intervals; separately report issue groups. "
            "Do not substitute issue-group ratio for end-to-end speedup.",
        files={p.name: p.stat().st_size for p in OUTPUT.iterdir() if p.is_file()})
    bone.json_write(OUTPUT / "fixture.json", report)
    bone.json_write(RESULT / "scan_fixture_check.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
