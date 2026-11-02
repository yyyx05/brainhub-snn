#!/usr/bin/env python3
"""Prepare the fixed shifted B4T12 natural crossing; never execute RTL or SSH.

Window-major, then sample-major scheduling matches the existing B4T4 TB layout.
Unlike that cold-window runner, all four sample contexts survive T4 commits.
Only raw inputs and initial parameters are stimulus; expectations are read-only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from supplement_scnn_same_dut_b4t4_reference import (
    ROOT, FEATURES, CLASSES, PARAMS, FRAME_SECONDS, BUILD_SECONDS,
    apply_update, write_signed_hex,
)
from supplement_scnn_graph_reference import softmax_error_q016
from analyze_scnn_b4t4_rtl_update_contract import round_nearest_even_ratio

ALGORITHM = ROOT / "results/paper_evidence/supplement_20261004/algorithm/scnn_integer_adaptation"
REPORT = ALGORITHM / "report_archive_order_cuda_20261008"
BASE = ROOT / "results/paper_evidence/supplement_20261004/rtl/scnn_same_dut/natural_visibility_prepare"
SEED, BATCH, WINDOW, STEPS, QUERY_STEPS = 38472, 4, 4, 12, 4
STATUS = "prepared_integer_reference_not_RTL_PASS"


def load_kernel():
    spec = importlib.util.spec_from_file_location("scnn_temporal_integer_classifier", ALGORITHM / "integer_classifier.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_episode():
    episode = json.loads((REPORT / "seed38472_29_to_28.json").read_text(encoding="utf-8"))
    plan = json.loads((REPORT / "plan.json").read_text(encoding="utf-8"))
    required = dict(mask="classifier", adapt_batch_size=4, window_timesteps=4,
                    optimizer_lr=5 * 2.0**-22, update_rule="fixed_sgd",
                    weight_fractional_bits=15, fixed_sgd_residual_bits=15,
                    temporal_jitter=1, support_domain="shifted", commit_scope="window",
                    classifier_row_mask="all", classifier_sample_gate="all",
                    shift_x=0, shift_y=0, event_drop_probability=0.0,
                    polarity_flip_probability=0.0, full_adaptation_batches_only=True)
    for key, value in required.items():
        if plan["selected"][key] != value:
            raise ValueError(f"unexpected formal protocol: {key}")
    if (episode["seed"], episode["support_subject"], episode["query_subject"]) != (SEED, 29, 28):
        raise ValueError("the requested fixed episode is unavailable")
    if episode["used_support_batches"][0] != [0, 1, 2, 3] or episode["query_indices"][0] != 22:
        raise ValueError("the formal first support batch/query member changed")
    return episode


def frame_schedule(support_ids, query_id):
    if support_ids != [0, 1, 2, 3] or query_id != 22:
        raise ValueError("use the fixed formal support and first query member")
    rows = []
    for window in range(STEPS // WINDOW):
        for sample, sample_id in enumerate(support_ids):
            for local_t in range(WINDOW):
                t = window * WINDOW + local_t
                rows.append(dict(frame=len(rows), query=False, sample_slot=sample,
                                 sample_id=sample_id, timestep=t, window_1based=window + 1,
                                 local_timestep=local_t, capture_row=sample * WINDOW + local_t,
                                 sample_state_reset=t == 0, context_restore=t != 0,
                                 context_save=True, learning=True,
                                 completed_windows=window,
                                 wait_for_commit_after=sample == BATCH-1 and local_t == WINDOW-1))
    for t in range(QUERY_STEPS):
        rows.append(dict(frame=len(rows), query=True, sample_slot=0, sample_id=query_id,
                         timestep=t, window_1based=None, local_timestep=t, capture_row=None,
                         sample_state_reset=t == 0, context_restore=t != 0,
                         context_save=True, learning=False, completed_windows=3,
                         wait_for_commit_after=False))
    return rows


def pack_frames(values, schedule):
    """Convert [sample, absolute time, ...] into the explicit DUT frame order."""
    return np.stack([values[r["sample_slot"], r["timestep"]] for r in schedule])


def scheduled_reference(features, labels, initial, schedule):
    """Independent frame-serial replay with four separate sample contexts."""
    if features.shape != (BATCH, STEPS, FEATURES) or initial.shape != (CLASSES, FEATURES):
        raise ValueError("expected main-size B4T12 features and 2048x11 codes")
    if len(labels) != BATCH or len(schedule) != BATCH * STEPS:
        raise ValueError("incomplete B4 schedule")
    codes, residual = initial.copy(), np.zeros_like(initial)
    traces = np.zeros((BATCH, FEATURES), dtype=np.int64)
    cumulative = np.zeros((BATCH, CLASSES), dtype=np.int64)
    gradient = np.zeros_like(initial)
    next_t = [0] * BATCH
    observations = dict(currents=[], centered=[], cumulative=[], traces=[])
    commits, snapshots = [], []
    for frame, row in enumerate(schedule):
        sample, t = row["sample_slot"], row["timestep"]
        if (row["query"] or row["frame"] != frame or t != next_t[sample]
                or row["completed_windows"] != len(commits)
                or row["sample_state_reset"] != (t == 0)
                or t // WINDOW != len(commits)):
            raise ValueError("reset, chronology or completed-window mismatch")
        x = features[sample, t]
        current = np.clip((codes @ x) >> 15, -32768, 32767)
        cumulative[sample] = (cumulative[sample] + current + (1 << 31)) % (1 << 32) - (1 << 31)
        centered_raw = (cumulative[sample] - cumulative[sample].max() + (1 << 31)) % (1 << 32) - (1 << 31)
        centered = np.clip(centered_raw, -32768, 32767)
        traces[sample] = (((traces[sample] * 250) >> 8) + x + 32768) % 65536 - 32768
        errors = np.asarray(softmax_error_q016(centered.tolist(), int(labels[sample])), dtype=np.int64)
        gradient = (gradient + errors[:, None] * (2 * traces[sample]) + (1 << 47)) % (1 << 48) - (1 << 47)
        for key, value in (("currents", current), ("centered", centered),
                           ("cumulative", cumulative[sample]), ("traces", traces[sample])):
            observations[key].append(value.copy())
        next_t[sample] += 1
        if row["wait_for_commit_after"]:
            window = len(commits) + 1
            if next_t != [window * WINDOW] * BATCH:
                raise ValueError("commit before all four samples finish the window")
            flat = gradient.reshape(-1).tolist()
            delta = np.asarray([round_nearest_even_ratio(-g * 640, 1 << 26) for g in flat], dtype=np.int64)
            after, carry = apply_update(codes.reshape(-1).tolist(), residual.reshape(-1).tolist(), flat)
            after, carry = np.asarray(after).reshape(initial.shape), np.asarray(carry).reshape(initial.shape)
            changed = np.flatnonzero(after.reshape(-1) != codes.reshape(-1))
            witness = None
            if len(changed):
                address = int(changed[0])
                c, f = divmod(address, FEATURES)
                witness = dict(address=address, class_id=c, feature=f, code_before=int(codes[c, f]),
                               code_after=int(after[c, f]), residual_before=int(residual[c, f]),
                               delta=int(delta[address]), residual_after=int(carry[c, f]))
            commits.append(dict(window_1based=window, after_frame=frame,
                                time_base=(window-1)*WINDOW, timestep_count=WINDOW, batch_count=BATCH,
                                trace_reset=window == 1,
                                code_changes=len(changed), residual_changes=int(np.count_nonzero(carry != residual)),
                                nonzero_delta_rmw=int(np.count_nonzero(delta)),
                                max_abs_delta=int(np.abs(delta).max()), max_abs_residual=int(np.abs(carry).max()),
                                witness=witness))
            snapshots.append(dict(codes=after.copy(), residuals=carry.copy(), gradient=gradient.copy(),
                                  delta=delta.reshape(initial.shape), rmw_mask=(delta != 0).astype(np.int64)))
            codes, residual = after, carry
            gradient.fill(0)
    if next_t != [STEPS] * BATCH or len(commits) != 3:
        raise ValueError("incomplete continuous trajectory")
    return dict(observations={k: np.stack(v) for k, v in observations.items()},
                commits=commits, snapshots=snapshots, codes=codes, residuals=residual)


def probe_locations(channels):
    return [dict(layer=i+1, points=[[0, 0, 0], [co//2, size//2, size//2], [co-1, size-1, size-1]])
            for i, (_, co) in enumerate(channels) for size in [32 >> (i//2)]]


def single_accumulator_negative_control(observations, schedule, *, reset_each_window=False):
    cumulative = np.zeros(CLASSES, dtype=np.int64)
    first = None
    changed = 0
    for row, current, expected in zip(schedule, observations["currents"], observations["centered"]):
        if row["sample_state_reset"] or (reset_each_window and row["local_timestep"] == 0):
            cumulative.fill(0)
        cumulative = (cumulative + current + (1 << 31)) % (1 << 32) - (1 << 31)
        centered = np.clip((cumulative - cumulative.max() + (1 << 31)) % (1 << 32) - (1 << 31), -32768, 32767)
        indices = np.flatnonzero(centered != expected)
        changed += len(indices)
        if len(indices) and first is None:
            c = int(indices[0])
            first = dict(frame=row["frame"], sample_id=row["sample_id"], timestep=row["timestep"],
                         class_id=c, expected=int(expected[c]), incorrect=int(centered[c]))
    return dict(changed_centered_values=changed, first_mismatch=first)


def scalar_checks(arrays, schedule):
    """Check the prepared head with Python integers, not the NumPy replay/update.

    Conv features remain the full-graph integer reference. The existing scalar
    softmax/LUT is shared; dot, trace, reduction, rounding and carry are separate.
    """
    def wrap(value, bits):
        return (value + (1 << (bits-1))) % (1 << bits) - (1 << (bits-1))

    def sat16(value):
        return max(-32768, min(32767, value))

    def check(actual, name, row=None):
        expected = arrays[name] if row is None else arrays[name][row]
        np.testing.assert_array_equal(actual, expected)

    def forward(features, codes, cumulative):
        raw = [sum(w*x for w, x in zip(row, features)) for row in codes]
        current = [sat16(value >> 15) for value in raw]
        cumulative[:] = [wrap(a+b, 32) for a, b in zip(cumulative, current)]
        centered = [sat16(wrap(value-max(cumulative), 32)) for value in cumulative]
        return raw, current, centered

    codes = arrays["initial_codes"].tolist()
    residual = [[0] * FEATURES for _ in range(CLASSES)]
    trace = [[0] * FEATURES for _ in range(BATCH)]
    cumulative = [[0] * CLASSES for _ in range(BATCH)]
    gradient = [[0] * FEATURES for _ in range(CLASSES)]
    witness_address = 18433
    wc, wf = divmod(witness_address, FEATURES)
    windows, terms = [], []
    for frame, row in enumerate(schedule):
        sample = row["sample_slot"]
        features = arrays["train_features"][frame].tolist()
        _, current, centered = forward(features, codes, cumulative[sample])
        check(current, "train_currents", frame)
        check(centered, "train_centered", frame)
        check(cumulative[sample], "train_cumulative", frame)
        trace[sample] = [wrap(((old*250) >> 8)+x, 16) for old, x in zip(trace[sample], features)]
        check(trace[sample], "train_traces", frame)
        errors = softmax_error_q016(centered, int(arrays["labels"][sample]))
        for c, error in enumerate(errors):
            gradient[c] = [wrap(g+error*(2*x), 48) for g, x in zip(gradient[c], trace[sample])]
        terms.append(dict(frame=frame, sample_slot=sample, timestep=row["timestep"],
                          trace=trace[sample][wf], error_q016=errors[wc],
                          contribution=errors[wc]*2*trace[sample][wf]))
        if row["wait_for_commit_after"]:
            window = len(windows)+1
            prefix = f"window{window:02d}_"
            check(gradient, prefix+"gradient")
            code_before, residual_before = codes[wc][wf], residual[wc][wf]
            delta = []
            for c in range(CLASSES):
                delta.append([])
                for f in range(FEATURES):
                    numerator = -gradient[c][f]*640
                    quotient, remainder = divmod(abs(numerator), 1 << 26)
                    quotient += (2*remainder > (1 << 26) or
                                 (2*remainder == (1 << 26) and quotient % 2 == 1))
                    d = -quotient if numerator < 0 else quotient
                    delta[c].append(d)
                    total = residual[c][f]+d
                    sign = -1 if total < 0 else 1
                    candidate = codes[c][f]+sign*(abs(total)//32768)
                    codes[c][f] = sat16(candidate)
                    residual[c][f] = 0 if codes[c][f] != candidate else sign*(abs(total) % 32768)
            check(delta, prefix+"delta")
            check(codes, prefix+"codes")
            check(residual, prefix+"residuals")
            windows.append(dict(window_1based=window, address=witness_address,
                                gradient=gradient[wc][wf], delta=delta[wc][wf],
                                code_before=code_before, code_after=codes[wc][wf],
                                residual_before=residual_before, residual_after=residual[wc][wf],
                                terms=terms))
            gradient = [[0] * FEATURES for _ in range(CLASSES)]
            terms = []
    check(codes, "final_codes")
    check(residual, "final_residuals")
    query = []
    for phase, weights, prefix in (("before", arrays["initial_codes"].tolist(), "query_before_"),
                                    ("after", codes, "query_")):
        sums = [0] * CLASSES
        for t, features in enumerate(arrays["query_features"].tolist()):
            raw, current, centered = forward(features, weights, sums)
            check(current, prefix+"currents", t)
            check(centered, prefix+"centered", t)
            query.append(dict(phase=phase, timestep=t, class_id=wc, raw_dot=raw[wc],
                              current=current[wc], cumulative=sums[wc], centered=centered[wc]))
    return dict(status="pass", windows=windows, query_class9=query,
                checked_train_dots=48*CLASSES, checked_query_dots=8*CLASSES,
                checked_gradient_parameters=3*PARAMS, checked_update_parameters=3*PARAMS,
                scope="Python integer head checks; shared scalar softmax/LUT and full-graph feature source, not a second independent Conv kernel")


def extract(network, raw, ids):
    features = np.empty((len(ids), raw.shape[1], FEATURES), dtype=np.int64)
    probes = np.empty((len(ids), raw.shape[1], 8, 3), dtype=np.int64)
    currents, centered = [], []
    from evaluate_scnn_full_integer import LAYER_CHANNELS
    points = probe_locations(LAYER_CHANNELS)

    def capture(t, layer, values):
        if layer < 8:
            for probe, (c, y, x) in enumerate(points[layer]["points"]):
                probes[:, t, layer, probe] = values["next"][:, c, y, x].cpu().numpy()
            if layer == 7:
                features[:, t] = values["pooled"].flatten(1).cpu().numpy()
        else:
            currents.append(values["current"].cpu().numpy())
            centered.append(values["centered"].cpu().numpy())

    logits = network.forward(raw, sample_ids=ids, verify_ids=ids, capture=capture)
    return dict(features=features, membrane_probes=probes, currents=np.stack(currents),
                centered=np.stack(centered), logits=logits.cpu().numpy(), probe_locations=points)


def compute():
    import torch
    from evaluate_scnn_full_integer import IntegerNetwork, check_prefix, MODEL_DIR
    from supplement_scnn_weight_quantization import PAYLOAD, input_batches
    import run_scnn_tp_selected_layer_shift_adaptation as api

    started = time.perf_counter()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    episode = read_episode()
    ids, query_id = episode["used_support_batches"][0], episode["query_indices"][0]
    schedule = frame_schedule(ids, query_id)
    train_schedule, query_schedule = schedule[:48], schedule[48:]
    data = PAYLOAD / "data"
    labels_all = api.read_labels(data / "test_264_labels.hex", 264)
    wanted = set(ids + [query_id])
    decoded = {}
    with (data / "test_264x20_inputs_q8_8.hex").open(encoding="ascii") as stream:
        for offset, frames in input_batches(stream, 264, 4, (20, 2, 32, 32)):
            for j, frame in enumerate(frames):
                if offset+j in wanted:
                    decoded[offset+j] = frame.copy()
            if wanted == set(decoded):
                break
    if set(decoded) != wanted:
        raise ValueError("missing fixed raw support/query")
    corruption = api.EventCorruption(temporal_jitter=1, seed=SEED)

    def shifted_q88(sample_ids):
        clean = torch.from_numpy(np.stack([decoded[i] for i in sample_ids]))
        # Transform full T20 before slicing: events at t12 can move into t11.
        shifted = api.transform_event_view(clean, sample_ids, dx=0, dy=0, corruption=corruption) * 256
        if not torch.equal(shifted, shifted.round()) or torch.equal(shifted, clean * 256):
            raise AssertionError("expected actual shifted Q8.8 inputs")
        return shifted.to(torch.int64)

    raw_support, raw_query = shifted_q88(ids), shifted_q88([query_id])[:, :QUERY_STEPS]
    network = IntegerNetwork("cpu")
    prefix = check_prefix(network)
    initial = network.codes[-1].copy()
    with np.load(REPORT / "seed38472_29_to_28_state.npz") as state:
        np.testing.assert_array_equal(initial, state["initial_codes"])
    kernel = load_kernel()
    support = extract(network, raw_support, ids)
    baseline = kernel.replay(support["features"], initial, capture=True)
    for name in ("currents", "centered"):
        np.testing.assert_array_equal(baseline["history"][name], support[name])
    np.testing.assert_array_equal(baseline["logits"], support["logits"])
    official_batch = kernel.replay(support["features"], initial, labels=labels_all[ids], capture=True)
    if official_batch["counters"] != episode["support_counters"][0]["counters"]:
        raise AssertionError("first B4T20 counters differ from the completed formal report")
    train_features = support["features"][:, :STEPS]
    path = scheduled_reference(train_features, labels_all[ids], initial, train_schedule)
    # Time-major official kernel and sample-major DUT schedule must agree.
    for name in ("currents", "centered", "traces"):
        by_sample = np.stack(official_batch["history"][name][:STEPS]).swapaxes(0, 1)
        np.testing.assert_array_equal(path["observations"][name], pack_frames(by_sample, train_schedule))
    for w, snapshot in enumerate(path["snapshots"]):
        for ours, theirs in (("codes", "codes"), ("residuals", "residuals"), ("gradient", "gradients")):
            np.testing.assert_array_equal(snapshot[ours], official_batch["history"][theirs][w])

    query_before = extract(network, raw_query, [query_id])
    q_before = kernel.replay(query_before["features"], initial, capture=True)
    network.codes[-1] = path["codes"].copy()
    network.weights[-1] = torch.tensor(path["codes"], dtype=torch.float64)
    # Fresh raw graph, reset sample state, retain the naturally learned codes.
    query_after = extract(network, raw_query, [query_id])
    np.testing.assert_array_equal(query_after["features"], query_before["features"])
    q_after = kernel.replay(query_after["features"], path["codes"], capture=True)
    for graph, head in ((query_before, q_before), (query_after, q_after)):
        for name in ("currents", "centered"):
            np.testing.assert_array_equal(head["history"][name], graph[name])
        np.testing.assert_array_equal(head["logits"], graph["logits"])
    visibility = {}
    for name in ("currents", "centered"):
        before, after = query_before[name][:, 0], query_after[name][:, 0]
        positions = np.argwhere(before != after)
        visibility[name] = dict(changed=int(len(positions)), first_witness=None)
        if len(positions):
            t, c = map(int, positions[0])
            visibility[name]["first_witness"] = dict(timestep=t, class_id=c,
                                                     before=int(before[t, c]), after=int(after[t, c]))
    arrays = {
        "initial_codes": initial, "initial_residuals": np.zeros_like(initial), "labels": labels_all[ids],
        "train_raw": pack_frames(raw_support.numpy(), train_schedule),
        "train_features": pack_frames(train_features, train_schedule),
        "train_membrane_post_probes": pack_frames(support["membrane_probes"], train_schedule),
        **{"train_" + k: v for k, v in path["observations"].items()},
        "final_codes": path["codes"], "final_residuals": path["residuals"],
        "query_raw": raw_query.numpy()[0], "query_features": query_after["features"][0],
        "query_membrane_post_probes": query_after["membrane_probes"][0],
        "query_currents": query_after["currents"][:, 0], "query_centered": query_after["centered"][:, 0],
        "query_before_currents": query_before["currents"][:, 0], "query_before_centered": query_before["centered"][:, 0],
    }
    for w, snapshot in enumerate(path["snapshots"], 1):
        arrays.update({f"window{w:02d}_" + k: v for k, v in snapshot.items()})
    scalar = scalar_checks(arrays, train_schedule)
    manifest = dict(status=STATUS, seed=SEED, support_subject=29, query_subject=28,
                    support_ids=ids, labels=labels_all[ids].tolist(), query_id=query_id,
                    query_selection="first formal held-out query index, fixed before computation; not selected for visibility",
                    support_domain="shifted", temporal_jitter=1, timesteps=STEPS,
                    windows=3, batch_count=BATCH, multiplier=640, vote_increment=0,
                    residual_threshold=32768, initial_residual=0, classifier_mask="all rows, all four samples",
                    train_graph_frames=48, query_graph_frames=4, total_graph_frames=52,
                    first_crossing_window=next((r["window_1based"] for r in path["commits"] if r["code_changes"]), None),
                    commits=path["commits"], query_visibility=visibility, frame_schedule=schedule,
                    frame_order="window, sample_slot, local_timestep; flattened features/inputs follow this order",
                    parameter_order="class*2048+feature", probe_locations=support["probe_locations"],
                    publication="not an algorithm oracle: completed_windows counts numerical updates only, never a cache pair or hardware version; the final-top TB must check its public commit/publication interface before the next window/query",
                    sources=dict(report=str(REPORT.relative_to(ROOT)).replace("\\", "/"),
                                 raw=str((data / "test_264x20_inputs_q8_8.hex").relative_to(ROOT)).replace("\\", "/"),
                                 labels=str((data / "test_264_labels.hex").relative_to(ROOT)).replace("\\", "/"),
                                 model_banks=str(MODEL_DIR.relative_to(ROOT)).replace("\\", "/")),
                    checks=dict(prefix_values=prefix["compared_values"], formal_first_batch_counters_match=True,
                                formal_first_batch_counters=official_batch["counters"],
                                independent_sample_major_replay_matches_time_major=True,
                                all_codes_residuals_gradients_checked=True,
                                fresh_raw_query_recomputed=True, cpu_integer_checks=network.checks),
                    negative_controls=dict(
                        one_global_logit_accumulator=single_accumulator_negative_control(path["observations"], train_schedule),
                        reset_logits_each_window=single_accumulator_negative_control(path["observations"], train_schedule, reset_each_window=True)),
                    scalar_checks=scalar,
                    dependencies=dict(
                        runtime=["Python >=3.10", "NumPy", "PyTorch with CPU float64 convolution"],
                        python_files=[
                            "scripts/prepare_scnn_temporal_visibility.py",
                            "scripts/supplement_scnn_same_dut_b4t4_reference.py",
                            "scripts/supplement_scnn_graph_reference.py",
                            "scripts/analyze_scnn_b4t4_rtl_update_contract.py",
                            "scripts/evaluate_scnn_full_integer.py",
                            "scripts/build_scnn_vgg9_full_frame_t2_reference.py",
                            "scripts/supplement_scnn_weight_quantization.py",
                            "scripts/run_scnn_tp_selected_layer_shift_adaptation.py",
                            str((ALGORITHM / "integer_classifier.py").relative_to(ROOT)).replace("\\", "/")],
                        report_files=[str((REPORT / name).relative_to(ROOT)).replace("\\", "/")
                                      for name in ("plan.json", "seed38472_29_to_28.json", "seed38472_29_to_28_state.npz")],
                        model_files=[str((MODEL_DIR / f"bank_{i:02d}.hex").relative_to(ROOT)).replace("\\", "/")
                                     for i in (*range(1, 9), 12)],
                        prefix_directory="results/paper_evidence/rtl/scnn_full_graph_same_array_20260927/t2_reference",
                        raw_files=[str((data / name).relative_to(ROOT)).replace("\\", "/")
                                   for name in ("test_264x20_inputs_q8_8.hex", "test_264_labels.hex")],
                        rtl_consumer="pre-generated fixture directory plus the final-top runner's existing RTL/model/ROM dependencies; Python/PyTorch and formal report are not needed to consume hex files",
                        inactive_instance_rom="rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem"),
                    resource=dict(cpu_threads=1, device="cpu",
                                  current_finaltop_estimate_hours=None,
                                  current_finaltop_estimate_status="pending measured final-top B1T2 frame timing; production per-word DDR/refill can substantially increase cost",
                                  legacy_graph_online_subtop_only=dict(
                                      observed_seconds_per_frame=FRAME_SECONDS,
                                      arithmetic_52_frame_hours=52*FRAME_SECONDS/3600,
                                      observed_build_seconds=BUILD_SECONDS,
                                      timing_basis="failed r1 two-frame direct-cache-fill graph-online subtop; NOT a current final-top estimate; excludes additional learning/checking/contention")),
                    reference_wall_seconds=time.perf_counter()-started,
                    boundary="prepare only; no RTL, final-top, dataset accuracy or measured hardware throughput claim")
    return dict(manifest=manifest, arrays=arrays)


INTEGRATION = """# Temporal Visibility Fixture Integration

Status: prepare only, not RTL PASS. No existing runner is modified or invoked.
Use manifest.json frame_schedule; do not point the cold B4T4 runner at this data.

1. Load existing Conv1..8 banks and initial_codes once; initialize learner residual
   to zero. Never load final/window expected codes, residuals, traces or features
   into the DUT. Those files are scoreboard-only expectations.
2. Execute 48 raw graph frames, eight Conv descriptors then classifier per frame.
   Order is window -> sample -> local time. Global time is window*4+local time.
   Each window contains all four distinct real samples, not 16 copies of one.
3. Maintain four independent observed-DUT membrane, TP and cumulative-logit
   contexts. Reset sample state only at global t0, not local t0 of later windows.
   Restore/save the correct sample context at switches, using completed public
   context barriers. Never populate backing state from expected probe files.
   TP traces continue across all three windows. A version bump is not a reset.
   The current logit adapter has sample-indexed cumulative storage. Select the
   correct sample slot and keep its state across commits. The manifest's global
   accumulator and per-window reset negative controls describe mutations, NOT
   current RTL defects. PE membrane transfer alone does not check logit state.
4. Configure each window as B4/T4, time bases 0/4/8, trace reset only at first
   window. capture_row=sample_slot*4+local_time; absolute timestep is also in the
   manifest. Map these explicitly to the integration's local/global metadata.
   Check train_traces in the same frame order; do not reuse a cold trace table.
   Observation replay already indexes TP state by sample*MAX_FEATURES+feature.
   Use existing public i_timestep=0/4/8, not a new time-base port. The current
   selected-layer owner raises cleanup on the SAME commit edge that enters
   bridge REARM. The drained learner is release-ready; release priority blocks
   replay configuration, so no base0 reset is accepted on cleanup. Replay abort
   does not reset trace storage. Base4/8 configuration preserves that storage.
   Keep owner/learning-enable until cleanup completes. Check the actual trace
   reset pulse and T4/T8 trace values; the read-only audit is not RTL PASS.
   The selected classifier-only path disables the external computed-trace tap;
   internal computed trace is still used. Use a read-only checker if the final
   integration does not export that tap, never inject the expected trace.
5. Only after all 16 classifier observations may that window reduce/commit.
   Check windowXX_gradient (signed48), delta (signed18), rmw_mask, all 22528
   codes/residuals. Expected RMW mask is delta != 0.
   Wait for actual commit sequence, publisher drain and ready/idle handshake.
   Do not expect the new cache pair/version at the graph-done edge itself.
   completed_windows is ONLY the number of numerical updates already applied,
   not a hardware version or bank selector. There are no pair/version oracle
   fields. Current final-top parameter_backend republishes the initialized
   classifier pair (pair1 in the final-top TB); it does NOT alternate pairs.
   In the current backend, accepted update publication advances the version;
   warm re-prime retains it. Check o_active_classifier_pair, o_parameter_version,
   cache valid and the real completion handshake in the final-top TB, using its
   actual initial configuration/version width. Never use old subtop XOR rules.
   No reload/reset of learner between windows, including zero-code windows.
6. After third commit, disable/detach learning through normal handshakes and
   run fresh raw query sample22 t0..3 in the SAME DUT. Reset its sample state,
   but retain learned codes/residuals and the last published version. Check actual
   Conv8 features/current/centered logits, and reject any query learning write.
   query_before files are read-only pretrained comparisons, not a second DUT.

All .hex files use one two's-complement word per line. manifest.json files lists
word widths, shapes and roles. C-order flattening applies within each frame.
Raw shape is [frame,2,32,32], pooled feature/trace [frame,2048], logits [frame,11].
Membrane probes are post-LIF [frame,8,3] at the fixed coordinates in the manifest;
they are a bounded check, not a substitute for complete state-transfer checking.
Only raw input, labels and initial parameter files are stimulus. Expected feature
files must never bypass Conv1..8. Final-top port wiring/descriptor control and
full state service remain the responsibility of the integration TB.

Preparation also recomputes the first full B4T20 batch (CPU only) to match the
completed formal report's aggregate counters. Only its first B4T12 prefix is
exported; support membrane/trace/cumulative state never resets at T4 boundaries.
Jitter is applied to full raw T20 before slicing the T12/T4 prefixes, with global
sample IDs and fixed seed38472. The held-out query is predetermined, not searched.
No full per-layer tensors, new weight banks or large replay caches are exported.

Read-only integration references at preparation time (not proof of later RTL):
- rtl/learning/unified_scnn_h40_classifier_logit_wave_adapter_v1.sv
- rtl/learning/unified_scnn_selected_layers_parameter_backend_v1.sv:393,400,509
- rtl/learning/unified_scnn_classifier_observation_replay.sv:168,250,291
- rtl/learning/unified_scnn_tp_classifier_final_core_learning_bridge.sv:377,445,453
- rtl/learning/unified_scnn_tp_selected_layers_t1_path.sv:1058,1100,1118
- rtl/learning/unified_scnn_tp_selected_layers_t1_path.sv:638,670

## Interface and Numeric Checks

Every frame requires eight real Conv descriptors. Training descriptors total
384; the four fresh query frames add 32, for 416 descriptors and 52 graph frames.
Only the last descriptor of a frame may advance the coordinator. Wait for tail
completion and state spill before changing its sample/timestep metadata.

train_* arrays have 48 rows. query_* arrays have FOUR rows indexed 0..3, even
though their schedule frame IDs are 48..51. Query dataset ID22 uses sample_slot0;
it is not a hardware context index22. Signed hex decoding is two's complement.
For nonlearning query frames, supply the correct public frame-index metadata;
do not assume the learning-only absolute-timestep selection is active.

Frame address formulas (C order): raw=frame*2048+channel*1024+y*32+x;
feature/trace=frame*2048+feature; class outputs=frame*11+class;
membrane probes=frame*24+(layer-1)*3+probe; parameter=class*2048+feature.
capture_row=sample_slot*4+local_timestep, while a flattened learner row is
local_timestep*4+sample_slot. Never interchange these two layouts.

Numerical equations: current=sat16(sum(code*x)>>15); per-sample cumulative is
signed32 wrap; centered=sat16(wrap32(cumulative-max(cumulative))). Trace is
wrap16((trace*250>>8)+x). Gradient is wrap48(sum(error_q016*2*trace)). Every T4
window applies delta=round_even(-gradient*640/2^26). Add delta to residual;
signed truncation by 32768 yields the code step, with signed16 code saturation
and residual zero on code saturation. Otherwise retain the signed remainder.
No artificial votes, label/input substitution or residual preload is used.

scalar_checks in manifest.json independently recomputes all 528 training and
88 before/after-query class dot products, all 67584 gradient entries and all
67584 parameter updates with Python integers. It exports the 16 per-window
gradient terms for address18433 and query class9 raw dots/current/centered values.
It shares the existing scalar softmax/LUT and integer Conv feature source; it
is not a second independent full Conv implementation. The sample-major replay
also matches the formal time-major kernel, including every code/residual and
the completed first B4T20 report counters. Its first three windows alone are
exported. Conv FP64 operations are exact integer carriers, not FP32 inference.

## Commands and Dependencies

Run from the project root, with an unused output directory:

```text
python -B scripts/prepare_scnn_temporal_visibility.py --output results/paper_evidence/supplement_20261004/rtl/scnn_same_dut/natural_visibility_prepare/seed38472_b4t12_q22_t4_finaltop_v2
python -B -m unittest discover -s tests -p test_scnn_temporal_visibility.py -v
```

Omit --output for a read-only numeric rerun. Existing directories are not
overwritten. Python preparation requires NumPy/PyTorch CPU and the repository
modules/data listed in manifest.dependencies. CPU compute uses one thread;
it never calls CUDA, RTL tools or SSH. No original FP32 checkpoint is loaded.

For an RTL run, transfer this complete fixture directory, existing Conv1..8
model banks and the final-top runner's own source/dependency package. No Python,
PyTorch, official report or full264 input file is required to consume the hex.
Include rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem in that package
even for classifier-only learning: an instantiated inactive Conv8 ROM can still
execute its initial readmem. This is not a standalone RTL runner or source list.

The historical 6.51h arithmetic for 52 frames and 527.84s build came from a
different graph-online direct-cache-fill subtop. They are NOT final-top runtime
or build estimates. Current final-top production per-word DDR/refill may cost
substantially more. Wait for actual B1T2 final-top frame timing before extrapolating;
warm-window, publication, context transfer and compile costs remain unmeasured.
"""


def write_fixture(result, output):
    output = Path(output).resolve()
    if output == BASE.resolve() or not output.is_relative_to(BASE.resolve()):
        raise ValueError("use a fresh child directory under scnn_same_dut/natural_visibility_prepare")
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(result["manifest"])
    inventory = []
    for name, array in result["arrays"].items():
        array = np.asarray(array)
        bits = 48 if name.endswith("_gradient") else 18 if name.endswith("_delta") else 32 if name.endswith("_cumulative") else 16
        signed = name != "labels" and not name.endswith("_rmw_mask")
        lower, upper = (-(1 << (bits-1)), (1 << (bits-1))-1) if signed else (0, (1 << bits)-1)
        if array.dtype.kind not in "iu" or np.any(array < lower) or np.any(array > upper):
            raise ValueError(f"invalid integer fixture width: {name}")
        path = output / f"{name}.hex"
        write_signed_hex(path, array.reshape(-1).tolist(), bits=bits)
        role = "stimulus" if name in ("train_raw", "query_raw", "labels", "initial_codes", "initial_residuals") else "scoreboard_only"
        inventory.append(dict(path=path.name, words=int(array.size), bits=bits, signed=signed,
                              shape=list(array.shape), role=role))
    manifest["files"] = inventory
    manifest["fixture_hex_bytes"] = sum((output / entry["path"]).stat().st_size for entry in inventory)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    (output / "INTEGRATION.md").write_text(INTEGRATION, encoding="ascii")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new preparation directory; omit for read-only assessment")
    args = parser.parse_args()
    result = compute()
    manifest = write_fixture(result, args.output) if args.output is not None else result["manifest"]
    print(json.dumps({k: manifest[k] for k in ("status", "support_ids", "query_id", "first_crossing_window",
                                             "commits", "query_visibility", "total_graph_frames",
                                             "resource", "reference_wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
