#!/usr/bin/env python3
"""Bounded raw700 INT8 backbone -> formal450/256/20 M2 reference/export.

No checkpoint load, simulation, compilation, remote access, or model updates.
The source-Q14 diagnostic is NOT the formal FP32 backbone. Only connect reads
observed RTL values; export produces Python expectations, never RTL evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/paper_evidence/supplement_20261004/algorithm/srnn_backbone_numeric"
MODEL = ROOT / "models/srnn_shd_700_480_20_int8_layer_scaled_v2_padded"
SOURCE = ROOT / "imports/srnn/data/shd/weights"
DATA = ROOT / "data/shd/official_test"
LEAK, THRESHOLD, LOGICAL, PHYSICAL = 27853, 32, 450, 480
FIELDS = ("ff_current", "recurrent_current", "current", "pre", "membrane", "spike")
sys.dont_write_bytecode = True


def project_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("all input paths must remain inside this workspace")
    return path


def new_output(path):
    path = project_path(path)
    if path == EVIDENCE or not path.is_relative_to(EVIDENCE):
        raise ValueError(f"output must be a child of {EVIDENCE}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def read_signed(path, bits, shape):
    with Path(path).open(encoding="ascii") as stream:
        raw = np.fromiter((int(line.strip(), 16) for line in stream if line.strip()), dtype=np.int64)
    if raw.size != int(np.prod(shape)) or np.any((raw < 0) | (raw >= 1 << bits)):
        raise ValueError(f"invalid {bits}-bit memory shape/codes: {path}")
    return np.where(raw >= 1 << (bits - 1), raw - (1 << bits), raw).reshape(shape)


def write_hex(path, values, bits):
    with Path(path).open("x", encoding="ascii") as stream:
        for value in np.asarray(values).flat:
            stream.write(f"{int(value) & ((1 << bits) - 1):0{(bits + 3) // 4}x}\n")


def refill_codes(source_major):
    """Scheduler addresses parameter_base + destination*source_count + source."""
    return np.asarray(source_major).T.copy()


def json_write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def read_prefix(path, steps):
    """Existing SHD sparse format, streamed with add-at semantics for repeats."""
    counts = np.zeros((steps, 700), dtype=np.int64)
    timestep = 0
    with Path(path).open(encoding="ascii") as stream:
        for line in stream:
            if not line.strip():
                continue
            word = int(line, 16)
            if not 0 <= word < 1 << 20:
                raise ValueError("SHD word exceeds 20 bits")
            tag = word >> 18
            if tag == 0:
                source, value = (word >> 8) & 1023, word & 255
                if source >= 700 or not value:
                    raise ValueError("invalid SHD source/count")
                counts[timestep, source] += value
            elif tag in (1, 2) and word & ((1 << 18) - 1) == 0:
                timestep += 1
                if timestep == steps:
                    return counts
                if tag == 2:
                    raise ValueError("sample ends before requested prefix; no invented padding")
            else:
                raise ValueError("invalid SHD boundary marker")
    raise ValueError("truncated SHD prefix")


def sat32(value):
    return np.clip(value, -(1 << 31), (1 << 31) - 1).astype(np.int64)


def lif_step(current, membrane, threshold=THRESHOLD):
    # Mode1: truncate negative products toward -infinity, saturate BEFORE compare.
    pre = sat32((np.asarray(membrane, dtype=np.int64) * LEAK >> 15) + sat32(current))
    spike = (pre >= threshold).astype(np.int64)
    return pre, pre - spike * threshold, spike


def backbone_reference(counts, w0, recurrent, threshold=THRESHOLD):
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 3 or counts.shape[2] != w0.shape[0] or np.any(counts < 0):
        raise ValueError("expected nonnegative [time,batch,source] counts")
    hidden = w0.shape[1]
    if recurrent.shape != (hidden, hidden):
        raise ValueError("recurrent matrix must be source-major square")
    membrane = np.zeros((counts.shape[1], hidden), dtype=np.int64)
    previous = np.zeros_like(membrane)
    history = {key: [] for key in FIELDS}
    for frame in counts:
        ff = frame @ w0
        rc = previous @ recurrent
        current = sat32(ff + rc)
        pre, membrane, previous = lif_step(current, membrane, threshold)
        for key, value in zip(FIELDS, (ff, rc, current, pre, membrane, previous)):
            history[key].append(value.copy())
    return {key: np.stack(values) for key, values in history.items()}


def p4_groups(counts, row_base=0):
    """Repeat each 4-source group inside ONE LIF transaction; never binarize."""
    counts = np.asarray(counts)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer) or np.any((counts < 0) | (counts > 255)):
        raise ValueError("P4 expansion requires uint8 counts")
    groups = []
    for start in range(0, len(counts), 4):
        block = counts[start:start + 4]
        addresses = [row_base + ((start + lane) // 2) for lane in range(4)]
        for repeat in range(int(block.max())):
            mask = sum((int(value) > repeat) << lane for lane, value in enumerate(block))
            groups.append((mask, *addresses))
    # The client requires a nonzero FF count, even for an empty input frame.
    return groups or [(0, row_base, row_base, row_base, row_base)]


def group_current(groups, weights, row_base=0):
    current = np.zeros(weights.shape[1], dtype=np.int64)
    for mask, *addresses in groups:
        for lane, address in enumerate(addresses):
            if mask & (1 << lane):
                source = 2 * (address - row_base) + (lane & 1)
                current += weights[source]
    return current


def difference(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch {left.shape} != {right.shape}")
    different = left != right
    where = np.argwhere(different)
    first = tuple(where[0]) if len(where) else None
    return dict(elements=int(left.size), count=int(different.sum()),
                first_index=list(map(int, first)) if first is not None else None,
                left=int(left[first]) if first is not None else None,
                right=int(right[first]) if first is not None else None,
                max_absolute_error=int(np.abs(left.astype(np.int64) - right.astype(np.int64)).max(initial=0)),
                per_timestep=[int(np.count_nonzero(row)) for row in different])


def peak_memory():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                 "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                 "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        value = Counters()
        value.cb = ctypes.sizeof(value)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return dict(bytes=int(value.PeakWorkingSetSize), metric="Windows process PeakWorkingSetSize")
    import resource
    size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(bytes=int(size * (1 if sys.platform == "darwin" else 1024)), metric="process ru_maxrss")


def m2_fixture(output, spikes, labels, source):
    # Reuse the already repaired reference and unchanged Wout-only update rule.
    # Its helper imports torch, but no checkpoint/model/tensor is loaded here.
    import build_srnn_formal_m2_fixture as m2

    if spikes.ndim != 3 or spikes.shape[1] != 2 or spikes.shape[2] not in (LOGICAL, PHYSICAL) or not np.isin(spikes, [0, 1]).all() or np.any(spikes[:, :, LOGICAL:]):
        raise ValueError("expected B2 binary H450, or H480 with zero padding")
    output.mkdir()
    wh1 = read_signed(m2.WH1, 16, (256, 450))
    wout = read_signed(m2.WOUT, 16, (20, 256))
    inputs = spikes[:, :, :LOGICAL] * m2.SPIKE
    write_hex(output / "input_q14.mem", inputs, 32)
    write_hex(output / "labels.mem", labels, 8)
    counters, arrays = {}, {}
    differences = {}
    for learning in (False, True):
        mode = "on" if learning else "off"
        production, traffic = m2.reference(inputs, labels, wh1, wout, learning=learning)
        formal, _ = m2.reference(inputs, labels, wh1, wout, learning=learning, arithmetic="formal_q14")
        counters["production_" + mode] = traffic
        arrays[mode] = production
        np.savez(output / f"production_{mode}.npz", **production)
        prefix = "expected_on" if learning else "expected"
        for key in ("pre", "membrane", "spike", "trace", "feature", "dot", "logit",
                    "wout_before", "wout_after", "residual", "gradient", "error"):
            bits = 64 if key in ("logit", "gradient", "error") else 16 if key in ("wout_before", "wout_after", "residual") else 32
            write_hex(output / f"{prefix}_{key}.mem", production[key], bits)
        differences[mode] = {key: difference(production[key], formal[key]) for key in
                             ("pre", "membrane", "spike", "trace", "feature", "dot", "logit")}
        if not learning:
            write_hex(output / "formal_pre.mem", formal["pre"], 32)
    info = dict(status="python_reference_not_rtl", batch=2, steps=len(spikes), backbone=450,
                hidden=256, classes=20, production_arithmetic="production", source=source,
                current_vs_formal_q14=differences["off"], on_current_vs_formal_q14=differences["on"],
                reference_counters=counters,
                input_nonzero_per_step=np.count_nonzero(inputs, axis=(1, 2)).tolist(),
                first_code_change_timestep={mode: next((row["timestep"] for row in rows if row["code_writes"]), None)
                                           for mode, rows in counters.items()},
                boundary="Real-size formal M2 integer equations fed by the explicitly named backbone lane; not FP32 equivalence, raw700 RTL or final-top execution")
    json_write(output / "fixture.json", info)
    return arrays, info


def export(output, steps):
    if not 1 <= steps <= 8:
        raise ValueError("this bounded probe accepts only contiguous prefixes T1..T8")
    started = time.perf_counter()
    output = new_output(output)
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"][:2]
    if [row["dataset_index"] for row in samples] != [0, 1] or any(row["clipped_feature_counts"] for row in samples):
        raise ValueError("expected existing, unclipped official sample0/1 streams")
    counts = np.stack([read_prefix(DATA / f"sample_{i:04d}.hex", steps) for i in (0, 1)], axis=1)
    labels = np.array([row["label"] for row in samples])
    w0 = read_signed(MODEL / "w0_int8.mem", 8, (700, 480))
    recurrent = read_signed(MODEL / "r_int8.mem", 8, (480, 480))
    q_w0 = read_signed(SOURCE / "w0_q14.mem", 16, (700, 450))
    q_r = read_signed(SOURCE / "r_q14.mem", 16, (450, 450))
    loaded = time.perf_counter()
    hardware = backbone_reference(counts, w0, recurrent)
    source_q14 = backbone_reference(counts, q_w0, q_r, 8192)
    binary = backbone_reference((counts > 0).astype(np.int64), w0, recurrent)
    if np.any(hardware["spike"][:, :, 450:]) or np.any(w0[:, 450:]) or np.any(recurrent[450:]) or np.any(recurrent[:, 450:]):
        raise ValueError("H480 padding is not zero")
    np.savez(output / "backbone_reference.npz", counts=counts, labels=labels, **hardware)
    np.savez(output / "source_q14_reference.npz", **source_q14)
    np.savez(output / "binary_diagnostic.npz", **binary)
    for key, array in hardware.items():
        write_hex(output / f"backbone_{key}.mem", array, 32)
    write_hex(output / "raw700_counts.mem", counts, 8)
    # The stored model is source-major, but production scalar refill is not.
    write_hex(output / "w0_refill_dest_major.mem", refill_codes(w0[:, :450]), 16)
    write_hex(output / "r_refill_dest_major.mem", refill_codes(recurrent[:450, :450]), 16)
    offsets = [0]
    words = []
    group_counts = []
    with (output / "feedforward_p4.csv").open("x", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestep", "sample", "source_valid", "addr0", "addr1", "addr2", "addr3"])
        for t in range(steps):
            for b in range(2):
                groups = p4_groups(counts[t, b])
                np.testing.assert_array_equal(group_current(groups, w0), hardware["ff_current"][t, b])
                group_counts.append(len(groups))
                for mask, *addresses in groups:
                    writer.writerow([t, b, mask, *addresses])
                    words.append(mask | sum(addr << (8 + lane * 11) for lane, addr in enumerate(addresses)))
                offsets.append(len(words))
    write_hex(output / "feedforward_p4.mem", words, 64)
    write_hex(output / "feedforward_offsets.mem", offsets, 32)
    write_hex(output / "feedforward_group_counts.mem", group_counts, 16)
    computed = time.perf_counter()
    source = dict(input_boundary="Python hardware-format INT8 count-preserving backbone, NOT observed RTL",
                  dataset="SHD official test", sample_indices=[0, 1], labels=labels.tolist(),
                  timestep_indices=list(range(steps)), shift_channels=0,
                  preprocessing="existing export_shd_h5_events.py sparse streams; 10000us counts, no binarization",
                  recurrent_reset="independent zero state for each sample, previous-step spikes only")
    m2_arrays, m2_info = m2_fixture(output / "m2_count_preserving", hardware["spike"], labels, source)
    import build_srnn_formal_m2_fixture as m2
    wh1 = read_signed(m2.WH1, 16, (256, 450))
    wout = read_signed(m2.WOUT, 16, (20, 256))
    m2_differences = {}
    for name, lane in (("source_q14", source_q14), ("native_binary", binary)):
        # Off-only causal diagnostics: do not train separate exploratory arms.
        other, _ = m2.reference(lane["spike"][:, :, :450] * 16384, labels, wh1, wout)
        m2_differences[name] = {key: difference(m2_arrays["off"][key], other[key]) for key in ("current", "spike", "logit")}
    differences = {
        "int8_counts_vs_source_q14_counts_spike": difference(hardware["spike"][:, :, :450], source_q14["spike"]),
        "int8_counts_scaled256_vs_source_q14_pre": difference(hardware["pre"][:, :, :450] * 256, source_q14["pre"]),
        "int8_counts_vs_native_binary_spike": difference(hardware["spike"], binary["spike"]),
        "int8_counts_vs_native_binary_current": difference(hardware["current"], binary["current"]),
    }
    dependencies = ["scripts/supplement_srnn_backbone_numeric.py", "scripts/build_srnn_formal_m2_fixture.py",
                    "scripts/screen_srnn_m2_direct_optimizer.py", "rtl/learning/unified_tp_row_softmax_q12_q24.sv",
                    "tb/data/srnn_m2_v28_wh1_initial_q14.mem", "tb/data/srnn_m2_v28_wout_initial_q14.mem",
                    "data/shd/official_test/manifest.json", "data/shd/official_test/sample_0000.hex", "data/shd/official_test/sample_0001.hex",
                    "models/srnn_shd_700_480_20_int8_layer_scaled_v2_padded/w0_int8.mem",
                    "models/srnn_shd_700_480_20_int8_layer_scaled_v2_padded/r_int8.mem",
                    "imports/srnn/data/shd/weights/w0_q14.mem", "imports/srnn/data/shd/weights/r_q14.mem"]
    result = dict(status="bounded_python_export_complete_not_rtl", source=source, steps=steps, batch=2,
                  shape=[steps, 2, 480], dependencies=dependencies,
                  counts=dict(events=int(counts.sum()), occupied=int(np.count_nonzero(counts)),
                              greater_than_one=int(np.count_nonzero(counts > 1)), maximum=int(counts.max()),
                              lost_if_binarized=int(counts.sum() - np.count_nonzero(counts))),
                  backbone_differences=differences, m2_off_differences=m2_differences,
                  backbone_spikes_per_timestep=np.count_nonzero(hardware["spike"], axis=(1, 2)).tolist(),
                  p4=dict(feedforward_groups=group_counts, feedforward_row_base=0, recurrent_row_base=350,
                          recurrent_source_count=450, recurrent_groups=113, last_recurrent_row=574,
                          neuron_count=450, physical_capacity=480,
                          refill_layout="signed16 sign-extended INT8; destination-major: W0[d*700+s], R[d*450+s]",
                          packed_word="source_valid[3:0], addr0[18:8], addr1[29:19], addr2[40:30], addr3[51:41]",
                          recurrence="TB MUST use previous DUT spikes, not expected recurrent masks. Source449 uses lane1 of last group; lanes2/3 masked.",
                          timestep="One descriptor: expanded FF groups then 113 recurrent groups, one LIF finalization. Empty FF emits one mask0 group.",
                          cache="M2 refill replaces rows; reload W0/R before each backbone phase when sharing the same array."),
                  numeric=dict(weight="signed INT8, 1/64 real scale (Q14 shift8)", current="sat32(sum(count*W0)+sum(previous_spike*R))",
                               pre="sat32((membrane*27853 arithmetic>>15)+current)",
                               spike="pre>=32", reset="soft subtract32 once per timestep", m2_input="first450 spike bits *16384"),
                  m2_reference_counters=m2_info["reference_counters"],
                  measured_seconds=dict(load=loaded-started, backbone_export=computed-loaded,
                                        m2_reference_export=time.perf_counter()-computed, total=time.perf_counter()-started),
                  peak_process_memory=peak_memory(), python=sys.version, numpy=np.__version__,
                  limitations=["No RTL executed; count-expanded ingress is an executable stimulus, not validated hardware.",
                               "Native fixed streamer only accepts bit vectors. Counts cannot be substituted with nonzero bits without a model change.",
                               "Q14 diagnostic reuses historical Q15/truncate recurrence, not FP32 formal-model execution.",
                               "No FP32 backbone fixture for these same raw sample0/1 prefixes is present in this export.",
                               "M2 on/off is a numeric/control probe, not support adaptation or accuracy on official test samples.",
                               "No old450->20 readout is used; W0/R/Wh1 stay unchanged."])
    json_write(output / "summary.json", result)
    return result


def load_observed(path, steps):
    """CSV is directly writable by an RTL TB; NPZ is a convenient alternative."""
    keys = ("current", "pre", "membrane", "spike")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if not all(key in archive for key in ("pre", "spike")):
                raise ValueError("observed NPZ requires pre and spike")
            return {key: archive[key] for key in keys if key in archive}
    if path.suffix.lower() != ".csv":
        raise ValueError("observed dump must be CSV or NPZ")
    shape = (steps, 2, LOGICAL)
    seen = np.zeros(shape, dtype=bool)
    with path.open(encoding="ascii", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = [["timestep", "sample", "neuron", *keys],
                   ["timestep", "sample", "neuron", "pre", "spike"]]
        if reader.fieldnames not in headers:
            raise ValueError("observed CSV requires coordinates and pre/spike, optionally current/membrane")
        keys = tuple(reader.fieldnames[3:])
        values = {key: np.zeros(shape, dtype=np.int64) for key in keys}
        for row in reader:
            index = tuple(int(row[key]) for key in ("timestep", "sample", "neuron"))
            if any(not 0 <= value < limit for value, limit in zip(index, shape)) or seen[index]:
                raise ValueError(f"invalid or duplicate observed index {index}")
            seen[index] = True
            for key in keys:
                values[key][index] = int(row[key])
    if not seen.all():
        raise ValueError(f"incomplete observed CSV: {int(seen.sum())}/{int(seen.size)} records")
    return values


def connect(fixture, observed, output):
    """Compare active H450 before M2 replay; only check padded lanes if supplied."""
    started = time.perf_counter()
    fixture, observed = project_path(fixture), project_path(observed)
    with np.load(fixture / "backbone_reference.npz", allow_pickle=False) as ref:
        labels = ref["labels"]
        expected_shape = ref["spike"].shape
        dut = load_observed(observed, expected_shape[0])
        shape = dut["spike"].shape
        if shape not in (expected_shape, (*expected_shape[:2], LOGICAL)):
            raise ValueError("observed must cover every timestep/sample and H450 or H480")
        for key in dut:
            if dut[key].shape != shape or not np.issubdtype(dut[key].dtype, np.integer):
                raise ValueError(f"observed {key} must be integer {shape}")
            if np.any((dut[key] < -(1 << 31)) | (dut[key] >= 1 << 31)):
                raise ValueError(f"observed {key} exceeds signed32")
        spike = dut["spike"].astype(np.int64)
        if not np.isin(spike, [0, 1]).all() or np.any(spike[:, :, LOGICAL:]):
            raise ValueError("observed spikes are nonbinary or padded lanes fired")
        differences = {key: difference(dut[key], ref[key][:, :, :shape[2]]) for key in dut}
    output = new_output(output)
    source = dict(input_boundary="caller-supplied observed RTL backbone dump; producer authenticity not inferred by this script",
                  observed_file=str(observed.relative_to(ROOT)), expected_fixture=str(fixture.relative_to(ROOT)),
                  observed_neurons=shape[2], inactive_padding_checked=shape[2] == PHYSICAL,
                  observed_fields=list(dut),
                  unobserved_fields=[key for key in ("current", "pre", "membrane", "spike") if key not in dut],
                  differences=differences)
    _, info = m2_fixture(output / "m2_observed", spike, labels, source)
    result = dict(status="observed_trace_matches_python_reference" if not any(row["count"] for row in differences.values())
                  else "observed_backbone_mismatch_not_equivalent", differences=differences, source=source,
                  m2_fixture=str((output / "m2_observed").relative_to(ROOT)),
                  m2_reference_counters=info["reference_counters"], measured_seconds=time.perf_counter()-started,
                  peak_process_memory=peak_memory(),
                  boundary="Observed backbone comparison and Python M2 replay only; M2 RTL/final-top still untested here")
    json_write(output / "summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("export")
    build.add_argument("--steps", type=int, default=8)
    build.add_argument("--output-dir", type=Path, required=True)
    check = sub.add_parser("connect")
    check.add_argument("--fixture", type=Path, required=True)
    check.add_argument("--observed", type=Path, required=True)
    check.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = export(args.output_dir, args.steps) if args.command == "export" else connect(args.fixture, args.observed, args.output_dir)
    print(json.dumps(result, indent=2))
    return 1 if result["status"] == "observed_backbone_mismatch_not_equivalent" else 0


if __name__ == "__main__":
    raise SystemExit(main())
