#!/usr/bin/env python3
"""Export one complete official SHD sample for a same-DUT performance probe.

Only existing NumPy integer equations are evaluated. No checkpoint, GPU, data
download, RTL simulation or model update outside the private reference occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/paper_evidence/supplement_20261004/performance/current_source_measurements/srnn"
DATA = ROOT / "data/shd/official_test"
MODEL = ROOT / "models/srnn_shd_700_480_20_int8_layer_scaled_v2_padded"
PAD_REPORT = ROOT / "results/paper_evidence/supplement_20261004/algorithm/srnn_weight_only_a1/fp32.json"


def sequence_definition(manifest, words, policy="native", padded_steps=None):
    """Validate the complete serialized sequence, including its final marker."""
    if policy not in ("native", "official-b64"):
        raise ValueError("unknown timestep policy")
    row = manifest["samples"][0]
    if (manifest["dataset"], manifest["time_window_us"], manifest["include_incomplete"],
            row["dataset_index"], row["clipped_feature_counts"]) != (
            "SHD official test", 10000, False, 0, 0):
        raise ValueError("expected unclipped official sample0, complete 10000us windows")
    markers = []
    for index, word in enumerate(words):
        if not 0 <= word < 1 << 20:
            raise ValueError("SHD word exceeds 20 bits")
        tag = word >> 18
        if tag == 0:
            if (word >> 8) & 1023 >= 700 or not word & 255:
                raise ValueError("invalid source/count")
        elif tag in (1, 2) and not word & ((1 << 18) - 1):
            markers.append(tag)
            if tag == 2 and index != len(words) - 1:
                raise ValueError("records follow final sample marker")
        else:
            raise ValueError("invalid boundary marker")
    if not markers or markers[-1] != 2 or markers.count(2) != 1:
        raise ValueError("missing or duplicate final sample marker")
    if len(markers) != row["time_steps"] or len(words) != row["records"]:
        raise ValueError("manifest and complete sparse stream disagree")
    native = len(markers)
    steps = native if policy == "native" else padded_steps
    if not isinstance(steps, int) or not native <= steps <= 128:
        raise ValueError("invalid formal padding or production timestep capacity")
    return dict(batch=1, samples=1, sample_index=0, label=row["label"], steps=steps,
                native_steps=native, padded_steps=steps-native, timestep_policy=policy,
                complete_sample=True, time_window_us=10000, include_incomplete=False,
                official_b64_first_batch_steps=padded_steps,
                timestep_boundary="Native sample ends at its final sparse marker; optional official-b64 policy adds the recorded first-batch zero tail, still executed as B1.")


def build_reference(output, policy="native"):
    import numpy as np
    import supplement_srnn_backbone_numeric as bone
    import build_srnn_formal_m2_fixture as m2

    output = output.resolve()
    if output == OUTPUT or not output.is_relative_to(OUTPUT.resolve()):
        raise ValueError("reference output must be a new child of the SRNN measurement directory")
    started = time.monotonic()
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    words = [int(line, 16) for line in (DATA / "sample_0000.hex").read_text(encoding="ascii").split()]
    formal_padding = json.loads(PAD_REPORT.read_text(encoding="utf-8"))["padded_timesteps_per_batch"][0]
    info = sequence_definition(manifest, words, policy, formal_padding)
    counts = np.zeros((info["steps"], 1, 700), dtype=np.int64)
    counts[:info["native_steps"], 0] = bone.read_prefix(DATA / "sample_0000.hex", info["native_steps"])
    w0 = bone.read_signed(MODEL / "w0_int8.mem", 8, (700, 480))
    recurrent = bone.read_signed(MODEL / "r_int8.mem", 8, (480, 480))
    if np.any(w0[:, 450:]) or np.any(recurrent[450:]) or np.any(recurrent[:, 450:]):
        raise ValueError("model padding is not zero")
    backbone = bone.backbone_reference(counts, w0, recurrent)
    labels = np.array([info["label"]], dtype=np.int64)
    wh1 = bone.read_signed(m2.WH1, 16, (256, 450))
    wout = bone.read_signed(m2.WOUT, 16, (20, 256))
    output.mkdir(parents=True, exist_ok=False)
    bone.write_hex(output / "raw700_counts.mem", counts, 8)
    bone.write_hex(output / "w0_refill_dest_major.mem", bone.refill_codes(w0[:, :450]), 16)
    bone.write_hex(output / "r_refill_dest_major.mem", bone.refill_codes(recurrent[:450, :450]), 16)
    for key in ("pre", "spike"):
        bone.write_hex(output / f"backbone_{key}.mem", backbone[key], 32)
    offsets, group_counts, packed = [0], [], []
    for frame in counts[:, 0]:
        groups = bone.p4_groups(frame)
        group_counts.append(len(groups))
        for mask, *addresses in groups:
            packed.append(mask | sum(addr << (8+11*lane) for lane, addr in enumerate(addresses)))
        offsets.append(len(packed))
    bone.write_hex(output / "feedforward_p4.mem", packed, 64)
    bone.write_hex(output / "feedforward_offsets.mem", offsets, 32)
    bone.write_hex(output / "feedforward_group_counts.mem", group_counts, 16)
    bone.write_hex(output / "labels.mem", labels, 8)
    traffic = {}
    for learning in (False, True):
        prefix = "on" if learning else "off"
        arrays, traffic[prefix] = m2.reference(backbone["spike"][:, :, :450] * 16384,
                                               labels, wh1, wout, learning=learning)
        for key in ("pre", "membrane", "spike", "trace", "feature", "logit",
                    "wout_before", "wout_after", "residual", "gradient"):
            bits = (64 if key in ("logit", "gradient") else
                    16 if key in ("wout_before", "wout_after", "residual") else 32)
            bone.write_hex(output / f"{prefix}_{key}.mem", arrays[key], bits)
    info.update(status="fixed_input_reference_only_not_rtl", feedforward_words=len(packed),
                feedforward_groups_per_timestep=group_counts, reference_counters=traffic,
                source_files=[p.relative_to(ROOT).as_posix() for p in (
                    DATA / "manifest.json", DATA / "sample_0000.hex", PAD_REPORT,
                    MODEL / "w0_int8.mem", MODEL / "r_int8.mem", m2.WH1, m2.WOUT)],
                elapsed_seconds=time.monotonic()-started,
                boundary="One unshifted official sample; Python expected values only, not dataset accuracy or measured performance. The DUT must create every backbone spike. Runtime learning toggle retains production fixed /64 reducer scaling even at B1.")
    (output / "reference.json").write_text(json.dumps(info, indent=2)+"\n", encoding="utf-8")
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timestep-policy", choices=("native", "official-b64"), default="native")
    args = parser.parse_args(argv)
    print(json.dumps(build_reference(args.output_dir, args.timestep_policy), indent=2))


if __name__ == "__main__":
    main()
