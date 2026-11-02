#!/usr/bin/env python3
"""Generate explicit Conv8 service input from the original natural raw frames.

The existing integer network is the oracle for omitted Conv1..7. This is not
RTL evidence for those omitted layers, nor a new end-to-end performance run.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_scnn_full_integer import IntegerNetwork, equal, read_hex
from scripts.build_scnn_vgg9_full_frame_t2_reference import write_i16_hex, write_bits
import numpy as np
import torch


def train_frame_index(sample, timestep):
    return (timestep // 4) * 16 + sample * 4 + timestep % 4


def prepare(fixture, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    network = IntegerNetwork("cpu")
    equal(network.codes[-1], read_hex(fixture / "initial_codes.hex"), "original model")
    raw = read_hex(fixture / "train_raw.hex").reshape(3, 4, 4, 2, 32, 32)
    train = raw.transpose(1, 0, 2, 3, 4, 5).reshape(4, 12, 2, 32, 32).copy()
    query = read_hex(fixture / "query_raw.hex").reshape(1, 4, 2, 32, 32)
    expected_features = np.concatenate([
        read_hex(fixture / "train_features.hex"),
        read_hex(fixture / "query_features.hex")]).reshape(52, 2048)
    expected_probes = np.concatenate([
        read_hex(fixture / "train_membrane_post_probes.hex"),
        read_hex(fixture / "query_membrane_post_probes.hex")]).reshape(52, 8, 3)
    inputs = np.zeros((52, 512, 4, 4), dtype=np.uint8)
    pre = np.zeros((52, 512, 4, 4), dtype=np.int16)
    post = np.zeros_like(pre)
    spikes = np.zeros_like(inputs)
    seen = set()
    counts = dict(features=0, probes=0)

    def capture(is_query):
        def observe(timestep, layer, values):
            if layer == 8:
                return
            for sample in range(1 if is_query else 4):
                frame = 48 + timestep if is_query else train_frame_index(sample, timestep)
                state = values["next"][sample].cpu().numpy()
                c, h, w = state.shape
                points = [state[0, 0, 0], state[c // 2, h // 2, w // 2],
                          state[-1, -1, -1]]
                counts["probes"] += equal(points, expected_probes[frame, layer],
                                         f"original probe frame={frame} layer={layer+1}")
                if layer == 6:
                    pooled = values["pooled"][sample].cpu().numpy()
                    if not np.isin(pooled, [0, 256]).all():
                        raise AssertionError("Conv7 output is not binary Q8.8")
                    inputs[frame] = (pooled != 0)
                if layer == 7:
                    if frame in seen:
                        raise AssertionError("duplicate reference frame")
                    seen.add(frame)
                    counts["features"] += equal(
                        values["pooled"][sample].cpu().numpy(), expected_features[frame],
                        f"original feature frame={frame}")
                    pre[frame] = values["pre"][sample].cpu().numpy()
                    post[frame] = state
                    spikes[frame] = values["spike"][sample].cpu().numpy()
            if layer == 7:
                print(json.dumps(dict(query=is_query, timestep=timestep,
                                      frames_checked=len(seen))), flush=True)
        return observe

    network.forward(torch.from_numpy(train), sample_ids=[0, 1, 2, 3],
                    verify_ids=[0, 1, 2, 3], capture=capture(False))
    network.forward(torch.from_numpy(query), sample_ids=[22],
                    verify_ids=[22], capture=capture(True))
    if seen != set(range(52)) or counts != dict(features=106496, probes=1248):
        raise AssertionError("incomplete original-oracle cross-check")
    write_bits(output / "conv8_input_bits.hex", inputs)
    write_i16_hex(output / "conv8_pre.hex", pre)
    write_i16_hex(output / "conv8_post.hex", post)
    write_bits(output / "conv8_spike.hex", spikes)
    summary = dict(status="PREPARED_INTEGER_REFERENCE_NOT_RTL_PASS",
                   original_fixture=str(fixture), frames=52, descriptors_per_frame=1,
                   original_oracle_checks=counts, integer_checks=network.checks,
                   elapsed_seconds=time.monotonic()-started, device="cpu", threads=4,
                   scope="Original cold B4T12 plus query4. Conv1..7 replaced by explicit integer-reference service input; Conv8 and classifier remain RTL. No model, mask, scale, raw input, or expected learning change. Not full raw Conv1..8 RTL or throughput evidence.")
    (output / "reference.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.fixture.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
