#!/usr/bin/env python3
"""Bounded formal FP32 vs actual raw700 integer-backbone observations, no training."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import supplement_srnn_backbone_numeric as numeric


def compare(fixture, observed, handoff, output):
    started = time.perf_counter()
    fixture, observed = numeric.project_path(fixture), numeric.project_path(observed)
    with np.load(fixture / "backbone_reference.npz", allow_pickle=False) as values:
        counts, labels = values["counts"], values["labels"]
        integer_spikes = values["spike"][:, :, :450]
    actual = numeric.load_observed(observed, len(counts))["spike"]
    np.testing.assert_array_equal(actual, integer_spikes)
    import torch
    import compare_srnn_shd_shifted_optimizers as original
    import build_srnn_formal_m2_fixture as m2

    sys.path.insert(0, str(handoff / "tp_exp/python"))
    trainer = importlib.import_module("multilayer_tp_train")
    base = importlib.import_module("m1_m2_train")
    config_path = handoff / "tp_exp/configs/B64_FORMAL_SEED_A.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["threads"] = 1
    model, optimizers = original.fresh_model(handoff, config, trainer, base)
    del optimizers
    model.eval()
    model.reset_potential()
    spikes = []
    with torch.no_grad():
        for frame in counts:
            spike, _, _ = model.network[0](torch.from_numpy(frame).float())
            spikes.append(spike.to(torch.int64).numpy())
    fp32_spikes = np.stack(spikes)
    wh1 = numeric.read_signed(m2.WH1, 16, (256, 450))
    wout = numeric.read_signed(m2.WOUT, 16, (20, 256))
    m2_actual, _ = m2.reference(actual * m2.SPIKE, labels, wh1, wout)
    m2_fp32_backbone, _ = m2.reference(fp32_spikes * m2.SPIKE, labels, wh1, wout)
    with np.load(fixture / "source_q14_reference.npz", allow_pickle=False) as q14:
        q14_spikes = q14["spike"]
    result = {
        "status": "bounded_formal_FP32_comparison_complete_not_full_dataset",
        "source": {"fixture": str(fixture.relative_to(numeric.ROOT)),
                   "observed": str(observed.relative_to(numeric.ROOT)),
                   "formal_config": str(config_path),
                   "checkpoint": str(handoff / "tp_exp/runs/formal_m2/checkpoints/best_checkpoint.pt"),
                   "sample_indices": [0, 1], "labels": labels.tolist(), "steps": len(counts),
                   "input": "same existing official SHD 10000us count prefix; no binary conversion"},
        "actual_INT8_backbone_vs_formal_FP32_spike": numeric.difference(actual, fp32_spikes),
        "Q14_diagnostic_vs_formal_FP32_spike": numeric.difference(q14_spikes, fp32_spikes),
        "M2_off_fixed_arithmetic_input_effect": {
            key: numeric.difference(m2_actual[key], m2_fp32_backbone[key])
            for key in ("current", "pre", "spike", "logit")},
        "spikes_per_timestep": {"actual_INT8": actual.sum(axis=(1, 2)).tolist(),
                                "formal_FP32": fp32_spikes.sum(axis=(1, 2)).tolist()},
        "limitations": ["B2T8 prefix is not a full-sequence or accuracy experiment.",
                        "M2 is held at identical integer equations to isolate the backbone input effect; this is not all-FP32 vs all-RTL comparison.",
                        "The earlier Q14 lane is a diagnostic, not the formal FP32 model.",
                        "This does not select a new precision or learning algorithm."],
        "seconds": time.perf_counter() - started,
        "peak_process_memory": numeric.peak_memory(),
        "torch": torch.__version__, "numpy": np.__version__, "device": "cpu", "threads": 1,
    }
    output = numeric.new_output(output)
    np.savez(output / "formal_fp32_prefix.npz", counts=counts, labels=labels, spike=fp32_spikes)
    numeric.json_write(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, default=numeric.ROOT.parent / "SHD_M2_WH1_TP_ONLINE_HANDOFF_20260829")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.fixture, args.observed, args.handoff, args.output), indent=2))
