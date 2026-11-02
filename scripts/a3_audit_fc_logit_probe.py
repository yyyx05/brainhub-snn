#!/usr/bin/env python3
"""CPU-only source-semantic and scalar-rounding probe; not an accuracy run."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACES = ROOT / "server_runs/scnn_subject_disjoint_ablation_20260919/payload/traces_propagation-main"
WEIGHTS = ROOT / "imports/fc/p3a_h160/data/weights/p3a_nmnist_h160_k8_int5_w2_int8_hash8/weights_h160_int5_w2_int8.npz"
OUTPUT = ROOT / "results/paper_evidence/supplement_20261004/algorithm/a3_audit/a3_audit_fc_logit_probe.json"


def rne_ratio(value: int, multiplier: int, shift: int = 16) -> int:
    product = value * multiplier
    q, r = divmod(abs(product), 1 << shift)
    half = 1 << (shift - 1)
    q += int(r > half or (r == half and q % 2 != 0))
    return -q if product < 0 else q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(args.traces_root.resolve()))
    from models.tp_mlp import TP_MLP
    from models.neuron_layers import LILayer
    from scripts.screen_fc_tp_update_candidates import _snn_params

    torch.set_num_threads(1)
    torch.manual_seed(0)
    params = _snn_params()
    model = TP_MLP([1], 1, 2, 1, params).to("cpu")
    with torch.no_grad():
        model.network[0].fc.weight.fill_(2.0)
        model.network[-1].fc.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        model.target_propagator.fc.weight.zero_()
    model.reset_potential()
    target = torch.tensor([[1.0, 0.0]])
    steps = []
    for index, x in enumerate((1.0, 0.0, 1.0)):
        if index == 2:
            with torch.no_grad():
                model.network[-1].fc.weight.copy_(torch.tensor([[2.0], [-2.0]]))
        logits, _ = model(torch.tensor([[x]]), target)
        raw = model.network[-1].fc(model.hs[-1]).detach()
        expected_error = torch.softmax(logits, 1) - target
        torch.testing.assert_close(model.e, expected_error, rtol=0, atol=0)
        before = model.network[-1].membrane_potential.detach().clone()
        model.detach_membrane_states()
        torch.testing.assert_close(model.network[-1].membrane_potential, before, rtol=0, atol=0)
        steps.append({"input": x, "hidden_spike": model.hs[-1].detach().tolist(),
                      "step_current": raw.tolist(), "learning_logits": logits.detach().tolist(),
                      "error": model.e.detach().tolist()})
    assert [s["learning_logits"][0][0] for s in steps] == [1.0, 1.0, 3.0]
    assert [s["step_current"][0][0] for s in steps] == [1.0, 0.0, 2.0]
    model.reset_potential()
    with torch.no_grad():
        model.network[-1].fc.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        inference = [model.inference(torch.tensor([[x]])).clone().tolist()
                     for x in (1.0, 0.0, 1.0)]
    assert [s[0][0] for s in inference] == [1.0, 1.0, 2.0]

    with np.load(WEIGHTS, allow_pickle=False) as packed:
        scale = float(packed["W2_scale"])
        w2 = packed["W2"].astype(np.int64)
        w0_scale = float(packed["W0_scale"])
        input_shift = int(packed["input_shift"])
    coefficient = scale * 64
    multiplier = round(coefficient * 65536)
    assert multiplier == 47303
    values = np.arange(-128 * 160, 127 * 160 + 1, dtype=np.int64)
    fixed = np.array([rne_ratio(int(v), multiplier) for v in values])
    ideal = np.rint(values * coefficient).astype(np.int64)
    mismatch = values[fixed != ideal]
    examples = [{"raw": int(v), "q6_coefficient_rne": rne_ratio(int(v), multiplier),
                 "q6_full_model_scale_rne": round(int(v) * coefficient)}
                for v in mismatch[:5]]
    effective_lr = 11 * scale * 16 / (1 << 15)
    result = {
        "status": "pass",
        "scope": "Synthetic 1-1-2 TP_MLP CPU probe plus H160 scale arithmetic; no dataset, no DUT simulation, no optimizer comparison",
        "python": sys.executable, "torch": torch.__version__, "device": "cpu", "threads": 1,
        "sources": {"tp_mlp": inspect.getsourcefile(TP_MLP), "li_layer": inspect.getsourcefile(LILayer),
                    "parameter_function": inspect.getsourcefile(_snn_params), "h160_weights": str(WEIGHTS)},
        "snn_params": params,
        "synthetic_learning_steps": steps,
        "synthetic_inference_logits": inference,
        "detach_preserves_output_membrane": True,
        "weight_change_does_not_recompute_past_contributions": True,
        "scaling": {
            "w0_scale": w0_scale, "w0_input_shift": input_shift,
            "w2_scale": scale, "w2_scale_times64": coefficient,
            "q16_coefficient": multiplier, "q16_coefficient_value": multiplier / 65536,
            "coefficient_error": multiplier / 65536 - coefficient,
            "signed_int8_h160_per_step_bound": [int(values[0]), int(values[-1])],
            "current_checkpoint_per_class_lower_bound": np.minimum(w2, 0).sum(0).tolist(),
            "current_checkpoint_per_class_upper_bound": np.maximum(w2, 0).sum(0).tolist(),
            "exhaustive_scalar_values": int(values.size),
            "full_scale_vs_quantized_coefficient_mismatches": int(mismatch.size),
            "max_q6_code_difference": int(np.abs(fixed - ideal).max()),
            "mismatch_examples": examples,
            "scope_note": "Scalar range enumeration is not a claim all values occur in actual hidden-spike patterns",
            "round_after_sum_example": {"raw_steps": [1, 1],
                "sum_of_rounded_steps": rne_ratio(1, multiplier) * 2,
                "rounded_sum": rne_ratio(2, multiplier)},
            "sat16_limits": [-32768, 32767],
            "fc_nominal_lr": 2 ** -14, "fc_multiplier11_effective_lr_before_update_rounding": effective_lr,
            "fc_lr_relative_difference": effective_lr / (2 ** -14) - 1,
        },
    }
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("output must remain inside the project")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(output), "status": result["status"], "scaling": result["scaling"]}))


if __name__ == "__main__":
    main()
