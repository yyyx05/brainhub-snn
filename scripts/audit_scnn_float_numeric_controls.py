"""Disentangle SCNN weight-code provenance and floating execution controls."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_scnn_full_integer import MODEL_DIR, read_hex, metric
from scripts.supplement_scnn_weight_quantization import (
    CHECKPOINT, PAYLOAD, build_weight_arm, input_batches,
)
from scripts import run_scnn_tp_selected_layer_shift_adaptation as api
from scripts import run_scnn_tp_subject_pair_ablation as subjects_api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve() / "result.json"
    if not out.is_relative_to(ROOT) or out.exists():
        raise ValueError("use a new output inside project")
    torch.set_num_threads(1)
    torch.manual_seed(38472)
    device = torch.device("cuda")
    data = PAYLOAD / "data"
    subjects, count = subjects_api.read_subjects(data / "test_264x20_metadata.json")
    if count != 264:
        raise ValueError("unexpected fixture")
    labels = api.read_labels(data / "test_264_labels.hex", count)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    codes = []
    audit = []
    for layer in range(9):
        name = f"network.{layer}.fc.weight"
        source = checkpoint[name]
        bank = layer + 1 if layer < 8 else 12
        values = read_hex(MODEL_DIR / f"bank_{bank:02d}.hex").reshape(source.shape)
        codes.append(values)
        entry = dict(bank=bank, words=values.size)
        for kind, tensor in (("cpu", source), ("cuda", source.to(device))):
            generated = (api.quantize_weight(api.effective_weight(tensor), fractional_bits=15) * 32768).to(torch.int64).cpu().numpy()
            entry[kind] = dict(differing_codes=int((values != generated).sum()),
                               max_abs_code_difference=int(np.abs(values - generated).max()))
        audit.append(entry)
    del checkpoint
    results = {}
    configurations = [("q015", "strict"), ("q015", "historical_default"),
                      ("banks", "strict"), ("banks", "historical_default"), ("fp32", "strict")]
    for source, arithmetic in configurations:
        strict = arithmetic == "strict"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = strict
        torch.use_deterministic_algorithms(strict)
        torch.backends.cudnn.allow_tf32 = not strict
        torch.backends.cuda.matmul.allow_tf32 = False
        model = build_weight_arm(api, PAYLOAD / "traces_propagation-main", CHECKPOINT,
                                 arm="fp32" if source == "fp32" else "q015", device=device)
        model.requires_grad_(False)
        if source == "banks":
            with torch.no_grad():
                for layer, values in enumerate(codes):
                    parameter = model.network[layer].fc.weight
                    parameter.copy_(torch.tensor(values / 32768.0, device=device, dtype=parameter.dtype))
        predictions = []
        with (data / "test_264x20_inputs_q8_8.hex").open(encoding="ascii") as stream:
            for offset, frames in input_batches(stream, count, 4, (20, 2, 32, 32)):
                x = torch.tensor(frames, device=device)
                y = torch.tensor(labels[offset:offset + len(frames)], device=device)
                result = api.evaluate(model, x, y, list(range(len(frames))), batch_size=4, dx=0, dy=0)
                predictions.extend(result["predictions"])
        key = f"{source}_{arithmetic}"
        results[key] = metric(predictions, labels, subjects)
        print(f"SCNN_FLOAT_CONTROL {key} correct={results[key]['correct']}/264", flush=True)
        del model, x, y
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        status="completed", weight_provenance=audit, arms=results, labels=labels.tolist(),
        settings=dict(strict="cudnn deterministic=true, use_deterministic_algorithms=true, cudnn/matmul TF32=false",
                      historical_default="cudnn deterministic=false, use_deterministic_algorithms=false, cudnn TF32=true, matmul TF32=false",
                      common="benchmark=false; FP32 tensors; same 264 inputs, B4T20, no adaptation"),
        scope="Control experiment, not selection of a best model. Hardware banks remain unchanged. Differences in floating predictions must not be attributed solely to weight bitwidth without holding numerical implementation constant.",
    ), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
