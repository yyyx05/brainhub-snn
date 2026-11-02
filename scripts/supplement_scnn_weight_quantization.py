"""Same-checkpoint clean FP32 versus Q0.15 effective weights, FP32 execution.

No adaptation, tuning, input changes, or model-source edits. Stream the existing
264-sample fixture in B4 through the original evaluation API, one arm at a time.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "server_runs/scnn_subject_disjoint_ablation_20260919/payload"
CHECKPOINT = ROOT / "imports/scnn/model/parameters.pt"
OUTPUT = ROOT / "results/paper_evidence/supplement_20261004/algorithm/scnn_weight_only_a1"


def input_batches(stream, count, batch_size, sample_shape):
    """Same signed Q8.8 decoding as read_signed_q8_8, bounded to one batch."""
    import numpy as np

    words = (int(line, 16) for line in stream if line.strip())
    sample_words = int(np.prod(sample_shape))
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        try:
            raw = np.fromiter(words, dtype=np.uint16, count=size * sample_words)
        except ValueError as error:
            raise ValueError(f"incomplete fixture batch starting at sample {start}") from error
        yield start, (raw.view(np.int16).astype(np.float32) / 256.0).reshape(
            size, *sample_shape)
    if next(words, None) is not None:
        raise ValueError("fixture contains extra words after the declared samples")


def build_weight_arm(api, traces_root, checkpoint, *, arm, device):
    if arm not in ("fp32", "q015"):
        raise ValueError(arm)
    quantize = api.quantize_weight

    def choose(value, *, fractional_bits):
        return value if arm == "fp32" else quantize(value, fractional_bits=fractional_bits)

    with patch.object(api, "quantize_weight", choose):
        return api.build_model(traces_root, checkpoint, batch_size=4,
                               device=device, weight_fractional_bits=15)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(dict(pid=os.getpid(), samples=264, batch_size=4, arms=["fp32", "q015"],
                          checkpoint=str(CHECKPOINT), run=args.run)), flush=True)
    if not args.run:
        return 0
    output = args.output_dir.resolve() / "clean_comparison.json"
    if not output.is_relative_to(ROOT):
        raise ValueError("new outputs must stay inside the project")
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from scripts import run_scnn_tp_selected_layer_shift_adaptation as api
    from scripts import run_scnn_tp_subject_pair_ablation as subjects_api

    torch.set_num_threads(1)
    torch.manual_seed(38472)
    device = torch.device(args.device)
    data_dir = PAYLOAD / "data"
    subjects, count = subjects_api.read_subjects(data_dir / "test_264x20_metadata.json")
    if count != 264:
        raise ValueError(f"expected existing 264-sample fixture, got {count}")
    labels = api.read_labels(data_dir / "test_264_labels.hex", count)
    results = {}
    started = time.perf_counter()
    for arm in ("fp32", "q015"):
        arm_started = time.perf_counter()
        model = build_weight_arm(api, PAYLOAD / "traces_propagation-main", CHECKPOINT,
                                 arm=arm, device=device)
        model.requires_grad_(False)
        predictions = []
        with (data_dir / "test_264x20_inputs_q8_8.hex").open("r", encoding="ascii") as stream:
            for start, frames in input_batches(stream, count, 4, (20, 2, 32, 32)):
                size = len(frames)
                x = torch.from_numpy(frames).to(device)
                y = torch.from_numpy(labels[start:start + size]).to(device)
                metric = api.evaluate(model, x, y, list(range(size)),
                                      batch_size=4, dx=0, dy=0, corruption=None)
                predictions.extend(metric["predictions"])
        prediction_array = np.asarray(predictions)
        correct = int(np.count_nonzero(prediction_array == labels))
        by_subject = {}
        for subject in sorted(set(int(value) for value in subjects)):
            selected = np.asarray(subjects) == subject
            subtotal = int(np.count_nonzero(selected))
            hits = int(np.count_nonzero(prediction_array[selected] == labels[selected]))
            by_subject[str(subject)] = dict(correct=hits, total=subtotal,
                                             accuracy_percent=100 * hits / subtotal)
        results[arm] = dict(correct=correct, total=count, accuracy_percent=100 * correct / count,
                            predictions=predictions, by_subject=by_subject,
                            wall_seconds=time.perf_counter() - arm_started)
        print(f"SCNN_WEIGHT_ONLY arm={arm} correct={correct}/{count} "
              f"seconds={results[arm]['wall_seconds']:.3f}", flush=True)
        del model, x, y
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = dict(
        status="completed", checkpoint=str(CHECKPOINT.relative_to(ROOT)), device=str(device),
        inputs=str((data_dir / "test_264x20_inputs_q8_8.hex").relative_to(ROOT)),
        metadata=str((data_dir / "test_264x20_metadata.json").relative_to(ROOT)),
        configuration=dict(samples=count, batch_size=4, timesteps=20, membrane_leak="135/256",
                           trace_leak="250/256", output_leak=1.0, norm_type="none",
                           weights="same effective_weight(WS checkpoint); quantizer identity vs original signed16 Q0.15",
                           forward="FP32 in both arms", adaptation=False, corruption=None),
        arms=results, labels=labels.tolist(),
        q015_minus_fp32_accuracy_points=results["q015"]["accuracy_percent"] - results["fp32"]["accuracy_percent"],
        prediction_disagreements=sum(a != b for a, b in zip(
            results["fp32"]["predictions"], results["q015"]["predictions"])),
        wall_seconds=time.perf_counter() - started,
        claim_boundary="Weight quantization loss only, both networks executed in FP32 at the same final neuron coefficients. Existing 264-sample clean fixture includes development and report subjects; not full DVS-Gesture test, retraining, or full-network fixed-point accuracy.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"SCNN_WEIGHT_ONLY_COMPLETE {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
