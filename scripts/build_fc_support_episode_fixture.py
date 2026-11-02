#!/usr/bin/env python3
"""Continue the approved support160 episode with persistent W2 and residuals."""
from contextlib import ExitStack
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.export_fc_formal_counts import EPISODE, load_formal_batch
from scripts.build_fc_final_counts_fixture import (
    BATCH, HIDDEN, CLASSES, MEM_LAYOUT, load_model, forward_step,
    quantized_error, batch_gradient, scale_gradient, apply_residual_update, _write_mem,
)
import numpy as np

OUTPUT = ROOT / "results/paper_evidence/supplement_20261004/algorithm/fc_final_counts_reference"


def run_batch(out, counts, labels, w0, w2, residual, initial_w2, scale, learning):
    """Reset sample states only; return the updated persistent parameter state."""
    steps = counts.shape[1]
    voltage = np.zeros((BATCH, HIDDEN), dtype=np.int64)
    trace = np.zeros_like(voltage)
    cumulative = np.zeros((BATCH, CLASSES), dtype=np.int64)
    records = []
    out.mkdir(parents=True)
    with ExitStack() as stack:
        streams = {name: stack.enter_context((out / f"{name}.mem").open("x", encoding="ascii"))
                   for name in (*MEM_LAYOUT, "parameters", "labels")}
        _write_mem(streams["parameters"], np.concatenate((w0.ravel(), w2.ravel())), 16)
        _write_mem(streams["labels"], labels, 8)
        for t in range(steps):
            frame = counts[:, t]
            forward = forward_step(frame, w0, w2, voltage, trace, cumulative, scale)
            voltage, trace, cumulative = (forward[key] for key in ("voltage", "trace", "cumulative"))
            _, error = quantized_error(forward["q6"], labels)
            gradient = batch_gradient(trace, error)
            delta, scale_saturated = scale_gradient(gradient)
            if learning:
                next_w2, next_residual, saturated = apply_residual_update(w2, residual, delta)
            else:
                next_w2, next_residual = w2.copy(), residual.copy()
                saturated = np.zeros_like(w2, dtype=bool)
            tensors = {"input_counts": frame, **forward, "w2_after": next_w2,
                       "residual_after": next_residual}
            for name, (bits, _, _, _) in MEM_LAYOUT.items():
                _write_mem(streams[name], tensors[name], bits)
            records.append({"t": t, "input_events": int(frame.sum()),
                            "hidden_spikes": int(forward["spike"].sum()),
                            "update_requests": int(np.count_nonzero(delta)) if learning else 0,
                            "code_changes": int(np.count_nonzero(next_w2 != w2)),
                            "residual_changes": int(np.count_nonzero(next_residual != residual)),
                            "code_saturations": int(np.count_nonzero(saturated)),
                            "scaler_saturations": int(np.count_nonzero(scale_saturated)),
                            "changed_from_initial": int(np.count_nonzero(next_w2 != initial_w2)),
                            "residual_abs_max": int(np.abs(next_residual).max())})
            w2, residual = next_w2, next_residual
    np.savez_compressed(out / "final_state.npz", w2=w2, residual=residual)
    return w2, residual, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning", choices=("on", "off"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(OUTPUT.resolve()):
        parser.error(f"output must be inside {OUTPUT}")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    provenance = json.loads((OUTPUT.parent / "fc_formal_counts/summary.json").read_text(encoding="utf-8"))
    episode = json.loads(EPISODE.read_text(encoding="utf-8"))
    indices = episode["support_indices"]
    if len(indices) != 160 or len(set(indices)) != 160:
        raise ValueError("expected the original 160 unique support samples")
    import torch
    torch.set_num_threads(1)
    from scripts.run_fc_tp_supervised_shift_adaptation import shift_nmnist_frames
    model = load_model()
    w0, initial_w2 = (model[name].astype(np.int64) for name in ("W0", "W2"))
    w2, residual = initial_w2.copy(), np.zeros_like(initial_w2)
    scale = round(float(model["W2_scale"]) * 64 * 65536)
    records, samples, batch_steps = [], [], []
    for batch in range(10):
        frames, labels, identities = load_formal_batch(
            traces_root=Path(provenance["traces_root"]), data_root=Path(provenance["data_root"]),
            role="support", indices=indices[batch*16:(batch+1)*16])
        counts = shift_nmnist_frames(frames, dx=2, dy=0).numpy().astype(np.int64)
        if counts.shape[1] > 99 or np.any(counts > 65535):
            raise ValueError("input exceeds the bounded testbench representation")
        batch_steps.append(counts.shape[1])
        w2, residual, items = run_batch(out / f"batch_{batch}", counts, labels.numpy(),
                                       w0, w2, residual, initial_w2, scale, args.learning == "on")
        for item in items:
            item["batch_index"] = batch
        records.extend(items)
        samples.extend(dict(item, batch_index=batch) for item in identities)
        print(json.dumps({"batch": batch, "steps": counts.shape[1],
                          "code_events": sum(i["code_changes"] for i in items),
                          "changed_w2": int(np.count_nonzero(w2 != initial_w2)),
                          "elapsed": round(time.monotonic()-started, 2)}), flush=True)
    with (out / "batch_steps.mem").open("x", encoding="ascii") as stream:
        _write_mem(stream, batch_steps, 32)
    summary = {"status": "PASS", "scope": "integer reference only; not RTL or dataset accuracy",
               "learning": args.learning == "on", "batch": 16, "batches": 10,
               "steps": batch_steps[0], "batch_steps": batch_steps, "total_steps": sum(batch_steps),
               "episode": str(EPISODE), "support_indices": indices, "samples": samples,
               "numeric_source": "build_fc_final_counts_fixture.py; unchanged multiplier11/threshold32768",
               "state_boundary": "reset vmem/trace/cumulative per batch; retain W2/residual across all ten batches",
               "records": records, "totals": {key: sum(item[key] for item in records) for key in (
                   "input_events", "hidden_spikes", "update_requests", "code_changes", "residual_changes",
                   "code_saturations", "scaler_saturations")},
               "final_changed_w2": int(np.count_nonzero(w2 != initial_w2)),
               "first_code_change": next(({"batch_index": i["batch_index"], "t": i["t"]}
                                          for i in records if i["code_changes"]), None),
               "wall_seconds": time.monotonic()-started}
    (out / "reference.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"totals": summary["totals"], "first_code_change": summary["first_code_change"],
                      "wall_seconds": summary["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
