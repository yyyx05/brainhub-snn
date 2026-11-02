"""Read-only numeric provenance from the formal M2 backbone to deployed codes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def read_signed(path, bits, shape):
    values = np.asarray([int(line.strip(), 16) for line in path.read_text(encoding="ascii").splitlines()
                         if line.strip()], dtype=np.int64)
    if np.any(values >= (1 << bits)):
        raise ValueError(f"out-of-range code in {path}")
    return np.where(values >= (1 << (bits-1)), values-(1 << bits), values).reshape(shape)


def compare(expected, actual):
    return {"elements": int(expected.size), "mismatches": int(np.count_nonzero(expected != actual)),
            "max_absolute_code_error": int(np.abs(expected-actual).max())}


def audit(handoff):
    checkpoint = handoff / "tp_exp/runs/formal_m2/checkpoints/best_checkpoint.pt"
    with torch.serialization.safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype,
                                          type(np.dtype(np.uint32))]):
        state = torch.load(checkpoint, weights_only=True, map_location="cpu")["parameter_state"]
    result = {"checkpoint": str(checkpoint), "matrices": {},
              "scope": "Numeric weight identity and documented quantization only; not forward equivalence or accuracy",
              "q14_rule": "FP32 multiply16384, ties-to-even round, signed16 saturation, destination/source transpose",
              "int8_rule": "Q14 shift8, sign-preserving magnitude half-away rounding, signed8 saturation",
              "h480_rule": "zero-pad source and/or destination dimensions; preserve first450 neurons"}
    for short, key, shape, padded_shape in (
            ("w0", "network.0.fc.weight", (700, 450), (700, 480)),
            ("r", "network.0.rc.weight", (450, 450), (480, 480))):
        q14 = torch.round(state[key]*16384).clamp(-32768, 32767).to(torch.int64).numpy().T
        q14_path = ROOT / f"imports/srnn/data/shd/weights/{short}_q14.mem"
        int8_path = ROOT / f"models/srnn_shd_700_450_20_int8_layer_scaled_v2/{short}_int8.mem"
        padded_path = ROOT / f"models/srnn_shd_700_480_20_int8_layer_scaled_v2_padded/{short}_int8.mem"
        actual_q14 = read_signed(q14_path, 16, shape)
        int8 = np.clip(np.sign(q14)*((np.abs(q14)+128) >> 8), -128, 127)
        actual_int8 = read_signed(int8_path, 8, shape)
        padded = np.zeros(padded_shape, dtype=np.int64)
        padded[:shape[0], :shape[1]] = int8
        actual_padded = read_signed(padded_path, 8, padded_shape)
        result["matrices"][short] = {
            "paths": [str(p.relative_to(ROOT)) for p in (q14_path, int8_path, padded_path)],
            "checkpoint_to_q14": compare(q14, actual_q14),
            "q14_to_int8": compare(int8, actual_int8),
            "h480_padding": compare(padded, actual_padded)}
    result["status"] = "PASS" if all(
        row[stage]["mismatches"] == 0 for row in result["matrices"].values()
        for stage in ("checkpoint_to_q14", "q14_to_int8", "h480_padding")) else "FAIL"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, default=Path("D:/Desktop/SHD_M2_WH1_TP_ONLINE_HANDOFF_20260829"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/paper_evidence/supplement_20261004/algorithm/srnn_backbone_origin/summary.json")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("use a new output within the project")
    result = audit(args.handoff)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
