#!/usr/bin/env python3
"""Prepare two native SHD sequences with persistent learning parameters only."""
import argparse
import difflib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if os.name == "nt" and not str(ROOT).startswith("\\\\?\\"):
    ROOT = Path("\\\\?\\" + str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import supplement_srnn_backbone_numeric as bone
import build_srnn_formal_m2_fixture as m2

EXPERIMENT = ROOT / "verification/experiments/srnn_resident_weight_context"
ACCEPTED = ROOT / "results/paper_evidence/supplement_20261015/performance_optimization/srnn_specialized_r1"
FIXTURE_REL = "results/paper_evidence/supplement_20261004/performance/current_source_measurements/srnn/fixture_sample0_t73"
FIELDS = ("pre", "membrane", "spike", "trace", "feature", "logit", "wout_before", "wout_after", "residual", "gradient")


def reference(inputs, label, wh1, initial_code, initial_residual, learning):
    """Reuse production integer primitives; reset dynamics, not learned state."""
    membrane = np.zeros((1, 256), dtype=np.int64)
    trace, feature = membrane.copy(), membrane.copy()
    logit = np.zeros((1, 20), dtype=np.int64)
    code, residual = initial_code.copy(), initial_residual.copy()
    records, traffic = [], []
    lut = m2.read_exp_lut()
    for frame in inputs:
        values = m2.forward_step(frame, wh1, code, membrane, trace, feature, logit)
        membrane, trace, feature, logit = (values[k] for k in ("membrane", "trace", "feature", "logit"))
        _, _, gradient = m2.wout_gradient(logit, trace, np.array([label]), lut)
        values.update(gradient=gradient, wout_before=code.copy())
        if learning:
            code, residual, stats = m2.direct_update_tensor(code, residual, gradient, 31, 65)
        else:
            stats = dict(rmw=0, code_writes=0, residual_writes=0, saturations=0, zero_skips=0)
        values.update(wout_after=code.copy(), residual=residual.copy())
        records.append(values)
        traffic.append(stats)
    return {k: np.stack([r[k] for r in records]) for k in FIELDS}, traffic


def pad_rows(values, size):
    result = np.zeros((size, *values.shape[1:]), dtype=values.dtype)
    result[:len(values)] = values
    return result


def fixture(data_dir, output):
    base = ACCEPTED / FIXTURE_REL
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["dataset"], manifest["time_window_us"], manifest["include_incomplete"]) != ("SHD official test", 10000, False):
        raise ValueError("requires the existing complete 10 ms official SHD serialization")
    samples = [manifest["samples"][i] for i in (0, 1)]
    if [r["dataset_index"] for r in samples] != [0, 1] or any(r["clipped_feature_counts"] for r in samples):
        raise ValueError("requires unchanged official samples 0 and 1")
    steps = [r["time_steps"] for r in samples]
    if max(steps) > 128:
        raise ValueError("sample exceeds the existing timestep capacity")
    for i, row in enumerate(samples):
        words = [int(x, 16) for x in (data_dir / f"sample_{i:04d}.hex").read_text(encoding="ascii").split()]
        if len(words) != row["records"] or not words or words[-1] != 2 << 18 or sum(w >> 18 in (1, 2) for w in words) != row["time_steps"]:
            raise ValueError("native stream length/final marker differs from the source manifest")
    # Recover the same source-major, padded model from the accepted fixture.
    w0 = np.zeros((700, 480), dtype=np.int64)
    w0[:, :450] = bone.read_signed(base / "w0_refill_dest_major.mem", 16, (450, 700)).T
    recurrent = np.zeros((480, 480), dtype=np.int64)
    recurrent[:450, :450] = bone.read_signed(base / "r_refill_dest_major.mem", 16, (450, 450)).T
    wh1 = bone.read_signed(ACCEPTED / "tb/data/srnn_m2_v28_wh1_initial_q14.mem", 16, (256, 450))
    initial = bone.read_signed(ACCEPTED / "tb/data/srnn_m2_v28_wout_initial_q14.mem", 16, (20, 256))
    counts = [bone.read_prefix(data_dir / f"sample_{i:04d}.hex", n)[:, None, :] for i, n in enumerate(steps)]
    backs = [bone.backbone_reference(c, w0, recurrent) for c in counts]
    # The first sequence must reproduce every previously accepted oracle field.
    for learning in (False, True):
        arm = "on" if learning else "off"
        arrays, _ = reference(backs[0]["spike"][:, :, :450]*16384, samples[0]["label"], wh1, initial, np.zeros_like(initial), learning)
        for key in FIELDS:
            bits = 64 if key in ("logit", "gradient") else 16 if key in ("wout_before", "wout_after", "residual") else 32
            expected = m2.read_signed_hex(base / f"{arm}_{key}.mem", bits, int(np.prod(arrays[key].shape))).reshape(arrays[key].shape)
            if not np.array_equal(expected, arrays[key]):
                raise ValueError("first-sequence oracle changed: "+arm+"/"+key)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(data_dir / "manifest.json", output / "source_manifest.json")
    groups = [list(g for frame in c[:, 0] for g in bone.p4_groups(frame)) for c in counts]
    maximum_groups = max(map(len, groups))
    for i, (c, back) in enumerate(zip(counts, backs)):
        directory = output / f"sequence_{i}"
        directory.mkdir()
        shutil.copy2(data_dir / f"sample_{i:04d}.hex", directory / "source_sample.hex")
        for name in ("w0_refill_dest_major.mem", "r_refill_dest_major.mem"):
            shutil.copy2(base / name, directory / name)
        bone.write_hex(directory / "raw700_counts.mem", c, 8)
        for key in ("pre", "spike"):
            bone.write_hex(directory / f"backbone_{key}.mem", pad_rows(back[key], max(steps)), 32)
        offsets, group_counts, packed = [0], [], []
        for frame in c[:, 0]:
            row = bone.p4_groups(frame)
            group_counts.append(len(row))
            packed += [mask | sum(a << (8+11*lane) for lane, a in enumerate(addresses)) for mask, *addresses in row]
            offsets.append(len(packed))
        bone.write_hex(directory / "feedforward_p4.mem", pad_rows(np.array(packed, dtype=np.int64), maximum_groups), 64)
        bone.write_hex(directory / "feedforward_offsets.mem", offsets+[offsets[-1]]*(max(steps)-steps[i]), 32)
        bone.write_hex(directory / "feedforward_group_counts.mem", group_counts+[0]*(max(steps)-steps[i]), 16)
        bone.write_hex(directory / "labels.mem", [samples[i]["label"]], 8)
    expected = {}
    for learning in (False, True):
        arm = "on" if learning else "off"
        code, residual, rows = initial.copy(), np.zeros_like(initial), []
        for i, back in enumerate(backs):
            starting_nonzero = int(np.count_nonzero(residual))
            arrays, traffic = reference(back["spike"][:, :, :450]*16384, samples[i]["label"], wh1, code, residual, learning)
            for key in FIELDS:
                bits = 64 if key in ("logit", "gradient") else 16 if key in ("wout_before", "wout_after", "residual") else 32
                bone.write_hex(output / f"sequence_{i}/{arm}_{key}.mem", pad_rows(arrays[key], max(steps)), bits)
            code, residual = arrays["wout_after"][-1], arrays["residual"][-1]
            rows.append(dict(sequence=i, dataset_index=samples[i]["dataset_index"], label=samples[i]["label"], steps=steps[i],
                bone=steps[i]*450, hidden=steps[i]*256, logits=steps[i]*20, store=steps[i]*120320,
                commits=steps[i] if learning else 0, code=sum(t["code_writes"] for t in traffic),
                residual=sum(t["residual_writes"] for t in traffic), initial_residual_nonzero=starting_nonzero,
                final_residual_nonzero=int(np.count_nonzero(residual))))
        expected[arm] = rows
    report = dict(status="REFERENCE_ONLY_NOT_RTL", native_steps=steps, capacity_steps=max(steps), maximum_feedforward_groups=maximum_groups,
        expected=expected, first_sequence_all_oracle_fields_match_accepted=True,
        data_input=str(data_dir), model_source="accepted srnn_specialized_r1 actual fixture and initial WH1/Wout",
        boundary="Two different official samples, independently reset logical dynamics; learned codes/residuals continue across samples. Only array capacity is zero-padded; padded timesteps are not executed. Not Test accuracy or board throughput.")
    (output / "reference.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def testbench(output, info):
    source = (ACCEPTED / "tb_srnn_profile_measurement.sv").read_text(encoding="utf-8")
    text = source
    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError("multisequence TB anchor not unique: "+old)
        text = text.replace(old, new, 1)
    replace("localparam int H = 450, T = 73, BONE_BASE = 256;", f"localparam int H = 450, T = {info['capacity_steps']}, BONE_BASE = 256;")
    replace("logic [63:0] ff_groups[0:5176];", f"logic [63:0] ff_groups[0:{info['maximum_feedforward_groups']-1}];")
    replace("    string raw, fixture, mode;", "    string raw, fixture, mode, fixture_root;\n    integer sequence_index=0, completed_steps=0;\n    integer native_steps[0:1] = '{"+','.join(map(str,info['native_steps']))+"};")
    load_start = text.index('        $readmemh({raw,"/backbone_pre.mem"}')
    load_end = text.index('        i_parameter_owner = ', load_start)
    reload = '\n'.join(line for line in text[load_start:load_end].splitlines() if 'model_mem' not in line)
    replace('        mode = learning ? "/on_" : "/off_";', '''        fixture_root = fixture;
        raw = {fixture_root,"/sequence_0"}; fixture = raw;
        mode = learning ? "/on_" : "/off_";''')
    anchor = '        initialized_cycle = cycles; profile_enter("boundary");'
    clear = '''        for (sequence_index=0; sequence_index<2; sequence_index++) begin
            longint setup_begin;
            setup_begin=cycles;
            if (sequence_index != 0) begin
                raw={fixture_root,"/sequence_1"}; fixture=raw;
RELOAD
                i_learning_srnn_label_class_id=labels[0][4:0];
                for (int tile=0; tile<12; tile++) for (int row=0; row<4; row++) begin
                    @(negedge i_clk); i_maintenance_tile=4'(tile);
                    i_state_maintenance_write_addr=10'(BONE_BASE+row);
                    i_trace_maintenance_write_addr=10'(512+BONE_BASE+row);
                    i_state_maintenance_write_valid=1; i_trace_maintenance_write_valid=1;
                    do @(posedge i_clk); while (!o_state_maintenance_write_ready || !o_trace_maintenance_write_ready);
                    @(negedge i_clk); i_state_maintenance_write_valid=0; i_trace_maintenance_write_valid=0;
                end
                previous_spikes='0; next_spikes='0;
                bone_seen=0; hidden_seen=0; logit_seen=0; layers=0; captures=0; replays=0;
                done_count=0; commits=0; detaches=0; code_writes=0; residual_writes=0;
                gradient_seen=0; store_checks=0;
                i_learning_srnn_sequence_version=sequence_index+1;
            end
'''.replace("RELOAD",reload)
    replace(anchor, clear+anchor)
    replace("for (timestep=0; timestep<T; timestep++)", "for (timestep=0; timestep<native_steps[sequence_index]; timestep++)")
    replace("if (timestep == 0) refill(0);", "if (sequence_index == 0 && timestep == 0) refill(0);")
    replace("if (timestep == 0) refill(1);", "if (sequence_index == 0 && timestep == 0) refill(1);")
    coverage_start = text.index("        if(bone_seen!=T*H")
    coverage_end = text.index("        profile_finish();", coverage_start)
    old = text[coverage_start:coverage_end]
    new = old.replace("T*", "native_steps[sequence_index]*").replace("? T : 0", "? native_steps[sequence_index] : 0")
    new = new.replace("resident_wout_loads > T", "resident_wout_loads > completed_steps+native_steps[sequence_index]")
    new = new.replace("steps=73 learning=%0d", "steps=%0d learning=%0d").replace("            learning,bone_seen,", "            native_steps[sequence_index],learning,bone_seen,")
    replace(old,new)
    replace("            T,learning,initialized_cycle,cycles,store_checks);",
            "            native_steps[sequence_index],learning,initialized_cycle,cycles,store_checks);")
    replace("        profile_finish();\n        $finish;", '''        profile_finish();
        $display("RESIDENT_NATIVE_SEQUENCE sequence=%0d steps=%0d learning=%0d cycles=%0d state_setup_cycles=%0d bone=%0d hidden=%0d logits=%0d store=%0d code=%0d residual=%0d commits=%0d backbone_loads=%0d wh1_loads=%0d wout_loads=%0d",
            sequence_index,native_steps[sequence_index],learning,cycles-initialized_cycle,initialized_cycle-setup_begin,
            bone_seen,hidden_seen,logit_seen,store_checks,code_writes,residual_writes,commits,
            resident_backbone_loads,resident_wh1_loads,resident_wout_loads);
        completed_steps+=native_steps[sequence_index];
        end
        $display("RESIDENT_NATIVE_PAIR_PASS sequences=2 learning=%0d steps=%0d",learning,completed_steps);
        $finish;''')
    (output / "tb_srnn_profile_measurement.sv").write_text(text, encoding="utf-8")
    (output / "testbench_changes.diff").write_text(''.join(difflib.unified_diff(source.splitlines(True), text.splitlines(True), fromfile="accepted/srnn_specialized_r1_tb", tofile="native_pair/tb")), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--name", default="multi_fixture_r1")
    args = parser.parse_args()
    if not args.name.replace('_','').isalnum():
        parser.error("simple new fixture name required")
    output = EXPERIMENT / args.name
    info = fixture(args.data_dir.resolve(), output)
    testbench(output, info)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
