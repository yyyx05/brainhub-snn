#!/usr/bin/env python3
"""Read existing B1 evidence; write only the scoped CSV. Never run a toolchain."""

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper_evidence/supplement_20261004/performance/fc_srnn_scope"
PAPER = "results/paper_evidence/"
FC = "results/fc_first/unified_snn_core_final_v1_fc_tp_online_real_fc_h160_w2only/"
SR = "results/learning_reference/actual_srnn_m2_online/"
TWO = "results/learning_reference/actual_srnn_m2_online_two_window/"
FCTB = "tb/tb_unified_snn_core_final_v1_real_fc.sv"
SRTB = "tb/tb_unified_snn_core_final_v1_srnn_m2_online.sv"
FINALTB = "tb/tb_unified_h40_final_composition_v1.sv"
FINALRUN = "scripts/run_verilator_unified_h40_final_composition_v1.ps1"
FCRUN = "scripts/run_verilator_unified_snn_core_final_v1_real_fc.ps1"
SRRUN = "scripts/run_verilator_unified_snn_core_final_v1_srnn_m2_online.ps1"
HEADERS = ["evidence_id", "field", "value", "unit", "source", "interpretation"]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8-sig")


def anchor(path, needle):
    for number, line in enumerate(read(path).splitlines(), 1):
        if needle in line:
            return f"{path}:{number} [contains {needle}]"
    raise ValueError(f"Missing source anchor: {path}: {needle}")


def marker(path, prefix):
    matches = [(n, line) for n, line in enumerate(read(path).splitlines(), 1)
               if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {prefix!r} in {path}, got {len(matches)}")
    number, line = matches[0]
    fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
    return line, fields, f"{path}:{number} [{prefix}]"


def build_rows():
    rows = []

    def add(eid, field, value, source="NA", note="", unit="NA"):
        rows.append(dict(zip(HEADERS, (eid, field, str(value), unit, source,
                                      note or "NA"))))

    def fact(eid, field, value, path, needle, note="", unit="NA"):
        add(eid, field, value, anchor(path, needle), note, unit)

    def json_fields(eid, path, keys):
        data = json.loads(read(path))
        for key in keys:
            value = data
            for part in key.split("."):
                value = value[part]
            if isinstance(value, (dict, list, bool)):
                value = json.dumps(value, ensure_ascii=False)
            add(eid, "reported." + key, value, f"{path}#/{key.replace('.', '/')}")
        return data

    def log_fields(eid, path, prefix, rerun=None):
        line, fields, source = marker(path, prefix)
        if rerun:
            other_line, _, other_source = marker(rerun, prefix)
            if line != other_line:
                raise ValueError(f"Non-deterministic terminal marker: {path}")
            add(eid, "terminal_marker_equal_between_runs", 1,
                source + "; " + other_source,
                "Only these terminal fields compared; not full transcripts.")
        for key, value in fields.items():
            add(eid, "raw." + key, value, source + f" field={key}")
        return fields, source

    closure = "docs/FINAL_ARCHITECTURE_AND_EXPERIMENT_CLOSURE_20260928.md"
    fact("AUTHORITY", "final_top", "unified_h40_final_composition_top_v1",
         closure, "rtl/top/unified_h40_final_composition_top_v1.sv")
    fact("AUTHORITY", "version", "V2 / RUN120 R2; RUN122 learning-off",
         closure, "RUN120 R2 is the current V2")
    fact("AUTHORITY", "fc_profile", "2312->160->10; W2-only",
         closure, "2312 -> 160 -> 10")
    fact("AUTHORITY", "srnn_profile", "700->450 recurrent; M2 450->256->20; Wout-only",
         closure, "450 -> 256 -> 20")

    fc_summary_path = PAPER + "rtl/fc_uq6_10_magnitude_20260924/performance_summary.json"
    fc_summary = json_fields("FC_COMPAT", fc_summary_path, [
        "status", "evidence_date", "model", "physical_tiles", "samples",
        "timesteps_per_sample", "update_contract", "counts_per_run",
        "target_frequency_hz"])
    json_fields("FC_COMPAT", FC + "source_manifest.json", [
        "top", "testbench", "mode", "fc_compat", "real_canonical_fc",
        "ddr_stub_present", "full_release_closure"])
    fc, fc_ref = log_fields("FC_COMPAT", FC + "final_v1_real_fc_official16_run1.log",
                           "FINAL_FC_TP_NATIVE_BATCH_PASS ",
                           FC + "final_v1_real_fc_official16_run2.log")
    for key in ("learning_on_run_cycles", "learning_off_run_cycles", "w2_reads",
                "writes", "code_writes", "residual_writes"):
        assert int(fc[key]) == fc_summary["counts_per_run"][key]
    add("FC_COMPAT", "ordered_cycle_difference", int(fc["learning_on_run_cycles"]) -
        int(fc["learning_off_run_cycles"]), fc_ref,
        "Subtraction, not isolated learning or publication latency.", "cycles")
    log_fields("FC_COMPAT", FC + "final_v1_real_fc_official16_run1.log",
               "FC_COMPAT_ACTIVE4_BOUND_COMPLETE ")
    log_fields("FC_COMPAT", FC + "final_v1_real_fc_official16_run1.log",
               "UNIFIED_SNN_CORE_FINAL_V1_FC_COMPAT_REAL_FC_PASS ")
    sample_lines = [line for line in read(FC + "final_v1_real_fc_official16_run1.log").splitlines()
                    if line.startswith("FC_COMPAT_ACTIVE4_BOUND sample=")]
    samples = [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in sample_lines]
    assert [int(s["sample"]) for s in samples] == list(range(16))
    assert sum(int(s["cycles"]) for s in samples) == 33264
    add("FC_COMPAT", "labels_in_logged_subset", ";".join(sorted({s["label"] for s in samples})),
        FC + "final_v1_real_fc_official16_run1.log [FC_COMPAT_ACTIVE4_BOUND:label, samples 0..15]",
        "Homogeneous label-0 subset; not representative dataset accuracy.")
    fact("FC_COMPAT", "final_physical_path", "NO_COMPAT",
         FCRUN, '"fc_compat_real_fc_ddr_stub_isolation"',
         "Model/numeric agreement does not transfer these cycles to final V2 P4.")
    fact("FC_COMPAT", "cycle_start", "after issue_fc_runtime_run",
         FCTB, "tp_learning_run_start_cycle = core_cycle_q;")
    fact("FC_COMPAT", "cycle_end", "runtime done and quiescent",
         FCTB, "tp_learning_run_cycles = core_cycle_q -")
    fact("FC_COMPAT", "recompute_check", "16 whole-logit vectors unchanged after residual-only commit",
         FCTB, "if (fc_logits_flat != tp_pre_update_logits[probe])",
         "Self-replay consistency, not an independent per-logit golden comparison.")
    fact("FC_COMPAT", "commit_count", 1, FCTB, "tp_commit_sequence != 32'd1")

    sr_summary_path = PAPER + "rtl/srnn_m2_current_cycles_20260928/summary.json"
    sr_summary = json_fields("SRNN_OLD_B2T2", sr_summary_path, [
        "selected_mask", "workload", "train_cycles", "learning_off_cycles",
        "incremental_learning_cycles", "changed_parameters", "commits",
        "model_version", "adapt_version"])
    sr, sr_ref = log_fields("SRNN_OLD_B2T2", SR + "run1.log",
                           "UNIFIED_SNN_CORE_FINAL_V1_SRNN_M2_ONLINE_PASS ", SR + "run2.log")
    sr_cycle_fields = {"train_cycles": "train_cycles",
                       "learning_off_cycles": "inference_cycles",
                       "incremental_learning_cycles": "incremental_learning_cycles"}
    for summary_key, raw_key in sr_cycle_fields.items():
        assert int(sr[raw_key]) == sr_summary[summary_key]
    assert int(sr["train_cycles"]) - int(sr["inference_cycles"]) == 38853
    with (ROOT / (PAPER + "rtl/srnn_m2_current_cycles_20260928/summary.csv")).open(
            encoding="utf-8-sig", newline="") as handle:
        old_csv = list(csv.DictReader(handle))
    assert len(old_csv) == 1
    for summary_key, raw_key in sr_cycle_fields.items():
        assert int(old_csv[0][summary_key]) == int(sr[raw_key])
    add("SRNN_OLD_B2T2", "summary_csv_reconciled", 1,
        PAPER + "rtl/srnn_m2_current_cycles_20260928/summary.csv [row topology=SRNN, train_cycles/learning_off_cycles/incremental_learning_cycles]",
        "Numeric agreement does not validate its selected_mask label.")
    for key, value, needle in [
        ("top", "unified_snn_core_final_v1", "unified_snn_core_final_v1 #("),
        ("m2_hidden", 2, "localparam int M2_HIDDEN = 2;"),
        ("m2_classes", 2, "localparam int M2_CLASSES = 2;"),
        ("m2_backbone", 450, "localparam int M2_BACKBONE = 450;"),
        ("batch", 2, "localparam int TP_REPLAY_BATCH_MAX = 2;"),
        ("timesteps_per_sample", 2, "localparam int TP_REPLAY_TIMESTEP_MAX = 2;"),
        ("weights", "synthetic TB constants", "code_value = (address_value & 1)"),
        ("targets", "synthetic TB traces and labels", "m2_target_trace_q14 = 32'sd1000"),
        ("labels", "sample index low bit, not read SHD labels", "m2_target_label_class_id = sample[0];"),
        ("wout_update_settings", "increment=1; threshold=1", ".i_cfg_wout_increment(16'd1), .i_cfg_wout_threshold(16'd1)"),
        ("timed_start", "after configuration, parameter and target load; before start_m2_run",
         "m2_learning_start_cycle = cycle_count;"),
        ("timed_end", "after capture, M2 transaction and idle", "m2_learning_end_cycle = cycle_count;"),
        ("unique_changed_addresses", "NA", "m2_changed_after_training++;"),
        ("replay_check", "parameter code scan equality; not logit golden parity",
         "m2_inference_mismatch_count++;"),
    ]:
        fact("SRNN_OLD_B2T2", key, value, SRTB, needle)
    binding = "rtl/learning/unified_srnn_m2_actual_runtime_binding.sv"
    coordinator = "rtl/learning/unified_srnn_m2_timestep_train_infer_coordinator.sv"
    fact("SRNN_OLD_B2T2", "binding_parameter_override", "no ENABLE_WH1_LEARNING override",
         binding, "unified_srnn_m2_timestep_train_infer_coordinator #(")
    fact("SRNN_OLD_B2T2", "effective_current_source_wh1_learning", 1,
         coordinator, "parameter bit ENABLE_WH1_LEARNING = 1'b1",
         "Current source audit; raw log does not separately count Wh1/Wout writes. Do not certify historical Wout-only.")
    fact("SRNN_OLD_B2T2", "reported_mask_origin", "historical summary label, not log field; parser since corrected",
         PAPER + "rtl/srnn_m2_current_cycles_20260928/summary.json", '"selected_mask": "Wout-only"')
    fact("SRNN_OLD_B2T2", "m2_numeric_forward", "standalone numeric consumer, not P4 physical M2 engine",
         coordinator, "unified_srnn_m2_numeric_forward_observation_consumer #(")
    fact("SRNN_OLD_B2T2", "final_physical_path", "NO_OLD_WAVE3_PLUS_NUMERIC_M2",
         "rtl/top/unified_wave3_tile_major_shared_extension_leaf_v2.sv",
         "unified_h40_shared_inference_array_v1 #(",
         "Old backbone path; final array module is unified_binary_p4_h40_array12_path_v1.")
    add("SRNN_OLD_B2T2", "input_sample_accounting", "2 unique SHD samples (0,1); 4 presentations total; 8 backbone timestep commits",
        sr_ref + "; " + anchor(SRTB, "run_short_actual_sample(0);"),
        "2 M2 commits are distinct from 8 backbone commits. Counts include on and off passes.")
    add("SRNN_OLD_B2T2", "changed_counter_scope", "scan1 plus scan2 versus scan0; not unique addresses",
        anchor(SRTB, "else if (m2_forward_scan_count == 1 &&") + "; " +
        anchor(SRTB, "if (!two_learning_windows && m2_forward_code !="),
        "reported changed_parameters=8 cannot be interpreted as 8 distinct Wout elements; Wout has only 4 elements.")
    fact("SRNN_OLD_B2T2", "missing_readmem_warning", "present",
         SR + "run1.log", "%Warning: /w0_cluster_00_bank_00.mem",
         "TB leaves canonical FC weight dir empty and asserts fc_activity=0; preserve warning, not a clean full-system load claim.")

    two, two_ref = log_fields("SRNN_OLD_TWO_WINDOWS", TWO + "run1.log",
                             "UNIFIED_SNN_CORE_FINAL_V1_SRNN_M2_TWO_WINDOW_PASS ", TWO + "run2.log")
    add("SRNN_OLD_TWO_WINDOWS", "final_physical_path", "NO_SAME_OLD_HARNESS", sr_ref)
    fact("SRNN_OLD_TWO_WINDOWS", "sample_accounting",
         "train (0,1), then train (2,3), then replay (2,3); B2xT2 per window",
         SRTB, "run_short_actual_sample(two_learning_windows ? 2 : 0);")
    add("SRNN_OLD_TWO_WINDOWS", "reuse", "bounded parameter continuity and sequence semantics only", two_ref)

    frontend = PAPER + "rtl/srnn_m2_wout_only_full_shape_20260928/"
    log_fields("SRNN_B64_FRONTEND", frontend + "verilator_formal_shape_wout_only.log",
               "SRNN_M2_TP_CE_FORMAL_SHAPE_WOUT_ONLY_PASS ")
    json_fields("SRNN_B64_FRONTEND", frontend + "summary.json", ["configuration", "measured"])
    front_tb = "tb/unified_srnn_m2_tp_ce_contribution_frontend_formal_shape_tb.sv"
    fact("SRNN_B64_FRONTEND", "input_type", "synthetic observations, logits and labels; no SHD event execution",
         front_tb, "label_class = CLASS_W'(sample % CLASSES);")
    fact("SRNN_B64_FRONTEND", "cycle_boundary", "reset release through fixture loading and contribution completion",
         front_tb, "cycle_count <= cycle_count + 1;")
    add("SRNN_B64_FRONTEND", "final_physical_path", "COMPONENT_ONLY_NO_ARRAY",
        anchor(front_tb, "module unified_srnn_m2_tp_ce_contribution_frontend_formal_shape_tb"))
    add("SRNN_B64_FRONTEND", "timesteps_per_window", 1, frontend + "README.md [Claim boundary: one B64 timestep/window]")

    backend = PAPER + "rtl/srnn_m2_wout_b64_two_window_store_20260928/"
    log_fields("SRNN_B64_BACKEND", backend + "verilator.log", "SRNN_M2_WOUT_B64_TWO_WINDOW_STORE_PASS ")
    json_fields("SRNN_B64_BACKEND", backend + "fixture_summary.json", [
        "profile", "batch", "hidden", "classes", "elements", "contributions_per_window",
        "increment", "threshold", "window1_code_changes", "window2_code_changes"])
    back_tb = "tb/unified_srnn_m2_wout_b64_two_window_store_tb.sv"
    fact("SRNN_B64_BACKEND", "input_type", "precomputed CE errors and hidden traces; not raw SHD",
         back_tb, '$readmemh("tb/data/srnn_m2_v26_ce_error_q40.mem"')
    add("SRNN_B64_BACKEND", "final_physical_path", "COMPONENT_ONLY_NO_ARRAY",
        anchor(back_tb, "unified_srnn_tp_wout_direct_online_learning_engine dut ("))
    add("SRNN_B64_BACKEND", "reuse", "two-window residual/code parity and commit continuity; do not add to frontend cycles",
        backend + "README.md [Claim boundary]")

    final_log = PAPER + "rtl/final_composition_uq6_10_magnitude_20260924/final_composition_run.log"
    log_fields("FINAL_SHARED_DIRECTED", final_log, "PASS: C17 FC W2 and SRNN M2 ")
    fact("FINAL_SHARED_DIRECTED", "top", "unified_h40_final_composition_top_v1", FINALTB,
         "unified_h40_final_composition_top_v1 #(")
    fact("FINAL_SHARED_DIRECTED", "srnn_actual_shape", "B1xT1; backbone 2 -> hidden 4 -> classes 2", FINALTB,
         ".i_learning_srnn_backbone_count(10'd2)")
    fact("FINAL_SHARED_DIRECTED", "fc_actual_shape", "B2xT1; 4 -> 4 -> 2", FINALTB,
         "refill_fc_cache(FC_W0_BASE, 4, 4, 4, 1);")
    fact("FINAL_SHARED_DIRECTED", "input_type", "TB-constructed parameters and vectors", FINALTB,
         "run_fc_layer(0, 4, 1'b0, 4'd1, 4'b1111);")
    fact("FINAL_SHARED_DIRECTED", "final_physical_path", "YES_SHARED_PATH_DIRECTED_ONLY",
         "rtl/top/unified_h40_array16_three_topology_top_v1.sv",
         "unified_binary_p4_h40_array12_path_v1 #(")
    fact("FINAL_SHARED_DIRECTED", "version_boundary", "Sep24 numeric log plus Sep28 V2 functional requalification; no new cycle run",
         PAPER + "rtl/fixed_shell_topology_mask_20260928/README.md",
         "The current mask-7 source passes the FC+SRNN no-reset directed integration")
    fact("FINAL_SHARED_DIRECTED", "reuse", "numeric sharing/visibility assertions; not H160 or SHD full-shape latency",
         FINALTB, "SRNN Wout-only production mode wrote the wrong tensor")

    roundtrip = PAPER + "rtl/current_v2_three_topology_roundtrip_20260928/verilator_roundtrip.log"
    fact("V2_ROUNDTRIP", "reuse", "FC->SRNN->SCNN->FC owner and state isolation; no dataset throughput",
         roundtrip, "PASS: descriptor-driven FC recurrence-off")
    fact("V2_ROUNDTRIP", "final_physical_path", "YES_SHARED_ARRAY_SUBTOP", roundtrip,
         "exactly one physical 12xH40/P4 array")

    official = "results/unified_core/final_v1_srnn_dataset/"
    official_json = json_fields("SRNN_OLD_OFFICIAL64", official + "official64_summary.json",
                               ["samples", "exact", "correct", "workers", "reruns", "cross_shard_state"])
    total_cycles = 0
    for start in (0, 16, 32, 48):
        fields, _ = log_fields("SRNN_OLD_OFFICIAL64", official + f"s{start:04d}_n0016_run1.log",
                               "UNIFIED_SNN_CORE_FINAL_V1_SRNN_DATASET_VERILATOR_PASS ",
                               official + f"s{start:04d}_n0016_run2.log")
        total_cycles += int(fields["total_cycles"])
    assert official_json["exact"] == 64 and total_cycles == 61236743
    add("SRNN_OLD_OFFICIAL64", "summed_forward_cycles", total_cycles,
        official + "official64_summary.json#/markers/shard_0..shard_3 [total_cycles]",
        "Four independently reset shards, not wall-clock duration or final P4+M2 throughput.", "cycles")
    fact("SRNN_OLD_OFFICIAL64", "final_physical_path", "NO_OLD_WAVE3_INFERENCE_NO_M2",
         "tb/tb_unified_snn_core_final_v1_srnn_dataset.sv", "unified_snn_core_final_v1 #(")

    projection = PAPER + "performance/current_workload_projection_20260926/summary.json"
    json_fields("PROJECTION_CAVEAT", projection, ["status", "clock_period_ns", "clock_source"])
    add("PROJECTION_CAVEAT", "reuse_decision", "Do not transfer old-harness cycles to final V2 by multiplying by 15 ns",
        projection + "#/workloads/fc_learning_on_b16; " + projection + "#/workloads/srnn_wout_learning_on_b2t2",
        "Clock projection is arithmetic, not proof of source/top/shape/mask matching RUN120.")

    # Absence is explicit: combined transactions do not yield disjoint phases.
    for eid in ("FC_COMPAT", "SRNN_OLD_B2T2", "SRNN_OLD_TWO_WINDOWS", "FINAL_SHARED_DIRECTED"):
        for field in ("isolated_initial_forward_cycles", "isolated_learning_cycles",
                      "isolated_publication_cycles", "parameter_load_cycles",
                      "final_v2_samples_per_second", "peak_process_tree_rss_bytes"):
            add(eid, field, "NA", note="Not established by this evidence; no extrapolation or new execution.")

    entries = [
        ("NEXT_FINAL_B1", FINALRUN, FINALTB, "selected integration starting point; not ready as an unchanged real-data B1 runner",
         "YES: regenerates DUT and TB", "NO: existing directed TB has constants",
         "-j 0; broad rtl source list; peak RSS NA; TB scalar store 64 entries but other default capacities remain",
         "Do not run original runner. Future scoped runner must bypass both generators, use existing RTL, explicit new output/build dirs and -j1. Add only TB counters/fixture driver; do not rerun unchanged historical PASS.",
         "generate_unified_h40_final_composition_top_v1.py"),
        ("NEXT_SRNN_PHYSICAL", "scripts/run_verilator_srnn_m2_h40_forward_physical_array_v1.ps1",
         "tb/tb_unified_srnn_m2_h40_forward_physical_array_v1.sv", "optional smaller physical forward fixture-adapter base; not full transaction",
         "NO", "NO: constants, B1, 2->4->2", "-j 0; PARAM_DEPTH=128; MEMORY_DEPTH=1024; RSS NA",
         "Already has numeric assertions on actual P4 array; needs production fixture and learning integration for B1. Not selected to repeat unchanged.",
         "--binary -j 0"),
        ("REUSE_FRONTEND", "scripts/run_verilator_srnn_m2_tp_ce_contribution_frontend_formal_shape_wout_only.ps1",
         front_tb, "reuse existing correct component result; no rerun", "NO", "NO: generated by TB formulas",
         "No -j override in runner; log one build thread; process-tree RSS NA",
         "B64/H256/C20 Wout contribution parity only, no array/RMW/commit.", "-GENABLE_WH1_LEARNING=0"),
        ("REUSE_BACKEND", "scripts/run_verilator_srnn_m2_wout_b64_two_window_store.ps1",
         back_tb, "reuse existing correct component result; no rerun", "NO", "local tb/data .mem fixtures",
         "-j 4; 5120 parameter elements; no giant raw-dataset buffer; RSS NA",
         "Runner does not rebuild fixture. Separate build_srnn_m2_wout_two_window_fixture.py writes tb/data and results/learning_reference; do not invoke it here.",
         "--binary --timing --assert -Wall -Wno-fatal -j 4"),
        ("REJECT_FC_COMPAT_RUNNER", FCRUN, FCTB, "not a final-path B1 entry", "NO DUT generation; rebuilds weight fixture",
         "local imported N-MNIST event/label files and weight package",
         "BuildJobs default 4; 1024*100*49 20-bit event slots (5017600); file 35123200 bytes; RSS NA",
         "Online mode requires FCCompat and Samples=16; overwrites historical output and C:/msys64/tmp build; ReuseBuild is not version proof.",
         "OnlineLearning requires -FCCompat -Samples 16"),
        ("REJECT_OLD_SRNN_RUNNER", SRRUN, SRTB, "not final physical path or production M2 shape", "NO DUT generation",
         "local SHD stream/model .mem; synthetic M2 weights/targets",
         "--build-jobs 4; may launch dataset runner to create missing base sources.f; RSS NA",
         "Fixed historical output/build paths and stale base-list dependency. Do not use as final Wout-only latency.",
         "run_verilator_unified_snn_core_final_v1_srnn_dataset.ps1"),
    ]
    for eid, runner, tb, selection, gen, inputs, memory, action, needle in entries:
        assert (ROOT / tb).is_file()
        source = anchor(runner, needle)
        for field, value in (("runner", runner), ("testbench", tb), ("selection", selection),
                             ("generates_rtl_or_fixture", gen), ("input_dependency", inputs),
                             ("memory_and_parallelism", memory), ("next_action", action),
                             ("executed_in_this_audit", "NO")):
            add(eid, field, value, source + "; " + tb)

    for path in (
        "imports/fc/p3a_h160/data/eval_1024_t100/eval1024_events.hex",
        "imports/fc/p3a_h160/data/eval_1024_t100/eval1024_labels.hex",
        "data/shd/official_test/sample_0000.hex", "data/shd/official_test/sample_0001.hex",
        "models/srnn_shd_700_480_20_int8_layer_scaled_v2_padded/w0_int8.mem",
        "models/srnn_shd_700_480_20_int8_layer_scaled_v2_padded/r_int8.mem",
        "models/srnn_shd_700_480_20_int8_layer_scaled_v2_padded/w2_int8.mem",
    ):
        add("INPUT_INVENTORY", "existing_file_size", (ROOT / path).stat().st_size,
            path + " [filesystem size; content not loaded by audit]", unit="bytes")
    return rows


def main():
    rows = build_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "scope.csv"
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    with target.open(encoding="utf-8-sig", newline="") as handle:
        restored = list(csv.DictReader(handle))
    assert restored == rows
    assert all(row["value"] == "NA" for row in rows if row["field"] == "final_v2_samples_per_second")
    print(f"Wrote {target.relative_to(ROOT).as_posix()}: {len(rows)} field-level rows")
    print("Validated existing log pairs, JSON/CSV counts, source anchors and CSV round trip; no simulation/build/GPU/EDA invoked.")


if __name__ == "__main__":
    main()
