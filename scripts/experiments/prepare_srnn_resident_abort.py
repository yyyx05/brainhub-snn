#!/usr/bin/env python3
"""Prepare one legal pre-learning abort/retry on the actual final composition."""
import argparse
import difflib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if os.name == "nt" and not str(ROOT).startswith("\\\\?\\"):
    ROOT = Path("\\\\?\\" + str(ROOT))
ACCEPTED = ROOT / "results/paper_evidence/supplement_20261015/performance_optimization/srnn_specialized_r1"
EXPERIMENT = ROOT / "verification/experiments/srnn_resident_weight_context"
FIXTURE = "results/paper_evidence/supplement_20261004/performance/current_source_measurements/srnn/fixture_sample0_t73"


def transform(source):
    text = source

    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError("abort TB anchor not unique: " + old)
        text = text.replace(old, new, 1)

    replace("    string raw, fixture, mode;", '''    string raw, fixture, mode;
    localparam int RECOVERY_STEPS = 1;
    integer attempt = 0, abort_seen = 0, reset_assertions = 0;
    bit reset_observed = 0;
    always @(posedge i_clk) begin
        if (!i_rst_n && !reset_observed) begin
            reset_assertions++;
            reset_observed = 1;
        end else if (i_rst_n) reset_observed = 0;
        if (i_rst_n && o_learning_srnn_timestep_aborted) abort_seen++;
    end''')
    replace("if (cycles > 367000000)", "if (cycles > 5000000)")
    replace('        initialized_cycle = cycles; profile_enter("boundary");', '''        for (attempt=0; attempt<2; attempt++) begin
            if (attempt != 0) begin
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
                i_learning_srnn_sequence_version=2;
            end
        initialized_cycle = cycles; profile_enter("boundary");''')
    replace("for (timestep=0; timestep<T; timestep++)", "for (timestep=0; timestep<RECOVERY_STEPS; timestep++)")
    replace("if (timestep == 0) refill(0);", "if (attempt == 0) refill(0);")
    replace("if (timestep == 0) refill(1);", "if (attempt == 0) refill(1);")
    replace("            i_srnn_phase_start_valid = 1;", '''            if (attempt == 0) i_learning_srnn_abort_timestep = 1;
            i_srnn_phase_start_valid = 1;''')
    replace("            while(done_count!=timestep+1) @(negedge i_clk);", '''            if (attempt == 0) begin
                do @(posedge i_clk); while (!o_learning_srnn_abort_ready);
                @(negedge i_clk); i_learning_srnn_abort_timestep=0;
                while (abort_seen != 1) @(negedge i_clk);
                if (bone_seen != H || hidden_seen != 0 || logit_seen != 0 || gradient_seen != 0 ||
                    done_count != 0 || commits != 0 || detaches != 0 || code_writes != 0 || residual_writes != 0)
                    $fatal(1,"abort boundary emitted unexpected numeric/update traffic");
                for (int a=0; a<120320; a++)
                    if (u_store.code_mem[a] !== model_mem[a] || u_store.residual_mem[a] !== 16'sd0)
                        $fatal(1,"aborted preparation changed parameter a=%0d",a);
                @(negedge i_clk); i_srnn_phase_abort=1;
                @(negedge i_clk); i_srnn_phase_abort=0;
                while (!o_srnn_phase_idle || o_learning_srnn_busy || !i_store_quiescent)
                    @(negedge i_clk);
                if (o_learning_srnn_recovery_required) $fatal(1,"abort unexpectedly requires parameter recovery");
                $display("RESIDENT_ABORT_ACCEPTED learning=%0d aborts=%0d bone=%0d hidden=%0d logits=%0d gradients=%0d code=%0d residual=%0d commits=%0d unchanged_parameters=120320",
                    learning,abort_seen,bone_seen,hidden_seen,logit_seen,gradient_seen,code_writes,residual_writes,commits);
                $fflush();
                continue;
            end
            while(done_count!=timestep+1) @(negedge i_clk);''')
    coverage_start = text.index("        if(bone_seen!=T*H")
    coverage_end = text.index("        profile_finish();", coverage_start)
    old = text[coverage_start:coverage_end]
    new = old.replace("T*", "RECOVERY_STEPS*").replace("? T : 0", "? RECOVERY_STEPS : 0")
    new = new.replace("steps=73 learning=%0d", "steps=1 learning=%0d")
    new = new.replace("resident_wout_loads > T", "resident_wout_loads != 1")
    replace(old, '''        if (attempt == 0) begin
            profile_enter("aborted_boundary");
            continue;
        end
'''+new)
    replace("        profile_finish();\n        $finish;", '''        profile_finish();
        if (reset_assertions != 1 || abort_seen != 1 || profile_count[0] != 120320 || profile_count[2] != 120320)
            $fatal(1,"retry used reset/reload or missing abort");
        $display("RESIDENT_ABORT_RECOVERY_PASS learning=%0d resets=%0d aborts=%0d bone=%0d hidden=%0d logits=%0d gradients=%0d store=%0d code=%0d residual=%0d commits=%0d detaches=%0d backbone_loads=%0d wh1_loads=%0d wout_loads=%0d parameter_loads=%0d store_loads=%0d",
            learning,reset_assertions,abort_seen,bone_seen,hidden_seen,logit_seen,gradient_seen,store_checks,
            code_writes,residual_writes,commits,detaches,resident_backbone_loads,resident_wh1_loads,resident_wout_loads,
            profile_count[0],profile_count[2]);
        end
        $finish;''')
    replace("            T,learning,initialized_cycle,cycles,store_checks);", "            RECOVERY_STEPS,learning,initialized_cycle,cycles,store_checks);")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="abort_before_learning_r1")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        parser.error("simple preparation name required")
    source = (ACCEPTED / "tb_srnn_profile_measurement.sv").read_text(encoding="utf-8")
    text = transform(source)
    original = json.loads((ACCEPTED / FIXTURE / "reference.json").read_text(encoding="utf-8"))
    expected = {}
    for arm in ("off", "on"):
        learning = int(arm == "on")
        traffic = original["reference_counters"][arm][0]
        expected[arm] = dict(learning=learning, resets=1, aborts=1, bone=450, hidden=256, logits=20,
            gradients=5120 if learning else 0, store=120320, code=traffic["code_writes"],
            residual=traffic["residual_writes"], commits=learning, detaches=learning,
            backbone_loads=2, wh1_loads=1, wout_loads=1, parameter_loads=120320, store_loads=120320)
    output = EXPERIMENT / args.name
    output.mkdir(parents=True, exist_ok=False)
    (output / "tb_srnn_profile_measurement.sv").write_text(text, encoding="utf-8")
    (output / "testbench_changes.diff").write_text("".join(difflib.unified_diff(source.splitlines(True), text.splitlines(True),
        fromfile="accepted/srnn_specialized_r1_tb", tofile="abort_before_learning/tb")), encoding="utf-8")
    report = dict(status="PREPARED_NOT_SIMULATED", expected=expected, fixture_source="srnn_specialized_r1/"+FIXTURE,
        scope="One first native sample0 timestep retried after a legal START_LEARNING abort. No global reset or parameter reload. Not mid-update rollback, arbitrary-cycle abort, a complete sample or throughput measurement.")
    (output / "reference.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
