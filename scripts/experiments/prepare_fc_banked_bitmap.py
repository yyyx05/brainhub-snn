#!/usr/bin/env python3
"""Make each trace bitmap bank an independently inferable one-dimensional RAM."""
import argparse
import difflib
from pathlib import Path

from prepare_fc_contribution_pipeline import ROOT, SOURCE, EXPERIMENT, replace_once


def candidate(source):
    text = replace_once(source,
        "    logic trace_nonzero_mem [0:MAX_BATCH-1][0:MAX_HIDDEN-1];",
        "    wire [MAX_BATCH-1:0] trace_nonzero_read;")
    text = replace_once(text,
        "begin : g_trace_mask\n        always_ff",
        """begin : g_trace_mask
        (* ram_style = "distributed" *) logic nonzero_mem [0:MAX_HIDDEN-1];
        assign trace_nonzero_read[sample_bank] = nonzero_mem[emit_hidden_q];
        always_ff""")
    text = replace_once(text,
        "trace_nonzero_mem[sample_bank][i_hidden_trace_hidden_index]",
        "nonzero_mem[i_hidden_trace_hidden_index]")
    return replace_once(text, "trace_nonzero_mem[s][emit_hidden_q]", "trace_nonzero_read[s]")


def testbench(source):
    text = replace_once(source, "B = 16, H = 160, O = 10", "B = 16, H = 2048, O = 10")
    text = replace_once(text, "logic [7:0] hidden_count", "logic [11:0] hidden_count")
    text = replace_once(text, "logic [7:0] trace_index", "logic [10:0] trace_index")
    text = replace_once(text, "hidden_count = 8'(nh)", "hidden_count = 12'(nh)")
    text = replace_once(text, "trace_index = 8'(h)", "trace_index = 11'(h)")
    text = replace_once(text, "(cycles + 7*d) % 29", "cycles % 29")
    text = replace_once(text,
        'if (!stalls && nb*nh*no >= 100 && finished[1] >= finished[0]) $fatal(1, "no scan improvement");',
        'if (finished[1] != finished[0]) $fatal(1, "bitmap layout changed cycles");')
    text = replace_once(text, "run_case(16, 160, 10, 0, 0, 0, 0)",
                        "run_case(16, 2048, 10, 0, 0, 0, 0)")
    text = replace_once(text, "run_case(16, 160, 10, 1, 0, 2, 0)",
                        "run_case(16, 2048, 10, 1, 0, 2, 0)")
    return replace_once(text, "    task automatic configure", """    // Same inputs/backpressure: every public control and valid payload must match.
    always @(negedge clk) begin
        if (rst_n) begin
            if ({cfg_ready[0], cfg_error[0], active[0], release_ready[0],
                 trace_ready[0], trace_error[0], error_ready[0], error_error[0],
                 observation_ready[0], emit_ready[0], valid[0], done[0], busy[0]} !==
                {cfg_ready[1], cfg_error[1], active[1], release_ready[1],
                 trace_ready[1], trace_error[1], error_ready[1], error_error[1],
                 observation_ready[1], emit_ready[1], valid[1], done[1], busy[1]})
                $fatal(1, "bitmap layout control cycle mismatch");
            if (valid[0] && {address[0], value_trace[0], value_error[0]} !==
                           {address[1], value_trace[1], value_error[1]})
                $fatal(1, "bitmap layout payload cycle mismatch");
        end
    end

    task automatic configure""")


def prepare(name):
    base = ROOT / EXPERIMENT / "candidate_sparse_r2"
    target = ROOT / EXPERIMENT / name
    target.mkdir(parents=True, exist_ok=False)
    original = (base / SOURCE).read_text(encoding="utf-8-sig")
    changed = candidate(original)
    output = target / SOURCE
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(changed, encoding="utf-8")
    (target / "reference_fc_contribution_scheduler.sv").write_text(replace_once(
        original, "module unified_fc_tp_w2_contribution_scheduler #(",
        "module reference_fc_contribution_scheduler #("), encoding="utf-8")
    tb = (ROOT / EXPERIMENT / "tb_fc_contribution_pipeline.sv").read_text(encoding="utf-8")
    (target / "tb_fc_contribution_pipeline.sv").write_text(testbench(tb), encoding="utf-8")
    (target / "source_changes.diff").write_text("".join(difflib.unified_diff(
        original.splitlines(True), changed.splitlines(True),
        fromfile="candidate_sparse_r2/" + SOURCE.as_posix(),
        tofile=name + "/" + SOURCE.as_posix())), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="candidate_banked_r3")
    prepare(parser.parse_args().name)
