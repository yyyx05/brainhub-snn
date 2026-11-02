#!/usr/bin/env python3
"""Prepare a trace-bitmap candidate on the accepted elastic scanner."""
import argparse
import difflib
import json
from pathlib import Path

from prepare_fc_contribution_pipeline import ROOT, SOURCE, EXPERIMENT, replace_once


def candidate(source):
    text = replace_once(source, "    logic read_available;", """    logic read_available;
    // One bit per observed trace; sample banks have independent write enables.
    logic trace_nonzero_mem [0:MAX_BATCH-1][0:MAX_HIDDEN-1];
    logic [MAX_BATCH-1:0] remaining_samples_q;
    logic mask_valid_q;
    logic sparse_mode;
    logic mask_last;
    logic address_advance;
    logic [BATCH_INDEX_WIDTH-1:0] selected_sample;

    for (genvar sample_bank = 0; sample_bank < MAX_BATCH; sample_bank++) begin : g_trace_mask
        always_ff @(posedge i_clk) begin
            if (i_rst_n && hidden_fire && hidden_input_matches &&
                (i_hidden_trace_sample_id == BATCH_INDEX_WIDTH'(sample_bank)))
                trace_nonzero_mem[sample_bank][i_hidden_trace_hidden_index] <=
                    (i_hidden_trace_q016 != '0);
        end
    end""")
    text = replace_once(text, "    logic emit_stream_last;\n", "")
    text = replace_once(text, """        emit_advance = (state_q == STATE_STREAM) &&
            !scan_exhausted_q && read_available;""", """        sparse_mode = batch_count_q > BATCH_COUNT_WIDTH'(2);
        selected_sample = emit_sample_q;
        if (sparse_mode) begin
            selected_sample = '0;
            for (int s = MAX_BATCH-1; s >= 0; s--)
                if (remaining_samples_q[s]) selected_sample = BATCH_INDEX_WIDTH'(s);
        end
        mask_last = (remaining_samples_q &
            (remaining_samples_q - MAX_BATCH'(1))) == '0;
        emit_advance = (state_q == STATE_STREAM) &&
            !scan_exhausted_q && read_available &&
            (!sparse_mode || (mask_valid_q && (remaining_samples_q != '0)));""")
    text = replace_once(text, "        emit_stream_last = emit_sample_last && emit_address_last;", """        address_advance = (state_q == STATE_STREAM) && !scan_exhausted_q &&
            (sparse_mode ? (mask_valid_q && ((remaining_samples_q == '0) ||
                (emit_advance && mask_last))) : (emit_advance && emit_sample_last));""")
    text = replace_once(text, "trace_address(emit_sample_q, emit_hidden_q)",
                        "trace_address(selected_sample, emit_hidden_q)")
    text = replace_once(text, "error_address(emit_sample_q, emit_output_q)",
                        "error_address(selected_sample, emit_output_q)")
    text = replace_once(text, "            read_local_addr_q <= '0;", """            read_local_addr_q <= '0;
            remaining_samples_q <= '0;
            mask_valid_q <= 1'b0;""")
    text = replace_once(text, """                        scan_exhausted_q <= 1'b0;
                        contribution_valid_q <= 1'b0;""", """                        scan_exhausted_q <= 1'b0;
                        mask_valid_q <= 1'b0;
                        remaining_samples_q <= '0;
                        contribution_valid_q <= 1'b0;""")
    text = replace_once(text, """                        read_valid_q <= !scan_exhausted_q;
                        if (!scan_exhausted_q) begin""", """                        read_valid_q <= emit_advance;
                        if (emit_advance) begin""")
    first = text.index("            // Advance on synchronous read issue")
    last = text.index("\n        end\n    end", first)
    text = text[:first] + """            if ((state_q == STATE_STREAM) && sparse_mode && !scan_exhausted_q) begin
                // All active bits are overwritten before seal; inactive banks are masked.
                if (!mask_valid_q) begin
                    for (int s = 0; s < MAX_BATCH; s++)
                        remaining_samples_q[s] <= (s < int'(batch_count_q)) &&
                            trace_nonzero_mem[s][emit_hidden_q];
                    mask_valid_q <= 1'b1;
                end else if (address_advance) begin
                    mask_valid_q <= 1'b0;
                end else if (emit_advance) begin
                    remaining_samples_q[selected_sample] <= 1'b0;
                end
            end
            if (address_advance) begin
                emit_sample_q <= '0;
                if (emit_address_last) begin
                    scan_exhausted_q <= 1'b1;
                end else if (output_major_q) begin
                    if (emit_hidden_last) begin
                        emit_hidden_q <= '0;
                        emit_output_q <= emit_output_q + 1'b1;
                    end else begin
                        emit_hidden_q <= emit_hidden_q + 1'b1;
                    end
                end else begin
                    if (emit_output_last) begin
                        emit_output_q <= '0;
                        emit_hidden_q <= emit_hidden_q + 1'b1;
                    end else begin
                        emit_output_q <= emit_output_q + 1'b1;
                    end
                end
            end else if (emit_advance && !sparse_mode) begin
                emit_sample_q <= emit_sample_q + 1'b1;
            end""" + text[last:]
    return text


def trace_profile():
    fixture = (ROOT / "results/paper_evidence/supplement_20261015/performance_optimization/"
               "fc_scanner_pipeline_r1/results/paper_evidence/supplement_20261004/"
               "algorithm/fc_final_counts_reference/t99_on/trace.mem")
    path = Path("\\\\?\\" + str(fixture)) if __import__("os").name == "nt" else fixture
    trace = [int(word, 16) for word in path.read_text(encoding="ascii").split()]
    assert len(trace) == 99 * 16 * 160
    groups = [sum(trace[(t * 16 + s) * 160 + h] != 0 for s in range(16))
              for t in range(99) for h in range(160)]
    return {
        "source": fixture.relative_to(ROOT).as_posix(),
        "scope": "Accepted FC B16T99 learning-on trace fixture, not measured new RTL cycles",
        "trace_conversion": "17-bit Q016 saturation to ffff preserves the zero predicate",
        "traces": len(trace), "nonzero_traces": sum(groups),
        "trace_zero_fraction": 1 - sum(groups) / len(trace),
        "dense_candidate_reads": len(trace) * 10,
        "trace_bitmap_candidate_reads": sum(groups) * 10,
        "mask_load_cycles_model": len(groups) * 10,
        "empty_mask_cycles_model": groups.count(0) * 10,
        "unstalled_issue_plus_mask_cycles_model": sum(1 + max(1, n) for n in groups) * 10,
        "limits": ["Excludes error filtering, downstream stalls, pipeline drain, and commit",
                   "Bitmap overhead can regress dense batches; B1/B2 retain the dense path",
                   "Storage is MAX_BATCH*MAX_HIDDEN bits; mapped LUT/FF/BRAM not yet measured"],
        "nonzero_per_window": [sum(groups[t * 160:(t + 1) * 160]) for t in range(99)],
        "active_samples_per_hidden_histogram": {str(n): groups.count(n) for n in range(17)},
    }


def prepare(name):
    target = ROOT / EXPERIMENT / name
    target.mkdir(parents=True, exist_ok=False)
    base = ROOT / EXPERIMENT / "candidate_r1"
    source = (base / SOURCE).read_text(encoding="utf-8-sig")
    changed = candidate(source)
    output = target / SOURCE
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(changed, encoding="utf-8")
    (target / "reference_fc_contribution_scheduler.sv").write_bytes(
        (base / "reference_fc_contribution_scheduler.sv").read_bytes())
    (target / "source_changes.diff").write_text("".join(difflib.unified_diff(
        source.splitlines(True), changed.splitlines(True),
        fromfile="candidate_r1/" + SOURCE.as_posix(),
        tofile=name + "/" + SOURCE.as_posix())), encoding="utf-8")
    (target / "trace_sparsity.json").write_text(json.dumps(trace_profile(), indent=2) + "\n",
                                               encoding="utf-8")
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="candidate_sparse_r2")
    prepare(parser.parse_args().name)
