#!/usr/bin/env python3
"""Prepare an isolated, elastic FC/SCNN contribution-scanner candidate."""
import argparse
import difflib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("rtl/learning/unified_fc_tp_w2_contribution_scheduler.sv")
EXPERIMENT = Path("verification/experiments/fc_contribution_pipeline")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"expected one source anchor: {old[:80]!r}")
    return text.replace(old, new, 1)


def candidate(source):
    text = replace_once(source,
        "// paired with output-error tokens only when the trace is nonzero, so the\n"
        "// existing outer-product accumulator receives sparse, ordered contributions.",
        "// paired with synchronous output-error reads in an elastic pipeline.\n"
        "// Zero terms are filtered without changing address-major contribution order.")
    first = text.index("    typedef enum logic [2:0]")
    last = text.index("    } state_t;", first) + len("    } state_t;")
    text = text[:first] + """    typedef enum logic [1:0] {
        STATE_IDLE, STATE_OBSERVE, STATE_SEALED, STATE_STREAM
    } state_t;""" + text[last:]
    text = replace_once(text, "    logic emit_advance;", """    logic emit_advance;
    logic read_valid_q;
    logic scan_exhausted_q;
    logic [PARAM_ADDR_WIDTH-1:0] read_local_addr_q;
    logic read_nonzero;
    logic output_available;
    logic read_available;""")
    first = text.index("        emit_advance = ((state_q == STATE_TRACE_CHECK)")
    last = text.index("        if (output_major_q)", first)
    text = text[:first] + """        read_nonzero = (emit_trace_q != '0) && (emit_error_q != '0);
        output_available = !contribution_valid_q || i_contribution_ready;
        read_available = !read_valid_q || !read_nonzero || output_available;
        emit_advance = (state_q == STATE_STREAM) &&
            !scan_exhausted_q && read_available;
""" + text[last:]
    text = replace_once(text, "            emit_error_q <= '0;", """            emit_error_q <= '0;
            read_valid_q <= 1'b0;
            scan_exhausted_q <= 1'b0;
            read_local_addr_q <= '0;""")
    # The original iterator has the same state assignment; target the sealed arm.
    text = replace_once(text, """                        emit_output_q <= '0;
                        state_q <= STATE_TRACE_READ;""", """                        emit_output_q <= '0;
                        read_valid_q <= 1'b0;
                        scan_exhausted_q <= 1'b0;
                        contribution_valid_q <= 1'b0;
                        state_q <= STATE_STREAM;""")
    first = text.index("                STATE_TRACE_READ: begin")
    last = text.index("                default: state_q <= STATE_IDLE;", first)
    text = text[:first] + """                STATE_STREAM: begin
                    // Each stage holds its payload while the downstream stage stalls.
                    if (output_available) begin
                        contribution_valid_q <= read_valid_q && read_nonzero;
                        if (read_valid_q && read_nonzero) begin
                            contribution_local_addr_q <= read_local_addr_q;
                            contribution_source_trace_q <= emit_trace_q;
                            contribution_error_q016_q <= emit_error_q;
                        end
                    end
                    if (read_available) begin
                        read_valid_q <= !scan_exhausted_q;
                        if (!scan_exhausted_q) begin
                            emit_trace_q <= trace_mem[emit_trace_addr];
                            emit_error_q <= error_mem[emit_error_addr];
                            read_local_addr_q <= contribution_local_addr_next;
                        end
                    end
                    if (scan_exhausted_q && !read_valid_q && !contribution_valid_q) begin
                        hidden_sample_q <= '0;
                        hidden_index_q <= '0;
                        error_sample_q <= '0;
                        error_index_q <= '0;
                        hidden_complete_q <= 1'b0;
                        error_complete_q <= 1'b0;
                        o_emit_done <= 1'b1;
                        state_q <= STATE_OBSERVE;
                    end
                end

""" + text[last:]
    first = text.index("            // Advance the address-major iterator")
    last = text.index("                    end else if (!emit_sample_last)", first)
    text = text[:first] + """            // Advance on synchronous read issue, not on downstream acceptance.
            if (emit_advance) begin
                    if (emit_stream_last) begin
                        scan_exhausted_q <= 1'b1;
""" + text[last:]
    text = text.replace("                        state_q <= STATE_TRACE_READ;\n", "")
    return text


def prepare(name):
    target = ROOT / EXPERIMENT / name
    target.mkdir(parents=True, exist_ok=False)
    source = (ROOT / SOURCE).read_text(encoding="utf-8-sig")
    changed = candidate(source)
    output = target / SOURCE
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(changed, encoding="utf-8")
    original = source.replace("module unified_fc_tp_w2_contribution_scheduler #(",
                              "module reference_fc_contribution_scheduler #(", 1)
    (target / "reference_fc_contribution_scheduler.sv").write_text(original, encoding="utf-8")
    (target / "source_changes.diff").write_text("".join(difflib.unified_diff(
        source.splitlines(True), changed.splitlines(True),
        fromfile="production/" + SOURCE.as_posix(), tofile="candidate/" + SOURCE.as_posix())),
        encoding="utf-8")
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="candidate_r1")
    prepare(parser.parse_args().name)
