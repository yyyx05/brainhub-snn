#!/usr/bin/env python3
"""Adapt the existing P4 bank-format regression to the dual-context candidate."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("ambiguous original TB anchor: " + old)
    return text.replace(old, new, 1)


def main():
    source = ROOT.parents[1] / "tb/tb_unified_dual_role_weight_store_p4all_candidate_v1.sv"
    original = source.read_text(encoding="utf-8")
    text = once(original, "module tb_unified_dual_role_weight_store_p4all_candidate_v1;", "module tb_srnn_resident_store_modes;")
    text = once(text, "    logic fill_ready;", "    logic fill_ready;\n    logic fill_context = 1'b0;\n    logic fc_weight_int8 = 1'b0;\n    integer checked_terms = 0;")
    text = once(text, "    unified_dual_role_weight_store_p4all_candidate_v1 u_dut (", "    experimental_dual_context_weight_store_p4all_v1 u_dut (")
    text = once(text, "        .i_fill_valid(fill_valid), .o_fill_ready(fill_ready),", "        .i_fill_context(fill_context), .i_fc_weight_int8(fc_weight_int8),\n        .i_fill_valid(fill_valid), .o_fill_ready(fill_ready),")
    text = once(text, "                actual[term] = $signed", "                checked_terms++;\n                actual[term] = $signed")
    fc8 = '''        // INT8 retains 25 packed lanes; the other fifteen lanes must be zero.
        fc_weight_int8 = 1'b1;
        for (int bank = 0; bank < 4; bank++) begin
            row[bank] = '0;
            for (int lane = 0; lane < 25; lane++)
                row[bank][lane*8 +: 8] = 8'(bank*40 + lane - 100);
            fill_row(2'(bank), ADDR_WIDTH'(20 + bank), row[bank]);
        end
        issue_read(2'd0, 1'b0, 20, 21, 22, 23, 0, 0, 0, 0);
        for (int lane = 0; lane < LANES; lane++) begin
            if (lane < 25)
                expect_terms(lane, lane-100, lane-60, lane-20, lane+20);
            else
                expect_terms(lane, 0, 0, 0, 0);
        end
        @(posedge clk);
        fc_weight_int8 = 1'b0;

'''
    anchor = "        // SRNN: bank pairs encode source parity; lanes split 20/20."
    text = once(text, anchor, fc8+anchor)
    anchor = "        // M2 Q2.14: four banks at one row form one signed 40-lane vector."
    text = once(text, anchor, "        fill_context = 1'b1;\n"+anchor)
    anchor = '        $display("UNIFIED_DUAL_ROLE_WEIGHT_STORE_P4ALL_CANDIDATE_V1_PASS");'
    text = once(text, anchor, '''        @(posedge clk);
        fill_context = 1'b0;
        issue_read(2'd2, 1'b1, 60, 60, 60, 60, 2, 3, 4, 5);
        for (int lane = 0; lane < LANES; lane++)
            expect_terms(lane, -1002, -2003, -1004, -2005);
        $display("RESIDENT_STORE_MODES_PASS checks=%0d", checked_terms);''')
    text = once(text, "    always #5 clk = ~clk;", "    always #5 clk = ~clk;\n    initial begin\n        repeat (10000) @(posedge clk);\n        $fatal(1, \"store modes timeout\");\n    end")
    root = Path("\\\\?\\"+str(ROOT)) if os.name == "nt" else ROOT
    output = root / "verification/experiments/srnn_resident_weight_context/tb"
    (output / "weight_store_modes_original_reference.sv").write_text(original, encoding="utf-8")
    target = output / "tb_srnn_resident_store_modes.sv"
    target.write_text(text, encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
