`timescale 1ns/1ps
`default_nettype none

// Defect characterization only; not correctness acceptance or evidence of
// production reachability through the unchanged idle/owner barriers.
// Runner must link the unmodified Vivado 2022.2 vendor xpm_memory.sv.
module tb_xpm_hold;
    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    logic fill_valid = 0, fill_ready;
    logic [1:0] fill_bank = 0;
    logic [9:0] fill_addr = 0;
    logic [199:0] fill_data = 0;
    logic read_valid = 0, read_ready, active_pair = 0;
    logic [9:0] addr0 = 1, addr1 = 1, addr2 = 2, addr3 = 2;
    logic result_valid, result_ready = 0;
    logic [999:0] term0, term1, term2, term3;
    wire [3999:0] all_terms = {term3, term2, term1, term0};
    logic [3999:0] held_before, held_after;
    logic signed [15:0] expected0;
    logic expect_fill;
    int checks = 0, vulnerable_cases = 0, guarded_cases = 0;
    int coalesced_safe_cases = 0, target_row;
    string tag;

    // No parameter overrides: original 40-lane, 200-bit, 578-row candidate.
    scnn_same_row_weight_store_v1 dut (
        .i_clk(clk), .i_rst_n(rst_n),
        .i_fill_valid(fill_valid), .o_fill_ready(fill_ready),
        .i_fill_bank(fill_bank), .i_fill_addr(fill_addr), .i_fill_data(fill_data),
        .i_read_valid(read_valid), .o_read_ready(read_ready),
        .i_mode(2'd2), .i_fc_weight_int8(1'b0), .i_scnn_active_pair(active_pair),
        .i_read_addr0(addr0), .i_read_addr1(addr1),
        .i_read_addr2(addr2), .i_read_addr3(addr3),
        .i_read_subword0(4'd0), .i_read_subword1(4'd1),
        .i_read_subword2(4'd2), .i_read_subword3(4'd3),
        .o_read_result_valid(result_valid), .i_read_result_ready(result_ready),
        .o_term0_flat(term0), .o_term1_flat(term1),
        .o_term2_flat(term2), .o_term3_flat(term3)
    );

    function automatic logic signed [15:0] seed(input int bank, row);
        case (row)
            1: seed = bank == 0 ? 16'sd101 : -16'sd1101;
            2: seed = bank == 0 ? -16'sd202 : 16'sd1202;
            3: seed = bank == 0 ? 16'sd303 : -16'sd1303;
            4: seed = bank == 0 ? -16'sd404 : 16'sd1404;
            default: seed = 0;
        endcase
    endfunction

    function automatic logic [199:0] pack_row(input logic signed [15:0] value);
        pack_row = {8'b0, {12{value}}};
    endfunction

    function automatic logic [999:0] broadcast(input logic signed [15:0] value);
        broadcast = {40{{9{value[15]}}, value}};
    endfunction

    task automatic check(input logic condition, input string message);
        if (condition !== 1'b1)
            $fatal(1, "XPM_HOLD_UNEXPECTED t=%0t %s", $time, message);
        checks++;
    endtask

    task automatic check_terms(input string context_name,
        input logic signed [15:0] v0, v1, v2, v3);
        check(result_valid && !result_ready && !read_ready,
            {context_name, ": response must remain held"});
        check(term0 === broadcast(v0), {context_name, ": term0 all 40 lanes"});
        check(term1 === broadcast(v1), {context_name, ": term1 all 40 lanes"});
        check(term2 === broadcast(v2), {context_name, ": term2 all 40 lanes"});
        check(term3 === broadcast(v3), {context_name, ": term3 all 40 lanes"});
    endtask

    task automatic restart_and_refill;
        // Reset explicitly aborts any stalled request/fill and held response.
        // XPM memory is not reset; refill removes dependence on previous cases.
        @(negedge clk);
        rst_n = 0; read_valid = 0; fill_valid = 0; result_ready = 0;
        repeat (2) begin @(posedge clk); #1; end
        check(result_valid === 1'b0, "reset clears response valid");
        @(negedge clk);
        rst_n = 1; active_pair = 0;
        addr0 = 1; addr1 = 1; addr2 = 2; addr3 = 2;
        for (int bank = 0; bank < 2; bank++) begin
            for (int row = 1; row <= 4; row++) begin
                @(negedge clk);
                fill_valid = 1; fill_bank = 2'(bank); fill_addr = 10'(row);
                fill_data = pack_row(seed(bank, row));
                #1; check(fill_ready === 1'b1, "initialization fill accepted");
                @(posedge clk); #1;
                check(result_valid === 1'b0, "fill must not create a response");
                @(negedge clk); fill_valid = 0;
            end
        end
    endtask

    task automatic hold_read(input bit coalesced);
        @(negedge clk);
        active_pair = 0; addr0 = 1; addr1 = 1;
        addr2 = coalesced ? 10'd1 : 10'd2; addr3 = addr2;
        read_valid = 1;
        #1; check(read_ready === 1'b1, "initial read request accepted");
        @(posedge clk); #1;
        check_terms("initial response", seed(0, 1), seed(1, 1),
            seed(0, coalesced ? 1 : 2), seed(1, coalesced ? 1 : 2));
        @(negedge clk); read_valid = 0;
    endtask

    initial begin
`ifndef UNIFIED_USE_XPM_MEMORY
        $fatal(1, "XPM_HOLD_UNEXPECTED requires UNIFIED_USE_XPM_MEMORY");
`endif
        // A: no new request; fallback port A is clobbered by a row-3 fill.
        // B: same-pair coalesced request; held fallback guard blocks row 4.
        // C: pair-1 coalesced request; current bank mask permits bank-0 row 4.
        // D: held coalesced response uses B; bank-0 row-3 fill is harmless.
        for (int scenario = 0; scenario < 4; scenario++) begin
            tag = $sformatf("CASE_%s", scenario == 0 ? "A" :
                scenario == 1 ? "B" : scenario == 2 ? "C" : "D");
            restart_and_refill();
            hold_read(scenario == 3);
            held_before = all_terms;
            target_row = (scenario == 0 || scenario == 3) ? 3 : 4;
            expect_fill = scenario != 1;
            expected0 = (scenario == 0 || scenario == 2) ?
                seed(0, target_row) : seed(0, 1);

            @(negedge clk);
            if (scenario == 1 || scenario == 2) begin
                read_valid = 1; active_pair = scenario == 2;
                addr0 = 1; addr1 = 1; addr2 = 1; addr3 = 1;
            end
            fill_valid = 1; fill_bank = 0; fill_addr = 10'(target_row);
            fill_data = pack_row(16'sd7777);
            #1;
            check(read_ready === 1'b0, {tag, ": no new read may be consumed"});
            check(fill_ready === expect_fill, {tag, ": fill acceptance"});
            check(all_terms === held_before, {tag, ": stable before write edge"});
            @(posedge clk); #1;
            check_terms(tag, expected0, seed(1, 1),
                seed(0, scenario == 3 ? 1 : 2), seed(1, scenario == 3 ? 1 : 2));
            held_after = all_terms;
            $display("%s accepted=%0b old_row=%0d write_value=7777 before_term0=%0d after_term0=%0d before_term2=%0d after_term2=%0d",
                tag, expect_fill, seed(0, target_row),
                $signed(held_before[24:0]), $signed(term0[24:0]),
                $signed(held_before[2024:2000]), $signed(term2[24:0]));
            if (scenario == 0 || scenario == 2) begin
                check(term0 !== held_before[999:0], {tag, ": clobber must occur"});
                check(held_after[3999:1000] === held_before[3999:1000],
                    {tag, ": other three terms remain stable"});
            end else begin
                check(all_terms === held_before, {tag, ": full 4000-bit stability"});
            end

            @(negedge clk);
            if (expect_fill) fill_valid = 0;
            // B's blocked fill and B/C's blocked read remain stable until reset.
            repeat (2) begin
                @(posedge clk); #1;
                check_terms(tag, expected0, seed(1, 1),
                    seed(0, scenario == 3 ? 1 : 2), seed(1, scenario == 3 ? 1 : 2));
                check(all_terms === held_after, {tag, ": continued hold"});
                if (scenario == 1)
                    check(fill_ready === 1'b0, {tag, ": fill stays blocked"});
            end
            case (scenario)
                0, 2: vulnerable_cases++;
                1: guarded_cases++;
                3: coalesced_safe_cases++;
            endcase
        end
        @(negedge clk); rst_n = 0;
        @(posedge clk); #1;
        check(result_valid === 1'b0, "final reset aborts held response");
        check(vulnerable_cases == 2 && guarded_cases == 1 && coalesced_safe_cases == 1,
            "all four characterization outcomes observed");
        $display("XPM_HOLD_CHARACTERIZED vulnerable_cases=%0d guarded_cases=%0d coalesced_safe_cases=%0d checks=%0d",
            vulnerable_cases, guarded_cases, coalesced_safe_cases, checks);
        $finish;
    end

    initial begin
        #10000;
        $fatal(1, "XPM_HOLD_UNEXPECTED watchdog timeout");
    end
endmodule

`default_nettype wire
