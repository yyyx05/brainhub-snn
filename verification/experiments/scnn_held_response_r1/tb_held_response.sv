`timescale 1ns/1ps
`default_nettype none

module tb_held_response;
    localparam int FC = 0, SRNN = 1, SCNN = 2, M2 = 3;
    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    logic read_valid = 0, result_ready = 0, fc_int8 = 0, pair_sel = 0;
    logic [1:0] mode = 0;
    logic [9:0] addr [0:3];
    logic [3:0] subword [0:3];
    logic fill_valid = 0;
    logic [1:0] fill_bank = 0;
    logic [9:0] fill_addr = 0;
    logic [199:0] fill_data = 0;
    wire read_ready, fill_ready, result_valid;
    wire [999:0] term [0:3];
    wire [3999:0] all_terms = {term[3], term[2], term[1], term[0]};
    wire [59:0] read_payload = {mode, fc_int8, pair_sel,
        addr[0], addr[1], addr[2], addr[3],
        subword[0], subword[1], subword[2], subword[3]};
    wire [211:0] fill_payload = {fill_bank, fill_addr, fill_data};

    // No overrides: both runner arms use 40 lanes and four 200 x 578 banks.
    scnn_held_response_weight_store_v1 dut (
        .i_clk(clk), .i_rst_n(rst_n),
        .i_fill_valid(fill_valid), .o_fill_ready(fill_ready),
        .i_fill_bank(fill_bank), .i_fill_addr(fill_addr), .i_fill_data(fill_data),
        .i_read_valid(read_valid), .o_read_ready(read_ready),
        .i_mode(mode), .i_fc_weight_int8(fc_int8), .i_scnn_active_pair(pair_sel),
        .i_read_addr0(addr[0]), .i_read_addr1(addr[1]),
        .i_read_addr2(addr[2]), .i_read_addr3(addr[3]),
        .i_read_subword0(subword[0]), .i_read_subword1(subword[1]),
        .i_read_subword2(subword[2]), .i_read_subword3(subword[3]),
        .o_read_result_valid(result_valid), .i_read_result_ready(result_ready),
        .o_term0_flat(term[0]), .o_term1_flat(term[1]),
        .o_term2_flat(term[2]), .o_term3_flat(term[3])
    );

    logic [199:0] golden [0:3][0:577];
    bit initialized [0:3][0:577];
    logic [3999:0] expected = '0;
    bit expected_valid = 0, accepted_pair = 0, accepted_coalesced = 0;
    int accepted_mode = FC;
    bit stalled_read = 0, stalled_fill = 0, previous_reset_n = 1;
    logic [59:0] saved_read;
    logic [211:0] saved_fill;
    int checks = 0, reads = 0, writes = 0, blocked = 0;
    int held_cycles = 0, simultaneous = 0, resets = 0;
    int read_first_checks = 0, discarded = 0, cases = 0;
    bit sign_edges [0:2][0:1];
    bit boundary [0:3][0:1];
    int start_r, start_w, start_b, start_h, start_s, start_z;
    string case_name;

    task automatic check(input logic ok, input string message);
        checks++;
        if (ok !== 1'b1)
            $fatal(1, "HELD_RESPONSE_FAIL case=%s t=%0t %s", case_name, $time, message);
    endtask

    function automatic logic [199:0] pattern(input int bank, row, salt);
        logic [199:0] value;
        for (int i = 0; i < 25; i++)
            value[i*8 +: 8] = 8'(bank*53 + row*17 + salt*29 + i*11);
        if (row == 0) begin
            value[0 +: 5] = 5'h10;
            value[5 +: 5] = 5'h0f;
        end else if (row == 1) begin
            value[0 +: 8] = 8'h80;
            value[8 +: 8] = 8'h7f;
        end else if (row == 2 || row == 577) begin
            value[0 +: 16] = 16'h8000;
            value[16 +: 16] = 16'h7fff;
        end
        return value;
    endfunction

    // Logical term/lane addressing only; no DUT port data enters the oracle.
    function automatic logic [24:0] golden_lane(input int t, lane);
        int bank, row, offset, bits, raw;
        logic [199:0] word_bits;
        row = int'(addr[t]);
        bank = 0;
        offset = 0;
        bits = 16;
        case (int'(mode))
            FC: begin
                if (fc_int8 && lane >= 25) return 25'd0;
                bank = t;
                bits = fc_int8 ? 8 : 5;
                offset = lane * bits;
            end
            SRNN: begin
                bank = (t % 2)*2 + lane/20;
                bits = 8;
                offset = (lane % 20)*8;
            end
            SCNN: begin
                if (subword[t] >= 12) return 25'd0;
                bank = int'(pair_sel)*2 + t%2;
                offset = int'(subword[t])*16;
            end
            M2: begin
                if (t != 0) return 25'd0;
                bank = lane/12;
                row = int'(addr[0]);
                offset = (lane%12)*16;
            end
        endcase
        if (row >= 578 || !initialized[bank][row])
            $fatal(1, "HELD_RESPONSE_FAIL uninitialized oracle bank=%0d row=%0d", bank, row);
        if (row == 0) boundary[bank][0] = 1;
        if (row == 577) boundary[bank][1] = 1;
        word_bits = golden[bank][row];
        case (bits)
            5: raw = int'(word_bits[offset +: 5]);
            8: raw = int'(word_bits[offset +: 8]);
            default: raw = int'(word_bits[offset +: 16]);
        endcase
        if (raw >= (1 << (bits-1))) raw -= 1 << bits;
        if (raw == -(1 << (bits-1))) sign_edges[bits == 5 ? 0 : bits == 8 ? 1 : 2][0] = 1;
        if (raw == (1 << (bits-1))-1) sign_edges[bits == 5 ? 0 : bits == 8 ? 1 : 2][1] = 1;
        return 25'(raw);
    endfunction

    task automatic check_output;
        check(result_valid === expected_valid, "response valid/latency");
        if (expected_valid)
            check(all_terms === expected, "all 4000 output bits match accepted request");
    endtask

    // Called only with inputs driven at a falling edge; returns at the next one.
    task automatic tick(input bit want_read, want_write);
        bit rf, wf, coalesced, protected_bank, current_conflict, probe_a;
        logic [199:0] prewrite;
        int written_bank;
        @(posedge clk);
        probe_a = 0;
        if (!rst_n) begin
            check(!read_valid && !fill_valid, "requests idle during explicit reset");
            if (previous_reset_n) begin
                resets++;
                if (expected_valid) discarded++;
            end
            expected_valid = 0;
            stalled_read = 0;
            stalled_fill = 0;
            check(!want_read && !want_write, "no reset transfers expected");
        end else begin
            check_output();
            if (stalled_read)
                check(read_valid && read_payload === saved_read, "stalled read remains valid and stable");
            if (stalled_fill)
                check(fill_valid && fill_payload === saved_fill, "stalled fill remains valid and stable");
            check(read_ready === (!expected_valid || result_ready), "read readiness");
            coalesced = mode == SCNN && addr[0] == addr[2] && addr[1] == addr[3];
            protected_bank = expected_valid && !result_ready &&
                (accepted_mode != SCNN || (!accepted_coalesced &&
                    int'(fill_bank)/2 == int'(accepted_pair)));
            current_conflict = read_valid &&
                (mode != SCNN || int'(fill_bank)/2 == int'(pair_sel)) &&
                (!coalesced || fill_addr == addr[int'(fill_bank)%2]);
            check(fill_ready === !(protected_bank || current_conflict), "fill readiness from accepted and current requests");
            rf = read_valid && read_ready;
            wf = fill_valid && fill_ready;
            check(rf === want_read && wf === want_write, "directed transfer expectation");
            stalled_read = read_valid && !read_ready;
            stalled_fill = fill_valid && !fill_ready;
            saved_read = read_payload;
            saved_fill = fill_payload;
            if (stalled_fill) blocked++;
            if (expected_valid && !result_ready) held_cycles++;
            if (rf && wf) simultaneous++;
            if (wf) begin
                check(fill_addr < 578, "fill address in range");
                written_bank = int'(fill_bank);
                probe_a = !expected_valid && !rf && initialized[written_bank][int'(fill_addr)];
                prewrite = golden[written_bank][int'(fill_addr)];
            end
            // Sample the complete read before applying ANY same-edge write.
            if (rf) begin
                for (int t = 0; t < 4; t++)
                    for (int lane = 0; lane < 40; lane++)
                        expected[t*1000 + lane*25 +: 25] = golden_lane(t, lane);
                expected_valid = 1;
                accepted_mode = int'(mode);
                accepted_pair = pair_sel;
                accepted_coalesced = coalesced;
                reads++;
            end else if (result_ready) expected_valid = 0;
            if (wf) begin
                golden[written_bank][int'(fill_addr)] = fill_data;
                initialized[written_bank][int'(fill_addr)] = 1;
                writes++;
            end
        end
        previous_reset_n = rst_n;
        #1;
        check_output();
        if (probe_a) begin
            check(dut.bank_read_data[written_bank][0] === prewrite,
                "idle A READ_FIRST output is the initialized pre-write row");
            read_first_checks++;
        end
        @(negedge clk);
    endtask

    task automatic restart;
        rst_n = 0;
        read_valid = 0;
        fill_valid = 0;
        result_ready = 0;
        tick(0, 0);
        tick(0, 0);
        rst_n = 1;
        tick(0, 0);
    endtask

    task automatic set_read(input int m, input bit eight, p,
        input int a0, a1, a2, a3, s0, s1, s2, s3);
        read_valid = 1;
        mode = 2'(m); fc_int8 = eight; pair_sel = p;
        addr[0] = 10'(a0); addr[1] = 10'(a1);
        addr[2] = 10'(a2); addr[3] = 10'(a3);
        subword[0] = 4'(s0); subword[1] = 4'(s1);
        subword[2] = 4'(s2); subword[3] = 4'(s3);
    endtask

    task automatic set_fill(input int bank, row, salt);
        fill_valid = 1;
        fill_bank = 2'(bank); fill_addr = 10'(row);
        fill_data = pattern(bank, row, salt);
    endtask

    task automatic consume;
        read_valid = 0; fill_valid = 0; result_ready = 1;
        tick(0, 0);
        result_ready = 0;
    endtask

    task automatic begin_case(input string name);
        case_name = name;
        start_r = reads; start_w = writes; start_b = blocked;
        start_h = held_cycles; start_s = simultaneous; start_z = resets;
    endtask

    task automatic end_case(input int r, w, b, h, s, z);
        check(reads-start_r == r && writes-start_w == w && blocked-start_b == b &&
            held_cycles-start_h == h && simultaneous-start_s == s && resets-start_z == z,
            $sformatf("counts got r/w/b/h/s/z=%0d/%0d/%0d/%0d/%0d/%0d expected=%0d/%0d/%0d/%0d/%0d/%0d",
                reads-start_r, writes-start_w, blocked-start_b, held_cycles-start_h,
                simultaneous-start_s, resets-start_z, r, w, b, h, s, z));
        check(!expected_valid && !stalled_read && !stalled_fill, "case drains or explicitly resets pending work");
        cases++;
        $display("CASE %s reads=%0d writes=%0d blocked=%0d held_cycles=%0d simultaneous=%0d resets=%0d",
            case_name, r, w, b, h, s, z);
    endtask

    task automatic packing_case(input string name, input int m, input bit eight, p,
        input int a0, a1, a2, a3, s0, s1, s2, s3);
        begin_case(name);
        set_read(m, eight, p, a0, a1, a2, a3, s0, s1, s2, s3);
        tick(1, 0);
        read_valid = 0;
        repeat (2) tick(0, 0);
        consume();
        end_case(1, 0, 0, 2, 0, 0);
    endtask

    initial begin : directed
        int row, before_probe, before_discard;
        for (int t = 0; t < 4; t++) begin addr[t] = 0; subword[t] = 0; end
        @(negedge clk);
        begin_case("initialize");
        restart();
        for (int b = 0; b < 4; b++) begin
            for (int r = 0; r < 6; r++) begin
                row = r == 5 ? 577 : r;
                set_fill(b, row, 0);
                tick(0, 1);
            end
        end
        fill_valid = 0;
        end_case(0, 24, 0, 0, 0, 1);

        begin_case("idle_read_first_all_banks");
        before_probe = read_first_checks;
        for (int b = 0; b < 4; b++) begin
            set_fill(b, 4, 11);
            check(fill_data !== golden[b][4], "READ_FIRST probe writes different data");
            tick(0, 1);
        end
        fill_valid = 0;
        check(read_first_checks-before_probe == 4, "four nonvacuous internal READ_FIRST probes");
        set_read(FC, 1, 0, 4, 4, 4, 4, 0, 0, 0, 0);
        tick(1, 0); read_valid = 0; tick(0, 0); consume();
        end_case(1, 4, 0, 1, 0, 0);

        packing_case("fc_int5_extrema", FC, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        packing_case("fc_int8_extrema_zero_tail", FC, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0);
        packing_case("srnn_extrema", SRNN, 0, 0, 1, 1, 577, 577, 0, 0, 0, 0);
        packing_case("m2_extrema_zero_terms", M2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0);
        packing_case("m2_last_row", M2, 0, 0, 577, 0, 0, 0, 0, 0, 0, 0);
        for (int p = 0; p < 2; p++) begin
            packing_case($sformatf("scnn_extrema_p%0d", p), SCNN, 0, 1'(p),
                2, 2, 577, 577, 0, 1, 0, 1);
            for (int c = 0; c < 2; c++)
                for (int k = 0; k < 4; k++)
                    packing_case($sformatf("invalid_subwords_p%0d_c%0d_rotation%0d", p, c, k),
                        SCNN, 0, 1'(p), 0, 577, c ? 0 : 2, c ? 577 : 2,
                        12+k, 12+(k+1)%4, 12+(k+2)%4, 12+(k+3)%4);
        end

        // Both old counterexamples and the existing same-pair guard, on each A bank.
        for (int b = 0; b < 4; b++) begin
            for (int kind = 0; kind < 3; kind++) begin
                begin_case($sformatf("fallback_bank%0d_%s", b,
                    kind == 0 ? "no_new_read" : kind == 1 ? "same_pair" : "new_pair"));
                restart();
                set_read(SCNN, 0, 1'(b/2), 1, 1, 2, 2, 0, 1, 2, 3);
                tick(1, 0); read_valid = 0;
                if (kind != 0)
                    set_read(SCNN, 0, 1'(kind == 1 ? b/2 : 1-b/2), 1, 1, 1, 1, 3, 2, 1, 0);
                set_fill(b, kind == 2 ? 4 : 3, 20+b+kind);
                repeat (3) tick(0, 0);
                result_ready = 1;
                tick(kind != 0, 1);
                read_valid = 0; fill_valid = 0; result_ready = 0;
                if (kind != 0) begin repeat (2) tick(0, 0); consume(); end
                // Read back the accepted fill after releasing the held response.
                set_read(SCNN, 0, 1'(b/2), kind == 2 ? 4 : 3, kind == 2 ? 4 : 3,
                    kind == 2 ? 4 : 3, kind == 2 ? 4 : 3, 0, 1, 2, 3);
                tick(1, 0); consume();
                end_case(kind == 0 ? 2 : 3, 1, 3, kind == 0 ? 3 : 5, kind == 0 ? 0 : 1, 1);
            end
        end

        for (int p = 0; p < 2; p++) begin
            begin_case($sformatf("fallback_inactive_pair_p%0d", p));
            restart();
            set_read(SCNN, 0, 1'(p), 1, 1, 2, 2, 0, 1, 2, 3);
            tick(1, 0); read_valid = 0;
            // Idle input metadata may change; accepted metadata must still govern.
            mode = FC; fc_int8 = 1; pair_sel = 1'(1-p);
            for (int k = 0; k < 2; k++) begin set_fill((1-p)*2+k, 3+k, 40+p); tick(0, 1); end
            fill_valid = 0; tick(0, 0); consume();
            end_case(1, 2, 0, 3, 0, 1);

            begin_case($sformatf("coalesced_held_same_row_fills_p%0d", p));
            restart();
            set_read(SCNN, 0, 1'(p), 1, 1, 1, 1, 0, 1, 2, 11);
            tick(1, 0); read_valid = 0;
            for (int k = 0; k < 2; k++) begin set_fill(p*2+k, 1, 50+p); tick(0, 1); end
            fill_valid = 0; repeat (2) tick(0, 0); consume();
            set_read(SCNN, 0, 1'(p), 1, 1, 1, 1, 0, 1, 2, 11);
            tick(1, 0); consume();
            end_case(2, 2, 0, 4, 0, 1);
        end

        for (int b = 0; b < 4; b++) begin
            for (int collision = 0; collision < 2; collision++) begin
                begin_case($sformatf("coalesced_bank%0d_collision%0d", b, collision));
                restart();
                set_read(SCNN, 0, 1'(b/2), 0, 577, 0, 577, 11, 11, 0, 1);
                set_fill(b, collision ? (b%2 ? 577 : 0) : 3, 60+b);
                tick(1, !collision);
                read_valid = 0;
                if (collision) tick(0, 1);
                fill_valid = 0; repeat (2) tick(0, 0); consume();
                end_case(1, 1, collision, collision ? 3 : 2, collision ? 0 : 1, 1);
            end
            // The current request reserves its banks even before it is accepted.
            for (int fallback = 0; fallback < 2; fallback++) begin
                begin_case($sformatf("stalled_current_guard_bank%0d_fallback%0d", b, fallback));
                restart();
                set_read(SCNN, 0, 1'(1-b/2), 1, 1, 1, 1, 0, 1, 2, 3);
                tick(1, 0);
                set_read(SCNN, 0, 1'(b/2), 1, 1, fallback ? 2 : 1, fallback ? 2 : 1, 0, 1, 2, 3);
                set_fill(b, fallback ? 3 : 1, 70+b);
                repeat (2) tick(0, 0);
                result_ready = 1; tick(1, 0);
                read_valid = 0; tick(0, 1);
                fill_valid = 0; result_ready = 0;
                end_case(2, 1, 3, 2, 0, 1);
            end
        end

        // FC (both precisions), SRNN and M2 protect all four A outputs.
        for (int format_id = 0; format_id < 4; format_id++) begin
            for (int b = 0; b < 4; b++) begin
                begin_case($sformatf("all_bank_hold_format%0d_bank%0d", format_id, b));
                restart();
                set_read(format_id < 2 ? FC : format_id == 2 ? SRNN : M2,
                    format_id == 1, 0, format_id == 0 ? 0 : 1, 1, 2, 577, 0, 1, 2, 3);
                tick(1, 0); read_valid = 0;
                mode = SCNN; pair_sel = 1'(1-b/2); fc_int8 = !fc_int8;
                set_fill(b, 3, 80+format_id*4+b);
                repeat (2) tick(0, 0);
                set_read(SCNN, 0, 1'(1-b/2), 1, 1, 1, 1, 3, 2, 1, 0);
                repeat (2) tick(0, 0);
                result_ready = 1; tick(1, 1);
                read_valid = 0; fill_valid = 0; result_ready = 0;
                tick(0, 0); consume();
                end_case(2, 1, 4, 5, 1, 1);
            end
        end

        for (int p = 0; p < 2; p++) begin
            begin_case($sformatf("reset_pending_retains_memory_p%0d", p));
            restart();
            before_discard = discarded;
            set_read(SCNN, 0, 1'(p), 1, 1, 2, 2, 0, 1, 2, 3);
            tick(1, 0);
            set_read(SCNN, 0, 1'(1-p), 3, 3, 3, 3, 0, 1, 2, 3);
            set_fill(p*2, 3, 101+p);
            repeat (2) tick(0, 0);
            restart();
            check(discarded-before_discard == 1, "reset discards exactly one pending response");
            set_read(FC, 1, 0, 3, 3, 3, 3, 0, 0, 0, 0);
            tick(1, 0); read_valid = 0; tick(0, 0); consume();
            end_case(2, 0, 2, 3, 0, 2);
        end
        packing_case("retained_first_rows", FC, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        packing_case("retained_last_rows", FC, 1, 0, 577, 577, 577, 577, 0, 0, 0, 0);
        for (int f = 0; f < 3; f++)
            check(sign_edges[f][0] && sign_edges[f][1], "signed minima and maxima observed for every width");
        for (int b = 0; b < 4; b++)
            check(boundary[b][0] && boundary[b][1], "first and last rows read in every bank");
        check(cases == 77, "all directed case records completed");
        check(checks > 0 && reads > 0 && writes > 0 && blocked > 0 &&
            held_cycles > 0 && simultaneous > 0 && resets > 0, "all required counters positive");
        $display("HELD_RESPONSE_PASS checks=%0d reads=%0d writes=%0d blocked=%0d held_cycles=%0d simultaneous=%0d resets=%0d",
            checks, reads, writes, blocked, held_cycles, simultaneous, resets);
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "HELD_RESPONSE_FAIL watchdog timeout");
    end
endmodule

`default_nettype wire
