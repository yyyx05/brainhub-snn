`timescale 1ns/1ps
`default_nettype none
module tb_gradient_retire;
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst_n = 0, cfg = 0, start = 0, abort_i = 0, iv = 0, gr = 0;
    logic [7:0] batch = 1, sample_id = 0;
    logic [23:0] elements = 1, addr = 0;
    logic signed [63:0] error_q40 = 0;
    logic signed [31:0] trace_q14 = 0;
    logic last = 0;
    wire cr, ce, sr, ir, gv, gl, busy, done, pe, ov;
    wire [23:0] ga;
    wire signed [63:0] gq;
    logic [195:0] vectors [0:7619];
    string filename;
    integer arm;
    unified_srnn_tp_wout_gradient_reducer dut (
        .i_clk(clk), .i_rst_n(rst_n), .i_cfg_valid(cfg), .o_cfg_ready(cr),
        .i_cfg_batch_count(batch), .i_cfg_element_count(elements), .o_cfg_error(ce),
        .i_start(start), .o_start_ready(sr), .i_abort(abort_i),
        .i_contribution_valid(iv), .o_contribution_ready(ir),
        .i_contribution_local_addr(addr), .i_contribution_sample_id(sample_id),
        .i_error_q40(error_q40), .i_hidden_trace_q14(trace_q14),
        .i_contribution_tensor_last(last), .o_gradient_valid(gv), .i_gradient_ready(gr),
        .o_gradient_local_addr(ga), .o_gradient_q40(gq), .o_gradient_tensor_last(gl),
        .o_busy(busy), .o_done(done), .o_protocol_error(pe), .o_overflow(ov)
    );
    task automatic reset_dut;
        @(negedge clk);
        rst_n = 0; cfg = 0; start = 0; abort_i = 0; iv = 0; gr = 0;
        repeat (2) @(posedge clk);
        @(negedge clk); rst_n = 1;
    endtask
    task automatic configure(input integer b, input integer n);
        reset_dut();
        @(negedge clk); cfg = 1; batch = 8'(b); elements = 24'(n);
        if (!cr) $fatal(1, "configuration unavailable");
        @(posedge clk); #1;
        if (ce) $fatal(1, "legal configuration rejected");
        @(negedge clk); cfg = 0;
        #1; if (!sr) $fatal(1, "start unavailable");
        start = 1;
        @(posedge clk); #1;
        @(negedge clk); start = 0;
    endtask
    task automatic send_first(input bit tail);
        @(negedge clk);
        iv = 1; addr = 0; sample_id = 0; last = tail;
        error_q40 = 64'sd1048576; trace_q14 = 1;
        #1; if (!ir) $fatal(1, "first contribution unavailable");
        @(posedge clk); #1;
        @(negedge clk); iv = 0;
    endtask
    task automatic wait_gradient;
        integer timeout_count;
        timeout_count = 0;
        while (!gv) begin
            @(posedge clk); #1;
            timeout_count++;
            if (timeout_count > 5) $fatal(1, "gradient absent");
        end
    endtask
    task automatic run_case(input integer b, input integer offset, input integer mode);
        integer sent, got, cycles, input_stalls, output_stalls, overlaps;
        bit infire, outfire, last_infire, held, expected_ov;
        logic [88:0] held_payload;
        logic [195:0] row, expected_row;
        configure(b, 20);
        sent = 0; got = 0; cycles = 0; input_stalls = 0; output_stalls = 0; overlaps = 0;
        held = 0; last_infire = 0; expected_ov = 0;
        while (got < 20) begin
            @(negedge clk);
            gr = (mode != 2) || ((cycles % 11) >= 4);
            if (!iv || last_infire) begin
                iv = sent < b * 20 && ((mode != 1) || (cycles % 2 == 0));
            end
            if (sent < b * 20) begin
                row = vectors[offset + sent];
                addr = row[195:172]; sample_id = row[171:164];
                error_q40 = row[163:100]; trace_q14 = row[99:68]; last = row[0];
            end
            #1;
            if (held && (!gv || {ga, gq, gl} !== held_payload))
                $fatal(1, "stalled gradient changed");
            infire = iv && ir; outfire = gv && gr;
            held = gv && !gr; held_payload = {ga, gq, gl};
            if (iv && !ir) input_stalls++;
            if (gv && !gr) output_stalls++;
            if (infire && outfire) overlaps++;
            if (outfire) begin
                expected_row = vectors[offset + (got + 1) * b - 1];
                expected_ov |= expected_row[1];
                if (ga !== 24'(got) || gq !== expected_row[67:4] || gl !== (got == 19) || ov !== expected_ov)
                    $fatal(1, "oracle mismatch b=%0d mode=%0d index=%0d", b, mode, got);
            end
            @(posedge clk); #1;
            if (infire) sent++;
            if (outfire) got++;
            if (ce || pe || (done && got != 20)) $fatal(1, "unexpected status");
            last_infire = infire;
            cycles++;
            if (cycles > 100000) $fatal(1, "case timeout");
        end
        if (sent != b * 20 || busy || !done) $fatal(1, "incomplete drain");
        @(negedge clk); iv = 0; gr = 0;
        $display("GRADIENT_RETIRE_CASE arm=%0d batch=%0d mode=%0d inputs=%0d outputs=%0d cycles=%0d input_stalls=%0d output_stalls=%0d overlaps=%0d", arm, b, mode, sent, got, cycles, input_stalls, output_stalls, overlaps);
    endtask
    task automatic directed;
        integer k, retired, timeout_count;
        bit infire, outfire;
        reset_dut();
        @(negedge clk); cfg = 1; batch = 0; elements = 1;
        @(posedge clk); #1;
        if (!ce || sr || busy) $fatal(1, "zero batch accepted");
        for (k = 0; k < 4; k++) begin
            configure(1, k == 2 ? 2 : 1);
            @(negedge clk); iv = 1;
            addr = k == 0 ? 1 : 0; sample_id = k == 1 ? 1 : 0;
            last = k != 3;
            #1; if (!ir) $fatal(1, "malformed item not consumed");
            @(posedge clk); #1;
            if (!pe || !done || busy || gv) $fatal(1, "malformed item not rejected");
        end
        configure(1, 2); send_first(0); wait_gradient();
        repeat (3) begin
            @(posedge clk); #1;
            if (!gv || ga != 0 || gq != 1 || gl) $fatal(1, "hold before abort");
        end
        @(negedge clk); abort_i = 1;
        @(posedge clk); #1;
        if (gv || busy || !done) $fatal(1, "abort failed");

        configure(1, 2); send_first(0);
        // send_first returns before the finalization edge.
        rst_n = 0; #1;
        if (gv || busy || ov || pe) $fatal(1, "in-flight reset failed");

        configure(1, 1); send_first(1); wait_gradient();
        @(negedge clk); gr = 1; iv = 1; addr = 0; sample_id = 0; last = 1;
        #1; if (ir) $fatal(1, "next tensor accepted on terminal retirement");
        @(posedge clk); #1;
        if (!done || busy || gv) $fatal(1, "terminal drain failed");

        configure(1, 2); send_first(0); wait_gradient();
        retired = 0; timeout_count = 0;
        while (!pe) begin
            @(negedge clk); gr = 1; iv = 1; addr = 77; sample_id = 0; last = 1;
            #1; infire = iv && ir; outfire = gv && gr;
            if (outfire && (ga != 0 || gq != 1 || gl)) $fatal(1, "old result corrupted by bad next item");
            @(posedge clk); #1;
            if (outfire) retired++;
            if (infire && !pe) $fatal(1, "bad successor accepted");
            timeout_count++;
            if (timeout_count > 4) $fatal(1, "bad successor not rejected");
        end
        if (retired != 1 || busy || gv || !done) $fatal(1, "bad successor drain failed");
        $display("GRADIENT_RETIRE_DIRECTED arm=%0d cases=9", arm);
    endtask
    integer b, offset, mode, i;
    initial begin
        if (!$value$plusargs("VECTORS=%s", filename)) $fatal(1, "vectors missing");
        if (!$value$plusargs("ARM=%d", arm)) $fatal(1, "arm missing");
        $readmemh(filename, vectors);
        offset = 0;
        for (i = 0; i < 8; i++) begin
            case (i)
                0: b = 1;
                1: b = 2;
                2: b = 3;
                3: b = 8;
                4: b = 16;
                5: b = 32;
                6: b = 64;
                default: b = 255;
            endcase
            for (mode = 0; mode < 3; mode++) run_case(b, offset, mode);
            offset += b * 20;
        end
        directed();
        $display("GRADIENT_RETIRE_PASS arm=%0d cases=24 directed=9 inputs=22860 outputs=480", arm);
        $finish;
    end
endmodule
`default_nettype wire
