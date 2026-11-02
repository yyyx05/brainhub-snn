`timescale 1ns/1ps
`default_nettype none

module tb_unified_dual_role_lif10_arithmetic_candidate_v3;
    localparam int LANES = 10;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic abort = 1'b0;
    logic valid = 1'b0;
    logic ready;
    logic [1:0] mode = '0;
    logic [319:0] current_flat = '0;
    logic [319:0] state_flat = '0;
    logic signed [15:0] leak = '0;
    logic signed [31:0] threshold = '0;
    logic reset_to_zero = 1'b0;
    logic result_valid;
    logic result_ready = 1'b1;
    logic [9:0] spike;
    logic [319:0] pre_state_flat;
    logic [319:0] next_state_flat;
    logic reference_input_valid = 1'b0;
    logic reference_ready;
    logic reference_result_valid;
    logic [9:0] reference_spike;
    logic [319:0] reference_pre_state_flat;
    logic [319:0] reference_next_state_flat;

    always #5 clk = ~clk;

    unified_dual_role_lif10_arithmetic_candidate_v3 dut (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(abort),
        .i_valid(valid), .o_ready(ready), .i_mode(mode),
        .i_current_flat(current_flat), .i_state_flat(state_flat),
        .i_leak(leak), .i_threshold(threshold),
        .i_reset_to_zero(reset_to_zero), .o_valid(result_valid),
        .i_ready(result_ready), .o_spike(spike),
        .o_pre_state_flat(pre_state_flat),
        .o_next_state_flat(next_state_flat)
    );

    unified_dual_role_lif10_arithmetic_candidate_v2 reference_dut (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(abort),
        .i_valid(reference_input_valid), .o_ready(reference_ready),
        .i_mode(mode), .i_current_flat(current_flat),
        .i_state_flat(state_flat), .i_leak(leak),
        .i_threshold(threshold), .i_reset_to_zero(reset_to_zero),
        .o_valid(reference_result_valid), .i_ready(result_ready),
        .o_spike(reference_spike),
        .o_pre_state_flat(reference_pre_state_flat),
        .o_next_state_flat(reference_next_state_flat)
    );

    task automatic compare_serial;
        begin
            result_ready = 1'b0;
            do @(negedge clk); while (!ready || !reference_ready);
            valid = 1'b1;
            reference_input_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            valid = 1'b0;
            reference_input_valid = 1'b0;
            do @(posedge clk);
            while (!result_valid || !reference_result_valid);
            if (spike !== reference_spike ||
                pre_state_flat !== reference_pre_state_flat ||
                next_state_flat !== reference_next_state_flat)
                $fatal(1, "elastic LIF differs from candidate-v2 mode=%0d", mode);
            @(negedge clk);
            result_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            result_ready = 1'b0;
        end
    endtask

    function automatic logic signed [31:0] clamp32(input longint signed value);
        if (value > 64'sd2147483647) return 32'sh7fffffff;
        if (value < -64'sd2147483648) return 32'sh80000000;
        return 32'(value);
    endfunction

    task automatic compare_m2;
        logic [9:0] expected_spike;
        logic [319:0] expected_pre, expected_next;
        longint signed product, quotient, remainder, sum_value, next_value;
        for (int lane = 0; lane < LANES; lane++) begin
            product = longint'($signed(state_flat[lane*32 +: 32])) * longint'(leak);
            quotient = product / 64'sd16384;
            remainder = product % 64'sd16384;
            if (remainder < 0) remainder = -remainder;
            if (remainder > 8192 || (remainder == 8192 && (quotient % 2 != 0)))
                quotient += product < 0 ? -1 : 1;
            sum_value = quotient + longint'($signed(current_flat[lane*32 +: 32]));
            expected_spike[lane] = sum_value >= longint'(threshold);
            expected_pre[lane*32 +: 32] = clamp32(sum_value);
            next_value = !expected_spike[lane] ? sum_value : reset_to_zero ? 0 :
                         sum_value-longint'(threshold);
            expected_next[lane*32 +: 32] = clamp32(next_value);
        end
        @(negedge clk);
        result_ready = 0;
        valid = 1;
        do @(posedge clk); while (!ready);
        @(negedge clk); valid = 0;
        while (!result_valid) @(negedge clk);
        repeat (3) begin
            if (!result_valid || spike !== expected_spike ||
                pre_state_flat !== expected_pre || next_state_flat !== expected_next)
                $fatal(1, "M2 Q14 arithmetic or stall mismatch: leak=%0d threshold=%0d", leak,threshold);
            @(negedge clk);
        end
        result_ready = 1;
        @(negedge clk);
        result_ready = 0;
    endtask

    task automatic check_result(input int index);
        logic [9:0] expected_spike;
        logic [319:0] expected_pre;
        logic [319:0] expected_next;
        begin
            expected_spike = '0;
            expected_pre = '0;
            expected_next = '0;
            case (index)
                0: begin
                    expected_spike = {LANES{1'b1}};
                    for (int lane = 0; lane < LANES; lane++) begin
                        expected_pre[lane*32 +: 32] = 32'd60;
                        expected_next[lane*32 +: 32] = 32'd0;
                    end
                end
                1: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        expected_pre[lane*32 +: 32] = 32'd75;
                        expected_next[lane*32 +: 32] = 32'd75;
                    end
                end
                default: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        expected_pre[lane*32 +: 32] = 32'd384;
                        expected_next[lane*32 +: 32] = 32'd384;
                    end
                end
            endcase
            if (!result_valid || spike !== expected_spike ||
                pre_state_flat !== expected_pre ||
                next_state_flat !== expected_next)
                $fatal(1, "elastic LIF result mismatch index=%0d", index);
        end
    endtask

    initial begin
        int observed;

        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        for (int test_index = 0; test_index < 24; test_index++) begin
            mode = 2'(test_index % 3);
            leak = 16'($urandom());
            threshold = 32'($urandom());
            reset_to_zero = test_index[0];
            for (int lane = 0; lane < LANES; lane++) begin
                state_flat[lane*32 +: 32] = 32'($urandom());
                current_flat[lane*32 +: 32] = 32'($urandom());
            end
            compare_serial();
        end

        for (int test_index = 0; test_index < 256; test_index++) begin
            mode = 2'd3;
            leak = 16'($urandom());
            threshold = 32'($urandom());
            reset_to_zero = test_index[0];
            for (int lane = 0; lane < LANES; lane++) begin
                state_flat[lane*32 +: 32] = 32'($urandom());
                current_flat[lane*32 +: 32] = 32'($urandom());
            end
            case (test_index)
                0: begin
                    leak = 13926; threshold = 8192; reset_to_zero = 0;
                    state_flat = {LANES{32'd16384}}; current_flat = 0;
                end
                1: begin
                    leak = 8192; threshold = 0; reset_to_zero = 0;
                    for (int lane = 0; lane < LANES; lane++) state_flat[lane*32 +: 32] = 32'(lane-5);
                    current_flat = 0;
                end
                2: begin
                    leak = 32767; threshold = 32'sh7fffffff; reset_to_zero = 0;
                    state_flat = {LANES{32'h7fffffff}}; current_flat = state_flat;
                end
                3: begin
                    leak = 32767; threshold = 32'sh80000000; reset_to_zero = 0;
                    state_flat = {LANES{32'h80000000}}; current_flat = state_flat;
                end
                4: begin
                    leak = -32768; threshold = 32'sh7fffffff; reset_to_zero = 0;
                    state_flat = {LANES{32'h80000000}}; current_flat = {LANES{32'h7fffffff}};
                end
                default: begin end
            endcase
            compare_m2();
        end

        result_ready = 1'b1;
        for (int lane = 0; lane < LANES; lane++) begin
            state_flat[lane*32 +: 32] = 32'd100;
            current_flat[lane*32 +: 32] = 32'd10;
        end
        mode = 2'd0;
        leak = 16'sd16384;
        threshold = 32'sd60;
        reset_to_zero = 1'b0;

        @(negedge clk);
        valid = 1'b1;
        @(negedge clk);
        if (!ready)
            $fatal(1, "elastic LIF lost one-cycle initiation rate");
        mode = 2'd1;
        leak = 16'sd16384;
        threshold = 32'sd100;
        reset_to_zero = 1'b1;
        for (int lane = 0; lane < LANES; lane++) begin
            state_flat[lane*32 +: 32] = 32'd100;
            current_flat[lane*32 +: 32] = 32'd25;
        end
        @(negedge clk);
        if (!ready)
            $fatal(1, "elastic LIF failed second consecutive input");
        mode = 2'd2;
        leak = 16'sd128;
        threshold = 32'sd512;
        reset_to_zero = 1'b0;
        for (int lane = 0; lane < LANES; lane++) begin
            state_flat[lane*32 +: 32] = 32'd256;
            current_flat[lane*32 +: 32] = 32'd256;
        end
        @(negedge clk);
        valid = 1'b0;

        observed = 0;
        while (observed < 3) begin
            if (result_valid) begin
                check_result(observed);
                observed++;
            end
            @(negedge clk);
        end

        while (!ready)
            @(negedge clk);
        mode = 2'd1;
        leak = 16'sd16384;
        threshold = 32'sd100;
        reset_to_zero = 1'b0;
        valid = 1'b1;
        @(negedge clk);
        valid = 1'b0;
        result_ready = 1'b0;
        while (!result_valid)
            @(negedge clk);
        repeat (3) begin
            if (!result_valid)
                $fatal(1, "elastic LIF result did not hold under backpressure");
            @(negedge clk);
        end
        abort = 1'b1;
        @(negedge clk);
        abort = 1'b0;
        result_ready = 1'b1;
        @(negedge clk);
        if (result_valid)
            $fatal(1, "elastic LIF abort did not flush pipeline");
        if (reference_result_valid)
            $fatal(1, "reference LIF retained an unexpected transaction");

        $display("PASS: unified dual-role elastic LIF10 candidate; M2_Q14_groups=256 lanes=10 legacy_groups=24");
        $finish;
    end
endmodule

`default_nettype wire
