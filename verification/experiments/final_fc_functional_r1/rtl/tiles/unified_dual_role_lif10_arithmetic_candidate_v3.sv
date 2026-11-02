`timescale 1ns/1ps
`default_nettype none

// Fully elastic version of the common ten-lane LIF arithmetic. It preserves
// candidate-v2 FC/backbone/SCNN semantics while allowing one accepted group per cycle,
// which makes one instance shareable by four neighboring H40 accumulators.
// The M2 role uses its formal Q14 leak and round-to-nearest-even convention.
module unified_dual_role_lif10_arithmetic_candidate_v3 #(
    parameter int LANES = 10
) (
    input  wire logic                    i_clk,
    input  wire logic                    i_rst_n,
    input  wire logic                    i_abort,
    input  wire logic                    i_valid,
    output logic                         o_ready,
    input  wire logic [1:0]              i_mode,
    input  wire logic [LANES*32-1:0]     i_current_flat,
    input  wire logic [LANES*32-1:0]     i_state_flat,
    input  wire logic signed [15:0]      i_leak,
    input  wire logic signed [31:0]      i_threshold,
    input  wire logic                    i_reset_to_zero,
    output logic                         o_valid,
    input  wire logic                    i_ready,
    output logic [LANES-1:0]             o_spike,
    output logic [LANES*32-1:0]          o_pre_state_flat,
    output logic [LANES*32-1:0]          o_next_state_flat
);
    localparam logic [1:0] MODE_FC = 2'd0;
    localparam logic [1:0] MODE_SRNN = 2'd1;
    localparam logic [1:0] MODE_SCNN = 2'd2;
    localparam logic [1:0] MODE_M2_Q14 = 2'd3;

    logic product_valid_q;
    logic sum_valid_q;
    logic result_valid_q;
    logic result_slot_ready;
    logic sum_slot_ready;
    logic product_slot_ready;
    logic input_fire;
    logic [1:0] product_mode_q;
    logic [1:0] sum_mode_q;
    logic [LANES*32-1:0] product_current_q;
    logic signed [31:0] product_threshold_q;
    logic signed [31:0] sum_threshold_q;
    logic product_reset_to_zero_q;
    logic sum_reset_to_zero_q;
    logic signed [31:0] state_operand [0:LANES-1];
    logic signed [47:0] scaled_leak [0:LANES-1];
    logic signed [47:0] normalized_current [0:LANES-1];
    (* use_dsp = "yes" *) logic signed [47:0] leak_product_q [0:LANES-1];
    (* use_dsp = "yes" *) logic signed [47:0] pre_sum_q [0:LANES-1];
    logic [LANES-1:0] spike_next;
    logic [LANES*32-1:0] pre_state_next_flat;
    logic [LANES*32-1:0] next_state_next_flat;

    function automatic logic signed [31:0] saturate48_to32(
        input logic signed [47:0] value
    );
        begin
            if (value > 48'sh00007fffffff)
                saturate48_to32 = 32'sh7fffffff;
            else if (value < -48'sh000080000000)
                saturate48_to32 = 32'sh80000000;
            else
                saturate48_to32 = value[31:0];
        end
    endfunction

    function automatic logic signed [15:0] saturate17_to16(
        input logic signed [16:0] value
    );
        begin
            if (value > 17'sd32767)
                saturate17_to16 = 16'sh7fff;
            else if (value < -17'sd32768)
                saturate17_to16 = 16'sh8000;
            else
                saturate17_to16 = value[15:0];
        end
    endfunction

    function automatic logic signed [47:0] round_even_q14(
        input logic signed [47:0] value
    );
        logic [47:0] magnitude;
        logic [47:0] quotient;
        logic [47:0] rounded;
        begin
            magnitude = value[47] ? $unsigned(-value) : $unsigned(value);
            quotient = magnitude >> 14;
            rounded = quotient + 48'((magnitude[13:0] > 14'd8192) ||
                ((magnitude[13:0] == 14'd8192) && quotient[0]));
            round_even_q14 = value[47] ? -$signed(rounded) : $signed(rounded);
        end
    endfunction

    assign result_slot_ready = !result_valid_q || i_ready;
    assign sum_slot_ready = !sum_valid_q || result_slot_ready;
    assign product_slot_ready = !product_valid_q || sum_slot_ready;
    assign o_ready = !i_abort && product_slot_ready;
    assign o_valid = !i_abort && result_valid_q;
    assign input_fire = i_valid && o_ready;

    always_comb begin
        for (int lane = 0; lane < LANES; lane++) begin
            logic signed [31:0] current32;
            logic signed [17:0] fc_leak;
            logic signed [16:0] scnn_leak;

            if (i_mode == MODE_FC) begin
                state_operand[lane] =
                    {{14{i_state_flat[lane*32 + 17]}},
                     i_state_flat[lane*32 +: 18]};
            end else if (i_mode == MODE_SCNN) begin
                state_operand[lane] =
                    {{16{i_state_flat[lane*32 + 15]}},
                     i_state_flat[lane*32 +: 16]};
            end else if (i_mode == MODE_SRNN || i_mode == MODE_M2_Q14) begin
                state_operand[lane] =
                    $signed(i_state_flat[lane*32 +: 32]);
            end else begin
                state_operand[lane] = '0;
            end

            current32 = $signed(product_current_q[lane*32 +: 32]);
            fc_leak = 18'((leak_product_q[lane] + 48'sd16384) >>> 15);
            scnn_leak = 17'(leak_product_q[lane] >>> 8);
            if (product_mode_q == MODE_FC) begin
                scaled_leak[lane] = {{30{fc_leak[17]}}, fc_leak};
                normalized_current[lane] =
                    {{30{current32[17]}}, current32[17:0]};
            end else if (product_mode_q == MODE_SCNN) begin
                scaled_leak[lane] = {{31{scnn_leak[16]}}, scnn_leak};
                normalized_current[lane] =
                    {{32{current32[15]}}, current32[15:0]};
            end else if (product_mode_q == MODE_SRNN) begin
                scaled_leak[lane] = leak_product_q[lane] >>> 15;
                normalized_current[lane] = {{16{current32[31]}}, current32};
            end else if (product_mode_q == MODE_M2_Q14) begin
                scaled_leak[lane] = round_even_q14(leak_product_q[lane]);
                normalized_current[lane] = {{16{current32[31]}}, current32};
            end else begin
                scaled_leak[lane] = '0;
                normalized_current[lane] = '0;
            end
        end
    end

    always_comb begin
        spike_next = '0;
        pre_state_next_flat = '0;
        next_state_next_flat = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            logic signed [31:0] compare_pre;
            logic signed [31:0] reported_pre;
            logic signed [31:0] compare_threshold;
            logic signed [31:0] next_state;
            logic signed [17:0] fc_pre;
            logic signed [17:0] fc_threshold;
            logic signed [17:0] fc_after;
            logic signed [16:0] scnn_pre;
            logic signed [16:0] scnn_threshold;
            logic signed [16:0] scnn_after;
            logic signed [15:0] scnn_next;
            logic mode_valid;

            compare_pre = '0;
            reported_pre = '0;
            compare_threshold = '0;
            next_state = '0;
            fc_pre = '0;
            fc_threshold = '0;
            fc_after = '0;
            scnn_pre = '0;
            scnn_threshold = '0;
            scnn_after = '0;
            scnn_next = '0;
            mode_valid = 1'b0;

            if (sum_mode_q == MODE_FC) begin
                mode_valid = 1'b1;
                fc_pre = pre_sum_q[lane][17:0];
                fc_threshold = sum_threshold_q[17:0];
                compare_pre = {{14{fc_pre[17]}}, fc_pre};
                reported_pre = compare_pre;
                compare_threshold = {{14{fc_threshold[17]}}, fc_threshold};
                fc_after = fc_pre - fc_threshold;
                next_state = {{14{fc_after[17]}}, fc_after};
            end else if (sum_mode_q == MODE_SCNN) begin
                mode_valid = 1'b1;
                scnn_pre = pre_sum_q[lane][16:0];
                scnn_threshold = {sum_threshold_q[15],
                                  sum_threshold_q[15:0]};
                compare_pre = {{15{scnn_pre[16]}}, scnn_pre};
                reported_pre = {{16{scnn_pre[15]}}, scnn_pre[15:0]};
                compare_threshold =
                    {{15{scnn_threshold[16]}}, scnn_threshold};
                scnn_after = scnn_pre - scnn_threshold;
                scnn_next = saturate17_to16(scnn_after);
                next_state = {{16{scnn_next[15]}}, scnn_next};
            end else if (sum_mode_q == MODE_SRNN) begin
                mode_valid = 1'b1;
                compare_pre = saturate48_to32(pre_sum_q[lane]);
                reported_pre = compare_pre;
                compare_threshold = sum_threshold_q;
                next_state = compare_pre - sum_threshold_q;
            end else if (sum_mode_q == MODE_M2_Q14) begin
                mode_valid = 1'b1;
                compare_pre = saturate48_to32(pre_sum_q[lane]);
                reported_pre = compare_pre;
                compare_threshold = sum_threshold_q;
                next_state = saturate48_to32(pre_sum_q[lane] -
                    {{16{sum_threshold_q[31]}}, sum_threshold_q});
            end

            spike_next[lane] = mode_valid && (sum_mode_q == MODE_M2_Q14 ?
                (pre_sum_q[lane] >= $signed({{16{sum_threshold_q[31]}}, sum_threshold_q})) :
                (compare_pre >= compare_threshold));
            pre_state_next_flat[lane*32 +: 32] = reported_pre;
            if (!spike_next[lane]) begin
                next_state_next_flat[lane*32 +: 32] = reported_pre;
            end else if ((sum_mode_q != MODE_FC) &&
                         sum_reset_to_zero_q) begin
                next_state_next_flat[lane*32 +: 32] = 32'sd0;
            end else begin
                next_state_next_flat[lane*32 +: 32] = next_state;
            end
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n || i_abort) begin
            product_valid_q <= 1'b0;
            sum_valid_q <= 1'b0;
            result_valid_q <= 1'b0;
            product_mode_q <= MODE_FC;
            sum_mode_q <= MODE_FC;
            product_current_q <= '0;
            product_threshold_q <= '0;
            sum_threshold_q <= '0;
            product_reset_to_zero_q <= 1'b0;
            sum_reset_to_zero_q <= 1'b0;
            o_spike <= '0;
            o_pre_state_flat <= '0;
            o_next_state_flat <= '0;
            for (int lane = 0; lane < LANES; lane++) begin
                leak_product_q[lane] <= '0;
                pre_sum_q[lane] <= '0;
            end
        end else begin
            if (result_slot_ready) begin
                result_valid_q <= sum_valid_q;
                if (sum_valid_q) begin
                    o_spike <= spike_next;
                    o_pre_state_flat <= pre_state_next_flat;
                    o_next_state_flat <= next_state_next_flat;
                end
            end

            if (sum_slot_ready) begin
                sum_valid_q <= product_valid_q;
                if (product_valid_q) begin
                    sum_mode_q <= product_mode_q;
                    sum_threshold_q <= product_threshold_q;
                    sum_reset_to_zero_q <= product_reset_to_zero_q;
                    for (int lane = 0; lane < LANES; lane++)
                        pre_sum_q[lane] <=
                            scaled_leak[lane] + normalized_current[lane];
                end
            end

            if (product_slot_ready) begin
                product_valid_q <= input_fire;
                if (input_fire) begin
                    product_mode_q <= i_mode;
                    product_current_q <= i_current_flat;
                    product_threshold_q <= i_threshold;
                    product_reset_to_zero_q <= i_reset_to_zero;
                    for (int lane = 0; lane < LANES; lane++)
                        leak_product_q[lane] <=
                            state_operand[lane] * $signed(i_leak);
                end
            end
        end
    end

    initial begin
        if (LANES != 10)
            $error("elastic normalized LIF candidate requires ten lanes");
    end
endmodule

`default_nettype wire
