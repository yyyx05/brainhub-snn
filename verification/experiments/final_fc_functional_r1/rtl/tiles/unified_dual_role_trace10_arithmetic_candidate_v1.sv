`timescale 1ns/1ps
`default_nettype none

// Ten-lane trace update datapath shared by FC, SRNN, and SCNN profiles.
// Storage and record transport stay outside this block so the same arithmetic
// can be placed either in each tile or in a regional observation collector.
module unified_dual_role_trace10_arithmetic_candidate_v1 #(
    parameter int LANES = 10,
    parameter int FC_TRACE_FRAC_BITS = 10
) (
    input  wire logic                    i_clk,
    input  wire logic                    i_rst_n,
    input  wire logic                    i_abort,
    input  wire logic                    i_valid,
    output logic                         o_ready,
    input  wire logic [1:0]              i_mode,
    input  wire logic [LANES-1:0]        i_lane_mask,
    input  wire logic [LANES-1:0]        i_spike,
    input  wire logic [LANES*32-1:0]     i_trace_old_flat,
    input  wire logic [15:0]             i_trace_decay,
    output logic                         o_valid,
    input  wire logic                    i_ready,
    output logic [LANES*32-1:0]          o_trace_next_flat,
    output logic                         o_mode_error
);
    localparam logic [1:0] MODE_FC = 2'd0;
    localparam logic [1:0] MODE_SRNN = 2'd1;
    localparam logic [1:0] MODE_SCNN = 2'd2;
    localparam logic [1:0] MODE_M2_Q14 = 2'd3;

    logic product_valid_q;
    logic result_valid_q;
    logic [1:0] mode_q;
    logic [LANES-1:0] lane_mask_q;
    logic [LANES-1:0] spike_q;
    logic [LANES*32-1:0] trace_old_flat_q;
    logic signed [32:0] trace_operand [0:LANES-1];
    logic signed [16:0] decay_operand;
    (* use_dsp = "yes" *) logic signed [49:0] trace_product_q [0:LANES-1];
    logic [LANES*32-1:0] trace_next_flat;
    logic mode_error_next;
    logic result_slot_ready;
    logic product_slot_ready;
    logic input_fire;

    function automatic logic signed [31:0] saturate50_to32(
        input logic signed [49:0] value
    );
        begin
            if (value > 50'sd2147483647)
                saturate50_to32 = 32'sh7fffffff;
            else if (value < -50'sd2147483648)
                saturate50_to32 = 32'sh80000000;
            else
                saturate50_to32 = value[31:0];
        end
    endfunction

    function automatic logic signed [49:0] round_shift_even14(
        input logic signed [49:0] value
    );
        logic negative;
        logic [49:0] magnitude;
        logic [49:0] quotient;
        logic [13:0] remainder;
        begin
            negative = value[49];
            magnitude = negative ? $unsigned(-value) : $unsigned(value);
            quotient = magnitude >> 14;
            remainder = magnitude[13:0];
            if ((remainder > 14'd8192) ||
                ((remainder == 14'd8192) && quotient[0]))
                quotient = quotient + 1'b1;
            round_shift_even14 = negative ?
                -$signed(quotient) : $signed(quotient);
        end
    endfunction

    function automatic logic [15:0] fc_trace_update(
        input logic signed [49:0] product,
        input logic spike
    );
        logic [16:0] trace_sum;
        begin
            trace_sum = {1'b0,
                16'(($unsigned(product) + 50'd32768) >> 16)} +
                (spike ? (17'(1) << FC_TRACE_FRAC_BITS) : 17'd0);
            fc_trace_update = trace_sum[16] ? 16'hffff : trace_sum[15:0];
        end
    endfunction

    function automatic logic signed [15:0] scnn_trace_update(
        input logic signed [49:0] product,
        input logic spike
    );
        begin
            scnn_trace_update = 16'((product >>> 8) +
                (spike ? 50'sd256 : 50'sd0));
        end
    endfunction

    assign result_slot_ready = !result_valid_q || i_ready;
    assign product_slot_ready = !product_valid_q || result_slot_ready;
    assign o_ready = !i_abort && product_slot_ready;
    assign o_valid = !i_abort && result_valid_q;
    assign input_fire = i_valid && o_ready;

    always_comb begin
        decay_operand = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            trace_operand[lane] = '0;
            if (i_mode == MODE_FC)
                trace_operand[lane] =
                    $signed({1'b0, 16'd0,
                             i_trace_old_flat[lane*32 +: 16]});
            else if (i_mode == MODE_SRNN || i_mode == MODE_M2_Q14)
                trace_operand[lane] =
                    $signed({i_trace_old_flat[lane*32 + 31],
                             i_trace_old_flat[lane*32 +: 32]});
            else if (i_mode == MODE_SCNN)
                trace_operand[lane] =
                    $signed({{17{i_trace_old_flat[lane*32 + 15]}},
                             i_trace_old_flat[lane*32 +: 16]});
        end

        if (i_mode == MODE_FC)
            decay_operand = $signed({1'b0, i_trace_decay[15:0]});
        else
            decay_operand = $signed({i_trace_decay[15],
                                     i_trace_decay[15:0]});
    end

    always_comb begin
        trace_next_flat = trace_old_flat_q;
        mode_error_next = !(mode_q == MODE_FC || mode_q == MODE_SRNN ||
                            mode_q == MODE_SCNN ||
                            mode_q == MODE_M2_Q14);

        for (int lane = 0; lane < LANES; lane++) begin
            logic [15:0] fc_next;
            logic signed [49:0] srnn_shifted;
            logic signed [49:0] srnn_next_wide;
            logic signed [15:0] scnn_next;

            fc_next = fc_trace_update(trace_product_q[lane], spike_q[lane]);
            srnn_shifted = round_shift_even14(trace_product_q[lane]);
            srnn_next_wide = srnn_shifted +
                (spike_q[lane] ? 50'sd16384 : 50'sd0);
            scnn_next = scnn_trace_update(trace_product_q[lane],
                                          spike_q[lane]);

            if (lane_mask_q[lane]) begin
                if (mode_q == MODE_FC)
                    trace_next_flat[lane*32 +: 32] = {16'd0, fc_next};
                else if (mode_q == MODE_SRNN ||
                         mode_q == MODE_M2_Q14)
                    trace_next_flat[lane*32 +: 32] =
                        saturate50_to32(srnn_next_wide);
                else if (mode_q == MODE_SCNN)
                    trace_next_flat[lane*32 +: 32] =
                        {{16{scnn_next[15]}}, scnn_next};
            end
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n || i_abort) begin
            product_valid_q <= 1'b0;
            result_valid_q <= 1'b0;
        end else begin
            if (result_slot_ready) begin
                result_valid_q <= product_valid_q;
            end

            if (product_slot_ready) begin
                product_valid_q <= input_fire;
            end
        end
    end

    // Valid bits are the ownership authority for both stages. The wide
    // payload stays reset-free so abort does not become a remote reset tree
    // on every trace bit; stale payload is never observable while valid=0.
    always_ff @(posedge i_clk) begin
        if (result_slot_ready && product_valid_q) begin
            o_trace_next_flat <= trace_next_flat;
            o_mode_error <= mode_error_next;
        end

        if (product_slot_ready && input_fire) begin
            mode_q <= i_mode;
            lane_mask_q <= i_lane_mask;
            spike_q <= i_spike;
            trace_old_flat_q <= i_trace_old_flat;
            for (int lane = 0; lane < LANES; lane++)
                trace_product_q[lane] <=
                    trace_operand[lane] * decay_operand;
        end
    end

    initial begin
        if (LANES != 10)
            $error("unified trace arithmetic candidate requires ten lanes");
        if ((FC_TRACE_FRAC_BITS < 0) || (FC_TRACE_FRAC_BITS >= 16))
            $error("FC trace fractional width is out of range");
    end
endmodule

`default_nettype wire
