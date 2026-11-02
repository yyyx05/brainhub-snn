`timescale 1ns/1ps
`default_nettype none

// Streams the SRNN M2 W_out CE reduction in row-major parameter order.
// For each class/hidden address, B Q40-error x Q14-trace products are summed
// and rounded exactly as the SHD M2 reference before one Q40 gradient is sent.
module unified_srnn_tp_wout_gradient_reducer #(
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int SAMPLE_ID_WIDTH = 8,
    parameter int ERROR_WIDTH = 64,
    parameter int TRACE_WIDTH = 32,
    parameter int GRADIENT_WIDTH = 64,
    parameter int ACC_WIDTH = 128
) (
    input  logic                                     i_clk,
    input  logic                                     i_rst_n,

    input  logic                                     i_cfg_valid,
    output logic                                     o_cfg_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]               i_cfg_batch_count,
    input  logic [PARAM_ADDR_WIDTH-1:0]              i_cfg_element_count,
    output logic                                     o_cfg_error,

    input  logic                                     i_start,
    output logic                                     o_start_ready,
    input  logic                                     i_abort,

    input  logic                                     i_contribution_valid,
    output logic                                     o_contribution_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]              i_contribution_local_addr,
    input  logic [SAMPLE_ID_WIDTH-1:0]               i_contribution_sample_id,
    input  logic signed [ERROR_WIDTH-1:0]            i_error_q40,
    input  logic signed [TRACE_WIDTH-1:0]            i_hidden_trace_q14,
    input  logic                                     i_contribution_tensor_last,

    output logic                                     o_gradient_valid,
    input  logic                                     i_gradient_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]              o_gradient_local_addr,
    output logic signed [GRADIENT_WIDTH-1:0]         o_gradient_q40,
    output logic                                     o_gradient_tensor_last,

    output logic                                     o_busy,
    output logic                                     o_done,
    output logic                                     o_protocol_error,
    output logic                                     o_overflow
);

    logic configured_q;
    logic active_q;
    logic finalize_q;
    logic pending_q;
    logic [SAMPLE_ID_WIDTH-1:0] batch_count_q;
    logic [PARAM_ADDR_WIDTH-1:0] element_count_q;
    logic [PARAM_ADDR_WIDTH-1:0] expected_addr_q;
    logic [SAMPLE_ID_WIDTH-1:0] expected_sample_q;
    logic signed [ACC_WIDTH-1:0] accumulator_q;
    logic [PARAM_ADDR_WIDTH-1:0] gradient_addr_q;
    logic signed [GRADIENT_WIDTH-1:0] gradient_q;
    logic gradient_last_q;

    logic cfg_legal;
    logic input_expected_last_sample;
    logic input_expected_tensor_last;
    logic input_legal;
    logic input_fire;
    logic output_fire;
    logic signed [ERROR_WIDTH+TRACE_WIDTH-1:0] contribution_product;
    logic signed [ACC_WIDTH-1:0] contribution_extended;
    logic signed [ACC_WIDTH-1:0] accumulated_with_current;
    logic signed [ACC_WIDTH-1:0] rounded_shifted;
    logic rounded_overflow;

    function automatic logic signed [ACC_WIDTH-1:0] round_shift_even_20(
        input logic signed [ACC_WIDTH-1:0] value
    );
        logic negative;
        logic [ACC_WIDTH-1:0] magnitude;
        logic [ACC_WIDTH-1:0] quotient;
        logic [19:0] remainder;
        begin
            negative = value[ACC_WIDTH-1];
            magnitude = negative ? -value : value;
            quotient = magnitude >> 20;
            remainder = magnitude[19:0];
            if ((remainder > 20'h80000) ||
                ((remainder == 20'h80000) && quotient[0]))
                quotient = quotient + 1'b1;
            round_shift_even_20 = negative ? -$signed(quotient) :
                $signed(quotient);
        end
    endfunction

    always_comb begin
        cfg_legal = (i_cfg_batch_count != '0) &&
            (i_cfg_element_count != '0);
        o_cfg_ready = !active_q && !finalize_q && !pending_q;
        o_start_ready = configured_q && !active_q && !finalize_q &&
            !pending_q && !i_cfg_valid;
        o_busy = active_q || finalize_q || pending_q;

        input_expected_last_sample =
            expected_sample_q == (batch_count_q - 1'b1);
        input_expected_tensor_last = input_expected_last_sample &&
            (expected_addr_q == (element_count_q - 1'b1));
        input_legal = (i_contribution_local_addr == expected_addr_q) &&
            (i_contribution_sample_id == expected_sample_q) &&
            (i_contribution_tensor_last == input_expected_tensor_last);
        o_contribution_ready = active_q && !finalize_q && !pending_q &&
            (input_legal || i_contribution_valid);
        input_fire = i_contribution_valid && o_contribution_ready;

        contribution_product = $signed(i_error_q40) *
            $signed(i_hidden_trace_q14);
        contribution_extended = {{(ACC_WIDTH-(ERROR_WIDTH+TRACE_WIDTH)){
            contribution_product[ERROR_WIDTH+TRACE_WIDTH-1]}},
            contribution_product};
        accumulated_with_current = accumulator_q + contribution_extended;
        rounded_shifted = round_shift_even_20(accumulator_q);
        rounded_overflow =
            rounded_shifted[ACC_WIDTH-1:GRADIENT_WIDTH] !=
            {(ACC_WIDTH-GRADIENT_WIDTH){rounded_shifted[GRADIENT_WIDTH-1]}};

        o_gradient_valid = pending_q;
        o_gradient_local_addr = gradient_addr_q;
        o_gradient_q40 = gradient_q;
        o_gradient_tensor_last = gradient_last_q;
        output_fire = o_gradient_valid && i_gradient_ready;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            configured_q <= 1'b0;
            active_q <= 1'b0;
            finalize_q <= 1'b0;
            pending_q <= 1'b0;
            batch_count_q <= '0;
            element_count_q <= '0;
            expected_addr_q <= '0;
            expected_sample_q <= '0;
            accumulator_q <= '0;
            gradient_addr_q <= '0;
            gradient_q <= '0;
            gradient_last_q <= 1'b0;
            o_cfg_error <= 1'b0;
            o_done <= 1'b0;
            o_protocol_error <= 1'b0;
            o_overflow <= 1'b0;
        end else begin
            o_cfg_error <= 1'b0;
            o_done <= 1'b0;

            if (i_cfg_valid && o_cfg_ready) begin
                if (!cfg_legal) begin
                    configured_q <= 1'b0;
                    o_cfg_error <= 1'b1;
                end else begin
                    configured_q <= 1'b1;
                    batch_count_q <= i_cfg_batch_count;
                    element_count_q <= i_cfg_element_count;
                end
            end

            if (i_start && o_start_ready) begin
                active_q <= 1'b1;
                finalize_q <= 1'b0;
                expected_addr_q <= '0;
                expected_sample_q <= '0;
                accumulator_q <= '0;
                o_protocol_error <= 1'b0;
                o_overflow <= 1'b0;
            end

            if (input_fire) begin
                if (!input_legal) begin
                    active_q <= 1'b0;
                    o_protocol_error <= 1'b1;
                    o_done <= 1'b1;
                end else if (input_expected_last_sample) begin
                    finalize_q <= 1'b1;
                    gradient_addr_q <= expected_addr_q;
                    gradient_last_q <= input_expected_tensor_last;
                    accumulator_q <= accumulated_with_current;
                    expected_sample_q <= '0;
                end else begin
                    accumulator_q <= accumulated_with_current;
                    expected_sample_q <= expected_sample_q + 1'b1;
                end
            end

            // Register the final multiply-accumulate value before Q20
            // rounding and overflow detection.
            if (finalize_q) begin
                finalize_q <= 1'b0;
                pending_q <= 1'b1;
                gradient_q <= rounded_shifted[GRADIENT_WIDTH-1:0];
                accumulator_q <= '0;
                if (rounded_overflow)
                    o_overflow <= 1'b1;
            end

            if (output_fire) begin
                pending_q <= 1'b0;
                if (gradient_last_q) begin
                    active_q <= 1'b0;
                    o_done <= 1'b1;
                end else begin
                    expected_addr_q <= expected_addr_q + 1'b1;
                end
            end

            if (i_abort && (active_q || finalize_q || pending_q)) begin
                active_q <= 1'b0;
                finalize_q <= 1'b0;
                pending_q <= 1'b0;
                accumulator_q <= '0;
                expected_addr_q <= '0;
                expected_sample_q <= '0;
                o_done <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
