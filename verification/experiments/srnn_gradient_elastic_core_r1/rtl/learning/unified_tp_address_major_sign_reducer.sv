`timescale 1ns/1ps
`default_nettype none

// Constant-state outer-product reducer for a nondecreasing parameter-address
// contribution stream. Only the current address accumulator is retained; a
// change of address emits one nonzero gradient sign before accepting the next
// address. Missing/zero-gradient addresses require no RMW transaction.
module unified_tp_address_major_sign_reducer #(
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int SOURCE_WIDTH = 16,
    parameter int ERROR_WIDTH = 18,
    parameter int ACC_WIDTH = 48,
    parameter int MAX_ELEMENTS = 32768
) (
    input  logic                                     i_clk,
    input  logic                                     i_rst_n,

    input  logic                                     i_cfg_valid,
    output logic                                     o_cfg_ready,
    input  logic [TENSOR_ID_WIDTH-1:0]               i_cfg_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]              i_cfg_element_count,
    output logic                                     o_cfg_error,
    output logic                                     o_active,
    output logic [TENSOR_ID_WIDTH-1:0]               o_active_tensor_id,
    input  logic                                     i_release,
    output logic                                     o_release_ready,

    input  logic                                     i_contribution_valid,
    output logic                                     o_contribution_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]              i_contribution_local_addr,
    input  logic [SOURCE_WIDTH-1:0]                  i_contribution_source_trace,
    input  logic signed [ERROR_WIDTH-1:0]            i_contribution_error,
    output logic                                     o_contribution_error,
    output logic                                     o_accumulator_overflow,

    input  logic                                     i_emit_start,
    output logic                                     o_emit_ready,
    output logic                                     o_sign_valid,
    input  logic                                     i_sign_ready,
    output logic [TENSOR_ID_WIDTH-1:0]               o_sign_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]              o_sign_local_addr,
    output logic signed [1:0]                        o_gradient_sign,
    output logic signed [ACC_WIDTH-1:0]              o_gradient_accumulator,
    output logic                                     o_emit_done,
    output logic                                     o_busy
);

    localparam int PRODUCT_WIDTH = SOURCE_WIDTH + 1 + ERROR_WIDTH;

    logic [TENSOR_ID_WIDTH-1:0] tensor_id_q;
    logic [PARAM_ADDR_WIDTH-1:0] element_count_q;
    logic have_accumulator_q;
    logic [PARAM_ADDR_WIDTH-1:0] accumulator_addr_q;
    logic signed [ACC_WIDTH-1:0] accumulator_q;
    logic sign_pending_q;
    logic sign_ends_window_q;
    logic [PARAM_ADDR_WIDTH-1:0] sign_addr_q;
    logic signed [ACC_WIDTH-1:0] sign_accumulator_q;

    logic contribution_fire;
    logic sign_fire;
    logic emit_fire;
    logic contribution_in_range;
    logic contribution_same_address;
    logic contribution_forward_address;
    logic contribution_reverse_address;
    logic signed [PRODUCT_WIDTH-1:0] contribution_product;
    logic signed [ACC_WIDTH:0] sum_extended;
    logic signed [ACC_WIDTH:0] accumulator_max_extended;
    logic signed [ACC_WIDTH:0] accumulator_min_extended;
    logic signed [ACC_WIDTH-1:0] accumulated_value;
    logic accumulated_overflow;

    always_comb begin
        contribution_in_range =
            i_contribution_local_addr < element_count_q;
        contribution_same_address = have_accumulator_q &&
            (i_contribution_local_addr == accumulator_addr_q);
        contribution_forward_address = have_accumulator_q &&
            (i_contribution_local_addr > accumulator_addr_q);
        contribution_reverse_address = have_accumulator_q &&
            (i_contribution_local_addr < accumulator_addr_q);

        o_cfg_ready = !o_active && !have_accumulator_q && !sign_pending_q;
        o_release_ready = o_active && !have_accumulator_q &&
            !sign_pending_q;
        o_emit_ready = o_active && !sign_pending_q;
        o_contribution_ready = o_active && !sign_pending_q && !i_emit_start &&
            (!have_accumulator_q || contribution_same_address ||
             contribution_reverse_address || !contribution_in_range);
        contribution_fire = i_contribution_valid && o_contribution_ready;
        emit_fire = i_emit_start && o_emit_ready;

        o_sign_valid = sign_pending_q;
        sign_fire = o_sign_valid && i_sign_ready;
        o_sign_tensor_id = tensor_id_q;
        o_sign_local_addr = sign_addr_q;
        o_gradient_accumulator = sign_accumulator_q;
        o_gradient_sign = (sign_accumulator_q > 0) ? 2'sd1 : -2'sd1;
        o_active_tensor_id = tensor_id_q;
        o_busy = sign_pending_q;

        contribution_product =
            $signed({1'b0, i_contribution_source_trace}) *
            $signed(i_contribution_error);
        sum_extended = $signed({accumulator_q[ACC_WIDTH-1], accumulator_q}) +
            $signed({{(ACC_WIDTH + 1 - PRODUCT_WIDTH){
                contribution_product[PRODUCT_WIDTH-1]}}, contribution_product});
        accumulator_max_extended =
            $signed({2'b00, {(ACC_WIDTH-1){1'b1}}});
        accumulator_min_extended =
            $signed({2'b11, {(ACC_WIDTH-1){1'b0}}});
        accumulated_overflow = 1'b0;
        if (sum_extended > accumulator_max_extended) begin
            accumulated_value = {1'b0, {(ACC_WIDTH-1){1'b1}}};
            accumulated_overflow = 1'b1;
        end else if (sum_extended < accumulator_min_extended) begin
            accumulated_value = {1'b1, {(ACC_WIDTH-1){1'b0}}};
            accumulated_overflow = 1'b1;
        end else begin
            accumulated_value = sum_extended[ACC_WIDTH-1:0];
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            tensor_id_q <= '0;
            element_count_q <= '0;
            have_accumulator_q <= 1'b0;
            accumulator_addr_q <= '0;
            accumulator_q <= '0;
            sign_pending_q <= 1'b0;
            sign_ends_window_q <= 1'b0;
            sign_addr_q <= '0;
            sign_accumulator_q <= '0;
            o_cfg_error <= 1'b0;
            o_active <= 1'b0;
            o_contribution_error <= 1'b0;
            o_accumulator_overflow <= 1'b0;
            o_emit_done <= 1'b0;
        end else begin
            o_cfg_error <= 1'b0;
            o_contribution_error <= 1'b0;
            o_accumulator_overflow <= 1'b0;
            o_emit_done <= 1'b0;

            if (i_cfg_valid && o_cfg_ready) begin
                if ((i_cfg_element_count == '0) ||
                    ({1'b0, i_cfg_element_count} >
                     (PARAM_ADDR_WIDTH+1)'(MAX_ELEMENTS))) begin
                    o_cfg_error <= 1'b1;
                end else begin
                    tensor_id_q <= i_cfg_tensor_id;
                    element_count_q <= i_cfg_element_count;
                    o_active <= 1'b1;
                end
            end

            if (i_release && o_release_ready) begin
                tensor_id_q <= '0;
                element_count_q <= '0;
                o_active <= 1'b0;
            end

            if (contribution_fire) begin
                if (!contribution_in_range || contribution_reverse_address) begin
                    o_contribution_error <= 1'b1;
                end else if (!have_accumulator_q) begin
                    have_accumulator_q <= 1'b1;
                    accumulator_addr_q <= i_contribution_local_addr;
                    accumulator_q <= {{(ACC_WIDTH-PRODUCT_WIDTH){
                        contribution_product[PRODUCT_WIDTH-1]}},
                        contribution_product};
                end else begin
                    accumulator_q <= accumulated_value;
                    if (accumulated_overflow)
                        o_accumulator_overflow <= 1'b1;
                end
            end else if (i_contribution_valid && o_active &&
                         !sign_pending_q && contribution_in_range &&
                         contribution_forward_address) begin
                if (accumulator_q == '0) begin
                    have_accumulator_q <= 1'b0;
                    accumulator_q <= '0;
                end else begin
                    sign_pending_q <= 1'b1;
                    sign_ends_window_q <= 1'b0;
                    sign_addr_q <= accumulator_addr_q;
                    sign_accumulator_q <= accumulator_q;
                end
            end

            if (emit_fire) begin
                if (have_accumulator_q && (accumulator_q != '0)) begin
                    sign_pending_q <= 1'b1;
                    sign_ends_window_q <= 1'b1;
                    sign_addr_q <= accumulator_addr_q;
                    sign_accumulator_q <= accumulator_q;
                end else begin
                    have_accumulator_q <= 1'b0;
                    accumulator_q <= '0;
                    o_emit_done <= 1'b1;
                end
            end

            if (sign_fire) begin
                sign_pending_q <= 1'b0;
                have_accumulator_q <= 1'b0;
                accumulator_q <= '0;
                if (sign_ends_window_q)
                    o_emit_done <= 1'b1;
                sign_ends_window_q <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
