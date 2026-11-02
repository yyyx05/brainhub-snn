`timescale 1ns/1ps
`default_nettype none

// Converts one emitted output-layer gradient into the shared RMW request
// format. Legacy mode forwards only the descent direction. Magnitude mode
// scales the full accumulator and retains its signed sub-LSB delta.
module unified_fc_tp_gradient_request_adapter #(
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int GRADIENT_WIDTH = 48,
    parameter int DELTA_WIDTH = 18
) (
    input  wire logic                                  i_clk,
    input  wire logic                                  i_rst_n,
    input  wire logic                                  i_active,
    input  wire logic [TENSOR_ID_WIDTH-1:0]            i_active_tensor_id,
    input  wire logic                                  i_magnitude_mode,
    input  wire logic [15:0]                           i_magnitude_multiplier,

    input  wire logic                                  i_gradient_valid,
    output logic                                       o_gradient_ready,
    input  wire logic [TENSOR_ID_WIDTH-1:0]            i_gradient_tensor_id,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]           i_gradient_local_parameter_addr,
    input  wire logic [TAG_WIDTH-1:0]                  i_gradient_tag,
    input  wire logic signed [1:0]                     i_gradient_sign,
    input  wire logic signed [GRADIENT_WIDTH-1:0]      i_gradient_accumulator,

    output logic                                       o_req_valid,
    input  wire logic                                  i_req_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]                o_req_local_parameter_addr,
    output logic [TAG_WIDTH-1:0]                       o_req_tag,
    output logic signed [1:0]                          o_req_update_direction,
    output logic                                       o_req_magnitude_mode,
    output logic signed [DELTA_WIDTH-1:0]              o_req_scaled_residual_delta,

    output logic                                       o_update_accepted,
    output logic                                       o_zero_skipped,
    output logic                                       o_tensor_error,
    output logic                                       o_sign_error,
    output logic                                       o_scale_saturated,
    output logic                                       o_busy
);

    logic tensor_match;
    logic sign_legal;
    logic sign_nonzero;
    logic scaler_input_valid;
    logic scaler_input_ready;
    logic scaler_output_valid;
    logic scaler_output_ready;
    logic signed [DELTA_WIDTH-1:0] scaler_delta;
    logic scaler_saturated;
    logic magnitude_zero;
    logic magnitude_metadata_fire;
    logic [PARAM_ADDR_WIDTH-1:0] magnitude_addr_q;
    logic [TAG_WIDTH-1:0] magnitude_tag_q;

    always_comb begin
        tensor_match = i_active &&
            (i_gradient_tensor_id == i_active_tensor_id);
        sign_legal = (i_gradient_sign == -2'sd1) ||
            (i_gradient_sign == 2'sd0) ||
            (i_gradient_sign == 2'sd1);
        sign_nonzero = i_gradient_sign != 2'sd0;
        scaler_input_valid = i_gradient_valid && i_magnitude_mode &&
            tensor_match && sign_legal;
        magnitude_metadata_fire = scaler_input_valid && scaler_input_ready;
        magnitude_zero = scaler_output_valid && (scaler_delta == '0);
        scaler_output_ready = magnitude_zero || i_req_ready;

        o_gradient_ready = i_magnitude_mode ? scaler_input_ready : i_req_ready;
        o_req_valid = i_magnitude_mode ?
            (scaler_output_valid && !magnitude_zero) :
            (i_gradient_valid && tensor_match && sign_legal && sign_nonzero);
        o_req_local_parameter_addr = i_magnitude_mode ?
            magnitude_addr_q : i_gradient_local_parameter_addr;
        o_req_tag = i_magnitude_mode ? magnitude_tag_q : i_gradient_tag;
        o_req_update_direction = i_magnitude_mode ? 2'sd0 : -i_gradient_sign;
        o_req_magnitude_mode = i_magnitude_mode;
        o_req_scaled_residual_delta = i_magnitude_mode ? scaler_delta : '0;
        o_busy = i_magnitude_mode &&
            (scaler_input_valid || scaler_output_valid);
    end

    unified_tp_q015_magnitude_scaler #(
        .GRADIENT_WIDTH(GRADIENT_WIDTH),
        .DELTA_WIDTH(DELTA_WIDTH)
    ) magnitude_scaler (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_valid(scaler_input_valid),
        .o_ready(scaler_input_ready),
        .i_gradient(i_gradient_accumulator),
        .i_scale_multiplier(i_magnitude_multiplier),
        .o_valid(scaler_output_valid),
        .i_ready(scaler_output_ready),
        .o_scaled_delta(scaler_delta),
        .o_saturated(scaler_saturated)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            magnitude_addr_q <= '0;
            magnitude_tag_q <= '0;
            o_update_accepted <= 1'b0;
            o_zero_skipped <= 1'b0;
            o_tensor_error <= 1'b0;
            o_sign_error <= 1'b0;
            o_scale_saturated <= 1'b0;
        end else begin
            o_update_accepted <= 1'b0;
            o_zero_skipped <= 1'b0;
            o_tensor_error <= 1'b0;
            o_sign_error <= 1'b0;
            o_scale_saturated <= 1'b0;

            if (magnitude_metadata_fire) begin
                magnitude_addr_q <= i_gradient_local_parameter_addr;
                magnitude_tag_q <= i_gradient_tag;
            end

            if (i_magnitude_mode) begin
                o_update_accepted <= scaler_output_valid &&
                    scaler_output_ready && !magnitude_zero;
                o_zero_skipped <= scaler_output_valid &&
                    scaler_output_ready && magnitude_zero;
                o_scale_saturated <= scaler_output_valid &&
                    scaler_output_ready && scaler_saturated;
                o_tensor_error <= i_gradient_valid && scaler_input_ready &&
                    !tensor_match;
                o_sign_error <= i_gradient_valid && scaler_input_ready &&
                    tensor_match && !sign_legal;
            end else begin
                o_update_accepted <= i_gradient_valid && i_req_ready &&
                    tensor_match && sign_legal && sign_nonzero;
                o_zero_skipped <= i_gradient_valid && i_req_ready &&
                    tensor_match && sign_legal && !sign_nonzero;
                o_tensor_error <= i_gradient_valid && i_req_ready &&
                    !tensor_match;
                o_sign_error <= i_gradient_valid && i_req_ready &&
                    tensor_match && !sign_legal;
            end
        end
    end

endmodule

`default_nettype wire
