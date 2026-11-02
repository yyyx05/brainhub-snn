`timescale 1ns/1ps
`default_nettype none

// Converts a shared trace/error gradient sum into a signed sub-LSB residual
// update. The small runtime multiplier lets one physical scaler serve numeric
// profiles with different trace and parameter scales:
//   delta = -RNE(gradient * scale_multiplier / 2^SCALE_SHIFT)
// The output is symmetrically saturated so positive and negative updates have
// the same representable magnitude.
module unified_tp_q015_magnitude_scaler #(
    parameter int GRADIENT_WIDTH = 48,
    parameter int DELTA_WIDTH = 18,
    parameter int MULTIPLIER_WIDTH = 16,
    parameter int SCALE_SHIFT = 26
) (
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,
    input  wire logic                              i_valid,
    output logic                                   o_ready,
    input  wire logic signed [GRADIENT_WIDTH-1:0] i_gradient,
    input  wire logic [MULTIPLIER_WIDTH-1:0]      i_scale_multiplier,
    output logic                                   o_valid,
    input  wire logic                              i_ready,
    output logic signed [DELTA_WIDTH-1:0]          o_scaled_delta,
    output logic                                   o_saturated
);

    localparam int PRODUCT_WIDTH = GRADIENT_WIDTH + MULTIPLIER_WIDTH + 1;
    localparam logic [DELTA_WIDTH-1:0] DELTA_MAGNITUDE_MAX =
        {1'b0, {(DELTA_WIDTH-1){1'b1}}};

    logic signed [PRODUCT_WIDTH-1:0] product_signed;
    logic [PRODUCT_WIDTH-1:0] product_abs;
    logic [PRODUCT_WIDTH-1:0] quotient;
    logic [SCALE_SHIFT-1:0] remainder;
    logic [PRODUCT_WIDTH-1:0] rounded_magnitude;
    logic round_up;
    logic magnitude_overflow;
    logic signed [DELTA_WIDTH-1:0] delta_next;

    logic valid_q;
    logic signed [DELTA_WIDTH-1:0] delta_q;
    logic saturated_q;

    assign o_ready = !valid_q || i_ready;
    assign o_valid = valid_q;
    assign o_scaled_delta = delta_q;
    assign o_saturated = saturated_q;

    always_comb begin
        product_signed = $signed(i_gradient) *
            $signed({1'b0, i_scale_multiplier});
        product_abs = product_signed[PRODUCT_WIDTH-1] ?
            $unsigned(-product_signed) : $unsigned(product_signed);
        quotient = product_abs >> SCALE_SHIFT;
        remainder = product_abs[SCALE_SHIFT-1:0];
        round_up = (remainder > (SCALE_SHIFT)'(1 << (SCALE_SHIFT-1))) ||
            ((remainder == (SCALE_SHIFT)'(1 << (SCALE_SHIFT-1))) &&
             quotient[0]);
        rounded_magnitude = quotient + PRODUCT_WIDTH'(round_up);
        magnitude_overflow = rounded_magnitude >
            PRODUCT_WIDTH'(DELTA_MAGNITUDE_MAX);

        if (rounded_magnitude == '0) begin
            delta_next = '0;
        end else if (product_signed[PRODUCT_WIDTH-1]) begin
            delta_next = magnitude_overflow ?
                $signed(DELTA_MAGNITUDE_MAX) :
                $signed(rounded_magnitude[DELTA_WIDTH-1:0]);
        end else begin
            delta_next = magnitude_overflow ?
                -$signed(DELTA_MAGNITUDE_MAX) :
                -$signed(rounded_magnitude[DELTA_WIDTH-1:0]);
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            valid_q <= 1'b0;
            delta_q <= '0;
            saturated_q <= 1'b0;
        end else begin
            if (valid_q && i_ready)
                valid_q <= 1'b0;
            if (i_valid && o_ready) begin
                valid_q <= 1'b1;
                delta_q <= delta_next;
                saturated_q <= magnitude_overflow;
            end
        end
    end

endmodule

`default_nettype wire
