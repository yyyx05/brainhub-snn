`timescale 1ns/1ps
`default_nettype none

// Forty destination-stationary four-term accumulation lanes. Each lane uses
// a two-DSP cascade and exposes one held result, avoiding a duplicated H40
// result bus and a later ten-lane merge stage.
module unified_dual_role_h40_tile_p4all_cascade_accum_candidate_v1 #(
    parameter int LANES = 40
) (
    input  wire logic                    i_clk,
    input  wire logic                    i_rst_n,
    input  wire logic                    i_abort,
    input  wire logic                    i_valid,
    output logic                         o_ready,
    input  wire logic                    i_first,
    input  wire logic                    i_last,
    input  wire logic [LANES-1:0]        i_lane_mask,
    input  wire logic [LANES-1:0]        i_enable0,
    input  wire logic [LANES-1:0]        i_enable1,
    input  wire logic [LANES-1:0]        i_enable2,
    input  wire logic [LANES-1:0]        i_enable3,
    input  wire logic [LANES*25-1:0]     i_term0_flat,
    input  wire logic [LANES*25-1:0]     i_term1_flat,
    input  wire logic [LANES*25-1:0]     i_term2_flat,
    input  wire logic [LANES*25-1:0]     i_term3_flat,
    input  wire logic [LANES*18-1:0]     i_factor_b_flat,
    output logic                         o_result_valid,
    input  wire logic                    i_result_ready,
    output logic [LANES-1:0]             o_result_lane_mask,
    output logic [LANES*48-1:0]          o_result_flat
);
    logic result_valid_q;
    logic [LANES-1:0] active_lane_mask_q;
    logic input_fire;

    assign o_ready = !i_abort && (!result_valid_q || i_result_ready);
    assign input_fire = i_valid && o_ready;
    assign o_result_valid = !i_abort && result_valid_q;
    assign o_result_lane_mask = active_lane_mask_q;

    generate
        for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
            unified_dual_role_dsp_lane_p4_cascade_candidate_v1 u_lane (
                .i_clk(i_clk),
                .i_fire(input_fire && i_lane_mask[lane]),
                .i_first(i_first),
                .i_enable0(i_enable0[lane]),
                .i_enable1(i_enable1[lane]),
                .i_enable2(i_enable2[lane]),
                .i_enable3(i_enable3[lane]),
                .i_term0($signed(i_term0_flat[lane*25 +: 25])),
                .i_term1($signed(i_term1_flat[lane*25 +: 25])),
                .i_term2($signed(i_term2_flat[lane*25 +: 25])),
                .i_term3($signed(i_term3_flat[lane*25 +: 25])),
                .i_factor($signed(i_factor_b_flat[lane*18 +: 18])),
                .o_result(o_result_flat[lane*48 +: 48])
            );
        end
    endgenerate

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n || i_abort) begin
            result_valid_q <= 1'b0;
            active_lane_mask_q <= '0;
        end else begin
            if (result_valid_q && i_result_ready)
                result_valid_q <= 1'b0;
            if (input_fire) begin
                if (i_first)
                    active_lane_mask_q <= i_lane_mask;
                result_valid_q <= i_last;
            end
        end
    end

    initial begin
        if (LANES != 40)
            $error("P4-all cascade accumulator requires 40 lanes");
    end
endmodule

`default_nettype wire
