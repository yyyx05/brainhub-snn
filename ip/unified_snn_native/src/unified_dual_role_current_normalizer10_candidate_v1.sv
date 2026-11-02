`timescale 1ns/1ps
`default_nettype none

// One elastic normalization stage between the 48-bit contribution
// accumulators and the common 32-bit LIF backend. FC preserves modulo-ACC18
// behavior, SRNN narrows with signed saturation, and SCNN converts the raw
// feature/weight product to Q8.8. Payload registers intentionally have no reset;
// the resettable valid bit is their sole ownership authority.
module unified_dual_role_current_normalizer10_candidate_v1 #(
    parameter int LANES = 10,
    parameter int SCNN_WEIGHT_FRACTIONAL_BITS = 8
) (
    input  wire logic                    i_clk,
    input  wire logic                    i_rst_n,
    input  wire logic                    i_abort,
    input  wire logic                    i_valid,
    output logic                         o_ready,
    input  wire logic [1:0]              i_mode,
    input  wire logic [LANES*48-1:0]     i_raw_current_flat,
    output logic                         o_valid,
    input  wire logic                    i_ready,
    output logic [1:0]                   o_mode,
    output logic [LANES*32-1:0]          o_current_flat,
    output logic [LANES-1:0]             o_range_event_mask,
    output logic                         o_mode_error
);
    localparam logic [1:0] MODE_FC = 2'd0;
    localparam logic [1:0] MODE_SRNN = 2'd1;
    localparam logic [1:0] MODE_SCNN = 2'd2;
    localparam logic [1:0] MODE_M2_Q14 = 2'd3;

    logic valid_q;
    logic [1:0] normalized_mode_next;
    logic [LANES*32-1:0] normalized_current_next;
    logic [LANES-1:0] range_event_next;
    logic mode_error_next;
    logic input_fire;

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

    assign o_ready = !i_abort && (!valid_q || i_ready);
    assign o_valid = !i_abort && valid_q;
    assign input_fire = i_valid && o_ready;

    always_comb begin
        normalized_mode_next = i_mode;
        normalized_current_next = '0;
        range_event_next = '0;
        mode_error_next = 1'b0;

        for (int lane = 0; lane < LANES; lane++) begin
            logic signed [47:0] raw_current;
            logic signed [47:0] scnn_shifted;

            raw_current = $signed(
                i_raw_current_flat[lane*48 +: 48]);
            scnn_shifted = raw_current >>> SCNN_WEIGHT_FRACTIONAL_BITS;

            if (i_mode == MODE_FC) begin
                normalized_current_next[lane*32 +: 32] =
                    {{14{raw_current[17]}}, raw_current[17:0]};
                range_event_next[lane] =
                    raw_current[47:18] != {30{raw_current[17]}};
            end else if (i_mode == MODE_SRNN || i_mode == MODE_M2_Q14) begin
                normalized_current_next[lane*32 +: 32] =
                    saturate48_to32(raw_current);
                range_event_next[lane] =
                    (raw_current > 48'sh00007fffffff) ||
                    (raw_current < -48'sh000080000000);
            end else if (i_mode == MODE_SCNN) begin
                if (scnn_shifted > 48'sd32767) begin
                    normalized_current_next[lane*32 +: 32] =
                        {{16{1'b0}}, 16'sh7fff};
                    range_event_next[lane] = 1'b1;
                end else if (scnn_shifted < -48'sd32768) begin
                    normalized_current_next[lane*32 +: 32] =
                        {{16{1'b1}}, 16'sh8000};
                    range_event_next[lane] = 1'b1;
                end else begin
                    normalized_current_next[lane*32 +: 32] =
                        {{16{scnn_shifted[15]}}, scnn_shifted[15:0]};
                end
            end else begin
                normalized_mode_next = MODE_FC;
                mode_error_next = 1'b1;
            end
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n || i_abort) begin
            valid_q <= 1'b0;
        end else begin
            if (valid_q && i_ready)
                valid_q <= 1'b0;
            if (input_fire)
                valid_q <= 1'b1;
        end
    end

    always_ff @(posedge i_clk) begin
        if (input_fire) begin
            o_mode <= normalized_mode_next;
            o_current_flat <= normalized_current_next;
            o_range_event_mask <= range_event_next;
            o_mode_error <= mode_error_next;
        end
    end

    initial begin
        if (LANES != 10)
            $error("current normalizer candidate requires ten lanes");
    end
endmodule

`default_nettype wire
