`timescale 1ns/1ps
`default_nettype none

// Converts one masked H40/region observation record into the scalar ordering
// used by the TP learning path. The record remains local while its active lanes
// are replayed, so learning backpressure never requires a second PE array or a
// duplicate state/trace read.
module unified_scnn_h40_tp_observation_serializer_v1 #(
    parameter int PHYSICAL_TILE_COUNT = 16
) (
    input  wire logic                 i_clk,
    input  wire logic                 i_rst_n,
    input  wire logic                 i_abort,
    input  wire logic                 i_clear_error,

    input  wire logic [3:0]           i_observation_valid,
    output logic [3:0]                o_observation_ready,
    input  wire logic [15:0]          i_observation_tile_flat,
    input  wire logic [7:0]           i_observation_mode_flat,
    input  wire logic [39:0]          i_observation_lane_mask_flat,
    input  wire logic [39:0]          i_observation_range_mask_flat,
    input  wire logic [39:0]          i_observation_spike_flat,
    input  wire logic [1279:0]        i_observation_pre_state_flat,
    input  wire logic [1279:0]        i_observation_trace_next_flat,
    input  wire logic [255:0]         i_observation_meta_flat,

    output logic                     o_scalar_valid,
    input  wire logic                i_scalar_ready,
    output logic [19:0]              o_scalar_layer_tag,
    output logic [9:0]               o_scalar_channel,
    output logic [31:0]              o_scalar_spatial,
    output logic                     o_scalar_spike,
    output logic signed [15:0]       o_scalar_pre_state_q8_8,
    output logic signed [15:0]       o_scalar_trace_next_q8_8,
    output logic                     o_scalar_last,

    output logic                     o_busy,
    output logic                     o_sticky_error
);
    localparam logic [1:0] MODE_SCNN = 2'd2;

    logic [1:0] selected_region;
    logic selected_valid;
    logic [3:0] selected_tile;
    logic [1:0] selected_mode;
    logic [9:0] selected_lane_mask;
    logic [9:0] selected_range_mask;
    logic [9:0] selected_spike;
    logic [319:0] selected_pre_state;
    logic [319:0] selected_trace_next;
    logic [63:0] selected_meta;
    logic selected_legal;
    logic input_fire;

    logic busy_q;
    logic [9:0] lane_mask_q;
    logic [9:0] spike_q;
    logic [319:0] pre_state_q;
    logic [319:0] trace_next_q;
    logic [19:0] layer_tag_q;
    logic [9:0] channel_q;
    logic [31:0] spatial_base_q;
    logic [1:0] group_q;
    logic [3:0] lane_q;
    logic [3:0] first_lane;
    logic [3:0] next_lane;
    logic next_lane_valid;
    logic scalar_fire;

    always_comb begin
        selected_region = '0;
        selected_valid = 1'b0;
        for (integer region = 0; region < 4; region++) begin
            if (i_observation_valid[region]) begin
                if (!selected_valid) begin
                    selected_valid = 1'b1;
                    selected_region = 2'(region);
                end
            end
        end

        selected_tile =
            i_observation_tile_flat[selected_region*4 +: 4];
        selected_mode =
            i_observation_mode_flat[selected_region*2 +: 2];
        selected_lane_mask =
            i_observation_lane_mask_flat[selected_region*10 +: 10];
        selected_range_mask =
            i_observation_range_mask_flat[selected_region*10 +: 10];
        selected_spike =
            i_observation_spike_flat[selected_region*10 +: 10];
        selected_pre_state =
            i_observation_pre_state_flat[selected_region*320 +: 320];
        selected_trace_next =
            i_observation_trace_next_flat[selected_region*320 +: 320];
        selected_meta =
            i_observation_meta_flat[selected_region*64 +: 64];

        first_lane = '0;
        for (integer lane = 9; lane >= 0; lane--)
            if (selected_lane_mask[lane])
                first_lane = 4'(lane);

        next_lane = '0;
        next_lane_valid = 1'b0;
        for (integer lane = 9; lane >= 0; lane--)
            if (lane > integer'(lane_q) && lane_mask_q[lane]) begin
                next_lane = 4'(lane);
                next_lane_valid = 1'b1;
            end

        selected_legal = selected_mode == MODE_SCNN &&
            selected_lane_mask != 0 &&
            (selected_range_mask & ~selected_lane_mask) == 0 &&
            (selected_spike & ~selected_lane_mask) == 0 &&
            selected_tile[3:2] == selected_region &&
            selected_tile == 4'(
                integer'(selected_meta[11:2]) % PHYSICAL_TILE_COUNT);

        o_observation_ready = '0;
        if (!i_abort && !busy_q && !o_sticky_error) begin
            // Keep the legacy ready-before-valid behavior while idle, then
            // select exactly one asserted region before the transfer edge.
            // Non-selected regions retain valid until the serializer drains
            // the selected record.
            if (selected_valid)
                o_observation_ready[selected_region] = 1'b1;
            else
                o_observation_ready = '1;
        end
        input_fire = selected_valid &&
            o_observation_ready[selected_region];

        o_scalar_valid = busy_q && !o_sticky_error && !i_abort;
        o_scalar_layer_tag = layer_tag_q;
        o_scalar_channel = channel_q;
        o_scalar_spatial = spatial_base_q +
            32'(integer'(group_q) * 10 + integer'(lane_q));
        o_scalar_spike = spike_q[lane_q];
        o_scalar_pre_state_q8_8 =
            $signed(pre_state_q[lane_q*32 +: 16]);
        o_scalar_trace_next_q8_8 =
            $signed(trace_next_q[lane_q*32 +: 16]);
        o_scalar_last = !next_lane_valid;
        scalar_fire = o_scalar_valid && i_scalar_ready;
        o_busy = busy_q;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            busy_q <= 1'b0;
            lane_mask_q <= '0;
            spike_q <= '0;
            pre_state_q <= '0;
            trace_next_q <= '0;
            layer_tag_q <= '0;
            channel_q <= '0;
            spatial_base_q <= '0;
            group_q <= '0;
            lane_q <= '0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            busy_q <= 1'b0;
            lane_mask_q <= '0;
            o_sticky_error <= 1'b0;
        end else begin
            if (i_clear_error && !busy_q)
                o_sticky_error <= 1'b0;

            if (input_fire) begin
                if (!selected_legal) begin
                    busy_q <= 1'b0;
                    o_sticky_error <= 1'b1;
                end else begin
                    busy_q <= 1'b1;
                    lane_mask_q <= selected_lane_mask;
                    spike_q <= selected_spike;
                    pre_state_q <= selected_pre_state;
                    trace_next_q <= selected_trace_next;
                    layer_tag_q <= selected_meta[63:44];
                    spatial_base_q <= selected_meta[43:12];
                    channel_q <= selected_meta[11:2];
                    group_q <= selected_meta[1:0];
                    lane_q <= first_lane;
                end
            end else if (scalar_fire) begin
                if (next_lane_valid)
                    lane_q <= next_lane;
                else
                    busy_q <= 1'b0;
            end
        end
    end

    initial begin
        if (PHYSICAL_TILE_COUNT < 1 || PHYSICAL_TILE_COUNT > 16)
            $error("SCNN TP serializer physical tile count mismatch");
    end
endmodule

`default_nettype wire
