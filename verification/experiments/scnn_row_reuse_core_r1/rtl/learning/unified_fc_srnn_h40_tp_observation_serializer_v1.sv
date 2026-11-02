`timescale 1ns/1ps
`default_nettype none

// Serializes committed FC/SRNN/M2 neuron records from the four physical regions.
// The 52-bit transaction tag is intentionally opaque here: topology-specific
// capture logic may encode sample, timestep, branch and layer identifiers
// without changing this shared physical boundary.
module unified_fc_srnn_h40_tp_observation_serializer_v1 (
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
    output logic [1:0]               o_scalar_mode,
    output logic [51:0]              o_scalar_transaction_tag,
    output logic [9:0]               o_scalar_neuron_id,
    output logic                     o_scalar_spike,
    output logic signed [31:0]       o_scalar_pre_state,
    output logic signed [31:0]       o_scalar_trace_next,
    output logic                     o_scalar_last_in_record,

    output logic                     o_busy,
    output logic                     o_sticky_error
);
    localparam logic [1:0] MODE_FC = 2'd0;
    localparam logic [1:0] MODE_SRNN = 2'd1;
    localparam logic [1:0] MODE_M2_Q14 = 2'd3;

    logic [1:0] round_robin_q;
    logic selected_valid;
    logic [1:0] selected_region;
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
    logic [1:0] mode_q;
    logic [51:0] transaction_tag_q;
    logic [9:0] unit_index_q;
    logic [1:0] group_q;
    logic [9:0] lane_mask_q;
    logic [9:0] spike_q;
    logic [319:0] pre_state_q;
    logic [319:0] trace_next_q;
    logic [3:0] lane_q;
    logic [3:0] first_lane;
    logic [3:0] next_lane;
    logic next_lane_valid;
    logic scalar_fire;

    // Payload is observed only while busy_q is set by a legal input_fire.
    // Sampling during idle removes the cross-system owner/quiescence chain
    // from the wide payload registers' clock enables without changing the
    // record handshake or the cycle in which a record becomes visible.
    always_ff @(posedge i_clk) begin
        if (!busy_q) begin
            pre_state_q <= selected_pre_state;
            trace_next_q <= selected_trace_next;
        end
    end

    always_comb begin
        logic found;
        integer candidate;

        selected_valid = 1'b0;
        selected_region = round_robin_q;
        found = 1'b0;
        for (integer offset = 0; offset < 4; offset++) begin
            candidate = integer'(round_robin_q) + offset;
            if (candidate >= 4)
                candidate = candidate - 4;
            if (!found && i_observation_valid[candidate]) begin
                found = 1'b1;
                selected_valid = 1'b1;
                selected_region = 2'(candidate);
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

        selected_legal =
            (selected_mode == MODE_FC || selected_mode == MODE_SRNN ||
             selected_mode == MODE_M2_Q14) &&
            selected_lane_mask != 0 &&
            (selected_range_mask & ~selected_lane_mask) == 0 &&
            (selected_spike & ~selected_lane_mask) == 0 &&
            // The record carries a layer-local logical unit index.  A layer
            // may start at a nonzero physical tile (for example the M2 Wout
            // layer at tile 7), so that index must not be compared directly
            // with the physical tile number.  Region ownership still proves
            // that the selected physical record came from the advertised
            // tile group.
            selected_tile[3:2] == selected_region;

        o_observation_ready = '0;
        if (!busy_q && !o_sticky_error && selected_valid)
            o_observation_ready[selected_region] = 1'b1;
        input_fire = selected_valid &&
            o_observation_ready[selected_region];

        o_scalar_valid = busy_q && !o_sticky_error;
        o_scalar_mode = mode_q;
        o_scalar_transaction_tag = transaction_tag_q;
        o_scalar_neuron_id = 10'(
            integer'(unit_index_q) * 40 + integer'(group_q) * 10 +
            integer'(lane_q));
        o_scalar_spike = spike_q[lane_q];
        o_scalar_pre_state =
            $signed(pre_state_q[lane_q*32 +: 32]);
        o_scalar_trace_next =
            $signed(trace_next_q[lane_q*32 +: 32]);
        o_scalar_last_in_record = !next_lane_valid;
        scalar_fire = o_scalar_valid && i_scalar_ready;
        o_busy = busy_q;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            round_robin_q <= '0;
            busy_q <= 1'b0;
            mode_q <= MODE_FC;
            transaction_tag_q <= '0;
            unit_index_q <= '0;
            group_q <= '0;
            lane_mask_q <= '0;
            spike_q <= '0;
            lane_q <= '0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            round_robin_q <= '0;
            busy_q <= 1'b0;
            lane_mask_q <= '0;
            o_sticky_error <= 1'b0;
        end else begin
            if (i_clear_error && !busy_q)
                o_sticky_error <= 1'b0;

            if (input_fire) begin
                round_robin_q <= selected_region + 1'b1;
                if (!selected_legal) begin
                    busy_q <= 1'b0;
                    o_sticky_error <= 1'b1;
                end else begin
                    busy_q <= 1'b1;
                    mode_q <= selected_mode;
                    transaction_tag_q <= selected_meta[63:12];
                    unit_index_q <= selected_meta[11:2];
                    group_q <= selected_meta[1:0];
                    lane_mask_q <= selected_lane_mask;
                    spike_q <= selected_spike;
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
endmodule

`default_nettype wire
