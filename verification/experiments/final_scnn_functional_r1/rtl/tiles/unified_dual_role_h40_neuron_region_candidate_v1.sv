`timescale 1ns/1ps
`default_nettype none

// Four destination-stationary H40 tiles share one fully elastic
// normalize -> LIF -> trace service. Each tile retains its accepted request
// until ready, so the region contains no per-tile input FIFO. State and trace
// commits emerge together as one transaction, preserving their common
// timestep/version boundary under arbitrary downstream backpressure.
module unified_dual_role_h40_neuron_region_candidate_v1 #(
    parameter int REGION_TILES = 4,
    parameter int META_WIDTH = 64,
    parameter int SCNN_WEIGHT_FRACTIONAL_BITS = 8,
    parameter bit CUT_RECORD_READY = 1'b0
) (
    input  wire logic                                i_clk,
    input  wire logic                                i_rst_n,
    input  wire logic                                i_abort,

    input  wire logic [REGION_TILES-1:0]             i_tile_valid,
    output logic [REGION_TILES-1:0]                  o_tile_ready,
    input  wire logic [REGION_TILES*2-1:0]           i_tile_mode_flat,
    input  wire logic [REGION_TILES*10-1:0]          i_tile_lane_mask_flat,
    input  wire logic [REGION_TILES*480-1:0]         i_tile_raw_current_flat,
    input  wire logic [REGION_TILES*320-1:0]         i_tile_state_flat,
    input  wire logic [REGION_TILES*320-1:0]         i_tile_trace_old_flat,
    input  wire logic [REGION_TILES*16-1:0]          i_tile_leak_flat,
    input  wire logic [REGION_TILES*32-1:0]          i_tile_threshold_flat,
    input  wire logic [REGION_TILES-1:0]             i_tile_reset_to_zero,
    input  wire logic [REGION_TILES*16-1:0]          i_tile_trace_decay_flat,
    input  wire logic [REGION_TILES*META_WIDTH-1:0]  i_tile_meta_flat,

    output logic                                     o_record_valid,
    input  wire logic                                i_record_ready,
    output logic [$clog2(REGION_TILES)-1:0]          o_record_tile,
    output logic [1:0]                               o_record_mode,
    output logic [9:0]                               o_record_lane_mask,
    output logic [9:0]                               o_record_range_mask,
    output logic [9:0]                               o_record_spike,
    output logic [319:0]                             o_record_pre_state,
    output logic [319:0]                             o_record_next_state,
    output logic [319:0]                             o_record_trace_next,
    output logic [META_WIDTH-1:0]                    o_record_meta,
    output logic                                     o_record_mode_error,
    output logic                                     o_idle
);
    localparam int TILE_ID_WIDTH = $clog2(REGION_TILES);
    localparam int NORMALIZER_SIDEBAND_WIDTH =
        TILE_ID_WIDTH + 10 + 320 + 320 + 16 + 32 + 1 + 16 + META_WIDTH;
    localparam int LIF_SIDEBAND_WIDTH =
        TILE_ID_WIDTH + 2 + 10 + 10 + 320 + 16 + META_WIDTH + 1;
    localparam int TRACE_SIDEBAND_WIDTH =
        TILE_ID_WIDTH + 2 + 10 + 10 + 10 + 320 + 320 + META_WIDTH + 1;
    localparam int RECORD_WIDTH = TILE_ID_WIDTH + 2 + 10 + 10 + 10 +
        320 + 320 + 320 + META_WIDTH + 1;

    logic [TILE_ID_WIDTH-1:0] round_robin_q;
    logic [3:0] inflight_q;
    logic selected_valid;
    logic [TILE_ID_WIDTH-1:0] selected_tile;
    logic [1:0] selected_mode;
    logic [9:0] selected_lane_mask;
    logic [479:0] selected_raw_current;
    logic [319:0] selected_state;
    logic [319:0] selected_trace_old;
    logic signed [15:0] selected_leak;
    logic signed [31:0] selected_threshold;
    logic selected_reset_to_zero;
    logic [15:0] selected_trace_decay;
    logic [META_WIDTH-1:0] selected_meta;
    logic selected_fire;

    logic normalizer_input_ready;
    logic normalizer_valid;
    logic normalizer_output_ready;
    logic [1:0] normalizer_mode;
    logic [319:0] normalizer_current;
    logic [9:0] normalizer_range_mask;
    logic normalizer_mode_error;

    logic normalizer_side_input_ready;
    logic normalizer_side_valid;
    logic [NORMALIZER_SIDEBAND_WIDTH-1:0] normalizer_side_input;
    logic [NORMALIZER_SIDEBAND_WIDTH-1:0] normalizer_side_output;
    logic [TILE_ID_WIDTH-1:0] normalizer_side_tile;
    logic [9:0] normalizer_side_lane_mask;
    logic [319:0] normalizer_side_state;
    logic [319:0] normalizer_side_trace_old;
    logic signed [15:0] normalizer_side_leak;
    logic signed [31:0] normalizer_side_threshold;
    logic normalizer_side_reset_to_zero;
    logic [15:0] normalizer_side_trace_decay;
    logic [META_WIDTH-1:0] normalizer_side_meta;

    logic lif_input_ready;
    logic lif_valid;
    logic lif_output_ready;
    logic [9:0] lif_spike;
    logic [319:0] lif_pre_state;
    logic [319:0] lif_next_state;
    logic lif_side_input_ready;
    logic lif_side_valid;
    logic [LIF_SIDEBAND_WIDTH-1:0] lif_side_input;
    logic [LIF_SIDEBAND_WIDTH-1:0] lif_side_output;
    logic [TILE_ID_WIDTH-1:0] lif_side_tile;
    logic [1:0] lif_side_mode;
    logic [9:0] lif_side_lane_mask;
    logic [9:0] lif_side_range_mask;
    logic [319:0] lif_side_trace_old;
    logic [15:0] lif_side_trace_decay;
    logic [META_WIDTH-1:0] lif_side_meta;
    logic lif_side_mode_error;

    logic trace_input_ready;
    logic trace_valid;
    logic [319:0] trace_next;
    logic trace_mode_error;
    logic trace_side_input_ready;
    logic trace_side_valid;
    logic [TRACE_SIDEBAND_WIDTH-1:0] trace_side_input;
    logic [TRACE_SIDEBAND_WIDTH-1:0] trace_side_output;
    logic [TILE_ID_WIDTH-1:0] trace_side_tile;
    logic [1:0] trace_side_mode;
    logic [9:0] trace_side_lane_mask;
    logic [9:0] trace_side_range_mask;
    logic [9:0] trace_side_spike;
    logic [319:0] trace_side_pre_state;
    logic [319:0] trace_side_next_state;
    logic [META_WIDTH-1:0] trace_side_meta;
    logic trace_side_mode_error;
    logic record_input_valid;
    logic record_input_ready;
    logic [RECORD_WIDTH-1:0] record_input_data;
    logic [RECORD_WIDTH-1:0] record_output_data;
    logic record_fire;

    always_comb begin
        logic found;
        int unsigned candidate;

        selected_valid = 1'b0;
        selected_tile = round_robin_q;
        selected_mode = '0;
        selected_lane_mask = '0;
        selected_raw_current = '0;
        selected_state = '0;
        selected_trace_old = '0;
        selected_leak = '0;
        selected_threshold = '0;
        selected_reset_to_zero = 1'b0;
        selected_trace_decay = '0;
        selected_meta = '0;
        o_tile_ready = '0;
        found = 1'b0;

        for (int offset = 0; offset < REGION_TILES; offset++) begin
            candidate = 32'(round_robin_q) + offset;
            if (candidate >= REGION_TILES)
                candidate = candidate - REGION_TILES;
            if (!found && i_tile_valid[candidate]) begin
                found = 1'b1;
                selected_valid = 1'b1;
                selected_tile = TILE_ID_WIDTH'(candidate);
            end
        end

        if (selected_valid) begin
            selected_mode = i_tile_mode_flat[selected_tile*2 +: 2];
            selected_lane_mask =
                i_tile_lane_mask_flat[selected_tile*10 +: 10];
            selected_raw_current =
                i_tile_raw_current_flat[selected_tile*480 +: 480];
            selected_state =
                i_tile_state_flat[selected_tile*320 +: 320];
            selected_trace_old =
                i_tile_trace_old_flat[selected_tile*320 +: 320];
            selected_leak =
                $signed(i_tile_leak_flat[selected_tile*16 +: 16]);
            selected_threshold =
                $signed(i_tile_threshold_flat[selected_tile*32 +: 32]);
            selected_reset_to_zero = i_tile_reset_to_zero[selected_tile];
            selected_trace_decay =
                i_tile_trace_decay_flat[selected_tile*16 +: 16];
            selected_meta =
                i_tile_meta_flat[selected_tile*META_WIDTH +: META_WIDTH];
            o_tile_ready[selected_tile] = normalizer_input_ready &&
                normalizer_side_input_ready && !i_abort;
        end
    end

    assign selected_fire = selected_valid && o_tile_ready[selected_tile];
    assign normalizer_side_input = {
        selected_tile, selected_lane_mask, selected_state,
        selected_trace_old, selected_leak, selected_threshold,
        selected_reset_to_zero, selected_trace_decay, selected_meta
    };
    assign {
        normalizer_side_tile, normalizer_side_lane_mask,
        normalizer_side_state, normalizer_side_trace_old,
        normalizer_side_leak, normalizer_side_threshold,
        normalizer_side_reset_to_zero, normalizer_side_trace_decay,
        normalizer_side_meta
    } = normalizer_side_output;

    assign normalizer_output_ready = lif_input_ready &&
        lif_side_input_ready;
    assign lif_side_input = {
        normalizer_side_tile, normalizer_mode,
        normalizer_side_lane_mask, normalizer_range_mask,
        normalizer_side_trace_old, normalizer_side_trace_decay,
        normalizer_side_meta, normalizer_mode_error
    };
    assign {
        lif_side_tile, lif_side_mode, lif_side_lane_mask,
        lif_side_range_mask, lif_side_trace_old, lif_side_trace_decay,
        lif_side_meta, lif_side_mode_error
    } = lif_side_output;

    assign lif_output_ready = trace_input_ready &&
        trace_side_input_ready;
    assign trace_side_input = {
        lif_side_tile, lif_side_mode, lif_side_lane_mask,
        lif_side_range_mask, lif_spike, lif_pre_state, lif_next_state,
        lif_side_meta, lif_side_mode_error
    };
    assign {
        trace_side_tile, trace_side_mode, trace_side_lane_mask,
        trace_side_range_mask, trace_side_spike, trace_side_pre_state,
        trace_side_next_state, trace_side_meta, trace_side_mode_error
    } = trace_side_output;

    assign record_input_valid = !i_abort && trace_valid && trace_side_valid;
    assign record_input_data = {
        trace_side_tile, trace_side_mode, trace_side_lane_mask,
        trace_side_range_mask, trace_side_spike, trace_side_pre_state,
        trace_side_next_state, trace_next, trace_side_meta,
        trace_side_mode_error || trace_mode_error
    };
    assign record_fire = o_record_valid && i_record_ready;
    assign o_idle = !selected_valid && (inflight_q == 0);

    generate
        if (CUT_RECORD_READY) begin : g_record_ready_cut
            unified_elastic_fifo2 #(
                .DATA_WIDTH(RECORD_WIDTH)
            ) u_record_fifo (
                .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
                .i_valid(record_input_valid), .o_ready(record_input_ready),
                .i_data(record_input_data), .o_valid(o_record_valid),
                .i_ready(i_record_ready), .o_data(record_output_data)
            );

            assign {
                o_record_tile, o_record_mode, o_record_lane_mask,
                o_record_range_mask, o_record_spike, o_record_pre_state,
                o_record_next_state, o_record_trace_next, o_record_meta,
                o_record_mode_error
            } = record_output_data;
        end else begin : g_direct_record
            assign record_input_ready = i_record_ready;
            assign o_record_valid = record_input_valid;
            assign record_output_data = record_input_data;
            assign {
                o_record_tile, o_record_mode, o_record_lane_mask,
                o_record_range_mask, o_record_spike, o_record_pre_state,
                o_record_next_state, o_record_trace_next, o_record_meta,
                o_record_mode_error
            } = record_output_data;
        end
    endgenerate

    unified_dual_role_current_normalizer10_candidate_v1 #(
        .SCNN_WEIGHT_FRACTIONAL_BITS(SCNN_WEIGHT_FRACTIONAL_BITS)
    ) u_normalizer (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_valid(selected_valid && normalizer_side_input_ready),
        .o_ready(normalizer_input_ready), .i_mode(selected_mode),
        .i_raw_current_flat(selected_raw_current),
        .o_valid(normalizer_valid), .i_ready(normalizer_output_ready),
        .o_mode(normalizer_mode), .o_current_flat(normalizer_current),
        .o_range_event_mask(normalizer_range_mask),
        .o_mode_error(normalizer_mode_error)
    );

    unified_elastic_pipeline #(
        .DATA_WIDTH(NORMALIZER_SIDEBAND_WIDTH), .STAGES(1)
    ) u_normalizer_sideband (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_valid(selected_valid && normalizer_input_ready),
        .o_ready(normalizer_side_input_ready),
        .i_data(normalizer_side_input), .o_valid(normalizer_side_valid),
        .i_ready(normalizer_output_ready), .o_data(normalizer_side_output)
    );

    unified_dual_role_lif10_arithmetic_candidate_v3 u_lif (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_valid(normalizer_valid && normalizer_side_valid &&
                 lif_side_input_ready),
        .o_ready(lif_input_ready), .i_mode(normalizer_mode),
        .i_current_flat(normalizer_current),
        .i_state_flat(normalizer_side_state),
        .i_leak(normalizer_side_leak),
        .i_threshold(normalizer_side_threshold),
        .i_reset_to_zero(normalizer_side_reset_to_zero),
        .o_valid(lif_valid), .i_ready(lif_output_ready),
        .o_spike(lif_spike), .o_pre_state_flat(lif_pre_state),
        .o_next_state_flat(lif_next_state)
    );

    unified_elastic_pipeline #(
        .DATA_WIDTH(LIF_SIDEBAND_WIDTH), .STAGES(3)
    ) u_lif_sideband (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_valid(normalizer_valid && normalizer_side_valid &&
                 lif_input_ready),
        .o_ready(lif_side_input_ready), .i_data(lif_side_input),
        .o_valid(lif_side_valid), .i_ready(lif_output_ready),
        .o_data(lif_side_output)
    );

    unified_dual_role_trace10_arithmetic_candidate_v1 u_trace (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_valid(lif_valid && lif_side_valid && trace_side_input_ready),
        .o_ready(trace_input_ready), .i_mode(lif_side_mode),
        .i_lane_mask(lif_side_lane_mask), .i_spike(lif_spike),
        .i_trace_old_flat(lif_side_trace_old),
        .i_trace_decay(lif_side_trace_decay),
        .o_valid(trace_valid),
        .i_ready(record_input_ready && trace_side_valid),
        .o_trace_next_flat(trace_next),
        .o_mode_error(trace_mode_error)
    );

    unified_elastic_pipeline #(
        .DATA_WIDTH(TRACE_SIDEBAND_WIDTH), .STAGES(2)
    ) u_trace_sideband (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_valid(lif_valid && lif_side_valid && trace_input_ready),
        .o_ready(trace_side_input_ready), .i_data(trace_side_input),
        .o_valid(trace_side_valid),
        .i_ready(record_input_ready && trace_valid),
        .o_data(trace_side_output)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n || i_abort) begin
            round_robin_q <= '0;
            inflight_q <= '0;
        end else begin
            case ({selected_fire, record_fire})
                2'b10: inflight_q <= inflight_q + 1'b1;
                2'b01: inflight_q <= inflight_q - 1'b1;
                default: inflight_q <= inflight_q;
            endcase

            if (selected_fire) begin
                if (selected_tile == TILE_ID_WIDTH'(REGION_TILES - 1))
                    round_robin_q <= '0;
                else
                    round_robin_q <= selected_tile + 1'b1;
            end
        end
    end

    initial begin
        if (REGION_TILES != 4 || META_WIDTH != 64)
            $error("neuron region candidate requires four tiles and 64-bit metadata");
    end
endmodule

`default_nettype wire
