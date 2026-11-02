`timescale 1ns/1ps
`default_nettype none

// Deterministic SRNN source-group streamer for the shared P4 H40 array.
// Feedforward and recurrent matrices occupy disjoint row ranges in the same
// four-bank cache. Four consecutive logical sources map to physical rows
// {r,r,r+1,r+1}; source parity selects the bank pair inside the weight store.
// Every logical group is emitted, while spike bits gate arithmetic terms. This
// gives a fixed correctness baseline without duplicating a topology datapath.
module unified_srnn_h40_p4_group_streamer_v1 #(
    parameter int FEEDFORWARD_MAX = 700,
    parameter int RECURRENT_MAX = 480,
    parameter int SOURCE_COUNT_WIDTH = 10,
    parameter int GROUP_COUNT_WIDTH = 16,
    parameter int WEIGHT_ADDR_WIDTH = 11,
    parameter int WEIGHT_ROW_DEPTH = 578,
    parameter int WORK_ADDR_WIDTH = 9,
    parameter int ARRAY_TILES = 12
) (
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,
    input  wire logic                              i_abort,
    input  wire logic                              i_clear_error,

    input  wire logic                              i_start_valid,
    output logic                                   o_start_ready,
    input  wire logic [SOURCE_COUNT_WIDTH-1:0]      i_feedforward_source_count,
    input  wire logic [SOURCE_COUNT_WIDTH-1:0]      i_recurrent_source_count,
    input  wire logic [9:0]                        i_feedforward_cache_row_base,
    input  wire logic [9:0]                        i_recurrent_cache_row_base,
    input  wire logic                              i_recurrent_enable,
    input  wire logic [FEEDFORWARD_MAX-1:0]         i_feedforward_spike_flat,
    input  wire logic [RECURRENT_MAX-1:0]           i_recurrent_spike_flat,
    input  wire logic [15:0]                        i_neuron_count,
    input  wire logic [15:0]                        i_leak,
    input  wire logic [31:0]                        i_threshold,
    input  wire logic                               i_reset_to_zero,
    input  wire logic [15:0]                        i_trace_decay,
    input  wire logic [51:0]                        i_transaction_tag,
    input  wire logic [WORK_ADDR_WIDTH-1:0]         i_work_base_addr,
    input  wire logic [3:0]                         i_destination_tile_base,

    output logic                                   o_layer_valid,
    input  wire logic                              i_layer_ready,
    output logic [GROUP_COUNT_WIDTH-1:0]           o_feedforward_group_count,
    output logic [GROUP_COUNT_WIDTH-1:0]           o_recurrent_group_count,
    output logic                                   o_recurrent_enable,
    output logic [15:0]                            o_neuron_count,
    output logic [15:0]                            o_leak,
    output logic [31:0]                            o_threshold,
    output logic                                   o_reset_to_zero,
    output logic [15:0]                            o_trace_decay,
    output logic [51:0]                            o_transaction_tag,
    output logic [WORK_ADDR_WIDTH-1:0]             o_work_base_addr,
    output logic [3:0]                             o_destination_tile_base,

    output logic                                   o_group_valid,
    input  wire logic                              i_group_ready,
    output logic                                   o_group_recurrent,
    output logic [3:0]                             o_group_source_valid,
    output logic [WEIGHT_ADDR_WIDTH-1:0]           o_group_weight_addr0,
    output logic [WEIGHT_ADDR_WIDTH-1:0]           o_group_weight_addr1,
    output logic [WEIGHT_ADDR_WIDTH-1:0]           o_group_weight_addr2,
    output logic [WEIGHT_ADDR_WIDTH-1:0]           o_group_weight_addr3,

    input  wire logic                              i_layer_done,
    output logic                                   o_done,
    output logic                                   o_busy,
    output logic                                   o_idle,
    output logic                                   o_sticky_error,
    output logic [3:0]                             o_error_code
);
    localparam logic [3:0] ERROR_NONE = 4'h0;
    localparam logic [3:0] ERROR_GEOMETRY = 4'h1;
    localparam logic [3:0] ERROR_EARLY_LAYER_DONE = 4'h2;

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_DESCRIPTOR,
        STATE_FEEDFORWARD,
        STATE_RECURRENT,
        STATE_DRAIN
    } state_t;

    state_t state_q;
    logic [SOURCE_COUNT_WIDTH-1:0] feedforward_source_count_q;
    logic [SOURCE_COUNT_WIDTH-1:0] recurrent_source_count_q;
    logic [GROUP_COUNT_WIDTH-1:0] feedforward_group_count_q;
    logic [GROUP_COUNT_WIDTH-1:0] recurrent_group_count_q;
    logic [9:0] feedforward_cache_row_base_q;
    logic [9:0] recurrent_cache_row_base_q;
    logic recurrent_enable_q;
    logic [FEEDFORWARD_MAX-1:0] feedforward_spike_q;
    logic [RECURRENT_MAX-1:0] recurrent_spike_q;
    logic [15:0] neuron_count_q;
    logic [15:0] leak_q;
    logic [31:0] threshold_q;
    logic reset_to_zero_q;
    logic [15:0] trace_decay_q;
    logic [51:0] transaction_tag_q;
    logic [WORK_ADDR_WIDTH-1:0] work_base_addr_q;
    logic [3:0] destination_tile_base_q;
    logic [GROUP_COUNT_WIDTH-1:0] group_index_q;

    logic request_legal;
    logic [GROUP_COUNT_WIDTH-1:0] requested_feedforward_groups;
    logic [GROUP_COUNT_WIDTH-1:0] requested_recurrent_groups;
    logic [10:0] feedforward_last_row;
    logic [10:0] recurrent_last_row;
    logic start_fire;
    logic layer_fire;
    logic group_fire;
    logic [SOURCE_COUNT_WIDTH+1:0] group_source_base;
    logic [SOURCE_COUNT_WIDTH+1:0] source_index [0:3];
    logic [10:0] selected_cache_base;
    logic [SOURCE_COUNT_WIDTH-1:0] selected_source_count;
    logic [4:0] requested_destination_tiles;
    logic [5:0] destination_tile_end;

    always_comb begin
        requested_feedforward_groups = GROUP_COUNT_WIDTH'(
            (i_feedforward_source_count + 3) >> 2);
        requested_recurrent_groups = GROUP_COUNT_WIDTH'(
            (i_recurrent_source_count + 3) >> 2);
        feedforward_last_row = {1'b0, i_feedforward_cache_row_base} +
            (11'(i_feedforward_source_count - 1'b1) >> 1);
        recurrent_last_row = {1'b0, i_recurrent_cache_row_base} +
            (11'(i_recurrent_source_count - 1'b1) >> 1);
        requested_destination_tiles = 5'((i_neuron_count + 39) / 40);
        destination_tile_end = {2'b0, i_destination_tile_base} +
            {1'b0, requested_destination_tiles};
        request_legal = i_feedforward_source_count != 0 &&
            i_feedforward_source_count <= SOURCE_COUNT_WIDTH'(FEEDFORWARD_MAX) &&
            requested_feedforward_groups != 0 &&
            feedforward_last_row < 11'(WEIGHT_ROW_DEPTH) &&
            ((!i_recurrent_enable && i_recurrent_source_count == 0) ||
             (i_recurrent_enable && i_recurrent_source_count != 0 &&
              i_recurrent_source_count <= SOURCE_COUNT_WIDTH'(RECURRENT_MAX) &&
              requested_recurrent_groups != 0 &&
              recurrent_last_row < 11'(WEIGHT_ROW_DEPTH))) &&
            i_neuron_count != 0 &&
            i_neuron_count <= 16'(ARRAY_TILES * 40) &&
            requested_destination_tiles != 0 &&
            destination_tile_end <= 6'(ARRAY_TILES);

        selected_cache_base = o_group_recurrent ?
            {1'b0, recurrent_cache_row_base_q} :
            {1'b0, feedforward_cache_row_base_q};
        selected_source_count = o_group_recurrent ?
            recurrent_source_count_q : feedforward_source_count_q;
        group_source_base = (SOURCE_COUNT_WIDTH + 2)'(group_index_q) << 2;
        for (int term = 0; term < 4; term++)
            source_index[term] = group_source_base +
                (SOURCE_COUNT_WIDTH + 2)'(term);

        o_group_source_valid = '0;
        o_group_weight_addr0 = '0;
        o_group_weight_addr1 = '0;
        o_group_weight_addr2 = '0;
        o_group_weight_addr3 = '0;
        for (int term = 0; term < 4; term++) begin
            if (source_index[term] < selected_source_count) begin
                if (o_group_recurrent)
                    o_group_source_valid[term] =
                        recurrent_spike_q[source_index[term]];
                else
                    o_group_source_valid[term] =
                        feedforward_spike_q[source_index[term]];
                case (term)
                    0: o_group_weight_addr0 = WEIGHT_ADDR_WIDTH'(
                        selected_cache_base + (source_index[term] >> 1));
                    1: o_group_weight_addr1 = WEIGHT_ADDR_WIDTH'(
                        selected_cache_base + (source_index[term] >> 1));
                    2: o_group_weight_addr2 = WEIGHT_ADDR_WIDTH'(
                        selected_cache_base + (source_index[term] >> 1));
                    default: o_group_weight_addr3 = WEIGHT_ADDR_WIDTH'(
                        selected_cache_base + (source_index[term] >> 1));
                endcase
            end
        end
    end

    assign o_start_ready = state_q == STATE_IDLE && !o_sticky_error;
    assign start_fire = i_start_valid && o_start_ready;
    assign o_layer_valid = state_q == STATE_DESCRIPTOR;
    assign layer_fire = o_layer_valid && i_layer_ready;
    assign o_feedforward_group_count = feedforward_group_count_q;
    assign o_recurrent_group_count = recurrent_group_count_q;
    assign o_recurrent_enable = recurrent_enable_q;
    assign o_neuron_count = neuron_count_q;
    assign o_leak = leak_q;
    assign o_threshold = threshold_q;
    assign o_reset_to_zero = reset_to_zero_q;
    assign o_trace_decay = trace_decay_q;
    assign o_transaction_tag = transaction_tag_q;
    assign o_work_base_addr = work_base_addr_q;
    assign o_destination_tile_base = destination_tile_base_q;
    assign o_group_valid = state_q == STATE_FEEDFORWARD ||
        state_q == STATE_RECURRENT;
    assign o_group_recurrent = state_q == STATE_RECURRENT;
    assign group_fire = o_group_valid && i_group_ready;
    assign o_busy = state_q != STATE_IDLE;
    assign o_idle = state_q == STATE_IDLE;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            feedforward_source_count_q <= '0;
            recurrent_source_count_q <= '0;
            feedforward_group_count_q <= '0;
            recurrent_group_count_q <= '0;
            feedforward_cache_row_base_q <= '0;
            recurrent_cache_row_base_q <= '0;
            recurrent_enable_q <= 1'b0;
            feedforward_spike_q <= '0;
            recurrent_spike_q <= '0;
            neuron_count_q <= '0;
            leak_q <= '0;
            threshold_q <= '0;
            reset_to_zero_q <= 1'b0;
            trace_decay_q <= '0;
            transaction_tag_q <= '0;
            work_base_addr_q <= '0;
            destination_tile_base_q <= '0;
            group_index_q <= '0;
            o_done <= 1'b0;
            o_sticky_error <= 1'b0;
            o_error_code <= ERROR_NONE;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            group_index_q <= '0;
            o_done <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_clear_error && state_q == STATE_IDLE) begin
                o_sticky_error <= 1'b0;
                o_error_code <= ERROR_NONE;
            end

            unique case (state_q)
                STATE_IDLE: begin
                    if (i_layer_done && !o_sticky_error) begin
                        o_sticky_error <= 1'b1;
                        o_error_code <= ERROR_EARLY_LAYER_DONE;
                    end else if (start_fire) begin
                        if (!request_legal) begin
                            o_sticky_error <= 1'b1;
                            o_error_code <= ERROR_GEOMETRY;
                        end else begin
                            feedforward_source_count_q <=
                                i_feedforward_source_count;
                            recurrent_source_count_q <=
                                i_recurrent_source_count;
                            feedforward_group_count_q <=
                                requested_feedforward_groups;
                            recurrent_group_count_q <= i_recurrent_enable ?
                                requested_recurrent_groups : '0;
                            feedforward_cache_row_base_q <=
                                i_feedforward_cache_row_base;
                            recurrent_cache_row_base_q <=
                                i_recurrent_cache_row_base;
                            recurrent_enable_q <= i_recurrent_enable;
                            feedforward_spike_q <= i_feedforward_spike_flat;
                            recurrent_spike_q <= i_recurrent_spike_flat;
                            neuron_count_q <= i_neuron_count;
                            leak_q <= i_leak;
                            threshold_q <= i_threshold;
                            reset_to_zero_q <= i_reset_to_zero;
                            trace_decay_q <= i_trace_decay;
                            transaction_tag_q <= i_transaction_tag;
                            work_base_addr_q <= i_work_base_addr;
                            destination_tile_base_q <=
                                i_destination_tile_base;
                            group_index_q <= '0;
                            state_q <= STATE_DESCRIPTOR;
                        end
                    end
                end

                STATE_DESCRIPTOR: begin
                    if (i_layer_done) begin
                        o_sticky_error <= 1'b1;
                        o_error_code <= ERROR_EARLY_LAYER_DONE;
                        state_q <= STATE_IDLE;
                    end else if (layer_fire) begin
                        group_index_q <= '0;
                        state_q <= STATE_FEEDFORWARD;
                    end
                end

                STATE_FEEDFORWARD: begin
                    if (i_layer_done) begin
                        o_sticky_error <= 1'b1;
                        o_error_code <= ERROR_EARLY_LAYER_DONE;
                        state_q <= STATE_IDLE;
                    end else if (group_fire) begin
                        if (group_index_q + 1'b1 ==
                                feedforward_group_count_q) begin
                            group_index_q <= '0;
                            state_q <= recurrent_enable_q ?
                                STATE_RECURRENT : STATE_DRAIN;
                        end else begin
                            group_index_q <= group_index_q + 1'b1;
                        end
                    end
                end

                STATE_RECURRENT: begin
                    if (i_layer_done) begin
                        o_sticky_error <= 1'b1;
                        o_error_code <= ERROR_EARLY_LAYER_DONE;
                        state_q <= STATE_IDLE;
                    end else if (group_fire) begin
                        if (group_index_q + 1'b1 == recurrent_group_count_q) begin
                            group_index_q <= '0;
                            state_q <= STATE_DRAIN;
                        end else begin
                            group_index_q <= group_index_q + 1'b1;
                        end
                    end
                end

                STATE_DRAIN: begin
                    if (i_layer_done) begin
                        o_done <= 1'b1;
                        state_q <= STATE_IDLE;
                    end
                end

                default: begin
                    state_q <= STATE_IDLE;
                    o_sticky_error <= 1'b1;
                    o_error_code <= ERROR_GEOMETRY;
                end
            endcase
        end
    end

    initial begin
        if (FEEDFORWARD_MAX < 1 || RECURRENT_MAX < 1 ||
                SOURCE_COUNT_WIDTH < 1 || GROUP_COUNT_WIDTH < 1 ||
                WEIGHT_ADDR_WIDTH != 11 || WEIGHT_ROW_DEPTH != 578 ||
                WORK_ADDR_WIDTH < 1 || ARRAY_TILES != 12)
            $error("SRNN H40 P4 group streamer geometry is invalid");
    end
endmodule

`default_nettype wire
