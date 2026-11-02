`timescale 1ns/1ps
`default_nettype none

// Captures one FC/SRNN layer observation in physical completion order and
// replays it in deterministic neuron-major order. Payload memory is reset-free;
// the per-transaction seen bitmap is the sole validity authority.
module unified_fc_srnn_h40_tp_observation_reorder_v1 #(
    parameter int MAX_NEURONS = 640,
    parameter int NEURON_ID_WIDTH = $clog2(MAX_NEURONS),
    parameter int NEURON_COUNT_WIDTH = $clog2(MAX_NEURONS + 1)
) (
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,
    input  wire logic                              i_abort,
    input  wire logic                              i_clear_error,

    input  wire logic                              i_cfg_valid,
    output logic                                   o_cfg_ready,
    input  wire logic [1:0]                        i_cfg_mode,
    input  wire logic [51:0]                       i_cfg_transaction_tag,
    input  wire logic [NEURON_ID_WIDTH-1:0]        i_cfg_neuron_base,
    input  wire logic [NEURON_COUNT_WIDTH-1:0]     i_cfg_neuron_count,

    input  wire logic                              i_scalar_valid,
    output logic                                   o_scalar_ready,
    input  wire logic [1:0]                        i_scalar_mode,
    input  wire logic [51:0]                       i_scalar_transaction_tag,
    input  wire logic [NEURON_ID_WIDTH-1:0]        i_scalar_neuron_id,
    input  wire logic                              i_scalar_spike,
    input  wire logic signed [31:0]                i_scalar_pre_state,
    input  wire logic signed [31:0]                i_scalar_trace_next,

    output logic                                   o_replay_valid,
    input  wire logic                              i_replay_ready,
    output logic [1:0]                             o_replay_mode,
    output logic [51:0]                            o_replay_transaction_tag,
    output logic [NEURON_COUNT_WIDTH-1:0]          o_replay_local_index,
    output logic [NEURON_ID_WIDTH-1:0]             o_replay_neuron_id,
    output logic                                   o_replay_spike,
    output logic signed [31:0]                     o_replay_pre_state,
    output logic signed [31:0]                     o_replay_trace_next,
    output logic                                   o_replay_last,

    output logic                                   o_capture_done,
    output logic                                   o_replay_done,
    output logic                                   o_busy,
    output logic                                   o_idle,
    output logic                                   o_sticky_error,
    output logic [3:0]                             o_error_code
);
    localparam logic [1:0] MODE_FC = 2'd0;
    localparam logic [1:0] MODE_SRNN = 2'd1;

    localparam logic [3:0] ERROR_NONE = 4'h0;
    localparam logic [3:0] ERROR_CONFIG = 4'h1;
    localparam logic [3:0] ERROR_METADATA = 4'h2;
    localparam logic [3:0] ERROR_RANGE = 4'h3;
    localparam logic [3:0] ERROR_DUPLICATE = 4'h4;
    localparam int LIMIT_WIDTH = NEURON_COUNT_WIDTH + 1;

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_CAPTURE,
        STATE_REPLAY
    } state_t;

    (* ram_style = "block" *) logic [64:0]
        payload_memory [0:MAX_NEURONS-1];

    state_t state_q;
    logic [1:0] mode_q;
    logic [51:0] transaction_tag_q;
    logic [NEURON_ID_WIDTH-1:0] neuron_base_q;
    logic [NEURON_COUNT_WIDTH-1:0] neuron_count_q;
    logic [NEURON_COUNT_WIDTH-1:0] accepted_count_q;
    logic [MAX_NEURONS-1:0] seen_q;

    logic [NEURON_COUNT_WIDTH-1:0] replay_next_index_q;
    logic [NEURON_COUNT_WIDTH-1:0] replay_output_index_q;
    logic [64:0] replay_payload_q;
    logic replay_valid_q;

    logic cfg_fire;
    logic cfg_legal;
    logic [LIMIT_WIDTH-1:0] cfg_limit;
    logic scalar_fire;
    logic scalar_metadata_legal;
    logic scalar_in_range;
    logic [NEURON_COUNT_WIDTH-1:0] scalar_local_index;
    logic scalar_duplicate;
    logic replay_fire;
    logic replay_advance;

    always_comb begin
        cfg_limit = LIMIT_WIDTH'(i_cfg_neuron_base) +
            LIMIT_WIDTH'(i_cfg_neuron_count);
        cfg_legal =
            (i_cfg_mode == MODE_FC || i_cfg_mode == MODE_SRNN) &&
            i_cfg_neuron_count != 0 &&
            cfg_limit <= LIMIT_WIDTH'(MAX_NEURONS);

        o_cfg_ready = state_q == STATE_IDLE && !o_sticky_error;
        cfg_fire = i_cfg_valid && o_cfg_ready;

        scalar_metadata_legal =
            i_scalar_mode == mode_q &&
            i_scalar_transaction_tag == transaction_tag_q;
        scalar_in_range =
            i_scalar_neuron_id >= neuron_base_q &&
            LIMIT_WIDTH'(i_scalar_neuron_id) <
                (LIMIT_WIDTH'(neuron_base_q) +
                 LIMIT_WIDTH'(neuron_count_q));
        scalar_local_index = NEURON_COUNT_WIDTH'(
            i_scalar_neuron_id - neuron_base_q);
        scalar_duplicate = 1'b0;
        if (scalar_in_range)
            scalar_duplicate = seen_q[scalar_local_index];

        o_scalar_ready = state_q == STATE_CAPTURE && !o_sticky_error;
        scalar_fire = i_scalar_valid && o_scalar_ready;

        o_replay_valid = replay_valid_q && !o_sticky_error;
        o_replay_mode = mode_q;
        o_replay_transaction_tag = transaction_tag_q;
        o_replay_local_index = replay_output_index_q;
        o_replay_neuron_id = neuron_base_q +
            NEURON_ID_WIDTH'(replay_output_index_q);
        o_replay_spike = replay_payload_q[0];
        o_replay_pre_state = $signed(replay_payload_q[32:1]);
        o_replay_trace_next = $signed(replay_payload_q[64:33]);
        o_replay_last = o_replay_valid &&
            replay_output_index_q + 1'b1 == neuron_count_q;
        replay_fire = o_replay_valid && i_replay_ready;
        replay_advance = state_q == STATE_REPLAY &&
            (!replay_valid_q || i_replay_ready);

        o_busy = state_q != STATE_IDLE;
        o_idle = state_q == STATE_IDLE;
    end

    // Keep payload RAM out of the asynchronous-reset control process.
    // Reset/abort suppress writes; seen_q remains the validity authority.
    always_ff @(posedge i_clk) begin
        if (i_rst_n && !i_abort && scalar_fire && scalar_metadata_legal &&
            scalar_in_range && !scalar_duplicate) begin
            payload_memory[scalar_local_index] <= {
                i_scalar_trace_next,
                i_scalar_pre_state,
                i_scalar_spike
            };
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            mode_q <= MODE_FC;
            transaction_tag_q <= '0;
            neuron_base_q <= '0;
            neuron_count_q <= '0;
            accepted_count_q <= '0;
            seen_q <= '0;
            replay_next_index_q <= '0;
            replay_output_index_q <= '0;
            replay_payload_q <= '0;
            replay_valid_q <= 1'b0;
            o_capture_done <= 1'b0;
            o_replay_done <= 1'b0;
            o_sticky_error <= 1'b0;
            o_error_code <= ERROR_NONE;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            accepted_count_q <= '0;
            seen_q <= '0;
            replay_next_index_q <= '0;
            replay_valid_q <= 1'b0;
            o_capture_done <= 1'b0;
            o_replay_done <= 1'b0;
        end else begin
            o_capture_done <= 1'b0;
            o_replay_done <= 1'b0;

            if (i_clear_error && state_q == STATE_IDLE) begin
                o_sticky_error <= 1'b0;
                o_error_code <= ERROR_NONE;
            end

            if (cfg_fire) begin
                if (!cfg_legal) begin
                    o_sticky_error <= 1'b1;
                    o_error_code <= ERROR_CONFIG;
                end else begin
                    state_q <= STATE_CAPTURE;
                    mode_q <= i_cfg_mode;
                    transaction_tag_q <= i_cfg_transaction_tag;
                    neuron_base_q <= i_cfg_neuron_base;
                    neuron_count_q <= i_cfg_neuron_count;
                    accepted_count_q <= '0;
                    seen_q <= '0;
                    replay_next_index_q <= '0;
                    replay_valid_q <= 1'b0;
                end
            end

            if (scalar_fire) begin
                if (!scalar_metadata_legal) begin
                    state_q <= STATE_IDLE;
                    replay_valid_q <= 1'b0;
                    o_sticky_error <= 1'b1;
                    o_error_code <= ERROR_METADATA;
                end else if (!scalar_in_range) begin
                    state_q <= STATE_IDLE;
                    replay_valid_q <= 1'b0;
                    o_sticky_error <= 1'b1;
                    o_error_code <= ERROR_RANGE;
                end else if (scalar_duplicate) begin
                    state_q <= STATE_IDLE;
                    replay_valid_q <= 1'b0;
                    o_sticky_error <= 1'b1;
                    o_error_code <= ERROR_DUPLICATE;
                end else begin
                    seen_q[scalar_local_index] <= 1'b1;
                    accepted_count_q <= accepted_count_q + 1'b1;
                    if (accepted_count_q + 1'b1 == neuron_count_q) begin
                        state_q <= STATE_REPLAY;
                        replay_next_index_q <= '0;
                        replay_valid_q <= 1'b0;
                        o_capture_done <= 1'b1;
                    end
                end
            end

            if (replay_advance) begin
                if (replay_next_index_q < neuron_count_q) begin
                    replay_payload_q <=
                        payload_memory[replay_next_index_q];
                    replay_output_index_q <= replay_next_index_q;
                    replay_next_index_q <= replay_next_index_q + 1'b1;
                    replay_valid_q <= 1'b1;
                end else if (replay_fire) begin
                    state_q <= STATE_IDLE;
                    replay_valid_q <= 1'b0;
                    o_replay_done <= 1'b1;
                end
            end
        end
    end

    initial begin
        if (MAX_NEURONS < 2 || NEURON_ID_WIDTH < 1 ||
            NEURON_COUNT_WIDTH < 2)
            $error("FC/SRNN observation reorder has invalid geometry");
    end
endmodule

`default_nettype wire
