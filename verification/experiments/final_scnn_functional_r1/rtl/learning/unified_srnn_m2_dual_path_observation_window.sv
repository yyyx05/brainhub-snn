`timescale 1ns/1ps
`default_nettype none

// Sequences actual and target SRNN backbone vectors into the path-major Q14
// observation stream consumed by the shared M2 W_h1 learning frontend.
module unified_srnn_m2_dual_path_observation_window #(
    parameter int BATCH_MAX = 64,
    parameter int TIMESTEP_MAX = 128,
    parameter int BACKBONE_MAX = 450,
    parameter int SOURCE_VECTOR_MAX = 480,
    parameter int SAMPLE_ID_WIDTH = $clog2(BATCH_MAX+1),
    parameter int TIMESTEP_ID_WIDTH = $clog2(TIMESTEP_MAX),
    parameter int BACKBONE_INDEX_WIDTH = $clog2(BACKBONE_MAX+1)
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_cfg_batch_count,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_cfg_backbone_count,
    output logic                                      o_cfg_error,

    input  logic                                      i_window_start,
    output logic                                      o_window_start_ready,
    input  logic [TIMESTEP_ID_WIDTH-1:0]              i_window_timestep_id,
    input  logic                                      i_window_include_target,
    input  logic                                      i_abort_window,

    output logic                                      o_actual_replay_start,
    input  logic                                      i_actual_replay_start_ready,
    output logic [TIMESTEP_ID_WIDTH-1:0]              o_actual_replay_timestep_id,
    input  logic                                      i_actual_replay_valid,
    output logic                                      o_actual_replay_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_actual_replay_sample_id,
    input  logic [SOURCE_VECTOR_MAX-1:0]              i_actual_replay_spikes,
    input  logic                                      i_actual_replay_done,
    input  logic                                      i_actual_replay_error,

    output logic                                      o_target_vector_start,
    input  logic                                      i_target_vector_start_ready,
    output logic [TIMESTEP_ID_WIDTH-1:0]              o_target_vector_timestep_id,
    input  logic                                      i_target_vector_valid,
    output logic                                      o_target_vector_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_target_vector_sample_id,
    input  logic [SOURCE_VECTOR_MAX-1:0]              i_target_vector_spikes,
    input  logic                                      i_target_vector_done,
    input  logic                                      i_target_vector_error,

    output logic                                      o_backbone_valid,
    input  logic                                      i_backbone_ready,
    output logic                                      o_backbone_path,
    output logic [SAMPLE_ID_WIDTH-1:0]                o_backbone_sample_id,
    output logic [BACKBONE_INDEX_WIDTH-1:0]           o_backbone_id,
    output logic signed [31:0]                        o_backbone_q14,
    output logic                                      o_backbone_last,

    output logic                                      o_busy,
    output logic                                      o_done,
    output logic                                      o_protocol_error
);

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_START_ACTUAL,
        STATE_ACTUAL,
        STATE_START_TARGET,
        STATE_TARGET,
        STATE_DRAIN,
        STATE_ERROR
    } state_t;

    state_t state_q;
    logic [TIMESTEP_ID_WIDTH-1:0] timestep_q;
    logic include_target_q;
    logic protocol_error_q;

    logic adapter_cfg_ready;
    logic adapter_cfg_error;
    logic adapter_window_start_ready;
    logic adapter_window_start;
    logic adapter_observation_valid;
    logic adapter_observation_ready;
    logic adapter_observation_path;
    logic [SAMPLE_ID_WIDTH-1:0] adapter_observation_sample_id;
    logic [SOURCE_VECTOR_MAX-1:0] adapter_observation_spikes;
    logic adapter_busy;
    logic adapter_done;
    logic adapter_protocol_error;

    logic start_fire;
    logic actual_start_fire;
    logic target_start_fire;
    logic source_error;

    always_comb begin
        source_error = i_actual_replay_error ||
            (((state_q == STATE_IDLE) ? i_window_include_target :
                include_target_q) && i_target_vector_error);

        o_cfg_ready = (state_q == STATE_IDLE) && adapter_cfg_ready;
        o_cfg_error = adapter_cfg_error;
        o_window_start_ready = (state_q == STATE_IDLE) &&
            adapter_window_start_ready && !source_error && !i_cfg_valid;
        start_fire = i_window_start && o_window_start_ready;
        adapter_window_start = start_fire;

        o_actual_replay_timestep_id = timestep_q;
        o_actual_replay_start = (state_q == STATE_START_ACTUAL) &&
            i_actual_replay_start_ready && !protocol_error_q;
        actual_start_fire = o_actual_replay_start;

        o_target_vector_timestep_id = timestep_q;
        o_target_vector_start = (state_q == STATE_START_TARGET) &&
            i_target_vector_start_ready && !protocol_error_q;
        target_start_fire = o_target_vector_start;

        adapter_observation_valid = 1'b0;
        adapter_observation_path = 1'b0;
        adapter_observation_sample_id = '0;
        adapter_observation_spikes = '0;
        o_actual_replay_ready = 1'b0;
        o_target_vector_ready = 1'b0;

        if (state_q == STATE_ACTUAL) begin
            adapter_observation_valid = i_actual_replay_valid;
            adapter_observation_path = 1'b0;
            adapter_observation_sample_id = i_actual_replay_sample_id;
            adapter_observation_spikes = i_actual_replay_spikes;
            o_actual_replay_ready = adapter_observation_ready &&
                !protocol_error_q && !source_error;
        end else if (state_q == STATE_TARGET) begin
            adapter_observation_valid = i_target_vector_valid;
            adapter_observation_path = 1'b1;
            adapter_observation_sample_id = i_target_vector_sample_id;
            adapter_observation_spikes = i_target_vector_spikes;
            o_target_vector_ready = adapter_observation_ready &&
                !protocol_error_q && !source_error;
        end

        o_busy = (state_q != STATE_IDLE) || adapter_busy;
        o_protocol_error = protocol_error_q || adapter_protocol_error ||
            source_error;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            timestep_q <= '0;
            include_target_q <= 1'b0;
            protocol_error_q <= 1'b0;
            o_done <= 1'b0;
        end else begin
            o_done <= 1'b0;

            if (i_clear_error) begin
                protocol_error_q <= 1'b0;
                if (state_q == STATE_ERROR)
                    state_q <= STATE_IDLE;
            end

            if (i_abort_window) begin
                state_q <= STATE_IDLE;
                include_target_q <= 1'b0;
                protocol_error_q <= 1'b0;
            end else if (adapter_protocol_error || source_error) begin
                state_q <= STATE_ERROR;
                protocol_error_q <= 1'b1;
            end else begin
                unique case (state_q)
                    STATE_IDLE: begin
                        if (start_fire) begin
                            timestep_q <= i_window_timestep_id;
                            include_target_q <= i_window_include_target;
                            state_q <= STATE_START_ACTUAL;
                        end
                    end

                    STATE_START_ACTUAL: begin
                        if (actual_start_fire)
                            state_q <= STATE_ACTUAL;
                    end

                    STATE_ACTUAL: begin
                        if (i_actual_replay_done) begin
                            if (include_target_q)
                                state_q <= STATE_START_TARGET;
                            else
                                state_q <= STATE_DRAIN;
                        end
                    end

                    STATE_START_TARGET: begin
                        if (target_start_fire)
                            state_q <= STATE_TARGET;
                    end

                    STATE_TARGET: begin
                        if (i_target_vector_done)
                            state_q <= STATE_DRAIN;
                    end

                    STATE_DRAIN: begin
                        if (adapter_done) begin
                            state_q <= STATE_IDLE;
                            o_done <= 1'b1;
                        end
                    end

                    STATE_ERROR: begin
                        if (i_clear_error)
                            state_q <= STATE_IDLE;
                    end

                    default: begin
                        state_q <= STATE_ERROR;
                        protocol_error_q <= 1'b1;
                    end
                endcase
            end
        end
    end

    unified_srnn_m2_backbone_observation_adapter #(
        .BATCH_MAX(BATCH_MAX),
        .BACKBONE_MAX(BACKBONE_MAX),
        .SOURCE_VECTOR_MAX(SOURCE_VECTOR_MAX),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .BACKBONE_INDEX_WIDTH(BACKBONE_INDEX_WIDTH)
    ) observation_adapter (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_cfg_valid(i_cfg_valid),
        .o_cfg_ready(adapter_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_backbone_count(i_cfg_backbone_count),
        .o_cfg_error(adapter_cfg_error),
        .i_window_start(adapter_window_start),
        .o_window_start_ready(adapter_window_start_ready),
        .i_window_include_target(i_window_include_target),
        .i_abort_window(i_abort_window),
        .i_observation_valid(adapter_observation_valid),
        .o_observation_ready(adapter_observation_ready),
        .i_observation_path(adapter_observation_path),
        .i_observation_sample_id(adapter_observation_sample_id),
        .i_observation_spikes(adapter_observation_spikes),
        .o_backbone_valid(o_backbone_valid),
        .i_backbone_ready(i_backbone_ready),
        .o_backbone_path(o_backbone_path),
        .o_backbone_sample_id(o_backbone_sample_id),
        .o_backbone_id(o_backbone_id),
        .o_backbone_q14(o_backbone_q14),
        .o_backbone_last(o_backbone_last),
        .o_busy(adapter_busy),
        .o_done(adapter_done),
        .o_protocol_error(adapter_protocol_error)
    );

endmodule

`default_nettype wire
