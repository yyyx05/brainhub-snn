`timescale 1ns/1ps
`default_nettype none

// Sequences a complete SCNN batch on one time-shared graph engine. A learning
// request opens one runtime capture window, then runs target/student passes for
// every sample before allowing one joint learning/commit sequence. Inference
// skips target passes and the learning runtime entirely.
module unified_scnn_selected_layers_graph_batch_coordinator_v1 #(
    parameter int MAX_BATCH = 10,
    parameter int MAX_TIMESTEPS = 20,
    parameter int MAX_WINDOW_BATCH = MAX_BATCH,
    parameter int MAX_WINDOW_TIMESTEPS = MAX_TIMESTEPS,
    parameter int MAX_FLATTENED_SAMPLES = MAX_BATCH,
    parameter bit ENABLE_TARGET_GRAPH = 1'b1,
    parameter int SAMPLE_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int TIMESTEP_WIDTH = 5,
    parameter int TIMESTEP_COUNT_WIDTH = $clog2(MAX_TIMESTEPS + 1),
    parameter int VERSION_WIDTH = 16,
    parameter int CLASS_WIDTH = 4
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_abort,
    input  wire logic                         i_clear_error,

    input  wire logic                         i_request_valid,
    output logic                              o_request_ready,
    input  wire logic                         i_learning_enable,
    input  wire logic [BATCH_COUNT_WIDTH-1:0] i_batch_count,
    input  wire logic [TIMESTEP_WIDTH-1:0]     i_timestep,
    input  wire logic [TIMESTEP_COUNT_WIDTH-1:0]
                                                i_timestep_count,
    input  wire logic [VERSION_WIDTH-1:0]      i_version,
    input  wire logic [MAX_BATCH*CLASS_WIDTH-1:0]
                                                i_labels_flat,
    output logic                              o_done,

    output logic                              o_runtime_run_valid,
    input  wire logic                         i_runtime_run_ready,
    output logic [TIMESTEP_WIDTH-1:0]          o_runtime_timestep,
    output logic [TIMESTEP_COUNT_WIDTH-1:0]    o_runtime_timestep_count,
    output logic [BATCH_COUNT_WIDTH-1:0]       o_runtime_batch_count,
    output logic [VERSION_WIDTH-1:0]           o_runtime_version,
    input  wire logic                         i_runtime_run_done,

    output logic                              o_sample_context_valid,
    input  wire logic                         i_sample_context_ready,
    output logic                              o_sample_context_target,
    output logic [SAMPLE_WIDTH-1:0]            o_sample_context_sample,
    output logic [VERSION_WIDTH-1:0]           o_sample_context_version,

    output logic                              o_graph_start_valid,
    input  wire logic                         i_graph_start_ready,
    output logic                              o_graph_target,
    output logic [SAMPLE_WIDTH-1:0]            o_graph_sample,
    output logic [TIMESTEP_WIDTH-1:0]          o_graph_timestep,
    output logic [VERSION_WIDTH-1:0]           o_graph_version,
    output logic [CLASS_WIDTH-1:0]             o_graph_label,
    output logic                              o_graph_sample_last,
    input  wire logic                         i_graph_done,

    output logic                              o_conv_observation_enable,
    output logic                              o_classifier_observation_enable,
    output logic                              o_observation_target,
    output logic                              o_busy,
    output logic                              o_sticky_error
);
    typedef enum logic [3:0] {
        BATCH_IDLE,
        BATCH_RUNTIME_START,
        BATCH_TARGET_CONTEXT,
        BATCH_TARGET_START,
        BATCH_TARGET_WAIT,
        BATCH_STUDENT_CONTEXT,
        BATCH_STUDENT_START,
        BATCH_STUDENT_WAIT,
        BATCH_LEARNING_WAIT,
        BATCH_RESPOND
    } batch_state_t;

    batch_state_t state_q;
    logic learning_q;
    logic [BATCH_COUNT_WIDTH-1:0] batch_count_q;
    logic [SAMPLE_WIDTH-1:0] sample_q;
    logic [TIMESTEP_WIDTH-1:0] timestep_q;
    logic [TIMESTEP_COUNT_WIDTH-1:0] timestep_count_q;
    logic [TIMESTEP_COUNT_WIDTH-1:0] timestep_offset_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic [MAX_BATCH*CLASS_WIDTH-1:0] labels_flat_q;
    logic runtime_started_q;
    logic runtime_done_seen_q;

    logic request_fire;
    logic runtime_start_fire;
    logic sample_context_fire;
    logic graph_start_fire;
    logic request_legal;
    logic last_sample;
    logic last_timestep;
    logic runtime_done_legal;

    assign request_legal = i_batch_count != 0 &&
        int'(i_batch_count) <= MAX_WINDOW_BATCH &&
        i_timestep_count != 0 &&
        int'(i_timestep_count) <= MAX_WINDOW_TIMESTEPS &&
        int'(i_batch_count) * int'(i_timestep_count) <=
            MAX_FLATTENED_SAMPLES &&
        int'(i_timestep) + int'(i_timestep_count) <= MAX_TIMESTEPS;
    assign o_request_ready = state_q == BATCH_IDLE && !o_sticky_error;
    assign request_fire = i_request_valid && o_request_ready && request_legal;

    assign o_runtime_run_valid = state_q == BATCH_RUNTIME_START;
    assign runtime_start_fire = o_runtime_run_valid && i_runtime_run_ready;
    assign o_runtime_timestep = timestep_q;
    assign o_runtime_timestep_count = timestep_count_q;
    assign o_runtime_batch_count = batch_count_q;
    assign o_runtime_version = version_q;

    assign o_sample_context_valid = state_q == BATCH_TARGET_CONTEXT ||
        state_q == BATCH_STUDENT_CONTEXT;
    assign o_sample_context_target = state_q == BATCH_TARGET_CONTEXT;
    assign o_sample_context_sample = sample_q;
    assign o_sample_context_version = version_q;
    assign sample_context_fire = o_sample_context_valid &&
        i_sample_context_ready;

    assign o_graph_start_valid = state_q == BATCH_TARGET_START ||
        state_q == BATCH_STUDENT_START;
    assign graph_start_fire = o_graph_start_valid && i_graph_start_ready;
    assign o_graph_target = state_q == BATCH_TARGET_START ||
        state_q == BATCH_TARGET_WAIT;
    assign o_graph_sample = sample_q;
    assign o_graph_timestep = TIMESTEP_WIDTH'(
        int'(timestep_q) + int'(timestep_offset_q));
    assign o_graph_version = version_q;
    assign o_graph_label = labels_flat_q[int'(sample_q)*CLASS_WIDTH +:
        CLASS_WIDTH];
    // The graph tail's sample_last is a temporal boundary for the current
    // physical sample, not the last sample in the batch.
    assign o_graph_sample_last = last_timestep;

    assign o_conv_observation_enable = ENABLE_TARGET_GRAPH && learning_q &&
        (state_q == BATCH_TARGET_START || state_q == BATCH_TARGET_WAIT ||
         state_q == BATCH_STUDENT_START || state_q == BATCH_STUDENT_WAIT);
    assign o_classifier_observation_enable = learning_q &&
        (state_q == BATCH_STUDENT_START || state_q == BATCH_STUDENT_WAIT);
    assign o_observation_target = state_q == BATCH_TARGET_START ||
        state_q == BATCH_TARGET_WAIT;
    assign o_busy = state_q != BATCH_IDLE;
    assign last_sample = BATCH_COUNT_WIDTH'(int'(sample_q) + 1) ==
        batch_count_q;
    assign last_timestep = TIMESTEP_COUNT_WIDTH'(
        int'(timestep_offset_q) + 1) == timestep_count_q;
    assign runtime_done_legal = state_q == BATCH_LEARNING_WAIT ||
        (state_q == BATCH_STUDENT_WAIT && last_sample && last_timestep &&
         i_graph_done);

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= BATCH_IDLE;
            learning_q <= 1'b0;
            batch_count_q <= '0;
            sample_q <= '0;
            timestep_q <= '0;
            timestep_count_q <= '0;
            timestep_offset_q <= '0;
            version_q <= '0;
            labels_flat_q <= '0;
            runtime_started_q <= 1'b0;
            runtime_done_seen_q <= 1'b0;
            o_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= BATCH_IDLE;
            runtime_started_q <= 1'b0;
            runtime_done_seen_q <= 1'b0;
            o_done <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_clear_error && state_q == BATCH_IDLE)
                o_sticky_error <= 1'b0;

            if (i_request_valid && o_request_ready && !request_legal)
                o_sticky_error <= 1'b1;
            if (i_runtime_run_done) begin
                if (!learning_q || !runtime_started_q || !runtime_done_legal)
                    o_sticky_error <= 1'b1;
                else
                    runtime_done_seen_q <= 1'b1;
            end
            if (i_graph_done && state_q != BATCH_TARGET_WAIT &&
                    state_q != BATCH_STUDENT_WAIT)
                o_sticky_error <= 1'b1;

            case (state_q)
                BATCH_IDLE: begin
                    if (request_fire) begin
                        learning_q <= i_learning_enable;
                        batch_count_q <= i_batch_count;
                        sample_q <= '0;
                        timestep_q <= i_timestep;
                        timestep_count_q <= i_timestep_count;
                        timestep_offset_q <= '0;
                        version_q <= i_version;
                        labels_flat_q <= i_labels_flat;
                        runtime_started_q <= 1'b0;
                        runtime_done_seen_q <= 1'b0;
                        state_q <= i_learning_enable ?
                            BATCH_RUNTIME_START : BATCH_STUDENT_START;
                    end
                end

                BATCH_RUNTIME_START: begin
                    if (runtime_start_fire) begin
                        runtime_started_q <= 1'b1;
                        state_q <= ENABLE_TARGET_GRAPH ?
                            BATCH_TARGET_CONTEXT : BATCH_STUDENT_CONTEXT;
                    end
                end

                BATCH_TARGET_CONTEXT: begin
                    if (sample_context_fire)
                        state_q <= BATCH_TARGET_START;
                end

                BATCH_TARGET_START: begin
                    if (graph_start_fire)
                        state_q <= BATCH_TARGET_WAIT;
                end

                BATCH_TARGET_WAIT: begin
                    if (i_graph_done)
                        state_q <= timestep_offset_q == '0 ?
                            BATCH_STUDENT_CONTEXT : BATCH_STUDENT_START;
                end

                BATCH_STUDENT_CONTEXT: begin
                    if (sample_context_fire)
                        state_q <= BATCH_STUDENT_START;
                end

                BATCH_STUDENT_START: begin
                    if (graph_start_fire)
                        state_q <= BATCH_STUDENT_WAIT;
                end

                BATCH_STUDENT_WAIT: begin
                    if (i_graph_done) begin
                        if (!last_timestep) begin
                            timestep_offset_q <= timestep_offset_q + 1'b1;
                            state_q <= learning_q ?
                                (ENABLE_TARGET_GRAPH ? BATCH_TARGET_START :
                                 BATCH_STUDENT_START) : BATCH_STUDENT_START;
                        end else if (!last_sample) begin
                            sample_q <= sample_q + 1'b1;
                            timestep_offset_q <= '0;
                            state_q <= learning_q ?
                                (ENABLE_TARGET_GRAPH ? BATCH_TARGET_CONTEXT :
                                 BATCH_STUDENT_CONTEXT) : BATCH_STUDENT_START;
                        end else if (!learning_q || runtime_done_seen_q ||
                                i_runtime_run_done) begin
                            state_q <= BATCH_RESPOND;
                        end else begin
                            state_q <= BATCH_LEARNING_WAIT;
                        end
                    end
                end

                BATCH_LEARNING_WAIT: begin
                    if (runtime_done_seen_q || i_runtime_run_done)
                        state_q <= BATCH_RESPOND;
                end

                BATCH_RESPOND: begin
                    o_done <= 1'b1;
                    state_q <= BATCH_IDLE;
                end

                default: begin
                    state_q <= BATCH_IDLE;
                    o_sticky_error <= 1'b1;
                end
            endcase
        end
    end

    initial begin
        if (MAX_BATCH < 1 || SAMPLE_WIDTH < 1 || BATCH_COUNT_WIDTH < 1 ||
                MAX_WINDOW_BATCH < 1 || MAX_WINDOW_BATCH > MAX_BATCH ||
                MAX_WINDOW_TIMESTEPS < 1 ||
                MAX_WINDOW_TIMESTEPS > MAX_TIMESTEPS ||
                MAX_FLATTENED_SAMPLES < 1)
            $error("SCNN batch coordinator parameters are invalid");
    end
endmodule

`default_nettype wire
