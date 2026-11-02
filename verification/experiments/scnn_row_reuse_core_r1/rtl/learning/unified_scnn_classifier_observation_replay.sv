`timescale 1ns/1ps
`default_nettype none

// Sample-major capture of the shared OUTPUT frontend's two observation ports.
// History contains raw signed Q8.8 temporal traces, not normalized or clipped
// classifier inputs. Only validity/control is reset; all history RAM is retained.
module unified_scnn_classifier_observation_replay #(
    parameter int MAX_BATCH = 10,
    parameter int MAX_TIMESTEPS = 20,
    parameter int CAPTURE_BATCH_DEPTH = MAX_BATCH,
    parameter int CAPTURE_TIMESTEP_DEPTH = MAX_TIMESTEPS,
    parameter int MAX_FEATURES = 2048,
    parameter int MAX_CLASSES = 11,
    parameter int LANES = 8,
    parameter int ENGINES = 12,
    parameter int VERSION_WIDTH = 16,
    parameter int FRAME_WIDTH = 8,
    parameter int BATCH_INDEX_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int TIMESTEP_INDEX_WIDTH = MAX_TIMESTEPS <= 2 ? 1 : $clog2(MAX_TIMESTEPS),
    parameter int TIMESTEP_COUNT_WIDTH = $clog2(MAX_TIMESTEPS + 1),
    parameter int FEATURE_INDEX_WIDTH = MAX_FEATURES <= 2 ? 1 : $clog2(MAX_FEATURES),
    parameter int FEATURE_COUNT_WIDTH = $clog2(MAX_FEATURES + 1),
    parameter int CLASS_INDEX_WIDTH = MAX_CLASSES <= 2 ? 1 : $clog2(MAX_CLASSES),
    parameter int CLASS_COUNT_WIDTH = $clog2(MAX_CLASSES + 1),
    // Override these two widths to match frontend OUTPUT_MAX independently.
    parameter int WAVE_CLASS_INDEX_WIDTH = CLASS_INDEX_WIDTH,
    parameter int WAVE_CLASS_COUNT_WIDTH = CLASS_COUNT_WIDTH,
    parameter int LANE_INDEX_WIDTH = LANES <= 2 ? 1 : $clog2(LANES),
    parameter int ENGINE_INDEX_WIDTH = ENGINES <= 2 ? 1 : $clog2(ENGINES),
    parameter int TRACE_ADDR_WIDTH = CAPTURE_BATCH_DEPTH * MAX_FEATURES <= 2 ? 1 :
        $clog2(CAPTURE_BATCH_DEPTH * MAX_FEATURES),
    parameter int TRACE_HISTORY_ADDR_WIDTH =
        CAPTURE_BATCH_DEPTH * CAPTURE_TIMESTEP_DEPTH * MAX_FEATURES <= 2 ? 1 :
        $clog2(CAPTURE_BATCH_DEPTH * CAPTURE_TIMESTEP_DEPTH * MAX_FEATURES),
    parameter int LOGIT_HISTORY_ADDR_WIDTH =
        CAPTURE_BATCH_DEPTH * CAPTURE_TIMESTEP_DEPTH * MAX_CLASSES <= 2 ? 1 :
        $clog2(CAPTURE_BATCH_DEPTH * CAPTURE_TIMESTEP_DEPTH * MAX_CLASSES),
    parameter logic signed [15:0] DEFAULT_TRACE_LEAK_Q8_8 = 16'sd250,
    parameter bit ENABLE_COMPUTED_TRACE_CAPTURE = 1'b0
) (
    input  wire logic i_clk,
    input  wire logic i_rst_n,
    input  wire logic i_abort,
    input  wire logic i_cfg_valid,
    output logic o_cfg_ready,
    input  wire logic [BATCH_COUNT_WIDTH-1:0] i_cfg_batch_count,
    input  wire logic [TIMESTEP_INDEX_WIDTH-1:0] i_cfg_timestep_base,
    input  wire logic [TIMESTEP_COUNT_WIDTH-1:0] i_cfg_timestep_count,
    input  wire logic i_cfg_trace_state_reset,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0] i_cfg_feature_count,
    input  wire logic [CLASS_COUNT_WIDTH-1:0] i_cfg_class_count,
    input  wire logic i_cfg_trace_leak_valid,
    input  wire logic signed [15:0] i_cfg_trace_leak_q8_8,
    input  wire logic i_cfg_computed_trace_capture_enable,
    output logic o_cfg_error,

    input  wire logic i_feature_trace_wave_valid,
    output logic o_feature_trace_wave_ready,
    input  wire logic [LANES-1:0] i_feature_trace_wave_mask,
    input  wire logic [FEATURE_INDEX_WIDTH-1:0] i_feature_trace_wave_base,
    input  wire logic [LANES*16-1:0] i_feature_trace_wave_q8_8_flat,
    input  wire logic [VERSION_WIDTH-1:0] i_feature_trace_wave_version,
    input  wire logic [FRAME_WIDTH-1:0] i_feature_trace_wave_frame_index,
    input  wire logic i_feature_trace_wave_first,
    input  wire logic i_feature_trace_wave_last,
    input  wire logic i_feature_trace_wave_sample_reset,
    input  wire logic i_feature_trace_wave_sample_last,
    // Valid with every feature wave; sampled at the first wave of each sample.
    input  wire logic [CLASS_INDEX_WIDTH-1:0] i_sample_label,

    input  wire logic i_logit_wave_valid,
    output logic o_logit_wave_ready,
    input  wire logic [ENGINES-1:0] i_logit_wave_mask,
    input  wire logic [WAVE_CLASS_INDEX_WIDTH-1:0] i_logit_wave_base,
    input  wire logic [WAVE_CLASS_COUNT_WIDTH-1:0] i_logit_wave_count,
    input  wire logic [ENGINES*16-1:0] i_logit_wave_q8_8_flat,
    input  wire logic [VERSION_WIDTH-1:0] i_logit_wave_version,
    input  wire logic [FRAME_WIDTH-1:0] i_logit_wave_frame_index,
    input  wire logic i_logit_wave_frame_last,
    input  wire logic i_logit_wave_sample_last,

    input  wire logic i_replay_valid,
    output logic o_replay_ready,
    input  wire logic [TIMESTEP_INDEX_WIDTH-1:0] i_replay_timestep,
    output logic o_replay_error,
    output logic o_capture_active,
    output logic o_capture_complete,
    output logic o_replay_busy,
    output logic o_idle,
    output logic o_protocol_error,

    // Direct counterparts of the classifier online-learning observation ports.
    output logic o_feature_trace_valid,
    input  wire logic i_feature_trace_ready,
    output logic [BATCH_INDEX_WIDTH-1:0] o_feature_trace_sample_id,
    output logic [FEATURE_INDEX_WIDTH-1:0] o_feature_trace_index,
    output logic signed [15:0] o_feature_trace_q8_8,
    output logic [VERSION_WIDTH-1:0] o_feature_trace_version,
    output logic o_logit_row_valid,
    input  wire logic i_logit_row_ready,
    output logic [BATCH_INDEX_WIDTH-1:0] o_logit_sample_id,
    output logic [CLASS_INDEX_WIDTH-1:0] o_logit_label,
    output logic [VERSION_WIDTH-1:0] o_logit_row_version,
    output logic o_logit_valid,
    input  wire logic i_logit_ready,
    output logic signed [15:0] o_logit_q8_8,
    output logic o_observation_done,
    input  wire logic i_observation_done_ready,

    // Live u_trace result fork, not the raw pooled feature wave or replay.
    // Sample id is the batch index; version is the raw sample-state version.
    output logic o_computed_trace_valid,
    input  wire logic i_computed_trace_ready,
    output logic [BATCH_INDEX_WIDTH-1:0] o_computed_trace_sample_id,
    output logic [FEATURE_INDEX_WIDTH-1:0] o_computed_trace_index,
    output logic [TIMESTEP_INDEX_WIDTH-1:0] o_computed_trace_timestep,
    output logic [FRAME_WIDTH-1:0] o_computed_trace_frame_index,
    output logic [VERSION_WIDTH-1:0] o_computed_trace_version,
    output logic signed [15:0] o_computed_trace_q8_8
);
    typedef enum logic [3:0] {
        IDLE, CAP_FEATURE, TRACE_ISSUE, TRACE_WAIT, CAP_LOGIT, LOGIT_STORE,
        CAPTURED, REPLAY_META, REPLAY_FEATURE_READ, REPLAY_FEATURE,
        REPLAY_ROW, REPLAY_LOGIT_READ, REPLAY_LOGIT, REPLAY_DONE, DRAIN
    } state_t;
    state_t state_q;
    logic [BATCH_COUNT_WIDTH-1:0] batch_count_q;
    logic [TIMESTEP_INDEX_WIDTH-1:0] timestep_base_q;
    logic [TIMESTEP_COUNT_WIDTH-1:0] timestep_count_q;
    logic [FEATURE_COUNT_WIDTH-1:0] feature_count_q;
    logic [CLASS_COUNT_WIDTH-1:0] class_count_q;
    logic signed [15:0] leak_q;
    logic computed_trace_capture_enable_q;
    logic [BATCH_INDEX_WIDTH-1:0] sample_q;
    logic [TIMESTEP_INDEX_WIDTH-1:0] timestep_q;
    logic [FEATURE_INDEX_WIDTH-1:0] feature_q;
    logic [CLASS_INDEX_WIDTH-1:0] class_q;
    logic [LANE_INDEX_WIDTH-1:0] lane_q;
    logic [ENGINE_INDEX_WIDTH-1:0] engine_q;
    logic [LANES*16-1:0] feature_wave_q;
    logic [ENGINES*16-1:0] logit_wave_q;
    logic [CLASS_INDEX_WIDTH-1:0] sample_label_q;
    logic [VERSION_WIDTH-1:0] sample_version_q;
    logic capture_complete_q;

    logic signed [15:0] trace_history
        [0:CAPTURE_BATCH_DEPTH*CAPTURE_TIMESTEP_DEPTH*MAX_FEATURES-1];
    logic signed [15:0] logit_history
        [0:CAPTURE_BATCH_DEPTH*CAPTURE_TIMESTEP_DEPTH*MAX_CLASSES-1];
    logic [CLASS_INDEX_WIDTH-1:0] label_memory [0:CAPTURE_BATCH_DEPTH-1];
    logic [VERSION_WIDTH-1:0] version_memory [0:CAPTURE_BATCH_DEPTH-1];
    logic [TRACE_ADDR_WIDTH-1:0] trace_addr;
    logic [TRACE_HISTORY_ADDR_WIDTH-1:0] trace_history_addr;
    logic [LOGIT_HISTORY_ADDR_WIDTH-1:0] logit_history_addr;
    logic trace_ready, trace_valid, trace_idle;
    logic trace_result_ready, trace_capture_fire;
    logic trace_reset_error, trace_address_error;
    logic [TRACE_ADDR_WIDTH-1:0] trace_result_addr;
    logic signed [15:0] trace_result;
    logic cfg_fire, cfg_legal, capture_fault;
    logic feature_legal, logit_legal;
    logic first_local_sample_wave, first_global_sample_wave;
    logic final_feature_wave, final_logit_wave, final_timestep;
    logic [LANES-1:0] expected_feature_mask;
    logic [ENGINES-1:0] expected_logit_mask;

    assign trace_addr = TRACE_ADDR_WIDTH'(int'(sample_q) * MAX_FEATURES + int'(feature_q));
    assign trace_history_addr = TRACE_HISTORY_ADDR_WIDTH'(
        (int'(sample_q) * CAPTURE_TIMESTEP_DEPTH + int'(timestep_q)) *
        MAX_FEATURES + int'(feature_q));
    assign logit_history_addr = LOGIT_HISTORY_ADDR_WIDTH'(
        (int'(sample_q) * CAPTURE_TIMESTEP_DEPTH + int'(timestep_q)) *
        MAX_CLASSES + int'(class_q));
    assign final_timestep = int'(timestep_q) + 1 == int'(timestep_count_q);
    assign first_local_sample_wave = timestep_q == 0 && feature_q == 0;
    assign first_global_sample_wave =
        int'(timestep_base_q) + int'(timestep_q) == 0 && feature_q == 0;
    assign final_feature_wave = int'(feature_q) + LANES >= int'(feature_count_q);
    assign final_logit_wave = int'(class_q) + ENGINES >= int'(class_count_q);

    always_comb begin
        expected_feature_mask = '0;
        expected_logit_mask = '0;
        for (int n = 0; n < LANES; n++)
            expected_feature_mask[n] = int'(feature_q) + n < int'(feature_count_q);
        for (int n = 0; n < ENGINES; n++)
            expected_logit_mask[n] = int'(class_q) + n < int'(class_count_q);
        cfg_legal = i_cfg_batch_count != 0 &&
            int'(i_cfg_batch_count) <= CAPTURE_BATCH_DEPTH &&
            i_cfg_timestep_count != 0 &&
            int'(i_cfg_timestep_count) <= CAPTURE_TIMESTEP_DEPTH &&
            int'(i_cfg_timestep_base) + int'(i_cfg_timestep_count) <= MAX_TIMESTEPS &&
            64'(i_cfg_timestep_base) + 64'(i_cfg_timestep_count) <=
                (64'd1 << FRAME_WIDTH) &&
            i_cfg_feature_count != 0 && int'(i_cfg_feature_count) <= MAX_FEATURES &&
            i_cfg_class_count != 0 && int'(i_cfg_class_count) <= MAX_CLASSES;
        feature_legal = i_feature_trace_wave_mask == expected_feature_mask &&
            i_feature_trace_wave_base == feature_q &&
            int'(i_feature_trace_wave_frame_index) ==
                int'(timestep_base_q) + int'(timestep_q) &&
            i_feature_trace_wave_first == (feature_q == 0) &&
            i_feature_trace_wave_last == final_feature_wave &&
            i_feature_trace_wave_sample_reset == first_global_sample_wave &&
            i_feature_trace_wave_sample_last == (final_timestep && final_feature_wave) &&
            int'(i_sample_label) < int'(class_count_q) &&
            (first_local_sample_wave || (i_sample_label == sample_label_q &&
                i_feature_trace_wave_version == sample_version_q));
        logit_legal = i_logit_wave_mask == expected_logit_mask &&
            int'(i_logit_wave_base) == int'(class_q) &&
            int'(i_logit_wave_count) == int'(class_count_q) &&
            i_logit_wave_version == sample_version_q &&
            int'(i_logit_wave_frame_index) ==
                int'(timestep_base_q) + int'(timestep_q) &&
            i_logit_wave_frame_last == final_logit_wave &&
            i_logit_wave_sample_last == (final_timestep && final_logit_wave);
    end

    assign o_cfg_ready = i_rst_n && !i_abort && trace_idle &&
        (state_q == IDLE || state_q == CAP_FEATURE || state_q == CAP_LOGIT || state_q == CAPTURED);
    assign cfg_fire = i_cfg_valid && o_cfg_ready;
    assign o_capture_active = i_rst_n && !i_abort &&
        (state_q == CAP_FEATURE || state_q == TRACE_ISSUE || state_q == TRACE_WAIT ||
         state_q == CAP_LOGIT || state_q == LOGIT_STORE);
    assign o_feature_trace_wave_ready = o_capture_active && state_q == CAP_FEATURE && !cfg_fire;
    assign o_logit_wave_ready = o_capture_active && state_q == CAP_LOGIT && !cfg_fire;
    assign o_capture_complete = capture_complete_q && i_rst_n && !i_abort;
    assign o_replay_ready = state_q == CAPTURED && o_capture_complete && !cfg_fire;
    assign o_replay_busy = i_rst_n && !i_abort &&
        (state_q == REPLAY_META || state_q == REPLAY_FEATURE_READ || state_q == REPLAY_FEATURE ||
         state_q == REPLAY_ROW || state_q == REPLAY_LOGIT_READ || state_q == REPLAY_LOGIT ||
         state_q == REPLAY_DONE);
    assign o_idle = i_rst_n && trace_idle && (state_q == IDLE || state_q == CAPTURED);
    assign o_feature_trace_valid = o_replay_busy && state_q == REPLAY_FEATURE;
    assign o_logit_row_valid = o_replay_busy && state_q == REPLAY_ROW;
    assign o_logit_valid = o_replay_busy && state_q == REPLAY_LOGIT;
    assign o_observation_done = o_replay_busy && state_q == REPLAY_DONE;
    assign o_feature_trace_sample_id = sample_q;
    assign o_feature_trace_index = feature_q;
    assign o_logit_sample_id = sample_q;

    generate
        if (ENABLE_COMPUTED_TRACE_CAPTURE) begin : g_computed_trace_capture
            assign o_computed_trace_valid = computed_trace_capture_enable_q &&
                i_rst_n && !i_abort && state_q == TRACE_WAIT && trace_valid &&
                trace_result_addr == trace_addr && !trace_reset_error && !trace_address_error;
            assign o_computed_trace_sample_id = sample_q;
            assign o_computed_trace_index = feature_q;
            assign o_computed_trace_timestep = TIMESTEP_INDEX_WIDTH'(
                int'(timestep_base_q) + int'(timestep_q));
            assign o_computed_trace_frame_index = FRAME_WIDTH'(
                int'(timestep_base_q) + int'(timestep_q));
            assign o_computed_trace_version = sample_version_q;
            assign o_computed_trace_q8_8 = trace_result;
        end else begin : g_no_computed_trace_capture
            assign o_computed_trace_valid = 1'b0;
            assign o_computed_trace_sample_id = '0;
            assign o_computed_trace_index = '0;
            assign o_computed_trace_timestep = '0;
            assign o_computed_trace_frame_index = '0;
            assign o_computed_trace_version = '0;
            assign o_computed_trace_q8_8 = '0;
        end
    endgenerate

    // History is always ready: commit both consumers on one result handshake.
    // Phase-local VALID checks avoid feeding configuration admission back into
    // the consumer's READY path through trace-idle.
    // Abort or the DRAIN phase consumes discarded work even with a stalled tap.
    assign trace_result_ready = !ENABLE_COMPUTED_TRACE_CAPTURE ||
        !computed_trace_capture_enable_q || state_q != TRACE_WAIT ||
        i_abort || i_computed_trace_ready;
    assign trace_capture_fire = state_q == TRACE_WAIT && trace_valid &&
        trace_result_ready && i_rst_n && !i_abort && !cfg_fire && !capture_fault;

    // Wrong-port traffic is rejected at wave boundaries. While serializing a
    // wave, the producer may legally hold the next wave against backpressure.
    assign capture_fault = !cfg_fire && o_capture_active &&
        ((state_q == CAP_FEATURE && (i_logit_wave_valid ||
            (i_feature_trace_wave_valid && !feature_legal))) ||
         (state_q == CAP_LOGIT && (i_feature_trace_wave_valid ||
            (i_logit_wave_valid && !logit_legal))) ||
         (state_q == TRACE_WAIT && trace_valid && trace_result_addr != trace_addr) ||
         trace_reset_error || trace_address_error);

    unified_scnn_q8_8_trace_state_engine #(
        .NUM_NEURONS(CAPTURE_BATCH_DEPTH * MAX_FEATURES),
        .ADDR_WIDTH(TRACE_ADDR_WIDTH)
    ) u_trace (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_sample_reset(cfg_fire && i_cfg_trace_state_reset),
        .i_valid(state_q == TRACE_ISSUE && !i_abort && !capture_fault),
        .o_ready(trace_ready), .i_addr(trace_addr),
        .i_spike_q8_8($signed(feature_wave_q[int'(lane_q)*16 +: 16])),
        .i_leak_q8_8(leak_q), .o_valid(trace_valid), .i_ready(trace_result_ready),
        .o_addr(trace_result_addr), .o_trace_q8_8(trace_result),
        .o_idle(trace_idle), .o_reset_while_busy_error(trace_reset_error),
        .o_address_error(trace_address_error)
    );

    // Separate synchronous RAM ports, with no reset or clear branch. Replay
    // loads a payload once, then its VALID state holds it until acceptance.
    always_ff @(posedge i_clk) begin
        if (i_rst_n && !i_abort && !cfg_fire && !capture_fault) begin
            if (trace_capture_fire)
                trace_history[trace_history_addr] <= trace_result;
            if (state_q == LOGIT_STORE)
                logit_history[logit_history_addr] <= logit_wave_q[int'(engine_q)*16 +: 16];
            if (state_q == CAP_FEATURE && i_feature_trace_wave_valid &&
                    first_local_sample_wave) begin
                label_memory[sample_q] <= i_sample_label;
                version_memory[sample_q] <= i_feature_trace_wave_version;
            end
            if (state_q == REPLAY_META) begin
                o_logit_label <= label_memory[sample_q];
                o_logit_row_version <= version_memory[sample_q];
                o_feature_trace_version <= version_memory[sample_q];
            end
            if (state_q == REPLAY_FEATURE_READ)
                o_feature_trace_q8_8 <= trace_history[trace_history_addr];
            if (state_q == REPLAY_LOGIT_READ)
                o_logit_q8_8 <= logit_history[logit_history_addr];
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= IDLE;
            batch_count_q <= '0;
            timestep_base_q <= '0;
            timestep_count_q <= '0;
            feature_count_q <= '0;
            class_count_q <= '0;
            leak_q <= DEFAULT_TRACE_LEAK_Q8_8;
            computed_trace_capture_enable_q <= 1'b0;
            sample_q <= '0;
            timestep_q <= '0;
            feature_q <= '0;
            class_q <= '0;
            lane_q <= '0;
            engine_q <= '0;
            feature_wave_q <= '0;
            logit_wave_q <= '0;
            sample_label_q <= '0;
            sample_version_q <= '0;
            capture_complete_q <= 1'b0;
            o_cfg_error <= 1'b0;
            o_protocol_error <= 1'b0;
            o_replay_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= DRAIN;
            capture_complete_q <= 1'b0;
        end else if (cfg_fire) begin
            state_q <= cfg_legal ? CAP_FEATURE : IDLE;
            batch_count_q <= i_cfg_batch_count;
            timestep_base_q <= i_cfg_timestep_base;
            timestep_count_q <= i_cfg_timestep_count;
            feature_count_q <= i_cfg_feature_count;
            class_count_q <= i_cfg_class_count;
            leak_q <= i_cfg_trace_leak_valid ? i_cfg_trace_leak_q8_8 : DEFAULT_TRACE_LEAK_Q8_8;
            computed_trace_capture_enable_q <= ENABLE_COMPUTED_TRACE_CAPTURE &&
                i_cfg_computed_trace_capture_enable;
            sample_q <= '0;
            timestep_q <= '0;
            feature_q <= '0;
            class_q <= '0;
            lane_q <= '0;
            engine_q <= '0;
            sample_label_q <= '0;
            sample_version_q <= '0;
            capture_complete_q <= 1'b0;
            o_cfg_error <= !cfg_legal;
            o_protocol_error <= 1'b0;
            o_replay_error <= 1'b0;
        end else if (capture_fault) begin
            o_protocol_error <= 1'b1;
            capture_complete_q <= 1'b0;
            state_q <= DRAIN;
        end else begin
            case (state_q)
                CAP_FEATURE: if (i_feature_trace_wave_valid) begin
                    feature_wave_q <= i_feature_trace_wave_q8_8_flat;
                    lane_q <= '0;
                    if (first_local_sample_wave) begin
                        sample_label_q <= i_sample_label;
                        sample_version_q <= i_feature_trace_wave_version;
                    end
                    state_q <= TRACE_ISSUE;
                end
                TRACE_ISSUE: if (trace_ready) state_q <= TRACE_WAIT;
                TRACE_WAIT: if (trace_capture_fire) begin
                    if (int'(feature_q) + 1 == int'(feature_count_q)) begin
                        class_q <= '0;
                        state_q <= CAP_LOGIT;
                    end else begin
                        feature_q <= feature_q + 1'b1;
                        if (int'(lane_q) + 1 == LANES) begin
                            state_q <= CAP_FEATURE;
                        end else begin
                            lane_q <= lane_q + 1'b1;
                            state_q <= TRACE_ISSUE;
                        end
                    end
                end
                CAP_LOGIT: if (i_logit_wave_valid) begin
                    logit_wave_q <= i_logit_wave_q8_8_flat;
                    engine_q <= '0;
                    state_q <= LOGIT_STORE;
                end
                LOGIT_STORE: begin
                    if (int'(class_q) + 1 == int'(class_count_q)) begin
                        feature_q <= '0;
                        class_q <= '0;
                        if (final_timestep) begin
                            timestep_q <= '0;
                            if (int'(sample_q) + 1 == int'(batch_count_q)) begin
                                capture_complete_q <= 1'b1;
                                state_q <= CAPTURED;
                            end else begin
                                sample_q <= sample_q + 1'b1;
                                state_q <= CAP_FEATURE;
                            end
                        end else begin
                            timestep_q <= timestep_q + 1'b1;
                            state_q <= CAP_FEATURE;
                        end
                    end else begin
                        class_q <= class_q + 1'b1;
                        if (int'(engine_q) + 1 == ENGINES) begin
                            state_q <= CAP_LOGIT;
                        end else begin
                            engine_q <= engine_q + 1'b1;
                        end
                    end
                end
                CAPTURED: if (i_replay_valid) begin
                    if (int'(i_replay_timestep) >= int'(timestep_count_q)) begin
                        o_replay_error <= 1'b1;
                    end else begin
                        sample_q <= '0;
                        feature_q <= '0;
                        class_q <= '0;
                        timestep_q <= i_replay_timestep;
                        state_q <= REPLAY_META;
                    end
                end
                REPLAY_META: state_q <= REPLAY_FEATURE_READ;
                REPLAY_FEATURE_READ: state_q <= REPLAY_FEATURE;
                REPLAY_FEATURE: if (i_feature_trace_ready) begin
                    if (int'(feature_q) + 1 == int'(feature_count_q))
                        state_q <= REPLAY_ROW;
                    else begin
                        feature_q <= feature_q + 1'b1;
                        state_q <= REPLAY_FEATURE_READ;
                    end
                end
                REPLAY_ROW: if (i_logit_row_ready) begin
                    class_q <= '0;
                    state_q <= REPLAY_LOGIT_READ;
                end
                REPLAY_LOGIT_READ: state_q <= REPLAY_LOGIT;
                REPLAY_LOGIT: if (i_logit_ready) begin
                    if (int'(class_q) + 1 == int'(class_count_q)) begin
                        if (int'(sample_q) + 1 == int'(batch_count_q))
                            state_q <= REPLAY_DONE;
                        else begin
                            sample_q <= sample_q + 1'b1;
                            feature_q <= '0;
                            state_q <= REPLAY_META;
                        end
                    end else begin
                        class_q <= class_q + 1'b1;
                        state_q <= REPLAY_LOGIT_READ;
                    end
                end
                REPLAY_DONE: if (i_observation_done_ready) state_q <= CAPTURED;
                DRAIN: if (trace_idle) state_q <= IDLE;
                default: state_q <= IDLE;
            endcase
        end
    end

    initial begin
        if (CAPTURE_BATCH_DEPTH < 1 || CAPTURE_BATCH_DEPTH > MAX_BATCH ||
                CAPTURE_TIMESTEP_DEPTH < 1 ||
                CAPTURE_TIMESTEP_DEPTH > MAX_TIMESTEPS)
            $error("SCNN classifier capture depths are invalid: max_batch=%0d max_timesteps=%0d capture_batch=%0d capture_timesteps=%0d",
                MAX_BATCH, MAX_TIMESTEPS,
                CAPTURE_BATCH_DEPTH, CAPTURE_TIMESTEP_DEPTH);
    end
endmodule

`default_nettype wire
