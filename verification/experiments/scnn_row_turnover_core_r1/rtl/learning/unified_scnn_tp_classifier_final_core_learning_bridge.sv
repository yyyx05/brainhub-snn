`timescale 1ns/1ps
`default_nettype none

// Production coordinator for SCNN classifier-only TP adaptation. The final
// core emits one complete sample-major BxT observation window. This bridge
// replays each timestep through the shared classifier learner, waits for every
// RMW window to retire, and exposes one batch commit before accepting a new
// observation window. Captured timesteps intentionally belong to one frozen
// forward-model version; this is windowed online adaptation, not immediate
// recurrent recomputation after every timestep update.
module unified_scnn_tp_classifier_final_core_learning_bridge #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int MAX_BATCH = 10,
    parameter int MAX_TIMESTEPS = 20,
    parameter int CLASSIFIER_CAPTURE_MAX_BATCH = MAX_BATCH,
    parameter int CLASSIFIER_CAPTURE_MAX_TIMESTEPS = MAX_TIMESTEPS,
    parameter int MAX_FEATURES = 2048,
    parameter int MAX_CLASSES = 11,
    parameter int LANES = 8,
    parameter int ENGINES = 12,
    parameter int VERSION_WIDTH = 16,
    parameter int FRAME_WIDTH = 8,
    parameter int MAX_GRADIENT_ELEMENTS = MAX_FEATURES * MAX_CLASSES,
    parameter int BATCH_INDEX_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int TIMESTEP_INDEX_WIDTH = MAX_TIMESTEPS <= 2 ? 1 : $clog2(MAX_TIMESTEPS),
    parameter int TIMESTEP_COUNT_WIDTH = $clog2(MAX_TIMESTEPS + 1),
    parameter int FEATURE_INDEX_WIDTH = MAX_FEATURES <= 2 ? 1 : $clog2(MAX_FEATURES),
    parameter int FEATURE_COUNT_WIDTH = $clog2(MAX_FEATURES + 1),
    parameter int CLASS_INDEX_WIDTH = MAX_CLASSES <= 2 ? 1 : $clog2(MAX_CLASSES),
    parameter int CLASS_COUNT_WIDTH = $clog2(MAX_CLASSES + 1),
    parameter int WAVE_CLASS_INDEX_WIDTH = CLASS_INDEX_WIDTH,
    parameter int WAVE_CLASS_COUNT_WIDTH = CLASS_COUNT_WIDTH,
    parameter bit ENABLE_COMPUTED_TRACE_CAPTURE = 1'b0,
    parameter bit USE_EXTERNAL_OUTPUT_LEARNER = 1'b0
) (
    input  wire logic                                  i_clk,
    input  wire logic                                  i_rst_n,
    input  wire logic                                  i_clear_error,

    input  wire logic                                  i_cfg_valid,
    output logic                                  o_cfg_ready,
    input  wire logic [PROFILE_ID_WIDTH-1:0]           i_cfg_profile_id,
    input  wire logic [TENSOR_ID_WIDTH-1:0]            i_cfg_tensor_id,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]           i_cfg_parameter_base,
    input  wire logic [BATCH_COUNT_WIDTH-1:0]          i_cfg_batch_count,
    input  wire logic [TIMESTEP_INDEX_WIDTH-1:0]       i_cfg_timestep_base,
    input  wire logic [TIMESTEP_COUNT_WIDTH-1:0]       i_cfg_timestep_count,
    input  wire logic                                  i_cfg_trace_state_reset,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0]        i_cfg_feature_count,
    input  wire logic [CLASS_COUNT_WIDTH-1:0]          i_cfg_class_count,
    input  wire logic                                  i_cfg_trace_leak_valid,
    input  wire logic signed [15:0]                    i_cfg_trace_leak_q8_8,
    input  wire logic                                  i_cfg_computed_trace_capture_enable,
    input  wire logic [VOTE_WIDTH-1:0]                 i_cfg_vote_increment,
    input  wire logic [VOTE_WIDTH-1:0]                 i_cfg_vote_threshold,
    input  wire logic signed [CODE_WIDTH-1:0]          i_cfg_code_min,
    input  wire logic signed [CODE_WIDTH-1:0]          i_cfg_code_max,
    output logic                                  o_cfg_error,
    output logic                                  o_active,
    input  wire logic                                  i_release,
    output logic                                  o_release_ready,
    input  wire logic                                  i_learning_enable,

    input  wire logic                                  i_feature_trace_wave_valid,
    output logic                                  o_feature_trace_wave_ready,
    input  wire logic [LANES-1:0]                     i_feature_trace_wave_mask,
    input  wire logic [FEATURE_INDEX_WIDTH-1:0]       i_feature_trace_wave_base,
    input  wire logic [LANES*16-1:0]                  i_feature_trace_wave_q8_8_flat,
    input  wire logic [VERSION_WIDTH-1:0]             i_feature_trace_wave_version,
    input  wire logic [FRAME_WIDTH-1:0]               i_feature_trace_wave_frame_index,
    input  wire logic                                  i_feature_trace_wave_first,
    input  wire logic                                  i_feature_trace_wave_last,
    input  wire logic                                  i_feature_trace_wave_sample_reset,
    input  wire logic                                  i_feature_trace_wave_sample_last,
    input  wire logic [CLASS_INDEX_WIDTH-1:0]         i_sample_label,

    input  wire logic                                  i_logit_wave_valid,
    output logic                                  o_logit_wave_ready,
    input  wire logic [ENGINES-1:0]                   i_logit_wave_mask,
    input  wire logic [WAVE_CLASS_INDEX_WIDTH-1:0]    i_logit_wave_base,
    input  wire logic [WAVE_CLASS_COUNT_WIDTH-1:0]    i_logit_wave_count,
    input  wire logic [ENGINES*16-1:0]                i_logit_wave_q8_8_flat,
    input  wire logic [VERSION_WIDTH-1:0]             i_logit_wave_version,
    input  wire logic [FRAME_WIDTH-1:0]               i_logit_wave_frame_index,
    input  wire logic                                  i_logit_wave_frame_last,
    input  wire logic                                  i_logit_wave_sample_last,

    output logic                                  o_window_commit_valid,
    input  wire logic                                  i_window_commit_ready,
    output logic [PROFILE_ID_WIDTH-1:0]          o_window_commit_profile_id,
    output logic [31:0]                           o_window_commit_sequence,
    output logic                                  o_timestep_learning_done,
    output logic                                  o_batch_learning_done,
    output logic [TIMESTEP_COUNT_WIDTH-1:0]       o_completed_timestep_count,
    output logic                                  o_inference_hold,
    output logic                                  o_window_busy,

    output logic                                  o_read_valid,
    input  wire logic                                  i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0]          o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]           o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]          o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]                 o_read_tag,
    input  wire logic                                  i_read_response_valid,
    output logic                                  o_read_response_ready,
    input  wire logic signed [CODE_WIDTH-1:0]         i_read_response_code,
    input  wire logic signed [RESIDUAL_WIDTH-1:0]     i_read_response_residual,
    input  wire logic                                  i_read_response_error,
    output logic                                  o_result_valid,
    input  wire logic                                  i_result_ready,
    output logic [PROFILE_ID_WIDTH-1:0]          o_result_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]           o_result_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]          o_result_parameter_addr,
    output logic [TAG_WIDTH-1:0]                 o_result_tag,
    output logic signed [CODE_WIDTH-1:0]         o_result_code,
    output logic signed [RESIDUAL_WIDTH-1:0]     o_result_residual,
    output logic                                  o_write_code_enable,
    output logic                                  o_write_residual_enable,
    output logic                                  o_write_code_fire,
    output logic                                  o_write_residual_fire,
    output logic                                  o_dirty,
    output logic                                  o_saturated,

    output logic                                  o_capture_complete,
    output logic                                  o_observation_error,
    output logic                                  o_context_error,
    output logic                                  o_gradient_overflow,
    output logic                                  o_parameter_error,
    output logic                                  o_protocol_error,

    // Computed capture-time trace with unnormalized sample-state metadata.
    output logic                                  o_computed_trace_valid,
    input  wire logic                                  i_computed_trace_ready,
    output logic [BATCH_INDEX_WIDTH-1:0]          o_computed_trace_sample_id,
    output logic [FEATURE_INDEX_WIDTH-1:0]        o_computed_trace_index,
    output logic [TIMESTEP_INDEX_WIDTH-1:0]       o_computed_trace_timestep,
    output logic [FRAME_WIDTH-1:0]                o_computed_trace_frame_index,
    output logic [VERSION_WIDTH-1:0]              o_computed_trace_version,
    output logic signed [15:0]                    o_computed_trace_q8_8,

    // Normalized scalar client boundary for the single shared FC/SCNN output
    // learner. These ports are active only when USE_EXTERNAL_OUTPUT_LEARNER=1.
    output logic                                  o_external_cfg_valid,
    input  wire logic                             i_external_cfg_ready,
    output logic [PROFILE_ID_WIDTH-1:0]           o_external_cfg_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]            o_external_cfg_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]           o_external_cfg_parameter_base,
    output logic [BATCH_COUNT_WIDTH-1:0]          o_external_cfg_batch_count,
    output logic [FEATURE_COUNT_WIDTH-1:0]        o_external_cfg_source_count,
    output logic [CLASS_COUNT_WIDTH-1:0]          o_external_cfg_output_count,
    output logic [VOTE_WIDTH-1:0]                 o_external_cfg_vote_increment,
    output logic [VOTE_WIDTH-1:0]                 o_external_cfg_vote_threshold,
    output logic signed [CODE_WIDTH-1:0]          o_external_cfg_code_min,
    output logic signed [CODE_WIDTH-1:0]          o_external_cfg_code_max,
    input  wire logic                             i_external_cfg_error,
    input  wire logic                             i_external_active,
    output logic                                  o_external_release,
    input  wire logic                             i_external_release_ready,
    output logic                                  o_external_source_trace_valid,
    input  wire logic                             i_external_source_trace_ready,
    output logic [BATCH_INDEX_WIDTH-1:0]          o_external_source_trace_sample_id,
    output logic [FEATURE_INDEX_WIDTH-1:0]        o_external_source_trace_index,
    output logic [15:0]                           o_external_source_trace_q016,
    input  wire logic                             i_external_source_trace_error,
    output logic                                  o_external_logit_row_valid,
    input  wire logic                             i_external_logit_row_ready,
    output logic [BATCH_INDEX_WIDTH-1:0]          o_external_logit_sample_id,
    output logic [CLASS_INDEX_WIDTH-1:0]          o_external_logit_label,
    output logic                                  o_external_logit_valid,
    input  wire logic                             i_external_logit_ready,
    output logic signed [15:0]                    o_external_logit_q6,
    input  wire logic                             i_external_logit_error,
    input  wire logic                             i_external_logit_row_done,
    output logic                                  o_external_observation_done,
    input  wire logic                             i_external_observation_done_ready,
    input  wire logic                             i_external_window_done,
    input  wire logic                             i_external_window_busy,
    input  wire logic                             i_external_context_error,
    input  wire logic                             i_external_gradient_overflow,
    input  wire logic                             i_external_update_config_error,
    input  wire logic                             i_external_request_error,
    input  wire logic                             i_external_store_error,
    input  wire logic                             i_external_internal_error
);
    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_CAPTURE,
        STATE_START_REPLAY,
        STATE_WAIT_WINDOW,
        STATE_COMMIT,
        STATE_REARM,
        STATE_RELEASE_DRAIN
    } state_t;

    state_t state_q;
    logic [PROFILE_ID_WIDTH-1:0] profile_id_q;
    logic [BATCH_COUNT_WIDTH-1:0] batch_count_q;
    logic [TIMESTEP_INDEX_WIDTH-1:0] timestep_base_q;
    logic [TIMESTEP_COUNT_WIDTH-1:0] timestep_count_q;
    logic [FEATURE_COUNT_WIDTH-1:0] feature_count_q;
    logic [CLASS_COUNT_WIDTH-1:0] class_count_q;
    logic trace_leak_valid_q;
    logic signed [15:0] trace_leak_q;
    logic trace_state_reset_q;
    logic computed_trace_capture_enable_q;
    logic [TIMESTEP_INDEX_WIDTH-1:0] replay_timestep_q;
    logic [TIMESTEP_COUNT_WIDTH-1:0] completed_timestep_count_q;
    logic [31:0] commit_sequence_q;
    logic capture_started_q;
    logic protocol_error_q;

    logic cfg_fire;
    logic release_fire;
    logic replay_cfg_valid;
    logic replay_cfg_ready;
    logic replay_cfg_error;
    logic replay_abort;
    logic replay_capture_complete;
    logic replay_start_valid;
    logic replay_start_ready;
    logic replay_error;
    logic replay_busy;
    logic replay_idle;
    logic replay_protocol_error;
    logic replay_feature_wave_ready;
    logic replay_logit_wave_ready;
    logic replay_feature_valid;
    logic replay_feature_ready;
    logic [BATCH_INDEX_WIDTH-1:0] replay_feature_sample;
    logic [FEATURE_INDEX_WIDTH-1:0] replay_feature_index;
    logic signed [15:0] replay_feature_trace;
    logic replay_row_valid;
    logic replay_row_ready;
    logic [BATCH_INDEX_WIDTH-1:0] replay_row_sample;
    logic [CLASS_INDEX_WIDTH-1:0] replay_row_label;
    logic replay_logit_valid;
    logic replay_logit_ready;
    logic signed [15:0] replay_logit;
    logic replay_observation_done;
    logic replay_observation_done_ready;

    logic learner_cfg_valid;
    logic learner_cfg_ready;
    logic learner_cfg_error;
    logic learner_active;
    logic learner_release;
    logic learner_release_ready;
    logic learner_feature_error;
    logic learner_logit_error;
    logic learner_observation_done;
    logic learner_observation_done_ready;
    logic learner_window_done;
    logic learner_window_busy;
    logic learner_context_error;
    logic learner_gradient_overflow;
    logic learner_update_config_error;
    logic learner_request_error;
    logic learner_store_error;
    logic learner_internal_error;
    logic capture_feature_fire;
    logic cfg_temporal_legal;
    logic cfg_reject_q;
    logic final_replay_timestep;
    logic replay_slice_done_fire;
    logic [BATCH_COUNT_WIDTH+TIMESTEP_COUNT_WIDTH-1:0]
        effective_batch_count_ext;
    logic [BATCH_COUNT_WIDTH-1:0] effective_batch_count;
    logic [BATCH_INDEX_WIDTH-1:0] learner_feature_sample;
    logic [BATCH_INDEX_WIDTH-1:0] learner_logit_sample;
    logic [15:0] external_feature_trace_q016;
    logic signed [15:0] external_logit_q6;

    function automatic logic [15:0] q8_8_trace_to_q016(
        input logic signed [15:0] value
    );
        begin
            if ($signed(value) <= 0)
                return 16'd0;
            // Exact range-preserving trace/128 representation. The shared
            // magnitude scaler restores the factor with SCNN multiplier 640.
            return {value[14:0], 1'b0};
        end
    endfunction

    function automatic logic signed [15:0] q8_8_logit_to_q6(
        input logic signed [15:0] value
    );
        logic signed [16:0] magnitude;
        logic signed [16:0] rounded;
        begin
            if ($signed(value) >= 0) begin
                rounded = $signed({1'b0, value}) + 17'sd2;
                return 16'($signed(rounded) >>> 2);
            end
            magnitude = -$signed({value[15], value});
            rounded = magnitude + 17'sd2;
            return 16'(-$signed(rounded >>> 2));
        end
    endfunction

    assign external_feature_trace_q016 =
        q8_8_trace_to_q016(replay_feature_trace);
    assign external_logit_q6 = q8_8_logit_to_q6(replay_logit);

    assign o_external_cfg_valid = USE_EXTERNAL_OUTPUT_LEARNER ?
        learner_cfg_valid : 1'b0;
    assign o_external_cfg_profile_id = i_cfg_profile_id;
    assign o_external_cfg_tensor_id = i_cfg_tensor_id;
    assign o_external_cfg_parameter_base = i_cfg_parameter_base;
    assign o_external_cfg_batch_count = effective_batch_count;
    assign o_external_cfg_source_count = i_cfg_feature_count;
    assign o_external_cfg_output_count = i_cfg_class_count;
    assign o_external_cfg_vote_increment = i_cfg_vote_increment;
    assign o_external_cfg_vote_threshold = i_cfg_vote_threshold;
    assign o_external_cfg_code_min = i_cfg_code_min;
    assign o_external_cfg_code_max = i_cfg_code_max;
    assign o_external_release = USE_EXTERNAL_OUTPUT_LEARNER ?
        learner_release : 1'b0;
    assign o_external_source_trace_valid = USE_EXTERNAL_OUTPUT_LEARNER ?
        replay_feature_valid : 1'b0;
    assign o_external_source_trace_sample_id = learner_feature_sample;
    assign o_external_source_trace_index = replay_feature_index;
    assign o_external_source_trace_q016 = external_feature_trace_q016;
    assign o_external_logit_row_valid = USE_EXTERNAL_OUTPUT_LEARNER ?
        replay_row_valid : 1'b0;
    assign o_external_logit_sample_id = learner_logit_sample;
    assign o_external_logit_label = replay_row_label;
    assign o_external_logit_valid = USE_EXTERNAL_OUTPUT_LEARNER ?
        replay_logit_valid : 1'b0;
    assign o_external_logit_q6 = external_logit_q6;
    assign o_external_observation_done = USE_EXTERNAL_OUTPUT_LEARNER ?
        learner_observation_done : 1'b0;

    // One software-selected adaptation window spans B samples by T timesteps.
    // Flatten (timestep, sample) into one learner batch so the address-major
    // reducer sees every contribution for an address before issuing one RMW.
    // This implements one RNE magnitude update per BxT window, matching the
    // fixed-SGD algorithm instead of rounding and committing once per timestep.
    assign effective_batch_count_ext =
        (BATCH_COUNT_WIDTH+TIMESTEP_COUNT_WIDTH)'(i_cfg_batch_count) *
        (BATCH_COUNT_WIDTH+TIMESTEP_COUNT_WIDTH)'(i_cfg_timestep_count);
    assign effective_batch_count = BATCH_COUNT_WIDTH'(effective_batch_count_ext);
    assign cfg_temporal_legal = (i_cfg_batch_count != '0) &&
        (i_cfg_timestep_count != '0) &&
        (int'(i_cfg_batch_count) <= CLASSIFIER_CAPTURE_MAX_BATCH) &&
        (int'(i_cfg_timestep_count) <= CLASSIFIER_CAPTURE_MAX_TIMESTEPS) &&
        (effective_batch_count_ext <=
         (BATCH_COUNT_WIDTH+TIMESTEP_COUNT_WIDTH)'(MAX_BATCH));
    assign learner_feature_sample = BATCH_INDEX_WIDTH'(
        int'(replay_timestep_q) * int'(batch_count_q) +
        int'(replay_feature_sample));
    assign learner_logit_sample = BATCH_INDEX_WIDTH'(
        int'(replay_timestep_q) * int'(batch_count_q) +
        int'(replay_row_sample));
    assign final_replay_timestep =
        int'(replay_timestep_q) + 1 >= int'(timestep_count_q);
    assign learner_observation_done = replay_observation_done &&
        final_replay_timestep;
    assign replay_observation_done_ready = final_replay_timestep ?
        learner_observation_done_ready : 1'b1;
    assign replay_slice_done_fire = replay_observation_done &&
        replay_observation_done_ready;

    assign o_cfg_ready = state_q == STATE_IDLE && replay_cfg_ready && learner_cfg_ready;
    assign cfg_fire = i_cfg_valid && o_cfg_ready;
    // A selected-layer owner releases this bridge after the joint commit.  Give
    // that release priority over the automatic same-descriptor rearm so a
    // following continuation descriptor can preserve temporal trace state.
    // If no release is requested, the bridge still rearms for another batch.
    assign replay_cfg_valid = (cfg_fire && cfg_temporal_legal) ||
        (state_q == STATE_REARM && !i_release);
    assign learner_cfg_valid = cfg_fire && cfg_temporal_legal;
    assign replay_abort = release_fire;
    assign learner_release = release_fire;
    assign o_release_ready =
        (state_q == STATE_CAPTURE || state_q == STATE_REARM) &&
        learner_release_ready;
    assign release_fire = i_release && o_release_ready;

    assign replay_start_valid = state_q == STATE_START_REPLAY;
    assign o_window_commit_valid = state_q == STATE_COMMIT;
    assign o_window_commit_profile_id = profile_id_q;
    assign o_window_commit_sequence = commit_sequence_q;
    assign o_completed_timestep_count = completed_timestep_count_q;
    assign o_capture_complete = replay_capture_complete;
    assign o_inference_hold = replay_capture_complete ||
        state_q == STATE_START_REPLAY || state_q == STATE_WAIT_WINDOW ||
        state_q == STATE_COMMIT || state_q == STATE_REARM;
    assign o_window_busy = replay_busy || learner_window_busy ||
        o_inference_hold || state_q == STATE_RELEASE_DRAIN;
    assign o_active = state_q != STATE_IDLE && learner_active;

    // Before a learning window starts, disabled learning is a true bypass:
    // observation traffic is accepted and discarded without modifying replay
    // state or parameter memory. A partially captured window is paused rather
    // than mixed with dropped observations.
    assign o_feature_trace_wave_ready = state_q == STATE_CAPTURE ?
        (i_learning_enable ? replay_feature_wave_ready : !capture_started_q) :
        ((state_q == STATE_IDLE && !i_learning_enable) ? 1'b1 : 1'b0);
    assign o_logit_wave_ready = state_q == STATE_CAPTURE ?
        (i_learning_enable ? replay_logit_wave_ready : !capture_started_q) :
        ((state_q == STATE_IDLE && !i_learning_enable) ? 1'b1 : 1'b0);
    assign capture_feature_fire = i_learning_enable &&
        i_feature_trace_wave_valid && replay_feature_wave_ready &&
        state_q == STATE_CAPTURE;

    assign o_cfg_error = cfg_reject_q || replay_cfg_error || learner_cfg_error;
    assign o_observation_error = replay_error || replay_protocol_error ||
        learner_feature_error || learner_logit_error;
    assign o_context_error = learner_context_error;
    assign o_gradient_overflow = learner_gradient_overflow;
    assign o_parameter_error = learner_update_config_error ||
        learner_request_error || learner_store_error || learner_internal_error;
    assign o_protocol_error = protocol_error_q;

    unified_scnn_classifier_observation_replay #(
        .MAX_BATCH(MAX_BATCH), .MAX_TIMESTEPS(MAX_TIMESTEPS),
        .CAPTURE_BATCH_DEPTH(CLASSIFIER_CAPTURE_MAX_BATCH),
        .CAPTURE_TIMESTEP_DEPTH(CLASSIFIER_CAPTURE_MAX_TIMESTEPS),
        .MAX_FEATURES(MAX_FEATURES), .MAX_CLASSES(MAX_CLASSES),
        .LANES(LANES), .ENGINES(ENGINES), .VERSION_WIDTH(VERSION_WIDTH),
        .FRAME_WIDTH(FRAME_WIDTH),
        .BATCH_INDEX_WIDTH(BATCH_INDEX_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .TIMESTEP_INDEX_WIDTH(TIMESTEP_INDEX_WIDTH),
        .TIMESTEP_COUNT_WIDTH(TIMESTEP_COUNT_WIDTH),
        .FEATURE_INDEX_WIDTH(FEATURE_INDEX_WIDTH),
        .FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
        .CLASS_INDEX_WIDTH(CLASS_INDEX_WIDTH),
        .CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH),
        .WAVE_CLASS_INDEX_WIDTH(WAVE_CLASS_INDEX_WIDTH),
        .WAVE_CLASS_COUNT_WIDTH(WAVE_CLASS_COUNT_WIDTH),
        .ENABLE_COMPUTED_TRACE_CAPTURE(ENABLE_COMPUTED_TRACE_CAPTURE)
    ) u_observation_replay (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(replay_abort),
        .i_cfg_valid(replay_cfg_valid), .o_cfg_ready(replay_cfg_ready),
        .i_cfg_batch_count(cfg_fire ? i_cfg_batch_count : batch_count_q),
        .i_cfg_timestep_base(cfg_fire ? i_cfg_timestep_base : timestep_base_q),
        .i_cfg_timestep_count(cfg_fire ? i_cfg_timestep_count : timestep_count_q),
        // Repeat the selected reset policy at each committed batch boundary.
        // A profile beginning at timestep zero therefore cannot inherit trace
        // state from the previous support batch; continuation profiles may
        // explicitly keep state by configuring reset low.
        .i_cfg_trace_state_reset(
            (cfg_fire && i_cfg_trace_state_reset) ||
            (state_q == STATE_REARM && trace_state_reset_q)),
        .i_cfg_feature_count(cfg_fire ? i_cfg_feature_count : feature_count_q),
        .i_cfg_class_count(cfg_fire ? i_cfg_class_count : class_count_q),
        .i_cfg_trace_leak_valid(cfg_fire ? i_cfg_trace_leak_valid : trace_leak_valid_q),
        .i_cfg_trace_leak_q8_8(cfg_fire ? i_cfg_trace_leak_q8_8 : trace_leak_q),
        .i_cfg_computed_trace_capture_enable(cfg_fire ?
            i_cfg_computed_trace_capture_enable : computed_trace_capture_enable_q),
        .o_cfg_error(replay_cfg_error),
        .i_feature_trace_wave_valid(i_feature_trace_wave_valid &&
            i_learning_enable && state_q == STATE_CAPTURE),
        .o_feature_trace_wave_ready(replay_feature_wave_ready),
        .i_feature_trace_wave_mask(i_feature_trace_wave_mask),
        .i_feature_trace_wave_base(i_feature_trace_wave_base),
        .i_feature_trace_wave_q8_8_flat(i_feature_trace_wave_q8_8_flat),
        .i_feature_trace_wave_version(i_feature_trace_wave_version),
        .i_feature_trace_wave_frame_index(i_feature_trace_wave_frame_index),
        .i_feature_trace_wave_first(i_feature_trace_wave_first),
        .i_feature_trace_wave_last(i_feature_trace_wave_last),
        .i_feature_trace_wave_sample_reset(i_feature_trace_wave_sample_reset),
        .i_feature_trace_wave_sample_last(i_feature_trace_wave_sample_last),
        .i_sample_label(i_sample_label),
        .i_logit_wave_valid(i_logit_wave_valid && i_learning_enable &&
            state_q == STATE_CAPTURE),
        .o_logit_wave_ready(replay_logit_wave_ready),
        .i_logit_wave_mask(i_logit_wave_mask),
        .i_logit_wave_base(i_logit_wave_base),
        .i_logit_wave_count(i_logit_wave_count),
        .i_logit_wave_q8_8_flat(i_logit_wave_q8_8_flat),
        .i_logit_wave_version(i_logit_wave_version),
        .i_logit_wave_frame_index(i_logit_wave_frame_index),
        .i_logit_wave_frame_last(i_logit_wave_frame_last),
        .i_logit_wave_sample_last(i_logit_wave_sample_last),
        .i_replay_valid(replay_start_valid), .o_replay_ready(replay_start_ready),
        .i_replay_timestep(replay_timestep_q), .o_replay_error(replay_error),
        .o_capture_active(),
        .o_capture_complete(replay_capture_complete),
        .o_replay_busy(replay_busy), .o_idle(replay_idle),
        .o_protocol_error(replay_protocol_error),
        .o_feature_trace_valid(replay_feature_valid),
        .i_feature_trace_ready(replay_feature_ready),
        .o_feature_trace_sample_id(replay_feature_sample),
        .o_feature_trace_index(replay_feature_index),
        .o_feature_trace_q8_8(replay_feature_trace),
        .o_feature_trace_version(),
        .o_logit_row_valid(replay_row_valid),
        .i_logit_row_ready(replay_row_ready),
        .o_logit_sample_id(replay_row_sample),
        .o_logit_label(replay_row_label),
        .o_logit_row_version(),
        .o_logit_valid(replay_logit_valid),
        .i_logit_ready(replay_logit_ready),
        .o_logit_q8_8(replay_logit),
        .o_observation_done(replay_observation_done),
        .i_observation_done_ready(replay_observation_done_ready),
        .o_computed_trace_valid(o_computed_trace_valid),
        .i_computed_trace_ready(i_computed_trace_ready),
        .o_computed_trace_sample_id(o_computed_trace_sample_id),
        .o_computed_trace_index(o_computed_trace_index),
        .o_computed_trace_timestep(o_computed_trace_timestep),
        .o_computed_trace_frame_index(o_computed_trace_frame_index),
        .o_computed_trace_version(o_computed_trace_version),
        .o_computed_trace_q8_8(o_computed_trace_q8_8)
    );

    generate
    if (!USE_EXTERNAL_OUTPUT_LEARNER) begin : gen_private_output_learner
    unified_scnn_tp_classifier_online_learning_path #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH), .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH), .MAX_BATCH(MAX_BATCH),
        .MAX_FEATURES(MAX_FEATURES), .MAX_CLASSES(MAX_CLASSES),
        .MAX_GRADIENT_ELEMENTS(MAX_GRADIENT_ELEMENTS),
        .BATCH_INDEX_WIDTH(BATCH_INDEX_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .FEATURE_INDEX_WIDTH(FEATURE_INDEX_WIDTH),
        .FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
        .CLASS_INDEX_WIDTH(CLASS_INDEX_WIDTH),
        .CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH)
    ) u_classifier_learner (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_cfg_valid(learner_cfg_valid), .o_cfg_ready(learner_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id),
        .i_cfg_tensor_id(i_cfg_tensor_id),
        .i_cfg_parameter_base(i_cfg_parameter_base),
        .i_cfg_batch_count(effective_batch_count),
        .i_cfg_feature_count(i_cfg_feature_count),
        .i_cfg_class_count(i_cfg_class_count),
        .i_cfg_vote_increment(i_cfg_vote_increment),
        .i_cfg_vote_threshold(i_cfg_vote_threshold),
        .i_cfg_code_min(i_cfg_code_min), .i_cfg_code_max(i_cfg_code_max),
        .o_cfg_error(learner_cfg_error), .o_active(learner_active),
        .i_release(learner_release), .o_release_ready(learner_release_ready),
        .i_feature_trace_valid(replay_feature_valid),
        .o_feature_trace_ready(replay_feature_ready),
        .i_feature_trace_sample_id(learner_feature_sample),
        .i_feature_trace_index(replay_feature_index),
        .i_feature_trace_q8_8(replay_feature_trace),
        .o_feature_trace_error(learner_feature_error),
        .i_logit_row_valid(replay_row_valid),
        .o_logit_row_ready(replay_row_ready),
        .i_logit_sample_id(learner_logit_sample),
        .i_logit_label(replay_row_label),
        .i_logit_valid(replay_logit_valid),
        .o_logit_ready(replay_logit_ready), .i_logit_q8_8(replay_logit),
        .o_logit_error(learner_logit_error),
        .o_logit_row_done(),
        .i_observation_done(learner_observation_done),
        .o_observation_done_ready(learner_observation_done_ready),
        .o_window_done(learner_window_done),
        .o_window_busy(learner_window_busy),
        .o_context_error(learner_context_error),
        .o_gradient_overflow(learner_gradient_overflow),
        .o_read_valid(o_read_valid), .i_read_ready(i_read_ready),
        .o_read_profile_id(o_read_profile_id),
        .o_read_tensor_id(o_read_tensor_id),
        .o_read_parameter_addr(o_read_parameter_addr),
        .o_read_tag(o_read_tag),
        .i_read_response_valid(i_read_response_valid),
        .o_read_response_ready(o_read_response_ready),
        .i_read_response_code(i_read_response_code),
        .i_read_response_residual(i_read_response_residual),
        .i_read_response_error(i_read_response_error),
        .o_result_valid(o_result_valid), .i_result_ready(i_result_ready),
        .o_result_profile_id(o_result_profile_id),
        .o_result_tensor_id(o_result_tensor_id),
        .o_result_parameter_addr(o_result_parameter_addr),
        .o_result_tag(o_result_tag), .o_result_code(o_result_code),
        .o_result_residual(o_result_residual),
        .o_write_code_enable(o_write_code_enable),
        .o_write_residual_enable(o_write_residual_enable),
        .o_write_code_fire(o_write_code_fire),
        .o_write_residual_fire(o_write_residual_fire),
        .o_dirty(o_dirty), .o_saturated(o_saturated),
        .o_update_config_error(learner_update_config_error),
        .o_request_error(learner_request_error),
        .o_store_error(learner_store_error),
        .o_internal_error(learner_internal_error)
    );
    end else begin : gen_external_output_learner
        assign learner_cfg_ready = i_external_cfg_ready;
        assign learner_cfg_error = i_external_cfg_error;
        assign learner_active = i_external_active;
        assign learner_release_ready = i_external_release_ready;
        assign replay_feature_ready = i_external_source_trace_ready;
        assign learner_feature_error = i_external_source_trace_error ||
            (replay_feature_valid && replay_feature_ready &&
             ($signed(replay_feature_trace) < 0));
        assign replay_row_ready = i_external_logit_row_ready;
        assign replay_logit_ready = i_external_logit_ready;
        assign learner_logit_error = i_external_logit_error;
        assign learner_observation_done_ready =
            i_external_observation_done_ready;
        assign learner_window_done = i_external_window_done;
        assign learner_window_busy = i_external_window_busy;
        assign learner_context_error = i_external_context_error;
        assign learner_gradient_overflow = i_external_gradient_overflow;
        assign learner_update_config_error = i_external_update_config_error;
        assign learner_request_error = i_external_request_error;
        assign learner_store_error = i_external_store_error;
        assign learner_internal_error = i_external_internal_error;

        assign o_read_valid = 1'b0;
        assign o_read_profile_id = '0;
        assign o_read_tensor_id = '0;
        assign o_read_parameter_addr = '0;
        assign o_read_tag = '0;
        assign o_read_response_ready = 1'b0;
        assign o_result_valid = 1'b0;
        assign o_result_profile_id = '0;
        assign o_result_tensor_id = '0;
        assign o_result_parameter_addr = '0;
        assign o_result_tag = '0;
        assign o_result_code = '0;
        assign o_result_residual = '0;
        assign o_write_code_enable = 1'b0;
        assign o_write_residual_enable = 1'b0;
        assign o_write_code_fire = 1'b0;
        assign o_write_residual_fire = 1'b0;
        assign o_dirty = 1'b0;
        assign o_saturated = 1'b0;
    end
    endgenerate

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            profile_id_q <= '0;
            batch_count_q <= '0;
            timestep_base_q <= '0;
            timestep_count_q <= '0;
            feature_count_q <= '0;
            class_count_q <= '0;
            trace_leak_valid_q <= 1'b0;
            trace_leak_q <= '0;
            trace_state_reset_q <= 1'b0;
            computed_trace_capture_enable_q <= 1'b0;
            replay_timestep_q <= '0;
            completed_timestep_count_q <= '0;
            commit_sequence_q <= '0;
            capture_started_q <= 1'b0;
            protocol_error_q <= 1'b0;
            cfg_reject_q <= 1'b0;
            o_timestep_learning_done <= 1'b0;
            o_batch_learning_done <= 1'b0;
        end else begin
            o_timestep_learning_done <= 1'b0;
            o_batch_learning_done <= 1'b0;
            cfg_reject_q <= cfg_fire && !cfg_temporal_legal;
            if (i_clear_error)
                protocol_error_q <= 1'b0;
            if ((i_release && !o_release_ready) ||
                (i_cfg_valid && !o_cfg_ready && state_q != STATE_REARM))
                protocol_error_q <= 1'b1;

            if (cfg_fire && cfg_temporal_legal) begin
                profile_id_q <= i_cfg_profile_id;
                batch_count_q <= i_cfg_batch_count;
                timestep_base_q <= i_cfg_timestep_base;
                timestep_count_q <= i_cfg_timestep_count;
                feature_count_q <= i_cfg_feature_count;
                class_count_q <= i_cfg_class_count;
                trace_leak_valid_q <= i_cfg_trace_leak_valid;
                trace_leak_q <= i_cfg_trace_leak_q8_8;
                trace_state_reset_q <= i_cfg_trace_state_reset;
                computed_trace_capture_enable_q <= ENABLE_COMPUTED_TRACE_CAPTURE &&
                    i_cfg_computed_trace_capture_enable;
                replay_timestep_q <= '0;
                completed_timestep_count_q <= '0;
                capture_started_q <= 1'b0;
                state_q <= STATE_CAPTURE;
            end else begin
                case (state_q)
                    STATE_IDLE: begin end
                    STATE_CAPTURE: begin
                        if (capture_feature_fire)
                            capture_started_q <= 1'b1;
                        if (release_fire)
                            state_q <= STATE_RELEASE_DRAIN;
                        else if (replay_capture_complete) begin
                            replay_timestep_q <= '0;
                            completed_timestep_count_q <= '0;
                            state_q <= STATE_START_REPLAY;
                        end
                    end
                    STATE_START_REPLAY: begin
                        if (replay_start_valid && replay_start_ready)
                            state_q <= STATE_WAIT_WINDOW;
                    end
                    STATE_WAIT_WINDOW: begin
                        if (!final_replay_timestep &&
                                replay_slice_done_fire) begin
                            o_timestep_learning_done <= 1'b1;
                            completed_timestep_count_q <=
                                completed_timestep_count_q + 1'b1;
                            replay_timestep_q <= replay_timestep_q + 1'b1;
                            state_q <= STATE_START_REPLAY;
                        end else if (final_replay_timestep &&
                                learner_window_done) begin
                            o_timestep_learning_done <= 1'b1;
                            completed_timestep_count_q <=
                                completed_timestep_count_q + 1'b1;
                            state_q <= STATE_COMMIT;
                        end
                    end
                    STATE_COMMIT: begin
                        if (i_window_commit_ready) begin
                            commit_sequence_q <= commit_sequence_q + 1'b1;
                            o_batch_learning_done <= 1'b1;
                            state_q <= STATE_REARM;
                        end
                    end
                    STATE_REARM: begin
                        if (release_fire) begin
                            state_q <= STATE_RELEASE_DRAIN;
                        end else if (replay_cfg_valid && replay_cfg_ready) begin
                            replay_timestep_q <= '0;
                            completed_timestep_count_q <= '0;
                            capture_started_q <= 1'b0;
                            state_q <= STATE_CAPTURE;
                        end
                    end
                    STATE_RELEASE_DRAIN: begin
                        if (replay_idle && !learner_active)
                            state_q <= STATE_IDLE;
                    end
                    default: state_q <= STATE_IDLE;
                endcase
            end
        end
    end

    initial begin
        if (MAX_TIMESTEPS < 1 || MAX_BATCH < 1 || MAX_FEATURES < 1 ||
                MAX_CLASSES < 1 || CLASSIFIER_CAPTURE_MAX_BATCH < 1 ||
                CLASSIFIER_CAPTURE_MAX_BATCH > MAX_BATCH ||
                CLASSIFIER_CAPTURE_MAX_TIMESTEPS < 1 ||
                CLASSIFIER_CAPTURE_MAX_TIMESTEPS > MAX_TIMESTEPS)
            $error("SCNN classifier final-core bridge parameters are invalid: max_batch=%0d max_timesteps=%0d capture_batch=%0d capture_timesteps=%0d",
                MAX_BATCH, MAX_TIMESTEPS,
                CLASSIFIER_CAPTURE_MAX_BATCH,
                CLASSIFIER_CAPTURE_MAX_TIMESTEPS);
    end
endmodule

`default_nettype wire
