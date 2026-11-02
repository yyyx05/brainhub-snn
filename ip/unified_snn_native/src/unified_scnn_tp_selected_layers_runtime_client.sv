`timescale 1ns/1ps
`default_nettype none

// Production runtime wrapper for one selected Conv + classifier TP update.
// It owns transaction lifetime and version visibility, but not observations,
// arithmetic, parameter storage, or topology ownership.
module unified_scnn_tp_selected_layers_runtime_client #(
    parameter bit ALLOW_UNORDERED_SURROGATE_CAPTURE = 1'b0,
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
    // Physical Conv8 learning tile capacity, not the full layer dimensions.
    parameter int MAX_INPUT_CHANNELS = 8,
    parameter int MAX_OUTPUT_CHANNELS = 12,
    parameter int OBS_INPUT_CHANNELS = 512,
    parameter int OBS_OUTPUT_CHANNELS = 512,
    parameter int SPATIAL = 4,
    parameter int POOLED_SPATIAL = 2,
    parameter int KERNEL_SIZE = 3,
    parameter int Q16_ADDR_WIDTH = 24,
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int VERSION_WIDTH = 16,
    parameter int MAX_FEATURES = 2048,
    parameter int MAX_CLASSES = 11,
    parameter int LANES = 8,
    parameter int ENGINES = 12,
    parameter int FRAME_WIDTH = 8,
    parameter int SAMPLE_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH+1),
    parameter int TIMESTEP_WIDTH = MAX_TIMESTEPS <= 2 ? 1 :
        $clog2(MAX_TIMESTEPS),
    parameter int TIMESTEP_COUNT_WIDTH = $clog2(MAX_TIMESTEPS + 1),
    parameter int CHANNEL_INDEX_WIDTH = 9,
    parameter int CHANNEL_COUNT_WIDTH = 10,
    parameter int FEATURE_INDEX_WIDTH = MAX_FEATURES <= 2 ? 1 :
        $clog2(MAX_FEATURES),
    parameter int FEATURE_COUNT_WIDTH = $clog2(MAX_FEATURES+1),
    parameter int CLASS_INDEX_WIDTH = MAX_CLASSES <= 2 ? 1 :
        $clog2(MAX_CLASSES),
    parameter int CLASS_COUNT_WIDTH = $clog2(MAX_CLASSES+1),
    parameter bit USE_SAMPLE_CONTEXT = 1'b1,
    parameter bit USE_CLASSIFIER_STUDENT_TRACE = 1'b1,
    parameter bit USE_INTERNAL_MODULATION = 1'b1,
    parameter bit USE_EXTERNAL_OUTPUT_LEARNER = 1'b0,
    parameter bit ENABLE_SCNN_CONV8_LEARNING = 1'b1,
    parameter bit USE_EXTERNAL_COMMIT_BARRIER = 1'b0,
    parameter int INTERNAL_MODULATION_BATCH = 2,
    parameter bit INTERNAL_MODULATION_TRACE_SPATIAL_MAJOR = 1'b0,
    parameter int SURROGATE_WAVE_CHANNELS = OBS_OUTPUT_CHANNELS,
    parameter int SURROGATE_SPATIAL_BLOCK_SIZE = 0,
    parameter int TRACE_WAVE_CHANNELS = OBS_OUTPUT_CHANNELS
) (
    input  wire logic                                i_clk,
    input  wire logic                                i_rst_n,
    input  wire logic                                i_clear_error,

    input  wire logic                                i_system_parameter_owner,
    input  wire logic                                i_system_learning_enable,
    input  wire logic                                i_system_profile_aligned,
    input  wire logic [PROFILE_ID_WIDTH-1:0]         i_system_profile_id,

    input  wire logic                                i_run_valid,
    output logic                                o_run_ready,
    input  wire logic [TIMESTEP_WIDTH-1:0]           i_run_timestep,
    input  wire logic [TIMESTEP_COUNT_WIDTH-1:0]     i_run_timestep_count,
    output logic                                o_run_done,
    output logic                                o_capture_complete,
    output logic                                o_learning_start_fire,

    input  wire logic [PROFILE_ID_WIDTH-1:0]         i_profile_id,
    input  wire logic [TENSOR_ID_WIDTH-1:0]          i_conv_tensor_id,
    input  wire logic [TENSOR_ID_WIDTH-1:0]          i_classifier_tensor_id,
    input  wire logic [BATCH_COUNT_WIDTH-1:0]        i_batch_count,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]      i_total_input_channels,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]      i_total_output_channels,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]         i_conv_parameter_base,
    input  wire logic [Q16_ADDR_WIDTH-1:0]           i_conv_weight_q16_base,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]         i_classifier_parameter_base,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0]      i_classifier_feature_count,
    input  wire logic [CLASS_COUNT_WIDTH-1:0]        i_classifier_class_count,
    input  wire logic                                i_classifier_trace_leak_valid,
    input  wire logic signed [15:0]                  i_classifier_trace_leak_q8_8,
    input  wire logic [VOTE_WIDTH-1:0]               i_conv_vote_increment,
    input  wire logic [VOTE_WIDTH-1:0]               i_conv_vote_threshold,
    input  wire logic [VOTE_WIDTH-1:0]               i_classifier_vote_increment,
    input  wire logic [VOTE_WIDTH-1:0]               i_classifier_vote_threshold,
    input  wire logic signed [CODE_WIDTH-1:0]        i_code_min,
    input  wire logic signed [CODE_WIDTH-1:0]        i_code_max,
    input  wire logic [VERSION_WIDTH-1:0]            i_sample_version,

    input  wire logic                                i_context_load_valid,
    output logic                                o_context_load_ready,
    input  wire logic [VERSION_WIDTH-1:0]            i_context_model_version,
    input  wire logic [VERSION_WIDTH-1:0]            i_context_adapt_version,
    input  wire logic                                i_forward_start_valid,
    output logic                                o_forward_start_ready,
    output logic                                o_forward_start_fire,
    input  wire logic                                i_forward_done,
    output logic                                o_forward_active,
    output logic [VERSION_WIDTH-1:0]            o_forward_model_version,
    output logic [VERSION_WIDTH-1:0]            o_forward_adapt_version,
    input  wire logic                                i_restore_complete,
    output logic                                o_restore_ready,
    output logic                                o_recovery_required,

    input  wire logic                                i_sample_context_valid,
    output logic                                o_sample_context_ready,
    input  wire logic                                i_sample_context_target,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_sample_context_sample,
    input  wire logic [VERSION_WIDTH-1:0]            i_sample_context_version,

    input  wire logic                                i_input_valid,
    output logic                                o_input_ready,
    input  wire logic                                i_input_target,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_input_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]           i_input_timestep,
    input  wire logic [VERSION_WIDTH-1:0]            i_input_sample_version,
    input  wire logic [CHANNEL_INDEX_WIDTH-1:0]      i_input_channel,
    input  wire logic [$clog2(SPATIAL*SPATIAL)-1:0] i_input_spatial,
    input  wire logic signed [15:0]                  i_input_q8_8,
    input  wire logic                                i_trace_valid,
    output logic                                o_trace_ready,
    input  wire logic                                i_trace_target,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_trace_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]           i_trace_timestep,
    input  wire logic [VERSION_WIDTH-1:0]            i_trace_sample_version,
    input  wire logic [CHANNEL_INDEX_WIDTH-1:0]      i_trace_channel,
    input  wire logic [$clog2(POOLED_SPATIAL*POOLED_SPATIAL)-1:0]
                                                i_trace_spatial,
    input  wire logic signed [15:0]                  i_trace_q8_8,
    input  wire logic                                i_surrogate_valid,
    output logic                                o_surrogate_ready,
    input  wire logic                                i_surrogate_target,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_surrogate_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]           i_surrogate_timestep,
    input  wire logic [VERSION_WIDTH-1:0]            i_surrogate_sample_version,
    input  wire logic [CHANNEL_INDEX_WIDTH-1:0]      i_surrogate_channel,
    input  wire logic [$clog2(SPATIAL*SPATIAL)-1:0] i_surrogate_spatial,
    input  wire logic signed [15:0]                  i_surrogate_q8_8,

    input  wire logic                                i_modulation_valid,
    output logic                                o_modulation_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_modulation_row,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_modulation_col,
    input  wire logic signed [15:0]                  i_modulation_q8_8,

    input  wire logic                                i_feature_trace_wave_valid,
    output logic                                o_feature_trace_wave_ready,
    input  wire logic [LANES-1:0]                    i_feature_trace_wave_mask,
    input  wire logic [FEATURE_INDEX_WIDTH-1:0]      i_feature_trace_wave_base,
    input  wire logic [LANES*16-1:0]                 i_feature_trace_wave_q8_8_flat,
    input  wire logic [VERSION_WIDTH-1:0]            i_feature_trace_wave_version,
    input  wire logic [FRAME_WIDTH-1:0]              i_feature_trace_wave_frame_index,
    input  wire logic                                i_feature_trace_wave_first,
    input  wire logic                                i_feature_trace_wave_last,
    input  wire logic                                i_feature_trace_wave_sample_reset,
    input  wire logic                                i_feature_trace_wave_sample_last,
    input  wire logic [CLASS_INDEX_WIDTH-1:0]        i_sample_label,
    input  wire logic                                i_logit_wave_valid,
    output logic                                o_logit_wave_ready,
    input  wire logic [ENGINES-1:0]                  i_logit_wave_mask,
    input  wire logic [CLASS_INDEX_WIDTH-1:0]        i_logit_wave_base,
    input  wire logic [CLASS_COUNT_WIDTH-1:0]        i_logit_wave_count,
    input  wire logic [ENGINES*16-1:0]               i_logit_wave_q8_8_flat,
    input  wire logic [VERSION_WIDTH-1:0]            i_logit_wave_version,
    input  wire logic [FRAME_WIDTH-1:0]              i_logit_wave_frame_index,
    input  wire logic                                i_logit_wave_frame_last,
    input  wire logic                                i_logit_wave_sample_last,

    output logic                                     o_external_cfg_valid,
    input  wire logic                                i_external_cfg_ready,
    output logic [PROFILE_ID_WIDTH-1:0]              o_external_cfg_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]               o_external_cfg_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]              o_external_cfg_parameter_base,
    output logic [BATCH_COUNT_WIDTH-1:0]             o_external_cfg_batch_count,
    output logic [FEATURE_COUNT_WIDTH-1:0]           o_external_cfg_source_count,
    output logic [CLASS_COUNT_WIDTH-1:0]             o_external_cfg_output_count,
    output logic [VOTE_WIDTH-1:0]                    o_external_cfg_vote_increment,
    output logic [VOTE_WIDTH-1:0]                    o_external_cfg_vote_threshold,
    output logic signed [CODE_WIDTH-1:0]             o_external_cfg_code_min,
    output logic signed [CODE_WIDTH-1:0]             o_external_cfg_code_max,
    input  wire logic                                i_external_cfg_error,
    input  wire logic                                i_external_active,
    output logic                                     o_external_release,
    input  wire logic                                i_external_release_ready,
    output logic                                     o_external_source_trace_valid,
    input  wire logic                                i_external_source_trace_ready,
    output logic [SAMPLE_WIDTH-1:0]                  o_external_source_trace_sample_id,
    output logic [FEATURE_INDEX_WIDTH-1:0]           o_external_source_trace_index,
    output logic [15:0]                              o_external_source_trace_q016,
    input  wire logic                                i_external_source_trace_error,
    output logic                                     o_external_logit_row_valid,
    input  wire logic                                i_external_logit_row_ready,
    output logic [SAMPLE_WIDTH-1:0]                  o_external_logit_sample_id,
    output logic [CLASS_INDEX_WIDTH-1:0]             o_external_logit_label,
    output logic                                     o_external_logit_valid,
    input  wire logic                                i_external_logit_ready,
    output logic signed [15:0]                       o_external_logit_q6,
    input  wire logic                                i_external_logit_error,
    input  wire logic                                i_external_logit_row_done,
    output logic                                     o_external_observation_done,
    input  wire logic                                i_external_observation_done_ready,
    input  wire logic                                i_external_window_done,
    input  wire logic                                i_external_window_busy,
    input  wire logic                                i_external_context_error,
    input  wire logic                                i_external_gradient_overflow,
    input  wire logic                                i_external_update_config_error,
    input  wire logic                                i_external_request_error,
    input  wire logic                                i_external_store_error,
    input  wire logic                                i_external_internal_error,

    output logic                                o_read_valid,
    input  wire logic                                i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0]         o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]          o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]                o_read_tag,
    input  wire logic                                i_read_response_valid,
    output logic                                o_read_response_ready,
    input  wire logic signed [CODE_WIDTH-1:0]        i_read_response_code,
    input  wire logic signed [RESIDUAL_WIDTH-1:0]    i_read_response_residual,
    input  wire logic                                i_read_response_error,
    output logic                                o_result_valid,
    input  wire logic                                i_result_ready,
    output logic [PROFILE_ID_WIDTH-1:0]         o_result_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]          o_result_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_result_parameter_addr,
    output logic [TAG_WIDTH-1:0]                o_result_tag,
    output logic signed [CODE_WIDTH-1:0]        o_result_code,
    output logic signed [RESIDUAL_WIDTH-1:0]    o_result_residual,
    output logic                                o_write_code_enable,
    output logic                                o_write_residual_enable,

    output logic [VERSION_WIDTH-1:0]            o_model_version,
    output logic [VERSION_WIDTH-1:0]            o_adapt_version,
    output logic                                o_commit_pulse,
    output logic                                o_commit_code_dirty,
    output logic                                o_commit_optimizer_dirty,
    output logic                                o_detach_pulse,
    output logic                                o_commit_pending,
    input  wire logic                                i_commit_publish_ready,
    output logic                                o_inference_hold,
    output logic                                o_busy,
    output logic                                o_quiescent,
    output logic                                o_error
);
    logic active_q;
    logic config_pending_q;
    logic commit_seen_q;
    logic wrapper_error_q;
    logic system_admit;
    logic run_fire;
    logic path_cfg_ready;
    logic path_cfg_valid;
    logic path_active;
    logic path_cfg_error;
    logic path_protocol_error;
    logic path_owner_quiescent;
    logic path_abort_request;
    logic path_abort_complete;
    logic path_learning_start_valid;
    logic boundary_learning_start_ready;
    logic boundary_learning_active;
    logic path_commit_valid;
    logic boundary_commit_ready;
    logic path_commit_fire;
    logic path_write_code_fire;
    logic path_write_residual_fire;
    logic boundary_error;
    logic boundary_forward_start_ready;
    logic boundary_forward_start_valid;
    logic [12:0] _unused_tiles_completed;
    logic _unused_gradient_update_accepted;
    logic _unused_gradient_zero_skipped;
    logic _unused_gradient_valid;
    logic _unused_gradient_ready;
    logic signed [1:0] _unused_gradient_sign;
    logic [PARAM_ADDR_WIDTH-1:0] _unused_gradient_local_addr;
    logic [PARAM_ADDR_WIDTH-1:0] _unused_gradient_parameter_addr;
    logic [Q16_ADDR_WIDTH-1:0] _unused_gradient_weight_q16_addr;
    logic [Q64_ADDR_WIDTH-1:0] _unused_gradient_weight_q64_addr;
    logic [1:0] _unused_gradient_weight_q64_lane;
    logic _unused_gradient_tile_last;
    logic [PROFILE_ID_WIDTH-1:0] _unused_active_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] _unused_active_tensor_id;
    logic [PROFILE_ID_WIDTH-1:0] _unused_commit_profile_id;
    logic [31:0] _unused_commit_sequence;
    logic _unused_dirty;
    logic _unused_saturated;
    logic [VERSION_WIDTH-1:0] _unused_learning_base_model_version;
    logic [VERSION_WIDTH-1:0] _unused_learning_base_adapt_version;
    logic [TIMESTEP_WIDTH-1:0] run_timestep_q;
    logic [TIMESTEP_COUNT_WIDTH-1:0] run_timestep_count_q;
    logic [PROFILE_ID_WIDTH-1:0] profile_id_q;
    logic [TENSOR_ID_WIDTH-1:0] conv_tensor_id_q;
    logic [TENSOR_ID_WIDTH-1:0] classifier_tensor_id_q;
    logic [BATCH_COUNT_WIDTH-1:0] batch_count_q;
    logic [CHANNEL_COUNT_WIDTH-1:0] total_input_channels_q;
    logic [CHANNEL_COUNT_WIDTH-1:0] total_output_channels_q;
    logic [PARAM_ADDR_WIDTH-1:0] conv_parameter_base_q;
    logic [Q16_ADDR_WIDTH-1:0] conv_weight_q16_base_q;
    logic [PARAM_ADDR_WIDTH-1:0] classifier_parameter_base_q;
    logic [FEATURE_COUNT_WIDTH-1:0] classifier_feature_count_q;
    logic [CLASS_COUNT_WIDTH-1:0] classifier_class_count_q;
    logic classifier_trace_leak_valid_q;
    logic signed [15:0] classifier_trace_leak_q8_8_q;
    logic [VOTE_WIDTH-1:0] conv_vote_increment_q;
    logic [VOTE_WIDTH-1:0] conv_vote_threshold_q;
    logic [VOTE_WIDTH-1:0] classifier_vote_increment_q;
    logic [VOTE_WIDTH-1:0] classifier_vote_threshold_q;
    logic signed [CODE_WIDTH-1:0] code_min_q;
    logic signed [CODE_WIDTH-1:0] code_max_q;
    logic [VERSION_WIDTH-1:0] sample_version_q;
    logic commit_barrier_ready;

    assign system_admit = i_system_parameter_owner &&
        i_system_learning_enable && i_system_profile_aligned &&
        i_system_profile_id == i_profile_id;
    assign o_run_ready = system_admit && !active_q && !o_error &&
        path_owner_quiescent;
    assign run_fire = i_run_valid && o_run_ready;
    assign path_cfg_valid = config_pending_q;
    assign boundary_forward_start_valid = i_forward_start_valid &&
        i_system_parameter_owner && i_system_profile_aligned && !o_error;
    assign o_forward_start_ready = boundary_forward_start_ready &&
        i_system_parameter_owner && i_system_profile_aligned && !o_error;
    assign o_busy = active_q || path_active || boundary_learning_active ||
        o_forward_active || !path_owner_quiescent;
    assign o_quiescent = !o_busy && !o_error;
    assign o_error = wrapper_error_q || path_cfg_error ||
        path_protocol_error || path_abort_request || boundary_error;
    assign commit_barrier_ready = !USE_EXTERNAL_COMMIT_BARRIER ||
        i_commit_publish_ready;
    assign o_commit_pending = path_commit_valid;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            active_q <= 1'b0;
            config_pending_q <= 1'b0;
            commit_seen_q <= 1'b0;
            wrapper_error_q <= 1'b0;
            o_run_done <= 1'b0;
            run_timestep_q <= '0;
            run_timestep_count_q <= '0;
            profile_id_q <= '0;
            conv_tensor_id_q <= '0;
            classifier_tensor_id_q <= '0;
            batch_count_q <= '0;
            total_input_channels_q <= '0;
            total_output_channels_q <= '0;
            conv_parameter_base_q <= '0;
            conv_weight_q16_base_q <= '0;
            classifier_parameter_base_q <= '0;
            classifier_feature_count_q <= '0;
            classifier_class_count_q <= '0;
            classifier_trace_leak_valid_q <= 1'b0;
            classifier_trace_leak_q8_8_q <= '0;
            conv_vote_increment_q <= '0;
            conv_vote_threshold_q <= '0;
            classifier_vote_increment_q <= '0;
            classifier_vote_threshold_q <= '0;
            code_min_q <= '0;
            code_max_q <= '0;
            sample_version_q <= '0;
        end else begin
            o_run_done <= 1'b0;
            if (i_clear_error && !active_q && path_owner_quiescent)
                wrapper_error_q <= 1'b0;
            if (run_fire) begin
                active_q <= 1'b1;
                config_pending_q <= 1'b1;
                commit_seen_q <= 1'b0;
                run_timestep_q <= i_run_timestep;
                run_timestep_count_q <= i_run_timestep_count;
                profile_id_q <= i_profile_id;
                conv_tensor_id_q <= i_conv_tensor_id;
                classifier_tensor_id_q <= i_classifier_tensor_id;
                batch_count_q <= i_batch_count;
                total_input_channels_q <= i_total_input_channels;
                total_output_channels_q <= i_total_output_channels;
                conv_parameter_base_q <= i_conv_parameter_base;
                conv_weight_q16_base_q <= i_conv_weight_q16_base;
                classifier_parameter_base_q <= i_classifier_parameter_base;
                classifier_feature_count_q <= i_classifier_feature_count;
                classifier_class_count_q <= i_classifier_class_count;
                classifier_trace_leak_valid_q <=
                    i_classifier_trace_leak_valid;
                classifier_trace_leak_q8_8_q <=
                    i_classifier_trace_leak_q8_8;
                conv_vote_increment_q <= i_conv_vote_increment;
                conv_vote_threshold_q <= i_conv_vote_threshold;
                classifier_vote_increment_q <= i_classifier_vote_increment;
                classifier_vote_threshold_q <= i_classifier_vote_threshold;
                code_min_q <= i_code_min;
                code_max_q <= i_code_max;
                sample_version_q <= i_sample_version;
            end
            if (path_cfg_valid && path_cfg_ready)
                config_pending_q <= 1'b0;
            if (path_commit_fire)
                commit_seen_q <= 1'b1;
            if (active_q && (commit_seen_q || path_commit_fire) &&
                path_owner_quiescent && !boundary_learning_active) begin
                active_q <= 1'b0;
                config_pending_q <= 1'b0;
                commit_seen_q <= 1'b0;
                o_run_done <= 1'b1;
            end
            if (active_q && (!i_system_parameter_owner ||
                !i_system_learning_enable || !i_system_profile_aligned ||
                i_system_profile_id != profile_id_q))
                wrapper_error_q <= 1'b1;
        end
    end

    unified_scnn_tp_selected_layers_t1_path #(
        .ALLOW_UNORDERED_SURROGATE_CAPTURE(
            ALLOW_UNORDERED_SURROGATE_CAPTURE),
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH), .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH), .MAX_BATCH(MAX_BATCH),
        .CLASSIFIER_CAPTURE_MAX_BATCH(CLASSIFIER_CAPTURE_MAX_BATCH),
        .MAX_INPUT_CHANNELS(MAX_INPUT_CHANNELS),
        .MAX_OUTPUT_CHANNELS(MAX_OUTPUT_CHANNELS),
        .SPATIAL(SPATIAL), .POOLED_SPATIAL(POOLED_SPATIAL),
        .KERNEL_SIZE(KERNEL_SIZE), .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH), .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .CHANNEL_INDEX_WIDTH(CHANNEL_INDEX_WIDTH),
        .CHANNEL_COUNT_WIDTH(CHANNEL_COUNT_WIDTH),
        .OBS_INPUT_CHANNELS(OBS_INPUT_CHANNELS),
        .OBS_OUTPUT_CHANNELS(OBS_OUTPUT_CHANNELS),
        .SURROGATE_WAVE_CHANNELS(SURROGATE_WAVE_CHANNELS),
        .SURROGATE_SPATIAL_BLOCK_SIZE(SURROGATE_SPATIAL_BLOCK_SIZE),
        .TRACE_WAVE_CHANNELS(TRACE_WAVE_CHANNELS),
        .CAPTURE_BEFORE_LEARNING(1'b1),
        .REGISTER_BANK_LAYOUT_CHECK(1'b1),
        .USE_SAMPLE_CONTEXT(USE_SAMPLE_CONTEXT),
        .USE_CLASSIFIER_STUDENT_TRACE(USE_CLASSIFIER_STUDENT_TRACE),
        .USE_SYSTEM_OWNER(1'b1),
        .USE_INTERNAL_MODULATION(
            USE_INTERNAL_MODULATION && ENABLE_SCNN_CONV8_LEARNING),
        .USE_EXTERNAL_OUTPUT_LEARNER(USE_EXTERNAL_OUTPUT_LEARNER),
        .ENABLE_SCNN_CONV8_LEARNING(ENABLE_SCNN_CONV8_LEARNING),
        .INTERNAL_MODULATION_BATCH(INTERNAL_MODULATION_BATCH),
        .INTERNAL_MODULATION_TRACE_SPATIAL_MAJOR(
            INTERNAL_MODULATION_TRACE_SPATIAL_MAJOR),
        .MAX_TIMESTEPS(MAX_TIMESTEPS),
        .CLASSIFIER_CAPTURE_MAX_TIMESTEPS(
            CLASSIFIER_CAPTURE_MAX_TIMESTEPS),
        .TIMESTEP_WIDTH(TIMESTEP_WIDTH),
        .TIMESTEP_COUNT_WIDTH(TIMESTEP_COUNT_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH),
        .MAX_FEATURES(MAX_FEATURES), .MAX_CLASSES(MAX_CLASSES),
        .LANES(LANES), .ENGINES(ENGINES), .FRAME_WIDTH(FRAME_WIDTH),
        .FEATURE_INDEX_WIDTH(FEATURE_INDEX_WIDTH),
        .FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
        .CLASS_INDEX_WIDTH(CLASS_INDEX_WIDTH),
        .CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH),
        .WAVE_CLASS_INDEX_WIDTH(CLASS_INDEX_WIDTH),
        .WAVE_CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH)
    ) u_selected_layers (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_cfg_valid(path_cfg_valid), .o_cfg_ready(path_cfg_ready),
        .i_cfg_profile_id(profile_id_q),
        .i_cfg_tensor_id(conv_tensor_id_q),
        .i_cfg_batch_count(batch_count_q),
        .i_cfg_timestep(run_timestep_q),
        .i_cfg_timestep_count(run_timestep_count_q),
        .i_cfg_total_input_channels(total_input_channels_q),
        .i_cfg_total_output_channels(total_output_channels_q),
        .i_cfg_parameter_base(conv_parameter_base_q),
        .i_cfg_weight_q16_base(conv_weight_q16_base_q),
        .i_cfg_vote_increment(conv_vote_increment_q),
        .i_cfg_vote_threshold(conv_vote_threshold_q),
        .i_cfg_code_min(code_min_q), .i_cfg_code_max(code_max_q),
        .o_cfg_error(path_cfg_error), .o_active(path_active),
        .o_active_profile_id(_unused_active_profile_id),
        .o_active_tensor_id(_unused_active_tensor_id),
        .i_modulation_valid(i_modulation_valid),
        .o_modulation_ready(o_modulation_ready),
        .i_modulation_row(i_modulation_row),
        .i_modulation_col(i_modulation_col),
        .i_modulation_q8_8(i_modulation_q8_8),
        .o_capture_complete(o_capture_complete),
        .o_gradient_update_accepted(_unused_gradient_update_accepted),
        .o_gradient_zero_skipped(_unused_gradient_zero_skipped),
        .o_gradient_valid(_unused_gradient_valid),
        .o_gradient_ready(_unused_gradient_ready),
        .o_gradient_sign(_unused_gradient_sign),
        .o_gradient_local_addr(_unused_gradient_local_addr),
        .o_gradient_parameter_addr(_unused_gradient_parameter_addr),
        .o_gradient_weight_q16_addr(_unused_gradient_weight_q16_addr),
        .o_gradient_weight_q64_addr(_unused_gradient_weight_q64_addr),
        .o_gradient_weight_q64_lane(_unused_gradient_weight_q64_lane),
        .o_gradient_tile_last(_unused_gradient_tile_last),
        .o_read_valid(o_read_valid), .i_read_ready(i_read_ready),
        .o_read_profile_id(o_read_profile_id),
        .o_read_tensor_id(o_read_tensor_id),
        .o_read_parameter_addr(o_read_parameter_addr), .o_read_tag(o_read_tag),
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
        .o_write_code_fire(path_write_code_fire),
        .o_write_residual_fire(path_write_residual_fire),
        .o_dirty(_unused_dirty), .o_saturated(_unused_saturated),
        .o_protocol_error(path_protocol_error),
        .i_input_valid(i_input_valid), .o_input_ready(o_input_ready),
        .i_input_target(i_input_target), .i_input_sample(i_input_sample),
        .i_input_timestep(i_input_timestep),
        .i_input_sample_version(i_input_sample_version),
        .i_input_channel(i_input_channel), .i_input_spatial(i_input_spatial),
        .i_input_q8_8(i_input_q8_8),
        .i_trace_valid(i_trace_valid), .o_trace_ready(o_trace_ready),
        .i_trace_target(i_trace_target), .i_trace_sample(i_trace_sample),
        .i_trace_timestep(i_trace_timestep),
        .i_trace_sample_version(i_trace_sample_version),
        .i_trace_channel(i_trace_channel), .i_trace_spatial(i_trace_spatial),
        .i_trace_q8_8(i_trace_q8_8),
        .i_surrogate_valid(i_surrogate_valid),
        .o_surrogate_ready(o_surrogate_ready),
        .i_surrogate_target(i_surrogate_target),
        .i_surrogate_sample(i_surrogate_sample),
        .i_surrogate_timestep(i_surrogate_timestep),
        .i_surrogate_sample_version(i_surrogate_sample_version),
        .i_surrogate_channel(i_surrogate_channel),
        .i_surrogate_spatial(i_surrogate_spatial),
        .i_surrogate_q8_8(i_surrogate_q8_8),
        .i_cfg_sample_version(sample_version_q),
        .i_sample_context_valid(i_sample_context_valid),
        .o_sample_context_ready(o_sample_context_ready),
        .i_sample_context_target(i_sample_context_target),
        .i_sample_context_sample(i_sample_context_sample),
        .i_sample_context_version(i_sample_context_version),
        .o_tiles_completed(_unused_tiles_completed),
        .i_cl_cfg_tensor_id(classifier_tensor_id_q),
        .i_cl_cfg_parameter_base(classifier_parameter_base_q),
        .i_cl_cfg_feature_count(classifier_feature_count_q),
        .i_cl_cfg_class_count(classifier_class_count_q),
        .i_cl_cfg_trace_leak_valid(classifier_trace_leak_valid_q),
        .i_cl_cfg_trace_leak_q8_8(classifier_trace_leak_q8_8_q),
        .i_cl_cfg_vote_increment(classifier_vote_increment_q),
        .i_cl_cfg_vote_threshold(classifier_vote_threshold_q),
        .i_cl_cfg_code_min(code_min_q), .i_cl_cfg_code_max(code_max_q),
        .i_feature_trace_wave_valid(i_feature_trace_wave_valid),
        .o_feature_trace_wave_ready(o_feature_trace_wave_ready),
        .i_feature_trace_wave_mask(i_feature_trace_wave_mask),
        .i_feature_trace_wave_base(i_feature_trace_wave_base),
        .i_feature_trace_wave_q8_8_flat(i_feature_trace_wave_q8_8_flat),
        .i_feature_trace_wave_version(i_feature_trace_wave_version),
        .i_feature_trace_wave_frame_index(i_feature_trace_wave_frame_index),
        .i_feature_trace_wave_first(i_feature_trace_wave_first),
        .i_feature_trace_wave_last(i_feature_trace_wave_last),
        .i_feature_trace_wave_sample_reset(i_feature_trace_wave_sample_reset),
        .i_feature_trace_wave_sample_last(i_feature_trace_wave_sample_last),
        .i_sample_label(i_sample_label), .i_logit_wave_valid(i_logit_wave_valid),
        .o_logit_wave_ready(o_logit_wave_ready),
        .i_logit_wave_mask(i_logit_wave_mask),
        .i_logit_wave_base(i_logit_wave_base),
        .i_logit_wave_count(i_logit_wave_count),
        .i_logit_wave_q8_8_flat(i_logit_wave_q8_8_flat),
        .i_logit_wave_version(i_logit_wave_version),
        .i_logit_wave_frame_index(i_logit_wave_frame_index),
        .i_logit_wave_frame_last(i_logit_wave_frame_last),
        .i_logit_wave_sample_last(i_logit_wave_sample_last),
        .o_external_cfg_valid(o_external_cfg_valid),
        .i_external_cfg_ready(i_external_cfg_ready),
        .o_external_cfg_profile_id(o_external_cfg_profile_id),
        .o_external_cfg_tensor_id(o_external_cfg_tensor_id),
        .o_external_cfg_parameter_base(o_external_cfg_parameter_base),
        .o_external_cfg_batch_count(o_external_cfg_batch_count),
        .o_external_cfg_source_count(o_external_cfg_source_count),
        .o_external_cfg_output_count(o_external_cfg_output_count),
        .o_external_cfg_vote_increment(o_external_cfg_vote_increment),
        .o_external_cfg_vote_threshold(o_external_cfg_vote_threshold),
        .o_external_cfg_code_min(o_external_cfg_code_min),
        .o_external_cfg_code_max(o_external_cfg_code_max),
        .i_external_cfg_error(i_external_cfg_error),
        .i_external_active(i_external_active),
        .o_external_release(o_external_release),
        .i_external_release_ready(i_external_release_ready),
        .o_external_source_trace_valid(o_external_source_trace_valid),
        .i_external_source_trace_ready(i_external_source_trace_ready),
        .o_external_source_trace_sample_id(
            o_external_source_trace_sample_id),
        .o_external_source_trace_index(o_external_source_trace_index),
        .o_external_source_trace_q016(o_external_source_trace_q016),
        .i_external_source_trace_error(i_external_source_trace_error),
        .o_external_logit_row_valid(o_external_logit_row_valid),
        .i_external_logit_row_ready(i_external_logit_row_ready),
        .o_external_logit_sample_id(o_external_logit_sample_id),
        .o_external_logit_label(o_external_logit_label),
        .o_external_logit_valid(o_external_logit_valid),
        .i_external_logit_ready(i_external_logit_ready),
        .o_external_logit_q6(o_external_logit_q6),
        .i_external_logit_error(i_external_logit_error),
        .i_external_logit_row_done(i_external_logit_row_done),
        .o_external_observation_done(o_external_observation_done),
        .i_external_observation_done_ready(
            i_external_observation_done_ready),
        .i_external_window_done(i_external_window_done),
        .i_external_window_busy(i_external_window_busy),
        .i_external_context_error(i_external_context_error),
        .i_external_gradient_overflow(i_external_gradient_overflow),
        .i_external_update_config_error(i_external_update_config_error),
        .i_external_request_error(i_external_request_error),
        .i_external_store_error(i_external_store_error),
        .i_external_internal_error(i_external_internal_error),
        .o_learning_start_valid(path_learning_start_valid),
        .i_learning_start_ready(boundary_learning_start_ready),
        .o_commit_valid(path_commit_valid),
        .i_commit_ready(boundary_commit_ready && commit_barrier_ready),
        .o_commit_profile_id(_unused_commit_profile_id),
        .o_commit_sequence(_unused_commit_sequence),
        .o_commit_fire(path_commit_fire),
        .o_inference_hold(o_inference_hold),
        .o_owner_quiescent(path_owner_quiescent),
        .o_abort_request(path_abort_request),
        .o_abort_complete(path_abort_complete),
        .i_system_parameter_owner(i_system_parameter_owner),
        .i_system_learning_enable(i_system_learning_enable),
        .i_system_profile_aligned(i_system_profile_aligned),
        .i_system_profile_id(i_system_profile_id)
    );

    unified_tp_train_infer_version_boundary #(
        .VERSION_WIDTH(VERSION_WIDTH)
    ) u_visibility (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_context_load_valid(i_context_load_valid),
        .o_context_load_ready(o_context_load_ready),
        .i_context_model_version(i_context_model_version),
        .i_context_adapt_version(i_context_adapt_version),
        .i_forward_start_valid(boundary_forward_start_valid),
        .o_forward_start_ready(boundary_forward_start_ready),
        .o_forward_start_fire(o_forward_start_fire),
        .i_forward_done(i_forward_done),
        .o_forward_active(o_forward_active),
        .o_forward_model_version(o_forward_model_version),
        .o_forward_adapt_version(o_forward_adapt_version),
        .i_learning_start_valid(path_learning_start_valid),
        .o_learning_start_ready(boundary_learning_start_ready),
        .o_learning_start_fire(o_learning_start_fire),
        .o_learning_active(boundary_learning_active),
        .o_learning_base_model_version(
            _unused_learning_base_model_version),
        .o_learning_base_adapt_version(
            _unused_learning_base_adapt_version),
        // The shared output learner performs its RMW outside this runtime's
        // private arbiter. Its terminal window pulse therefore supplies the
        // dirty event that keeps the runtime model/adapt versions aligned
        // with the externally published parameter version.
        .i_forward_code_write_fire(path_write_code_fire ||
            (USE_EXTERNAL_OUTPUT_LEARNER && i_external_window_done)),
        .i_optimizer_state_write_fire(path_write_residual_fire ||
            (USE_EXTERNAL_OUTPUT_LEARNER && i_external_window_done)),
        .i_learning_commit_valid(path_commit_valid && commit_barrier_ready),
        .o_learning_commit_ready(boundary_commit_ready),
        .i_learning_abort_complete(path_abort_complete),
        .i_restore_complete(i_restore_complete),
        .o_restore_ready(o_restore_ready),
        .o_recovery_required(o_recovery_required),
        .o_model_version(o_model_version),
        .o_adapt_version(o_adapt_version),
        .o_commit_pulse(o_commit_pulse),
        .o_commit_code_dirty(o_commit_code_dirty),
        .o_commit_optimizer_dirty(o_commit_optimizer_dirty),
        .o_detach_pulse(o_detach_pulse),
        .i_clear_error(i_clear_error),
        .o_protocol_error(boundary_error)
    );

    initial begin
        if (MAX_TIMESTEPS < 1 || MAX_BATCH < 1)
            $error("SCNN selected-layer runtime dimensions must be positive");
    end
endmodule

`default_nettype wire
