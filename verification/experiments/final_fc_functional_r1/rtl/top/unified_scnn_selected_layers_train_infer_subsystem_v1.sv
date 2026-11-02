`timescale 1ns/1ps
`default_nettype none

// Complete selected-layer SCNN train/infer subsystem. The graph runtime and
// the persistent parameter/cache plane share one commit boundary: updated
// classifier weights are republished and Conv residency is invalidated before
// the new model version becomes visible to the next forward transaction.
module unified_scnn_selected_layers_train_infer_subsystem_v1 #(
    parameter bit ALLOW_UNORDERED_SURROGATE_CAPTURE = 1'b0,
    parameter bit USE_INTERNAL_PARAMETER_STORE = 1'b1,
    parameter bit USE_CONTEXT_PARAMETER_BASE = 1'b0,
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_DEPTH = 32768,
    parameter int PARAM_ADDR_WIDTH = $clog2(PARAM_DEPTH),
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int MAX_BATCH = 10,
    parameter int MAX_TIMESTEPS = 20,
    parameter int CLASSIFIER_CAPTURE_MAX_BATCH = MAX_BATCH,
    parameter int CLASSIFIER_CAPTURE_MAX_TIMESTEPS = MAX_TIMESTEPS,
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
    parameter int ARRAY_WEIGHT_ADDR_WIDTH = 11,
    parameter int SAMPLE_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int TIMESTEP_WIDTH = MAX_TIMESTEPS <= 2 ? 1 :
        $clog2(MAX_TIMESTEPS),
    parameter int TIMESTEP_COUNT_WIDTH = $clog2(MAX_TIMESTEPS + 1),
    parameter int CHANNEL_COUNT_WIDTH = $clog2(OBS_INPUT_CHANNELS + 1),
    parameter int FEATURE_WIDTH = MAX_FEATURES <= 2 ? 1 :
        $clog2(MAX_FEATURES),
    parameter int FEATURE_COUNT_WIDTH = $clog2(MAX_FEATURES + 1),
    parameter int CLASS_WIDTH = MAX_CLASSES <= 2 ? 1 :
        $clog2(MAX_CLASSES),
    parameter int CLASS_COUNT_WIDTH = $clog2(MAX_CLASSES + 1),
    parameter int SELECTED_LAYER = 7,
    parameter int CONV_SURROGATE_WAVE_CHANNELS = OBS_OUTPUT_CHANNELS,
    parameter int CONV_SURROGATE_SPATIAL_BLOCK_SIZE = 0,
    parameter bit USE_EXTERNAL_OUTPUT_LEARNER = 1'b0,
    parameter bit ENABLE_SCNN_CONV8_LEARNING = 1'b1,
    parameter int INTERNAL_MODULATION_BATCH = 2
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_clear_error,
    input  wire logic                         i_cache_invalidate,

    input  wire logic                         i_parameter_load_valid,
    output logic                              o_parameter_load_ready,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]  i_parameter_load_addr,
    input  wire logic signed [15:0]           i_parameter_load_code,
    input  wire logic signed [15:0]           i_parameter_load_residual,
    output logic                              o_store_load_valid,
    input  wire logic                         i_store_load_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]       o_store_load_addr,
    output logic signed [15:0]                o_store_load_code,
    output logic signed [15:0]                o_store_load_residual,
    output logic                              o_store_learning_read_valid,
    input  wire logic                         i_store_learning_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]       o_store_learning_read_addr,
    output logic [TAG_WIDTH-1:0]              o_store_learning_read_tag,
    input  wire logic                         i_store_learning_response_valid,
    output logic                              o_store_learning_response_ready,
    input  wire logic [TAG_WIDTH-1:0]         i_store_learning_response_tag,
    input  wire logic signed [15:0]           i_store_learning_response_code,
    input  wire logic signed [15:0]           i_store_learning_response_residual,
    input  wire logic                         i_store_learning_response_error,
    output logic                              o_store_learning_write_valid,
    input  wire logic                         i_store_learning_write_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]       o_store_learning_write_addr,
    output logic signed [15:0]                o_store_learning_write_code,
    output logic signed [15:0]                o_store_learning_write_residual,
    output logic                              o_store_learning_write_code_enable,
    output logic                              o_store_learning_write_residual_enable,
    output logic                              o_store_forward_read_valid,
    input  wire logic                         i_store_forward_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]       o_store_forward_read_addr,
    output logic [TAG_WIDTH-1:0]              o_store_forward_read_tag,
    input  wire logic                         i_store_forward_response_valid,
    output logic                              o_store_forward_response_ready,
    input  wire logic [TAG_WIDTH-1:0]         i_store_forward_response_tag,
    input  wire logic signed [15:0]           i_store_forward_response_code,
    input  wire logic                         i_store_forward_response_error,
    input  wire logic                         i_store_protocol_error,
    input  wire logic                         i_initialize_valid,
    output logic                              o_initialize_ready,
    output logic                              o_initialize_done,
    input  wire logic                         i_classifier_pair,
    input  wire logic [VERSION_WIDTH-1:0]     i_context_model_version,
    input  wire logic [VERSION_WIDTH-1:0]     i_context_adapt_version,

    input  wire logic                         i_system_parameter_owner,
    input  wire logic                         i_system_learning_enable,
    input  wire logic                         i_system_profile_aligned,
    input  wire logic [PROFILE_ID_WIDTH-1:0]  i_system_profile_id,
    input  wire logic                         i_request_valid,
    output logic                              o_request_ready,
    input  wire logic                         i_learning_enable,
    input  wire logic [BATCH_COUNT_WIDTH-1:0] i_batch_count,
    input  wire logic [TIMESTEP_WIDTH-1:0]     i_timestep,
    input  wire logic [TIMESTEP_COUNT_WIDTH-1:0]
                                                i_timestep_count,
    input  wire logic [VERSION_WIDTH-1:0]      i_sample_version,
    input  wire logic [MAX_BATCH*CLASS_WIDTH-1:0]
                                                i_labels_flat,
    output logic                              o_done,

    input  wire logic [PROFILE_ID_WIDTH-1:0]  i_profile_id,
    input  wire logic [TENSOR_ID_WIDTH-1:0]   i_conv_tensor_id,
    input  wire logic [TENSOR_ID_WIDTH-1:0]   i_classifier_tensor_id,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]
                                                i_total_input_channels,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]
                                                i_total_output_channels,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]  i_conv_parameter_base,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]  i_conv_context_parameter_base,
    input  wire logic [Q16_ADDR_WIDTH-1:0]    i_conv_weight_q16_base,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]  i_classifier_parameter_base,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0]
                                                i_classifier_feature_count,
    input  wire logic [CLASS_COUNT_WIDTH-1:0] i_classifier_class_count,
    input  wire logic                         i_classifier_trace_leak_valid,
    input  wire logic signed [15:0]           i_classifier_trace_leak_q8_8,
    input  wire logic [VOTE_WIDTH-1:0]        i_conv_vote_increment,
    input  wire logic [VOTE_WIDTH-1:0]        i_conv_vote_threshold,
    input  wire logic [VOTE_WIDTH-1:0]        i_classifier_vote_increment,
    input  wire logic [VOTE_WIDTH-1:0]        i_classifier_vote_threshold,
    input  wire logic signed [CODE_WIDTH-1:0] i_code_min,
    input  wire logic signed [CODE_WIDTH-1:0] i_code_max,

    input  wire logic                         i_forward_start_valid,
    output logic                              o_forward_start_ready,
    output logic                              o_forward_start_fire,
    input  wire logic                         i_forward_done,
    output logic                              o_forward_active,
    output logic [VERSION_WIDTH-1:0]           o_forward_model_version,
    output logic [VERSION_WIDTH-1:0]           o_forward_adapt_version,
    input  wire logic                         i_restore_complete,
    output logic                              o_restore_ready,
    output logic                              o_recovery_required,

    input  wire logic [15:0]                  i_conv_in_width,
    input  wire logic [15:0]                  i_conv_in_height,
    input  wire logic [15:0]                  i_conv_in_channels,
    input  wire logic [7:0]                   i_conv_padding,
    input  wire logic [19:0]                  i_conv_layer_tag,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0]
                                                i_target_feature_count,
    input  wire logic signed [15:0]           i_target_trace_leak_q8_8,
    input  wire logic                         i_target_trace_state_reset,

    output logic                              o_graph_start_valid,
    input  wire logic                         i_graph_start_ready,
    output logic                              o_graph_target,
    output logic [SAMPLE_WIDTH-1:0]           o_graph_sample,
    output logic [TIMESTEP_WIDTH-1:0]         o_graph_timestep,
    output logic [VERSION_WIDTH-1:0]           o_graph_version,
    output logic [CLASS_WIDTH-1:0]            o_graph_label,
    output logic                              o_graph_sample_last,
    input  wire logic                         i_graph_done,
    input  wire logic                         i_graph_error,
    output logic                              o_conv_observation_enable,
    output logic                              o_classifier_observation_enable,
    output logic                              o_observation_target,

    output logic                              o_prescan_feature_request_valid,
    input  wire logic                         i_prescan_feature_request_ready,
    output logic [3:0]                        o_prescan_feature_term_valid,
    output logic [39:0]                       o_prescan_feature_input_channel_flat,
    output logic [7:0]                        o_prescan_feature_kernel_row_flat,
    output logic [7:0]                        o_prescan_feature_kernel_col_flat,
    output logic [31:0]                       o_prescan_feature_spatial_base,
    output logic [5:0]                        o_prescan_feature_spatial_count,
    output logic [15:0]                       o_prescan_feature_output_row,
    output logic [15:0]                       o_prescan_feature_output_col,
    input  wire logic                         i_prescan_feature_response_valid,
    output logic                              o_prescan_feature_response_ready,
    input  wire logic [39:0]                  i_prescan_feature_mask0,
    input  wire logic [39:0]                  i_prescan_feature_mask1,
    input  wire logic [39:0]                  i_prescan_feature_mask2,
    input  wire logic [39:0]                  i_prescan_feature_mask3,
    input  wire logic                         i_prescan_feature_service_error,

    input  wire logic                         i_surrogate_valid,
    output logic                              o_surrogate_source_ready,
    input  wire logic                         i_surrogate_target,
    input  wire logic [SAMPLE_WIDTH-1:0]      i_surrogate_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]    i_surrogate_timestep,
    input  wire logic [VERSION_WIDTH-1:0]     i_surrogate_version,
    input  wire logic [19:0]                  i_surrogate_layer_tag,
    input  wire logic [9:0]                   i_surrogate_channel,
    input  wire logic [31:0]                  i_surrogate_spatial,
    input  wire logic signed [15:0]           i_surrogate_q8_8,
    input  wire logic                         i_feature_wave_valid,
    output logic                              o_feature_wave_ready,
    input  wire logic [LANES-1:0]             i_feature_wave_mask,
    input  wire logic [FEATURE_WIDTH-1:0]     i_feature_wave_base,
    input  wire logic [LANES*16-1:0]          i_feature_wave_q8_8_flat,
    input  wire logic [VERSION_WIDTH-1:0]     i_feature_wave_version,
    input  wire logic [FRAME_WIDTH-1:0]       i_feature_wave_frame_index,
    input  wire logic                         i_feature_wave_first,
    input  wire logic                         i_feature_wave_last,
    input  wire logic                         i_feature_wave_sample_reset,
    input  wire logic                         i_feature_wave_sample_last,
    input  wire logic [CLASS_WIDTH-1:0]       i_feature_sample_label,
    input  wire logic                         i_logit_wave_valid,
    output logic                              o_logit_wave_ready,
    input  wire logic [ENGINES-1:0]           i_logit_wave_mask,
    input  wire logic [CLASS_WIDTH-1:0]       i_logit_wave_base,
    input  wire logic [CLASS_COUNT_WIDTH-1:0] i_logit_wave_count,
    input  wire logic [ENGINES*16-1:0]        i_logit_wave_q8_8_flat,
    input  wire logic [VERSION_WIDTH-1:0]     i_logit_wave_version,
    input  wire logic [FRAME_WIDTH-1:0]       i_logit_wave_frame_index,
    input  wire logic                         i_logit_wave_frame_last,
    input  wire logic                         i_logit_wave_sample_last,

    output logic                              o_external_cfg_valid,
    input  wire logic                         i_external_cfg_ready,
    output logic [PROFILE_ID_WIDTH-1:0]       o_external_cfg_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]        o_external_cfg_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]       o_external_cfg_parameter_base,
    output logic [BATCH_COUNT_WIDTH-1:0]      o_external_cfg_batch_count,
    output logic [FEATURE_COUNT_WIDTH-1:0]    o_external_cfg_source_count,
    output logic [CLASS_COUNT_WIDTH-1:0]      o_external_cfg_output_count,
    output logic [VOTE_WIDTH-1:0]             o_external_cfg_vote_increment,
    output logic [VOTE_WIDTH-1:0]             o_external_cfg_vote_threshold,
    output logic signed [CODE_WIDTH-1:0]      o_external_cfg_code_min,
    output logic signed [CODE_WIDTH-1:0]      o_external_cfg_code_max,
    input  wire logic                         i_external_cfg_error,
    input  wire logic                         i_external_active,
    output logic                              o_external_release,
    input  wire logic                         i_external_release_ready,
    output logic                              o_external_source_trace_valid,
    input  wire logic                         i_external_source_trace_ready,
    output logic [SAMPLE_WIDTH-1:0]           o_external_source_trace_sample_id,
    output logic [FEATURE_WIDTH-1:0]          o_external_source_trace_index,
    output logic [15:0]                       o_external_source_trace_q016,
    input  wire logic                         i_external_source_trace_error,
    output logic                              o_external_logit_row_valid,
    input  wire logic                         i_external_logit_row_ready,
    output logic [SAMPLE_WIDTH-1:0]           o_external_logit_sample_id,
    output logic [CLASS_WIDTH-1:0]            o_external_logit_label,
    output logic                              o_external_logit_valid,
    input  wire logic                         i_external_logit_ready,
    output logic signed [15:0]                o_external_logit_q6,
    input  wire logic                         i_external_logit_error,
    input  wire logic                         i_external_logit_row_done,
    output logic                              o_external_observation_done,
    input  wire logic                         i_external_observation_done_ready,
    input  wire logic                         i_external_window_done,
    input  wire logic                         i_external_window_busy,
    input  wire logic                         i_external_context_error,
    input  wire logic                         i_external_gradient_overflow,
    input  wire logic                         i_external_update_config_error,
    input  wire logic                         i_external_request_error,
    input  wire logic                         i_external_store_error,
    input  wire logic                         i_external_internal_error,

    input  wire logic                         i_conv_context_valid,
    output logic                              o_conv_context_ready,
    input  wire logic                         i_conv_context_load_weights,
    input  wire logic [9:0]                   i_conv_output_channel_base,
    input  wire logic [4:0]                   i_conv_output_channel_count,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]
                                                i_conv_input_channel_count,
    input  wire logic                         i_conv_context_pair,
    input  wire logic [VERSION_WIDTH-1:0]     i_conv_context_version,
    input  wire logic                         i_array_idle,
    output logic [3:0]                        o_weight_fill_valid,
    input  wire logic [3:0]                   i_weight_fill_ready,
    output logic [7:0]                        o_weight_fill_tile_flat,
    output logic [3:0]                        o_weight_fill_bank,
    output logic [4*ARRAY_WEIGHT_ADDR_WIDTH-1:0]
                                                o_weight_fill_addr_flat,
    output logic [799:0]                      o_weight_fill_data_flat,
    output logic                              o_weight_fill_busy,

    output logic                              o_classifier_cache_valid,
    output logic                              o_active_classifier_pair,
    output logic [VERSION_WIDTH-1:0]           o_parameter_version,
    output logic                              o_conv_cache_valid,
    output logic [9:0]                        o_conv_cached_output_base,
    output logic [4:0]                        o_conv_cached_output_count,
    output logic [VERSION_WIDTH-1:0]           o_conv_cached_version,
    output logic                              o_conv_cache_invalidate,
    output logic [VERSION_WIDTH-1:0]           o_model_version,
    output logic [VERSION_WIDTH-1:0]           o_adapt_version,
    output logic                              o_capture_complete,
    output logic                              o_learning_start_fire,
    output logic                              o_commit_pulse,
    output logic                              o_commit_code_dirty,
    output logic                              o_commit_optimizer_dirty,
    output logic                              o_detach_pulse,
    output logic                              o_commit_pending,
    output logic                              o_inference_hold,
    output logic                              o_initialized,
    output logic                              o_versions_aligned,
    output logic                              o_busy,
    output logic                              o_quiescent,
    output logic                              o_sticky_error
);
    logic runtime_context_valid;
    logic runtime_context_ready;
    logic runtime_request_valid;
    logic runtime_request_ready;
    logic runtime_forward_start_valid;
    logic runtime_forward_start_ready;
    logic runtime_busy;
    logic runtime_quiescent;
    logic runtime_error;

    logic plane_prime_valid;
    logic plane_prime_ready;
    logic plane_prime_done;
    logic plane_commit_ready;
    logic plane_busy;
    logic plane_error;
    logic initialized_q;
    logic initialize_inflight_q;
    logic initialize_fire;
    logic refresh_request;
    logic forward_admission_allowed;

    logic runtime_read_valid;
    logic runtime_read_ready;
    logic [PROFILE_ID_WIDTH-1:0] runtime_read_profile;
    logic [TENSOR_ID_WIDTH-1:0] runtime_read_tensor;
    logic [PARAM_ADDR_WIDTH-1:0] runtime_read_addr;
    logic [TAG_WIDTH-1:0] runtime_read_tag;
    logic runtime_response_valid;
    logic runtime_response_ready;
    logic [TAG_WIDTH-1:0] runtime_response_tag;
    logic signed [15:0] runtime_response_code;
    logic signed [15:0] runtime_response_residual;
    logic runtime_response_error;
    logic runtime_result_valid;
    logic runtime_result_ready;
    logic [PROFILE_ID_WIDTH-1:0] runtime_result_profile;
    logic [TENSOR_ID_WIDTH-1:0] runtime_result_tensor;
    logic [PARAM_ADDR_WIDTH-1:0] runtime_result_addr;
    logic [TAG_WIDTH-1:0] runtime_result_tag;
    logic signed [15:0] runtime_result_code;
    logic signed [15:0] runtime_result_residual;
    logic runtime_write_code_enable;
    logic runtime_write_residual_enable;

    assign o_initialize_ready = !initialized_q && !initialize_inflight_q &&
        runtime_context_ready && plane_prime_ready;
    assign initialize_fire = i_initialize_valid && o_initialize_ready;
    assign runtime_context_valid = initialize_fire;
    // A returning client refreshes weights without reinitializing its context.
    assign refresh_request = initialized_q && !o_classifier_cache_valid &&
        (i_request_valid || i_forward_start_valid) && i_array_idle &&
        !i_cache_invalidate;
    assign plane_prime_valid = initialize_fire || refresh_request;

    assign forward_admission_allowed = initialized_q &&
        o_classifier_cache_valid && i_array_idle && !i_cache_invalidate;
    assign runtime_request_valid = i_request_valid && forward_admission_allowed;
    assign o_request_ready = runtime_request_ready && forward_admission_allowed;
    assign runtime_forward_start_valid = i_forward_start_valid &&
        forward_admission_allowed;
    assign o_forward_start_ready = runtime_forward_start_ready &&
        forward_admission_allowed;

    assign o_initialized = initialized_q;
    assign o_versions_aligned = !initialized_q ||
        (o_model_version == o_parameter_version);
    assign o_busy = runtime_busy || plane_busy || initialize_inflight_q;
    assign o_quiescent = initialized_q && runtime_quiescent && !plane_busy &&
        !initialize_inflight_q;
    assign o_sticky_error = runtime_error || plane_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            initialized_q <= 1'b0;
            initialize_inflight_q <= 1'b0;
            o_initialize_done <= 1'b0;
        end else begin
            o_initialize_done <= 1'b0;
            if (initialize_fire)
                initialize_inflight_q <= 1'b1;
            if (plane_prime_done && initialize_inflight_q) begin
                initialize_inflight_q <= 1'b0;
                initialized_q <= 1'b1;
                o_initialize_done <= 1'b1;
            end
        end
    end

    unified_scnn_selected_layers_graph_runtime_v1 #(
        .ALLOW_UNORDERED_SURROGATE_CAPTURE(
            ALLOW_UNORDERED_SURROGATE_CAPTURE),
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH), .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH), .MAX_BATCH(MAX_BATCH),
        .MAX_TIMESTEPS(MAX_TIMESTEPS),
        .CLASSIFIER_CAPTURE_MAX_BATCH(CLASSIFIER_CAPTURE_MAX_BATCH),
        .CLASSIFIER_CAPTURE_MAX_TIMESTEPS(
            CLASSIFIER_CAPTURE_MAX_TIMESTEPS),
        .MAX_INPUT_CHANNELS(MAX_INPUT_CHANNELS),
        .MAX_OUTPUT_CHANNELS(MAX_OUTPUT_CHANNELS),
        .OBS_INPUT_CHANNELS(OBS_INPUT_CHANNELS),
        .OBS_OUTPUT_CHANNELS(OBS_OUTPUT_CHANNELS),
        .CONV_SURROGATE_WAVE_CHANNELS(CONV_SURROGATE_WAVE_CHANNELS),
        .CONV_SURROGATE_SPATIAL_BLOCK_SIZE(
            CONV_SURROGATE_SPATIAL_BLOCK_SIZE),
        .SPATIAL(SPATIAL), .POOLED_SPATIAL(POOLED_SPATIAL),
        .KERNEL_SIZE(KERNEL_SIZE), .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH), .VERSION_WIDTH(VERSION_WIDTH),
        .MAX_FEATURES(MAX_FEATURES), .MAX_CLASSES(MAX_CLASSES),
        .LANES(LANES), .ENGINES(ENGINES), .FRAME_WIDTH(FRAME_WIDTH),
        .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .TIMESTEP_WIDTH(TIMESTEP_WIDTH),
        .TIMESTEP_COUNT_WIDTH(TIMESTEP_COUNT_WIDTH),
        .CHANNEL_COUNT_WIDTH(CHANNEL_COUNT_WIDTH),
        .FEATURE_WIDTH(FEATURE_WIDTH),
        .FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
        .CLASS_WIDTH(CLASS_WIDTH), .CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH),
        .SELECTED_LAYER(SELECTED_LAYER),
        .USE_EXTERNAL_OUTPUT_LEARNER(USE_EXTERNAL_OUTPUT_LEARNER),
        .ENABLE_SCNN_CONV8_LEARNING(ENABLE_SCNN_CONV8_LEARNING),
        .INTERNAL_MODULATION_BATCH(INTERNAL_MODULATION_BATCH)
    ) u_runtime (
        .i_request_valid(runtime_request_valid),
        .o_request_ready(runtime_request_ready),
        .i_context_load_valid(runtime_context_valid),
        .o_context_load_ready(runtime_context_ready),
        .i_forward_start_valid(runtime_forward_start_valid),
        .o_forward_start_ready(runtime_forward_start_ready),
        .o_read_valid(runtime_read_valid),
        .i_read_ready(runtime_read_ready),
        .o_read_profile_id(runtime_read_profile),
        .o_read_tensor_id(runtime_read_tensor),
        .o_read_parameter_addr(runtime_read_addr),
        .o_read_tag(runtime_read_tag),
        .i_read_response_valid(runtime_response_valid),
        .o_read_response_ready(runtime_response_ready),
        .i_read_response_code(runtime_response_code),
        .i_read_response_residual(runtime_response_residual),
        .i_read_response_error(runtime_response_error),
        .o_result_valid(runtime_result_valid),
        .i_result_ready(runtime_result_ready),
        .o_result_profile_id(runtime_result_profile),
        .o_result_tensor_id(runtime_result_tensor),
        .o_result_parameter_addr(runtime_result_addr),
        .o_result_tag(runtime_result_tag),
        .o_result_code(runtime_result_code),
        .o_result_residual(runtime_result_residual),
        .o_write_code_enable(runtime_write_code_enable),
        .o_write_residual_enable(runtime_write_residual_enable),
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
        .o_external_source_trace_sample_id(o_external_source_trace_sample_id),
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
        .i_external_observation_done_ready(i_external_observation_done_ready),
        .i_external_window_done(i_external_window_done),
        .i_external_window_busy(i_external_window_busy),
        .i_external_context_error(i_external_context_error),
        .i_external_gradient_overflow(i_external_gradient_overflow),
        .i_external_update_config_error(i_external_update_config_error),
        .i_external_request_error(i_external_request_error),
        .i_external_store_error(i_external_store_error),
        .i_external_internal_error(i_external_internal_error),
        .i_commit_publish_ready(plane_commit_ready),
        .o_busy(runtime_busy), .o_quiescent(runtime_quiescent),
        .o_sticky_error(runtime_error),
        .*
    );

    unified_scnn_selected_layers_parameter_plane_v1 #(
        .USE_INTERNAL_STORE(USE_INTERNAL_PARAMETER_STORE),
        .PARAM_DEPTH(PARAM_DEPTH), .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH), .VERSION_WIDTH(VERSION_WIDTH),
        .MAX_INPUT_CHANNELS(OBS_INPUT_CHANNELS),
        .MAX_CLASSIFIER_FEATURES(MAX_FEATURES),
        .MAX_CLASSES(MAX_CLASSES),
        .ARRAY_WEIGHT_ADDR_WIDTH(ARRAY_WEIGHT_ADDR_WIDTH),
        .INPUT_CHANNEL_COUNT_WIDTH(CHANNEL_COUNT_WIDTH),
        .CLASSIFIER_FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
        .CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH)
    ) u_parameter_plane (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(1'b0),
        .i_clear_error(i_clear_error),
        .i_cache_invalidate(i_cache_invalidate),
        .i_parameter_load_valid(i_parameter_load_valid),
        .o_parameter_load_ready(o_parameter_load_ready),
        .i_parameter_load_addr(i_parameter_load_addr),
        .i_parameter_load_code(i_parameter_load_code),
        .i_parameter_load_residual(i_parameter_load_residual),
        .o_store_load_valid(o_store_load_valid),
        .i_store_load_ready(i_store_load_ready),
        .o_store_load_addr(o_store_load_addr),
        .o_store_load_code(o_store_load_code),
        .o_store_load_residual(o_store_load_residual),
        .o_store_learning_read_valid(o_store_learning_read_valid),
        .i_store_learning_read_ready(i_store_learning_read_ready),
        .o_store_learning_read_addr(o_store_learning_read_addr),
        .o_store_learning_read_tag(o_store_learning_read_tag),
        .i_store_learning_response_valid(i_store_learning_response_valid),
        .o_store_learning_response_ready(o_store_learning_response_ready),
        .i_store_learning_response_tag(i_store_learning_response_tag),
        .i_store_learning_response_code(i_store_learning_response_code),
        .i_store_learning_response_residual(
            i_store_learning_response_residual),
        .i_store_learning_response_error(i_store_learning_response_error),
        .o_store_learning_write_valid(o_store_learning_write_valid),
        .i_store_learning_write_ready(i_store_learning_write_ready),
        .o_store_learning_write_addr(o_store_learning_write_addr),
        .o_store_learning_write_code(o_store_learning_write_code),
        .o_store_learning_write_residual(o_store_learning_write_residual),
        .o_store_learning_write_code_enable(
            o_store_learning_write_code_enable),
        .o_store_learning_write_residual_enable(
            o_store_learning_write_residual_enable),
        .o_store_forward_read_valid(o_store_forward_read_valid),
        .i_store_forward_read_ready(i_store_forward_read_ready),
        .o_store_forward_read_addr(o_store_forward_read_addr),
        .o_store_forward_read_tag(o_store_forward_read_tag),
        .i_store_forward_response_valid(i_store_forward_response_valid),
        .o_store_forward_response_ready(o_store_forward_response_ready),
        .i_store_forward_response_tag(i_store_forward_response_tag),
        .i_store_forward_response_code(i_store_forward_response_code),
        .i_store_forward_response_error(i_store_forward_response_error),
        .i_store_protocol_error(i_store_protocol_error),
        .i_classifier_prime_valid(plane_prime_valid),
        .o_classifier_prime_ready(plane_prime_ready),
        .i_classifier_pair(i_classifier_pair),
        .i_initial_version(i_context_model_version),
        .i_classifier_parameter_base(i_classifier_parameter_base),
        .i_classifier_feature_count(i_classifier_feature_count),
        .i_classifier_class_count(i_classifier_class_count),
        .o_classifier_prime_done(plane_prime_done),
        .i_commit_pending(o_commit_pending),
        .o_commit_publish_ready(plane_commit_ready),
        .i_commit_pulse(o_commit_pulse),
        .i_learning_read_valid(runtime_read_valid),
        .o_learning_read_ready(runtime_read_ready),
        .i_learning_read_addr(runtime_read_addr),
        .i_learning_read_tag(runtime_read_tag),
        .o_learning_response_valid(runtime_response_valid),
        .i_learning_response_ready(runtime_response_ready),
        .o_learning_response_tag(runtime_response_tag),
        .o_learning_response_code(runtime_response_code),
        .o_learning_response_residual(runtime_response_residual),
        .o_learning_response_error(runtime_response_error),
        .i_learning_write_valid(runtime_result_valid),
        .o_learning_write_ready(runtime_result_ready),
        .i_learning_write_addr(runtime_result_addr),
        .i_learning_write_code(runtime_result_code),
        .i_learning_write_residual(runtime_result_residual),
        .i_learning_write_code_enable(runtime_write_code_enable),
        .i_learning_write_residual_enable(runtime_write_residual_enable),
        .i_conv_context_valid(i_conv_context_valid),
        .o_conv_context_ready(o_conv_context_ready),
        .i_conv_context_load_weights(i_conv_context_load_weights),
        .i_conv_output_channel_base(i_conv_output_channel_base),
        .i_conv_output_channel_count(i_conv_output_channel_count),
        .i_conv_input_channel_count(i_conv_input_channel_count),
        .i_conv_context_pair(i_conv_context_pair),
        // Learning captures its selected-layer base at request admission;
        // each forward descriptor can refill a different layer afterwards.
        .i_conv_parameter_base(USE_CONTEXT_PARAMETER_BASE ?
            i_conv_context_parameter_base : i_conv_parameter_base),
        .i_conv_context_version(i_conv_context_version),
        .i_array_idle(i_array_idle),
        .o_weight_fill_valid(o_weight_fill_valid),
        .i_weight_fill_ready(i_weight_fill_ready),
        .o_weight_fill_tile_flat(o_weight_fill_tile_flat),
        .o_weight_fill_bank(o_weight_fill_bank),
        .o_weight_fill_addr_flat(o_weight_fill_addr_flat),
        .o_weight_fill_data_flat(o_weight_fill_data_flat),
        .o_weight_fill_busy(o_weight_fill_busy),
        .o_classifier_cache_valid(o_classifier_cache_valid),
        .o_active_classifier_pair(o_active_classifier_pair),
        .o_active_version(o_parameter_version),
        .o_conv_cache_valid(o_conv_cache_valid),
        .o_conv_cached_output_base(o_conv_cached_output_base),
        .o_conv_cached_output_count(o_conv_cached_output_count),
        .o_conv_cached_version(o_conv_cached_version),
        .o_conv_cache_invalidate(o_conv_cache_invalidate),
        .o_busy(plane_busy), .o_sticky_error(plane_error)
    );

    initial begin
        if (CODE_WIDTH != 16 || RESIDUAL_WIDTH != 16)
            $error("SCNN selected-layer parameter plane requires 16-bit code/residual");
    end
endmodule

`default_nettype wire
