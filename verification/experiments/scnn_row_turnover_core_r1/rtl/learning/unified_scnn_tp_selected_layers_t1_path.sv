`timescale 1ns/1ps
`default_nettype none

// One-timestep numerical transaction fed by student/target observations.
// Reuses both learners and a single shared RMW port. Capture-first mode keeps
// forward available until both captures finish, then acquires learning ownership.
// The external version boundary drains forward and consumes the joint commit.
// A temporal coordinator may re-admit this path for successive absolute
// timesteps; both selected banks commit before the next transaction starts.
// Active child faults fail-stop here: no rollback or child-reset recovery is claimed.
module unified_scnn_tp_selected_layers_t1_path #(
    parameter bit ALLOW_UNORDERED_SURROGATE_CAPTURE = 1'b0,
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int MAX_BATCH = 10,
    parameter int CLASSIFIER_CAPTURE_MAX_BATCH = MAX_BATCH,
    parameter int MAX_INPUT_CHANNELS = 8,
    parameter int MAX_OUTPUT_CHANNELS = 12,
    parameter int SPATIAL = 4,
    parameter int POOLED_SPATIAL = 2,
    parameter int KERNEL_SIZE = 3,
    parameter int Q16_ADDR_WIDTH = 24,
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int SAMPLE_WIDTH = (MAX_BATCH <= 2) ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int CHANNEL_INDEX_WIDTH = 9,
    parameter int CHANNEL_COUNT_WIDTH = 10,
    parameter int INPUT_INDEX_WIDTH =
        (MAX_INPUT_CHANNELS*SPATIAL*SPATIAL <= 2) ? 1 :
        $clog2(MAX_INPUT_CHANNELS*SPATIAL*SPATIAL),
    parameter int TRACE_INDEX_WIDTH =
        (MAX_OUTPUT_CHANNELS*POOLED_SPATIAL*POOLED_SPATIAL <= 2) ? 1 :
        $clog2(MAX_OUTPUT_CHANNELS*POOLED_SPATIAL*POOLED_SPATIAL),
    parameter int SURROGATE_INDEX_WIDTH =
        (MAX_OUTPUT_CHANNELS*SPATIAL*SPATIAL <= 2) ? 1 :
        $clog2(MAX_OUTPUT_CHANNELS*SPATIAL*SPATIAL),
    parameter int OBS_INPUT_CHANNELS = 512,
    parameter int OBS_OUTPUT_CHANNELS = 512,
    parameter int SURROGATE_WAVE_CHANNELS = 0,
    parameter int SURROGATE_SPATIAL_BLOCK_SIZE = 0,
    parameter int TRACE_WAVE_CHANNELS = 0,
    parameter bit CAPTURE_BEFORE_LEARNING = 1'b0,
    parameter bit REGISTER_BANK_LAYOUT_CHECK = 1'b0,
    parameter bit USE_SAMPLE_CONTEXT = 1'b0,
    parameter bit USE_CLASSIFIER_STUDENT_TRACE = 1'b0,
    parameter bit USE_SYSTEM_OWNER = 1'b0,
    parameter bit USE_INTERNAL_MODULATION = 1'b0,
    parameter bit USE_EXTERNAL_OUTPUT_LEARNER = 1'b0,
    parameter bit ENABLE_SCNN_CONV8_LEARNING = 1'b1,
    parameter int INTERNAL_MODULATION_BATCH = 2,
    parameter bit INTERNAL_MODULATION_TRACE_SPATIAL_MAJOR = 1'b0,
    parameter int MAX_TIMESTEPS = 1,
    parameter int CLASSIFIER_CAPTURE_MAX_TIMESTEPS = MAX_TIMESTEPS,
    parameter int TIMESTEP_WIDTH = MAX_TIMESTEPS <= 2 ? 1 : $clog2(MAX_TIMESTEPS),
    parameter int VERSION_WIDTH = 16,
    parameter int MAX_FEATURES = OBS_OUTPUT_CHANNELS*POOLED_SPATIAL*POOLED_SPATIAL,
    parameter int MAX_CLASSES = 11,
    parameter int LANES = 8,
    parameter int ENGINES = 12,
    parameter int FRAME_WIDTH = 8,
    parameter int MAX_GRADIENT_ELEMENTS = MAX_FEATURES*MAX_CLASSES,
    parameter int BATCH_INDEX_WIDTH = SAMPLE_WIDTH,
    parameter int TIMESTEP_INDEX_WIDTH = TIMESTEP_WIDTH,
    parameter int TIMESTEP_COUNT_WIDTH = $clog2(MAX_TIMESTEPS + 1),
    parameter int FEATURE_INDEX_WIDTH = MAX_FEATURES <= 2 ? 1 : $clog2(MAX_FEATURES),
    parameter int FEATURE_COUNT_WIDTH = $clog2(MAX_FEATURES+1),
    parameter int CLASS_INDEX_WIDTH = MAX_CLASSES <= 2 ? 1 : $clog2(MAX_CLASSES),
    parameter int CLASS_COUNT_WIDTH = $clog2(MAX_CLASSES+1),
    parameter int WAVE_CLASS_INDEX_WIDTH = CLASS_INDEX_WIDTH,
    parameter int WAVE_CLASS_COUNT_WIDTH = CLASS_COUNT_WIDTH
) (
    input wire logic i_clk,
    input wire logic i_rst_n,
    input wire logic i_cfg_valid,
    output logic o_cfg_ready,
    input wire logic [PROFILE_ID_WIDTH-1:0] i_cfg_profile_id,
    input wire logic [TENSOR_ID_WIDTH-1:0] i_cfg_tensor_id,
    input wire logic [BATCH_COUNT_WIDTH-1:0] i_cfg_batch_count,
    input wire logic [TIMESTEP_WIDTH-1:0] i_cfg_timestep,
    input wire logic [TIMESTEP_COUNT_WIDTH-1:0] i_cfg_timestep_count,
    input wire logic [CHANNEL_COUNT_WIDTH-1:0] i_cfg_total_input_channels,
    input wire logic [CHANNEL_COUNT_WIDTH-1:0] i_cfg_total_output_channels,
    input wire logic [PARAM_ADDR_WIDTH-1:0] i_cfg_parameter_base,
    input wire logic [Q16_ADDR_WIDTH-1:0] i_cfg_weight_q16_base,
    input wire logic [VOTE_WIDTH-1:0] i_cfg_vote_increment,
    input wire logic [VOTE_WIDTH-1:0] i_cfg_vote_threshold,
    input wire logic signed [CODE_WIDTH-1:0] i_cfg_code_min,
    input wire logic signed [CODE_WIDTH-1:0] i_cfg_code_max,
    output logic o_cfg_error,
    output logic o_active,
    output logic [PROFILE_ID_WIDTH-1:0] o_active_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0] o_active_tensor_id,
    input wire logic i_modulation_valid,
    output logic o_modulation_ready,
    input wire logic [SAMPLE_WIDTH-1:0] i_modulation_row,
    input wire logic [SAMPLE_WIDTH-1:0] i_modulation_col,
    input wire logic signed [15:0] i_modulation_q8_8,
    output logic o_capture_complete,
    output logic o_gradient_update_accepted,
    output logic o_gradient_zero_skipped,
    output logic o_gradient_valid,
    output logic o_gradient_ready,
    output logic signed [1:0] o_gradient_sign,
    output logic [PARAM_ADDR_WIDTH-1:0] o_gradient_local_addr,
    output logic [PARAM_ADDR_WIDTH-1:0] o_gradient_parameter_addr,
    output logic [Q16_ADDR_WIDTH-1:0] o_gradient_weight_q16_addr,
    output logic [Q64_ADDR_WIDTH-1:0] o_gradient_weight_q64_addr,
    output logic [1:0] o_gradient_weight_q64_lane,
    output logic o_gradient_tile_last,
    output logic o_read_valid,
    input wire logic i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0] o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0] o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0] o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0] o_read_tag,
    input wire logic i_read_response_valid,
    output logic o_read_response_ready,
    input wire logic signed [CODE_WIDTH-1:0] i_read_response_code,
    input wire logic signed [RESIDUAL_WIDTH-1:0] i_read_response_residual,
    input wire logic i_read_response_error,
    output logic o_result_valid,
    input wire logic i_result_ready,
    output logic [PROFILE_ID_WIDTH-1:0] o_result_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0] o_result_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0] o_result_parameter_addr,
    output logic [TAG_WIDTH-1:0] o_result_tag,
    output logic signed [CODE_WIDTH-1:0] o_result_code,
    output logic signed [RESIDUAL_WIDTH-1:0] o_result_residual,
    output logic o_write_code_enable,
    output logic o_write_residual_enable,
    output logic o_write_code_fire,
    output logic o_write_residual_fire,
    output logic o_dirty,
    output logic o_saturated,
    output logic o_protocol_error,
    input wire logic i_input_valid,
    output logic o_input_ready,
    input wire logic i_input_target,
    input wire logic [SAMPLE_WIDTH-1:0] i_input_sample,
    input wire logic [TIMESTEP_WIDTH-1:0] i_input_timestep,
    input wire logic [VERSION_WIDTH-1:0] i_input_sample_version,
    input wire logic [CHANNEL_INDEX_WIDTH-1:0] i_input_channel,
    input wire logic [$clog2(SPATIAL*SPATIAL)-1:0] i_input_spatial,
    input wire logic signed [15:0] i_input_q8_8,
    input wire logic i_trace_valid,
    output logic o_trace_ready,
    input wire logic i_trace_target,
    input wire logic [SAMPLE_WIDTH-1:0] i_trace_sample,
    input wire logic [TIMESTEP_WIDTH-1:0] i_trace_timestep,
    input wire logic [VERSION_WIDTH-1:0] i_trace_sample_version,
    input wire logic [CHANNEL_INDEX_WIDTH-1:0] i_trace_channel,
    input wire logic [$clog2(POOLED_SPATIAL*POOLED_SPATIAL)-1:0] i_trace_spatial,
    input wire logic signed [15:0] i_trace_q8_8,
    input wire logic i_surrogate_valid,
    output logic o_surrogate_ready,
    input wire logic i_surrogate_target,
    input wire logic [SAMPLE_WIDTH-1:0] i_surrogate_sample,
    input wire logic [TIMESTEP_WIDTH-1:0] i_surrogate_timestep,
    input wire logic [VERSION_WIDTH-1:0] i_surrogate_sample_version,
    input wire logic [CHANNEL_INDEX_WIDTH-1:0] i_surrogate_channel,
    input wire logic [$clog2(SPATIAL*SPATIAL)-1:0] i_surrogate_spatial,
    input wire logic signed [15:0] i_surrogate_q8_8,
    input wire logic [VERSION_WIDTH-1:0] i_cfg_sample_version,
    // Register the raw state epoch of each student/target forward sample.
    // i_cfg_sample_version remains the common learning-window epoch.
    input wire logic i_sample_context_valid,
    output logic o_sample_context_ready,
    input wire logic i_sample_context_target,
    input wire logic [SAMPLE_WIDTH-1:0] i_sample_context_sample,
    input wire logic [VERSION_WIDTH-1:0] i_sample_context_version,
    output logic [12:0] o_tiles_completed,
    input wire logic [TENSOR_ID_WIDTH-1:0]            i_cl_cfg_tensor_id,
    input wire logic [PARAM_ADDR_WIDTH-1:0]           i_cl_cfg_parameter_base,
    input wire logic [FEATURE_COUNT_WIDTH-1:0]        i_cl_cfg_feature_count,
    input wire logic [CLASS_COUNT_WIDTH-1:0]          i_cl_cfg_class_count,
    input wire logic i_cl_cfg_trace_leak_valid,
    input wire logic signed [15:0]                    i_cl_cfg_trace_leak_q8_8,
    input wire logic [VOTE_WIDTH-1:0]                 i_cl_cfg_vote_increment,
    input wire logic [VOTE_WIDTH-1:0]                 i_cl_cfg_vote_threshold,
    input wire logic signed [CODE_WIDTH-1:0]          i_cl_cfg_code_min,
    input wire logic signed [CODE_WIDTH-1:0]          i_cl_cfg_code_max,
    input wire logic i_feature_trace_wave_valid,
    output logic o_feature_trace_wave_ready,
    input wire logic [LANES-1:0]                     i_feature_trace_wave_mask,
    input wire logic [FEATURE_INDEX_WIDTH-1:0]       i_feature_trace_wave_base,
    input wire logic [LANES*16-1:0]                  i_feature_trace_wave_q8_8_flat,
    input wire logic [VERSION_WIDTH-1:0]             i_feature_trace_wave_version,
    input wire logic [FRAME_WIDTH-1:0]               i_feature_trace_wave_frame_index,
    input wire logic i_feature_trace_wave_first,
    input wire logic i_feature_trace_wave_last,
    input wire logic i_feature_trace_wave_sample_reset,
    input wire logic i_feature_trace_wave_sample_last,
    input wire logic [CLASS_INDEX_WIDTH-1:0]         i_sample_label,
    input wire logic i_logit_wave_valid,
    output logic o_logit_wave_ready,
    input wire logic [ENGINES-1:0]                   i_logit_wave_mask,
    input wire logic [WAVE_CLASS_INDEX_WIDTH-1:0]    i_logit_wave_base,
    input wire logic [WAVE_CLASS_COUNT_WIDTH-1:0]    i_logit_wave_count,
    input wire logic [ENGINES*16-1:0]                i_logit_wave_q8_8_flat,
    input wire logic [VERSION_WIDTH-1:0]             i_logit_wave_version,
    input wire logic [FRAME_WIDTH-1:0]               i_logit_wave_frame_index,
    input wire logic i_logit_wave_frame_last,
    input wire logic i_logit_wave_sample_last,
    output logic o_learning_start_valid,
    input wire logic i_learning_start_ready,
    output logic o_commit_valid,
    input wire logic i_commit_ready,
    output logic [7:0] o_commit_profile_id,
    output logic [31:0] o_commit_sequence,
    output logic o_commit_fire,
    output logic o_inference_hold,
    output logic o_owner_quiescent,
    output logic o_abort_request,
    output logic o_abort_complete,

    output logic                                  o_external_cfg_valid,
    input wire logic                              i_external_cfg_ready,
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
    input wire logic                              i_external_cfg_error,
    input wire logic                              i_external_active,
    output logic                                  o_external_release,
    input wire logic                              i_external_release_ready,
    output logic                                  o_external_source_trace_valid,
    input wire logic                              i_external_source_trace_ready,
    output logic [BATCH_INDEX_WIDTH-1:0]          o_external_source_trace_sample_id,
    output logic [FEATURE_INDEX_WIDTH-1:0]        o_external_source_trace_index,
    output logic [15:0]                           o_external_source_trace_q016,
    input wire logic                              i_external_source_trace_error,
    output logic                                  o_external_logit_row_valid,
    input wire logic                              i_external_logit_row_ready,
    output logic [BATCH_INDEX_WIDTH-1:0]          o_external_logit_sample_id,
    output logic [CLASS_INDEX_WIDTH-1:0]          o_external_logit_label,
    output logic                                  o_external_logit_valid,
    input wire logic                              i_external_logit_ready,
    output logic signed [15:0]                    o_external_logit_q6,
    input wire logic                              i_external_logit_error,
    input wire logic                              i_external_logit_row_done,
    output logic                                  o_external_observation_done,
    input wire logic                              i_external_observation_done_ready,
    input wire logic                              i_external_window_done,
    input wire logic                              i_external_window_busy,
    input wire logic                              i_external_context_error,
    input wire logic                              i_external_gradient_overflow,
    input wire logic                              i_external_update_config_error,
    input wire logic                              i_external_request_error,
    input wire logic                              i_external_store_error,
    input wire logic                              i_external_internal_error,

    input wire logic i_system_parameter_owner,
    input wire logic i_system_learning_enable,
    input wire logic i_system_profile_aligned,
    input wire logic [PROFILE_ID_WIDTH-1:0] i_system_profile_id
);
    logic cv_i_cfg_valid;
    logic cv_o_cfg_ready;
    logic [PROFILE_ID_WIDTH-1:0] cv_i_cfg_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] cv_i_cfg_tensor_id;
    logic [BATCH_COUNT_WIDTH-1:0] cv_i_cfg_batch_count;
    logic [CHANNEL_COUNT_WIDTH-1:0] cv_i_cfg_total_input_channels;
    logic [CHANNEL_COUNT_WIDTH-1:0] cv_i_cfg_total_output_channels;
    logic [PARAM_ADDR_WIDTH-1:0] cv_i_cfg_parameter_base;
    logic [Q16_ADDR_WIDTH-1:0] cv_i_cfg_weight_q16_base;
    logic [VOTE_WIDTH-1:0] cv_i_cfg_vote_increment;
    logic [VOTE_WIDTH-1:0] cv_i_cfg_vote_threshold;
    logic signed [CODE_WIDTH-1:0] cv_i_cfg_code_min;
    logic signed [CODE_WIDTH-1:0] cv_i_cfg_code_max;
    logic cv_o_cfg_error;
    logic cv_o_active;
    logic [PROFILE_ID_WIDTH-1:0] cv_o_active_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] cv_o_active_tensor_id;
    logic cv_i_release;
    logic cv_o_release_ready;
    logic cv_i_modulation_valid;
    logic cv_o_modulation_ready;
    logic [SAMPLE_WIDTH-1:0] cv_i_modulation_row;
    logic [SAMPLE_WIDTH-1:0] cv_i_modulation_col;
    logic signed [15:0] cv_i_modulation_q8_8;
    logic cv_o_capture_complete;
    logic cv_o_window_done;
    logic cv_o_window_busy;
    logic cv_o_gradient_update_accepted;
    logic cv_o_gradient_zero_skipped;
    logic cv_o_gradient_valid;
    logic cv_o_gradient_ready;
    logic signed [1:0] cv_o_gradient_sign;
    logic [PARAM_ADDR_WIDTH-1:0] cv_o_gradient_local_addr;
    logic [PARAM_ADDR_WIDTH-1:0] cv_o_gradient_parameter_addr;
    logic [Q16_ADDR_WIDTH-1:0] cv_o_gradient_weight_q16_addr;
    logic [Q64_ADDR_WIDTH-1:0] cv_o_gradient_weight_q64_addr;
    logic [1:0] cv_o_gradient_weight_q64_lane;
    logic cv_o_gradient_tile_last;
    logic cv_o_read_valid;
    logic cv_i_read_ready;
    logic [PROFILE_ID_WIDTH-1:0] cv_o_read_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] cv_o_read_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] cv_o_read_parameter_addr;
    logic [TAG_WIDTH-1:0] cv_o_read_tag;
    logic cv_i_read_response_valid;
    logic cv_o_read_response_ready;
    logic signed [CODE_WIDTH-1:0] cv_i_read_response_code;
    logic signed [RESIDUAL_WIDTH-1:0] cv_i_read_response_residual;
    logic cv_i_read_response_error;
    logic cv_o_result_valid;
    logic cv_i_result_ready;
    logic [PROFILE_ID_WIDTH-1:0] cv_o_result_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] cv_o_result_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] cv_o_result_parameter_addr;
    logic [TAG_WIDTH-1:0] cv_o_result_tag;
    logic signed [CODE_WIDTH-1:0] cv_o_result_code;
    logic signed [RESIDUAL_WIDTH-1:0] cv_o_result_residual;
    logic cv_o_write_code_enable;
    logic cv_o_write_residual_enable;
    logic cv_o_write_code_fire;
    logic cv_o_write_residual_fire;
    logic cv_o_dirty;
    logic cv_o_saturated;
    logic cv_o_protocol_error;
    logic cv_i_input_valid;
    logic cv_o_input_ready;
    logic cv_i_input_target;
    logic [SAMPLE_WIDTH-1:0] cv_i_input_sample;
    logic [TIMESTEP_WIDTH-1:0] cv_i_input_timestep;
    logic [VERSION_WIDTH-1:0] cv_i_input_sample_version;
    logic [CHANNEL_INDEX_WIDTH-1:0] cv_i_input_channel;
    logic [$clog2(SPATIAL*SPATIAL)-1:0] cv_i_input_spatial;
    logic signed [15:0] cv_i_input_q8_8;
    logic cv_i_trace_valid;
    logic cv_o_trace_ready;
    logic cv_i_trace_target;
    logic [SAMPLE_WIDTH-1:0] cv_i_trace_sample;
    logic [TIMESTEP_WIDTH-1:0] cv_i_trace_timestep;
    logic [VERSION_WIDTH-1:0] cv_i_trace_sample_version;
    logic [CHANNEL_INDEX_WIDTH-1:0] cv_i_trace_channel;
    logic [$clog2(POOLED_SPATIAL*POOLED_SPATIAL)-1:0] cv_i_trace_spatial;
    logic signed [15:0] cv_i_trace_q8_8;
    logic cv_i_surrogate_valid;
    logic cv_o_surrogate_ready;
    logic cv_i_surrogate_target;
    logic [SAMPLE_WIDTH-1:0] cv_i_surrogate_sample;
    logic [TIMESTEP_WIDTH-1:0] cv_i_surrogate_timestep;
    logic [VERSION_WIDTH-1:0] cv_i_surrogate_sample_version;
    logic [CHANNEL_INDEX_WIDTH-1:0] cv_i_surrogate_channel;
    logic [$clog2(SPATIAL*SPATIAL)-1:0] cv_i_surrogate_spatial;
    logic signed [15:0] cv_i_surrogate_q8_8;
    logic [TIMESTEP_WIDTH-1:0] cv_i_cfg_timestep;
    logic [VERSION_WIDTH-1:0] cv_i_cfg_sample_version;
    logic [12:0] cv_o_tiles_completed;
    logic cl_i_clear_error;
    logic cl_i_cfg_valid;
    logic cl_o_cfg_ready;
    logic [PROFILE_ID_WIDTH-1:0]           cl_i_cfg_profile_id;
    logic [TENSOR_ID_WIDTH-1:0]            cl_i_cfg_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0]           cl_i_cfg_parameter_base;
    logic [BATCH_COUNT_WIDTH-1:0]          cl_i_cfg_batch_count;
    logic [TIMESTEP_COUNT_WIDTH-1:0]       cl_i_cfg_timestep_count;
    logic [FEATURE_COUNT_WIDTH-1:0]        cl_i_cfg_feature_count;
    logic [CLASS_COUNT_WIDTH-1:0]          cl_i_cfg_class_count;
    logic cl_i_cfg_trace_leak_valid;
    logic signed [15:0]                    cl_i_cfg_trace_leak_q8_8;
    logic [VOTE_WIDTH-1:0]                 cl_i_cfg_vote_increment;
    logic [VOTE_WIDTH-1:0]                 cl_i_cfg_vote_threshold;
    logic signed [CODE_WIDTH-1:0]          cl_i_cfg_code_min;
    logic signed [CODE_WIDTH-1:0]          cl_i_cfg_code_max;
    logic cl_o_cfg_error;
    logic cl_o_active;
    logic cl_i_release;
    logic cl_o_release_ready;
    logic cl_i_learning_enable;
    logic cl_i_feature_trace_wave_valid;
    logic cl_o_feature_trace_wave_ready;
    logic [LANES-1:0]                     cl_i_feature_trace_wave_mask;
    logic [FEATURE_INDEX_WIDTH-1:0]       cl_i_feature_trace_wave_base;
    logic [LANES*16-1:0]                  cl_i_feature_trace_wave_q8_8_flat;
    logic [VERSION_WIDTH-1:0]             cl_i_feature_trace_wave_version;
    logic [FRAME_WIDTH-1:0]               cl_i_feature_trace_wave_frame_index;
    logic cl_i_feature_trace_wave_first;
    logic cl_i_feature_trace_wave_last;
    logic cl_i_feature_trace_wave_sample_reset;
    logic cl_i_feature_trace_wave_sample_last;
    logic [CLASS_INDEX_WIDTH-1:0]         cl_i_sample_label;
    logic cl_i_logit_wave_valid;
    logic cl_o_logit_wave_ready;
    logic [ENGINES-1:0]                   cl_i_logit_wave_mask;
    logic [WAVE_CLASS_INDEX_WIDTH-1:0]    cl_i_logit_wave_base;
    logic [WAVE_CLASS_COUNT_WIDTH-1:0]    cl_i_logit_wave_count;
    logic [ENGINES*16-1:0]                cl_i_logit_wave_q8_8_flat;
    logic [VERSION_WIDTH-1:0]             cl_i_logit_wave_version;
    logic [FRAME_WIDTH-1:0]               cl_i_logit_wave_frame_index;
    logic cl_i_logit_wave_frame_last;
    logic cl_i_logit_wave_sample_last;
    logic cl_o_window_commit_valid;
    logic cl_i_window_commit_ready;
    logic [PROFILE_ID_WIDTH-1:0]          cl_o_window_commit_profile_id;
    logic [31:0]                           cl_o_window_commit_sequence;
    logic cl_o_timestep_learning_done;
    logic cl_o_batch_learning_done;
    logic [TIMESTEP_COUNT_WIDTH-1:0]       cl_o_completed_timestep_count;
    logic cl_o_inference_hold;
    logic cl_o_window_busy;
    logic cl_o_read_valid;
    logic cl_i_read_ready;
    logic [PROFILE_ID_WIDTH-1:0]          cl_o_read_profile_id;
    logic [TENSOR_ID_WIDTH-1:0]           cl_o_read_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0]          cl_o_read_parameter_addr;
    logic [TAG_WIDTH-1:0]                 cl_o_read_tag;
    logic cl_i_read_response_valid;
    logic cl_o_read_response_ready;
    logic signed [CODE_WIDTH-1:0]         cl_i_read_response_code;
    logic signed [RESIDUAL_WIDTH-1:0]     cl_i_read_response_residual;
    logic cl_i_read_response_error;
    logic cl_o_result_valid;
    logic cl_i_result_ready;
    logic [PROFILE_ID_WIDTH-1:0]          cl_o_result_profile_id;
    logic [TENSOR_ID_WIDTH-1:0]           cl_o_result_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0]          cl_o_result_parameter_addr;
    logic [TAG_WIDTH-1:0]                 cl_o_result_tag;
    logic signed [CODE_WIDTH-1:0]         cl_o_result_code;
    logic signed [RESIDUAL_WIDTH-1:0]     cl_o_result_residual;
    logic cl_o_write_code_enable;
    logic cl_o_write_residual_enable;
    logic cl_o_write_code_fire;
    logic cl_o_write_residual_fire;
    logic cl_o_dirty;
    logic cl_o_saturated;
    logic cl_o_capture_complete;
    logic cl_o_observation_error;
    logic cl_o_context_error;
    logic cl_o_gradient_overflow;
    logic cl_o_parameter_error;
    logic cl_o_protocol_error;
    logic arb_busy, arb_error, joint_ready, joint_start_fire, joint_hold, joint_idle;
    logic joint_conv_release, joint_classifier_ack, joint_fault, joint_cfg_error;
    logic cleanup_q, error_q, child_error;
    logic [VERSION_WIDTH-1:0] observation_version_q;
    logic observation_version_error;
    logic cfg_fire, capture_pending_q, learning_owned_q;
    logic system_admit;
    logic [PROFILE_ID_WIDTH-1:0] captured_profile_q;
    logic [BATCH_COUNT_WIDTH-1:0] captured_batch_q;
    logic [BATCH_COUNT_WIDTH-1:0] classifier_sample_q;
    logic [MAX_BATCH-1:0] context_valid_q [0:1];
    logic [VERSION_WIDTH-1:0] context_version_q [0:1][0:MAX_BATCH-1];
    logic context_fire, context_legal;
    logic input_version_legal, trace_version_legal, surrogate_version_legal;
    logic feature_version_legal, logit_version_legal;
    logic arb_read_valid, arb_read_ready;
    logic join_start_valid, join_abort_request;
    logic shared_trace_valid, shared_trace_ready;
    logic [BATCH_INDEX_WIDTH-1:0] shared_trace_sample;
    logic [FEATURE_INDEX_WIDTH-1:0] shared_trace_index;
    logic [TIMESTEP_INDEX_WIDTH-1:0] shared_trace_timestep;
    logic [VERSION_WIDTH-1:0] shared_trace_version;
    logic signed [15:0] shared_trace_value;
    logic shared_trace_selected, shared_trace_legal;
    localparam int INTERNAL_MOD_SAMPLE_WIDTH =
        INTERNAL_MODULATION_BATCH <= 2 ? 1 : $clog2(INTERNAL_MODULATION_BATCH);
    localparam int INTERNAL_MOD_TRACE_VALUES =
        OBS_OUTPUT_CHANNELS * POOLED_SPATIAL * POOLED_SPATIAL;
    localparam int INTERNAL_MOD_TRACE_INDEX_WIDTH =
        INTERNAL_MOD_TRACE_VALUES <= 2 ? 1 : $clog2(INTERNAL_MOD_TRACE_VALUES);
    localparam int INTERNAL_MOD_PREV_VALUES =
        OBS_INPUT_CHANNELS * SPATIAL * SPATIAL;
    localparam int INTERNAL_MOD_PREV_INDEX_WIDTH =
        INTERNAL_MOD_PREV_VALUES <= 2 ? 1 : $clog2(INTERNAL_MOD_PREV_VALUES);
    logic input_source_valid, trace_source_valid, trace_source_target;
    logic internal_mod_student_ready, internal_mod_target_ready;
    logic internal_mod_prev_ready, internal_mod_start_ready;
    logic internal_mod_capture_complete, internal_mod_error;
    logic internal_mod_valid, internal_mod_ready;
    logic [INTERNAL_MOD_SAMPLE_WIDTH-1:0] internal_mod_row;
    logic [INTERNAL_MOD_SAMPLE_WIDTH-1:0] internal_mod_col;
    logic signed [15:0] internal_mod_q8_8;
    logic internal_mod_last, internal_mod_done, internal_mod_busy;
    logic internal_mod_student_valid, internal_mod_target_valid;
    logic internal_mod_prev_valid;
    logic [INTERNAL_MOD_TRACE_INDEX_WIDTH-1:0] internal_mod_trace_index;
    logic [INTERNAL_MOD_PREV_INDEX_WIDTH-1:0] internal_mod_prev_index;
    localparam bit CONV_MODULATION_ACTIVE =
        USE_INTERNAL_MODULATION && ENABLE_SCNN_CONV8_LEARNING;

    function automatic logic version_matches(
        input logic target,
        input logic [SAMPLE_WIDTH-1:0] sample_index,
        input logic [VERSION_WIDTH-1:0] raw_version
    );
        if (!USE_SAMPLE_CONTEXT)
            return raw_version == observation_version_q;
        // Keep batch indices unsigned. With MAX_BATCH=1, casting the one-bit
        // count to signed int can be narrowed by elaboration and turn 1 into -1.
        if ($unsigned(sample_index) >= MAX_BATCH ||
            $unsigned(sample_index) >= $unsigned(captured_batch_q))
            return 1'b0;
        return context_valid_q[target][sample_index] &&
            context_version_q[target][sample_index] == raw_version;
    endfunction
    logic bank_layout_legal;
    logic bank_layout_admission_legal;
    logic bank_layout_admission_error;
    logic [32:0] conv_end, classifier_end;
    // The backing store is address-based, so distinct tensor IDs alone cannot
    // prevent aliasing. Use widened exclusive ends to catch address wrap too.
    assign conv_end = 33'(i_cfg_parameter_base) +
        33'(i_cfg_total_input_channels) * 33'(i_cfg_total_output_channels) *
        33'(KERNEL_SIZE*KERNEL_SIZE);
    assign classifier_end = 33'(i_cl_cfg_parameter_base) +
        33'(i_cl_cfg_feature_count) * 33'(i_cl_cfg_class_count);
    assign bank_layout_legal = (!ENABLE_SCNN_CONV8_LEARNING ||
        (i_cfg_tensor_id != i_cl_cfg_tensor_id &&
         conv_end > 33'(i_cfg_parameter_base) &&
         conv_end <= (33'd1 << PARAM_ADDR_WIDTH) &&
         (conv_end <= 33'(i_cl_cfg_parameter_base) ||
          classifier_end <= 33'(i_cfg_parameter_base)))) &&
        classifier_end > 33'(i_cl_cfg_parameter_base) &&
        classifier_end <= (33'd1 << PARAM_ADDR_WIDTH) &&
        (!CONV_MODULATION_ACTIVE ||
         (int'(i_cfg_batch_count) == INTERNAL_MODULATION_BATCH &&
          int'(i_cfg_total_input_channels) == OBS_INPUT_CHANNELS &&
          int'(i_cfg_total_output_channels) == OBS_OUTPUT_CHANNELS)) &&
        i_cfg_timestep_count != '0 &&
        int'(i_cfg_timestep_count) <= CLASSIFIER_CAPTURE_MAX_TIMESTEPS &&
        int'(i_cfg_timestep) + int'(i_cfg_timestep_count) <= MAX_TIMESTEPS &&
        int'(i_cfg_batch_count) * int'(i_cfg_timestep_count) <= MAX_BATCH &&
        (!ENABLE_SCNN_CONV8_LEARNING || i_cfg_timestep_count == 1) &&
        (!USE_CLASSIFIER_STUDENT_TRACE ||
         int'(i_cl_cfg_feature_count) == int'(i_cfg_total_output_channels) *
            POOLED_SPATIAL * POOLED_SPATIAL);
    generate
        if (REGISTER_BANK_LAYOUT_CHECK) begin : g_registered_layout_check
            logic checked_q;
            logic legal_q;

            // The runtime client holds its descriptor while cfg_valid waits.
            // Capture the wide geometry/range check before it can drive the
            // shared learner's configuration enables.
            always_ff @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n) begin
                    checked_q <= 1'b0;
                    legal_q <= 1'b0;
                end else if (!i_cfg_valid || cfg_fire) begin
                    checked_q <= 1'b0;
                end else if (!checked_q) begin
                    checked_q <= 1'b1;
                    legal_q <= bank_layout_legal;
                end
            end

            assign bank_layout_admission_legal = checked_q && legal_q;
            assign bank_layout_admission_error = checked_q && !legal_q;
        end else begin : g_direct_layout_check
            assign bank_layout_admission_legal = bank_layout_legal;
            assign bank_layout_admission_error = !bank_layout_legal;
        end
    endgenerate
    assign child_error = cv_o_cfg_error || cv_o_protocol_error ||
        cl_o_cfg_error || cl_o_observation_error || cl_o_context_error ||
        cl_o_gradient_overflow || cl_o_parameter_error || cl_o_protocol_error || arb_error ||
        observation_version_error || internal_mod_error;
    assign input_version_legal = version_matches(i_input_target,
        i_input_sample, i_input_sample_version);
    assign trace_version_legal = version_matches(i_trace_target,
        i_trace_sample, i_trace_sample_version);
    assign surrogate_version_legal = version_matches(i_surrogate_target,
        i_surrogate_sample, i_surrogate_sample_version);
    assign feature_version_legal = classifier_sample_q < captured_batch_q &&
        version_matches(1'b0, SAMPLE_WIDTH'(classifier_sample_q),
            i_feature_trace_wave_version);
    assign logit_version_legal = classifier_sample_q < captured_batch_q &&
        version_matches(1'b0, SAMPLE_WIDTH'(classifier_sample_q), i_logit_wave_version);
    assign shared_trace_legal = version_matches(1'b0,
        SAMPLE_WIDTH'(shared_trace_sample), shared_trace_version);
    assign context_legal =
        $unsigned(i_sample_context_sample) < $unsigned(captured_batch_q) &&
        $unsigned(i_sample_context_sample) < MAX_BATCH &&
        !context_valid_q[i_sample_context_target][i_sample_context_sample];
    assign o_sample_context_ready = USE_SAMPLE_CONTEXT && o_active &&
        !cleanup_q && !error_q && !learning_owned_q;
    assign context_fire = i_sample_context_valid && o_sample_context_ready;
    assign observation_version_error =
        (context_fire && !context_legal) ||
        (USE_SAMPLE_CONTEXT && i_input_valid && o_input_ready && !input_version_legal) ||
        (USE_SAMPLE_CONTEXT && i_trace_valid && o_trace_ready && !trace_version_legal) ||
        (USE_SAMPLE_CONTEXT && i_surrogate_valid && o_surrogate_ready && !surrogate_version_legal) ||
        (shared_trace_valid && shared_trace_ready && !shared_trace_legal) ||
        (i_feature_trace_wave_valid && o_feature_trace_wave_ready &&
         !feature_version_legal) ||
        (i_logit_wave_valid && o_logit_wave_ready &&
         !logit_version_legal);
    // Context validity is reset, but the tiny version table is only written
    // on admission. Raw epochs are checked before conversion to window epochs.
    always_ff @(posedge i_clk) begin
        if (context_fire && context_legal)
            context_version_q[i_sample_context_target][i_sample_context_sample] <=
                i_sample_context_version;
    end
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            cleanup_q <= 1'b0; error_q <= 1'b0; observation_version_q <= '0;
            capture_pending_q <= 1'b0; learning_owned_q <= 1'b0;
            captured_profile_q <= '0; captured_batch_q <= '0;
            classifier_sample_q <= '0;
            context_valid_q[0] <= '0; context_valid_q[1] <= '0;
        end
        else begin
            if (cfg_fire) begin
                observation_version_q <= i_cfg_sample_version;
                captured_profile_q <= i_cfg_profile_id;
                captured_batch_q <= i_cfg_batch_count;
                classifier_sample_q <= '0;
                context_valid_q[0] <= '0; context_valid_q[1] <= '0;
                capture_pending_q <= CAPTURE_BEFORE_LEARNING;
            end else begin
                if (context_fire && context_legal)
                    context_valid_q[i_sample_context_target][i_sample_context_sample] <= 1'b1;
                if (cl_i_logit_wave_valid && cl_o_logit_wave_ready &&
                    i_logit_wave_sample_last)
                    classifier_sample_q <= classifier_sample_q + 1'b1;
            end
            if (joint_start_fire) begin
                capture_pending_q <= 1'b0;
                learning_owned_q <= 1'b1;
            end
            if (o_commit_fire) learning_owned_q <= 1'b0;
            if (child_error) error_q <= 1'b1;
            if (o_commit_fire) cleanup_q <= 1'b1;
            else if (cleanup_q && !cv_o_active && !cl_o_active) cleanup_q <= 1'b0;
        end
    end
    // Only admission follows the requested profile. An accepted window must
    // finish even when a switch is pending; o_owner_quiescent delays release.
    assign system_admit = !USE_SYSTEM_OWNER || (i_system_parameter_owner &&
        i_system_learning_enable && i_system_profile_aligned &&
        i_system_profile_id == i_cfg_profile_id);
    assign o_cfg_ready = system_admit && (CAPTURE_BEFORE_LEARNING ? !capture_pending_q &&
        !learning_owned_q : joint_ready) && cv_o_cfg_ready && cl_o_cfg_ready &&
        !cleanup_q && !error_q && bank_layout_admission_legal;
    assign cfg_fire = i_cfg_valid && o_cfg_ready;
    assign o_cfg_error = cv_o_cfg_error || cl_o_cfg_error || joint_cfg_error ||
        (i_cfg_valid && bank_layout_admission_error);
    assign o_protocol_error = error_q || child_error || joint_fault;
    assign o_active = joint_hold || cv_o_active || cl_o_active || cleanup_q ||
        capture_pending_q || (CONV_MODULATION_ACTIVE && internal_mod_busy);
    assign o_inference_hold = CAPTURE_BEFORE_LEARNING ?
        joint_hold || learning_owned_q || cleanup_q || error_q : o_active;
    assign o_owner_quiescent = joint_idle && !cleanup_q && !cv_o_active &&
        !cl_o_active && !arb_busy && !error_q &&
        (!CONV_MODULATION_ACTIVE ||
         (!internal_mod_busy && !internal_mod_capture_complete));
    assign o_active_profile_id = ENABLE_SCNN_CONV8_LEARNING ?
        cv_o_active_profile_id : captured_profile_q;
    assign o_active_tensor_id = ENABLE_SCNN_CONV8_LEARNING ?
        cv_o_active_tensor_id : i_cl_cfg_tensor_id;
    assign cv_i_cfg_valid = cfg_fire;
    assign cl_i_cfg_valid = cfg_fire;
    assign cv_i_cfg_timestep = i_cfg_timestep;
    assign cv_i_release = joint_conv_release;
    assign cl_i_release = cleanup_q && cl_o_release_ready;
    assign cl_i_clear_error = 1'b0;
    assign cl_i_learning_enable = 1'b1;
    assign cl_i_cfg_timestep_count = i_cfg_timestep_count;
    assign cl_i_window_commit_ready = joint_classifier_ack;
    assign cv_i_cfg_profile_id = i_cfg_profile_id;
    assign cv_i_cfg_tensor_id = i_cfg_tensor_id;
    assign cv_i_cfg_batch_count = i_cfg_batch_count;
    assign cv_i_cfg_total_input_channels = i_cfg_total_input_channels;
    assign cv_i_cfg_total_output_channels = i_cfg_total_output_channels;
    assign cv_i_cfg_parameter_base = i_cfg_parameter_base;
    assign cv_i_cfg_weight_q16_base = i_cfg_weight_q16_base;
    assign cv_i_cfg_vote_increment = i_cfg_vote_increment;
    assign cv_i_cfg_vote_threshold = i_cfg_vote_threshold;
    assign cv_i_cfg_code_min = i_cfg_code_min;
    assign cv_i_cfg_code_max = i_cfg_code_max;
    assign cv_i_modulation_valid = CONV_MODULATION_ACTIVE ?
        internal_mod_valid : ENABLE_SCNN_CONV8_LEARNING && i_modulation_valid;
    assign o_modulation_ready = ENABLE_SCNN_CONV8_LEARNING &&
        !CONV_MODULATION_ACTIVE && cv_o_modulation_ready;
    assign cv_i_modulation_row = CONV_MODULATION_ACTIVE ?
        SAMPLE_WIDTH'(internal_mod_row) : i_modulation_row;
    assign cv_i_modulation_col = CONV_MODULATION_ACTIVE ?
        SAMPLE_WIDTH'(internal_mod_col) : i_modulation_col;
    assign cv_i_modulation_q8_8 = CONV_MODULATION_ACTIVE ?
        internal_mod_q8_8 : i_modulation_q8_8;
    assign internal_mod_ready = CONV_MODULATION_ACTIVE && cv_o_modulation_ready;
    assign o_capture_complete = ENABLE_SCNN_CONV8_LEARNING ?
        cv_o_capture_complete && cl_o_capture_complete : cl_o_capture_complete;
    assign o_gradient_update_accepted = cv_o_gradient_update_accepted;
    assign o_gradient_zero_skipped = cv_o_gradient_zero_skipped;
    assign o_gradient_valid = cv_o_gradient_valid;
    assign o_gradient_ready = cv_o_gradient_ready;
    assign o_gradient_sign = cv_o_gradient_sign;
    assign o_gradient_local_addr = cv_o_gradient_local_addr;
    assign o_gradient_parameter_addr = cv_o_gradient_parameter_addr;
    assign o_gradient_weight_q16_addr = cv_o_gradient_weight_q16_addr;
    assign o_gradient_weight_q64_addr = cv_o_gradient_weight_q64_addr;
    assign o_gradient_weight_q64_lane = cv_o_gradient_weight_q64_lane;
    assign o_gradient_tile_last = cv_o_gradient_tile_last;
    assign input_source_valid = i_input_valid && !cleanup_q && !error_q &&
        (!USE_SAMPLE_CONTEXT || input_version_legal);
    assign cv_i_input_valid = input_source_valid &&
        (!CONV_MODULATION_ACTIVE || !i_input_target || internal_mod_prev_ready);
    assign o_input_ready = cv_o_input_ready && !cleanup_q && !error_q &&
        (!CONV_MODULATION_ACTIVE || !i_input_target || internal_mod_prev_ready);
    assign internal_mod_prev_valid = CONV_MODULATION_ACTIVE &&
        input_source_valid && i_input_target && cv_o_input_ready;
    assign internal_mod_prev_index = INTERNAL_MOD_PREV_INDEX_WIDTH'(
        int'(i_input_channel) * SPATIAL * SPATIAL + int'(i_input_spatial));
    assign cv_i_input_target = i_input_target;
    assign cv_i_input_sample = i_input_sample;
    assign cv_i_input_timestep = i_input_timestep;
    assign cv_i_input_sample_version = USE_SAMPLE_CONTEXT ?
        observation_version_q : i_input_sample_version;
    assign cv_i_input_channel = i_input_channel;
    assign cv_i_input_spatial = i_input_spatial;
    assign cv_i_input_q8_8 = i_input_q8_8;
    assign shared_trace_selected = USE_CLASSIFIER_STUDENT_TRACE && shared_trace_valid;
    assign trace_source_valid = !cleanup_q && !error_q &&
        (shared_trace_selected ? shared_trace_legal : i_trace_valid &&
         (!USE_CLASSIFIER_STUDENT_TRACE || i_trace_target) &&
         (!USE_SAMPLE_CONTEXT || trace_version_legal));
    assign trace_source_target = shared_trace_selected ? 1'b0 : i_trace_target;
    assign shared_trace_ready = USE_CLASSIFIER_STUDENT_TRACE && cv_o_trace_ready &&
        (!CONV_MODULATION_ACTIVE || internal_mod_student_ready) &&
        !cleanup_q && !error_q;
    assign cv_i_trace_valid = trace_source_valid &&
        (!CONV_MODULATION_ACTIVE ||
         (trace_source_target ? internal_mod_target_ready :
          internal_mod_student_ready));
    assign o_trace_ready = cv_o_trace_ready && !shared_trace_selected &&
        (!CONV_MODULATION_ACTIVE ||
         (i_trace_target ? internal_mod_target_ready :
          internal_mod_student_ready)) &&
        (!USE_CLASSIFIER_STUDENT_TRACE || i_trace_target) && !cleanup_q && !error_q;
    assign cv_i_trace_target = trace_source_target;
    assign cv_i_trace_sample = shared_trace_selected ?
        SAMPLE_WIDTH'(shared_trace_sample) : i_trace_sample;
    assign cv_i_trace_timestep = shared_trace_selected ?
        TIMESTEP_WIDTH'(shared_trace_timestep) : i_trace_timestep;
    assign cv_i_trace_sample_version = USE_SAMPLE_CONTEXT ? observation_version_q :
        shared_trace_selected ? shared_trace_version : i_trace_sample_version;
    assign cv_i_trace_channel = shared_trace_selected ? CHANNEL_INDEX_WIDTH'(
        int'(shared_trace_index) / (POOLED_SPATIAL*POOLED_SPATIAL)) : i_trace_channel;
    assign cv_i_trace_spatial = shared_trace_selected ? $clog2(POOLED_SPATIAL*POOLED_SPATIAL)'(
        int'(shared_trace_index) % (POOLED_SPATIAL*POOLED_SPATIAL)) : i_trace_spatial;
    assign cv_i_trace_q8_8 = shared_trace_selected ? shared_trace_value : i_trace_q8_8;
    assign internal_mod_trace_index = INTERNAL_MOD_TRACE_INDEX_WIDTH'(
        int'(cv_i_trace_channel) * POOLED_SPATIAL * POOLED_SPATIAL +
        int'(cv_i_trace_spatial));
    assign internal_mod_student_valid = CONV_MODULATION_ACTIVE &&
        trace_source_valid && !trace_source_target && cv_o_trace_ready;
    assign internal_mod_target_valid = CONV_MODULATION_ACTIVE &&
        trace_source_valid && trace_source_target && cv_o_trace_ready;
    assign cv_i_surrogate_valid = i_surrogate_valid && !cleanup_q && !error_q &&
        (!USE_SAMPLE_CONTEXT || surrogate_version_legal);
    assign o_surrogate_ready = cv_o_surrogate_ready && !cleanup_q && !error_q;
    assign cv_i_surrogate_target = i_surrogate_target;
    assign cv_i_surrogate_sample = i_surrogate_sample;
    assign cv_i_surrogate_timestep = i_surrogate_timestep;
    assign cv_i_surrogate_sample_version = USE_SAMPLE_CONTEXT ?
        observation_version_q : i_surrogate_sample_version;
    assign cv_i_surrogate_channel = i_surrogate_channel;
    assign cv_i_surrogate_spatial = i_surrogate_spatial;
    assign cv_i_surrogate_q8_8 = i_surrogate_q8_8;
    assign cv_i_cfg_sample_version = i_cfg_sample_version;
    assign o_tiles_completed = cv_o_tiles_completed;
    assign cl_i_cfg_profile_id = i_cfg_profile_id;
    assign cl_i_cfg_tensor_id = i_cl_cfg_tensor_id;
    assign cl_i_cfg_parameter_base = i_cl_cfg_parameter_base;
    assign cl_i_cfg_batch_count = i_cfg_batch_count;
    assign cl_i_cfg_feature_count = i_cl_cfg_feature_count;
    assign cl_i_cfg_class_count = i_cl_cfg_class_count;
    assign cl_i_cfg_trace_leak_valid = i_cl_cfg_trace_leak_valid;
    assign cl_i_cfg_trace_leak_q8_8 = i_cl_cfg_trace_leak_q8_8;
    assign cl_i_cfg_vote_increment = i_cl_cfg_vote_increment;
    assign cl_i_cfg_vote_threshold = i_cl_cfg_vote_threshold;
    assign cl_i_cfg_code_min = i_cl_cfg_code_min;
    assign cl_i_cfg_code_max = i_cl_cfg_code_max;
    assign cl_i_feature_trace_wave_valid = i_feature_trace_wave_valid && !cleanup_q && !error_q &&
        feature_version_legal;
    assign o_feature_trace_wave_ready = cl_o_feature_trace_wave_ready && !cleanup_q && !error_q;
    assign cl_i_feature_trace_wave_mask = i_feature_trace_wave_mask;
    assign cl_i_feature_trace_wave_base = i_feature_trace_wave_base;
    assign cl_i_feature_trace_wave_q8_8_flat = i_feature_trace_wave_q8_8_flat;
    assign cl_i_feature_trace_wave_version = i_feature_trace_wave_version;
    assign cl_i_feature_trace_wave_frame_index = i_feature_trace_wave_frame_index;
    assign cl_i_feature_trace_wave_first = i_feature_trace_wave_first;
    assign cl_i_feature_trace_wave_last = i_feature_trace_wave_last;
    assign cl_i_feature_trace_wave_sample_reset = i_feature_trace_wave_sample_reset;
    assign cl_i_feature_trace_wave_sample_last = i_feature_trace_wave_sample_last;
    assign cl_i_sample_label = i_sample_label;
    assign cl_i_logit_wave_valid = i_logit_wave_valid && !cleanup_q && !error_q &&
        logit_version_legal;
    assign o_logit_wave_ready = cl_o_logit_wave_ready && !cleanup_q && !error_q;
    assign cl_i_logit_wave_mask = i_logit_wave_mask;
    assign cl_i_logit_wave_base = i_logit_wave_base;
    assign cl_i_logit_wave_count = i_logit_wave_count;
    assign cl_i_logit_wave_q8_8_flat = i_logit_wave_q8_8_flat;
    assign cl_i_logit_wave_version = i_logit_wave_version;
    assign cl_i_logit_wave_frame_index = i_logit_wave_frame_index;
    assign cl_i_logit_wave_frame_last = i_logit_wave_frame_last;
    assign cl_i_logit_wave_sample_last = i_logit_wave_sample_last;

    generate
        if (CONV_MODULATION_ACTIVE) begin : g_internal_modulation
            unified_scnn_tp_modulation_generator_q8_8 #(
                .BATCH_SIZE(INTERNAL_MODULATION_BATCH),
                .VECTOR_SIZE(INTERNAL_MOD_TRACE_VALUES),
                .SPATIAL_SIZE(POOLED_SPATIAL * POOLED_SPATIAL),
                .PREV_CHANNELS(OBS_INPUT_CHANNELS),
                .PREV_SPATIAL_SIZE(SPATIAL * SPATIAL),
                .TRACE_STREAM_SPATIAL_MAJOR(
                    INTERNAL_MODULATION_TRACE_SPATIAL_MAJOR)
            ) u_modulation_generator (
                .i_clk(i_clk),
                .i_rst_n(i_rst_n),
                .i_student_trace_valid(internal_mod_student_valid),
                .o_student_trace_ready(internal_mod_student_ready),
                .i_student_trace_sample(INTERNAL_MOD_SAMPLE_WIDTH'(
                    cv_i_trace_sample)),
                .i_student_trace_index(internal_mod_trace_index),
                .i_student_trace_q8_8(cv_i_trace_q8_8),
                .i_target_trace_valid(internal_mod_target_valid),
                .o_target_trace_ready(internal_mod_target_ready),
                .i_target_trace_sample(INTERNAL_MOD_SAMPLE_WIDTH'(
                    cv_i_trace_sample)),
                .i_target_trace_index(internal_mod_trace_index),
                .i_target_trace_q8_8(cv_i_trace_q8_8),
                .i_prev_target_trace_valid(internal_mod_prev_valid),
                .o_prev_target_trace_ready(internal_mod_prev_ready),
                .i_prev_target_trace_sample(INTERNAL_MOD_SAMPLE_WIDTH'(
                    i_input_sample)),
                .i_prev_target_trace_index(internal_mod_prev_index),
                .i_prev_target_trace_q8_8(i_input_q8_8),
                .i_start_valid(internal_mod_capture_complete),
                .o_start_ready(internal_mod_start_ready),
                .o_capture_complete(internal_mod_capture_complete),
                .o_protocol_error(internal_mod_error),
                .o_modulation_valid(internal_mod_valid),
                .i_modulation_ready(internal_mod_ready),
                .o_modulation_row(internal_mod_row),
                .o_modulation_col(internal_mod_col),
                .o_modulation_q8_8(internal_mod_q8_8),
                .o_modulation_last(internal_mod_last),
                .o_done(internal_mod_done),
                .o_busy(internal_mod_busy)
            );
        end else begin : g_external_modulation
            assign internal_mod_student_ready = 1'b0;
            assign internal_mod_target_ready = 1'b0;
            assign internal_mod_prev_ready = 1'b0;
            assign internal_mod_start_ready = 1'b0;
            assign internal_mod_capture_complete = 1'b0;
            assign internal_mod_error = 1'b0;
            assign internal_mod_valid = 1'b0;
            assign internal_mod_row = '0;
            assign internal_mod_col = '0;
            assign internal_mod_q8_8 = '0;
            assign internal_mod_last = 1'b0;
            assign internal_mod_done = 1'b0;
            assign internal_mod_busy = 1'b0;
        end
    endgenerate

    generate
    if (ENABLE_SCNN_CONV8_LEARNING) begin : g_conv8_learning
    unified_scnn_tp_conv8_learning_replay_path #(
        .ALLOW_UNORDERED_SURROGATE_CAPTURE(
            ALLOW_UNORDERED_SURROGATE_CAPTURE),
        .SURROGATE_WAVE_CHANNELS(SURROGATE_WAVE_CHANNELS),
        .SURROGATE_SPATIAL_BLOCK_SIZE(SURROGATE_SPATIAL_BLOCK_SIZE),
        .TRACE_WAVE_CHANNELS(TRACE_WAVE_CHANNELS),
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH),
        .MAX_BATCH(MAX_BATCH),
        .MAX_INPUT_CHANNELS(MAX_INPUT_CHANNELS),
        .MAX_OUTPUT_CHANNELS(MAX_OUTPUT_CHANNELS),
        .SPATIAL(SPATIAL),
        .POOLED_SPATIAL(POOLED_SPATIAL),
        .KERNEL_SIZE(KERNEL_SIZE),
        .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH),
        .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .CHANNEL_INDEX_WIDTH(CHANNEL_INDEX_WIDTH),
        .CHANNEL_COUNT_WIDTH(CHANNEL_COUNT_WIDTH),
        .INPUT_INDEX_WIDTH(INPUT_INDEX_WIDTH),
        .TRACE_INDEX_WIDTH(TRACE_INDEX_WIDTH),
        .SURROGATE_INDEX_WIDTH(SURROGATE_INDEX_WIDTH),
        .OBS_INPUT_CHANNELS(OBS_INPUT_CHANNELS),
        .OBS_OUTPUT_CHANNELS(OBS_OUTPUT_CHANNELS),
        .MAX_TIMESTEPS(MAX_TIMESTEPS),
        .TIMESTEP_WIDTH(TIMESTEP_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH)
    ) u_cv (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(cv_i_cfg_valid),
        .o_cfg_ready(cv_o_cfg_ready),
        .i_cfg_profile_id(cv_i_cfg_profile_id),
        .i_cfg_tensor_id(cv_i_cfg_tensor_id),
        .i_cfg_batch_count(cv_i_cfg_batch_count),
        .i_cfg_total_input_channels(cv_i_cfg_total_input_channels),
        .i_cfg_total_output_channels(cv_i_cfg_total_output_channels),
        .i_cfg_parameter_base(cv_i_cfg_parameter_base),
        .i_cfg_weight_q16_base(cv_i_cfg_weight_q16_base),
        .i_cfg_vote_increment(cv_i_cfg_vote_increment),
        .i_cfg_vote_threshold(cv_i_cfg_vote_threshold),
        .i_cfg_code_min(cv_i_cfg_code_min),
        .i_cfg_code_max(cv_i_cfg_code_max),
        .o_cfg_error(cv_o_cfg_error),
        .o_active(cv_o_active),
        .o_active_profile_id(cv_o_active_profile_id),
        .o_active_tensor_id(cv_o_active_tensor_id),
        .i_release(cv_i_release),
        .o_release_ready(cv_o_release_ready),
        .i_modulation_valid(cv_i_modulation_valid),
        .o_modulation_ready(cv_o_modulation_ready),
        .i_modulation_row(cv_i_modulation_row),
        .i_modulation_col(cv_i_modulation_col),
        .i_modulation_q8_8(cv_i_modulation_q8_8),
        .o_capture_complete(cv_o_capture_complete),
        .o_window_done(cv_o_window_done),
        .o_window_busy(cv_o_window_busy),
        .o_gradient_update_accepted(cv_o_gradient_update_accepted),
        .o_gradient_zero_skipped(cv_o_gradient_zero_skipped),
        .o_gradient_valid(cv_o_gradient_valid),
        .o_gradient_ready(cv_o_gradient_ready),
        .o_gradient_sign(cv_o_gradient_sign),
        .o_gradient_local_addr(cv_o_gradient_local_addr),
        .o_gradient_parameter_addr(cv_o_gradient_parameter_addr),
        .o_gradient_weight_q16_addr(cv_o_gradient_weight_q16_addr),
        .o_gradient_weight_q64_addr(cv_o_gradient_weight_q64_addr),
        .o_gradient_weight_q64_lane(cv_o_gradient_weight_q64_lane),
        .o_gradient_tile_last(cv_o_gradient_tile_last),
        .o_read_valid(cv_o_read_valid),
        .i_read_ready(cv_i_read_ready),
        .o_read_profile_id(cv_o_read_profile_id),
        .o_read_tensor_id(cv_o_read_tensor_id),
        .o_read_parameter_addr(cv_o_read_parameter_addr),
        .o_read_tag(cv_o_read_tag),
        .i_read_response_valid(cv_i_read_response_valid),
        .o_read_response_ready(cv_o_read_response_ready),
        .i_read_response_code(cv_i_read_response_code),
        .i_read_response_residual(cv_i_read_response_residual),
        .i_read_response_error(cv_i_read_response_error),
        .o_result_valid(cv_o_result_valid),
        .i_result_ready(cv_i_result_ready),
        .o_result_profile_id(cv_o_result_profile_id),
        .o_result_tensor_id(cv_o_result_tensor_id),
        .o_result_parameter_addr(cv_o_result_parameter_addr),
        .o_result_tag(cv_o_result_tag),
        .o_result_code(cv_o_result_code),
        .o_result_residual(cv_o_result_residual),
        .o_write_code_enable(cv_o_write_code_enable),
        .o_write_residual_enable(cv_o_write_residual_enable),
        .o_write_code_fire(cv_o_write_code_fire),
        .o_write_residual_fire(cv_o_write_residual_fire),
        .o_dirty(cv_o_dirty),
        .o_saturated(cv_o_saturated),
        .o_protocol_error(cv_o_protocol_error),
        .i_input_valid(cv_i_input_valid),
        .o_input_ready(cv_o_input_ready),
        .i_input_target(cv_i_input_target),
        .i_input_sample(cv_i_input_sample),
        .i_input_timestep(cv_i_input_timestep),
        .i_input_sample_version(cv_i_input_sample_version),
        .i_input_channel(cv_i_input_channel),
        .i_input_spatial(cv_i_input_spatial),
        .i_input_q8_8(cv_i_input_q8_8),
        .i_trace_valid(cv_i_trace_valid),
        .o_trace_ready(cv_o_trace_ready),
        .i_trace_target(cv_i_trace_target),
        .i_trace_sample(cv_i_trace_sample),
        .i_trace_timestep(cv_i_trace_timestep),
        .i_trace_sample_version(cv_i_trace_sample_version),
        .i_trace_channel(cv_i_trace_channel),
        .i_trace_spatial(cv_i_trace_spatial),
        .i_trace_q8_8(cv_i_trace_q8_8),
        .i_surrogate_valid(cv_i_surrogate_valid),
        .o_surrogate_ready(cv_o_surrogate_ready),
        .i_surrogate_target(cv_i_surrogate_target),
        .i_surrogate_sample(cv_i_surrogate_sample),
        .i_surrogate_timestep(cv_i_surrogate_timestep),
        .i_surrogate_sample_version(cv_i_surrogate_sample_version),
        .i_surrogate_channel(cv_i_surrogate_channel),
        .i_surrogate_spatial(cv_i_surrogate_spatial),
        .i_surrogate_q8_8(cv_i_surrogate_q8_8),
        .i_cfg_timestep(cv_i_cfg_timestep),
        .i_cfg_sample_version(cv_i_cfg_sample_version),
        .o_tiles_completed(cv_o_tiles_completed)
    );
    end else begin : g_classifier_only
        assign cv_o_cfg_ready = 1'b1;
        assign cv_o_cfg_error = 1'b0;
        assign cv_o_active = 1'b0;
        assign cv_o_active_profile_id = '0;
        assign cv_o_active_tensor_id = '0;
        assign cv_o_release_ready = 1'b1;
        assign cv_o_modulation_ready = 1'b1;
        assign cv_o_capture_complete = 1'b1;
        assign cv_o_window_done = 1'b0;
        assign cv_o_window_busy = 1'b0;
        assign cv_o_gradient_update_accepted = 1'b0;
        assign cv_o_gradient_zero_skipped = 1'b0;
        assign cv_o_gradient_valid = 1'b0;
        assign cv_o_gradient_ready = 1'b0;
        assign cv_o_gradient_sign = '0;
        assign cv_o_gradient_local_addr = '0;
        assign cv_o_gradient_parameter_addr = '0;
        assign cv_o_gradient_weight_q16_addr = '0;
        assign cv_o_gradient_weight_q64_addr = '0;
        assign cv_o_gradient_weight_q64_lane = '0;
        assign cv_o_gradient_tile_last = 1'b0;
        assign cv_o_read_valid = 1'b0;
        assign cv_o_read_profile_id = '0;
        assign cv_o_read_tensor_id = '0;
        assign cv_o_read_parameter_addr = '0;
        assign cv_o_read_tag = '0;
        assign cv_o_read_response_ready = 1'b0;
        assign cv_o_result_valid = 1'b0;
        assign cv_o_result_profile_id = '0;
        assign cv_o_result_tensor_id = '0;
        assign cv_o_result_parameter_addr = '0;
        assign cv_o_result_tag = '0;
        assign cv_o_result_code = '0;
        assign cv_o_result_residual = '0;
        assign cv_o_write_code_enable = 1'b0;
        assign cv_o_write_residual_enable = 1'b0;
        assign cv_o_write_code_fire = 1'b0;
        assign cv_o_write_residual_fire = 1'b0;
        assign cv_o_dirty = 1'b0;
        assign cv_o_saturated = 1'b0;
        assign cv_o_protocol_error = 1'b0;
        assign cv_o_input_ready = 1'b1;
        assign cv_o_trace_ready = 1'b1;
        assign cv_o_surrogate_ready = 1'b1;
        assign cv_o_tiles_completed = '0;
    end
    endgenerate
    unified_scnn_tp_classifier_final_core_learning_bridge #(
        .ENABLE_COMPUTED_TRACE_CAPTURE(
            USE_CLASSIFIER_STUDENT_TRACE && ENABLE_SCNN_CONV8_LEARNING),
        .USE_EXTERNAL_OUTPUT_LEARNER(USE_EXTERNAL_OUTPUT_LEARNER),
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH),
        .MAX_BATCH(MAX_BATCH),
        .MAX_TIMESTEPS(MAX_TIMESTEPS),
        .CLASSIFIER_CAPTURE_MAX_BATCH(CLASSIFIER_CAPTURE_MAX_BATCH),
        .CLASSIFIER_CAPTURE_MAX_TIMESTEPS(
            CLASSIFIER_CAPTURE_MAX_TIMESTEPS),
        .MAX_FEATURES(MAX_FEATURES),
        .MAX_CLASSES(MAX_CLASSES),
        .LANES(LANES),
        .ENGINES(ENGINES),
        .VERSION_WIDTH(VERSION_WIDTH),
        .FRAME_WIDTH(FRAME_WIDTH),
        .MAX_GRADIENT_ELEMENTS(MAX_GRADIENT_ELEMENTS),
        .BATCH_INDEX_WIDTH(BATCH_INDEX_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .TIMESTEP_INDEX_WIDTH(TIMESTEP_INDEX_WIDTH),
        .TIMESTEP_COUNT_WIDTH(TIMESTEP_COUNT_WIDTH),
        .FEATURE_INDEX_WIDTH(FEATURE_INDEX_WIDTH),
        .FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
        .CLASS_INDEX_WIDTH(CLASS_INDEX_WIDTH),
        .CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH),
        .WAVE_CLASS_INDEX_WIDTH(WAVE_CLASS_INDEX_WIDTH),
        .WAVE_CLASS_COUNT_WIDTH(WAVE_CLASS_COUNT_WIDTH)
    ) u_cl (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_clear_error(cl_i_clear_error),
        .i_cfg_valid(cl_i_cfg_valid),
        .o_cfg_ready(cl_o_cfg_ready),
        .i_cfg_profile_id(cl_i_cfg_profile_id),
        .i_cfg_tensor_id(cl_i_cfg_tensor_id),
        .i_cfg_parameter_base(cl_i_cfg_parameter_base),
        .i_cfg_batch_count(cl_i_cfg_batch_count),
        .i_cfg_timestep_base(TIMESTEP_INDEX_WIDTH'(i_cfg_timestep)),
        .i_cfg_timestep_count(cl_i_cfg_timestep_count),
        .i_cfg_trace_state_reset(i_cfg_timestep == '0),
        .i_cfg_feature_count(cl_i_cfg_feature_count),
        .i_cfg_class_count(cl_i_cfg_class_count),
        .i_cfg_trace_leak_valid(cl_i_cfg_trace_leak_valid),
        .i_cfg_trace_leak_q8_8(cl_i_cfg_trace_leak_q8_8),
        .i_cfg_vote_increment(cl_i_cfg_vote_increment),
        .i_cfg_vote_threshold(cl_i_cfg_vote_threshold),
        .i_cfg_code_min(cl_i_cfg_code_min),
        .i_cfg_code_max(cl_i_cfg_code_max),
        .o_cfg_error(cl_o_cfg_error),
        .o_active(cl_o_active),
        .i_release(cl_i_release),
        .o_release_ready(cl_o_release_ready),
        .i_learning_enable(cl_i_learning_enable),
        .i_feature_trace_wave_valid(cl_i_feature_trace_wave_valid),
        .i_cfg_computed_trace_capture_enable(
            USE_CLASSIFIER_STUDENT_TRACE && ENABLE_SCNN_CONV8_LEARNING),
        .o_computed_trace_valid(shared_trace_valid),
        .i_computed_trace_ready(shared_trace_ready),
        .o_computed_trace_sample_id(shared_trace_sample),
        .o_computed_trace_index(shared_trace_index),
        .o_computed_trace_timestep(shared_trace_timestep),
        .o_computed_trace_frame_index(),
        .o_computed_trace_version(shared_trace_version),
        .o_computed_trace_q8_8(shared_trace_value),
        .o_feature_trace_wave_ready(cl_o_feature_trace_wave_ready),
        .i_feature_trace_wave_mask(cl_i_feature_trace_wave_mask),
        .i_feature_trace_wave_base(cl_i_feature_trace_wave_base),
        .i_feature_trace_wave_q8_8_flat(cl_i_feature_trace_wave_q8_8_flat),
        .i_feature_trace_wave_version(cl_i_feature_trace_wave_version),
        .i_feature_trace_wave_frame_index(cl_i_feature_trace_wave_frame_index),
        .i_feature_trace_wave_first(cl_i_feature_trace_wave_first),
        .i_feature_trace_wave_last(cl_i_feature_trace_wave_last),
        .i_feature_trace_wave_sample_reset(cl_i_feature_trace_wave_sample_reset),
        .i_feature_trace_wave_sample_last(cl_i_feature_trace_wave_sample_last),
        .i_sample_label(cl_i_sample_label),
        .i_logit_wave_valid(cl_i_logit_wave_valid),
        .o_logit_wave_ready(cl_o_logit_wave_ready),
        .i_logit_wave_mask(cl_i_logit_wave_mask),
        .i_logit_wave_base(cl_i_logit_wave_base),
        .i_logit_wave_count(cl_i_logit_wave_count),
        .i_logit_wave_q8_8_flat(cl_i_logit_wave_q8_8_flat),
        .i_logit_wave_version(cl_i_logit_wave_version),
        .i_logit_wave_frame_index(cl_i_logit_wave_frame_index),
        .i_logit_wave_frame_last(cl_i_logit_wave_frame_last),
        .i_logit_wave_sample_last(cl_i_logit_wave_sample_last),
        .o_window_commit_valid(cl_o_window_commit_valid),
        .i_window_commit_ready(cl_i_window_commit_ready),
        .o_window_commit_profile_id(cl_o_window_commit_profile_id),
        .o_window_commit_sequence(cl_o_window_commit_sequence),
        .o_timestep_learning_done(cl_o_timestep_learning_done),
        .o_batch_learning_done(cl_o_batch_learning_done),
        .o_completed_timestep_count(cl_o_completed_timestep_count),
        .o_inference_hold(cl_o_inference_hold),
        .o_window_busy(cl_o_window_busy),
        .o_read_valid(cl_o_read_valid),
        .i_read_ready(cl_i_read_ready),
        .o_read_profile_id(cl_o_read_profile_id),
        .o_read_tensor_id(cl_o_read_tensor_id),
        .o_read_parameter_addr(cl_o_read_parameter_addr),
        .o_read_tag(cl_o_read_tag),
        .i_read_response_valid(cl_i_read_response_valid),
        .o_read_response_ready(cl_o_read_response_ready),
        .i_read_response_code(cl_i_read_response_code),
        .i_read_response_residual(cl_i_read_response_residual),
        .i_read_response_error(cl_i_read_response_error),
        .o_result_valid(cl_o_result_valid),
        .i_result_ready(cl_i_result_ready),
        .o_result_profile_id(cl_o_result_profile_id),
        .o_result_tensor_id(cl_o_result_tensor_id),
        .o_result_parameter_addr(cl_o_result_parameter_addr),
        .o_result_tag(cl_o_result_tag),
        .o_result_code(cl_o_result_code),
        .o_result_residual(cl_o_result_residual),
        .o_write_code_enable(cl_o_write_code_enable),
        .o_write_residual_enable(cl_o_write_residual_enable),
        .o_write_code_fire(cl_o_write_code_fire),
        .o_write_residual_fire(cl_o_write_residual_fire),
        .o_dirty(cl_o_dirty),
        .o_saturated(cl_o_saturated),
        .o_capture_complete(cl_o_capture_complete),
        .o_observation_error(cl_o_observation_error),
        .o_context_error(cl_o_context_error),
        .o_gradient_overflow(cl_o_gradient_overflow),
        .o_parameter_error(cl_o_parameter_error),
        .o_protocol_error(cl_o_protocol_error),
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
        .i_external_internal_error(i_external_internal_error)
    );
    unified_tp_two_client_parameter_rmw_arbiter #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH)
    ) u_rmw (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_c0_read_valid(cv_o_read_valid),
        .o_c0_read_ready(cv_i_read_ready),
        .i_c0_read_profile_id(cv_o_read_profile_id),
        .i_c0_read_tensor_id(cv_o_read_tensor_id),
        .i_c0_read_parameter_addr(cv_o_read_parameter_addr),
        .i_c0_read_tag(cv_o_read_tag),
        .o_c0_read_response_valid(cv_i_read_response_valid),
        .i_c0_read_response_ready(cv_o_read_response_ready),
        .o_c0_read_response_code(cv_i_read_response_code),
        .o_c0_read_response_residual(cv_i_read_response_residual),
        .o_c0_read_response_error(cv_i_read_response_error),
        .i_c0_result_valid(cv_o_result_valid),
        .o_c0_result_ready(cv_i_result_ready),
        .i_c0_result_profile_id(cv_o_result_profile_id),
        .i_c0_result_tensor_id(cv_o_result_tensor_id),
        .i_c0_result_parameter_addr(cv_o_result_parameter_addr),
        .i_c0_result_tag(cv_o_result_tag),
        .i_c0_result_code(cv_o_result_code),
        .i_c0_result_residual(cv_o_result_residual),
        .i_c0_write_code_enable(cv_o_write_code_enable),
        .i_c0_write_residual_enable(cv_o_write_residual_enable),
        .i_c0_write_code_fire(cv_o_write_code_fire),
        .i_c0_write_residual_fire(cv_o_write_residual_fire),
        .i_c0_dirty(cv_o_dirty),
        .i_c0_saturated(cv_o_saturated),
        .i_c1_read_valid(cl_o_read_valid),
        .o_c1_read_ready(cl_i_read_ready),
        .i_c1_read_profile_id(cl_o_read_profile_id),
        .i_c1_read_tensor_id(cl_o_read_tensor_id),
        .i_c1_read_parameter_addr(cl_o_read_parameter_addr),
        .i_c1_read_tag(cl_o_read_tag),
        .o_c1_read_response_valid(cl_i_read_response_valid),
        .i_c1_read_response_ready(cl_o_read_response_ready),
        .o_c1_read_response_code(cl_i_read_response_code),
        .o_c1_read_response_residual(cl_i_read_response_residual),
        .o_c1_read_response_error(cl_i_read_response_error),
        .i_c1_result_valid(cl_o_result_valid),
        .o_c1_result_ready(cl_i_result_ready),
        .i_c1_result_profile_id(cl_o_result_profile_id),
        .i_c1_result_tensor_id(cl_o_result_tensor_id),
        .i_c1_result_parameter_addr(cl_o_result_parameter_addr),
        .i_c1_result_tag(cl_o_result_tag),
        .i_c1_result_code(cl_o_result_code),
        .i_c1_result_residual(cl_o_result_residual),
        .i_c1_write_code_enable(cl_o_write_code_enable),
        .i_c1_write_residual_enable(cl_o_write_residual_enable),
        .i_c1_write_code_fire(cl_o_write_code_fire),
        .i_c1_write_residual_fire(cl_o_write_residual_fire),
        .i_c1_dirty(cl_o_dirty),
        .i_c1_saturated(cl_o_saturated),
        .o_read_valid(arb_read_valid),
        .i_read_ready(arb_read_ready),
        .o_read_profile_id(o_read_profile_id),
        .o_read_tensor_id(o_read_tensor_id),
        .o_read_parameter_addr(o_read_parameter_addr),
        .o_read_tag(o_read_tag),
        .i_read_response_valid(i_read_response_valid),
        .o_read_response_ready(o_read_response_ready),
        .i_read_response_code(i_read_response_code),
        .i_read_response_residual(i_read_response_residual),
        .i_read_response_error(i_read_response_error),
        .o_result_valid(o_result_valid),
        .i_result_ready(i_result_ready),
        .o_result_profile_id(o_result_profile_id),
        .o_result_tensor_id(o_result_tensor_id),
        .o_result_parameter_addr(o_result_parameter_addr),
        .o_result_tag(o_result_tag),
        .o_result_code(o_result_code),
        .o_result_residual(o_result_residual),
        .o_write_code_enable(o_write_code_enable),
        .o_write_residual_enable(o_write_residual_enable),
        .o_write_code_fire(o_write_code_fire),
        .o_write_residual_fire(o_write_residual_fire),
        .o_dirty(o_dirty),
        .o_saturated(o_saturated),
        .o_busy(arb_busy),
        .o_protocol_error(arb_error)
    );
    // A live capture must finish before any learner can read or change the
    // parameter image. The external boundary also waits for the last forward
    // transaction to drain before granting learning ownership.
    assign o_read_valid = arb_read_valid && (!CAPTURE_BEFORE_LEARNING || learning_owned_q);
    assign arb_read_ready = i_read_ready && (!CAPTURE_BEFORE_LEARNING || learning_owned_q);
    assign join_start_valid = CAPTURE_BEFORE_LEARNING ?
        capture_pending_q && cv_o_capture_complete && cl_o_capture_complete && !error_q :
        i_cfg_valid && system_admit && cv_o_cfg_ready && cl_o_cfg_ready &&
            !cleanup_q && !error_q && bank_layout_legal;
    assign o_abort_request = join_abort_request || error_q;
    unified_scnn_tp_joint_commit u_join (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_start_valid(join_start_valid),
        .o_start_ready(joint_ready),
        .i_selected_banks(ENABLE_SCNN_CONV8_LEARNING ? 2'b11 : 2'b10),
        .i_profile_id(CAPTURE_BEFORE_LEARNING ? captured_profile_q : i_cfg_profile_id),
        .i_sequence(cl_o_window_commit_sequence),
        .o_learning_start_valid(o_learning_start_valid),
        .i_learning_start_ready(i_learning_start_ready), .o_start_fire(joint_start_fire),
        .i_conv8_done(cv_o_window_done), .i_conv8_release_ready(cv_o_release_ready),
        .o_conv8_release(joint_conv_release),
        .i_classifier_commit_valid(cl_o_window_commit_valid),
        .i_classifier_profile_id(cl_o_window_commit_profile_id),
        .i_classifier_sequence(cl_o_window_commit_sequence),
        .o_classifier_commit_ready(joint_classifier_ack),
        .i_rmw_quiescent(!arb_busy && !o_read_valid && !o_result_valid && !i_read_response_valid),
        .i_abort(1'b0), .i_child_error(error_q),
        .o_abort_request(join_abort_request),
        .i_children_quiescent(!cv_o_active && !cl_o_active),
        .o_abort_complete(o_abort_complete), .i_recovery_ack(1'b0),
        .o_commit_valid(o_commit_valid), .i_commit_ready(i_commit_ready),
        .o_commit_profile_id(o_commit_profile_id), .o_commit_sequence(o_commit_sequence),
        .o_commit_fire(o_commit_fire), .o_inference_hold(joint_hold),
        .o_owner_quiescent(joint_idle), .o_fault(joint_fault), .o_config_error(joint_cfg_error)
    );
    initial begin
        if (MAX_TIMESTEPS < 1 || TIMESTEP_COUNT_WIDTH < 1)
            $error("Selected-layer timestep parameters are invalid");
        if (PARAM_ADDR_WIDTH > 32)
            $error("Selected-layer parameter addresses must be at most 32 bits");
        if (USE_SAMPLE_CONTEXT && !CAPTURE_BEFORE_LEARNING)
            $error("Per-sample context requires the live-capture phase");
    end
endmodule
`default_nettype wire
