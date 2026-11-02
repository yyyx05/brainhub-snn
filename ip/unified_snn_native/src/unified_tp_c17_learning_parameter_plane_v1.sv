`timescale 1ns/1ps
`default_nettype none

// Learning and persistent-parameter plane paired with the native C17 H40
// system.  C17 remains the only inference array and the only SCNN learner.
// This module adds only the FC W2 and SRNN M2 clients, then routes all three
// topology clients to one external scalar code/residual store.
module unified_tp_c17_learning_parameter_plane_v1 #(
    parameter bit ENABLE_ONLINE_LEARNING = 1'b1,
    parameter bit ENABLE_FC_CLIENT = 1'b1,
    parameter bit ENABLE_SRNN_CLIENT = 1'b1,
    parameter bit ENABLE_SCNN_CLIENT = 1'b1,
    parameter bit USE_SHARED_SRNN_H40_FORWARD = 1'b0,
    parameter bit SHARE_FC_SCNN_OUTPUT_LEARNER = 1'b0,
    parameter bit ENABLE_SRNN_WH1_LEARNING = 1'b1,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int FC_MAX_HIDDEN = 480,
    parameter int FC_MAX_OUTPUTS = 16,
    parameter int MAX_BATCH = 10,
    parameter int MAX_FEATURES = 2048,
    parameter int MAX_CLASSES = 11,
    parameter int SRNN_BATCH_MAX = 64,
    parameter int SRNN_TIMESTEP_MAX = 128,
    parameter int SRNN_HIDDEN_MAX = 256,
    parameter int SRNN_BACKBONE_MAX = 450,
    parameter int SRNN_CLASS_MAX = 20
) (
    input  logic i_clk,
    input  logic i_rst_n,
    input  logic i_clear_error,
    input  logic [2:0] i_parameter_owner,
    input  logic [2:0] i_topology_learning_enable,

    input  logic i_fc_cfg_valid,
    output logic o_fc_cfg_ready,
    input  logic [7:0] i_fc_profile_id,
    input  logic [7:0] i_fc_w2_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0] i_fc_w2_parameter_base,
    input  logic [4:0] i_fc_batch_count,
    input  logic [8:0] i_fc_hidden_count,
    input  logic [4:0] i_fc_output_count,
    input  logic [15:0] i_fc_vote_increment,
    input  logic [15:0] i_fc_vote_threshold,
    input  logic signed [CODE_WIDTH-1:0] i_fc_code_min,
    input  logic signed [CODE_WIDTH-1:0] i_fc_code_max,
    output logic o_fc_cfg_error,
    input  logic i_fc_release,
    output logic o_fc_release_ready,
    input  logic i_fc_student_trace_valid,
    output logic o_fc_student_trace_ready,
    input  logic [5:0] i_fc_student_trace_sample,
    input  logic [9:0] i_fc_student_trace_hidden,
    input  logic [16:0] i_fc_student_trace_q016,
    input  logic i_fc_logit_row_valid,
    output logic o_fc_logit_row_ready,
    input  logic [5:0] i_fc_logit_sample,
    input  logic [3:0] i_fc_logit_label,
    input  logic i_fc_logit_valid,
    output logic o_fc_logit_ready,
    input  logic signed [15:0] i_fc_logit_q6,
    input  logic i_fc_observation_done,
    output logic o_fc_window_commit_valid,
    input  logic i_fc_window_commit_ready,
    output logic [31:0] o_fc_window_commit_sequence,
    output logic o_fc_window_busy,
    output logic o_fc_error,
    input  logic i_fc_parameter_load_valid,
    output logic o_fc_parameter_load_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0] i_fc_parameter_load_addr,
    input  logic signed [CODE_WIDTH-1:0] i_fc_parameter_load_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]
        i_fc_parameter_load_residual,
    input  logic i_fc_forward_read_valid,
    output logic o_fc_forward_read_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0] i_fc_forward_read_addr,
    input  logic [TAG_WIDTH-1:0] i_fc_forward_read_tag,
    output logic o_fc_forward_response_valid,
    input  logic i_fc_forward_response_ready,
    output logic [TAG_WIDTH-1:0] o_fc_forward_response_tag,
    output logic signed [CODE_WIDTH-1:0] o_fc_forward_response_code,
    output logic o_fc_forward_response_error,

    input  logic i_srnn_cfg_valid,
    output logic o_srnn_cfg_ready,
    input  logic [7:0] i_srnn_profile_id,
    input  logic [7:0] i_srnn_wh1_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0] i_srnn_wh1_base,
    input  logic [15:0] i_srnn_wh1_increment,
    input  logic [15:0] i_srnn_wh1_threshold,
    input  logic [7:0] i_srnn_wout_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0] i_srnn_wout_base,
    input  logic [15:0] i_srnn_wout_increment,
    input  logic [15:0] i_srnn_wout_threshold,
    input  logic signed [CODE_WIDTH-1:0] i_srnn_code_min,
    input  logic signed [CODE_WIDTH-1:0] i_srnn_code_max,
    input  logic [7:0] i_srnn_batch_count,
    input  logic [8:0] i_srnn_hidden_count,
    input  logic [9:0] i_srnn_backbone_count,
    input  logic [4:0] i_srnn_class_count,
    input  logic [PARAM_ADDR_WIDTH-1:0] i_srnn_wout_element_count,
    output logic o_srnn_cfg_error,
    input  logic i_srnn_parameter_load_valid,
    output logic o_srnn_parameter_load_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0] i_srnn_parameter_load_addr,
    input  logic signed [CODE_WIDTH-1:0] i_srnn_parameter_load_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]
        i_srnn_parameter_load_residual,
    input  logic i_srnn_timestep_start,
    output logic o_srnn_timestep_start_ready,
    input  logic [$clog2(SRNN_TIMESTEP_MAX)-1:0] i_srnn_timestep_id,
    input  logic [15:0] i_srnn_sequence_version,
    input  logic i_srnn_learning_enable,
    input  logic i_srnn_abort_timestep,
    output logic o_srnn_abort_ready,
    input  logic i_srnn_backbone_valid,
    output logic o_srnn_backbone_ready,
    input  logic i_srnn_backbone_path,
    input  logic [5:0] i_srnn_backbone_sample_id,
    input  logic [9:0] i_srnn_backbone_id,
    input  logic signed [31:0] i_srnn_backbone_q14,
    input  logic i_srnn_backbone_last,
    input  logic i_srnn_backbone_done,
    input  logic i_srnn_backbone_error,
    output logic o_srnn_target_trace_start,
    input  logic i_srnn_target_trace_start_ready,
    output logic [$clog2(SRNN_TIMESTEP_MAX)-1:0]
        o_srnn_target_trace_timestep_id,
    input  logic i_srnn_target_trace_valid,
    output logic o_srnn_target_trace_ready,
    input  logic [7:0] i_srnn_target_trace_sample_id,
    input  logic [9:0] i_srnn_target_trace_backbone_id,
    input  logic signed [31:0] i_srnn_target_trace_q14,
    input  logic i_srnn_target_trace_last,
    input  logic i_srnn_target_trace_done,
    input  logic i_srnn_target_trace_error,
    output logic o_srnn_label_replay_start,
    input  logic i_srnn_label_replay_start_ready,
    input  logic i_srnn_label_valid,
    output logic o_srnn_label_ready,
    input  logic [7:0] i_srnn_label_sample_id,
    input  logic [4:0] i_srnn_label_class_id,
    input  logic i_srnn_label_last,
    input  logic i_srnn_restore_complete,
    output logic o_srnn_restore_ready,
    output logic o_srnn_recovery_required,
    output logic o_srnn_learning_commit,
    output logic o_srnn_detach,
    output logic o_srnn_logit_valid,
    input  logic i_srnn_logit_ready,
    output logic [7:0] o_srnn_logit_sample_id,
    output logic [4:0] o_srnn_logit_class_id,
    output logic signed [63:0] o_srnn_logit_q28,
    output logic o_srnn_logit_last,
    output logic o_srnn_timestep_done,
    output logic o_srnn_timestep_aborted,
    output logic o_srnn_busy,
    output logic o_srnn_error,
    output logic [PARAM_ADDR_WIDTH:0] o_srnn_zero_gradient_count,
    output logic [PARAM_ADDR_WIDTH:0] o_srnn_update_count,

    output logic o_srnn_h40_cfg_valid,
    input  logic i_srnn_h40_cfg_ready,
    input  logic i_srnn_h40_cfg_error,
    output logic [7:0] o_srnn_h40_cfg_batch_count,
    output logic [9:0] o_srnn_h40_cfg_backbone_count,
    output logic [8:0] o_srnn_h40_cfg_hidden_count,
    output logic [4:0] o_srnn_h40_cfg_class_count,
    output logic [PARAM_ADDR_WIDTH-1:0]
        o_srnn_h40_cfg_wh1_parameter_base,
    output logic [PARAM_ADDR_WIDTH-1:0]
        o_srnn_h40_cfg_wout_parameter_base,
    output logic o_srnn_h40_prepare_valid,
    input  logic i_srnn_h40_prepare_ready,
    output logic [15:0] o_srnn_h40_prepare_version,
    input  logic i_srnn_h40_prepare_done,
    input  logic i_srnn_h40_cache_valid,
    output logic o_srnn_h40_cache_invalidate,
    input  logic i_srnn_h40_forward_read_valid,
    output logic o_srnn_h40_forward_read_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0] i_srnn_h40_forward_read_addr,
    input  logic [TAG_WIDTH-1:0] i_srnn_h40_forward_read_tag,
    output logic o_srnn_h40_forward_response_valid,
    input  logic i_srnn_h40_forward_response_ready,
    output logic [TAG_WIDTH-1:0] o_srnn_h40_forward_response_tag,
    output logic signed [CODE_WIDTH-1:0]
        o_srnn_h40_forward_response_code,
    output logic o_srnn_h40_forward_response_error,
    output logic o_srnn_h40_timestep_start,
    input  logic i_srnn_h40_timestep_start_ready,
    output logic [$clog2(SRNN_TIMESTEP_MAX)-1:0]
        o_srnn_h40_timestep_id,
    output logic [15:0] o_srnn_h40_sequence_version,
    output logic o_srnn_h40_include_target_path,
    output logic o_srnn_h40_observation_valid,
    input  logic i_srnn_h40_observation_ready,
    output logic o_srnn_h40_observation_path,
    output logic [7:0] o_srnn_h40_observation_sample_id,
    output logic [9:0] o_srnn_h40_observation_backbone_id,
    output logic signed [31:0] o_srnn_h40_observation_q14,
    output logic o_srnn_h40_observation_last,
    output logic o_srnn_h40_observation_done,
    input  logic i_srnn_h40_hidden_valid,
    output logic o_srnn_h40_hidden_ready,
    input  logic i_srnn_h40_hidden_path,
    input  logic [7:0] i_srnn_h40_hidden_sample_id,
    input  logic [8:0] i_srnn_h40_hidden_id,
    input  logic signed [31:0] i_srnn_h40_hidden_membrane_pre_q14,
    input  logic signed [31:0] i_srnn_h40_hidden_membrane_next_q14,
    input  logic i_srnn_h40_hidden_spike,
    input  logic signed [31:0] i_srnn_h40_hidden_trace_q14,
    input  logic signed [31:0] i_srnn_h40_hidden_output_feature_q14,
    input  logic i_srnn_h40_hidden_last,
    input  logic i_srnn_h40_logit_valid,
    output logic o_srnn_h40_logit_ready,
    input  logic [7:0] i_srnn_h40_logit_sample_id,
    input  logic [4:0] i_srnn_h40_logit_class_id,
    input  logic signed [63:0] i_srnn_h40_logit_q28,
    input  logic i_srnn_h40_logit_last,
    input  logic [$clog2(SRNN_TIMESTEP_MAX)-1:0]
        i_srnn_h40_timestep_id,
    input  logic i_srnn_h40_timestep_done,
    input  logic i_srnn_h40_busy,
    input  logic i_srnn_h40_sticky_error,
    input  logic i_srnn_h40_protocol_error,
    input  logic i_srnn_h40_overflow,

    input  logic i_scnn_store_load_valid,
    output logic o_scnn_store_load_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0] i_scnn_store_load_addr,
    input  logic signed [CODE_WIDTH-1:0] i_scnn_store_load_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]
        i_scnn_store_load_residual,
    input  logic i_scnn_store_learning_read_valid,
    output logic o_scnn_store_learning_read_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0] i_scnn_store_learning_read_addr,
    input  logic [TAG_WIDTH-1:0] i_scnn_store_learning_read_tag,
    output logic o_scnn_store_learning_response_valid,
    input  logic i_scnn_store_learning_response_ready,
    output logic [TAG_WIDTH-1:0] o_scnn_store_learning_response_tag,
    output logic signed [CODE_WIDTH-1:0]
        o_scnn_store_learning_response_code,
    output logic signed [RESIDUAL_WIDTH-1:0]
        o_scnn_store_learning_response_residual,
    output logic o_scnn_store_learning_response_error,
    input  logic i_scnn_store_learning_write_valid,
    output logic o_scnn_store_learning_write_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0] i_scnn_store_learning_write_addr,
    input  logic signed [CODE_WIDTH-1:0] i_scnn_store_learning_write_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]
        i_scnn_store_learning_write_residual,
    input  logic i_scnn_store_learning_write_code_enable,
    input  logic i_scnn_store_learning_write_residual_enable,
    input  logic i_scnn_store_forward_read_valid,
    output logic o_scnn_store_forward_read_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0] i_scnn_store_forward_read_addr,
    input  logic [TAG_WIDTH-1:0] i_scnn_store_forward_read_tag,
    output logic o_scnn_store_forward_response_valid,
    input  logic i_scnn_store_forward_response_ready,
    output logic [TAG_WIDTH-1:0] o_scnn_store_forward_response_tag,
    output logic signed [CODE_WIDTH-1:0]
        o_scnn_store_forward_response_code,
    output logic o_scnn_store_forward_response_error,

    input  logic i_scnn_external_cfg_valid,
    output logic o_scnn_external_cfg_ready,
    input  logic [7:0] i_scnn_external_cfg_profile_id,
    input  logic [7:0] i_scnn_external_cfg_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]
        i_scnn_external_cfg_parameter_base,
    input  logic [$clog2(MAX_BATCH+1)-1:0]
        i_scnn_external_cfg_batch_count,
    input  logic [$clog2(MAX_FEATURES+1)-1:0]
        i_scnn_external_cfg_source_count,
    input  logic [$clog2(MAX_CLASSES+1)-1:0]
        i_scnn_external_cfg_output_count,
    input  logic [15:0] i_scnn_external_cfg_vote_increment,
    input  logic [15:0] i_scnn_external_cfg_vote_threshold,
    input  logic signed [CODE_WIDTH-1:0]
        i_scnn_external_cfg_code_min,
    input  logic signed [CODE_WIDTH-1:0]
        i_scnn_external_cfg_code_max,
    output logic o_scnn_external_cfg_error,
    output logic o_scnn_external_active,
    input  logic i_scnn_external_release,
    output logic o_scnn_external_release_ready,
    input  logic i_scnn_external_source_trace_valid,
    output logic o_scnn_external_source_trace_ready,
    input  logic [(MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH))-1:0]
        i_scnn_external_source_trace_sample_id,
    input  logic [(MAX_FEATURES <= 2 ? 1 : $clog2(MAX_FEATURES))-1:0]
        i_scnn_external_source_trace_index,
    input  logic [15:0] i_scnn_external_source_trace_q016,
    output logic o_scnn_external_source_trace_error,
    input  logic i_scnn_external_logit_row_valid,
    output logic o_scnn_external_logit_row_ready,
    input  logic [(MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH))-1:0]
        i_scnn_external_logit_sample_id,
    input  logic [(MAX_CLASSES <= 2 ? 1 : $clog2(MAX_CLASSES))-1:0]
        i_scnn_external_logit_label,
    input  logic i_scnn_external_logit_valid,
    output logic o_scnn_external_logit_ready,
    input  logic signed [15:0] i_scnn_external_logit_q6,
    output logic o_scnn_external_logit_error,
    output logic o_scnn_external_logit_row_done,
    input  logic i_scnn_external_observation_done,
    output logic o_scnn_external_observation_done_ready,
    output logic o_scnn_external_window_done,
    output logic o_scnn_external_window_busy,
    output logic o_scnn_external_context_error,
    output logic o_scnn_external_gradient_overflow,
    output logic o_scnn_external_update_config_error,
    output logic o_scnn_external_request_error,
    output logic o_scnn_external_store_error,
    output logic o_scnn_external_internal_error,

    output logic o_store_load_valid,
    input  logic i_store_load_ready,
    output logic [PARAM_ADDR_WIDTH-1:0] o_store_load_addr,
    output logic signed [CODE_WIDTH-1:0] o_store_load_code,
    output logic signed [RESIDUAL_WIDTH-1:0] o_store_load_residual,
    output logic o_store_learning_read_valid,
    input  logic i_store_learning_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0] o_store_learning_read_addr,
    output logic [TAG_WIDTH-1:0] o_store_learning_read_tag,
    input  logic i_store_learning_response_valid,
    output logic o_store_learning_response_ready,
    input  logic [TAG_WIDTH-1:0] i_store_learning_response_tag,
    input  logic signed [CODE_WIDTH-1:0] i_store_learning_response_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]
        i_store_learning_response_residual,
    input  logic i_store_learning_response_error,
    output logic o_store_learning_write_valid,
    input  logic i_store_learning_write_ready,
    output logic [PARAM_ADDR_WIDTH-1:0] o_store_learning_write_addr,
    output logic signed [CODE_WIDTH-1:0] o_store_learning_write_code,
    output logic signed [RESIDUAL_WIDTH-1:0] o_store_learning_write_residual,
    output logic o_store_learning_write_code_enable,
    output logic o_store_learning_write_residual_enable,
    output logic o_store_forward_read_valid,
    input  logic i_store_forward_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0] o_store_forward_read_addr,
    output logic [TAG_WIDTH-1:0] o_store_forward_read_tag,
    input  logic i_store_forward_response_valid,
    output logic o_store_forward_response_ready,
    input  logic [TAG_WIDTH-1:0] i_store_forward_response_tag,
    input  logic signed [CODE_WIDTH-1:0] i_store_forward_response_code,
    input  logic i_store_forward_response_error,
    input  logic i_store_quiescent,
    input  logic i_store_protocol_error,
    output logic o_backend_quiescent,
    output logic o_backend_error,
    output logic o_owner_encoding_error,
    output logic o_router_protocol_error
);
    localparam int SHARED_MAX_BATCH = (MAX_BATCH > 16) ? MAX_BATCH : 16;
    localparam int SHARED_MAX_SOURCES =
        (MAX_FEATURES > FC_MAX_HIDDEN) ? MAX_FEATURES : FC_MAX_HIDDEN;
    localparam int SHARED_MAX_OUTPUTS =
        (MAX_CLASSES > FC_MAX_OUTPUTS) ? MAX_CLASSES : FC_MAX_OUTPUTS;
    localparam int SHARED_MAX_GRADIENT_ELEMENTS =
        (FC_MAX_HIDDEN * FC_MAX_OUTPUTS >
         MAX_FEATURES * MAX_CLASSES) ?
        FC_MAX_HIDDEN * FC_MAX_OUTPUTS : MAX_FEATURES * MAX_CLASSES;
    localparam int SHARED_BATCH_INDEX_WIDTH =
        (SHARED_MAX_BATCH <= 2) ? 1 : $clog2(SHARED_MAX_BATCH);
    localparam int SHARED_BATCH_COUNT_WIDTH =
        $clog2(SHARED_MAX_BATCH + 1);
    localparam int SHARED_SOURCE_INDEX_WIDTH =
        (SHARED_MAX_SOURCES <= 2) ? 1 : $clog2(SHARED_MAX_SOURCES);
    localparam int SHARED_SOURCE_COUNT_WIDTH =
        $clog2(SHARED_MAX_SOURCES + 1);
    localparam int SHARED_OUTPUT_INDEX_WIDTH =
        (SHARED_MAX_OUTPUTS <= 2) ? 1 : $clog2(SHARED_MAX_OUTPUTS);
    localparam int SHARED_OUTPUT_COUNT_WIDTH =
        $clog2(SHARED_MAX_OUTPUTS + 1);

    logic fc_selected;
    logic fc_learning_selected;
    logic srnn_selected;
    logic scnn_selected;
    logic scnn_learning_selected;
    logic fc_cfg_ready_internal;
    logic fc_cfg_error_internal;
    logic fc_release_ready_internal;
    logic fc_student_ready_internal;
    logic fc_logit_row_ready_internal;
    logic fc_logit_ready_internal;
    logic fc_observation_done_ready;
    logic fc_observation_done_pending_q;
    logic fc_done_protocol_error_q;
    logic fc_observation_error;
    logic fc_context_error;
    logic fc_arithmetic_overflow;
    logic fc_parameter_error;
    logic fc_protocol_error;
    logic fc_engine_error;
    logic fc_read_valid;
    logic fc_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] fc_read_addr;
    logic [TAG_WIDTH-1:0] fc_read_tag;
    logic fc_read_response_valid;
    logic fc_read_response_ready;
    logic fc_result_valid;
    logic fc_result_ready;
    logic [PARAM_ADDR_WIDTH-1:0] fc_result_addr;
    logic signed [CODE_WIDTH-1:0] fc_result_code;
    logic signed [RESIDUAL_WIDTH-1:0] fc_result_residual;
    logic fc_write_code_enable;
    logic fc_write_residual_enable;
    logic fc_window_commit_valid_internal;
    logic [31:0] fc_window_commit_sequence_internal;
    logic fc_window_busy_internal;

    logic [1:0] shared_output_owner;
    logic shared_output_busy;
    logic shared_output_protocol_error;
    logic shared_output_read_valid;
    logic shared_output_read_ready;
    logic [7:0] shared_output_read_profile_id;
    logic [7:0] shared_output_read_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] shared_output_read_addr;
    logic [TAG_WIDTH-1:0] shared_output_read_tag;
    logic shared_output_response_valid;
    logic shared_output_response_ready;
    logic signed [CODE_WIDTH-1:0] shared_output_response_code;
    logic signed [RESIDUAL_WIDTH-1:0] shared_output_response_residual;
    logic shared_output_response_error;
    logic shared_output_result_valid;
    logic shared_output_result_ready;
    logic [7:0] shared_output_result_profile_id;
    logic [7:0] shared_output_result_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] shared_output_result_addr;
    logic [TAG_WIDTH-1:0] shared_output_result_tag;
    logic signed [CODE_WIDTH-1:0] shared_output_result_code;
    logic signed [RESIDUAL_WIDTH-1:0] shared_output_result_residual;
    logic shared_output_write_code_enable;
    logic shared_output_write_residual_enable;
    logic shared_output_write_code_fire;
    logic shared_output_write_residual_fire;
    logic shared_output_dirty;
    logic shared_output_saturated;
    logic shared_output_update_config_error;
    logic shared_output_request_error;
    logic shared_output_store_error;
    logic shared_output_internal_error;
    logic shared_fc_window_done;
    logic shared_fc_window_busy;
    logic shared_fc_commit_valid_q;
    logic [31:0] shared_fc_commit_sequence_q;
    logic shared_fc_commit_protocol_error_q;

    logic scnn_merged_read_valid;
    logic scnn_merged_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] scnn_merged_read_addr;
    logic [TAG_WIDTH-1:0] scnn_merged_read_tag;
    logic scnn_merged_response_valid;
    logic scnn_merged_response_ready;
    logic scnn_merged_write_valid;
    logic scnn_merged_write_ready;
    logic [PARAM_ADDR_WIDTH-1:0] scnn_merged_write_addr;
    logic signed [CODE_WIDTH-1:0] scnn_merged_write_code;
    logic signed [RESIDUAL_WIDTH-1:0] scnn_merged_write_residual;
    logic scnn_merged_write_code_enable;
    logic scnn_merged_write_residual_enable;
    logic scnn_merge_busy;
    logic scnn_merge_protocol_error;
    logic shared_scnn_read_ready;
    logic shared_scnn_response_valid;
    logic signed [CODE_WIDTH-1:0] shared_scnn_response_code;
    logic signed [RESIDUAL_WIDTH-1:0] shared_scnn_response_residual;
    logic shared_scnn_response_error;
    logic shared_scnn_result_ready;
    logic scnn_external_cfg_ready_internal;
    logic scnn_external_release_ready_internal;
    logic scnn_external_source_trace_ready_internal;
    logic scnn_external_logit_row_ready_internal;
    logic scnn_external_logit_ready_internal;
    logic scnn_external_observation_done_ready_internal;
    logic backend_quiescent_internal;
    logic backend_error_internal;

    logic srnn_store_load_valid;
    logic srnn_store_load_ready;
    logic [PARAM_ADDR_WIDTH-1:0] srnn_store_load_addr;
    logic signed [CODE_WIDTH-1:0] srnn_store_load_code;
    logic signed [RESIDUAL_WIDTH-1:0] srnn_store_load_residual;
    logic srnn_store_read_valid;
    logic srnn_store_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] srnn_store_read_addr;
    logic [TAG_WIDTH-1:0] srnn_store_read_tag;
    logic srnn_store_response_valid;
    logic srnn_store_response_ready;
    logic srnn_store_write_valid;
    logic srnn_store_write_ready;
    logic [PARAM_ADDR_WIDTH-1:0] srnn_store_write_addr;
    logic signed [CODE_WIDTH-1:0] srnn_store_write_code;
    logic signed [RESIDUAL_WIDTH-1:0] srnn_store_write_residual;
    logic srnn_store_write_code_enable;
    logic srnn_store_write_residual_enable;
    logic srnn_store_forward_valid;
    logic srnn_store_forward_ready;
    logic [PARAM_ADDR_WIDTH-1:0] srnn_store_forward_addr;
    logic [TAG_WIDTH-1:0] srnn_store_forward_tag;
    logic srnn_store_forward_response_valid;
    logic srnn_store_forward_response_ready;
    logic srnn_cfg_ready_internal;
    logic srnn_load_ready_internal;
    logic srnn_timestep_start_ready_internal;
    logic srnn_abort_ready_internal;
    logic srnn_backbone_ready_internal;
    logic srnn_target_trace_ready_internal;
    logic srnn_label_ready_internal;
    logic srnn_restore_ready_internal;
    logic srnn_logit_valid_internal;

    logic [2:0] load_valid;
    logic [2:0] load_ready;
    logic [3*PARAM_ADDR_WIDTH-1:0] load_addr;
    logic signed [3*CODE_WIDTH-1:0] load_code;
    logic signed [3*RESIDUAL_WIDTH-1:0] load_residual;
    logic [2:0] learning_read_valid;
    logic [2:0] learning_read_ready;
    logic [3*PARAM_ADDR_WIDTH-1:0] learning_read_addr;
    logic [3*TAG_WIDTH-1:0] learning_read_tag;
    logic [2:0] learning_response_valid;
    logic [2:0] learning_response_ready;
    logic [3*TAG_WIDTH-1:0] learning_response_tag;
    logic signed [3*CODE_WIDTH-1:0] learning_response_code;
    logic signed [3*RESIDUAL_WIDTH-1:0] learning_response_residual;
    logic [2:0] learning_response_error;
    logic [2:0] learning_write_valid;
    logic [2:0] learning_write_ready;
    logic [3*PARAM_ADDR_WIDTH-1:0] learning_write_addr;
    logic signed [3*CODE_WIDTH-1:0] learning_write_code;
    logic signed [3*RESIDUAL_WIDTH-1:0] learning_write_residual;
    logic [2:0] learning_write_code_enable;
    logic [2:0] learning_write_residual_enable;
    logic [2:0] forward_read_valid;
    logic [2:0] forward_read_ready;
    logic [3*PARAM_ADDR_WIDTH-1:0] forward_read_addr;
    logic [3*TAG_WIDTH-1:0] forward_read_tag;
    logic [2:0] forward_response_valid;
    logic [2:0] forward_response_ready;
    logic [3*TAG_WIDTH-1:0] forward_response_tag;
    logic signed [3*CODE_WIDTH-1:0] forward_response_code;
    logic [2:0] forward_response_error;

    assign fc_selected = ENABLE_FC_CLIENT &&
        i_parameter_owner == 3'b001;
    assign fc_learning_selected = ENABLE_ONLINE_LEARNING && fc_selected &&
        i_topology_learning_enable[0];
    assign srnn_selected = ENABLE_SRNN_CLIENT &&
        i_parameter_owner == 3'b010;
    assign scnn_selected = ENABLE_SCNN_CLIENT &&
        i_parameter_owner == 3'b100;
    assign scnn_learning_selected = ENABLE_ONLINE_LEARNING && scnn_selected &&
        i_topology_learning_enable[2];
    assign o_fc_cfg_ready = ENABLE_FC_CLIENT &&
        fc_cfg_ready_internal && fc_learning_selected;
    assign o_fc_cfg_error = ENABLE_FC_CLIENT &&
        (fc_cfg_error_internal || fc_done_protocol_error_q);
    assign o_fc_release_ready = ENABLE_FC_CLIENT &&
        fc_release_ready_internal && fc_learning_selected;
    assign o_fc_student_trace_ready = ENABLE_FC_CLIENT &&
        fc_student_ready_internal && fc_learning_selected;
    assign o_fc_logit_row_ready = ENABLE_FC_CLIENT &&
        fc_logit_row_ready_internal && fc_learning_selected;
    assign o_fc_logit_ready = ENABLE_FC_CLIENT &&
        fc_logit_ready_internal && fc_learning_selected;
    assign fc_engine_error = fc_cfg_error_internal ||
        fc_done_protocol_error_q || fc_observation_error ||
        fc_context_error || fc_arithmetic_overflow || fc_parameter_error ||
        fc_protocol_error || shared_fc_commit_protocol_error_q;
    assign o_fc_error = ENABLE_FC_CLIENT && fc_engine_error;
    assign o_fc_window_commit_valid = ENABLE_FC_CLIENT ?
        fc_window_commit_valid_internal : 1'b0;
    assign o_fc_window_commit_sequence = ENABLE_FC_CLIENT ?
        fc_window_commit_sequence_internal : '0;
    assign o_fc_window_busy = ENABLE_FC_CLIENT &&
        fc_window_busy_internal;

    assign o_backend_quiescent = backend_quiescent_internal &&
        (!SHARE_FC_SCNN_OUTPUT_LEARNER ||
         (!shared_output_busy && !scnn_merge_busy));
    assign o_backend_error = backend_error_internal ||
        (SHARE_FC_SCNN_OUTPUT_LEARNER &&
         (shared_output_protocol_error || scnn_merge_protocol_error));

    assign o_srnn_cfg_ready = ENABLE_SRNN_CLIENT &&
        srnn_cfg_ready_internal && srnn_selected;
    assign o_srnn_parameter_load_ready = ENABLE_SRNN_CLIENT &&
        srnn_load_ready_internal && srnn_selected;
    assign o_srnn_timestep_start_ready = ENABLE_SRNN_CLIENT &&
        srnn_timestep_start_ready_internal && srnn_selected;
    assign o_srnn_abort_ready = ENABLE_SRNN_CLIENT &&
        srnn_abort_ready_internal && srnn_selected;
    assign o_srnn_backbone_ready = ENABLE_SRNN_CLIENT &&
        srnn_backbone_ready_internal && srnn_selected;
    assign o_srnn_target_trace_ready = ENABLE_SRNN_CLIENT &&
        srnn_target_trace_ready_internal && srnn_selected;
    assign o_srnn_label_ready = ENABLE_SRNN_CLIENT &&
        srnn_label_ready_internal && srnn_selected;
    assign o_srnn_restore_ready = ENABLE_SRNN_CLIENT &&
        srnn_restore_ready_internal && srnn_selected;
    assign o_srnn_logit_valid = ENABLE_SRNN_CLIENT &&
        srnn_logit_valid_internal && srnn_selected;

    generate
    if (ENABLE_FC_CLIENT) begin : g_fc_completion_state
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            fc_observation_done_pending_q <= 1'b0;
            fc_done_protocol_error_q <= 1'b0;
        end else begin
            if (i_clear_error)
                fc_done_protocol_error_q <= 1'b0;
            if (i_fc_observation_done && fc_learning_selected &&
                fc_observation_done_pending_q &&
                !fc_observation_done_ready)
                fc_done_protocol_error_q <= 1'b1;
            unique case ({i_fc_observation_done && fc_learning_selected,
                    fc_observation_done_pending_q &&
                    fc_observation_done_ready})
                2'b01: fc_observation_done_pending_q <= 1'b0;
                2'b10, 2'b11: fc_observation_done_pending_q <= 1'b1;
                default: fc_observation_done_pending_q <=
                    fc_observation_done_pending_q;
            endcase
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            shared_fc_commit_valid_q <= 1'b0;
            shared_fc_commit_sequence_q <= '0;
            shared_fc_commit_protocol_error_q <= 1'b0;
        end else begin
            if (i_clear_error)
                shared_fc_commit_protocol_error_q <= 1'b0;
            if (SHARE_FC_SCNN_OUTPUT_LEARNER) begin
                if (i_fc_cfg_valid && o_fc_cfg_ready) begin
                    shared_fc_commit_valid_q <= 1'b0;
                    shared_fc_commit_sequence_q <= '0;
                end
                if (shared_fc_window_done) begin
                    if (shared_fc_commit_valid_q &&
                        !i_fc_window_commit_ready)
                        shared_fc_commit_protocol_error_q <= 1'b1;
                    shared_fc_commit_valid_q <= 1'b1;
                end else if (shared_fc_commit_valid_q &&
                        i_fc_window_commit_ready) begin
                    shared_fc_commit_valid_q <= 1'b0;
                    shared_fc_commit_sequence_q <=
                        shared_fc_commit_sequence_q + 1'b1;
                end
                if (i_fc_release && fc_release_ready_internal)
                    shared_fc_commit_valid_q <= 1'b0;
            end else begin
                shared_fc_commit_valid_q <= 1'b0;
                shared_fc_commit_sequence_q <= '0;
            end
        end
    end
    end else begin : g_no_fc_completion_state
        assign fc_observation_done_pending_q = 1'b0;
        assign fc_done_protocol_error_q = 1'b0;
        assign shared_fc_commit_valid_q = 1'b0;
        assign shared_fc_commit_sequence_q = '0;
        assign shared_fc_commit_protocol_error_q = 1'b0;
    end
    endgenerate

    generate
    if (!SHARE_FC_SCNN_OUTPUT_LEARNER) begin : gen_private_fc_output_learner
    if (ENABLE_FC_CLIENT) begin : g_fc_client
    unified_fc_tp_online_learning_engine #(
        .ENABLE_W0_LEARNING(1'b0),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH), .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .MAX_BATCH(16), .MAX_HIDDEN(FC_MAX_HIDDEN),
        .MAX_FEATURES(2312), .MAX_TARGET_DIMENSION(16),
        .MAX_OUTPUTS(FC_MAX_OUTPUTS), .W2_MAX_GRADIENT_ELEMENTS(32768)
    ) u_fc_w2 (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_cfg_valid(i_fc_cfg_valid && fc_learning_selected),
        .o_cfg_ready(fc_cfg_ready_internal),
        .i_cfg_profile_id(i_fc_profile_id),
        .i_cfg_w0_tensor_id(8'h00), .i_cfg_w0_parameter_base('0),
        .i_cfg_w0_vote_increment(16'd0),
        .i_cfg_w0_vote_threshold(16'd1),
        .i_cfg_w2_tensor_id(i_fc_w2_tensor_id),
        .i_cfg_w2_parameter_base(i_fc_w2_parameter_base),
        .i_cfg_w2_vote_increment(i_fc_vote_increment),
        .i_cfg_w2_vote_threshold(i_fc_vote_threshold),
        .i_cfg_batch_count(i_fc_batch_count),
        .i_cfg_hidden_count(i_fc_hidden_count),
        .i_cfg_feature_count(12'd1), .i_cfg_target_dimension(5'd1),
        .i_cfg_output_count(i_fc_output_count),
        .i_cfg_w0_code_min('0), .i_cfg_w0_code_max('0),
        .i_cfg_w2_code_min(i_fc_code_min),
        .i_cfg_w2_code_max(i_fc_code_max),
        .o_cfg_error(fc_cfg_error_internal), .o_active(),
        .i_release(i_fc_release && fc_learning_selected),
        .o_release_ready(fc_release_ready_internal),
        .i_student_trace_valid(i_fc_student_trace_valid &&
            fc_learning_selected),
        .o_student_trace_ready(fc_student_ready_internal),
        .i_student_trace_sample(i_fc_student_trace_sample[3:0]),
        .i_student_trace_hidden(i_fc_student_trace_hidden[8:0]),
        .i_student_trace_q016(i_fc_student_trace_q016),
        .i_target_hidden_trace_valid(1'b0),
        .o_target_hidden_trace_ready(), .i_target_hidden_trace_sample('0),
        .i_target_hidden_trace_hidden('0),
        .i_target_hidden_trace_q016('0),
        .i_target_input_trace_valid(1'b0),
        .o_target_input_trace_ready(), .i_target_input_trace_sample('0),
        .i_target_input_trace_index('0), .i_target_input_trace_q016('0),
        .i_surrogate_valid(1'b0), .o_surrogate_ready(),
        .i_surrogate_sample('0), .i_surrogate_hidden('0),
        .i_surrogate_q016('0), .i_event_valid(1'b0), .o_event_ready(),
        .i_event_sample_id('0), .i_event_feature_index('0),
        .i_logit_row_valid(i_fc_logit_row_valid && fc_learning_selected),
        .o_logit_row_ready(fc_logit_row_ready_internal),
        .i_logit_sample_id(i_fc_logit_sample[3:0]),
        .i_logit_label(i_fc_logit_label),
        .i_logit_valid(i_fc_logit_valid && fc_learning_selected),
        .o_logit_ready(fc_logit_ready_internal),
        .i_logit_q6(i_fc_logit_q6), .o_logit_row_done(),
        .i_observation_done(fc_observation_done_pending_q &&
            fc_learning_selected),
        .o_observation_done_ready(fc_observation_done_ready),
        .o_window_commit_valid(fc_window_commit_valid_internal),
        .i_window_commit_ready(i_fc_window_commit_ready),
        .o_window_commit_profile_id(),
        .o_window_commit_sequence(fc_window_commit_sequence_internal),
        .o_window_busy(fc_window_busy_internal),
        .o_observation_error(fc_observation_error),
        .o_context_error(fc_context_error),
        .o_arithmetic_overflow(fc_arithmetic_overflow),
        .o_parameter_error(fc_parameter_error),
        .o_protocol_error(fc_protocol_error),
        .o_read_valid(fc_read_valid), .i_read_ready(fc_read_ready),
        .o_read_profile_id(), .o_read_tensor_id(),
        .o_read_parameter_addr(fc_read_addr), .o_read_tag(fc_read_tag),
        .i_read_response_valid(fc_read_response_valid),
        .o_read_response_ready(fc_read_response_ready),
        .i_read_response_code(learning_response_code[0 +: CODE_WIDTH]),
        .i_read_response_residual(
            learning_response_residual[0 +: RESIDUAL_WIDTH]),
        .i_read_response_error(learning_response_error[0]),
        .o_result_valid(fc_result_valid), .i_result_ready(fc_result_ready),
        .o_result_profile_id(), .o_result_tensor_id(),
        .o_result_parameter_addr(fc_result_addr), .o_result_tag(),
        .o_result_code(fc_result_code),
        .o_result_residual(fc_result_residual),
        .o_write_code_enable(fc_write_code_enable),
        .o_write_residual_enable(fc_write_residual_enable),
        .o_write_code_fire(), .o_write_residual_fire(),
        .o_dirty(), .o_saturated()
    );
    end else begin : g_no_fc_client
        assign fc_cfg_ready_internal = 1'b0;
        assign fc_cfg_error_internal = 1'b0;
        assign fc_release_ready_internal = 1'b0;
        assign fc_student_ready_internal = 1'b0;
        assign fc_logit_row_ready_internal = 1'b0;
        assign fc_logit_ready_internal = 1'b0;
        assign fc_observation_done_ready = 1'b0;
        assign fc_observation_error = 1'b0;
        assign fc_context_error = 1'b0;
        assign fc_arithmetic_overflow = 1'b0;
        assign fc_parameter_error = 1'b0;
        assign fc_protocol_error = 1'b0;
        assign fc_read_valid = 1'b0;
        assign fc_read_addr = '0;
        assign fc_read_tag = '0;
        assign fc_read_response_ready = 1'b0;
        assign fc_result_valid = 1'b0;
        assign fc_result_addr = '0;
        assign fc_result_code = '0;
        assign fc_result_residual = '0;
        assign fc_write_code_enable = 1'b0;
        assign fc_write_residual_enable = 1'b0;
        assign fc_window_commit_valid_internal = 1'b0;
        assign fc_window_commit_sequence_internal = '0;
        assign fc_window_busy_internal = 1'b0;
    end
    assign scnn_merged_read_valid = ENABLE_SCNN_CLIENT ?
        i_scnn_store_learning_read_valid : 1'b0;
    assign o_scnn_store_learning_read_ready = ENABLE_SCNN_CLIENT ?
        scnn_merged_read_ready : 1'b0;
    assign scnn_merged_read_addr = i_scnn_store_learning_read_addr;
    assign scnn_merged_read_tag = i_scnn_store_learning_read_tag;
    assign o_scnn_store_learning_response_valid = ENABLE_SCNN_CLIENT ?
        scnn_merged_response_valid : 1'b0;
    assign scnn_merged_response_ready = ENABLE_SCNN_CLIENT ?
        i_scnn_store_learning_response_ready : 1'b0;
    assign o_scnn_store_learning_response_tag = ENABLE_SCNN_CLIENT ?
        learning_response_tag[2*TAG_WIDTH +: TAG_WIDTH] : '0;
    assign o_scnn_store_learning_response_code = ENABLE_SCNN_CLIENT ?
        learning_response_code[2*CODE_WIDTH +: CODE_WIDTH] : '0;
    assign o_scnn_store_learning_response_residual = ENABLE_SCNN_CLIENT ?
        learning_response_residual[2*RESIDUAL_WIDTH +: RESIDUAL_WIDTH] : '0;
    assign o_scnn_store_learning_response_error = ENABLE_SCNN_CLIENT &&
        learning_response_error[2];
    assign scnn_merged_write_valid = ENABLE_SCNN_CLIENT ?
        i_scnn_store_learning_write_valid : 1'b0;
    assign o_scnn_store_learning_write_ready = ENABLE_SCNN_CLIENT ?
        scnn_merged_write_ready : 1'b0;
    assign scnn_merged_write_addr = i_scnn_store_learning_write_addr;
    assign scnn_merged_write_code = i_scnn_store_learning_write_code;
    assign scnn_merged_write_residual =
        i_scnn_store_learning_write_residual;
    assign scnn_merged_write_code_enable = ENABLE_SCNN_CLIENT &&
        i_scnn_store_learning_write_code_enable;
    assign scnn_merged_write_residual_enable = ENABLE_SCNN_CLIENT &&
        i_scnn_store_learning_write_residual_enable;
    assign scnn_merge_busy = 1'b0;
    assign scnn_merge_protocol_error = 1'b0;
    assign shared_output_owner = 2'd0;
    assign shared_output_busy = 1'b0;
    assign shared_output_protocol_error = 1'b0;
    assign shared_output_read_valid = 1'b0;
    assign shared_output_read_profile_id = '0;
    assign shared_output_read_tensor_id = '0;
    assign shared_output_read_addr = '0;
    assign shared_output_read_tag = '0;
    assign shared_output_response_ready = 1'b0;
    assign shared_output_result_valid = 1'b0;
    assign shared_output_result_profile_id = '0;
    assign shared_output_result_tensor_id = '0;
    assign shared_output_result_addr = '0;
    assign shared_output_result_tag = '0;
    assign shared_output_result_code = '0;
    assign shared_output_result_residual = '0;
    assign shared_output_write_code_enable = 1'b0;
    assign shared_output_write_residual_enable = 1'b0;
    assign shared_output_write_code_fire = 1'b0;
    assign shared_output_write_residual_fire = 1'b0;
    assign shared_output_dirty = 1'b0;
    assign shared_output_saturated = 1'b0;
    assign shared_output_update_config_error = 1'b0;
    assign shared_output_request_error = 1'b0;
    assign shared_output_store_error = 1'b0;
    assign shared_output_internal_error = 1'b0;
    assign shared_fc_window_done = 1'b0;
    assign shared_fc_window_busy = 1'b0;
    assign o_scnn_external_cfg_ready = 1'b0;
    assign o_scnn_external_cfg_error = 1'b0;
    assign o_scnn_external_active = 1'b0;
    assign o_scnn_external_release_ready = 1'b0;
    assign o_scnn_external_source_trace_ready = 1'b0;
    assign o_scnn_external_source_trace_error = 1'b0;
    assign o_scnn_external_logit_row_ready = 1'b0;
    assign o_scnn_external_logit_ready = 1'b0;
    assign o_scnn_external_logit_error = 1'b0;
    assign o_scnn_external_logit_row_done = 1'b0;
    assign o_scnn_external_observation_done_ready = 1'b0;
    assign o_scnn_external_window_done = 1'b0;
    assign o_scnn_external_window_busy = 1'b0;
    assign o_scnn_external_context_error = 1'b0;
    assign o_scnn_external_gradient_overflow = 1'b0;
    assign o_scnn_external_update_config_error = 1'b0;
    assign o_scnn_external_request_error = 1'b0;
    assign o_scnn_external_store_error = 1'b0;
    assign o_scnn_external_internal_error = 1'b0;
    end else begin : gen_shared_fc_scnn_output_learner

    unified_tp_shared_output_learning_service_v1 #(
        .ENABLE_FC_CLIENT(ENABLE_FC_CLIENT),
        .ENABLE_SCNN_CLIENT(ENABLE_SCNN_CLIENT),
        .PROFILE_ID_WIDTH(8), .TENSOR_ID_WIDTH(8),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH), .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(16), .MAX_BATCH(SHARED_MAX_BATCH),
        .MAX_SOURCES(SHARED_MAX_SOURCES),
        .MAX_OUTPUTS(SHARED_MAX_OUTPUTS),
        .MAX_GRADIENT_ELEMENTS(SHARED_MAX_GRADIENT_ELEMENTS),
        .BATCH_INDEX_WIDTH(SHARED_BATCH_INDEX_WIDTH),
        .BATCH_COUNT_WIDTH(SHARED_BATCH_COUNT_WIDTH),
        .SOURCE_INDEX_WIDTH(SHARED_SOURCE_INDEX_WIDTH),
        .SOURCE_COUNT_WIDTH(SHARED_SOURCE_COUNT_WIDTH),
        .OUTPUT_INDEX_WIDTH(SHARED_OUTPUT_INDEX_WIDTH),
        .OUTPUT_COUNT_WIDTH(SHARED_OUTPUT_COUNT_WIDTH)
    ) u_shared_output_learner (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_protocol_error(i_clear_error),
        .i_fc_cfg_valid(i_fc_cfg_valid && fc_learning_selected),
        .o_fc_cfg_ready(fc_cfg_ready_internal),
        .i_fc_cfg_profile_id(i_fc_profile_id),
        .i_fc_cfg_tensor_id(i_fc_w2_tensor_id),
        .i_fc_cfg_parameter_base(i_fc_w2_parameter_base),
        .i_fc_cfg_batch_count(SHARED_BATCH_COUNT_WIDTH'(
            i_fc_batch_count)),
        .i_fc_cfg_source_count(SHARED_SOURCE_COUNT_WIDTH'(
            i_fc_hidden_count)),
        .i_fc_cfg_output_count(SHARED_OUTPUT_COUNT_WIDTH'(
            i_fc_output_count)),
        .i_fc_cfg_output_major(1'b0),
        .i_fc_cfg_source_stride('0),
        .i_fc_cfg_output_stride(PARAM_ADDR_WIDTH'(i_fc_output_count)),
        .i_fc_cfg_vote_increment(i_fc_vote_increment),
        .i_fc_cfg_vote_threshold(i_fc_vote_threshold),
        .i_fc_cfg_code_min(i_fc_code_min),
        .i_fc_cfg_code_max(i_fc_code_max),
        .o_fc_cfg_error(fc_cfg_error_internal),
        .o_fc_active(),
        .i_fc_release(i_fc_release && fc_learning_selected),
        .o_fc_release_ready(fc_release_ready_internal),
        .i_fc_source_trace_valid(i_fc_student_trace_valid &&
            fc_learning_selected),
        .o_fc_source_trace_ready(fc_student_ready_internal),
        .i_fc_source_trace_sample_id(SHARED_BATCH_INDEX_WIDTH'(
            i_fc_student_trace_sample)),
        .i_fc_source_trace_index(SHARED_SOURCE_INDEX_WIDTH'(
            i_fc_student_trace_hidden)),
        .i_fc_source_trace_q016(i_fc_student_trace_q016[16] ?
            16'hffff : i_fc_student_trace_q016[15:0]),
        .o_fc_source_trace_error(fc_observation_error),
        .i_fc_logit_row_valid(i_fc_logit_row_valid &&
            fc_learning_selected),
        .o_fc_logit_row_ready(fc_logit_row_ready_internal),
        .i_fc_logit_sample_id(SHARED_BATCH_INDEX_WIDTH'(
            i_fc_logit_sample)),
        .i_fc_logit_label(SHARED_OUTPUT_INDEX_WIDTH'(
            i_fc_logit_label)),
        .i_fc_logit_valid(i_fc_logit_valid && fc_learning_selected),
        .o_fc_logit_ready(fc_logit_ready_internal),
        .i_fc_logit_q6(i_fc_logit_q6),
        .o_fc_logit_error(fc_context_error),
        .o_fc_logit_row_done(),
        .i_fc_observation_done(fc_observation_done_pending_q &&
            fc_learning_selected),
        .o_fc_observation_done_ready(fc_observation_done_ready),
        .o_fc_window_done(shared_fc_window_done),
        .o_fc_window_busy(shared_fc_window_busy),
        .o_fc_context_error(fc_arithmetic_overflow),
        .o_fc_gradient_overflow(fc_parameter_error),
        .i_scnn_cfg_valid(i_scnn_external_cfg_valid &&
            scnn_learning_selected),
        .o_scnn_cfg_ready(scnn_external_cfg_ready_internal),
        .i_scnn_cfg_profile_id(i_scnn_external_cfg_profile_id),
        .i_scnn_cfg_tensor_id(i_scnn_external_cfg_tensor_id),
        .i_scnn_cfg_parameter_base(
            i_scnn_external_cfg_parameter_base),
        .i_scnn_cfg_batch_count(SHARED_BATCH_COUNT_WIDTH'(
            i_scnn_external_cfg_batch_count)),
        .i_scnn_cfg_source_count(SHARED_SOURCE_COUNT_WIDTH'(
            i_scnn_external_cfg_source_count)),
        .i_scnn_cfg_output_count(SHARED_OUTPUT_COUNT_WIDTH'(
            i_scnn_external_cfg_output_count)),
        .i_scnn_cfg_output_major(1'b1),
        .i_scnn_cfg_source_stride(PARAM_ADDR_WIDTH'(
            i_scnn_external_cfg_source_count)),
        .i_scnn_cfg_output_stride('0),
        .i_scnn_cfg_vote_increment(
            i_scnn_external_cfg_vote_increment),
        .i_scnn_cfg_vote_threshold(
            i_scnn_external_cfg_vote_threshold),
        .i_scnn_cfg_code_min(i_scnn_external_cfg_code_min),
        .i_scnn_cfg_code_max(i_scnn_external_cfg_code_max),
        .o_scnn_cfg_error(o_scnn_external_cfg_error),
        .o_scnn_active(o_scnn_external_active),
        .i_scnn_release(i_scnn_external_release &&
            scnn_learning_selected),
        .o_scnn_release_ready(scnn_external_release_ready_internal),
        .i_scnn_source_trace_valid(
            i_scnn_external_source_trace_valid &&
            scnn_learning_selected),
        .o_scnn_source_trace_ready(
            scnn_external_source_trace_ready_internal),
        .i_scnn_source_trace_sample_id(SHARED_BATCH_INDEX_WIDTH'(
            i_scnn_external_source_trace_sample_id)),
        .i_scnn_source_trace_index(SHARED_SOURCE_INDEX_WIDTH'(
            i_scnn_external_source_trace_index)),
        .i_scnn_source_trace_q016(
            i_scnn_external_source_trace_q016),
        .o_scnn_source_trace_error(
            o_scnn_external_source_trace_error),
        .i_scnn_logit_row_valid(i_scnn_external_logit_row_valid &&
            scnn_learning_selected),
        .o_scnn_logit_row_ready(
            scnn_external_logit_row_ready_internal),
        .i_scnn_logit_sample_id(SHARED_BATCH_INDEX_WIDTH'(
            i_scnn_external_logit_sample_id)),
        .i_scnn_logit_label(SHARED_OUTPUT_INDEX_WIDTH'(
            i_scnn_external_logit_label)),
        .i_scnn_logit_valid(i_scnn_external_logit_valid &&
            scnn_learning_selected),
        .o_scnn_logit_ready(scnn_external_logit_ready_internal),
        .i_scnn_logit_q6(i_scnn_external_logit_q6),
        .o_scnn_logit_error(o_scnn_external_logit_error),
        .o_scnn_logit_row_done(o_scnn_external_logit_row_done),
        .i_scnn_observation_done(
            i_scnn_external_observation_done && scnn_learning_selected),
        .o_scnn_observation_done_ready(
            scnn_external_observation_done_ready_internal),
        .o_scnn_window_done(o_scnn_external_window_done),
        .o_scnn_window_busy(o_scnn_external_window_busy),
        .o_scnn_context_error(o_scnn_external_context_error),
        .o_scnn_gradient_overflow(
            o_scnn_external_gradient_overflow),
        .o_active_owner(shared_output_owner),
        .o_busy(shared_output_busy),
        .o_protocol_error(shared_output_protocol_error),
        .o_read_valid(shared_output_read_valid),
        .i_read_ready(shared_output_read_ready),
        .o_read_profile_id(shared_output_read_profile_id),
        .o_read_tensor_id(shared_output_read_tensor_id),
        .o_read_parameter_addr(shared_output_read_addr),
        .o_read_tag(shared_output_read_tag),
        .i_read_response_valid(shared_output_response_valid),
        .o_read_response_ready(shared_output_response_ready),
        .i_read_response_code(shared_output_response_code),
        .i_read_response_residual(shared_output_response_residual),
        .i_read_response_error(shared_output_response_error),
        .o_result_valid(shared_output_result_valid),
        .i_result_ready(shared_output_result_ready),
        .o_result_profile_id(shared_output_result_profile_id),
        .o_result_tensor_id(shared_output_result_tensor_id),
        .o_result_parameter_addr(shared_output_result_addr),
        .o_result_tag(shared_output_result_tag),
        .o_result_code(shared_output_result_code),
        .o_result_residual(shared_output_result_residual),
        .o_write_code_enable(shared_output_write_code_enable),
        .o_write_residual_enable(
            shared_output_write_residual_enable),
        .o_write_code_fire(shared_output_write_code_fire),
        .o_write_residual_fire(shared_output_write_residual_fire),
        .o_dirty(shared_output_dirty),
        .o_saturated(shared_output_saturated),
        .o_update_config_error(shared_output_update_config_error),
        .o_request_error(shared_output_request_error),
        .o_store_error(shared_output_store_error),
        .o_internal_error(shared_output_internal_error)
    );

    // The runtime parameter allocator gives independently trainable SCNN
    // tensors disjoint address intervals. This arbiter serializes their port
    // traffic; it intentionally does not infer tensor ownership from addresses.
    unified_tp_two_client_parameter_port_arbiter #(
        .PROFILE_ID_WIDTH(8), .TENSOR_ID_WIDTH(8),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH), .RESIDUAL_WIDTH(RESIDUAL_WIDTH)
    ) u_scnn_conv_classifier_rmw (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_c0_read_valid(i_scnn_store_learning_read_valid &&
            scnn_learning_selected),
        .o_c0_read_ready(o_scnn_store_learning_read_ready),
        .i_c0_read_profile_id('0), .i_c0_read_tensor_id('0),
        .i_c0_read_parameter_addr(i_scnn_store_learning_read_addr),
        .i_c0_read_tag(i_scnn_store_learning_read_tag),
        .o_c0_read_response_valid(
            o_scnn_store_learning_response_valid),
        .i_c0_read_response_ready(
            i_scnn_store_learning_response_ready),
        .o_c0_read_response_code(o_scnn_store_learning_response_code),
        .o_c0_read_response_residual(
            o_scnn_store_learning_response_residual),
        .o_c0_read_response_error(
            o_scnn_store_learning_response_error),
        .i_c0_result_valid(i_scnn_store_learning_write_valid &&
            scnn_learning_selected),
        .o_c0_result_ready(o_scnn_store_learning_write_ready),
        .i_c0_result_profile_id('0), .i_c0_result_tensor_id('0),
        .i_c0_result_parameter_addr(i_scnn_store_learning_write_addr),
        .i_c0_result_tag('0),
        .i_c0_result_code(i_scnn_store_learning_write_code),
        .i_c0_result_residual(i_scnn_store_learning_write_residual),
        .i_c0_write_code_enable(
            i_scnn_store_learning_write_code_enable),
        .i_c0_write_residual_enable(
            i_scnn_store_learning_write_residual_enable),
        .i_c0_write_code_fire(i_scnn_store_learning_write_valid &&
            o_scnn_store_learning_write_ready &&
            i_scnn_store_learning_write_code_enable),
        .i_c0_write_residual_fire(i_scnn_store_learning_write_valid &&
            o_scnn_store_learning_write_ready &&
            i_scnn_store_learning_write_residual_enable),
        .i_c0_dirty(1'b0), .i_c0_saturated(1'b0),
        .i_c1_read_valid(shared_output_read_valid &&
            shared_output_owner == 2'd2),
        .o_c1_read_ready(shared_scnn_read_ready),
        .i_c1_read_profile_id(shared_output_read_profile_id),
        .i_c1_read_tensor_id(shared_output_read_tensor_id),
        .i_c1_read_parameter_addr(shared_output_read_addr),
        .i_c1_read_tag(shared_output_read_tag),
        .o_c1_read_response_valid(shared_scnn_response_valid),
        .i_c1_read_response_ready(shared_output_response_ready),
        .o_c1_read_response_code(shared_scnn_response_code),
        .o_c1_read_response_residual(
            shared_scnn_response_residual),
        .o_c1_read_response_error(shared_scnn_response_error),
        .i_c1_result_valid(shared_output_result_valid &&
            shared_output_owner == 2'd2),
        .o_c1_result_ready(shared_scnn_result_ready),
        .i_c1_result_profile_id(shared_output_result_profile_id),
        .i_c1_result_tensor_id(shared_output_result_tensor_id),
        .i_c1_result_parameter_addr(shared_output_result_addr),
        .i_c1_result_tag(shared_output_result_tag),
        .i_c1_result_code(shared_output_result_code),
        .i_c1_result_residual(shared_output_result_residual),
        .i_c1_write_code_enable(shared_output_write_code_enable),
        .i_c1_write_residual_enable(
            shared_output_write_residual_enable),
        .i_c1_write_code_fire(shared_output_write_code_fire),
        .i_c1_write_residual_fire(
            shared_output_write_residual_fire),
        .i_c1_dirty(shared_output_dirty),
        .i_c1_saturated(shared_output_saturated),
        .o_read_valid(scnn_merged_read_valid),
        .i_read_ready(scnn_merged_read_ready),
        .o_read_profile_id(), .o_read_tensor_id(),
        .o_read_parameter_addr(scnn_merged_read_addr),
        .o_read_tag(scnn_merged_read_tag),
        .i_read_response_valid(scnn_merged_response_valid),
        .o_read_response_ready(scnn_merged_response_ready),
        .i_read_response_code(
            learning_response_code[2*CODE_WIDTH +: CODE_WIDTH]),
        .i_read_response_residual(
            learning_response_residual[2*RESIDUAL_WIDTH +:
                RESIDUAL_WIDTH]),
        .i_read_response_error(learning_response_error[2]),
        .o_result_valid(scnn_merged_write_valid),
        .i_result_ready(scnn_merged_write_ready),
        .o_result_profile_id(), .o_result_tensor_id(),
        .o_result_parameter_addr(scnn_merged_write_addr),
        .o_result_tag(), .o_result_code(scnn_merged_write_code),
        .o_result_residual(scnn_merged_write_residual),
        .o_write_code_enable(scnn_merged_write_code_enable),
        .o_write_residual_enable(
            scnn_merged_write_residual_enable),
        .o_write_code_fire(), .o_write_residual_fire(),
        .o_dirty(), .o_saturated(), .o_busy(scnn_merge_busy),
        .o_protocol_error(scnn_merge_protocol_error)
    );

    assign fc_window_commit_valid_internal = shared_fc_commit_valid_q;
    assign fc_window_commit_sequence_internal =
        shared_fc_commit_sequence_q;
    assign fc_window_busy_internal = shared_fc_window_busy ||
        shared_fc_commit_valid_q;
    assign fc_read_valid = shared_output_read_valid &&
        shared_output_owner == 2'd1;
    assign fc_read_addr = shared_output_read_addr;
    assign fc_read_tag = shared_output_read_tag;
    assign fc_read_response_ready = shared_output_response_ready &&
        shared_output_owner == 2'd1;
    assign fc_result_valid = shared_output_result_valid &&
        shared_output_owner == 2'd1;
    assign fc_result_addr = shared_output_result_addr;
    assign fc_result_code = shared_output_result_code;
    assign fc_result_residual = shared_output_result_residual;
    assign fc_write_code_enable = shared_output_write_code_enable;
    assign fc_write_residual_enable =
        shared_output_write_residual_enable;
    assign shared_output_read_ready =
        (shared_output_owner == 2'd1) ? fc_read_ready :
        ((shared_output_owner == 2'd2) ? shared_scnn_read_ready : 1'b0);
    assign shared_output_response_valid =
        (shared_output_owner == 2'd1) ? fc_read_response_valid :
        ((shared_output_owner == 2'd2) ?
         shared_scnn_response_valid : 1'b0);
    assign shared_output_response_code =
        (shared_output_owner == 2'd1) ?
        learning_response_code[0 +: CODE_WIDTH] :
        shared_scnn_response_code;
    assign shared_output_response_residual =
        (shared_output_owner == 2'd1) ?
        learning_response_residual[0 +: RESIDUAL_WIDTH] :
        shared_scnn_response_residual;
    assign shared_output_response_error =
        (shared_output_owner == 2'd1) ? learning_response_error[0] :
        shared_scnn_response_error;
    assign shared_output_result_ready =
        (shared_output_owner == 2'd1) ? fc_result_ready :
        ((shared_output_owner == 2'd2) ?
         shared_scnn_result_ready : 1'b0);
    assign o_scnn_external_update_config_error =
        shared_output_update_config_error;
    assign o_scnn_external_request_error = shared_output_request_error;
    assign o_scnn_external_store_error = shared_output_store_error;
    assign o_scnn_external_internal_error =
        shared_output_internal_error || shared_output_protocol_error ||
        scnn_merge_protocol_error;
    assign o_scnn_external_cfg_ready =
        scnn_external_cfg_ready_internal && scnn_learning_selected;
    assign o_scnn_external_release_ready =
        scnn_external_release_ready_internal && scnn_learning_selected;
    assign o_scnn_external_source_trace_ready =
        scnn_external_source_trace_ready_internal &&
        scnn_learning_selected;
    assign o_scnn_external_logit_row_ready =
        scnn_external_logit_row_ready_internal &&
        scnn_learning_selected;
    assign o_scnn_external_logit_ready =
        scnn_external_logit_ready_internal && scnn_learning_selected;
    assign o_scnn_external_observation_done_ready =
        scnn_external_observation_done_ready_internal &&
        scnn_learning_selected;
    assign o_scnn_store_learning_response_tag =
        learning_response_tag[2*TAG_WIDTH +: TAG_WIDTH];
    assign fc_protocol_error = shared_output_request_error ||
        shared_output_internal_error || shared_output_protocol_error;
    end
    endgenerate

    generate
    if (ENABLE_SRNN_CLIENT) begin : g_srnn_client
    unified_srnn_m2_timestep_train_infer_coordinator #(
        .USE_INTERNAL_STORE(1'b0),
        .USE_EXTERNAL_BACKBONE_STREAM(1'b1),
        .USE_SHARED_H40_FORWARD(USE_SHARED_SRNN_H40_FORWARD),
        .ENABLE_WH1_LEARNING(ENABLE_SRNN_WH1_LEARNING),
        .BATCH_MAX(SRNN_BATCH_MAX),
        .TIMESTEP_MAX(SRNN_TIMESTEP_MAX),
        .HIDDEN_MAX(SRNN_HIDDEN_MAX),
        .BACKBONE_MAX(SRNN_BACKBONE_MAX),
        .CLASS_MAX(SRNN_CLASS_MAX), .SOURCE_VECTOR_MAX(480),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH), .RESIDUAL_WIDTH(RESIDUAL_WIDTH)
    ) u_srnn_m2 (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_cfg_valid(i_srnn_cfg_valid && srnn_selected),
        .o_cfg_ready(srnn_cfg_ready_internal),
        .i_cfg_profile_id(i_srnn_profile_id),
        .i_cfg_wh1_tensor_id(i_srnn_wh1_tensor_id),
        .i_cfg_wh1_base(i_srnn_wh1_base),
        .i_cfg_wh1_increment(i_srnn_wh1_increment),
        .i_cfg_wh1_threshold(i_srnn_wh1_threshold),
        .i_cfg_wout_tensor_id(i_srnn_wout_tensor_id),
        .i_cfg_wout_base(i_srnn_wout_base),
        .i_cfg_wout_increment(i_srnn_wout_increment),
        .i_cfg_wout_threshold(i_srnn_wout_threshold),
        .i_cfg_code_min(i_srnn_code_min), .i_cfg_code_max(i_srnn_code_max),
        .i_cfg_batch_count(i_srnn_batch_count),
        .i_cfg_hidden_count(i_srnn_hidden_count),
        .i_cfg_backbone_count(i_srnn_backbone_count),
        .i_cfg_class_count(i_srnn_class_count),
        .i_cfg_wout_element_count(i_srnn_wout_element_count),
        .o_cfg_error(o_srnn_cfg_error),
        .i_load_valid(i_srnn_parameter_load_valid && srnn_selected),
        .o_load_ready(srnn_load_ready_internal),
        .i_load_addr(i_srnn_parameter_load_addr),
        .i_load_code(i_srnn_parameter_load_code),
        .i_load_residual(i_srnn_parameter_load_residual),
        .i_timestep_start(i_srnn_timestep_start && srnn_selected),
        .o_timestep_start_ready(srnn_timestep_start_ready_internal),
        .i_timestep_id(i_srnn_timestep_id),
        .i_sequence_version(i_srnn_sequence_version),
        .i_learning_enable(i_srnn_learning_enable &&
            i_topology_learning_enable[1] && srnn_selected),
        .i_abort_timestep(i_srnn_abort_timestep && srnn_selected),
        .o_abort_ready(srnn_abort_ready_internal),
        .o_actual_replay_start(), .i_actual_replay_start_ready(1'b1),
        .o_actual_replay_timestep_id(), .i_actual_replay_valid(1'b0),
        .o_actual_replay_ready(), .i_actual_replay_sample_id('0),
        .i_actual_replay_spikes('0), .i_actual_replay_done(1'b0),
        .i_actual_replay_error(1'b0),
        .o_target_vector_start(), .i_target_vector_start_ready(1'b1),
        .o_target_vector_timestep_id(), .i_target_vector_valid(1'b0),
        .o_target_vector_ready(), .i_target_vector_sample_id('0),
        .i_target_vector_spikes('0), .i_target_vector_done(1'b0),
        .i_target_vector_error(1'b0),
        .i_external_backbone_valid(i_srnn_backbone_valid && srnn_selected),
        .o_external_backbone_ready(srnn_backbone_ready_internal),
        .i_external_backbone_path(i_srnn_backbone_path),
        .i_external_backbone_sample_id({2'b00,
            i_srnn_backbone_sample_id}),
        .i_external_backbone_id(i_srnn_backbone_id),
        .i_external_backbone_q14(i_srnn_backbone_q14),
        .i_external_backbone_last(i_srnn_backbone_last),
        .i_external_backbone_done(i_srnn_backbone_done && srnn_selected),
        .i_external_backbone_error(i_srnn_backbone_error && srnn_selected),
        .o_target_trace_start(o_srnn_target_trace_start),
        .i_target_trace_start_ready(i_srnn_target_trace_start_ready),
        .o_target_trace_timestep_id(o_srnn_target_trace_timestep_id),
        .i_target_trace_valid(i_srnn_target_trace_valid && srnn_selected),
        .o_target_trace_ready(srnn_target_trace_ready_internal),
        .i_target_trace_sample_id(i_srnn_target_trace_sample_id),
        .i_target_trace_backbone_id(i_srnn_target_trace_backbone_id),
        .i_target_trace_q14(i_srnn_target_trace_q14),
        .i_target_trace_last(i_srnn_target_trace_last),
        .i_target_trace_done(i_srnn_target_trace_done && srnn_selected),
        .i_target_trace_error(i_srnn_target_trace_error && srnn_selected),
        .o_target_trace_valid(), .i_target_trace_ready(1'b1),
        .o_target_trace_sample_id(), .o_target_trace_backbone_id(),
        .o_target_trace_q14(), .o_target_trace_last(),
        .o_target_trace_done(), .o_forward_code_valid(),
        .i_forward_code_ready(1'b1), .o_forward_tensor_id(),
        .o_forward_local_addr(), .o_forward_parameter_addr(),
        .o_forward_code(), .o_forward_last(),
        .o_forward_parameter_scan_done(),
        .o_h40_cfg_valid(o_srnn_h40_cfg_valid),
        .i_h40_cfg_ready(i_srnn_h40_cfg_ready),
        .i_h40_cfg_error(i_srnn_h40_cfg_error),
        .o_h40_cfg_batch_count(o_srnn_h40_cfg_batch_count),
        .o_h40_cfg_backbone_count(o_srnn_h40_cfg_backbone_count),
        .o_h40_cfg_hidden_count(o_srnn_h40_cfg_hidden_count),
        .o_h40_cfg_class_count(o_srnn_h40_cfg_class_count),
        .o_h40_cfg_wh1_parameter_base(
            o_srnn_h40_cfg_wh1_parameter_base),
        .o_h40_cfg_wout_parameter_base(
            o_srnn_h40_cfg_wout_parameter_base),
        .o_h40_prepare_valid(o_srnn_h40_prepare_valid),
        .i_h40_prepare_ready(i_srnn_h40_prepare_ready),
        .o_h40_prepare_version(o_srnn_h40_prepare_version),
        .i_h40_prepare_done(i_srnn_h40_prepare_done),
        .i_h40_cache_valid(i_srnn_h40_cache_valid),
        .o_h40_cache_invalidate(o_srnn_h40_cache_invalidate),
        .i_h40_forward_read_valid(i_srnn_h40_forward_read_valid),
        .o_h40_forward_read_ready(o_srnn_h40_forward_read_ready),
        .i_h40_forward_read_addr(i_srnn_h40_forward_read_addr),
        .i_h40_forward_read_tag(i_srnn_h40_forward_read_tag),
        .o_h40_forward_response_valid(o_srnn_h40_forward_response_valid),
        .i_h40_forward_response_ready(i_srnn_h40_forward_response_ready),
        .o_h40_forward_response_tag(o_srnn_h40_forward_response_tag),
        .o_h40_forward_response_code(o_srnn_h40_forward_response_code),
        .o_h40_forward_response_error(o_srnn_h40_forward_response_error),
        .o_h40_timestep_start(o_srnn_h40_timestep_start),
        .i_h40_timestep_start_ready(i_srnn_h40_timestep_start_ready),
        .o_h40_timestep_id(o_srnn_h40_timestep_id),
        .o_h40_sequence_version(o_srnn_h40_sequence_version),
        .o_h40_include_target_path(o_srnn_h40_include_target_path),
        .o_h40_observation_valid(o_srnn_h40_observation_valid),
        .i_h40_observation_ready(i_srnn_h40_observation_ready),
        .o_h40_observation_path(o_srnn_h40_observation_path),
        .o_h40_observation_sample_id(o_srnn_h40_observation_sample_id),
        .o_h40_observation_backbone_id(
            o_srnn_h40_observation_backbone_id),
        .o_h40_observation_q14(o_srnn_h40_observation_q14),
        .o_h40_observation_last(o_srnn_h40_observation_last),
        .o_h40_observation_done(o_srnn_h40_observation_done),
        .i_h40_hidden_valid(i_srnn_h40_hidden_valid),
        .o_h40_hidden_ready(o_srnn_h40_hidden_ready),
        .i_h40_hidden_path(i_srnn_h40_hidden_path),
        .i_h40_hidden_sample_id(i_srnn_h40_hidden_sample_id),
        .i_h40_hidden_id(i_srnn_h40_hidden_id),
        .i_h40_hidden_membrane_pre_q14(
            i_srnn_h40_hidden_membrane_pre_q14),
        .i_h40_hidden_membrane_next_q14(
            i_srnn_h40_hidden_membrane_next_q14),
        .i_h40_hidden_spike(i_srnn_h40_hidden_spike),
        .i_h40_hidden_trace_q14(i_srnn_h40_hidden_trace_q14),
        .i_h40_hidden_output_feature_q14(
            i_srnn_h40_hidden_output_feature_q14),
        .i_h40_hidden_last(i_srnn_h40_hidden_last),
        .i_h40_logit_valid(i_srnn_h40_logit_valid),
        .o_h40_logit_ready(o_srnn_h40_logit_ready),
        .i_h40_logit_sample_id(i_srnn_h40_logit_sample_id),
        .i_h40_logit_class_id(i_srnn_h40_logit_class_id),
        .i_h40_logit_q28(i_srnn_h40_logit_q28),
        .i_h40_logit_last(i_srnn_h40_logit_last),
        .i_h40_timestep_id(i_srnn_h40_timestep_id),
        .i_h40_timestep_done(i_srnn_h40_timestep_done),
        .i_h40_busy(i_srnn_h40_busy),
        .i_h40_sticky_error(i_srnn_h40_sticky_error),
        .i_h40_protocol_error(i_srnn_h40_protocol_error),
        .i_h40_overflow(i_srnn_h40_overflow),
        .o_hidden_observation_valid(), .i_hidden_observation_ready(1'b1),
        .o_hidden_observation_path(), .o_hidden_observation_sample_id(),
        .o_hidden_observation_hidden_id(), .o_hidden_membrane_pre_q14(),
        .o_hidden_membrane_next_q14(), .o_hidden_spike(),
        .o_hidden_trace_q14(), .o_hidden_output_feature_q14(),
        .o_hidden_observation_last(),
        .o_logit_valid(srnn_logit_valid_internal),
        .i_logit_ready(i_srnn_logit_ready && srnn_selected),
        .o_logit_sample_id(o_srnn_logit_sample_id),
        .o_logit_class_id(o_srnn_logit_class_id),
        .o_logit_q28(o_srnn_logit_q28),
        .o_logit_last(o_srnn_logit_last),
        .o_label_replay_start(o_srnn_label_replay_start),
        .i_label_replay_start_ready(i_srnn_label_replay_start_ready),
        .i_label_valid(i_srnn_label_valid && srnn_selected),
        .o_label_ready(srnn_label_ready_internal),
        .i_label_sample_id(i_srnn_label_sample_id),
        .i_label_class_id(i_srnn_label_class_id),
        .i_label_last(i_srnn_label_last),
        .i_restore_complete(i_srnn_restore_complete && srnn_selected),
        .o_restore_ready(srnn_restore_ready_internal),
        .o_recovery_required(o_srnn_recovery_required),
        .o_learning_commit_pulse(o_srnn_learning_commit),
        .o_detach_pulse(o_srnn_detach), .o_model_version(),
        .o_adapt_version(), .o_forward_model_version(),
        .o_forward_adapt_version(), .o_learning_base_model_version(),
        .o_learning_base_adapt_version(),
        .o_learning_commit_window_sequence(), .o_commit_code_dirty(),
        .o_commit_optimizer_dirty(), .o_learning_result_dirty(),
        .o_learning_saturated(), .o_timestep_done(o_srnn_timestep_done),
        .o_timestep_aborted(o_srnn_timestep_aborted), .o_busy(o_srnn_busy),
        .o_error(o_srnn_error),
        .o_zero_gradient_count(o_srnn_zero_gradient_count),
        .o_update_count(o_srnn_update_count),
        .o_store_load_valid(srnn_store_load_valid),
        .i_store_load_ready(srnn_store_load_ready),
        .o_store_load_addr(srnn_store_load_addr),
        .o_store_load_code(srnn_store_load_code),
        .o_store_load_residual(srnn_store_load_residual),
        .o_store_learning_read_valid(srnn_store_read_valid),
        .i_store_learning_read_ready(srnn_store_read_ready),
        .o_store_learning_read_addr(srnn_store_read_addr),
        .o_store_learning_read_tag(srnn_store_read_tag),
        .i_store_learning_response_valid(srnn_store_response_valid),
        .o_store_learning_response_ready(srnn_store_response_ready),
        .i_store_learning_response_tag(
            learning_response_tag[TAG_WIDTH +: TAG_WIDTH]),
        .i_store_learning_response_code(
            learning_response_code[CODE_WIDTH +: CODE_WIDTH]),
        .i_store_learning_response_residual(
            learning_response_residual[RESIDUAL_WIDTH +: RESIDUAL_WIDTH]),
        .i_store_learning_response_error(learning_response_error[1]),
        .o_store_learning_write_valid(srnn_store_write_valid),
        .i_store_learning_write_ready(srnn_store_write_ready),
        .o_store_learning_write_addr(srnn_store_write_addr),
        .o_store_learning_write_code(srnn_store_write_code),
        .o_store_learning_write_residual(srnn_store_write_residual),
        .o_store_learning_write_code_enable(srnn_store_write_code_enable),
        .o_store_learning_write_residual_enable(
            srnn_store_write_residual_enable),
        .o_store_forward_read_valid(srnn_store_forward_valid),
        .i_store_forward_read_ready(srnn_store_forward_ready),
        .o_store_forward_read_addr(srnn_store_forward_addr),
        .o_store_forward_read_tag(srnn_store_forward_tag),
        .i_store_forward_response_valid(srnn_store_forward_response_valid),
        .o_store_forward_response_ready(srnn_store_forward_response_ready),
        .i_store_forward_response_tag(
            forward_response_tag[TAG_WIDTH +: TAG_WIDTH]),
        .i_store_forward_response_code(
            forward_response_code[CODE_WIDTH +: CODE_WIDTH]),
        .i_store_forward_response_error(forward_response_error[1]),
        .i_store_protocol_error(i_store_protocol_error)
    );
    end else begin : g_no_srnn_client
        assign srnn_cfg_ready_internal = 1'b0;
        assign o_srnn_cfg_error = 1'b0;
        assign srnn_load_ready_internal = 1'b0;
        assign srnn_timestep_start_ready_internal = 1'b0;
        assign srnn_abort_ready_internal = 1'b0;
        assign srnn_backbone_ready_internal = 1'b0;
        assign o_srnn_target_trace_start = 1'b0;
        assign o_srnn_target_trace_timestep_id = '0;
        assign srnn_target_trace_ready_internal = 1'b0;
        assign o_srnn_h40_cfg_valid = 1'b0;
        assign o_srnn_h40_cfg_batch_count = '0;
        assign o_srnn_h40_cfg_backbone_count = '0;
        assign o_srnn_h40_cfg_hidden_count = '0;
        assign o_srnn_h40_cfg_class_count = '0;
        assign o_srnn_h40_cfg_wh1_parameter_base = '0;
        assign o_srnn_h40_cfg_wout_parameter_base = '0;
        assign o_srnn_h40_prepare_valid = 1'b0;
        assign o_srnn_h40_prepare_version = '0;
        assign o_srnn_h40_cache_invalidate = 1'b0;
        assign o_srnn_h40_forward_read_ready = 1'b0;
        assign o_srnn_h40_forward_response_valid = 1'b0;
        assign o_srnn_h40_forward_response_tag = '0;
        assign o_srnn_h40_forward_response_code = '0;
        assign o_srnn_h40_forward_response_error = 1'b0;
        assign o_srnn_h40_timestep_start = 1'b0;
        assign o_srnn_h40_timestep_id = '0;
        assign o_srnn_h40_sequence_version = '0;
        assign o_srnn_h40_include_target_path = 1'b0;
        assign o_srnn_h40_observation_valid = 1'b0;
        assign o_srnn_h40_observation_path = 1'b0;
        assign o_srnn_h40_observation_sample_id = '0;
        assign o_srnn_h40_observation_backbone_id = '0;
        assign o_srnn_h40_observation_q14 = '0;
        assign o_srnn_h40_observation_last = 1'b0;
        assign o_srnn_h40_observation_done = 1'b0;
        assign o_srnn_h40_hidden_ready = 1'b0;
        assign o_srnn_h40_logit_ready = 1'b0;
        assign srnn_logit_valid_internal = 1'b0;
        assign o_srnn_logit_sample_id = '0;
        assign o_srnn_logit_class_id = '0;
        assign o_srnn_logit_q28 = '0;
        assign o_srnn_logit_last = 1'b0;
        assign o_srnn_label_replay_start = 1'b0;
        assign srnn_label_ready_internal = 1'b0;
        assign srnn_restore_ready_internal = 1'b0;
        assign o_srnn_recovery_required = 1'b0;
        assign o_srnn_learning_commit = 1'b0;
        assign o_srnn_detach = 1'b0;
        assign o_srnn_timestep_done = 1'b0;
        assign o_srnn_timestep_aborted = 1'b0;
        assign o_srnn_busy = 1'b0;
        assign o_srnn_error = 1'b0;
        assign o_srnn_zero_gradient_count = '0;
        assign o_srnn_update_count = '0;
        assign srnn_store_load_valid = 1'b0;
        assign srnn_store_load_addr = '0;
        assign srnn_store_load_code = '0;
        assign srnn_store_load_residual = '0;
        assign srnn_store_read_valid = 1'b0;
        assign srnn_store_read_addr = '0;
        assign srnn_store_read_tag = '0;
        assign srnn_store_response_ready = 1'b0;
        assign srnn_store_write_valid = 1'b0;
        assign srnn_store_write_addr = '0;
        assign srnn_store_write_code = '0;
        assign srnn_store_write_residual = '0;
        assign srnn_store_write_code_enable = 1'b0;
        assign srnn_store_write_residual_enable = 1'b0;
        assign srnn_store_forward_valid = 1'b0;
        assign srnn_store_forward_addr = '0;
        assign srnn_store_forward_tag = '0;
        assign srnn_store_forward_response_ready = 1'b0;
    end
    endgenerate

    assign load_valid = {i_scnn_store_load_valid,
        srnn_store_load_valid, i_fc_parameter_load_valid};
    assign load_addr = {i_scnn_store_load_addr, srnn_store_load_addr,
        i_fc_parameter_load_addr};
    assign load_code = {i_scnn_store_load_code, srnn_store_load_code,
        i_fc_parameter_load_code};
    assign load_residual = {i_scnn_store_load_residual,
        srnn_store_load_residual, i_fc_parameter_load_residual};
    assign o_fc_parameter_load_ready = load_ready[0];
    assign srnn_store_load_ready = load_ready[1];
    assign o_scnn_store_load_ready = load_ready[2];

    assign learning_read_valid = {scnn_merged_read_valid,
        srnn_store_read_valid, fc_read_valid};
    assign learning_read_addr = {scnn_merged_read_addr,
        srnn_store_read_addr, fc_read_addr};
    assign learning_read_tag = {scnn_merged_read_tag,
        srnn_store_read_tag, fc_read_tag};
    assign fc_read_ready = learning_read_ready[0];
    assign srnn_store_read_ready = learning_read_ready[1];
    assign scnn_merged_read_ready = learning_read_ready[2];
    assign learning_response_ready = {
        scnn_merged_response_ready,
        srnn_store_response_ready, fc_read_response_ready};
    assign fc_read_response_valid = learning_response_valid[0];
    assign srnn_store_response_valid = learning_response_valid[1];
    assign scnn_merged_response_valid = learning_response_valid[2];

    assign learning_write_valid = {scnn_merged_write_valid,
        srnn_store_write_valid, fc_result_valid};
    assign learning_write_addr = {scnn_merged_write_addr,
        srnn_store_write_addr, fc_result_addr};
    assign learning_write_code = {scnn_merged_write_code,
        srnn_store_write_code, fc_result_code};
    assign learning_write_residual = {
        scnn_merged_write_residual,
        srnn_store_write_residual, fc_result_residual};
    assign learning_write_code_enable = {
        scnn_merged_write_code_enable,
        srnn_store_write_code_enable, fc_write_code_enable};
    assign learning_write_residual_enable = {
        scnn_merged_write_residual_enable,
        srnn_store_write_residual_enable, fc_write_residual_enable};
    assign fc_result_ready = learning_write_ready[0];
    assign srnn_store_write_ready = learning_write_ready[1];
    assign scnn_merged_write_ready = learning_write_ready[2];

    assign forward_read_valid = {i_scnn_store_forward_read_valid,
        srnn_store_forward_valid, i_fc_forward_read_valid};
    assign forward_read_addr = {i_scnn_store_forward_read_addr,
        srnn_store_forward_addr, i_fc_forward_read_addr};
    assign forward_read_tag = {i_scnn_store_forward_read_tag,
        srnn_store_forward_tag, i_fc_forward_read_tag};
    assign o_fc_forward_read_ready = forward_read_ready[0];
    assign srnn_store_forward_ready = forward_read_ready[1];
    assign o_scnn_store_forward_read_ready = forward_read_ready[2];
    assign forward_response_ready = {i_scnn_store_forward_response_ready,
        srnn_store_forward_response_ready, i_fc_forward_response_ready};
    assign o_fc_forward_response_valid = forward_response_valid[0];
    assign o_fc_forward_response_tag =
        forward_response_tag[0 +: TAG_WIDTH];
    assign o_fc_forward_response_code =
        forward_response_code[0 +: CODE_WIDTH];
    assign o_fc_forward_response_error = forward_response_error[0];
    assign srnn_store_forward_response_valid = forward_response_valid[1];
    assign o_scnn_store_forward_response_valid = forward_response_valid[2];
    assign o_scnn_store_forward_response_tag =
        forward_response_tag[2*TAG_WIDTH +: TAG_WIDTH];
    assign o_scnn_store_forward_response_code =
        forward_response_code[2*CODE_WIDTH +: CODE_WIDTH];
    assign o_scnn_store_forward_response_error = forward_response_error[2];

    unified_tp_external_parameter_backend #(
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH), .RESIDUAL_WIDTH(RESIDUAL_WIDTH)
    ) u_parameter_backend (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_parameter_owner(i_parameter_owner),
        .i_topology_learning_enable(i_topology_learning_enable),
        .i_load_valid(load_valid), .o_load_ready(load_ready),
        .i_load_addr(load_addr), .i_load_code(load_code),
        .i_load_residual(load_residual),
        .i_learning_read_valid(learning_read_valid),
        .o_learning_read_ready(learning_read_ready),
        .i_learning_read_addr(learning_read_addr),
        .i_learning_read_tag(learning_read_tag),
        .o_learning_response_valid(learning_response_valid),
        .i_learning_response_ready(learning_response_ready),
        .o_learning_response_tag(learning_response_tag),
        .o_learning_response_code(learning_response_code),
        .o_learning_response_residual(learning_response_residual),
        .o_learning_response_error(learning_response_error),
        .i_learning_write_valid(learning_write_valid),
        .o_learning_write_ready(learning_write_ready),
        .i_learning_write_addr(learning_write_addr),
        .i_learning_write_code(learning_write_code),
        .i_learning_write_residual(learning_write_residual),
        .i_learning_write_code_enable(learning_write_code_enable),
        .i_learning_write_residual_enable(
            learning_write_residual_enable),
        .i_forward_read_valid(forward_read_valid),
        .o_forward_read_ready(forward_read_ready),
        .i_forward_read_addr(forward_read_addr),
        .i_forward_read_tag(forward_read_tag),
        .o_forward_response_valid(forward_response_valid),
        .i_forward_response_ready(forward_response_ready),
        .o_forward_response_tag(forward_response_tag),
        .o_forward_response_code(forward_response_code),
        .o_forward_response_error(forward_response_error),
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
        .i_store_quiescent(i_store_quiescent),
        .i_store_protocol_error(i_store_protocol_error),
        .o_backend_quiescent(backend_quiescent_internal),
        .o_backend_error(backend_error_internal),
        .o_owner_encoding_error(o_owner_encoding_error),
        .o_router_protocol_error(o_router_protocol_error)
    );

    initial begin
        if (PARAM_ADDR_WIDTH != 24 || FC_MAX_HIDDEN < 160 ||
            SRNN_BACKBONE_MAX < 450 || SRNN_HIDDEN_MAX < 256 ||
            SRNN_CLASS_MAX < 20)
            $error("C17 learning/parameter plane geometry is invalid");
    end
endmodule

`default_nettype wire
