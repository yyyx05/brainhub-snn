`timescale 1ns/1ps
`default_nettype none

// Coordinates one SRNN M2 timestep around the shared W_h1/W_out parameter
// store.  One shared M2 numeric consumer uses the same forward parameter scan
// and observation stream as learning; no second trainable parameter copy or
// optimizer state is introduced.
module unified_srnn_m2_timestep_train_infer_coordinator #(
    parameter bit USE_INTERNAL_STORE = 1'b1,
    parameter bit USE_EXTERNAL_BACKBONE_STREAM = 1'b0,
    parameter bit USE_SHARED_H40_FORWARD = 1'b0,
    parameter bit ENABLE_WH1_LEARNING = 1'b1,
    parameter int STORE_DEPTH = 1024,
    parameter int BATCH_MAX = 64,
    parameter int TIMESTEP_MAX = 128,
    parameter int HIDDEN_MAX = 256,
    parameter int BACKBONE_MAX = 450,
    parameter int CLASS_MAX = 20,
    parameter int SOURCE_VECTOR_MAX = 480,
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int SAMPLE_ID_WIDTH = 8,
    parameter int TIMESTEP_ID_WIDTH = $clog2(TIMESTEP_MAX),
    parameter int HIDDEN_INDEX_WIDTH = 9,
    parameter int BACKBONE_INDEX_WIDTH = 10,
    parameter int CLASS_INDEX_WIDTH = 5,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int VERSION_WIDTH = 16,
    parameter int WINDOW_SEQUENCE_WIDTH = 32
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]               i_cfg_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_cfg_wh1_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_wh1_base,
    input  logic [VOTE_WIDTH-1:0]                     i_cfg_wh1_increment,
    input  logic [VOTE_WIDTH-1:0]                     i_cfg_wh1_threshold,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_cfg_wout_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_wout_base,
    input  logic [VOTE_WIDTH-1:0]                     i_cfg_wout_increment,
    input  logic [VOTE_WIDTH-1:0]                     i_cfg_wout_threshold,
    input  logic signed [CODE_WIDTH-1:0]              i_cfg_code_min,
    input  logic signed [CODE_WIDTH-1:0]              i_cfg_code_max,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_cfg_batch_count,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]             i_cfg_hidden_count,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_cfg_backbone_count,
    input  logic [CLASS_INDEX_WIDTH-1:0]              i_cfg_class_count,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_wout_element_count,
    output logic                                      o_cfg_error,

    input  logic                                      i_load_valid,
    output logic                                      o_load_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_load_addr,
    input  logic signed [CODE_WIDTH-1:0]              i_load_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]          i_load_residual,

    input  logic                                      i_timestep_start,
    output logic                                      o_timestep_start_ready,
    input  logic [TIMESTEP_ID_WIDTH-1:0]              i_timestep_id,
    input  logic [VERSION_WIDTH-1:0]                  i_sequence_version,
    input  logic                                      i_learning_enable,
    input  logic                                      i_abort_timestep,
    output logic                                      o_abort_ready,

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

    // The native H40 array already emits ordered H450 Q2.14 observations.
    // Enabling this boundary bypasses only the spike-vector-to-backbone
    // replay window; M2 forward, TP/CE learning and parameter RMW are shared.
    input  logic                                      i_external_backbone_valid,
    output logic                                      o_external_backbone_ready,
    input  logic                                      i_external_backbone_path,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_external_backbone_sample_id,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_external_backbone_id,
    input  logic signed [31:0]                        i_external_backbone_q14,
    input  logic                                      i_external_backbone_last,
    input  logic                                      i_external_backbone_done,
    input  logic                                      i_external_backbone_error,

    output logic                                      o_target_trace_start,
    input  logic                                      i_target_trace_start_ready,
    output logic [TIMESTEP_ID_WIDTH-1:0]              o_target_trace_timestep_id,
    input  logic                                      i_target_trace_valid,
    output logic                                      o_target_trace_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_target_trace_sample_id,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_target_trace_backbone_id,
    input  logic signed [31:0]                        i_target_trace_q14,
    input  logic                                      i_target_trace_last,
    input  logic                                      i_target_trace_done,
    input  logic                                      i_target_trace_error,

    output logic                                      o_target_trace_valid,
    input  logic                                      i_target_trace_ready,
    output logic [SAMPLE_ID_WIDTH-1:0]                o_target_trace_sample_id,
    output logic [BACKBONE_INDEX_WIDTH-1:0]           o_target_trace_backbone_id,
    output logic signed [31:0]                        o_target_trace_q14,
    output logic                                      o_target_trace_last,
    output logic                                      o_target_trace_done,

    output logic                                      o_forward_code_valid,
    input  logic                                      i_forward_code_ready,
    output logic [TENSOR_ID_WIDTH-1:0]                o_forward_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_forward_local_addr,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_forward_parameter_addr,
    output logic signed [CODE_WIDTH-1:0]              o_forward_code,
    output logic                                      o_forward_last,
    output logic                                      o_forward_parameter_scan_done,

    // Shared-H40 forward boundary.  The physical engine remains beside the
    // common array; this coordinator owns sequencing and the parameter-store
    // read service without introducing another trainable parameter copy.
    output logic                                      o_h40_cfg_valid,
    input  logic                                      i_h40_cfg_ready,
    input  logic                                      i_h40_cfg_error,
    output logic [SAMPLE_ID_WIDTH-1:0]                o_h40_cfg_batch_count,
    output logic [BACKBONE_INDEX_WIDTH-1:0]           o_h40_cfg_backbone_count,
    output logic [HIDDEN_INDEX_WIDTH-1:0]             o_h40_cfg_hidden_count,
    output logic [CLASS_INDEX_WIDTH-1:0]              o_h40_cfg_class_count,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_h40_cfg_wh1_parameter_base,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_h40_cfg_wout_parameter_base,
    output logic                                      o_h40_prepare_valid,
    input  logic                                      i_h40_prepare_ready,
    output logic [VERSION_WIDTH-1:0]                  o_h40_prepare_version,
    input  logic                                      i_h40_prepare_done,
    input  logic                                      i_h40_cache_valid,
    output logic                                      o_h40_cache_invalidate,
    output logic                                      o_h40_wh1_cache_invalidate,
    input  logic                                      i_h40_forward_read_valid,
    output logic                                      o_h40_forward_read_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_h40_forward_read_addr,
    input  logic [TAG_WIDTH-1:0]                      i_h40_forward_read_tag,
    output logic                                      o_h40_forward_response_valid,
    input  logic                                      i_h40_forward_response_ready,
    output logic [TAG_WIDTH-1:0]                      o_h40_forward_response_tag,
    output logic signed [CODE_WIDTH-1:0]              o_h40_forward_response_code,
    output logic                                      o_h40_forward_response_error,
    output logic                                      o_h40_timestep_start,
    input  logic                                      i_h40_timestep_start_ready,
    output logic [TIMESTEP_ID_WIDTH-1:0]              o_h40_timestep_id,
    output logic [VERSION_WIDTH-1:0]                  o_h40_sequence_version,
    output logic                                      o_h40_include_target_path,
    output logic                                      o_h40_observation_valid,
    input  logic                                      i_h40_observation_ready,
    output logic                                      o_h40_observation_path,
    output logic [SAMPLE_ID_WIDTH-1:0]                o_h40_observation_sample_id,
    output logic [BACKBONE_INDEX_WIDTH-1:0]           o_h40_observation_backbone_id,
    output logic signed [31:0]                        o_h40_observation_q14,
    output logic                                      o_h40_observation_last,
    output logic                                      o_h40_observation_done,
    input  logic                                      i_h40_hidden_valid,
    output logic                                      o_h40_hidden_ready,
    input  logic                                      i_h40_hidden_path,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_h40_hidden_sample_id,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]             i_h40_hidden_id,
    input  logic signed [31:0]                        i_h40_hidden_membrane_pre_q14,
    input  logic signed [31:0]                        i_h40_hidden_membrane_next_q14,
    input  logic                                      i_h40_hidden_spike,
    input  logic signed [31:0]                        i_h40_hidden_trace_q14,
    input  logic signed [31:0]                        i_h40_hidden_output_feature_q14,
    input  logic                                      i_h40_hidden_last,
    input  logic                                      i_h40_logit_valid,
    output logic                                      o_h40_logit_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_h40_logit_sample_id,
    input  logic [CLASS_INDEX_WIDTH-1:0]              i_h40_logit_class_id,
    input  logic signed [63:0]                        i_h40_logit_q28,
    input  logic                                      i_h40_logit_last,
    input  logic [TIMESTEP_ID_WIDTH-1:0]              i_h40_timestep_id,
    input  logic                                      i_h40_timestep_done,
    input  logic                                      i_h40_busy,
    input  logic                                      i_h40_sticky_error,
    input  logic                                      i_h40_protocol_error,
    input  logic                                      i_h40_overflow,

    output logic                                      o_hidden_observation_valid,
    input  logic                                      i_hidden_observation_ready,
    output logic                                      o_hidden_observation_path,
    output logic [SAMPLE_ID_WIDTH-1:0]                o_hidden_observation_sample_id,
    output logic [HIDDEN_INDEX_WIDTH-1:0]             o_hidden_observation_hidden_id,
    output logic signed [31:0]                        o_hidden_membrane_pre_q14,
    output logic signed [31:0]                        o_hidden_membrane_next_q14,
    output logic                                      o_hidden_spike,
    output logic signed [31:0]                        o_hidden_trace_q14,
    output logic signed [31:0]                        o_hidden_output_feature_q14,
    output logic                                      o_hidden_observation_last,

    output logic                                      o_logit_valid,
    input  logic                                      i_logit_ready,
    output logic [SAMPLE_ID_WIDTH-1:0]                o_logit_sample_id,
    output logic [CLASS_INDEX_WIDTH-1:0]              o_logit_class_id,
    output logic signed [63:0]                        o_logit_q28,
    output logic                                      o_logit_last,

    output logic                                      o_label_replay_start,
    input  logic                                      i_label_replay_start_ready,
    input  logic                                      i_label_valid,
    output logic                                      o_label_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_label_sample_id,
    input  logic [CLASS_INDEX_WIDTH-1:0]              i_label_class_id,
    input  logic                                      i_label_last,

    input  logic                                      i_restore_complete,
    output logic                                      o_restore_ready,
    output logic                                      o_recovery_required,
    output logic                                      o_learning_commit_pulse,
    output logic                                      o_detach_pulse,
    output logic [VERSION_WIDTH-1:0]                  o_model_version,
    output logic [VERSION_WIDTH-1:0]                  o_adapt_version,
    output logic [VERSION_WIDTH-1:0]                  o_forward_model_version,
    output logic [VERSION_WIDTH-1:0]                  o_forward_adapt_version,
    output logic [VERSION_WIDTH-1:0]                  o_learning_base_model_version,
    output logic [VERSION_WIDTH-1:0]                  o_learning_base_adapt_version,
    output logic [WINDOW_SEQUENCE_WIDTH-1:0]          o_learning_commit_window_sequence,
    output logic                                      o_commit_code_dirty,
    output logic                                      o_commit_optimizer_dirty,
    output logic                                      o_learning_result_dirty,
    output logic                                      o_learning_saturated,
    output logic                                      o_timestep_done,
    output logic                                      o_timestep_aborted,
    output logic                                      o_busy,
    output logic                                      o_error,
    output logic [PARAM_ADDR_WIDTH:0]                 o_zero_gradient_count,
    output logic [PARAM_ADDR_WIDTH:0]                 o_update_count,

    output logic                                      o_store_load_valid,
    input  logic                                      i_store_load_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_store_load_addr,
    output logic signed [CODE_WIDTH-1:0]              o_store_load_code,
    output logic signed [RESIDUAL_WIDTH-1:0]          o_store_load_residual,
    output logic                                      o_store_learning_read_valid,
    input  logic                                      i_store_learning_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_store_learning_read_addr,
    output logic [TAG_WIDTH-1:0]                      o_store_learning_read_tag,
    input  logic                                      i_store_learning_response_valid,
    output logic                                      o_store_learning_response_ready,
    input  logic [TAG_WIDTH-1:0]                      i_store_learning_response_tag,
    input  logic signed [CODE_WIDTH-1:0]              i_store_learning_response_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]          i_store_learning_response_residual,
    input  logic                                      i_store_learning_response_error,
    output logic                                      o_store_learning_write_valid,
    input  logic                                      i_store_learning_write_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_store_learning_write_addr,
    output logic signed [CODE_WIDTH-1:0]              o_store_learning_write_code,
    output logic signed [RESIDUAL_WIDTH-1:0]          o_store_learning_write_residual,
    output logic                                      o_store_learning_write_code_enable,
    output logic                                      o_store_learning_write_residual_enable,
    output logic                                      o_store_forward_read_valid,
    input  logic                                      i_store_forward_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_store_forward_read_addr,
    output logic [TAG_WIDTH-1:0]                      o_store_forward_read_tag,
    input  logic                                      i_store_forward_response_valid,
    output logic                                      o_store_forward_response_ready,
    input  logic [TAG_WIDTH-1:0]                      i_store_forward_response_tag,
    input  logic signed [CODE_WIDTH-1:0]              i_store_forward_response_code,
    input  logic                                      i_store_forward_response_error,
    input  logic                                      i_store_protocol_error
);

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_FORWARD,
        STATE_PREPARE_ISSUE,
        STATE_PREPARE_WAIT,
        STATE_FORWARD_CLOSE,
        STATE_START_LEARNING,
        STATE_LEARNING,
        STATE_ERROR
    } state_t;

    state_t state_q;
    logic configured_q;
    logic learning_enable_q;
    logic [TIMESTEP_ID_WIDTH-1:0] timestep_q;
    logic [VERSION_WIDTH-1:0] sequence_version_q;
    logic observation_done_q;
    logic target_trace_started_q;
    logic target_trace_done_q;
    logic learning_done_q;
    logic commit_seen_q;
    logic detach_seen_q;
    logic numeric_done_q;
    logic contribution_done_q;

    logic store_cfg_ready;
    logic store_cfg_error;
    logic observation_cfg_ready;
    logic observation_cfg_error;
    logic numeric_cfg_ready;
    logic numeric_cfg_error;
    logic shared_cfg_fire;

    logic store_forward_start;
    logic store_forward_start_ready;
    logic store_forward_done;
    logic store_forward_error;
    logic store_forward_code_valid;
    logic store_forward_code_ready;
    logic [TENSOR_ID_WIDTH-1:0] store_forward_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] store_forward_local_addr;
    logic [PARAM_ADDR_WIDTH-1:0] store_forward_parameter_addr;
    logic signed [CODE_WIDTH-1:0] store_forward_code;
    logic store_forward_last;
    logic store_forward_consumer_done_ready;
    logic store_forward_consumer_done_valid;
    logic store_learning_start;
    logic store_learning_start_ready;
    logic store_learning_abort;
    logic store_learning_done;
    logic store_learning_aborted;
    logic store_busy;
    logic store_error;

    logic observation_start;
    logic observation_start_ready;
    logic observation_abort;
    logic observation_backbone_valid;
    logic observation_backbone_ready;
    logic observation_backbone_path;
    logic [SAMPLE_ID_WIDTH-1:0] observation_backbone_sample;
    logic [BACKBONE_INDEX_WIDTH-1:0] observation_backbone_id;
    logic signed [31:0] observation_backbone_q14;
    logic observation_backbone_last;
    logic observation_busy;
    logic observation_done;
    logic observation_error;
    logic window_cfg_ready;
    logic window_cfg_error;
    logic window_start_ready;
    logic window_backbone_valid;
    logic window_backbone_ready;
    logic window_backbone_path;
    logic [SAMPLE_ID_WIDTH-1:0] window_backbone_sample;
    logic [BACKBONE_INDEX_WIDTH-1:0] window_backbone_id;
    logic signed [31:0] window_backbone_q14;
    logic window_backbone_last;
    logic window_busy;
    logic window_done;
    logic window_error;

    logic numeric_parameter_valid;
    logic numeric_parameter_ready;
    logic numeric_parameter_scan_done;
    logic numeric_parameter_done_pending_q;
    logic numeric_timestep_start_ready;
    logic numeric_observation_valid;
    logic numeric_observation_ready;
    logic [TIMESTEP_ID_WIDTH-1:0] numeric_timestep_id;
    logic numeric_timestep_done;
    logic numeric_busy;
    logic numeric_protocol_error;
    logic numeric_overflow;
    logic numeric_timestep_id_error;
    logic numeric_hidden_valid;
    logic numeric_hidden_ready;
    logic numeric_hidden_path;
    logic [SAMPLE_ID_WIDTH-1:0] numeric_hidden_sample_id;
    logic [HIDDEN_INDEX_WIDTH-1:0] numeric_hidden_id;
    logic signed [31:0] numeric_hidden_membrane_pre_q14;
    logic signed [31:0] numeric_hidden_membrane_next_q14;
    logic numeric_hidden_spike;
    logic signed [31:0] numeric_hidden_trace_q14;
    logic signed [31:0] numeric_hidden_output_feature_q14;
    logic numeric_hidden_last;
    logic numeric_logit_valid;
    logic numeric_logit_ready;
    logic [SAMPLE_ID_WIDTH-1:0] numeric_logit_sample_id;
    logic [CLASS_INDEX_WIDTH-1:0] numeric_logit_class_id;
    logic signed [63:0] numeric_logit_q28;
    logic numeric_logit_last;

    logic contribution_cfg_ready;
    logic contribution_cfg_error;
    logic contribution_start_ready;
    logic contribution_hidden_valid;
    logic contribution_hidden_ready;
    logic contribution_target_trace_valid;
    logic contribution_target_trace_ready;
    logic contribution_logit_valid;
    logic contribution_logit_ready;
    logic contribution_label_valid;
    logic contribution_label_ready;
    logic contribution_wh1_valid;
    logic contribution_wh1_ready;
    logic contribution_wh1_path;
    logic [SAMPLE_ID_WIDTH-1:0] contribution_wh1_sample_id;
    logic [HIDDEN_INDEX_WIDTH-1:0] contribution_wh1_hidden_id;
    logic [SAMPLE_ID_WIDTH-1:0] contribution_wh1_column_id;
    logic signed [63:0] contribution_wh1_error_q40;
    logic signed [31:0] contribution_wh1_opposite_trace_q14;
    logic [24:0] contribution_wh1_surrogate_uq24;
    logic contribution_wh1_last;
    logic contribution_wout_valid;
    logic contribution_wout_ready;
    logic [PARAM_ADDR_WIDTH-1:0] contribution_wout_addr;
    logic [SAMPLE_ID_WIDTH-1:0] contribution_wout_sample_id;
    logic signed [63:0] contribution_wout_error_q40;
    logic signed [31:0] contribution_wout_hidden_trace_q14;
    logic contribution_wout_last;
    logic contribution_busy;
    logic contribution_done;
    logic contribution_aborted;
    logic contribution_protocol_error;
    logic contribution_overflow;
    logic [PARAM_ADDR_WIDTH:0] cfg_expected_wout_count;
    logic cfg_shape_legal;

    logic start_fire;
    logic learning_pair_fire;
    logic abort_fire;
    logic completion_seen;
    logic target_trace_start_fire;
    logic store_wh1_backbone_ready;

    always_comb begin
        cfg_expected_wout_count =
            (PARAM_ADDR_WIDTH + 1)'(i_cfg_hidden_count) *
            (PARAM_ADDR_WIDTH + 1)'(i_cfg_class_count);
        cfg_shape_legal = !cfg_expected_wout_count[PARAM_ADDR_WIDTH] &&
            (cfg_expected_wout_count ==
                {1'b0, i_cfg_wout_element_count});
        o_cfg_ready = (state_q == STATE_IDLE) && store_cfg_ready &&
            observation_cfg_ready && numeric_cfg_ready &&
            contribution_cfg_ready;
        shared_cfg_fire = i_cfg_valid && o_cfg_ready && cfg_shape_legal;
        o_cfg_error = store_cfg_error || observation_cfg_error ||
            numeric_cfg_error || contribution_cfg_error ||
            (i_cfg_valid && o_cfg_ready && !cfg_shape_legal);

        o_timestep_start_ready = (state_q == STATE_IDLE) && configured_q &&
            store_forward_start_ready && !i_cfg_valid && !o_cfg_error &&
            !o_recovery_required && (i_sequence_version != '0);
        start_fire = i_timestep_start && o_timestep_start_ready;
        store_forward_start = start_fire;

        learning_pair_fire = (state_q == STATE_START_LEARNING) &&
            observation_start_ready && numeric_timestep_start_ready &&
            (!learning_enable_q || (store_learning_start_ready &&
                contribution_start_ready && i_label_replay_start_ready)) &&
            !i_abort_timestep;
        store_learning_start = learning_pair_fire && learning_enable_q;
        observation_start = learning_pair_fire;
        o_label_replay_start = learning_pair_fire && learning_enable_q;

        o_abort_ready = (state_q == STATE_START_LEARNING) ||
            (state_q == STATE_LEARNING);
        abort_fire = i_abort_timestep && o_abort_ready;
        store_learning_abort = abort_fire && (state_q == STATE_LEARNING) &&
            learning_enable_q;
        observation_abort = abort_fire && (state_q == STATE_LEARNING);

        observation_backbone_ready = (state_q == STATE_LEARNING) ?
            (numeric_observation_ready && (!learning_enable_q ||
                !ENABLE_WH1_LEARNING ||
                store_wh1_backbone_ready)) : 1'b0;
        numeric_observation_valid = observation_backbone_valid &&
            (!learning_enable_q || !ENABLE_WH1_LEARNING ||
                store_wh1_backbone_ready) &&
            (state_q == STATE_LEARNING);

        numeric_parameter_valid = store_forward_code_valid &&
            i_forward_code_ready;
        store_forward_code_ready = i_forward_code_ready &&
            numeric_parameter_ready;
        o_forward_code_valid = store_forward_code_valid &&
            numeric_parameter_ready;
        o_forward_tensor_id = store_forward_tensor_id;
        o_forward_local_addr = store_forward_local_addr;
        o_forward_parameter_addr = store_forward_parameter_addr;
        o_forward_code = store_forward_code;
        o_forward_last = store_forward_last;
        store_forward_consumer_done_valid =
            USE_SHARED_H40_FORWARD ?
                ((state_q == STATE_PREPARE_WAIT) && i_h40_prepare_done) :
                (numeric_parameter_done_pending_q ||
                    numeric_parameter_scan_done);
        // H40 busy includes cache preparation, before the adapter latches the
        // new timestep. Keep checking the ID throughout the launched window.
        numeric_timestep_id_error = (numeric_timestep_id != timestep_q) &&
            (USE_SHARED_H40_FORWARD ? state_q == STATE_LEARNING : numeric_busy);

        // Atomic forks keep the public observation ports and the real learning
        // frontend aligned under independent backpressure.
        numeric_hidden_ready = i_hidden_observation_ready &&
            (!learning_enable_q || contribution_hidden_ready);
        o_hidden_observation_valid = numeric_hidden_valid &&
            (!learning_enable_q || contribution_hidden_ready);
        contribution_hidden_valid = numeric_hidden_valid &&
            i_hidden_observation_ready && learning_enable_q;
        o_hidden_observation_path = numeric_hidden_path;
        o_hidden_observation_sample_id = numeric_hidden_sample_id;
        o_hidden_observation_hidden_id = numeric_hidden_id;
        o_hidden_membrane_pre_q14 = numeric_hidden_membrane_pre_q14;
        o_hidden_membrane_next_q14 = numeric_hidden_membrane_next_q14;
        o_hidden_spike = numeric_hidden_spike;
        o_hidden_trace_q14 = numeric_hidden_trace_q14;
        o_hidden_output_feature_q14 =
            numeric_hidden_output_feature_q14;
        o_hidden_observation_last = numeric_hidden_last;

        numeric_logit_ready = i_logit_ready &&
            (!learning_enable_q || contribution_logit_ready);
        o_logit_valid = numeric_logit_valid &&
            (!learning_enable_q || contribution_logit_ready);
        contribution_logit_valid = numeric_logit_valid && i_logit_ready &&
            learning_enable_q;
        o_logit_sample_id = numeric_logit_sample_id;
        o_logit_class_id = numeric_logit_class_id;
        o_logit_q28 = numeric_logit_q28;
        o_logit_last = numeric_logit_last;

        o_target_trace_start = ENABLE_WH1_LEARNING && learning_enable_q &&
            (state_q == STATE_LEARNING) &&
            (observation_done_q || observation_done) &&
            !target_trace_started_q && i_target_trace_start_ready &&
            !i_abort_timestep;
        target_trace_start_fire = o_target_trace_start;
        o_target_trace_timestep_id = timestep_q;
        o_target_trace_valid = ENABLE_WH1_LEARNING && i_target_trace_valid &&
            contribution_target_trace_ready &&
            (state_q == STATE_LEARNING) && target_trace_started_q;
        contribution_target_trace_valid = ENABLE_WH1_LEARNING &&
            i_target_trace_valid &&
            i_target_trace_ready && (state_q == STATE_LEARNING) &&
            target_trace_started_q;
        o_target_trace_ready = ENABLE_WH1_LEARNING && i_target_trace_ready &&
            contribution_target_trace_ready &&
            (state_q == STATE_LEARNING) && target_trace_started_q;
        o_target_trace_sample_id = i_target_trace_sample_id;
        o_target_trace_backbone_id = i_target_trace_backbone_id;
        o_target_trace_q14 = i_target_trace_q14;
        o_target_trace_last = i_target_trace_last;
        o_target_trace_done = ENABLE_WH1_LEARNING && i_target_trace_done &&
            (state_q == STATE_LEARNING) && target_trace_started_q;
        contribution_label_valid = i_label_valid && learning_enable_q &&
            (state_q == STATE_LEARNING);
        o_label_ready = contribution_label_ready && learning_enable_q &&
            (state_q == STATE_LEARNING);

        completion_seen = (observation_done_q || observation_done) &&
            (numeric_done_q || numeric_timestep_done) &&
            (!learning_enable_q ||
                ((!ENABLE_WH1_LEARNING || target_trace_done_q ||
                    o_target_trace_done) &&
                 (learning_done_q || store_learning_done) &&
                 (commit_seen_q || o_learning_commit_pulse) &&
                 (detach_seen_q || o_detach_pulse) &&
                 (contribution_done_q || contribution_done)));

        o_busy = (state_q != STATE_IDLE) || store_busy || observation_busy ||
            numeric_busy || contribution_busy;
        o_error = (state_q == STATE_ERROR) || store_error ||
            store_forward_error || observation_error ||
            (ENABLE_WH1_LEARNING && learning_enable_q &&
                i_target_trace_error) ||
            numeric_protocol_error || numeric_overflow ||
            numeric_timestep_id_error || (learning_enable_q &&
                (contribution_protocol_error || contribution_overflow));
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            configured_q <= 1'b0;
            learning_enable_q <= 1'b0;
            timestep_q <= '0;
            sequence_version_q <= '0;
            observation_done_q <= 1'b0;
            target_trace_started_q <= 1'b0;
            target_trace_done_q <= 1'b0;
            learning_done_q <= 1'b0;
            commit_seen_q <= 1'b0;
            detach_seen_q <= 1'b0;
            numeric_done_q <= 1'b0;
            contribution_done_q <= 1'b0;
            numeric_parameter_done_pending_q <= 1'b0;
            o_timestep_done <= 1'b0;
            o_timestep_aborted <= 1'b0;
        end else begin
            o_timestep_done <= 1'b0;
            o_timestep_aborted <= 1'b0;

            if (numeric_parameter_scan_done)
                numeric_parameter_done_pending_q <= 1'b1;
            if (store_forward_done)
                numeric_parameter_done_pending_q <= 1'b0;

            if (shared_cfg_fire)
                configured_q <= 1'b1;

            if (i_clear_error && (state_q == STATE_ERROR)) begin
                state_q <= STATE_IDLE;
            end else if (store_error || store_forward_error ||
                    observation_error ||
                    (ENABLE_WH1_LEARNING && learning_enable_q &&
                        i_target_trace_error) ||
                    numeric_protocol_error || numeric_overflow ||
                    numeric_timestep_id_error || (learning_enable_q &&
                        (contribution_protocol_error ||
                            contribution_overflow))) begin
                state_q <= STATE_ERROR;
            end else begin
                unique case (state_q)
                    STATE_IDLE: begin
                        if (start_fire) begin
                            timestep_q <= i_timestep_id;
                            sequence_version_q <= i_sequence_version;
                            learning_enable_q <= i_learning_enable;
                            observation_done_q <= 1'b0;
                            target_trace_started_q <= 1'b0;
                            target_trace_done_q <= 1'b0;
                            learning_done_q <= 1'b0;
                            commit_seen_q <= 1'b0;
                            detach_seen_q <= 1'b0;
                            numeric_done_q <= 1'b0;
                            contribution_done_q <= 1'b0;
                            state_q <= STATE_FORWARD;
                        end
                    end

                    STATE_FORWARD: begin
                        if (USE_SHARED_H40_FORWARD &&
                                o_forward_parameter_scan_done)
                            state_q <= STATE_PREPARE_ISSUE;
                        else if (!USE_SHARED_H40_FORWARD &&
                                store_forward_done)
                            state_q <= STATE_START_LEARNING;
                    end

                    STATE_PREPARE_ISSUE: begin
                        if (o_h40_prepare_valid && i_h40_prepare_ready)
                            state_q <= STATE_PREPARE_WAIT;
                    end

                    STATE_PREPARE_WAIT: begin
                        if (i_h40_prepare_done) begin
                            if (!i_h40_cache_valid)
                                state_q <= STATE_ERROR;
                            else
                                state_q <= STATE_FORWARD_CLOSE;
                        end
                    end

                    STATE_FORWARD_CLOSE: begin
                        if (store_forward_done)
                            state_q <= STATE_START_LEARNING;
                    end

                    STATE_START_LEARNING: begin
                        if (abort_fire) begin
                            state_q <= STATE_IDLE;
                            o_timestep_aborted <= 1'b1;
                        end else if (learning_pair_fire) begin
                            state_q <= STATE_LEARNING;
                        end
                    end

                    STATE_LEARNING: begin
                        if (observation_done)
                            observation_done_q <= 1'b1;
                        if (target_trace_start_fire)
                            target_trace_started_q <= 1'b1;
                        if (o_target_trace_done)
                            target_trace_done_q <= 1'b1;
                        if (store_learning_done)
                            learning_done_q <= 1'b1;
                        if (o_learning_commit_pulse)
                            commit_seen_q <= 1'b1;
                        if (o_detach_pulse)
                            detach_seen_q <= 1'b1;
                        if (numeric_timestep_done)
                            numeric_done_q <= 1'b1;
                        if (contribution_done)
                            contribution_done_q <= 1'b1;

                        if (learning_enable_q &&
                            (store_learning_aborted || contribution_aborted)) begin
                            state_q <= STATE_IDLE;
                            o_timestep_aborted <= 1'b1;
                        end else if (completion_seen) begin
                            state_q <= STATE_IDLE;
                            o_timestep_done <= 1'b1;
                        end
                    end

                    STATE_ERROR: begin
                        if (i_clear_error)
                            state_q <= STATE_IDLE;
                    end

                    default: state_q <= STATE_ERROR;
                endcase
            end
        end
    end

    assign o_h40_cfg_batch_count = i_cfg_batch_count;
    assign o_h40_cfg_backbone_count = i_cfg_backbone_count;
    assign o_h40_cfg_hidden_count = i_cfg_hidden_count;
    assign o_h40_cfg_class_count = i_cfg_class_count;
    assign o_h40_cfg_wh1_parameter_base = i_cfg_wh1_base;
    assign o_h40_cfg_wout_parameter_base = i_cfg_wout_base;

    generate
    if (!USE_SHARED_H40_FORWARD) begin : g_legacy_numeric_forward
    unified_srnn_m2_numeric_forward_observation_consumer #(
        .BATCH_MAX(BATCH_MAX),
        .TIMESTEP_MAX(TIMESTEP_MAX),
        .BACKBONE_MAX(BACKBONE_MAX),
        .HIDDEN_MAX(HIDDEN_MAX),
        .CLASS_MAX(CLASS_MAX),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .TIMESTEP_ID_WIDTH(TIMESTEP_ID_WIDTH),
        .BACKBONE_INDEX_WIDTH(BACKBONE_INDEX_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .CLASS_INDEX_WIDTH(CLASS_INDEX_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH)
    ) numeric_consumer (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_cfg_valid(shared_cfg_fire),
        .o_cfg_ready(numeric_cfg_ready),
        .i_cfg_wh1_tensor_id(i_cfg_wh1_tensor_id),
        .i_cfg_wout_tensor_id(i_cfg_wout_tensor_id),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_backbone_count(i_cfg_backbone_count),
        .i_cfg_hidden_count(i_cfg_hidden_count),
        .i_cfg_class_count(i_cfg_class_count),
        .o_cfg_error(numeric_cfg_error),
        .i_parameter_valid(numeric_parameter_valid),
        .o_parameter_ready(numeric_parameter_ready),
        .i_parameter_tensor_id(store_forward_tensor_id),
        .i_parameter_local_addr(store_forward_local_addr),
        .i_parameter_code_q14(store_forward_code),
        .i_parameter_last(store_forward_last),
        .o_parameter_scan_done(numeric_parameter_scan_done),
        .i_timestep_start(learning_pair_fire),
        .o_timestep_start_ready(numeric_timestep_start_ready),
        .i_timestep_id(timestep_q),
        .i_sequence_version(i_sequence_version),
        .i_include_target_path(learning_enable_q && ENABLE_WH1_LEARNING),
        .i_abort_timestep(store_learning_abort),
        .i_observation_valid(numeric_observation_valid),
        .o_observation_ready(numeric_observation_ready),
        .i_observation_path(observation_backbone_path),
        .i_observation_sample_id(observation_backbone_sample),
        .i_observation_backbone_id(observation_backbone_id),
        .i_observation_q14(observation_backbone_q14),
        .i_observation_last(observation_backbone_last),
        .i_observation_done(observation_done),
        .o_hidden_valid(numeric_hidden_valid),
        .i_hidden_ready(numeric_hidden_ready),
        .o_hidden_path(numeric_hidden_path),
        .o_hidden_sample_id(numeric_hidden_sample_id),
        .o_hidden_id(numeric_hidden_id),
        .o_hidden_membrane_pre_q14(numeric_hidden_membrane_pre_q14),
        .o_hidden_membrane_next_q14(numeric_hidden_membrane_next_q14),
        .o_hidden_spike(numeric_hidden_spike),
        .o_hidden_trace_q14(numeric_hidden_trace_q14),
        .o_hidden_output_feature_q14(
            numeric_hidden_output_feature_q14),
        .o_hidden_last(numeric_hidden_last),
        .o_logit_valid(numeric_logit_valid),
        .i_logit_ready(numeric_logit_ready),
        .o_logit_sample_id(numeric_logit_sample_id),
        .o_logit_class_id(numeric_logit_class_id),
        .o_logit_q28(numeric_logit_q28),
        .o_logit_last(numeric_logit_last),
        .o_timestep_id(numeric_timestep_id),
        .o_timestep_done(numeric_timestep_done),
        .o_busy(numeric_busy),
        .o_protocol_error(numeric_protocol_error),
        .o_overflow(numeric_overflow)
    );
    assign o_h40_cfg_valid = 1'b0;
    assign o_h40_prepare_valid = 1'b0;
    assign o_h40_prepare_version = '0;
    assign o_h40_cache_invalidate = 1'b0;
    assign o_h40_wh1_cache_invalidate = 1'b0;
    assign o_h40_timestep_start = 1'b0;
    assign o_h40_timestep_id = '0;
    assign o_h40_sequence_version = '0;
    assign o_h40_include_target_path = 1'b0;
    assign o_h40_observation_valid = 1'b0;
    assign o_h40_observation_path = 1'b0;
    assign o_h40_observation_sample_id = '0;
    assign o_h40_observation_backbone_id = '0;
    assign o_h40_observation_q14 = '0;
    assign o_h40_observation_last = 1'b0;
    assign o_h40_observation_done = 1'b0;
    assign o_h40_hidden_ready = 1'b0;
    assign o_h40_logit_ready = 1'b0;
    end else begin : g_shared_h40_forward
        assign numeric_cfg_ready = i_h40_cfg_ready;
        assign numeric_cfg_error = i_h40_cfg_error;
        assign numeric_parameter_ready = 1'b1;
        assign numeric_parameter_scan_done = 1'b0;
        assign numeric_timestep_start_ready = i_h40_timestep_start_ready;
        assign numeric_observation_ready = i_h40_observation_ready;
        assign numeric_timestep_id = i_h40_timestep_id;
        assign numeric_timestep_done = i_h40_timestep_done;
        assign numeric_busy = i_h40_busy;
        assign numeric_protocol_error = i_h40_sticky_error ||
            i_h40_protocol_error;
        assign numeric_overflow = i_h40_overflow;
        assign numeric_hidden_valid = i_h40_hidden_valid;
        assign numeric_hidden_path = i_h40_hidden_path;
        assign numeric_hidden_sample_id = i_h40_hidden_sample_id;
        assign numeric_hidden_id = i_h40_hidden_id;
        assign numeric_hidden_membrane_pre_q14 =
            i_h40_hidden_membrane_pre_q14;
        assign numeric_hidden_membrane_next_q14 =
            i_h40_hidden_membrane_next_q14;
        assign numeric_hidden_spike = i_h40_hidden_spike;
        assign numeric_hidden_trace_q14 = i_h40_hidden_trace_q14;
        assign numeric_hidden_output_feature_q14 =
            i_h40_hidden_output_feature_q14;
        assign numeric_hidden_last = i_h40_hidden_last;
        assign numeric_logit_valid = i_h40_logit_valid;
        assign numeric_logit_sample_id = i_h40_logit_sample_id;
        assign numeric_logit_class_id = i_h40_logit_class_id;
        assign numeric_logit_q28 = i_h40_logit_q28;
        assign numeric_logit_last = i_h40_logit_last;

        assign o_h40_cfg_valid = shared_cfg_fire;
        assign o_h40_prepare_valid = state_q == STATE_PREPARE_ISSUE;
        assign o_h40_prepare_version = o_forward_adapt_version;
        assign o_h40_wh1_cache_invalidate = shared_cfg_fire ||
            (i_load_valid && o_load_ready) ||
            (ENABLE_WH1_LEARNING && o_learning_commit_pulse && o_commit_code_dirty);
        assign o_h40_cache_invalidate = shared_cfg_fire ||
            (i_load_valid && o_load_ready) ||
            (o_learning_commit_pulse && o_commit_code_dirty);
        assign o_h40_timestep_start = learning_pair_fire;
        assign o_h40_timestep_id = timestep_q;
        assign o_h40_sequence_version = sequence_version_q;
        assign o_h40_include_target_path = learning_enable_q &&
            ENABLE_WH1_LEARNING;
        assign o_h40_observation_valid = numeric_observation_valid;
        assign o_h40_observation_path = observation_backbone_path;
        assign o_h40_observation_sample_id = observation_backbone_sample;
        assign o_h40_observation_backbone_id = observation_backbone_id;
        assign o_h40_observation_q14 = observation_backbone_q14;
        assign o_h40_observation_last = observation_backbone_last;
        assign o_h40_observation_done = observation_done;
        assign o_h40_hidden_ready = numeric_hidden_ready;
        assign o_h40_logit_ready = numeric_logit_ready;
    end
    endgenerate

    unified_srnn_m2_tp_ce_contribution_frontend #(
        .ENABLE_WH1_LEARNING(ENABLE_WH1_LEARNING),
        .BATCH_MAX(BATCH_MAX),
        .HIDDEN_MAX(HIDDEN_MAX),
        .BACKBONE_MAX(BACKBONE_MAX),
        .CLASS_MAX(CLASS_MAX),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .BACKBONE_INDEX_WIDTH(BACKBONE_INDEX_WIDTH),
        .CLASS_INDEX_WIDTH(CLASS_INDEX_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH)
    ) contribution_frontend (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_cfg_valid(shared_cfg_fire),
        .o_cfg_ready(contribution_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_hidden_count(i_cfg_hidden_count),
        .i_cfg_backbone_count(i_cfg_backbone_count),
        .i_cfg_class_count(i_cfg_class_count),
        .o_cfg_error(contribution_cfg_error),
        .i_start_window(learning_pair_fire && learning_enable_q),
        .o_start_ready(contribution_start_ready),
        .i_abort_window(store_learning_abort),
        .i_hidden_valid(contribution_hidden_valid),
        .o_hidden_ready(contribution_hidden_ready),
        .i_hidden_path(numeric_hidden_path),
        .i_hidden_sample_id(numeric_hidden_sample_id),
        .i_hidden_id(numeric_hidden_id),
        .i_hidden_membrane_pre_q14(numeric_hidden_membrane_pre_q14),
        .i_hidden_trace_q14(numeric_hidden_trace_q14),
        .i_hidden_last(numeric_hidden_last),
        .i_target_trace_valid(contribution_target_trace_valid),
        .o_target_trace_ready(contribution_target_trace_ready),
        .i_target_trace_sample_id(i_target_trace_sample_id),
        .i_target_trace_backbone_id(i_target_trace_backbone_id),
        .i_target_trace_q14(i_target_trace_q14),
        .i_target_trace_last(i_target_trace_last),
        .i_logit_valid(contribution_logit_valid),
        .o_logit_ready(contribution_logit_ready),
        .i_logit_sample_id(numeric_logit_sample_id),
        .i_logit_class_id(numeric_logit_class_id),
        .i_logit_q28(numeric_logit_q28),
        .i_logit_last(numeric_logit_last),
        .i_label_valid(contribution_label_valid),
        .o_label_ready(contribution_label_ready),
        .i_label_sample_id(i_label_sample_id),
        .i_label_class_id(i_label_class_id),
        .i_label_last(i_label_last),
        .o_wh1_contribution_valid(contribution_wh1_valid),
        .i_wh1_contribution_ready(contribution_wh1_ready),
        .o_wh1_contribution_path(contribution_wh1_path),
        .o_wh1_contribution_sample_id(contribution_wh1_sample_id),
        .o_wh1_contribution_hidden_id(contribution_wh1_hidden_id),
        .o_wh1_contribution_column_id(contribution_wh1_column_id),
        .o_wh1_error_q40(contribution_wh1_error_q40),
        .o_wh1_opposite_trace_q14(
            contribution_wh1_opposite_trace_q14),
        .o_wh1_surrogate_uq24(contribution_wh1_surrogate_uq24),
        .o_wh1_contribution_last(contribution_wh1_last),
        .o_wout_contribution_valid(contribution_wout_valid),
        .i_wout_contribution_ready(contribution_wout_ready),
        .o_wout_contribution_addr(contribution_wout_addr),
        .o_wout_contribution_sample_id(contribution_wout_sample_id),
        .o_wout_error_q40(contribution_wout_error_q40),
        .o_wout_hidden_trace_q14(contribution_wout_hidden_trace_q14),
        .o_wout_contribution_last(contribution_wout_last),
        .o_busy(contribution_busy),
        .o_done(contribution_done),
        .o_aborted(contribution_aborted),
        .o_protocol_error(contribution_protocol_error),
        .o_overflow(contribution_overflow)
    );

    generate
        if (USE_EXTERNAL_BACKBONE_STREAM) begin : gen_external_backbone
            assign observation_cfg_ready = 1'b1;
            assign observation_cfg_error = 1'b0;
            assign observation_start_ready = 1'b1;
            assign observation_backbone_valid = i_external_backbone_valid;
            assign o_external_backbone_ready = observation_backbone_ready;
            assign window_backbone_ready = 1'b0;
            assign observation_backbone_path = i_external_backbone_path;
            assign observation_backbone_sample = i_external_backbone_sample_id;
            assign observation_backbone_id = i_external_backbone_id;
            assign observation_backbone_q14 = i_external_backbone_q14;
            assign observation_backbone_last = i_external_backbone_last;
            assign observation_busy = 1'b0;
            assign observation_done = i_external_backbone_done;
            assign observation_error = i_external_backbone_error;
        end else begin : gen_vector_replay_backbone
            assign o_external_backbone_ready = 1'b0;
            assign observation_cfg_ready = window_cfg_ready;
            assign observation_cfg_error = window_cfg_error;
            assign observation_start_ready = window_start_ready;
            assign observation_backbone_valid = window_backbone_valid;
            assign window_backbone_ready = observation_backbone_ready;
            assign observation_backbone_path = window_backbone_path;
            assign observation_backbone_sample = window_backbone_sample;
            assign observation_backbone_id = window_backbone_id;
            assign observation_backbone_q14 = window_backbone_q14;
            assign observation_backbone_last = window_backbone_last;
            assign observation_busy = window_busy;
            assign observation_done = window_done;
            assign observation_error = window_error;
        end
    endgenerate

    unified_srnn_m2_dual_path_observation_window #(
        .BATCH_MAX(BATCH_MAX),
        .TIMESTEP_MAX(TIMESTEP_MAX),
        .BACKBONE_MAX(BACKBONE_MAX),
        .SOURCE_VECTOR_MAX(SOURCE_VECTOR_MAX),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .TIMESTEP_ID_WIDTH(TIMESTEP_ID_WIDTH),
        .BACKBONE_INDEX_WIDTH(BACKBONE_INDEX_WIDTH)
    ) observation_window (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_cfg_valid(shared_cfg_fire && !USE_EXTERNAL_BACKBONE_STREAM),
        .o_cfg_ready(window_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_backbone_count(i_cfg_backbone_count),
        .o_cfg_error(window_cfg_error),
        .i_window_start(observation_start &&
            !USE_EXTERNAL_BACKBONE_STREAM),
        .o_window_start_ready(window_start_ready),
        .i_window_timestep_id(timestep_q),
        .i_window_include_target(learning_enable_q &&
            ENABLE_WH1_LEARNING),
        .i_abort_window(observation_abort),
        .o_actual_replay_start(o_actual_replay_start),
        .i_actual_replay_start_ready(i_actual_replay_start_ready),
        .o_actual_replay_timestep_id(o_actual_replay_timestep_id),
        .i_actual_replay_valid(i_actual_replay_valid),
        .o_actual_replay_ready(o_actual_replay_ready),
        .i_actual_replay_sample_id(i_actual_replay_sample_id),
        .i_actual_replay_spikes(i_actual_replay_spikes),
        .i_actual_replay_done(i_actual_replay_done),
        .i_actual_replay_error(i_actual_replay_error),
        .o_target_vector_start(o_target_vector_start),
        .i_target_vector_start_ready(i_target_vector_start_ready),
        .o_target_vector_timestep_id(o_target_vector_timestep_id),
        .i_target_vector_valid(i_target_vector_valid),
        .o_target_vector_ready(o_target_vector_ready),
        .i_target_vector_sample_id(i_target_vector_sample_id),
        .i_target_vector_spikes(i_target_vector_spikes),
        .i_target_vector_done(i_target_vector_done),
        .i_target_vector_error(i_target_vector_error),
        .o_backbone_valid(window_backbone_valid),
        .i_backbone_ready(window_backbone_ready),
        .o_backbone_path(window_backbone_path),
        .o_backbone_sample_id(window_backbone_sample),
        .o_backbone_id(window_backbone_id),
        .o_backbone_q14(window_backbone_q14),
        .o_backbone_last(window_backbone_last),
        .o_busy(window_busy),
        .o_done(window_done),
        .o_protocol_error(window_error)
    );

    /* verilator lint_off PINCONNECTEMPTY */
    unified_srnn_m2_forward_learning_store_slice #(
        .USE_INTERNAL_STORE(USE_INTERNAL_STORE),
        .USE_RANDOM_ACCESS_FORWARD(USE_SHARED_H40_FORWARD),
        .ENABLE_WH1_LEARNING(ENABLE_WH1_LEARNING),
        .STORE_DEPTH(STORE_DEPTH),
        .BATCH_MAX(BATCH_MAX),
        .HIDDEN_MAX(HIDDEN_MAX),
        .BACKBONE_MAX(BACKBONE_MAX),
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .BACKBONE_INDEX_WIDTH(BACKBONE_INDEX_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH),
        .WINDOW_SEQUENCE_WIDTH(WINDOW_SEQUENCE_WIDTH)
    ) parameter_store (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_cfg_valid(shared_cfg_fire),
        .o_cfg_ready(store_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id),
        .i_cfg_wh1_tensor_id(i_cfg_wh1_tensor_id),
        .i_cfg_wh1_base(i_cfg_wh1_base),
        .i_cfg_wh1_increment(i_cfg_wh1_increment),
        .i_cfg_wh1_threshold(i_cfg_wh1_threshold),
        .i_cfg_wout_tensor_id(i_cfg_wout_tensor_id),
        .i_cfg_wout_base(i_cfg_wout_base),
        .i_cfg_wout_increment(i_cfg_wout_increment),
        .i_cfg_wout_threshold(i_cfg_wout_threshold),
        .i_cfg_code_min(i_cfg_code_min),
        .i_cfg_code_max(i_cfg_code_max),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_hidden_count(i_cfg_hidden_count),
        .i_cfg_backbone_count(i_cfg_backbone_count),
        .i_cfg_wout_element_count(i_cfg_wout_element_count),
        .o_cfg_error(store_cfg_error),
        .i_load_valid(i_load_valid),
        .o_load_ready(o_load_ready),
        .i_load_addr(i_load_addr),
        .i_load_code(i_load_code),
        .i_load_residual(i_load_residual),
        .i_forward_start(store_forward_start),
        .o_forward_start_ready(store_forward_start_ready),
        .o_forward_code_valid(store_forward_code_valid),
        .i_forward_code_ready(store_forward_code_ready),
        .o_forward_tensor_id(store_forward_tensor_id),
        .o_forward_local_addr(store_forward_local_addr),
        .o_forward_parameter_addr(store_forward_parameter_addr),
        .o_forward_code(store_forward_code),
        .o_forward_last(store_forward_last),
        .o_forward_parameter_scan_done(o_forward_parameter_scan_done),
        .i_forward_consumer_done_valid(
            store_forward_consumer_done_valid),
        .o_forward_consumer_done_ready(
            store_forward_consumer_done_ready),
        .o_forward_done(store_forward_done),
        .o_forward_error(store_forward_error),
        .i_cache_read_valid(USE_SHARED_H40_FORWARD ?
            i_h40_forward_read_valid : 1'b0),
        .o_cache_read_ready(o_h40_forward_read_ready),
        .i_cache_read_addr(i_h40_forward_read_addr),
        .i_cache_read_tag(i_h40_forward_read_tag),
        .o_cache_response_valid(o_h40_forward_response_valid),
        .i_cache_response_ready(USE_SHARED_H40_FORWARD ?
            i_h40_forward_response_ready : 1'b0),
        .o_cache_response_tag(o_h40_forward_response_tag),
        .o_cache_response_code(o_h40_forward_response_code),
        .o_cache_response_error(o_h40_forward_response_error),
        .i_learning_start(store_learning_start),
        .o_learning_start_ready(store_learning_start_ready),
        .i_learning_abort(store_learning_abort),
        .i_restore_complete(i_restore_complete),
        .i_wh1_tp_contribution_valid(ENABLE_WH1_LEARNING &&
            contribution_wh1_valid &&
            learning_enable_q && (state_q == STATE_LEARNING)),
        .o_wh1_tp_contribution_ready(contribution_wh1_ready),
        .i_wh1_tp_contribution_path(contribution_wh1_path),
        .i_wh1_tp_contribution_sample_id(contribution_wh1_sample_id),
        .i_wh1_tp_contribution_hidden_id(contribution_wh1_hidden_id),
        .i_wh1_tp_contribution_column_id(contribution_wh1_column_id),
        .i_wh1_tp_error_q40(contribution_wh1_error_q40),
        .i_wh1_tp_opposite_trace_q14(
            contribution_wh1_opposite_trace_q14),
        .i_wh1_tp_surrogate_uq24(contribution_wh1_surrogate_uq24),
        .i_wh1_tp_contribution_last(contribution_wh1_last),
        .i_wh1_backbone_valid(ENABLE_WH1_LEARNING &&
            observation_backbone_valid &&
            numeric_observation_ready && learning_enable_q &&
            (state_q == STATE_LEARNING)),
        .o_wh1_backbone_ready(store_wh1_backbone_ready),
        .i_wh1_backbone_path(observation_backbone_path),
        .i_wh1_backbone_sample_id(observation_backbone_sample),
        .i_wh1_backbone_id(observation_backbone_id),
        .i_wh1_backbone_q14(observation_backbone_q14),
        .i_wh1_backbone_last(observation_backbone_last),
        .i_wout_contribution_valid(contribution_wout_valid &&
            learning_enable_q && (state_q == STATE_LEARNING)),
        .o_wout_contribution_ready(contribution_wout_ready),
        .i_wout_contribution_addr(contribution_wout_addr),
        .i_wout_contribution_sample_id(contribution_wout_sample_id),
        .i_wout_error_q40(contribution_wout_error_q40),
        .i_wout_hidden_trace_q14(contribution_wout_hidden_trace_q14),
        .i_wout_contribution_last(contribution_wout_last),
        .o_learning_commit_pulse(o_learning_commit_pulse),
        .o_detach_pulse(o_detach_pulse),
        .o_model_version(o_model_version),
        .o_adapt_version(o_adapt_version),
        .o_forward_model_version(o_forward_model_version),
        .o_forward_adapt_version(o_forward_adapt_version),
        .o_learning_base_model_version(o_learning_base_model_version),
        .o_learning_base_adapt_version(o_learning_base_adapt_version),
        .o_learning_commit_window_sequence(o_learning_commit_window_sequence),
        .o_commit_code_dirty(o_commit_code_dirty),
        .o_commit_optimizer_dirty(o_commit_optimizer_dirty),
        .o_learning_result_dirty(o_learning_result_dirty),
        .o_learning_saturated(o_learning_saturated),
        .o_restore_ready(o_restore_ready),
        .o_recovery_required(o_recovery_required),
        .o_busy(store_busy),
        .o_learning_done(store_learning_done),
        .o_learning_aborted(store_learning_aborted),
        .o_error(store_error),
        .o_zero_gradient_count(o_zero_gradient_count),
        .o_update_count(o_update_count),
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
        .i_store_protocol_error(i_store_protocol_error)
    );
    /* verilator lint_on PINCONNECTEMPTY */

endmodule

`default_nettype wire
