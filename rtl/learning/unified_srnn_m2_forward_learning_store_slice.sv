`timescale 1ns/1ps
`default_nettype none

// SRNN M2 slice that binds trainable W_h1/W_out to one shared Q2.14
// parameter store.  Forward scans read only the code field; TP learning uses
// the same addresses through the RMW path and publishes visibility through the
// shared train/infer version boundary.
module unified_srnn_m2_forward_learning_store_slice #(
    parameter bit USE_INTERNAL_STORE = 1'b1,
    parameter bit USE_RANDOM_ACCESS_FORWARD = 1'b0,
    parameter bit ENABLE_WH1_LEARNING = 1'b1,
    parameter int STORE_DEPTH = 1024,
    parameter int BATCH_MAX = 64,
    parameter int HIDDEN_MAX = 256,
    parameter int BACKBONE_MAX = 450,
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int SAMPLE_ID_WIDTH = 8,
    parameter int HIDDEN_INDEX_WIDTH = 9,
    parameter int BACKBONE_INDEX_WIDTH = 10,
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
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_wout_element_count,
    output logic                                      o_cfg_error,

    input  logic                                      i_load_valid,
    output logic                                      o_load_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_load_addr,
    input  logic signed [CODE_WIDTH-1:0]              i_load_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]          i_load_residual,

    input  logic                                      i_forward_start,
    output logic                                      o_forward_start_ready,
    output logic                                      o_forward_code_valid,
    input  logic                                      i_forward_code_ready,
    output logic [TENSOR_ID_WIDTH-1:0]                o_forward_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_forward_local_addr,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_forward_parameter_addr,
    output logic signed [CODE_WIDTH-1:0]              o_forward_code,
    output logic                                      o_forward_last,
    output logic                                      o_forward_parameter_scan_done,
    input  logic                                      i_forward_consumer_done_valid,
    output logic                                      o_forward_consumer_done_ready,
    output logic                                      o_forward_done,
    output logic                                      o_forward_error,

    // Random-access inference service for the shared H40 working-set refill.
    // It uses the same physical forward read port as the legacy parameter
    // scan and is legal only while a random-access forward boundary is open.
    input  logic                                      i_cache_read_valid,
    output logic                                      o_cache_read_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cache_read_addr,
    input  logic [TAG_WIDTH-1:0]                      i_cache_read_tag,
    output logic                                      o_cache_response_valid,
    input  logic                                      i_cache_response_ready,
    output logic [TAG_WIDTH-1:0]                      o_cache_response_tag,
    output logic signed [CODE_WIDTH-1:0]              o_cache_response_code,
    output logic                                      o_cache_response_error,

    input  logic                                      i_learning_start,
    output logic                                      o_learning_start_ready,
    input  logic                                      i_learning_abort,
    input  logic                                      i_restore_complete,

    input  logic                                      i_wh1_tp_contribution_valid,
    output logic                                      o_wh1_tp_contribution_ready,
    input  logic                                      i_wh1_tp_contribution_path,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_wh1_tp_contribution_sample_id,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]             i_wh1_tp_contribution_hidden_id,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_wh1_tp_contribution_column_id,
    input  logic signed [63:0]                        i_wh1_tp_error_q40,
    input  logic signed [31:0]                        i_wh1_tp_opposite_trace_q14,
    input  logic [24:0]                               i_wh1_tp_surrogate_uq24,
    input  logic                                      i_wh1_tp_contribution_last,

    input  logic                                      i_wh1_backbone_valid,
    output logic                                      o_wh1_backbone_ready,
    input  logic                                      i_wh1_backbone_path,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_wh1_backbone_sample_id,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_wh1_backbone_id,
    input  logic signed [31:0]                        i_wh1_backbone_q14,
    input  logic                                      i_wh1_backbone_last,

    input  logic                                      i_wout_contribution_valid,
    output logic                                      o_wout_contribution_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_wout_contribution_addr,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_wout_contribution_sample_id,
    input  logic signed [63:0]                        i_wout_error_q40,
    input  logic signed [31:0]                        i_wout_hidden_trace_q14,
    input  logic                                      i_wout_contribution_last,

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
    output logic                                      o_restore_ready,
    output logic                                      o_recovery_required,
    output logic                                      o_busy,
    output logic                                      o_learning_done,
    output logic                                      o_learning_aborted,
    output logic                                      o_error,
    output logic [PARAM_ADDR_WIDTH:0]                 o_zero_gradient_count,
    output logic [PARAM_ADDR_WIDTH:0]                 o_update_count,

    // Optional external parameter-context backend.  These ports expose the
    // same single-outstanding protocol used by the internal behavioral store,
    // so a higher-level owner can share one physical service across profiles.
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
        FWD_IDLE,
        FWD_REQ,
        FWD_WAIT,
        FWD_EMIT,
        FWD_DRAIN
    } forward_state_t;

    localparam logic [PARAM_ADDR_WIDTH:0] STORE_DEPTH_EXT =
        (PARAM_ADDR_WIDTH + 1)'(STORE_DEPTH);
    localparam logic [PARAM_ADDR_WIDTH:0] EXTERNAL_ADDRESS_LIMIT =
        {1'b1, {PARAM_ADDR_WIDTH{1'b0}}};

    forward_state_t forward_state_q;
    logic configured_q;
    logic cfg_error_q;
    logic [PROFILE_ID_WIDTH-1:0] profile_id_q;
    logic [TENSOR_ID_WIDTH-1:0] wh1_tensor_id_q;
    logic [TENSOR_ID_WIDTH-1:0] wout_tensor_id_q;
    logic [PARAM_ADDR_WIDTH-1:0] wh1_base_q;
    logic [PARAM_ADDR_WIDTH-1:0] wout_base_q;
    logic [PARAM_ADDR_WIDTH-1:0] wh1_count_q;
    logic [PARAM_ADDR_WIDTH-1:0] wout_count_q;
    logic forward_phase_q;
    logic [PARAM_ADDR_WIDTH-1:0] forward_local_addr_q;
    logic [TENSOR_ID_WIDTH-1:0] forward_token_tensor_q;
    logic [PARAM_ADDR_WIDTH-1:0] forward_token_local_q;
    logic [PARAM_ADDR_WIDTH-1:0] forward_token_global_q;
    logic signed [CODE_WIDTH-1:0] forward_token_code_q;
    logic forward_token_last_q;
    logic forward_token_valid_q;
    logic forward_error_q;

    logic [PARAM_ADDR_WIDTH:0] cfg_wh1_count_ext;
    logic [PARAM_ADDR_WIDTH:0] cfg_wh1_end_ext;
    logic [PARAM_ADDR_WIDTH:0] cfg_wout_end_ext;
    logic cfg_ranges_before_wout;
    logic cfg_ranges_before_wh1;
    logic cfg_legal;
    logic cfg_fire;
    logic direct_cfg_ready;
    logic direct_cfg_error;
    logic boundary_context_ready;

    logic boundary_forward_start_valid;
    logic boundary_forward_start_ready;
    logic boundary_forward_start_fire;
    logic boundary_forward_active;
    logic boundary_learning_start_valid;
    logic boundary_learning_start_ready;
    logic boundary_learning_start_fire;
    logic boundary_learning_active;
    logic boundary_learning_commit_ready;
    logic boundary_commit_pulse;
    logic boundary_protocol_error;
    logic store_protocol_error;

    logic engine_start_ready;
    logic engine_commit_valid;
    logic engine_commit_fire;
    logic [PROFILE_ID_WIDTH-1:0] engine_commit_profile_id;
    logic [WINDOW_SEQUENCE_WIDTH-1:0] engine_commit_window_sequence;
    logic [WINDOW_SEQUENCE_WIDTH-1:0] commit_window_sequence_q;
    logic engine_busy;
    logic engine_done;
    logic engine_aborted;
    logic engine_error;
    logic engine_write_code_fire;
    logic engine_write_residual_fire;
    logic engine_read_valid;
    logic engine_read_ready;
    logic [PROFILE_ID_WIDTH-1:0] engine_read_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] engine_read_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] engine_read_addr;
    logic [TAG_WIDTH-1:0] engine_read_tag;
    logic engine_read_response_valid;
    logic engine_read_response_ready;
    logic signed [CODE_WIDTH-1:0] engine_read_response_code;
    logic signed [RESIDUAL_WIDTH-1:0] engine_read_response_residual;
    logic engine_read_response_error;
    logic engine_result_valid;
    logic engine_result_ready;
    logic [PROFILE_ID_WIDTH-1:0] engine_result_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] engine_result_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] engine_result_addr;
    logic [TAG_WIDTH-1:0] engine_result_tag;
    logic signed [CODE_WIDTH-1:0] engine_result_code;
    logic signed [RESIDUAL_WIDTH-1:0] engine_result_residual;
    logic engine_write_code_enable;
    logic engine_write_residual_enable;
    logic engine_dirty;
    logic engine_saturated;

    logic store_forward_read_valid;
    logic store_forward_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] store_forward_read_addr;
    logic [TAG_WIDTH-1:0] store_forward_read_tag;
    logic store_forward_response_valid;
    logic store_forward_response_ready;
    logic [TAG_WIDTH-1:0] store_forward_response_tag;
    logic signed [CODE_WIDTH-1:0] store_forward_response_code;
    logic store_forward_response_error;
    logic [TAG_WIDTH-1:0] store_learning_response_tag;
    logic store_learning_write_ready;
    logic learning_metadata_error;
    logic [WINDOW_SEQUENCE_WIDTH-1:0] expected_commit_sequence_q;
    logic forward_read_fire;
    logic forward_response_fire;
    logic forward_token_fire;
    logic cache_access_allowed;
    logic cache_read_fire;
    logic cache_response_fire;
    logic cache_active_q;
    logic forward_last_address;
    logic [PARAM_ADDR_WIDTH-1:0] next_forward_addr;
    logic next_forward_phase;

    always_comb begin
        cfg_wh1_count_ext = (PARAM_ADDR_WIDTH + 1)'(i_cfg_hidden_count) *
            (PARAM_ADDR_WIDTH + 1)'(i_cfg_backbone_count);
        cfg_wh1_end_ext = {1'b0, i_cfg_wh1_base} + cfg_wh1_count_ext;
        cfg_wout_end_ext = {1'b0, i_cfg_wout_base} +
            {1'b0, i_cfg_wout_element_count};
        cfg_ranges_before_wout = cfg_wh1_end_ext <= {1'b0, i_cfg_wout_base};
        cfg_ranges_before_wh1 = cfg_wout_end_ext <= {1'b0, i_cfg_wh1_base};
        cfg_legal = (cfg_wh1_count_ext != '0) &&
            (i_cfg_wout_element_count != '0) &&
            (i_cfg_wh1_tensor_id != i_cfg_wout_tensor_id) &&
            !cfg_wh1_count_ext[PARAM_ADDR_WIDTH] &&
            !cfg_wh1_end_ext[PARAM_ADDR_WIDTH] &&
            !cfg_wout_end_ext[PARAM_ADDR_WIDTH] &&
            (cfg_wh1_end_ext <= (USE_INTERNAL_STORE ? STORE_DEPTH_EXT :
                EXTERNAL_ADDRESS_LIMIT)) &&
            (cfg_wout_end_ext <= (USE_INTERNAL_STORE ? STORE_DEPTH_EXT :
                EXTERNAL_ADDRESS_LIMIT)) &&
            (cfg_ranges_before_wout || cfg_ranges_before_wh1) &&
            ($signed(i_cfg_code_min) <= $signed(i_cfg_code_max));

        o_cfg_ready = direct_cfg_ready && boundary_context_ready &&
            (forward_state_q == FWD_IDLE) && !cache_active_q &&
            !(USE_RANDOM_ACCESS_FORWARD && i_cache_read_valid);
        cfg_fire = i_cfg_valid && o_cfg_ready;
        o_cfg_error = cfg_error_q || direct_cfg_error;

        boundary_forward_start_valid = i_forward_start && configured_q &&
            (forward_state_q == FWD_IDLE) && !engine_busy &&
            !cache_active_q &&
            !(USE_RANDOM_ACCESS_FORWARD && i_cache_read_valid);
        o_forward_start_ready = configured_q &&
            (forward_state_q == FWD_IDLE) && !engine_busy &&
            boundary_forward_start_ready && !cache_active_q &&
            !(USE_RANDOM_ACCESS_FORWARD && i_cache_read_valid);

        boundary_learning_start_valid = i_learning_start && configured_q &&
            (forward_state_q == FWD_IDLE) && engine_start_ready;
        o_learning_start_ready = configured_q &&
            (forward_state_q == FWD_IDLE) && engine_start_ready &&
            boundary_learning_start_ready;

        cache_access_allowed = USE_RANDOM_ACCESS_FORWARD &&
            (forward_state_q == FWD_DRAIN);
        store_forward_read_valid = (forward_state_q == FWD_REQ) ||
            (cache_access_allowed && i_cache_read_valid && !cache_active_q);
        store_forward_read_addr = cache_access_allowed ?
            i_cache_read_addr : ((forward_phase_q ? wout_base_q :
                wh1_base_q) + forward_local_addr_q);
        store_forward_read_tag = cache_access_allowed ?
            i_cache_read_tag : TAG_WIDTH'(forward_local_addr_q);
        o_cache_read_ready = cache_access_allowed && !cache_active_q &&
            store_forward_read_ready;
        cache_read_fire = i_cache_read_valid && o_cache_read_ready;
        forward_read_fire = (forward_state_q == FWD_REQ) &&
            store_forward_read_ready;
        store_forward_response_ready = cache_active_q ?
            i_cache_response_ready :
            ((forward_state_q == FWD_WAIT) && !forward_token_valid_q);
        forward_response_fire = store_forward_response_valid &&
            store_forward_response_ready && !cache_active_q;
        cache_response_fire = store_forward_response_valid &&
            store_forward_response_ready && cache_active_q;
        forward_token_fire = forward_token_valid_q && i_forward_code_ready;
        forward_last_address = forward_phase_q &&
            (forward_local_addr_q == (wout_count_q - 1'b1));

        next_forward_phase = forward_phase_q;
        next_forward_addr = forward_local_addr_q + 1'b1;
        if (!forward_phase_q && (forward_local_addr_q == wh1_count_q - 1'b1)) begin
            next_forward_phase = 1'b1;
            next_forward_addr = '0;
        end

        o_forward_code_valid = forward_token_valid_q;
        o_forward_tensor_id = forward_token_tensor_q;
        o_forward_local_addr = forward_token_local_q;
        o_forward_parameter_addr = forward_token_global_q;
        o_forward_code = forward_token_code_q;
        o_forward_last = forward_token_last_q;
        o_forward_consumer_done_ready =
            forward_state_q == FWD_DRAIN;
        o_forward_error = forward_error_q;
        o_cache_response_valid = store_forward_response_valid &&
            cache_active_q;
        o_cache_response_tag = store_forward_response_tag;
        o_cache_response_code = store_forward_response_code;
        o_cache_response_error = store_forward_response_error;

        o_learning_commit_pulse = boundary_commit_pulse;
        o_learning_commit_window_sequence = commit_window_sequence_q;
        o_learning_result_dirty = engine_dirty;
        o_learning_saturated = engine_saturated;
        o_busy = (forward_state_q != FWD_IDLE) || cache_active_q || engine_busy ||
            boundary_forward_active || boundary_learning_active;
        o_learning_aborted = engine_aborted;
        o_error = o_cfg_error || forward_error_q || engine_error ||
            boundary_protocol_error || store_protocol_error;

        learning_metadata_error =
            (engine_read_valid &&
                ((engine_read_profile_id != profile_id_q) ||
                 ((engine_read_tensor_id != wh1_tensor_id_q) &&
                  (engine_read_tensor_id != wout_tensor_id_q)))) ||
            (engine_read_response_valid &&
                (store_learning_response_tag != engine_read_tag)) ||
            (engine_result_valid &&
                ((engine_result_profile_id != profile_id_q) ||
                 (engine_result_profile_id != engine_read_profile_id) ||
                 (engine_result_tensor_id != engine_read_tensor_id) ||
                 (engine_result_addr != engine_read_addr) ||
                 (engine_result_tag != engine_read_tag))) ||
            (engine_commit_valid &&
                ((engine_commit_profile_id != profile_id_q) ||
                 (engine_commit_window_sequence !=
                    expected_commit_sequence_q)));
        engine_commit_fire = engine_commit_valid &&
            boundary_learning_commit_ready;
    end

    unified_tp_train_infer_version_boundary #(
        .VERSION_WIDTH(VERSION_WIDTH)
    ) visibility_boundary (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_context_load_valid(1'b0),
        .o_context_load_ready(boundary_context_ready),
        .i_context_model_version('0),
        .i_context_adapt_version('0),
        .i_forward_start_valid(boundary_forward_start_valid),
        .o_forward_start_ready(boundary_forward_start_ready),
        .o_forward_start_fire(boundary_forward_start_fire),
        .i_forward_done(o_forward_done),
        .o_forward_active(boundary_forward_active),
        .o_forward_model_version(o_forward_model_version),
        .o_forward_adapt_version(o_forward_adapt_version),
        .i_learning_start_valid(boundary_learning_start_valid),
        .o_learning_start_ready(boundary_learning_start_ready),
        .o_learning_start_fire(boundary_learning_start_fire),
        .o_learning_active(boundary_learning_active),
        .o_learning_base_model_version(o_learning_base_model_version),
        .o_learning_base_adapt_version(o_learning_base_adapt_version),
        .i_forward_code_write_fire(engine_write_code_fire),
        .i_optimizer_state_write_fire(engine_write_residual_fire),
        .i_learning_commit_valid(engine_commit_valid),
        .o_learning_commit_ready(boundary_learning_commit_ready),
        .i_learning_abort_complete(engine_aborted),
        .i_restore_complete(i_restore_complete),
        .o_restore_ready(o_restore_ready),
        .o_recovery_required(o_recovery_required),
        .o_model_version(o_model_version),
        .o_adapt_version(o_adapt_version),
        .o_commit_pulse(boundary_commit_pulse),
        .o_commit_code_dirty(o_commit_code_dirty),
        .o_commit_optimizer_dirty(o_commit_optimizer_dirty),
        .o_detach_pulse(o_detach_pulse),
        .i_clear_error(i_clear_error),
        .o_protocol_error(boundary_protocol_error)
    );

    generate
    if (ENABLE_WH1_LEARNING) begin : g_dual_tensor_learning
    unified_srnn_tp_direct_online_learning_engine #(
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
        .WINDOW_SEQUENCE_WIDTH(WINDOW_SEQUENCE_WIDTH)
    ) learning_engine (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(cfg_fire && cfg_legal),
        .o_cfg_ready(direct_cfg_ready),
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
        .o_cfg_error(direct_cfg_error),
        .i_start_window(boundary_learning_start_fire),
        .o_start_ready(engine_start_ready),
        .i_abort_window(i_learning_abort),
        .i_wh1_tp_contribution_valid(i_wh1_tp_contribution_valid),
        .o_wh1_tp_contribution_ready(o_wh1_tp_contribution_ready),
        .i_wh1_tp_contribution_path(i_wh1_tp_contribution_path),
        .i_wh1_tp_contribution_sample_id(i_wh1_tp_contribution_sample_id),
        .i_wh1_tp_contribution_hidden_id(i_wh1_tp_contribution_hidden_id),
        .i_wh1_tp_contribution_column_id(i_wh1_tp_contribution_column_id),
        .i_wh1_tp_error_q40(i_wh1_tp_error_q40),
        .i_wh1_tp_opposite_trace_q14(i_wh1_tp_opposite_trace_q14),
        .i_wh1_tp_surrogate_uq24(i_wh1_tp_surrogate_uq24),
        .i_wh1_tp_contribution_last(i_wh1_tp_contribution_last),
        .i_wh1_backbone_valid(i_wh1_backbone_valid),
        .o_wh1_backbone_ready(o_wh1_backbone_ready),
        .i_wh1_backbone_path(i_wh1_backbone_path),
        .i_wh1_backbone_sample_id(i_wh1_backbone_sample_id),
        .i_wh1_backbone_id(i_wh1_backbone_id),
        .i_wh1_backbone_q14(i_wh1_backbone_q14),
        .i_wh1_backbone_last(i_wh1_backbone_last),
        .i_wout_contribution_valid(i_wout_contribution_valid),
        .o_wout_contribution_ready(o_wout_contribution_ready),
        .i_wout_contribution_addr(i_wout_contribution_addr),
        .i_wout_contribution_sample_id(i_wout_contribution_sample_id),
        .i_wout_error_q40(i_wout_error_q40),
        .i_wout_hidden_trace_q14(i_wout_hidden_trace_q14),
        .i_wout_contribution_last(i_wout_contribution_last),
        .o_read_valid(engine_read_valid),
        .i_read_ready(engine_read_ready),
        .o_read_profile_id(engine_read_profile_id),
        .o_read_tensor_id(engine_read_tensor_id),
        .o_read_parameter_addr(engine_read_addr),
        .o_read_tag(engine_read_tag),
        .i_read_response_valid(engine_read_response_valid),
        .o_read_response_ready(engine_read_response_ready),
        .i_read_response_code(engine_read_response_code),
        .i_read_response_residual(engine_read_response_residual),
        .i_read_response_error(engine_read_response_error),
        .o_result_valid(engine_result_valid),
        .i_result_ready(engine_result_ready),
        .o_result_profile_id(engine_result_profile_id),
        .o_result_tensor_id(engine_result_tensor_id),
        .o_result_parameter_addr(engine_result_addr),
        .o_result_tag(engine_result_tag),
        .o_result_code(engine_result_code),
        .o_result_residual(engine_result_residual),
        .o_write_code_enable(engine_write_code_enable),
        .o_write_residual_enable(engine_write_residual_enable),
        .o_write_code_fire(engine_write_code_fire),
        .o_write_residual_fire(engine_write_residual_fire),
        .o_dirty(engine_dirty),
        .o_saturated(engine_saturated),
        .o_commit_valid(engine_commit_valid),
        .i_commit_ready(boundary_learning_commit_ready),
        .o_commit_profile_id(engine_commit_profile_id),
        .o_commit_window_sequence(engine_commit_window_sequence),
        .o_busy(engine_busy),
        .o_done(engine_done),
        .o_window_aborted(engine_aborted),
        .o_error(engine_error),
        .o_zero_gradient_count(o_zero_gradient_count),
        .o_update_count(o_update_count)
    );
    end else begin : g_wout_only_learning
    typedef struct packed {
        logic [PROFILE_ID_WIDTH-1:0] profile_id;
        logic [TENSOR_ID_WIDTH-1:0] tensor_id;
        logic [PARAM_ADDR_WIDTH-1:0] parameter_base;
        logic [VOTE_WIDTH-1:0] increment;
        logic [VOTE_WIDTH-1:0] threshold;
        logic signed [CODE_WIDTH-1:0] code_min;
        logic signed [CODE_WIDTH-1:0] code_max;
        logic [SAMPLE_ID_WIDTH-1:0] batch_count;
        logic [PARAM_ADDR_WIDTH-1:0] element_count;
    } learner_cfg_t;

    learner_cfg_t incoming_cfg;
    learner_cfg_t saved_cfg_q;
    learner_cfg_t learner_cfg;
    logic saved_cfg_valid_q;
    logic learner_cfg_legal;
    logic rearm_needed_q;
    logic rearm_pending_q;
    logic auto_cfg_valid;
    logic auto_cfg_fire;
    logic learner_cfg_valid;
    logic learner_start_ready;
    logic learner_start_valid;
    logic deferred_start_q;
    logic deferred_abort_q;

    always_comb begin
        incoming_cfg.profile_id = i_cfg_profile_id;
        incoming_cfg.tensor_id = i_cfg_wout_tensor_id;
        incoming_cfg.parameter_base = i_cfg_wout_base;
        incoming_cfg.increment = i_cfg_wout_increment;
        incoming_cfg.threshold = i_cfg_wout_threshold;
        incoming_cfg.code_min = i_cfg_code_min;
        incoming_cfg.code_max = i_cfg_code_max;
        incoming_cfg.batch_count = i_cfg_batch_count;
        incoming_cfg.element_count = i_cfg_wout_element_count;
        learner_cfg_legal = cfg_legal && (i_cfg_batch_count != '0) &&
            (i_cfg_wout_threshold != '0) &&
            ({1'b0, i_cfg_wout_increment} <
                ({1'b0, i_cfg_wout_threshold} + {1'b0, i_cfg_wout_threshold}));

        // Ready accepts one buffered learning start. Forward-only work must
        // leave a released context available for the next external cfg.
        engine_start_ready = !deferred_start_q && !i_cfg_valid &&
            (learner_start_ready || ((rearm_needed_q || engine_done) &&
                saved_cfg_valid_q && direct_cfg_ready && !o_error &&
                !o_recovery_required));
        // Once learning is accepted, a queued external cfg cannot stall it:
        // boundary_context_ready blocks cfg_fire until this window retires.
        auto_cfg_valid = rearm_pending_q && saved_cfg_valid_q && configured_q &&
            boundary_learning_active && !engine_busy &&
            !o_recovery_required && !o_error;
        auto_cfg_fire = auto_cfg_valid && direct_cfg_ready;
        // External cfg still clears the engine/reducer's latched faults, but
        // acquires no update context until an accepted learning transaction.
        learner_cfg_valid = (cfg_fire && cfg_legal) || auto_cfg_valid;
        learner_cfg = cfg_fire ? incoming_cfg : saved_cfg_q;
        learner_start_valid = (boundary_learning_start_fire && learner_start_ready) ||
            (deferred_start_q && !rearm_pending_q);
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            saved_cfg_q <= '0;
            saved_cfg_valid_q <= 1'b0;
            rearm_needed_q <= 1'b0;
            rearm_pending_q <= 1'b0;
            deferred_start_q <= 1'b0;
            deferred_abort_q <= 1'b0;
        end else if (cfg_fire) begin
            saved_cfg_valid_q <= learner_cfg_legal;
            if (learner_cfg_legal)
                saved_cfg_q <= incoming_cfg;
            rearm_needed_q <= learner_cfg_legal;
            rearm_pending_q <= 1'b0;
            deferred_start_q <= 1'b0;
            deferred_abort_q <= 1'b0;
        end else begin
            if (engine_done) begin
                rearm_needed_q <= 1'b1;
                deferred_abort_q <= 1'b0;
            end
            if (boundary_learning_start_fire && !learner_start_ready) begin
                rearm_pending_q <= 1'b1;
                deferred_start_q <= 1'b1;
            end
            if (i_learning_abort && (deferred_start_q ||
                    (boundary_learning_start_fire && !learner_start_ready)))
                deferred_abort_q <= 1'b1;
            if (learner_start_valid && learner_start_ready)
                deferred_start_q <= 1'b0;
            if (auto_cfg_fire) begin
                rearm_needed_q <= 1'b0;
                rearm_pending_q <= 1'b0;
            end
        end
    end

    assign o_wh1_tp_contribution_ready = 1'b0;
    assign o_wh1_backbone_ready = 1'b0;

    unified_srnn_tp_wout_direct_online_learning_engine #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH),
        .WINDOW_SEQUENCE_WIDTH(WINDOW_SEQUENCE_WIDTH)
    ) learning_engine (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(learner_cfg_valid),
        .i_cfg_defer_context(cfg_fire),
        .o_cfg_ready(direct_cfg_ready),
        .i_cfg_profile_id(learner_cfg.profile_id),
        .i_cfg_wout_tensor_id(learner_cfg.tensor_id),
        .i_cfg_wout_base(learner_cfg.parameter_base),
        .i_cfg_wout_increment(learner_cfg.increment),
        .i_cfg_wout_threshold(learner_cfg.threshold),
        .i_cfg_code_min(learner_cfg.code_min),
        .i_cfg_code_max(learner_cfg.code_max),
        .i_cfg_batch_count(learner_cfg.batch_count),
        .i_cfg_wout_element_count(learner_cfg.element_count),
        .o_cfg_error(direct_cfg_error),
        .i_start_window(learner_start_valid),
        .o_start_ready(learner_start_ready),
        .i_abort_window(i_learning_abort || deferred_abort_q),
        .i_wout_contribution_valid(i_wout_contribution_valid),
        .o_wout_contribution_ready(o_wout_contribution_ready),
        .i_wout_contribution_addr(i_wout_contribution_addr),
        .i_wout_contribution_sample_id(i_wout_contribution_sample_id),
        .i_wout_error_q40(i_wout_error_q40),
        .i_wout_hidden_trace_q14(i_wout_hidden_trace_q14),
        .i_wout_contribution_last(i_wout_contribution_last),
        .o_read_valid(engine_read_valid),
        .i_read_ready(engine_read_ready),
        .o_read_profile_id(engine_read_profile_id),
        .o_read_tensor_id(engine_read_tensor_id),
        .o_read_parameter_addr(engine_read_addr),
        .o_read_tag(engine_read_tag),
        .i_read_response_valid(engine_read_response_valid),
        .o_read_response_ready(engine_read_response_ready),
        .i_read_response_code(engine_read_response_code),
        .i_read_response_residual(engine_read_response_residual),
        .i_read_response_error(engine_read_response_error),
        .o_result_valid(engine_result_valid),
        .i_result_ready(engine_result_ready),
        .o_result_profile_id(engine_result_profile_id),
        .o_result_tensor_id(engine_result_tensor_id),
        .o_result_parameter_addr(engine_result_addr),
        .o_result_tag(engine_result_tag),
        .o_result_code(engine_result_code),
        .o_result_residual(engine_result_residual),
        .o_write_code_enable(engine_write_code_enable),
        .o_write_residual_enable(engine_write_residual_enable),
        .o_write_code_fire(engine_write_code_fire),
        .o_write_residual_fire(engine_write_residual_fire),
        .o_dirty(engine_dirty),
        .o_saturated(engine_saturated),
        .o_commit_valid(engine_commit_valid),
        .i_commit_ready(boundary_learning_commit_ready),
        .o_commit_profile_id(engine_commit_profile_id),
        .o_commit_window_sequence(engine_commit_window_sequence),
        .o_busy(engine_busy),
        .o_done(engine_done),
        .o_window_aborted(engine_aborted),
        .o_error(engine_error),
        .o_zero_gradient_count(o_zero_gradient_count),
        .o_update_count(o_update_count)
    );
    end
    endgenerate

    generate
        if (USE_INTERNAL_STORE) begin : g_internal_store
            logic unused_external_store_inputs;

            unified_tp_shared_forward_rmw_store #(
                .DEPTH(STORE_DEPTH),
                .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
                .TAG_WIDTH(TAG_WIDTH),
                .CODE_WIDTH(CODE_WIDTH),
                .RESIDUAL_WIDTH(RESIDUAL_WIDTH)
            ) shared_store (
                .i_clk(i_clk),
                .i_rst_n(i_rst_n),
                .i_clear_error(i_clear_error),
                .i_load_valid(i_load_valid),
                .o_load_ready(o_load_ready),
                .i_load_addr(i_load_addr),
                .i_load_code(i_load_code),
                .i_load_residual(i_load_residual),
                .i_learning_read_valid(engine_read_valid),
                .o_learning_read_ready(engine_read_ready),
                .i_learning_read_addr(engine_read_addr),
                .i_learning_read_tag(engine_read_tag),
                .o_learning_response_valid(engine_read_response_valid),
                .i_learning_response_ready(engine_read_response_ready),
                .o_learning_response_tag(store_learning_response_tag),
                .o_learning_response_code(engine_read_response_code),
                .o_learning_response_residual(engine_read_response_residual),
                .o_learning_response_error(engine_read_response_error),
                .i_learning_write_valid(engine_result_valid),
                .o_learning_write_ready(store_learning_write_ready),
                .i_learning_write_addr(engine_result_addr),
                .i_learning_write_code(engine_result_code),
                .i_learning_write_residual(engine_result_residual),
                .i_learning_write_code_enable(engine_write_code_enable),
                .i_learning_write_residual_enable(engine_write_residual_enable),
                .i_forward_read_valid(store_forward_read_valid),
                .o_forward_read_ready(store_forward_read_ready),
                .i_forward_read_addr(store_forward_read_addr),
                .i_forward_read_tag(store_forward_read_tag),
                .o_forward_response_valid(store_forward_response_valid),
                .i_forward_response_ready(store_forward_response_ready),
                .o_forward_response_tag(store_forward_response_tag),
                .o_forward_response_code(store_forward_response_code),
                .o_forward_response_error(store_forward_response_error),
                .o_protocol_error(store_protocol_error)
            );

            always_comb begin
                o_store_load_valid = 1'b0;
                o_store_load_addr = '0;
                o_store_load_code = '0;
                o_store_load_residual = '0;
                o_store_learning_read_valid = 1'b0;
                o_store_learning_read_addr = '0;
                o_store_learning_read_tag = '0;
                o_store_learning_response_ready = 1'b0;
                o_store_learning_write_valid = 1'b0;
                o_store_learning_write_addr = '0;
                o_store_learning_write_code = '0;
                o_store_learning_write_residual = '0;
                o_store_learning_write_code_enable = 1'b0;
                o_store_learning_write_residual_enable = 1'b0;
                o_store_forward_read_valid = 1'b0;
                o_store_forward_read_addr = '0;
                o_store_forward_read_tag = '0;
                o_store_forward_response_ready = 1'b0;
                unused_external_store_inputs = &{1'b0,
                    i_store_load_ready, i_store_learning_read_ready,
                    i_store_learning_response_valid,
                    i_store_learning_response_tag,
                    i_store_learning_response_code,
                    i_store_learning_response_residual,
                    i_store_learning_response_error,
                    i_store_learning_write_ready,
                    i_store_forward_read_ready,
                    i_store_forward_response_valid,
                    i_store_forward_response_tag,
                    i_store_forward_response_code,
                    i_store_forward_response_error,
                    i_store_protocol_error};
            end
        end else begin : g_external_store
            always_comb begin
                o_load_ready = i_store_load_ready;
                o_store_load_valid = i_load_valid;
                o_store_load_addr = i_load_addr;
                o_store_load_code = i_load_code;
                o_store_load_residual = i_load_residual;

                o_store_learning_read_valid = engine_read_valid;
                engine_read_ready = i_store_learning_read_ready;
                o_store_learning_read_addr = engine_read_addr;
                o_store_learning_read_tag = engine_read_tag;
                engine_read_response_valid =
                    i_store_learning_response_valid;
                o_store_learning_response_ready =
                    engine_read_response_ready;
                store_learning_response_tag =
                    i_store_learning_response_tag;
                engine_read_response_code =
                    i_store_learning_response_code;
                engine_read_response_residual =
                    i_store_learning_response_residual;
                engine_read_response_error =
                    i_store_learning_response_error;

                o_store_learning_write_valid = engine_result_valid;
                store_learning_write_ready =
                    i_store_learning_write_ready;
                o_store_learning_write_addr = engine_result_addr;
                o_store_learning_write_code = engine_result_code;
                o_store_learning_write_residual = engine_result_residual;
                o_store_learning_write_code_enable =
                    engine_write_code_enable;
                o_store_learning_write_residual_enable =
                    engine_write_residual_enable;

                o_store_forward_read_valid = store_forward_read_valid;
                store_forward_read_ready = i_store_forward_read_ready;
                o_store_forward_read_addr = store_forward_read_addr;
                o_store_forward_read_tag = store_forward_read_tag;
                store_forward_response_valid =
                    i_store_forward_response_valid;
                o_store_forward_response_ready =
                    store_forward_response_ready;
                store_forward_response_tag =
                    i_store_forward_response_tag;
                store_forward_response_code =
                    i_store_forward_response_code;
                store_forward_response_error =
                    i_store_forward_response_error;
                store_protocol_error = i_store_protocol_error;
            end
        end
    endgenerate

    assign engine_result_ready = store_learning_write_ready;
    assign o_learning_done = engine_done;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            configured_q <= 1'b0;
            cfg_error_q <= 1'b0;
            profile_id_q <= '0;
            wh1_tensor_id_q <= '0;
            wout_tensor_id_q <= '0;
            wh1_base_q <= '0;
            wout_base_q <= '0;
            wh1_count_q <= '0;
            wout_count_q <= '0;
            forward_state_q <= FWD_IDLE;
            forward_phase_q <= 1'b0;
            forward_local_addr_q <= '0;
            forward_token_tensor_q <= '0;
            forward_token_local_q <= '0;
            forward_token_global_q <= '0;
            forward_token_code_q <= '0;
            forward_token_last_q <= 1'b0;
            forward_token_valid_q <= 1'b0;
            forward_error_q <= 1'b0;
            cache_active_q <= 1'b0;
            expected_commit_sequence_q <= '0;
            commit_window_sequence_q <= '0;
            o_forward_parameter_scan_done <= 1'b0;
            o_forward_done <= 1'b0;
        end else begin
            o_forward_parameter_scan_done <= 1'b0;
            o_forward_done <= 1'b0;
            cfg_error_q <= 1'b0;
            if (i_clear_error)
                forward_error_q <= 1'b0;

            if (cfg_fire) begin
                if (!cfg_legal) begin
                    configured_q <= 1'b0;
                    cfg_error_q <= 1'b1;
                end else begin
                    configured_q <= 1'b1;
                    profile_id_q <= i_cfg_profile_id;
                    wh1_tensor_id_q <= i_cfg_wh1_tensor_id;
                    wout_tensor_id_q <= i_cfg_wout_tensor_id;
                    wh1_base_q <= i_cfg_wh1_base;
                    wout_base_q <= i_cfg_wout_base;
                    wh1_count_q <= cfg_wh1_count_ext[PARAM_ADDR_WIDTH-1:0];
                    wout_count_q <= i_cfg_wout_element_count;
                end
            end

            if (store_forward_response_valid && store_forward_response_ready &&
                store_forward_response_error)
                forward_error_q <= 1'b1;
            if (cache_read_fire)
                cache_active_q <= 1'b1;
            if (cache_response_fire)
                cache_active_q <= 1'b0;
            if (learning_metadata_error)
                forward_error_q <= 1'b1;
            if (engine_commit_fire) begin
                commit_window_sequence_q <= engine_commit_window_sequence;
                expected_commit_sequence_q <=
                    expected_commit_sequence_q + 1'b1;
            end

            case (forward_state_q)
                FWD_IDLE: begin
                    if (boundary_forward_start_fire) begin
                        forward_phase_q <= 1'b0;
                        forward_local_addr_q <= '0;
                        forward_token_valid_q <= 1'b0;
                        if (USE_RANDOM_ACCESS_FORWARD) begin
                            o_forward_parameter_scan_done <= 1'b1;
                            forward_state_q <= FWD_DRAIN;
                        end else begin
                            forward_state_q <= FWD_REQ;
                        end
                    end
                end
                FWD_REQ: begin
                    if (forward_read_fire)
                        forward_state_q <= FWD_WAIT;
                end
                FWD_WAIT: begin
                    if (forward_response_fire) begin
                        forward_token_valid_q <= 1'b1;
                        forward_token_tensor_q <= forward_phase_q ?
                            wout_tensor_id_q : wh1_tensor_id_q;
                        forward_token_local_q <= forward_local_addr_q;
                        forward_token_global_q <= store_forward_read_addr;
                        forward_token_code_q <= store_forward_response_code;
                        forward_token_last_q <= forward_last_address;
                        if (store_forward_response_tag !=
                            TAG_WIDTH'(forward_local_addr_q))
                            forward_error_q <= 1'b1;
                        forward_state_q <= FWD_EMIT;
                    end
                end
                FWD_EMIT: begin
                    if (forward_token_fire) begin
                        forward_token_valid_q <= 1'b0;
                        if (forward_token_last_q) begin
                            o_forward_parameter_scan_done <= 1'b1;
                            forward_state_q <= FWD_DRAIN;
                        end else begin
                            forward_phase_q <= next_forward_phase;
                            forward_local_addr_q <= next_forward_addr;
                            forward_state_q <= FWD_REQ;
                        end
                    end
                end
                FWD_DRAIN: begin
                    if (i_forward_consumer_done_valid &&
                        o_forward_consumer_done_ready) begin
                        o_forward_done <= 1'b1;
                        forward_state_q <= FWD_IDLE;
                    end
                end
                default: forward_state_q <= FWD_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
