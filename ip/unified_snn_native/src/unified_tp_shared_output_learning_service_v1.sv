`timescale 1ns/1ps
`default_nettype none

// One physical TP output-learning datapath shared by the FC and SCNN clients.
// Both clients present normalized Q0.16 source traces and Q6 logits. Runtime
// layout selects hidden-major FC addressing or output-major SCNN addressing;
// ownership remains locked from accepted configuration through release.
module unified_tp_shared_output_learning_service_v1 #(
    parameter bit ENABLE_FC_CLIENT = 1'b1,
    parameter bit ENABLE_SCNN_CLIENT = 1'b1,
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int MAX_BATCH = 16,
    parameter int MAX_SOURCES = 2048,
    parameter int MAX_OUTPUTS = 16,
    parameter int MAX_GRADIENT_ELEMENTS = 32768,
    parameter int BATCH_INDEX_WIDTH = (MAX_BATCH <= 2) ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int SOURCE_INDEX_WIDTH = (MAX_SOURCES <= 2) ? 1 : $clog2(MAX_SOURCES),
    parameter int SOURCE_COUNT_WIDTH = $clog2(MAX_SOURCES + 1),
    parameter int OUTPUT_INDEX_WIDTH = (MAX_OUTPUTS <= 2) ? 1 : $clog2(MAX_OUTPUTS),
    parameter int OUTPUT_COUNT_WIDTH = $clog2(MAX_OUTPUTS + 1)
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_clear_protocol_error,

    input  logic                                  i_fc_cfg_valid,
    output logic                                  o_fc_cfg_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]           i_fc_cfg_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]            i_fc_cfg_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]           i_fc_cfg_parameter_base,
    input  logic [BATCH_COUNT_WIDTH-1:0]          i_fc_cfg_batch_count,
    input  logic [SOURCE_COUNT_WIDTH-1:0]         i_fc_cfg_source_count,
    input  logic [OUTPUT_COUNT_WIDTH-1:0]         i_fc_cfg_output_count,
    input  logic                                  i_fc_cfg_output_major,
    input  logic [PARAM_ADDR_WIDTH-1:0]           i_fc_cfg_source_stride,
    input  logic [PARAM_ADDR_WIDTH-1:0]           i_fc_cfg_output_stride,
    input  logic [VOTE_WIDTH-1:0]                 i_fc_cfg_vote_increment,
    input  logic [VOTE_WIDTH-1:0]                 i_fc_cfg_vote_threshold,
    input  logic signed [CODE_WIDTH-1:0]          i_fc_cfg_code_min,
    input  logic signed [CODE_WIDTH-1:0]          i_fc_cfg_code_max,
    output logic                                  o_fc_cfg_error,
    output logic                                  o_fc_active,
    input  logic                                  i_fc_release,
    output logic                                  o_fc_release_ready,
    input  logic                                  i_fc_source_trace_valid,
    output logic                                  o_fc_source_trace_ready,
    input  logic [BATCH_INDEX_WIDTH-1:0]          i_fc_source_trace_sample_id,
    input  logic [SOURCE_INDEX_WIDTH-1:0]         i_fc_source_trace_index,
    input  logic [15:0]                           i_fc_source_trace_q016,
    output logic                                  o_fc_source_trace_error,
    input  logic                                  i_fc_logit_row_valid,
    output logic                                  o_fc_logit_row_ready,
    input  logic [BATCH_INDEX_WIDTH-1:0]          i_fc_logit_sample_id,
    input  logic [OUTPUT_INDEX_WIDTH-1:0]         i_fc_logit_label,
    input  logic                                  i_fc_logit_valid,
    output logic                                  o_fc_logit_ready,
    input  logic signed [15:0]                    i_fc_logit_q6,
    output logic                                  o_fc_logit_error,
    output logic                                  o_fc_logit_row_done,
    input  logic                                  i_fc_observation_done,
    output logic                                  o_fc_observation_done_ready,
    output logic                                  o_fc_window_done,
    output logic                                  o_fc_window_busy,
    output logic                                  o_fc_context_error,
    output logic                                  o_fc_gradient_overflow,

    input  logic                                  i_scnn_cfg_valid,
    output logic                                  o_scnn_cfg_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]           i_scnn_cfg_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]            i_scnn_cfg_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]           i_scnn_cfg_parameter_base,
    input  logic [BATCH_COUNT_WIDTH-1:0]          i_scnn_cfg_batch_count,
    input  logic [SOURCE_COUNT_WIDTH-1:0]         i_scnn_cfg_source_count,
    input  logic [OUTPUT_COUNT_WIDTH-1:0]         i_scnn_cfg_output_count,
    input  logic                                  i_scnn_cfg_output_major,
    input  logic [PARAM_ADDR_WIDTH-1:0]           i_scnn_cfg_source_stride,
    input  logic [PARAM_ADDR_WIDTH-1:0]           i_scnn_cfg_output_stride,
    input  logic [VOTE_WIDTH-1:0]                 i_scnn_cfg_vote_increment,
    input  logic [VOTE_WIDTH-1:0]                 i_scnn_cfg_vote_threshold,
    input  logic signed [CODE_WIDTH-1:0]          i_scnn_cfg_code_min,
    input  logic signed [CODE_WIDTH-1:0]          i_scnn_cfg_code_max,
    output logic                                  o_scnn_cfg_error,
    output logic                                  o_scnn_active,
    input  logic                                  i_scnn_release,
    output logic                                  o_scnn_release_ready,
    input  logic                                  i_scnn_source_trace_valid,
    output logic                                  o_scnn_source_trace_ready,
    input  logic [BATCH_INDEX_WIDTH-1:0]          i_scnn_source_trace_sample_id,
    input  logic [SOURCE_INDEX_WIDTH-1:0]         i_scnn_source_trace_index,
    input  logic [15:0]                           i_scnn_source_trace_q016,
    output logic                                  o_scnn_source_trace_error,
    input  logic                                  i_scnn_logit_row_valid,
    output logic                                  o_scnn_logit_row_ready,
    input  logic [BATCH_INDEX_WIDTH-1:0]          i_scnn_logit_sample_id,
    input  logic [OUTPUT_INDEX_WIDTH-1:0]         i_scnn_logit_label,
    input  logic                                  i_scnn_logit_valid,
    output logic                                  o_scnn_logit_ready,
    input  logic signed [15:0]                    i_scnn_logit_q6,
    output logic                                  o_scnn_logit_error,
    output logic                                  o_scnn_logit_row_done,
    input  logic                                  i_scnn_observation_done,
    output logic                                  o_scnn_observation_done_ready,
    output logic                                  o_scnn_window_done,
    output logic                                  o_scnn_window_busy,
    output logic                                  o_scnn_context_error,
    output logic                                  o_scnn_gradient_overflow,

    output logic [1:0]                            o_active_owner,
    output logic                                  o_busy,
    output logic                                  o_protocol_error,

    output logic                                  o_read_valid,
    input  logic                                  i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0]           o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]            o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]           o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]                  o_read_tag,
    input  logic                                  i_read_response_valid,
    output logic                                  o_read_response_ready,
    input  logic signed [CODE_WIDTH-1:0]          i_read_response_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]      i_read_response_residual,
    input  logic                                  i_read_response_error,
    output logic                                  o_result_valid,
    input  logic                                  i_result_ready,
    output logic [PROFILE_ID_WIDTH-1:0]           o_result_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]            o_result_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]           o_result_parameter_addr,
    output logic [TAG_WIDTH-1:0]                  o_result_tag,
    output logic signed [CODE_WIDTH-1:0]          o_result_code,
    output logic signed [RESIDUAL_WIDTH-1:0]      o_result_residual,
    output logic                                  o_write_code_enable,
    output logic                                  o_write_residual_enable,
    output logic                                  o_write_code_fire,
    output logic                                  o_write_residual_fire,
    output logic                                  o_dirty,
    output logic                                  o_saturated,
    output logic                                  o_update_config_error,
    output logic                                  o_request_error,
    output logic                                  o_store_error,
    output logic                                  o_internal_error
);

    typedef enum logic [1:0] {
        OWNER_NONE = 2'd0,
        OWNER_FC = 2'd1,
        OWNER_SCNN = 2'd2
    } owner_t;

    owner_t owner_q;
    logic select_fc_cfg;
    logic select_scnn_cfg;
    logic core_cfg_valid;
    logic core_cfg_ready;
    logic core_cfg_error;
    logic core_active;
    logic core_release;
    logic core_release_ready;
    logic core_source_trace_valid;
    logic core_source_trace_ready;
    logic [BATCH_INDEX_WIDTH-1:0] core_source_trace_sample_id;
    logic [SOURCE_INDEX_WIDTH-1:0] core_source_trace_index;
    logic [15:0] core_source_trace_q016;
    logic core_source_trace_error;
    logic core_logit_row_valid;
    logic core_logit_row_ready;
    logic [BATCH_INDEX_WIDTH-1:0] core_logit_sample_id;
    logic [OUTPUT_INDEX_WIDTH-1:0] core_logit_label;
    logic core_logit_valid;
    logic core_logit_ready;
    logic signed [15:0] core_logit_q6;
    logic core_logit_error;
    logic core_logit_row_done;
    logic core_observation_done;
    logic core_observation_done_ready;
    logic core_window_done;
    logic core_window_busy;
    logic core_context_error;
    logic core_gradient_overflow;
    logic core_cfg_fire;
    logic core_release_fire;
    logic fc_cfg_numeric_legal;
    logic fc_cfg_reject;
    logic fc_cfg_reject_q;
    logic fc_non_cfg_activity;
    logic scnn_non_cfg_activity;

    logic [PROFILE_ID_WIDTH-1:0] selected_cfg_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] selected_cfg_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] selected_cfg_parameter_base;
    logic [BATCH_COUNT_WIDTH-1:0] selected_cfg_batch_count;
    logic [SOURCE_COUNT_WIDTH-1:0] selected_cfg_source_count;
    logic [OUTPUT_COUNT_WIDTH-1:0] selected_cfg_output_count;
    logic selected_cfg_output_major;
    logic [PARAM_ADDR_WIDTH-1:0] selected_cfg_source_stride;
    logic [PARAM_ADDR_WIDTH-1:0] selected_cfg_output_stride;
    logic [VOTE_WIDTH-1:0] selected_cfg_vote_increment;
    logic [VOTE_WIDTH-1:0] selected_cfg_vote_threshold;
    logic signed [CODE_WIDTH-1:0] selected_cfg_code_min;
    logic signed [CODE_WIDTH-1:0] selected_cfg_code_max;

    always_comb begin
        // Threshold 0x8000 reserves the shared magnitude-residual mode. A
        // zero increment selects the B4xT4 SCNN multiplier 640; a nonzero
        // increment supplies the profile multiplier (FC UQ6.10 uses 11).
        fc_cfg_numeric_legal = 1'b1;
        select_fc_cfg = ENABLE_FC_CLIENT && (owner_q == OWNER_NONE) &&
            i_fc_cfg_valid && fc_cfg_numeric_legal;
        select_scnn_cfg = ENABLE_SCNN_CLIENT &&
            (owner_q == OWNER_NONE) &&
            !(ENABLE_FC_CLIENT && i_fc_cfg_valid) && i_scnn_cfg_valid;
        core_cfg_valid = select_fc_cfg || select_scnn_cfg;
        o_fc_cfg_ready = ENABLE_FC_CLIENT && (owner_q == OWNER_NONE) &&
            core_cfg_ready;
        o_scnn_cfg_ready = ENABLE_SCNN_CLIENT &&
            (owner_q == OWNER_NONE) &&
            !(ENABLE_FC_CLIENT && i_fc_cfg_valid) && core_cfg_ready;
        core_cfg_fire = core_cfg_valid && core_cfg_ready;
        fc_cfg_reject = ENABLE_FC_CLIENT && (owner_q == OWNER_NONE) &&
            i_fc_cfg_valid && core_cfg_ready && !fc_cfg_numeric_legal;

        selected_cfg_profile_id = select_scnn_cfg ? i_scnn_cfg_profile_id :
            i_fc_cfg_profile_id;
        selected_cfg_tensor_id = select_scnn_cfg ? i_scnn_cfg_tensor_id :
            i_fc_cfg_tensor_id;
        selected_cfg_parameter_base = select_scnn_cfg ?
            i_scnn_cfg_parameter_base : i_fc_cfg_parameter_base;
        selected_cfg_batch_count = select_scnn_cfg ? i_scnn_cfg_batch_count :
            i_fc_cfg_batch_count;
        selected_cfg_source_count = select_scnn_cfg ? i_scnn_cfg_source_count :
            i_fc_cfg_source_count;
        selected_cfg_output_count = select_scnn_cfg ? i_scnn_cfg_output_count :
            i_fc_cfg_output_count;
        selected_cfg_output_major = select_scnn_cfg ? i_scnn_cfg_output_major :
            i_fc_cfg_output_major;
        selected_cfg_source_stride = select_scnn_cfg ?
            i_scnn_cfg_source_stride : i_fc_cfg_source_stride;
        selected_cfg_output_stride = select_scnn_cfg ?
            i_scnn_cfg_output_stride : i_fc_cfg_output_stride;
        selected_cfg_vote_increment = select_scnn_cfg ?
            i_scnn_cfg_vote_increment : i_fc_cfg_vote_increment;
        selected_cfg_vote_threshold = select_scnn_cfg ?
            i_scnn_cfg_vote_threshold : i_fc_cfg_vote_threshold;
        selected_cfg_code_min = select_scnn_cfg ? i_scnn_cfg_code_min :
            i_fc_cfg_code_min;
        selected_cfg_code_max = select_scnn_cfg ? i_scnn_cfg_code_max :
            i_fc_cfg_code_max;

        core_release = (owner_q == OWNER_FC) ? i_fc_release :
            ((owner_q == OWNER_SCNN) ? i_scnn_release : 1'b0);
        core_release_fire = core_release && core_release_ready;
        core_source_trace_valid = (owner_q == OWNER_FC) ?
            i_fc_source_trace_valid : ((owner_q == OWNER_SCNN) ?
            i_scnn_source_trace_valid : 1'b0);
        core_source_trace_sample_id = (owner_q == OWNER_SCNN) ?
            i_scnn_source_trace_sample_id : i_fc_source_trace_sample_id;
        core_source_trace_index = (owner_q == OWNER_SCNN) ?
            i_scnn_source_trace_index : i_fc_source_trace_index;
        core_source_trace_q016 = (owner_q == OWNER_SCNN) ?
            i_scnn_source_trace_q016 : i_fc_source_trace_q016;
        core_logit_row_valid = (owner_q == OWNER_FC) ? i_fc_logit_row_valid :
            ((owner_q == OWNER_SCNN) ? i_scnn_logit_row_valid : 1'b0);
        core_logit_sample_id = (owner_q == OWNER_SCNN) ?
            i_scnn_logit_sample_id : i_fc_logit_sample_id;
        core_logit_label = (owner_q == OWNER_SCNN) ? i_scnn_logit_label :
            i_fc_logit_label;
        core_logit_valid = (owner_q == OWNER_FC) ? i_fc_logit_valid :
            ((owner_q == OWNER_SCNN) ? i_scnn_logit_valid : 1'b0);
        core_logit_q6 = (owner_q == OWNER_SCNN) ? i_scnn_logit_q6 :
            i_fc_logit_q6;
        core_observation_done = (owner_q == OWNER_FC) ? i_fc_observation_done :
            ((owner_q == OWNER_SCNN) ? i_scnn_observation_done : 1'b0);

        o_fc_cfg_error = ENABLE_FC_CLIENT && (fc_cfg_reject_q ||
            ((owner_q == OWNER_FC) && core_cfg_error));
        o_fc_active = ENABLE_FC_CLIENT && (owner_q == OWNER_FC) &&
            core_active;
        o_fc_release_ready = ENABLE_FC_CLIENT &&
            (owner_q == OWNER_FC) && core_release_ready;
        o_fc_source_trace_ready = ENABLE_FC_CLIENT &&
            (owner_q == OWNER_FC) &&
            core_source_trace_ready;
        o_fc_source_trace_error = ENABLE_FC_CLIENT &&
            (owner_q == OWNER_FC) &&
            core_source_trace_error;
        o_fc_logit_row_ready = ENABLE_FC_CLIENT &&
            (owner_q == OWNER_FC) && core_logit_row_ready;
        o_fc_logit_ready = ENABLE_FC_CLIENT &&
            (owner_q == OWNER_FC) && core_logit_ready;
        o_fc_logit_error = ENABLE_FC_CLIENT &&
            (owner_q == OWNER_FC) && core_logit_error;
        o_fc_logit_row_done = ENABLE_FC_CLIENT &&
            (owner_q == OWNER_FC) && core_logit_row_done;
        o_fc_observation_done_ready = ENABLE_FC_CLIENT &&
            (owner_q == OWNER_FC) &&
            core_observation_done_ready;
        o_fc_window_done = ENABLE_FC_CLIENT &&
            (owner_q == OWNER_FC) && core_window_done;
        o_fc_window_busy = ENABLE_FC_CLIENT &&
            (owner_q == OWNER_FC) && core_window_busy;
        o_fc_context_error = ENABLE_FC_CLIENT &&
            (owner_q == OWNER_FC) && core_context_error;
        o_fc_gradient_overflow = ENABLE_FC_CLIENT &&
            (owner_q == OWNER_FC) &&
            core_gradient_overflow;

        o_scnn_cfg_error = ENABLE_SCNN_CLIENT &&
            (owner_q == OWNER_SCNN) && core_cfg_error;
        o_scnn_active = ENABLE_SCNN_CLIENT &&
            (owner_q == OWNER_SCNN) && core_active;
        o_scnn_release_ready = ENABLE_SCNN_CLIENT &&
            (owner_q == OWNER_SCNN) && core_release_ready;
        o_scnn_source_trace_ready = ENABLE_SCNN_CLIENT &&
            (owner_q == OWNER_SCNN) &&
            core_source_trace_ready;
        o_scnn_source_trace_error = ENABLE_SCNN_CLIENT &&
            (owner_q == OWNER_SCNN) &&
            core_source_trace_error;
        o_scnn_logit_row_ready = ENABLE_SCNN_CLIENT &&
            (owner_q == OWNER_SCNN) &&
            core_logit_row_ready;
        o_scnn_logit_ready = ENABLE_SCNN_CLIENT &&
            (owner_q == OWNER_SCNN) && core_logit_ready;
        o_scnn_logit_error = ENABLE_SCNN_CLIENT &&
            (owner_q == OWNER_SCNN) && core_logit_error;
        o_scnn_logit_row_done = ENABLE_SCNN_CLIENT &&
            (owner_q == OWNER_SCNN) && core_logit_row_done;
        o_scnn_observation_done_ready = ENABLE_SCNN_CLIENT &&
            (owner_q == OWNER_SCNN) &&
            core_observation_done_ready;
        o_scnn_window_done = ENABLE_SCNN_CLIENT &&
            (owner_q == OWNER_SCNN) && core_window_done;
        o_scnn_window_busy = ENABLE_SCNN_CLIENT &&
            (owner_q == OWNER_SCNN) && core_window_busy;
        o_scnn_context_error = ENABLE_SCNN_CLIENT &&
            (owner_q == OWNER_SCNN) && core_context_error;
        o_scnn_gradient_overflow = ENABLE_SCNN_CLIENT &&
            (owner_q == OWNER_SCNN) &&
            core_gradient_overflow;

        o_active_owner = owner_q;
        o_busy = (owner_q != OWNER_NONE) || core_window_busy;
        fc_non_cfg_activity = ENABLE_FC_CLIENT &&
            (i_fc_release || i_fc_source_trace_valid ||
             i_fc_logit_row_valid || i_fc_logit_valid ||
             i_fc_observation_done);
        scnn_non_cfg_activity = ENABLE_SCNN_CLIENT &&
            (i_scnn_release || i_scnn_source_trace_valid ||
             i_scnn_logit_row_valid || i_scnn_logit_valid ||
             i_scnn_observation_done);
    end

    generate
    if (ENABLE_FC_CLIENT || ENABLE_SCNN_CLIENT) begin : g_output_learning_core
    unified_fc_tp_w2_online_learning_path #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH),
        .MAX_BATCH(MAX_BATCH),
        .MAX_HIDDEN(MAX_SOURCES),
        .MAX_OUTPUTS(MAX_OUTPUTS),
        .MAX_GRADIENT_ELEMENTS(MAX_GRADIENT_ELEMENTS),
        .BATCH_INDEX_WIDTH(BATCH_INDEX_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .HIDDEN_INDEX_WIDTH(SOURCE_INDEX_WIDTH),
        .HIDDEN_COUNT_WIDTH(SOURCE_COUNT_WIDTH),
        .OUTPUT_INDEX_WIDTH(OUTPUT_INDEX_WIDTH),
        .OUTPUT_COUNT_WIDTH(OUTPUT_COUNT_WIDTH)
    ) u_shared_output_learning_path (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(core_cfg_valid),
        .o_cfg_ready(core_cfg_ready),
        .i_cfg_profile_id(selected_cfg_profile_id),
        .i_cfg_tensor_id(selected_cfg_tensor_id),
        .i_cfg_parameter_base(selected_cfg_parameter_base),
        .i_cfg_batch_count(selected_cfg_batch_count),
        .i_cfg_hidden_count(selected_cfg_source_count),
        .i_cfg_output_count(selected_cfg_output_count),
        .i_cfg_output_major(selected_cfg_output_major),
        .i_cfg_hidden_stride(selected_cfg_source_stride),
        .i_cfg_output_stride(selected_cfg_output_stride),
        .i_cfg_vote_increment(selected_cfg_vote_increment),
        .i_cfg_vote_threshold(selected_cfg_vote_threshold),
        .i_cfg_code_min(selected_cfg_code_min),
        .i_cfg_code_max(selected_cfg_code_max),
        .o_cfg_error(core_cfg_error),
        .o_active(core_active),
        .i_release(core_release),
        .o_release_ready(core_release_ready),
        .i_hidden_trace_valid(core_source_trace_valid),
        .o_hidden_trace_ready(core_source_trace_ready),
        .i_hidden_trace_sample_id(core_source_trace_sample_id),
        .i_hidden_trace_hidden_index(core_source_trace_index),
        .i_hidden_trace_q016(core_source_trace_q016),
        .o_hidden_trace_error(core_source_trace_error),
        .i_logit_row_valid(core_logit_row_valid),
        .o_logit_row_ready(core_logit_row_ready),
        .i_logit_sample_id(core_logit_sample_id),
        .i_logit_label(core_logit_label),
        .i_logit_valid(core_logit_valid),
        .o_logit_ready(core_logit_ready),
        .i_logit_q6(core_logit_q6),
        .o_logit_error(core_logit_error),
        .o_logit_row_done(core_logit_row_done),
        .i_observation_done(core_observation_done),
        .o_observation_done_ready(core_observation_done_ready),
        .o_window_done(core_window_done),
        .o_window_busy(core_window_busy),
        .o_context_error(core_context_error),
        .o_gradient_overflow(core_gradient_overflow),
        .o_read_valid(o_read_valid),
        .i_read_ready(i_read_ready),
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
        .o_update_config_error(o_update_config_error),
        .o_request_error(o_request_error),
        .o_store_error(o_store_error),
        .o_internal_error(o_internal_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            owner_q <= OWNER_NONE;
            fc_cfg_reject_q <= 1'b0;
            o_protocol_error <= 1'b0;
        end else begin
            fc_cfg_reject_q <= fc_cfg_reject;
            if (i_clear_protocol_error)
                o_protocol_error <= 1'b0;

            if (owner_q == OWNER_NONE) begin
                if (core_cfg_fire)
                    owner_q <= select_fc_cfg ? OWNER_FC : OWNER_SCNN;
                if (fc_non_cfg_activity || scnn_non_cfg_activity)
                    o_protocol_error <= 1'b1;
            end else begin
                if (core_release_fire || (core_cfg_error && !core_active))
                    owner_q <= OWNER_NONE;
                if (((owner_q == OWNER_FC) && scnn_non_cfg_activity) ||
                    ((owner_q == OWNER_SCNN) && fc_non_cfg_activity))
                    o_protocol_error <= 1'b1;
            end
        end
    end
    end else begin : g_no_output_learning_clients
        assign owner_q = OWNER_NONE;
        assign fc_cfg_reject_q = 1'b0;
        assign o_protocol_error = 1'b0;
        assign core_cfg_ready = 1'b0;
        assign core_cfg_error = 1'b0;
        assign core_active = 1'b0;
        assign core_release_ready = 1'b0;
        assign core_source_trace_ready = 1'b0;
        assign core_source_trace_error = 1'b0;
        assign core_logit_row_ready = 1'b0;
        assign core_logit_ready = 1'b0;
        assign core_logit_error = 1'b0;
        assign core_logit_row_done = 1'b0;
        assign core_observation_done_ready = 1'b0;
        assign core_window_done = 1'b0;
        assign core_window_busy = 1'b0;
        assign core_context_error = 1'b0;
        assign core_gradient_overflow = 1'b0;
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
        assign o_update_config_error = 1'b0;
        assign o_request_error = 1'b0;
        assign o_store_error = 1'b0;
        assign o_internal_error = 1'b0;
    end
    endgenerate

endmodule

`default_nettype wire
