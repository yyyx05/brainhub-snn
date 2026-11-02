`timescale 1ns/1ps
`default_nettype none

// Complete FC TP W2 online-learning path. Hidden traces and labelled logits
// are converted into sparse outer-product contributions and then committed
// through the shared direct-residual parameter RMW service.
module unified_fc_tp_w2_online_learning_path #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int MAX_BATCH = 16,
    parameter int MAX_HIDDEN = 400,
    parameter int MAX_OUTPUTS = 16,
    parameter int MAX_GRADIENT_ELEMENTS = 32768,
    parameter bit OUTPUT_MAJOR_ADDRESS = 1'b0,
    parameter int PARAMETER_HIDDEN_STRIDE = 0,
    parameter int PARAMETER_OUTPUT_STRIDE = 0,
    parameter int BATCH_INDEX_WIDTH = (MAX_BATCH <= 2) ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int HIDDEN_INDEX_WIDTH = (MAX_HIDDEN <= 2) ? 1 : $clog2(MAX_HIDDEN),
    parameter int HIDDEN_COUNT_WIDTH = $clog2(MAX_HIDDEN + 1),
    parameter int OUTPUT_INDEX_WIDTH = (MAX_OUTPUTS <= 2) ? 1 : $clog2(MAX_OUTPUTS),
    parameter int OUTPUT_COUNT_WIDTH = $clog2(MAX_OUTPUTS + 1)
) (
    input  wire logic                                  i_clk,
    input  wire logic                                  i_rst_n,
    input  wire logic                                  i_cfg_valid,
    output logic                                  o_cfg_ready,
    input  wire logic [PROFILE_ID_WIDTH-1:0]           i_cfg_profile_id,
    input  wire logic [TENSOR_ID_WIDTH-1:0]            i_cfg_tensor_id,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]           i_cfg_parameter_base,
    input  wire logic [BATCH_COUNT_WIDTH-1:0]          i_cfg_batch_count,
    input  wire logic [HIDDEN_COUNT_WIDTH-1:0]         i_cfg_hidden_count,
    input  wire logic [OUTPUT_COUNT_WIDTH-1:0]         i_cfg_output_count,
    input  wire logic                                  i_cfg_output_major,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]           i_cfg_hidden_stride,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]           i_cfg_output_stride,
    input  wire logic [VOTE_WIDTH-1:0]                 i_cfg_vote_increment,
    input  wire logic [VOTE_WIDTH-1:0]                 i_cfg_vote_threshold,
    input  wire logic signed [CODE_WIDTH-1:0]          i_cfg_code_min,
    input  wire logic signed [CODE_WIDTH-1:0]          i_cfg_code_max,
    output logic                                  o_cfg_error,
    output logic                                  o_active,
    input  wire logic                                  i_release,
    output logic                                  o_release_ready,

    input  wire logic                                  i_hidden_trace_valid,
    output logic                                  o_hidden_trace_ready,
    input  wire logic [BATCH_INDEX_WIDTH-1:0]          i_hidden_trace_sample_id,
    input  wire logic [HIDDEN_INDEX_WIDTH-1:0]         i_hidden_trace_hidden_index,
    input  wire logic [15:0]                           i_hidden_trace_q016,
    output logic                                  o_hidden_trace_error,
    input  wire logic                                  i_logit_row_valid,
    output logic                                  o_logit_row_ready,
    input  wire logic [BATCH_INDEX_WIDTH-1:0]          i_logit_sample_id,
    input  wire logic [OUTPUT_INDEX_WIDTH-1:0]         i_logit_label,
    input  wire logic                                  i_logit_valid,
    output logic                                  o_logit_ready,
    input  wire logic signed [15:0]                    i_logit_q6,
    output logic                                  o_logit_error,
    output logic                                  o_logit_row_done,
    input  wire logic                                  i_observation_done,
    output logic                                  o_observation_done_ready,

    output logic                                  o_window_done,
    output logic                                  o_window_busy,
    output logic                                  o_context_error,
    output logic                                  o_gradient_overflow,
    output logic                                  o_read_valid,
    input  wire logic                                  i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0]           o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]            o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]           o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]                  o_read_tag,
    input  wire logic                                  i_read_response_valid,
    output logic                                  o_read_response_ready,
    input  wire logic signed [CODE_WIDTH-1:0]          i_read_response_code,
    input  wire logic signed [RESIDUAL_WIDTH-1:0]      i_read_response_residual,
    input  wire logic                                  i_read_response_error,
    output logic                                  o_result_valid,
    input  wire logic                                  i_result_ready,
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

    localparam logic [PARAM_ADDR_WIDTH:0] ADDRESS_SPACE_LIMIT =
        {1'b1, {PARAM_ADDR_WIDTH{1'b0}}};

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_OBSERVE,
        STATE_START_FRONTEND,
        STATE_CONTRIBUTE,
        STATE_START_UPDATE,
        STATE_UPDATE
    } state_t;

    state_t state_q;
    logic frontend_cfg_ready;
    logic frontend_cfg_error;
    logic frontend_active;
    logic frontend_release_ready;
    logic frontend_hidden_trace_ready;
    logic frontend_logit_row_ready;
    logic frontend_logit_ready;
    logic frontend_observation_done_ready;
    logic frontend_emit_ready;
    logic frontend_emit_start;
    logic frontend_contribution_valid;
    logic frontend_contribution_ready;
    logic [PARAM_ADDR_WIDTH-1:0] frontend_contribution_local_addr;
    logic [15:0] frontend_contribution_source_trace;
    logic signed [17:0] frontend_contribution_error;
    logic frontend_emit_done;
    logic frontend_busy;

    logic learning_cfg_ready;
    logic learning_cfg_error;
    logic learning_active;
    logic learning_release_ready;
    logic learning_contribution_error;
    logic learning_window_ready;
    logic learning_window_emit;
    logic learning_window_done;
    logic learning_window_busy;
    logic learning_context_error;

    logic cfg_legal;
    logic cfg_fire;
    logic shared_cfg_valid;
    logic release_fire;
    logic observation_fire;
    logic cfg_reject_q;
    logic [PARAM_ADDR_WIDTH:0] element_count_ext;
    logic [PARAM_ADDR_WIDTH:0] end_exclusive_ext;
    logic [VOTE_WIDTH:0] twice_threshold_ext;
    logic [PARAM_ADDR_WIDTH:0] address_stride_ext;

    always_comb begin
        address_stride_ext = i_cfg_output_major ?
            ((i_cfg_hidden_stride != '0) ? {1'b0, i_cfg_hidden_stride} :
             ((PARAMETER_HIDDEN_STRIDE != 0) ?
              (PARAM_ADDR_WIDTH+1)'(PARAMETER_HIDDEN_STRIDE) :
              (PARAM_ADDR_WIDTH+1)'(i_cfg_hidden_count))) :
            ((i_cfg_output_stride != '0) ? {1'b0, i_cfg_output_stride} :
             ((PARAMETER_OUTPUT_STRIDE != 0) ?
              (PARAM_ADDR_WIDTH+1)'(PARAMETER_OUTPUT_STRIDE) :
              (PARAM_ADDR_WIDTH+1)'(i_cfg_output_count)));
        element_count_ext = '0;
        if ((i_cfg_hidden_count != '0) && (i_cfg_output_count != '0)) begin
            if (i_cfg_output_major)
                element_count_ext =
                    ((PARAM_ADDR_WIDTH+1)'(i_cfg_output_count - 1'b1) *
                     address_stride_ext) +
                    (PARAM_ADDR_WIDTH+1)'(i_cfg_hidden_count);
            else
                element_count_ext =
                    ((PARAM_ADDR_WIDTH+1)'(i_cfg_hidden_count - 1'b1) *
                     address_stride_ext) +
                    (PARAM_ADDR_WIDTH+1)'(i_cfg_output_count);
        end
        end_exclusive_ext = {1'b0, i_cfg_parameter_base} + element_count_ext;
        twice_threshold_ext = {1'b0, i_cfg_vote_threshold} +
            {1'b0, i_cfg_vote_threshold};
        cfg_legal = (i_cfg_batch_count != '0) &&
            (i_cfg_batch_count <= BATCH_COUNT_WIDTH'(MAX_BATCH)) &&
            (i_cfg_hidden_count != '0) &&
            (i_cfg_hidden_count <= HIDDEN_COUNT_WIDTH'(MAX_HIDDEN)) &&
            (i_cfg_output_count != '0) &&
            (i_cfg_output_count <= OUTPUT_COUNT_WIDTH'(MAX_OUTPUTS)) &&
            (address_stride_ext >= (i_cfg_output_major ?
             (PARAM_ADDR_WIDTH+1)'(i_cfg_hidden_count) :
             (PARAM_ADDR_WIDTH+1)'(i_cfg_output_count))) &&
            (element_count_ext <= (PARAM_ADDR_WIDTH+1)'(MAX_GRADIENT_ELEMENTS)) &&
            (end_exclusive_ext <= ADDRESS_SPACE_LIMIT) &&
            (i_cfg_vote_threshold != '0) &&
            ({1'b0, i_cfg_vote_increment} < twice_threshold_ext) &&
            ($signed(i_cfg_code_min) <= $signed(i_cfg_code_max));
        o_cfg_ready = (state_q == STATE_IDLE) && frontend_cfg_ready &&
            learning_cfg_ready;
        cfg_fire = i_cfg_valid && o_cfg_ready;
        shared_cfg_valid = cfg_fire && cfg_legal;
        o_cfg_error = cfg_reject_q || frontend_cfg_error || learning_cfg_error;
        o_active = (state_q != STATE_IDLE) && frontend_active && learning_active;
        o_release_ready = (state_q == STATE_OBSERVE) &&
            frontend_release_ready && learning_release_ready;
        release_fire = i_release && o_release_ready;

        o_observation_done_ready = (state_q == STATE_OBSERVE) &&
            frontend_observation_done_ready;
        observation_fire = i_observation_done && o_observation_done_ready;
        o_hidden_trace_ready = (state_q == STATE_OBSERVE) &&
            frontend_hidden_trace_ready;
        o_logit_row_ready = (state_q == STATE_OBSERVE) &&
            frontend_logit_row_ready;
        o_logit_ready = (state_q == STATE_OBSERVE) && frontend_logit_ready;
        frontend_emit_start = (state_q == STATE_START_FRONTEND) &&
            frontend_emit_ready;
        learning_window_emit = (state_q == STATE_START_UPDATE) &&
            learning_window_ready;

        o_window_done = learning_window_done;
        o_window_busy = frontend_busy || learning_window_busy ||
            ((state_q != STATE_IDLE) && (state_q != STATE_OBSERVE));
        o_context_error = learning_context_error || learning_contribution_error;
    end

    unified_fc_tp_w2_online_frontend #(
        .MAX_BATCH(MAX_BATCH),
        .MAX_HIDDEN(MAX_HIDDEN),
        .MAX_OUTPUTS(MAX_OUTPUTS),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .OUTPUT_MAJOR_ADDRESS(OUTPUT_MAJOR_ADDRESS),
        .PARAMETER_HIDDEN_STRIDE(PARAMETER_HIDDEN_STRIDE),
        .PARAMETER_OUTPUT_STRIDE(PARAMETER_OUTPUT_STRIDE),
        .BATCH_INDEX_WIDTH(BATCH_INDEX_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .HIDDEN_COUNT_WIDTH(HIDDEN_COUNT_WIDTH),
        .OUTPUT_INDEX_WIDTH(OUTPUT_INDEX_WIDTH),
        .OUTPUT_COUNT_WIDTH(OUTPUT_COUNT_WIDTH)
    ) u_frontend (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_valid),
        .o_cfg_ready(frontend_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_hidden_count(i_cfg_hidden_count),
        .i_cfg_output_count(i_cfg_output_count),
        .i_cfg_output_major(i_cfg_output_major),
        .i_cfg_hidden_stride(i_cfg_hidden_stride),
        .i_cfg_output_stride(i_cfg_output_stride),
        .o_cfg_error(frontend_cfg_error),
        .o_active(frontend_active),
        .i_release(release_fire),
        .o_release_ready(frontend_release_ready),
        .i_hidden_trace_valid(i_hidden_trace_valid &&
                              (state_q == STATE_OBSERVE)),
        .o_hidden_trace_ready(frontend_hidden_trace_ready),
        .i_hidden_trace_sample_id(i_hidden_trace_sample_id),
        .i_hidden_trace_hidden_index(i_hidden_trace_hidden_index),
        .i_hidden_trace_q016(i_hidden_trace_q016),
        .o_hidden_trace_error(o_hidden_trace_error),
        .i_logit_row_valid(i_logit_row_valid && (state_q == STATE_OBSERVE)),
        .o_logit_row_ready(frontend_logit_row_ready),
        .i_logit_sample_id(i_logit_sample_id),
        .i_logit_label(i_logit_label),
        .i_logit_valid(i_logit_valid && (state_q == STATE_OBSERVE)),
        .o_logit_ready(frontend_logit_ready),
        .i_logit_q6(i_logit_q6),
        .o_logit_error(o_logit_error),
        .o_logit_row_done(o_logit_row_done),
        .i_observation_done(observation_fire),
        .o_observation_done_ready(frontend_observation_done_ready),
        .i_emit_start(frontend_emit_start),
        .o_emit_ready(frontend_emit_ready),
        .o_contribution_valid(frontend_contribution_valid),
        .i_contribution_ready(frontend_contribution_ready),
        .o_contribution_local_addr(frontend_contribution_local_addr),
        .o_contribution_source_trace(frontend_contribution_source_trace),
        .o_contribution_error_q016(frontend_contribution_error),
        .o_emit_done(frontend_emit_done),
        .o_busy(frontend_busy)
    );

    unified_fc_tp_w2_learning_slice #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH),
        .MAX_GRADIENT_ELEMENTS(MAX_GRADIENT_ELEMENTS)
    ) u_learning_slice (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_valid),
        .o_cfg_ready(learning_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id),
        .i_cfg_tensor_id(i_cfg_tensor_id),
        .i_cfg_parameter_base(i_cfg_parameter_base),
        .i_cfg_element_count(element_count_ext[PARAM_ADDR_WIDTH-1:0]),
        .i_cfg_vote_increment(i_cfg_vote_increment),
        .i_cfg_vote_threshold(i_cfg_vote_threshold),
        .i_cfg_code_min(i_cfg_code_min),
        .i_cfg_code_max(i_cfg_code_max),
        .o_cfg_error(learning_cfg_error),
        .o_active(learning_active),
        .i_release(release_fire),
        .o_release_ready(learning_release_ready),
        .i_contribution_valid(frontend_contribution_valid &&
                              (state_q == STATE_CONTRIBUTE)),
        .o_contribution_ready(frontend_contribution_ready),
        .i_contribution_local_addr(frontend_contribution_local_addr),
        .i_contribution_source_trace(frontend_contribution_source_trace),
        .i_contribution_error(frontend_contribution_error),
        .o_contribution_error(learning_contribution_error),
        .o_gradient_overflow(o_gradient_overflow),
        .i_window_emit(learning_window_emit),
        .o_window_ready(learning_window_ready),
        .o_window_done(learning_window_done),
        .o_window_busy(learning_window_busy),
        .o_context_error(learning_context_error),
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
            state_q <= STATE_IDLE;
            cfg_reject_q <= 1'b0;
        end else begin
            cfg_reject_q <= cfg_fire && !cfg_legal;
            case (state_q)
                STATE_IDLE: begin
                    if (shared_cfg_valid)
                        state_q <= STATE_OBSERVE;
                end

                STATE_OBSERVE: begin
                    if (release_fire)
                        state_q <= STATE_IDLE;
                    else if (observation_fire)
                        state_q <= STATE_START_FRONTEND;
                end

                STATE_START_FRONTEND: begin
                    if (frontend_emit_start)
                        state_q <= STATE_CONTRIBUTE;
                end

                STATE_CONTRIBUTE: begin
                    if (frontend_emit_done)
                        state_q <= STATE_START_UPDATE;
                end

                STATE_START_UPDATE: begin
                    if (learning_window_emit)
                        state_q <= STATE_UPDATE;
                end

                STATE_UPDATE: begin
                    if (learning_window_done)
                        state_q <= STATE_OBSERVE;
                end

                default: state_q <= STATE_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
