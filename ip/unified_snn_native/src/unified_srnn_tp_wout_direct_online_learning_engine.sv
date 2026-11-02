`timescale 1ns/1ps
`default_nettype none

// Wout-only SRNN TP update path. W_h1 remains available to inference but owns
// no mutable learning state in this build. Address-major Q40 reduction keeps
// the original SRNN arithmetic, while one topology-neutral residual RMW path
// applies the sign update and retires the window through one commit fence.
module unified_srnn_tp_wout_direct_online_learning_engine #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int SAMPLE_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int WINDOW_SEQUENCE_WIDTH = 32
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,

    input  logic                                      i_cfg_valid,
    input  logic                                      i_cfg_defer_context = 1'b0,
    output logic                                      o_cfg_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]               i_cfg_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_cfg_wout_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_wout_base,
    input  logic [VOTE_WIDTH-1:0]                     i_cfg_wout_increment,
    input  logic [VOTE_WIDTH-1:0]                     i_cfg_wout_threshold,
    input  logic signed [CODE_WIDTH-1:0]              i_cfg_code_min,
    input  logic signed [CODE_WIDTH-1:0]              i_cfg_code_max,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_cfg_batch_count,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_wout_element_count,
    output logic                                      o_cfg_error,

    input  logic                                      i_start_window,
    output logic                                      o_start_ready,
    input  logic                                      i_abort_window,

    input  logic                                      i_wout_contribution_valid,
    output logic                                      o_wout_contribution_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_wout_contribution_addr,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_wout_contribution_sample_id,
    input  logic signed [63:0]                        i_wout_error_q40,
    input  logic signed [31:0]                        i_wout_hidden_trace_q14,
    input  logic                                      i_wout_contribution_last,

    output logic                                      o_read_valid,
    input  logic                                      i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0]               o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]                o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]                      o_read_tag,
    input  logic                                      i_read_response_valid,
    output logic                                      o_read_response_ready,
    input  logic signed [CODE_WIDTH-1:0]              i_read_response_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]          i_read_response_residual,
    input  logic                                      i_read_response_error,

    output logic                                      o_result_valid,
    input  logic                                      i_result_ready,
    output logic [PROFILE_ID_WIDTH-1:0]               o_result_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]                o_result_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_result_parameter_addr,
    output logic [TAG_WIDTH-1:0]                      o_result_tag,
    output logic signed [CODE_WIDTH-1:0]              o_result_code,
    output logic signed [RESIDUAL_WIDTH-1:0]          o_result_residual,
    output logic                                      o_write_code_enable,
    output logic                                      o_write_residual_enable,
    output logic                                      o_write_code_fire,
    output logic                                      o_write_residual_fire,
    output logic                                      o_dirty,
    output logic                                      o_saturated,

    output logic                                      o_commit_valid,
    input  logic                                      i_commit_ready,
    output logic [PROFILE_ID_WIDTH-1:0]               o_commit_profile_id,
    output logic [WINDOW_SEQUENCE_WIDTH-1:0]          o_commit_window_sequence,
    output logic                                      o_busy,
    output logic                                      o_done,
    output logic                                      o_window_aborted,
    output logic                                      o_error,
    output logic [PARAM_ADDR_WIDTH:0]                 o_zero_gradient_count,
    output logic [PARAM_ADDR_WIDTH:0]                 o_update_count
);

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_STREAM,
        STATE_DRAIN,
        STATE_RELEASE,
        STATE_COMMIT,
        STATE_ABORT_DRAIN,
        STATE_ABORT_RELEASE,
        STATE_ABORT_DONE
    } state_t;

    state_t state_q;
    logic configured_q;
    logic [PROFILE_ID_WIDTH-1:0] profile_id_q;
    logic [WINDOW_SEQUENCE_WIDTH-1:0] window_sequence_q;
    logic cfg_reject_q;
    logic error_q;

    logic reducer_cfg_ready;
    logic reducer_cfg_error;
    logic reducer_start_ready;
    logic reducer_gradient_valid;
    logic reducer_gradient_ready;
    logic [PARAM_ADDR_WIDTH-1:0] reducer_gradient_addr;
    logic signed [63:0] reducer_gradient_q40;
    logic reducer_gradient_last;
    logic reducer_busy;
    logic reducer_done;
    logic reducer_protocol_error;
    logic reducer_overflow;

    logic update_cfg_ready;
    logic update_cfg_error;
    logic update_active;
    logic [PROFILE_ID_WIDTH-1:0] update_active_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] update_active_tensor_id;
    logic update_release_ready;
    logic update_gradient_ready;
    logic update_accepted;
    logic update_zero_skipped;
    logic update_tensor_error;
    logic update_sign_error;
    logic update_config_error;
    logic update_request_error;
    logic update_store_error;
    logic update_internal_error;
    logic update_busy;

    logic cfg_legal;
    logic shared_cfg_fire;
    logic shared_start_fire;
    logic release_request;
    logic release_fire;
    logic gradient_fire;
    logic signed [1:0] gradient_sign;
    logic [TAG_WIDTH-1:0] gradient_tag;
    logic datapath_error;

    always_comb begin
        cfg_legal = (i_cfg_wout_element_count != '0) &&
            (i_cfg_wout_threshold != '0) &&
            ({1'b0, i_cfg_wout_increment} <
                ({1'b0, i_cfg_wout_threshold} +
                 {1'b0, i_cfg_wout_threshold})) &&
            ($signed(i_cfg_code_min) <= $signed(i_cfg_code_max));
        o_cfg_ready = (state_q == STATE_IDLE) && reducer_cfg_ready &&
            update_cfg_ready;
        shared_cfg_fire = i_cfg_valid && o_cfg_ready && cfg_legal;
        o_start_ready = (state_q == STATE_IDLE) && configured_q &&
            reducer_start_ready && update_active && !i_cfg_valid;
        shared_start_fire = i_start_window && o_start_ready;

        if (reducer_gradient_q40 > 0)
            gradient_sign = 2'sd1;
        else if (reducer_gradient_q40 < 0)
            gradient_sign = -2'sd1;
        else
            gradient_sign = 2'sd0;

        reducer_gradient_ready = (state_q == STATE_STREAM) ?
            update_gradient_ready : 1'b0;
        gradient_fire = reducer_gradient_valid && reducer_gradient_ready;
        release_request = (state_q == STATE_RELEASE) ||
            (state_q == STATE_ABORT_RELEASE);
        release_fire = release_request && update_release_ready;

        o_commit_valid = state_q == STATE_COMMIT;
        o_commit_profile_id = profile_id_q;
        o_commit_window_sequence = window_sequence_q;
        o_busy = (state_q != STATE_IDLE) || reducer_busy || update_busy;
        o_cfg_error = cfg_reject_q || reducer_cfg_error || update_cfg_error;
        datapath_error = reducer_protocol_error || reducer_overflow ||
            update_tensor_error || update_sign_error || update_config_error ||
            update_request_error || update_store_error || update_internal_error;
        o_error = error_q || o_cfg_error || datapath_error;
    end

    generate
        if (TAG_WIDTH >= PARAM_ADDR_WIDTH) begin : gen_wide_tag
            assign gradient_tag = {
                {(TAG_WIDTH-PARAM_ADDR_WIDTH){1'b0}}, reducer_gradient_addr};
        end else begin : gen_narrow_tag
            assign gradient_tag = reducer_gradient_addr[TAG_WIDTH-1:0];
        end
    endgenerate

    unified_srnn_tp_wout_gradient_reducer #(
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH)
    ) u_wout_reducer (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_fire),
        .o_cfg_ready(reducer_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_element_count(i_cfg_wout_element_count),
        .o_cfg_error(reducer_cfg_error),
        .i_start(shared_start_fire),
        .o_start_ready(reducer_start_ready),
        .i_abort(i_abort_window || datapath_error),
        .i_contribution_valid(i_wout_contribution_valid &&
            (state_q == STATE_STREAM)),
        .o_contribution_ready(o_wout_contribution_ready),
        .i_contribution_local_addr(i_wout_contribution_addr),
        .i_contribution_sample_id(i_wout_contribution_sample_id),
        .i_error_q40(i_wout_error_q40),
        .i_hidden_trace_q14(i_wout_hidden_trace_q14),
        .i_contribution_tensor_last(i_wout_contribution_last),
        .o_gradient_valid(reducer_gradient_valid),
        .i_gradient_ready(reducer_gradient_ready),
        .o_gradient_local_addr(reducer_gradient_addr),
        .o_gradient_q40(reducer_gradient_q40),
        .o_gradient_tensor_last(reducer_gradient_last),
        .o_busy(reducer_busy),
        .o_done(reducer_done),
        .o_protocol_error(reducer_protocol_error),
        .o_overflow(reducer_overflow)
    );

    unified_fc_tp_gradient_rmw_path #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH)
    ) u_update_path (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_fire && !i_cfg_defer_context),
        .o_cfg_ready(update_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id),
        .i_cfg_tensor_id(i_cfg_wout_tensor_id),
        .i_cfg_parameter_base(i_cfg_wout_base),
        .i_cfg_element_count(i_cfg_wout_element_count),
        .i_cfg_vote_increment(i_cfg_wout_increment),
        .i_cfg_vote_threshold(i_cfg_wout_threshold),
        .i_cfg_code_min(i_cfg_code_min),
        .i_cfg_code_max(i_cfg_code_max),
        .o_cfg_error(update_cfg_error),
        .i_release(release_request),
        .o_release_ready(update_release_ready),
        .o_active(update_active),
        .o_active_profile_id(update_active_profile_id),
        .o_active_tensor_id(update_active_tensor_id),
        .i_gradient_valid(reducer_gradient_valid &&
            (state_q == STATE_STREAM)),
        .o_gradient_ready(update_gradient_ready),
        .i_gradient_tensor_id(update_active_tensor_id),
        .i_gradient_local_parameter_addr(reducer_gradient_addr),
        .i_gradient_tag(gradient_tag),
        .i_gradient_sign(gradient_sign),
        .i_gradient_accumulator('0),
        .i_magnitude_mode(1'b0),
        .i_magnitude_multiplier('0),
        .o_gradient_update_accepted(update_accepted),
        .o_gradient_zero_skipped(update_zero_skipped),
        .o_gradient_tensor_error(update_tensor_error),
        .o_gradient_sign_error(update_sign_error),
        .o_gradient_scale_saturated(),
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
        .o_update_config_error(update_config_error),
        .o_request_error(update_request_error),
        .o_store_error(update_store_error),
        .o_internal_error(update_internal_error),
        .o_busy(update_busy)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            configured_q <= 1'b0;
            profile_id_q <= '0;
            window_sequence_q <= '0;
            cfg_reject_q <= 1'b0;
            error_q <= 1'b0;
            o_done <= 1'b0;
            o_window_aborted <= 1'b0;
            o_zero_gradient_count <= '0;
            o_update_count <= '0;
        end else begin
            cfg_reject_q <= 1'b0;
            o_done <= 1'b0;
            o_window_aborted <= 1'b0;

            if (i_cfg_valid && o_cfg_ready) begin
                configured_q <= cfg_legal;
                cfg_reject_q <= !cfg_legal;
                error_q <= 1'b0;
                if (cfg_legal)
                    profile_id_q <= i_cfg_profile_id;
            end

            if (update_zero_skipped)
                o_zero_gradient_count <= o_zero_gradient_count + 1'b1;
            if (update_accepted)
                o_update_count <= o_update_count + 1'b1;

            if (datapath_error) begin
                error_q <= 1'b1;
                if ((state_q != STATE_IDLE) &&
                    (state_q != STATE_ABORT_DRAIN) &&
                    (state_q != STATE_ABORT_RELEASE) &&
                    (state_q != STATE_ABORT_DONE))
                    state_q <= STATE_ABORT_DRAIN;
            end else begin
                unique case (state_q)
                    STATE_IDLE: begin
                        if (shared_start_fire) begin
                            o_zero_gradient_count <= '0;
                            o_update_count <= '0;
                            state_q <= STATE_STREAM;
                        end
                    end
                    STATE_STREAM: begin
                        if (i_abort_window)
                            state_q <= STATE_ABORT_DRAIN;
                        else if (reducer_done ||
                                (gradient_fire && reducer_gradient_last))
                            state_q <= STATE_DRAIN;
                    end
                    STATE_DRAIN: begin
                        if (i_abort_window)
                            state_q <= STATE_ABORT_DRAIN;
                        else if (!update_busy && !reducer_busy &&
                                 !reducer_gradient_valid)
                            state_q <= STATE_RELEASE;
                    end
                    STATE_RELEASE: begin
                        if (release_fire)
                            state_q <= STATE_COMMIT;
                    end
                    STATE_COMMIT: begin
                        if (o_commit_valid && i_commit_ready) begin
                            window_sequence_q <= window_sequence_q + 1'b1;
                            o_done <= 1'b1;
                            state_q <= STATE_IDLE;
                        end
                    end
                    STATE_ABORT_DRAIN: begin
                        if (!update_busy && !reducer_busy &&
                            !reducer_gradient_valid)
                            state_q <= STATE_ABORT_RELEASE;
                    end
                    STATE_ABORT_RELEASE: begin
                        if (release_fire)
                            state_q <= STATE_ABORT_DONE;
                    end
                    STATE_ABORT_DONE: begin
                        o_done <= 1'b1;
                        o_window_aborted <= 1'b1;
                        state_q <= STATE_IDLE;
                    end
                    default: state_q <= STATE_ABORT_DRAIN;
                endcase
            end
        end
    end

    logic _unused_update_context;
    always_comb begin
        _unused_update_context = ^{update_active_profile_id,
            update_active_tensor_id};
    end

endmodule

`default_nettype wire
