`timescale 1ns/1ps
`default_nettype none

// FC/SCNN-classifier online-learning slice. Address-major contributions are
// reduced with constant state, then the shared parameter RMW path applies the
// configured direct-residual update without a full gradient tensor RAM.
module unified_fc_tp_w2_learning_slice #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int SOURCE_WIDTH = 16,
    parameter int ERROR_WIDTH = 18,
    parameter int GRAD_ACC_WIDTH = 48,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int MAX_GRADIENT_ELEMENTS = 32768
) (
    input  wire logic                               i_clk,
    input  wire logic                               i_rst_n,

    input  wire logic                               i_cfg_valid,
    output logic                               o_cfg_ready,
    input  wire logic [PROFILE_ID_WIDTH-1:0]         i_cfg_profile_id,
    input  wire logic [TENSOR_ID_WIDTH-1:0]          i_cfg_tensor_id,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]         i_cfg_parameter_base,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]         i_cfg_element_count,
    input  wire logic [VOTE_WIDTH-1:0]               i_cfg_vote_increment,
    input  wire logic [VOTE_WIDTH-1:0]               i_cfg_vote_threshold,
    input  wire logic signed [CODE_WIDTH-1:0]        i_cfg_code_min,
    input  wire logic signed [CODE_WIDTH-1:0]        i_cfg_code_max,
    output logic                               o_cfg_error,
    output logic                               o_active,
    input  wire logic                               i_release,
    output logic                               o_release_ready,

    input  wire logic                               i_contribution_valid,
    output logic                               o_contribution_ready,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]         i_contribution_local_addr,
    input  wire logic [SOURCE_WIDTH-1:0]             i_contribution_source_trace,
    input  wire logic signed [ERROR_WIDTH-1:0]       i_contribution_error,
    output logic                               o_contribution_error,
    output logic                               o_gradient_overflow,

    input  wire logic                               i_window_emit,
    output logic                               o_window_ready,
    output logic                               o_window_done,
    output logic                               o_window_busy,
    output logic                               o_context_error,

    output logic                               o_read_valid,
    input  wire logic                               i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0]         o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]          o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]                o_read_tag,
    input  wire logic                               i_read_response_valid,
    output logic                               o_read_response_ready,
    input  wire logic signed [CODE_WIDTH-1:0]        i_read_response_code,
    input  wire logic signed [RESIDUAL_WIDTH-1:0]    i_read_response_residual,
    input  wire logic                               i_read_response_error,

    output logic                               o_result_valid,
    input  wire logic                               i_result_ready,
    output logic [PROFILE_ID_WIDTH-1:0]         o_result_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]          o_result_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_result_parameter_addr,
    output logic [TAG_WIDTH-1:0]                o_result_tag,
    output logic signed [CODE_WIDTH-1:0]        o_result_code,
    output logic signed [RESIDUAL_WIDTH-1:0]    o_result_residual,
    output logic                               o_write_code_enable,
    output logic                               o_write_residual_enable,
    output logic                               o_write_code_fire,
    output logic                               o_write_residual_fire,
    output logic                               o_dirty,
    output logic                               o_saturated,
    output logic                               o_update_config_error,
    output logic                               o_request_error,
    output logic                               o_store_error,
    output logic                               o_internal_error
);

    localparam logic [PARAM_ADDR_WIDTH:0] ADDRESS_SPACE_LIMIT =
        {1'b1, {PARAM_ADDR_WIDTH{1'b0}}};

    logic gradient_cfg_ready;
    logic gradient_cfg_error;
    logic gradient_active;
    logic [TENSOR_ID_WIDTH-1:0] gradient_active_tensor_id;
    logic gradient_release_ready;
    logic gradient_emit_ready;
    logic gradient_sign_valid;
    logic gradient_sign_ready;
    logic [TENSOR_ID_WIDTH-1:0] gradient_sign_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] gradient_sign_local_addr;
    logic signed [1:0] gradient_sign;
    logic signed [GRAD_ACC_WIDTH-1:0] gradient_accumulator;
    logic gradient_emit_done;
    logic gradient_busy;

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
    logic update_scale_saturated;
    logic update_saturated;
    logic update_busy;
    logic requested_magnitude_mode;
    logic [VOTE_WIDTH-1:0] requested_magnitude_multiplier;
    logic magnitude_mode_q;
    logic [VOTE_WIDTH-1:0] magnitude_multiplier_q;

    logic shared_cfg_valid;
    logic shared_cfg_legal;
    logic shared_release;
    logic window_active_q;
    logic window_flush_requested_q;
    logic gradient_done_seen_q;
    logic cfg_reject_q;
    logic [PROFILE_ID_WIDTH-1:0] active_profile_id_q;
    logic [TAG_WIDTH-1:0] gradient_tag;
    logic [PARAM_ADDR_WIDTH:0] cfg_end_exclusive_ext;
    logic [VOTE_WIDTH:0] cfg_twice_threshold_ext;

    always_comb begin
        cfg_end_exclusive_ext = {1'b0, i_cfg_parameter_base} +
            {1'b0, i_cfg_element_count};
        cfg_twice_threshold_ext = {1'b0, i_cfg_vote_threshold} +
            {1'b0, i_cfg_vote_threshold};
        requested_magnitude_mode =
            i_cfg_vote_threshold == VOTE_WIDTH'(16'h8000);
        // The SCNN adapter stores a Q8.8 trace as code*2 at the Q0.16 port,
        // i.e. real trace / 128.  Its locked B4 fixed-SGD profile therefore
        // needs 5*128 here. FC supplies its explicit UQ6.10 multiplier (11).
        requested_magnitude_multiplier = (i_cfg_vote_increment == '0) ?
            VOTE_WIDTH'(640) : i_cfg_vote_increment;
        shared_cfg_legal = (i_cfg_element_count != '0) &&
            ({1'b0, i_cfg_element_count} <=
             (PARAM_ADDR_WIDTH+1)'(MAX_GRADIENT_ELEMENTS)) &&
            (cfg_end_exclusive_ext <= ADDRESS_SPACE_LIMIT) &&
            (i_cfg_vote_threshold != '0) &&
            ({1'b0, i_cfg_vote_increment} < cfg_twice_threshold_ext) &&
            ($signed(i_cfg_code_min) <= $signed(i_cfg_code_max));
        o_cfg_ready = gradient_cfg_ready && update_cfg_ready &&
            !window_active_q && !window_flush_requested_q;
        shared_cfg_valid = i_cfg_valid && o_cfg_ready && shared_cfg_legal;
        o_release_ready = gradient_release_ready && update_release_ready &&
            !window_active_q && !window_flush_requested_q;
        shared_release = i_release && o_release_ready;
        o_active = gradient_active && update_active &&
            (gradient_active_tensor_id == update_active_tensor_id);
        o_context_error = (gradient_active && !update_active) ||
            (gradient_active && update_active &&
             (gradient_active_tensor_id != update_active_tensor_id)) ||
            (update_active &&
             (update_active_profile_id != active_profile_id_q)) ||
            (gradient_sign_valid && (gradient_accumulator == '0)) ||
            (update_accepted && !window_active_q) ||
            update_tensor_error || update_sign_error ||
            // A nonzero accumulator can quantize to zero after magnitude
            // scaling. That is a valid no-op; direct-sign zero requests remain
            // diagnostic because the reducer must not emit them.
            (!magnitude_mode_q && update_zero_skipped);
        o_cfg_error = cfg_reject_q || gradient_cfg_error || update_cfg_error;
        o_window_ready = o_active && gradient_emit_ready &&
            !window_flush_requested_q;
        o_window_busy = window_active_q || window_flush_requested_q ||
            gradient_busy || update_busy;
        gradient_sign_ready = update_gradient_ready;
        o_saturated = update_saturated || update_scale_saturated;
    end

    generate
        if (TAG_WIDTH >= PARAM_ADDR_WIDTH) begin : gen_wide_tag
            assign gradient_tag = {
                {(TAG_WIDTH-PARAM_ADDR_WIDTH){1'b0}}, gradient_sign_local_addr};
        end else begin : gen_narrow_tag
            assign gradient_tag = gradient_sign_local_addr[TAG_WIDTH-1:0];
        end
    endgenerate

    unified_tp_address_major_sign_reducer #(
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .SOURCE_WIDTH(SOURCE_WIDTH),
        .ERROR_WIDTH(ERROR_WIDTH),
        .ACC_WIDTH(GRAD_ACC_WIDTH),
        .MAX_ELEMENTS(MAX_GRADIENT_ELEMENTS)
    ) gradient_accumulator_i (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_valid),
        .o_cfg_ready(gradient_cfg_ready),
        .i_cfg_tensor_id(i_cfg_tensor_id),
        .i_cfg_element_count(i_cfg_element_count),
        .o_cfg_error(gradient_cfg_error),
        .o_active(gradient_active),
        .o_active_tensor_id(gradient_active_tensor_id),
        .i_release(shared_release),
        .o_release_ready(gradient_release_ready),
        .i_contribution_valid(i_contribution_valid && o_active &&
                              !window_flush_requested_q),
        .o_contribution_ready(o_contribution_ready),
        .i_contribution_local_addr(i_contribution_local_addr),
        .i_contribution_source_trace(i_contribution_source_trace),
        .i_contribution_error(i_contribution_error),
        .o_contribution_error(o_contribution_error),
        .o_accumulator_overflow(o_gradient_overflow),
        .i_emit_start(i_window_emit && o_window_ready),
        .o_emit_ready(gradient_emit_ready),
        .o_sign_valid(gradient_sign_valid),
        .i_sign_ready(gradient_sign_ready),
        .o_sign_tensor_id(gradient_sign_tensor_id),
        .o_sign_local_addr(gradient_sign_local_addr),
        .o_gradient_sign(gradient_sign),
        .o_gradient_accumulator(gradient_accumulator),
        .o_emit_done(gradient_emit_done),
        .o_busy(gradient_busy)
    );

    unified_fc_tp_gradient_rmw_path #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH)
    ) update_path_i (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_valid),
        .o_cfg_ready(update_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id),
        .i_cfg_tensor_id(i_cfg_tensor_id),
        .i_cfg_parameter_base(i_cfg_parameter_base),
        .i_cfg_element_count(i_cfg_element_count),
        .i_cfg_vote_increment(i_cfg_vote_increment),
        .i_cfg_vote_threshold(i_cfg_vote_threshold),
        .i_cfg_code_min(i_cfg_code_min),
        .i_cfg_code_max(i_cfg_code_max),
        .o_cfg_error(update_cfg_error),
        .i_release(shared_release),
        .o_release_ready(update_release_ready),
        .o_active(update_active),
        .o_active_profile_id(update_active_profile_id),
        .o_active_tensor_id(update_active_tensor_id),
        .i_gradient_valid(gradient_sign_valid),
        .o_gradient_ready(update_gradient_ready),
        .i_gradient_tensor_id(gradient_sign_tensor_id),
        .i_gradient_local_parameter_addr(gradient_sign_local_addr),
        .i_gradient_tag(gradient_tag),
        .i_gradient_sign(gradient_sign),
        .i_gradient_accumulator(gradient_accumulator),
        .i_magnitude_mode(magnitude_mode_q),
        .i_magnitude_multiplier(magnitude_multiplier_q),
        .o_gradient_update_accepted(update_accepted),
        .o_gradient_zero_skipped(update_zero_skipped),
        .o_gradient_tensor_error(update_tensor_error),
        .o_gradient_sign_error(update_sign_error),
        .o_gradient_scale_saturated(update_scale_saturated),
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
        .o_saturated(update_saturated),
        .o_update_config_error(o_update_config_error),
        .o_request_error(o_request_error),
        .o_store_error(o_store_error),
        .o_internal_error(o_internal_error),
        .o_busy(update_busy)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            window_active_q <= 1'b0;
            window_flush_requested_q <= 1'b0;
            gradient_done_seen_q <= 1'b0;
            cfg_reject_q <= 1'b0;
            active_profile_id_q <= '0;
            magnitude_mode_q <= 1'b0;
            magnitude_multiplier_q <= '0;
            o_window_done <= 1'b0;
        end else begin
            o_window_done <= 1'b0;
            cfg_reject_q <= i_cfg_valid && o_cfg_ready && !shared_cfg_legal;
            if (shared_cfg_valid) begin
                active_profile_id_q <= i_cfg_profile_id;
                magnitude_mode_q <= requested_magnitude_mode;
                magnitude_multiplier_q <= requested_magnitude_multiplier;
            end
            if (shared_release) begin
                active_profile_id_q <= '0;
                magnitude_mode_q <= 1'b0;
                magnitude_multiplier_q <= '0;
            end
            if (i_contribution_valid && o_contribution_ready)
                window_active_q <= 1'b1;
            if (i_window_emit && o_window_ready) begin
                window_active_q <= 1'b1;
                window_flush_requested_q <= 1'b1;
                gradient_done_seen_q <= 1'b0;
            end
            if (window_flush_requested_q && gradient_emit_done)
                gradient_done_seen_q <= 1'b1;
            if (window_flush_requested_q &&
                (gradient_done_seen_q || gradient_emit_done) &&
                !update_busy && !gradient_sign_valid) begin
                window_active_q <= 1'b0;
                window_flush_requested_q <= 1'b0;
                gradient_done_seen_q <= 1'b0;
                o_window_done <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
