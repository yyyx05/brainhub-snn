`timescale 1ns/1ps
`default_nettype none

module unified_fc_tp_gradient_rmw_path #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int GRADIENT_WIDTH = 48,
    parameter int DELTA_WIDTH = 18
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
    input  wire logic                               i_release,
    output logic                               o_release_ready,
    output logic                               o_active,
    output logic [PROFILE_ID_WIDTH-1:0]         o_active_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]          o_active_tensor_id,

    input  wire logic                               i_gradient_valid,
    output logic                               o_gradient_ready,
    input  wire logic [TENSOR_ID_WIDTH-1:0]          i_gradient_tensor_id,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]         i_gradient_local_parameter_addr,
    input  wire logic [TAG_WIDTH-1:0]                i_gradient_tag,
    input  wire logic signed [1:0]                   i_gradient_sign,
    input  wire logic signed [GRADIENT_WIDTH-1:0]    i_gradient_accumulator,
    input  wire logic                                i_magnitude_mode,
    input  wire logic [VOTE_WIDTH-1:0]               i_magnitude_multiplier,
    output logic                               o_gradient_update_accepted,
    output logic                               o_gradient_zero_skipped,
    output logic                               o_gradient_tensor_error,
    output logic                               o_gradient_sign_error,
    output logic                               o_gradient_scale_saturated,

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
    output logic                               o_internal_error,
    output logic                               o_busy
);

    logic req_valid;
    logic req_ready;
    logic [PARAM_ADDR_WIDTH-1:0] req_local_parameter_addr;
    logic [TAG_WIDTH-1:0] req_tag;
    logic signed [1:0] req_update_direction;
    logic req_magnitude_mode;
    logic signed [DELTA_WIDTH-1:0] req_scaled_residual_delta;
    logic request_adapter_busy;
    logic rmw_service_busy;

    assign o_busy = request_adapter_busy || rmw_service_busy;

    unified_fc_tp_gradient_request_adapter #(
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .GRADIENT_WIDTH(GRADIENT_WIDTH),
        .DELTA_WIDTH(DELTA_WIDTH)
    ) request_adapter (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_active(o_active),
        .i_active_tensor_id(o_active_tensor_id),
        .i_magnitude_mode(i_magnitude_mode),
        .i_magnitude_multiplier(i_magnitude_multiplier),
        .i_gradient_valid(i_gradient_valid),
        .o_gradient_ready(o_gradient_ready),
        .i_gradient_tensor_id(i_gradient_tensor_id),
        .i_gradient_local_parameter_addr(i_gradient_local_parameter_addr),
        .i_gradient_tag(i_gradient_tag),
        .i_gradient_sign(i_gradient_sign),
        .i_gradient_accumulator(i_gradient_accumulator),
        .o_req_valid(req_valid),
        .i_req_ready(req_ready),
        .o_req_local_parameter_addr(req_local_parameter_addr),
        .o_req_tag(req_tag),
        .o_req_update_direction(req_update_direction),
        .o_req_magnitude_mode(req_magnitude_mode),
        .o_req_scaled_residual_delta(req_scaled_residual_delta),
        .o_update_accepted(o_gradient_update_accepted),
        .o_zero_skipped(o_gradient_zero_skipped),
        .o_tensor_error(o_gradient_tensor_error),
        .o_sign_error(o_gradient_sign_error),
        .o_scale_saturated(o_gradient_scale_saturated),
        .o_busy(request_adapter_busy)
    );

    unified_tp_parameter_rmw_service #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH)
    ) rmw_service (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(i_cfg_valid),
        .o_cfg_ready(o_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id),
        .i_cfg_tensor_id(i_cfg_tensor_id),
        .i_cfg_parameter_base(i_cfg_parameter_base),
        .i_cfg_element_count(i_cfg_element_count),
        .i_cfg_vote_increment(i_cfg_vote_increment),
        .i_cfg_vote_threshold(i_cfg_vote_threshold),
        .i_cfg_code_min(i_cfg_code_min),
        .i_cfg_code_max(i_cfg_code_max),
        .o_cfg_error(o_cfg_error),
        .i_release(i_release),
        .o_release_ready(o_release_ready),
        .o_active(o_active),
        .o_active_profile_id(o_active_profile_id),
        .o_active_tensor_id(o_active_tensor_id),
        .i_req_valid(req_valid),
        .o_req_ready(req_ready),
        .i_req_local_parameter_addr(req_local_parameter_addr),
        .i_req_tag(req_tag),
        .i_req_update_direction(req_update_direction),
        .i_req_magnitude_mode(req_magnitude_mode),
        .i_req_scaled_residual_delta(req_scaled_residual_delta),
        .o_req_error(o_request_error),
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
        .o_store_error(o_store_error),
        .o_internal_error(o_internal_error),
        .o_busy(rmw_service_busy)
    );

endmodule

`default_nettype wire
