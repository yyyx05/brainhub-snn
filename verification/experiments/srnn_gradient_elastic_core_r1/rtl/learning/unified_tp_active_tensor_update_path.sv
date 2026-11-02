`timescale 1ns/1ps
`default_nettype none

// Topology-neutral direct-update slice. A scheduler loads one active tensor
// descriptor, then FC/SRNN/SCNN adapters stream local parameter requests through
// the same context, arithmetic leaf and writeback response interface.
module unified_tp_active_tensor_update_path #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int DELTA_WIDTH = 18
) (
    input  wire logic                               i_clk,
    input  wire logic                               i_rst_n,

    input  wire logic                               i_cfg_valid,
    output logic                               o_cfg_ready,
    input  wire logic [PROFILE_ID_WIDTH-1:0]        i_cfg_profile_id,
    input  wire logic [TENSOR_ID_WIDTH-1:0]         i_cfg_tensor_id,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_cfg_parameter_base,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_cfg_element_count,
    input  wire logic        [VOTE_WIDTH-1:0]       i_cfg_vote_increment,
    input  wire logic        [VOTE_WIDTH-1:0]       i_cfg_vote_threshold,
    input  wire logic signed [CODE_WIDTH-1:0]       i_cfg_code_min,
    input  wire logic signed [CODE_WIDTH-1:0]       i_cfg_code_max,
    output logic                               o_cfg_error,

    input  wire logic                               i_release,
    output logic                               o_release_ready,
    output logic                               o_active,
    output logic [PROFILE_ID_WIDTH-1:0]        o_active_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]         o_active_tensor_id,

    input  wire logic                               i_req_valid,
    output logic                               o_req_ready,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_req_local_parameter_addr,
    input  wire logic [TAG_WIDTH-1:0]               i_req_tag,
    input  wire logic signed [CODE_WIDTH-1:0]       i_req_code,
    input  wire logic signed [RESIDUAL_WIDTH-1:0]   i_req_residual,
    input  wire logic signed [1:0]                  i_req_update_direction,
    input  wire logic                               i_req_magnitude_mode,
    input  wire logic signed [DELTA_WIDTH-1:0]      i_req_scaled_residual_delta,
    output logic                               o_req_error,

    output logic                               o_valid,
    input  wire logic                               i_ready,
    output logic [PROFILE_ID_WIDTH-1:0]        o_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]         o_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]        o_parameter_addr,
    output logic [TAG_WIDTH-1:0]               o_tag,
    output logic signed [CODE_WIDTH-1:0]       o_code,
    output logic signed [RESIDUAL_WIDTH-1:0]   o_residual,
    output logic                               o_write_code_enable,
    output logic                               o_write_residual_enable,
    output logic                               o_write_code_fire,
    output logic                               o_write_residual_fire,
    output logic                               o_dirty,
    output logic                               o_saturated,
    output logic                               o_update_config_error
);

    logic context_valid;
    logic context_ready;
    logic [PROFILE_ID_WIDTH-1:0] context_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] context_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] context_parameter_addr;
    logic [TAG_WIDTH-1:0] context_tag;
    logic signed [CODE_WIDTH-1:0] context_code;
    logic signed [RESIDUAL_WIDTH-1:0] context_residual;
    logic signed [1:0] context_update_direction;
    logic context_magnitude_mode;
    logic signed [DELTA_WIDTH-1:0] context_scaled_residual_delta;
    logic [VOTE_WIDTH-1:0] context_vote_increment;
    logic [VOTE_WIDTH-1:0] context_vote_threshold;
    logic signed [CODE_WIDTH-1:0] context_code_min;
    logic signed [CODE_WIDTH-1:0] context_code_max;

    unified_tp_active_tensor_context #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH),
        .DELTA_WIDTH(DELTA_WIDTH)
    ) active_context (
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
        .i_req_valid(i_req_valid),
        .o_req_ready(o_req_ready),
        .i_req_local_parameter_addr(i_req_local_parameter_addr),
        .i_req_tag(i_req_tag),
        .i_req_code(i_req_code),
        .i_req_residual(i_req_residual),
        .i_req_update_direction(i_req_update_direction),
        .i_req_magnitude_mode(i_req_magnitude_mode),
        .i_req_scaled_residual_delta(i_req_scaled_residual_delta),
        .o_req_error(o_req_error),
        .o_valid(context_valid),
        .i_ready(context_ready),
        .o_profile_id(context_profile_id),
        .o_tensor_id(context_tensor_id),
        .o_parameter_addr(context_parameter_addr),
        .o_tag(context_tag),
        .o_code(context_code),
        .o_residual(context_residual),
        .o_update_direction(context_update_direction),
        .o_magnitude_mode(context_magnitude_mode),
        .o_scaled_residual_delta(context_scaled_residual_delta),
        .o_vote_increment(context_vote_increment),
        .o_vote_threshold(context_vote_threshold),
        .o_code_min(context_code_min),
        .o_code_max(context_code_max)
    );

    unified_tp_parameter_update_adapter #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH),
        .DELTA_WIDTH(DELTA_WIDTH)
    ) parameter_update (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_valid(context_valid),
        .o_ready(context_ready),
        .i_profile_id(context_profile_id),
        .i_tensor_id(context_tensor_id),
        .i_parameter_addr(context_parameter_addr),
        .i_tag(context_tag),
        .i_code(context_code),
        .i_residual(context_residual),
        .i_update_direction(context_update_direction),
        .i_magnitude_mode(context_magnitude_mode),
        .i_scaled_residual_delta(context_scaled_residual_delta),
        .i_vote_increment(context_vote_increment),
        .i_vote_threshold(context_vote_threshold),
        .i_code_min(context_code_min),
        .i_code_max(context_code_max),
        .o_valid(o_valid),
        .i_ready(i_ready),
        .o_profile_id(o_profile_id),
        .o_tensor_id(o_tensor_id),
        .o_parameter_addr(o_parameter_addr),
        .o_tag(o_tag),
        .o_code(o_code),
        .o_residual(o_residual),
        .o_write_code_enable(o_write_code_enable),
        .o_write_residual_enable(o_write_residual_enable),
        .o_write_code_fire(o_write_code_fire),
        .o_write_residual_fire(o_write_residual_fire),
        .o_dirty(o_dirty),
        .o_saturated(o_saturated),
        .o_config_error(o_update_config_error)
    );

endmodule

`default_nettype wire
