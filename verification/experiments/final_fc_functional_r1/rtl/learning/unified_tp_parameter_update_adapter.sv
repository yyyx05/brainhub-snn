`timescale 1ns/1ps
`default_nettype none

module unified_tp_parameter_update_adapter #(
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

    input  wire logic                               i_valid,
    output logic                               o_ready,
    input  wire logic [PROFILE_ID_WIDTH-1:0]        i_profile_id,
    input  wire logic [TENSOR_ID_WIDTH-1:0]         i_tensor_id,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_parameter_addr,
    input  wire logic [TAG_WIDTH-1:0]               i_tag,
    input  wire logic signed [CODE_WIDTH-1:0]       i_code,
    input  wire logic signed [RESIDUAL_WIDTH-1:0]   i_residual,
    input  wire logic signed [1:0]                  i_update_direction,
    input  wire logic                               i_magnitude_mode,
    input  wire logic signed [DELTA_WIDTH-1:0]      i_scaled_residual_delta,
    input  wire logic        [VOTE_WIDTH-1:0]       i_vote_increment,
    input  wire logic        [VOTE_WIDTH-1:0]       i_vote_threshold,
    input  wire logic signed [CODE_WIDTH-1:0]       i_code_min,
    input  wire logic signed [CODE_WIDTH-1:0]       i_code_max,

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
    output logic                               o_config_error
);

    logic leaf_ready;
    logic leaf_valid;
    logic signed [CODE_WIDTH-1:0] leaf_code;
    logic signed [RESIDUAL_WIDTH-1:0] leaf_residual;
    logic leaf_code_changed;
    logic leaf_saturated;
    logic leaf_config_error;
    logic direct_ready;
    logic direct_valid;
    logic signed [CODE_WIDTH-1:0] direct_code;
    logic signed [RESIDUAL_WIDTH-1:0] direct_residual;
    logic direct_code_changed;
    logic direct_saturated;
    logic direct_config_error;
    logic magnitude_ready;
    logic magnitude_valid;
    logic signed [CODE_WIDTH-1:0] magnitude_code;
    logic signed [RESIDUAL_WIDTH-1:0] magnitude_residual;
    logic magnitude_code_changed;
    logic magnitude_saturated;
    logic magnitude_config_error;

    logic [PROFILE_ID_WIDTH-1:0] profile_id_q;
    logic [TENSOR_ID_WIDTH-1:0] tensor_id_q;
    logic [PARAM_ADDR_WIDTH-1:0] parameter_addr_q;
    logic [TAG_WIDTH-1:0] tag_q;
    logic signed [RESIDUAL_WIDTH-1:0] residual_in_q;

    logic stage_valid_q;
    logic [PROFILE_ID_WIDTH-1:0] stage_profile_id_q;
    logic [TENSOR_ID_WIDTH-1:0] stage_tensor_id_q;
    logic [PARAM_ADDR_WIDTH-1:0] stage_parameter_addr_q;
    logic [TAG_WIDTH-1:0] stage_tag_q;
    logic signed [CODE_WIDTH-1:0] stage_code_q;
    logic signed [RESIDUAL_WIDTH-1:0] stage_residual_q;
    logic signed [1:0] stage_update_direction_q;
    logic stage_magnitude_mode_q;
    logic signed [DELTA_WIDTH-1:0] stage_scaled_residual_delta_q;
    logic [VOTE_WIDTH-1:0] stage_vote_increment_q;
    logic [VOTE_WIDTH-1:0] stage_vote_threshold_q;
    logic signed [CODE_WIDTH-1:0] stage_code_min_q;
    logic signed [CODE_WIDTH-1:0] stage_code_max_q;

    wire stage_ready = !stage_valid_q || leaf_ready;
    wire stage_accept = i_valid && stage_ready;
    wire leaf_accept = stage_valid_q && leaf_ready;

    always_comb begin
        leaf_ready = stage_magnitude_mode_q ? magnitude_ready : direct_ready;
        leaf_valid = direct_valid || magnitude_valid;
        leaf_code = magnitude_valid ? magnitude_code : direct_code;
        leaf_residual = magnitude_valid ? magnitude_residual : direct_residual;
        leaf_code_changed = magnitude_valid ?
            magnitude_code_changed : direct_code_changed;
        leaf_saturated = magnitude_valid ?
            magnitude_saturated : direct_saturated;
        leaf_config_error = magnitude_valid ?
            magnitude_config_error : direct_config_error;
    end

    assign o_ready = stage_ready;
    assign o_valid = leaf_valid;
    assign o_profile_id = profile_id_q;
    assign o_tensor_id = tensor_id_q;
    assign o_parameter_addr = parameter_addr_q;
    assign o_tag = tag_q;
    assign o_code = leaf_code;
    assign o_residual = leaf_residual;
    assign o_saturated = leaf_saturated;
    assign o_config_error = leaf_config_error;
    assign o_write_code_enable = leaf_valid && !leaf_config_error &&
                                 leaf_code_changed;
    assign o_write_residual_enable = leaf_valid && !leaf_config_error &&
                                     (leaf_residual != residual_in_q);
    assign o_write_code_fire = o_write_code_enable && i_ready;
    assign o_write_residual_fire = o_write_residual_enable && i_ready;
    assign o_dirty = o_write_code_enable || o_write_residual_enable;

    unified_tp_direct_residual_update #(
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH)
    ) update_leaf (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_valid(stage_valid_q && !stage_magnitude_mode_q),
        .o_ready(direct_ready),
        .i_code(stage_code_q),
        .i_residual(stage_residual_q),
        .i_update_direction(stage_update_direction_q),
        .i_vote_increment(stage_vote_increment_q),
        .i_vote_threshold(stage_vote_threshold_q),
        .i_code_min(stage_code_min_q),
        .i_code_max(stage_code_max_q),
        .o_valid(direct_valid),
        .i_ready(i_ready),
        .o_code(direct_code),
        .o_residual(direct_residual),
        .o_code_changed(direct_code_changed),
        .o_saturated(direct_saturated),
        .o_config_error(direct_config_error)
    );

    unified_tp_q015_magnitude_residual_update #(
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .DELTA_WIDTH(DELTA_WIDTH)
    ) magnitude_update_leaf (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_valid(stage_valid_q && stage_magnitude_mode_q),
        .o_ready(magnitude_ready),
        .i_code(stage_code_q),
        .i_residual(stage_residual_q),
        .i_scaled_delta(stage_scaled_residual_delta_q),
        .i_code_min(stage_code_min_q),
        .i_code_max(stage_code_max_q),
        .o_valid(magnitude_valid),
        .i_ready(i_ready),
        .o_code(magnitude_code),
        .o_residual(magnitude_residual),
        .o_code_changed(magnitude_code_changed),
        .o_saturated(magnitude_saturated),
        .o_config_error(magnitude_config_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            stage_valid_q <= 1'b0;
            stage_profile_id_q <= '0;
            stage_tensor_id_q <= '0;
            stage_parameter_addr_q <= '0;
            stage_tag_q <= '0;
            stage_code_q <= '0;
            stage_residual_q <= '0;
            stage_update_direction_q <= '0;
            stage_magnitude_mode_q <= 1'b0;
            stage_scaled_residual_delta_q <= '0;
            stage_vote_increment_q <= '0;
            stage_vote_threshold_q <= '0;
            stage_code_min_q <= '0;
            stage_code_max_q <= '0;
            profile_id_q <= '0;
            tensor_id_q <= '0;
            parameter_addr_q <= '0;
            tag_q <= '0;
            residual_in_q <= '0;
        end else begin
            if (leaf_accept) begin
                profile_id_q <= stage_profile_id_q;
                tensor_id_q <= stage_tensor_id_q;
                parameter_addr_q <= stage_parameter_addr_q;
                tag_q <= stage_tag_q;
                residual_in_q <= stage_residual_q;
            end

            if (leaf_ready) begin
                stage_valid_q <= stage_accept;
            end else if (stage_accept) begin
                stage_valid_q <= 1'b1;
            end

            if (stage_accept) begin
                stage_profile_id_q <= i_profile_id;
                stage_tensor_id_q <= i_tensor_id;
                stage_parameter_addr_q <= i_parameter_addr;
                stage_tag_q <= i_tag;
                stage_code_q <= i_code;
                stage_residual_q <= i_residual;
                stage_update_direction_q <= i_update_direction;
                stage_magnitude_mode_q <= i_magnitude_mode;
                stage_scaled_residual_delta_q <= i_scaled_residual_delta;
                stage_vote_increment_q <= i_vote_increment;
                stage_vote_threshold_q <= i_vote_threshold;
                stage_code_min_q <= i_code_min;
                stage_code_max_q <= i_code_max;
            end
        end
    end

endmodule

`default_nettype wire
