`timescale 1ns/1ps
`default_nettype none

// Holds the descriptor for the one trainable tensor currently owned by the
// learning scheduler. Requests use tensor-local addresses; this block checks
// the range, resolves the global parameter address and inserts one elastic
// register before the shared parameter-update service.
module unified_tp_active_tensor_context #(
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
    output logic signed [1:0]                  o_update_direction,
    output logic                               o_magnitude_mode,
    output logic signed [DELTA_WIDTH-1:0]      o_scaled_residual_delta,
    output logic        [VOTE_WIDTH-1:0]       o_vote_increment,
    output logic        [VOTE_WIDTH-1:0]       o_vote_threshold,
    output logic signed [CODE_WIDTH-1:0]       o_code_min,
    output logic signed [CODE_WIDTH-1:0]       o_code_max
);

    localparam logic [PARAM_ADDR_WIDTH:0] ADDRESS_SPACE_LIMIT =
        {1'b1, {PARAM_ADDR_WIDTH{1'b0}}};

    logic active_q;
    logic pending_q;
    logic [PROFILE_ID_WIDTH-1:0] active_profile_id_q;
    logic [TENSOR_ID_WIDTH-1:0] active_tensor_id_q;
    logic [PARAM_ADDR_WIDTH-1:0] parameter_base_q;
    logic [PARAM_ADDR_WIDTH-1:0] element_count_q;
    logic [VOTE_WIDTH-1:0] vote_increment_q;
    logic [VOTE_WIDTH-1:0] vote_threshold_q;
    logic signed [CODE_WIDTH-1:0] code_min_q;
    logic signed [CODE_WIDTH-1:0] code_max_q;

    logic [PROFILE_ID_WIDTH-1:0] out_profile_id_q;
    logic [TENSOR_ID_WIDTH-1:0] out_tensor_id_q;
    logic [PARAM_ADDR_WIDTH-1:0] out_parameter_addr_q;
    logic [TAG_WIDTH-1:0] out_tag_q;
    logic signed [CODE_WIDTH-1:0] out_code_q;
    logic signed [RESIDUAL_WIDTH-1:0] out_residual_q;
    logic signed [1:0] out_update_direction_q;
    logic out_magnitude_mode_q;
    logic signed [DELTA_WIDTH-1:0] out_scaled_residual_delta_q;
    logic [VOTE_WIDTH-1:0] out_vote_increment_q;
    logic [VOTE_WIDTH-1:0] out_vote_threshold_q;
    logic signed [CODE_WIDTH-1:0] out_code_min_q;
    logic signed [CODE_WIDTH-1:0] out_code_max_q;

    logic [PARAM_ADDR_WIDTH:0] cfg_end_exclusive_ext;
    logic [VOTE_WIDTH:0] cfg_twice_threshold_ext;
    logic cfg_legal;
    logic output_slot_ready;
    logic req_address_legal;

    always_comb begin
        cfg_end_exclusive_ext = {1'b0, i_cfg_parameter_base} +
            {1'b0, i_cfg_element_count};
        cfg_twice_threshold_ext = {1'b0, i_cfg_vote_threshold} +
            {1'b0, i_cfg_vote_threshold};
        cfg_legal = (i_cfg_element_count != '0) &&
            (cfg_end_exclusive_ext <= ADDRESS_SPACE_LIMIT) &&
            (i_cfg_vote_threshold != '0) &&
            ({1'b0, i_cfg_vote_increment} < cfg_twice_threshold_ext) &&
            ($signed(i_cfg_code_min) <= $signed(i_cfg_code_max));

        output_slot_ready = !pending_q || i_ready;
        o_cfg_ready = !active_q && output_slot_ready;
        o_release_ready = active_q && output_slot_ready;
        o_req_ready = output_slot_ready && !i_release && !i_cfg_valid;
        req_address_legal = active_q &&
            (i_req_local_parameter_addr < element_count_q);
    end

    assign o_active = active_q;
    assign o_active_profile_id = active_profile_id_q;
    assign o_active_tensor_id = active_tensor_id_q;
    assign o_valid = pending_q;
    assign o_profile_id = out_profile_id_q;
    assign o_tensor_id = out_tensor_id_q;
    assign o_parameter_addr = out_parameter_addr_q;
    assign o_tag = out_tag_q;
    assign o_code = out_code_q;
    assign o_residual = out_residual_q;
    assign o_update_direction = out_update_direction_q;
    assign o_magnitude_mode = out_magnitude_mode_q;
    assign o_scaled_residual_delta = out_scaled_residual_delta_q;
    assign o_vote_increment = out_vote_increment_q;
    assign o_vote_threshold = out_vote_threshold_q;
    assign o_code_min = out_code_min_q;
    assign o_code_max = out_code_max_q;

    // active_q is the validity bit for this descriptor. The count is therefore
    // irrelevant after reset/release and can remain reset-free, allowing the
    // configuration arithmetic to use the DSP output register directly.
    always_ff @(posedge i_clk) begin
        if (i_cfg_valid && o_cfg_ready && cfg_legal)
            element_count_q <= i_cfg_element_count;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            active_q <= 1'b0;
            pending_q <= 1'b0;
            active_profile_id_q <= '0;
            active_tensor_id_q <= '0;
            parameter_base_q <= '0;
            vote_increment_q <= '0;
            vote_threshold_q <= '0;
            code_min_q <= '0;
            code_max_q <= '0;
            out_profile_id_q <= '0;
            out_tensor_id_q <= '0;
            out_parameter_addr_q <= '0;
            out_tag_q <= '0;
            out_code_q <= '0;
            out_residual_q <= '0;
            out_update_direction_q <= '0;
            out_magnitude_mode_q <= 1'b0;
            out_scaled_residual_delta_q <= '0;
            out_vote_increment_q <= '0;
            out_vote_threshold_q <= '0;
            out_code_min_q <= '0;
            out_code_max_q <= '0;
            o_cfg_error <= 1'b0;
            o_req_error <= 1'b0;
        end else begin
            o_cfg_error <= 1'b0;
            o_req_error <= 1'b0;

            if (pending_q && i_ready)
                pending_q <= 1'b0;

            if (i_cfg_valid && o_cfg_ready) begin
                if (cfg_legal) begin
                    active_q <= 1'b1;
                    active_profile_id_q <= i_cfg_profile_id;
                    active_tensor_id_q <= i_cfg_tensor_id;
                    parameter_base_q <= i_cfg_parameter_base;
                    vote_increment_q <= i_cfg_vote_increment;
                    vote_threshold_q <= i_cfg_vote_threshold;
                    code_min_q <= i_cfg_code_min;
                    code_max_q <= i_cfg_code_max;
                end else begin
                    active_q <= 1'b0;
                    o_cfg_error <= 1'b1;
                end
            end

            if (i_release && o_release_ready)
                active_q <= 1'b0;

            if (i_req_valid && o_req_ready) begin
                if (!req_address_legal) begin
                    o_req_error <= 1'b1;
                end else begin
                    pending_q <= 1'b1;
                    out_profile_id_q <= active_profile_id_q;
                    out_tensor_id_q <= active_tensor_id_q;
                    out_parameter_addr_q <= parameter_base_q +
                        i_req_local_parameter_addr;
                    out_tag_q <= i_req_tag;
                    out_code_q <= i_req_code;
                    out_residual_q <= i_req_residual;
                    out_update_direction_q <= i_req_update_direction;
                    out_magnitude_mode_q <= i_req_magnitude_mode;
                    out_scaled_residual_delta_q <=
                        i_req_scaled_residual_delta;
                    out_vote_increment_q <= vote_increment_q;
                    out_vote_threshold_q <= vote_threshold_q;
                    out_code_min_q <= code_min_q;
                    out_code_max_q <= code_max_q;
                end
            end
        end
    end

endmodule

`default_nettype wire
