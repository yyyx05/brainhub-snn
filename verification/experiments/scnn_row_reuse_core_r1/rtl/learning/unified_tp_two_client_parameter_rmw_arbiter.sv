`timescale 1ns/1ps
`default_nettype none

// Two-client arbiter for the shared parameter RMW port. Ownership is acquired
// when a read request is accepted and is retained through the matching result
// handshake, so read responses cannot be delivered to the wrong tensor path.
// An accepted error response terminates the transaction: the RMW client
// returns idle without generating a result on that path.
module unified_tp_two_client_parameter_rmw_arbiter #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16
) (
    input  wire logic                               i_clk,
    input  wire logic                               i_rst_n,

    input  wire logic                               i_c0_read_valid,
    output logic                               o_c0_read_ready,
    input  wire logic [PROFILE_ID_WIDTH-1:0]        i_c0_read_profile_id,
    input  wire logic [TENSOR_ID_WIDTH-1:0]         i_c0_read_tensor_id,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_c0_read_parameter_addr,
    input  wire logic [TAG_WIDTH-1:0]               i_c0_read_tag,
    output logic                               o_c0_read_response_valid,
    input  wire logic                               i_c0_read_response_ready,
    output logic signed [CODE_WIDTH-1:0]       o_c0_read_response_code,
    output logic signed [RESIDUAL_WIDTH-1:0]   o_c0_read_response_residual,
    output logic                               o_c0_read_response_error,
    input  wire logic                               i_c0_result_valid,
    output logic                               o_c0_result_ready,
    input  wire logic [PROFILE_ID_WIDTH-1:0]        i_c0_result_profile_id,
    input  wire logic [TENSOR_ID_WIDTH-1:0]         i_c0_result_tensor_id,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_c0_result_parameter_addr,
    input  wire logic [TAG_WIDTH-1:0]               i_c0_result_tag,
    input  wire logic signed [CODE_WIDTH-1:0]       i_c0_result_code,
    input  wire logic signed [RESIDUAL_WIDTH-1:0]   i_c0_result_residual,
    input  wire logic                               i_c0_write_code_enable,
    input  wire logic                               i_c0_write_residual_enable,
    input  wire logic                               i_c0_write_code_fire,
    input  wire logic                               i_c0_write_residual_fire,
    input  wire logic                               i_c0_dirty,
    input  wire logic                               i_c0_saturated,

    input  wire logic                               i_c1_read_valid,
    output logic                               o_c1_read_ready,
    input  wire logic [PROFILE_ID_WIDTH-1:0]        i_c1_read_profile_id,
    input  wire logic [TENSOR_ID_WIDTH-1:0]         i_c1_read_tensor_id,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_c1_read_parameter_addr,
    input  wire logic [TAG_WIDTH-1:0]               i_c1_read_tag,
    output logic                               o_c1_read_response_valid,
    input  wire logic                               i_c1_read_response_ready,
    output logic signed [CODE_WIDTH-1:0]       o_c1_read_response_code,
    output logic signed [RESIDUAL_WIDTH-1:0]   o_c1_read_response_residual,
    output logic                               o_c1_read_response_error,
    input  wire logic                               i_c1_result_valid,
    output logic                               o_c1_result_ready,
    input  wire logic [PROFILE_ID_WIDTH-1:0]        i_c1_result_profile_id,
    input  wire logic [TENSOR_ID_WIDTH-1:0]         i_c1_result_tensor_id,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_c1_result_parameter_addr,
    input  wire logic [TAG_WIDTH-1:0]               i_c1_result_tag,
    input  wire logic signed [CODE_WIDTH-1:0]       i_c1_result_code,
    input  wire logic signed [RESIDUAL_WIDTH-1:0]   i_c1_result_residual,
    input  wire logic                               i_c1_write_code_enable,
    input  wire logic                               i_c1_write_residual_enable,
    input  wire logic                               i_c1_write_code_fire,
    input  wire logic                               i_c1_write_residual_fire,
    input  wire logic                               i_c1_dirty,
    input  wire logic                               i_c1_saturated,

    output logic                               o_read_valid,
    input  wire logic                               i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0]        o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]         o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]        o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]               o_read_tag,
    input  wire logic                               i_read_response_valid,
    output logic                               o_read_response_ready,
    input  wire logic signed [CODE_WIDTH-1:0]       i_read_response_code,
    input  wire logic signed [RESIDUAL_WIDTH-1:0]   i_read_response_residual,
    input  wire logic                               i_read_response_error,
    output logic                               o_result_valid,
    input  wire logic                               i_result_ready,
    output logic [PROFILE_ID_WIDTH-1:0]        o_result_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]         o_result_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]        o_result_parameter_addr,
    output logic [TAG_WIDTH-1:0]               o_result_tag,
    output logic signed [CODE_WIDTH-1:0]       o_result_code,
    output logic signed [RESIDUAL_WIDTH-1:0]   o_result_residual,
    output logic                               o_write_code_enable,
    output logic                               o_write_residual_enable,
    output logic                               o_write_code_fire,
    output logic                               o_write_residual_fire,
    output logic                               o_dirty,
    output logic                               o_saturated,
    output logic                               o_busy,
    output logic                               o_protocol_error
);

    logic owner_q;
    logic last_grant_q;
    logic transaction_active_q;
    logic selected_owner;
    logic selected_valid;
    logic read_fire;
    logic result_fire;

    always_comb begin
        selected_valid = i_c0_read_valid || i_c1_read_valid;
        if (i_c0_read_valid && i_c1_read_valid)
            selected_owner = ~last_grant_q;
        else
            selected_owner = i_c1_read_valid;

        o_c0_read_ready = !transaction_active_q && selected_valid &&
            !selected_owner && i_read_ready;
        o_c1_read_ready = !transaction_active_q && selected_valid &&
            selected_owner && i_read_ready;
        o_read_valid = !transaction_active_q && selected_valid;
        o_read_profile_id = selected_owner ? i_c1_read_profile_id :
            i_c0_read_profile_id;
        o_read_tensor_id = selected_owner ? i_c1_read_tensor_id :
            i_c0_read_tensor_id;
        o_read_parameter_addr = selected_owner ? i_c1_read_parameter_addr :
            i_c0_read_parameter_addr;
        o_read_tag = selected_owner ? i_c1_read_tag : i_c0_read_tag;
        read_fire = o_read_valid && i_read_ready;

        o_c0_read_response_valid = transaction_active_q && !owner_q &&
            i_read_response_valid;
        o_c1_read_response_valid = transaction_active_q && owner_q &&
            i_read_response_valid;
        o_c0_read_response_code = i_read_response_code;
        o_c1_read_response_code = i_read_response_code;
        o_c0_read_response_residual = i_read_response_residual;
        o_c1_read_response_residual = i_read_response_residual;
        o_c0_read_response_error = i_read_response_error;
        o_c1_read_response_error = i_read_response_error;
        o_read_response_ready = transaction_active_q &&
            (owner_q ? i_c1_read_response_ready : i_c0_read_response_ready);

        o_c0_result_ready = transaction_active_q && !owner_q && i_result_ready;
        o_c1_result_ready = transaction_active_q && owner_q && i_result_ready;
        o_result_valid = transaction_active_q &&
            (owner_q ? i_c1_result_valid : i_c0_result_valid);
        o_result_profile_id = owner_q ? i_c1_result_profile_id :
            i_c0_result_profile_id;
        o_result_tensor_id = owner_q ? i_c1_result_tensor_id :
            i_c0_result_tensor_id;
        o_result_parameter_addr = owner_q ? i_c1_result_parameter_addr :
            i_c0_result_parameter_addr;
        o_result_tag = owner_q ? i_c1_result_tag : i_c0_result_tag;
        o_result_code = owner_q ? i_c1_result_code : i_c0_result_code;
        o_result_residual = owner_q ? i_c1_result_residual :
            i_c0_result_residual;
        o_write_code_enable = owner_q ? i_c1_write_code_enable :
            i_c0_write_code_enable;
        o_write_residual_enable = owner_q ? i_c1_write_residual_enable :
            i_c0_write_residual_enable;
        o_write_code_fire = owner_q ? i_c1_write_code_fire :
            i_c0_write_code_fire;
        o_write_residual_fire = owner_q ? i_c1_write_residual_fire :
            i_c0_write_residual_fire;
        o_dirty = owner_q ? i_c1_dirty : i_c0_dirty;
        o_saturated = owner_q ? i_c1_saturated : i_c0_saturated;
        result_fire = o_result_valid && i_result_ready;
        o_busy = transaction_active_q;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            owner_q <= 1'b0;
            last_grant_q <= 1'b1;
            transaction_active_q <= 1'b0;
            o_protocol_error <= 1'b0;
        end else begin
            o_protocol_error <= (!transaction_active_q &&
                                 (i_read_response_valid || i_c0_result_valid ||
                                  i_c1_result_valid));
            if (read_fire) begin
                owner_q <= selected_owner;
                last_grant_q <= selected_owner;
                transaction_active_q <= 1'b1;
            end
            if (result_fire || (i_read_response_valid &&
                    o_read_response_ready && i_read_response_error))
                transaction_active_q <= 1'b0;
        end
    end

endmodule

`default_nettype wire
