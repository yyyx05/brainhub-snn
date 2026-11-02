`timescale 1ns/1ps
`default_nettype none

// Merge two clients onto one parameter-store port. Read ownership is retained
// only until the matching response; writes are an independent channel. This
// supports both direct read/write clients and clients that implement RMW.
module unified_tp_two_client_parameter_port_arbiter #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16
) (
    input  logic                               i_clk,
    input  logic                               i_rst_n,

    input  logic                               i_c0_read_valid,
    output logic                               o_c0_read_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]        i_c0_read_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]         i_c0_read_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]        i_c0_read_parameter_addr,
    input  logic [TAG_WIDTH-1:0]               i_c0_read_tag,
    output logic                               o_c0_read_response_valid,
    input  logic                               i_c0_read_response_ready,
    output logic signed [CODE_WIDTH-1:0]        o_c0_read_response_code,
    output logic signed [RESIDUAL_WIDTH-1:0]    o_c0_read_response_residual,
    output logic                               o_c0_read_response_error,
    input  logic                               i_c0_result_valid,
    output logic                               o_c0_result_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]        i_c0_result_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]         i_c0_result_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]        i_c0_result_parameter_addr,
    input  logic [TAG_WIDTH-1:0]               i_c0_result_tag,
    input  logic signed [CODE_WIDTH-1:0]        i_c0_result_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]    i_c0_result_residual,
    input  logic                               i_c0_write_code_enable,
    input  logic                               i_c0_write_residual_enable,
    input  logic                               i_c0_write_code_fire,
    input  logic                               i_c0_write_residual_fire,
    input  logic                               i_c0_dirty,
    input  logic                               i_c0_saturated,

    input  logic                               i_c1_read_valid,
    output logic                               o_c1_read_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]        i_c1_read_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]         i_c1_read_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]        i_c1_read_parameter_addr,
    input  logic [TAG_WIDTH-1:0]               i_c1_read_tag,
    output logic                               o_c1_read_response_valid,
    input  logic                               i_c1_read_response_ready,
    output logic signed [CODE_WIDTH-1:0]        o_c1_read_response_code,
    output logic signed [RESIDUAL_WIDTH-1:0]    o_c1_read_response_residual,
    output logic                               o_c1_read_response_error,
    input  logic                               i_c1_result_valid,
    output logic                               o_c1_result_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]        i_c1_result_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]         i_c1_result_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]        i_c1_result_parameter_addr,
    input  logic [TAG_WIDTH-1:0]               i_c1_result_tag,
    input  logic signed [CODE_WIDTH-1:0]        i_c1_result_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]    i_c1_result_residual,
    input  logic                               i_c1_write_code_enable,
    input  logic                               i_c1_write_residual_enable,
    input  logic                               i_c1_write_code_fire,
    input  logic                               i_c1_write_residual_fire,
    input  logic                               i_c1_dirty,
    input  logic                               i_c1_saturated,

    output logic                               o_read_valid,
    input  logic                               i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0]        o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]         o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]        o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]               o_read_tag,
    input  logic                               i_read_response_valid,
    output logic                               o_read_response_ready,
    input  logic signed [CODE_WIDTH-1:0]        i_read_response_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]    i_read_response_residual,
    input  logic                               i_read_response_error,
    output logic                               o_result_valid,
    input  logic                               i_result_ready,
    output logic [PROFILE_ID_WIDTH-1:0]        o_result_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]         o_result_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]        o_result_parameter_addr,
    output logic [TAG_WIDTH-1:0]               o_result_tag,
    output logic signed [CODE_WIDTH-1:0]        o_result_code,
    output logic signed [RESIDUAL_WIDTH-1:0]    o_result_residual,
    output logic                               o_write_code_enable,
    output logic                               o_write_residual_enable,
    output logic                               o_write_code_fire,
    output logic                               o_write_residual_fire,
    output logic                               o_dirty,
    output logic                               o_saturated,
    output logic                               o_busy,
    output logic                               o_protocol_error
);

    logic read_owner_q;
    logic read_pending_q;
    logic last_read_grant_q;
    logic last_write_grant_q;
    logic last_request_was_write_q;
    logic request_hold_q;
    logic request_hold_is_write_q;
    logic request_hold_owner_q;
    logic selected_read_owner;
    logic selected_write_owner;
    logic selected_request_is_write;
    logic selected_request_owner;
    logic read_selected_valid;
    logic write_selected_valid;
    logic read_available;
    logic write_available;
    logic read_fire;
    logic read_response_fire;
    logic write_fire;
    logic selected_request_valid;
    logic selected_request_ready;
    logic selected_client_still_valid;

    always_comb begin
        read_selected_valid = i_c0_read_valid || i_c1_read_valid;
        if (request_hold_q && !request_hold_is_write_q)
            selected_read_owner = request_hold_owner_q;
        else if (i_c0_read_valid && i_c1_read_valid)
            selected_read_owner = ~last_read_grant_q;
        else
            selected_read_owner = i_c1_read_valid;

        write_selected_valid = i_c0_result_valid || i_c1_result_valid;
        if (request_hold_q && request_hold_is_write_q)
            selected_write_owner = request_hold_owner_q;
        else if (i_c0_result_valid && i_c1_result_valid)
            selected_write_owner = ~last_write_grant_q;
        else
            selected_write_owner = i_c1_result_valid;

        read_available = !read_pending_q && read_selected_valid;
        write_available = write_selected_valid;
        if (request_hold_q) begin
            selected_request_is_write = request_hold_is_write_q;
            selected_request_owner = request_hold_owner_q;
        end else begin
            if (read_available && write_available)
                selected_request_is_write = ~last_request_was_write_q;
            else
                selected_request_is_write = write_available;
            selected_request_owner = selected_request_is_write ?
                selected_write_owner : selected_read_owner;
        end

        // The shared store accepts at most one learning-port operation per
        // cycle. A stalled selection remains locked until its handshake.
        o_read_valid = !selected_request_is_write && read_available;
        o_c0_read_ready = o_read_valid && !selected_request_owner &&
            i_read_ready;
        o_c1_read_ready = o_read_valid && selected_request_owner &&
            i_read_ready;
        o_read_profile_id = selected_request_owner ? i_c1_read_profile_id :
            i_c0_read_profile_id;
        o_read_tensor_id = selected_request_owner ? i_c1_read_tensor_id :
            i_c0_read_tensor_id;
        o_read_parameter_addr = selected_request_owner ?
            i_c1_read_parameter_addr : i_c0_read_parameter_addr;
        o_read_tag = selected_request_owner ? i_c1_read_tag : i_c0_read_tag;
        read_fire = o_read_valid && i_read_ready;

        o_c0_read_response_valid = read_pending_q && !read_owner_q &&
            i_read_response_valid;
        o_c1_read_response_valid = read_pending_q && read_owner_q &&
            i_read_response_valid;
        o_c0_read_response_code = i_read_response_code;
        o_c1_read_response_code = i_read_response_code;
        o_c0_read_response_residual = i_read_response_residual;
        o_c1_read_response_residual = i_read_response_residual;
        o_c0_read_response_error = i_read_response_error;
        o_c1_read_response_error = i_read_response_error;
        o_read_response_ready = read_pending_q &&
            (read_owner_q ? i_c1_read_response_ready :
             i_c0_read_response_ready);
        read_response_fire = i_read_response_valid && o_read_response_ready;

        o_result_valid = selected_request_is_write && write_available;
        o_c0_result_ready = o_result_valid && !selected_request_owner &&
            i_result_ready;
        o_c1_result_ready = o_result_valid && selected_request_owner &&
            i_result_ready;
        o_result_profile_id = selected_request_owner ?
            i_c1_result_profile_id : i_c0_result_profile_id;
        o_result_tensor_id = selected_request_owner ?
            i_c1_result_tensor_id : i_c0_result_tensor_id;
        o_result_parameter_addr = selected_request_owner ?
            i_c1_result_parameter_addr : i_c0_result_parameter_addr;
        o_result_tag = selected_request_owner ? i_c1_result_tag :
            i_c0_result_tag;
        o_result_code = selected_request_owner ? i_c1_result_code :
            i_c0_result_code;
        o_result_residual = selected_request_owner ? i_c1_result_residual :
            i_c0_result_residual;
        o_write_code_enable = selected_request_owner ?
            i_c1_write_code_enable : i_c0_write_code_enable;
        o_write_residual_enable = selected_request_owner ?
            i_c1_write_residual_enable : i_c0_write_residual_enable;
        o_write_code_fire = selected_request_owner ? i_c1_write_code_fire :
            i_c0_write_code_fire;
        o_write_residual_fire = selected_request_owner ?
            i_c1_write_residual_fire : i_c0_write_residual_fire;
        o_dirty = selected_request_owner ? i_c1_dirty : i_c0_dirty;
        o_saturated = selected_request_owner ? i_c1_saturated :
            i_c0_saturated;
        write_fire = o_result_valid && i_result_ready;

        selected_request_valid = selected_request_is_write ?
            o_result_valid : o_read_valid;
        selected_request_ready = selected_request_is_write ?
            i_result_ready : i_read_ready;
        selected_client_still_valid = selected_request_is_write ?
            (selected_request_owner ? i_c1_result_valid : i_c0_result_valid) :
            (selected_request_owner ? i_c1_read_valid : i_c0_read_valid);
        o_busy = read_pending_q || request_hold_q || read_available ||
            write_available;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            read_owner_q <= 1'b0;
            read_pending_q <= 1'b0;
            last_read_grant_q <= 1'b1;
            last_write_grant_q <= 1'b1;
            last_request_was_write_q <= 1'b1;
            request_hold_q <= 1'b0;
            request_hold_is_write_q <= 1'b0;
            request_hold_owner_q <= 1'b0;
            o_protocol_error <= 1'b0;
        end else begin
            if ((i_read_response_valid && !read_pending_q) ||
                (request_hold_q && !selected_client_still_valid))
                o_protocol_error <= 1'b1;

            if (request_hold_q && !selected_client_still_valid) begin
                request_hold_q <= 1'b0;
            end else if (!request_hold_q && selected_request_valid &&
                !selected_request_ready) begin
                request_hold_q <= 1'b1;
                request_hold_is_write_q <= selected_request_is_write;
                request_hold_owner_q <= selected_request_owner;
            end else if (request_hold_q && selected_request_valid &&
                         selected_request_ready) begin
                request_hold_q <= 1'b0;
            end

            if (read_fire) begin
                read_owner_q <= selected_request_owner;
                last_read_grant_q <= selected_request_owner;
                last_request_was_write_q <= 1'b0;
                read_pending_q <= 1'b1;
            end
            if (read_response_fire)
                read_pending_q <= 1'b0;
            if (write_fire) begin
                last_write_grant_q <= selected_request_owner;
                last_request_was_write_q <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
