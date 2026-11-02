`timescale 1ns/1ps
`default_nettype none

// Arbitrates the target/student input prescan and the active SCNN graph over
// one untagged feature service. Response ownership is retained from request
// acceptance through response acceptance, so arbitrary service latency and
// consumer backpressure cannot cross-connect the two clients.
module unified_scnn_two_client_feature_service_arbiter_v1 (
    input  wire logic        i_clk,
    input  wire logic        i_rst_n,
    input  wire logic        i_clear_error,

    input  wire logic        i_c0_request_valid,
    output logic             o_c0_request_ready,
    input  wire logic [3:0]  i_c0_term_valid,
    input  wire logic [39:0] i_c0_input_channel_flat,
    input  wire logic [7:0]  i_c0_kernel_row_flat,
    input  wire logic [7:0]  i_c0_kernel_col_flat,
    input  wire logic [31:0] i_c0_spatial_base,
    input  wire logic [5:0]  i_c0_spatial_count,
    input  wire logic [15:0] i_c0_output_row,
    input  wire logic [15:0] i_c0_output_col,
    input  wire logic [3:0]  i_c0_value_bit,
    output logic             o_c0_response_valid,
    input  wire logic        i_c0_response_ready,

    input  wire logic        i_c1_request_valid,
    output logic             o_c1_request_ready,
    input  wire logic [3:0]  i_c1_term_valid,
    input  wire logic [39:0] i_c1_input_channel_flat,
    input  wire logic [7:0]  i_c1_kernel_row_flat,
    input  wire logic [7:0]  i_c1_kernel_col_flat,
    input  wire logic [31:0] i_c1_spatial_base,
    input  wire logic [5:0]  i_c1_spatial_count,
    input  wire logic [15:0] i_c1_output_row,
    input  wire logic [15:0] i_c1_output_col,
    input  wire logic [3:0]  i_c1_value_bit,
    output logic             o_c1_response_valid,
    input  wire logic        i_c1_response_ready,

    output logic             o_service_request_valid,
    input  wire logic        i_service_request_ready,
    output logic [3:0]       o_service_term_valid,
    output logic [39:0]      o_service_input_channel_flat,
    output logic [7:0]       o_service_kernel_row_flat,
    output logic [7:0]       o_service_kernel_col_flat,
    output logic [31:0]      o_service_spatial_base,
    output logic [5:0]       o_service_spatial_count,
    output logic [15:0]      o_service_output_row,
    output logic [15:0]      o_service_output_col,
    output logic [3:0]       o_service_value_bit,
    input  wire logic        i_service_response_valid,
    output logic             o_service_response_ready,
    input  wire logic [39:0] i_service_mask0,
    input  wire logic [39:0] i_service_mask1,
    input  wire logic [39:0] i_service_mask2,
    input  wire logic [39:0] i_service_mask3,
    input  wire logic        i_service_error,

    output logic [39:0]      o_response_mask0,
    output logic [39:0]      o_response_mask1,
    output logic [39:0]      o_response_mask2,
    output logic [39:0]      o_response_mask3,
    output logic             o_response_error,
    output logic             o_busy,
    output logic             o_protocol_error
);
    logic pending_q;
    logic owner_q;
    logic selected_owner;
    logic request_fire;
    logic response_fire;

    always_comb begin
        selected_owner = i_c1_request_valid && !i_c0_request_valid;
        o_service_request_valid = !pending_q &&
            (i_c0_request_valid || i_c1_request_valid);
        o_service_term_valid = selected_owner ?
            i_c1_term_valid : i_c0_term_valid;
        o_service_input_channel_flat = selected_owner ?
            i_c1_input_channel_flat : i_c0_input_channel_flat;
        o_service_kernel_row_flat = selected_owner ?
            i_c1_kernel_row_flat : i_c0_kernel_row_flat;
        o_service_kernel_col_flat = selected_owner ?
            i_c1_kernel_col_flat : i_c0_kernel_col_flat;
        o_service_spatial_base = selected_owner ?
            i_c1_spatial_base : i_c0_spatial_base;
        o_service_spatial_count = selected_owner ?
            i_c1_spatial_count : i_c0_spatial_count;
        o_service_output_row = selected_owner ?
            i_c1_output_row : i_c0_output_row;
        o_service_output_col = selected_owner ?
            i_c1_output_col : i_c0_output_col;
        o_service_value_bit = selected_owner ?
            i_c1_value_bit : i_c0_value_bit;

        o_c0_request_ready = o_service_request_valid && !selected_owner &&
            i_service_request_ready;
        o_c1_request_ready = o_service_request_valid && selected_owner &&
            i_service_request_ready;
        request_fire = o_service_request_valid && i_service_request_ready;

        o_c0_response_valid = pending_q && !owner_q &&
            i_service_response_valid;
        o_c1_response_valid = pending_q && owner_q &&
            i_service_response_valid;
        o_service_response_ready = pending_q &&
            (owner_q ? i_c1_response_ready : i_c0_response_ready);
        response_fire = i_service_response_valid &&
            o_service_response_ready;

        o_response_mask0 = i_service_mask0;
        o_response_mask1 = i_service_mask1;
        o_response_mask2 = i_service_mask2;
        o_response_mask3 = i_service_mask3;
        o_response_error = i_service_error;
        o_busy = pending_q;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            pending_q <= 1'b0;
            owner_q <= 1'b0;
            o_protocol_error <= 1'b0;
        end else begin
            if (i_clear_error && !pending_q &&
                    !i_c0_request_valid && !i_c1_request_valid)
                o_protocol_error <= 1'b0;
            if (i_c0_request_valid && i_c1_request_valid)
                o_protocol_error <= 1'b1;
            if (i_service_response_valid && !pending_q)
                o_protocol_error <= 1'b1;
            if (request_fire) begin
                pending_q <= 1'b1;
                owner_q <= selected_owner;
            end
            if (response_fire)
                pending_q <= 1'b0;
        end
    end
endmodule

`default_nettype wire
