`timescale 1ns/1ps
`default_nettype none

// Three-topology parameter router with an external storage boundary.  The
// runtime remains the sole ownership authority, while the board integration
// may bind this interface to DDR, a cache or a simulation memory.  This avoids
// inferring one monolithic on-chip array for full SRNN/SCNN parameter spaces.
module unified_tp_external_parameter_backend #(
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16
) (
    input  logic                                i_clk,
    input  logic                                i_rst_n,
    input  logic                                i_clear_error,
    input  logic [2:0]                          i_parameter_owner,
    input  logic [2:0]                          i_topology_learning_enable,

    input  logic [2:0]                          i_load_valid,
    output logic [2:0]                          o_load_ready,
    input  logic [3*PARAM_ADDR_WIDTH-1:0]       i_load_addr,
    input  logic signed [3*CODE_WIDTH-1:0]      i_load_code,
    input  logic signed [3*RESIDUAL_WIDTH-1:0]  i_load_residual,

    input  logic [2:0]                          i_learning_read_valid,
    output logic [2:0]                          o_learning_read_ready,
    input  logic [3*PARAM_ADDR_WIDTH-1:0]       i_learning_read_addr,
    input  logic [3*TAG_WIDTH-1:0]              i_learning_read_tag,
    output logic [2:0]                          o_learning_response_valid,
    input  logic [2:0]                          i_learning_response_ready,
    output logic [3*TAG_WIDTH-1:0]              o_learning_response_tag,
    output logic signed [3*CODE_WIDTH-1:0]
                                                    o_learning_response_code,
    output logic signed [3*RESIDUAL_WIDTH-1:0]
                                                    o_learning_response_residual,
    output logic [2:0]                          o_learning_response_error,

    input  logic [2:0]                          i_learning_write_valid,
    output logic [2:0]                          o_learning_write_ready,
    input  logic [3*PARAM_ADDR_WIDTH-1:0]       i_learning_write_addr,
    input  logic signed [3*CODE_WIDTH-1:0]      i_learning_write_code,
    input  logic signed [3*RESIDUAL_WIDTH-1:0]
                                                    i_learning_write_residual,
    input  logic [2:0]                          i_learning_write_code_enable,
    input  logic [2:0]
                                                    i_learning_write_residual_enable,

    input  logic [2:0]                          i_forward_read_valid,
    output logic [2:0]                          o_forward_read_ready,
    input  logic [3*PARAM_ADDR_WIDTH-1:0]       i_forward_read_addr,
    input  logic [3*TAG_WIDTH-1:0]              i_forward_read_tag,
    output logic [2:0]                          o_forward_response_valid,
    input  logic [2:0]                          i_forward_response_ready,
    output logic [3*TAG_WIDTH-1:0]              o_forward_response_tag,
    output logic signed [3*CODE_WIDTH-1:0]      o_forward_response_code,
    output logic [2:0]                          o_forward_response_error,

    output logic                                o_store_load_valid,
    input  logic                                i_store_load_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_store_load_addr,
    output logic signed [CODE_WIDTH-1:0]        o_store_load_code,
    output logic signed [RESIDUAL_WIDTH-1:0]    o_store_load_residual,
    output logic                                o_store_learning_read_valid,
    input  logic                                i_store_learning_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_store_learning_read_addr,
    output logic [TAG_WIDTH-1:0]                o_store_learning_read_tag,
    input  logic                                i_store_learning_response_valid,
    output logic                                o_store_learning_response_ready,
    input  logic [TAG_WIDTH-1:0]                i_store_learning_response_tag,
    input  logic signed [CODE_WIDTH-1:0]        i_store_learning_response_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]
                                                    i_store_learning_response_residual,
    input  logic                                i_store_learning_response_error,
    output logic                                o_store_learning_write_valid,
    input  logic                                i_store_learning_write_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_store_learning_write_addr,
    output logic signed [CODE_WIDTH-1:0]        o_store_learning_write_code,
    output logic signed [RESIDUAL_WIDTH-1:0]
                                                    o_store_learning_write_residual,
    output logic                                o_store_learning_write_code_enable,
    output logic
                                                    o_store_learning_write_residual_enable,
    output logic                                o_store_forward_read_valid,
    input  logic                                i_store_forward_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_store_forward_read_addr,
    output logic [TAG_WIDTH-1:0]                o_store_forward_read_tag,
    input  logic                                i_store_forward_response_valid,
    output logic                                o_store_forward_response_ready,
    input  logic [TAG_WIDTH-1:0]                i_store_forward_response_tag,
    input  logic signed [CODE_WIDTH-1:0]        i_store_forward_response_code,
    input  logic                                i_store_forward_response_error,
    input  logic                                i_store_quiescent,
    input  logic                                i_store_protocol_error,

    output logic                                o_backend_quiescent,
    output logic                                o_backend_error,
    output logic                                o_owner_encoding_error,
    output logic                                o_router_protocol_error
);
    logic owner_active;
    logic [1:0] active_owner;
    logic [2:0] gated_learning_read_valid;
    logic [2:0] gated_learning_write_valid;
    logic [2:0] router_learning_read_ready;
    logic [2:0] router_learning_write_ready;
    logic router_quiescent;

    always_comb begin
        owner_active = 1'b1;
        active_owner = 2'd0;
        o_owner_encoding_error = 1'b0;
        unique case (i_parameter_owner)
            3'b000: owner_active = 1'b0;
            3'b001: active_owner = 2'd1;
            3'b010: active_owner = 2'd2;
            3'b100: active_owner = 2'd3;
            default: begin
                owner_active = 1'b0;
                o_owner_encoding_error = 1'b1;
            end
        endcase
    end

    assign gated_learning_read_valid = i_learning_read_valid &
        i_topology_learning_enable;
    assign gated_learning_write_valid = i_learning_write_valid &
        i_topology_learning_enable;
    assign o_learning_read_ready = router_learning_read_ready &
        i_topology_learning_enable;
    assign o_learning_write_ready = router_learning_write_ready &
        i_topology_learning_enable;
    assign o_backend_quiescent = router_quiescent && i_store_quiescent;
    assign o_backend_error = o_owner_encoding_error ||
        o_router_protocol_error || i_store_protocol_error;

    unified_tp_three_client_parameter_backend_router #(
        .CLIENTS(3), .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH), .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH)
    ) u_router (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_owner_active(owner_active), .i_active_owner(active_owner),
        .i_load_valid(i_load_valid), .o_load_ready(o_load_ready),
        .i_load_addr(i_load_addr), .i_load_code(i_load_code),
        .i_load_residual(i_load_residual),
        .i_read_valid(gated_learning_read_valid),
        .o_read_ready(router_learning_read_ready),
        .i_read_addr(i_learning_read_addr),
        .i_read_tag(i_learning_read_tag),
        .o_response_valid(o_learning_response_valid),
        .i_response_ready(i_learning_response_ready),
        .o_response_tag(o_learning_response_tag),
        .o_response_code(o_learning_response_code),
        .o_response_residual(o_learning_response_residual),
        .o_response_error(o_learning_response_error),
        .i_write_valid(gated_learning_write_valid),
        .o_write_ready(router_learning_write_ready),
        .i_write_addr(i_learning_write_addr),
        .i_write_code(i_learning_write_code),
        .i_write_residual(i_learning_write_residual),
        .i_write_code_enable(i_learning_write_code_enable),
        .i_write_residual_enable(i_learning_write_residual_enable),
        .i_forward_read_valid(i_forward_read_valid),
        .o_forward_read_ready(o_forward_read_ready),
        .i_forward_read_addr(i_forward_read_addr),
        .i_forward_read_tag(i_forward_read_tag),
        .o_forward_response_valid(o_forward_response_valid),
        .i_forward_response_ready(i_forward_response_ready),
        .o_forward_response_tag(o_forward_response_tag),
        .o_forward_response_code(o_forward_response_code),
        .o_forward_response_error(o_forward_response_error),
        .o_store_load_valid(o_store_load_valid),
        .i_store_load_ready(i_store_load_ready),
        .o_store_load_addr(o_store_load_addr),
        .o_store_load_code(o_store_load_code),
        .o_store_load_residual(o_store_load_residual),
        .o_store_learning_read_valid(o_store_learning_read_valid),
        .i_store_learning_read_ready(i_store_learning_read_ready),
        .o_store_learning_read_addr(o_store_learning_read_addr),
        .o_store_learning_read_tag(o_store_learning_read_tag),
        .i_store_learning_response_valid(i_store_learning_response_valid),
        .o_store_learning_response_ready(o_store_learning_response_ready),
        .i_store_learning_response_tag(i_store_learning_response_tag),
        .i_store_learning_response_code(i_store_learning_response_code),
        .i_store_learning_response_residual(
            i_store_learning_response_residual),
        .i_store_learning_response_error(i_store_learning_response_error),
        .o_store_learning_write_valid(o_store_learning_write_valid),
        .i_store_learning_write_ready(i_store_learning_write_ready),
        .o_store_learning_write_addr(o_store_learning_write_addr),
        .o_store_learning_write_code(o_store_learning_write_code),
        .o_store_learning_write_residual(o_store_learning_write_residual),
        .o_store_learning_write_code_enable(
            o_store_learning_write_code_enable),
        .o_store_learning_write_residual_enable(
            o_store_learning_write_residual_enable),
        .o_store_forward_read_valid(o_store_forward_read_valid),
        .i_store_forward_read_ready(i_store_forward_read_ready),
        .o_store_forward_read_addr(o_store_forward_read_addr),
        .o_store_forward_read_tag(o_store_forward_read_tag),
        .i_store_forward_response_valid(i_store_forward_response_valid),
        .o_store_forward_response_ready(o_store_forward_response_ready),
        .i_store_forward_response_tag(i_store_forward_response_tag),
        .i_store_forward_response_code(i_store_forward_response_code),
        .i_store_forward_response_error(i_store_forward_response_error),
        .i_store_protocol_error(i_store_protocol_error),
        .o_quiescent(router_quiescent),
        .o_protocol_error(o_router_protocol_error)
    );

    initial begin
        if (PARAM_ADDR_WIDTH < 1 || TAG_WIDTH < 1 || CODE_WIDTH < 2 ||
            RESIDUAL_WIDTH < 2)
            $error("external parameter backend dimensions are invalid");
    end
endmodule

`default_nettype wire
