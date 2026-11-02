`timescale 1ns/1ps
`default_nettype none

// Retained-grant transaction arbiter for the single physical array weight-fill
// fabric. SCNN, M2 and the frozen SRNN backbone may leave bubbles between rows;
// the grant therefore changes only after the selected publisher drops busy.
module unified_array16_three_client_weight_fill_arbiter_v1 #(
    parameter int ARRAY_WEIGHT_ADDR_WIDTH = 11
) (
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,
    input  wire logic                              i_abort,
    input  wire logic                              i_clear_error,

    input  wire logic                              i_client0_busy,
    input  wire logic [3:0]                        i_client0_fill_valid,
    output logic [3:0]                             o_client0_fill_ready,
    input  wire logic [7:0]                        i_client0_fill_tile_flat,
    input  wire logic [3:0]                        i_client0_fill_bank,
    input  wire logic [4*ARRAY_WEIGHT_ADDR_WIDTH-1:0]
                                                    i_client0_fill_addr_flat,
    input  wire logic [799:0]                      i_client0_fill_data_flat,

    input  wire logic                              i_client1_busy,
    input  wire logic [3:0]                        i_client1_fill_valid,
    output logic [3:0]                             o_client1_fill_ready,
    input  wire logic [7:0]                        i_client1_fill_tile_flat,
    input  wire logic [3:0]                        i_client1_fill_bank,
    input  wire logic [4*ARRAY_WEIGHT_ADDR_WIDTH-1:0]
                                                    i_client1_fill_addr_flat,
    input  wire logic [799:0]                      i_client1_fill_data_flat,

    input  wire logic                              i_client2_busy,
    input  wire logic [3:0]                        i_client2_fill_valid,
    output logic [3:0]                             o_client2_fill_ready,
    input  wire logic [7:0]                        i_client2_fill_tile_flat,
    input  wire logic [3:0]                        i_client2_fill_bank,
    input  wire logic [4*ARRAY_WEIGHT_ADDR_WIDTH-1:0]
                                                    i_client2_fill_addr_flat,
    input  wire logic [799:0]                      i_client2_fill_data_flat,

    output logic [3:0]                             o_fill_valid,
    input  wire logic [3:0]                        i_fill_ready,
    output logic [7:0]                             o_fill_tile_flat,
    output logic [3:0]                             o_fill_bank,
    output logic [4*ARRAY_WEIGHT_ADDR_WIDTH-1:0]    o_fill_addr_flat,
    output logic [799:0]                           o_fill_data_flat,

    output logic [1:0]                             o_grant,
    output logic                                   o_busy,
    output logic                                   o_quiescent,
    output logic                                   o_sticky_error
);
    localparam logic [1:0] GRANT_NONE = 2'd0;
    localparam logic [1:0] GRANT_CLIENT0 = 2'd1;
    localparam logic [1:0] GRANT_CLIENT1 = 2'd2;
    localparam logic [1:0] GRANT_CLIENT2 = 2'd3;

    logic [1:0] grant_q;
    logic [1:0] last_client_q;
    logic any_protocol_error;

    function automatic logic [1:0] select_next(
        input logic [1:0] last_client,
        input logic client0_busy,
        input logic client1_busy,
        input logic client2_busy
    );
        begin
            select_next = GRANT_NONE;
            unique case (last_client)
                2'd0: begin
                    if (client1_busy) select_next = GRANT_CLIENT1;
                    else if (client2_busy) select_next = GRANT_CLIENT2;
                    else if (client0_busy) select_next = GRANT_CLIENT0;
                end
                2'd1: begin
                    if (client2_busy) select_next = GRANT_CLIENT2;
                    else if (client0_busy) select_next = GRANT_CLIENT0;
                    else if (client1_busy) select_next = GRANT_CLIENT1;
                end
                default: begin
                    if (client0_busy) select_next = GRANT_CLIENT0;
                    else if (client1_busy) select_next = GRANT_CLIENT1;
                    else if (client2_busy) select_next = GRANT_CLIENT2;
                end
            endcase
        end
    endfunction

    assign o_grant = grant_q;
    assign o_busy = grant_q != GRANT_NONE || i_client0_busy ||
        i_client1_busy || i_client2_busy;
    assign o_quiescent = grant_q == GRANT_NONE &&
        !i_client0_busy && !i_client1_busy && !i_client2_busy &&
        i_client0_fill_valid == 4'b0000 &&
        i_client1_fill_valid == 4'b0000 &&
        i_client2_fill_valid == 4'b0000;
    assign any_protocol_error =
        (i_client0_fill_valid != 4'b0000 && !i_client0_busy) ||
        (i_client1_fill_valid != 4'b0000 && !i_client1_busy) ||
        (i_client2_fill_valid != 4'b0000 && !i_client2_busy);

    always_comb begin
        o_client0_fill_ready = '0;
        o_client1_fill_ready = '0;
        o_client2_fill_ready = '0;
        o_fill_valid = '0;
        o_fill_tile_flat = '0;
        o_fill_bank = '0;
        o_fill_addr_flat = '0;
        o_fill_data_flat = '0;

        unique case (grant_q)
            GRANT_CLIENT0: begin
                o_client0_fill_ready = i_fill_ready;
                o_fill_valid = i_client0_fill_valid;
                o_fill_tile_flat = i_client0_fill_tile_flat;
                o_fill_bank = i_client0_fill_bank;
                o_fill_addr_flat = i_client0_fill_addr_flat;
                o_fill_data_flat = i_client0_fill_data_flat;
            end
            GRANT_CLIENT1: begin
                o_client1_fill_ready = i_fill_ready;
                o_fill_valid = i_client1_fill_valid;
                o_fill_tile_flat = i_client1_fill_tile_flat;
                o_fill_bank = i_client1_fill_bank;
                o_fill_addr_flat = i_client1_fill_addr_flat;
                o_fill_data_flat = i_client1_fill_data_flat;
            end
            GRANT_CLIENT2: begin
                o_client2_fill_ready = i_fill_ready;
                o_fill_valid = i_client2_fill_valid;
                o_fill_tile_flat = i_client2_fill_tile_flat;
                o_fill_bank = i_client2_fill_bank;
                o_fill_addr_flat = i_client2_fill_addr_flat;
                o_fill_data_flat = i_client2_fill_data_flat;
            end
            default: begin
            end
        endcase
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            grant_q <= GRANT_NONE;
            last_client_q <= 2'd2;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            grant_q <= GRANT_NONE;
        end else begin
            if (i_clear_error && o_quiescent)
                o_sticky_error <= 1'b0;
            if (any_protocol_error)
                o_sticky_error <= 1'b1;

            unique case (grant_q)
                GRANT_NONE: grant_q <= select_next(
                    last_client_q, i_client0_busy,
                    i_client1_busy, i_client2_busy);
                GRANT_CLIENT0: begin
                    if (!i_client0_busy) begin
                        last_client_q <= 2'd0;
                        grant_q <= select_next(2'd0, 1'b0,
                            i_client1_busy, i_client2_busy);
                    end
                end
                GRANT_CLIENT1: begin
                    if (!i_client1_busy) begin
                        last_client_q <= 2'd1;
                        grant_q <= select_next(2'd1, i_client0_busy,
                            1'b0, i_client2_busy);
                    end
                end
                GRANT_CLIENT2: begin
                    if (!i_client2_busy) begin
                        last_client_q <= 2'd2;
                        grant_q <= select_next(2'd2, i_client0_busy,
                            i_client1_busy, 1'b0);
                    end
                end
                default: begin
                    grant_q <= GRANT_NONE;
                    o_sticky_error <= 1'b1;
                end
            endcase
        end
    end

    initial begin
        if (ARRAY_WEIGHT_ADDR_WIDTH != 11)
            $error("array16 three-client fill arbiter requires 11-bit address");
    end
endmodule

`default_nettype wire
