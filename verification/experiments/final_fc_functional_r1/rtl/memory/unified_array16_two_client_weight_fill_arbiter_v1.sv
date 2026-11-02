`timescale 1ns/1ps
`default_nettype none

// Serializes complete FC/SRNN and SCNN cache-maintenance transactions onto the
// one physical 16xH40 weight-fill fabric. A grant is retained across gaps in a
// publisher's row stream and changes only after that publisher reports idle.
module unified_array16_two_client_weight_fill_arbiter_v1 #(
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
    localparam logic [1:0] GRANT_NONE = 2'b00;
    localparam logic [1:0] GRANT_CLIENT0 = 2'b01;
    localparam logic [1:0] GRANT_CLIENT1 = 2'b10;

    logic [1:0] grant_q;
    logic last_client_q;
    logic client0_protocol_error;
    logic client1_protocol_error;
    logic grant_encoding_error;

    assign o_grant = grant_q;
    assign o_busy = grant_q != GRANT_NONE || i_client0_busy || i_client1_busy;
    assign o_quiescent = grant_q == GRANT_NONE &&
        !i_client0_busy && !i_client1_busy &&
        i_client0_fill_valid == 4'b0000 &&
        i_client1_fill_valid == 4'b0000;
    assign client0_protocol_error =
        i_client0_fill_valid != 4'b0000 && !i_client0_busy;
    assign client1_protocol_error =
        i_client1_fill_valid != 4'b0000 && !i_client1_busy;
    assign grant_encoding_error = grant_q == 2'b11;

    always_comb begin
        o_client0_fill_ready = '0;
        o_client1_fill_ready = '0;
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
            default: begin
            end
        endcase
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            grant_q <= GRANT_NONE;
            // Client 0 receives the first simultaneous request after reset.
            last_client_q <= 1'b1;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            grant_q <= GRANT_NONE;
        end else begin
            if (i_clear_error && o_quiescent)
                o_sticky_error <= 1'b0;
            if (client0_protocol_error || client1_protocol_error ||
                    grant_encoding_error)
                o_sticky_error <= 1'b1;

            unique case (grant_q)
                GRANT_NONE: begin
                    if (i_client0_busy && i_client1_busy) begin
                        if (last_client_q)
                            grant_q <= GRANT_CLIENT0;
                        else
                            grant_q <= GRANT_CLIENT1;
                    end else if (i_client0_busy) begin
                        grant_q <= GRANT_CLIENT0;
                    end else if (i_client1_busy) begin
                        grant_q <= GRANT_CLIENT1;
                    end
                end
                GRANT_CLIENT0: begin
                    if (!i_client0_busy) begin
                        last_client_q <= 1'b0;
                        if (i_client1_busy)
                            grant_q <= GRANT_CLIENT1;
                        else
                            grant_q <= GRANT_NONE;
                    end
                end
                GRANT_CLIENT1: begin
                    if (!i_client1_busy) begin
                        last_client_q <= 1'b1;
                        if (i_client0_busy)
                            grant_q <= GRANT_CLIENT0;
                        else
                            grant_q <= GRANT_NONE;
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
            $error("array16 weight-fill arbiter requires the 11-bit P4 cache address");
    end
endmodule

`default_nettype wire
