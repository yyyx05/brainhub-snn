`timescale 1ns/1ps
`default_nettype none

// Routes one ordered 200-bit cache-row stream to one bank in the 16-tile
// array. A completion token is published only after the full command range is
// accepted, so a higher-level residency table can swap cache versions safely.
module unified_array16_weight_refill_dispatcher_v1 #(
    parameter int BANK_DEPTH = 578,
    parameter int BANK_ADDR_WIDTH = 10,
    parameter int ARRAY_ADDR_WIDTH = 11,
    parameter int TAG_WIDTH = 16
) (
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,
    input  wire logic                              i_abort,
    input  wire logic                              i_clear_error,

    input  wire logic                              i_command_valid,
    output logic                                   o_command_ready,
    input  wire logic [3:0]                        i_command_tile,
    input  wire logic [1:0]                        i_command_bank,
    input  wire logic [BANK_ADDR_WIDTH-1:0]        i_command_base_addr,
    input  wire logic [BANK_ADDR_WIDTH:0]          i_command_rows,
    input  wire logic [TAG_WIDTH-1:0]              i_command_tag,

    input  wire logic                              i_row_valid,
    output logic                                   o_row_ready,
    input  wire logic [199:0]                      i_row_data,

    output logic [3:0]                             o_fill_valid,
    input  wire logic [3:0]                        i_fill_ready,
    output logic [7:0]                             o_fill_tile_flat,
    output logic [3:0]                             o_fill_bank,
    output logic [4*ARRAY_ADDR_WIDTH-1:0]          o_fill_addr_flat,
    output logic [799:0]                           o_fill_data_flat,

    output logic                                   o_complete_valid,
    input  wire logic                              i_complete_ready,
    output logic [3:0]                             o_complete_tile,
    output logic [1:0]                             o_complete_bank,
    output logic [TAG_WIDTH-1:0]                   o_complete_tag,
    output logic                                   o_busy,
    output logic                                   o_sticky_error
);
    logic active_q;
    logic [3:0] tile_q;
    logic [1:0] bank_q;
    logic [BANK_ADDR_WIDTH-1:0] addr_q;
    logic [BANK_ADDR_WIDTH:0] rows_remaining_q;
    logic [TAG_WIDTH-1:0] tag_q;
    logic complete_valid_q;
    logic command_fire;
    logic command_legal;
    logic row_fire;
    logic [BANK_ADDR_WIDTH:0] command_end;
    wire logic [1:0] selected_region = tile_q[3:2];

    assign command_end = {1'b0, i_command_base_addr} + i_command_rows;
    assign command_legal = i_command_rows != 0 &&
        command_end <= (BANK_ADDR_WIDTH+1)'(BANK_DEPTH);
    assign o_command_ready = !i_abort && !active_q &&
        !complete_valid_q && !o_sticky_error;
    assign command_fire = i_command_valid && o_command_ready;

    always_comb begin
        o_fill_valid = '0;
        o_fill_tile_flat = '0;
        o_fill_bank = '0;
        o_fill_addr_flat = '0;
        o_fill_data_flat = '0;
        o_row_ready = 1'b0;
        if (active_q && !i_abort) begin
            o_fill_valid[selected_region] = i_row_valid;
            o_fill_tile_flat[selected_region*2 +: 2] = tile_q[1:0];
            o_fill_bank[selected_region] = bank_q[0];
            o_fill_addr_flat[
                selected_region*ARRAY_ADDR_WIDTH +: ARRAY_ADDR_WIDTH] =
                {bank_q[1], addr_q};
            o_fill_data_flat[selected_region*200 +: 200] = i_row_data;
            o_row_ready = i_fill_ready[selected_region];
        end
    end

    assign row_fire = i_row_valid && o_row_ready;
    assign o_complete_valid = complete_valid_q;
    assign o_complete_tile = tile_q;
    assign o_complete_bank = bank_q;
    assign o_complete_tag = tag_q;
    assign o_busy = active_q || complete_valid_q;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            active_q <= 1'b0;
            tile_q <= '0;
            bank_q <= '0;
            addr_q <= '0;
            rows_remaining_q <= '0;
            tag_q <= '0;
            complete_valid_q <= 1'b0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            active_q <= 1'b0;
            rows_remaining_q <= '0;
            complete_valid_q <= 1'b0;
        end else begin
            if (i_clear_error && !o_busy)
                o_sticky_error <= 1'b0;
            if (complete_valid_q && i_complete_ready)
                complete_valid_q <= 1'b0;

            if (command_fire) begin
                if (!command_legal) begin
                    o_sticky_error <= 1'b1;
                end else begin
                    active_q <= 1'b1;
                    tile_q <= i_command_tile;
                    bank_q <= i_command_bank;
                    addr_q <= i_command_base_addr;
                    rows_remaining_q <= i_command_rows;
                    tag_q <= i_command_tag;
                end
            end

            if (row_fire) begin
                if (rows_remaining_q == 1) begin
                    active_q <= 1'b0;
                    rows_remaining_q <= '0;
                    complete_valid_q <= 1'b1;
                end else begin
                    addr_q <= addr_q + 1'b1;
                    rows_remaining_q <= rows_remaining_q - 1'b1;
                end
            end
        end
    end

    initial begin
        if (BANK_DEPTH != 578 || BANK_ADDR_WIDTH != 10 ||
            ARRAY_ADDR_WIDTH != 11)
            $error("array16 refill dispatcher requires 4x200x578 bank geometry");
    end
endmodule

`default_nettype wire
