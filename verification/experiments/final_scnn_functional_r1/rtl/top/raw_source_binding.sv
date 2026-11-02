`timescale 1ns/1ps
`default_nettype none

module raw_source_binding #(
    parameter int POLICY = 1,
    parameter int ADDR_WIDTH = 11
) (
    input wire i_clk, i_rst_n,
    input wire i_raw_source_active, i_raw_count_valid,
    input wire [31:0] i_raw_counts,
    output wire o_raw_count_ready,
    input wire i_manual_group_valid, i_manual_group_recurrent,
    input wire [3:0] i_manual_group_source_valid,
    input wire [ADDR_WIDTH-1:0] i_manual_group_weight_addr0,
    input wire [ADDR_WIDTH-1:0] i_manual_group_weight_addr1,
    input wire [ADDR_WIDTH-1:0] i_manual_group_weight_addr2,
    input wire [ADDR_WIDTH-1:0] i_manual_group_weight_addr3,
    input wire i_manual_group_feedforward_end,
    input wire i_core_group_ready, i_core_layer_done,
    output wire o_group_valid, o_group_recurrent,
    output wire [3:0] o_group_source_valid,
    output wire [ADDR_WIDTH-1:0] o_group_weight_addr0,
    output wire [ADDR_WIDTH-1:0] o_group_weight_addr1,
    output wire [ADDR_WIDTH-1:0] o_group_weight_addr2,
    output wire [ADDR_WIDTH-1:0] o_group_weight_addr3,
    output wire o_group_feedforward_end
);
    wire producer_valid, producer_last;
    wire [3:0] producer_mask;
    wire [9:0] producer_row;

    // Static extraction of the validated TB producer, with no active gate on
    // count_valid or layer_done. POLICY is an elaboration-time choice only.
    generate
        if (POLICY == 0) begin : g_buffered
            source_scan producer (
                .clk(i_clk), .rst_n(i_rst_n), .sparse(1'b1),
                .count_valid(i_raw_count_valid), .count_ready(o_raw_count_ready),
                .counts(i_raw_counts), .group_valid(producer_valid),
                .group_ready(i_raw_source_active && i_core_group_ready),
                .first(), .last(producer_last), .mask(producer_mask),
                .row(producer_row), .result_fire(i_core_layer_done), .phase()
            );
        end else begin : g_stream
            source_stream producer (
                .clk(i_clk), .rst_n(i_rst_n),
                .count_valid(i_raw_count_valid), .count_ready(o_raw_count_ready),
                .counts(i_raw_counts), .group_valid(producer_valid),
                .group_ready(i_raw_source_active && i_core_group_ready),
                .first(), .last(producer_last), .mask(producer_mask),
                .row(producer_row), .result_fire(i_core_layer_done), .phase()
            );
        end
    endgenerate

    assign o_group_valid = i_raw_source_active ? producer_valid : i_manual_group_valid;
    assign o_group_recurrent = i_raw_source_active ? 1'b0 : i_manual_group_recurrent;
    assign o_group_source_valid = i_raw_source_active ? producer_mask : i_manual_group_source_valid;
    assign o_group_weight_addr0 = i_raw_source_active ? ADDR_WIDTH'(producer_row) : i_manual_group_weight_addr0;
    assign o_group_weight_addr1 = i_raw_source_active ? ADDR_WIDTH'(producer_row) : i_manual_group_weight_addr1;
    assign o_group_weight_addr2 = i_raw_source_active ? ADDR_WIDTH'(producer_row) + ADDR_WIDTH'(1) : i_manual_group_weight_addr2;
    assign o_group_weight_addr3 = i_raw_source_active ? ADDR_WIDTH'(producer_row) + ADDR_WIDTH'(1) : i_manual_group_weight_addr3;
    assign o_group_feedforward_end = i_raw_source_active ? producer_last : i_manual_group_feedforward_end;
endmodule

`default_nettype wire
