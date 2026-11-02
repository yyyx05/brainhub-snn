`timescale 1ns/1ps
`default_nettype none

// Common cache-maintenance plane for the one 16xH40/P4 weight store. Client 0
// converts scalar FC W0 parameters into packed H40 rows; client 1 accepts the
// already-packed rows produced by the SCNN Conv/classifier cache publishers.
module unified_array16_shared_weight_cache_refill_plane_v1 #(
    parameter int PARAM_DEPTH = 120320,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int VERSION_WIDTH = 16,
    parameter int FEATURE_COUNT_WIDTH = 12,
    parameter int HIDDEN_COUNT_WIDTH = 10,
    parameter int ARRAY_WEIGHT_ADDR_WIDTH = 11,
    parameter int ARRAY_TILES = 16,
    parameter bit ENABLE_FC_REFILL = 1'b1
) (
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,
    input  wire logic                              i_abort,
    input  wire logic                              i_clear_error,

    input  wire logic                              i_fc_refill_valid,
    output logic                                   o_fc_refill_ready,
    input  wire logic                              i_fc_refill_all,
    input  wire logic                              i_fc_weight_int8,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0]     i_fc_dirty_feature,
    input  wire logic [3:0]                        i_fc_dirty_tile,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_fc_parameter_base,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_fc_hidden_stride,
    input  wire logic [9:0]                         i_fc_cache_row_base,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0]     i_fc_feature_count,
    input  wire logic [HIDDEN_COUNT_WIDTH-1:0]      i_fc_hidden_count,
    input  wire logic [4:0]                        i_fc_tile_count,
    input  wire logic [3:0]                        i_fc_destination_tile_base,
    input  wire logic [VERSION_WIDTH-1:0]           i_fc_version,
    input  wire logic                              i_array_idle,

    output logic                                   o_forward_read_valid,
    input  wire logic                              i_forward_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]             o_forward_read_addr,
    output logic [TAG_WIDTH-1:0]                    o_forward_read_tag,
    input  wire logic                              i_forward_response_valid,
    output logic                                   o_forward_response_ready,
    input  wire logic [TAG_WIDTH-1:0]               i_forward_response_tag,
    input  wire logic signed [15:0]                 i_forward_response_code,
    input  wire logic                              i_forward_response_error,

    input  wire logic                              i_topology_fill_busy,
    input  wire logic [3:0]                        i_topology_fill_valid,
    output logic [3:0]                             o_topology_fill_ready,
    input  wire logic [7:0]                        i_topology_fill_tile_flat,
    input  wire logic [3:0]                        i_topology_fill_bank,
    input  wire logic [4*ARRAY_WEIGHT_ADDR_WIDTH-1:0]
                                                    i_topology_fill_addr_flat,
    input  wire logic [799:0]                      i_topology_fill_data_flat,

    output logic [3:0]                             o_weight_fill_valid,
    input  wire logic [3:0]                        i_weight_fill_ready,
    output logic [7:0]                             o_weight_fill_tile_flat,
    output logic [3:0]                             o_weight_fill_bank,
    output logic [4*ARRAY_WEIGHT_ADDR_WIDTH-1:0]    o_weight_fill_addr_flat,
    output logic [799:0]                           o_weight_fill_data_flat,

    output logic                                   o_fc_refill_done,
    output logic                                   o_fc_refill_busy,
    output logic [31:0]                            o_fc_rows_published,
    output logic [VERSION_WIDTH-1:0]               o_fc_published_version,
    output logic [1:0]                             o_fill_grant,
    output logic                                   o_busy,
    output logic                                   o_quiescent,
    output logic                                   o_sticky_error
);
    logic fc_refill_busy;
    logic fc_refill_error;
    logic [3:0] fc_fill_valid;
    logic [3:0] fc_fill_ready;
    logic [7:0] fc_fill_tile_flat;
    logic [3:0] fc_fill_bank;
    logic [4*ARRAY_WEIGHT_ADDR_WIDTH-1:0] fc_fill_addr_flat;
    logic [799:0] fc_fill_data_flat;
    logic fill_arbiter_busy;
    logic fill_arbiter_quiescent;
    logic fill_arbiter_error;

    assign o_fc_refill_busy = fc_refill_busy;
    assign o_busy = fc_refill_busy || fill_arbiter_busy;
    assign o_quiescent = !fc_refill_busy && fill_arbiter_quiescent;
    assign o_sticky_error = fc_refill_error || fill_arbiter_error;

    generate
        if (ENABLE_FC_REFILL) begin : g_fc_refill
            unified_fc_h40_w0_cache_refill_scheduler_v1 #(
                .PARAM_DEPTH(PARAM_DEPTH),
                .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
                .VERSION_WIDTH(VERSION_WIDTH),
                .FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
                .HIDDEN_COUNT_WIDTH(HIDDEN_COUNT_WIDTH),
                .ARRAY_WEIGHT_ADDR_WIDTH(ARRAY_WEIGHT_ADDR_WIDTH),
                .ARRAY_TILES(ARRAY_TILES)
            ) u_fc_refill (
                .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
                .i_clear_error(i_clear_error),
                .i_refill_valid(i_fc_refill_valid),
                .o_refill_ready(o_fc_refill_ready),
                .i_refill_all(i_fc_refill_all),
                .i_fc_weight_int8(i_fc_weight_int8),
                .i_dirty_feature(i_fc_dirty_feature),
                .i_dirty_tile(i_fc_dirty_tile),
                .i_parameter_base(i_fc_parameter_base),
                .i_hidden_stride(i_fc_hidden_stride),
                .i_cache_row_base(i_fc_cache_row_base),
                .i_feature_count(i_fc_feature_count),
                .i_hidden_count(i_fc_hidden_count),
                .i_tile_count(i_fc_tile_count), .i_version(i_fc_version),
                .i_destination_tile_base(i_fc_destination_tile_base),
                .i_array_idle(i_array_idle),
                .o_forward_read_valid(o_forward_read_valid),
                .i_forward_read_ready(i_forward_read_ready),
                .o_forward_read_addr(o_forward_read_addr),
                .o_forward_read_tag(o_forward_read_tag),
                .i_forward_response_valid(i_forward_response_valid),
                .o_forward_response_ready(o_forward_response_ready),
                .i_forward_response_tag(i_forward_response_tag),
                .i_forward_response_code(i_forward_response_code),
                .i_forward_response_error(i_forward_response_error),
                .o_weight_fill_valid(fc_fill_valid),
                .i_weight_fill_ready(fc_fill_ready),
                .o_weight_fill_tile_flat(fc_fill_tile_flat),
                .o_weight_fill_bank(fc_fill_bank),
                .o_weight_fill_addr_flat(fc_fill_addr_flat),
                .o_weight_fill_data_flat(fc_fill_data_flat),
                .o_refill_done(o_fc_refill_done),
                .o_rows_published(o_fc_rows_published),
                .o_published_version(o_fc_published_version),
                .o_busy(fc_refill_busy), .o_sticky_error(fc_refill_error)
            );
        end else begin : g_no_fc_refill
            assign o_fc_refill_ready = 1'b0;
            assign o_forward_read_valid = 1'b0;
            assign o_forward_read_addr = '0;
            assign o_forward_read_tag = '0;
            assign o_forward_response_ready = 1'b0;
            assign fc_fill_valid = '0;
            assign fc_fill_tile_flat = '0;
            assign fc_fill_bank = '0;
            assign fc_fill_addr_flat = '0;
            assign fc_fill_data_flat = '0;
            assign o_fc_refill_done = 1'b0;
            assign o_fc_rows_published = '0;
            assign o_fc_published_version = '0;
            assign fc_refill_busy = 1'b0;
            assign fc_refill_error = 1'b0;
        end
    endgenerate

    unified_array16_two_client_weight_fill_arbiter_v1 #(
        .ARRAY_WEIGHT_ADDR_WIDTH(ARRAY_WEIGHT_ADDR_WIDTH)
    ) u_fill_arbiter (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_client0_busy(fc_refill_busy),
        .i_client0_fill_valid(fc_fill_valid),
        .o_client0_fill_ready(fc_fill_ready),
        .i_client0_fill_tile_flat(fc_fill_tile_flat),
        .i_client0_fill_bank(fc_fill_bank),
        .i_client0_fill_addr_flat(fc_fill_addr_flat),
        .i_client0_fill_data_flat(fc_fill_data_flat),
        .i_client1_busy(i_topology_fill_busy),
        .i_client1_fill_valid(i_topology_fill_valid),
        .o_client1_fill_ready(o_topology_fill_ready),
        .i_client1_fill_tile_flat(i_topology_fill_tile_flat),
        .i_client1_fill_bank(i_topology_fill_bank),
        .i_client1_fill_addr_flat(i_topology_fill_addr_flat),
        .i_client1_fill_data_flat(i_topology_fill_data_flat),
        .o_fill_valid(o_weight_fill_valid),
        .i_fill_ready(i_weight_fill_ready),
        .o_fill_tile_flat(o_weight_fill_tile_flat),
        .o_fill_bank(o_weight_fill_bank),
        .o_fill_addr_flat(o_weight_fill_addr_flat),
        .o_fill_data_flat(o_weight_fill_data_flat),
        .o_grant(o_fill_grant), .o_busy(fill_arbiter_busy),
        .o_quiescent(fill_arbiter_quiescent),
        .o_sticky_error(fill_arbiter_error)
    );

    initial begin
        if (PARAM_DEPTH < 1 || PARAM_ADDR_WIDTH < 1 ||
                ARRAY_WEIGHT_ADDR_WIDTH != 11 || ARRAY_TILES < 1 ||
                ARRAY_TILES > 16)
            $error("shared array16 weight-cache refill plane geometry is invalid");
    end
endmodule

`default_nettype wire
