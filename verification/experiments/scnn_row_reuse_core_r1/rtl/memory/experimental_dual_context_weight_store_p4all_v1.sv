`timescale 1ns/1ps
`default_nettype none

// Mechanism prototype only. Context 0 retains the shared FC/SRNN/SCNN image;
// context 1 retains M2 Q2.14 weights. Arithmetic and read latency are unchanged.
module experimental_dual_context_weight_store_p4all_v1 #(
    parameter int M2_CACHE_DEPTH = 578,
    parameter bit ENABLE_M2_CONTEXT = 1'b1,
    parameter int LANES = 40,
    parameter int BANK_WIDTH = 200,
    parameter int BANK_DEPTH = 578,
    parameter int ADDR_WIDTH = $clog2(BANK_DEPTH)
) (
    input  wire logic                    i_clk,
    input  wire logic                    i_rst_n,

    input  wire logic                    i_fill_valid,
    output logic                         o_fill_ready,
    input  wire logic                    i_fill_context,
    input  wire logic [1:0]              i_fill_bank,
    input  wire logic [ADDR_WIDTH-1:0]   i_fill_addr,
    input  wire logic [BANK_WIDTH-1:0]   i_fill_data,

    input  wire logic                    i_read_valid,
    output logic                         o_read_ready,
    input  wire logic [1:0]              i_mode,
    input  wire logic                    i_fc_weight_int8,
    input  wire logic                    i_scnn_active_pair,
    input  wire logic [ADDR_WIDTH-1:0]   i_read_addr0,
    input  wire logic [ADDR_WIDTH-1:0]   i_read_addr1,
    input  wire logic [ADDR_WIDTH-1:0]   i_read_addr2,
    input  wire logic [ADDR_WIDTH-1:0]   i_read_addr3,
    input  wire logic [3:0]              i_read_subword0,
    input  wire logic [3:0]              i_read_subword1,
    input  wire logic [3:0]              i_read_subword2,
    input  wire logic [3:0]              i_read_subword3,
    output logic                         o_read_result_valid,
    input  wire logic                    i_read_result_ready,
    output logic [LANES*25-1:0]          o_term0_flat,
    output logic [LANES*25-1:0]          o_term1_flat,
    output logic [LANES*25-1:0]          o_term2_flat,
    output logic [LANES*25-1:0]          o_term3_flat
);
    localparam logic [1:0] MODE_M2_Q14 = 2'd3;

    logic fill_ready [0:1];
    logic read_ready [0:1];
    logic read_result_valid [0:1];
    logic [LANES*25-1:0] term0_flat [0:1];
    logic [LANES*25-1:0] term1_flat [0:1];
    logic [LANES*25-1:0] term2_flat [0:1];
    logic [LANES*25-1:0] term3_flat [0:1];
    logic read_context;
    logic read_context_q;
    logic read_fire;
    logic response_pending_q;
    logic request_slot_ready;

    assign read_context = ENABLE_M2_CONTEXT && i_mode == MODE_M2_Q14;
    assign o_fill_ready = fill_ready[i_fill_context];
    assign request_slot_ready = !response_pending_q || i_read_result_ready;
    assign o_read_ready = request_slot_ready && read_ready[read_context];
    assign read_fire = i_read_valid && o_read_ready;
    assign o_read_result_valid = read_result_valid[read_context_q];
    assign o_term0_flat = term0_flat[read_context_q];
    assign o_term1_flat = term1_flat[read_context_q];
    assign o_term2_flat = term2_flat[read_context_q];
    assign o_term3_flat = term3_flat[read_context_q];

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            read_context_q <= 1'b0;
            response_pending_q <= 1'b0;
        end else begin
            if (o_read_result_valid && i_read_result_ready)
                response_pending_q <= 1'b0;
            if (read_fire) begin
                read_context_q <= read_context;
                response_pending_q <= 1'b1;
            end
        end
    end

    generate
        for (genvar ctx_idx = 0; ctx_idx < 2; ctx_idx++) begin : g_context
            if (ctx_idx == 0 || ENABLE_M2_CONTEXT) begin : g_enabled
            if (ctx_idx == 1) begin : g_compact
                m2_compact_store #(.DEPTH(M2_CACHE_DEPTH)) u_store (
                    .i_clk(i_clk), .i_rst_n(i_rst_n),
                    .i_fill_valid(i_fill_valid && i_fill_context == ctx_idx),
                    .o_fill_ready(fill_ready[ctx_idx]),
                    .i_fill_bank(i_fill_bank), .i_fill_addr(i_fill_addr),
                    .i_fill_data(i_fill_data),
                    .i_read_valid(i_read_valid && request_slot_ready && read_context == ctx_idx),
                    .o_read_ready(read_ready[ctx_idx]), .i_read_addr(i_read_addr0),
                    .o_read_result_valid(read_result_valid[ctx_idx]),
                    .i_read_result_ready(i_read_result_ready && read_context_q == ctx_idx),
                    .o_term0_flat(term0_flat[ctx_idx])
                );
                assign term1_flat[ctx_idx] = '0;
                assign term2_flat[ctx_idx] = '0;
                assign term3_flat[ctx_idx] = '0;
            end else begin : g_original
            unified_dual_role_weight_store_p4all_candidate_v1 #(
                .LANES(LANES), .BANK_WIDTH(BANK_WIDTH),
                .BANK_DEPTH(BANK_DEPTH), .ADDR_WIDTH(ADDR_WIDTH)
            ) u_store (
                .i_clk(i_clk), .i_rst_n(i_rst_n),
                .i_fill_valid(i_fill_valid && i_fill_context == ctx_idx),
                .o_fill_ready(fill_ready[ctx_idx]),
                .i_fill_bank(i_fill_bank), .i_fill_addr(i_fill_addr),
                .i_fill_data(i_fill_data),
                .i_read_valid(i_read_valid && request_slot_ready && read_context == ctx_idx),
                .o_read_ready(read_ready[ctx_idx]),
                .i_mode(ctx_idx == 1 ? MODE_M2_Q14 : i_mode),
                .i_fc_weight_int8(i_fc_weight_int8),
                .i_scnn_active_pair(i_scnn_active_pair),
                .i_read_addr0(i_read_addr0), .i_read_addr1(i_read_addr1),
                .i_read_addr2(i_read_addr2), .i_read_addr3(i_read_addr3),
                .i_read_subword0(i_read_subword0),
                .i_read_subword1(i_read_subword1),
                .i_read_subword2(i_read_subword2),
                .i_read_subword3(i_read_subword3),
                .o_read_result_valid(read_result_valid[ctx_idx]),
                .i_read_result_ready(i_read_result_ready &&
                    read_context_q == ctx_idx),
                .o_term0_flat(term0_flat[ctx_idx]),
                .o_term1_flat(term1_flat[ctx_idx]),
                .o_term2_flat(term2_flat[ctx_idx]),
                .o_term3_flat(term3_flat[ctx_idx])
            );
            end
            end else begin : g_unused
                assign fill_ready[ctx_idx] = 1'b0;
                assign read_ready[ctx_idx] = 1'b0;
                assign read_result_valid[ctx_idx] = 1'b0;
                assign term0_flat[ctx_idx] = '0;
                assign term1_flat[ctx_idx] = '0;
                assign term2_flat[ctx_idx] = '0;
                assign term3_flat[ctx_idx] = '0;
            end
        end
    endgenerate
endmodule

`default_nettype wire
