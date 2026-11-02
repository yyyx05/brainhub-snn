`timescale 1ns/1ps
`default_nettype none

// Repackages one scalar parameter sequence into one physical H40 cache row.
// Topology adapters own tensor-to-row address mapping; this module owns the
// shared forward-read, precision packing and atomic whole-row publication.
module unified_h40_scalar_parameter_row_publisher_v1 #(
    parameter int PARAM_DEPTH = 120320,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int VERSION_WIDTH = 16,
    parameter int ARRAY_WEIGHT_ADDR_WIDTH = 11
) (
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,
    input  wire logic                              i_abort,
    input  wire logic                              i_clear_error,

    input  wire logic                              i_command_valid,
    output logic                                   o_command_ready,
    input  wire logic [3:0]                        i_target_tile,
    input  wire logic [1:0]                        i_target_bank,
    input  wire logic [9:0]                        i_target_row,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_scalar_base,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_scalar_stride,
    input  wire logic [5:0]                        i_scalar_count,
    input  wire logic [1:0]                        i_element_format,
    input  wire logic [VERSION_WIDTH-1:0]           i_version,
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

    output logic [3:0]                             o_weight_fill_valid,
    input  wire logic [3:0]                        i_weight_fill_ready,
    output logic [7:0]                             o_weight_fill_tile_flat,
    output logic [3:0]                             o_weight_fill_bank,
    output logic [4*ARRAY_WEIGHT_ADDR_WIDTH-1:0]    o_weight_fill_addr_flat,
    output logic [799:0]                           o_weight_fill_data_flat,

    output logic                                   o_done,
    output logic [3:0]                             o_published_tile,
    output logic [1:0]                             o_published_bank,
    output logic [9:0]                             o_published_row,
    output logic [VERSION_WIDTH-1:0]               o_published_version,
    output logic                                   o_busy,
    output logic                                   o_sticky_error
);
    localparam logic [1:0] FORMAT_INT5 = 2'd0;
    localparam logic [1:0] FORMAT_INT8 = 2'd1;
    localparam logic [1:0] FORMAT_INT16 = 2'd2;
    localparam int RANGE_WIDTH = PARAM_ADDR_WIDTH + 6;

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_DISPATCH,
        STATE_READ_REQUEST,
        STATE_READ_RESPONSE,
        STATE_SEND_ROW,
        STATE_WAIT_COMPLETE
    } state_t;

    state_t state_q;
    logic [3:0] target_tile_q;
    logic [1:0] target_bank_q;
    logic [9:0] target_row_q;
    logic [PARAM_ADDR_WIDTH-1:0] scalar_addr_q;
    logic [PARAM_ADDR_WIDTH-1:0] scalar_stride_q;
    logic [5:0] scalar_count_q;
    logic [5:0] scalar_index_q;
    logic [1:0] element_format_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic [TAG_WIDTH-1:0] read_tag_q;
    logic [199:0] row_data_q;

    logic dispatcher_command_valid;
    logic dispatcher_command_ready;
    logic dispatcher_row_valid;
    logic dispatcher_row_ready;
    logic dispatcher_complete_valid;
    logic dispatcher_complete_ready;
    logic [3:0] dispatcher_complete_tile;
    logic [1:0] dispatcher_complete_bank;
    logic [VERSION_WIDTH-1:0] dispatcher_complete_version;
    logic dispatcher_busy;
    logic dispatcher_error;

    logic command_fire;
    logic read_fire;
    logic response_fire;
    logic row_fire;
    logic complete_fire;
    logic command_legal;
    logic response_code_legal;
    logic unsafe_refill;
    logic publisher_abort;
    logic dispatcher_abort;
    logic response_fault;
    logic next_read_valid;
    logic [5:0] element_capacity;
    logic [RANGE_WIDTH-1:0] last_scalar_addr_ext;
    logic [RANGE_WIDTH-1:0] scalar_offset_ext;

    always_comb begin
        unique case (i_element_format)
            FORMAT_INT5: element_capacity = 6'd40;
            FORMAT_INT8: element_capacity = 6'd25;
            FORMAT_INT16: element_capacity = 6'd12;
            default: element_capacity = '0;
        endcase

        scalar_offset_ext = RANGE_WIDTH'(i_scalar_stride) *
            RANGE_WIDTH'(i_scalar_count - 1'b1);
        last_scalar_addr_ext = RANGE_WIDTH'(i_scalar_base) +
            scalar_offset_ext;
        command_legal = i_target_row < 10'd578 && i_scalar_stride != 0 &&
            i_scalar_count != 0 && i_scalar_count <= element_capacity &&
            last_scalar_addr_ext < RANGE_WIDTH'(PARAM_DEPTH);

        unique case (element_format_q)
            FORMAT_INT5: response_code_legal =
                $signed(i_forward_response_code) >= -16'sd16 &&
                $signed(i_forward_response_code) <= 16'sd15;
            FORMAT_INT8: response_code_legal =
                $signed(i_forward_response_code) >= -16'sd128 &&
                $signed(i_forward_response_code) <= 16'sd127;
            FORMAT_INT16: response_code_legal = 1'b1;
            default: response_code_legal = 1'b0;
        endcase
    end

    assign unsafe_refill = state_q != STATE_IDLE && !i_array_idle;
    assign publisher_abort = i_abort || unsafe_refill;
    assign response_fault = response_fire &&
        (i_forward_response_error ||
         i_forward_response_tag != read_tag_q ||
         !response_code_legal);
    assign dispatcher_abort = publisher_abort || response_fault;
    assign o_command_ready = state_q == STATE_IDLE && !dispatcher_busy &&
        !o_sticky_error && i_array_idle;
    assign command_fire = i_command_valid && o_command_ready;
    assign o_busy = state_q != STATE_IDLE || dispatcher_busy;

    assign dispatcher_command_valid = state_q == STATE_DISPATCH;
    assign dispatcher_row_valid = state_q == STATE_SEND_ROW;
    assign dispatcher_complete_ready = state_q == STATE_WAIT_COMPLETE;
    assign read_fire = o_forward_read_valid && i_forward_read_ready;
    assign response_fire = i_forward_response_valid &&
        o_forward_response_ready;
    assign row_fire = dispatcher_row_valid && dispatcher_row_ready;
    assign complete_fire = dispatcher_complete_valid &&
        dispatcher_complete_ready;

    // Retiring the old scalar frees the only credit. If the backend stalls,
    // the incremented request persists in STATE_READ_REQUEST next cycle.
    assign next_read_valid = response_fire && !response_fault &&
        !publisher_abort && (scalar_index_q + 1'b1 != scalar_count_q);
    assign o_forward_read_valid = state_q == STATE_READ_REQUEST || next_read_valid;
    assign o_forward_read_addr = next_read_valid ?
        scalar_addr_q + scalar_stride_q : scalar_addr_q;
    assign o_forward_read_tag = next_read_valid ? read_tag_q + 1'b1 : read_tag_q;
    assign o_forward_response_ready = state_q == STATE_READ_RESPONSE;

    unified_array16_weight_refill_dispatcher_v1 #(
        .BANK_DEPTH(578), .BANK_ADDR_WIDTH(10),
        .ARRAY_ADDR_WIDTH(ARRAY_WEIGHT_ADDR_WIDTH),
        .TAG_WIDTH(VERSION_WIDTH)
    ) u_refill_dispatcher (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_abort(dispatcher_abort), .i_clear_error(i_clear_error),
        .i_command_valid(dispatcher_command_valid),
        .o_command_ready(dispatcher_command_ready),
        .i_command_tile(target_tile_q),
        .i_command_bank(target_bank_q),
        .i_command_base_addr(target_row_q), .i_command_rows(11'd1),
        .i_command_tag(version_q),
        .i_row_valid(dispatcher_row_valid),
        .o_row_ready(dispatcher_row_ready), .i_row_data(row_data_q),
        .o_fill_valid(o_weight_fill_valid),
        .i_fill_ready(i_weight_fill_ready),
        .o_fill_tile_flat(o_weight_fill_tile_flat),
        .o_fill_bank(o_weight_fill_bank),
        .o_fill_addr_flat(o_weight_fill_addr_flat),
        .o_fill_data_flat(o_weight_fill_data_flat),
        .o_complete_valid(dispatcher_complete_valid),
        .i_complete_ready(dispatcher_complete_ready),
        .o_complete_tile(dispatcher_complete_tile),
        .o_complete_bank(dispatcher_complete_bank),
        .o_complete_tag(dispatcher_complete_version),
        .o_busy(dispatcher_busy), .o_sticky_error(dispatcher_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            target_tile_q <= '0;
            target_bank_q <= '0;
            target_row_q <= '0;
            scalar_addr_q <= '0;
            scalar_stride_q <= '0;
            scalar_count_q <= '0;
            scalar_index_q <= '0;
            element_format_q <= FORMAT_INT5;
            version_q <= '0;
            read_tag_q <= '0;
            row_data_q <= '0;
            o_done <= 1'b0;
            o_published_tile <= '0;
            o_published_bank <= '0;
            o_published_row <= '0;
            o_published_version <= '0;
            o_sticky_error <= 1'b0;
        end else if (publisher_abort) begin
            state_q <= STATE_IDLE;
            scalar_index_q <= '0;
            row_data_q <= '0;
            o_done <= 1'b0;
            if (unsafe_refill)
                o_sticky_error <= 1'b1;
        end else begin
            o_done <= 1'b0;
            if (i_clear_error && state_q == STATE_IDLE && !dispatcher_busy)
                o_sticky_error <= 1'b0;
            if (dispatcher_error)
                o_sticky_error <= 1'b1;

            unique case (state_q)
                STATE_IDLE: begin
                    if (command_fire) begin
                        if (!command_legal) begin
                            o_sticky_error <= 1'b1;
                        end else begin
                            target_tile_q <= i_target_tile;
                            target_bank_q <= i_target_bank;
                            target_row_q <= i_target_row;
                            scalar_addr_q <= i_scalar_base;
                            scalar_stride_q <= i_scalar_stride;
                            scalar_count_q <= i_scalar_count;
                            scalar_index_q <= '0;
                            element_format_q <= i_element_format;
                            version_q <= i_version;
                            read_tag_q <= '0;
                            row_data_q <= '0;
                            state_q <= STATE_DISPATCH;
                        end
                    end
                end

                STATE_DISPATCH: begin
                    if (dispatcher_command_valid && dispatcher_command_ready)
                        state_q <= STATE_READ_REQUEST;
                end

                STATE_READ_REQUEST: begin
                    if (read_fire)
                        state_q <= STATE_READ_RESPONSE;
                end

                STATE_READ_RESPONSE: begin
                    if (response_fire) begin
                        if (i_forward_response_error ||
                                i_forward_response_tag != read_tag_q ||
                                !response_code_legal) begin
                            state_q <= STATE_IDLE;
                            o_sticky_error <= 1'b1;
                        end else begin
                            unique case (element_format_q)
                                FORMAT_INT5:
                                    row_data_q[scalar_index_q*5 +: 5] <=
                                        i_forward_response_code[4:0];
                                FORMAT_INT8:
                                    row_data_q[scalar_index_q*8 +: 8] <=
                                        i_forward_response_code[7:0];
                                default:
                                    row_data_q[scalar_index_q*16 +: 16] <=
                                        i_forward_response_code;
                            endcase
                            if (scalar_index_q + 1'b1 == scalar_count_q) begin
                                state_q <= STATE_SEND_ROW;
                            end else begin
                                scalar_index_q <= scalar_index_q + 1'b1;
                                scalar_addr_q <= scalar_addr_q +
                                    scalar_stride_q;
                                read_tag_q <= read_tag_q + 1'b1;
                                state_q <= read_fire ? STATE_READ_RESPONSE : STATE_READ_REQUEST;
                            end
                        end
                    end
                end

                STATE_SEND_ROW: begin
                    if (row_fire)
                        state_q <= STATE_WAIT_COMPLETE;
                end

                STATE_WAIT_COMPLETE: begin
                    if (complete_fire) begin
                        if (dispatcher_complete_tile != target_tile_q ||
                                dispatcher_complete_bank != target_bank_q ||
                                dispatcher_complete_version != version_q) begin
                            o_sticky_error <= 1'b1;
                        end else begin
                            o_done <= 1'b1;
                            o_published_tile <= target_tile_q;
                            o_published_bank <= target_bank_q;
                            o_published_row <= target_row_q;
                            o_published_version <= version_q;
                        end
                        state_q <= STATE_IDLE;
                    end
                end

                default: begin
                    state_q <= STATE_IDLE;
                    o_sticky_error <= 1'b1;
                end
            endcase
        end
    end

    initial begin
        if (PARAM_DEPTH < 1 || PARAM_ADDR_WIDTH < 1 || TAG_WIDTH < 1 ||
                VERSION_WIDTH < 1 || ARRAY_WEIGHT_ADDR_WIDTH != 11)
            $error("H40 scalar row publisher parameter mismatch");
    end
endmodule

`default_nettype wire
