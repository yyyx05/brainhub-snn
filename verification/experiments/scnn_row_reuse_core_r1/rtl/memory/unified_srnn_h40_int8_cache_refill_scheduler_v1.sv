`timescale 1ns/1ps
`default_nettype none

// Packs a destination-major signed INT8 SRNN matrix into the shared H40
// weight cache. Source parity selects bank pair {0,1} or {2,3}; the lower
// and upper twenty destination lanes select the bank within that pair. Two
// consecutive sources therefore share one physical row address while still
// supplying the four independent terms consumed by the P4 SRNN datapath.
module unified_srnn_h40_int8_cache_refill_scheduler_v1 #(
    parameter int PARAM_DEPTH = 120320,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int VERSION_WIDTH = 16,
    parameter int SOURCE_COUNT_WIDTH = 10,
    parameter int DESTINATION_COUNT_WIDTH = 10,
    parameter int ARRAY_WEIGHT_ADDR_WIDTH = 11,
    parameter int ARRAY_TILES = 12
) (
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,
    input  wire logic                              i_abort,
    input  wire logic                              i_clear_error,

    input  wire logic                              i_refill_valid,
    output logic                                   o_refill_ready,
    input  wire logic                              i_refill_all,
    input  wire logic [SOURCE_COUNT_WIDTH-1:0]      i_dirty_source,
    input  wire logic [3:0]                        i_dirty_destination_tile,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_parameter_base,
    input  wire logic [9:0]                        i_cache_row_base,
    input  wire logic [SOURCE_COUNT_WIDTH-1:0]      i_source_count,
    input  wire logic [DESTINATION_COUNT_WIDTH-1:0] i_destination_count,
    input  wire logic [3:0]                        i_destination_tile_base,
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

    output logic                                   o_refill_done,
    output logic [31:0]                            o_rows_published,
    output logic [VERSION_WIDTH-1:0]               o_published_version,
    output logic                                   o_busy,
    output logic                                   o_sticky_error
);
    localparam int RANGE_WIDTH = PARAM_ADDR_WIDTH +
        SOURCE_COUNT_WIDTH + DESTINATION_COUNT_WIDTH + 2;

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_ISSUE,
        STATE_WAIT
    } state_t;

    state_t state_q;
    logic refill_all_q;
    logic [PARAM_ADDR_WIDTH-1:0] parameter_base_q;
    logic [9:0] cache_row_base_q;
    logic [SOURCE_COUNT_WIDTH-1:0] source_count_q;
    logic [DESTINATION_COUNT_WIDTH-1:0] destination_count_q;
    logic [4:0] destination_tile_count_q;
    logic [3:0] destination_tile_base_q;
    logic [SOURCE_COUNT_WIDTH-1:0] source_q;
    logic [3:0] destination_tile_q;
    logic destination_half_q;
    logic [VERSION_WIDTH-1:0] version_q;

    logic row_command_valid;
    logic row_command_ready;
    logic row_done;
    logic row_busy;
    logic row_error;
    logic [3:0] row_published_tile;
    logic [1:0] row_published_bank;
    logic [9:0] row_published_row;
    logic [VERSION_WIDTH-1:0] row_published_version;

    logic request_fire;
    logic request_legal;
    logic row_command_fire;
    logic [4:0] requested_tile_count;
    logic [DESTINATION_COUNT_WIDTH:0] tile_destination_base;
    logic [DESTINATION_COUNT_WIDTH:0] tile_destination_remaining;
    logic [5:0] tile_lane_count;
    logic [DESTINATION_COUNT_WIDTH:0] half_destination_base;
    logic [5:0] row_scalar_count;
    logic [PARAM_ADDR_WIDTH-1:0] row_scalar_base;
    logic [9:0] target_row;
    logic [3:0] target_tile;
    logic [1:0] target_bank;
    logic last_half;
    logic last_tile;
    logic last_source;
    logic [RANGE_WIDTH-1:0] matrix_end_ext;
    logic [RANGE_WIDTH-1:0] row_scalar_base_ext;
    logic [10:0] cache_last_row_ext;
    logic [5:0] physical_tile_end;

    always_comb begin
        requested_tile_count = '0;
        for (int tile = 0; tile < ARRAY_TILES; tile++) begin
            if (16'(i_destination_count) > 16'(tile * 40))
                requested_tile_count = 5'(tile + 1);
        end

        matrix_end_ext = RANGE_WIDTH'(i_parameter_base) +
            RANGE_WIDTH'(i_source_count) *
                RANGE_WIDTH'(i_destination_count);
        cache_last_row_ext = {1'b0, i_cache_row_base} +
            (11'(i_source_count - 1'b1) >> 1);
        physical_tile_end = {2'b0, i_destination_tile_base} +
            {1'b0, requested_tile_count};
        request_legal = i_source_count != 0 &&
            i_destination_count != 0 && requested_tile_count != 0 &&
            16'(i_destination_count) <= 16'(ARRAY_TILES * 40) &&
            physical_tile_end <= 6'(ARRAY_TILES) &&
            matrix_end_ext <= RANGE_WIDTH'(PARAM_DEPTH) &&
            cache_last_row_ext < 11'd578 &&
            (i_refill_all ||
             (i_dirty_source < i_source_count &&
              5'(i_dirty_destination_tile) < requested_tile_count));

        tile_destination_base =
            (DESTINATION_COUNT_WIDTH + 1)'(destination_tile_q) *
                (DESTINATION_COUNT_WIDTH + 1)'(40);
        tile_destination_remaining =
            (DESTINATION_COUNT_WIDTH + 1)'(destination_count_q) -
                tile_destination_base;
        tile_lane_count = tile_destination_remaining >=
            (DESTINATION_COUNT_WIDTH + 1)'(40) ? 6'd40 :
            6'(tile_destination_remaining);
        half_destination_base = tile_destination_base +
            (destination_half_q ?
                (DESTINATION_COUNT_WIDTH + 1)'(20) : '0);
        row_scalar_count = destination_half_q ?
            (tile_lane_count > 6'd20 ? tile_lane_count - 6'd20 : 6'd0) :
            (tile_lane_count >= 6'd20 ? 6'd20 : tile_lane_count);
        row_scalar_base_ext = RANGE_WIDTH'(parameter_base_q) +
            RANGE_WIDTH'(half_destination_base) *
                RANGE_WIDTH'(source_count_q) +
            RANGE_WIDTH'(source_q);
        row_scalar_base = PARAM_ADDR_WIDTH'(row_scalar_base_ext);
        target_row = cache_row_base_q + 10'(source_q >> 1);
        target_tile = destination_tile_base_q + destination_tile_q;
        target_bank = {source_q[0], destination_half_q};
        last_half = destination_half_q || tile_lane_count <= 6'd20;
        last_tile = destination_tile_q + 1'b1 ==
            destination_tile_count_q;
        last_source = source_q + 1'b1 == source_count_q;
    end

    assign o_refill_ready = state_q == STATE_IDLE && !row_busy &&
        !o_sticky_error && i_array_idle;
    assign request_fire = i_refill_valid && o_refill_ready;
    assign row_command_valid = state_q == STATE_ISSUE;
    assign row_command_fire = row_command_valid && row_command_ready;
    assign o_busy = state_q != STATE_IDLE || row_busy;

    unified_h40_scalar_parameter_row_publisher_v1 #(
        .PARAM_DEPTH(PARAM_DEPTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH),
        .ARRAY_WEIGHT_ADDR_WIDTH(ARRAY_WEIGHT_ADDR_WIDTH)
    ) u_row_publisher (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_command_valid(row_command_valid),
        .o_command_ready(row_command_ready),
        .i_target_tile(target_tile), .i_target_bank(target_bank),
        .i_target_row(target_row),
        .i_scalar_base(row_scalar_base),
        .i_scalar_stride(PARAM_ADDR_WIDTH'(source_count_q)),
        .i_scalar_count(row_scalar_count), .i_element_format(2'd1),
        .i_version(version_q), .i_array_idle(i_array_idle),
        .o_forward_read_valid(o_forward_read_valid),
        .i_forward_read_ready(i_forward_read_ready),
        .o_forward_read_addr(o_forward_read_addr),
        .o_forward_read_tag(o_forward_read_tag),
        .i_forward_response_valid(i_forward_response_valid),
        .o_forward_response_ready(o_forward_response_ready),
        .i_forward_response_tag(i_forward_response_tag),
        .i_forward_response_code(i_forward_response_code),
        .i_forward_response_error(i_forward_response_error),
        .o_weight_fill_valid(o_weight_fill_valid),
        .i_weight_fill_ready(i_weight_fill_ready),
        .o_weight_fill_tile_flat(o_weight_fill_tile_flat),
        .o_weight_fill_bank(o_weight_fill_bank),
        .o_weight_fill_addr_flat(o_weight_fill_addr_flat),
        .o_weight_fill_data_flat(o_weight_fill_data_flat),
        .o_done(row_done), .o_published_tile(row_published_tile),
        .o_published_bank(row_published_bank),
        .o_published_row(row_published_row),
        .o_published_version(row_published_version),
        .o_busy(row_busy), .o_sticky_error(row_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            refill_all_q <= 1'b0;
            parameter_base_q <= '0;
            cache_row_base_q <= '0;
            source_count_q <= '0;
            destination_count_q <= '0;
            destination_tile_count_q <= '0;
            destination_tile_base_q <= '0;
            source_q <= '0;
            destination_tile_q <= '0;
            destination_half_q <= 1'b0;
            version_q <= '0;
            o_refill_done <= 1'b0;
            o_rows_published <= '0;
            o_published_version <= '0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            destination_half_q <= 1'b0;
            o_refill_done <= 1'b0;
        end else begin
            o_refill_done <= 1'b0;
            if (i_clear_error && state_q == STATE_IDLE && !row_busy)
                o_sticky_error <= 1'b0;
            if (row_error)
                o_sticky_error <= 1'b1;

            unique case (state_q)
                STATE_IDLE: begin
                    if (request_fire) begin
                        if (!request_legal) begin
                            o_sticky_error <= 1'b1;
                        end else begin
                            refill_all_q <= i_refill_all;
                            parameter_base_q <= i_parameter_base;
                            cache_row_base_q <= i_cache_row_base;
                            source_count_q <= i_source_count;
                            destination_count_q <= i_destination_count;
                            destination_tile_count_q <=
                                requested_tile_count;
                            destination_tile_base_q <=
                                i_destination_tile_base;
                            source_q <= i_refill_all ? '0 : i_dirty_source;
                            destination_tile_q <= i_refill_all ? '0 :
                                i_dirty_destination_tile;
                            destination_half_q <= 1'b0;
                            version_q <= i_version;
                            o_rows_published <= '0;
                            state_q <= STATE_ISSUE;
                        end
                    end
                end

                STATE_ISSUE: begin
                    if (row_command_fire)
                        state_q <= STATE_WAIT;
                end

                STATE_WAIT: begin
                    if (row_error) begin
                        // A malformed scalar response terminates the current
                        // row without producing row_done. Return ownership to
                        // the command boundary while preserving the sticky
                        // diagnostic so software can clear and retry.
                        state_q <= STATE_IDLE;
                    end else if (row_done) begin
                        if (row_published_tile != target_tile ||
                                row_published_bank != target_bank ||
                                row_published_row != target_row ||
                                row_published_version != version_q) begin
                            o_sticky_error <= 1'b1;
                            state_q <= STATE_IDLE;
                        end else begin
                            o_rows_published <= o_rows_published + 1'b1;
                            if (!last_half) begin
                                destination_half_q <= 1'b1;
                                state_q <= STATE_ISSUE;
                            end else if (!refill_all_q ||
                                    (last_tile && last_source)) begin
                                o_refill_done <= 1'b1;
                                o_published_version <= version_q;
                                state_q <= STATE_IDLE;
                            end else if (!last_tile) begin
                                destination_tile_q <=
                                    destination_tile_q + 1'b1;
                                destination_half_q <= 1'b0;
                                state_q <= STATE_ISSUE;
                            end else begin
                                source_q <= source_q + 1'b1;
                                destination_tile_q <= '0;
                                destination_half_q <= 1'b0;
                                state_q <= STATE_ISSUE;
                            end
                        end
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
                VERSION_WIDTH < 1 || ARRAY_WEIGHT_ADDR_WIDTH != 11 ||
                ARRAY_TILES < 1 || ARRAY_TILES > 16)
            $error("SRNN H40 INT8 cache refill geometry is invalid");
    end
endmodule

`default_nettype wire
