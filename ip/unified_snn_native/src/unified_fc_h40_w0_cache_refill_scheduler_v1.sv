`timescale 1ns/1ps
`default_nettype none

// Maps feature-major FC parameters onto the shared H40/P4 cache. A full
// command primes every active feature/tile row; a targeted command republishes
// one dirty row after online learning without copying an FC-only weight store.
module unified_fc_h40_w0_cache_refill_scheduler_v1 #(
    parameter int PARAM_DEPTH = 120320,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int VERSION_WIDTH = 16,
    parameter int FEATURE_COUNT_WIDTH = 12,
    parameter int HIDDEN_COUNT_WIDTH = 10,
    parameter int ARRAY_WEIGHT_ADDR_WIDTH = 11,
    parameter int ARRAY_TILES = 16
) (
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,
    input  wire logic                              i_abort,
    input  wire logic                              i_clear_error,

    input  wire logic                              i_refill_valid,
    output logic                                   o_refill_ready,
    input  wire logic                              i_refill_all,
    input  wire logic                              i_fc_weight_int8,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0]     i_dirty_feature,
    input  wire logic [3:0]                        i_dirty_tile,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_parameter_base,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_hidden_stride,
    input  wire logic [9:0]                         i_cache_row_base,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0]     i_feature_count,
    input  wire logic [HIDDEN_COUNT_WIDTH-1:0]      i_hidden_count,
    input  wire logic [4:0]                        i_tile_count,
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
        FEATURE_COUNT_WIDTH + 1;

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_ISSUE,
        STATE_WAIT
    } state_t;

    state_t state_q;
    logic refill_all_q;
    logic fc_weight_int8_q;
    logic [FEATURE_COUNT_WIDTH-1:0] feature_count_q;
    logic [HIDDEN_COUNT_WIDTH-1:0] hidden_count_q;
    logic [4:0] tile_count_q;
    logic [FEATURE_COUNT_WIDTH-1:0] feature_q;
    logic [3:0] tile_q;
    logic [3:0] destination_tile_base_q;
    logic [PARAM_ADDR_WIDTH-1:0] hidden_stride_q;
    logic [PARAM_ADDR_WIDTH-1:0] feature_base_q;
    logic [PARAM_ADDR_WIDTH-1:0] row_scalar_base_q;
    logic [9:0] cache_row_base_q;
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
    logic last_tile;
    logic last_feature;
    logic [HIDDEN_COUNT_WIDTH:0] tile_hidden_base;
    logic [HIDDEN_COUNT_WIDTH:0] remaining_hidden;
    logic [5:0] row_scalar_count;
    logic [RANGE_WIDTH-1:0] matrix_end_ext;
    logic [RANGE_WIDTH-1:0] target_base_ext;
    logic [RANGE_WIDTH-1:0] dirty_feature_offset_ext;
    logic [10:0] cache_last_row_ext;
    logic [9:0] target_row;
    logic hidden_geometry_legal;
    logic [3:0] destination_tile_base;
    logic [5:0] destination_tile_end;
    logic [3:0] physical_tile;

    always_comb begin
        // Legacy omitted/X/Z bits map to zero without hiding known-one bits.
        for (int bit_index = 0; bit_index < 4; bit_index++)
            destination_tile_base[bit_index] =
                i_destination_tile_base[bit_index] === 1'b1;
        destination_tile_end = {2'b0, destination_tile_base} + {1'b0, i_tile_count};
        // tile_q and dirty-tile parameter offsets remain local to the matrix.
        physical_tile = destination_tile_base_q + tile_q;
        matrix_end_ext = RANGE_WIDTH'(i_parameter_base) +
            RANGE_WIDTH'(i_feature_count - 1'b1) *
                RANGE_WIDTH'(i_hidden_stride) +
            RANGE_WIDTH'(i_hidden_count);
        dirty_feature_offset_ext = RANGE_WIDTH'(i_dirty_feature) *
            RANGE_WIDTH'(i_hidden_stride);
        target_base_ext = RANGE_WIDTH'(i_parameter_base) +
            dirty_feature_offset_ext +
            RANGE_WIDTH'(i_dirty_tile) * RANGE_WIDTH'(40);
        cache_last_row_ext = {1'b0, i_cache_row_base} +
            11'((RANGE_WIDTH'(i_feature_count) - RANGE_WIDTH'(1)) >> 2);
        target_row = cache_row_base_q + 10'(feature_q >> 2);
        hidden_geometry_legal = i_tile_count != 0 &&
            i_tile_count <= 5'(ARRAY_TILES) && i_hidden_count != 0 &&
            HIDDEN_COUNT_WIDTH'(i_tile_count - 1'b1) *
                HIDDEN_COUNT_WIDTH'(40) < i_hidden_count &&
            i_hidden_count <= HIDDEN_COUNT_WIDTH'(i_tile_count) *
                HIDDEN_COUNT_WIDTH'(40);
        // INT8 W2 uses one 200-bit row for up to 25 classes, not 40 INT5 lanes.
        if (i_fc_weight_int8 === 1'b1)
            hidden_geometry_legal = i_tile_count == 5'd1 &&
                i_hidden_count != 0 && i_hidden_count <= HIDDEN_COUNT_WIDTH'(25);
        request_legal = i_feature_count != 0 &&
            RANGE_WIDTH'(i_feature_count) <= RANGE_WIDTH'(2312) &&
            hidden_geometry_legal &&
            destination_tile_end <= 6'(ARRAY_TILES) &&
            i_hidden_stride >= PARAM_ADDR_WIDTH'(i_hidden_count) &&
            matrix_end_ext <= RANGE_WIDTH'(PARAM_DEPTH) &&
            cache_last_row_ext < 11'd578 &&
            (i_refill_all ||
             (i_dirty_feature < i_feature_count &&
              5'(i_dirty_tile) < i_tile_count &&
              target_base_ext < RANGE_WIDTH'(PARAM_DEPTH)));

        tile_hidden_base = (HIDDEN_COUNT_WIDTH + 1)'(tile_q) *
            (HIDDEN_COUNT_WIDTH + 1)'(40);
        remaining_hidden =
            (HIDDEN_COUNT_WIDTH + 1)'(hidden_count_q) -
            tile_hidden_base;
        row_scalar_count = remaining_hidden >=
            (HIDDEN_COUNT_WIDTH + 1)'(40) ? 6'd40 :
            6'(remaining_hidden);

        last_tile = tile_q + 1'b1 == tile_count_q;
        last_feature = feature_q + 1'b1 == feature_count_q;
    end

    assign o_refill_ready = state_q == STATE_IDLE && !row_busy &&
        !o_sticky_error && i_array_idle;
    assign request_fire = i_refill_valid && o_refill_ready;
    assign row_command_valid = state_q == STATE_ISSUE && !row_error;
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
        .i_target_tile(physical_tile), .i_target_bank(feature_q[1:0]),
        .i_target_row(target_row),
        .i_scalar_base(row_scalar_base_q),
        .i_scalar_stride(PARAM_ADDR_WIDTH'(1)),
        .i_scalar_count(row_scalar_count),
        .i_element_format(fc_weight_int8_q ? 2'd1 : 2'd0),
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
            fc_weight_int8_q <= 1'b0;
            feature_count_q <= '0;
            hidden_count_q <= '0;
            tile_count_q <= '0;
            feature_q <= '0;
            tile_q <= '0;
            destination_tile_base_q <= '0;
            hidden_stride_q <= '0;
            feature_base_q <= '0;
            row_scalar_base_q <= '0;
            cache_row_base_q <= '0;
            version_q <= '0;
            o_refill_done <= 1'b0;
            o_rows_published <= '0;
            o_published_version <= '0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            o_refill_done <= 1'b0;
        end else begin
            o_refill_done <= 1'b0;
            if (i_clear_error && state_q == STATE_IDLE && !row_busy)
                o_sticky_error <= 1'b0;
            else if (row_error)
                o_sticky_error <= 1'b1;

            unique case (state_q)
                STATE_IDLE: begin
                    if (request_fire) begin
                        if (!request_legal) begin
                            o_sticky_error <= 1'b1;
                        end else begin
                            refill_all_q <= i_refill_all;
                            fc_weight_int8_q <= i_fc_weight_int8 === 1'b1;
                            feature_count_q <= i_feature_count;
                            hidden_count_q <= i_hidden_count;
                            tile_count_q <= i_tile_count;
                            destination_tile_base_q <= destination_tile_base;
                            hidden_stride_q <= i_hidden_stride;
                            cache_row_base_q <= i_cache_row_base;
                            version_q <= i_version;
                            o_rows_published <= '0;
                            if (i_refill_all) begin
                                feature_q <= '0;
                                tile_q <= '0;
                                feature_base_q <= i_parameter_base;
                                row_scalar_base_q <= i_parameter_base;
                            end else begin
                                feature_q <= i_dirty_feature;
                                tile_q <= i_dirty_tile;
                                feature_base_q <= PARAM_ADDR_WIDTH'(
                                    RANGE_WIDTH'(i_parameter_base) +
                                    dirty_feature_offset_ext);
                                row_scalar_base_q <= PARAM_ADDR_WIDTH'(
                                    target_base_ext);
                            end
                            state_q <= STATE_ISSUE;
                        end
                    end
                end

                STATE_ISSUE: begin
                    if (row_command_fire)
                        state_q <= STATE_WAIT;
                end

                STATE_WAIT: begin
                    if (row_done) begin
                        if (row_published_tile != physical_tile ||
                                row_published_bank != feature_q[1:0] ||
                                row_published_row != target_row ||
                                row_published_version != version_q) begin
                            o_sticky_error <= 1'b1;
                            state_q <= STATE_IDLE;
                        end else begin
                            o_rows_published <= o_rows_published + 1'b1;
                            if (!refill_all_q ||
                                    (last_tile && last_feature)) begin
                                o_refill_done <= 1'b1;
                                o_published_version <= version_q;
                                state_q <= STATE_IDLE;
                            end else if (!last_tile) begin
                                tile_q <= tile_q + 1'b1;
                                row_scalar_base_q <= row_scalar_base_q +
                                    PARAM_ADDR_WIDTH'(40);
                                state_q <= STATE_ISSUE;
                            end else begin
                                feature_q <= feature_q + 1'b1;
                                tile_q <= '0;
                                feature_base_q <= feature_base_q +
                                    hidden_stride_q;
                                row_scalar_base_q <= feature_base_q +
                                    hidden_stride_q;
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
            // A rejected scalar aborts its publisher without producing row_done.
            // Leave WAIT so an explicit clear can recover without a global reset.
            if (row_error)
                state_q <= STATE_IDLE;
        end
    end

    initial begin
        if (PARAM_DEPTH < 1 || PARAM_ADDR_WIDTH < 1 ||
                FEATURE_COUNT_WIDTH < 2 || HIDDEN_COUNT_WIDTH < 7 ||
                ARRAY_WEIGHT_ADDR_WIDTH != 11 || ARRAY_TILES < 1 ||
                ARRAY_TILES > 16)
            $error("FC H40 W0 refill scheduler parameter mismatch");
    end
endmodule

`default_nettype wire
