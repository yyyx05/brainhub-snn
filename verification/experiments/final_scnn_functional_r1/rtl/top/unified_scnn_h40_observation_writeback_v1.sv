`timescale 1ns/1ps
`default_nettype none

// Collects the final ten-lane observations of one SCNN spatial workset and
// commits their spike-domain values to the next feature-map slot and optional
// classifier tap. AAVG2 preserves the exact 0/64/128/192/256 Q8.8 levels while
// the binary feature RAM retains nonzero occupancy for convolution compatibility.
module unified_scnn_h40_observation_writeback_v1 #(
    parameter int WORD_DEPTH = 8192,
    parameter int ADDR_WIDTH = $clog2(WORD_DEPTH),
    parameter int MAX_POOL_WIDTH = 256,
    parameter int PHYSICAL_TILE_COUNT = 16,
    parameter bit ENABLE_FEATURE_COMMIT_TAP = 1'b0
) (
    input  wire logic                       i_clk,
    input  wire logic                       i_rst_n,
    input  wire logic                       i_abort,
    input  wire logic                       i_clear_error,

    input  wire logic                       i_context_valid,
    output logic                            o_context_ready,
    input  wire logic [19:0]                i_layer_tag,
    input  wire logic [9:0]                 i_output_channel_base,
    input  wire logic [9:0]                 i_output_channel_count,
    input  wire logic [31:0]                i_spatial_base,
    input  wire logic [15:0]                i_spatial_row,
    input  wire logic [15:0]                i_spatial_col,
    input  wire logic [5:0]                 i_spatial_count,
    input  wire logic [15:0]                i_out_width,
    input  wire logic [15:0]                i_out_height,
    input  wire logic [1:0]                 i_pool_mode,
    input  wire logic                       i_write_slot,
    input  wire logic [ADDR_WIDTH-1:0]      i_feature_base_addr,
    input  wire logic [15:0]                i_row_stride_words,
    input  wire logic [ADDR_WIDTH-1:0]      i_channel_stride_words,

    input  wire logic [3:0]                 i_observation_valid,
    input  wire logic [3:0]                 i_observation_accept,
    output logic [3:0]                      o_observation_ready,
    input  wire logic [15:0]                i_observation_tile_flat,
    input  wire logic [7:0]                 i_observation_mode_flat,
    input  wire logic [39:0]                i_observation_lane_mask_flat,
    input  wire logic [39:0]                i_observation_spike_flat,
    input  wire logic [255:0]               i_observation_meta_flat,

    input  wire logic                       i_close_valid,
    input  wire logic                       i_close_accept,
    output logic                            o_close_ready,

    output logic                            o_update_valid,
    input  wire logic                       i_update_ready,
    output logic                            o_update_slot,
    output logic [ADDR_WIDTH-1:0]           o_update_addr,
    output logic [63:0]                     o_update_data,
    output logic [63:0]                     o_update_mask,
    input  wire logic                       i_update_done,
    input  wire logic                       i_update_error,

    output logic                            o_feature_commit_valid,
    input  wire logic                       i_feature_commit_ready,
    output logic [19:0]                     o_feature_commit_layer_tag,
    output logic [9:0]                      o_feature_commit_channel,
    output logic [31:0]                     o_feature_commit_spatial_base,
    output logic [5:0]                      o_feature_commit_spatial_count,
    output logic [39:0]                     o_feature_commit_spike,
    output logic [639:0]                    o_feature_commit_q8_8_flat,

    output logic                            o_busy,
    output logic                            o_sticky_error
);
    localparam logic [1:0] MODE_SCNN = 2'd2;
    localparam logic [1:0] POOL_BYPASS = 2'd0;
    localparam logic [1:0] POOL_MAX2 = 2'd1;
    localparam logic [1:0] POOL_AAVG2 = 2'd2;
    // Reserved encoding used by the shared-array Conv1 adapter.  Input
    // activations are replayed as Q8.8 bit-planes, while output writeback is
    // identical to the ordinary unpooled path.
    localparam logic [1:0] POOL_VALUE_BYPASS = 2'd3;
    localparam int POOL_RESULT_LANES = 20;
    localparam int POOL_ROW_BITS = MAX_POOL_WIDTH / 2;
    localparam int POOL_CHUNK_WIDTH = POOL_RESULT_LANES * 2;
    localparam int POOL_CHUNK_COUNT =
        (POOL_ROW_BITS + POOL_RESULT_LANES - 1) / POOL_RESULT_LANES;
    localparam int POOL_CHUNK_ADDR_WIDTH = POOL_CHUNK_COUNT <= 2 ? 1 :
        $clog2(POOL_CHUNK_COUNT);

    typedef enum logic [2:0] {
        IDLE,
        COLLECT,
        WRITE_REQUEST,
        WRITE_WAIT,
        CLOSE_READY
    } state_t;

    state_t state_q;
    logic [19:0] layer_tag_q;
    logic [9:0] output_channel_base_q;
    logic [4:0] output_channel_count_q;
    logic [31:0] spatial_base_q;
    logic [15:0] spatial_row_q;
    logic [15:0] spatial_col_q;
    logic [5:0] spatial_count_q;
    logic [15:0] out_width_q;
    logic [1:0] pool_mode_q;
    logic write_slot_q;
    logic [ADDR_WIDTH-1:0] feature_base_addr_q;
    logic [15:0] row_stride_words_q;
    logic [ADDR_WIDTH-1:0] channel_stride_words_q;
    logic [2:0] group_count_q;
    logic [6:0] expected_observation_count_q;
    logic [6:0] accepted_observation_count_q;
    logic [1:0] observation_region_q;
    logic [39:0] bitmap_q [0:15];
    logic [3:0] seen_group_q [0:15];
    logic [POOL_CHUNK_WIDTH-1:0]
        pool_row_chunk_q [0:15][0:POOL_CHUNK_COUNT-1];
    logic [POOL_CHUNK_COUNT-1:0] pool_chunk_valid_q;
    logic [POOL_CHUNK_ADDR_WIDTH-1:0] pool_chunk_index_q;
    logic [9:0] pool_output_channel_base_q;
    logic [4:0] pool_output_channel_count_q;
    logic [15:0] pool_even_row_q;
    logic [4:0] write_channel_q;
    logic write_second_word_q;
    logic update_store_sent_q;
    logic feature_commit_sent_q;

    logic context_fire;
    logic observation_fire;
    logic update_store_fire;
    logic feature_commit_fire;
    logic write_request_complete;
    logic feature_commit_required;
    logic close_fire;
    logic selected_valid;
    logic [1:0] selected_region;
    logic [3:0] selected_tile;
    logic [1:0] selected_mode;
    logic [9:0] selected_lane_mask;
    logic [9:0] selected_spike;
    logic [63:0] selected_meta;
    logic [19:0] selected_layer_tag;
    logic [31:0] selected_spatial_base;
    logic [9:0] selected_channel;
    logic [1:0] selected_group;
    logic [3:0] selected_channel_offset;
    logic [9:0] expected_lane_mask;
    logic selected_channel_in_range;
    logic selected_duplicate;
    logic selected_observation_legal;
    logic context_legal;
    logic all_observations_collected;
    logic need_second_word;
    logic pool_context_even_row;
    logic pool_pending_match;
    logic context_pooling;
    logic write_pooling;
    logic context_pool_chunk_in_range;
    logic [16*POOL_CHUNK_WIDTH-1:0] pool_capture_chunk_flat;
    logic [16*POOL_CHUNK_WIDTH-1:0] pool_read_chunk_flat;
    logic [POOL_CHUNK_WIDTH-1:0] selected_pool_chunk;

    integer unsigned context_total_spatial;
    integer unsigned context_row;
    integer unsigned context_col;
    integer unsigned required_row_words;
    integer unsigned required_channel_words;
    integer unsigned context_last_word;
    integer unsigned first_word_index;
    integer unsigned second_word_index;
    integer unsigned update_address_integer;
    integer unsigned update_channel_integer;
    integer unsigned output_bit_integer;
    integer unsigned stored_output_width;
    integer unsigned stored_output_height;
    integer unsigned write_context_row;
    integer unsigned write_context_col;
    integer unsigned write_spatial_count;
    integer unsigned write_output_width;
    integer unsigned context_pool_chunk_index;

    assign all_observations_collected =
        accepted_observation_count_q == expected_observation_count_q;

    // The selected region is registered. Keep ready independent of input valid
    // so the array-to-writeback fork cannot form a combinational handshake loop.
    assign o_observation_ready =
        (state_q == COLLECT && !o_sticky_error &&
         !all_observations_collected) ?
            (4'b0001 << observation_region_q) : 4'b0000;

    assign selected_pool_chunk = pool_read_chunk_flat[
        write_channel_q[3:0]*POOL_CHUNK_WIDTH +: POOL_CHUNK_WIDTH];

    always_comb begin
        context_total_spatial = integer'(i_out_width) *
            integer'(i_out_height);
        context_row = integer'(i_spatial_row);
        context_col = integer'(i_spatial_col);
        context_pooling = i_pool_mode == POOL_MAX2 ||
            i_pool_mode == POOL_AAVG2;
        stored_output_width = context_pooling ?
            integer'(i_out_width) / 2 : integer'(i_out_width);
        stored_output_height = context_pooling ?
            integer'(i_out_height) / 2 : integer'(i_out_height);
        required_row_words = (stored_output_width + 63) / 64;
        required_channel_words = integer'(i_row_stride_words) *
            stored_output_height;
        context_last_word = integer'(i_feature_base_addr) +
            (integer'(i_output_channel_base) +
             integer'(i_output_channel_count)) *
                integer'(i_channel_stride_words);
        pool_context_even_row = !context_row[0];
        context_pool_chunk_index = context_col / 40;
        context_pool_chunk_in_range =
            context_pool_chunk_index < POOL_CHUNK_COUNT;
        pool_pending_match =
            pool_output_channel_base_q == i_output_channel_base &&
            pool_output_channel_count_q == 5'(i_output_channel_count);
        context_legal = i_output_channel_count != 0 &&
            i_output_channel_count <= 10'd16 && i_spatial_count != 0 &&
            i_spatial_count <= 6'd40 && i_out_width != 0 &&
            i_out_height != 0 && integer'(i_spatial_base) <
                context_total_spatial && i_spatial_row < i_out_height &&
            i_spatial_col < i_out_width &&
            context_col + integer'(i_spatial_count) <=
                integer'(i_out_width) &&
            integer'(i_row_stride_words) >= required_row_words &&
            integer'(i_channel_stride_words) >= required_channel_words &&
            context_last_word <= WORD_DEPTH &&
            (i_pool_mode == POOL_BYPASS ||
             i_pool_mode == POOL_MAX2 ||
             i_pool_mode == POOL_AAVG2 ||
             i_pool_mode == POOL_VALUE_BYPASS);
        if (context_pooling) begin
            context_legal = context_legal &&
                i_out_width <= 16'(MAX_POOL_WIDTH) &&
                !i_out_width[0] && !i_out_height[0] &&
                context_col % 40 == 0 && !i_spatial_count[0] &&
                context_pool_chunk_in_range;
            if (!context_pool_chunk_in_range)
                context_legal = 1'b0;
            else if (pool_context_even_row)
                context_legal = context_legal &&
                    !pool_chunk_valid_q[context_pool_chunk_index] &&
                    (pool_chunk_valid_q == '0 ||
                     (pool_pending_match &&
                      pool_even_row_q == 16'(context_row)));
            else
                context_legal = context_legal && pool_pending_match &&
                    pool_even_row_q + 1'b1 == 16'(context_row) &&
                    pool_chunk_valid_q[context_pool_chunk_index];
        end

        o_context_ready = state_q == IDLE && !o_sticky_error;
        context_fire = i_context_valid && o_context_ready;
        o_busy = state_q != IDLE;

        selected_region = observation_region_q;
        selected_valid = i_observation_valid[observation_region_q];
        observation_fire = selected_valid &&
            o_observation_ready[selected_region] &&
            i_observation_accept[selected_region];
        selected_tile = i_observation_tile_flat[selected_region*4 +: 4];
        selected_mode = i_observation_mode_flat[selected_region*2 +: 2];
        selected_lane_mask =
            i_observation_lane_mask_flat[selected_region*10 +: 10];
        selected_spike =
            i_observation_spike_flat[selected_region*10 +: 10];
        selected_meta =
            i_observation_meta_flat[selected_region*64 +: 64];
        selected_layer_tag = selected_meta[63:44];
        selected_spatial_base = selected_meta[43:12];
        selected_channel = selected_meta[11:2];
        selected_group = selected_meta[1:0];
        selected_channel_offset = 4'(selected_channel -
            output_channel_base_q);
        selected_channel_in_range =
            selected_channel >= output_channel_base_q &&
            {1'b0, selected_channel} <
                ({1'b0, output_channel_base_q} +
                 11'(output_channel_count_q));
        selected_duplicate = 1'b1;
        if (selected_channel_in_range)
            selected_duplicate =
                seen_group_q[selected_channel_offset][selected_group];

        expected_lane_mask = '0;
        for (integer lane = 0; lane < 10; lane++)
            if (integer'(selected_group) * 10 + lane <
                integer'(spatial_count_q))
                expected_lane_mask[lane] = 1'b1;
        selected_observation_legal = selected_mode == MODE_SCNN &&
            selected_layer_tag == layer_tag_q &&
            selected_spatial_base == spatial_base_q &&
            selected_channel_in_range &&
            3'(selected_group) < group_count_q &&
            selected_tile == 4'(
                integer'(selected_channel) % PHYSICAL_TILE_COUNT) &&
            selected_tile[3:2] == selected_region &&
            selected_lane_mask == expected_lane_mask &&
            (selected_spike & ~selected_lane_mask) == 0 &&
            !selected_duplicate;

        context_row = integer'(spatial_row_q);
        context_col = integer'(spatial_col_q);
        write_pooling = pool_mode_q == POOL_MAX2 ||
            pool_mode_q == POOL_AAVG2;
        write_context_row = write_pooling ?
            context_row / 2 : context_row;
        write_context_col = write_pooling ?
            context_col / 2 : context_col;
        write_spatial_count = write_pooling ?
            integer'(spatial_count_q) / 2 : integer'(spatial_count_q);
        write_output_width = write_pooling ?
            integer'(out_width_q) / 2 : integer'(out_width_q);
        pool_capture_chunk_flat = '0;
        for (integer channel = 0; channel < 16; channel++)
            for (integer lane = 0; lane < POOL_RESULT_LANES; lane++)
                if (channel < output_channel_count_q &&
                    lane < write_spatial_count)
                    pool_capture_chunk_flat[
                        channel*POOL_CHUNK_WIDTH + lane*2 +: 2] =
                            {1'b0, bitmap_q[channel][lane*2]} +
                            {1'b0, bitmap_q[channel][lane*2 + 1]};
        first_word_index = write_context_col / 64;
        second_word_index = (write_context_col +
            write_spatial_count - 1) / 64;
        need_second_word = first_word_index != second_word_index;
        update_channel_integer = integer'(output_channel_base_q) +
            integer'(write_channel_q);
        update_address_integer = integer'(feature_base_addr_q) +
            update_channel_integer * integer'(channel_stride_words_q) +
            write_context_row * integer'(row_stride_words_q) +
            (write_second_word_q ? second_word_index : first_word_index);

        feature_commit_required = ENABLE_FEATURE_COMMIT_TAP &&
            !write_second_word_q;
        o_update_valid = state_q == WRITE_REQUEST && !o_sticky_error &&
            !update_store_sent_q;
        o_update_slot = write_slot_q;
        o_update_addr = ADDR_WIDTH'(update_address_integer);
        o_update_data = '0;
        o_update_mask = '0;
        if (write_pooling) begin
            for (integer lane = 0; lane < POOL_RESULT_LANES; lane++) begin
                output_bit_integer = write_context_col + lane;
                if (lane < write_spatial_count &&
                    (output_bit_integer / 64) ==
                        (write_second_word_q ? second_word_index :
                         first_word_index)) begin
                    o_update_mask[output_bit_integer % 64] = 1'b1;
                    o_update_data[output_bit_integer % 64] =
                        (selected_pool_chunk[lane*2 +: 2] +
                         bitmap_q[write_channel_q[3:0]][lane*2] +
                         bitmap_q[write_channel_q[3:0]][lane*2 + 1]) != 0;
                end
            end
        end else begin
            for (integer lane = 0; lane < 40; lane++) begin
                output_bit_integer = write_context_col + lane;
                if (lane < write_spatial_count &&
                    (output_bit_integer / 64) ==
                        (write_second_word_q ? second_word_index :
                         first_word_index)) begin
                    o_update_mask[output_bit_integer % 64] = 1'b1;
                    o_update_data[output_bit_integer % 64] =
                        bitmap_q[write_channel_q[3:0]][lane];
                end
            end
        end
        o_feature_commit_valid = state_q == WRITE_REQUEST &&
            !o_sticky_error && feature_commit_required &&
            !feature_commit_sent_q;
        o_feature_commit_layer_tag = layer_tag_q;
        o_feature_commit_channel = 10'(update_channel_integer);
        o_feature_commit_spatial_base = 32'(
            write_context_row * write_output_width + write_context_col);
        o_feature_commit_spatial_count = 6'(write_spatial_count);
        o_feature_commit_spike = '0;
        o_feature_commit_q8_8_flat = '0;
        if (write_pooling) begin
            for (integer lane = 0; lane < POOL_RESULT_LANES; lane++)
                if (lane < write_spatial_count) begin
                    o_feature_commit_spike[lane] =
                        (selected_pool_chunk[lane*2 +: 2] +
                         bitmap_q[write_channel_q[3:0]][lane*2] +
                         bitmap_q[write_channel_q[3:0]][lane*2 + 1]) != 0;
                    if (pool_mode_q == POOL_AAVG2)
                        o_feature_commit_q8_8_flat[lane*16 +: 16] =
                            16'((integer'(selected_pool_chunk[
                                lane*2 +: 2]) +
                                 integer'(bitmap_q[
                                    write_channel_q[3:0]][lane*2]) +
                                 integer'(bitmap_q[
                                    write_channel_q[3:0]][lane*2 + 1])) * 64);
                    else
                        o_feature_commit_q8_8_flat[lane*16 +: 16] =
                            o_feature_commit_spike[lane] ? 16'd256 : 16'd0;
                end
        end else begin
            for (integer lane = 0; lane < 40; lane++)
                if (lane < write_spatial_count) begin
                    o_feature_commit_spike[lane] =
                        bitmap_q[write_channel_q[3:0]][lane];
                    o_feature_commit_q8_8_flat[lane*16 +: 16] =
                        bitmap_q[write_channel_q[3:0]][lane] ?
                            16'd256 : 16'd0;
                end
        end
        update_store_fire = o_update_valid && i_update_ready;
        feature_commit_fire = o_feature_commit_valid &&
            i_feature_commit_ready;
        write_request_complete =
            (update_store_sent_q || update_store_fire) &&
            (!feature_commit_required || feature_commit_sent_q ||
             feature_commit_fire);
        o_close_ready = state_q == CLOSE_READY && !o_sticky_error;
        close_fire = i_close_accept && o_close_ready;
    end

    // H40 pooling contexts are aligned to 40 input positions, so one context
    // produces at most one fixed-width chunk of 20 two-bit row counts. Keeping
    // one small, reset-free bank per output channel avoids a 16x128 arbitrary
    // bit-addressed register array and its large read/write mux fabric.
    for (genvar channel = 0; channel < 16; channel++) begin : g_pool_bank
        assign pool_read_chunk_flat[
            channel*POOL_CHUNK_WIDTH +: POOL_CHUNK_WIDTH] =
                pool_row_chunk_q[channel][pool_chunk_index_q];

        always_ff @(posedge i_clk) begin
            if (i_rst_n && !i_abort && state_q == COLLECT &&
                i_close_valid && all_observations_collected &&
                write_pooling && !context_row[0] &&
                channel < output_channel_count_q)
                pool_row_chunk_q[channel][pool_chunk_index_q] <=
                    pool_capture_chunk_flat[
                        channel*POOL_CHUNK_WIDTH +: POOL_CHUNK_WIDTH];
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= IDLE;
            layer_tag_q <= '0;
            output_channel_base_q <= '0;
            output_channel_count_q <= '0;
            spatial_base_q <= '0;
            spatial_row_q <= '0;
            spatial_col_q <= '0;
            spatial_count_q <= '0;
            out_width_q <= '0;
            pool_mode_q <= POOL_BYPASS;
            write_slot_q <= 1'b0;
            feature_base_addr_q <= '0;
            row_stride_words_q <= '0;
            channel_stride_words_q <= '0;
            group_count_q <= '0;
            expected_observation_count_q <= '0;
            accepted_observation_count_q <= '0;
            observation_region_q <= '0;
            pool_chunk_valid_q <= '0;
            pool_chunk_index_q <= '0;
            pool_output_channel_base_q <= '0;
            pool_output_channel_count_q <= '0;
            pool_even_row_q <= '0;
            write_channel_q <= '0;
            write_second_word_q <= 1'b0;
            update_store_sent_q <= 1'b0;
            feature_commit_sent_q <= 1'b0;
            for (integer channel = 0; channel < 16; channel++) begin
                bitmap_q[channel] <= '0;
                seen_group_q[channel] <= '0;
            end
            o_sticky_error <= 1'b0;
        end else begin
            if (i_abort) begin
                state_q <= IDLE;
                pool_chunk_valid_q <= '0;
            end else begin
                if (i_clear_error && state_q == IDLE) begin
                    o_sticky_error <= 1'b0;
                    pool_chunk_valid_q <= '0;
                end
                if (i_update_error && state_q != IDLE) begin
                    state_q <= IDLE;
                    o_sticky_error <= 1'b1;
                end else begin
                    case (state_q)
                        IDLE: begin
                            if (context_fire) begin
                                if (!context_legal) begin
                                    o_sticky_error <= 1'b1;
                                end else begin
                                    layer_tag_q <= i_layer_tag;
                                    output_channel_base_q <=
                                        i_output_channel_base;
                                    output_channel_count_q <=
                                        5'(i_output_channel_count);
                                    spatial_base_q <= i_spatial_base;
                                    spatial_row_q <= i_spatial_row;
                                    spatial_col_q <= i_spatial_col;
                                    spatial_count_q <= i_spatial_count;
                                    out_width_q <= i_out_width;
                                    pool_mode_q <= i_pool_mode;
                                    pool_chunk_index_q <=
                                        POOL_CHUNK_ADDR_WIDTH'(
                                            context_pool_chunk_index);
                                    write_slot_q <= i_write_slot;
                                    feature_base_addr_q <=
                                        i_feature_base_addr;
                                    row_stride_words_q <=
                                        i_row_stride_words;
                                    channel_stride_words_q <=
                                        i_channel_stride_words;
                                    group_count_q <=
                                        3'((integer'(i_spatial_count) + 9) /
                                           10);
                                    expected_observation_count_q <=
                                        7'(integer'(i_output_channel_count) *
                                           ((integer'(i_spatial_count) + 9) /
                                            10));
                                    accepted_observation_count_q <= '0;
                                    observation_region_q <= '0;
                                    write_channel_q <= '0;
                                    write_second_word_q <= 1'b0;
                                    update_store_sent_q <= 1'b0;
                                    feature_commit_sent_q <= 1'b0;
                                    for (integer channel = 0;
                                         channel < 16; channel++) begin
                                        bitmap_q[channel] <= '0;
                                        seen_group_q[channel] <= '0;
                                    end
                                    state_q <= COLLECT;
                                end
                            end
                        end
                        COLLECT: begin
                            if ((!selected_valid || observation_fire) &&
                                !all_observations_collected)
                                observation_region_q <=
                                    observation_region_q + 1'b1;
                            if (observation_fire) begin
                                if (!selected_observation_legal) begin
                                    state_q <= IDLE;
                                    o_sticky_error <= 1'b1;
                                end else begin
                                    bitmap_q[selected_channel_offset][
                                        selected_group*10 +: 10] <=
                                            selected_spike &
                                            selected_lane_mask;
                                    seen_group_q[selected_channel_offset][
                                        selected_group] <= 1'b1;
                                    accepted_observation_count_q <=
                                        accepted_observation_count_q + 1'b1;
                                end
                            end
                            if (i_close_valid && all_observations_collected) begin
                                write_channel_q <= '0;
                                write_second_word_q <= 1'b0;
                                if (write_pooling &&
                                    !context_row[0]) begin
                                    pool_chunk_valid_q[
                                        pool_chunk_index_q] <= 1'b1;
                                    if (pool_chunk_valid_q == '0) begin
                                        pool_output_channel_base_q <=
                                            output_channel_base_q;
                                        pool_output_channel_count_q <=
                                            output_channel_count_q;
                                        pool_even_row_q <= 16'(context_row);
                                    end
                                    state_q <= CLOSE_READY;
                                end else begin
                                    update_store_sent_q <= 1'b0;
                                    feature_commit_sent_q <= 1'b0;
                                    state_q <= WRITE_REQUEST;
                                end
                            end
                        end
                        WRITE_REQUEST: begin
                            if (update_store_fire)
                                update_store_sent_q <= 1'b1;
                            if (feature_commit_fire)
                                feature_commit_sent_q <= 1'b1;
                            if (write_request_complete)
                                state_q <= WRITE_WAIT;
                        end
                        WRITE_WAIT: begin
                            if (i_update_done) begin
                                if (need_second_word &&
                                    !write_second_word_q) begin
                                    write_second_word_q <= 1'b1;
                                    update_store_sent_q <= 1'b0;
                                    feature_commit_sent_q <= 1'b0;
                                    state_q <= WRITE_REQUEST;
                                end else if (write_channel_q + 1'b1 <
                                             output_channel_count_q) begin
                                    write_channel_q <=
                                        write_channel_q + 1'b1;
                                    write_second_word_q <= 1'b0;
                                    update_store_sent_q <= 1'b0;
                                    feature_commit_sent_q <= 1'b0;
                                    state_q <= WRITE_REQUEST;
                                end else begin
                                    if (write_pooling)
                                        pool_chunk_valid_q[
                                            pool_chunk_index_q] <= 1'b0;
                                    state_q <= CLOSE_READY;
                                end
                            end
                        end
                        CLOSE_READY: begin
                            if (close_fire)
                                state_q <= IDLE;
                        end
                        default: begin
                            state_q <= IDLE;
                            o_sticky_error <= 1'b1;
                        end
                    endcase
                end
            end
        end
    end

    initial begin
        if (WORD_DEPTH < 2 || ADDR_WIDTH != $clog2(WORD_DEPTH) ||
            MAX_POOL_WIDTH < 2 || MAX_POOL_WIDTH[0] ||
            PHYSICAL_TILE_COUNT < 1 || PHYSICAL_TILE_COUNT > 16)
            $error("SCNN observation writeback parameter mismatch");
    end
endmodule

`default_nettype wire
