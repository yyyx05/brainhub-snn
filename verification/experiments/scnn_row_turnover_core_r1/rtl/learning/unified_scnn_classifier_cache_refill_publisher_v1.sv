`timescale 1ns/1ps
`default_nettype none

// Repackages the persistent class-major Q8.8 classifier matrix into the
// tile-local SCNN cache layout. One class owns one tile; even and odd feature
// indices occupy the two banks of the selected pair, with twelve Q8.8 values
// packed into each 200-bit row. Publication is legal only while the array is
// idle, so a completed pair can be selected atomically by the runtime owner.
module unified_scnn_classifier_cache_refill_publisher_v1 #(
    parameter int MAX_FEATURES = 2048,
    parameter int MAX_CLASSES = 16,
    parameter int PARAM_DEPTH = MAX_FEATURES * MAX_CLASSES,
    parameter int PARAM_ADDR_WIDTH = $clog2(PARAM_DEPTH),
    parameter int TAG_WIDTH = 16,
    parameter int VERSION_WIDTH = 16,
    parameter int ARRAY_WEIGHT_ADDR_WIDTH = 11,
    parameter int FEATURE_COUNT_WIDTH = $clog2(MAX_FEATURES + 1),
    parameter int CLASS_COUNT_WIDTH = $clog2(MAX_CLASSES + 1)
) (
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,
    input  wire logic                              i_abort,
    input  wire logic                              i_clear_error,

    input  wire logic                              i_start_valid,
    output logic                                   o_start_ready,
    input  wire logic                              i_target_pair,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_parameter_base,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0]     i_feature_count,
    input  wire logic [CLASS_COUNT_WIDTH-1:0]       i_class_count,
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
    output logic                                   o_published_pair,
    output logic [VERSION_WIDTH-1:0]               o_published_version,
    output logic                                   o_busy,
    output logic                                   o_sticky_error
);
    localparam int ROW_SUBWORDS = 12;
    localparam int FEATURES_PER_ROW = ROW_SUBWORDS * 2;
    localparam int BANK_DEPTH = 578;
    localparam int BANK_ADDR_WIDTH = 10;
    localparam logic [PARAM_ADDR_WIDTH:0] PARAM_DEPTH_EXT =
        (PARAM_ADDR_WIDTH + 1)'(PARAM_DEPTH);

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_COMMAND,
        STATE_GATHER,
        STATE_READ_REQUEST,
        STATE_READ_RESPONSE,
        STATE_SEND_ROW,
        STATE_WAIT_COMPLETE
    } state_t;

    state_t state_q;
    logic target_pair_q;
    logic [PARAM_ADDR_WIDTH-1:0] class_parameter_base_q;
    logic [FEATURE_COUNT_WIDTH-1:0] feature_count_q;
    logic [CLASS_COUNT_WIDTH-1:0] class_count_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic [3:0] class_q;
    logic bank_parity_q;
    logic [BANK_ADDR_WIDTH-1:0] row_q;
    logic [BANK_ADDR_WIDTH:0] rows_per_bank_q;
    logic [3:0] subword_q;
    logic [199:0] row_data_q;
    logic [TAG_WIDTH-1:0] read_tag_q;

    logic refill_command_valid;
    logic refill_command_ready;
    logic refill_row_valid;
    logic refill_row_ready;
    logic refill_complete_valid;
    logic refill_complete_ready;
    logic refill_busy;
    logic refill_error;
    logic unsafe_refill;
    logic publisher_abort;
    logic start_fire;
    logic read_fire;
    logic response_fire;
    logic turnover_valid;
    logic row_turnover_valid;
    logic row_fire;
    logic complete_fire;
    logic start_legal;
    logic [PARAM_ADDR_WIDTH:0] parameter_end;
    integer unsigned feature_index_int;
    integer unsigned parameter_addr_int;

    always_comb begin
        feature_index_int = integer'(row_q) * FEATURES_PER_ROW +
            integer'(subword_q) * 2 + integer'(bank_parity_q);
        parameter_addr_int = integer'(class_parameter_base_q) +
            feature_index_int;
        parameter_end = {1'b0, i_parameter_base} +
            (PARAM_ADDR_WIDTH + 1)'(
                integer'(i_feature_count) * integer'(i_class_count));
        start_legal = i_feature_count != 0 &&
            i_feature_count <= FEATURE_COUNT_WIDTH'(MAX_FEATURES) &&
            i_class_count != 0 &&
            i_class_count <= CLASS_COUNT_WIDTH'(MAX_CLASSES) &&
            ((integer'(i_feature_count) + FEATURES_PER_ROW - 1) /
                FEATURES_PER_ROW) <= BANK_DEPTH &&
            parameter_end <= PARAM_DEPTH_EXT;
    end

    assign unsafe_refill = state_q != STATE_IDLE && !i_array_idle;
    assign publisher_abort = i_abort || unsafe_refill;
    assign o_start_ready = state_q == STATE_IDLE && !refill_busy &&
        !o_sticky_error && i_array_idle;
    assign start_fire = i_start_valid && o_start_ready;
    assign o_busy = state_q != STATE_IDLE || refill_busy;

    assign refill_command_valid = state_q == STATE_COMMAND;
    assign refill_row_valid = state_q == STATE_SEND_ROW;
    assign refill_complete_ready = state_q == STATE_WAIT_COMPLETE;
    assign read_fire = o_forward_read_valid && i_forward_read_ready;
    assign response_fire = i_forward_response_valid &&
        o_forward_response_ready;
    assign row_fire = refill_row_valid && refill_row_ready;
    assign complete_fire = refill_complete_valid && refill_complete_ready;

    // One response retires before its replacement becomes outstanding.
    assign turnover_valid = state_q == STATE_READ_RESPONSE &&
        i_forward_response_valid && !i_forward_response_error &&
        i_forward_response_tag == read_tag_q && !publisher_abort &&
        subword_q < ROW_SUBWORDS - 1 &&
        feature_index_int + 2 < integer'(feature_count_q);

    // Reuse the row buffer only when its previous contents retire.
    // The replacement response cannot arrive before the following cycle.
    assign row_turnover_valid = row_fire && !publisher_abort &&
        row_q + 1'b1 < rows_per_bank_q &&
        (integer'(row_q) + 1) * FEATURES_PER_ROW +
            integer'(bank_parity_q) < integer'(feature_count_q);

    always_comb begin
        // Issue the current valid scalar directly; address/tag hold under backpressure.
        o_forward_read_valid = state_q == STATE_GATHER &&
            feature_index_int < integer'(feature_count_q) ||
            turnover_valid || row_turnover_valid;
        o_forward_read_addr = PARAM_ADDR_WIDTH'(parameter_addr_int);
        o_forward_read_tag = read_tag_q;
        if (turnover_valid) begin
            o_forward_read_addr = PARAM_ADDR_WIDTH'(parameter_addr_int + 2);
            o_forward_read_tag = read_tag_q + 1'b1;
        end else if (row_turnover_valid) begin
            o_forward_read_addr = PARAM_ADDR_WIDTH'(
                integer'(class_parameter_base_q) +
                (integer'(row_q) + 1) * FEATURES_PER_ROW +
                integer'(bank_parity_q));
        end
        o_forward_response_ready = state_q == STATE_READ_RESPONSE;
    end

    unified_array16_weight_refill_dispatcher_v1 #(
        .BANK_DEPTH(BANK_DEPTH),
        .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH),
        .ARRAY_ADDR_WIDTH(ARRAY_WEIGHT_ADDR_WIDTH),
        .TAG_WIDTH(VERSION_WIDTH)
    ) u_refill_dispatcher (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_abort(publisher_abort), .i_clear_error(i_clear_error),
        .i_command_valid(refill_command_valid),
        .o_command_ready(refill_command_ready),
        .i_command_tile(class_q),
        .i_command_bank({target_pair_q, bank_parity_q}),
        .i_command_base_addr('0), .i_command_rows(rows_per_bank_q),
        .i_command_tag(version_q),
        .i_row_valid(refill_row_valid), .o_row_ready(refill_row_ready),
        .i_row_data(row_data_q),
        .o_fill_valid(o_weight_fill_valid),
        .i_fill_ready(i_weight_fill_ready),
        .o_fill_tile_flat(o_weight_fill_tile_flat),
        .o_fill_bank(o_weight_fill_bank),
        .o_fill_addr_flat(o_weight_fill_addr_flat),
        .o_fill_data_flat(o_weight_fill_data_flat),
        .o_complete_valid(refill_complete_valid),
        .i_complete_ready(refill_complete_ready),
        .o_complete_tile(), .o_complete_bank(), .o_complete_tag(),
        .o_busy(refill_busy), .o_sticky_error(refill_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            target_pair_q <= 1'b0;
            class_parameter_base_q <= '0;
            feature_count_q <= '0;
            class_count_q <= '0;
            version_q <= '0;
            class_q <= '0;
            bank_parity_q <= 1'b0;
            row_q <= '0;
            rows_per_bank_q <= '0;
            subword_q <= '0;
            row_data_q <= '0;
            read_tag_q <= '0;
            o_done <= 1'b0;
            o_published_pair <= 1'b0;
            o_published_version <= '0;
            o_sticky_error <= 1'b0;
        end else if (publisher_abort) begin
            state_q <= STATE_IDLE;
            row_data_q <= '0;
            o_done <= 1'b0;
            if (unsafe_refill)
                o_sticky_error <= 1'b1;
        end else begin
            o_done <= 1'b0;
            if (i_clear_error && state_q == STATE_IDLE && !refill_busy)
                o_sticky_error <= 1'b0;
            if (refill_error)
                o_sticky_error <= 1'b1;

            case (state_q)
                STATE_IDLE: begin
                    if (start_fire) begin
                        if (!start_legal) begin
                            o_sticky_error <= 1'b1;
                        end else begin
                            target_pair_q <= i_target_pair;
                            class_parameter_base_q <= i_parameter_base;
                            feature_count_q <= i_feature_count;
                            class_count_q <= i_class_count;
                            version_q <= i_version;
                            class_q <= '0;
                            bank_parity_q <= 1'b0;
                            row_q <= '0;
                            rows_per_bank_q <= BANK_ADDR_WIDTH'(
                                (integer'(i_feature_count) +
                                 FEATURES_PER_ROW - 1) /
                                FEATURES_PER_ROW);
                            subword_q <= '0;
                            row_data_q <= '0;
                            state_q <= STATE_COMMAND;
                        end
                    end
                end

                STATE_COMMAND: begin
                    if (refill_command_valid && refill_command_ready)
                        state_q <= STATE_GATHER;
                end

                STATE_GATHER: begin
                    if (feature_index_int >= integer'(feature_count_q)) begin
                        row_data_q[subword_q*16 +: 16] <= '0;
                        if (subword_q == ROW_SUBWORDS - 1)
                            state_q <= STATE_SEND_ROW;
                        else
                            subword_q <= subword_q + 1'b1;
                    end else if (read_fire) begin
                        state_q <= STATE_READ_RESPONSE;
                    end
                end

                STATE_READ_REQUEST: begin
                    if (read_fire)
                        state_q <= STATE_READ_RESPONSE;
                end

                STATE_READ_RESPONSE: begin
                    if (response_fire) begin
                        if (i_forward_response_error ||
                            i_forward_response_tag != read_tag_q) begin
                            state_q <= STATE_IDLE;
                            o_sticky_error <= 1'b1;
                        end else begin
                            row_data_q[subword_q*16 +: 16] <=
                                i_forward_response_code;
                            read_tag_q <= read_tag_q + 1'b1;
                            if (subword_q == ROW_SUBWORDS - 1)
                                state_q <= STATE_SEND_ROW;
                            else begin
                                subword_q <= subword_q + 1'b1;
                                state_q <= read_fire ? STATE_READ_RESPONSE : STATE_GATHER;
                            end
                        end
                    end
                end

                STATE_SEND_ROW: begin
                    if (row_fire) begin
                        row_data_q <= '0;
                        subword_q <= '0;
                        if (row_q + 1'b1 < rows_per_bank_q) begin
                            row_q <= row_q + 1'b1;
                            state_q <= read_fire ? STATE_READ_RESPONSE : STATE_GATHER;
                        end else begin
                            state_q <= STATE_WAIT_COMPLETE;
                        end
                    end
                end

                STATE_WAIT_COMPLETE: begin
                    if (complete_fire) begin
                        row_q <= '0;
                        subword_q <= '0;
                        row_data_q <= '0;
                        if (!bank_parity_q) begin
                            bank_parity_q <= 1'b1;
                            state_q <= STATE_COMMAND;
                        end else if (class_q + 1'b1 < class_count_q) begin
                            class_q <= class_q + 1'b1;
                            bank_parity_q <= 1'b0;
                            class_parameter_base_q <=
                                class_parameter_base_q + feature_count_q;
                            state_q <= STATE_COMMAND;
                        end else begin
                            o_published_pair <= target_pair_q;
                            o_published_version <= version_q;
                            o_done <= 1'b1;
                            state_q <= STATE_IDLE;
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
        if (MAX_FEATURES > 13872 || MAX_CLASSES > 16 ||
            ARRAY_WEIGHT_ADDR_WIDTH != 11 || TAG_WIDTH < 1)
            $error("SCNN classifier cache publisher parameter mismatch");
    end
endmodule

`default_nettype wire
