`timescale 1ns/1ps
`default_nettype none

// Maps the SCNN classifier feature stream onto the same P4 H40 array as
// convolution. AAVG2 features use five Q8.8 values (0, 1/4, 1/2, 3/4, 1).
// Terms with the same nonzero level are grouped into one micro-command so the
// existing common DSP factor can scale four weights without extra multipliers.
// In SCNN mode o_command_source_valid is a tagged level code (1..4), not the
// FC/SRNN per-term-valid bitmap; term presence remains in the feature masks.
module unified_scnn_h40_classifier_p4_scheduler_v1 #(
    parameter int MAX_FEATURES = 2048,
    parameter int MAX_CLASSES = 16,
    parameter int LANES = 8,
    parameter int WEIGHT_ADDR_WIDTH = 10,
    parameter int WORK_ADDR_WIDTH = 9,
    parameter int FEATURE_WIDTH = MAX_FEATURES <= 2 ? 1 : $clog2(MAX_FEATURES),
    parameter int FEATURE_COUNT_WIDTH = $clog2(MAX_FEATURES + 1),
    parameter int CLASS_COUNT_WIDTH = $clog2(MAX_CLASSES + 1)
) (
    input  wire logic                           i_clk,
    input  wire logic                           i_rst_n,
    input  wire logic                           i_abort,
    input  wire logic                           i_clear_error,

    input  wire logic                           i_start_valid,
    output logic                                o_start_ready,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0] i_feature_count,
    input  wire logic [CLASS_COUNT_WIDTH-1:0]   i_class_count,
    input  wire logic                           i_active_pair,
    input  wire logic [WORK_ADDR_WIDTH-1:0]     i_work_base_addr,
    input  wire logic [15:0]                    i_trace_decay,
    input  wire logic [19:0]                    i_layer_tag,

    input  wire logic                           i_feature_wave_valid,
    output logic                                o_feature_wave_ready,
    input  wire logic [LANES-1:0]               i_feature_wave_mask,
    input  wire logic [FEATURE_WIDTH-1:0]       i_feature_wave_base,
    input  wire logic [LANES*16-1:0]            i_feature_wave_q8_8_flat,
    input  wire logic                           i_feature_wave_first,
    input  wire logic                           i_feature_wave_last,

    output logic                                o_command_valid,
    input  wire logic                           i_command_ready,
    output logic [15:0]                         o_command_neuron_count,
    output logic                                o_command_first,
    output logic                                o_command_last,
    output logic [3:0]                          o_command_source_valid,
    output logic [5:0]                          o_command_spatial_count,
    output logic [9:0]                          o_command_output_channel_base,
    output logic [39:0]                         o_command_feature_mask0,
    output logic [39:0]                         o_command_feature_mask1,
    output logic [39:0]                         o_command_feature_mask2,
    output logic [39:0]                         o_command_feature_mask3,
    output logic                                o_command_active_pair,
    output logic [3:0]                          o_command_weight_subword0,
    output logic [3:0]                          o_command_weight_subword1,
    output logic [3:0]                          o_command_weight_subword2,
    output logic [3:0]                          o_command_weight_subword3,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_command_weight_addr0,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_command_weight_addr1,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_command_weight_addr2,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_command_weight_addr3,
    output logic [WORK_ADDR_WIDTH-1:0]          o_command_work_base_addr,
    output logic [15:0]                         o_command_leak,
    output logic [31:0]                         o_command_threshold,
    output logic                                o_command_reset_to_zero,
    output logic [15:0]                         o_command_trace_decay,
    output logic [51:0]                         o_command_transaction_tag,
    input  wire logic                           i_command_done,
    input  wire logic                           i_child_error,

    output logic                                o_done,
    output logic                                o_busy,
    output logic                                o_sticky_error
);
    localparam int WEIGHTS_PER_ROW = 12;

    typedef enum logic [2:0] {
        IDLE, ACCEPT_WAVE, COMMAND_REQUEST, COMMAND_WAIT
    } state_t;
    state_t state_q;

    logic [FEATURE_COUNT_WIDTH-1:0] feature_count_q;
    logic [CLASS_COUNT_WIDTH-1:0] class_count_q;
    logic active_pair_q;
    logic [WORK_ADDR_WIDTH-1:0] work_base_addr_q;
    logic [15:0] trace_decay_q;
    logic [19:0] layer_tag_q;
    logic [LANES-1:0] wave_mask_q;
    logic [FEATURE_WIDTH-1:0] wave_base_q;
    logic [FEATURE_WIDTH-1:0] expected_feature_base_q;
    logic [LANES*16-1:0] wave_q8_8_q;
    logic wave_last_q;
    logic half_q;
    logic [2:0] level_q;
    logic frame_command_started_q;

    logic start_fire;
    logic wave_fire;
    logic command_fire;
    logic config_legal;
    logic wave_legal;
    logic [LANES-1:0] expected_wave_mask;
    logic initial_group_found;
    logic initial_half;
    logic [2:0] initial_level;
    logic next_group_found;
    logic next_half;
    logic [2:0] next_level;
    integer unsigned feature_index [0:3];
    integer unsigned pair_index [0:3];

    always_comb begin
        config_legal = i_feature_count != 0 &&
            integer'(i_feature_count) <= MAX_FEATURES &&
            i_class_count != 0 && integer'(i_class_count) <= MAX_CLASSES;
        o_start_ready = state_q == IDLE && !o_sticky_error;
        start_fire = i_start_valid && o_start_ready;
        o_busy = state_q != IDLE;

        expected_wave_mask = '0;
        for (integer lane = 0; lane < LANES; lane++)
            if (integer'(i_feature_wave_base) + lane <
                integer'(feature_count_q))
                expected_wave_mask[lane] = 1'b1;
        wave_legal = i_feature_wave_base == expected_feature_base_q &&
            i_feature_wave_mask == expected_wave_mask &&
            i_feature_wave_first == (i_feature_wave_base == 0) &&
            i_feature_wave_last ==
                (integer'(i_feature_wave_base) + LANES >=
                 integer'(feature_count_q));
        for (integer lane = 0; lane < LANES; lane++) begin
            logic signed [15:0] feature_value;
            feature_value =
                $signed(i_feature_wave_q8_8_flat[lane*16 +: 16]);
            if (i_feature_wave_mask[lane] &&
                (feature_value < 0 || feature_value > 16'sd256 ||
                 feature_value[5:0] != 0))
                wave_legal = 1'b0;
        end

        // Select the first populated quarter-level group in the incoming
        // wave. An all-zero wave still emits one no-op command so first/last
        // transaction boundaries and classifier state initialization remain
        // explicit.
        initial_group_found = 1'b0;
        initial_half = 1'b0;
        initial_level = 3'd4;
        for (integer candidate = 1; candidate <= 4; candidate++) begin
            for (integer lane = 0; lane < 4; lane++) begin
                if (!initial_group_found && i_feature_wave_mask[lane] &&
                    $unsigned(i_feature_wave_q8_8_flat[lane*16 +: 16]) ==
                        candidate * 64) begin
                    initial_group_found = 1'b1;
                    initial_half = 1'b0;
                    initial_level = 3'(candidate);
                end
            end
        end
        for (integer candidate = 1; candidate <= 4; candidate++) begin
            for (integer lane = 4; lane < 8; lane++) begin
                if (!initial_group_found && i_feature_wave_mask[lane] &&
                    $unsigned(i_feature_wave_q8_8_flat[lane*16 +: 16]) ==
                        candidate * 64) begin
                    initial_group_found = 1'b1;
                    initial_half = 1'b1;
                    initial_level = 3'(candidate);
                end
            end
        end

        // Find the next populated group after the command currently being
        // issued: higher levels in this half, then levels in the second half.
        next_group_found = 1'b0;
        next_half = half_q;
        next_level = level_q;
        for (integer candidate = 1; candidate <= 4; candidate++) begin
            for (integer lane = 0; lane < 4; lane++) begin
                if (!next_group_found && candidate > integer'(level_q) &&
                    wave_mask_q[(half_q ? 4 : 0) + lane] &&
                    $unsigned(wave_q8_8_q[
                        ((half_q ? 4 : 0) + lane)*16 +: 16]) ==
                        candidate * 64) begin
                    next_group_found = 1'b1;
                    next_half = half_q;
                    next_level = 3'(candidate);
                end
            end
        end
        if (!half_q) begin
            for (integer candidate = 1; candidate <= 4; candidate++) begin
                for (integer lane = 4; lane < 8; lane++) begin
                    if (!next_group_found && wave_mask_q[lane] &&
                        $unsigned(wave_q8_8_q[lane*16 +: 16]) ==
                            candidate * 64) begin
                        next_group_found = 1'b1;
                        next_half = 1'b1;
                        next_level = 3'(candidate);
                    end
                end
            end
        end

        o_feature_wave_ready = state_q == ACCEPT_WAVE &&
            !o_sticky_error;
        wave_fire = i_feature_wave_valid && o_feature_wave_ready;

        o_command_valid = state_q == COMMAND_REQUEST && !o_sticky_error;
        o_command_neuron_count = 16'(class_count_q);
        o_command_first = !frame_command_started_q;
        o_command_last = wave_last_q && !next_group_found;
        o_command_source_valid = 4'(level_q);
        o_command_spatial_count = 6'd1;
        o_command_output_channel_base = 10'd0;
        o_command_feature_mask0 = '0;
        o_command_feature_mask1 = '0;
        o_command_feature_mask2 = '0;
        o_command_feature_mask3 = '0;
        o_command_active_pair = active_pair_q;
        o_command_weight_subword0 = '0;
        o_command_weight_subword1 = '0;
        o_command_weight_subword2 = '0;
        o_command_weight_subword3 = '0;
        o_command_weight_addr0 = '0;
        o_command_weight_addr1 = '0;
        o_command_weight_addr2 = '0;
        o_command_weight_addr3 = '0;
        for (integer term = 0; term < 4; term++) begin
            feature_index[term] = integer'(wave_base_q) +
                (half_q ? 4 : 0) + term;
            pair_index[term] = feature_index[term] / 2;
            if (feature_index[term] < integer'(feature_count_q) &&
                wave_mask_q[(half_q ? 4 : 0) + term]) begin
                case (term)
                    0: begin
                        o_command_weight_addr0 = WEIGHT_ADDR_WIDTH'(
                            pair_index[term] / WEIGHTS_PER_ROW);
                        o_command_weight_subword0 = 4'(
                            pair_index[term] % WEIGHTS_PER_ROW);
                        o_command_feature_mask0[0] =
                            $unsigned(wave_q8_8_q[
                                ((half_q ? 4 : 0) + term)*16 +: 16]) ==
                                integer'(level_q) * 64;
                    end
                    1: begin
                        o_command_weight_addr1 = WEIGHT_ADDR_WIDTH'(
                            pair_index[term] / WEIGHTS_PER_ROW);
                        o_command_weight_subword1 = 4'(
                            pair_index[term] % WEIGHTS_PER_ROW);
                        o_command_feature_mask1[0] =
                            $unsigned(wave_q8_8_q[
                                ((half_q ? 4 : 0) + term)*16 +: 16]) ==
                                integer'(level_q) * 64;
                    end
                    2: begin
                        o_command_weight_addr2 = WEIGHT_ADDR_WIDTH'(
                            pair_index[term] / WEIGHTS_PER_ROW);
                        o_command_weight_subword2 = 4'(
                            pair_index[term] % WEIGHTS_PER_ROW);
                        o_command_feature_mask2[0] =
                            $unsigned(wave_q8_8_q[
                                ((half_q ? 4 : 0) + term)*16 +: 16]) ==
                                integer'(level_q) * 64;
                    end
                    default: begin
                        o_command_weight_addr3 = WEIGHT_ADDR_WIDTH'(
                            pair_index[term] / WEIGHTS_PER_ROW);
                        o_command_weight_subword3 = 4'(
                            pair_index[term] % WEIGHTS_PER_ROW);
                        o_command_feature_mask3[0] =
                            $unsigned(wave_q8_8_q[
                                ((half_q ? 4 : 0) + term)*16 +: 16]) ==
                                integer'(level_q) * 64;
                    end
                endcase
            end
        end
        o_command_work_base_addr = work_base_addr_q;
        o_command_leak = 16'd0;
        o_command_threshold = 32'h00007fff;
        o_command_reset_to_zero = 1'b0;
        o_command_trace_decay = trace_decay_q;
        o_command_transaction_tag = {layer_tag_q, 32'd0};
        command_fire = o_command_valid && i_command_ready;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= IDLE;
            feature_count_q <= '0;
            class_count_q <= '0;
            active_pair_q <= 1'b0;
            work_base_addr_q <= '0;
            trace_decay_q <= '0;
            layer_tag_q <= '0;
            wave_mask_q <= '0;
            wave_base_q <= '0;
            expected_feature_base_q <= '0;
            wave_q8_8_q <= '0;
            wave_last_q <= 1'b0;
            half_q <= 1'b0;
            level_q <= 3'd4;
            frame_command_started_q <= 1'b0;
            o_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= IDLE;
            half_q <= 1'b0;
            level_q <= 3'd4;
            frame_command_started_q <= 1'b0;
            o_done <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_clear_error && state_q == IDLE)
                o_sticky_error <= 1'b0;

            if (i_child_error && state_q != IDLE) begin
                state_q <= IDLE;
                o_sticky_error <= 1'b1;
            end else begin
                case (state_q)
                    IDLE: begin
                        if (start_fire) begin
                            if (!config_legal) begin
                                o_sticky_error <= 1'b1;
                            end else begin
                                feature_count_q <= i_feature_count;
                                class_count_q <= i_class_count;
                                active_pair_q <= i_active_pair;
                                work_base_addr_q <= i_work_base_addr;
                                trace_decay_q <= i_trace_decay;
                                layer_tag_q <= i_layer_tag;
                                expected_feature_base_q <= '0;
                                frame_command_started_q <= 1'b0;
                                state_q <= ACCEPT_WAVE;
                            end
                        end
                    end
                    ACCEPT_WAVE: begin
                        if (wave_fire) begin
                            if (!wave_legal) begin
                                state_q <= IDLE;
                                o_sticky_error <= 1'b1;
                            end else begin
                                wave_mask_q <= i_feature_wave_mask;
                                wave_base_q <= i_feature_wave_base;
                                wave_q8_8_q <= i_feature_wave_q8_8_flat;
                                wave_last_q <= i_feature_wave_last;
                                half_q <= initial_half;
                                level_q <= initial_level;
                                state_q <= COMMAND_REQUEST;
                            end
                        end
                    end
                    COMMAND_REQUEST: begin
                        if (command_fire) begin
                            frame_command_started_q <= 1'b1;
                            state_q <= COMMAND_WAIT;
                        end
                    end
                    COMMAND_WAIT: begin
                        if (i_command_done) begin
                            if (o_command_last) begin
                                state_q <= IDLE;
                                o_done <= 1'b1;
                            end else if (next_group_found) begin
                                half_q <= next_half;
                                level_q <= next_level;
                                state_q <= COMMAND_REQUEST;
                            end else begin
                                expected_feature_base_q <=
                                    expected_feature_base_q + FEATURE_WIDTH'(LANES);
                                state_q <= ACCEPT_WAVE;
                            end
                        end
                    end
                    default: begin
                        state_q <= IDLE;
                        o_sticky_error <= 1'b1;
                    end
                endcase
            end
        end
    end

    initial begin
        if (LANES != 8 || MAX_CLASSES > 16 || MAX_FEATURES > 13872 ||
            WEIGHT_ADDR_WIDTH < 10)
            $error("SCNN H40 classifier scheduler parameter mismatch");
    end
endmodule

`default_nettype wire
