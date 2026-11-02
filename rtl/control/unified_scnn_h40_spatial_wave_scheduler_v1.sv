`timescale 1ns/1ps
`default_nettype none

// Expands one 3x3 SCNN layer into commands for the shared H40 array.
// A physical tile owns one output channel and its forty lanes own consecutive
// spatial positions. Four input-channel/tap contributions are issued per
// command; arithmetic and neuron state remain in the common H40 datapath.
module unified_scnn_h40_spatial_wave_scheduler_v1 #(
    parameter int WEIGHT_ADDR_WIDTH = 11,
    parameter int WORK_ADDR_WIDTH = 9,
    parameter int WEIGHT_BANK_DEPTH = 578,
    parameter int PHYSICAL_TILE_COUNT = 16,
    parameter int VALUE_INPUT_BITS = 15,
    parameter bit HOLD_ACTIVE_PAIR = 1'b0
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_abort,
    input  wire logic                         i_clear_error,

    input  wire logic                         i_start_valid,
    output logic                              o_start_ready,
    input  wire logic [15:0]                  i_in_width,
    input  wire logic [15:0]                  i_in_height,
    input  wire logic [15:0]                  i_in_channels,
    input  wire logic [15:0]                  i_out_width,
    input  wire logic [15:0]                  i_out_height,
    input  wire logic [15:0]                  i_out_channels,
    input  wire logic [7:0]                   i_stride,
    input  wire logic [7:0]                   i_padding,
    input  wire logic [1:0]                   i_pool_mode,
    input  wire logic                         i_initial_active_pair,
    input  wire logic [WORK_ADDR_WIDTH-1:0]   i_work_base_addr,
    input  wire logic [15:0]                  i_leak,
    input  wire logic [31:0]                  i_threshold,
    input  wire logic                         i_reset_to_zero,
    input  wire logic [15:0]                  i_trace_decay,
    input  wire logic [19:0]                  i_layer_tag,

    // The workset manager refills the selected output-channel weights and the
    // state/trace rows for this spatial group before accepting the request.
    output logic                              o_context_load_valid,
    input  wire logic                         i_context_load_ready,
    output logic                              o_context_load_weights,
    output logic [9:0]                        o_context_output_channel_base,
    output logic [9:0]                        o_context_output_channel_count,
    output logic [31:0]                       o_context_spatial_base,
    output logic [5:0]                        o_context_spatial_count,
    output logic                              o_context_active_pair,
    output logic [WORK_ADDR_WIDTH-1:0]        o_context_work_base_addr,

    // The feature reader converts these four coordinates into one binary
    // 40-position activity mask per contribution, including padding effects.
    output logic                              o_feature_request_valid,
    input  wire logic                         i_feature_request_ready,
    output logic [3:0]                        o_feature_term_valid,
    output logic [39:0]                       o_feature_input_channel_flat,
    output logic [7:0]                        o_feature_kernel_row_flat,
    output logic [7:0]                        o_feature_kernel_col_flat,
    output logic [31:0]                       o_feature_spatial_base,
    output logic [5:0]                        o_feature_spatial_count,
    output logic [15:0]                       o_feature_output_row,
    output logic [15:0]                       o_feature_output_col,
    // POOL_VALUE_BYPASS requests one unsigned Q8.8 input bit-plane at a
    // time.  Four independent contribution masks still share one power-of-two
    // factor, so the physical P4 MACs are reused without a second datapath.
    output logic [3:0]                        o_feature_value_bit,
    input  wire logic                         i_feature_response_valid,
    output logic                              o_feature_response_ready,
    input  wire logic [39:0]                  i_feature_mask0,
    input  wire logic [39:0]                  i_feature_mask1,
    input  wire logic [39:0]                  i_feature_mask2,
    input  wire logic [39:0]                  i_feature_mask3,

    output logic                              o_command_valid,
    input  wire logic                         i_command_ready,
    output logic [15:0]                       o_command_output_channels,
    output logic                              o_command_first,
    output logic                              o_command_last,
    output logic [5:0]                        o_command_spatial_count,
    output logic [9:0]                        o_command_output_channel_base,
    output logic [39:0]                       o_command_feature_mask0,
    output logic [39:0]                       o_command_feature_mask1,
    output logic [39:0]                       o_command_feature_mask2,
    output logic [39:0]                       o_command_feature_mask3,
    output logic signed [17:0]                o_command_factor_b,
    output logic                              o_command_active_pair,
    output logic [3:0]                        o_command_weight_subword0,
    output logic [3:0]                        o_command_weight_subword1,
    output logic [3:0]                        o_command_weight_subword2,
    output logic [3:0]                        o_command_weight_subword3,
    output logic [WEIGHT_ADDR_WIDTH-1:0]      o_command_weight_addr0,
    output logic [WEIGHT_ADDR_WIDTH-1:0]      o_command_weight_addr1,
    output logic [WEIGHT_ADDR_WIDTH-1:0]      o_command_weight_addr2,
    output logic [WEIGHT_ADDR_WIDTH-1:0]      o_command_weight_addr3,
    output logic [WORK_ADDR_WIDTH-1:0]        o_command_work_base_addr,
    output logic [15:0]                       o_command_leak,
    output logic [31:0]                       o_command_threshold,
    output logic                              o_command_reset_to_zero,
    output logic [15:0]                       o_command_trace_decay,
    output logic [51:0]                       o_command_transaction_tag,
    input  wire logic                         i_command_done,
    input  wire logic                         i_child_error,

    // Release acceptance is the writeback barrier for output, state and trace
    // before the same physical tile/work rows are reused by another workset.
    output logic                              o_context_release_valid,
    input  wire logic                         i_context_release_ready,
    output logic [9:0]                        o_release_output_channel_base,
    output logic [9:0]                        o_release_output_channel_count,
    output logic [31:0]                       o_release_spatial_base,
    output logic [5:0]                        o_release_spatial_count,
    output logic                              o_release_active_pair,
    output logic [WORK_ADDR_WIDTH-1:0]        o_release_work_base_addr,

    output logic                              o_busy,
    output logic                              o_done,
    output logic                              o_sticky_error
);
    localparam int KERNEL_SIZE = 3;
    localparam int CONTRIBUTIONS_PER_CHANNEL = 9;
    localparam int WEIGHTS_PER_BANK_ROW = 12;
    localparam logic [1:0] POOL_VALUE_BYPASS = 2'd3;

    typedef enum logic [2:0] {
        IDLE,
        CONTEXT_LOAD,
        FEATURE_REQUEST,
        FEATURE_RESPONSE,
        COMMAND_REQUEST,
        COMMAND_WAIT,
        CONTEXT_RELEASE
    } state_t;

    state_t state_q;
    logic [10:0] out_channels_q;
    logic [15:0] out_width_q;
    logic [13:0] contribution_count_q;
    logic [31:0] spatial_count_q;
    logic [9:0] output_base_q;
    logic [31:0] spatial_base_q;
    logic [15:0] output_row_q;
    logic [15:0] output_col_q;
    logic [13:0] contribution_base_q;
    logic active_pair_q;
    logic [WORK_ADDR_WIDTH-1:0] work_base_addr_q;
    logic [15:0] leak_q;
    logic [31:0] threshold_q;
    logic reset_to_zero_q;
    logic [15:0] trace_decay_q;
    logic [19:0] layer_tag_q;
    logic value_input_q;
    logic [3:0] value_bit_q;
    logic [39:0] feature_mask_q [0:3];

    logic config_legal;
    logic [10:0] output_remaining;
    logic [31:0] spatial_remaining;
    logic [9:0] output_wave_count;
    logic [5:0] spatial_wave_count;
    logic final_contribution;
    logic start_fire;
    logic context_load_fire;
    logic feature_request_fire;
    logic feature_response_fire;
    logic command_fire;
    logic context_release_fire;

    integer unsigned padded_width;
    integer unsigned padded_height;
    integer unsigned output_width_lower_bound;
    integer unsigned output_width_upper_bound;
    integer unsigned output_height_lower_bound;
    integer unsigned output_height_upper_bound;
    integer unsigned total_contributions_input;
    integer unsigned total_spatial_input;
    integer unsigned contribution_index;
    integer unsigned paired_index;
    integer unsigned tap_index;
    integer unsigned output_row_remaining;

    always_comb begin
        padded_width = integer'(i_in_width) + 2 * integer'(i_padding);
        padded_height = integer'(i_in_height) + 2 * integer'(i_padding);
        output_width_lower_bound = i_out_width == 0 ? 0 :
            (integer'(i_out_width) - 1) * integer'(i_stride);
        output_width_upper_bound =
            integer'(i_out_width) * integer'(i_stride);
        output_height_lower_bound = i_out_height == 0 ? 0 :
            (integer'(i_out_height) - 1) * integer'(i_stride);
        output_height_upper_bound =
            integer'(i_out_height) * integer'(i_stride);
        total_contributions_input =
            integer'(i_in_channels) * CONTRIBUTIONS_PER_CHANNEL;
        total_spatial_input = integer'(i_out_width) * integer'(i_out_height);
        config_legal = i_in_width != 0 && i_in_height != 0 &&
            i_in_channels != 0 && i_in_channels <= 16'd1024 &&
            i_out_width != 0 && i_out_height != 0 &&
            i_out_channels != 0 && i_out_channels <= 16'd1024 &&
            i_stride != 0 && padded_width >= KERNEL_SIZE &&
            padded_height >= KERNEL_SIZE &&
            output_width_lower_bound <= padded_width - KERNEL_SIZE &&
            output_width_upper_bound > padded_width - KERNEL_SIZE &&
            output_height_lower_bound <= padded_height - KERNEL_SIZE &&
            output_height_upper_bound > padded_height - KERNEL_SIZE &&
            total_contributions_input <=
                2 * WEIGHT_BANK_DEPTH * WEIGHTS_PER_BANK_ROW &&
            total_spatial_input != 0 &&
            integer'(i_work_base_addr) + 1 < (1 << WORK_ADDR_WIDTH) &&
            VALUE_INPUT_BITS >= 1 && VALUE_INPUT_BITS <= 15;
    end

    always_comb begin
        output_remaining = out_channels_q - {1'b0, output_base_q};
        spatial_remaining = spatial_count_q - spatial_base_q;
        output_row_remaining = integer'(out_width_q) -
            integer'(output_col_q);
        output_wave_count = output_remaining > 11'(PHYSICAL_TILE_COUNT) ?
            10'(PHYSICAL_TILE_COUNT) : output_remaining[9:0];
        if (spatial_remaining > 32'd40 && output_row_remaining > 40)
            spatial_wave_count = 6'd40;
        else if (spatial_remaining < output_row_remaining)
            spatial_wave_count = spatial_remaining[5:0];
        else
            spatial_wave_count = 6'(output_row_remaining);
        final_contribution =
            {1'b0, contribution_base_q} + 15'd4 >=
            {1'b0, contribution_count_q};

        o_start_ready = state_q == IDLE && !o_sticky_error;
        o_busy = state_q != IDLE;
        o_context_load_valid = state_q == CONTEXT_LOAD;
        o_context_load_weights = spatial_base_q == 0;
        o_context_output_channel_base = output_base_q;
        o_context_output_channel_count = output_wave_count;
        o_context_spatial_base = spatial_base_q;
        o_context_spatial_count = spatial_wave_count;
        o_context_active_pair = active_pair_q;
        o_context_work_base_addr = work_base_addr_q;

        o_feature_request_valid = state_q == FEATURE_REQUEST;
        o_feature_term_valid = '0;
        o_feature_input_channel_flat = '0;
        o_feature_kernel_row_flat = '0;
        o_feature_kernel_col_flat = '0;
        o_feature_spatial_base = spatial_base_q;
        o_feature_spatial_count = spatial_wave_count;
        o_feature_output_row = output_row_q;
        o_feature_output_col = output_col_q;
        o_feature_value_bit = value_input_q ? value_bit_q : 4'd0;

        o_command_weight_addr0 = '0;
        o_command_weight_addr1 = '0;
        o_command_weight_addr2 = '0;
        o_command_weight_addr3 = '0;
        o_command_weight_subword0 = '0;
        o_command_weight_subword1 = '0;
        o_command_weight_subword2 = '0;
        o_command_weight_subword3 = '0;
        for (integer term = 0; term < 4; term++) begin
            contribution_index = integer'(contribution_base_q) + term;
            if (contribution_index < integer'(contribution_count_q)) begin
                o_feature_term_valid[term] = 1'b1;
                o_feature_input_channel_flat[term*10 +: 10] =
                    10'(contribution_index / CONTRIBUTIONS_PER_CHANNEL);
                tap_index = contribution_index % CONTRIBUTIONS_PER_CHANNEL;
                o_feature_kernel_row_flat[term*2 +: 2] =
                    2'(tap_index / KERNEL_SIZE);
                o_feature_kernel_col_flat[term*2 +: 2] =
                    2'(tap_index % KERNEL_SIZE);
                paired_index = contribution_index / 2;
                case (term)
                    0: begin
                        o_command_weight_addr0 =
                            WEIGHT_ADDR_WIDTH'(
                                paired_index / WEIGHTS_PER_BANK_ROW);
                        o_command_weight_subword0 =
                            4'(paired_index % WEIGHTS_PER_BANK_ROW);
                    end
                    1: begin
                        o_command_weight_addr1 =
                            WEIGHT_ADDR_WIDTH'(
                                paired_index / WEIGHTS_PER_BANK_ROW);
                        o_command_weight_subword1 =
                            4'(paired_index % WEIGHTS_PER_BANK_ROW);
                    end
                    2: begin
                        o_command_weight_addr2 =
                            WEIGHT_ADDR_WIDTH'(
                                paired_index / WEIGHTS_PER_BANK_ROW);
                        o_command_weight_subword2 =
                            4'(paired_index % WEIGHTS_PER_BANK_ROW);
                    end
                    default: begin
                        o_command_weight_addr3 =
                            WEIGHT_ADDR_WIDTH'(
                                paired_index / WEIGHTS_PER_BANK_ROW);
                        o_command_weight_subword3 =
                            4'(paired_index % WEIGHTS_PER_BANK_ROW);
                    end
                endcase
            end
        end
        o_feature_response_ready = state_q == FEATURE_RESPONSE;

        o_command_valid = state_q == COMMAND_REQUEST;
        o_command_output_channels = {6'd0, output_wave_count};
        o_command_first = contribution_base_q == 0 &&
            (!value_input_q || value_bit_q == 0);
        o_command_last = final_contribution &&
            (!value_input_q ||
             value_bit_q + 4'd1 == 4'(VALUE_INPUT_BITS));
        o_command_spatial_count = spatial_wave_count;
        o_command_output_channel_base = output_base_q;
        o_command_feature_mask0 = feature_mask_q[0];
        o_command_feature_mask1 = feature_mask_q[1];
        o_command_feature_mask2 = feature_mask_q[2];
        o_command_feature_mask3 = feature_mask_q[3];
        o_command_factor_b = value_input_q ?
            (18'sd1 <<< value_bit_q) : 18'sd256;
        o_command_active_pair = active_pair_q;
        o_command_work_base_addr = work_base_addr_q;
        o_command_leak = leak_q;
        o_command_threshold = threshold_q;
        o_command_reset_to_zero = reset_to_zero_q;
        o_command_trace_decay = trace_decay_q;
        o_command_transaction_tag = {layer_tag_q, spatial_base_q};

        o_context_release_valid = state_q == CONTEXT_RELEASE;
        o_release_output_channel_base = output_base_q;
        o_release_output_channel_count = output_wave_count;
        o_release_spatial_base = spatial_base_q;
        o_release_spatial_count = spatial_wave_count;
        o_release_active_pair = active_pair_q;
        o_release_work_base_addr = work_base_addr_q;

        start_fire = i_start_valid && o_start_ready;
        context_load_fire = o_context_load_valid && i_context_load_ready;
        feature_request_fire =
            o_feature_request_valid && i_feature_request_ready;
        feature_response_fire =
            i_feature_response_valid && o_feature_response_ready;
        command_fire = o_command_valid && i_command_ready;
        context_release_fire =
            o_context_release_valid && i_context_release_ready;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= IDLE;
            out_channels_q <= '0;
            out_width_q <= '0;
            contribution_count_q <= '0;
            spatial_count_q <= '0;
            output_base_q <= '0;
            spatial_base_q <= '0;
            output_row_q <= '0;
            output_col_q <= '0;
            contribution_base_q <= '0;
            active_pair_q <= 1'b0;
            work_base_addr_q <= '0;
            leak_q <= '0;
            threshold_q <= '0;
            reset_to_zero_q <= 1'b0;
            trace_decay_q <= '0;
            layer_tag_q <= '0;
            value_input_q <= 1'b0;
            value_bit_q <= '0;
            for (integer term = 0; term < 4; term++)
                feature_mask_q[term] <= '0;
            o_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_abort) begin
                state_q <= IDLE;
            end else if (i_clear_error && state_q == IDLE) begin
                o_sticky_error <= 1'b0;
            end

            if (!i_abort && i_child_error && state_q != IDLE) begin
                state_q <= IDLE;
                o_sticky_error <= 1'b1;
            end else if (!i_abort) begin
                case (state_q)
                    IDLE: begin
                        if (start_fire) begin
                            if (!config_legal) begin
                                o_sticky_error <= 1'b1;
                            end else begin
                                out_channels_q <= 11'(i_out_channels);
                                out_width_q <= i_out_width;
                                contribution_count_q <=
                                    14'(total_contributions_input);
                                spatial_count_q <= 32'(total_spatial_input);
                                output_base_q <= '0;
                                spatial_base_q <= '0;
                                output_row_q <= '0;
                                output_col_q <= '0;
                                contribution_base_q <= '0;
                                active_pair_q <= i_initial_active_pair;
                                work_base_addr_q <= i_work_base_addr;
                                leak_q <= i_leak;
                                threshold_q <= i_threshold;
                                reset_to_zero_q <= i_reset_to_zero;
                                trace_decay_q <= i_trace_decay;
                                layer_tag_q <= i_layer_tag;
                                value_input_q <= i_pool_mode ==
                                    POOL_VALUE_BYPASS;
                                value_bit_q <= '0;
                                state_q <= CONTEXT_LOAD;
                            end
                        end
                    end
                    CONTEXT_LOAD: begin
                        if (context_load_fire)
                            state_q <= FEATURE_REQUEST;
                    end
                    FEATURE_REQUEST: begin
                        if (feature_request_fire)
                            state_q <= FEATURE_RESPONSE;
                    end
                    FEATURE_RESPONSE: begin
                        if (feature_response_fire) begin
                            feature_mask_q[0] <= i_feature_mask0;
                            feature_mask_q[1] <= i_feature_mask1;
                            feature_mask_q[2] <= i_feature_mask2;
                            feature_mask_q[3] <= i_feature_mask3;
                            state_q <= COMMAND_REQUEST;
                        end
                    end
                    COMMAND_REQUEST: begin
                        if (command_fire)
                            state_q <= COMMAND_WAIT;
                    end
                    COMMAND_WAIT: begin
                        if (i_command_done) begin
                            if (value_input_q &&
                                    value_bit_q + 4'd1 <
                                    4'(VALUE_INPUT_BITS)) begin
                                value_bit_q <= value_bit_q + 4'd1;
                                state_q <= FEATURE_REQUEST;
                            end else if (final_contribution) begin
                                value_bit_q <= '0;
                                state_q <= CONTEXT_RELEASE;
                            end else begin
                                value_bit_q <= '0;
                                contribution_base_q <=
                                    contribution_base_q + 14'd4;
                                state_q <= FEATURE_REQUEST;
                            end
                        end
                    end
                    CONTEXT_RELEASE: begin
                        if (context_release_fire) begin
                            contribution_base_q <= '0;
                            if (spatial_base_q + 32'(spatial_wave_count) >=
                                spatial_count_q) begin
                                spatial_base_q <= '0;
                                output_row_q <= '0;
                                output_col_q <= '0;
                                if ({1'b0, output_base_q} +
                                    {1'b0, output_wave_count} >=
                                    out_channels_q) begin
                                    state_q <= IDLE;
                                    o_done <= 1'b1;
                                end else begin
                                    output_base_q <= output_base_q +
                                        output_wave_count;
                                    if (!HOLD_ACTIVE_PAIR)
                                        active_pair_q <= !active_pair_q;
                                    state_q <= CONTEXT_LOAD;
                                end
                            end else begin
                                spatial_base_q <= spatial_base_q +
                                    32'(spatial_wave_count);
                                if ({1'b0, output_col_q} +
                                    {11'd0, spatial_wave_count} >=
                                    {1'b0, out_width_q}) begin
                                    output_row_q <= output_row_q + 16'd1;
                                    output_col_q <= '0;
                                end else begin
                                    output_col_q <= output_col_q +
                                        16'(spatial_wave_count);
                                end
                                state_q <= CONTEXT_LOAD;
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
        if (WEIGHT_BANK_DEPTH != 578 || WORK_ADDR_WIDTH != 9 ||
                PHYSICAL_TILE_COUNT < 1 || PHYSICAL_TILE_COUNT > 16)
            $error("SCNN H40 scheduler requires the selected P4 cache/workset geometry");
    end
endmodule

`default_nettype wire
