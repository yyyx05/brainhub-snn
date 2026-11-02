`timescale 1ns/1ps
`default_nettype none

// SCNN layer-control client for the common 16xH40/P4 array. The client owns
// topology traversal and compact feature writeback, but no PE, neuron state,
// trace or weight storage. Its command and observation ports attach to the
// same physical array used by FC and SRNN.
module unified_scnn_h40_array16_layer_client_v1 #(
    parameter int WEIGHT_DEPTH = 1156,
    parameter int WEIGHT_ADDR_WIDTH = $clog2(WEIGHT_DEPTH),
    parameter int WORK_ADDR_WIDTH = 9,
    parameter int FEATURE_WORD_DEPTH = 8192,
    parameter int FEATURE_ADDR_WIDTH = $clog2(FEATURE_WORD_DEPTH),
    parameter int PHYSICAL_TILE_COUNT = 16,
    parameter bit ENABLE_FEATURE_COMMIT_TAP = 1'b0,
    parameter bit HOLD_ACTIVE_PAIR = 1'b0
) (
    input  wire logic                           i_clk,
    input  wire logic                           i_rst_n,
    input  wire logic                           i_abort,
    input  wire logic                           i_clear_error,
    input  wire logic                           i_start_valid,
    output logic                                o_start_ready,
    input  wire logic [15:0]                    i_in_width,
    input  wire logic [15:0]                    i_in_height,
    input  wire logic [15:0]                    i_in_channels,
    input  wire logic [15:0]                    i_out_width,
    input  wire logic [15:0]                    i_out_height,
    input  wire logic [15:0]                    i_out_channels,
    input  wire logic [7:0]                     i_stride,
    input  wire logic [7:0]                     i_padding,
    input  wire logic [1:0]                     i_pool_mode,
    input  wire logic                           i_initial_active_pair,
    input  wire logic [WORK_ADDR_WIDTH-1:0]     i_work_base_addr,
    input  wire logic [15:0]                    i_leak,
    input  wire logic [31:0]                    i_threshold,
    input  wire logic                           i_reset_to_zero,
    input  wire logic [15:0]                    i_trace_decay,
    input  wire logic [19:0]                    i_layer_tag,

    output logic                                o_context_load_valid,
    input  wire logic                           i_context_load_ready,
    output logic                                o_context_load_weights,
    output logic [9:0]                          o_context_output_channel_base,
    output logic [9:0]                          o_context_output_channel_count,
    output logic [31:0]                         o_context_spatial_base,
    output logic [5:0]                          o_context_spatial_count,
    output logic                                o_context_active_pair,
    output logic [WORK_ADDR_WIDTH-1:0]          o_context_work_base_addr,

    output logic                                o_feature_request_valid,
    input  wire logic                           i_feature_request_ready,
    output logic [3:0]                          o_feature_term_valid,
    output logic [39:0]                         o_feature_input_channel_flat,
    output logic [7:0]                          o_feature_kernel_row_flat,
    output logic [7:0]                          o_feature_kernel_col_flat,
    output logic [31:0]                         o_feature_spatial_base,
    output logic [5:0]                          o_feature_spatial_count,
    output logic [15:0]                         o_feature_output_row,
    output logic [15:0]                         o_feature_output_col,
    output logic [3:0]                          o_feature_value_bit,
    input  wire logic                           i_feature_response_valid,
    output logic                                o_feature_response_ready,
    input  wire logic [39:0]                    i_feature_mask0,
    input  wire logic [39:0]                    i_feature_mask1,
    input  wire logic [39:0]                    i_feature_mask2,
    input  wire logic [39:0]                    i_feature_mask3,
    input  wire logic                           i_feature_service_error,

    input  wire logic                           i_feature_write_slot,
    input  wire logic [FEATURE_ADDR_WIDTH-1:0]  i_feature_write_base_addr,
    input  wire logic [15:0]                    i_feature_write_row_stride_words,
    input  wire logic [FEATURE_ADDR_WIDTH-1:0]  i_feature_write_channel_stride_words,
    output logic                                o_feature_update_valid,
    input  wire logic                           i_feature_update_ready,
    output logic                                o_feature_update_slot,
    output logic [FEATURE_ADDR_WIDTH-1:0]       o_feature_update_addr,
    output logic [63:0]                         o_feature_update_data,
    output logic [63:0]                         o_feature_update_mask,
    input  wire logic                           i_feature_update_done,

    output logic                                o_feature_commit_valid,
    input  wire logic                           i_feature_commit_ready,
    output logic [19:0]                         o_feature_commit_layer_tag,
    output logic [9:0]                          o_feature_commit_channel,
    output logic [31:0]                         o_feature_commit_spatial_base,
    output logic [5:0]                          o_feature_commit_spatial_count,
    output logic [39:0]                         o_feature_commit_spike,
    output logic [639:0]                        o_feature_commit_q8_8_flat,

    output logic                                o_context_release_valid,
    input  wire logic                           i_context_release_ready,
    output logic [9:0]                          o_release_output_channel_base,
    output logic [9:0]                          o_release_output_channel_count,
    output logic [31:0]                         o_release_spatial_base,
    output logic [5:0]                          o_release_spatial_count,
    output logic                                o_release_active_pair,
    output logic [WORK_ADDR_WIDTH-1:0]          o_release_work_base_addr,

    output logic                                o_array_command_valid,
    input  wire logic                           i_array_command_ready,
    output logic [15:0]                         o_array_command_neuron_count,
    output logic                                o_array_command_first,
    output logic                                o_array_command_last,
    output logic [5:0]                          o_array_command_spatial_count,
    output logic [9:0]                          o_array_command_output_channel_base,
    output logic [39:0]                         o_array_command_feature_mask0,
    output logic [39:0]                         o_array_command_feature_mask1,
    output logic [39:0]                         o_array_command_feature_mask2,
    output logic [39:0]                         o_array_command_feature_mask3,
    output logic signed [17:0]                  o_array_command_factor_b,
    output logic                                o_array_command_active_pair,
    output logic [3:0]                          o_array_command_weight_subword0,
    output logic [3:0]                          o_array_command_weight_subword1,
    output logic [3:0]                          o_array_command_weight_subword2,
    output logic [3:0]                          o_array_command_weight_subword3,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_array_command_weight_addr0,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_array_command_weight_addr1,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_array_command_weight_addr2,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_array_command_weight_addr3,
    output logic [WORK_ADDR_WIDTH-1:0]          o_array_command_work_base_addr,
    output logic [15:0]                         o_array_command_leak,
    output logic [31:0]                         o_array_command_threshold,
    output logic                                o_array_command_reset_to_zero,
    output logic [15:0]                         o_array_command_trace_decay,
    output logic [51:0]                         o_array_command_transaction_tag,
    input  wire logic                           i_array_command_done,
    input  wire logic                           i_array_busy,
    input  wire logic                           i_array_idle,
    input  wire logic                           i_array_error,

    input  wire logic [3:0]                     i_array_observation_valid,
    output logic [3:0]                          o_array_observation_ready,
    input  wire logic [15:0]                    i_array_observation_tile_flat,
    input  wire logic [7:0]                     i_array_observation_mode_flat,
    input  wire logic [39:0]                    i_array_observation_lane_mask_flat,
    input  wire logic [39:0]                    i_array_observation_range_mask_flat,
    input  wire logic [39:0]                    i_array_observation_spike_flat,
    input  wire logic [1279:0]                  i_array_observation_pre_state_flat,
    input  wire logic [1279:0]                  i_array_observation_trace_next_flat,
    input  wire logic [255:0]                   i_array_observation_meta_flat,

    output logic [3:0]                          o_observation_valid,
    input  wire logic [3:0]                     i_observation_ready,
    output logic [15:0]                         o_observation_tile_flat,
    output logic [7:0]                          o_observation_mode_flat,
    output logic [39:0]                         o_observation_lane_mask_flat,
    output logic [39:0]                         o_observation_range_mask_flat,
    output logic [39:0]                         o_observation_spike_flat,
    output logic [1279:0]                       o_observation_pre_state_flat,
    output logic [1279:0]                       o_observation_trace_next_flat,
    output logic [255:0]                        o_observation_meta_flat,

    output logic                                o_busy,
    output logic                                o_done,
    output logic                                o_idle,
    output logic                                o_sticky_error
);
    logic scheduler_busy;
    logic scheduler_done;
    logic scheduler_error;
    logic scheduler_context_load_valid;
    logic scheduler_context_release_valid;
    logic writeback_context_ready;
    logic writeback_close_ready;
    logic writeback_busy;
    logic writeback_error;
    logic [3:0] writeback_observation_select;

    assign o_busy = scheduler_busy || i_array_busy || writeback_busy;
    assign o_done = scheduler_done;
    assign o_idle = !scheduler_busy && i_array_idle && !writeback_busy;
    assign o_sticky_error = scheduler_error || i_array_error ||
                            writeback_error || i_feature_service_error;
    assign o_context_load_valid = scheduler_context_load_valid &&
                                  writeback_context_ready;
    assign o_context_release_valid = scheduler_context_release_valid &&
                                     writeback_close_ready;

    assign o_observation_valid = i_array_observation_valid &
                                 writeback_observation_select;
    assign o_array_observation_ready = writeback_observation_select &
                                       i_observation_ready;
    assign o_observation_tile_flat = i_array_observation_tile_flat;
    assign o_observation_mode_flat = i_array_observation_mode_flat;
    assign o_observation_lane_mask_flat =
        i_array_observation_lane_mask_flat;
    assign o_observation_range_mask_flat =
        i_array_observation_range_mask_flat;
    assign o_observation_spike_flat = i_array_observation_spike_flat;
    assign o_observation_pre_state_flat =
        i_array_observation_pre_state_flat;
    assign o_observation_trace_next_flat =
        i_array_observation_trace_next_flat;
    assign o_observation_meta_flat = i_array_observation_meta_flat;

    unified_scnn_h40_spatial_wave_scheduler_v1 #(
        .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH),
        .WEIGHT_BANK_DEPTH(WEIGHT_DEPTH / 2),
        .PHYSICAL_TILE_COUNT(PHYSICAL_TILE_COUNT),
        .HOLD_ACTIVE_PAIR(HOLD_ACTIVE_PAIR)
    ) u_scheduler (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_start_valid(i_start_valid), .o_start_ready(o_start_ready),
        .i_in_width(i_in_width), .i_in_height(i_in_height),
        .i_in_channels(i_in_channels), .i_out_width(i_out_width),
        .i_out_height(i_out_height), .i_out_channels(i_out_channels),
        .i_stride(i_stride), .i_padding(i_padding),
        .i_pool_mode(i_pool_mode),
        .i_initial_active_pair(i_initial_active_pair),
        .i_work_base_addr(i_work_base_addr), .i_leak(i_leak),
        .i_threshold(i_threshold), .i_reset_to_zero(i_reset_to_zero),
        .i_trace_decay(i_trace_decay), .i_layer_tag(i_layer_tag),
        .o_context_load_valid(scheduler_context_load_valid),
        .i_context_load_ready(i_context_load_ready &&
                              writeback_context_ready),
        .o_context_load_weights(o_context_load_weights),
        .o_context_output_channel_base(o_context_output_channel_base),
        .o_context_output_channel_count(o_context_output_channel_count),
        .o_context_spatial_base(o_context_spatial_base),
        .o_context_spatial_count(o_context_spatial_count),
        .o_context_active_pair(o_context_active_pair),
        .o_context_work_base_addr(o_context_work_base_addr),
        .o_feature_request_valid(o_feature_request_valid),
        .i_feature_request_ready(i_feature_request_ready),
        .o_feature_term_valid(o_feature_term_valid),
        .o_feature_input_channel_flat(o_feature_input_channel_flat),
        .o_feature_kernel_row_flat(o_feature_kernel_row_flat),
        .o_feature_kernel_col_flat(o_feature_kernel_col_flat),
        .o_feature_spatial_base(o_feature_spatial_base),
        .o_feature_spatial_count(o_feature_spatial_count),
        .o_feature_output_row(o_feature_output_row),
        .o_feature_output_col(o_feature_output_col),
        .o_feature_value_bit(o_feature_value_bit),
        .i_feature_response_valid(i_feature_response_valid),
        .o_feature_response_ready(o_feature_response_ready),
        .i_feature_mask0(i_feature_mask0),
        .i_feature_mask1(i_feature_mask1),
        .i_feature_mask2(i_feature_mask2),
        .i_feature_mask3(i_feature_mask3),
        .o_command_valid(o_array_command_valid),
        .i_command_ready(i_array_command_ready),
        .o_command_output_channels(o_array_command_neuron_count),
        .o_command_first(o_array_command_first),
        .o_command_last(o_array_command_last),
        .o_command_spatial_count(o_array_command_spatial_count),
        .o_command_output_channel_base(
            o_array_command_output_channel_base),
        .o_command_feature_mask0(o_array_command_feature_mask0),
        .o_command_feature_mask1(o_array_command_feature_mask1),
        .o_command_feature_mask2(o_array_command_feature_mask2),
        .o_command_feature_mask3(o_array_command_feature_mask3),
        .o_command_factor_b(o_array_command_factor_b),
        .o_command_active_pair(o_array_command_active_pair),
        .o_command_weight_subword0(o_array_command_weight_subword0),
        .o_command_weight_subword1(o_array_command_weight_subword1),
        .o_command_weight_subword2(o_array_command_weight_subword2),
        .o_command_weight_subword3(o_array_command_weight_subword3),
        .o_command_weight_addr0(o_array_command_weight_addr0),
        .o_command_weight_addr1(o_array_command_weight_addr1),
        .o_command_weight_addr2(o_array_command_weight_addr2),
        .o_command_weight_addr3(o_array_command_weight_addr3),
        .o_command_work_base_addr(o_array_command_work_base_addr),
        .o_command_leak(o_array_command_leak),
        .o_command_threshold(o_array_command_threshold),
        .o_command_reset_to_zero(o_array_command_reset_to_zero),
        .o_command_trace_decay(o_array_command_trace_decay),
        .o_command_transaction_tag(o_array_command_transaction_tag),
        .i_command_done(i_array_command_done),
        .i_child_error(i_array_error || writeback_error ||
                       i_feature_service_error),
        .o_context_release_valid(scheduler_context_release_valid),
        .i_context_release_ready(i_context_release_ready &&
                                 writeback_close_ready),
        .o_release_output_channel_base(o_release_output_channel_base),
        .o_release_output_channel_count(o_release_output_channel_count),
        .o_release_spatial_base(o_release_spatial_base),
        .o_release_spatial_count(o_release_spatial_count),
        .o_release_active_pair(o_release_active_pair),
        .o_release_work_base_addr(o_release_work_base_addr),
        .o_busy(scheduler_busy), .o_done(scheduler_done),
        .o_sticky_error(scheduler_error)
    );

    unified_scnn_h40_observation_writeback_v1 #(
        .WORD_DEPTH(FEATURE_WORD_DEPTH),
        .ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .PHYSICAL_TILE_COUNT(PHYSICAL_TILE_COUNT),
        .ENABLE_FEATURE_COMMIT_TAP(ENABLE_FEATURE_COMMIT_TAP)
    ) u_observation_writeback (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_context_valid(scheduler_context_load_valid &&
                         i_context_load_ready),
        .o_context_ready(writeback_context_ready),
        .i_layer_tag(i_layer_tag),
        .i_output_channel_base(o_context_output_channel_base),
        .i_output_channel_count(o_context_output_channel_count),
        .i_spatial_base(o_context_spatial_base),
        .i_spatial_row(o_feature_output_row),
        .i_spatial_col(o_feature_output_col),
        .i_spatial_count(o_context_spatial_count),
        .i_out_width(i_out_width), .i_out_height(i_out_height),
        .i_pool_mode(i_pool_mode),
        .i_write_slot(i_feature_write_slot),
        .i_feature_base_addr(i_feature_write_base_addr),
        .i_row_stride_words(i_feature_write_row_stride_words),
        .i_channel_stride_words(
            i_feature_write_channel_stride_words),
        .i_observation_valid(i_array_observation_valid),
        .i_observation_accept(i_observation_ready),
        .o_observation_ready(writeback_observation_select),
        .i_observation_tile_flat(i_array_observation_tile_flat),
        .i_observation_mode_flat(i_array_observation_mode_flat),
        .i_observation_lane_mask_flat(
            i_array_observation_lane_mask_flat),
        .i_observation_spike_flat(i_array_observation_spike_flat),
        .i_observation_meta_flat(i_array_observation_meta_flat),
        .i_close_valid(scheduler_context_release_valid),
        .i_close_accept(scheduler_context_release_valid &&
                        writeback_close_ready &&
                        i_context_release_ready),
        .o_close_ready(writeback_close_ready),
        .o_update_valid(o_feature_update_valid),
        .i_update_ready(i_feature_update_ready),
        .o_update_slot(o_feature_update_slot),
        .o_update_addr(o_feature_update_addr),
        .o_update_data(o_feature_update_data),
        .o_update_mask(o_feature_update_mask),
        .i_update_done(i_feature_update_done),
        .i_update_error(i_feature_service_error),
        .o_feature_commit_valid(o_feature_commit_valid),
        .i_feature_commit_ready(i_feature_commit_ready),
        .o_feature_commit_layer_tag(o_feature_commit_layer_tag),
        .o_feature_commit_channel(o_feature_commit_channel),
        .o_feature_commit_spatial_base(o_feature_commit_spatial_base),
        .o_feature_commit_spatial_count(o_feature_commit_spatial_count),
        .o_feature_commit_spike(o_feature_commit_spike),
        .o_feature_commit_q8_8_flat(o_feature_commit_q8_8_flat),
        .o_busy(writeback_busy), .o_sticky_error(writeback_error)
    );
endmodule

`default_nettype wire
