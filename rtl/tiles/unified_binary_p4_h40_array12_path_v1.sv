`timescale 1ns/1ps
`default_nettype none

// Complete binary FC/SRNN/SCNN path through one physical 12xH40 array. The
// external four-region vectors are retained only as a compatibility boundary;
// slot three is permanently inactive and owns no compute or storage.
module unified_binary_p4_h40_array12_path_v1 #(
    parameter int SCNN_WEIGHT_FRACTIONAL_BITS = 8,
    parameter int WEIGHT_DEPTH = 1156,
    parameter int WEIGHT_ADDR_WIDTH = $clog2(WEIGHT_DEPTH),
    parameter int MEMORY_DEPTH = 1024,
    parameter int MEMORY_ADDR_WIDTH = $clog2(MEMORY_DEPTH),
    parameter int WORK_ADDR_WIDTH = $clog2(MEMORY_DEPTH / 2),
    parameter bit CUT_RECORD_READY = 1'b1
) (
    input  wire logic                           i_clk,
    input  wire logic                           i_rst_n,
    input  wire logic                           i_abort,
    input  wire logic                           i_clear_error,

    input  wire logic                           i_command_valid,
    output logic                                o_command_ready,
    input  wire logic [1:0]                     i_mode,
    input  wire logic                           i_fc_weight_int8,
    input  wire logic                           i_fc_current_shift6,
    input  wire logic [15:0]                    i_neuron_count,
    input  wire logic                           i_first,
    input  wire logic                           i_last,
    input  wire logic [3:0]                     i_source_valid,
    input  wire logic [5:0]                     i_scnn_spatial_count,
    input  wire logic [9:0]                     i_scnn_output_channel_base,
    input  wire logic [39:0]                    i_scnn_feature_mask0,
    input  wire logic [39:0]                    i_scnn_feature_mask1,
    input  wire logic [39:0]                    i_scnn_feature_mask2,
    input  wire logic [39:0]                    i_scnn_feature_mask3,
    input  wire logic signed [17:0]             i_scnn_factor_b,
    input  wire logic                           i_scnn_active_pair,
    input  wire logic [3:0]                     i_scnn_weight_subword0,
    input  wire logic [3:0]                     i_scnn_weight_subword1,
    input  wire logic [3:0]                     i_scnn_weight_subword2,
    input  wire logic [3:0]                     i_scnn_weight_subword3,
    input  wire logic [WORK_ADDR_WIDTH-1:0]     i_scnn_work_base_addr,
    input  wire logic [3:0]                     i_destination_tile_base,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]   i_weight_addr0,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]   i_weight_addr1,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]   i_weight_addr2,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]   i_weight_addr3,
    input  wire logic [15:0]                    i_leak,
    input  wire logic [31:0]                    i_threshold,
    input  wire logic                           i_reset_to_zero,
    input  wire logic [15:0]                    i_trace_decay,
    input  wire logic [51:0]                    i_transaction_tag,

    input  wire logic [3:0]                     i_weight_fill_valid,
    output logic [3:0]                          o_weight_fill_ready,
    input  wire logic [7:0]                     i_weight_fill_tile_flat,
    input  wire logic [3:0]                     i_weight_fill_bank,
    input  wire logic [4*WEIGHT_ADDR_WIDTH-1:0] i_weight_fill_addr_flat,
    input  wire logic [799:0]                   i_weight_fill_data_flat,

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

    input  wire logic [3:0]                     i_maintenance_tile,
    input  wire logic                           i_state_maintenance_write_valid,
    output logic                                o_state_maintenance_write_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]   i_state_maintenance_write_addr,
    input  wire logic [319:0]                   i_state_maintenance_write_data,
    input  wire logic [39:0]                    i_state_maintenance_write_mask,
    input  wire logic                           i_state_maintenance_read_valid,
    output logic                                o_state_maintenance_read_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]   i_state_maintenance_read_addr,
    output logic                                o_state_maintenance_result_valid,
    input  wire logic                           i_state_maintenance_result_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]        o_state_maintenance_result_addr,
    output logic [319:0]                        o_state_maintenance_read_data,

    input  wire logic                           i_trace_maintenance_write_valid,
    output logic                                o_trace_maintenance_write_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]   i_trace_maintenance_write_addr,
    input  wire logic [319:0]                   i_trace_maintenance_write_data,
    input  wire logic [39:0]                    i_trace_maintenance_write_mask,
    input  wire logic                           i_trace_maintenance_read_valid,
    output logic                                o_trace_maintenance_read_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]   i_trace_maintenance_read_addr,
    output logic                                o_trace_maintenance_result_valid,
    input  wire logic                           i_trace_maintenance_result_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]        o_trace_maintenance_result_addr,
    output logic [319:0]                        o_trace_maintenance_read_data,

    output logic                                o_busy,
    output logic                                o_done,
    output logic                                o_idle,
    output logic [15:0]                         o_tile_done,
    output logic [15:0]                         o_tile_idle,
    output logic [15:0]                         o_tile_sticky_error,
    output logic                                o_sticky_error
);
    logic walker_command_ready;
    logic walker_busy;
    logic walker_done;
    logic walker_sticky_error;
    logic [15:0] issue_valid;
    logic [15:0] issue_ready;
    logic [15:0] issue_first;
    logic [15:0] issue_last;
    logic [639:0] lane_mask_flat;
    logic [639:0] enable_a_flat;
    logic [639:0] enable_d_flat;
    logic [639:0] enable2_flat;
    logic [639:0] enable3_flat;
    logic [11519:0] factor_b_flat;
    logic [31:0] mode_flat;
    logic [15:0] fc_weight_int8;
    logic [15:0] scnn_active_bank;
    logic [16*WEIGHT_ADDR_WIDTH-1:0] weight_addr0_flat;
    logic [16*WEIGHT_ADDR_WIDTH-1:0] weight_addr1_flat;
    logic [16*WEIGHT_ADDR_WIDTH-1:0] weight_addr2_flat;
    logic [16*WEIGHT_ADDR_WIDTH-1:0] weight_addr3_flat;
    logic [63:0] weight_subword0_flat;
    logic [63:0] weight_subword1_flat;
    logic [63:0] weight_subword2_flat;
    logic [63:0] weight_subword3_flat;
    logic [16*WORK_ADDR_WIDTH-1:0] work_base_addr_flat;
    logic [255:0] leak_flat;
    logic [511:0] threshold_flat;
    logic [15:0] reset_to_zero_flat;
    logic [255:0] trace_decay_flat;
    logic [991:0] transaction_meta_flat;
    logic array_idle;
    logic [2:0] array_weight_fill_ready;
    logic [11:0] array_issue_ready;
    logic [2:0] array_observation_valid;
    logic [11:0] array_observation_tile_flat;
    logic [5:0] array_observation_mode_flat;
    logic [29:0] array_observation_lane_mask_flat;
    logic [29:0] array_observation_range_mask_flat;
    logic [29:0] array_observation_spike_flat;
    logic [959:0] array_observation_pre_state_flat;
    logic [959:0] array_observation_trace_next_flat;
    logic [191:0] array_observation_meta_flat;
    logic [11:0] array_tile_done;
    logic [11:0] array_tile_idle;
    logic [11:0] array_tile_sticky_error;
    logic active_q;
    logic transaction_open_q;
    logic dispatch_done_q;
    logic done_q;
    logic command_last_q;
    logic path_sticky_error_q;
    logic command_fire;
    logic command_context_legal;
    logic walker_command_valid;

    logic [1:0] transaction_mode_q;
    logic transaction_fc_weight_int8_q;
    logic transaction_fc_current_shift6_q;
    logic [15:0] transaction_neuron_count_q;
    logic [5:0] transaction_scnn_spatial_count_q;
    logic [9:0] transaction_scnn_output_channel_base_q;
    logic transaction_scnn_active_pair_q;
    logic [WORK_ADDR_WIDTH-1:0] transaction_scnn_work_base_addr_q;
    logic [3:0] transaction_destination_tile_base_q;
    logic [15:0] transaction_leak_q;
    logic [31:0] transaction_threshold_q;
    logic transaction_reset_to_zero_q;
    logic [15:0] transaction_trace_decay_q;
    logic [51:0] transaction_tag_q;

    assign command_context_legal = transaction_open_q ?
        (!i_first && i_mode == transaction_mode_q &&
         (i_fc_weight_int8 === 1'b1) == transaction_fc_weight_int8_q &&
         (i_fc_current_shift6 === 1'b1) == transaction_fc_current_shift6_q &&
         i_neuron_count == transaction_neuron_count_q &&
         i_scnn_spatial_count == transaction_scnn_spatial_count_q &&
         i_scnn_output_channel_base ==
             transaction_scnn_output_channel_base_q &&
         i_scnn_active_pair == transaction_scnn_active_pair_q &&
          i_scnn_work_base_addr == transaction_scnn_work_base_addr_q &&
          i_destination_tile_base ==
              transaction_destination_tile_base_q &&
         i_leak == transaction_leak_q &&
         i_threshold == transaction_threshold_q &&
         i_reset_to_zero == transaction_reset_to_zero_q &&
         i_trace_decay == transaction_trace_decay_q &&
         i_transaction_tag == transaction_tag_q) : i_first;
    assign o_command_ready = !active_q && walker_command_ready && !i_abort &&
        !path_sticky_error_q && (transaction_open_q || array_idle);
    assign command_fire = i_command_valid && o_command_ready;
    assign walker_command_valid = command_fire && command_context_legal;
    assign o_busy = active_q || transaction_open_q;
    assign o_done = done_q;
    assign o_idle = !active_q && !transaction_open_q &&
        !walker_busy && array_idle;
    assign o_sticky_error = path_sticky_error_q || walker_sticky_error |
        (|o_tile_sticky_error);
    assign issue_ready = {4'b0000, array_issue_ready};
    assign o_weight_fill_ready = {1'b0, array_weight_fill_ready};
    assign o_observation_valid = {1'b0, array_observation_valid};
    assign o_observation_tile_flat = {4'b0000,
        array_observation_tile_flat};
    assign o_observation_mode_flat = {2'b00,
        array_observation_mode_flat};
    assign o_observation_lane_mask_flat = {10'b0,
        array_observation_lane_mask_flat};
    assign o_observation_range_mask_flat = {10'b0,
        array_observation_range_mask_flat};
    assign o_observation_spike_flat = {10'b0,
        array_observation_spike_flat};
    assign o_observation_pre_state_flat = {320'b0,
        array_observation_pre_state_flat};
    assign o_observation_trace_next_flat = {320'b0,
        array_observation_trace_next_flat};
    assign o_observation_meta_flat = {64'b0,
        array_observation_meta_flat};
    assign o_tile_done = {4'b0000, array_tile_done};
    assign o_tile_idle = {4'b1111, array_tile_idle};
    assign o_tile_sticky_error = {4'b0000, array_tile_sticky_error};

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n || i_abort) begin
            active_q <= 1'b0;
            transaction_open_q <= 1'b0;
            dispatch_done_q <= 1'b0;
            done_q <= 1'b0;
            command_last_q <= 1'b0;
            path_sticky_error_q <= 1'b0;
            transaction_mode_q <= '0;
            transaction_fc_weight_int8_q <= 1'b0;
            transaction_fc_current_shift6_q <= 1'b0;
            transaction_neuron_count_q <= '0;
            transaction_scnn_spatial_count_q <= '0;
            transaction_scnn_output_channel_base_q <= '0;
            transaction_scnn_active_pair_q <= 1'b0;
            transaction_scnn_work_base_addr_q <= '0;
            transaction_destination_tile_base_q <= '0;
            transaction_leak_q <= '0;
            transaction_threshold_q <= '0;
            transaction_reset_to_zero_q <= 1'b0;
            transaction_trace_decay_q <= '0;
            transaction_tag_q <= '0;
        end else begin
            done_q <= 1'b0;

            if (i_clear_error && !active_q && !transaction_open_q &&
                array_idle)
                path_sticky_error_q <= 1'b0;

            if (command_fire && !command_context_legal) begin
                path_sticky_error_q <= 1'b1;
                done_q <= 1'b1;
            end else if (command_fire) begin
                active_q <= 1'b1;
                dispatch_done_q <= 1'b0;
                command_last_q <= i_last;
                if (!transaction_open_q) begin
                    transaction_open_q <= !i_last;
                    transaction_mode_q <= i_mode;
                    transaction_fc_weight_int8_q <=
                        i_fc_weight_int8 === 1'b1;
                    transaction_fc_current_shift6_q <=
                        i_fc_current_shift6 === 1'b1;
                    transaction_neuron_count_q <= i_neuron_count;
                    transaction_scnn_spatial_count_q <=
                        i_scnn_spatial_count;
                    transaction_scnn_output_channel_base_q <=
                        i_scnn_output_channel_base;
                    transaction_scnn_active_pair_q <= i_scnn_active_pair;
                    transaction_scnn_work_base_addr_q <=
                        i_scnn_work_base_addr;
                    transaction_destination_tile_base_q <=
                        i_destination_tile_base;
                    transaction_leak_q <= i_leak;
                    transaction_threshold_q <= i_threshold;
                    transaction_reset_to_zero_q <= i_reset_to_zero;
                    transaction_trace_decay_q <= i_trace_decay;
                    transaction_tag_q <= i_transaction_tag;
                end
            end else if (active_q) begin
                if (walker_done && !command_last_q) begin
                    active_q <= 1'b0;
                    dispatch_done_q <= 1'b0;
                    done_q <= 1'b1;
                end else begin
                    if (walker_done)
                        dispatch_done_q <= 1'b1;
                    if ((dispatch_done_q || walker_done) && array_idle) begin
                        active_q <= 1'b0;
                        transaction_open_q <= 1'b0;
                        dispatch_done_q <= 1'b0;
                        done_q <= 1'b1;
                    end
                end
            end
        end
    end

    unified_binary_p4_destination_walker_v1 #(
        .ARRAY_TILES(12),
        .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH)
    ) u_walker (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_command_valid(walker_command_valid),
        .o_command_ready(walker_command_ready),
        .i_mode(i_mode), .i_neuron_count(i_neuron_count),
        .i_fc_weight_int8(i_fc_weight_int8),
        .i_fc_current_shift6(i_fc_current_shift6),
        .i_first(i_first), .i_last(i_last),
        .i_source_valid(i_source_valid),
        .i_scnn_spatial_count(i_scnn_spatial_count),
        .i_scnn_output_channel_base(i_scnn_output_channel_base),
        .i_scnn_feature_mask0(i_scnn_feature_mask0),
        .i_scnn_feature_mask1(i_scnn_feature_mask1),
        .i_scnn_feature_mask2(i_scnn_feature_mask2),
        .i_scnn_feature_mask3(i_scnn_feature_mask3),
        .i_scnn_factor_b(i_scnn_factor_b),
        .i_scnn_active_pair(i_scnn_active_pair),
        .i_scnn_weight_subword0(i_scnn_weight_subword0),
        .i_scnn_weight_subword1(i_scnn_weight_subword1),
        .i_scnn_weight_subword2(i_scnn_weight_subword2),
        .i_scnn_weight_subword3(i_scnn_weight_subword3),
        .i_scnn_work_base_addr(i_scnn_work_base_addr),
        .i_destination_tile_base(i_destination_tile_base),
        .i_weight_addr0(i_weight_addr0),
        .i_weight_addr1(i_weight_addr1),
        .i_weight_addr2(i_weight_addr2),
        .i_weight_addr3(i_weight_addr3),
        .i_leak(i_leak), .i_threshold(i_threshold),
        .i_reset_to_zero(i_reset_to_zero),
        .i_trace_decay(i_trace_decay),
        .i_transaction_tag(i_transaction_tag),
        .o_issue_valid(issue_valid), .i_issue_ready(issue_ready),
        .o_first(issue_first), .o_last(issue_last),
        .o_lane_mask_flat(lane_mask_flat),
        .o_enable_a_flat(enable_a_flat),
        .o_enable_d_flat(enable_d_flat),
        .o_enable2_flat(enable2_flat),
        .o_enable3_flat(enable3_flat),
        .o_factor_b_flat(factor_b_flat),
        .o_mode_flat(mode_flat),
        .o_fc_weight_int8(fc_weight_int8),
        .o_scnn_active_bank(scnn_active_bank),
        .o_weight_addr0_flat(weight_addr0_flat),
        .o_weight_addr1_flat(weight_addr1_flat),
        .o_weight_addr2_flat(weight_addr2_flat),
        .o_weight_addr3_flat(weight_addr3_flat),
        .o_weight_subword0_flat(weight_subword0_flat),
        .o_weight_subword1_flat(weight_subword1_flat),
        .o_weight_subword2_flat(weight_subword2_flat),
        .o_weight_subword3_flat(weight_subword3_flat),
        .o_work_base_addr_flat(work_base_addr_flat),
        .o_leak_flat(leak_flat),
        .o_threshold_flat(threshold_flat),
        .o_reset_to_zero(reset_to_zero_flat),
        .o_trace_decay_flat(trace_decay_flat),
        .o_transaction_meta_flat(transaction_meta_flat),
        .o_busy(walker_busy), .o_done(walker_done),
        .o_sticky_error(walker_sticky_error)
    );

    unified_dual_role_h40_stationary_array12_candidate_v1 #(
        .SCNN_WEIGHT_FRACTIONAL_BITS(SCNN_WEIGHT_FRACTIONAL_BITS),
        .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .MEMORY_DEPTH(MEMORY_DEPTH),
        .MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH),
        .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH),
        .CUT_RECORD_READY(CUT_RECORD_READY)
    ) u_array (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_weight_fill_valid(i_weight_fill_valid[2:0]),
        .o_weight_fill_ready(array_weight_fill_ready),
        .i_weight_fill_tile_flat(i_weight_fill_tile_flat[5:0]),
        .i_weight_fill_bank(i_weight_fill_bank[2:0]),
        .i_weight_fill_addr_flat(
            i_weight_fill_addr_flat[3*WEIGHT_ADDR_WIDTH-1:0]),
        .i_weight_fill_data_flat(i_weight_fill_data_flat[599:0]),
        .i_issue_valid(issue_valid[11:0]),
        .o_issue_ready(array_issue_ready),
        .i_first(issue_first[11:0]), .i_last(issue_last[11:0]),
        .i_lane_mask_flat(lane_mask_flat[479:0]),
        .i_enable_a_flat(enable_a_flat[479:0]),
        .i_enable_d_flat(enable_d_flat[479:0]),
        .i_enable2_flat(enable2_flat[479:0]),
        .i_enable3_flat(enable3_flat[479:0]),
        .i_factor_b_flat(factor_b_flat[8639:0]),
        .i_mode_flat(mode_flat[23:0]),
        .i_fc_weight_int8(fc_weight_int8[11:0]),
        .i_scnn_active_bank(scnn_active_bank[11:0]),
        .i_weight_addr0_flat(weight_addr0_flat[
            12*WEIGHT_ADDR_WIDTH-1:0]),
        .i_weight_addr1_flat(weight_addr1_flat[
            12*WEIGHT_ADDR_WIDTH-1:0]),
        .i_weight_addr2_flat(weight_addr2_flat[
            12*WEIGHT_ADDR_WIDTH-1:0]),
        .i_weight_addr3_flat(weight_addr3_flat[
            12*WEIGHT_ADDR_WIDTH-1:0]),
        .i_weight_subword0_flat(weight_subword0_flat[47:0]),
        .i_weight_subword1_flat(weight_subword1_flat[47:0]),
        .i_weight_subword2_flat(weight_subword2_flat[47:0]),
        .i_weight_subword3_flat(weight_subword3_flat[47:0]),
        .i_work_base_addr_flat(work_base_addr_flat[
            12*WORK_ADDR_WIDTH-1:0]),
        .i_leak_flat(leak_flat[191:0]),
        .i_threshold_flat(threshold_flat[383:0]),
        .i_reset_to_zero(reset_to_zero_flat[11:0]),
        .i_trace_decay_flat(trace_decay_flat[191:0]),
        .i_transaction_meta_flat(transaction_meta_flat[743:0]),
        .o_observation_valid(array_observation_valid),
        .i_observation_ready(i_observation_ready[2:0]),
        .o_observation_tile_flat(array_observation_tile_flat),
        .o_observation_mode_flat(array_observation_mode_flat),
        .o_observation_lane_mask_flat(array_observation_lane_mask_flat),
        .o_observation_range_mask_flat(array_observation_range_mask_flat),
        .o_observation_spike_flat(array_observation_spike_flat),
        .o_observation_pre_state_flat(array_observation_pre_state_flat),
        .o_observation_trace_next_flat(array_observation_trace_next_flat),
        .o_observation_meta_flat(array_observation_meta_flat),
        .i_maintenance_tile(i_maintenance_tile),
        .i_state_maintenance_write_valid(i_state_maintenance_write_valid),
        .o_state_maintenance_write_ready(o_state_maintenance_write_ready),
        .i_state_maintenance_write_addr(i_state_maintenance_write_addr),
        .i_state_maintenance_write_data(i_state_maintenance_write_data),
        .i_state_maintenance_write_mask(i_state_maintenance_write_mask),
        .i_state_maintenance_read_valid(i_state_maintenance_read_valid),
        .o_state_maintenance_read_ready(o_state_maintenance_read_ready),
        .i_state_maintenance_read_addr(i_state_maintenance_read_addr),
        .o_state_maintenance_result_valid(o_state_maintenance_result_valid),
        .i_state_maintenance_result_ready(i_state_maintenance_result_ready),
        .o_state_maintenance_result_addr(o_state_maintenance_result_addr),
        .o_state_maintenance_read_data(o_state_maintenance_read_data),
        .i_trace_maintenance_write_valid(i_trace_maintenance_write_valid),
        .o_trace_maintenance_write_ready(o_trace_maintenance_write_ready),
        .i_trace_maintenance_write_addr(i_trace_maintenance_write_addr),
        .i_trace_maintenance_write_data(i_trace_maintenance_write_data),
        .i_trace_maintenance_write_mask(i_trace_maintenance_write_mask),
        .i_trace_maintenance_read_valid(i_trace_maintenance_read_valid),
        .o_trace_maintenance_read_ready(o_trace_maintenance_read_ready),
        .i_trace_maintenance_read_addr(i_trace_maintenance_read_addr),
        .o_trace_maintenance_result_valid(o_trace_maintenance_result_valid),
        .i_trace_maintenance_result_ready(i_trace_maintenance_result_ready),
        .o_trace_maintenance_result_addr(o_trace_maintenance_result_addr),
        .o_trace_maintenance_read_data(o_trace_maintenance_read_data),
        .o_done(array_tile_done), .o_tile_idle(array_tile_idle),
        .o_idle(array_idle), .o_sticky_error(array_tile_sticky_error)
    );
endmodule

`default_nettype wire
