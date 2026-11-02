`timescale 1ns/1ps
`default_nettype none

// Native H480 array: three identical four-tile regions, with no inactive
// fourth region retained in RTL. FC, SRNN and SCNN share all twelve H40 tiles.
module unified_dual_role_h40_stationary_array12_candidate_v1 #(
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

    input  wire logic [2:0]                     i_weight_fill_valid,
    output logic [2:0]                          o_weight_fill_ready,
    input  wire logic [5:0]                     i_weight_fill_tile_flat,
    input  wire logic [2:0]                     i_weight_fill_bank,
    input  wire logic [3*WEIGHT_ADDR_WIDTH-1:0] i_weight_fill_addr_flat,
    input  wire logic [599:0]                   i_weight_fill_data_flat,

    input  wire logic [11:0]                    i_issue_valid,
    output logic [11:0]                         o_issue_ready,
    input  wire logic [11:0]                    i_first,
    input  wire logic [11:0]                    i_last,
    input  wire logic [479:0]                   i_lane_mask_flat,
    input  wire logic [479:0]                   i_enable_a_flat,
    input  wire logic [479:0]                   i_enable_d_flat,
    input  wire logic [479:0]                   i_enable2_flat,
    input  wire logic [479:0]                   i_enable3_flat,
    input  wire logic [8639:0]                  i_factor_b_flat,
    input  wire logic [23:0]                    i_mode_flat,
    input  wire logic [11:0]                    i_fc_weight_int8,
    input  wire logic [11:0]                    i_scnn_active_bank,
    input  wire logic [12*WEIGHT_ADDR_WIDTH-1:0] i_weight_addr0_flat,
    input  wire logic [12*WEIGHT_ADDR_WIDTH-1:0] i_weight_addr1_flat,
    input  wire logic [12*WEIGHT_ADDR_WIDTH-1:0] i_weight_addr2_flat,
    input  wire logic [12*WEIGHT_ADDR_WIDTH-1:0] i_weight_addr3_flat,
    input  wire logic [47:0]                    i_weight_subword0_flat,
    input  wire logic [47:0]                    i_weight_subword1_flat,
    input  wire logic [47:0]                    i_weight_subword2_flat,
    input  wire logic [47:0]                    i_weight_subword3_flat,
    input  wire logic [12*WORK_ADDR_WIDTH-1:0]  i_work_base_addr_flat,
    input  wire logic [191:0]                   i_leak_flat,
    input  wire logic [383:0]                   i_threshold_flat,
    input  wire logic [11:0]                    i_reset_to_zero,
    input  wire logic [191:0]                   i_trace_decay_flat,
    input  wire logic [743:0]                   i_transaction_meta_flat,

    output logic [2:0]                          o_observation_valid,
    input  wire logic [2:0]                     i_observation_ready,
    output logic [11:0]                         o_observation_tile_flat,
    output logic [5:0]                          o_observation_mode_flat,
    output logic [29:0]                         o_observation_lane_mask_flat,
    output logic [29:0]                         o_observation_range_mask_flat,
    output logic [29:0]                         o_observation_spike_flat,
    output logic [959:0]                        o_observation_pre_state_flat,
    output logic [959:0]                        o_observation_trace_next_flat,
    output logic [191:0]                        o_observation_meta_flat,

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

    output logic [11:0]                         o_done,
    output logic [11:0]                         o_tile_idle,
    output logic                                o_idle,
    output logic [11:0]                         o_sticky_error
);
    localparam int REGIONS = 3;
    localparam int TILES_PER_REGION = 4;

    logic [REGIONS-1:0] region_weight_fill_ready;
    logic [11:0] region_issue_ready_flat;
    logic [REGIONS-1:0] region_idle;

    logic [REGIONS-1:0] region_state_write_ready;
    logic [REGIONS-1:0] region_state_read_ready;
    logic [REGIONS-1:0] region_state_result_valid;
    logic [REGIONS*MEMORY_ADDR_WIDTH-1:0] region_state_result_addr_flat;
    logic [REGIONS*320-1:0] region_state_read_data_flat;
    logic [REGIONS-1:0] region_trace_write_ready;
    logic [REGIONS-1:0] region_trace_read_ready;
    logic [REGIONS-1:0] region_trace_result_valid;
    logic [REGIONS*MEMORY_ADDR_WIDTH-1:0] region_trace_result_addr_flat;
    logic [REGIONS*320-1:0] region_trace_read_data_flat;

    logic state_read_pending_q;
    logic trace_read_pending_q;
    logic [1:0] state_read_region_q;
    logic [1:0] trace_read_region_q;
    logic state_read_fire;
    logic trace_read_fire;
    logic state_result_fire;
    logic trace_result_fire;
    logic maintenance_region_valid;

    wire logic [1:0] selected_maintenance_region = i_maintenance_tile[3:2];

    assign maintenance_region_valid =
        selected_maintenance_region < 2'(REGIONS);
    assign o_weight_fill_ready = region_weight_fill_ready & {3{!i_abort}};
    assign o_issue_ready = region_issue_ready_flat & {12{!i_abort}};

    assign o_state_maintenance_write_ready = !i_abort &&
        maintenance_region_valid &&
        region_state_write_ready[selected_maintenance_region];
    assign o_trace_maintenance_write_ready = !i_abort &&
        maintenance_region_valid &&
        region_trace_write_ready[selected_maintenance_region];
    assign o_state_maintenance_read_ready = !i_abort &&
        maintenance_region_valid && !state_read_pending_q &&
        region_state_read_ready[selected_maintenance_region];
    assign o_trace_maintenance_read_ready = !i_abort &&
        maintenance_region_valid && !trace_read_pending_q &&
        region_trace_read_ready[selected_maintenance_region];
    assign state_read_fire = i_state_maintenance_read_valid &&
        o_state_maintenance_read_ready;
    assign trace_read_fire = i_trace_maintenance_read_valid &&
        o_trace_maintenance_read_ready;

    assign o_state_maintenance_result_valid = !i_abort &&
        state_read_pending_q &&
        region_state_result_valid[state_read_region_q];
    assign o_trace_maintenance_result_valid = !i_abort &&
        trace_read_pending_q &&
        region_trace_result_valid[trace_read_region_q];
    assign o_state_maintenance_result_addr =
        region_state_result_addr_flat[
            state_read_region_q*MEMORY_ADDR_WIDTH +: MEMORY_ADDR_WIDTH];
    assign o_trace_maintenance_result_addr =
        region_trace_result_addr_flat[
            trace_read_region_q*MEMORY_ADDR_WIDTH +: MEMORY_ADDR_WIDTH];
    assign o_state_maintenance_read_data =
        region_state_read_data_flat[state_read_region_q*320 +: 320];
    assign o_trace_maintenance_read_data =
        region_trace_read_data_flat[trace_read_region_q*320 +: 320];
    assign state_result_fire = o_state_maintenance_result_valid &&
        i_state_maintenance_result_ready;
    assign trace_result_fire = o_trace_maintenance_result_valid &&
        i_trace_maintenance_result_ready;
    assign o_idle = (&region_idle) && !state_read_pending_q &&
        !trace_read_pending_q;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n || i_abort) begin
            state_read_pending_q <= 1'b0;
            trace_read_pending_q <= 1'b0;
            state_read_region_q <= '0;
            trace_read_region_q <= '0;
        end else begin
            if (state_result_fire)
                state_read_pending_q <= 1'b0;
            if (trace_result_fire)
                trace_read_pending_q <= 1'b0;
            if (state_read_fire) begin
                state_read_pending_q <= 1'b1;
                state_read_region_q <= selected_maintenance_region;
            end
            if (trace_read_fire) begin
                trace_read_pending_q <= 1'b1;
                trace_read_region_q <= selected_maintenance_region;
            end
        end
    end

    generate
        for (genvar region = 0; region < REGIONS; region++) begin : g_region
            logic [1:0] observation_local_tile;
            logic state_result_ready;
            logic trace_result_ready;

            assign o_observation_tile_flat[region*4 +: 4] =
                {2'(region), observation_local_tile};
            assign state_result_ready = state_read_pending_q &&
                state_read_region_q == region &&
                i_state_maintenance_result_ready;
            assign trace_result_ready = trace_read_pending_q &&
                trace_read_region_q == region &&
                i_trace_maintenance_result_ready;

            unified_dual_role_h40_stationary_region4_candidate_v1 #(
                .SCNN_WEIGHT_FRACTIONAL_BITS(SCNN_WEIGHT_FRACTIONAL_BITS),
                .WEIGHT_DEPTH(WEIGHT_DEPTH),
                .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
                .MEMORY_DEPTH(MEMORY_DEPTH),
                .MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH),
                .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH),
                .SUPPORT_P4_ALL(1'b1),
                .P4_ALL_CASCADE(1'b1),
                .REGISTER_MAINTENANCE_REQUESTS(1'b1),
                .LOCALIZE_ABORT(1'b1),
                .CUT_RECORD_READY(CUT_RECORD_READY)
            ) u_region (
                .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
                .i_clear_error(i_clear_error),
                .i_weight_fill_valid(i_weight_fill_valid[region]),
                .o_weight_fill_ready(region_weight_fill_ready[region]),
                .i_weight_fill_tile(i_weight_fill_tile_flat[region*2 +: 2]),
                .i_weight_fill_bank(i_weight_fill_bank[region]),
                .i_weight_fill_addr(i_weight_fill_addr_flat[
                    region*WEIGHT_ADDR_WIDTH +: WEIGHT_ADDR_WIDTH]),
                .i_weight_fill_data(i_weight_fill_data_flat[
                    region*200 +: 200]),
                .i_issue_valid(i_issue_valid[
                    region*TILES_PER_REGION +: TILES_PER_REGION]),
                .o_issue_ready(region_issue_ready_flat[
                    region*TILES_PER_REGION +: TILES_PER_REGION]),
                .i_first(i_first[
                    region*TILES_PER_REGION +: TILES_PER_REGION]),
                .i_last(i_last[
                    region*TILES_PER_REGION +: TILES_PER_REGION]),
                .i_lane_mask_flat(i_lane_mask_flat[region*160 +: 160]),
                .i_enable_a_flat(i_enable_a_flat[region*160 +: 160]),
                .i_enable_d_flat(i_enable_d_flat[region*160 +: 160]),
                .i_enable2_flat(i_enable2_flat[region*160 +: 160]),
                .i_enable3_flat(i_enable3_flat[region*160 +: 160]),
                .i_factor_b_flat(i_factor_b_flat[region*2880 +: 2880]),
                .i_mode_flat(i_mode_flat[region*8 +: 8]),
                .i_fc_weight_int8(i_fc_weight_int8[region*4 +: 4]),
                .i_scnn_active_bank(i_scnn_active_bank[region*4 +: 4]),
                .i_weight_addr0_flat(i_weight_addr0_flat[
                    region*4*WEIGHT_ADDR_WIDTH +: 4*WEIGHT_ADDR_WIDTH]),
                .i_weight_addr1_flat(i_weight_addr1_flat[
                    region*4*WEIGHT_ADDR_WIDTH +: 4*WEIGHT_ADDR_WIDTH]),
                .i_weight_addr2_flat(i_weight_addr2_flat[
                    region*4*WEIGHT_ADDR_WIDTH +: 4*WEIGHT_ADDR_WIDTH]),
                .i_weight_addr3_flat(i_weight_addr3_flat[
                    region*4*WEIGHT_ADDR_WIDTH +: 4*WEIGHT_ADDR_WIDTH]),
                .i_weight_subword0_flat(i_weight_subword0_flat[
                    region*16 +: 16]),
                .i_weight_subword1_flat(i_weight_subword1_flat[
                    region*16 +: 16]),
                .i_weight_subword2_flat(i_weight_subword2_flat[
                    region*16 +: 16]),
                .i_weight_subword3_flat(i_weight_subword3_flat[
                    region*16 +: 16]),
                .i_work_base_addr_flat(i_work_base_addr_flat[
                    region*4*WORK_ADDR_WIDTH +: 4*WORK_ADDR_WIDTH]),
                .i_leak_flat(i_leak_flat[region*64 +: 64]),
                .i_threshold_flat(i_threshold_flat[region*128 +: 128]),
                .i_reset_to_zero(i_reset_to_zero[region*4 +: 4]),
                .i_trace_decay_flat(i_trace_decay_flat[region*64 +: 64]),
                .i_transaction_meta_flat(i_transaction_meta_flat[
                    region*248 +: 248]),
                .o_observation_valid(o_observation_valid[region]),
                .i_observation_ready(i_observation_ready[region]),
                .o_observation_tile(observation_local_tile),
                .o_observation_mode(o_observation_mode_flat[
                    region*2 +: 2]),
                .o_observation_lane_mask(o_observation_lane_mask_flat[
                    region*10 +: 10]),
                .o_observation_range_mask(o_observation_range_mask_flat[
                    region*10 +: 10]),
                .o_observation_spike(o_observation_spike_flat[
                    region*10 +: 10]),
                .o_observation_pre_state(o_observation_pre_state_flat[
                    region*320 +: 320]),
                .o_observation_trace_next(o_observation_trace_next_flat[
                    region*320 +: 320]),
                .o_observation_meta(o_observation_meta_flat[
                    region*64 +: 64]),
                .i_maintenance_tile(i_maintenance_tile[1:0]),
                .i_state_maintenance_write_valid(
                    i_state_maintenance_write_valid &&
                    maintenance_region_valid &&
                    selected_maintenance_region == region),
                .o_state_maintenance_write_ready(
                    region_state_write_ready[region]),
                .i_state_maintenance_write_addr(
                    i_state_maintenance_write_addr),
                .i_state_maintenance_write_data(
                    i_state_maintenance_write_data),
                .i_state_maintenance_write_mask(
                    i_state_maintenance_write_mask),
                .i_state_maintenance_read_valid(
                    i_state_maintenance_read_valid &&
                    maintenance_region_valid && !state_read_pending_q &&
                    selected_maintenance_region == region),
                .o_state_maintenance_read_ready(
                    region_state_read_ready[region]),
                .i_state_maintenance_read_addr(
                    i_state_maintenance_read_addr),
                .o_state_maintenance_result_valid(
                    region_state_result_valid[region]),
                .i_state_maintenance_result_ready(state_result_ready),
                .o_state_maintenance_result_addr(
                    region_state_result_addr_flat[
                        region*MEMORY_ADDR_WIDTH +: MEMORY_ADDR_WIDTH]),
                .o_state_maintenance_read_data(region_state_read_data_flat[
                    region*320 +: 320]),
                .i_trace_maintenance_write_valid(
                    i_trace_maintenance_write_valid &&
                    maintenance_region_valid &&
                    selected_maintenance_region == region),
                .o_trace_maintenance_write_ready(
                    region_trace_write_ready[region]),
                .i_trace_maintenance_write_addr(
                    i_trace_maintenance_write_addr),
                .i_trace_maintenance_write_data(
                    i_trace_maintenance_write_data),
                .i_trace_maintenance_write_mask(
                    i_trace_maintenance_write_mask),
                .i_trace_maintenance_read_valid(
                    i_trace_maintenance_read_valid &&
                    maintenance_region_valid && !trace_read_pending_q &&
                    selected_maintenance_region == region),
                .o_trace_maintenance_read_ready(
                    region_trace_read_ready[region]),
                .i_trace_maintenance_read_addr(
                    i_trace_maintenance_read_addr),
                .o_trace_maintenance_result_valid(
                    region_trace_result_valid[region]),
                .i_trace_maintenance_result_ready(trace_result_ready),
                .o_trace_maintenance_result_addr(
                    region_trace_result_addr_flat[
                        region*MEMORY_ADDR_WIDTH +: MEMORY_ADDR_WIDTH]),
                .o_trace_maintenance_read_data(region_trace_read_data_flat[
                    region*320 +: 320]),
                .o_done(o_done[region*4 +: 4]),
                .o_tile_idle(o_tile_idle[region*4 +: 4]),
                .o_idle(region_idle[region]),
                .o_sticky_error(o_sticky_error[region*4 +: 4])
            );
        end
    endgenerate

    initial begin
        if (WEIGHT_DEPTH != 1156 || MEMORY_DEPTH != 1024)
            $error("array12 candidate requires measured cache geometries");
    end
endmodule

`default_nettype wire
