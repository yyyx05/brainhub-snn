`timescale 1ns/1ps
`default_nettype none

// Homogeneous destination-stationary H40 tile for the regional-neuron
// architecture. Weight cache, forty contribution accumulators, and the
// active state/trace working set remain local. Normalization, LIF, and trace
// arithmetic leave the tile through one ten-lane regional request stream.
module unified_dual_role_h40_stationary_tile_candidate_v1 #(
    parameter int M2_CACHE_DEPTH = 578,
    parameter int WEIGHT_DEPTH = 1156,
    parameter int WEIGHT_ADDR_WIDTH = $clog2(WEIGHT_DEPTH),
    parameter int MEMORY_DEPTH = 1024,
    parameter int MEMORY_ADDR_WIDTH = $clog2(MEMORY_DEPTH),
    parameter int WORK_ADDR_WIDTH = $clog2(MEMORY_DEPTH / 2),
    parameter bit SUPPORT_FC_P4 = 1'b0,
    parameter bit LOCAL_FC_P4_MERGE = 1'b0,
    parameter bit SUPPORT_P4_ALL = 1'b0,
    parameter bit LOCAL_P4_ALL_MERGE = 1'b0,
    parameter bit P4_ALL_CASCADE = 1'b0,
    parameter bit M2_WEIGHT_CONTEXT = 1'b0
) (
    input  wire logic                          i_clk,
    input  wire logic                          i_rst_n,
    input  wire logic                          i_abort,
    input  wire logic                          i_clear_error,

    input  wire logic                          i_weight_fill_context,
    input  wire logic                          i_weight_fill_valid,
    output logic                               o_weight_fill_ready,
    input  wire logic                          i_weight_fill_bank,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]  i_weight_fill_addr,
    input  wire logic [199:0]                  i_weight_fill_data,

    input  wire logic                          i_issue_valid,
    output logic                               o_issue_ready,
    input  wire logic                          i_first,
    input  wire logic                          i_last,
    input  wire logic [39:0]                   i_lane_mask,
    input  wire logic [39:0]                   i_enable_a,
    input  wire logic [39:0]                   i_enable_d,
    input  wire logic [39:0]                   i_enable2,
    input  wire logic [39:0]                   i_enable3,
    input  wire logic [40*18-1:0]              i_factor_b_flat,
    input  wire logic [1:0]                    i_mode,
    input  wire logic                          i_fc_weight_int8,
    input  wire logic                          i_scnn_active_bank,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]  i_weight_addr0,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]  i_weight_addr1,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]  i_weight_addr2,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]  i_weight_addr3,
    input  wire logic [3:0]                    i_weight_subword0,
    input  wire logic [3:0]                    i_weight_subword1,
    input  wire logic [3:0]                    i_weight_subword2,
    input  wire logic [3:0]                    i_weight_subword3,
    input  wire logic [WORK_ADDR_WIDTH-1:0]    i_work_base_addr,
    input  wire logic signed [15:0]            i_leak,
    input  wire logic signed [31:0]            i_threshold,
    input  wire logic                          i_reset_to_zero,
    input  wire logic [15:0]                   i_trace_decay,
    input  wire logic [61:0]                   i_transaction_meta,

    output logic                               o_region_valid,
    input  wire logic                          i_region_ready,
    output logic [1:0]                         o_region_mode,
    output logic [9:0]                         o_region_lane_mask,
    output logic [10*48-1:0]                   o_region_raw_current_flat,
    output logic                               o_region_p4_fc,
    output logic [10*48-1:0]                   o_region_raw_current_secondary_flat,
    output logic [10*32-1:0]                   o_region_state_flat,
    output logic [10*32-1:0]                   o_region_trace_old_flat,
    output logic signed [15:0]                 o_region_leak,
    output logic signed [31:0]                 o_region_threshold,
    output logic                               o_region_reset_to_zero,
    output logic [15:0]                        o_region_trace_decay,
    output logic [63:0]                        o_region_meta,

    input  wire logic                          i_commit_valid,
    output logic                               o_commit_ready,
    input  wire logic [1:0]                    i_commit_mode,
    input  wire logic [9:0]                    i_commit_lane_mask,
    input  wire logic [9:0]                    i_commit_range_mask,
    input  wire logic [9:0]                    i_commit_spike,
    input  wire logic [10*32-1:0]              i_commit_pre_state_flat,
    input  wire logic [10*32-1:0]              i_commit_next_state_flat,
    input  wire logic [10*32-1:0]              i_commit_trace_next_flat,
    input  wire logic [63:0]                   i_commit_meta,
    input  wire logic                          i_commit_mode_error,

    output logic                               o_observation_valid,
    input  wire logic                          i_observation_ready,
    output logic [1:0]                         o_observation_mode,
    output logic [9:0]                         o_observation_lane_mask,
    output logic [9:0]                         o_observation_range_mask,
    output logic [9:0]                         o_observation_spike,
    output logic [10*32-1:0]                   o_observation_pre_state_flat,
    output logic [10*32-1:0]                   o_observation_trace_next_flat,
    output logic [63:0]                        o_observation_meta,

    input  wire logic                          i_state_maintenance_write_valid,
    output logic                               o_state_maintenance_write_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]  i_state_maintenance_write_addr,
    input  wire logic [319:0]                  i_state_maintenance_write_data,
    input  wire logic [39:0]                   i_state_maintenance_write_mask,
    input  wire logic                          i_state_maintenance_read_valid,
    output logic                               o_state_maintenance_read_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]  i_state_maintenance_read_addr,
    output logic                               o_state_maintenance_result_valid,
    input  wire logic                          i_state_maintenance_result_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]       o_state_maintenance_result_addr,
    output logic [319:0]                       o_state_maintenance_read_data,

    input  wire logic                          i_trace_maintenance_write_valid,
    output logic                               o_trace_maintenance_write_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]  i_trace_maintenance_write_addr,
    input  wire logic [319:0]                  i_trace_maintenance_write_data,
    input  wire logic [39:0]                   i_trace_maintenance_write_mask,
    input  wire logic                          i_trace_maintenance_read_valid,
    output logic                               o_trace_maintenance_read_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]  i_trace_maintenance_read_addr,
    output logic                               o_trace_maintenance_result_valid,
    input  wire logic                          i_trace_maintenance_result_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]       o_trace_maintenance_result_addr,
    output logic [319:0]                       o_trace_maintenance_read_data,

    output logic                               o_done,
    output logic                               o_idle,
    output logic                               o_sticky_error
);
    logic weight_read_ready;
    logic weight_result_valid;
    logic weight_result_ready;
    logic weight_read_pending_q;
    logic discard_weight_q;
    logic [40*25-1:0] weight_term0_flat;
    logic [40*25-1:0] weight_term1_flat;
    logic [40*5-1:0] weight_term2_flat;
    logic [40*5-1:0] weight_term3_flat;
    logic [40*25-1:0] weight_term2_wide_flat;
    logic [40*25-1:0] weight_term3_wide_flat;
    logic issue_fire;

    logic first_q;
    logic last_q;
    logic [39:0] lane_mask_q;
    logic [39:0] enable_a_q;
    logic [39:0] enable_d_q;
    logic [39:0] enable2_q;
    logic [39:0] enable3_q;
    logic [40*18-1:0] factor_b_flat_q;
    logic [1:0] issue_mode_q;
    logic [WORK_ADDR_WIDTH-1:0] issue_work_base_addr_q;
    logic signed [15:0] issue_leak_q;
    logic signed [31:0] issue_threshold_q;
    logic issue_reset_to_zero_q;
    logic [15:0] issue_trace_decay_q;
    logic [61:0] issue_transaction_meta_q;

    logic transaction_active_q;
    logic contribution_open_q;
    logic neuron_started_q;
    logic protocol_error_q;
    logic transaction_sequence_legal;
    logic contribution_fire;
    logic accumulator_input_valid;
    logic accumulator_ready;
    logic accumulator_result_valid;
    logic accumulator_result_ready;
    logic [39:0] accumulator_result_lane_mask;
    logic [40*48-1:0] accumulator_result_flat;
    logic accumulator_result_p4_fc;
    logic [40*48-1:0] accumulator_result_secondary_flat;

    logic [1:0] transaction_mode_q;
    logic [WORK_ADDR_WIDTH-1:0] transaction_work_base_addr_q;
    logic signed [15:0] transaction_leak_q;
    logic signed [31:0] transaction_threshold_q;
    logic transaction_reset_to_zero_q;
    logic [15:0] transaction_trace_decay_q;
    logic [61:0] transaction_meta_q;

    logic sequencer_start_valid;
    logic sequencer_start_ready;
    logic sequencer_done;
    logic sequencer_idle;
    logic sequencer_error;

    logic seq_state_read_valid;
    logic seq_state_read_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] seq_state_read_addr;
    logic seq_state_read_result_valid;
    logic seq_state_read_result_ready;
    logic [319:0] seq_state_read_data;
    logic seq_state_write_valid;
    logic seq_state_write_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] seq_state_write_addr;
    logic [319:0] seq_state_write_data;
    logic [39:0] seq_state_write_mask;
    logic seq_trace_read_valid;
    logic seq_trace_read_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] seq_trace_read_addr;
    logic seq_trace_read_result_valid;
    logic seq_trace_read_result_ready;
    logic [319:0] seq_trace_read_data;
    logic seq_trace_write_valid;
    logic seq_trace_write_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] seq_trace_write_addr;
    logic [319:0] seq_trace_write_data;
    logic [39:0] seq_trace_write_mask;

    logic state_store_write_valid;
    logic state_store_write_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] state_store_write_addr;
    logic [319:0] state_store_write_data;
    logic [39:0] state_store_write_mask;
    logic state_store_read_valid;
    logic state_store_read_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] state_store_read_addr;
    logic state_store_result_valid;
    logic state_store_result_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] state_store_result_addr;
    logic [319:0] state_store_read_data;
    logic trace_store_write_valid;
    logic trace_store_write_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] trace_store_write_addr;
    logic [319:0] trace_store_write_data;
    logic [39:0] trace_store_write_mask;
    logic trace_store_read_valid;
    logic trace_store_read_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] trace_store_read_addr;
    logic trace_store_result_valid;
    logic trace_store_result_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] trace_store_result_addr;
    logic [319:0] trace_store_read_data;

    logic maintenance_owner;
    logic maintenance_request;
    logic maintenance_response_pending;
    logic maintenance_quiescent;

    assign o_sticky_error = protocol_error_q || sequencer_error;
    assign maintenance_request = i_state_maintenance_write_valid ||
        i_state_maintenance_read_valid ||
        i_trace_maintenance_write_valid ||
        i_trace_maintenance_read_valid;
    assign maintenance_response_pending = state_store_result_valid ||
        trace_store_result_valid;
    assign maintenance_quiescent = !transaction_active_q &&
        !contribution_open_q && !neuron_started_q &&
        !weight_read_pending_q && !discard_weight_q &&
        !weight_result_valid &&
        !accumulator_result_valid;
    assign maintenance_owner = maintenance_quiescent && !i_abort &&
        (maintenance_request || maintenance_response_pending);

    assign o_issue_ready = weight_read_ready && !i_abort &&
        !o_sticky_error && !discard_weight_q && !maintenance_owner;
    assign issue_fire = i_issue_valid && o_issue_ready;
    assign weight_result_ready = i_abort || discard_weight_q ||
        accumulator_ready;
    assign contribution_fire = weight_result_valid && weight_result_ready;
    assign transaction_sequence_legal =
        (!transaction_active_q && first_q) ||
        (transaction_active_q && contribution_open_q && !first_q);
    assign accumulator_input_valid = weight_result_valid &&
        transaction_sequence_legal && !discard_weight_q && !i_abort;

    assign sequencer_start_valid = accumulator_result_valid &&
        !neuron_started_q;
    assign accumulator_result_ready = sequencer_done && neuron_started_q;
    assign o_done = sequencer_done;
    assign o_idle = !transaction_active_q && !contribution_open_q &&
        !neuron_started_q && !weight_read_pending_q &&
        !discard_weight_q && !weight_result_valid &&
        !accumulator_result_valid &&
        !maintenance_request && !maintenance_response_pending &&
        sequencer_idle;

    generate
        if (SUPPORT_P4_ALL) begin : g_p4_all_frontend
            experimental_dual_context_weight_store_p4all_v1 #(
                .M2_CACHE_DEPTH(M2_CACHE_DEPTH),
                .ENABLE_M2_CONTEXT(M2_WEIGHT_CONTEXT)
            ) u_weight_store (
                .i_clk(i_clk), .i_rst_n(i_rst_n),
                .i_fill_context(i_weight_fill_context),
                .i_fill_valid(i_weight_fill_valid),
                .o_fill_ready(o_weight_fill_ready),
                .i_fill_bank({
                    i_weight_fill_addr[WEIGHT_ADDR_WIDTH-1],
                    i_weight_fill_bank}),
                .i_fill_addr(i_weight_fill_addr[9:0]),
                .i_fill_data(i_weight_fill_data),
                .i_read_valid(i_issue_valid && !i_abort &&
                              !o_sticky_error && !maintenance_owner),
                .o_read_ready(weight_read_ready), .i_mode(i_mode),
                .i_scnn_active_pair(i_scnn_active_bank),
                .i_fc_weight_int8(i_fc_weight_int8 === 1'b1),
                .i_read_addr0(i_weight_addr0[9:0]),
                .i_read_addr1(i_weight_addr1[9:0]),
                .i_read_addr2(i_weight_addr2[9:0]),
                .i_read_addr3(i_weight_addr3[9:0]),
                .i_read_subword0(i_weight_subword0),
                .i_read_subword1(i_weight_subword1),
                .i_read_subword2(i_weight_subword2),
                .i_read_subword3(i_weight_subword3),
                .o_read_result_valid(weight_result_valid),
                .i_read_result_ready(weight_result_ready),
                .o_term0_flat(weight_term0_flat),
                .o_term1_flat(weight_term1_flat),
                .o_term2_flat(weight_term2_wide_flat),
                .o_term3_flat(weight_term3_wide_flat)
            );

            assign weight_term2_flat = '0;
            assign weight_term3_flat = '0;

            if (P4_ALL_CASCADE) begin : g_cascade_accumulator
                unified_dual_role_h40_tile_p4all_cascade_accum_candidate_v1
                u_accumulator (
                    .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
                    .i_valid(accumulator_input_valid),
                    .o_ready(accumulator_ready),
                    .i_first(first_q), .i_last(last_q),
                    .i_lane_mask(lane_mask_q),
                    .i_enable0(enable_a_q), .i_enable1(enable_d_q),
                    .i_enable2(enable2_q), .i_enable3(enable3_q),
                    .i_term0_flat(weight_term0_flat),
                    .i_term1_flat(weight_term1_flat),
                    .i_term2_flat(weight_term2_wide_flat),
                    .i_term3_flat(weight_term3_wide_flat),
                    .i_factor_b_flat(factor_b_flat_q),
                    .o_result_valid(accumulator_result_valid),
                    .i_result_ready(accumulator_result_ready),
                    .o_result_lane_mask(accumulator_result_lane_mask),
                    .o_result_flat(accumulator_result_flat)
                );

                assign accumulator_result_p4_fc = 1'b0;
                assign accumulator_result_secondary_flat = '0;
            end else begin : g_dual_accumulator
                unified_dual_role_h40_tile_p4all_dual_accum_candidate_v1
                u_accumulator (
                    .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
                    .i_valid(accumulator_input_valid),
                    .o_ready(accumulator_ready),
                    .i_first(first_q), .i_last(last_q),
                    .i_use_secondary(1'b1),
                    .i_lane_mask(lane_mask_q),
                    .i_enable0(enable_a_q), .i_enable1(enable_d_q),
                    .i_enable2(enable2_q), .i_enable3(enable3_q),
                    .i_term0_flat(weight_term0_flat),
                    .i_term1_flat(weight_term1_flat),
                    .i_term2_flat(weight_term2_wide_flat),
                    .i_term3_flat(weight_term3_wide_flat),
                    .i_factor_b_flat(factor_b_flat_q),
                    .o_result_valid(accumulator_result_valid),
                    .i_result_ready(accumulator_result_ready),
                    .o_result_has_secondary(accumulator_result_p4_fc),
                    .o_result_lane_mask(accumulator_result_lane_mask),
                    .o_result_primary_flat(accumulator_result_flat),
                    .o_result_secondary_flat(
                        accumulator_result_secondary_flat)
                );
            end
        end else if (SUPPORT_FC_P4) begin : g_p4_frontend
            unified_dual_role_weight_store_p4fc_candidate_v1 #(
                .BANK_DEPTH(WEIGHT_DEPTH), .ADDR_WIDTH(WEIGHT_ADDR_WIDTH)
            ) u_weight_store (
                .i_clk(i_clk), .i_rst_n(i_rst_n),
                .i_fill_valid(i_weight_fill_valid),
                .o_fill_ready(o_weight_fill_ready),
                .i_fill_bank(i_weight_fill_bank),
                .i_fill_addr(i_weight_fill_addr),
                .i_fill_data(i_weight_fill_data),
                .i_read_valid(i_issue_valid && !i_abort &&
                              !o_sticky_error && !maintenance_owner),
                .o_read_ready(weight_read_ready), .i_mode(i_mode),
                .i_scnn_active_bank(i_scnn_active_bank),
                .i_read_addr0(i_weight_addr0),
                .i_read_addr1(i_weight_addr1),
                .i_read_addr2(i_weight_addr2),
                .i_read_addr3(i_weight_addr3),
                .i_read_subword0(i_weight_subword0),
                .i_read_subword1(i_weight_subword1),
                .o_read_result_valid(weight_result_valid),
                .i_read_result_ready(weight_result_ready),
                .o_term0_flat(weight_term0_flat),
                .o_term1_flat(weight_term1_flat),
                .o_term2_flat(weight_term2_flat),
                .o_term3_flat(weight_term3_flat)
            );

            assign weight_term2_wide_flat = '0;
            assign weight_term3_wide_flat = '0;

            unified_dual_role_h40_tile_p4fc_dual_accum_candidate_v1
            u_accumulator (
                .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
                .i_valid(accumulator_input_valid),
                .o_ready(accumulator_ready),
                .i_first(first_q), .i_last(last_q),
                .i_p4_fc(issue_mode_q == 2'd0),
                .i_lane_mask(lane_mask_q),
                .i_enable0(enable_a_q), .i_enable1(enable_d_q),
                .i_enable2(enable2_q), .i_enable3(enable3_q),
                .i_term0_flat(weight_term0_flat),
                .i_term1_flat(weight_term1_flat),
                .i_term2_flat(weight_term2_flat),
                .i_term3_flat(weight_term3_flat),
                .i_factor_b_flat(factor_b_flat_q),
                .o_result_valid(accumulator_result_valid),
                .i_result_ready(accumulator_result_ready),
                .o_result_p4_fc(accumulator_result_p4_fc),
                .o_result_lane_mask(accumulator_result_lane_mask),
                .o_result_primary_flat(accumulator_result_flat),
                .o_result_secondary_flat(
                    accumulator_result_secondary_flat)
            );
        end else begin : g_p2_frontend
            unified_dual_role_weight_store_candidate_v1 #(
                .BANK_DEPTH(WEIGHT_DEPTH), .ADDR_WIDTH(WEIGHT_ADDR_WIDTH)
            ) u_weight_store (
                .i_clk(i_clk), .i_rst_n(i_rst_n),
                .i_fill_valid(i_weight_fill_valid),
                .o_fill_ready(o_weight_fill_ready),
                .i_fill_bank(i_weight_fill_bank),
                .i_fill_addr(i_weight_fill_addr),
                .i_fill_data(i_weight_fill_data),
                .i_read_valid(i_issue_valid && !i_abort &&
                              !o_sticky_error && !maintenance_owner),
                .o_read_ready(weight_read_ready), .i_mode(i_mode),
                .i_scnn_active_bank(i_scnn_active_bank),
                .i_read_addr0(i_weight_addr0),
                .i_read_addr1(i_weight_addr1),
                .i_read_subword0(i_weight_subword0),
                .i_read_subword1(i_weight_subword1),
                .o_read_result_valid(weight_result_valid),
                .i_read_result_ready(weight_result_ready),
                .o_term0_flat(weight_term0_flat),
                .o_term1_flat(weight_term1_flat)
            );

            assign weight_term2_flat = '0;
            assign weight_term3_flat = '0;
            assign weight_term2_wide_flat = '0;
            assign weight_term3_wide_flat = '0;
            assign accumulator_result_p4_fc = 1'b0;
            assign accumulator_result_secondary_flat = '0;

            unified_dual_role_h40_tile_candidate_v3 u_accumulator (
                .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
                .i_valid(accumulator_input_valid),
                .o_ready(accumulator_ready),
                .i_first(first_q), .i_last(last_q),
                .i_lane_mask(lane_mask_q),
                .i_enable_a(enable_a_q), .i_enable_d(enable_d_q),
                .i_term_a_flat(weight_term0_flat),
                .i_term_d_flat(weight_term1_flat),
                .i_factor_b_flat(factor_b_flat_q),
                .o_result_valid(accumulator_result_valid),
                .i_result_ready(accumulator_result_ready),
                .o_result_lane_mask(accumulator_result_lane_mask),
                .o_result_flat(accumulator_result_flat)
            );
        end
    endgenerate

    unified_dual_role_h40_group_sequencer_candidate_v1 #(
        .LOCAL_MERGE(
            (SUPPORT_FC_P4 && LOCAL_FC_P4_MERGE) ||
            (SUPPORT_P4_ALL && LOCAL_P4_ALL_MERGE &&
             !P4_ALL_CASCADE))
    ) u_group_sequencer (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_start_valid(sequencer_start_valid),
        .o_start_ready(sequencer_start_ready),
        .i_mode(transaction_mode_q),
        .i_active_lane_mask(accumulator_result_lane_mask),
        .i_work_base_addr(transaction_work_base_addr_q),
        .i_raw_current_flat(accumulator_result_flat),
        .i_p4_fc(accumulator_result_p4_fc),
        .i_raw_current_secondary_flat(
            accumulator_result_secondary_flat),
        .i_leak(transaction_leak_q),
        .i_threshold(transaction_threshold_q),
        .i_reset_to_zero(transaction_reset_to_zero_q),
        .i_trace_decay(transaction_trace_decay_q),
        .i_transaction_meta(transaction_meta_q),
        .o_region_valid(o_region_valid), .i_region_ready(i_region_ready),
        .o_region_mode(o_region_mode),
        .o_region_lane_mask(o_region_lane_mask),
        .o_region_raw_current_flat(o_region_raw_current_flat),
        .o_region_p4_fc(o_region_p4_fc),
        .o_region_raw_current_secondary_flat(
            o_region_raw_current_secondary_flat),
        .o_region_state_flat(o_region_state_flat),
        .o_region_trace_old_flat(o_region_trace_old_flat),
        .o_region_leak(o_region_leak),
        .o_region_threshold(o_region_threshold),
        .o_region_reset_to_zero(o_region_reset_to_zero),
        .o_region_trace_decay(o_region_trace_decay),
        .o_region_meta(o_region_meta),
        .i_commit_valid(i_commit_valid), .o_commit_ready(o_commit_ready),
        .i_commit_mode(i_commit_mode),
        .i_commit_lane_mask(i_commit_lane_mask),
        .i_commit_range_mask(i_commit_range_mask),
        .i_commit_spike(i_commit_spike),
        .i_commit_pre_state_flat(i_commit_pre_state_flat),
        .i_commit_next_state_flat(i_commit_next_state_flat),
        .i_commit_trace_next_flat(i_commit_trace_next_flat),
        .i_commit_meta(i_commit_meta),
        .i_commit_mode_error(i_commit_mode_error),
        .o_observation_valid(o_observation_valid),
        .i_observation_ready(i_observation_ready),
        .o_observation_mode(o_observation_mode),
        .o_observation_lane_mask(o_observation_lane_mask),
        .o_observation_range_mask(o_observation_range_mask),
        .o_observation_spike(o_observation_spike),
        .o_observation_pre_state_flat(o_observation_pre_state_flat),
        .o_observation_trace_next_flat(o_observation_trace_next_flat),
        .o_observation_meta(o_observation_meta),
        .o_state_read_valid(seq_state_read_valid),
        .i_state_read_ready(seq_state_read_ready),
        .o_state_read_addr(seq_state_read_addr),
        .i_state_read_result_valid(seq_state_read_result_valid),
        .o_state_read_result_ready(seq_state_read_result_ready),
        .i_state_read_data(seq_state_read_data),
        .o_state_write_valid(seq_state_write_valid),
        .i_state_write_ready(seq_state_write_ready),
        .o_state_write_addr(seq_state_write_addr),
        .o_state_write_data(seq_state_write_data),
        .o_state_write_mask(seq_state_write_mask),
        .o_trace_read_valid(seq_trace_read_valid),
        .i_trace_read_ready(seq_trace_read_ready),
        .o_trace_read_addr(seq_trace_read_addr),
        .i_trace_read_result_valid(seq_trace_read_result_valid),
        .o_trace_read_result_ready(seq_trace_read_result_ready),
        .i_trace_read_data(seq_trace_read_data),
        .o_trace_write_valid(seq_trace_write_valid),
        .i_trace_write_ready(seq_trace_write_ready),
        .o_trace_write_addr(seq_trace_write_addr),
        .o_trace_write_data(seq_trace_write_data),
        .o_trace_write_mask(seq_trace_write_mask),
        .o_done(sequencer_done), .o_idle(sequencer_idle),
        .o_sticky_error(sequencer_error)
    );

    assign state_store_write_valid = maintenance_owner ?
        i_state_maintenance_write_valid : seq_state_write_valid;
    assign state_store_write_addr = maintenance_owner ?
        i_state_maintenance_write_addr : seq_state_write_addr;
    assign state_store_write_data = maintenance_owner ?
        i_state_maintenance_write_data : seq_state_write_data;
    assign state_store_write_mask = maintenance_owner ?
        i_state_maintenance_write_mask : seq_state_write_mask;
    assign state_store_read_valid = maintenance_owner ?
        i_state_maintenance_read_valid : seq_state_read_valid;
    assign state_store_read_addr = maintenance_owner ?
        i_state_maintenance_read_addr : seq_state_read_addr;
    assign state_store_result_ready = maintenance_owner ?
        i_state_maintenance_result_ready : seq_state_read_result_ready;
    assign seq_state_write_ready = !maintenance_owner &&
        state_store_write_ready;
    assign seq_state_read_ready = !maintenance_owner &&
        state_store_read_ready;
    assign seq_state_read_result_valid = !maintenance_owner &&
        state_store_result_valid;
    assign seq_state_read_data = state_store_read_data;
    assign o_state_maintenance_write_ready = maintenance_owner &&
        state_store_write_ready;
    assign o_state_maintenance_read_ready = maintenance_owner &&
        state_store_read_ready;
    assign o_state_maintenance_result_valid = maintenance_owner &&
        state_store_result_valid;
    assign o_state_maintenance_result_addr = state_store_result_addr;
    assign o_state_maintenance_read_data = state_store_read_data;

    assign trace_store_write_valid = maintenance_owner ?
        i_trace_maintenance_write_valid : seq_trace_write_valid;
    assign trace_store_write_addr = maintenance_owner ?
        i_trace_maintenance_write_addr : seq_trace_write_addr;
    assign trace_store_write_data = maintenance_owner ?
        i_trace_maintenance_write_data : seq_trace_write_data;
    assign trace_store_write_mask = maintenance_owner ?
        i_trace_maintenance_write_mask : seq_trace_write_mask;
    assign trace_store_read_valid = maintenance_owner ?
        i_trace_maintenance_read_valid : seq_trace_read_valid;
    assign trace_store_read_addr = maintenance_owner ?
        i_trace_maintenance_read_addr : seq_trace_read_addr;
    assign trace_store_result_ready = maintenance_owner ?
        i_trace_maintenance_result_ready : seq_trace_read_result_ready;
    assign seq_trace_write_ready = !maintenance_owner &&
        trace_store_write_ready;
    assign seq_trace_read_ready = !maintenance_owner &&
        trace_store_read_ready;
    assign seq_trace_read_result_valid = !maintenance_owner &&
        trace_store_result_valid;
    assign seq_trace_read_data = trace_store_read_data;
    assign o_trace_maintenance_write_ready = maintenance_owner &&
        trace_store_write_ready;
    assign o_trace_maintenance_read_ready = maintenance_owner &&
        trace_store_read_ready;
    assign o_trace_maintenance_result_valid = maintenance_owner &&
        trace_store_result_valid;
    assign o_trace_maintenance_result_addr = trace_store_result_addr;
    assign o_trace_maintenance_read_data = trace_store_read_data;

    unified_dual_role_state_trace_store_candidate_v2 #(
        .DEPTH(MEMORY_DEPTH), .ADDR_WIDTH(MEMORY_ADDR_WIDTH)
    ) u_state_trace_store (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_state_write_valid(state_store_write_valid),
        .o_state_write_ready(state_store_write_ready),
        .i_state_write_addr(state_store_write_addr),
        .i_state_write_data(state_store_write_data),
        .i_state_write_mask(state_store_write_mask),
        .i_state_read_valid(state_store_read_valid),
        .o_state_read_ready(state_store_read_ready),
        .i_state_read_addr(state_store_read_addr),
        .o_state_read_result_valid(state_store_result_valid),
        .i_state_read_result_ready(state_store_result_ready),
        .o_state_read_result_addr(state_store_result_addr),
        .o_state_read_data(state_store_read_data),
        .i_trace_write_valid(trace_store_write_valid),
        .o_trace_write_ready(trace_store_write_ready),
        .i_trace_write_addr(trace_store_write_addr),
        .i_trace_write_data(trace_store_write_data),
        .i_trace_write_mask(trace_store_write_mask),
        .i_trace_read_valid(trace_store_read_valid),
        .o_trace_read_ready(trace_store_read_ready),
        .i_trace_read_addr(trace_store_read_addr),
        .o_trace_read_result_valid(trace_store_result_valid),
        .i_trace_read_result_ready(trace_store_result_ready),
        .o_trace_read_result_addr(trace_store_result_addr),
        .o_trace_read_data(trace_store_read_data)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            first_q <= 1'b0;
            last_q <= 1'b0;
            lane_mask_q <= '0;
            enable_a_q <= '0;
            enable_d_q <= '0;
            enable2_q <= '0;
            enable3_q <= '0;
            factor_b_flat_q <= '0;
            issue_mode_q <= 2'd0;
            issue_work_base_addr_q <= '0;
            issue_leak_q <= '0;
            issue_threshold_q <= '0;
            issue_reset_to_zero_q <= 1'b0;
            issue_trace_decay_q <= '0;
            issue_transaction_meta_q <= '0;
            weight_read_pending_q <= 1'b0;
            discard_weight_q <= 1'b0;
            transaction_active_q <= 1'b0;
            contribution_open_q <= 1'b0;
            neuron_started_q <= 1'b0;
            protocol_error_q <= 1'b0;
            transaction_mode_q <= 2'd0;
            transaction_work_base_addr_q <= '0;
            transaction_leak_q <= '0;
            transaction_threshold_q <= '0;
            transaction_reset_to_zero_q <= 1'b0;
            transaction_trace_decay_q <= '0;
            transaction_meta_q <= '0;
        end else begin
            if (i_clear_error && o_idle)
                protocol_error_q <= 1'b0;

            if (i_abort) begin
                transaction_active_q <= 1'b0;
                contribution_open_q <= 1'b0;
                neuron_started_q <= 1'b0;
                if (weight_result_valid) begin
                    weight_read_pending_q <= 1'b0;
                    discard_weight_q <= 1'b0;
                end else if (weight_read_pending_q) begin
                    discard_weight_q <= 1'b1;
                end
            end else begin
                if (issue_fire)
                    weight_read_pending_q <= 1'b1;
                else if (contribution_fire) begin
                    weight_read_pending_q <= 1'b0;
                    if (discard_weight_q)
                        discard_weight_q <= 1'b0;
                end

                if (issue_fire) begin
                    first_q <= i_first;
                    last_q <= i_last;
                    lane_mask_q <= i_lane_mask;
                    enable_a_q <= i_enable_a;
                    enable_d_q <= i_enable_d;
                    enable2_q <= i_enable2;
                    enable3_q <= i_enable3;
                    factor_b_flat_q <= i_factor_b_flat;
                    issue_mode_q <= i_mode;
                    issue_work_base_addr_q <= i_work_base_addr;
                    issue_leak_q <= i_leak;
                    issue_threshold_q <= i_threshold;
                    issue_reset_to_zero_q <= i_reset_to_zero;
                    issue_trace_decay_q <= i_trace_decay;
                    issue_transaction_meta_q <= i_transaction_meta;
                end

                if (contribution_fire && !discard_weight_q &&
                    !transaction_sequence_legal)
                    protocol_error_q <= 1'b1;

                if (accumulator_input_valid) begin
                    if (first_q) begin
                        transaction_active_q <= 1'b1;
                        transaction_mode_q <= issue_mode_q;
                        transaction_work_base_addr_q <=
                            issue_work_base_addr_q;
                        transaction_leak_q <= issue_leak_q;
                        transaction_threshold_q <= issue_threshold_q;
                        transaction_reset_to_zero_q <=
                            issue_reset_to_zero_q;
                        transaction_trace_decay_q <= issue_trace_decay_q;
                        transaction_meta_q <= issue_transaction_meta_q;
                    end
                    contribution_open_q <= !last_q;
                end

                if (sequencer_start_valid && sequencer_start_ready)
                    neuron_started_q <= 1'b1;

                if (sequencer_done) begin
                    transaction_active_q <= 1'b0;
                    contribution_open_q <= 1'b0;
                    neuron_started_q <= 1'b0;
                end
            end

            if (sequencer_error)
                protocol_error_q <= 1'b1;
        end
    end

    initial begin
        if (SUPPORT_FC_P4 && SUPPORT_P4_ALL)
            $error("Select only one four-contribution weight frontend");
        if (P4_ALL_CASCADE && !SUPPORT_P4_ALL)
            $error("P4-all cascade requires the P4-all weight frontend");
        if (WEIGHT_ADDR_WIDTH < 11 && SUPPORT_P4_ALL)
            $error("P4-all refill encoding requires an 11-bit address");
        if (WEIGHT_DEPTH != 1156 || MEMORY_DEPTH != 1024)
            $error("stationary tile requires measured weight/workset geometries");
    end
endmodule

`default_nettype wire
