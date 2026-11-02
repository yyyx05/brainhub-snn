`timescale 1ns/1ps
`default_nettype none

// Four homogeneous destination-stationary H40 tiles form one physical
// placement region. Contribution streams remain independent, while weight
// refill, working-set maintenance, neuron arithmetic, and observation egress
// are regional services.
module unified_dual_role_h40_stationary_region4_candidate_v1 #(
    parameter int SCNN_WEIGHT_FRACTIONAL_BITS = 8,
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
    parameter bit REGISTER_MAINTENANCE_REQUESTS = 1'b0,
    parameter bit LOCALIZE_ABORT = 1'b0,
    parameter bit CUT_RECORD_READY = 1'b0
) (
    input  wire logic                          i_clk,
    input  wire logic                          i_rst_n,
    input  wire logic                          i_abort,
    input  wire logic                          i_clear_error,

    input  wire logic                          i_weight_fill_valid,
    output logic                               o_weight_fill_ready,
    input  wire logic [1:0]                    i_weight_fill_tile,
    input  wire logic                          i_weight_fill_bank,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]  i_weight_fill_addr,
    input  wire logic [199:0]                  i_weight_fill_data,

    input  wire logic [3:0]                    i_issue_valid,
    output logic [3:0]                         o_issue_ready,
    input  wire logic [3:0]                    i_first,
    input  wire logic [3:0]                    i_last,
    input  wire logic [159:0]                  i_lane_mask_flat,
    input  wire logic [159:0]                  i_enable_a_flat,
    input  wire logic [159:0]                  i_enable_d_flat,
    input  wire logic [159:0]                  i_enable2_flat,
    input  wire logic [159:0]                  i_enable3_flat,
    input  wire logic [2879:0]                 i_factor_b_flat,
    input  wire logic [7:0]                    i_mode_flat,
    input  wire logic [3:0]                    i_fc_weight_int8,
    input  wire logic [3:0]                    i_scnn_active_bank,
    input  wire logic [4*WEIGHT_ADDR_WIDTH-1:0] i_weight_addr0_flat,
    input  wire logic [4*WEIGHT_ADDR_WIDTH-1:0] i_weight_addr1_flat,
    input  wire logic [4*WEIGHT_ADDR_WIDTH-1:0] i_weight_addr2_flat,
    input  wire logic [4*WEIGHT_ADDR_WIDTH-1:0] i_weight_addr3_flat,
    input  wire logic [15:0]                   i_weight_subword0_flat,
    input  wire logic [15:0]                   i_weight_subword1_flat,
    input  wire logic [15:0]                   i_weight_subword2_flat,
    input  wire logic [15:0]                   i_weight_subword3_flat,
    input  wire logic [4*WORK_ADDR_WIDTH-1:0]  i_work_base_addr_flat,
    input  wire logic [63:0]                   i_leak_flat,
    input  wire logic [127:0]                  i_threshold_flat,
    input  wire logic [3:0]                    i_reset_to_zero,
    input  wire logic [63:0]                   i_trace_decay_flat,
    input  wire logic [247:0]                  i_transaction_meta_flat,

    output logic                               o_observation_valid,
    input  wire logic                          i_observation_ready,
    output logic [1:0]                         o_observation_tile,
    output logic [1:0]                         o_observation_mode,
    output logic [9:0]                         o_observation_lane_mask,
    output logic [9:0]                         o_observation_range_mask,
    output logic [9:0]                         o_observation_spike,
    output logic [319:0]                       o_observation_pre_state,
    output logic [319:0]                       o_observation_trace_next,
    output logic [63:0]                        o_observation_meta,

    input  wire logic [1:0]                    i_maintenance_tile,
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

    output logic [3:0]                         o_done,
    output logic [3:0]                         o_tile_idle,
    output logic                               o_idle,
    output logic [3:0]                         o_sticky_error
);
    logic [3:0] tile_weight_fill_ready;
    logic [3:0] tile_issue_ready;
    logic [3:0] tile_abort;
    logic neuron_region_abort;
    logic [3:0] tile_region_valid;
    logic [3:0] tile_region_ready;
    logic [7:0] tile_region_mode_flat;
    logic [39:0] tile_region_lane_mask_flat;
    logic [1919:0] tile_region_raw_current_flat;
    logic [3:0] tile_region_p4_fc;
    logic [1919:0] tile_region_raw_current_secondary_flat;
    logic [1279:0] tile_region_state_flat;
    logic [1279:0] tile_region_trace_old_flat;
    logic [63:0] tile_region_leak_flat;
    logic [127:0] tile_region_threshold_flat;
    logic [3:0] tile_region_reset_to_zero;
    logic [63:0] tile_region_trace_decay_flat;
    logic [255:0] tile_region_meta_flat;

    logic region_record_valid;
    logic region_record_ready;
    logic [1:0] region_record_tile;
    logic [1:0] region_record_mode;
    logic [9:0] region_record_lane_mask;
    logic [9:0] region_record_range_mask;
    logic [9:0] region_record_spike;
    logic [319:0] region_record_pre_state;
    logic [319:0] region_record_next_state;
    logic [319:0] region_record_trace_next;
    logic [63:0] region_record_meta;
    logic region_record_mode_error;
    logic region_idle;
    logic [3:0] tile_commit_ready;

    logic [3:0] tile_observation_valid;
    logic [3:0] tile_observation_ready;
    logic [7:0] tile_observation_mode_flat;
    logic [39:0] tile_observation_lane_mask_flat;
    logic [39:0] tile_observation_range_mask_flat;
    logic [39:0] tile_observation_spike_flat;
    logic [1279:0] tile_observation_pre_state_flat;
    logic [1279:0] tile_observation_trace_next_flat;
    logic [255:0] tile_observation_meta_flat;

    logic [3:0] tile_state_write_ready;
    logic [3:0] tile_state_read_ready;
    logic [3:0] tile_state_result_valid;
    logic [4*MEMORY_ADDR_WIDTH-1:0] tile_state_result_addr_flat;
    logic [1279:0] tile_state_read_data_flat;
    logic [3:0] tile_trace_write_ready;
    logic [3:0] tile_trace_read_ready;
    logic [3:0] tile_trace_result_valid;
    logic [4*MEMORY_ADDR_WIDTH-1:0] tile_trace_result_addr_flat;
    logic [1279:0] tile_trace_read_data_flat;

    logic state_read_pending_q;
    logic trace_read_pending_q;
    logic [1:0] state_read_tile_q;
    logic [1:0] trace_read_tile_q;
    logic state_read_can_issue;
    logic trace_read_can_issue;
    logic state_read_fire;
    logic trace_read_fire;
    logic state_result_fire;
    logic trace_result_fire;

    logic [1:0] state_maintenance_tile;
    logic state_maintenance_write_valid;
    logic [MEMORY_ADDR_WIDTH-1:0] state_maintenance_write_addr;
    logic [319:0] state_maintenance_write_data;
    logic [39:0] state_maintenance_write_mask;
    logic state_maintenance_read_valid;
    logic [MEMORY_ADDR_WIDTH-1:0] state_maintenance_read_addr;
    logic [1:0] trace_maintenance_tile;
    logic trace_maintenance_write_valid;
    logic [MEMORY_ADDR_WIDTH-1:0] trace_maintenance_write_addr;
    logic [319:0] trace_maintenance_write_data;
    logic [39:0] trace_maintenance_write_mask;
    logic trace_maintenance_read_valid;
    logic [MEMORY_ADDR_WIDTH-1:0] trace_maintenance_read_addr;
    logic maintenance_commands_idle;
    logic observation_valid_unmasked;

    assign o_weight_fill_ready = !i_abort &&
        tile_weight_fill_ready[i_weight_fill_tile];
    assign o_issue_ready = tile_issue_ready & {4{!i_abort}};
    assign region_record_ready =
        tile_commit_ready[region_record_tile];

    // Maintenance is a refill/debug path, so keep one read outstanding per
    // memory class. Avoiding same-cycle response replacement also keeps ready
    // independent of the selected tile's response-valid path.
    assign state_read_can_issue = !state_read_pending_q;
    assign trace_read_can_issue = !trace_read_pending_q;
    assign state_read_fire = state_maintenance_read_valid &&
        state_read_can_issue && tile_state_read_ready[state_maintenance_tile];
    assign trace_read_fire = trace_maintenance_read_valid &&
        trace_read_can_issue && tile_trace_read_ready[trace_maintenance_tile];

    assign o_state_maintenance_result_valid = !i_abort &&
        state_read_pending_q &&
        tile_state_result_valid[state_read_tile_q];
    assign o_trace_maintenance_result_valid = !i_abort &&
        trace_read_pending_q &&
        tile_trace_result_valid[trace_read_tile_q];
    assign o_state_maintenance_result_addr =
        tile_state_result_addr_flat[
            state_read_tile_q*MEMORY_ADDR_WIDTH +: MEMORY_ADDR_WIDTH];
    assign o_trace_maintenance_result_addr =
        tile_trace_result_addr_flat[
            trace_read_tile_q*MEMORY_ADDR_WIDTH +: MEMORY_ADDR_WIDTH];
    assign o_state_maintenance_read_data =
        tile_state_read_data_flat[state_read_tile_q*320 +: 320];
    assign o_trace_maintenance_read_data =
        tile_trace_read_data_flat[trace_read_tile_q*320 +: 320];
    assign state_result_fire = o_state_maintenance_result_valid &&
        i_state_maintenance_result_ready;
    assign trace_result_fire = o_trace_maintenance_result_valid &&
        i_trace_maintenance_result_ready;
    assign o_observation_valid = observation_valid_unmasked && !i_abort;
    assign o_idle = (&o_tile_idle) && region_idle &&
        !state_read_pending_q && !trace_read_pending_q &&
        maintenance_commands_idle;

    // Abort is a soft transaction cancellation, not a rollback. External
    // ready/valid is suppressed immediately above, while each physical tile
    // and the regional neuron service receive a local one-cycle token. Work
    // already accepted before abort may commit; owner/valid state is flushed
    // locally without routing the external abort onto every memory enable.
    generate
        if (LOCALIZE_ABORT) begin : g_local_abort
            (* DONT_TOUCH = "yes" *) logic [3:0] tile_abort_q;
            (* DONT_TOUCH = "yes" *) logic neuron_region_abort_q;

            always_ff @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n) begin
                    tile_abort_q <= '0;
                    neuron_region_abort_q <= 1'b0;
                end else begin
                    tile_abort_q <= {4{i_abort}};
                    neuron_region_abort_q <= i_abort;
                end
            end

            assign tile_abort = tile_abort_q;
            assign neuron_region_abort = neuron_region_abort_q;
        end else begin : g_direct_abort
            assign tile_abort = {4{i_abort}};
            assign neuron_region_abort = i_abort;
        end
    endgenerate

    generate
        if (REGISTER_MAINTENANCE_REQUESTS) begin : g_registered_maintenance
            logic state_command_valid_q;
            logic state_command_write_q;
            logic [1:0] state_command_tile_q;
            logic [MEMORY_ADDR_WIDTH-1:0] state_command_addr_q;
            logic [319:0] state_command_data_q;
            logic [39:0] state_command_mask_q;
            logic state_command_issue;
            logic trace_command_valid_q;
            logic trace_command_write_q;
            logic [1:0] trace_command_tile_q;
            logic [MEMORY_ADDR_WIDTH-1:0] trace_command_addr_q;
            logic [319:0] trace_command_data_q;
            logic [39:0] trace_command_mask_q;
            logic trace_command_issue;

            assign o_state_maintenance_write_ready =
                !i_abort && !state_command_valid_q;
            assign o_state_maintenance_read_ready =
                !i_abort && !state_command_valid_q &&
                !i_state_maintenance_write_valid && !state_read_pending_q;
            assign o_trace_maintenance_write_ready =
                !i_abort && !trace_command_valid_q;
            assign o_trace_maintenance_read_ready =
                !i_abort && !trace_command_valid_q &&
                !i_trace_maintenance_write_valid && !trace_read_pending_q;

            assign state_maintenance_tile = state_command_tile_q;
            assign state_maintenance_write_valid =
                state_command_valid_q && state_command_write_q;
            assign state_maintenance_write_addr = state_command_addr_q;
            assign state_maintenance_write_data = state_command_data_q;
            assign state_maintenance_write_mask = state_command_mask_q;
            assign state_maintenance_read_valid =
                state_command_valid_q && !state_command_write_q;
            assign state_maintenance_read_addr = state_command_addr_q;

            assign trace_maintenance_tile = trace_command_tile_q;
            assign trace_maintenance_write_valid =
                trace_command_valid_q && trace_command_write_q;
            assign trace_maintenance_write_addr = trace_command_addr_q;
            assign trace_maintenance_write_data = trace_command_data_q;
            assign trace_maintenance_write_mask = trace_command_mask_q;
            assign trace_maintenance_read_valid =
                trace_command_valid_q && !trace_command_write_q;
            assign trace_maintenance_read_addr = trace_command_addr_q;

            assign state_command_issue = state_command_valid_q &&
                (state_command_write_q ?
                    tile_state_write_ready[state_command_tile_q] :
                    (state_read_can_issue &&
                     tile_state_read_ready[state_command_tile_q]));
            assign trace_command_issue = trace_command_valid_q &&
                (trace_command_write_q ?
                    tile_trace_write_ready[trace_command_tile_q] :
                    (trace_read_can_issue &&
                     tile_trace_read_ready[trace_command_tile_q]));
            assign maintenance_commands_idle =
                !state_command_valid_q && !trace_command_valid_q;

            always_ff @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n || i_abort) begin
                    state_command_valid_q <= 1'b0;
                    trace_command_valid_q <= 1'b0;
                end else begin
                    if (state_command_issue)
                        state_command_valid_q <= 1'b0;
                    if (i_state_maintenance_write_valid &&
                        o_state_maintenance_write_ready)
                        state_command_valid_q <= 1'b1;
                    else if (i_state_maintenance_read_valid &&
                             o_state_maintenance_read_ready)
                        state_command_valid_q <= 1'b1;

                    if (trace_command_issue)
                        trace_command_valid_q <= 1'b0;
                    if (i_trace_maintenance_write_valid &&
                        o_trace_maintenance_write_ready)
                        trace_command_valid_q <= 1'b1;
                    else if (i_trace_maintenance_read_valid &&
                             o_trace_maintenance_read_ready)
                        trace_command_valid_q <= 1'b1;
                end
            end

            // Payload registers are reset-free; their valid bits above own
            // the contents and suppress every stale command after reset.
            always_ff @(posedge i_clk) begin
                if (i_state_maintenance_write_valid &&
                    o_state_maintenance_write_ready) begin
                    state_command_write_q <= 1'b1;
                    state_command_tile_q <= i_maintenance_tile;
                    state_command_addr_q <= i_state_maintenance_write_addr;
                    state_command_data_q <= i_state_maintenance_write_data;
                    state_command_mask_q <= i_state_maintenance_write_mask;
                end else if (i_state_maintenance_read_valid &&
                             o_state_maintenance_read_ready) begin
                    state_command_write_q <= 1'b0;
                    state_command_tile_q <= i_maintenance_tile;
                    state_command_addr_q <= i_state_maintenance_read_addr;
                end

                if (i_trace_maintenance_write_valid &&
                    o_trace_maintenance_write_ready) begin
                    trace_command_write_q <= 1'b1;
                    trace_command_tile_q <= i_maintenance_tile;
                    trace_command_addr_q <= i_trace_maintenance_write_addr;
                    trace_command_data_q <= i_trace_maintenance_write_data;
                    trace_command_mask_q <= i_trace_maintenance_write_mask;
                end else if (i_trace_maintenance_read_valid &&
                             o_trace_maintenance_read_ready) begin
                    trace_command_write_q <= 1'b0;
                    trace_command_tile_q <= i_maintenance_tile;
                    trace_command_addr_q <= i_trace_maintenance_read_addr;
                end
            end
        end else begin : g_direct_maintenance
            assign o_state_maintenance_write_ready =
                !i_abort && tile_state_write_ready[i_maintenance_tile];
            assign o_state_maintenance_read_ready = !i_abort &&
                state_read_can_issue &&
                tile_state_read_ready[i_maintenance_tile];
            assign o_trace_maintenance_write_ready =
                !i_abort && tile_trace_write_ready[i_maintenance_tile];
            assign o_trace_maintenance_read_ready = !i_abort &&
                trace_read_can_issue &&
                tile_trace_read_ready[i_maintenance_tile];

            assign state_maintenance_tile = i_maintenance_tile;
            assign state_maintenance_write_valid =
                i_state_maintenance_write_valid;
            assign state_maintenance_write_addr =
                i_state_maintenance_write_addr;
            assign state_maintenance_write_data =
                i_state_maintenance_write_data;
            assign state_maintenance_write_mask =
                i_state_maintenance_write_mask;
            assign state_maintenance_read_valid =
                i_state_maintenance_read_valid;
            assign state_maintenance_read_addr =
                i_state_maintenance_read_addr;

            assign trace_maintenance_tile = i_maintenance_tile;
            assign trace_maintenance_write_valid =
                i_trace_maintenance_write_valid;
            assign trace_maintenance_write_addr =
                i_trace_maintenance_write_addr;
            assign trace_maintenance_write_data =
                i_trace_maintenance_write_data;
            assign trace_maintenance_write_mask =
                i_trace_maintenance_write_mask;
            assign trace_maintenance_read_valid =
                i_trace_maintenance_read_valid;
            assign trace_maintenance_read_addr =
                i_trace_maintenance_read_addr;
            assign maintenance_commands_idle = 1'b1;
        end
    endgenerate

    always_comb begin
        logic found;

        observation_valid_unmasked = 1'b0;
        o_observation_tile = '0;
        o_observation_mode = '0;
        o_observation_lane_mask = '0;
        o_observation_range_mask = '0;
        o_observation_spike = '0;
        o_observation_pre_state = '0;
        o_observation_trace_next = '0;
        o_observation_meta = '0;
        found = 1'b0;
        for (int tile = 0; tile < 4; tile++) begin
            // Abort masks the external valid immediately, but must not enter
            // this internal ready chain. The delayed local abort token flushes
            // valid ownership one cycle later; any record consumed here during
            // cancellation is intentionally discarded by the external mask.
            if (!found && tile_observation_valid[tile]) begin
                found = 1'b1;
                observation_valid_unmasked = 1'b1;
                o_observation_tile = tile[1:0];
                o_observation_mode =
                    tile_observation_mode_flat[tile*2 +: 2];
                o_observation_lane_mask =
                    tile_observation_lane_mask_flat[tile*10 +: 10];
                o_observation_range_mask =
                    tile_observation_range_mask_flat[tile*10 +: 10];
                o_observation_spike =
                    tile_observation_spike_flat[tile*10 +: 10];
                o_observation_pre_state =
                    tile_observation_pre_state_flat[tile*320 +: 320];
                o_observation_trace_next =
                    tile_observation_trace_next_flat[tile*320 +: 320];
                o_observation_meta =
                    tile_observation_meta_flat[tile*64 +: 64];
            end
        end
    end

    // Keep the request/payload selection process independent of downstream
    // ready. This is the same combinational arbiter, but its dependency graph
    // now preserves the standard valid-must-not-depend-on-ready contract.
    always_comb begin
        tile_observation_ready = '0;
        if (observation_valid_unmasked)
            tile_observation_ready[o_observation_tile] =
                i_observation_ready;
    end

    generate
        for (genvar tile = 0; tile < 4; tile++) begin : g_tile
            logic state_result_ready;
            logic trace_result_ready;

            assign state_result_ready = state_read_pending_q &&
                state_read_tile_q == tile &&
                i_state_maintenance_result_ready;
            assign trace_result_ready = trace_read_pending_q &&
                trace_read_tile_q == tile &&
                i_trace_maintenance_result_ready;

            unified_dual_role_h40_stationary_tile_candidate_v1 #(
                .WEIGHT_DEPTH(WEIGHT_DEPTH),
                .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
                .MEMORY_DEPTH(MEMORY_DEPTH),
                .MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH),
                .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH),
                .SUPPORT_FC_P4(SUPPORT_FC_P4),
                .LOCAL_FC_P4_MERGE(LOCAL_FC_P4_MERGE),
                .SUPPORT_P4_ALL(SUPPORT_P4_ALL),
                .LOCAL_P4_ALL_MERGE(LOCAL_P4_ALL_MERGE),
                .P4_ALL_CASCADE(P4_ALL_CASCADE)
            ) u_tile (
                .i_clk(i_clk), .i_rst_n(i_rst_n),
                .i_abort(tile_abort[tile]),
                .i_clear_error(i_clear_error),
                .i_weight_fill_valid(i_weight_fill_valid &&
                    i_weight_fill_tile == tile),
                .o_weight_fill_ready(tile_weight_fill_ready[tile]),
                .i_weight_fill_bank(i_weight_fill_bank),
                .i_weight_fill_addr(i_weight_fill_addr),
                .i_weight_fill_data(i_weight_fill_data),
                .i_issue_valid(i_issue_valid[tile]),
                .o_issue_ready(tile_issue_ready[tile]),
                .i_first(i_first[tile]), .i_last(i_last[tile]),
                .i_lane_mask(i_lane_mask_flat[tile*40 +: 40]),
                .i_enable_a(i_enable_a_flat[tile*40 +: 40]),
                .i_enable_d(i_enable_d_flat[tile*40 +: 40]),
                .i_enable2(i_enable2_flat[tile*40 +: 40]),
                .i_enable3(i_enable3_flat[tile*40 +: 40]),
                .i_factor_b_flat(i_factor_b_flat[tile*720 +: 720]),
                .i_mode(i_mode_flat[tile*2 +: 2]),
                .i_fc_weight_int8(i_fc_weight_int8[tile]),
                .i_scnn_active_bank(i_scnn_active_bank[tile]),
                .i_weight_addr0(i_weight_addr0_flat[
                    tile*WEIGHT_ADDR_WIDTH +: WEIGHT_ADDR_WIDTH]),
                .i_weight_addr1(i_weight_addr1_flat[
                    tile*WEIGHT_ADDR_WIDTH +: WEIGHT_ADDR_WIDTH]),
                .i_weight_addr2(i_weight_addr2_flat[
                    tile*WEIGHT_ADDR_WIDTH +: WEIGHT_ADDR_WIDTH]),
                .i_weight_addr3(i_weight_addr3_flat[
                    tile*WEIGHT_ADDR_WIDTH +: WEIGHT_ADDR_WIDTH]),
                .i_weight_subword0(i_weight_subword0_flat[tile*4 +: 4]),
                .i_weight_subword1(i_weight_subword1_flat[tile*4 +: 4]),
                .i_weight_subword2(i_weight_subword2_flat[tile*4 +: 4]),
                .i_weight_subword3(i_weight_subword3_flat[tile*4 +: 4]),
                .i_work_base_addr(i_work_base_addr_flat[
                    tile*WORK_ADDR_WIDTH +: WORK_ADDR_WIDTH]),
                .i_leak(i_leak_flat[tile*16 +: 16]),
                .i_threshold(i_threshold_flat[tile*32 +: 32]),
                .i_reset_to_zero(i_reset_to_zero[tile]),
                .i_trace_decay(i_trace_decay_flat[tile*16 +: 16]),
                .i_transaction_meta(i_transaction_meta_flat[
                    tile*62 +: 62]),
                .o_region_valid(tile_region_valid[tile]),
                .i_region_ready(tile_region_ready[tile]),
                .o_region_mode(tile_region_mode_flat[tile*2 +: 2]),
                .o_region_lane_mask(tile_region_lane_mask_flat[
                    tile*10 +: 10]),
                .o_region_raw_current_flat(tile_region_raw_current_flat[
                    tile*480 +: 480]),
                .o_region_p4_fc(tile_region_p4_fc[tile]),
                .o_region_raw_current_secondary_flat(
                    tile_region_raw_current_secondary_flat[
                        tile*480 +: 480]),
                .o_region_state_flat(tile_region_state_flat[
                    tile*320 +: 320]),
                .o_region_trace_old_flat(tile_region_trace_old_flat[
                    tile*320 +: 320]),
                .o_region_leak(tile_region_leak_flat[tile*16 +: 16]),
                .o_region_threshold(tile_region_threshold_flat[
                    tile*32 +: 32]),
                .o_region_reset_to_zero(tile_region_reset_to_zero[tile]),
                .o_region_trace_decay(tile_region_trace_decay_flat[
                    tile*16 +: 16]),
                .o_region_meta(tile_region_meta_flat[tile*64 +: 64]),
                .i_commit_valid(region_record_valid &&
                    region_record_tile == tile),
                .o_commit_ready(tile_commit_ready[tile]),
                .i_commit_mode(region_record_mode),
                .i_commit_lane_mask(region_record_lane_mask),
                .i_commit_range_mask(region_record_range_mask),
                .i_commit_spike(region_record_spike),
                .i_commit_pre_state_flat(region_record_pre_state),
                .i_commit_next_state_flat(region_record_next_state),
                .i_commit_trace_next_flat(region_record_trace_next),
                .i_commit_meta(region_record_meta),
                .i_commit_mode_error(region_record_mode_error),
                .o_observation_valid(tile_observation_valid[tile]),
                .i_observation_ready(tile_observation_ready[tile]),
                .o_observation_mode(tile_observation_mode_flat[
                    tile*2 +: 2]),
                .o_observation_lane_mask(tile_observation_lane_mask_flat[
                    tile*10 +: 10]),
                .o_observation_range_mask(tile_observation_range_mask_flat[
                    tile*10 +: 10]),
                .o_observation_spike(tile_observation_spike_flat[
                    tile*10 +: 10]),
                .o_observation_pre_state_flat(
                    tile_observation_pre_state_flat[tile*320 +: 320]),
                .o_observation_trace_next_flat(
                    tile_observation_trace_next_flat[tile*320 +: 320]),
                .o_observation_meta(tile_observation_meta_flat[
                    tile*64 +: 64]),
                .i_state_maintenance_write_valid(
                    state_maintenance_write_valid &&
                    state_maintenance_tile == tile),
                .o_state_maintenance_write_ready(
                    tile_state_write_ready[tile]),
                .i_state_maintenance_write_addr(
                    state_maintenance_write_addr),
                .i_state_maintenance_write_data(
                    state_maintenance_write_data),
                .i_state_maintenance_write_mask(
                    state_maintenance_write_mask),
                .i_state_maintenance_read_valid(
                    state_maintenance_read_valid &&
                    state_read_can_issue && state_maintenance_tile == tile),
                .o_state_maintenance_read_ready(
                    tile_state_read_ready[tile]),
                .i_state_maintenance_read_addr(
                    state_maintenance_read_addr),
                .o_state_maintenance_result_valid(
                    tile_state_result_valid[tile]),
                .i_state_maintenance_result_ready(state_result_ready),
                .o_state_maintenance_result_addr(
                    tile_state_result_addr_flat[
                        tile*MEMORY_ADDR_WIDTH +: MEMORY_ADDR_WIDTH]),
                .o_state_maintenance_read_data(
                    tile_state_read_data_flat[tile*320 +: 320]),
                .i_trace_maintenance_write_valid(
                    trace_maintenance_write_valid &&
                    trace_maintenance_tile == tile),
                .o_trace_maintenance_write_ready(
                    tile_trace_write_ready[tile]),
                .i_trace_maintenance_write_addr(
                    trace_maintenance_write_addr),
                .i_trace_maintenance_write_data(
                    trace_maintenance_write_data),
                .i_trace_maintenance_write_mask(
                    trace_maintenance_write_mask),
                .i_trace_maintenance_read_valid(
                    trace_maintenance_read_valid &&
                    trace_read_can_issue && trace_maintenance_tile == tile),
                .o_trace_maintenance_read_ready(
                    tile_trace_read_ready[tile]),
                .i_trace_maintenance_read_addr(
                    trace_maintenance_read_addr),
                .o_trace_maintenance_result_valid(
                    tile_trace_result_valid[tile]),
                .i_trace_maintenance_result_ready(trace_result_ready),
                .o_trace_maintenance_result_addr(
                    tile_trace_result_addr_flat[
                        tile*MEMORY_ADDR_WIDTH +: MEMORY_ADDR_WIDTH]),
                .o_trace_maintenance_read_data(
                    tile_trace_read_data_flat[tile*320 +: 320]),
                .o_done(o_done[tile]), .o_idle(o_tile_idle[tile]),
                .o_sticky_error(o_sticky_error[tile])
            );
        end
    endgenerate

    generate
        if ((SUPPORT_FC_P4 && !LOCAL_FC_P4_MERGE) ||
            (SUPPORT_P4_ALL && !P4_ALL_CASCADE &&
             !LOCAL_P4_ALL_MERGE)) begin : g_p4_neuron_region
            unified_dual_role_h40_neuron_region_p4fc_candidate_v1 #(
                .SCNN_WEIGHT_FRACTIONAL_BITS(SCNN_WEIGHT_FRACTIONAL_BITS)
            )
            u_neuron_region (
                .i_clk(i_clk), .i_rst_n(i_rst_n),
                .i_abort(neuron_region_abort),
                .i_tile_valid(tile_region_valid),
                .o_tile_ready(tile_region_ready),
                .i_tile_mode_flat(tile_region_mode_flat),
                .i_tile_lane_mask_flat(tile_region_lane_mask_flat),
                .i_tile_raw_current_flat(tile_region_raw_current_flat),
                .i_tile_p4_fc(tile_region_p4_fc),
                .i_tile_raw_current_secondary_flat(
                    tile_region_raw_current_secondary_flat),
                .i_tile_state_flat(tile_region_state_flat),
                .i_tile_trace_old_flat(tile_region_trace_old_flat),
                .i_tile_leak_flat(tile_region_leak_flat),
                .i_tile_threshold_flat(tile_region_threshold_flat),
                .i_tile_reset_to_zero(tile_region_reset_to_zero),
                .i_tile_trace_decay_flat(tile_region_trace_decay_flat),
                .i_tile_meta_flat(tile_region_meta_flat),
                .o_record_valid(region_record_valid),
                .i_record_ready(region_record_ready),
                .o_record_tile(region_record_tile),
                .o_record_mode(region_record_mode),
                .o_record_lane_mask(region_record_lane_mask),
                .o_record_range_mask(region_record_range_mask),
                .o_record_spike(region_record_spike),
                .o_record_pre_state(region_record_pre_state),
                .o_record_next_state(region_record_next_state),
                .o_record_trace_next(region_record_trace_next),
                .o_record_meta(region_record_meta),
                .o_record_mode_error(region_record_mode_error),
                .o_idle(region_idle)
            );
        end else begin : g_p2_neuron_region
            unified_dual_role_h40_neuron_region_candidate_v1 #(
                .SCNN_WEIGHT_FRACTIONAL_BITS(SCNN_WEIGHT_FRACTIONAL_BITS),
                .CUT_RECORD_READY(CUT_RECORD_READY)
            )
            u_neuron_region (
                .i_clk(i_clk), .i_rst_n(i_rst_n),
                .i_abort(neuron_region_abort),
                .i_tile_valid(tile_region_valid),
                .o_tile_ready(tile_region_ready),
                .i_tile_mode_flat(tile_region_mode_flat),
                .i_tile_lane_mask_flat(tile_region_lane_mask_flat),
                .i_tile_raw_current_flat(tile_region_raw_current_flat),
                .i_tile_state_flat(tile_region_state_flat),
                .i_tile_trace_old_flat(tile_region_trace_old_flat),
                .i_tile_leak_flat(tile_region_leak_flat),
                .i_tile_threshold_flat(tile_region_threshold_flat),
                .i_tile_reset_to_zero(tile_region_reset_to_zero),
                .i_tile_trace_decay_flat(tile_region_trace_decay_flat),
                .i_tile_meta_flat(tile_region_meta_flat),
                .o_record_valid(region_record_valid),
                .i_record_ready(region_record_ready),
                .o_record_tile(region_record_tile),
                .o_record_mode(region_record_mode),
                .o_record_lane_mask(region_record_lane_mask),
                .o_record_range_mask(region_record_range_mask),
                .o_record_spike(region_record_spike),
                .o_record_pre_state(region_record_pre_state),
                .o_record_next_state(region_record_next_state),
                .o_record_trace_next(region_record_trace_next),
                .o_record_meta(region_record_meta),
                .o_record_mode_error(region_record_mode_error),
                .o_idle(region_idle)
            );
        end
    endgenerate

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n || i_abort) begin
            state_read_pending_q <= 1'b0;
            trace_read_pending_q <= 1'b0;
            state_read_tile_q <= '0;
            trace_read_tile_q <= '0;
        end else begin
            if (state_result_fire)
                state_read_pending_q <= 1'b0;
            if (trace_result_fire)
                trace_read_pending_q <= 1'b0;
            if (state_read_fire) begin
                state_read_pending_q <= 1'b1;
                state_read_tile_q <= state_maintenance_tile;
            end
            if (trace_read_fire) begin
                trace_read_pending_q <= 1'b1;
                trace_read_tile_q <= trace_maintenance_tile;
            end
        end
    end

    initial begin
        if (P4_ALL_CASCADE && !SUPPORT_P4_ALL)
            $error("P4-all cascade requires the P4-all weight frontend");
        if (WEIGHT_DEPTH != 1156 || MEMORY_DEPTH != 1024)
            $error("stationary region requires measured cache geometries");
    end
endmodule

`default_nettype wire
