`timescale 1ns/1ps
`default_nettype none

// Tile-local controller between one held H40 accumulator result, the shared
// state/trace RAM, and a regional ten-lane neuron service. The low two bits of
// request metadata carry the group index; the upper 62 bits remain available
// to identify profile, layer, timestep, sample, and parameter version.
module unified_dual_role_h40_group_sequencer_candidate_v1 #(
    parameter int MEMORY_DEPTH = 1024,
    parameter int MEMORY_ADDR_WIDTH = $clog2(MEMORY_DEPTH),
    parameter int WORK_ADDR_WIDTH = $clog2(MEMORY_DEPTH / 2),
    parameter bit LOCAL_MERGE = 1'b0
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_abort,
    input  wire logic                         i_clear_error,

    input  wire logic                         i_start_valid,
    output logic                              o_start_ready,
    input  wire logic [1:0]                   i_mode,
    input  wire logic [39:0]                  i_active_lane_mask,
    input  wire logic [WORK_ADDR_WIDTH-1:0]   i_work_base_addr,
    input  wire logic [40*48-1:0]             i_raw_current_flat,
    input  wire logic                         i_p4_fc,
    input  wire logic [40*48-1:0]             i_raw_current_secondary_flat,
    input  wire logic signed [15:0]           i_leak,
    input  wire logic signed [31:0]           i_threshold,
    input  wire logic                         i_reset_to_zero,
    input  wire logic [15:0]                  i_trace_decay,
    input  wire logic [61:0]                  i_transaction_meta,

    output logic                              o_region_valid,
    input  wire logic                         i_region_ready,
    output logic [1:0]                        o_region_mode,
    output logic [9:0]                        o_region_lane_mask,
    output logic [10*48-1:0]                  o_region_raw_current_flat,
    output logic                              o_region_p4_fc,
    output logic [10*48-1:0]                  o_region_raw_current_secondary_flat,
    output logic [10*32-1:0]                  o_region_state_flat,
    output logic [10*32-1:0]                  o_region_trace_old_flat,
    output logic signed [15:0]                o_region_leak,
    output logic signed [31:0]                o_region_threshold,
    output logic                              o_region_reset_to_zero,
    output logic [15:0]                       o_region_trace_decay,
    output logic [63:0]                       o_region_meta,

    input  wire logic                         i_commit_valid,
    output logic                              o_commit_ready,
    input  wire logic [1:0]                   i_commit_mode,
    input  wire logic [9:0]                   i_commit_lane_mask,
    input  wire logic [9:0]                   i_commit_range_mask,
    input  wire logic [9:0]                   i_commit_spike,
    input  wire logic [10*32-1:0]             i_commit_pre_state_flat,
    input  wire logic [10*32-1:0]             i_commit_next_state_flat,
    input  wire logic [10*32-1:0]             i_commit_trace_next_flat,
    input  wire logic [63:0]                  i_commit_meta,
    input  wire logic                         i_commit_mode_error,

    output logic                              o_observation_valid,
    input  wire logic                         i_observation_ready,
    output logic [1:0]                        o_observation_mode,
    output logic [9:0]                        o_observation_lane_mask,
    output logic [9:0]                        o_observation_range_mask,
    output logic [9:0]                        o_observation_spike,
    output logic [10*32-1:0]                  o_observation_pre_state_flat,
    output logic [10*32-1:0]                  o_observation_trace_next_flat,
    output logic [63:0]                       o_observation_meta,

    output logic                              o_state_read_valid,
    input  wire logic                         i_state_read_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]      o_state_read_addr,
    input  wire logic                         i_state_read_result_valid,
    output logic                              o_state_read_result_ready,
    input  wire logic [10*32-1:0]             i_state_read_data,
    output logic                              o_state_write_valid,
    input  wire logic                         i_state_write_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]      o_state_write_addr,
    output logic [10*32-1:0]                  o_state_write_data,
    output logic [39:0]                       o_state_write_mask,

    output logic                              o_trace_read_valid,
    input  wire logic                         i_trace_read_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]      o_trace_read_addr,
    input  wire logic                         i_trace_read_result_valid,
    output logic                              o_trace_read_result_ready,
    input  wire logic [10*32-1:0]             i_trace_read_data,
    output logic                              o_trace_write_valid,
    input  wire logic                         i_trace_write_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]      o_trace_write_addr,
    output logic [10*32-1:0]                  o_trace_write_data,
    output logic [39:0]                       o_trace_write_mask,

    output logic                              o_done,
    output logic                              o_idle,
    output logic                              o_sticky_error
);
    localparam logic [1:0] MODE_FC = 2'd0;
    localparam logic [1:0] MODE_SRNN = 2'd1;
    localparam logic [1:0] MODE_SCNN = 2'd2;
    localparam logic [1:0] MODE_M2_Q14 = 2'd3;
    localparam int HALF_DEPTH = MEMORY_DEPTH / 2;

    logic active_q;
    logic [1:0] mode_q;
    logic [39:0] active_lane_mask_q;
    logic [WORK_ADDR_WIDTH-1:0] work_base_addr_q;
    logic signed [15:0] leak_q;
    logic signed [31:0] threshold_q;
    logic reset_to_zero_q;
    logic [15:0] trace_decay_q;
    logic [61:0] transaction_meta_q;
    logic [2:0] groups_issued_q;
    logic [2:0] groups_committed_q;
    logic [3:0] active_group_mask_q;
    logic [2:0] active_group_count_q;
    logic read_outstanding_q;
    logic [1:0] pending_group_q;
    logic [3:0] inflight_groups_q;

    logic request_valid_q;
    logic [1:0] request_group_q;
    logic [9:0] request_lane_mask_q;
    logic [10*48-1:0] request_raw_current_flat_q;
    logic [10*48-1:0] request_raw_current_secondary_flat_q;
    logic [10*48-1:0] local_merged_current_flat;
    logic [10*32-1:0] request_state_flat_q;
    logic [10*32-1:0] request_trace_old_flat_q;

    logic start_legal;
    logic [WORK_ADDR_WIDTH:0] start_last_addr;
    logic start_fire;
    logic request_slot_ready;
    logic read_join_fire;
    logic can_issue_read;
    logic read_issue_fire;
    logic region_fire;
    logic [WORK_ADDR_WIDTH-1:0] read_logical_addr;
    logic [1:0] commit_group;
    logic commit_matches;
    logic commit_fire;
    logic commit_good_fire;
    logic commit_state_done_q;
    logic commit_trace_done_q;
    logic commit_observation_done_q;
    logic state_write_fire;
    logic trace_write_fire;
    logic observation_fire;
    logic commit_all_complete;
    logic [WORK_ADDR_WIDTH-1:0] commit_logical_addr;
    logic [3:0] inflight_groups_next;
    logic [3:0] start_group_mask;
    logic [2:0] start_group_count;
    logic current_issue_group_active;

    always_comb begin
        start_group_mask[0] = |i_active_lane_mask[9:0];
        start_group_mask[1] = |i_active_lane_mask[19:10];
        start_group_mask[2] = |i_active_lane_mask[29:20];
        start_group_mask[3] = |i_active_lane_mask[39:30];
        start_group_count = 3'(start_group_mask[0]) +
            3'(start_group_mask[1]) + 3'(start_group_mask[2]) +
            3'(start_group_mask[3]);

        start_last_addr = {1'b0, i_work_base_addr};
        if (i_mode == MODE_SCNN) begin
            if (start_group_mask[3] || start_group_mask[2])
                start_last_addr = start_last_addr +
                    (WORK_ADDR_WIDTH+1)'(1);
        end else if (start_group_mask[3]) begin
            start_last_addr = start_last_addr + (WORK_ADDR_WIDTH+1)'(3);
        end else if (start_group_mask[2]) begin
            start_last_addr = start_last_addr + (WORK_ADDR_WIDTH+1)'(2);
        end else if (start_group_mask[1]) begin
            start_last_addr = start_last_addr + (WORK_ADDR_WIDTH+1)'(1);
        end
        start_legal = (i_mode == MODE_FC || i_mode == MODE_SRNN ||
            i_mode == MODE_SCNN || i_mode == MODE_M2_Q14) &&
            start_group_count != 3'd0 &&
            start_last_addr < (WORK_ADDR_WIDTH+1)'(HALF_DEPTH);
    end

    assign o_idle = !active_q && !request_valid_q &&
        !read_outstanding_q && !i_state_read_result_valid &&
        !i_trace_read_result_valid;
    assign o_start_ready = o_idle && !o_sticky_error;
    assign start_fire = i_start_valid && o_start_ready;

    assign request_slot_ready = !request_valid_q || i_region_ready;
    assign read_join_fire = active_q && i_state_read_result_valid &&
        i_trace_read_result_valid && request_slot_ready;
    assign o_state_read_result_ready = active_q && request_slot_ready &&
        i_trace_read_result_valid;
    assign o_trace_read_result_ready = active_q && request_slot_ready &&
        i_state_read_result_valid;

    assign current_issue_group_active =
        active_group_mask_q[groups_issued_q[1:0]];
    assign can_issue_read = active_q && groups_issued_q < 3'd4 &&
        current_issue_group_active &&
        (!read_outstanding_q || read_join_fire);
    assign o_state_read_valid = can_issue_read && i_trace_read_ready;
    assign o_trace_read_valid = can_issue_read && i_state_read_ready;
    assign read_issue_fire = can_issue_read && i_state_read_ready &&
        i_trace_read_ready;

    always_comb begin
        if (mode_q == MODE_SCNN)
            read_logical_addr = work_base_addr_q +
                WORK_ADDR_WIDTH'(groups_issued_q[1]);
        else
            read_logical_addr = work_base_addr_q +
                WORK_ADDR_WIDTH'(groups_issued_q[1:0]);
    end
    assign o_state_read_addr = {1'b0, read_logical_addr};
    assign o_trace_read_addr = {1'b1, read_logical_addr};

    assign o_region_valid = !i_abort && request_valid_q;
    assign o_region_mode = mode_q;
    assign o_region_lane_mask = request_lane_mask_q;
    assign o_region_raw_current_flat = LOCAL_MERGE ?
        local_merged_current_flat : request_raw_current_flat_q;
    // The parent accumulator holds its result until o_done. Keep only the
    // active ten-lane group here instead of copying both complete H40 buses.
    assign o_region_p4_fc = LOCAL_MERGE ? 1'b0 : i_p4_fc;
    assign o_region_raw_current_secondary_flat = LOCAL_MERGE ? '0 :
        request_raw_current_secondary_flat_q;
    assign o_region_state_flat = request_state_flat_q;
    assign o_region_trace_old_flat = request_trace_old_flat_q;
    assign o_region_leak = leak_q;
    assign o_region_threshold = threshold_q;
    assign o_region_reset_to_zero = reset_to_zero_q;
    assign o_region_trace_decay = trace_decay_q;
    assign o_region_meta = {transaction_meta_q, request_group_q};
    assign region_fire = o_region_valid && i_region_ready;

    generate
        if (LOCAL_MERGE) begin : g_local_merge
            for (genvar lane = 0; lane < 10; lane++) begin : g_lane
                unified_dual_role_dsp_add48_lane_candidate_v1 u_merge (
                    .i_clk(i_clk),
                    .i_fire(read_join_fire &&
                        active_lane_mask_q[pending_group_q*10 + lane]),
                    .i_add_secondary(i_p4_fc),
                    .i_primary($signed(i_raw_current_flat[
                        (pending_group_q*10 + lane)*48 +: 48])),
                    .i_secondary($signed(i_raw_current_secondary_flat[
                        (pending_group_q*10 + lane)*48 +: 48])),
                    .o_result(local_merged_current_flat[
                        lane*48 +: 48])
                );
            end
        end else begin : g_no_local_merge
            assign local_merged_current_flat = '0;
        end
    endgenerate

    assign commit_group = i_commit_meta[1:0];
    assign commit_matches = active_q &&
        i_commit_meta[63:2] == transaction_meta_q &&
        i_commit_mode == mode_q && inflight_groups_q[commit_group] &&
        !i_commit_mode_error;
    // The commit payload remains stable in the upstream elastic stage until
    // all three consumers have accepted it. Per-sink completion bits make the
    // fork atomic without feeding downstream ready through another sink's
    // valid or duplicating the wide state/trace payload.
    assign o_state_write_valid = !i_abort && i_commit_valid &&
        commit_matches && !commit_state_done_q;
    assign o_trace_write_valid = !i_abort && i_commit_valid &&
        commit_matches && !commit_trace_done_q;
    assign o_observation_valid = !i_abort && i_commit_valid &&
        commit_matches && !commit_observation_done_q;
    assign state_write_fire = o_state_write_valid && i_state_write_ready;
    assign trace_write_fire = o_trace_write_valid && i_trace_write_ready;
    assign observation_fire = o_observation_valid && i_observation_ready;
    assign commit_all_complete =
        (commit_state_done_q || state_write_fire) &&
        (commit_trace_done_q || trace_write_fire) &&
        (commit_observation_done_q || observation_fire);
    assign o_commit_ready = !i_abort && active_q &&
        (!commit_matches || commit_all_complete);
    assign commit_fire = i_commit_valid && o_commit_ready;
    assign commit_good_fire = commit_fire && commit_matches;

    assign o_observation_mode = i_commit_mode;
    assign o_observation_lane_mask = i_commit_lane_mask;
    assign o_observation_range_mask = i_commit_range_mask;
    assign o_observation_spike = i_commit_spike & i_commit_lane_mask;
    assign o_observation_pre_state_flat = i_commit_pre_state_flat;
    assign o_observation_trace_next_flat = i_commit_trace_next_flat;
    assign o_observation_meta = i_commit_meta;

    always_comb begin
        if (mode_q == MODE_SCNN)
            commit_logical_addr = work_base_addr_q +
                WORK_ADDR_WIDTH'(commit_group[1]);
        else
            commit_logical_addr = work_base_addr_q +
                WORK_ADDR_WIDTH'(commit_group);

        o_state_write_addr = {1'b0, commit_logical_addr};
        o_trace_write_addr = {1'b1, commit_logical_addr};
        o_state_write_data = '0;
        o_trace_write_data = '0;
        o_state_write_mask = '0;
        o_trace_write_mask = '0;

        for (int lane = 0; lane < 10; lane++) begin
            logic [8:0] bit_offset;
            logic [5:0] byte_offset;
            bit_offset = '0;
            byte_offset = '0;
            if (i_commit_lane_mask[lane]) begin
                if (mode_q == MODE_SCNN) begin
                    bit_offset = 9'(lane) << 4;
                    byte_offset = 6'(lane) << 1;
                    if (commit_group[0]) begin
                        bit_offset = bit_offset + 9'd160;
                        byte_offset = byte_offset + 6'd20;
                    end
                    o_state_write_data[bit_offset +: 16] =
                        i_commit_next_state_flat[lane*32 +: 16];
                    o_trace_write_data[bit_offset +: 16] =
                        i_commit_trace_next_flat[lane*32 +: 16];
                    o_state_write_mask[byte_offset +: 2] = 2'b11;
                    o_trace_write_mask[byte_offset +: 2] = 2'b11;
                end else begin
                    o_state_write_data[lane*32 +: 32] =
                        i_commit_next_state_flat[lane*32 +: 32];
                    o_trace_write_data[lane*32 +: 32] =
                        i_commit_trace_next_flat[lane*32 +: 32];
                    o_state_write_mask[lane*4 +: 4] = 4'b1111;
                    o_trace_write_mask[lane*4 +: 4] = 4'b1111;
                end
            end
        end
    end

    always_comb begin
        inflight_groups_next = inflight_groups_q;
        if (region_fire)
            inflight_groups_next[request_group_q] = 1'b1;
        if (commit_good_fire)
            inflight_groups_next[commit_group] = 1'b0;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            active_q <= 1'b0;
            mode_q <= MODE_FC;
            active_lane_mask_q <= '0;
            work_base_addr_q <= '0;
            leak_q <= '0;
            threshold_q <= '0;
            reset_to_zero_q <= 1'b0;
            trace_decay_q <= '0;
            transaction_meta_q <= '0;
            groups_issued_q <= '0;
            groups_committed_q <= '0;
            active_group_mask_q <= '0;
            active_group_count_q <= '0;
            read_outstanding_q <= 1'b0;
            pending_group_q <= '0;
            inflight_groups_q <= '0;
            request_valid_q <= 1'b0;
            request_group_q <= '0;
            request_lane_mask_q <= '0;
            commit_state_done_q <= 1'b0;
            commit_trace_done_q <= 1'b0;
            commit_observation_done_q <= 1'b0;
            o_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_clear_error && o_idle)
                o_sticky_error <= 1'b0;

            if (i_abort) begin
                active_q <= 1'b0;
                groups_issued_q <= '0;
                groups_committed_q <= '0;
                active_group_mask_q <= '0;
                active_group_count_q <= '0;
                read_outstanding_q <= 1'b0;
                inflight_groups_q <= '0;
                request_valid_q <= 1'b0;
                commit_state_done_q <= 1'b0;
                commit_trace_done_q <= 1'b0;
                commit_observation_done_q <= 1'b0;
            end else begin
                if (start_fire) begin
                    if (!start_legal) begin
                        o_sticky_error <= 1'b1;
                    end else begin
                        active_q <= 1'b1;
                        mode_q <= i_mode;
                        active_lane_mask_q <= i_active_lane_mask;
                        work_base_addr_q <= i_work_base_addr;
                        leak_q <= i_leak;
                        threshold_q <= i_threshold;
                        reset_to_zero_q <= i_reset_to_zero;
                        trace_decay_q <= i_trace_decay;
                        transaction_meta_q <= i_transaction_meta;
                        groups_issued_q <= '0;
                        groups_committed_q <= '0;
                        active_group_mask_q <= start_group_mask;
                        active_group_count_q <= start_group_count;
                        read_outstanding_q <= 1'b0;
                        inflight_groups_q <= '0;
                        request_valid_q <= 1'b0;
                        commit_state_done_q <= 1'b0;
                        commit_trace_done_q <= 1'b0;
                        commit_observation_done_q <= 1'b0;
                    end
                end

                if (commit_fire) begin
                    commit_state_done_q <= 1'b0;
                    commit_trace_done_q <= 1'b0;
                    commit_observation_done_q <= 1'b0;
                end else begin
                    if (state_write_fire)
                        commit_state_done_q <= 1'b1;
                    if (trace_write_fire)
                        commit_trace_done_q <= 1'b1;
                    if (observation_fire)
                        commit_observation_done_q <= 1'b1;
                end

                if (request_slot_ready) begin
                    request_valid_q <= read_join_fire;
                    if (read_join_fire) begin
                        request_group_q <= pending_group_q;
                        request_lane_mask_q <=
                            active_lane_mask_q[pending_group_q*10 +: 10];
                    end
                end

                case ({read_issue_fire, read_join_fire})
                    2'b10: read_outstanding_q <= 1'b1;
                    2'b01: read_outstanding_q <= 1'b0;
                    default: read_outstanding_q <= read_outstanding_q;
                endcase
                if (read_issue_fire) begin
                    pending_group_q <= groups_issued_q[1:0];
                    groups_issued_q <= groups_issued_q + 1'b1;
                end else if (active_q && groups_issued_q < 3'd4 &&
                        !current_issue_group_active) begin
                    groups_issued_q <= groups_issued_q + 1'b1;
                end

                inflight_groups_q <= inflight_groups_next;
                if (commit_fire && !commit_matches)
                    o_sticky_error <= 1'b1;
                if (commit_good_fire) begin
                    groups_committed_q <= groups_committed_q + 1'b1;
                    if (groups_committed_q + 1'b1 ==
                            active_group_count_q) begin
                        active_q <= 1'b0;
                        o_done <= 1'b1;
                    end
                end
            end
        end
    end

    // The request payload is valid only when request_valid_q is asserted.  Keep
    // these wide data registers reset-free so abort/reset controls do not fan
    // out into the physical trace/state/current payload banks.
    always_ff @(posedge i_clk) begin
        if (request_slot_ready && read_join_fire) begin
            for (int lane = 0; lane < 10; lane++) begin
                request_raw_current_flat_q[lane*48 +: 48] <=
                    i_raw_current_flat[(pending_group_q*10 + lane)*48 +: 48];
                request_raw_current_secondary_flat_q[lane*48 +: 48] <=
                    i_raw_current_secondary_flat[
                        (pending_group_q*10 + lane)*48 +: 48];
                if (mode_q == MODE_SCNN) begin
                    if (pending_group_q[0]) begin
                        request_state_flat_q[lane*32 +: 32] <=
                            {{16{i_state_read_data[160 + lane*16 + 15]}},
                             i_state_read_data[160 + lane*16 +: 16]};
                        request_trace_old_flat_q[lane*32 +: 32] <=
                            {{16{i_trace_read_data[160 + lane*16 + 15]}},
                             i_trace_read_data[160 + lane*16 +: 16]};
                    end else begin
                        request_state_flat_q[lane*32 +: 32] <=
                            {{16{i_state_read_data[lane*16 + 15]}},
                             i_state_read_data[lane*16 +: 16]};
                        request_trace_old_flat_q[lane*32 +: 32] <=
                            {{16{i_trace_read_data[lane*16 + 15]}},
                             i_trace_read_data[lane*16 +: 16]};
                    end
                end else begin
                    request_state_flat_q[lane*32 +: 32] <=
                        i_state_read_data[lane*32 +: 32];
                    request_trace_old_flat_q[lane*32 +: 32] <=
                        i_trace_read_data[lane*32 +: 32];
                end
            end
        end
    end

    initial begin
        if (MEMORY_DEPTH != 1024 || MEMORY_ADDR_WIDTH != 10 ||
            WORK_ADDR_WIDTH != 9)
            $error("group sequencer requires a 512+512 row working-set map");
    end
endmodule

`default_nettype wire
