`timescale 1ns/1ps
`default_nettype none

// Replays one group of up to four source terms across every resident
// destination H40 unit. FC/SRNN use one scalar source-valid bit per term.
// SCNN supplies one forty-position feature bitmap per tap plus one shared
// Q8.8 factor. Binary layers use 1.0; a value-coded first layer replays the
// same four P4 terms at successive power-of-two factors. The arithmetic and
// destination-state interfaces are identical in all three modes. Commands
// larger than the configured resident H40 context are rejected until an outer
// weight/state refill barrier has installed the next context. Each tile
// handshakes independently so backpressure cannot duplicate a contribution.
module unified_binary_p4_destination_walker_v1 #(
    parameter int ARRAY_TILES = 16,
    parameter int WEIGHT_DEPTH = 1156,
    parameter int WEIGHT_ADDR_WIDTH = $clog2(WEIGHT_DEPTH),
    parameter int WORK_ADDR_WIDTH = 9
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
    // Upper metadata is sample/layer/version context. The walker appends the
    // ten-bit logical H40 unit index for unambiguous observation routing.
    input  wire logic [51:0]                    i_transaction_tag,

    output logic [15:0]                         o_issue_valid,
    input  wire logic [15:0]                    i_issue_ready,
    output logic [15:0]                         o_first,
    output logic [15:0]                         o_last,
    output logic [639:0]                        o_lane_mask_flat,
    output logic [639:0]                        o_enable_a_flat,
    output logic [639:0]                        o_enable_d_flat,
    output logic [639:0]                        o_enable2_flat,
    output logic [639:0]                        o_enable3_flat,
    output logic [11519:0]                      o_factor_b_flat,
    output logic [31:0]                         o_mode_flat,
    output logic [15:0]                         o_fc_weight_int8,
    output logic [15:0]                         o_scnn_active_bank,
    output logic [16*WEIGHT_ADDR_WIDTH-1:0]     o_weight_addr0_flat,
    output logic [16*WEIGHT_ADDR_WIDTH-1:0]     o_weight_addr1_flat,
    output logic [16*WEIGHT_ADDR_WIDTH-1:0]     o_weight_addr2_flat,
    output logic [16*WEIGHT_ADDR_WIDTH-1:0]     o_weight_addr3_flat,
    output logic [63:0]                         o_weight_subword0_flat,
    output logic [63:0]                         o_weight_subword1_flat,
    output logic [63:0]                         o_weight_subword2_flat,
    output logic [63:0]                         o_weight_subword3_flat,
    output logic [16*WORK_ADDR_WIDTH-1:0]       o_work_base_addr_flat,
    output logic [255:0]                        o_leak_flat,
    output logic [511:0]                        o_threshold_flat,
    output logic [15:0]                         o_reset_to_zero,
    output logic [255:0]                        o_trace_decay_flat,
    output logic [991:0]                        o_transaction_meta_flat,

    output logic                                o_busy,
    output logic                                o_done,
    output logic                                o_sticky_error
);
    localparam logic [1:0] MODE_FC = 2'd0;
    localparam logic [1:0] MODE_SRNN = 2'd1;
    localparam logic [1:0] MODE_SCNN = 2'd2;
    localparam logic [1:0] MODE_M2_Q14 = 2'd3;
    localparam int HALF_DEPTH = 1 << WORK_ADDR_WIDTH;

    logic [9:0] geometry_units;
    logic [5:0] geometry_tail_lanes;
    logic [39:0] geometry_tail_lane_mask;
    logic geometry_legal;
    logic [10:0] scnn_channel_end;
    logic [10:0] direct_tile_end;

    logic busy_q;
    logic done_q;
    logic sticky_error_q;
    logic [5:0] wave_q;
    logic [15:0] pending_q;
    logic [9:0] unit_count_q;
    logic [39:0] tail_lane_mask_q;
    logic [1:0] mode_q;
    logic fc_weight_int8_q;
    logic fc_current_shift6_q;
    logic first_q;
    logic last_q;
    logic [3:0] source_valid_q;
    logic [39:0] scnn_feature_mask_q [0:3];
    logic signed [17:0] scnn_factor_b_q;
    logic [9:0] scnn_output_channel_base_q;
    logic scnn_active_pair_q;
    logic [3:0] scnn_weight_subword_q [0:3];
    logic [WORK_ADDR_WIDTH-1:0] scnn_work_base_addr_q;
    logic [3:0] destination_tile_base_q;
    logic [WEIGHT_ADDR_WIDTH-1:0] weight_addr_q [0:3];
    logic [15:0] leak_q;
    logic [31:0] threshold_q;
    logic reset_to_zero_q;
    logic [15:0] trace_decay_q;
    logic [51:0] transaction_tag_q;

    // The shared schedulers may perform their own runtime geometry arithmetic.
    // Capture the accepted command first so the walker's geometry and legality
    // logic cannot feed the high-fanout enables of the active configuration
    // registers in the same cycle.
    logic command_stage_valid_q;
    logic [1:0] command_mode_q;
    logic command_fc_weight_int8_q;
    logic command_fc_current_shift6_q;
    logic [15:0] command_neuron_count_q;
    logic command_first_q;
    logic command_last_q;
    logic [3:0] command_source_valid_q;
    logic [5:0] command_scnn_spatial_count_q;
    logic [9:0] command_scnn_output_channel_base_q;
    logic [39:0] command_scnn_feature_mask_q [0:3];
    logic signed [17:0] command_scnn_factor_b_q;
    logic command_scnn_active_pair_q;
    logic [3:0] command_scnn_weight_subword_q [0:3];
    logic [WORK_ADDR_WIDTH-1:0] command_scnn_work_base_addr_q;
    logic [3:0] command_destination_tile_base_q;
    logic [WEIGHT_ADDR_WIDTH-1:0] command_weight_addr_q [0:3];
    logic [15:0] command_leak_q;
    logic [31:0] command_threshold_q;
    logic command_reset_to_zero_q;
    logic [15:0] command_trace_decay_q;
    logic [51:0] command_transaction_tag_q;

    logic command_fire;
    logic command_legal;
    logic [15:0] accepted_mask;
    logic [15:0] pending_after_accept;

    function automatic logic [15:0] mask_for_wave(
        input logic [9:0] units,
        input logic [5:0] wave,
        input logic [3:0] destination_tile_base,
        input logic [1:0] command_mode
    );
        integer unsigned base;
        begin
            base = integer'(wave) * ARRAY_TILES;
            mask_for_wave = '0;
            for (integer tile = 0; tile < ARRAY_TILES; tile++) begin
                if (command_mode == MODE_SCNN ?
                        base + tile < integer'(units) :
                        tile >= integer'(destination_tile_base) &&
                        tile < integer'(destination_tile_base) +
                            integer'(units))
                    mask_for_wave[tile] = 1'b1;
            end
        end
    endfunction

    // This boundary accepts one resident wave, not a folded workset. Constant
    // H40 bounds give the same legal commands without a generic divide-by-40
    // followed by divide-by-ARRAY_TILES and a row-capacity product.
    always_comb begin
        geometry_units = '0;
        geometry_tail_lanes = '0;
        geometry_legal = 1'b0;
        if (command_mode_q == MODE_FC || command_mode_q == MODE_SRNN ||
                command_mode_q == MODE_M2_Q14) begin
            for (int tile = 0; tile < ARRAY_TILES; tile++) begin
                if (command_neuron_count_q > 16'(tile * 40)) begin
                    geometry_units = 10'(tile + 1);
                    geometry_tail_lanes =
                        6'(command_neuron_count_q - 16'(tile * 40));
                end
            end
            geometry_legal = command_neuron_count_q != 0 &&
                command_neuron_count_q <= 16'(ARRAY_TILES * 40);
        end else if (command_mode_q == MODE_SCNN) begin
            geometry_units = command_neuron_count_q[9:0];
            geometry_tail_lanes = command_scnn_spatial_count_q;
            geometry_legal = command_neuron_count_q != 0 &&
                command_neuron_count_q <= 16'(ARRAY_TILES) &&
                command_scnn_spatial_count_q != 0 &&
                command_scnn_spatial_count_q <= 6'd40;
        end
        geometry_tail_lane_mask = '0;
        for (int lane = 0; lane < 40; lane++) begin
            if (6'(lane) < geometry_tail_lanes)
                geometry_tail_lane_mask[lane] = 1'b1;
        end
    end

    assign scnn_channel_end =
        {1'b0, command_scnn_output_channel_base_q} +
        {1'b0, geometry_units};
    assign direct_tile_end =
        {7'b0, command_destination_tile_base_q} +
        {1'b0, geometry_units};
    assign command_legal = geometry_legal &&
        (!command_fc_current_shift6_q || command_mode_q == MODE_FC) &&
        (!command_fc_weight_int8_q || (command_mode_q == MODE_FC &&
            command_neuron_count_q <= 16'd25)) &&
        (command_mode_q == MODE_SCNN ?
         (command_source_valid_q <= 4'd4 &&
          command_scnn_work_base_addr_q < WORK_ADDR_WIDTH'(HALF_DEPTH - 1) &&
          scnn_channel_end <= 11'd1024) :
         direct_tile_end <= 11'(ARRAY_TILES));
    assign o_command_ready = !busy_q;
    assign command_fire = i_command_valid && o_command_ready;
    assign accepted_mask = pending_q & i_issue_ready;
    assign pending_after_accept = pending_q & ~accepted_mask;
    assign o_busy = busy_q;
    assign o_done = done_q;
    assign o_sticky_error = sticky_error_q;

    // Payload registers are deliberately reset-free. command_stage_valid_q is
    // the sole authority for consuming them after an accepted handshake.
    always_ff @(posedge i_clk) begin
        if (command_fire) begin
            command_mode_q <= i_mode;
            command_fc_weight_int8_q <= i_fc_weight_int8 === 1'b1;
            command_fc_current_shift6_q <= i_fc_current_shift6 === 1'b1;
            command_neuron_count_q <= i_neuron_count;
            command_first_q <= i_first;
            command_last_q <= i_last;
            command_source_valid_q <= i_source_valid;
            command_scnn_spatial_count_q <= i_scnn_spatial_count;
            command_scnn_output_channel_base_q <=
                i_scnn_output_channel_base;
            command_scnn_feature_mask_q[0] <= i_scnn_feature_mask0;
            command_scnn_feature_mask_q[1] <= i_scnn_feature_mask1;
            command_scnn_feature_mask_q[2] <= i_scnn_feature_mask2;
            command_scnn_feature_mask_q[3] <= i_scnn_feature_mask3;
            command_scnn_factor_b_q <= i_scnn_factor_b;
            command_scnn_active_pair_q <= i_scnn_active_pair;
            command_scnn_weight_subword_q[0] <= i_scnn_weight_subword0;
            command_scnn_weight_subword_q[1] <= i_scnn_weight_subword1;
            command_scnn_weight_subword_q[2] <= i_scnn_weight_subword2;
            command_scnn_weight_subword_q[3] <= i_scnn_weight_subword3;
            command_scnn_work_base_addr_q <= i_scnn_work_base_addr;
            command_destination_tile_base_q <= i_destination_tile_base;
            command_weight_addr_q[0] <= i_weight_addr0;
            command_weight_addr_q[1] <= i_weight_addr1;
            command_weight_addr_q[2] <= i_weight_addr2;
            command_weight_addr_q[3] <= i_weight_addr3;
            command_leak_q <= i_leak;
            command_threshold_q <= i_threshold;
            command_reset_to_zero_q <= i_reset_to_zero;
            command_trace_decay_q <= i_trace_decay;
            command_transaction_tag_q <= i_transaction_tag;
        end
    end

    always_comb begin
        o_issue_valid = pending_q;
        o_first = '0;
        o_last = '0;
        o_lane_mask_flat = '0;
        o_enable_a_flat = '0;
        o_enable_d_flat = '0;
        o_enable2_flat = '0;
        o_enable3_flat = '0;
        o_factor_b_flat = '0;
        o_mode_flat = '0;
        o_fc_weight_int8 = '0;
        o_scnn_active_bank = '0;
        o_weight_addr0_flat = '0;
        o_weight_addr1_flat = '0;
        o_weight_addr2_flat = '0;
        o_weight_addr3_flat = '0;
        o_weight_subword0_flat = '0;
        o_weight_subword1_flat = '0;
        o_weight_subword2_flat = '0;
        o_weight_subword3_flat = '0;
        o_work_base_addr_flat = '0;
        o_leak_flat = '0;
        o_threshold_flat = '0;
        o_reset_to_zero = '0;
        o_trace_decay_flat = '0;
        o_transaction_meta_flat = '0;

        for (integer tile = 0; tile < ARRAY_TILES; tile++) begin
            integer unsigned unit_index;
            logic [39:0] lane_mask;

            unit_index = integer'(wave_q) * ARRAY_TILES + tile;
            if (mode_q == MODE_SCNN) begin
                unit_index = integer'(scnn_output_channel_base_q) + tile;
                lane_mask = tail_lane_mask_q;
            end else begin
                unit_index = tile >= integer'(destination_tile_base_q) ?
                    tile - integer'(destination_tile_base_q) : 0;
                lane_mask = unit_index + 1 == integer'(unit_count_q) ?
                    tail_lane_mask_q : 40'hffffffffff;
            end

            o_first[tile] = first_q;
            o_last[tile] = last_q;
            o_lane_mask_flat[tile*40 +: 40] = lane_mask;
            if (mode_q == MODE_SCNN) begin
                o_enable_a_flat[tile*40 +: 40] =
                    lane_mask & scnn_feature_mask_q[0];
                o_enable_d_flat[tile*40 +: 40] =
                    lane_mask & scnn_feature_mask_q[1];
                o_enable2_flat[tile*40 +: 40] =
                    lane_mask & scnn_feature_mask_q[2];
                o_enable3_flat[tile*40 +: 40] =
                    lane_mask & scnn_feature_mask_q[3];
            end else begin
                o_enable_a_flat[tile*40 +: 40] =
                    lane_mask & {40{source_valid_q[0]}};
                o_enable_d_flat[tile*40 +: 40] =
                    lane_mask & {40{source_valid_q[1]}};
                o_enable2_flat[tile*40 +: 40] =
                    lane_mask & {40{source_valid_q[2]}};
                o_enable3_flat[tile*40 +: 40] =
                    lane_mask & {40{source_valid_q[3]}};
            end
            for (integer lane = 0; lane < 40; lane++) begin
                if (lane_mask[lane])
                    o_factor_b_flat[(tile*40 + lane)*18 +: 18] =
                        mode_q == MODE_SCNN ? scnn_factor_b_q :
                        (mode_q == MODE_FC && fc_current_shift6_q) ?
                            18'sd64 : 18'sd1;
            end
            o_mode_flat[tile*2 +: 2] = mode_q;
            o_fc_weight_int8[tile] = fc_weight_int8_q;
            o_scnn_active_bank[tile] = scnn_active_pair_q;
            o_weight_addr0_flat[tile*WEIGHT_ADDR_WIDTH +:
                WEIGHT_ADDR_WIDTH] = weight_addr_q[0];
            o_weight_addr1_flat[tile*WEIGHT_ADDR_WIDTH +:
                WEIGHT_ADDR_WIDTH] = weight_addr_q[1];
            o_weight_addr2_flat[tile*WEIGHT_ADDR_WIDTH +:
                WEIGHT_ADDR_WIDTH] = weight_addr_q[2];
            o_weight_addr3_flat[tile*WEIGHT_ADDR_WIDTH +:
                WEIGHT_ADDR_WIDTH] = weight_addr_q[3];
            o_weight_subword0_flat[tile*4 +: 4] =
                scnn_weight_subword_q[0];
            o_weight_subword1_flat[tile*4 +: 4] =
                scnn_weight_subword_q[1];
            o_weight_subword2_flat[tile*4 +: 4] =
                scnn_weight_subword_q[2];
            o_weight_subword3_flat[tile*4 +: 4] =
                scnn_weight_subword_q[3];
            o_work_base_addr_flat[tile*WORK_ADDR_WIDTH +:
                WORK_ADDR_WIDTH] = mode_q == MODE_SCNN ?
                    scnn_work_base_addr_q :
                    scnn_work_base_addr_q +
                        WORK_ADDR_WIDTH'(integer'(wave_q) * 4);
            o_leak_flat[tile*16 +: 16] = leak_q;
            o_threshold_flat[tile*32 +: 32] = threshold_q;
            o_reset_to_zero[tile] = reset_to_zero_q;
            o_trace_decay_flat[tile*16 +: 16] = trace_decay_q;
            o_transaction_meta_flat[tile*62 +: 62] =
                {transaction_tag_q, 10'(unit_index)};
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            busy_q <= 1'b0;
            done_q <= 1'b0;
            sticky_error_q <= 1'b0;
            command_stage_valid_q <= 1'b0;
            wave_q <= '0;
            pending_q <= '0;
            unit_count_q <= '0;
            tail_lane_mask_q <= '0;
            mode_q <= MODE_FC;
            fc_weight_int8_q <= 1'b0;
            fc_current_shift6_q <= 1'b0;
            first_q <= 1'b0;
            last_q <= 1'b0;
            source_valid_q <= '0;
            for (integer term = 0; term < 4; term++) begin
                scnn_feature_mask_q[term] <= '0;
                scnn_weight_subword_q[term] <= '0;
            end
            scnn_factor_b_q <= 18'sd256;
            scnn_output_channel_base_q <= '0;
            scnn_active_pair_q <= 1'b0;
            scnn_work_base_addr_q <= '0;
            destination_tile_base_q <= '0;
            for (integer term = 0; term < 4; term++)
                weight_addr_q[term] <= '0;
            leak_q <= '0;
            threshold_q <= '0;
            reset_to_zero_q <= 1'b0;
            trace_decay_q <= '0;
            transaction_tag_q <= '0;
        end else begin
            done_q <= 1'b0;

            if (i_clear_error && !busy_q)
                sticky_error_q <= 1'b0;

            if (i_abort) begin
                busy_q <= 1'b0;
                command_stage_valid_q <= 1'b0;
                pending_q <= '0;
            end else if (busy_q) begin
                if (command_stage_valid_q) begin
                    command_stage_valid_q <= 1'b0;
                    wave_q <= '0;
                    pending_q <= command_legal ?
                        mask_for_wave(geometry_units, 6'd0,
                            command_destination_tile_base_q,
                            command_mode_q) : '0;
                    unit_count_q <= geometry_units;
                    tail_lane_mask_q <= geometry_tail_lane_mask;
                    mode_q <= command_mode_q;
                    fc_weight_int8_q <= command_fc_weight_int8_q;
                    fc_current_shift6_q <= command_fc_current_shift6_q;
                    first_q <= command_first_q;
                    last_q <= command_last_q;
                    source_valid_q <= command_source_valid_q;
                    scnn_feature_mask_q[0] <=
                        command_scnn_feature_mask_q[0];
                    scnn_feature_mask_q[1] <=
                        command_scnn_feature_mask_q[1];
                    scnn_feature_mask_q[2] <=
                        command_scnn_feature_mask_q[2];
                    scnn_feature_mask_q[3] <=
                        command_scnn_feature_mask_q[3];
                    scnn_factor_b_q <= command_scnn_factor_b_q;
                    scnn_output_channel_base_q <=
                        command_scnn_output_channel_base_q;
                    scnn_active_pair_q <= command_scnn_active_pair_q;
                    scnn_weight_subword_q[0] <=
                        command_scnn_weight_subword_q[0];
                    scnn_weight_subword_q[1] <=
                        command_scnn_weight_subword_q[1];
                    scnn_weight_subword_q[2] <=
                        command_scnn_weight_subword_q[2];
                    scnn_weight_subword_q[3] <=
                        command_scnn_weight_subword_q[3];
                    scnn_work_base_addr_q <=
                        command_scnn_work_base_addr_q;
                    destination_tile_base_q <=
                        command_destination_tile_base_q;
                    weight_addr_q[0] <= command_weight_addr_q[0];
                    weight_addr_q[1] <= command_weight_addr_q[1];
                    weight_addr_q[2] <= command_weight_addr_q[2];
                    weight_addr_q[3] <= command_weight_addr_q[3];
                    leak_q <= command_leak_q;
                    threshold_q <= command_threshold_q;
                    reset_to_zero_q <= command_reset_to_zero_q;
                    trace_decay_q <= command_trace_decay_q;
                    transaction_tag_q <= command_transaction_tag_q;
                    if (!command_legal) begin
                        busy_q <= 1'b0;
                        sticky_error_q <= 1'b1;
                        done_q <= 1'b1;
                    end
                end else if (pending_after_accept != 0) begin
                    pending_q <= pending_after_accept;
                end else if ((integer'(wave_q) + 1) * ARRAY_TILES <
                             integer'(unit_count_q)) begin
                    wave_q <= wave_q + 1'b1;
                    pending_q <= mask_for_wave(unit_count_q,
                                               wave_q + 1'b1,
                                               destination_tile_base_q,
                                               mode_q);
                end else begin
                    busy_q <= 1'b0;
                    pending_q <= '0;
                    done_q <= 1'b1;
                end
            end else if (command_fire) begin
                busy_q <= 1'b1;
                command_stage_valid_q <= 1'b1;
            end
        end
    end

    initial begin
        if (ARRAY_TILES < 4 || ARRAY_TILES > 16 ||
            (ARRAY_TILES % 4) != 0)
            $error("binary P4 destination walker requires 4, 8, 12 or 16 H40 tiles");
    end
endmodule

`default_nettype wire
