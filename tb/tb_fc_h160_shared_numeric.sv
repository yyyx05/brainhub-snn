`timescale 1ns/1ps
`default_nettype none

// One real sample, replayed with and without deterministic observation stalls.
// No hierarchy writes or forced arithmetic results: weights/state use public ports.
module tb_fc_h160_shared_numeric;
    localparam int AW = 11;
    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    logic command_valid = 0, command_ready;
    logic fc_int8 = 0, fc_shift6 = 1;
    logic [15:0] neurons = 160;
    logic first = 0, last = 0;
    logic [3:0] source_valid = 0, tile_base = 0;
    logic [AW-1:0] row_addr = 0;
    logic [15:0] leak = 32113;
    logic [31:0] threshold = 1024;
    logic [51:0] tag = 0;
    logic [3:0] fill_valid, fill_ready, fill_bank;
    logic [7:0] fill_tile;
    logic [43:0] fill_addr;
    logic [799:0] fill_data;
    logic [3:0] obs_valid, obs_ready;
    logic [15:0] obs_tile;
    logic [7:0] obs_mode;
    logic [39:0] obs_mask, obs_range, obs_spike;
    logic [1279:0] obs_pre, obs_trace;
    logic [255:0] obs_meta;
    logic [3:0] maintenance_tile = 0;
    logic state_wr = 0, state_ready, trace_wr = 0, trace_ready;
    logic [9:0] state_addr = 0, trace_addr = 0;
    logic busy, done, idle, error;
    logic [15:0] tile_error;
    logic numeric_valid = 0, numeric_ready, q6_valid, q6_ready = 0;
    logic signed [23:0] numeric_raw = 0;
    logic signed [15:0] q6;
    logic router_cfg_valid = 0, router_cfg_ready, router_error;
    logic [9:0] numeric_class = 0;
    logic refill_valid = 0, refill_ready, refill_done, refill_error;
    logic refill_int8 = 0;
    logic [3:0] refill_tile_base = 0;
    logic [23:0] refill_parameter_base = 0, refill_stride = 160;
    logic [11:0] refill_features = 2312;
    logic [9:0] refill_hidden = 160;
    logic [4:0] refill_tiles = 4;
    logic read_valid, read_ready, response_valid = 0, response_ready;
    logic [23:0] read_addr;
    logic [15:0] read_tag, response_tag = 0;
    logic signed [15:0] response_code = 0;
    logic signed [15:0] parameters [0:371519];
    integer parameter_reads = 0, parameter_responses = 0, load_start;

    logic [199:0] weights0 [0:9247], weights2 [0:159];
    logic [3:0] input_masks [0:57799];
    logic [15:0] input_counts [0:231199];
    integer counts_mode = 0;
    logic signed [31:0] expected_pre [0:15999], expected_trace [0:15999];
    logic expected_spike [0:15999];
    logic signed [31:0] expected_dot [0:999], expected_cumulative [0:999];
    logic signed [15:0] expected_q6 [0:999];
    logic [159:0] hidden_spikes;
    logic [159:0] hidden_seen;
    logic [9:0] output_seen;
    integer output_dot [0:9], temporal_sum [0:9];
    integer steps = 100, pass_index = 0, timestep = 0, phase = 0;
    integer cycles = 0, commands = 0, input_terms = 0, hidden_terms = 0;
    integer hidden_checks = 0, output_checks = 0, q6_checks = 0;
    integer fills = 0, issue_count = 0, enabled_terms = 0;
    integer observation_wait = 0, start_cycle, end_cycle;
    integer issue_start, enabled_start, command_start, wait_start;
    integer w0_cycles, w2_cycles, adapter_cycles, segment_start;
    wire [11:0] weight_read_fire, state_read_fire, state_write_fire;
    wire [11:0] trace_read_fire, trace_write_fire;
    integer weight_reads = 0, state_reads = 0, state_writes = 0;
    integer trace_reads = 0, trace_writes = 0;
    integer weight_read_start, state_read_start, state_write_start;
    integer trace_read_start, trace_write_start, init_start, init_cycles;
    string fixture;

    unified_binary_p4_h40_array12_path_v1 u_dut (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(1'b0), .i_clear_error(1'b0),
        .i_command_valid(command_valid), .o_command_ready(command_ready),
        .i_mode(2'd0), .i_fc_weight_int8(fc_int8), .i_fc_current_shift6(fc_shift6),
        .i_neuron_count(neurons), .i_first(first), .i_last(last),
        .i_source_valid(source_valid), .i_scnn_spatial_count(6'd0),
        .i_scnn_output_channel_base(10'd0), .i_scnn_feature_mask0(40'd0),
        .i_scnn_feature_mask1(40'd0), .i_scnn_feature_mask2(40'd0),
        .i_scnn_feature_mask3(40'd0), .i_scnn_factor_b(18'sd256),
        .i_scnn_active_pair(1'b0), .i_scnn_weight_subword0(4'd0),
        .i_scnn_weight_subword1(4'd0), .i_scnn_weight_subword2(4'd0),
        .i_scnn_weight_subword3(4'd0), .i_scnn_work_base_addr(9'd0),
        .i_destination_tile_base(tile_base),
        .i_weight_addr0(row_addr), .i_weight_addr1(row_addr),
        .i_weight_addr2(row_addr), .i_weight_addr3(row_addr),
        .i_leak(leak), .i_threshold(threshold), .i_reset_to_zero(1'b0),
        .i_trace_decay(16'd64225), .i_transaction_tag(tag),
        .i_weight_fill_valid(fill_valid), .o_weight_fill_ready(fill_ready),
        .i_weight_fill_tile_flat(fill_tile), .i_weight_fill_bank(fill_bank),
        .i_weight_fill_addr_flat(fill_addr), .i_weight_fill_data_flat(fill_data),
        .o_observation_valid(obs_valid), .i_observation_ready(obs_ready),
        .o_observation_tile_flat(obs_tile), .o_observation_mode_flat(obs_mode),
        .o_observation_lane_mask_flat(obs_mask), .o_observation_range_mask_flat(obs_range),
        .o_observation_spike_flat(obs_spike), .o_observation_pre_state_flat(obs_pre),
        .o_observation_trace_next_flat(obs_trace), .o_observation_meta_flat(obs_meta),
        .i_maintenance_tile(maintenance_tile),
        .i_state_maintenance_write_valid(state_wr), .o_state_maintenance_write_ready(state_ready),
        .i_state_maintenance_write_addr(state_addr), .i_state_maintenance_write_data(320'd0),
        .i_state_maintenance_write_mask(40'hffffffffff),
        .i_state_maintenance_read_valid(1'b0), .o_state_maintenance_read_ready(),
        .i_state_maintenance_read_addr(10'd0), .o_state_maintenance_result_valid(),
        .i_state_maintenance_result_ready(1'b1), .o_state_maintenance_result_addr(),
        .o_state_maintenance_read_data(),
        .i_trace_maintenance_write_valid(trace_wr), .o_trace_maintenance_write_ready(trace_ready),
        .i_trace_maintenance_write_addr(trace_addr), .i_trace_maintenance_write_data(320'd0),
        .i_trace_maintenance_write_mask(40'hffffffffff),
        .i_trace_maintenance_read_valid(1'b0), .o_trace_maintenance_read_ready(),
        .i_trace_maintenance_read_addr(10'd0), .o_trace_maintenance_result_valid(),
        .i_trace_maintenance_result_ready(1'b1), .o_trace_maintenance_result_addr(),
        .o_trace_maintenance_read_data(), .o_busy(busy), .o_done(done), .o_idle(idle),
        .o_tile_done(), .o_tile_idle(), .o_tile_sticky_error(tile_error), .o_sticky_error(error)
    );

    unified_fc_srnn_h40_tp_replay_router_v1 u_numeric (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(1'b0), .i_clear_error(1'b0),
        .i_cfg_valid(router_cfg_valid), .o_cfg_ready(router_cfg_ready),
        .i_cfg_mode(2'd0), .i_cfg_sample_id(6'd0), .i_cfg_neuron_count(10'd10),
        .i_cfg_fc_threshold(18'sd131071), .i_cfg_fc_output_layer(1'b1),
        .i_cfg_fc_label(4'd0), .i_cfg_srnn_path(1'b0),
        .i_cfg_fc_logit_scale_enable(1'b1), .i_cfg_fc_logit_scale_q16(24'd47303),
        .i_cfg_fc_logit_accumulate(1'b1), .i_cfg_fc_sample_start(timestep == 0),
        .i_replay_valid(numeric_valid), .o_replay_ready(numeric_ready),
        .i_replay_mode(2'd0), .i_replay_local_index(numeric_class), .i_replay_spike(1'b0),
        .i_replay_pre_state({{8{numeric_raw[23]}}, numeric_raw}),
        .i_replay_trace_next(32'sd0), .i_replay_last(numeric_class == 9),
        .o_fc_student_trace_valid(), .i_fc_student_trace_ready(1'b1),
        .o_fc_student_trace_sample(), .o_fc_student_trace_hidden(), .o_fc_student_trace_q016(),
        .o_fc_surrogate_valid(), .i_fc_surrogate_ready(1'b1), .o_fc_surrogate_sample(),
        .o_fc_surrogate_hidden(), .o_fc_surrogate_q016(),
        .o_fc_logit_row_valid(), .i_fc_logit_row_ready(1'b1), .o_fc_logit_sample(),
        .o_fc_logit_label(), .o_fc_logit_valid(q6_valid), .i_fc_logit_ready(q6_ready),
        .o_fc_logit_q6(q6), .o_fc_logit_last(),
        .o_srnn_backbone_valid(), .i_srnn_backbone_ready(1'b1), .o_srnn_backbone_path(),
        .o_srnn_backbone_sample_id(), .o_srnn_backbone_id(), .o_srnn_backbone_q14(),
        .o_srnn_backbone_last(), .o_done(), .o_busy(), .o_idle(),
        .o_sticky_error(router_error), .o_error_code()
    );

    unified_fc_h40_w0_cache_refill_scheduler_v1 #(
        .PARAM_DEPTH(371520), .ARRAY_TILES(12)
    ) u_refill (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(1'b0), .i_clear_error(1'b0),
        .i_refill_valid(refill_valid), .o_refill_ready(refill_ready), .i_refill_all(1'b1),
        .i_fc_weight_int8(refill_int8), .i_destination_tile_base(refill_tile_base),
        .i_dirty_feature(12'd0), .i_dirty_tile(4'd0), .i_parameter_base(refill_parameter_base),
        .i_hidden_stride(refill_stride), .i_cache_row_base(10'd0),
        .i_feature_count(refill_features), .i_hidden_count(refill_hidden),
        .i_tile_count(refill_tiles), .i_version(16'd0), .i_array_idle(idle),
        .o_forward_read_valid(read_valid), .i_forward_read_ready(read_ready),
        .o_forward_read_addr(read_addr), .o_forward_read_tag(read_tag),
        .i_forward_response_valid(response_valid), .o_forward_response_ready(response_ready),
        .i_forward_response_tag(response_tag), .i_forward_response_code(response_code),
        .i_forward_response_error(1'b0), .o_weight_fill_valid(fill_valid),
        .i_weight_fill_ready(fill_ready), .o_weight_fill_tile_flat(fill_tile),
        .o_weight_fill_bank(fill_bank), .o_weight_fill_addr_flat(fill_addr),
        .o_weight_fill_data_flat(fill_data), .o_refill_done(refill_done),
        .o_rows_published(), .o_published_version(), .o_busy(), .o_sticky_error(refill_error)
    );
    assign read_ready = !response_valid;
    always_ff @(posedge clk) begin
        if (read_valid && read_ready) begin
            if (read_addr >= 371520) $fatal(1, "parameter range %0d", read_addr);
            response_valid <= 1;
            response_tag <= read_tag;
            response_code <= parameters[read_addr];
        end else if (response_valid && response_ready) response_valid <= 0;
    end

    // Read-only handshake monitors. A FC cache request reads four 200-bit banks,
    // including lanes whose arithmetic enable is zero; count those separately.
    for (genvar r = 0; r < 3; r++) begin : g_monitor_region
        for (genvar t = 0; t < 4; t++) begin : g_monitor_tile
            assign weight_read_fire[r*4+t] = u_dut.u_array.g_region[r].u_region.g_tile[t].u_tile.g_p4_all_frontend.u_weight_store.read_fire;
            assign state_read_fire[r*4+t] = u_dut.u_array.g_region[r].u_region.g_tile[t].u_tile.seq_state_read_valid && u_dut.u_array.g_region[r].u_region.g_tile[t].u_tile.seq_state_read_ready;
            assign state_write_fire[r*4+t] = u_dut.u_array.g_region[r].u_region.g_tile[t].u_tile.seq_state_write_valid && u_dut.u_array.g_region[r].u_region.g_tile[t].u_tile.seq_state_write_ready;
            assign trace_read_fire[r*4+t] = u_dut.u_array.g_region[r].u_region.g_tile[t].u_tile.seq_trace_read_valid && u_dut.u_array.g_region[r].u_region.g_tile[t].u_tile.seq_trace_read_ready;
            assign trace_write_fire[r*4+t] = u_dut.u_array.g_region[r].u_region.g_tile[t].u_tile.seq_trace_write_valid && u_dut.u_array.g_region[r].u_region.g_tile[t].u_tile.seq_trace_write_ready;
        end
    end

    always_comb begin
        obs_ready = '1;
        if (pass_index == 1)
            for (int r = 0; r < 4; r++)
                obs_ready[r] = (cycles + r * 3) % 13 >= 4;
    end

    always @(posedge clk) begin
        cycles++;
        if (cycles > 5000000) $fatal(1, "timeout phase=%0d t=%0d", phase, timestep);
        if (rst_n && (error || tile_error != 0 || router_error || refill_error))
            $fatal(1, "DUT error phase=%0d timestep=%0d error=%h tiles=%h", phase, timestep, error, tile_error);
        if (command_valid && command_ready) commands++;
        if (read_valid && read_ready) parameter_reads++;
        if (response_valid && response_ready) parameter_responses++;
        weight_reads += $countones(weight_read_fire);
        state_reads += $countones(state_read_fire);
        state_writes += $countones(state_write_fire);
        trace_reads += $countones(trace_read_fire);
        trace_writes += $countones(trace_write_fire);
        for (int r = 0; r < 4; r++) begin
            if (fill_valid[r] && fill_ready[r]) begin
                int tile, source;
                tile = r*4 + int'(fill_tile[r*2 +: 2]);
                source = int'(fill_addr[r*AW +: 10])*4 + int'({fill_addr[r*AW+10], fill_bank[r]});
                if (refill_int8) begin
                    if (tile != 4 || source >= 160 || fill_data[r*200 +: 200] != weights2[source])
                        $fatal(1, "W2 refill mapping");
                end else if (tile >= 4 || source >= 2312 || fill_data[r*200 +: 200] != weights0[tile*2312+source])
                    $fatal(1, "W0 refill mapping");
                fills++;
            end
            if (obs_valid[r] && !obs_ready[r]) observation_wait++;
            if (obs_valid[r] && obs_ready[r]) begin
                int tile, group_id, h, base;
                tile = int'(obs_tile[r*4 +: 4]);
                group_id = int'(obs_meta[r*64 +: 2]);
                if (obs_mode[r*2 +: 2] != 0 || obs_range[r*10 +: 10] != 0)
                    $fatal(1, "observation mode/range mismatch");
                for (int lane = 0; lane < 10; lane++) begin
                    if (obs_mask[r*10 + lane]) begin
                        if (phase == 1) begin
                            h = tile*40 + group_id*10 + lane;
                            base = timestep*160 + h;
                            if (h >= 160 || hidden_seen[h]) $fatal(1, "hidden id/duplicate %0d", h);
                            if ($signed(obs_pre[r*320 + lane*32 +: 32]) != expected_pre[base] ||
                                obs_spike[r*10+lane] != expected_spike[base] ||
                                $signed(obs_trace[r*320 + lane*32 +: 32]) != expected_trace[base])
                                $fatal(1, "hidden mismatch t=%0d h=%0d got pre=%0d spike=%0d trace=%0d expected=%0d/%0d/%0d",
                                    timestep, h, $signed(obs_pre[r*320+lane*32 +: 32]), obs_spike[r*10+lane],
                                    $signed(obs_trace[r*320+lane*32 +: 32]), expected_pre[base], expected_spike[base], expected_trace[base]);
                            hidden_seen[h] = 1;
                            hidden_spikes[h] = obs_spike[r*10+lane];
                            hidden_checks++;
                        end else if (phase == 2) begin
                            if (tile != 4 || group_id != 0 || output_seen[lane]) $fatal(1, "output route/duplicate");
                            if (obs_spike[r*10+lane] ||
                                $signed(obs_pre[r*320 + lane*32 +: 32]) != expected_dot[timestep*10+lane])
                                $fatal(1, "linear dot mismatch t=%0d c=%0d got=%0d expected=%0d spike=%0d",
                                    timestep, lane, $signed(obs_pre[r*320+lane*32 +: 32]),
                                    expected_dot[timestep*10+lane], obs_spike[r*10+lane]);
                            output_dot[lane] = $signed(obs_pre[r*320+lane*32 +: 32]);
                            temporal_sum[lane] = 32'($signed(24'(temporal_sum[lane] + output_dot[lane])));
                            if (temporal_sum[lane] != expected_cumulative[timestep*10+lane]) $fatal(1, "host temporal sum mismatch");
                            output_seen[lane] = 1;
                            output_checks++;
                        end else $fatal(1, "observation outside active layer");
                    end
                end
            end
        end
        for (int tile = 0; tile < 12; tile++) begin
            if (u_dut.u_walker.o_issue_valid[tile] && u_dut.u_walker.i_issue_ready[tile]) begin
                issue_count++;
                enabled_terms += $countones(u_dut.u_walker.o_enable_a_flat[tile*40 +: 40]);
                enabled_terms += $countones(u_dut.u_walker.o_enable_d_flat[tile*40 +: 40]);
                enabled_terms += $countones(u_dut.u_walker.o_enable2_flat[tile*40 +: 40]);
                enabled_terms += $countones(u_dut.u_walker.o_enable3_flat[tile*40 +: 40]);
            end
        end
    end

    task automatic refill;
        @(negedge clk); refill_valid = 1;
        do @(posedge clk); while (!refill_ready);
        @(negedge clk); refill_valid = 0;
        while (!refill_done) @(negedge clk);
        @(negedge clk);
    endtask

    task automatic zero_states;
        for (int tile = 0; tile < 5; tile++) begin
            for (int row = 0; row < 4; row++) begin
                @(negedge clk);
                maintenance_tile = 4'(tile);
                state_addr = 10'(row);
                trace_addr = {1'b1, 9'(row)};
                state_wr = 1;
                trace_wr = 1;
                do @(posedge clk); while (!state_ready || !trace_ready);
                @(negedge clk);
                state_wr = 0;
                trace_wr = 0;
            end
        end
    endtask

    task automatic issue(input int row, input logic [3:0] mask, input bit is_first, input bit is_last);
        @(negedge clk);
        row_addr = AW'(row);
        source_valid = mask;
        first = is_first;
        last = is_last;
        command_valid = 1;
        do @(posedge clk); while (!command_ready);
        @(negedge clk);
        command_valid = 0;
        while (!command_ready) @(negedge clk);
    endtask

    function automatic int source_rounds(input bit output_layer, input int row);
        int rounds;
        rounds = 0;
        if (output_layer || !counts_mode) return 1;
        for (int lane = 0; lane < 4; lane++) begin
            if (int'(input_counts[timestep*2312+row*4+lane]) > rounds)
                rounds = int'(input_counts[timestep*2312+row*4+lane]);
        end
        return rounds;
    endfunction

    function automatic logic [3:0] source_mask(input bit output_layer, input int row, input int rep);
        logic [3:0] mask;
        if (output_layer) return hidden_spikes[row*4 +: 4];
        if (!counts_mode) return input_masks[timestep*578+row];
        for (int lane = 0; lane < 4; lane++)
            mask[lane] = int'(input_counts[timestep*2312+row*4+lane]) > rep;
        return mask;
    endfunction

    task automatic forward_layer(input bit output_layer);
        int n, issued, row_limit;
        logic [3:0] mask;
        phase = output_layer ? 2 : 1;
        fc_int8 = output_layer;
        fc_shift6 = !output_layer;
        neurons = output_layer ? 16'd10 : 16'd160;
        tile_base = output_layer ? 4'd4 : 4'd0;
        // leak=0 and unreachable positive threshold produce exact H160 linear dots.
        // Max |dot| <=160*128=20480 < signed18 positive limit.
        leak = output_layer ? 16'd0 : 16'd32113;
        threshold = output_layer ? 32'd131071 : 32'd1024;
        tag = {35'd0, 16'(timestep), output_layer};
        n = 0;
        row_limit = output_layer ? 40 : 578;
        for (int row = 0; row < row_limit; row++) begin
            for (int rep = 0; rep < source_rounds(output_layer, row); rep++) begin
                mask = source_mask(output_layer, row, rep);
                if (mask != 0) n++;
            end
        end
        issued = 0;
        if (n == 0) issue(0, 0, 1, 1);
        else for (int row = 0; row < row_limit; row++) begin
            for (int rep = 0; rep < source_rounds(output_layer, row); rep++) begin
                mask = source_mask(output_layer, row, rep);
                if (mask != 0) begin
                    if (output_layer) hidden_terms += $countones(mask);
                    else input_terms += $countones(mask);
                    issue(row, mask, issued == 0, issued == n-1);
                    issued++;
                end
            end
        end
        while (!idle || (output_layer ? output_seen != 10'h3ff : hidden_seen != {160{1'b1}})) @(negedge clk);
        @(negedge clk);
    endtask

    initial begin
        if (!$value$plusargs("FIXTURE=%s", fixture)) $fatal(1, "fixture required");
        if (!$value$plusargs("STEPS=%d", steps)) steps = 100;
        if (steps < 1 || steps > 100) $fatal(1, "invalid steps");
        if (!$value$plusargs("COUNTS=%d", counts_mode)) counts_mode = 0;
        $readmemh({fixture, "/w0.mem"}, weights0);
        $readmemh({fixture, "/w2.mem"}, weights2);
        $readmemh({fixture, "/parameters.mem"}, parameters);
        if (counts_mode) $readmemh({fixture, "/input_counts.mem"}, input_counts);
        else $readmemh({fixture, "/input.mem"}, input_masks);
        $readmemh({fixture, "/pre.mem"}, expected_pre);
        $readmemh({fixture, "/trace.mem"}, expected_trace);
        $readmemh({fixture, "/spike.mem"}, expected_spike);
        $readmemh({fixture, "/dot.mem"}, expected_dot);
        $readmemh({fixture, "/q6.mem"}, expected_q6);
        $readmemh({fixture, "/cumulative.mem"}, expected_cumulative);
        repeat (6) @(negedge clk);
        rst_n = 1;
        load_start = cycles;
        refill();
        refill_int8 = 1; refill_tile_base = 4; refill_parameter_base = 369920;
        refill_stride = 10; refill_features = 160; refill_hidden = 10; refill_tiles = 1;
        refill();
        if (parameter_reads != 371520 || parameter_responses != 371520) $fatal(1, "parameter response count");
        $display("FC_H160_LOAD fills=%0d cycles=%0d payload_bytes=%0d parameter_reads=%0d parameter_responses=%0d parameter_bytes=%0d",
            fills, cycles-load_start, fills*25, parameter_reads, parameter_responses, parameter_responses*2);
        for (pass_index = 0; pass_index < 2; pass_index++) begin
            init_start = cycles;
            zero_states();
            init_cycles = cycles-init_start;
            for (int c = 0; c < 10; c++) temporal_sum[c] = 0;
            start_cycle = cycles;
            command_start = commands;
            issue_start = issue_count;
            enabled_start = enabled_terms;
            wait_start = observation_wait;
            weight_read_start = weight_reads;
            state_read_start = state_reads;
            state_write_start = state_writes;
            trace_read_start = trace_reads;
            trace_write_start = trace_writes;
            w0_cycles = 0; w2_cycles = 0; adapter_cycles = 0;
            input_terms = 0; hidden_terms = 0;
            for (timestep = 0; timestep < steps; timestep++) begin
                hidden_spikes = 0; hidden_seen = 0; output_seen = 0;
                segment_start = cycles;
                forward_layer(0);
                w0_cycles += cycles - segment_start;
                segment_start = cycles;
                forward_layer(1);
                w2_cycles += cycles - segment_start;
                segment_start = cycles;
                @(negedge clk); router_cfg_valid = 1;
                do @(posedge clk); while (!router_cfg_ready);
                @(negedge clk); router_cfg_valid = 0;
                for (int c = 0; c < 10; c++) begin
                    @(negedge clk);
                    numeric_raw = 24'(output_dot[c]); numeric_class = 10'(c); numeric_valid = 1;
                    do @(posedge clk); while (!numeric_ready);
                    @(negedge clk); numeric_valid = 0;
                    while (!q6_valid) @(negedge clk);
                    if (pass_index == 1) repeat (2) @(negedge clk);
                    if (q6 != expected_q6[timestep*10+c]) $fatal(1, "Q6 mismatch t=%0d c=%0d got=%0d expected=%0d", timestep,c,q6,expected_q6[timestep*10+c]);
                    q6_checks++; q6_ready = 1;
                    @(posedge clk); @(negedge clk); q6_ready = 0;
                end
                adapter_cycles += cycles-segment_start;
            end
            end_cycle = cycles;
            phase = 0;
            $display("FC_H160_PASS_METRICS pass=%0d steps=%0d cycles=%0d init_cycles=%0d w0_cycles=%0d w2_cycles=%0d adapter_cycles=%0d commands=%0d tile_issues=%0d effective_sops=%0d input_events=%0d hidden_spikes=%0d observation_wait_cycles=%0d weight_read_requests=%0d weight_read_bytes=%0d state_read_words=%0d state_write_words=%0d trace_read_words=%0d trace_write_words=%0d",
                pass_index, steps, end_cycle-start_cycle, init_cycles,w0_cycles,w2_cycles,adapter_cycles,
                commands-command_start,issue_count-issue_start,enabled_terms-enabled_start,input_terms,hidden_terms,observation_wait-wait_start,
                weight_reads-weight_read_start, (weight_reads-weight_read_start)*100,
                state_reads-state_read_start,state_writes-state_write_start,trace_reads-trace_read_start,trace_writes-trace_write_start);
        end
        if (fills != 9408 || hidden_checks != 2*steps*160 || output_checks != 2*steps*10 || q6_checks != 2*steps*10)
            $fatal(1, "coverage mismatch");
        $display("FC_H160_SHARED_NUMERIC_PASS steps=%0d hidden_checks=%0d output_checks=%0d q6_checks=%0d", steps,hidden_checks,output_checks,q6_checks);
        $finish;
    end
endmodule
`default_nettype wire
