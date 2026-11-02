`timescale 1ns/1ps
`default_nettype none

// Standalone command-replay harness. Numerical fixture/tasks below are reused
// from tb_unified_h40_array16_three_topology_graph_top_v1; no catalog model load.
module tb_frontend_interface_smoke;
    localparam int AW = $clog2(1156);
    localparam logic [19:0] CLASS_TAG = 20'hc1a53;
    logic clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;

    logic start_valid = 1'b0, start_ready, scnn_done;
    logic context_load_valid, context_load_ready = 1'b0;
    logic context_load_weights, context_active_pair;
    logic context_refill_safe, context_release_valid;
    logic [9:0] context_channel_count;
    logic feature_request_valid, feature_request_ready = 1'b1;
    logic [3:0] feature_term_valid;
    logic feature_response_valid = 1'b0, feature_response_ready;
    logic [3:0] response_term_q = '0;
    logic [39:0] feature_mask0, feature_mask1, feature_mask2, feature_mask3;
    logic feature_update_valid, feature_update_ready = 1'b1;
    logic feature_update_done = 1'b0;

    logic feature_wave_valid, feature_wave_ready;
    logic [7:0] feature_wave_mask;
    logic [3:0] feature_wave_base;
    logic [127:0] feature_wave_flat;
    logic logit_wave_valid, logit_wave_ready;
    logic [11:0] logit_wave_mask;
    logic [3:0] logit_wave_base;
    logic [4:0] logit_wave_count;
    logic [191:0] logit_wave_flat;

    logic [3:0] weight_fill_valid = '0, weight_fill_ready;
    logic [7:0] weight_fill_tile_flat = '0;
    logic [3:0] weight_fill_bank = '0;
    logic [4*AW-1:0] weight_fill_addr_flat = '0;
    logic [799:0] weight_fill_data_flat = '0;
    logic maintenance_start_valid = 1'b0, maintenance_start_ready;
    logic maintenance_release_valid = 1'b0, maintenance_release_ready;
    logic maintenance_done;
    logic [3:0] maintenance_tile = '0;
    logic state_write_valid = 1'b0, state_write_ready;
    logic [9:0] state_write_addr = '0;
    logic [319:0] state_write_data = '0;
    logic [39:0] state_write_mask = '0;
    logic trace_write_valid = 1'b0, trace_write_ready;
    logic [9:0] trace_write_addr = '0;
    logic [319:0] trace_write_data = '0;
    logic [39:0] trace_write_mask = '0;
    logic [1:0] active_owner;
    logic busy, idle, sticky_error;
    logic [15:0] tile_idle, tile_error;
    logic [7:0] cycle_q = '0;
    integer loads = 0, releases = 0, feature_waves = 0, logit_waves = 0;
    integer dense_phase = 0;
    integer fc_observations = 0, fc_surrogate_observations = 0;
    integer srnn_observations = 0;
    logic [79:0] fc_seen = '0, fc_surrogate_seen = '0, srnn_seen = '0;

    logic dense_layer_valid = 1'b0, dense_layer_ready;
    logic [1:0] dense_mode = '0;
    logic [15:0] dense_neuron_count = '0;
    logic [15:0] dense_feedforward_groups = '0;
    logic [15:0] dense_recurrent_groups = '0;
    logic dense_recurrent_enable = 1'b0;
    logic [31:0] dense_threshold = '0;
    logic [51:0] dense_transaction_tag = '0;
    logic dense_group_valid = 1'b0, dense_group_ready;
    logic dense_group_recurrent = 1'b0;
    logic [3:0] dense_group_source_valid = '0;
    logic [AW-1:0] dense_group_addr [0:3];
    logic dense_layer_done, dense_busy, dense_idle, dense_error;
    logic [7:0] dense_error_code;
    logic [3:0] direct_observation_valid;
    logic [7:0] direct_observation_mode_flat;
    logic [39:0] direct_observation_lane_mask_flat;
    logic [39:0] direct_observation_spike_flat;
    logic [1279:0] direct_observation_pre_state_flat;
    logic dense_observation_valid;
    logic dense_observation_ready;
    logic [1:0] dense_observation_mode;
    logic [51:0] dense_observation_tag;
    logic [9:0] dense_observation_neuron;
    logic dense_observation_spike;
    logic signed [31:0] dense_observation_pre_state;
    logic signed [31:0] dense_observation_trace_next;
    logic dense_observation_last;
    logic dense_observation_busy;
    logic dense_observation_error;
    logic dense_reorder_cfg_valid = 1'b0, dense_reorder_cfg_ready;
    logic [1:0] dense_reorder_cfg_mode = '0;
    logic [51:0] dense_reorder_cfg_tag = '0;
    logic [9:0] dense_reorder_cfg_base = '0;
    logic [9:0] dense_reorder_cfg_count = '0;
    logic [5:0] dense_reorder_cfg_sample = '0;
    logic signed [17:0] dense_reorder_cfg_threshold = '0;
    logic dense_reorder_cfg_srnn_path = 1'b0;
    logic dense_fc_trace_valid, dense_fc_trace_ready;
    logic [5:0] dense_fc_trace_sample;
    logic [9:0] dense_fc_trace_hidden;
    logic [16:0] dense_fc_trace_q016;
    logic dense_fc_surrogate_valid, dense_fc_surrogate_ready;
    logic [5:0] dense_fc_surrogate_sample;
    logic [9:0] dense_fc_surrogate_hidden;
    logic [16:0] dense_fc_surrogate_q016;
    logic dense_srnn_backbone_valid, dense_srnn_backbone_ready;
    logic dense_srnn_backbone_path;
    logic [5:0] dense_srnn_backbone_sample;
    logic [9:0] dense_srnn_backbone_id;
    logic signed [31:0] dense_srnn_backbone_q14;
    logic dense_srnn_backbone_last;
    logic dense_capture_done, dense_replay_done;
    logic dense_reorder_busy, dense_reorder_idle, dense_reorder_error;
    logic dense_reorder_error_source;
    logic [3:0] dense_reorder_error_code;
    integer dense_capture_dones = 0, dense_replay_dones = 0;

    logic profile_valid = 1'b0, profile_ready, profile_activate, profile_reject;
    logic [6:0] profile_code = '0, active_profile_code;
    logic [1:0] profile_slot;
    logic [2:0] profile_mode;
    logic profile_active, profile_error;
    logic [15:0] profile_version;
    logic hold_dense_drain = 1'b0, hold_scnn_drain = 1'b0;
    logic frame_valid = 1'b0, frame_ready, frame_done, frame_busy, frame_error;
    logic descriptor_valid = 1'b0, descriptor_ready, descriptor_done, tail_active;
    logic [31:0] descriptor_data = '0, active_descriptor;
    bit force_timeout = $test$plusargs("force_profile_timeout");
    bit force_illegal = $test$plusargs("force_illegal_opcode");
    wire global_idle = idle && active_owner == 0 && !dense_reorder_busy &&
                       !frame_busy && !force_timeout;
    integer profile_accepts = 0, dense_accepts = 0, descriptor_accepts = 0;
    integer group_accepts = 0, recurrent_accepts = 0, maintenance_accepts = 0;
    integer reset_releases = 0, command_index = 0, expected_stage = 1;
    integer current_opcode = 0;
    bit reset_seen = 1'b0;
    bit previous_profile_stall = 1'b0, previous_descriptor_stall = 1'b0;
    logic [6:0] stalled_profile;
    logic [31:0] stalled_descriptor;

    unified_tb_profile_code_lookup_v1 u_profile (
        .i_clk(clk), .i_rst_n(rst_n), .i_clear_error(1'b0),
        .i_global_idle(global_idle),
        .i_profile_valid(profile_valid), .o_profile_ready(profile_ready),
        .i_profile_code(profile_code), .o_lookup_legal(),
        .o_profile_activate_pulse(profile_activate),
        .o_profile_reject_pulse(profile_reject), .o_active_valid(profile_active),
        .o_active_profile_code(active_profile_code), .o_active_slot(profile_slot),
        .o_active_mode(profile_mode), .o_active_cluster_mask(),
        .o_input_dim(), .o_hidden_or_channel_dim(), .o_output_dim(),
        .o_layer_count(), .o_fixed_timesteps(), .o_numeric_profile(),
        .o_fc_w2_update_enable(), .o_profile_version(profile_version),
        .o_sticky_error(profile_error), .o_error_code()
    );

    // Real final-top descriptor sequencer, but a bounded synthetic tag payload
    // and Conv8+classifier client, not final-composition descriptor packing.
    unified_scnn_frame_descriptor_sequencer_v1 #(.DESCRIPTOR_WIDTH(32)) u_frame (
        .i_clk(clk), .i_rst_n(rst_n), .i_clear_error(1'b0),
        .i_frame_start_valid(frame_valid), .o_frame_start_ready(frame_ready),
        .o_frame_done(frame_done), .i_descriptor_valid(descriptor_valid),
        .o_descriptor_ready(descriptor_ready), .i_descriptor(descriptor_data),
        .i_descriptor_last(1'b1), .o_descriptor_done(descriptor_done),
        .o_descriptor(active_descriptor), .o_tail_active(tail_active),
        .o_layer_start_valid(start_valid), .i_layer_start_ready(start_ready),
        .i_layer_done(scnn_done), .o_busy(frame_busy), .o_sticky_error(frame_error)
    );

    always @(posedge clk) begin
        if (rst_n) begin
            if (!reset_seen) begin
                reset_releases++;
                reset_seen = 1'b1;
            end
            if (previous_profile_stall &&
                (!profile_valid || profile_code != stalled_profile))
                $fatal(1, "IF_FATAL profile payload changed while stalled");
            if (previous_descriptor_stall &&
                (!descriptor_valid || descriptor_data != stalled_descriptor))
                $fatal(1, "IF_FATAL descriptor payload changed while stalled");
            previous_profile_stall = profile_valid && !profile_ready;
            previous_descriptor_stall = descriptor_valid && !descriptor_ready;
            stalled_profile = profile_code;
            stalled_descriptor = descriptor_data;
            if (profile_valid && profile_ready) profile_accepts++;
            if (dense_layer_valid && dense_layer_ready) dense_accepts++;
            if (descriptor_valid && descriptor_ready) descriptor_accepts++;
            if (dense_group_valid && dense_group_ready) begin
                group_accepts++;
                if (dense_group_recurrent) recurrent_accepts++;
            end
            if (maintenance_start_valid && maintenance_start_ready) maintenance_accepts++;
            if (sticky_error || tile_error != 0 || dense_error ||
                dense_observation_error || dense_reorder_error || frame_error)
                $fatal(1, "IF_FATAL native DUT reported an error");
        end else if (reset_seen)
            $fatal(1, "IF_FATAL global reset reasserted after initial startup");
    end

    assign feature_wave_ready = cycle_q[1:0] != 2'b01;
    assign logit_wave_ready = !hold_scnn_drain && cycle_q[2:0] != 3'b011;
    assign dense_fc_trace_ready = cycle_q[2:0] != 3'b101;
    assign dense_fc_surrogate_ready = cycle_q[2:0] != 3'b011;
    assign dense_srnn_backbone_ready = cycle_q[2:0] != 3'b110;
    assign feature_mask0 = response_term_q[0] ? 40'b1 : '0;
    assign feature_mask1 = response_term_q[1] ? 40'b1 : '0;
    assign feature_mask2 = response_term_q[2] ? 40'b1 : '0;
    assign feature_mask3 = response_term_q[3] ? 40'b1 : '0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_q <= '0;
            feature_response_valid <= 1'b0;
            response_term_q <= '0;
            feature_update_done <= 1'b0;
        end else begin
            cycle_q <= cycle_q + 1'b1;
            feature_update_done <= 1'b0;
            if (feature_request_valid && feature_request_ready) begin
                feature_response_valid <= 1'b1;
                response_term_q <= feature_term_valid;
            end
            if (feature_response_valid && feature_response_ready)
                feature_response_valid <= 1'b0;
            if (feature_update_valid && feature_update_ready)
                feature_update_done <= 1'b1;
        end
    end

    always_ff @(posedge clk) begin
        if (rst_n) begin
            if (context_load_valid && context_load_ready) begin
                loads <= loads + 1;
                if (loads == 0 && context_channel_count != 10'd2)
                    $fatal(1, "wrong real-array convolution context");
                if (loads == 1 && context_channel_count != 10'd3)
                    $fatal(1, "wrong real-array classifier context");
            end
            if (context_release_valid)
                releases <= releases + 1;
            if (feature_wave_valid && feature_wave_ready) begin
                feature_waves <= feature_waves + 1;
                if (feature_wave_mask != 8'b00000011 ||
                    feature_wave_flat[15:0] != 16'h0100 ||
                    feature_wave_flat[31:16] != 16'h0100)
                    $fatal(1, "real array did not produce two Conv8 spikes");
            end
            if (logit_wave_valid && logit_wave_ready) begin
                logit_waves <= logit_waves + 1;
                if (logit_wave_mask != 12'b000000000111 ||
                    logit_wave_base != 0 || logit_wave_count != 5'd3)
                    $fatal(1, "bad real-array classifier logit envelope");
            end
            if (dense_capture_done)
                dense_capture_dones <= dense_capture_dones + 1;
            if (dense_replay_done)
                dense_replay_dones <= dense_replay_dones + 1;
            if (dense_fc_trace_valid && dense_fc_trace_ready) begin
                if (dense_phase != 1 || dense_fc_trace_sample != 6'd5 ||
                    dense_fc_trace_hidden != 10'(fc_observations) ||
                    // FC keeps its resident trace in UQ6.10.  The shared
                    // learning service interprets this raw code with the
                    // FC-specific multiplier, so one spike is 1.0 = 1024.
                    dense_fc_trace_q016 != 17'd1024 ||
                    dense_fc_trace_hidden >= 80 ||
                    fc_seen[dense_fc_trace_hidden])
                    $fatal(1,
                        "FC trace learning ingress mismatch phase=%0d sample=%0d hidden=%0d expected_hidden=%0d trace=%0d seen=%0b",
                        dense_phase, dense_fc_trace_sample,
                        dense_fc_trace_hidden, fc_observations,
                        dense_fc_trace_q016,
                        fc_seen[dense_fc_trace_hidden]);
                fc_seen[dense_fc_trace_hidden] <= 1'b1;
                fc_observations <= fc_observations + 1;
            end
            if (dense_fc_surrogate_valid && dense_fc_surrogate_ready) begin
                if (dense_phase != 1 || dense_fc_surrogate_sample != 6'd5 ||
                    dense_fc_surrogate_hidden !=
                        10'(fc_surrogate_observations) ||
                    dense_fc_surrogate_q016 != 17'd65536 ||
                    dense_fc_surrogate_hidden >= 80 ||
                    fc_surrogate_seen[dense_fc_surrogate_hidden])
                    $fatal(1, "FC surrogate learning ingress mismatch");
                fc_surrogate_seen[dense_fc_surrogate_hidden] <= 1'b1;
                fc_surrogate_observations <=
                    fc_surrogate_observations + 1;
            end
            if (dense_srnn_backbone_valid && dense_srnn_backbone_ready) begin
                if (dense_phase != 2 || !dense_srnn_backbone_path ||
                    dense_srnn_backbone_sample != 6'd1 ||
                    dense_srnn_backbone_id != 10'(srnn_observations) ||
                    dense_srnn_backbone_q14 != 32'sd16384 ||
                    dense_srnn_backbone_last != (srnn_observations == 79) ||
                    dense_srnn_backbone_id >= 80 ||
                    srnn_seen[dense_srnn_backbone_id])
                    $fatal(1, "SRNN backbone learning ingress mismatch");
                srnn_seen[dense_srnn_backbone_id] <= 1'b1;
                srnn_observations <= srnn_observations + 1;
            end
        end
    end

    task automatic fill_weight(
        input int tile, input int logical_bank, input int row,
        input logic [199:0] data
    );
        int region;
        begin
            region = tile / 4;
            @(negedge clk);
            weight_fill_valid[region] = 1'b1;
            weight_fill_tile_flat[region*2 +: 2] = 2'(tile % 4);
            weight_fill_bank[region] = logical_bank[0];
            weight_fill_addr_flat[region*AW +: AW] =
                {1'(logical_bank >> 1), 10'(row)};
            weight_fill_data_flat[region*200 +: 200] = data;
            #1;
            if (!weight_fill_ready[region])
                $fatal(1, "weight fill stalled tile=%0d bank=%0d",
                    tile, logical_bank);
            @(posedge clk); @(negedge clk);
            weight_fill_valid[region] = 1'b0;
        end
    endtask

    task automatic acquire_maintenance;
        begin
            @(negedge clk);
            maintenance_start_valid = 1'b1;
            do @(posedge clk); while (!maintenance_start_ready);
            @(negedge clk);
            maintenance_start_valid = 1'b0;
            if (active_owner != 2'd3)
                $fatal(1, "maintenance lease was not acquired");
        end
    endtask

    task automatic release_maintenance;
        begin
            @(negedge clk);
            maintenance_release_valid = 1'b1;
            do @(posedge clk); while (!maintenance_release_ready);
            @(negedge clk);
            maintenance_release_valid = 1'b0;
            if (!maintenance_done || active_owner != 2'd0)
                $fatal(1, "maintenance lease was not released atomically");
        end
    endtask

    task automatic clear_work(input int tile, input int work_base);
        begin
            @(negedge clk);
            maintenance_tile = 4'(tile);
            state_write_valid = 1'b1;
            state_write_addr = {1'b0, 9'(work_base)};
            state_write_data = '0;
            state_write_mask = '1;
            trace_write_valid = 1'b1;
            trace_write_addr = {1'b1, 9'(work_base)};
            trace_write_data = '0;
            trace_write_mask = '1;
            #1;
            if (!state_write_ready || !trace_write_ready)
                $fatal(1, "work clear stalled tile=%0d", tile);
            @(posedge clk); @(negedge clk);
            state_write_valid = 1'b0;
            trace_write_valid = 1'b0;
        end
    endtask

    task automatic launch_dense_layer(
        input logic [1:0] launch_mode,
        input logic recurrence_enable,
        input logic [15:0] feedforward_groups,
        input logic [15:0] recurrent_groups,
        input logic [51:0] tag
    );
        begin
            @(negedge clk);
            dense_reorder_cfg_mode = launch_mode;
            dense_reorder_cfg_tag = tag;
            dense_reorder_cfg_base = 10'd0;
            dense_reorder_cfg_count = 10'd80;
            dense_reorder_cfg_sample = launch_mode == 2'd0 ? 6'd5 : 6'd1;
            dense_reorder_cfg_threshold = dense_threshold[17:0];
            dense_reorder_cfg_srnn_path = recurrence_enable;
            dense_reorder_cfg_valid = 1'b1;
            #1;
            if (!dense_reorder_cfg_ready)
                $fatal(1, "dense observation reorder not ready");
            @(posedge clk); @(negedge clk);
            dense_reorder_cfg_valid = 1'b0;

            dense_mode = launch_mode;
            dense_neuron_count = 16'd80;
            dense_feedforward_groups = feedforward_groups;
            dense_recurrent_groups = recurrent_groups;
            dense_recurrent_enable = recurrence_enable;
            dense_transaction_tag = tag;
            dense_layer_valid = 1'b1;
            #1;
            if (!dense_layer_ready)
                $fatal(1, "dense layer client not ready");
            @(posedge clk); @(negedge clk);
            dense_layer_valid = 1'b0;
            if (!dense_busy || dense_idle || dense_error)
                $fatal(1,
                    "dense layer launch failed mode=%0d ready=%0b busy=%0b idle=%0b error=%0h client_state=%0d descriptor_legal=%0b array_error=%0b",
                    launch_mode, dense_layer_ready, dense_busy, dense_idle,
                    dense_error_code,
                    dut.g_fc_srnn_layer.u_fc_srnn_client.state_q,
                    dut.g_fc_srnn_layer.u_fc_srnn_client.descriptor_legal,
                    dut.array_error);
        end
    endtask

    task automatic push_dense_group(
        input logic recurrent,
        input logic [3:0] valid_mask,
        input int row
    );
        int timeout;
        logic accepted;
        begin
            @(negedge clk);
            dense_group_recurrent = recurrent;
            dense_group_source_valid = valid_mask;
            for (int term = 0; term < 4; term++)
                dense_group_addr[term] = AW'(row);
            dense_group_valid = 1'b1;
            timeout = 0;
            accepted = 1'b0;
            while (!accepted && timeout < 20000) begin
                @(posedge clk);
                accepted = dense_group_valid && dense_group_ready;
                timeout++;
            end
            if (!accepted)
                $fatal(1,
                    "dense group timed out error=%0h client_state=%0d client_valid=%0b owner=%0d owner_idle=%0b select_direct=%0b direct_ready=%0b array_ready=%0b array_idle=%0b scnn_idle=%0b sticky=%0b",
                    dense_error_code,
                    dut.g_fc_srnn_layer.u_fc_srnn_client.state_q,
                    dut.direct_command_valid, dut.u_owner.owner_q,
                    dut.u_owner.acquisition_idle, dut.select_direct,
                    dut.direct_command_ready, dut.array_command_ready,
                    dut.array_idle, dut.scnn_client_idle, sticky_error);
            @(negedge clk);
            dense_group_valid = 1'b0;
        end
    endtask

    task automatic wait_dense_done;
        int timeout;
        begin
            timeout = 0;
            while (!dense_layer_done && timeout < 20000) begin
                @(negedge clk);
                timeout++;
            end
            if (!dense_layer_done || dense_error || sticky_error)
                $fatal(1, "dense layer failed error=%0h", dense_error_code);
            @(posedge clk); @(negedge clk);
            if (dense_busy || !dense_idle)
                $fatal(1, "dense client did not return idle");
        end
    endtask

    unified_h40_array16_three_topology_top_v1 #(
        .ENABLE_FC_SRNN_LAYER_CLIENT(1'b1),
        .ENABLE_FC_SRNN_TP_OBSERVATION_STREAM(1'b1),
        .ENABLE_SCNN_CLASSIFIER_GRAPH(1'b1),
        .MAX_SCNN_FEATURE_CHANNELS(16),
        .SCNN_CLASSIFIER_FEATURE_SPATIAL(1), .MAX_SCNN_CLASSES(16),
        .SCNN_CHANNEL_WIDTH(4), .SCNN_CHANNEL_COUNT_WIDTH(5),
        .SCNN_FEATURE_WIDTH(4), .SCNN_CLASS_WIDTH(4),
        .SCNN_CLASS_COUNT_WIDTH(5)
    ) dut (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(1'b0),
        .i_clear_error(1'b0),
        .i_direct_command_valid(1'b0), .i_direct_mode(2'd0),
        .i_direct_neuron_count(16'd0), .i_direct_first(1'b0),
        .i_direct_last(1'b0), .i_direct_source_valid(4'b0),
        .i_direct_weight_addr0('0), .i_direct_weight_addr1('0),
        .i_direct_weight_addr2('0), .i_direct_weight_addr3('0),
        .i_direct_leak('0), .i_direct_threshold('0),
        .i_direct_reset_to_zero(1'b0), .i_direct_trace_decay('0),
        .i_direct_transaction_tag('0),
        .i_direct_work_base_addr('0),
        .i_direct_destination_tile_base(4'd0),
        .i_direct_observation_ready(4'b1111),
        .i_fc_srnn_layer_valid(dense_layer_valid),
        .o_fc_srnn_layer_ready(dense_layer_ready),
        .i_fc_srnn_mode(dense_mode),
        .i_fc_srnn_neuron_count(dense_neuron_count),
        .i_fc_srnn_feedforward_group_count(dense_feedforward_groups),
        .i_fc_srnn_recurrent_group_count(dense_recurrent_groups),
        .i_fc_srnn_recurrent_enable(dense_recurrent_enable),
        .i_fc_srnn_leak(16'd0),
        .i_fc_srnn_threshold(dense_threshold),
        .i_fc_srnn_reset_to_zero(1'b1),
        .i_fc_srnn_trace_decay(16'd0),
        .i_fc_srnn_transaction_tag(dense_transaction_tag),
        .i_fc_srnn_work_base_addr(9'd20),
        .i_fc_srnn_destination_tile_base(4'd0),
        .i_fc_srnn_group_valid(dense_group_valid),
        .o_fc_srnn_group_ready(dense_group_ready),
        .i_fc_srnn_group_recurrent(dense_group_recurrent),
        .i_fc_srnn_group_source_valid(dense_group_source_valid),
        .i_fc_srnn_group_weight_addr0(dense_group_addr[0]),
        .i_fc_srnn_group_weight_addr1(dense_group_addr[1]),
        .i_fc_srnn_group_weight_addr2(dense_group_addr[2]),
        .i_fc_srnn_group_weight_addr3(dense_group_addr[3]),
        .o_fc_srnn_layer_done(dense_layer_done),
        .o_fc_srnn_busy(dense_busy), .o_fc_srnn_idle(dense_idle),
        .o_fc_srnn_sticky_error(dense_error),
        .o_fc_srnn_error_code(dense_error_code),
        .o_fc_srnn_observation_valid(dense_observation_valid),
        .i_fc_srnn_observation_ready(dense_observation_ready),
        .i_fc_srnn_observation_sink_idle(dense_reorder_idle && !hold_dense_drain),
        .o_fc_srnn_observation_mode(dense_observation_mode),
        .o_fc_srnn_observation_transaction_tag(dense_observation_tag),
        .o_fc_srnn_observation_neuron_id(dense_observation_neuron),
        .o_fc_srnn_observation_spike(dense_observation_spike),
        .o_fc_srnn_observation_pre_state(dense_observation_pre_state),
        .o_fc_srnn_observation_trace_next(dense_observation_trace_next),
        .o_fc_srnn_observation_last_in_record(dense_observation_last),
        .o_fc_srnn_observation_busy(dense_observation_busy),
        .o_fc_srnn_observation_sticky_error(dense_observation_error),
        .o_direct_observation_valid(direct_observation_valid),
        .o_direct_observation_mode_flat(direct_observation_mode_flat),
        .o_direct_observation_lane_mask_flat(
            direct_observation_lane_mask_flat),
        .o_direct_observation_spike_flat(direct_observation_spike_flat),
        .o_direct_observation_pre_state_flat(
            direct_observation_pre_state_flat),
        .i_scnn_start_valid(start_valid), .o_scnn_start_ready(start_ready),
        .i_scnn_in_width(16'd3), .i_scnn_in_height(16'd3),
        .i_scnn_in_channels(16'd1), .i_scnn_out_width(16'd1),
        .i_scnn_out_height(16'd1), .i_scnn_out_channels(16'd2),
        .i_scnn_stride(8'd1), .i_scnn_padding(8'd0),
        .i_scnn_pool_mode(2'd0), .i_scnn_initial_active_pair(1'b0),
        .i_scnn_work_base_addr(9'd4), .i_scnn_leak(16'd0),
        .i_scnn_threshold(32'd1), .i_scnn_reset_to_zero(1'b1),
        .i_scnn_trace_decay(16'd0), .i_scnn_layer_tag(active_descriptor[19:0]),
        .i_scnn_classifier_feature_channel_base(4'd0),
        .i_scnn_classifier_feature_channel_count(5'd2),
        .i_scnn_classifier_class_count(5'd3),
        .i_scnn_classifier_active_pair(1'b1),
        .i_scnn_classifier_work_base_addr(9'd20),
        .i_scnn_classifier_trace_decay(16'd200),
        .i_scnn_classifier_layer_tag(CLASS_TAG),
        .i_scnn_version(16'h1234), .i_scnn_frame_index(8'd7),
        .i_scnn_sample_reset(1'b1), .i_scnn_sample_last(1'b1),
        .i_scnn_sample_label(4'd2), .i_scnn_learning_enable(1'b0),
        .i_scnn_conv8_observation_enable(1'b0),
        .i_scnn_conv8_observation_target(1'b0),
        .i_scnn_sample_index('0), .i_scnn_timestep_index('0),
        .o_scnn_context_load_valid(context_load_valid),
        .i_scnn_context_load_ready(context_load_ready),
        .o_scnn_context_load_weights(context_load_weights),
        .o_scnn_context_active_pair(context_active_pair),
        .o_scnn_context_output_channel_count(context_channel_count),
        .o_scnn_context_release_valid(context_release_valid),
        .i_scnn_context_release_ready(1'b1),
        .o_scnn_feature_request_valid(feature_request_valid),
        .i_scnn_feature_request_ready(feature_request_ready),
        .o_scnn_feature_term_valid(feature_term_valid),
        .i_scnn_feature_response_valid(feature_response_valid),
        .o_scnn_feature_response_ready(feature_response_ready),
        .i_scnn_feature_mask0(feature_mask0),
        .i_scnn_feature_mask1(feature_mask1),
        .i_scnn_feature_mask2(feature_mask2),
        .i_scnn_feature_mask3(feature_mask3),
        .i_scnn_feature_service_error(1'b0),
        .i_scnn_feature_write_slot(1'b1),
        .i_scnn_feature_write_base_addr(13'd32),
        .i_scnn_feature_write_row_stride_words(16'd1),
        .i_scnn_feature_write_channel_stride_words(13'd1),
        .o_scnn_feature_update_valid(feature_update_valid),
        .i_scnn_feature_update_ready(feature_update_ready),
        .i_scnn_feature_update_done(feature_update_done),
        .i_scnn_observation_ready(4'b1111),
        .i_scnn_observation_sink_idle(!hold_scnn_drain),
        .o_scnn_done(scnn_done),
        .o_scnn_feature_trace_wave_valid(feature_wave_valid),
        .i_scnn_feature_trace_wave_ready(feature_wave_ready),
        .o_scnn_feature_trace_wave_mask(feature_wave_mask),
        .o_scnn_feature_trace_wave_base(feature_wave_base),
        .o_scnn_feature_trace_wave_q8_8_flat(feature_wave_flat),
        .o_scnn_conv8_surrogate_valid(),
        .i_scnn_conv8_surrogate_ready(1'b1),
        .o_scnn_conv8_surrogate_target(), .o_scnn_conv8_surrogate_sample(),
        .o_scnn_conv8_surrogate_timestep(), .o_scnn_conv8_surrogate_version(),
        .o_scnn_conv8_surrogate_layer_tag(), .o_scnn_conv8_surrogate_channel(),
        .o_scnn_conv8_surrogate_spatial(), .o_scnn_conv8_membrane_pre_q8_8(),
        .o_scnn_conv8_surrogate_q8_8(),
        .o_scnn_logit_wave_valid(logit_wave_valid),
        .i_scnn_logit_wave_ready(logit_wave_ready),
        .o_scnn_logit_wave_mask(logit_wave_mask),
        .o_scnn_logit_wave_base(logit_wave_base),
        .o_scnn_logit_wave_count(logit_wave_count),
        .o_scnn_logit_wave_q8_8_flat(logit_wave_flat),
        .i_weight_fill_valid(weight_fill_valid),
        .o_weight_fill_ready(weight_fill_ready),
        .o_scnn_context_refill_safe(context_refill_safe),
        .i_weight_fill_tile_flat(weight_fill_tile_flat),
        .i_weight_fill_bank(weight_fill_bank),
        .i_weight_fill_addr_flat(weight_fill_addr_flat),
        .i_weight_fill_data_flat(weight_fill_data_flat),
        .i_maintenance_start_valid(maintenance_start_valid),
        .o_maintenance_start_ready(maintenance_start_ready),
        .i_maintenance_release_valid(maintenance_release_valid),
        .o_maintenance_release_ready(maintenance_release_ready),
        .o_maintenance_done(maintenance_done),
        .i_maintenance_tile(maintenance_tile),
        .i_state_maintenance_write_valid(state_write_valid),
        .o_state_maintenance_write_ready(state_write_ready),
        .i_state_maintenance_write_addr(state_write_addr),
        .i_state_maintenance_write_data(state_write_data),
        .i_state_maintenance_write_mask(state_write_mask),
        .i_state_maintenance_read_valid(1'b0),
        .i_state_maintenance_read_addr('0),
        .i_state_maintenance_result_ready(1'b1),
        .i_trace_maintenance_write_valid(trace_write_valid),
        .o_trace_maintenance_write_ready(trace_write_ready),
        .i_trace_maintenance_write_addr(trace_write_addr),
        .i_trace_maintenance_write_data(trace_write_data),
        .i_trace_maintenance_write_mask(trace_write_mask),
        .i_trace_maintenance_read_valid(1'b0),
        .i_trace_maintenance_read_addr('0),
        .i_trace_maintenance_result_ready(1'b1),
        .o_active_owner(active_owner), .o_busy(busy), .o_idle(idle),
        .o_tile_idle(tile_idle), .o_tile_sticky_error(tile_error),
        .o_sticky_error(sticky_error)
    );

    unified_fc_srnn_h40_tp_observation_ingress_v1 u_dense_reorder (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(1'b0),
        .i_clear_error(1'b0),
        .i_cfg_valid(dense_reorder_cfg_valid),
        .o_cfg_ready(dense_reorder_cfg_ready),
        .i_cfg_mode(dense_reorder_cfg_mode),
        .i_cfg_transaction_tag(dense_reorder_cfg_tag),
        .i_cfg_sample_id(dense_reorder_cfg_sample),
        .i_cfg_neuron_base(dense_reorder_cfg_base),
        .i_cfg_neuron_count(dense_reorder_cfg_count),
        .i_cfg_fc_threshold(dense_reorder_cfg_threshold),
        .i_cfg_srnn_path(dense_reorder_cfg_srnn_path),
        .i_scalar_valid(dense_observation_valid),
        .o_scalar_ready(dense_observation_ready),
        .i_scalar_mode(dense_observation_mode),
        .i_scalar_transaction_tag(dense_observation_tag),
        .i_scalar_neuron_id(dense_observation_neuron),
        .i_scalar_spike(dense_observation_spike),
        .i_scalar_pre_state(dense_observation_pre_state),
        .i_scalar_trace_next(dense_observation_trace_next),
        .o_fc_student_trace_valid(dense_fc_trace_valid),
        .i_fc_student_trace_ready(dense_fc_trace_ready),
        .o_fc_student_trace_sample(dense_fc_trace_sample),
        .o_fc_student_trace_hidden(dense_fc_trace_hidden),
        .o_fc_student_trace_q016(dense_fc_trace_q016),
        .o_fc_surrogate_valid(dense_fc_surrogate_valid),
        .i_fc_surrogate_ready(dense_fc_surrogate_ready),
        .o_fc_surrogate_sample(dense_fc_surrogate_sample),
        .o_fc_surrogate_hidden(dense_fc_surrogate_hidden),
        .o_fc_surrogate_q016(dense_fc_surrogate_q016),
        .o_srnn_backbone_valid(dense_srnn_backbone_valid),
        .i_srnn_backbone_ready(dense_srnn_backbone_ready),
        .o_srnn_backbone_path(dense_srnn_backbone_path),
        .o_srnn_backbone_sample_id(dense_srnn_backbone_sample),
        .o_srnn_backbone_id(dense_srnn_backbone_id),
        .o_srnn_backbone_q14(dense_srnn_backbone_q14),
        .o_srnn_backbone_last(dense_srnn_backbone_last),
        .o_capture_done(dense_capture_done),
        .o_done(dense_replay_done),
        .o_busy(dense_reorder_busy), .o_idle(dense_reorder_idle),
        .o_sticky_error(dense_reorder_error),
        .o_error_source(dense_reorder_error_source),
        .o_error_code(dense_reorder_error_code)
    );

    task automatic wait_quiescent;
        int timeout;
        begin
            timeout = 0;
            while ((!idle || active_owner != 0 || dense_reorder_busy || frame_busy) &&
                   timeout < 20000) begin
                @(negedge clk);
                timeout++;
            end
            if (!idle || active_owner != 0 || dense_reorder_busy || frame_busy)
                $fatal(1, "IF_FATAL OWNER_DRAIN_TIMEOUT");
        end
    endtask

    task automatic load_fixture(input int opcode);
        begin
            acquire_maintenance();
            if (command_index == 2) begin
                @(negedge clk);
                maintenance_tile = 4'd12;
                state_write_valid = 1'b1;
                state_write_mask = '1;
                #1;
                if (state_write_ready || weight_fill_ready[3])
                    $fatal(1, "IF_FATAL absent compatibility region accepted a write");
                @(posedge clk); @(negedge clk);
                state_write_valid = 1'b0;
            end
            if (opcode == 1 || opcode == 16) begin
                for (int tile = 0; tile < 2; tile++) begin
                    for (int bank = 0; bank < 4; bank++)
                        if (opcode == 1)
                            fill_weight(tile, bank, 1, {40{5'sd1}});
                        else
                            fill_weight(tile, bank, 2, {25{8'sd1}});
                    for (int row = 0; row < 4; row++) clear_work(tile, row);
                end
            end else if (opcode == 32) begin
                for (int tile = 0; tile < 3; tile++) begin
                    for (int bank = 0; bank < 4; bank++)
                        fill_weight(tile, bank, 0, {25{8'sd1}});
                    clear_work(tile, 4); clear_work(tile, 5);
                    clear_work(tile, 20); clear_work(tile, 21);
                end
            end else
                $fatal(1, "IF_FATAL no fixture for opcode");
            // Retain the real maintenance lease until the profile request is
            // held stable and backpressured against that owner's release.
        end
    endtask

    task automatic request_profile(input int opcode);
        int timeout;
        logic [15:0] old_version;
        begin
            @(negedge clk);
            old_version = profile_version;
            profile_code = force_illegal ? 7'h7f : 7'(opcode);
            profile_valid = 1'b1;
            repeat (5) begin
                @(posedge clk); @(negedge clk);
                if (profile_ready || active_owner != 3 ||
                    profile_version != old_version)
                    $fatal(1, "IF_FATAL profile changed before maintenance drain");
            end
            release_maintenance();
            timeout = 0;
            do begin
                @(posedge clk);
                timeout++;
            end while (!profile_ready && timeout < 2000);
            if (!profile_ready) $fatal(1, "IF_FATAL PROFILE_TIMEOUT");
            @(negedge clk);
            profile_valid = 1'b0;
            if (profile_reject || profile_error)
                $fatal(1, "IF_FATAL PROFILE_REJECT");
            if (!profile_activate || !profile_active ||
                active_profile_code != 7'(opcode) || profile_version != old_version + 1)
                $fatal(1, "IF_FATAL PROFILE_ALIGNMENT");
            $display("IF_PROFILE,%0d,%0d,5", opcode, profile_version);
        end
    endtask

    task automatic hold_and_release_drain(input int opcode);
        begin
            repeat (5) begin
                @(posedge clk); @(negedge clk);
                if (active_owner != (opcode == 32 ? 2 : 1) ||
                    profile_ready || idle || scnn_done || dense_layer_done)
                    $fatal(1, "IF_FATAL owner released before sink drain opcode=%0d owner=%0d idle=%0b profile_ready=%0b",
                        opcode, active_owner, idle, profile_ready);
            end
            $display("IF_DRAIN,%0d,5", opcode);
            hold_dense_drain = 1'b0;
            hold_scnn_drain = 1'b0;
        end
    endtask

    task automatic dispatch_dense(input int opcode, input int payload);
        int timeout;
        int previous_replays;
        begin
            if (payload != 80 || (profile_mode != 1 && profile_mode != 2))
                $fatal(1, "IF_FATAL wrong dense descriptor");
            @(negedge clk);
            fc_observations = 0; fc_surrogate_observations = 0; srnn_observations = 0;
            fc_seen = '0; fc_surrogate_seen = '0; srnn_seen = '0;
            previous_replays = dense_replay_dones;
            dense_phase = opcode == 1 ? 1 : 2;
            dense_threshold = opcode == 1 ? 32'd3 : 32'd7;
            hold_dense_drain = 1'b1;
            launch_dense_layer(2'(profile_mode - 1), profile_mode == 2,
                16'd1, opcode == 1 ? 16'd7 : 16'd1, 52'(command_index));
            push_dense_group(1'b0, 4'b1111, opcode == 1 ? 1 : 2);
            if (opcode == 16) push_dense_group(1'b1, 4'b1111, 2);
            timeout = 0;
            while ((dense_replay_dones == previous_replays || dense_reorder_busy ||
                    dense_observation_busy || !dut.array_idle) && timeout < 20000) begin
                @(negedge clk);
                timeout++;
            end
            if (timeout >= 20000) $fatal(1, "IF_FATAL dense observation timeout");
            hold_and_release_drain(opcode);
            wait_dense_done();
            if (opcode == 1 &&
                (fc_observations != 80 || fc_surrogate_observations != 80 ||
                 fc_seen != '1 || fc_surrogate_seen != '1))
                $fatal(1, "IF_FATAL FC fixture observation mismatch");
            if (opcode == 16 && (srnn_observations != 80 || srnn_seen != '1))
                $fatal(1, "IF_FATAL SRNN fixture observation mismatch");
            $display("IF_OBSERVATIONS,%0d,%0d,%0d,%0d,0",
                opcode, fc_observations, fc_surrogate_observations, srnn_observations);
            dense_phase = 0;
        end
    endtask

    task automatic serve_contexts;
        int served;
        begin
            served = 0;
            while (served < 2) begin
                wait(context_load_valid);
                if (served == 0) begin
                    if (!context_load_weights || context_active_pair != 0)
                        $fatal(1, "IF_FATAL expected Conv refill");
                    wait(context_refill_safe);
                    if (active_owner != 2)
                        $fatal(1, "IF_FATAL refill outside SCNN owner");
                    fill_weight(0, 0, 0, {25{8'sd1}});
                end else if (context_active_pair != 1 || context_refill_safe)
                    $fatal(1, "IF_FATAL classifier exposed a refill window");
                @(negedge clk);
                context_load_ready = 1'b1;
                @(posedge clk); @(negedge clk);
                context_load_ready = 1'b0;
                served++;
                wait(!context_load_valid);
            end
        end
    endtask

    task automatic dispatch_scnn(input int payload);
        int timeout;
        begin
            @(negedge clk);
            descriptor_data = 32'(payload);
            descriptor_valid = 1'b1;
            repeat (3) begin
                @(posedge clk); @(negedge clk);
                if (descriptor_ready)
                    $fatal(1, "IF_FATAL descriptor accepted outside a frame");
            end
            frame_valid = 1'b1;
            do @(posedge clk); while (!frame_ready);
            @(negedge clk); frame_valid = 1'b0;
            do @(posedge clk); while (!descriptor_ready);
            @(negedge clk); descriptor_valid = 1'b0;
            do @(posedge clk); while (!(start_valid && start_ready));
            @(negedge clk);
            if (!tail_active || active_descriptor != descriptor_data)
                $fatal(1, "IF_FATAL descriptor capture mismatch");
            hold_scnn_drain = 1'b1;
            fork serve_contexts(); join_none
            timeout = 0;
            // Graph-enabled ownership drains through the graph client's
            // actual outputs, not i_scnn_observation_sink_idle.
            while (!logit_wave_valid && timeout < 20000) begin
                @(negedge clk);
                timeout++;
            end
            if (!logit_wave_valid) $fatal(1, "IF_FATAL SCNN compute timeout");
            hold_and_release_drain(32);
            timeout = 0;
            while (!frame_done && timeout < 20000) begin
                @(negedge clk);
                timeout++;
            end
            if (!frame_done || !descriptor_done || loads != 2 || releases != 2 ||
                feature_waves != 0 || logit_waves != 1)
                $fatal(1, "IF_FATAL SCNN descriptor/frame completion mismatch");
            @(posedge clk); @(negedge clk);
            $display("IF_OBSERVATIONS,32,0,0,0,1");
        end
    endtask

    initial begin : replay
        string program_path;
        int fd, scan_count, operation, opcode, payload;
        bit stopped;
        stopped = 1'b0;
        for (int term = 0; term < 4; term++) dense_group_addr[term] = '0;
        if (!$value$plusargs("program=%s", program_path))
            $fatal(1, "IF_FATAL program path is required");
        fd = $fopen(program_path, "r");
        if (fd == 0) $fatal(1, "IF_FATAL cannot read program");
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        while (!$feof(fd)) begin
            scan_count = $fscanf(fd, "%d %d %d\n", operation, opcode, payload);
            if (scan_count != 3) $fatal(1, "IF_FATAL malformed program");
            if (operation != expected_stage || (opcode != 1 && opcode != 16 && opcode != 32))
                $fatal(1, "IF_FATAL program operation/opcode mismatch");
            if (operation == 1) current_opcode = opcode;
            else if (opcode != current_opcode)
                $fatal(1, "IF_FATAL opcode changed within transaction");
            case (operation)
                1: stopped = 1'b1;
                2: wait_quiescent();
                3: load_fixture(opcode);
                4: request_profile(opcode);
                5: if (!profile_active || active_profile_code != 7'(opcode) ||
                       profile_slot != (opcode == 1 ? 0 : opcode == 16 ? 2 : 3))
                       $fatal(1, "IF_FATAL active profile not visible");
                6: begin
                    if (!stopped) $fatal(1, "IF_FATAL work admitted before stop");
                    if (opcode == 32) dispatch_scnn(payload);
                    else dispatch_dense(opcode, payload);
                end
                7: wait_quiescent();
                8: begin
                    wait_quiescent();
                    stopped = 1'b0;
                end
                default: $fatal(1, "IF_FATAL unknown operation");
            endcase
            $display("IF_ACK,%0d,%0d,%0d,%0d", command_index, operation, opcode, payload);
            command_index++;
            expected_stage = operation == 8 ? 1 : operation + 1;
        end
        $fclose(fd);
        if (command_index != 32 || expected_stage != 1 || stopped ||
            active_owner != 0 || !idle || busy || tile_idle != 16'hffff ||
            profile_version != 4 || dense_capture_dones != 3 || dense_replay_dones != 3)
            $fatal(1, "IF_FATAL incomplete roundtrip");
        $display("IF_COUNTS,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
            reset_releases, profile_accepts, dense_accepts, descriptor_accepts,
            group_accepts, recurrent_accepts, maintenance_accepts, loads);
        $display("IF_PASS");
        $finish;
    end

    initial begin
        repeat (200000) @(posedge clk);
        $fatal(1, "IF_FATAL GLOBAL_WATCHDOG");
    end
endmodule

`default_nettype wire

