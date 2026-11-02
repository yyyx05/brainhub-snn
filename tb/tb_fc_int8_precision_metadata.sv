`timescale 1ns/1ps
`default_nettype none

module tb_fc_int8_precision_metadata;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic abort = 1'b0;
    logic clear_error = 1'b0;
    logic layer_valid = 1'b0;
    logic layer_ready;
    logic owner_ready = 1'b0;
    logic owner_valid;
    logic [1:0] mode = '0;
    logic int8_flag = 1'b0;
    logic shift6_flag = 1'b0;
    logic [15:0] neurons = 16'd10;
    logic [3:0] tile_base = '0;
    logic group_valid = 1'b0;
    logic group_ready;
    logic dispatch_enable = 1'b0;
    logic layer_done;
    logic client_idle;
    logic client_error;
    logic [7:0] client_error_code;
    logic transaction_done = 1'b0;
    logic command_valid;
    logic [1:0] command_mode;
    logic command_int8;
    logic command_shift6;
    logic [15:0] command_neurons;
    logic [3:0] command_tile_base;
    logic command_first;
    logic command_last;
    logic walker_ready;
    logic walker_done;
    logic walker_error;
    logic [15:0] issue_valid;
    logic [15:0] issue_ready = '0;
    logic [15:0] issue_int8;
    logic [31:0] issue_mode;
    logic [639:0] issue_lanes;
    logic [11519:0] issue_factor;
    logic signed [17:0] scnn_factor = -18'sd257;
    logic [15:0] issue_first;
    logic [15:0] issue_last;
    logic bypass = 1'b0;
    logic raw_valid = 1'b0;
    logic legacy_valid = 1'b0;
    logic legacy_int8;
    logic legacy_shift6;
    logic legacy_command_valid;
    integer layer_count = 0;
    integer rejected_count = 0;
    integer raw_rejected_count = 0;
    integer abort_count = 0;

    always #5 clk = ~clk;
    initial begin
        #100000;
        $fatal(1, "metadata watchdog expired");
    end

    unified_fc_srnn_h40_p4_layer_client_v1 u_client (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(abort),
        .i_clear_error(clear_error), .i_layer_valid(layer_valid),
        .o_layer_ready(layer_ready), .o_owner_request_valid(owner_valid),
        .i_owner_request_ready(owner_ready), .i_mode(mode),
        .i_fc_weight_int8(int8_flag), .i_neuron_count(neurons),
        .i_fc_current_shift6(shift6_flag),
        .i_feedforward_group_count(16'd2), .i_recurrent_group_count(16'd0),
        .i_recurrent_enable(1'b0), .i_leak(16'd0), .i_threshold(32'd100),
        .i_reset_to_zero(1'b1), .i_trace_decay(16'd0),
        .i_transaction_tag(52'd1), .i_work_base_addr(9'd0),
        .i_destination_tile_base(tile_base), .i_group_valid(group_valid),
        .o_group_ready(group_ready), .i_group_recurrent(1'b0),
        .i_group_source_valid(4'b1111), .i_group_weight_addr0(11'd0),
        .i_group_weight_addr1(11'd1), .i_group_weight_addr2(11'd2),
        .i_group_weight_addr3(11'd3), .o_array_command_valid(command_valid),
        .i_array_command_ready(dispatch_enable && walker_ready && !bypass),
        .o_array_command_mode(command_mode),
        .o_array_command_fc_weight_int8(command_int8),
        .o_array_command_fc_current_shift6(command_shift6),
        .o_array_command_neuron_count(command_neurons),
        .o_array_command_first(command_first), .o_array_command_last(command_last),
        .o_array_command_destination_tile_base(command_tile_base),
        .i_array_transaction_done(transaction_done), .i_array_sticky_error(1'b0),
        .o_layer_done(layer_done), .o_idle(client_idle),
        .o_sticky_error(client_error), .o_error_code(client_error_code)
    );

    // Deliberately omit both opt-in inputs on an accepted legacy descriptor.
    unified_fc_srnn_h40_p4_layer_client_v1 u_legacy_client (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(abort), .i_clear_error(clear_error),
        .i_layer_valid(legacy_valid), .i_owner_request_ready(1'b1),
        .i_mode(2'd0), .i_neuron_count(16'd160),
        .i_feedforward_group_count(16'd1), .i_recurrent_group_count(16'd0),
        .i_recurrent_enable(1'b0), .i_leak(16'd0), .i_threshold(32'd100),
        .i_reset_to_zero(1'b1), .i_trace_decay(16'd0),
        .i_transaction_tag(52'd0), .i_work_base_addr(9'd0),
        .i_destination_tile_base(4'd0), .i_group_valid(1'b1),
        .i_group_recurrent(1'b0), .i_group_source_valid(4'd1),
        .i_group_weight_addr0(11'd0), .i_group_weight_addr1(11'd0),
        .i_group_weight_addr2(11'd0), .i_group_weight_addr3(11'd0),
        .o_array_command_valid(legacy_command_valid), .i_array_command_ready(1'b0),
        .o_array_command_fc_weight_int8(legacy_int8),
        .o_array_command_fc_current_shift6(legacy_shift6),
        .i_array_transaction_done(1'b0), .i_array_sticky_error(1'b0)
    );

    unified_binary_p4_destination_walker_v1 #(.ARRAY_TILES(12)) u_walker (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(abort), .i_clear_error(clear_error),
        .i_command_valid(bypass ? raw_valid : command_valid && dispatch_enable),
        .o_command_ready(walker_ready), .i_mode(bypass ? mode : command_mode),
        .i_fc_weight_int8(bypass ? int8_flag : command_int8),
        .i_fc_current_shift6(bypass ? shift6_flag : command_shift6),
        .i_neuron_count(bypass ? neurons : command_neurons),
        .i_first(command_first), .i_last(command_last),
        .i_source_valid(bypass ? 4'd1 : 4'b1111),
        .i_scnn_spatial_count(6'd1), .i_scnn_output_channel_base(10'd0),
        .i_scnn_feature_mask0('0), .i_scnn_feature_mask1('0),
        .i_scnn_feature_mask2('0), .i_scnn_feature_mask3('0),
        .i_scnn_factor_b(scnn_factor), .i_scnn_active_pair(1'b0),
        .i_scnn_weight_subword0(4'd0), .i_scnn_weight_subword1(4'd0),
        .i_scnn_weight_subword2(4'd0), .i_scnn_weight_subword3(4'd0),
        .i_scnn_work_base_addr(9'd0),
        .i_destination_tile_base(bypass ? tile_base : command_tile_base),
        .i_weight_addr0(11'd0), .i_weight_addr1(11'd1),
        .i_weight_addr2(11'd2), .i_weight_addr3(11'd3),
        .i_leak(16'd0), .i_threshold(32'd100), .i_reset_to_zero(1'b1),
        .i_trace_decay(16'd0), .i_transaction_tag(52'd1),
        .o_issue_valid(issue_valid), .i_issue_ready(issue_ready),
        .o_first(issue_first), .o_last(issue_last), .o_lane_mask_flat(issue_lanes),
        .o_mode_flat(issue_mode), .o_fc_weight_int8(issue_int8),
        .o_factor_b_flat(issue_factor),
        .o_done(walker_done), .o_sticky_error(walker_error)
    );

    task automatic tick;
        @(posedge clk);
        #1;
    endtask

    task automatic check_issue(input logic [15:0] mask,
                               input logic [1:0] expected_mode,
                               input logic expected_int8,
                               input integer count, input integer base,
                               input integer group_index,
                               input logic expected_shift6);
        if (issue_valid !== mask)
            $fatal(1, "issue mask got=%h expected=%h", issue_valid, mask);
        for (integer tile = 0; tile < 12; tile++) begin
            if (mask[tile]) begin
                if (issue_int8[tile] !== expected_int8 ||
                    issue_mode[tile*2 +: 2] !== expected_mode ||
                    issue_first[tile] !== (group_index == 0) ||
                    issue_last[tile] !== (group_index == 1))
                    $fatal(1, "metadata changed under stall tile=%0d", tile);
                for (integer lane = 0; lane < 40; lane++) begin
                    if (issue_lanes[tile*40+lane] !==
                        ((tile-base)*40+lane < count))
                        $fatal(1, "incorrect lane mask tile=%0d lane=%0d", tile, lane);
                    if ($signed(issue_factor[(tile*40+lane)*18 +: 18]) !==
                        (((tile-base)*40+lane < count) ?
                            (expected_shift6 ? 18'sd64 : 18'sd1) : 18'sd0))
                        $fatal(1, "incorrect current factor tile=%0d lane=%0d", tile, lane);
                end
            end
        end
    endtask

    task automatic run_layer(input logic [1:0] expected_mode,
                             input logic expected_int8,
                             input integer count, input integer base,
                             input logic expected_shift6 = 1'b0);
        logic [15:0] expected_mask;
        logic [15:0] pending_mask;
        expected_mask = 16'(((1 << ((count+39)/40))-1) << base);
        mode = expected_mode;
        int8_flag = expected_int8;
        shift6_flag = expected_shift6;
        neurons = 16'(count);
        tile_base = 4'(base);
        owner_ready = 1'b0;
        layer_valid = 1'b1;
        repeat (3) begin
            tick();
            if (layer_ready || !owner_valid || !client_idle)
                $fatal(1, "layer accepted without ownership");
        end
        owner_ready = 1'b1;
        tick();
        layer_valid = 1'b0;
        // Change the live descriptor after acceptance, including both opt-ins.
        mode = 2'd2;
        int8_flag = !expected_int8;
        shift6_flag = !expected_shift6;
        neurons = 16'd600;
        tile_base = 4'd15;
        for (integer group_index = 0; group_index < 2; group_index++) begin
            group_valid = 1'b1;
            repeat (3) begin
                tick();
                if (!command_valid || group_ready || command_int8 !== expected_int8 ||
                    command_shift6 !== expected_shift6 ||
                    command_mode !== expected_mode || command_neurons !== 16'(count))
                    $fatal(1, "layer precision or mode changed before command handshake");
            end
            dispatch_enable = 1'b1;
            tick();
            dispatch_enable = 1'b0;
            group_valid = 1'b0;
            // Scramble walker inputs immediately after its command handshake.
            bypass = 1'b1;
            while (issue_valid == 0) tick();
            repeat (5) begin
                check_issue(expected_mask, expected_mode, expected_int8, count, base,
                            group_index, expected_shift6);
                tick();
            end
            pending_mask = expected_mask;
            for (integer tile = 0; tile < 12; tile++) begin
                if (pending_mask[tile]) begin
                    issue_ready = 16'(1 << tile);
                    tick();
                    pending_mask[tile] = 1'b0;
                    check_issue(pending_mask, expected_mode, expected_int8, count, base,
                                group_index, expected_shift6);
                end
            end
            issue_ready = '0;
            while (!walker_done) tick();
            bypass = 1'b0;
            if (walker_error || client_error)
                $fatal(1, "legal layer raised error");
        end
        transaction_done = 1'b1;
        tick();
        transaction_done = 1'b0;
        if (!layer_done || !client_idle)
            $fatal(1, "layer did not retire");
        layer_count++;
        $display("METADATA_LAYER mode=%0d int8=%0d shift6=%0d neurons=%0d base=%0d PASS",
                 expected_mode, expected_int8, expected_shift6, count, base);
    endtask

    task automatic reject_layer(input logic [1:0] bad_mode,
                                input integer count, input integer base,
                                input logic bad_int8 = 1'b1,
                                input logic bad_shift6 = 1'b0);
        mode = bad_mode;
        neurons = 16'(count);
        tile_base = 4'(base);
        int8_flag = bad_int8;
        shift6_flag = bad_shift6;
        owner_ready = 1'b0;
        layer_valid = 1'b1;
        #1;
        if (!layer_ready || owner_valid)
            $fatal(1, "illegal metadata descriptor requested ownership");
        tick();
        layer_valid = 1'b0;
        if (!client_error || client_error_code != 8'h01 || !client_idle || command_valid)
            $fatal(1, "illegal metadata descriptor escaped layer client");
        clear_error = 1'b1;
        tick();
        clear_error = 1'b0;
        if (client_error) $fatal(1, "client error clear failed");
        rejected_count++;
        $display("METADATA_REJECT_CLIENT mode=%0d int8=%0d shift6=%0d PASS",
                 bad_mode, bad_int8, bad_shift6);
    endtask

    task automatic reject_raw(input logic [1:0] bad_mode, input integer count,
                              input logic bad_int8 = 1'b1,
                              input logic bad_shift6 = 1'b0);
        bypass = 1'b1;
        mode = bad_mode;
        neurons = 16'(count);
        tile_base = 4'd0;
        int8_flag = bad_int8;
        shift6_flag = bad_shift6;
        raw_valid = 1'b1;
        tick();
        raw_valid = 1'b0;
        while (!walker_done) begin
            if (issue_valid != 0) $fatal(1, "illegal raw metadata command issued");
            tick();
        end
        if (!walker_error || issue_valid != 0)
            $fatal(1, "walker did not reject unsupported metadata command");
        clear_error = 1'b1;
        tick();
        clear_error = 1'b0;
        bypass = 1'b0;
        if (walker_error) $fatal(1, "walker error clear failed");
        raw_rejected_count++;
        $display("METADATA_REJECT_WALKER mode=%0d int8=%0d shift6=%0d PASS",
                 bad_mode, bad_int8, bad_shift6);
    endtask

    task automatic run_scnn;
        bypass = 1'b1;
        mode = 2'd2;
        neurons = 16'd1;
        tile_base = 4'd0;
        int8_flag = 1'b0;
        shift6_flag = 1'b0;
        scnn_factor = -18'sd257;
        raw_valid = 1'b1;
        tick();
        raw_valid = 1'b0;
        mode = 2'd0;
        shift6_flag = 1'b1;
        scnn_factor = 18'sd64;
        while (issue_valid == 0) tick();
        repeat (4) begin
            if (issue_valid !== 16'd1 || issue_mode[1:0] !== 2'd2 ||
                $signed(issue_factor[17:0]) !== -18'sd257 ||
                issue_factor[719:18] !== '0)
                $fatal(1, "SCNN factor was changed by FC scaling");
            tick();
        end
        issue_ready = '1;
        tick();
        issue_ready = '0;
        while (!walker_done) tick();
        if (walker_error) $fatal(1, "legal SCNN command raised error");
        bypass = 1'b0;
        $display("METADATA_SCNN_FACTOR_UNCHANGED_PASS");
    endtask

    task automatic abort_shift6(input integer phase);
        mode = 2'd0;
        neurons = 16'd80;
        tile_base = 4'd0;
        int8_flag = 1'b0;
        shift6_flag = 1'b1;
        owner_ready = 1'b1;
        layer_valid = 1'b1;
        tick();
        layer_valid = 1'b0;
        shift6_flag = 1'b0;
        group_valid = 1'b1;
        if (phase != 0) begin
            dispatch_enable = 1'b1;
            tick();
            dispatch_enable = 1'b0;
            group_valid = 1'b0;
            if (phase == 2) begin
                while (issue_valid == 0) tick();
                check_issue(16'd3, 2'd0, 1'b0, 80, 0, 0, 1'b1);
                issue_ready = 16'd1;
                tick();
                issue_ready = '0;
                check_issue(16'd2, 2'd0, 1'b0, 80, 0, 0, 1'b1);
            end
        end
        abort = 1'b1;
        tick();
        abort = 1'b0;
        group_valid = 1'b0;
        repeat (3) begin
            if (!client_idle || command_valid || issue_valid != 0 ||
                !walker_ready || layer_done || walker_done || client_error || walker_error)
                $fatal(1, "abort left live metadata phase=%0d", phase);
            tick();
        end
        abort_count++;
        $display("METADATA_SHIFT6_ABORT phase=%0d PASS", phase);
        run_layer(2'd0, 1'b0, 80, 0);
    endtask

    initial begin
        repeat (3) tick();
        rst_n = 1'b1;
        legacy_valid = 1'b1;
        tick();
        legacy_valid = 1'b0;
        if (!legacy_command_valid || legacy_int8 !== 1'b0 || legacy_shift6 !== 1'b0)
            $fatal(1, "unconnected legacy inputs did not select INT5/raw");
        $display("METADATA_LEGACY_DEFAULT_RAW_PASS");
        run_layer(2'd0, 1'b0, 160, 0);
        run_layer(2'd0, 1'b1, 10, 0);
        run_layer(2'd0, 1'b1, 25, 11);
        run_layer(2'd1, 1'b0, 41, 3);
        run_layer(2'd3, 1'b0, 20, 4);
        run_layer(2'd0, 1'b0, 80, 2);
        // Scaling and precision are independent; no reset separates descriptors.
        run_layer(2'd0, 1'b0, 160, 0, 1'b1);
        run_layer(2'd0, 1'b0, 160, 0);
        run_layer(2'd0, 1'b1, 25, 11, 1'b1);
        run_layer(2'd0, 1'b1, 25, 11);
        run_layer(2'd0, 1'b0, 41, 3, 1'b1);
        // W0 uses tiles 0..3; raw INT8 W2 targets the reserved tile 4.
        // run_layer scrambles live base after acceptance and stalls both stages.
        run_layer(2'd0, 1'b0, 160, 0, 1'b1);
        run_layer(2'd0, 1'b1, 10, 4);
        run_layer(2'd0, 1'b0, 160, 0, 1'b1);
        $display("FC_DESTINATION_BASE4_METADATA_PASS");
        run_layer(2'd1, 1'b0, 41, 3);
        run_layer(2'd3, 1'b0, 20, 4);
        run_scnn();
        reject_layer(2'd0, 26, 0);
        reject_layer(2'd0, 0, 0);
        reject_layer(2'd0, 25, 12);
        reject_layer(2'd1, 10, 0);
        reject_layer(2'd2, 10, 0);
        reject_layer(2'd3, 10, 0);
        reject_raw(2'd0, 26);
        reject_raw(2'd1, 10);
        reject_raw(2'd2, 10);
        reject_raw(2'd3, 10);
        for (integer bad_mode = 1; bad_mode < 4; bad_mode++) begin
            reject_layer(2'(bad_mode), 10, 0, 1'b0, 1'b1);
            reject_raw(2'(bad_mode), 10, 1'b0, 1'b1);
        end
        for (integer phase = 0; phase < 3; phase++) abort_shift6(phase);
        // Abort a staged INT8 command before any tile accepts it.
        bypass = 1'b1;
        mode = 2'd0;
        neurons = 16'd10;
        int8_flag = 1'b1;
        shift6_flag = 1'b0;
        raw_valid = 1'b1;
        tick();
        raw_valid = 1'b0;
        abort = 1'b1;
        tick();
        abort = 1'b0;
        bypass = 1'b0;
        if (issue_valid != 0 || !walker_ready) $fatal(1, "abort left pending metadata");
        run_layer(2'd0, 1'b1, 10, 0);
        run_layer(2'd0, 1'b0, 160, 0);
        $display("FC_CURRENT_SHIFT6_METADATA_PASS layers=%0d client_rejects=%0d walker_rejects=%0d aborts=%0d",
                 layer_count, rejected_count, raw_rejected_count, abort_count);
        $display("FC_INT8_PRECISION_METADATA_PASS layers=%0d rejects=%0d", layer_count, rejected_count);
        $finish;
    end
endmodule

// Exercise the real transaction guard; numeric H160 comparison belongs to
// the separate full-model bench. The stationary tiles require native geometry.
module tb_fc_current_shift6_array12_metadata;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic abort = 1'b0;
    logic command_valid = 1'b0;
    logic command_ready;
    logic shift6 = 1'b0;
    logic first = 1'b1;
    logic last = 1'b0;
    logic done, idle, error;
    logic [3:0] observation_valid;
    logic maintenance_valid = 1'b0;
    logic [9:0] maintenance_addr = '0;
    logic state_ready, trace_ready;
    integer issue_count = 0;
    integer transaction_count = 0;

    always #5 clk = ~clk;
    initial begin
        #100000;
        $fatal(1, "array12 metadata watchdog expired");
    end

    unified_binary_p4_h40_array12_path_v1 u_path (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(abort), .i_clear_error(1'b0),
        .i_command_valid(command_valid), .o_command_ready(command_ready),
        .i_mode(2'd0), .i_fc_weight_int8(1'b0), .i_fc_current_shift6(shift6),
        .i_neuron_count(16'd1), .i_first(first), .i_last(last),
        .i_source_valid(4'd0), .i_scnn_spatial_count(6'd0),
        .i_scnn_output_channel_base(10'd0),
        .i_scnn_feature_mask0('0), .i_scnn_feature_mask1('0),
        .i_scnn_feature_mask2('0), .i_scnn_feature_mask3('0),
        .i_scnn_factor_b(18'sd256), .i_scnn_active_pair(1'b0),
        .i_scnn_weight_subword0(4'd0), .i_scnn_weight_subword1(4'd0),
        .i_scnn_weight_subword2(4'd0), .i_scnn_weight_subword3(4'd0),
        .i_scnn_work_base_addr(9'd0), .i_destination_tile_base(4'd0),
        .i_weight_addr0(11'd0), .i_weight_addr1(11'd0),
        .i_weight_addr2(11'd0), .i_weight_addr3(11'd0),
        .i_leak(16'd0), .i_threshold(32'd100), .i_reset_to_zero(1'b1),
        .i_trace_decay(16'd0), .i_transaction_tag(52'd1),
        .i_weight_fill_valid('0), .i_weight_fill_tile_flat('0),
        .i_weight_fill_bank('0), .i_weight_fill_addr_flat('0),
        .i_weight_fill_data_flat('0),
        .o_observation_valid(observation_valid), .i_observation_ready(4'hf),
        .i_maintenance_tile(4'd0),
        .i_state_maintenance_write_valid(maintenance_valid),
        .o_state_maintenance_write_ready(state_ready),
        .i_state_maintenance_write_addr(maintenance_addr),
        .i_state_maintenance_write_data('0), .i_state_maintenance_write_mask('1),
        .i_state_maintenance_read_valid(1'b0), .i_state_maintenance_read_addr('0),
        .i_state_maintenance_result_ready(1'b1),
        .i_trace_maintenance_write_valid(maintenance_valid),
        .o_trace_maintenance_write_ready(trace_ready),
        .i_trace_maintenance_write_addr({1'b1, maintenance_addr[8:0]}),
        .i_trace_maintenance_write_data('0), .i_trace_maintenance_write_mask('1),
        .i_trace_maintenance_read_valid(1'b0), .i_trace_maintenance_read_addr('0),
        .i_trace_maintenance_result_ready(1'b1),
        .o_done(done), .o_idle(idle), .o_sticky_error(error)
    );

    always @(posedge clk) begin
        if (rst_n && !abort && |(u_path.issue_valid & u_path.issue_ready)) begin
            issue_count++;
            if (u_path.issue_valid !== 16'd1 ||
                u_path.factor_b_flat[17:0] !==
                    (u_path.transaction_fc_current_shift6_q ? 18'd64 : 18'd1))
                $fatal(1, "array12 issued incorrect scale metadata");
        end
    end

    task automatic tick;
        @(posedge clk);
        #1;
    endtask

    task automatic send(input logic scale, input logic is_first, input logic is_last);
        while (!command_ready) tick();
        shift6 = scale;
        first = is_first;
        last = is_last;
        command_valid = 1'b1;
        tick();
        command_valid = 1'b0;
        shift6 = !scale;
    endtask

    task automatic finish_group;
        while (!done) tick();
        if (error) $fatal(1, "legal array12 group raised error");
    endtask

    task automatic run_transaction(input logic scale);
        integer before_count;
        before_count = issue_count;
        send(scale, 1'b1, 1'b0);
        finish_group();
        if (idle || u_path.transaction_fc_current_shift6_q !== scale)
            $fatal(1, "array12 lost accepted transaction scale");
        send(scale, 1'b0, 1'b1);
        finish_group();
        if (!idle || issue_count != before_count + 2)
            $fatal(1, "array12 transaction failed to retire");
        transaction_count++;
        $display("ARRAY12_SCALE_TRANSACTION shift6=%0d PASS", scale);
    endtask

    task automatic reject_change(input logic initial_scale);
        integer before_count;
        send(initial_scale, 1'b1, 1'b0);
        finish_group();
        before_count = issue_count;
        send(!initial_scale, 1'b0, 1'b1);
        if (!error || !done || u_path.walker_command_valid)
            $fatal(1, "array12 accepted a mid-transaction scale change");
        repeat (4) tick();
        if (issue_count != before_count || command_ready)
            $fatal(1, "array12 dispatched inconsistent scale metadata");
        abort = 1'b1;
        tick();
        abort = 1'b0;
        tick();
        if (!idle || error || u_path.transaction_fc_current_shift6_q !== 1'b0)
            $fatal(1, "array12 abort failed to clear transaction scale");
        $display("ARRAY12_SCALE_MISMATCH_REJECT initial_shift6=%0d PASS", initial_scale);
        run_transaction(1'b0);
    endtask

    initial begin
        repeat (3) tick();
        rst_n = 1'b1;
        // Initialize only the exercised FC state/trace rows through maintenance.
        for (integer row = 0; row < 4; row++) begin
            maintenance_addr = 10'(row);
            maintenance_valid = 1'b1;
            #1;
            if (!state_ready || !trace_ready) $fatal(1, "maintenance not ready");
            tick();
            maintenance_valid = 1'b0;
            tick();
        end
        maintenance_valid = 1'b0;
        tick();
        run_transaction(1'b1);
        run_transaction(1'b0);
        reject_change(1'b1);
        reject_change(1'b0);
        run_transaction(1'b1);
        run_transaction(1'b0);
        $display("FC_CURRENT_SHIFT6_ARRAY12_METADATA_PASS transactions=%0d issues=%0d",
                 transaction_count, issue_count);
        $finish;
    end
endmodule

`default_nettype wire
