`timescale 1ns/1ps
`default_nettype none

module tb_unified_scnn_selected_layers_context_frame_service_v1;
    logic i_clk = 1'b0;
    always #5 i_clk = ~i_clk;
    logic i_rst_n = 1'b0, i_abort = 1'b0, i_clear_error = 1'b0;
    logic i_context_load_valid = 1'b0, o_context_load_ready;
    logic i_context_load_weights = 1'b0;
    logic [9:0] i_context_output_channel_base = '0;
    logic [9:0] i_context_output_channel_count = '0;
    logic [31:0] i_context_spatial_base = '0;
    logic [5:0] i_context_spatial_count = '0;
    logic i_context_active_pair = 1'b0;
    logic [8:0] i_context_work_base_addr = '0;
    logic i_context_release_valid = 1'b0, o_context_release_ready;
    logic [9:0] i_release_output_channel_base = '0;
    logic [9:0] i_release_output_channel_count = '0;
    logic [31:0] i_release_spatial_base = '0;
    logic [5:0] i_release_spatial_count = '0;
    logic i_release_active_pair = 1'b0;
    logic [8:0] i_release_work_base_addr = '0;
    logic [15:0] i_graph_version = 16'd7;
    logic i_classifier_cache_valid = 1'b1, i_classifier_pair = 1'b1;
    logic [15:0] i_classifier_version = 16'd7;
    logic [3:0] i_classifier_class_count = 4'd3;
    logic [8:0] i_classifier_work_base_addr = 9'd96;
    logic [9:0] i_conv_input_channel_count = 10'd512;
    logic [23:0] i_conv_parameter_base = 24'd120;
    logic o_conv_context_valid, i_conv_context_ready = 1'b0;
    logic o_conv_context_load_weights;
    logic [9:0] o_conv_output_channel_base;
    logic [4:0] o_conv_output_channel_count;
    logic [9:0] o_conv_input_channel_count;
    logic o_conv_context_pair;
    logic [23:0] o_conv_parameter_base;
    logic [15:0] o_conv_context_version;
    logic o_state_service_valid, i_state_service_ready = 1'b0;
    logic i_state_service_drained = 1'b0;
    logic o_state_service_release, o_state_service_is_classifier;
    logic [9:0] o_state_service_output_channel_base;
    logic [9:0] o_state_service_output_channel_count;
    logic [31:0] o_state_service_spatial_base;
    logic [5:0] o_state_service_spatial_count;
    logic o_state_service_active_pair;
    logic [8:0] o_state_service_work_base_addr;
    logic o_context_active, o_context_is_classifier, o_sticky_error;
    integer cache_fires = 0, service_fires = 0;
    integer load_fires = 0, release_fires = 0, cases = 0;

    unified_scnn_selected_layers_context_bridge_v1 #(
        .PARAM_ADDR_WIDTH(24), .WORK_ADDR_WIDTH(9),
        .MAX_INPUT_CHANNELS(512), .ENABLE_FRAME_SERVICE(1'b1)
    ) dut (.*);

    always @(posedge i_clk) begin
        if (i_rst_n && !i_abort) begin
            if (o_conv_context_valid && i_conv_context_ready) cache_fires++;
            if (o_state_service_valid && i_state_service_ready) service_fires++;
            if (i_context_load_valid && o_context_load_ready) load_fires++;
            if (i_context_release_valid && o_context_release_ready) release_fires++;
        end
    end

    task automatic tick;
        @(posedge i_clk);
        #1;
        @(negedge i_clk);
        #1;
    endtask

    task automatic check(input bit condition, input string message);
        if (!condition) $fatal(1, "%s", message);
    endtask

    task automatic mark(input string name);
        cases++;
        $display("SCNN_CONTEXT_FRAME_CASE_PASS %s", name);
    endtask

    task automatic no_core_handshake;
        check(!o_context_load_ready && !o_context_release_ready,
              "core barrier released before service completion/drain");
    endtask

    task automatic check_metadata(input bit classifier, input bit release_op);
        check(o_state_service_is_classifier == classifier &&
              o_state_service_release == release_op &&
              o_state_service_output_channel_base == i_context_output_channel_base &&
              o_state_service_output_channel_count == i_context_output_channel_count &&
              o_state_service_spatial_base == i_context_spatial_base &&
              o_state_service_spatial_count == i_context_spatial_count &&
              o_state_service_active_pair == i_context_active_pair &&
              o_state_service_work_base_addr == i_context_work_base_addr,
              "state-service metadata changed");
    endtask

    task automatic begin_load(input bit classifier, input bit weights);
        i_context_load_weights = weights;
        i_context_output_channel_base = classifier ? 10'd0 : 10'd16;
        i_context_output_channel_count = classifier ? 10'd3 : 10'd12;
        i_context_spatial_base = classifier ? 32'd0 : 32'd80;
        i_context_spatial_count = classifier ? 6'd1 : 6'd40;
        i_context_active_pair = classifier;
        i_context_work_base_addr = classifier ? 9'd96 : 9'd24;
        i_context_load_valid = 1'b1;
        i_context_release_valid = 1'b0;
        i_conv_context_ready = 1'b0;
        i_state_service_ready = 1'b0;
        i_state_service_drained = 1'b0;
        tick();
        no_core_handshake();
        check_metadata(classifier, 1'b0);
        check(!o_sticky_error, "legal load raised error");
        check(o_conv_context_valid == !classifier,
              "classifier requested cache or Conv skipped cache");
        check(o_state_service_valid == classifier,
              "classifier skipped state restore or Conv restored before cache");
    endtask

    task automatic finish_cache;
        integer before_count;
        before_count = cache_fires;
        repeat (3) begin
            check(o_conv_context_valid && !o_state_service_valid,
                  "cache request not held until completion");
            check(o_conv_context_load_weights == i_context_load_weights &&
                  o_conv_output_channel_base == 10'd16 &&
                  o_conv_output_channel_count == 5'd12 &&
                  o_conv_input_channel_count == 10'd512 &&
                  o_conv_context_pair == 1'b0 &&
                  o_conv_parameter_base == 24'd120 && o_conv_context_version == 16'd7,
                  "cache request metadata mismatch");
            no_core_handshake();
            tick();
        end
        i_conv_context_ready = 1'b1;
        tick();
        check(cache_fires == before_count + 1 && !o_conv_context_valid &&
              o_state_service_valid, "cache completion was not retired once");
        repeat (3) begin
            tick();
            check(cache_fires == before_count + 1 && !o_conv_context_valid,
                  "cache request was accepted again while restore stalled");
        end
    endtask

    task automatic complete_transfer(input bit classifier, input bit release_op);
        integer before_count;
        before_count = service_fires;
        repeat (4) begin
            check(o_state_service_valid, "service request dropped before completion");
            check_metadata(classifier, release_op);
            no_core_handshake();
            tick();
        end
        i_state_service_ready = 1'b1;
        tick();
        check(service_fires == before_count + 1 && !o_state_service_valid,
              "service completion was not accepted once");
        // Completion remains high, but reads are deliberately still pending.
        repeat (4) begin
            no_core_handshake();
            check_metadata(classifier, release_op);
            tick();
            check(!o_state_service_valid && service_fires == before_count + 1 &&
                  !o_conv_context_valid, "completed transfer was restarted");
        end
        i_state_service_ready = 1'b0;
    endtask

    task automatic drain_load(input bit classifier);
        integer before_count;
        before_count = load_fires;
        i_state_service_drained = 1'b1;
        #1;
        check(o_context_load_ready && !o_context_release_ready,
              "drained restore did not release load barrier");
        tick();
        i_context_load_valid = 1'b0;
        i_state_service_drained = 1'b0;
        i_conv_context_ready = 1'b0;
        tick();
        check(load_fires == before_count + 1 && o_context_active &&
              o_context_is_classifier == classifier && !o_sticky_error,
              "load ownership/handshake mismatch");
    endtask

    task automatic begin_release;
        i_release_output_channel_base = i_context_output_channel_base;
        i_release_output_channel_count = i_context_output_channel_count;
        i_release_spatial_base = i_context_spatial_base;
        i_release_spatial_count = i_context_spatial_count;
        i_release_active_pair = i_context_active_pair;
        i_release_work_base_addr = i_context_work_base_addr;
        i_context_release_valid = 1'b1;
        tick();
        check(o_state_service_valid && o_context_active,
              "release did not retain ownership for spill");
        no_core_handshake();
    endtask

    task automatic drain_release;
        integer before_count;
        before_count = release_fires;
        i_state_service_drained = 1'b1;
        #1;
        check(o_context_release_ready && !o_context_load_ready,
              "drained spill did not release core barrier");
        tick();
        i_context_release_valid = 1'b0;
        i_state_service_drained = 1'b0;
        tick();
        check(release_fires == before_count + 1 && !o_context_active &&
              !o_context_is_classifier && !o_sticky_error,
              "release ownership/handshake mismatch");
    endtask

    task automatic full_context(input bit classifier, input bit weights);
        integer old_cache;
        old_cache = cache_fires;
        begin_load(classifier, weights);
        if (!classifier) finish_cache();
        complete_transfer(classifier, 1'b0);
        drain_load(classifier);
        begin_release();
        complete_transfer(classifier, 1'b1);
        drain_release();
        check(cache_fires == old_cache + (classifier ? 0 : 1),
              "unexpected cache request count for context lifecycle");
    endtask

    task automatic cancel_pending(input bit use_abort);
        if (use_abort) i_abort = 1'b1;
        else i_rst_n = 1'b0;
        #1;
        no_core_handshake();
        check(!o_state_service_valid && !o_conv_context_valid,
              "cancel did not suppress outward request");
        tick();
        i_context_load_valid = 1'b0;
        i_context_release_valid = 1'b0;
        i_conv_context_ready = 1'b0;
        i_state_service_ready = 1'b0;
        i_state_service_drained = 1'b0;
        tick();
        i_abort = 1'b0;
        i_rst_n = 1'b1;
        i_clear_error = 1'b1;
        tick();
        i_clear_error = 1'b0;
        check(!o_context_active && !o_state_service_valid && !o_conv_context_valid &&
              !o_sticky_error && o_state_service_output_channel_count == 0,
              "cancel left pending ownership/metadata/error");
        i_state_service_drained = 1'b1;
        tick();
        no_core_handshake();
        i_state_service_drained = 1'b0;
    endtask

    task automatic wait_stage(input integer stage);
        begin_load(1'b0, 1'b1);
        if (stage >= 1) finish_cache();
        if (stage >= 2) complete_transfer(1'b0, 1'b0);
        if (stage >= 3) begin
            drain_load(1'b0);
            begin_release();
        end
        if (stage >= 4) complete_transfer(1'b0, 1'b1);
    endtask

    initial begin
        repeat (2) tick();
        i_rst_n = 1'b1;
        tick();
        full_context(1'b0, 1'b1);
        mark("conv_refill_slow_restore_spill");
        full_context(1'b0, 1'b0);
        mark("conv_resident_state_only");
        repeat (2) full_context(1'b1, 1'b0);
        mark("classifier_two_timesteps_restore_spill");
        for (int kind = 0; kind < 2; kind++) begin
            for (int stage = 0; stage < 5; stage++) begin
                wait_stage(stage);
                cancel_pending(kind != 0);
                full_context(1'b0, 1'b1);
                mark($sformatf("%s_wait_stage_%0d_fresh_context",
                              kind != 0 ? "abort" : "reset", stage));
            end
        end
        begin_load(1'b0, 1'b1);
        i_context_spatial_base = 32'd120;
        #1;
        check(o_state_service_spatial_base == 32'd80 && !o_conv_context_valid,
              "pending metadata was not held on invalid producer change");
        tick();
        check(o_sticky_error, "changed held request metadata was not rejected");
        i_clear_error = 1'b1;
        tick();
        check(o_sticky_error, "clear_error discarded an active pending context");
        i_clear_error = 1'b0;
        cancel_pending(1'b1);
        full_context(1'b1, 1'b0);
        mark("metadata_change_abort_recovery");
        check(cases == 14, "missing directed cases");
        $display("SCNN_CONTEXT_FRAME_SERVICE_PASS cases=%0d cache=%0d service=%0d load=%0d release=%0d",
                 cases, cache_fires, service_fires, load_fires, release_fires);
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "context frame-service test timed out");
    end
endmodule

`default_nettype wire
