    localparam int LEARNING_BASE = 32;
    localparam int CLASSIFIER_BASE = 80;
    int conv_reads, conv_writes, classifier_writes, commits, refills, learning_configs;
    int expected_refill_base, refill_reads, forward_reads;
    bit refill_active, reject_traffic;
    bit [35:0] refill_seen, conv_read_seen, conv_write_seen;
    bit [23:0] classifier_write_seen;

    always #5 i_clk = ~i_clk;
    assign i_prescan_feature_request_ready = !i_prescan_feature_response_valid ||
        o_prescan_feature_response_ready;
    assign i_prescan_feature_mask0 = '1;
    assign i_prescan_feature_mask1 = '0;
    assign i_prescan_feature_mask2 = '0;
    assign i_prescan_feature_mask3 = '0;

    always @(posedge i_clk) begin
        if (!i_rst_n) begin
            i_prescan_feature_response_valid <= 0;
            conv_reads = 0;
            conv_writes = 0;
            classifier_writes = 0;
            learning_configs = 0;
            commits = 0;
            forward_reads = 0;
            conv_read_seen = '0;
            conv_write_seen = '0;
            classifier_write_seen = '0;
        end else begin
            if (dut.u_runtime.u_runtime.u_selected_layers.i_cfg_valid &&
                dut.u_runtime.u_runtime.u_selected_layers.o_cfg_ready) begin
                if (dut.u_runtime.u_runtime.u_selected_layers.i_cfg_parameter_base != LEARNING_BASE)
                    $fatal(1, "learning configuration used refill base");
                learning_configs++;
            end
            if (learning_configs != 0 && dut.u_runtime.u_runtime.active_q &&
                dut.u_runtime.u_runtime.conv_parameter_base_q != LEARNING_BASE)
                $fatal(1, "captured learning base changed during window");
            if (i_prescan_feature_response_valid && o_prescan_feature_response_ready)
                i_prescan_feature_response_valid <= 0;
            if (o_prescan_feature_request_valid && i_prescan_feature_request_ready)
                i_prescan_feature_response_valid <= 1;
            if (reject_traffic && (o_store_learning_read_valid ||
                    o_store_learning_write_valid || o_store_forward_read_valid ||
                    (|o_weight_fill_valid) || o_commit_pulse))
                $fatal(1, "invalid metadata generated parameter traffic");
            if (o_store_forward_read_valid && i_store_forward_read_ready) begin
                forward_reads++;
                if (refill_active) begin
                    if (int'(o_store_forward_read_addr) < expected_refill_base ||
                            int'(o_store_forward_read_addr) >= expected_refill_base + 36)
                        $fatal(1, "refill address got=%0d expected=[%0d,%0d)",
                            o_store_forward_read_addr, expected_refill_base, expected_refill_base+36);
                    if (refill_seen[int'(o_store_forward_read_addr)-expected_refill_base])
                        $fatal(1, "duplicate refill address");
                    refill_seen[int'(o_store_forward_read_addr)-expected_refill_base] = 1;
                    refill_reads++;
                end else if (int'(o_store_forward_read_addr) < CLASSIFIER_BASE ||
                        int'(o_store_forward_read_addr) >= CLASSIFIER_BASE+24)
                    $fatal(1, "unexpected classifier publication address");
            end
            if (o_store_learning_read_valid && i_store_learning_read_ready) begin
                if (int'(o_store_learning_read_addr) >= LEARNING_BASE &&
                        int'(o_store_learning_read_addr) < LEARNING_BASE+36) begin
                    if (conv_read_seen[int'(o_store_learning_read_addr)-LEARNING_BASE])
                        $fatal(1, "duplicate conv RMW read");
                    conv_read_seen[int'(o_store_learning_read_addr)-LEARNING_BASE] = 1;
                    conv_reads++;
                end else if (int'(o_store_learning_read_addr) < CLASSIFIER_BASE ||
                        int'(o_store_learning_read_addr) >= CLASSIFIER_BASE+24)
                    $fatal(1, "learning base contaminated: read=%0d expected Conv32..67 or CL80..103",
                        o_store_learning_read_addr);
            end
            if (o_store_learning_write_valid && i_store_learning_write_ready) begin
                if (int'(o_store_learning_write_addr) >= LEARNING_BASE &&
                        int'(o_store_learning_write_addr) < LEARNING_BASE+36) begin
                    if (conv_write_seen[int'(o_store_learning_write_addr)-LEARNING_BASE])
                        $fatal(1, "duplicate conv RMW write");
                    conv_write_seen[int'(o_store_learning_write_addr)-LEARNING_BASE] = 1;
                    conv_writes++;
                end else if (int'(o_store_learning_write_addr) >= CLASSIFIER_BASE &&
                        int'(o_store_learning_write_addr) < CLASSIFIER_BASE+24) begin
                    if (classifier_write_seen[int'(o_store_learning_write_addr)-CLASSIFIER_BASE])
                        $fatal(1, "duplicate classifier RMW write");
                    classifier_write_seen[int'(o_store_learning_write_addr)-CLASSIFIER_BASE] = 1;
                    classifier_writes++;
                end else
                    $fatal(1, "learning base contaminated: write=%0d", o_store_learning_write_addr);
            end
            if (o_commit_pulse) commits++;
        end
    end

    task automatic tick;
        @(posedge i_clk);
        #1;
    endtask

    task automatic initialize;
        @(negedge i_clk);
        i_rst_n = 0;
        i_request_valid = 0;
        i_conv_context_valid = 0;
        i_graph_done = 0;
        i_graph_start_ready = 0;
        i_feature_wave_valid = 0;
        i_logit_wave_valid = 0;
        i_surrogate_valid = 0;
        reject_traffic = 0;
        refill_active = 0;
        i_conv_parameter_base = 8'(LEARNING_BASE);
        i_conv_context_parameter_base = 128;
        i_batch_count = 2;
        i_conv_context_version = 5;
        repeat (4) tick();
        @(negedge i_clk); i_rst_n = 1;
        tick();
        for (int addr = 0; addr < 256; addr++) begin
            @(negedge i_clk);
            i_parameter_load_valid = 1;
            i_parameter_load_addr = 8'(addr);
            i_parameter_load_code = 0;
            while (!o_parameter_load_ready) @(negedge i_clk);
            tick();
        end
        @(negedge i_clk); i_parameter_load_valid = 0;
        i_initialize_valid = 1;
        while (!o_initialize_ready) @(negedge i_clk);
        tick();
        @(negedge i_clk); i_initialize_valid = 0;
        while (!o_initialize_done) tick();
        tick();
        if (o_sticky_error || !o_initialized || o_parameter_version != 5)
            $fatal(1, "initialization failed");
    endtask

    task automatic request_batch;
        @(negedge i_clk);
        i_request_valid = 1;
        while (!o_request_ready) @(negedge i_clk);
        tick();
        @(negedge i_clk);
        i_request_valid = 0;
        // Configuration is consumed at the existing runtime run handshake.
        // Change the live base only after that capture, before both refills.
        while (!dut.u_runtime.u_runtime.active_q) tick();
        @(negedge i_clk);
        if (USE_CONTEXT_PARAMETER_BASE) i_conv_parameter_base = 112;
    endtask

    task automatic refill(input int base_addr);
        @(negedge i_clk);
        expected_refill_base = USE_CONTEXT_PARAMETER_BASE ? base_addr : LEARNING_BASE;
        refill_reads = 0;
        refill_seen = '0;
        refill_active = 1;
        i_conv_context_parameter_base = 8'(base_addr);
        i_conv_context_valid = 1;
        while (!o_conv_context_ready) @(negedge i_clk);
        tick();
        @(negedge i_clk); i_conv_context_valid = 0;
        refill_active = 0;
        if (refill_reads != 36 || !(&refill_seen))
            $fatal(1, "incomplete refill count=%0d seen=%h", refill_reads, refill_seen);
        refills++;
        $display("CASE_PASS refill requested=%0d effective=%0d reads=%0d",
            base_addr, expected_refill_base, refill_reads);
    endtask

    task automatic run_graph;
        @(negedge i_clk);
        while (!o_graph_start_valid) @(negedge i_clk);
        i_graph_start_ready = 1;
        tick();
        @(negedge i_clk); i_graph_start_ready = 0;
        for (int spatial = 0; spatial < 16; spatial++)
            for (int channel = 0; channel < 2; channel++) begin
                @(negedge i_clk);
                i_surrogate_target = o_graph_target;
                i_surrogate_sample = o_graph_sample;
                i_surrogate_timestep = o_graph_timestep;
                i_surrogate_version = o_graph_version;
                i_surrogate_layer_tag = 7;
                i_surrogate_channel = 10'(channel);
                i_surrogate_spatial = 32'(spatial);
                i_surrogate_q8_8 = 16'sh100;
                i_surrogate_valid = 1;
                while (!o_surrogate_source_ready) @(negedge i_clk);
                tick();
                @(negedge i_clk); i_surrogate_valid = 0;
            end
        i_feature_wave_mask = '1;
        i_feature_wave_base = 0;
        for (int f = 0; f < 8; f++)
            i_feature_wave_q8_8_flat[f*16 +: 16] = 16'(256+f+int'(o_graph_sample));
        i_feature_wave_version = o_graph_version;
        i_feature_wave_frame_index = 8'(o_graph_timestep);
        i_feature_wave_first = 1;
        i_feature_wave_last = 1;
        i_feature_wave_sample_reset = 1;
        i_feature_wave_sample_last = 1;
        i_feature_sample_label = o_graph_label;
        i_feature_wave_valid = 1;
        while (!o_feature_wave_ready) @(negedge i_clk);
        tick();
        @(negedge i_clk); i_feature_wave_valid = 0;
        i_logit_wave_mask = 12'h007;
        i_logit_wave_base = 0;
        i_logit_wave_count = 3;
        i_logit_wave_q8_8_flat = 0;
        i_logit_wave_version = o_graph_version;
        i_logit_wave_frame_index = 8'(o_graph_timestep);
        i_logit_wave_frame_last = 1;
        i_logit_wave_sample_last = 1;
        i_logit_wave_valid = 1;
        while (!o_logit_wave_ready) @(negedge i_clk);
        tick();
        @(negedge i_clk); i_logit_wave_valid = 0;
        i_graph_done = 1;
        tick();
        @(negedge i_clk); i_graph_done = 0;
    endtask

    initial begin
        i_system_parameter_owner = 1;
        i_system_learning_enable = 1;
        i_system_profile_aligned = 1;
        i_system_profile_id = 8'h20;
        i_profile_id = 8'h20;
        i_conv_tensor_id = 8'h08;
        i_classifier_tensor_id = 8'h12;
        i_learning_enable = 1;
        i_timestep_count = 1;
        i_sample_version = 16'h6100;
        i_labels_flat = {2'd1, 2'd0};
        i_total_input_channels = 2;
        i_total_output_channels = 2;
        i_classifier_parameter_base = 8'(CLASSIFIER_BASE);
        i_classifier_feature_count = 8;
        i_classifier_class_count = 3;
        i_classifier_trace_leak_valid = 1;
        i_conv_vote_increment = 1;
        i_conv_vote_threshold = 1;
        i_classifier_vote_increment = 1;
        i_classifier_vote_threshold = 1;
        i_code_min = -127;
        i_code_max = 127;
        i_classifier_pair = 1;
        i_context_model_version = 5;
        i_context_adapt_version = 7;
        i_conv_in_width = 4;
        i_conv_in_height = 4;
        i_conv_in_channels = 2;
        i_conv_padding = 1;
        i_conv_layer_tag = 7;
        i_target_feature_count = 8;
        i_target_trace_state_reset = 1;
        i_array_idle = 1;
        i_weight_fill_ready = '1;
        i_conv_context_load_weights = 1;
        i_conv_output_channel_count = 2;
        i_conv_input_channel_count = 2;
        refills = 0;

        initialize();
        request_batch();
        refill(128);
        run_graph();
        run_graph();
        refill(176);
        run_graph();
        run_graph();
        while (!o_done) tick();
        if (o_sticky_error || o_recovery_required || conv_reads != 0 || conv_writes != 0 ||
                classifier_writes != 24 || commits != 1 || learning_configs != 1 || !(&classifier_write_seen))
            $fatal(1, "RMW coverage/error reads=%0d writes=%0d classifier=%0d commits=%0d error=%b",
                conv_reads, conv_writes, classifier_writes, commits, o_sticky_error);
        $display("CASE_PASS captured_learning_base=32 zero_conv_gradient=1 conv_reads=%0d conv_writes=%0d classifier_writes=%0d commits=%0d",
            conv_reads, conv_writes, classifier_writes, commits);

        initialize();
        i_batch_count = 0;
        reject_traffic = 1;
        @(negedge i_clk); i_request_valid = 1;
        while (!o_request_ready) @(negedge i_clk);
        tick();
        @(negedge i_clk); i_request_valid = 0;
        repeat (100) tick();
        if (!o_sticky_error || conv_reads || conv_writes || classifier_writes)
            $fatal(1, "invalid batch was not rejected without RMW");
        $display("CASE_PASS invalid_batch no_parameter_traffic");

        initialize();
        @(negedge i_clk);
        reject_traffic = 1;
        i_conv_context_version = 6;
        i_conv_context_valid = 1;
        repeat (100) begin
            tick();
            if (o_conv_context_ready) $fatal(1, "wrong-version refill was accepted");
        end
        if (!o_sticky_error || conv_reads || conv_writes || classifier_writes)
            $fatal(1, "invalid refill version was not rejected without RMW");
        $display("CASE_PASS invalid_context_version no_parameter_traffic");
        $display("SCNN_PARAMETER_BASE_PASS split=%0d conv_reads=0 conv_writes=0 classifier_writes=24 refills=%0d negative_cases=2",
            USE_CONTEXT_PARAMETER_BASE, refills);
        $finish;
    end

    initial begin
        repeat (100000) tick();
        $fatal(1, "parameter-base timeout reads=%0d writes=%0d error=%b runtime_busy=%b plane_busy=%b",
            conv_reads, conv_writes, o_sticky_error, dut.runtime_busy, dut.plane_busy);
    end
