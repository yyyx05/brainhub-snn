    longint cycles = 0;
    int frame = 0, layer = 0, retired = 0, restores = 0, spills = 0;
    int classifier_restores = 0, classifier_spills = 0;
    int learn_on = 0, commits = 0, windows = 1, window_index = 0;
    logic frame_done_seen = 0;
    logic feature_pending = 0;
    always #5 i_clk = ~i_clk;
    always @(posedge i_clk) begin
        cycles++;
        if (cycles > 400000) $fatal(1, "frame-service timeout f=%0d l=%0d", frame, layer);
        if (i_rst_n && (o_sticky_error || o_backend_error || i_store_protocol_error))
            $fatal(1, "final-top error f=%0d l=%0d system=%0b backend=%0b store=%0b",
                frame,layer,o_sticky_error,o_backend_error,i_store_protocol_error);
        if (o_done) frame_done_seen = 1;
        if (o_commit_pulse) commits++;
        if (o_graph_descriptor_done) retired++;
    end
    assign i_feature_request_ready = !feature_pending;
    assign i_feature_response_valid = feature_pending;
    always @(posedge i_clk) begin
        if (!i_rst_n) feature_pending <= 0;
        else if (o_feature_request_valid && i_feature_request_ready) feature_pending <= 1;
        else if (feature_pending && o_feature_response_ready) feature_pending <= 0;
        i_feature_update_done <= o_feature_update_valid && i_feature_update_ready;
    end
    task automatic tick;
        @(posedge i_clk); #1;
    endtask
    task automatic service_state;
        logic release_context, classifier_context, completion_sent;
        int tiles, base;
        release_context = o_state_service_release;
        classifier_context = o_state_service_is_classifier;
        tiles = int'(o_state_service_output_channel_count);
        base = int'(o_state_service_work_base_addr);
        completion_sent = 0;
        if (!classifier_context && (tiles != 1 || o_state_service_spatial_count != 2 ||
                (o_state_service_spatial_base != 0 && o_state_service_spatial_base != 2)))
            $fatal(1, "unexpected row-bounded convolution context");
        repeat (3) tick();
        for (int tile = 0; tile < tiles; tile++) begin
            for (int row = 0; row < 2; row++) begin
                @(negedge i_clk);
                i_maintenance_tile = 4'(tile);
                if (!release_context) begin
                    i_state_maintenance_write_addr = MEMORY_ADDR_WIDTH'(base+row);
                    i_trace_maintenance_write_addr = MEMORY_ADDR_WIDTH'(MEMORY_DEPTH/2+base+row);
                    i_state_maintenance_write_valid = 1;
                    #1;
                    while (!o_state_maintenance_write_ready) tick();
                    tick();
                    @(negedge i_clk);
                    i_state_maintenance_write_valid = 0;
                    i_trace_maintenance_write_valid = 1;
                    #1;
                    while (!o_trace_maintenance_write_ready) tick();
                    tick();
                    @(negedge i_clk);
                    i_trace_maintenance_write_valid = 0;
                end else begin
                    i_state_maintenance_read_addr = MEMORY_ADDR_WIDTH'(base+row);
                    i_state_maintenance_read_valid = 1;
                    #1; while (!o_state_maintenance_read_ready) tick();
                    tick();
                    @(negedge i_clk); i_state_maintenance_read_valid = 0;
                    while (!o_state_maintenance_result_valid) tick();
                    if (tile == tiles-1 && row == 1) begin
                        @(negedge i_clk); i_state_service_ready = 1;
                        tick(); @(negedge i_clk); i_state_service_ready = 0;
                        completion_sent = 1;
                    end
                    repeat (3) begin
                        tick();
                        if ((!completion_sent && !o_state_service_valid) ||
                                o_graph_descriptor_done ||
                                !u_final.u_system.array_context_release_valid ||
                                u_final.u_system.array_context_release_ready)
                            $fatal(1, "state barrier lost during response backpressure");
                    end
                    @(negedge i_clk); i_state_maintenance_result_ready = 1;
                    tick(); @(negedge i_clk); i_state_maintenance_result_ready = 0;
                end
            end
        end
        if (release_context) begin
            spills++; if (classifier_context) classifier_spills++;
        end else begin
            restores++; if (classifier_context) classifier_restores++;
        end
        if (!completion_sent) begin
            @(negedge i_clk); i_state_service_ready = 1;
            tick(); @(negedge i_clk); i_state_service_ready = 0;
        end
    endtask
    initial begin
        if (!$value$plusargs("LEARNING=%d", learn_on)) learn_on = 0;
        if (!$value$plusargs("WINDOWS=%d", windows)) windows = 1;
        i_learning_enable = 1'(learn_on);
        i_system_learning_enable = 1'(learn_on);
        i_parameter_owner = 3'b001;
        i_topology_learning_enable = 3'b100;
        i_system_parameter_owner = 1;
        i_system_profile_aligned = 1;
        i_system_profile_id = 8'h20;
        i_profile_id = 8'h20;
        i_context_model_version = 5;
        i_context_adapt_version = 7;
        i_sample_version = 5;
        i_classifier_pair = 1;
        i_total_input_channels = 1;
        i_total_output_channels = 1;
        i_classifier_parameter_base = 32;
        i_classifier_feature_count = 4;
        i_classifier_class_count = 2;
        i_classifier_feature_channel_count = 1;
        i_classifier_work_base_addr = 20;
        i_classifier_trace_decay = 16'h7fff;
        i_classifier_layer_tag = 20'h80001;
        i_target_feature_count = 4;
        i_target_trace_leak_q8_8 = 128;
        i_classifier_trace_leak_valid = 1;
        i_classifier_trace_leak_q8_8 = 128;
        i_classifier_vote_increment = 0;
        i_classifier_vote_threshold = 16'h8000;
        i_code_min = -32768; i_code_max = 32767;
        i_batch_count = 2; i_timestep_count = 2;
        i_conv_in_width = 2; i_conv_in_height = 2; i_conv_in_channels = 1;
        i_conv_out_width = 2; i_conv_out_height = 2; i_conv_out_channels = 1;
        i_conv_stride = 1; i_conv_padding = 1;
        i_conv_leak = 16'h7fff; i_conv_threshold = 256;
        i_conv_trace_decay = 16'h7fff;
        i_feature_write_row_stride_words = 1;
        i_feature_write_channel_stride_words = 2;
        i_feature_update_ready = 1;
        i_state_maintenance_write_mask = '1;
        i_trace_maintenance_write_mask = '1;
        repeat (4) tick(); @(negedge i_clk); i_rst_n = 1; tick();
        for (int address = 0; address < 40; address++) begin
            @(negedge i_clk); i_learning_fc_parameter_load_addr = PARAM_ADDR_WIDTH'(address);
            i_learning_fc_parameter_load_valid = 1;
            #1; while (!o_learning_fc_parameter_load_ready) tick();
            tick();
            @(negedge i_clk); i_learning_fc_parameter_load_valid = 0;
        end
        while (!i_store_quiescent) tick();
        @(negedge i_clk); i_parameter_owner = 3'b100;
        repeat (4) tick();
        @(negedge i_clk); i_initialize_valid = 1;
        #1; while (!o_initialize_ready) tick();
        tick();
        @(negedge i_clk); i_initialize_valid = 0;
        while (!o_initialize_done) tick();
        for (window_index = 0; window_index < windows; window_index++) begin
        @(negedge i_clk);
        frame_done_seen = 0;
        i_timestep = TIMESTEP_WIDTH'(window_index*2);
        i_sample_version = o_model_version;
        @(negedge i_clk); i_request_valid = 1;
        #1; while (!o_request_ready) tick();
        tick();
        @(negedge i_clk); i_request_valid = 0;
        for (frame = 0; frame < 4; frame++) begin
            for (layer = 0; layer < 8; layer++) begin
                while (!o_graph_descriptor_ready) tick();
                if (int'(o_graph_frame_sample) != frame/2 ||
                        int'(o_graph_frame_timestep) != window_index*2+frame%2)
                    $fatal(1, "final-top frame advanced on intermediate descriptor");
                @(negedge i_clk);
                i_conv_layer_tag = 20'(layer+1);
                i_graph_descriptor_last = layer == 7;
                i_graph_descriptor_valid = 1;
                tick(); @(negedge i_clk); i_graph_descriptor_valid = 0;
                while (!o_graph_descriptor_done) begin
                    if (o_state_service_valid) service_state();
                    else tick();
                end
                @(negedge i_clk);
            end
        end
        while (!frame_done_seen) tick();
        end
        if (commits != learn_on*windows)
            $fatal(1, "unexpected learning commit count %0d expected %0d", commits, learn_on*windows);
        // Each 2x2 convolution uses two row-bounded spatial contexts.
        if (retired != 32*windows || restores != 68*windows || spills != 68*windows ||
            classifier_restores != 4*windows || classifier_spills != 4*windows)
            $fatal(1, "bad counts descriptors=%0d restore/spill=%0d/%0d classifier=%0d/%0d",
                retired,restores,spills,classifier_restores,classifier_spills);
        $display("PASS: SCNN_FINAL_FRAME_SERVICE_B2T2 windows=%0d descriptors=%0d frames=%0d restores=%0d spills=%0d classifier_contexts=%0d learning=%0d commits=%0d cycles=%0d",windows,retired,4*windows,restores,spills,classifier_restores,learn_on,commits,cycles);
        $finish;
    end
