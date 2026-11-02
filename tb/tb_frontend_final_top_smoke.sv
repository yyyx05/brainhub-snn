// Stimulus body consumed by run_fc_final_counts.generate_tb. The generated
// standalone wrapper instantiates the unmodified final top and shared store.
    longint cycles = 0;
    int resets = 0, layer_done_count = 0, capture_count = 0;
    int groups = 0, observations = 0, descriptors = 0, retired = 0;
    int restores = 0, spills = 0, classifier_restores = 0, classifier_spills = 0;
    int state_writes = 0, trace_writes = 0, state_reads = 0, trace_reads = 0;
    int descriptor_stalls = 0, feature_stalls = 0, request_stalls = 0;
    int commits = 0, learning_writes = 0, code_writes = 0, residual_writes = 0;
    int program_index = 0, operation = 0, opcode = 0, payload_count = 0;
    int fd, scanned, stopped = 0, stage = 0, initialized_once = 0;
    longint unsigned payload [0:767];
    string program_path;
    logic previous_reset = 0, frame_done_seen = 0, dense_active = 0;
    logic saw_dense_owner = 0, saw_scnn_owner = 0, saw_maintenance_owner = 0;
    logic feature_pending = 0, srnn_pending = 0;
    int feature_delay = 0;
    logic [TAG_WIDTH-1:0] srnn_tag = '0;
    logic signed [15:0] srnn_codes [0:3];
    logic [319:0] state_shadow [0:1][0:8][0:1][0:1][0:1];
    logic [319:0] trace_shadow [0:1][0:8][0:1][0:1][0:1];
    logic [$bits(u_final.u_system.descriptor_input)-1:0] stalled_descriptor;
    logic descriptor_was_stalled = 0, stalled_last = 0;
    int dense_expected_pre = 0, dense_expected_spike = 0, dense_expected_mode = 0;
    int dense_observations = 0;

    always #5 i_clk = ~i_clk;
    assign i_feature_request_ready = !feature_pending && cycles % 4 == 0;
    assign i_feature_response_valid = feature_pending && feature_delay == 0;
    assign i_srnn_backbone_forward_read_ready = !srnn_pending;
    assign i_srnn_backbone_forward_response_valid = srnn_pending;
    assign i_srnn_backbone_forward_response_tag = srnn_tag;

    always @(posedge i_clk) begin
        cycles++;
        if (cycles > 50000)
            $fatal(1, "FT_FATAL TIMEOUT index=%0d op=%0d code=%0d q=%b owner=%0d capture_idle=%b phase_idle=%b",
                program_index, operation, opcode, o_quiescent, o_array_active_owner,
                o_fc_srnn_learning_idle, o_srnn_phase_idle);
        if (i_rst_n && !previous_reset) resets++;
        if (!i_rst_n && previous_reset) $fatal(1, "FT_FATAL second global reset");
        previous_reset = i_rst_n;
        if (i_rst_n && (o_sticky_error || o_backend_error || i_store_protocol_error ||
                o_owner_encoding_error || o_fc_srnn_sticky_error ||
                o_fc_srnn_learning_sticky_error || o_fc_srnn_observation_sticky_error ||
                o_cache_refill_sticky_error || o_srnn_phase_sticky_error))
            $fatal(1, "FT_FATAL DUT_ERROR index=%0d op=%0d system/backend/store=%b/%b/%b dense=%b code=%0d capture=%b code=%0d cache=%b phase=%b",
                program_index,operation,o_sticky_error,o_backend_error,i_store_protocol_error,
                o_fc_srnn_sticky_error,o_fc_srnn_error_code,o_fc_srnn_learning_sticky_error,
                o_fc_srnn_learning_error_code,o_cache_refill_sticky_error,o_srnn_phase_sticky_error);
        if (i_rst_n) begin
            if (i_learning_enable || i_system_learning_enable || i_topology_learning_enable != 0 ||
                    i_srnn_phase_learning_enable || i_learning_fc_cfg_valid || i_learning_srnn_cfg_valid)
                $fatal(1, "FT_FATAL learning unexpectedly enabled");
            if (o_commit_pulse) commits++;
            if (o_store_learning_write_valid) begin
                learning_writes++;
                if (o_store_learning_write_code_enable) code_writes++;
                if (o_store_learning_write_residual_enable) residual_writes++;
                $fatal(1, "FT_FATAL learning write while off");
            end
            if (o_fc_srnn_layer_done) layer_done_count++;
            if (o_fc_srnn_learning_capture_done) capture_count++;
            if (i_fc_srnn_group_valid && o_fc_srnn_group_ready) groups++;
            if (o_done) frame_done_seen = 1;
            if (i_graph_descriptor_valid && o_graph_descriptor_ready) descriptors++;
            if (o_graph_descriptor_done) retired++;
            if (o_array_active_owner == 1) saw_dense_owner = 1;
            if (o_array_active_owner == 2) saw_scnn_owner = 1;
            if (o_array_active_owner == 3) saw_maintenance_owner = 1;
            if (i_graph_descriptor_valid && !o_graph_descriptor_ready) descriptor_stalls++;
            if (o_feature_request_valid && !i_feature_request_ready) feature_stalls++;
            if (i_request_valid && !o_request_ready) request_stalls++;
            if (i_state_maintenance_write_valid && i_state_maintenance_write_mask != '1)
                $fatal(1, "FT_FATAL state write mask changed");
            if (i_trace_maintenance_write_valid && i_trace_maintenance_write_mask != '1)
                $fatal(1, "FT_FATAL trace write mask changed");
            if (descriptor_was_stalled && (u_final.u_system.descriptor_input != stalled_descriptor ||
                    i_graph_descriptor_last != stalled_last || !i_graph_descriptor_valid))
                $fatal(1, "FT_FATAL descriptor changed under backpressure");
            descriptor_was_stalled = i_graph_descriptor_valid && !o_graph_descriptor_ready;
            stalled_descriptor = u_final.u_system.descriptor_input;
            stalled_last = i_graph_descriptor_last;
            if (dense_active && o_fc_srnn_observation_valid &&
                    u_final.u_system.array_fc_srnn_observation_ready) begin
                if (o_array_active_owner != 1 || o_fc_srnn_observation_mode != dense_expected_mode ||
                        o_fc_srnn_observation_transaction_tag != i_fc_srnn_transaction_tag ||
                        o_fc_srnn_observation_neuron_id != dense_observations ||
                        o_fc_srnn_observation_pre_state != dense_expected_pre ||
                        o_fc_srnn_observation_spike != dense_expected_spike ||
                        o_fc_srnn_observation_last_in_record != (dense_observations == 1))
                    $fatal(1, "FT_FATAL dense observation mode=%0d neuron=%0d pre=%0d spike=%b expected=%0d/%0d",
                        o_fc_srnn_observation_mode,o_fc_srnn_observation_neuron_id,
                        o_fc_srnn_observation_pre_state,o_fc_srnn_observation_spike,
                        dense_expected_pre,dense_expected_spike);
                observations++; dense_observations++;
            end
        end
        if (!i_rst_n) begin
            feature_pending <= 0; srnn_pending <= 0; feature_delay <= 0;
        end else begin
            if (o_feature_request_valid && i_feature_request_ready) begin
                feature_pending <= 1; feature_delay <= 2;
            end else if (feature_delay > 0) feature_delay <= feature_delay - 1;
            else if (feature_pending && o_feature_response_ready) feature_pending <= 0;
            if (o_srnn_backbone_forward_read_valid && i_srnn_backbone_forward_read_ready) begin
                if (o_srnn_backbone_forward_read_addr >= 4)
                    $fatal(1, "FT_FATAL synthetic SRNN address out of range");
                srnn_pending <= 1;
                srnn_tag <= o_srnn_backbone_forward_read_tag;
                i_srnn_backbone_forward_response_code <= srnn_codes[int'(o_srnn_backbone_forward_read_addr)];
            end else if (srnn_pending && o_srnn_backbone_forward_response_ready) srnn_pending <= 0;
        end
        i_feature_update_done <= o_feature_update_valid && i_feature_update_ready;
        if (i_rst_n && o_feature_update_valid && i_feature_update_ready && o_feature_update_data != 0)
            $fatal(1, "FT_FATAL zero-input/zero-weight SCNN generated nonzero features");
    end

    task automatic tick;
        @(posedge i_clk); #1;
    endtask

    task automatic drain;
        while (!o_quiescent || !o_backend_quiescent || !i_store_quiescent ||
                o_array_active_owner != 0 || !o_fc_srnn_learning_idle ||
                !o_srnn_phase_idle || feature_pending || srnn_pending) tick();
        repeat (2) tick();
        if (!o_quiescent || o_array_active_owner != 0) $fatal(1, "FT_FATAL unstable drain");
    endtask

    task automatic load_fc(input int base, count, code, residual);
        for (int address = base; address < base + count; address++) begin
            @(negedge i_clk);
            i_learning_fc_parameter_load_addr = PARAM_ADDR_WIDTH'(address);
            i_learning_fc_parameter_load_code = CODE_WIDTH'(code);
            i_learning_fc_parameter_load_residual = RESIDUAL_WIDTH'(residual);
            i_learning_fc_parameter_load_valid = 1;
            do @(posedge i_clk); while (!o_learning_fc_parameter_load_ready);
            #1; @(negedge i_clk); i_learning_fc_parameter_load_valid = 0;
        end
        while (!i_store_quiescent || !o_backend_quiescent) tick();
    endtask

    task automatic clear_dense_state;
        @(negedge i_clk);
        i_maintenance_tile = 0;
        i_state_maintenance_write_addr = 0; i_state_maintenance_write_data = 0;
        i_trace_maintenance_write_addr = MEMORY_ADDR_WIDTH'(MEMORY_DEPTH/2);
        i_trace_maintenance_write_data = 0;
        i_state_maintenance_write_valid = 1;
        do @(posedge i_clk); while (!o_state_maintenance_write_ready);
        #1; @(negedge i_clk); i_state_maintenance_write_valid = 0;
        i_trace_maintenance_write_valid = 1;
        do @(posedge i_clk); while (!o_trace_maintenance_write_ready);
        #1; @(negedge i_clk); i_trace_maintenance_write_valid = 0;
        while (o_array_active_owner != 0) tick();
    endtask

    task automatic dense_dispatch;
        int done_base, capture_base;
        if (payload_count != 26) $fatal(1, "FT_FATAL dense payload size");
        @(negedge i_clk);
        i_fc_srnn_mode = 2'(payload[0]);
        i_fc_srnn_neuron_count = 16'(payload[1]);
        i_fc_srnn_feedforward_group_count = 16'(payload[2]);
        i_fc_srnn_recurrent_group_count = 16'(payload[3]);
        i_fc_srnn_recurrent_enable = 1'(payload[4]);
        i_fc_srnn_weight_int8 = 1'(payload[5]); i_fc_srnn_current_shift6 = 1'(payload[6]);
        i_fc_srnn_leak = 16'(payload[7]); i_fc_srnn_threshold = 32'(payload[8]);
        i_fc_srnn_reset_to_zero = 1'(payload[9]); i_fc_srnn_trace_decay = 16'(payload[10]);
        i_fc_srnn_transaction_tag = 52'(payload[11]);
        i_fc_srnn_work_base_addr = WORK_ADDR_WIDTH'(payload[12]);
        i_fc_srnn_destination_tile_base = 4'(payload[13]);
        i_fc_srnn_group_source_valid = 4'(payload[14]);
        i_fc_srnn_group_weight_addr0 = WEIGHT_ADDR_WIDTH'(payload[15]);
        i_fc_srnn_group_weight_addr1 = WEIGHT_ADDR_WIDTH'(payload[16]);
        i_fc_srnn_group_weight_addr2 = WEIGHT_ADDR_WIDTH'(payload[17]);
        i_fc_srnn_group_weight_addr3 = WEIGHT_ADDR_WIDTH'(payload[18]);
        if (opcode == 1) begin
            i_fc_w0_refill_all = 1;
            i_fc_w0_parameter_base = PARAM_ADDR_WIDTH'(payload[19]);
            i_fc_w0_hidden_stride = PARAM_ADDR_WIDTH'(payload[20]);
            i_fc_w0_feature_count = FEATURE_COUNT_WIDTH'(payload[21]);
            i_fc_w0_hidden_count = 10'(payload[1]); i_fc_w0_tile_count = 5'(payload[22]);
            i_fc_w0_version = VERSION_WIDTH'(payload[23]);
            i_fc_w0_refill_valid = 1;
            do @(posedge i_clk); while (!o_fc_w0_refill_ready);
            #1; @(negedge i_clk); i_fc_w0_refill_valid = 0;
            while (!o_fc_w0_refill_done) tick();
            if (o_fc_w0_rows_published != payload[21] || o_fc_w0_published_version != payload[23])
                $fatal(1, "FT_FATAL FC native refill");
        end else begin
            i_srnn_phase_record_count = $bits(i_srnn_phase_record_count)'(payload[1]);
            i_srnn_phase_capture_cfg_valid = 1;
            do @(posedge i_clk); while (!o_srnn_phase_capture_cfg_ready);
            #1; @(negedge i_clk); i_srnn_phase_capture_cfg_valid = 0;
            i_srnn_backbone_refill_all = 1;
            i_srnn_backbone_parameter_base = SRNN_BACKBONE_PARAM_ADDR_WIDTH'(payload[19]);
            i_srnn_backbone_source_count = 10'(payload[21]);
            i_srnn_backbone_destination_count = 10'(payload[1]);
            i_srnn_backbone_version = VERSION_WIDTH'(payload[23]);
            i_srnn_backbone_refill_valid = 1;
            do @(posedge i_clk); while (!o_srnn_backbone_refill_ready);
            #1; @(negedge i_clk); i_srnn_backbone_refill_valid = 0;
            while (!o_srnn_backbone_refill_done) tick();
            if (o_srnn_backbone_rows_published != payload[21] || o_srnn_backbone_published_version != payload[23])
                $fatal(1, "FT_FATAL SRNN native refill");
        end
        clear_dense_state();
        @(negedge i_clk);
        i_fc_srnn_learning_cfg_mode = i_fc_srnn_mode;
        i_fc_srnn_learning_cfg_tag = i_fc_srnn_transaction_tag;
        i_fc_srnn_learning_neuron_count = 10'(payload[1]);
        i_fc_srnn_learning_fc_threshold = 18'(payload[8]);
        i_fc_srnn_learning_cfg_valid = 1;
        do @(posedge i_clk); while (!o_fc_srnn_learning_cfg_ready);
        #1; @(negedge i_clk); i_fc_srnn_learning_cfg_valid = 0;
        done_base = layer_done_count; capture_base = capture_count;
        dense_observations = 0; dense_active = 1;
        dense_expected_mode = int'(payload[0]); dense_expected_pre = int'(payload[24]);
        dense_expected_spike = int'(payload[25]);
        i_fc_srnn_layer_valid = 1;
        do @(posedge i_clk); while (!o_fc_srnn_layer_ready);
        #1; @(negedge i_clk); i_fc_srnn_layer_valid = 0;
        i_fc_srnn_group_valid = 1;
        do @(posedge i_clk); while (!o_fc_srnn_group_ready);
        #1; @(negedge i_clk); i_fc_srnn_group_valid = 0;
        while (layer_done_count == done_base || capture_count == capture_base) tick();
        if (dense_observations != 2 || layer_done_count != done_base + 1 || capture_count != capture_base + 1)
            $fatal(1, "FT_FATAL dense completion counts");
        dense_active = 0;
        if (opcode == 16) begin
            if (!o_srnn_phase_capture_complete) $fatal(1, "FT_FATAL SRNN tape capture missing");
            // No M2 dispatch is claimed: retire the captured synthetic tape via
            // the native phase abort, not a global reset or a forced RTL signal.
            @(negedge i_clk); i_srnn_phase_abort = 1;
            tick(); @(negedge i_clk); i_srnn_phase_abort = 0;
        end
        drain();
        $display("FT_DENSE,%0d,%0d,%0d,%0d",opcode,dense_observations,
            layer_done_count-done_base,capture_count-capture_base);
    endtask

    task automatic service_state;
        logic release_context, classifier_context, completion_sent;
        logic [319:0] saved_data;
        int base, sample_id, slot, spatial, tiles;
        release_context = o_state_service_release;
        classifier_context = o_state_service_is_classifier;
        base = int'(o_state_service_work_base_addr);
        tiles = int'(o_state_service_output_channel_count);
        sample_id = int'(o_graph_frame_sample);
        slot = classifier_context ? 8 : int'(u_final.u_system.descriptor_active.layer_tag)-1;
        spatial = classifier_context ? 0 : int'(o_state_service_spatial_base)/2;
        completion_sent = 0;
        if (tiles != (classifier_context ? 2 : 1) || sample_id > 1 || slot < 0 || slot > 8 ||
                spatial > 1 || (!classifier_context && o_state_service_spatial_count != 2))
            $fatal(1, "FT_FATAL unexpected context classifier=%b tiles=%0d sample=%0d slot=%0d spatial=%0d",
                classifier_context,tiles,sample_id,slot,spatial);
        repeat (3) tick();
        for (int tile = 0; tile < tiles; tile++) begin
        for (int row = 0; row < 2; row++) begin
            @(negedge i_clk); i_maintenance_tile = 4'(tile);
            if (!release_context) begin
                i_state_maintenance_write_addr = MEMORY_ADDR_WIDTH'(base+row);
                i_state_maintenance_write_data = state_shadow[sample_id][slot][spatial][tile][row];
                i_trace_maintenance_write_addr = MEMORY_ADDR_WIDTH'(MEMORY_DEPTH/2+base+row);
                i_trace_maintenance_write_data = trace_shadow[sample_id][slot][spatial][tile][row];
                i_state_maintenance_write_valid = 1;
                do @(posedge i_clk); while (!o_state_maintenance_write_ready);
                #1; state_writes++; @(negedge i_clk); i_state_maintenance_write_valid = 0;
                i_trace_maintenance_write_valid = 1;
                do @(posedge i_clk); while (!o_trace_maintenance_write_ready);
                #1; trace_writes++; @(negedge i_clk); i_trace_maintenance_write_valid = 0;
            end else begin
                i_state_maintenance_read_addr = MEMORY_ADDR_WIDTH'(base+row);
                i_state_maintenance_read_valid = 1;
                do @(posedge i_clk); while (!o_state_maintenance_read_ready);
                #1; @(negedge i_clk); i_state_maintenance_read_valid = 0;
                while (!o_state_maintenance_result_valid) tick();
                saved_data = o_state_maintenance_read_data;
                repeat (3) begin
                    tick();
                    if (!o_state_maintenance_result_valid || o_state_maintenance_read_data != saved_data ||
                            o_state_maintenance_result_addr != MEMORY_ADDR_WIDTH'(base+row) ||
                            o_array_active_owner != 2 || i_parameter_owner != 4 ||
                            u_final.u_parameter_plane.u_parameter_backend.active_owner != 3 ||
                            !u_final.u_system.array_context_release_valid ||
                            u_final.u_system.array_context_release_ready || o_graph_descriptor_done)
                        $fatal(1, "FT_FATAL state response/owner barrier");
                end
                if (saved_data != 0) $fatal(1, "FT_FATAL synthetic zero-state mismatch");
                state_shadow[sample_id][slot][spatial][tile][row] = saved_data;
                state_reads++;
                @(negedge i_clk); i_state_maintenance_result_ready = 1;
                tick(); @(negedge i_clk); i_state_maintenance_result_ready = 0;
                i_trace_maintenance_read_addr = MEMORY_ADDR_WIDTH'(MEMORY_DEPTH/2+base+row);
                i_trace_maintenance_read_valid = 1;
                do @(posedge i_clk); while (!o_trace_maintenance_read_ready);
                #1; @(negedge i_clk); i_trace_maintenance_read_valid = 0;
                while (!o_trace_maintenance_result_valid) tick();
                saved_data = o_trace_maintenance_read_data;
                if (row == 1 && tile == tiles-1) begin
                    @(negedge i_clk); i_state_service_ready = 1;
                    tick(); @(negedge i_clk); i_state_service_ready = 0;
                    completion_sent = 1;
                end
                repeat (3) begin
                    tick();
                    if (!o_trace_maintenance_result_valid || o_trace_maintenance_read_data != saved_data ||
                            o_trace_maintenance_result_addr != MEMORY_ADDR_WIDTH'(MEMORY_DEPTH/2+base+row) ||
                            o_array_active_owner != 2 || i_parameter_owner != 4 ||
                            u_final.u_parameter_plane.u_parameter_backend.active_owner != 3 ||
                            !u_final.u_system.array_context_release_valid ||
                            u_final.u_system.array_context_release_ready || o_graph_descriptor_done)
                        $fatal(1, "FT_FATAL trace response/early-completion owner barrier");
                end
                if (saved_data != 0) $fatal(1, "FT_FATAL synthetic zero-trace mismatch");
                trace_shadow[sample_id][slot][spatial][tile][row] = saved_data;
                trace_reads++;
                @(negedge i_clk); i_trace_maintenance_result_ready = 1;
                tick(); @(negedge i_clk); i_trace_maintenance_result_ready = 0;
            end
        end
        end
        if (release_context) begin spills++; if (classifier_context) classifier_spills++; end
        else begin restores++; if (classifier_context) classifier_restores++; end
        if (!completion_sent) begin
            @(negedge i_clk); i_state_service_ready = 1;
            tick(); @(negedge i_clk); i_state_service_ready = 0;
        end
    endtask

    task automatic scnn_dispatch;
        int offset, sample_id, timestep_id, is_last;
        logic [$bits(u_final.u_system.descriptor_input)-1:0] expected_descriptor;
        if (payload_count != 740 || payload[3] != 32) $fatal(1, "FT_FATAL SCNN payload size");
        @(negedge i_clk);
        i_batch_count = BATCH_COUNT_WIDTH'(payload[0]);
        i_timestep_count = TIMESTEP_COUNT_WIDTH'(payload[1]);
        i_timestep = TIMESTEP_WIDTH'(payload[2]); i_sample_version = o_model_version;
        frame_done_seen = 0; i_request_valid = 1;
        do @(posedge i_clk); while (!o_request_ready);
        #1; @(negedge i_clk); i_request_valid = 0;
        for (int record = 0; record < 32; record++) begin
            offset = 4 + record*23;
            sample_id = int'(payload[offset]); timestep_id = int'(payload[offset+1]);
            is_last = int'(payload[offset+2]); offset += 3;
            @(negedge i_clk);
            i_conv_parameter_base = PARAM_ADDR_WIDTH'(payload[offset]);
            i_conv_in_width = 16'(payload[offset+1]); i_conv_in_height = 16'(payload[offset+2]);
            i_conv_in_channels = 16'(payload[offset+3]); i_conv_out_width = 16'(payload[offset+4]);
            i_conv_out_height = 16'(payload[offset+5]); i_conv_out_channels = 16'(payload[offset+6]);
            i_conv_stride = 8'(payload[offset+7]); i_conv_padding = 8'(payload[offset+8]);
            i_conv_pool_mode = 2'(payload[offset+9]); i_conv_work_base_addr = WORK_ADDR_WIDTH'(payload[offset+10]);
            i_conv_leak = 16'(payload[offset+11]); i_conv_threshold = 32'(payload[offset+12]);
            i_conv_reset_to_zero = 1'(payload[offset+13]); i_conv_trace_decay = 16'(payload[offset+14]);
            i_conv_layer_tag = 20'(payload[offset+15]); i_feature_write_slot = 1'(payload[offset+16]);
            i_feature_write_base_addr = FEATURE_ADDR_WIDTH'(payload[offset+17]);
            i_feature_write_row_stride_words = 16'(payload[offset+18]);
            i_feature_write_channel_stride_words = FEATURE_ADDR_WIDTH'(payload[offset+19]);
            expected_descriptor = {i_conv_parameter_base,i_conv_in_width,i_conv_in_height,i_conv_in_channels,
                i_conv_out_width,i_conv_out_height,i_conv_out_channels,i_conv_stride,i_conv_padding,
                i_conv_pool_mode,i_conv_work_base_addr,i_conv_leak,i_conv_threshold,i_conv_reset_to_zero,
                i_conv_trace_decay,i_conv_layer_tag,i_feature_write_slot,i_feature_write_base_addr,
                i_feature_write_row_stride_words,i_feature_write_channel_stride_words};
            i_graph_descriptor_last = 1'(is_last); i_graph_descriptor_valid = 1;
            #1;
            while (!o_graph_descriptor_ready) begin
                if (o_state_service_valid) service_state(); else tick();
            end
            if (int'(o_graph_frame_sample) != sample_id || int'(o_graph_frame_timestep) != timestep_id ||
                    o_graph_frame_version != o_model_version)
                $fatal(1, "FT_FATAL frame metadata record=%0d sample=%0d time=%0d",record,
                    o_graph_frame_sample,o_graph_frame_timestep);
            tick();
            if (u_final.u_system.descriptor_active != expected_descriptor)
                $fatal(1, "FT_FATAL complete native descriptor capture record=%0d",record);
            @(negedge i_clk); i_graph_descriptor_valid = 0;
            while (!o_graph_descriptor_done) begin
                if (o_state_service_valid) service_state(); else tick();
            end
        end
        while (!frame_done_seen) begin
            if (o_state_service_valid) service_state(); else tick();
        end
        drain();
        if (descriptors != 32 || retired != 32 || restores != 68 || spills != 68 ||
                classifier_restores != 4 || classifier_spills != 4 || !saw_scnn_owner ||
                !saw_maintenance_owner || descriptor_stalls == 0 || feature_stalls == 0)
            $fatal(1, "FT_FATAL SCNN coverage descriptors/retired=%0d/%0d restore/spill=%0d/%0d stalls=%0d/%0d",
                descriptors,retired,restores,spills,descriptor_stalls,feature_stalls);
        $display("FT_SCNN,%0d,4,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
            retired,restores,spills,classifier_restores,classifier_spills,
            state_writes,trace_writes,state_reads,trace_reads,descriptor_stalls,feature_stalls,request_stalls);
    endtask

    initial begin
        if (!$value$plusargs("program=%s", program_path)) $fatal(1, "FT_FATAL missing program");
        fd = $fopen(program_path,"r"); if (!fd) $fatal(1, "FT_FATAL program open");
        for (int s=0;s<2;s++) for (int l=0;l<9;l++) for (int p=0;p<2;p++)
                for (int t=0;t<2;t++) for (int r=0;r<2;r++) begin
            state_shadow[s][l][p][t][r] = 0; trace_shadow[s][l][p][t][r] = 0;
        end
        i_parameter_owner = 1;
        i_system_parameter_owner = 1; i_system_profile_aligned = 1;
        i_system_profile_id = 32; i_profile_id = 32;
        i_context_model_version = 5; i_context_adapt_version = 7; i_sample_version = 5;
        i_classifier_pair = 1; i_total_input_channels = 1; i_total_output_channels = 1;
        i_classifier_parameter_base = 32; i_classifier_feature_count = 4;
        i_classifier_class_count = 2; i_classifier_feature_channel_count = 1;
        i_classifier_work_base_addr = 20; i_classifier_trace_decay = 16'h7fff;
        i_classifier_layer_tag = 20'h80001; i_target_feature_count = 4;
        i_target_trace_leak_q8_8 = 128; i_classifier_trace_leak_valid = 1;
        i_classifier_trace_leak_q8_8 = 128; i_classifier_vote_threshold = 16'h8000;
        i_code_min = -32768; i_code_max = 32767;
        i_feature_update_ready = 1;
        i_state_maintenance_write_mask = '1; i_trace_maintenance_write_mask = '1;
        repeat (4) tick(); @(negedge i_clk); i_rst_n = 1; tick();
        // Bootstrap SCNN once so the first frontend DRAIN uses real initialized
        // quiescence rather than waiving the final top's initialized condition.
        load_fc(0,40,0,0);
        @(negedge i_clk); i_parameter_owner = 4;
        repeat (4) tick(); @(negedge i_clk); i_initialize_valid = 1;
        do @(posedge i_clk); while (!o_initialize_ready);
        #1; @(negedge i_clk); i_initialize_valid = 0;
        while (!o_initialize_done) tick();
        initialized_once++;
        drain(); $display("FT_BOOT,1,%0d,%0d",o_model_version,o_adapt_version);
        while (!$feof(fd)) begin
            scanned = $fscanf(fd,"%d %d %d",operation,opcode,payload_count);
            if (scanned == -1 || scanned == 0) break;
            if (scanned != 3 || payload_count < 0 || payload_count > 768 ||
                    (opcode != 1 && opcode != 16 && opcode != 32) || operation != stage+1)
                $fatal(1, "FT_FATAL invalid frontend command");
            for (int k=0;k<payload_count;k++)
                if ($fscanf(fd,"%d",payload[k]) != 1) $fatal(1, "FT_FATAL truncated payload");
            $display("FT_STEP,%0d,%0d,%0d",program_index,operation,opcode);
            case (operation)
                1: begin if (stopped) $fatal(1,"FT_FATAL duplicate STOP"); stopped=1; end
                2: begin drain(); $display("FT_DRAIN,%0d,%0d",opcode,o_array_active_owner); end
                3: begin
                    drain();
                    if (payload_count != 4) $fatal(1,"FT_FATAL asset payload");
                    if (opcode == 16) begin
                        if (payload[0] != 0 || payload[1] != 4 || payload[3] != 0)
                            $fatal(1,"FT_FATAL SRNN synthetic asset geometry");
                        for (int k=0;k<4;k++) srnn_codes[k] = 16'(payload[2]);
                    end else begin
                        @(negedge i_clk); i_parameter_owner = 1;
                        repeat (2) tick();
                        load_fc(int'(payload[0]),int'(payload[1]),int'(payload[2]),int'(payload[3]));
                    end
                end
                4: begin
                    drain();
                    if (payload_count != 7 || payload[4] != 0) $fatal(1,"FT_FATAL config payload");
                    @(negedge i_clk);
                    i_parameter_owner = 3'(payload[0]); i_system_parameter_owner = 1'(payload[1]);
                    i_profile_id = PROFILE_ID_WIDTH'(payload[2]);
                    i_system_profile_id = PROFILE_ID_WIDTH'(payload[3]);
                    i_context_model_version = VERSION_WIDTH'(payload[5]);
                    i_context_adapt_version = VERSION_WIDTH'(payload[6]);
                    repeat (2) tick();
                    if (i_profile_id != opcode || i_system_profile_id != opcode ||
                            u_final.u_parameter_plane.u_parameter_backend.active_owner != (opcode==1?1:opcode==16?2:3) ||
                            o_array_active_owner != 0)
                        $fatal(1,"FT_FATAL parameter/array owner configuration");
                    $display("FT_CONFIG,%0d,%0d,%0d,%0d",opcode,i_parameter_owner,
                        u_final.u_parameter_plane.u_parameter_backend.active_owner,o_array_active_owner);
                end
                5: begin
                    while (!o_versions_aligned) tick();
                    if (!o_initialized || o_model_version != 5 || o_parameter_version != 5 || o_adapt_version != 7)
                        $fatal(1,"FT_FATAL initialization/version visibility");
                end
                6: begin if (opcode==32) scnn_dispatch(); else dense_dispatch(); end
                7: begin drain(); if (commits || learning_writes) $fatal(1,"FT_FATAL learning publication"); end
                8: begin drain(); stopped=0; end
                default: $fatal(1,"FT_FATAL unknown operation");
            endcase
            $display("FT_ACK,%0d,%0d,%0d,%0d",program_index,operation,opcode,payload_count);
            stage = operation == 8 ? 0 : operation;
            program_index++;
        end
        $fclose(fd);
        if (program_index != 32 || stage || stopped || resets != 1 || initialized_once != 1 ||
                layer_done_count != 3 || groups != 3 || observations != 6 || !saw_dense_owner)
            $fatal(1,"FT_FATAL terminal command/reset/dense counts");
        $display("FT_COUNTS,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",resets,
            layer_done_count,groups,observations,retired,commits,learning_writes,code_writes,residual_writes);
        $display("FT_CYCLES,%0d",cycles);
        $display("FT_PASS"); $finish;
    end
