// Read-only hierarchical monitors. All stimulus and state transfers use ports.
    string b4_directory;
    integer b4_frame = 0, b4_sample = 0, b4_t = 0, b4_window = 0;
    bit b4_query = 0;
    logic b4_release = 0;
    wire b4_release_ready;
    wire b4_learning = !b4_query && layer_phase == 8;
    logic [15:0] b4_labels [0:3], b4_query_sample [0:0];
    logic signed [15:0] b4_features [0:32767], b4_traces [0:32767];
    logic signed [15:0] b4_currents [0:175], b4_logits [0:175];
    logic signed [15:0] b4_query_currents [0:43], b4_query_logits [0:43];
    logic signed [15:0] b4_baseline_currents [0:43], b4_baseline_logits [0:43];
    logic signed [15:0] b4_codes [0:PARAMS-1], b4_residuals [0:PARAMS-1];
    logic signed [15:0] b4_before_codes [0:PARAMS-1], b4_before_residuals [0:PARAMS-1];
    logic [15:0] b4_counts [0:2], b4_rmw_mask [0:PARAMS-1], b4_query_counts [0:1];
    bit b4_feature_seen [0:B4_GRAPH_FRAMES*CLASSIFIER_FEATURES-1];
    bit b4_current_seen [0:B4_GRAPH_FRAMES*CLASSES-1];
    bit b4_trace_seen [0:32767], b4_rmw_seen [0:PARAMS-1];
    bit b4_publish_seen [0:B4_WINDOWS][0:1891];
    integer b4_features_checked = 0, b4_currents_checked = 0, b4_traces_checked = 0;
    integer b4_rmw = 0, b4_code_writes = 0, b4_residual_writes = 0;
    integer b4_previous_rmw = 0, b4_previous_codes = 0, b4_previous_residuals = 0;
    integer b4_publish_words = 0, b4_prescan_requests = 0, b4_prescan_values = 0;
    integer b4_commits = 0, b4_releases = 0;
    integer b4_current_changes = 0, b4_logit_changes = 0;

`define B4_TAIL dut.u_core.g_scnn_graph.u_scnn_graph_client.u_classifier_tail
`define B4_CLIENT dut.u_learning_client

    task automatic b4_load_window;
        string directory;
        begin
            directory = $sformatf("%s/window%02d", b4_directory, b4_window);
            $readmemh({directory, "/currents.hex"}, b4_currents);
            $readmemh({directory, "/logits.hex"}, b4_logits);
            $readmemh({directory, "/traces.hex"}, b4_traces);
            $readmemh({directory, "/codes.hex"}, b4_codes);
            $readmemh({directory, "/residuals.hex"}, b4_residuals);
            $readmemh({directory, "/before_codes.hex"}, b4_before_codes);
            $readmemh({directory, "/before_residuals.hex"}, b4_before_residuals);
            $readmemh({directory, "/rmw_mask.hex"}, b4_rmw_mask);
            $readmemh({directory, "/counts.hex"}, b4_counts);
            for (int i = 0; i < 32768; i++) b4_trace_seen[i] = 0;
            for (int i = 0; i < PARAMS; i++) b4_rmw_seen[i] = 0;
            b4_previous_rmw = b4_rmw;
            b4_previous_codes = b4_code_writes;
            b4_previous_residuals = b4_residual_writes;
        end
    endtask

    task automatic b4_load_initial;
        begin
            if (!$value$plusargs("SCNN_B4T4_FIXTURE=%s", b4_directory))
                $fatal(1, "SCNN_B4T4_FIXTURE is required");
            $readmemh({b4_directory, "/labels.hex"}, b4_labels);
            $readmemh({b4_directory, "/expected_features.hex"}, b4_features);
            $readmemh({b4_directory, "/query_sample.hex"}, b4_query_sample);
            $readmemh({b4_directory, "/query_currents.hex"}, b4_query_currents);
            $readmemh({b4_directory, "/query_logits.hex"}, b4_query_logits);
            $readmemh({b4_directory, "/query_baseline_currents.hex"}, b4_baseline_currents);
            $readmemh({b4_directory, "/query_baseline_logits.hex"}, b4_baseline_logits);
            $readmemh({b4_directory, "/query_counts.hex"}, b4_query_counts);
            if (b4_query_sample[0] > 3) $fatal(1, "invalid query sample");
            b4_load_window();
        end
    endtask

    task automatic b4_begin_frame;
        begin
            @(negedge clk);
            b4_query = timestep_phase >= B4_WINDOWS*16;
            b4_t = timestep_phase % 4;
            b4_sample = b4_query ? int'(b4_query_sample[0]) : (timestep_phase % 16)/4;
            b4_frame = b4_sample*4 + b4_t;
            if (!b4_query && timestep_phase != 0 && timestep_phase % 16 == 0) begin
                b4_window = timestep_phase / 16;
                b4_load_window();
                // Only configuration is reset: persistent code/residual and
                // publication state stay in this same DUT across all windows.
                configure_learning_client();
            end
            if (b4_t == 0)
                for (int tile = 0; tile < 12; tile++)
                    clear_tile_state_at_work_base(tile, CLASSIFIER_WORK_BASE);
            $display("SCNN_B4T4_FRAME_BEGIN global_frame=%0d window=%0d sample=%0d t=%0d query=%0d version=%0d",
                timestep_phase, b4_window+1, b4_sample, b4_t, b4_query, classifier_version);
        end
    endtask

    always @(posedge clk) begin : b4_observe
        integer index, local_index, feature, class_id, address, publication, row_index, pair_index, parity, word_index;
        logic signed [15:0] expected_weight;
        if (rst_n) begin
            if (`B4_TAIL.u_scheduler.i_feature_wave_valid && `B4_TAIL.u_scheduler.o_feature_wave_ready) begin
                if (layer_phase != 8 || timestep_phase >= B4_GRAPH_FRAMES)
                    $fatal(1, "B4T4 unexpected feature phase");
                for (int lane = 0; lane < 8; lane++) begin
                    if (`B4_TAIL.u_scheduler.i_feature_wave_mask[lane]) begin
                        feature = int'(`B4_TAIL.u_scheduler.i_feature_wave_base) + lane;
                        index = timestep_phase*CLASSIFIER_FEATURES + feature;
                        if (feature >= CLASSIFIER_FEATURES || b4_feature_seen[index] ||
                                $signed(`B4_TAIL.u_scheduler.i_feature_wave_q8_8_flat[lane*16 +: 16]) !== b4_features[b4_frame*CLASSIFIER_FEATURES+feature])
                            $fatal(1, "B4T4 actual feature mismatch frame=%0d feature=%0d", timestep_phase, feature);
                        b4_feature_seen[index] = 1;
                        b4_features_checked++;
                    end
                end
            end
            if (`B4_TAIL.serializer_scalar_valid && `B4_TAIL.serializer_scalar_ready) begin
                class_id = int'(`B4_TAIL.serializer_scalar_channel);
                index = timestep_phase*CLASSES + class_id;
                if (layer_phase != 8 || class_id >= CLASSES || timestep_phase >= B4_GRAPH_FRAMES ||
                        b4_current_seen[index] || `B4_TAIL.serializer_scalar_layer_tag != CLASSIFIER_TAG ||
                        `B4_TAIL.serializer_scalar_spatial != 0 ||
                        `B4_TAIL.serializer_scalar_pre_state !== classifier_expected_current[class_id])
                    $fatal(1, "B4T4 actual current mismatch frame=%0d class=%0d", timestep_phase, class_id);
                b4_current_seen[index] = 1;
                b4_currents_checked++;
                if (b4_query && `B4_TAIL.serializer_scalar_pre_state != b4_baseline_currents[b4_t*CLASSES+class_id])
                    b4_current_changes++;
            end
            if (logit_wave_fire && b4_query)
                for (int c = 0; c < CLASSES; c++)
                    if ($signed(logit_wave_flat[c*16 +: 16]) != b4_baseline_logits[b4_t*CLASSES+c])
                        b4_logit_changes++;
            if (dut.o_classifier_computed_trace_valid && dut.i_classifier_computed_trace_ready) begin
                feature = int'(dut.o_classifier_computed_trace_index);
                local_index = (int'(dut.o_classifier_computed_trace_sample)*4 +
                    int'(dut.o_classifier_computed_trace_timestep))*CLASSIFIER_FEATURES + feature;
                if (!b4_learning || feature >= CLASSIFIER_FEATURES || local_index >= 32768 ||
                        b4_trace_seen[local_index] || dut.o_classifier_computed_trace_sample != b4_sample ||
                        dut.o_classifier_computed_trace_timestep != b4_t ||
                        dut.o_classifier_computed_trace_frame_index != b4_t ||
                        dut.o_classifier_computed_trace_version != CLASSIFIER_VERSION + 16'(b4_window) ||
                        dut.o_classifier_computed_trace_q8_8 !== b4_traces[local_index])
                    $fatal(1, "B4T4 trace mismatch sample=%0d t=%0d feature=%0d", b4_sample, b4_t, feature);
                b4_trace_seen[local_index] = 1;
                b4_traces_checked++;
            end
            if (dut.prescan_busy && feature_request_valid && feature_request_ready)
                b4_prescan_requests++;
            if (dut.o_conv8_input_valid && dut.i_conv8_input_ready) begin
                index = int'(dut.o_conv8_input_channel)*16 + int'(dut.o_conv8_input_spatial);
                if (!b4_learning || index >= CONV7_SCALARS || dut.o_conv8_input_layer_tag != CONV8_TAG ||
                        dut.o_conv8_input_version != CLASSIFIER_VERSION + 16'(b4_window) ||
                        dut.o_conv8_input_sample != b4_sample || dut.o_conv8_input_timestep != b4_t ||
                        dut.o_conv8_input_q8_8 !== (conv7_expected_spike[index] ? 16'sd256 : 16'sd0))
                    $fatal(1, "B4T4 prescan mismatch frame=%0d index=%0d", timestep_phase, index);
                b4_prescan_values++;
            end
            if (`B4_CLIENT.bridge_result_valid && `B4_CLIENT.bridge_result_ready) begin
                address = int'(`B4_CLIENT.bridge_result_addr);
                if (b4_query || b4_frame != 15 || classifier_logit_wave_count != (b4_window+1)*16 ||
                        address >= PARAMS || b4_rmw_seen[address] || b4_rmw_mask[address] != 1 ||
                        `B4_CLIENT.bridge_result_code !== b4_codes[address] ||
                        `B4_CLIENT.bridge_result_residual !== b4_residuals[address] ||
                        `B4_CLIENT.bridge_write_code_enable != (b4_codes[address] != b4_before_codes[address]) ||
                        `B4_CLIENT.bridge_write_residual_enable != (b4_residuals[address] != b4_before_residuals[address]))
                    $fatal(1, "B4T4 RMW mismatch window=%0d address=%0d", b4_window+1, address);
                b4_rmw_seen[address] = 1;
                b4_rmw++;
                if (`B4_CLIENT.bridge_write_code_enable) b4_code_writes++;
                if (`B4_CLIENT.bridge_write_residual_enable) b4_residual_writes++;
            end
            for (int region = 0; region < 4; region++) begin
                if (dut.learning_fill_valid[region] && dut.learning_fill_ready[region]) begin
                    class_id = region*4 + int'(dut.learning_fill_tile_flat[region*2 +: 2]);
                    pair_index = int'(dut.learning_fill_addr_flat[region*WEIGHT_AW + WEIGHT_AW-1]);
                    row_index = int'(dut.learning_fill_addr_flat[region*WEIGHT_AW +: WEIGHT_AW-1]);
                    parity = int'(dut.learning_fill_bank[region]);
                    publication = cache_prime_count == 0 ? 0 : b4_window+1;
                    word_index = class_id*172 + row_index*2 + parity;
                    if (b4_query || class_id >= CLASSES || row_index >= 86 ||
                            pair_index != (1 ^ (publication & 1)) || b4_publish_seen[publication][word_index] ||
                            (publication != 0 && (b4_frame != 15 || b4_rmw - b4_previous_rmw != int'(b4_counts[2]))))
                        $fatal(1, "B4T4 publication envelope window=%0d word=%0d", b4_window+1, word_index);
                    for (int subword = 0; subword < 12; subword++) begin
                        feature = row_index*24 + subword*2 + parity;
                        expected_weight = 0;
                        if (feature < CLASSIFIER_FEATURES)
                            expected_weight = publication == 0 ? classifier_weights[class_id*CLASSIFIER_FEATURES+feature] :
                                b4_codes[class_id*CLASSIFIER_FEATURES+feature];
                        if ($signed(dut.learning_fill_data_flat[region*200+subword*16 +: 16]) !== expected_weight)
                            $fatal(1, "B4T4 publication data mismatch class=%0d feature=%0d", class_id, feature);
                    end
                    b4_publish_seen[publication][word_index] = 1;
                    b4_publish_words++;
                end
            end
            if (dut.o_learning_commit_sequence != b4_commits) begin
                if (b4_query || b4_frame != 15 || b4_commits != b4_window ||
                        dut.o_learning_commit_sequence != b4_commits+1 ||
                        b4_publish_words != (b4_commits+2)*1892 ||
                        classifier_pair != (1 ^ ((b4_commits+1) & 1)) ||
                        classifier_version != CLASSIFIER_VERSION + 16'(b4_commits+1))
                    $fatal(1, "B4T4 non-atomic/premature commit window=%0d", b4_window+1);
                b4_commits++;
            end
            if (b4_query && (parameter_load_valid || cache_prime_valid ||
                    `B4_CLIENT.bridge_result_valid || dut.cache_publish_busy || dut.learning_active))
                $fatal(1, "B4T4 next forward changed learning state or reloaded parameters");
            if (timestep_phase != 0 && (parameter_load_valid || cache_prime_valid))
                $fatal(1, "B4T4 parameters/cache re-primed after first frame");
        end
    end

    task automatic b4_check_store;
        begin
            for (int address = 0; address < PARAMS; address++)
                if (`B4_CLIENT.u_parameter_store.code_mem[address] !== b4_codes[address] ||
                        `B4_CLIENT.u_parameter_store.residual_mem[address] !== b4_residuals[address] ||
                        b4_rmw_seen[address] != (b4_rmw_mask[address] != 0))
                    $fatal(1, "B4T4 store mismatch window=%0d address=%0d", b4_window+1, address);
        end
    endtask

    task automatic b4_check_frame;
        integer watchdog;
        begin
            if (b4_features_checked != (timestep_phase+1)*CLASSIFIER_FEATURES ||
                    b4_currents_checked != (timestep_phase+1)*CLASSES)
                $fatal(1, "B4T4 incomplete classifier frame=%0d", timestep_phase);
            if (!b4_query && b4_frame < 15 && (b4_rmw != b4_previous_rmw || b4_commits != b4_window))
                $fatal(1, "B4T4 committed before all four samples/four steps");
            if (!b4_query && b4_frame == 15) begin
                // A graph completion can precede the capture/replay hold.
                // Wait for the actual publication, not a transient idle gap.
                watchdog = 0;
                while ((b4_commits != b4_window+1 ||
                        dut.o_learning_commit_sequence != b4_window+1 ||
                        dut.cache_publish_busy || dut.learning_inference_hold || !idle) &&
                        watchdog < 8000000) begin
                    @(negedge clk);
                    watchdog++;
                end
                if (b4_commits != b4_window+1 ||
                        dut.o_learning_commit_sequence != b4_window+1 ||
                        dut.cache_publish_busy || dut.learning_inference_hold || !idle)
                    $fatal(1, "B4T4 commit wait timeout window=%0d capture=%0b state=%0d rmw=%0d traces=%0d",
                        b4_window+1, `B4_CLIENT.u_learning_bridge.o_capture_complete,
                        `B4_CLIENT.u_learning_bridge.state_q, b4_rmw, b4_traces_checked);
                b4_check_store();
                if (b4_commits != b4_window+1 || b4_traces_checked != (b4_window+1)*32768 ||
                        b4_rmw - b4_previous_rmw != int'(b4_counts[2]) ||
                        b4_code_writes - b4_previous_codes != int'(b4_counts[0]) ||
                        b4_residual_writes - b4_previous_residuals != int'(b4_counts[1]))
                    $fatal(1, "B4T4 window counters mismatch window=%0d", b4_window+1);
                @(negedge clk);
                watchdog = 0;
                while (!b4_release_ready && watchdog < 8192) begin
                    @(negedge clk);
                    watchdog++;
                end
                if (!b4_release_ready) $fatal(1, "B4T4 release timeout");
                b4_release = 1;
                @(posedge clk);
                @(negedge clk);
                b4_release = 0;
                b4_releases++;
                watchdog = 0;
                while (dut.learning_active && watchdog < 8192) begin
                    @(negedge clk);
                    watchdog++;
                end
                if (dut.learning_active) $fatal(1, "B4T4 detach timeout");
                $display("SCNN_B4T4_COMMIT window=%0d codes=%0d residuals=%0d rmw=%0d pair=%0d version=%0d cycles=%0d",
                    b4_window+1, b4_counts[0], b4_counts[1], b4_counts[2], classifier_pair, classifier_version, cycle_count);
            end
            if (b4_query) b4_check_store();
        end
    endtask

    task automatic b4_finish;
        begin
            if (b4_features_checked != B4_GRAPH_FRAMES*CLASSIFIER_FEATURES ||
                    b4_currents_checked != B4_GRAPH_FRAMES*CLASSES || classifier_logit_wave_count != B4_GRAPH_FRAMES ||
                    b4_traces_checked != B4_WINDOWS*32768 || b4_commits != B4_WINDOWS || b4_releases != B4_WINDOWS ||
                    b4_prescan_requests != B4_WINDOWS*32768 || b4_prescan_values != B4_WINDOWS*131072 ||
                    b4_publish_words != (B4_WINDOWS+1)*1892 || b4_current_changes != int'(b4_query_counts[0]) ||
                    b4_logit_changes != int'(b4_query_counts[1]) || dut.learning_active || cache_prime_count != 1 ||
                    sticky_error || tile_error != 0)
                $fatal(1, "B4T4 incomplete terminal evidence");
            b4_check_store();
            $display("SCNN_SAME_DUT_B4T4_GRAPH_LEARN_NEXT_PASS windows=%0d frames=%0d features=%0d currents=%0d logits=%0d traces=%0d rmw=%0d code_writes=%0d residual_writes=%0d commits=%0d publisher_words=%0d query_current_changes=%0d query_logit_changes=%0d cycles=%0d",
                B4_WINDOWS, B4_GRAPH_FRAMES, b4_features_checked, b4_currents_checked, classifier_logit_wave_count*CLASSES,
                b4_traces_checked, b4_rmw, b4_code_writes, b4_residual_writes, b4_commits, b4_publish_words,
                b4_current_changes, b4_logit_changes, cycle_count);
        end
    endtask
`undef B4_TAIL
`undef B4_CLIENT
