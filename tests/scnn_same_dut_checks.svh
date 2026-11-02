// Included in the NEW generated test module, not in any production RTL.
// Hierarchical accesses below observe data only. Stimulus uses DUT ports.
    integer sd_learn_on = 1;
    logic sd_release = 1'b0;
    wire sd_release_ready;
    wire sd_learning = sd_learn_on != 0 && timestep_phase < 2 && layer_phase == 8;
    logic signed [15:0] sd_features [0:6143];
    logic signed [15:0] sd_current [0:32];
    logic signed [15:0] sd_logits [0:32];
    logic signed [15:0] sd_traces [0:4095];
    logic signed [15:0] sd_codes [0:PARAMS-1];
    logic signed [15:0] sd_residuals [0:PARAMS-1];
    logic [15:0] sd_counts [0:2];
    logic [15:0] sd_rmw_mask [0:PARAMS-1];
    logic [15:0] sd_label [0:0];
    bit sd_feature_seen [0:6143];
    bit sd_current_seen [0:32];
    bit sd_trace_seen [0:4095];
    bit sd_rmw_seen [0:PARAMS-1];
    bit sd_publish_seen [0:1][0:1891];
    integer sd_features_checked = 0;
    integer sd_currents_checked = 0;
    integer sd_traces_checked = 0;
    integer sd_rmw_count = 0;
    integer sd_code_writes = 0;
    integer sd_residual_writes = 0;
    integer sd_publish_words = 0;
    integer sd_prescan_requests = 0;
    integer sd_prescan_values = 0;
    integer sd_releases = 0;
    integer sd_commits = 0;
    logic [31:0] sd_previous_sequence = 0;
    longint unsigned sd_commit_cycle = 0;

`define SD_TAIL dut.u_core.g_scnn_graph.u_scnn_graph_client.u_classifier_tail
`define SD_CLIENT dut.u_learning_client

    task automatic sd_load_expected;
        string directory;
        integer unused_arg;
        begin
            if (!$value$plusargs("SCNN_SAME_DUT_FIXTURE=%s", directory))
                $fatal(1, "SCNN_SAME_DUT_FIXTURE is required");
            unused_arg = $value$plusargs("SCNN_LEARNING=%d", sd_learn_on);
            if (sd_learn_on != 0 && sd_learn_on != 1)
                $fatal(1, "SCNN_LEARNING must be 0 or 1");
            $readmemh({directory, "/expected_features.hex"}, sd_features);
            $readmemh({directory, "/expected_currents.hex"}, sd_current);
            $readmemh({directory, "/expected_logits.hex"}, sd_logits);
            $readmemh({directory, "/expected_traces.hex"}, sd_traces);
            $readmemh({directory, "/expected_codes.hex"}, sd_codes);
            $readmemh({directory, "/expected_residuals.hex"}, sd_residuals);
            $readmemh({directory, "/expected_counts.hex"}, sd_counts);
            $readmemh({directory, "/expected_rmw_mask.hex"}, sd_rmw_mask);
            $readmemh({directory, "/label.hex"}, sd_label);
        end
    endtask

    always @(posedge clk) begin : sd_observe
        integer index, address, feature, class_id, pair_index, row_index;
        integer parity, publication, word_index;
        logic signed [15:0] expected_weight;
        if (rst_n) begin
            if (`SD_TAIL.u_scheduler.i_feature_wave_valid &&
                    `SD_TAIL.u_scheduler.o_feature_wave_ready) begin
                if (layer_phase != 8 || timestep_phase < 0 || timestep_phase > 2)
                    $fatal(1, "same-DUT unexpected scheduler feature phase");
                for (int lane = 0; lane < 8; lane++) begin
                    if (`SD_TAIL.u_scheduler.i_feature_wave_mask[lane]) begin
                        feature = int'(`SD_TAIL.u_scheduler.i_feature_wave_base) + lane;
                        index = timestep_phase * CLASSIFIER_FEATURES + feature;
                        if (feature >= CLASSIFIER_FEATURES || sd_feature_seen[index] ||
                                $signed(`SD_TAIL.u_scheduler.i_feature_wave_q8_8_flat[lane*16 +: 16]) !== sd_features[index])
                            $fatal(1, "same-DUT feature mismatch frame=%0d feature=%0d", timestep_phase, feature);
                        sd_feature_seen[index] = 1;
                        sd_features_checked++;
                    end
                end
            end
            if (`SD_TAIL.serializer_scalar_valid && `SD_TAIL.serializer_scalar_ready) begin
                class_id = int'(`SD_TAIL.serializer_scalar_channel);
                index = timestep_phase * CLASSES + class_id;
                if (class_id >= CLASSES || timestep_phase > 2 || sd_current_seen[index] ||
                        `SD_TAIL.serializer_scalar_layer_tag != CLASSIFIER_TAG ||
                        `SD_TAIL.serializer_scalar_spatial != 0 ||
                        `SD_TAIL.serializer_scalar_pre_state !== sd_current[index])
                    $fatal(1, "same-DUT classifier current mismatch frame=%0d class=%0d got=%0d expected=%0d",
                        timestep_phase, class_id, int'(`SD_TAIL.serializer_scalar_pre_state), int'(sd_current[index]));
                sd_current_seen[index] = 1;
                sd_currents_checked++;
            end
            if (dut.o_classifier_computed_trace_valid && dut.i_classifier_computed_trace_ready) begin
                feature = int'(dut.o_classifier_computed_trace_index);
                index = int'(dut.o_classifier_computed_trace_timestep) * CLASSIFIER_FEATURES + feature;
                if (!sd_learning || feature >= CLASSIFIER_FEATURES || index >= 4096 ||
                        sd_trace_seen[index] || dut.o_classifier_computed_trace_sample != 0 ||
                        dut.o_classifier_computed_trace_frame_index != timestep_phase ||
                        dut.o_classifier_computed_trace_version != CLASSIFIER_VERSION ||
                        dut.o_classifier_computed_trace_q8_8 !== sd_traces[index])
                    $fatal(1, "same-DUT trace mismatch frame=%0d feature=%0d", timestep_phase, feature);
                sd_trace_seen[index] = 1;
                sd_traces_checked++;
            end
            if (dut.prescan_busy && feature_request_valid && feature_request_ready)
                sd_prescan_requests++;
            if (dut.o_conv8_input_valid && dut.i_conv8_input_ready) begin
                index = int'(dut.o_conv8_input_channel) * 16 + int'(dut.o_conv8_input_spatial);
                if (!sd_learning || index >= CONV7_SCALARS ||
                        dut.o_conv8_input_layer_tag != CONV8_TAG ||
                        dut.o_conv8_input_version != CLASSIFIER_VERSION ||
                        dut.o_conv8_input_sample != 0 || dut.o_conv8_input_timestep != timestep_phase ||
                        dut.o_conv8_input_q8_8 !== (conv7_expected_spike[index] ? 16'sd256 : 16'sd0))
                    $fatal(1, "same-DUT Conv8 prescan mismatch frame=%0d index=%0d", timestep_phase, index);
                sd_prescan_values++;
            end
            if (`SD_CLIENT.bridge_result_valid && `SD_CLIENT.bridge_result_ready) begin
                address = int'(`SD_CLIENT.bridge_result_addr);
                if (!sd_learn_on || timestep_phase != 1 || classifier_logit_wave_count != 2 ||
                        address >= PARAMS || sd_rmw_seen[address] || sd_rmw_mask[address] != 1 ||
                        `SD_CLIENT.bridge_result_code !== sd_codes[address] ||
                        `SD_CLIENT.bridge_result_residual !== sd_residuals[address] ||
                        `SD_CLIENT.bridge_write_code_enable != (sd_codes[address] != classifier_weights[address]) ||
                        `SD_CLIENT.bridge_write_residual_enable != (sd_residuals[address] != 0))
                    $fatal(1, "same-DUT RMW mismatch address=%0d frame=%0d code=%0d/%0d residual=%0d/%0d",
                        address, timestep_phase, int'(`SD_CLIENT.bridge_result_code), int'(sd_codes[address]),
                        int'(`SD_CLIENT.bridge_result_residual), int'(sd_residuals[address]));
                sd_rmw_seen[address] = 1;
                sd_rmw_count++;
                if (`SD_CLIENT.bridge_write_code_enable) sd_code_writes++;
                if (`SD_CLIENT.bridge_write_residual_enable) sd_residual_writes++;
            end
            // Check every word of initial publication and post-learning refill.
            for (int region = 0; region < 4; region++) begin
                if (dut.learning_fill_valid[region] && dut.learning_fill_ready[region]) begin
                    class_id = region*4 + int'(dut.learning_fill_tile_flat[region*2 +: 2]);
                    pair_index = int'(dut.learning_fill_addr_flat[region*WEIGHT_AW + WEIGHT_AW-1]);
                    row_index = int'(dut.learning_fill_addr_flat[region*WEIGHT_AW +: WEIGHT_AW-1]);
                    parity = int'(dut.learning_fill_bank[region]);
                    publication = cache_prime_count == 0 ? 0 : 1;
                    word_index = class_id*172 + row_index*2 + parity;
                    if (class_id >= CLASSES || row_index >= 86 || pair_index != (1-publication) ||
                            sd_publish_seen[publication][word_index] ||
                            (publication == 1 && (!sd_learn_on || timestep_phase != 1 || sd_rmw_count != int'(sd_counts[2]))))
                        $fatal(1, "same-DUT publication envelope class=%0d row=%0d pair=%0d stage=%0d",
                            class_id, row_index, pair_index, publication);
                    for (int subword = 0; subword < 12; subword++) begin
                        feature = row_index*24 + subword*2 + parity;
                        expected_weight = 0;
                        if (feature < CLASSIFIER_FEATURES)
                            expected_weight = publication == 0 ? classifier_weights[class_id*CLASSIFIER_FEATURES+feature] :
                                sd_codes[class_id*CLASSIFIER_FEATURES+feature];
                        if ($signed(dut.learning_fill_data_flat[region*200+subword*16 +: 16]) !== expected_weight)
                            $fatal(1, "same-DUT publication value class=%0d feature=%0d", class_id, feature);
                    end
                    sd_publish_seen[publication][word_index] = 1;
                    sd_publish_words++;
                end
            end
            if (dut.o_learning_commit_sequence != sd_previous_sequence) begin
                if (!sd_learn_on || timestep_phase != 1 || sd_commits != 0 ||
                        dut.o_learning_commit_sequence != 1 || sd_publish_words != 3784 ||
                        classifier_pair != 0 || classifier_version != CLASSIFIER_VERSION+1)
                    $fatal(1, "same-DUT premature/non-atomic commit");
                sd_previous_sequence = dut.o_learning_commit_sequence;
                sd_commits++;
                sd_commit_cycle = cycle_count;
            end
            if (timestep_phase == 2 && (parameter_load_valid || cache_prime_valid ||
                    `SD_CLIENT.bridge_result_valid || dut.cache_publish_busy || !rst_n))
                $fatal(1, "same-DUT next forward reloaded or changed learning state");
        end
    end

    task automatic sd_check_store;
        begin
            for (int address = 0; address < PARAMS; address++) begin
                if (`SD_CLIENT.u_parameter_store.code_mem[address] !== sd_codes[address] ||
                    `SD_CLIENT.u_parameter_store.residual_mem[address] !== sd_residuals[address] ||
                        sd_rmw_seen[address] != (sd_rmw_mask[address] != 0))
                    $fatal(1, "same-DUT committed store mismatch address=%0d", address);
            end
        end
    endtask

    task automatic sd_check_frame;
        integer watchdog;
        begin
            if (sd_features_checked != (timestep_phase+1)*CLASSIFIER_FEATURES ||
                    sd_currents_checked != (timestep_phase+1)*CLASSES)
                $fatal(1, "same-DUT incomplete classifier frame=%0d features=%0d currents=%0d",
                    timestep_phase, sd_features_checked, sd_currents_checked);
            if (timestep_phase == 0 && (sd_rmw_count != 0 || sd_commits != 0))
                $fatal(1, "same-DUT committed before full T2 window");
            // Graph done/idle can precede replay capture completion. Do not
            // inspect the committed store or detach until the DUT publishes.
            if (timestep_phase == 1 && sd_learn_on) begin
                watchdog = 0;
                while ((sd_commits != 1 || dut.o_learning_commit_sequence != 1 ||
                        dut.cache_publish_busy || dut.learning_inference_hold || !idle) &&
                        watchdog < 4000000) begin
                    @(negedge clk);
                    watchdog++;
                end
                if (sd_commits != 1 || dut.o_learning_commit_sequence != 1 ||
                        dut.cache_publish_busy || dut.learning_inference_hold || !idle)
                    $fatal(1, "same-DUT commit wait timeout sequence=%0d monitored=%0d capture=%0b replay_state=%0d hold=%0b publish=%0b rmw=%0d traces=%0d",
                        dut.o_learning_commit_sequence, sd_commits,
                        `SD_CLIENT.u_learning_bridge.o_capture_complete,
                        `SD_CLIENT.u_learning_bridge.state_q,
                        dut.learning_inference_hold, dut.cache_publish_busy, sd_rmw_count, sd_traces_checked);
            end
            if (timestep_phase >= 1) begin
                sd_check_store();
                if (sd_commits != sd_learn_on || sd_rmw_count != int'(sd_counts[2]) ||
                        sd_code_writes != int'(sd_counts[0]) || sd_residual_writes != int'(sd_counts[1]))
                    $fatal(1, "same-DUT learning result counts mismatch");
            end
            if (timestep_phase == 1) begin
                @(negedge clk);
                watchdog = 0;
                while (!sd_release_ready && watchdog < 4096) begin
                    @(negedge clk);
                    watchdog++;
                end
                if (!sd_release_ready) $fatal(1, "same-DUT learning detach timeout");
                sd_release = 1;
                @(posedge clk);
                @(negedge clk);
                sd_release = 0;
                sd_releases++;
                watchdog = 0;
                while (dut.learning_active && watchdog < 4096) begin
                    @(negedge clk);
                    watchdog++;
                end
                if (dut.learning_active) $fatal(1, "same-DUT learner did not detach");
                $display("SCNN_SAME_DUT_COMMIT learning=%0d sequence=%0d pair=%0d version=%0d code_writes=%0d residual_writes=%0d cycle=%0d",
                    sd_learn_on, sd_commits, classifier_pair, classifier_version, sd_code_writes, sd_residual_writes, sd_commit_cycle);
            end
        end
    endtask

    task automatic sd_finish;
        begin
            if (sd_features_checked != 6144 || sd_currents_checked != 33 ||
                    classifier_logit_wave_count != 3 || sd_traces_checked != sd_learn_on*4096 ||
                    sd_prescan_requests != sd_learn_on*4096 || sd_prescan_values != sd_learn_on*16384 ||
                    sd_publish_words != 1892*(1+sd_learn_on) || sd_releases != 1 ||
                    dut.learning_active || cache_prime_count != 1 || sticky_error || tile_error != 0)
                $fatal(1, "same-DUT terminal completeness failure");
            sd_check_store();
            $display("SCNN_SAME_DUT_GRAPH_LEARN_NEXT_PASS learning=%0d frames=3 features=%0d currents=%0d logits=%0d traces=%0d rmw=%0d code_writes=%0d residual_writes=%0d commits=%0d publisher_words=%0d cycles=%0d",
                sd_learn_on, sd_features_checked, sd_currents_checked, classifier_logit_wave_count*CLASSES,
                sd_traces_checked, sd_rmw_count, sd_code_writes, sd_residual_writes, sd_commits, sd_publish_words, cycle_count);
        end
    endtask
`undef SD_TAIL
`undef SD_CLIENT
