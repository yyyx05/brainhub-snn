// Exclusive helper for tb_srnn_formal_m2_numeric; no alternate learning model.
logic c_cfg_ready, c_cfg_error, c_engine_cfg_valid;
logic c_prepare_valid, c_cache_invalidate, c_engine_start, c_include_target;
logic [7:0] c_prepare_version, c_engine_sequence, c_model_version, c_adapt_version;
logic [6:0] c_engine_timestep;
logic c_obs_valid, c_obs_path, c_obs_last, c_obs_done, c_external_ready;
logic [SAMPLE_ID_WIDTH-1:0] c_obs_sample;
logic [8:0] c_obs_backbone;
logic signed [31:0] c_obs_q14;
logic c_response_error, c_start_ready, c_done, c_busy, c_error;
logic c_load_valid = 0, c_load_ready;
logic [PARAM_ADDR_WIDTH-1:0] c_load_addr = 0;
logic c_label_start, c_label_valid = 0, c_label_ready;
logic [SAMPLE_ID_WIDTH-1:0] c_label_sample = 0;
logic [7:0] labels [0:1];
logic c_commit, c_detach, c_code_dirty, c_optimizer_dirty;
logic [PARAM_ADDR_WIDTH:0] c_zero_count, c_update_count;
integer commit_count = 0, detach_count = 0, gradient_checks = 0;
integer rmw_reads = 0, code_writes = 0, residual_writes = 0;
integer changed_readbacks = 0, frozen_wh1_checks = 0, store_element_checks = 0;
integer first_code_step = -1, first_changed_read_step = -1, first_logit_change_step = -1;
integer changed_logit_checks = 0;
integer prepare_stale_id_cycles = 0;
integer last_progress_cycle = 0;

`ifdef SRNN_FORMAL_M2_ONLINE
generate if (WITH_COORDINATOR) begin : g_online
    logic signed [15:0] expected_before [0:128*5120-1];
    logic signed [15:0] expected_after [0:128*5120-1];
    logic signed [15:0] expected_residual [0:128*5120-1];
    logic signed [63:0] expected_gradient [0:128*5120-1];
    logic signed [63:0] off_logit [0:128*40-1];
    logic [PARAM_ADDR_WIDTH-1:0] response_addr_q = 0;
    unified_srnn_m2_timestep_train_infer_coordinator #(
        .USE_INTERNAL_STORE(1'b1), .USE_EXTERNAL_BACKBONE_STREAM(1'b1),
        .USE_SHARED_H40_FORWARD(1'b1), .ENABLE_WH1_LEARNING(1'b0),
        .STORE_DEPTH(PARAM_DEPTH), .BATCH_MAX(2), .TIMESTEP_MAX(128),
        .HIDDEN_MAX(256), .BACKBONE_MAX(450), .CLASS_MAX(20), .SOURCE_VECTOR_MAX(450),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH), .TIMESTEP_ID_WIDTH(7), .HIDDEN_INDEX_WIDTH(9),
        .BACKBONE_INDEX_WIDTH(9), .CLASS_INDEX_WIDTH(5),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(8), .VERSION_WIDTH(8)
    ) u_coordinator (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_clear_error(clear_error),
        .i_cfg_valid(cfg_valid),
        .o_cfg_ready(c_cfg_ready),
        .i_cfg_profile_id(8'd7),
        .i_cfg_wh1_tensor_id(8'd16),
        .i_cfg_wh1_base(17'd0),
        .i_cfg_wh1_increment(16'd31),
        .i_cfg_wh1_threshold(16'd65),
        .i_cfg_wout_tensor_id(8'd17),
        .i_cfg_wout_base(17'd115200),
        .i_cfg_wout_increment(16'd31),
        .i_cfg_wout_threshold(16'd65),
        .i_cfg_code_min(16'sh8000),
        .i_cfg_code_max(16'sh7fff),
        .i_cfg_batch_count(SAMPLE_ID_WIDTH'(2)),
        .i_cfg_hidden_count(9'd256),
        .i_cfg_backbone_count(9'd450),
        .i_cfg_class_count(5'd20),
        .i_cfg_wout_element_count(17'd5120),
        .o_cfg_error(c_cfg_error),
        .i_load_valid(c_load_valid),
        .o_load_ready(c_load_ready),
        .i_load_addr(c_load_addr),
        .i_load_code(model_mem[c_load_addr]),
        .i_load_residual(16'sd0),
        .i_timestep_start(timestep_start),
        .o_timestep_start_ready(c_start_ready),
        .i_timestep_id(timestep_id),
        .i_sequence_version(sequence_version),
        .i_learning_enable(learning_enable),
        .i_abort_timestep('0),
        .o_abort_ready(),
        .o_actual_replay_start(),
        .i_actual_replay_start_ready('0),
        .o_actual_replay_timestep_id(),
        .i_actual_replay_valid('0),
        .o_actual_replay_ready(),
        .i_actual_replay_sample_id('0),
        .i_actual_replay_spikes('0),
        .i_actual_replay_done('0),
        .i_actual_replay_error('0),
        .o_target_vector_start(),
        .i_target_vector_start_ready('0),
        .o_target_vector_timestep_id(),
        .i_target_vector_valid('0),
        .o_target_vector_ready(),
        .i_target_vector_sample_id('0),
        .i_target_vector_spikes('0),
        .i_target_vector_done('0),
        .i_target_vector_error('0),
        .i_external_backbone_valid(observation_valid),
        .o_external_backbone_ready(c_external_ready),
        .i_external_backbone_path(observation_path),
        .i_external_backbone_sample_id(observation_sample),
        .i_external_backbone_id(observation_backbone),
        .i_external_backbone_q14(observation_q14),
        .i_external_backbone_last(observation_last),
        .i_external_backbone_done(observation_done),
        .i_external_backbone_error('0),
        .o_target_trace_start(),
        .i_target_trace_start_ready('0),
        .o_target_trace_timestep_id(),
        .i_target_trace_valid('0),
        .o_target_trace_ready(),
        .i_target_trace_sample_id('0),
        .i_target_trace_backbone_id('0),
        .i_target_trace_q14('0),
        .i_target_trace_last('0),
        .i_target_trace_done('0),
        .i_target_trace_error('0),
        .o_target_trace_valid(),
        .i_target_trace_ready(1'b1),
        .o_target_trace_sample_id(),
        .o_target_trace_backbone_id(),
        .o_target_trace_q14(),
        .o_target_trace_last(),
        .o_target_trace_done(),
        .o_forward_code_valid(),
        .i_forward_code_ready(1'b1),
        .o_forward_tensor_id(),
        .o_forward_local_addr(),
        .o_forward_parameter_addr(),
        .o_forward_code(),
        .o_forward_last(),
        .o_forward_parameter_scan_done(),
        .o_h40_cfg_valid(c_engine_cfg_valid),
        .i_h40_cfg_ready(cfg_ready),
        .i_h40_cfg_error(cfg_error),
        .o_h40_cfg_batch_count(),
        .o_h40_cfg_backbone_count(),
        .o_h40_cfg_hidden_count(),
        .o_h40_cfg_class_count(),
        .o_h40_cfg_wh1_parameter_base(),
        .o_h40_cfg_wout_parameter_base(),
        .o_h40_prepare_valid(c_prepare_valid),
        .i_h40_prepare_ready(prepare_ready),
        .o_h40_prepare_version(c_prepare_version),
        .i_h40_prepare_done(prepare_done),
        .i_h40_cache_valid(cache_valid),
        .o_h40_cache_invalidate(c_cache_invalidate),
        .i_h40_forward_read_valid(refill_read_valid),
        .o_h40_forward_read_ready(refill_read_ready),
        .i_h40_forward_read_addr(refill_read_addr),
        .i_h40_forward_read_tag(refill_read_tag),
        .o_h40_forward_response_valid(refill_response_valid),
        .i_h40_forward_response_ready(refill_response_ready),
        .o_h40_forward_response_tag(refill_response_tag),
        .o_h40_forward_response_code(refill_response_code),
        .o_h40_forward_response_error(c_response_error),
        .o_h40_timestep_start(c_engine_start),
        .i_h40_timestep_start_ready(timestep_start_ready),
        .o_h40_timestep_id(c_engine_timestep),
        .o_h40_sequence_version(c_engine_sequence),
        .o_h40_include_target_path(c_include_target),
        .o_h40_observation_valid(c_obs_valid),
        .i_h40_observation_ready(observation_ready),
        .o_h40_observation_path(c_obs_path),
        .o_h40_observation_sample_id(c_obs_sample),
        .o_h40_observation_backbone_id(c_obs_backbone),
        .o_h40_observation_q14(c_obs_q14),
        .o_h40_observation_last(c_obs_last),
        .o_h40_observation_done(c_obs_done),
        .i_h40_hidden_valid(hidden_valid),
        .o_h40_hidden_ready(hidden_ready),
        .i_h40_hidden_path(hidden_path),
        .i_h40_hidden_sample_id(hidden_sample),
        .i_h40_hidden_id(hidden_id),
        .i_h40_hidden_membrane_pre_q14(hidden_pre),
        .i_h40_hidden_membrane_next_q14(hidden_next),
        .i_h40_hidden_spike(hidden_spike),
        .i_h40_hidden_trace_q14(hidden_trace),
        .i_h40_hidden_output_feature_q14(hidden_feature),
        .i_h40_hidden_last(hidden_last),
        .i_h40_logit_valid(logit_valid),
        .o_h40_logit_ready(logit_ready),
        .i_h40_logit_sample_id(logit_sample),
        .i_h40_logit_class_id(logit_class),
        .i_h40_logit_q28(logit_q28),
        .i_h40_logit_last(logit_last),
        .i_h40_timestep_id(observed_timestep ^
            (((inject_timestep_id_error && hidden_valid) ||
              (inject_done_timestep_id_error && timestep_done && !engine_busy)) ? 7'd1 : 7'd0)),
        .i_h40_timestep_done(timestep_done),
        .i_h40_busy(engine_busy),
        .i_h40_sticky_error(engine_error),
        .i_h40_protocol_error(protocol_error),
        .i_h40_overflow(overflow),
        .o_hidden_observation_valid(),
        .i_hidden_observation_ready(1'b1),
        .o_hidden_observation_path(),
        .o_hidden_observation_sample_id(),
        .o_hidden_observation_hidden_id(),
        .o_hidden_membrane_pre_q14(),
        .o_hidden_membrane_next_q14(),
        .o_hidden_spike(),
        .o_hidden_trace_q14(),
        .o_hidden_output_feature_q14(),
        .o_hidden_observation_last(),
        .o_logit_valid(),
        .i_logit_ready(1'b1),
        .o_logit_sample_id(),
        .o_logit_class_id(),
        .o_logit_q28(),
        .o_logit_last(),
        .o_label_replay_start(c_label_start),
        .i_label_replay_start_ready(1'b1),
        .i_label_valid(c_label_valid),
        .o_label_ready(c_label_ready),
        .i_label_sample_id(c_label_sample),
        .i_label_class_id(labels[c_label_sample][4:0]),
        .i_label_last(c_label_sample == 1),
        .i_restore_complete('0),
        .o_restore_ready(),
        .o_recovery_required(),
        .o_learning_commit_pulse(c_commit),
        .o_detach_pulse(c_detach),
        .o_model_version(c_model_version),
        .o_adapt_version(c_adapt_version),
        .o_forward_model_version(),
        .o_forward_adapt_version(),
        .o_learning_base_model_version(),
        .o_learning_base_adapt_version(),
        .o_learning_commit_window_sequence(),
        .o_commit_code_dirty(c_code_dirty),
        .o_commit_optimizer_dirty(c_optimizer_dirty),
        .o_learning_result_dirty(),
        .o_learning_saturated(),
        .o_timestep_done(c_done),
        .o_timestep_aborted(),
        .o_busy(c_busy),
        .o_error(c_error),
        .o_zero_gradient_count(c_zero_count),
        .o_update_count(c_update_count),
        .o_store_load_valid(),
        .i_store_load_ready('0),
        .o_store_load_addr(),
        .o_store_load_code(),
        .o_store_load_residual(),
        .o_store_learning_read_valid(),
        .i_store_learning_read_ready('0),
        .o_store_learning_read_addr(),
        .o_store_learning_read_tag(),
        .i_store_learning_response_valid('0),
        .o_store_learning_response_ready(),
        .i_store_learning_response_tag('0),
        .i_store_learning_response_code('0),
        .i_store_learning_response_residual('0),
        .i_store_learning_response_error('0),
        .o_store_learning_write_valid(),
        .i_store_learning_write_ready('0),
        .o_store_learning_write_addr(),
        .o_store_learning_write_code(),
        .o_store_learning_write_residual(),
        .o_store_learning_write_code_enable(),
        .o_store_learning_write_residual_enable(),
        .o_store_forward_read_valid(),
        .i_store_forward_read_ready('0),
        .o_store_forward_read_addr(),
        .o_store_forward_read_tag(),
        .i_store_forward_response_valid('0),
        .o_store_forward_response_ready(),
        .i_store_forward_response_tag('0),
        .i_store_forward_response_code('0),
        .i_store_forward_response_error('0),
        .i_store_protocol_error('0)
    );

    task automatic load_real_parameters;
        $readmemh({fixture, "/labels.mem"}, labels, 0, 1);
        $readmemh({fixture, expectation_prefix, "wout_before.mem"}, expected_before, 0, steps*5120-1);
        $readmemh({fixture, expectation_prefix, "wout_after.mem"}, expected_after, 0, steps*5120-1);
        $readmemh({fixture, expectation_prefix, "residual.mem"}, expected_residual, 0, steps*5120-1);
        $readmemh({fixture, expectation_prefix, "gradient.mem"}, expected_gradient, 0, steps*5120-1);
        $readmemh({fixture, "/expected_logit.mem"}, off_logit, 0, steps*40-1);
        for (int address = 0; address < PARAM_DEPTH; address++) begin
            @(negedge clk);
            c_load_addr = PARAM_ADDR_WIDTH'(address);
            c_load_valid = 1;
            do @(posedge clk); while (!c_load_ready);
        end
        @(negedge clk);
        c_load_valid = 0;
        $display("SRNN_FORMAL_M2_STORE_LOADED elements=%0d cycles=%0d", PARAM_DEPTH, cycles);
        $fflush();
    endtask

    task automatic check_store_after(input int timestep);
        for (int address = 0; address < 115200; address++) begin
            if (tb_srnn_formal_m2_numeric.g_online.u_coordinator.parameter_store.g_internal_store.shared_store.code_mem[address] !== model_mem[address] ||
                tb_srnn_formal_m2_numeric.g_online.u_coordinator.parameter_store.g_internal_store.shared_store.residual_mem[address] !== 0)
                $fatal(1, "frozen Wh1 changed address=%0d", address);
        end
        frozen_wh1_checks = frozen_wh1_checks + 115200;
        for (int address = 0; address < 5120; address++) begin
            if (tb_srnn_formal_m2_numeric.g_online.u_coordinator.parameter_store.g_internal_store.shared_store.code_mem[115200+address] !== expected_after[timestep*5120+address] ||
                tb_srnn_formal_m2_numeric.g_online.u_coordinator.parameter_store.g_internal_store.shared_store.residual_mem[115200+address] !== expected_residual[timestep*5120+address])
                $fatal(1, "M2_STORE_MISMATCH t=%0d addr=%0d code=%0d expected=%0d residual=%0d expected_residual=%0d",
                    timestep, address,
                    tb_srnn_formal_m2_numeric.g_online.u_coordinator.parameter_store.g_internal_store.shared_store.code_mem[115200+address],
                    expected_after[timestep*5120+address],
                    tb_srnn_formal_m2_numeric.g_online.u_coordinator.parameter_store.g_internal_store.shared_store.residual_mem[115200+address],
                    expected_residual[timestep*5120+address]);
        end
        store_element_checks = store_element_checks + 5120;
        $display("SRNN_FORMAL_M2_ONLINE_STEP t=%0d learning=%0d zero_gradients=%0d updates=%0d rmw_total=%0d code_writes_total=%0d residual_writes_total=%0d commits=%0d detach=%0d model=%0d adapt=%0d",
            timestep, learning_enable, c_zero_count, c_update_count, rmw_reads, code_writes,
            residual_writes, commit_count, detach_count, c_model_version, c_adapt_version);
    endtask

    initial begin
        wait (rst_n);
        forever begin
            @(posedge clk);
            if (c_label_start) begin
                $display("SRNN_FORMAL_M2_LABEL_START t=%0d", timestep_id);
                $fflush();
                for (int sample_id = 0; sample_id < 2; sample_id++) begin
                    @(negedge clk);
                    c_label_sample = SAMPLE_ID_WIDTH'(sample_id);
                    c_label_valid = 1;
                    do @(posedge clk); while (!c_label_ready);
                    $display("SRNN_FORMAL_M2_LABEL sample=%0d class=%0d", c_label_sample, labels[c_label_sample]);
                    $fflush();
                    @(negedge clk);
                    c_label_valid = 0;
                end
            end
        end
    end

    always @(posedge clk) if (rst_n) begin
        if ((c_load_valid && c_load_ready) || (refill_read_valid && refill_read_ready) ||
            (hidden_valid && hidden_ready) || (logit_valid && logit_ready) ||
            (c_label_valid && c_label_ready) || c_commit || c_detach ||
            (u_coordinator.parameter_store.g_wout_only_learning.learning_engine.reducer_gradient_valid &&
             u_coordinator.parameter_store.g_wout_only_learning.learning_engine.reducer_gradient_ready))
            last_progress_cycle <= cycles;
        if (cycles - last_progress_cycle > 100000) begin
            $display("M2_LEARNER_READY configured=%0d forward_state=%0d engine_start=%0d boundary_start=%0d learner_state=%0d learner_configured=%0d reducer_start=%0d update_active=%0d cfg_valid=%0d",
                u_coordinator.parameter_store.configured_q,
                u_coordinator.parameter_store.forward_state_q,
                u_coordinator.parameter_store.engine_start_ready,
                u_coordinator.parameter_store.boundary_learning_start_ready,
                u_coordinator.parameter_store.g_wout_only_learning.learning_engine.state_q,
                u_coordinator.parameter_store.g_wout_only_learning.learning_engine.configured_q,
                u_coordinator.parameter_store.g_wout_only_learning.learning_engine.reducer_start_ready,
                u_coordinator.parameter_store.g_wout_only_learning.learning_engine.update_active,
                u_coordinator.parameter_store.g_wout_only_learning.learning_engine.i_cfg_valid);
            $fatal(1, "M2_NO_PROGRESS t=%0d coordinator=%0d adapter=%0d frontend=%0d start_ready[obs,num,store,contrib]=%0d%0d%0d%0d labels_complete=%0d labels_expected=%0d label_valid=%0d label_ready=%0d hidden=%0d logits=%0d gradients=%0d commits=%0d detach=%0d",
                timestep_id, u_coordinator.state_q, u_engine.u_adapter.state_q,
                u_coordinator.contribution_frontend.state_q,
                u_coordinator.observation_start_ready, u_coordinator.numeric_timestep_start_ready,
                u_coordinator.store_learning_start_ready, u_coordinator.contribution_start_ready,
                u_coordinator.contribution_frontend.label_complete_q,
                u_coordinator.contribution_frontend.label_expected_sample_q,
                c_label_valid, c_label_ready, hidden_seen, logit_seen, gradient_checks, commit_count, detach_count);
        end
        if (inject_done_timestep_id_error && timestep_done && !engine_busy)
            $display("M2_ID_INJECTION kind=2 busy=0 done=1");
        if (inject_timestep_id_error && hidden_valid)
            $display("M2_ID_INJECTION kind=1 busy=%0d done=%0d", engine_busy, timestep_done);
        if (u_coordinator.state_q == 3 && engine_busy &&
            observed_timestep != u_coordinator.timestep_q && !c_error)
            prepare_stale_id_cycles <= prepare_stale_id_cycles + 1;
        if (c_error || c_cfg_error || c_response_error)
            $fatal(1, "M2_COORDINATOR_ERROR t=%0d state=%0d store=%0b forward=%0b observation=%0b numeric_protocol=%0b numeric_overflow=%0b timestep_id=%0b frontend=%0b engine=%0b prep_state=%0d refill=%0b clear=%0b local=%0b layer=%0b layer_code=%0h array=%0b owner=%0b scalar=%0b cache_valid=%0b cache_version=%0d prepare_version=%0d model=%0d adapt=%0d sequence=%0d observed_timestep=%0d",
                timestep_id, u_coordinator.state_q, u_coordinator.store_error,
                u_coordinator.store_forward_error, u_coordinator.observation_error,
                u_coordinator.numeric_protocol_error, u_coordinator.numeric_overflow,
                u_coordinator.numeric_timestep_id_error, u_coordinator.contribution_protocol_error,
                engine_error, u_engine.prep_state_q, u_engine.refill_error,
                u_engine.clear_error, u_engine.local_error_q, layer_error, layer_error_code,
                array_error, owner_error, scalar_error, cache_valid, cache_version,
                c_prepare_version, c_model_version, c_adapt_version, c_engine_sequence, observed_timestep);
        if (c_prepare_valid && prepare_ready) begin
            $display("M2_PREPARE t=%0d model=%0d adapt=%0d requested_version=%0d cache_valid=%0b cache_version=%0d sequence=%0d array_idle=%0b owner_idle=%0b",
                timestep_id, c_model_version, c_adapt_version, c_prepare_version,
                cache_valid, cache_version, c_engine_sequence, array_idle, owner_idle);
            $fflush();
        end
        if (c_include_target) $fatal(1, "Wout-only path unexpectedly requested target backbone");
        if (timestep_id == 2 && u_coordinator.contribution_frontend.softmax_value_fire)
            $display("M2_T2_SOFTMAX_INPUT row=%0d class=%0d metric=%0d",
                u_coordinator.contribution_frontend.row_q,
                u_coordinator.contribution_frontend.softmax_load_index_q,
                u_coordinator.contribution_frontend.softmax_value_q12);
        if (timestep_id == 2 && u_coordinator.contribution_frontend.softmax_probability_fire)
            $display("M2_T2_SOFTMAX_PROB row=%0d class=%0d probability=%0d",
                u_coordinator.contribution_frontend.row_q,
                u_coordinator.contribution_frontend.softmax_probability_index,
                u_coordinator.contribution_frontend.softmax_probability_q024);
        if (timestep_id == 2 && u_coordinator.parameter_store.i_wout_contribution_valid &&
            u_coordinator.parameter_store.o_wout_contribution_ready &&
            u_coordinator.parameter_store.i_wout_contribution_addr == 200) begin
            $display("M2_T2_GRADIENT_INPUT address=200 sample=%0d error=%0d trace=%0d",
                u_coordinator.parameter_store.i_wout_contribution_sample_id,
                u_coordinator.parameter_store.i_wout_error_q40,
                u_coordinator.parameter_store.i_wout_hidden_trace_q14);
            $fflush();
        end
        if (c_commit) begin
            commit_count <= commit_count + 1;
            $display("SRNN_FORMAL_M2_COMMIT t=%0d code_dirty=%0d optimizer_dirty=%0d model=%0d adapt=%0d",
                timestep_id, c_code_dirty, c_optimizer_dirty, c_model_version, c_adapt_version);
        end
        if (c_detach) detach_count <= detach_count + 1;
        if (u_coordinator.parameter_store.engine_read_valid && u_coordinator.parameter_store.engine_read_ready)
            rmw_reads <= rmw_reads + 1;
        if (u_coordinator.parameter_store.engine_result_valid && u_coordinator.parameter_store.engine_result_ready) begin
            if (u_coordinator.parameter_store.engine_result_addr < 115200)
                $fatal(1, "learner wrote frozen Wh1");
            if (u_coordinator.parameter_store.engine_write_code_enable) begin
                code_writes <= code_writes + 1;
                if (first_code_step < 0) begin
                    first_code_step <= int'(timestep_id);
                    $display("M2_FIRST_CODE_CHANGE t=%0d address=%0d", timestep_id,
                        u_coordinator.parameter_store.engine_result_addr);
                end
            end
            if (u_coordinator.parameter_store.engine_write_residual_enable) residual_writes <= residual_writes + 1;
        end
        if (refill_read_valid && refill_read_ready) response_addr_q <= refill_read_addr;
        if (refill_response_valid && refill_response_ready) begin
            if (response_addr_q < 115200) begin
                if (refill_response_code !== model_mem[response_addr_q])
                    $fatal(1, "Wh1 shared-store read differs from formal input");
            end else begin
                if (refill_response_code !== expected_before[int'(timestep_id)*5120+int'(response_addr_q)-115200])
                    $fatal(1, "updated Wout not visible at next refill t=%0d address=%0d got=%0d expected=%0d",
                        timestep_id, response_addr_q, refill_response_code,
                        expected_before[int'(timestep_id)*5120+int'(response_addr_q)-115200]);
                if (refill_response_code != model_mem[response_addr_q]) begin
                    changed_readbacks <= changed_readbacks + 1;
                    if (first_changed_read_step < 0) first_changed_read_step <= int'(timestep_id);
                end
            end
        end
        if (logit_valid && logit_ready && logit_q28 !== off_logit[logit_seen]) begin
            changed_logit_checks <= changed_logit_checks + 1;
            if (first_logit_change_step < 0) begin
                first_logit_change_step <= int'(timestep_id);
                $display("M2_FIRST_FORWARD_EFFECT t=%0d sample=%0d class=%0d on=%0d off=%0d",
                    timestep_id, logit_sample, logit_class, logit_q28, off_logit[logit_seen]);
            end
        end
        if (u_coordinator.parameter_store.g_wout_only_learning.learning_engine.reducer_gradient_valid &&
            u_coordinator.parameter_store.g_wout_only_learning.learning_engine.reducer_gradient_ready) begin
            if (u_coordinator.parameter_store.g_wout_only_learning.learning_engine.reducer_gradient_addr != gradient_checks % 5120 ||
                u_coordinator.parameter_store.g_wout_only_learning.learning_engine.reducer_gradient_q40 !== expected_gradient[gradient_checks])
                $fatal(1, "M2_GRADIENT_MISMATCH index=%0d actual=%0d expected=%0d", gradient_checks,
                    u_coordinator.parameter_store.g_wout_only_learning.learning_engine.reducer_gradient_q40, expected_gradient[gradient_checks]);
            gradient_checks <= gradient_checks + 1;
        end
    end
end endgenerate
`endif
