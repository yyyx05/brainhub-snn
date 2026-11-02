`timescale 1ns/1ps
`default_nettype none

module tb_unified_h40_final_composition_scnn_v1 #(
    parameter bit CLASSIFIER_ONLY = 1'b0,
    parameter int SCNN_WEIGHT_FRACTIONAL_BITS = 8,
    parameter bit ACCUMULATE_SCNN_SAMPLE_LOGITS = 1'b0
);
    localparam int CONV_BASE = 0;
    localparam int CLASSIFIER_BASE = 24;
    localparam int FEATURES = 8;
    localparam int CLASSES = 3;
    localparam logic [19:0] CONV_TAG = 20'd7;
    localparam logic [19:0] CLASS_TAG = 20'hc1a53;
    localparam logic [51:0] FC_PRE_TAG = 52'hfc00000000001;
    localparam logic [51:0] FC_OUT_TAG = 52'hfc00000000002;
    localparam logic [51:0] SRNN_TAG = 52'h5a00000000002;
    localparam logic [51:0] FC_POST_TAG = 52'hfc00000000003;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic clear_error = 1'b0;
    logic parameter_load_valid = 1'b0;
    logic parameter_load_ready;
    logic [7:0] parameter_load_addr = '0;
    logic signed [15:0] parameter_load_code = '0;
    logic [2:0] parameter_owner = 3'b001;
    logic scnn_store_load_valid;
    logic [7:0] scnn_store_load_addr;
    logic signed [15:0] scnn_store_load_code;
    logic signed [15:0] scnn_store_load_residual;
    logic scnn_store_learning_read_valid;
    logic [7:0] scnn_store_learning_read_addr;
    logic [15:0] scnn_store_learning_read_tag;
    logic scnn_store_learning_response_ready;
    logic scnn_store_learning_write_valid;
    logic [7:0] scnn_store_learning_write_addr;
    logic signed [15:0] scnn_store_learning_write_code;
    logic signed [15:0] scnn_store_learning_write_residual;
    logic scnn_store_learning_write_code_enable;
    logic scnn_store_learning_write_residual_enable;
    logic scnn_store_forward_read_valid;
    logic [7:0] scnn_store_forward_read_addr;
    logic [15:0] scnn_store_forward_read_tag;
    logic scnn_store_forward_response_ready;
    logic [2:0] shared_load_ready;
    logic [2:0] shared_learning_read_ready;
    logic [2:0] shared_learning_response_valid;
    logic [47:0] shared_learning_response_tag;
    logic signed [47:0] shared_learning_response_code;
    logic signed [47:0] shared_learning_response_residual;
    logic [2:0] shared_learning_response_error;
    logic [2:0] shared_learning_write_ready;
    logic [2:0] shared_forward_read_ready;
    logic [2:0] shared_forward_response_valid;
    logic [47:0] shared_forward_response_tag;
    logic signed [47:0] shared_forward_response_code;
    logic [2:0] shared_forward_response_error;
    logic shared_backend_quiescent;
    logic shared_backend_error;
    logic initialize_valid = 1'b0;
    logic initialize_ready;
    logic initialize_done;
    logic request_valid = 1'b0;
    logic request_ready;
    logic learning_enable = 1'b1;
    logic done;

    logic fc_refill_valid = 1'b0;
    logic fc_refill_ready;
    logic fc_refill_done;
    logic [31:0] fc_rows_published;
    logic fc_cache_read_allow = 1'b0;
    logic fc_cache_read_valid;
    logic fc_cache_read_ready;
    logic [7:0] fc_cache_read_addr;
    logic [15:0] fc_cache_read_tag;
    logic fc_cache_response_valid = 1'b0;
    logic fc_cache_response_ready;
    logic [15:0] fc_cache_response_tag = '0;

    logic fc_srnn_layer_valid = 1'b0;
    logic fc_srnn_layer_ready;
    logic [1:0] fc_srnn_mode = '0;
    logic [15:0] fc_srnn_neuron_count = 16'd480;
    logic [15:0] fc_srnn_feedforward_group_count = 16'd1;
    logic [15:0] fc_srnn_recurrent_group_count = '0;
    logic fc_srnn_recurrent_enable = 1'b0;
    logic [15:0] fc_srnn_leak = '0;
    logic [31:0] fc_srnn_threshold = 32'd1;
    logic fc_srnn_reset_to_zero = 1'b1;
    logic [15:0] fc_srnn_trace_decay = '0;
    logic [51:0] fc_srnn_transaction_tag = '0;
    logic fc_srnn_group_valid = 1'b0;
    logic fc_srnn_group_ready;
    logic fc_srnn_group_recurrent = 1'b0;
    logic [3:0] fc_srnn_group_source_valid = 4'b1111;
    logic [10:0] fc_srnn_group_weight_addr0 = 11'd0;
    logic [10:0] fc_srnn_group_weight_addr1 = 11'd1;
    logic [10:0] fc_srnn_group_weight_addr2 = 11'd2;
    logic [10:0] fc_srnn_group_weight_addr3 = 11'd3;
    logic srnn_group_streamer_select = 1'b0;
    logic srnn_group_streamer_start_valid = 1'b0;
    logic srnn_group_streamer_start_ready;
    logic [9:0] srnn_group_feedforward_source_count = 10'd700;
    logic [9:0] srnn_group_recurrent_source_count = 10'd450;
    logic [9:0] srnn_group_feedforward_cache_row_base = 10'd0;
    logic [9:0] srnn_group_recurrent_cache_row_base = 10'd350;
    logic [699:0] srnn_group_feedforward_spike_flat = '0;
    logic [479:0] srnn_group_recurrent_spike_flat = '0;
    logic srnn_group_streamer_done;
    logic srnn_group_streamer_busy;
    logic srnn_group_streamer_error;
    logic [3:0] srnn_group_streamer_error_code;
    logic fc_srnn_layer_done;
    logic fc_srnn_busy;
    logic fc_srnn_idle;
    logic fc_srnn_error;
    logic [7:0] fc_srnn_error_code;
    logic fc_srnn_observation_valid;
    logic [1:0] fc_srnn_observation_mode;
    logic [51:0] fc_srnn_observation_tag;
    logic [9:0] fc_srnn_observation_neuron;
    logic fc_srnn_observation_spike;
    logic signed [31:0] fc_srnn_observation_pre_state;
    logic signed [31:0] fc_srnn_observation_trace_next;
    logic fc_srnn_observation_last_in_record;
    logic fc_srnn_observation_busy;
    logic fc_srnn_observation_error;
    logic fc_srnn_learning_cfg_valid = 1'b0;
    logic fc_srnn_learning_cfg_ready;
    logic [1:0] fc_srnn_learning_cfg_mode = '0;
    logic [51:0] fc_srnn_learning_cfg_tag = '0;
    logic [5:0] fc_srnn_learning_sample_id = '0;
    logic [9:0] fc_srnn_learning_neuron_count = '0;
    logic fc_srnn_learning_fc_output_layer = 1'b0;
    logic [3:0] fc_srnn_learning_fc_label = '0;
    logic fc_student_trace_valid;
    logic [5:0] fc_student_trace_sample;
    logic [9:0] fc_student_trace_hidden;
    logic [16:0] fc_student_trace_q016;
    logic fc_surrogate_valid;
    logic [5:0] fc_surrogate_sample;
    logic [9:0] fc_surrogate_hidden;
    logic [16:0] fc_surrogate_q016;
    logic fc_logit_row_valid;
    logic fc_logit_row_ready;
    logic [5:0] fc_logit_sample;
    logic [3:0] fc_logit_label;
    logic fc_logit_valid;
    logic fc_logit_ready;
    logic signed [15:0] fc_logit_q6;
    logic fc_logit_last;
    logic srnn_backbone_valid;
    logic srnn_backbone_path;
    logic [5:0] srnn_backbone_sample_id;
    logic [9:0] srnn_backbone_id;
    logic signed [31:0] srnn_backbone_q14;
    logic srnn_backbone_last;
    logic fc_srnn_learning_capture_done;
    logic fc_srnn_learning_done;
    logic fc_srnn_learning_busy;
    logic fc_srnn_learning_idle;
    logic fc_srnn_learning_error;
    logic fc_srnn_learning_error_source;
    logic [3:0] fc_srnn_learning_error_code;

    logic feature_request_valid;
    logic feature_request_ready = 1'b1;
    logic [3:0] feature_term_valid;
    logic [31:0] feature_spatial_base;
    logic [5:0] feature_spatial_count;
    logic feature_response_valid = 1'b0;
    logic feature_response_ready;
    logic [3:0] response_term_q = '0;
    logic [39:0] response_feature_mask_q = '0;
    logic [39:0] feature_mask0;
    logic [39:0] feature_mask1;
    logic [39:0] feature_mask2;
    logic [39:0] feature_mask3;
    logic feature_update_valid;
    logic feature_update_done = 1'b0;

    logic [3:0] maintenance_tile = '0;
    logic state_write_valid = 1'b0;
    logic state_write_ready;
    logic [9:0] state_write_addr = '0;
    logic [319:0] state_write_data = '0;
    logic [39:0] state_write_mask = '0;
    logic trace_write_valid = 1'b0;
    logic trace_write_ready;
    logic [9:0] trace_write_addr = '0;
    logic [319:0] trace_write_data = '0;
    logic [39:0] trace_write_mask = '0;

    logic classifier_cache_valid;
    logic active_classifier_pair;
    logic [15:0] parameter_version;
    logic conv_cache_valid;
    logic [15:0] conv_cached_version;
    logic conv_cache_invalidate;
    logic [15:0] model_version;
    logic [15:0] adapt_version;
    logic commit_pulse;
    logic initialized;
    logic versions_aligned;
    logic [1:0] array_active_owner;
    logic [15:0] array_tile_idle;
    logic [15:0] array_tile_error;
    logic busy;
    logic quiescent;
    logic sticky_error;

    integer done_count = 0;
    integer graph_count = 0;
    integer target_graph_count = 0;
    integer student_graph_count = 0;
    integer feature_request_count = 0;
    integer context_load_count = 0;
    integer weight_fill_count = 0;
    integer conv_write_count = 0;
    integer classifier_write_count = 0;
    integer commit_count = 0;
    integer logit_count = 0;
    integer feature_wave_count = 0;
    integer surrogate_count = 0;
    integer fc_observation_count = 0;
    integer srnn_observation_count = 0;
    integer fc_pre_tag_count = 0;
    integer fc_out_tag_count = 0;
    integer fc_post_tag_count = 0;
    integer srnn_tag_count = 0;
    integer dense_metadata_error_count = 0;
    integer fc_student_trace_count = 0;
    integer fc_surrogate_count = 0;
    integer fc_logit_row_count = 0;
    integer fc_logit_token_count = 0;
    integer srnn_backbone_count = 0;
    integer fc_srnn_learning_done_count = 0;
    integer progress_stage = 0;
    integer watchdog = 0;
    integer initialize_done_count = 0;
    integer runtime_context_count = 0;
    integer classifier_prime_count = 0;
    integer foreign_fill_count = 0;
    integer scnn_classifier_fill_count = 0;
    logic check_warm_return = 1'b0;
    logic check_learning_refill_block = 1'b0;
    integer learning_refill_block_cycles = 0;
    integer publication_refill_block_cycles = 0;
    logic saw_conv_invalidate = 1'b0;
    logic saw_array_error_report = 1'b0;
    logic saw_child_error_report = 1'b0;
    logic saw_subsystem_error_report = 1'b0;
    logic signed [31:0] fc_output_pre_state [0:15];
    logic signed [15:0] learned_logits [0:CLASSES-1];
    logic signed [15:0] inference_logits [0:CLASSES-1];
    logic signed [15:0] warm_return_logits [0:CLASSES-1];
    logic signed [15:0] saved_parameter_code [0:255];
    logic signed [15:0] saved_parameter_residual [0:255];

    always #5 clk = ~clk;

    function automatic bit desired_conv_spike(input int spatial);
        int row;
        int col;
        int pool_row;
        int pool_col;
        int local_row;
        int local_col;
        int desired_count;
        begin
            row = spatial / 4;
            col = spatial % 4;
            pool_row = row / 2;
            pool_col = col / 2;
            local_row = row % 2;
            local_col = col % 2;
            desired_count = pool_row * 2 + pool_col;
            desired_conv_spike = local_row * 2 + local_col < desired_count;
        end
    endfunction

    assign feature_mask0 = response_term_q[0] ? response_feature_mask_q : '0;
    assign feature_mask1 = response_term_q[1] ? response_feature_mask_q : '0;
    assign feature_mask2 = response_term_q[2] ? response_feature_mask_q : '0;
    assign feature_mask3 = response_term_q[3] ? response_feature_mask_q : '0;

    // Four scalar FC rows overwrite both SCNN bank pairs through the real
    // publisher and physical fill fabric, without touching the SCNN store.
    assign fc_cache_read_ready = fc_cache_read_allow && !fc_cache_response_valid;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fc_cache_response_valid <= 1'b0;
            fc_cache_response_tag <= '0;
        end else begin
            if (fc_cache_read_valid && fc_cache_read_ready) begin
                if (fc_cache_read_addr < 8'd128 || fc_cache_read_addr > 8'd131)
                    $fatal(1, "foreign FC refill read outside directed matrix");
                fc_cache_response_valid <= 1'b1;
                fc_cache_response_tag <= fc_cache_read_tag;
            end
            if (fc_cache_response_valid && fc_cache_response_ready)
                fc_cache_response_valid <= 1'b0;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            feature_response_valid <= 1'b0;
            response_term_q <= '0;
            response_feature_mask_q <= '0;
            feature_update_done <= 1'b0;
        end else begin
            feature_update_done <= 1'b0;
            if (feature_request_valid && feature_request_ready) begin
                feature_response_valid <= 1'b1;
                response_term_q <= feature_term_valid;
                response_feature_mask_q <= '0;
                for (int lane = 0; lane < 40; lane++)
                    if (lane < int'(feature_spatial_count))
                        response_feature_mask_q[lane] <= desired_conv_spike(
                            int'(feature_spatial_base) + lane);
            end
            if (feature_response_valid && feature_response_ready)
                feature_response_valid <= 1'b0;
            if (feature_update_valid)
                feature_update_done <= 1'b1;
        end
    end

    always_ff @(posedge clk) begin
        if (check_learning_refill_block) begin
            learning_refill_block_cycles <= learning_refill_block_cycles + 1;
            if (dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.commit_inflight_q)
                publication_refill_block_cycles <= publication_refill_block_cycles + 1;
            if (!fc_refill_valid || fc_refill_ready || dut.u_system.fc_refill_busy ||
                    fc_cache_read_valid || fc_refill_done || fc_rows_published != 0)
                $fatal(1, "foreign FC refill admitted during SCNN learning/publication");
            if ((|(dut.u_system.weight_fill_valid & dut.u_system.weight_fill_ready)) &&
                    !(|(dut.u_system.scnn_weight_fill_valid & dut.u_system.scnn_weight_fill_ready)))
                $fatal(1, "foreign physical fill accepted during SCNN transaction");
        end
        if (initialize_done)
            initialize_done_count <= initialize_done_count + 1;
        if (dut.u_system.g_scnn_client.u_train_infer.runtime_context_valid &&
                dut.u_system.g_scnn_client.u_train_infer.runtime_context_ready)
            runtime_context_count <= runtime_context_count + 1;
        if (dut.u_system.g_scnn_client.u_train_infer.plane_prime_done)
            classifier_prime_count <= classifier_prime_count + 1;
        if (check_warm_return) begin
            if (!initialized || initialize_done ||
                    dut.u_system.g_scnn_client.u_train_infer.runtime_context_valid ||
                    model_version != 16'd6 || parameter_version != 16'd6 ||
                    adapt_version != 16'd8 || !versions_aligned ||
                    commit_pulse || scnn_store_load_valid ||
                    scnn_store_learning_write_valid)
                $fatal(1, "warm return reinitialized context or mutated model/store");
            if (dut.u_system.fc_refill_busy &&
                    (request_ready || dut.u_system.g_scnn_client.u_train_infer.plane_prime_valid ||
                     dut.u_system.scnn_weight_fill_busy))
                $fatal(1, "SCNN return admitted during foreign refill");
            if (!classifier_cache_valid && request_ready)
                $fatal(1, "SCNN return admitted before classifier residency");
        end
        for (int region = 0; region < 4; region++) begin
            if (check_warm_return && dut.u_system.fc_refill_busy &&
                    dut.u_system.weight_fill_valid[region] && dut.u_system.weight_fill_ready[region]) begin
                if (!dut.u_system.scnn_cache_evicted)
                    $fatal(1, "accepted foreign physical write did not evict SCNN");
                foreign_fill_count <= foreign_fill_count + 1;
            end
            if (dut.u_system.scnn_weight_fill_valid[region] &&
                    dut.u_system.scnn_weight_fill_ready[region] &&
                    dut.u_system.scnn_weight_fill_addr_flat[region*11 + 10]) begin
                if (dut.u_system.scnn_cache_evicted)
                    $fatal(1, "SCNN classifier refill evicted itself");
                scnn_classifier_fill_count <= scnn_classifier_fill_count + 1;
            end
        end
        if (done)
            done_count <= done_count + 1;
        if (dut.u_system.graph_start_valid && dut.u_system.graph_start_ready) begin
            graph_count <= graph_count + 1;
            if (dut.u_system.graph_target)
                target_graph_count <= target_graph_count + 1;
            else
                student_graph_count <= student_graph_count + 1;
        end
        if (feature_request_valid && feature_request_ready)
            feature_request_count <= feature_request_count + 1;
        if (dut.u_system.array_context_load_valid && dut.u_system.array_context_load_ready)
            context_load_count <= context_load_count + 1;
        for (int region = 0; region < 4; region++)
            if (dut.u_system.weight_fill_valid[region] && dut.u_system.weight_fill_ready[region])
                weight_fill_count <= weight_fill_count + 1;
        if (scnn_store_learning_write_valid && shared_learning_write_ready[2]) begin
            if (scnn_store_learning_write_addr >= CLASSIFIER_BASE &&
                    scnn_store_learning_write_addr <
                        CLASSIFIER_BASE + FEATURES * CLASSES)
                classifier_write_count <= classifier_write_count + 1;
            else if (scnn_store_learning_write_addr < CLASSIFIER_BASE)
                conv_write_count <= conv_write_count + 1;
            else
                $fatal(1, "shared output learner wrote outside the selected SCNN banks: addr=%0d",
                    scnn_store_learning_write_addr);
            if (!scnn_store_learning_write_code_enable &&
                    !scnn_store_learning_write_residual_enable)
                $fatal(1, "shared output learner issued a no-op write");
        end
        if (commit_pulse)
            commit_count <= commit_count + 1;
        if (conv_cache_invalidate)
            saw_conv_invalidate <= 1'b1;
        if (dut.u_system.feature_wave_valid && dut.u_system.feature_wave_ready)
            feature_wave_count <= feature_wave_count + 1;
        if (dut.u_system.surrogate_valid && dut.u_system.surrogate_ready)
            surrogate_count <= surrogate_count + 1;
        if (dut.u_system.logit_wave_valid && dut.u_system.logit_wave_ready) begin
            if (dut.u_system.logit_wave_mask != 12'h007 || dut.u_system.logit_wave_count != 3)
                $fatal(1, "classifier logit envelope mismatch");
            // The target graph also executes the shared classifier tail so it
            // can export its pooled feature trace. Compare only student logits
            // before and after the atomic learning publication.
            if (!dut.u_system.observation_target) begin
                for (int class_id = 0; class_id < CLASSES; class_id++) begin
                    if (logit_count == 0)
                        learned_logits[class_id] <= $signed(
                            dut.u_system.logit_wave_flat[class_id*16 +: 16]);
                    if (logit_count == 1)
                        inference_logits[class_id] <= $signed(
                            dut.u_system.logit_wave_flat[class_id*16 +: 16]);
                    if (logit_count == 2)
                        warm_return_logits[class_id] <= $signed(
                            dut.u_system.logit_wave_flat[class_id*16 +: 16]);
                end
                logit_count <= logit_count + 1;
            end
        end
        if (fc_srnn_observation_valid &&
                dut.u_system.fc_srnn_learning_scalar_ready) begin
            if (fc_srnn_observation_neuron >= 10'd480)
                dense_metadata_error_count <= dense_metadata_error_count + 1;
            case (fc_srnn_observation_mode)
                2'd0: begin
                    fc_observation_count <= fc_observation_count + 1;
                    if (fc_srnn_observation_tag == FC_PRE_TAG)
                        fc_pre_tag_count <= fc_pre_tag_count + 1;
                    else if (fc_srnn_observation_tag == FC_OUT_TAG) begin
                        fc_out_tag_count <= fc_out_tag_count + 1;
                        if (fc_srnn_observation_neuron < 10'd16)
                            fc_output_pre_state[
                                fc_srnn_observation_neuron[3:0]] <=
                                    fc_srnn_observation_pre_state;
                        else
                            dense_metadata_error_count <=
                                dense_metadata_error_count + 1;
                    end
                    else if (fc_srnn_observation_tag == FC_POST_TAG)
                        fc_post_tag_count <= fc_post_tag_count + 1;
                    else
                        dense_metadata_error_count <=
                            dense_metadata_error_count + 1;
                end
                2'd1: begin
                    srnn_observation_count <= srnn_observation_count + 1;
                    if (fc_srnn_observation_tag == SRNN_TAG)
                        srnn_tag_count <= srnn_tag_count + 1;
                    else
                        dense_metadata_error_count <=
                            dense_metadata_error_count + 1;
                end
                default: dense_metadata_error_count <=
                    dense_metadata_error_count + 1;
            endcase
        end
        if (fc_student_trace_valid) begin
            if (fc_student_trace_sample != 6'd3 ||
                    fc_student_trace_hidden != 10'(
                        fc_student_trace_count % 400))
                dense_metadata_error_count <= dense_metadata_error_count + 1;
            fc_student_trace_count <= fc_student_trace_count + 1;
        end
        if (fc_surrogate_valid) begin
            if (fc_surrogate_sample != 6'd3 ||
                    fc_surrogate_hidden != 10'(fc_surrogate_count % 400))
                dense_metadata_error_count <= dense_metadata_error_count + 1;
            fc_surrogate_count <= fc_surrogate_count + 1;
        end
        if (fc_logit_row_valid && fc_logit_row_ready) begin
            if (fc_logit_sample != 6'd3 || fc_logit_label != 4'd4 ||
                    fc_logit_row_count != 0)
                dense_metadata_error_count <= dense_metadata_error_count + 1;
            fc_logit_row_count <= fc_logit_row_count + 1;
        end
        if (fc_logit_valid && fc_logit_ready) begin
            if (fc_logit_sample != 6'd3 || fc_logit_token_count >= 10 ||
                    fc_logit_q6 != 16'($signed(
                        fc_output_pre_state[fc_logit_token_count]) >>> 4) ||
                    fc_logit_last != (fc_logit_token_count == 9))
                dense_metadata_error_count <= dense_metadata_error_count + 1;
            fc_logit_token_count <= fc_logit_token_count + 1;
        end
        if (srnn_backbone_valid) begin
            if (!srnn_backbone_path || srnn_backbone_sample_id != 6'd7 ||
                    srnn_backbone_id != 10'(srnn_backbone_count) ||
                    (srnn_backbone_q14 != 32'sd0 &&
                     srnn_backbone_q14 != 32'sd16384) ||
                    srnn_backbone_last != (srnn_backbone_id == 10'd449))
                dense_metadata_error_count <= dense_metadata_error_count + 1;
            srnn_backbone_count <= srnn_backbone_count + 1;
        end
        if (fc_srnn_learning_done)
            fc_srnn_learning_done_count <= fc_srnn_learning_done_count + 1;
    end

    if (!CLASSIFIER_ONLY) begin : g_dual_mode_diagnostics
    always_ff @(posedge clk) begin
        if (rst_n &&
                !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.local_error_q &&
                (dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.i_active_target &&
                     dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.i_active_student ||
                 dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.i_target_trace_cfg_valid &&
                     !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.phase_target ||
                 dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.i_prescan_valid &&
                     !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.prescan_metadata_legal ||
                 dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.i_surrogate_valid &&
                     !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.surrogate_metadata_legal ||
                 dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.i_feature_wave_valid &&
                     !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.phase_target &&
                     !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.phase_student ||
                 dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.i_logit_wave_valid &&
                     !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.phase_target &&
                     !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.phase_student))
            $display("router local trigger both/cfg/prescan/surr/feature/logit=%0b/%0b/%0b/%0b/%0b/%0b phase=%0b/%0b graph(done/startv/startr)=%0b/%0b/%0b coord_state=%0d prep_state=%0d",
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.i_active_target &&
                    dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.i_active_student,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.i_target_trace_cfg_valid &&
                    !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.phase_target,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.i_prescan_valid &&
                    !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.prescan_metadata_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.i_surrogate_valid &&
                    !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.surrogate_metadata_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.i_feature_wave_valid &&
                    !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.phase_target &&
                    !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.phase_student,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.i_logit_wave_valid &&
                    !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.phase_target &&
                    !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.phase_student,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.phase_target,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.phase_student,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.i_graph_done,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.o_graph_start_valid,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.i_graph_start_ready,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_batch.state_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_prep.state_q);
        if (rst_n && dut.u_system.subsystem_error &&
                !saw_subsystem_error_report) begin
            saw_subsystem_error_report <= 1'b1;
            $display("subsystem first error runtime/plane=%0b/%0b graph(frontend/runtime)=%0b/%0b plane(backend/refill/overlap/invalid)=%0b/%0b/%0b/%0b stage=%0d",
                dut.u_system.g_scnn_client.u_train_infer.runtime_error,
                dut.u_system.g_scnn_client.u_train_infer.plane_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.frontend_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.runtime_error,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.backend_error,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.refill_error,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.fill_overlap,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.invalid_context_request,
                progress_stage);
            $display("frontend parts coord/prep/prescan/router=%0b/%0b/%0b/%0b router(local/target/binding)=%0b/%0b/%0b phase(target/student)=%0b/%0b enables(conv/classifier)=%0b/%0b streams(prescan/surr/feature/logit)=%0b/%0b/%0b/%0b",
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.coord_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.prep_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.prescan_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.router_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.local_error_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.target_trace_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.binding_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.phase_target,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_router.phase_student,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.o_conv_observation_enable,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.o_classifier_observation_enable,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.prescan_observation_valid,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.i_surrogate_valid,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.i_feature_wave_valid,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.i_logit_wave_valid);
            $display("plane context valid/ready/load/idle/start/common/resident=%0b/%0b/%0b/%0b/%0b/%0b/%0b cfg(base/count/in/pair/ver)=%0d/%0d/%0d/%0b/%0d cache(valid/base/count/in/pair/ver)=%0b/%0d/%0d/%0d/%0b/%0d classifier(valid/pair/ver)=%0b/%0b/%0d",
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.i_conv_context_valid,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.o_conv_context_ready,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.i_conv_context_load_weights,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.i_array_idle,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.context_start_allowed,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.context_common_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.resident_match,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.i_conv_output_channel_base,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.i_conv_output_channel_count,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.i_conv_input_channel_count,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.i_conv_context_pair,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.i_conv_context_version,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.conv_cache_valid_q,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.conv_output_base_q,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.conv_output_count_q,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.conv_input_count_q,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.conv_pair_q,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.conv_version_q,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.o_classifier_cache_valid,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.o_active_classifier_pair,
                dut.u_system.g_scnn_client.u_train_infer.u_parameter_plane.o_active_version);
        end
        if (rst_n && dut.u_system.array_error && !saw_array_error_report) begin
            saw_array_error_report <= 1'b1;
            $display("array first error path/walker/tiles/owner/scnn=%0b/%0b/%h/%0b/%0b command(fire/legal/first/last/mode/count/spatial/ocbase/pair/work/tag)=%0b/%0b/%0b/%0b/%0d/%0d/%0d/%0d/%0b/%0d/%h tx(open/mode/count/spatial/ocbase/pair/work/tag)=%0b/%0d/%0d/%0d/%0d/%0b/%0d/%h",
                dut.u_system.u_array.u_array.path_sticky_error_q,
                dut.u_system.u_array.u_array.walker_sticky_error,
                array_tile_error,
                dut.u_system.u_array.owner_error,
                dut.u_system.u_array.scnn_client_error,
                dut.u_system.u_array.u_array.command_fire,
                dut.u_system.u_array.u_array.command_context_legal,
                dut.u_system.u_array.array_command_first,
                dut.u_system.u_array.array_command_last,
                dut.u_system.u_array.array_command_mode,
                dut.u_system.u_array.array_command_neuron_count,
                dut.u_system.u_array.array_command_spatial_count,
                dut.u_system.u_array.array_command_output_channel_base,
                dut.u_system.u_array.array_command_active_pair,
                dut.u_system.u_array.array_command_work_base_addr,
                dut.u_system.u_array.array_command_transaction_tag,
                dut.u_system.u_array.u_array.transaction_open_q,
                dut.u_system.u_array.u_array.transaction_mode_q,
                dut.u_system.u_array.u_array.transaction_neuron_count_q,
                dut.u_system.u_array.u_array.transaction_scnn_spatial_count_q,
                dut.u_system.u_array.u_array.transaction_scnn_output_channel_base_q,
                dut.u_system.u_array.u_array.transaction_scnn_active_pair_q,
                dut.u_system.u_array.u_array.transaction_scnn_work_base_addr_q,
                dut.u_system.u_array.u_array.transaction_tag_q);
            $display("graph client state/local/conv/serializer/fork/classifier/feature/array=%0d/%0b/%0b/%0b/%0b/%0b/%0b/%0b conv(busy/done/idle/error)=%0b/%0b/%0b/%0b classifier(busy/done/idle/error)=%0b/%0b/%0b/%0b",
                dut.u_system.u_array.g_scnn_graph.u_scnn_graph_client.state_q,
                dut.u_system.u_array.g_scnn_graph.u_scnn_graph_client.local_error_q,
                dut.u_system.u_array.g_scnn_graph.u_scnn_graph_client.conv_error,
                dut.u_system.u_array.g_scnn_graph.u_scnn_graph_client.serializer_error,
                dut.u_system.u_array.g_scnn_graph.u_scnn_graph_client.fork_error,
                dut.u_system.u_array.g_scnn_graph.u_scnn_graph_client.classifier_error,
                dut.u_system.u_array.i_scnn_feature_service_error,
                dut.u_system.u_array.array_error,
                dut.u_system.u_array.g_scnn_graph.u_scnn_graph_client.conv_busy,
                dut.u_system.u_array.g_scnn_graph.u_scnn_graph_client.conv_done,
                dut.u_system.u_array.g_scnn_graph.u_scnn_graph_client.conv_idle,
                dut.u_system.u_array.g_scnn_graph.u_scnn_graph_client.conv_error,
                dut.u_system.u_array.g_scnn_graph.u_scnn_graph_client.classifier_busy,
                dut.u_system.u_array.g_scnn_graph.u_scnn_graph_client.classifier_done,
                dut.u_system.u_array.g_scnn_graph.u_scnn_graph_client.classifier_idle,
                dut.u_system.u_array.g_scnn_graph.u_scnn_graph_client.classifier_error);
        end
        if (rst_n && !saw_child_error_report &&
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.child_error) begin
            saw_child_error_report <= 1'b1;
            $display("child_error pulse cv(cfg/proto)=%0b/%0b cl(cfg/obs/context/grad/param/proto)=%0b/%0b/%0b/%0b/%0b/%0b arb/version/mod=%0b/%0b/%0b cv_parts(store/ctl/lr_cfg/lr_proto/lr_latched)=%0b/%0b/%0b/%0b/%0b",
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.cv_o_cfg_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.cv_o_protocol_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.cl_o_cfg_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.cl_o_observation_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.cl_o_context_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.cl_o_gradient_overflow,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.cl_o_parameter_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.cl_o_protocol_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.arb_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.observation_version_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.internal_mod_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.st_o_protocol_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.ctl_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.lr_o_cfg_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.lr_o_protocol_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.learner_cfg_error_q);
            $display("conv store cfg(fire/valid/ready/legal/active batch/time/ver/in/out)=%0b/%0b/%0b/%0b/%0b %0d/%0d/%h/%0d/%0d streams input(v/r/legal/order tgt/s/t/v/c/p)=%0b/%0b/%0b/%0b %0b/%0d/%0d/%h/%0d/%0d trace=%0b/%0b/%0b/%0b %0b/%0d/%0d/%h/%0d/%0d surrogate=%0b/%0b/%0b/%0b %0b/%0d/%0d/%h/%0d/%0d counts(i/t/s student,target)=%0d,%0d/%0d,%0d/%0d,%0d done(i/t/s student,target)=%0b,%0b/%0b,%0b/%0b,%0b",
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.cfg_fire,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_cfg_valid,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.st_o_cfg_ready,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.cfg_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.active_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.batch_count_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.timestep_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.sample_version_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.input_channels_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.output_channels_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_input_valid,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.st_o_input_ready,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.input_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.input_order_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_input_target,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_input_sample,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_input_timestep,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_input_sample_version,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_input_channel,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_input_spatial,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_trace_valid,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.st_o_trace_ready,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.trace_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.trace_order_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_trace_target,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_trace_sample,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_trace_timestep,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_trace_sample_version,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_trace_channel,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_trace_spatial,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_surrogate_valid,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.st_o_surrogate_ready,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.surrogate_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.surrogate_order_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_surrogate_target,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_surrogate_sample,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_surrogate_timestep,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_surrogate_sample_version,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_surrogate_channel,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.i_surrogate_spatial,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.student_input_count_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.target_input_count_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.student_trace_count_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.target_trace_count_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.student_surrogate_count_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.target_surrogate_count_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.input_done_q[0],
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.input_done_q[1],
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.trace_done[0],
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.trace_done[1],
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.surrogate_done[0],
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.surrogate_done[1]);
        end
        if (rst_n &&
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.cl_o_cfg_error)
            $display("classifier cfg pulse replay(err/legal)=%0b/%0b learner_err=%0b cfg(batch/tbase/tcount/features/classes/base/inc/thr/min/max)=%0d/%0d/%0d/%0d/%0d/%0d/%0d/%0d/%0d/%0d",
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.u_cl.replay_cfg_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.u_cl.u_observation_replay.cfg_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.u_cl.learner_cfg_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.u_cl.i_cfg_batch_count,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.u_cl.i_cfg_timestep_base,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.u_cl.i_cfg_timestep_count,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.u_cl.i_cfg_feature_count,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.u_cl.i_cfg_class_count,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.u_cl.i_cfg_parameter_base,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.u_cl.i_cfg_vote_increment,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.u_cl.i_cfg_vote_threshold,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.u_cl.i_cfg_code_min,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.u_cl.i_cfg_code_max);
    end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            watchdog <= 0;
        end else begin
            watchdog <= watchdog + 1;
            if (watchdog == 800000)
                $fatal(1, "system timeout stage=%0d owner=%0d busy=%0b init=%0b graph=%0d writes=%0d/%0d fills=%0d error=%0b sub=%0b array=%0b bridge=%0b arb=%0b",
                    progress_stage, array_active_owner, busy, initialized,
                    graph_count, conv_write_count, classifier_write_count,
                    weight_fill_count, sticky_error, dut.u_system.subsystem_error,
                    dut.u_system.array_error, dut.u_system.context_bridge_error,
                    dut.u_system.feature_arbiter_error);
        end
    end

    task automatic tick;
        @(posedge clk);
        @(negedge clk);
    endtask

    task automatic load_parameter(input int addr, input int value);
        @(negedge clk);
        parameter_load_addr = 8'(addr);
        parameter_load_code = 16'(value);
        parameter_load_valid = 1'b1;
        while (!parameter_load_ready) @(negedge clk);
        tick();
        parameter_load_valid = 1'b0;
    endtask

    task automatic clear_work(input int tile, input int work_base);
        @(negedge clk);
        maintenance_tile = 4'(tile);
        state_write_addr = {1'b0, 9'(work_base)};
        state_write_data = '0;
        state_write_mask = '1;
        trace_write_addr = {1'b1, 9'(work_base)};
        trace_write_data = '0;
        trace_write_mask = '1;
        state_write_valid = 1'b1;
        trace_write_valid = 1'b1;
        while (!state_write_ready || !trace_write_ready) @(negedge clk);
        tick();
        state_write_valid = 1'b0;
        trace_write_valid = 1'b0;
    endtask

    task automatic initialize_system;
        @(negedge clk);
        initialize_valid = 1'b1;
        while (!initialize_ready) @(negedge clk);
        tick();
        initialize_valid = 1'b0;
        while (!initialize_done) tick();
    endtask

    task automatic push_dense_group(input logic recurrent_group);
        int local_wait;
        logic accepted;
        begin
            @(negedge clk);
            fc_srnn_group_recurrent = recurrent_group;
            fc_srnn_group_valid = 1'b1;
            local_wait = 0;
            accepted = 1'b0;
            while (!accepted && local_wait < 20000) begin
                @(posedge clk);
                accepted = fc_srnn_group_valid && fc_srnn_group_ready;
                local_wait++;
            end
            if (!accepted)
                $fatal(1, "dense group timeout recurrent=%0b client_state=%0d issued/total=%0d/%0d owner=%0d array(active/open/ready)=%0b/%0b/%0b",
                    recurrent_group,
                    dut.u_system.u_array.g_fc_srnn_layer.u_fc_srnn_client.state_q,
                    dut.u_system.u_array.g_fc_srnn_layer.u_fc_srnn_client.issued_group_count_q,
                    dut.u_system.u_array.g_fc_srnn_layer.u_fc_srnn_client.total_group_count_q,
                    dut.u_system.u_array.u_owner.owner_q,
                    dut.u_system.u_array.u_array.active_q,
                    dut.u_system.u_array.u_array.transaction_open_q,
                    dut.u_system.u_array.array_command_ready);
            @(negedge clk);
            fc_srnn_group_valid = 1'b0;
        end
    endtask

    task automatic configure_dense_learning(
        input logic [1:0] mode,
        input logic [51:0] transaction_tag,
        input int neuron_count,
        input logic fc_output_layer,
        input logic [3:0] fc_label
    );
        int local_wait;
        logic accepted;
        begin
            @(negedge clk);
            fc_srnn_learning_cfg_mode = mode;
            fc_srnn_learning_cfg_tag = transaction_tag;
            fc_srnn_learning_sample_id = mode == 2'd0 ? 6'd3 : 6'd7;
            fc_srnn_learning_neuron_count = 10'(neuron_count);
            fc_srnn_learning_fc_output_layer = fc_output_layer;
            fc_srnn_learning_fc_label = fc_label;
            fc_srnn_learning_cfg_valid = 1'b1;
            local_wait = 0;
            accepted = 1'b0;
            while (!accepted && local_wait < 1000) begin
                @(posedge clk);
                accepted = fc_srnn_learning_cfg_valid &&
                    fc_srnn_learning_cfg_ready;
                local_wait++;
            end
            if (!accepted)
                $fatal(1, "dense learning ingress config timeout mode=%0d busy=%0b error=%0b/%0b/%h",
                    mode, fc_srnn_learning_busy, fc_srnn_learning_error,
                    fc_srnn_learning_error_source,
                    fc_srnn_learning_error_code);
            @(negedge clk);
            fc_srnn_learning_cfg_valid = 1'b0;
        end
    endtask

    task automatic run_dense_layer(
        input logic [1:0] mode,
        input logic recurrent_enable,
        input logic [51:0] transaction_tag,
        input int neuron_count,
        input logic fc_output_layer,
        input logic [3:0] fc_label
    );
        int expected_observations;
        int expected_fc_streams;
        int expected_srnn_stream;
        int expected_fc_logit_rows;
        int expected_fc_logit_tokens;
        int expected_learning_done;
        int local_wait;
        begin
            expected_observations = (mode == 2'd0 ?
                fc_observation_count : srnn_observation_count) + neuron_count;
            expected_fc_streams = fc_student_trace_count +
                (mode == 2'd0 && !fc_output_layer ? neuron_count : 0);
            expected_srnn_stream = srnn_backbone_count +
                (mode == 2'd1 ? neuron_count : 0);
            expected_fc_logit_rows = fc_logit_row_count +
                (mode == 2'd0 && fc_output_layer ? 1 : 0);
            expected_fc_logit_tokens = fc_logit_token_count +
                (mode == 2'd0 && fc_output_layer ? neuron_count : 0);
            expected_learning_done = fc_srnn_learning_done_count + 1;
            configure_dense_learning(mode, transaction_tag, neuron_count,
                fc_output_layer, fc_label);
            @(negedge clk);
            fc_srnn_mode = mode;
            fc_srnn_neuron_count = 16'(neuron_count);
            fc_srnn_recurrent_enable = recurrent_enable;
            fc_srnn_recurrent_group_count = recurrent_enable ? 16'd1 : 16'd0;
            fc_srnn_transaction_tag = transaction_tag;
            fc_srnn_layer_valid = 1'b1;
            local_wait = 0;
            while (!fc_srnn_layer_ready && local_wait < 1000) begin
                @(negedge clk);
                local_wait++;
            end
            if (local_wait == 1000)
                $fatal(1, "dense layer admission timeout mode=%0d owner=%0d busy=%0b",
                    mode, array_active_owner, busy);
            tick();
            fc_srnn_layer_valid = 1'b0;

            push_dense_group(1'b0);
            if (recurrent_enable)
                push_dense_group(1'b1);

            local_wait = 0;
            while (!fc_srnn_layer_done && local_wait < 20000) begin
                tick();
                local_wait++;
            end
            if (local_wait == 20000 ||
                    (mode == 2'd0 ? fc_observation_count :
                        srnn_observation_count) != expected_observations ||
                    fc_student_trace_count != expected_fc_streams ||
                    fc_surrogate_count != expected_fc_streams ||
                    fc_logit_row_count != expected_fc_logit_rows ||
                    fc_logit_token_count != expected_fc_logit_tokens ||
                    srnn_backbone_count != expected_srnn_stream ||
                    fc_srnn_learning_done_count != expected_learning_done ||
                    array_active_owner != 0 || fc_srnn_busy ||
                    fc_srnn_error || fc_srnn_observation_error ||
                    fc_srnn_learning_error || sticky_error) begin
                $display("dense client state=%0d issued/ff/total=%0d/%0d/%0d expected_recurrent=%0b group(v/r/kind)=%0b/%0b/%0b direct(v/r/first/last/done)=%0b/%0b/%0b/%0b/%0b",
                    dut.u_system.u_array.g_fc_srnn_layer.u_fc_srnn_client.state_q,
                    dut.u_system.u_array.g_fc_srnn_layer.u_fc_srnn_client.issued_group_count_q,
                    dut.u_system.u_array.g_fc_srnn_layer.u_fc_srnn_client.feedforward_group_count_q,
                    dut.u_system.u_array.g_fc_srnn_layer.u_fc_srnn_client.total_group_count_q,
                    dut.u_system.u_array.g_fc_srnn_layer.u_fc_srnn_client.expected_recurrent,
                    fc_srnn_group_valid, fc_srnn_group_ready,
                    fc_srnn_group_recurrent, dut.u_system.u_array.direct_command_valid,
                    dut.u_system.u_array.direct_command_ready, dut.u_system.u_array.direct_first,
                    dut.u_system.u_array.direct_last, dut.u_system.u_array.direct_transaction_done);
                $display("dense owner state=%0d select=%0b last_inflight=%0b done_seen=%0b sink_idle=%0b array(v/r/done/idle/active/open/last)=%0b/%0b/%0b/%0b/%0b/%0b/%0b walker(busy/done)=%0b/%0b tiles(done/idle/error)=%h/%h/%h serializer(busy/error)=%0b/%0b",
                    dut.u_system.u_array.u_owner.owner_q, dut.u_system.u_array.select_direct,
                    dut.u_system.u_array.u_owner.direct_last_inflight_q,
                    dut.u_system.u_array.u_owner.direct_done_seen_q,
                    dut.u_system.u_array.fc_srnn_direct_sink_idle,
                    dut.u_system.u_array.array_command_valid,
                    dut.u_system.u_array.array_command_ready, dut.u_system.u_array.array_done,
                    dut.u_system.u_array.array_idle, dut.u_system.u_array.u_array.active_q,
                    dut.u_system.u_array.u_array.transaction_open_q,
                    dut.u_system.u_array.u_array.command_last_q,
                    dut.u_system.u_array.u_array.walker_busy,
                    dut.u_system.u_array.u_array.walker_done,
                    dut.u_system.u_array.o_tile_done, dut.u_system.u_array.o_tile_idle,
                    dut.u_system.u_array.o_tile_sticky_error,
                    dut.u_system.u_array.fc_srnn_observation_busy,
                    dut.u_system.u_array.fc_srnn_observation_error);
                $fatal(1, "dense layer failure mode=%0d output=%0b observations=%0d expected=%0d routed(fc_trace/fc_surr/logit_rows/logit_tokens/srnn/done)=%0d/%0d/%0d/%0d/%0d/%0d expected=%0d/%0d/%0d/%0d/%0d owner=%0d busy=%0b client=%0b error=%0b/%h obs/ingress/system=%0b/%0b/%0b",
                    mode,
                    fc_output_layer,
                    mode == 2'd0 ? fc_observation_count :
                        srnn_observation_count,
                    expected_observations, fc_student_trace_count,
                    fc_surrogate_count, fc_logit_row_count,
                    fc_logit_token_count, srnn_backbone_count,
                    fc_srnn_learning_done_count, expected_fc_streams,
                    expected_fc_logit_rows, expected_fc_logit_tokens,
                    expected_srnn_stream, expected_learning_done,
                    array_active_owner, busy,
                    fc_srnn_busy, fc_srnn_error, fc_srnn_error_code,
                    fc_srnn_observation_error, fc_srnn_learning_error,
                    sticky_error);
            end
        end
    endtask

    task automatic run_streamed_srnn_layer;
        int expected_observations;
        int expected_srnn_stream;
        int expected_learning_done;
        int local_wait;
        logic accepted;
        begin
            expected_observations = srnn_observation_count + 450;
            expected_srnn_stream = srnn_backbone_count + 450;
            expected_learning_done = fc_srnn_learning_done_count + 1;
            configure_dense_learning(2'd1, SRNN_TAG, 450, 1'b0, 4'd0);

            @(negedge clk);
            fc_srnn_mode = 2'd1;
            fc_srnn_neuron_count = 16'd450;
            fc_srnn_recurrent_enable = 1'b1;
            fc_srnn_transaction_tag = SRNN_TAG;
            srnn_group_feedforward_source_count = 10'd700;
            srnn_group_recurrent_source_count = 10'd450;
            srnn_group_feedforward_cache_row_base = 10'd0;
            srnn_group_recurrent_cache_row_base = 10'd350;
            srnn_group_feedforward_spike_flat = '0;
            srnn_group_recurrent_spike_flat = '0;
            srnn_group_streamer_select = 1'b1;
            srnn_group_streamer_start_valid = 1'b1;
            local_wait = 0;
            accepted = 1'b0;
            while (!accepted && local_wait < 1000) begin
                @(posedge clk);
                accepted = srnn_group_streamer_start_valid &&
                    srnn_group_streamer_start_ready;
                local_wait++;
            end
            if (!accepted)
                $fatal(1, "SRNN streamer admission timeout owner=%0d busy=%0b error=%0b/%h",
                    array_active_owner, srnn_group_streamer_busy,
                    srnn_group_streamer_error,
                    srnn_group_streamer_error_code);
            @(negedge clk);
            srnn_group_streamer_start_valid = 1'b0;
            // Prove ownership is retained by the admitted transaction rather
            // than by a profile-select level held for hundreds of groups.
            srnn_group_streamer_select = 1'b0;

            local_wait = 0;
            while (!srnn_group_streamer_done && local_wait < 200000) begin
                tick();
                local_wait++;
            end
            if (local_wait == 200000 ||
                    srnn_observation_count != expected_observations ||
                    srnn_backbone_count != expected_srnn_stream ||
                    fc_srnn_learning_done_count != expected_learning_done ||
                    srnn_group_streamer_busy || srnn_group_streamer_error ||
                    fc_srnn_error || fc_srnn_observation_error ||
                    fc_srnn_learning_error || sticky_error)
                $fatal(1, "streamed SRNN failure wait=%0d observations=%0d/%0d backbone=%0d/%0d learning_done=%0d/%0d streamer=%0b/%0b/%h array=%0b/%h obs/ingress/system=%0b/%0b/%0b",
                    local_wait, srnn_observation_count,
                    expected_observations, srnn_backbone_count,
                    expected_srnn_stream, fc_srnn_learning_done_count,
                    expected_learning_done, srnn_group_streamer_busy,
                    srnn_group_streamer_error,
                    srnn_group_streamer_error_code, fc_srnn_error,
                    fc_srnn_error_code, fc_srnn_observation_error,
                    fc_srnn_learning_error, sticky_error);
        end
    endtask

    task automatic request_batch(input logic enable_learning,
                                 input int expected_done_count);
        int local_wait;
        @(negedge clk);
        learning_enable = enable_learning;
        request_valid = 1'b1;
        while (!request_ready) @(negedge clk);
        tick();
        request_valid = 1'b0;
        local_wait = 0;
        while (done_count != expected_done_count && local_wait < 200000) begin
            tick();
            local_wait++;
        end
        if (done_count != expected_done_count) begin
            $display("graph client state/conv/classifier/done=%0d/%0b/%0b/%0b counts feature/surrogate/logit=%0d/%0d/%0d wave(v/r)=%0b/%0b logit(v/r)=%0b/%0b classifier capture/error=%0b/%0b",
                dut.u_system.u_array.g_scnn_graph.u_scnn_graph_client.state_q,
                dut.u_system.u_array.g_scnn_graph.u_scnn_graph_client.conv_done,
                dut.u_system.u_array.g_scnn_graph.u_scnn_graph_client.classifier_done,
                dut.u_system.graph_done, feature_wave_count, surrogate_count,
                logit_count, dut.u_system.feature_wave_valid,
                dut.u_system.feature_wave_ready, dut.u_system.logit_wave_valid,
                dut.u_system.logit_wave_ready,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.cl_o_capture_complete,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.cl_o_protocol_error);
            $display("internal frontend=%0b coord/prep/prescan/router=%0b/%0b/%0b/%0b batch_state=%0d runtime(active/pending/error)=%0b/%0b/%0b path(cfg/proto/abort/boundary)=%0b/%0b/%0b/%0b selected(cfg_cv/cfg_cl/joint/error/child/mod/version)=%0b/%0b/%0b/%0b/%0b/%0b/%0b",
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.frontend_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.coord_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.prep_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.prescan_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.router_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_frontend.u_batch.state_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.active_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.config_pending_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.runtime_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.path_cfg_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.path_protocol_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.path_abort_request,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.boundary_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.cv_o_cfg_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.cl_o_cfg_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.joint_cfg_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.error_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.child_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.internal_mod_error,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.observation_version_error);
            $display("capture cv/cl/mod=%0b/%0b/%0b pending/owned=%0b/%0b join_start=%0b/%0b",
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.cv_o_capture_complete,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.cl_o_capture_complete,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.internal_mod_capture_complete,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.capture_pending_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.learning_owned_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.join_start_valid,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.joint_ready);
            $fatal(1, "batch timeout expected=%0d graph=%0d start=%0b/%0b done=%0b owner=%0d array(b/i/e)=%0b/%0b/%0b context(v/r/pair/base/count/active)=%0b/%0b/%0b/%0d/%0d/%0b conv(v/r/cache/version)=%0b/%0b/%0b/%0d fill(v/r)=%b/%b maintenance(lease/req/start/ready/release/ready)=%0b/%0b/%0b/%0b/%0b/%0b feature(req/ready/resp/ready/busy)=%0b/%0b/%0b/%0b/%0b subsystem(b/q/e)=%0b/%0b/%0b commit=%0b",
                expected_done_count, graph_count, dut.u_system.graph_start_valid,
                dut.u_system.graph_start_ready, dut.u_system.graph_done, array_active_owner,
                dut.u_system.array_busy, dut.u_system.array_idle, dut.u_system.array_error,
                dut.u_system.array_context_load_valid, dut.u_system.array_context_load_ready,
                dut.u_system.array_context_pair, dut.u_system.array_context_output_base,
                dut.u_system.array_context_output_count, dut.u_system.o_context_active,
                dut.u_system.conv_context_valid, dut.u_system.conv_context_ready,
                conv_cache_valid, conv_cached_version,
                dut.u_system.weight_fill_valid, dut.u_system.weight_fill_ready,
                dut.u_system.maintenance_lease_q, dut.u_system.maintenance_request,
                dut.u_system.maintenance_start_valid, dut.u_system.maintenance_start_ready,
                dut.u_system.maintenance_release_valid,
                dut.u_system.maintenance_release_ready, feature_request_valid,
                feature_request_ready, feature_response_valid,
                feature_response_ready, dut.u_system.feature_arbiter_busy,
                dut.u_system.subsystem_busy, dut.u_system.subsystem_quiescent,
                dut.u_system.subsystem_error, dut.u_system.o_commit_pending);
        end
    endtask

    always @(posedge clk) begin
        if (rst_n &&
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.
                    observation_version_error) begin
            $display("version_error terms context/input/trace/surr/shared/feature/logit=%0b/%0b/%0b/%0b/%0b/%0b/%0b classifier_sample=%0d captured_batch=%0d feature(v/r/legal/ver)=%0b/%0b/%0b/%h logit(v/r/legal/ver)=%0b/%0b/%0b/%h shared(v/r/legal/sample/ver)=%0b/%0b/%0b/%0d/%h contexts target/student=%b/%b ctx_versions=%h/%h",
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.context_fire &&
                    !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.context_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.i_input_valid &&
                    dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.o_input_ready &&
                    !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.input_version_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.i_trace_valid &&
                    dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.o_trace_ready &&
                    !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.trace_version_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.i_surrogate_valid &&
                    dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.o_surrogate_ready &&
                    !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.surrogate_version_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.shared_trace_valid &&
                    dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.shared_trace_ready &&
                    !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.shared_trace_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.i_feature_trace_wave_valid &&
                    dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.o_feature_trace_wave_ready &&
                    !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.feature_version_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.i_logit_wave_valid &&
                    dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.o_logit_wave_ready &&
                    !dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.logit_version_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.classifier_sample_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.captured_batch_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.i_feature_trace_wave_valid,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.o_feature_trace_wave_ready,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.feature_version_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.i_feature_trace_wave_version,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.i_logit_wave_valid,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.o_logit_wave_ready,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.logit_version_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.i_logit_wave_version,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.shared_trace_valid,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.shared_trace_ready,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.shared_trace_legal,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.shared_trace_sample,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.shared_trace_version,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.context_valid_q[1],
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.context_valid_q[0],
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.context_version_q[1][0],
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.context_version_q[0][0]);
            $display("shared direct bounds(max/batch)=%0b/%0b context_valid=%0b version_equal=%0b sample_bits=%b batch_bits=%b context_version=%h shared_version=%h",
                int'(dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.
                    shared_trace_sample) >= 1,
                int'(dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.
                    shared_trace_sample) >=
                    int'(dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.
                        captured_batch_q),
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.
                    context_valid_q[0][dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.
                        u_selected_layers.shared_trace_sample],
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.
                    context_version_q[0][dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.
                        u_selected_layers.shared_trace_sample] ==
                    dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.
                        shared_trace_version,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.
                    shared_trace_sample,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.
                    captured_batch_q,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.
                    context_version_q[0][dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.
                        u_selected_layers.shared_trace_sample],
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.
                    shared_trace_version);
        end
    end

        // Generated exact-composition adaptation of the validated selected-system
    // sequence. The SCNN path uses the production shared output learner and store.
    unified_h40_final_composition_top_v1 #(
        .ENABLE_SCNN_CONV8_LEARNING(!CLASSIFIER_ONLY),
        .SCNN_WEIGHT_FRACTIONAL_BITS(SCNN_WEIGHT_FRACTIONAL_BITS),
        .ACCUMULATE_SCNN_SAMPLE_LOGITS(ACCUMULATE_SCNN_SAMPLE_LOGITS),
        .PARAM_DEPTH(4194304),
        .PARAM_ADDR_WIDTH(24),
        .MAX_BATCH(1),
        .MAX_TIMESTEPS(1),
        .MAX_INPUT_CHANNELS(2),
        .MAX_OUTPUT_CHANNELS(2),
        .OBS_INPUT_CHANNELS(1),
        .OBS_OUTPUT_CHANNELS(2),
        .SPATIAL(4),
        .POOLED_SPATIAL(2),
        .MAX_FEATURES(16),
        .MAX_CLASSES(16),
        .SAMPLE_WIDTH(1),
        .BATCH_COUNT_WIDTH(1),
        .TIMESTEP_WIDTH(1),
        .CHANNEL_WIDTH(4),
        .CHANNEL_COUNT_WIDTH(5),
        .FEATURE_WIDTH(4),
        .FEATURE_COUNT_WIDTH(5),
        .CLASS_WIDTH(4),
        .CLASS_COUNT_WIDTH(5),
        .INTERNAL_MODULATION_BATCH(1),
        .ENABLE_SCNN_FRAME_SERVICE(1'b0)
    ) dut (
        .i_initialize_valid(initialize_valid),
        .o_initialize_ready(initialize_ready),
        .o_initialize_done(initialize_done),
        .i_classifier_pair(1'b1),
        .i_context_model_version(16'd5),
        .i_context_adapt_version(16'd7),
        .i_system_parameter_owner(1'b1),
        .i_system_learning_enable(1'b1),
        .i_system_profile_aligned(1'b1),
        .i_system_profile_id(8'h31),
        .i_request_valid(request_valid),
        .o_request_ready(request_ready),
        .i_learning_enable(learning_enable),
        .i_batch_count(1'b1),
        .i_timestep(1'b0),
        .i_timestep_count(1'b1),
        .i_sample_version(16'h6100),
        .i_labels_flat(4'd0),
        .o_done(done),
        .i_fc_srnn_layer_valid(fc_srnn_layer_valid),
        .o_fc_srnn_layer_ready(fc_srnn_layer_ready),
        .i_fc_srnn_mode(fc_srnn_mode),
        .i_fc_srnn_weight_int8('0),
        .i_fc_srnn_current_shift6('0),
        .i_fc_srnn_neuron_count(fc_srnn_neuron_count),
        .i_fc_srnn_feedforward_group_count(fc_srnn_feedforward_group_count),
        .i_fc_srnn_recurrent_group_count(fc_srnn_recurrent_group_count),
        .i_fc_srnn_recurrent_enable(fc_srnn_recurrent_enable),
        .i_fc_srnn_leak(fc_srnn_leak),
        .i_fc_srnn_threshold(fc_srnn_threshold),
        .i_fc_srnn_reset_to_zero(fc_srnn_reset_to_zero),
        .i_fc_srnn_trace_decay(fc_srnn_trace_decay),
        .i_fc_srnn_transaction_tag(fc_srnn_transaction_tag),
        .i_fc_srnn_work_base_addr('0),
        .i_fc_srnn_destination_tile_base('0),
        .i_fc_srnn_group_valid(fc_srnn_group_valid),
        .o_fc_srnn_group_ready(fc_srnn_group_ready),
        .i_fc_srnn_group_recurrent(fc_srnn_group_recurrent),
        .i_fc_srnn_group_source_valid(fc_srnn_group_source_valid),
        .i_fc_srnn_group_weight_addr0(fc_srnn_group_weight_addr0),
        .i_fc_srnn_group_weight_addr1(fc_srnn_group_weight_addr1),
        .i_fc_srnn_group_weight_addr2(fc_srnn_group_weight_addr2),
        .i_fc_srnn_group_weight_addr3(fc_srnn_group_weight_addr3),
        .i_srnn_group_streamer_select(srnn_group_streamer_select),
        .i_srnn_group_streamer_start_valid(srnn_group_streamer_start_valid),
        .o_srnn_group_streamer_start_ready(srnn_group_streamer_start_ready),
        .i_srnn_group_feedforward_source_count(srnn_group_feedforward_source_count),
        .i_srnn_group_recurrent_source_count(srnn_group_recurrent_source_count),
        .i_srnn_group_feedforward_cache_row_base(srnn_group_feedforward_cache_row_base),
        .i_srnn_group_recurrent_cache_row_base(srnn_group_recurrent_cache_row_base),
        .i_srnn_group_feedforward_spike_flat(srnn_group_feedforward_spike_flat),
        .i_srnn_group_recurrent_spike_flat(srnn_group_recurrent_spike_flat),
        .o_srnn_group_streamer_done(srnn_group_streamer_done),
        .o_srnn_group_streamer_busy(srnn_group_streamer_busy),
        .o_srnn_group_streamer_sticky_error(srnn_group_streamer_error),
        .o_srnn_group_streamer_error_code(srnn_group_streamer_error_code),
        .o_fc_srnn_layer_done(fc_srnn_layer_done),
        .o_fc_srnn_busy(fc_srnn_busy),
        .o_fc_srnn_idle(fc_srnn_idle),
        .o_fc_srnn_sticky_error(fc_srnn_error),
        .o_fc_srnn_error_code(fc_srnn_error_code),
        .o_fc_srnn_observation_valid(fc_srnn_observation_valid),
        .i_fc_srnn_observation_ready(1'b1),
        .i_fc_srnn_observation_sink_idle(1'b1),
        .o_fc_srnn_observation_mode(fc_srnn_observation_mode),
        .o_fc_srnn_observation_transaction_tag(fc_srnn_observation_tag),
        .o_fc_srnn_observation_neuron_id(fc_srnn_observation_neuron),
        .o_fc_srnn_observation_spike(fc_srnn_observation_spike),
        .o_fc_srnn_observation_pre_state(fc_srnn_observation_pre_state),
        .o_fc_srnn_observation_trace_next(fc_srnn_observation_trace_next),
        .o_fc_srnn_observation_last_in_record(fc_srnn_observation_last_in_record),
        .o_fc_srnn_observation_busy(fc_srnn_observation_busy),
        .o_fc_srnn_observation_sticky_error(fc_srnn_observation_error),
        .i_fc_srnn_learning_cfg_valid(fc_srnn_learning_cfg_valid),
        .o_fc_srnn_learning_cfg_ready(fc_srnn_learning_cfg_ready),
        .i_fc_srnn_learning_cfg_mode(fc_srnn_learning_cfg_mode),
        .i_fc_srnn_learning_cfg_tag(fc_srnn_learning_cfg_tag),
        .i_fc_srnn_learning_sample_id(fc_srnn_learning_sample_id),
        .i_fc_srnn_learning_neuron_base(10'd0),
        .i_fc_srnn_learning_neuron_count(fc_srnn_learning_neuron_count),
        .i_fc_srnn_learning_fc_threshold(18'sd1),
        .i_fc_srnn_learning_fc_output_layer(fc_srnn_learning_fc_output_layer),
        .i_fc_srnn_learning_fc_label(fc_srnn_learning_fc_label),
        .i_fc_srnn_learning_fc_logit_scale_enable('0),
        .i_fc_srnn_learning_fc_logit_scale_q16('0),
        .i_fc_srnn_learning_fc_logit_accumulate('0),
        .i_fc_srnn_learning_fc_sample_start('0),
        .i_fc_srnn_learning_srnn_path(1'b1),
        .o_fc_surrogate_valid(fc_surrogate_valid),
        .o_fc_surrogate_sample(fc_surrogate_sample),
        .o_fc_surrogate_hidden(fc_surrogate_hidden),
        .o_fc_surrogate_q016(fc_surrogate_q016),
        .o_fc_logit_last(fc_logit_last),
        .o_fc_srnn_learning_capture_done(fc_srnn_learning_capture_done),
        .o_fc_srnn_learning_done(fc_srnn_learning_done),
        .o_fc_srnn_learning_busy(fc_srnn_learning_busy),
        .o_fc_srnn_learning_idle(fc_srnn_learning_idle),
        .o_fc_srnn_learning_sticky_error(fc_srnn_learning_error),
        .o_fc_srnn_learning_error_source(fc_srnn_learning_error_source),
        .o_fc_srnn_learning_error_code(fc_srnn_learning_error_code),
        .i_fc_w0_refill_valid(fc_refill_valid),
        .o_fc_w0_refill_ready(fc_refill_ready),
        .i_fc_w0_refill_all(1'b1),
        .i_fc_cache_refill_weight_int8(1'b0),
        .i_fc_cache_refill_destination_tile_base(4'd0),
        .i_fc_w0_dirty_feature('0),
        .i_fc_w0_dirty_tile('0),
        .i_fc_w0_parameter_base(8'd128),
        .i_fc_w0_hidden_stride(8'd1),
        .i_fc_cache_row_base(10'd0),
        .i_fc_w0_feature_count(5'd4),
        .i_fc_w0_hidden_count(10'd1),
        .i_fc_w0_tile_count(5'd1),
        .i_fc_w0_version(16'd77),
        .o_fc_w0_refill_done(fc_refill_done),
        .o_fc_w0_rows_published(fc_rows_published),
        .o_fc_w0_published_version(),
        .o_cache_fill_grant(),
        .o_cache_refill_busy(),
        .o_cache_refill_quiescent(),
        .o_cache_refill_sticky_error(),
        .i_srnn_backbone_refill_valid(1'b0),
        .o_srnn_backbone_refill_ready(),
        .i_srnn_backbone_refill_all(1'b0),
        .i_srnn_backbone_dirty_source('0),
        .i_srnn_backbone_dirty_tile('0),
        .i_srnn_backbone_parameter_base('0),
        .i_srnn_backbone_cache_row_base('0),
        .i_srnn_backbone_source_count('0),
        .i_srnn_backbone_destination_count('0),
        .i_srnn_backbone_tile_base('0),
        .i_srnn_backbone_version('0),
        .o_srnn_backbone_forward_read_valid(),
        .i_srnn_backbone_forward_read_ready(1'b0),
        .o_srnn_backbone_forward_read_addr(),
        .o_srnn_backbone_forward_read_tag(),
        .i_srnn_backbone_forward_response_valid(1'b0),
        .o_srnn_backbone_forward_response_ready(),
        .i_srnn_backbone_forward_response_tag('0),
        .i_srnn_backbone_forward_response_code('0),
        .i_srnn_backbone_forward_response_error(1'b0),
        .o_srnn_backbone_refill_done(),
        .o_srnn_backbone_rows_published(),
        .o_srnn_backbone_published_version(),
        .i_profile_id(8'h31),
        .i_conv_tensor_id(8'h08),
        .i_classifier_tensor_id(8'h12),
        .i_total_input_channels(5'd1),
        .i_total_output_channels(5'd2),
        .i_conv_parameter_base(8'(CONV_BASE)),
        .i_conv_weight_q16_base('0),
        .i_classifier_parameter_base(8'(CLASSIFIER_BASE)),
        .i_classifier_feature_count(5'(FEATURES)),
        .i_classifier_class_count(5'(CLASSES)),
        .i_classifier_trace_leak_valid(1'b1),
        .i_classifier_trace_leak_q8_8(16'sd0),
        .i_conv_vote_increment(16'd1),
        .i_conv_vote_threshold(16'd1),
        .i_classifier_vote_increment(16'd1),
        .i_classifier_vote_threshold(16'd1),
        .i_code_min(SCNN_WEIGHT_FRACTIONAL_BITS == 15 ?
            16'sh8000 : -16'sd127),
        .i_code_max(SCNN_WEIGHT_FRACTIONAL_BITS == 15 ?
            16'sh7fff : 16'sd127),
        .i_forward_start_valid(1'b0),
        .o_forward_start_ready(),
        .o_forward_start_fire(),
        .i_forward_done(1'b0),
        .o_forward_active(),
        .o_forward_model_version(),
        .o_forward_adapt_version(),
        .i_restore_complete(1'b0),
        .o_restore_ready(),
        .o_recovery_required(),
        .i_conv_in_width(16'd4),
        .i_conv_in_height(16'd4),
        .i_conv_in_channels(16'd1),
        .i_conv_out_width(16'd4),
        .i_conv_out_height(16'd4),
        .i_conv_out_channels(16'd2),
        .i_conv_stride(8'd1),
        .i_conv_padding(8'd1),
        .i_conv_pool_mode(2'd2),
        .i_conv_work_base_addr(9'd4),
        .i_conv_leak(16'd0),
        .i_conv_threshold(32'd1),
        .i_conv_reset_to_zero(1'b1),
        .i_conv_trace_decay(16'd0),
        .i_conv_layer_tag(CONV_TAG),
        .i_target_feature_count(5'(FEATURES)),
        .i_target_trace_leak_q8_8(16'sd0),
        .i_target_trace_state_reset(1'b1),
        .i_classifier_feature_channel_base(4'd0),
        .i_classifier_feature_channel_count(5'd2),
        .i_classifier_work_base_addr(9'd20),
        .i_classifier_trace_decay(16'd0),
        .i_classifier_layer_tag(CLASS_TAG),
        .i_frame_index(8'd0),
        .i_graph_descriptor_valid('0),
        .o_graph_descriptor_ready(),
        .i_graph_descriptor_last('0),
        .o_graph_descriptor_done(),
        .o_graph_frame_active(),
        .o_graph_frame_target(),
        .o_graph_frame_sample(),
        .o_graph_frame_timestep(),
        .o_graph_frame_version(),
        .o_state_service_valid(),
        .i_state_service_ready('0),
        .o_state_service_release(),
        .o_state_service_is_classifier(),
        .o_state_service_output_channel_base(),
        .o_state_service_output_channel_count(),
        .o_state_service_spatial_base(),
        .o_state_service_spatial_count(),
        .o_state_service_active_pair(),
        .o_state_service_work_base_addr(),
        .o_feature_request_valid(feature_request_valid),
        .i_feature_request_ready(feature_request_ready),
        .o_feature_term_valid(feature_term_valid),
        .o_feature_input_channel_flat(),
        .o_feature_kernel_row_flat(),
        .o_feature_kernel_col_flat(),
        .o_feature_spatial_base(feature_spatial_base),
        .o_feature_spatial_count(feature_spatial_count),
        .o_feature_output_row(),
        .o_feature_output_col(),
        .o_feature_value_bit(),
        .i_feature_response_valid(feature_response_valid),
        .o_feature_response_ready(feature_response_ready),
        .i_feature_mask0(feature_mask0),
        .i_feature_mask1(feature_mask1),
        .i_feature_mask2(feature_mask2),
        .i_feature_mask3(feature_mask3),
        .i_feature_service_error(1'b0),
        .i_feature_write_slot(1'b1),
        .i_feature_write_base_addr(13'd32),
        .i_feature_write_row_stride_words(16'd1),
        .i_feature_write_channel_stride_words(13'd2),
        .o_feature_update_valid(feature_update_valid),
        .i_feature_update_ready(1'b1),
        .o_feature_update_slot(),
        .o_feature_update_addr(),
        .o_feature_update_data(),
        .o_feature_update_mask(),
        .i_feature_update_done(feature_update_done),
        .i_maintenance_tile(maintenance_tile),
        .i_state_maintenance_write_valid(state_write_valid),
        .o_state_maintenance_write_ready(state_write_ready),
        .i_state_maintenance_write_addr(state_write_addr),
        .i_state_maintenance_write_data(state_write_data),
        .i_state_maintenance_write_mask(state_write_mask),
        .i_state_maintenance_read_valid(1'b0),
        .o_state_maintenance_read_ready(),
        .i_state_maintenance_read_addr('0),
        .o_state_maintenance_result_valid(),
        .i_state_maintenance_result_ready(1'b1),
        .o_state_maintenance_result_addr(),
        .o_state_maintenance_read_data(),
        .i_trace_maintenance_write_valid(trace_write_valid),
        .o_trace_maintenance_write_ready(trace_write_ready),
        .i_trace_maintenance_write_addr(trace_write_addr),
        .i_trace_maintenance_write_data(trace_write_data),
        .i_trace_maintenance_write_mask(trace_write_mask),
        .i_trace_maintenance_read_valid(1'b0),
        .o_trace_maintenance_read_ready(),
        .i_trace_maintenance_read_addr('0),
        .o_trace_maintenance_result_valid(),
        .i_trace_maintenance_result_ready(1'b1),
        .o_trace_maintenance_result_addr(),
        .o_trace_maintenance_read_data(),
        .o_classifier_cache_valid(classifier_cache_valid),
        .o_active_classifier_pair(active_classifier_pair),
        .o_parameter_version(parameter_version),
        .o_conv_cache_valid(conv_cache_valid),
        .o_conv_cached_output_base(),
        .o_conv_cached_output_count(),
        .o_conv_cached_version(conv_cached_version),
        .o_conv_cache_invalidate(conv_cache_invalidate),
        .o_model_version(model_version),
        .o_adapt_version(adapt_version),
        .o_capture_complete(),
        .o_learning_start_fire(),
        .o_commit_pulse(commit_pulse),
        .o_commit_code_dirty(),
        .o_commit_optimizer_dirty(),
        .o_detach_pulse(),
        .o_commit_pending(),
        .o_inference_hold(),
        .o_initialized(initialized),
        .o_versions_aligned(versions_aligned),
        .o_array_active_owner(array_active_owner),
        .o_array_tile_idle(array_tile_idle),
        .o_array_tile_sticky_error(array_tile_error),
        .o_context_active(),
        .o_context_is_classifier(),
        .o_busy(busy),
        .o_quiescent(quiescent),
        .o_sticky_error(sticky_error),
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_clear_error(clear_error),
        .i_srnn_phase_abort('0),
        .i_srnn_phase_capture_cfg_valid('0),
        .o_srnn_phase_capture_cfg_ready(),
        .i_srnn_phase_record_count('0),
        .i_srnn_phase_start_valid('0),
        .o_srnn_phase_start_ready(),
        .i_srnn_phase_learning_enable('0),
        .i_srnn_phase_target_valid('0),
        .o_srnn_phase_target_ready(),
        .i_srnn_phase_target_sample_id('0),
        .i_srnn_phase_target_backbone_id('0),
        .i_srnn_phase_target_q14('0),
        .i_srnn_phase_target_last('0),
        .i_srnn_phase_target_done('0),
        .i_srnn_phase_target_error('0),
        .o_srnn_phase_capture_done(),
        .o_srnn_phase_capture_complete(),
        .o_srnn_phase_delivery_done(),
        .o_srnn_phase_busy(),
        .o_srnn_phase_idle(),
        .o_srnn_phase_sticky_error(),
        .i_parameter_owner(parameter_owner),
        .i_topology_learning_enable(3'b100),
        .i_learning_fc_cfg_valid('0),
        .o_learning_fc_cfg_ready(),
        .i_learning_fc_profile_id('0),
        .i_learning_fc_w2_tensor_id('0),
        .i_learning_fc_w2_parameter_base('0),
        .i_learning_fc_batch_count('0),
        .i_learning_fc_hidden_count('0),
        .i_learning_fc_output_count('0),
        .i_learning_fc_vote_increment('0),
        .i_learning_fc_vote_threshold('0),
        .i_learning_fc_code_min('0),
        .i_learning_fc_code_max('0),
        .o_learning_fc_cfg_error(),
        .i_learning_fc_release('0),
        .o_learning_fc_release_ready(),
        .o_learning_fc_window_commit_valid(),
        .i_learning_fc_window_commit_ready('0),
        .o_learning_fc_window_commit_sequence(),
        .o_learning_fc_window_busy(),
        .o_learning_fc_error(),
        .i_learning_fc_parameter_load_valid('0),
        .o_learning_fc_parameter_load_ready(),
        .i_learning_fc_parameter_load_addr('0),
        .i_learning_fc_parameter_load_code('0),
        .i_learning_fc_parameter_load_residual('0),
        .i_learning_srnn_cfg_valid('0),
        .o_learning_srnn_cfg_ready(),
        .i_learning_srnn_profile_id('0),
        .i_learning_srnn_wh1_tensor_id('0),
        .i_learning_srnn_wh1_base('0),
        .i_learning_srnn_wh1_increment('0),
        .i_learning_srnn_wh1_threshold('0),
        .i_learning_srnn_wout_tensor_id('0),
        .i_learning_srnn_wout_base('0),
        .i_learning_srnn_wout_increment('0),
        .i_learning_srnn_wout_threshold('0),
        .i_learning_srnn_code_min('0),
        .i_learning_srnn_code_max('0),
        .i_learning_srnn_batch_count('0),
        .i_learning_srnn_hidden_count('0),
        .i_learning_srnn_backbone_count('0),
        .i_learning_srnn_class_count('0),
        .i_learning_srnn_wout_element_count('0),
        .o_learning_srnn_cfg_error(),
        .i_learning_srnn_parameter_load_valid('0),
        .o_learning_srnn_parameter_load_ready(),
        .i_learning_srnn_parameter_load_addr('0),
        .i_learning_srnn_parameter_load_code('0),
        .i_learning_srnn_parameter_load_residual('0),
        .i_learning_srnn_timestep_id('0),
        .i_learning_srnn_sequence_version('0),
        .i_learning_srnn_abort_timestep('0),
        .o_learning_srnn_abort_ready(),
        .o_learning_srnn_target_trace_start(),
        .i_learning_srnn_target_trace_start_ready('0),
        .o_learning_srnn_target_trace_timestep_id(),
        .i_learning_srnn_target_trace_valid('0),
        .o_learning_srnn_target_trace_ready(),
        .i_learning_srnn_target_trace_sample_id('0),
        .i_learning_srnn_target_trace_backbone_id('0),
        .i_learning_srnn_target_trace_q14('0),
        .i_learning_srnn_target_trace_last('0),
        .i_learning_srnn_target_trace_done('0),
        .i_learning_srnn_target_trace_error('0),
        .o_learning_srnn_label_replay_start(),
        .i_learning_srnn_label_replay_start_ready('0),
        .i_learning_srnn_label_valid('0),
        .o_learning_srnn_label_ready(),
        .i_learning_srnn_label_sample_id('0),
        .i_learning_srnn_label_class_id('0),
        .i_learning_srnn_label_last('0),
        .i_learning_srnn_restore_complete('0),
        .o_learning_srnn_restore_ready(),
        .o_learning_srnn_recovery_required(),
        .o_learning_srnn_learning_commit(),
        .o_learning_srnn_detach(),
        .o_learning_srnn_logit_valid(),
        .i_learning_srnn_logit_ready('0),
        .o_learning_srnn_logit_sample_id(),
        .o_learning_srnn_logit_class_id(),
        .o_learning_srnn_logit_q28(),
        .o_learning_srnn_logit_last(),
        .o_learning_srnn_timestep_done(),
        .o_learning_srnn_timestep_aborted(),
        .o_learning_srnn_busy(),
        .o_learning_srnn_error(),
        .o_learning_srnn_zero_gradient_count(),
        .o_learning_srnn_update_count(),
        .o_store_load_valid(scnn_store_load_valid),
        .i_store_load_ready(shared_load_ready[2]),
        .o_store_load_addr(scnn_store_load_addr),
        .o_store_load_code(scnn_store_load_code),
        .o_store_load_residual(scnn_store_load_residual),
        .o_store_learning_read_valid(scnn_store_learning_read_valid),
        .i_store_learning_read_ready(shared_learning_read_ready[2]),
        .o_store_learning_read_addr(scnn_store_learning_read_addr),
        .o_store_learning_read_tag(scnn_store_learning_read_tag),
        .i_store_learning_response_valid(shared_learning_response_valid[2]),
        .o_store_learning_response_ready(scnn_store_learning_response_ready),
        .i_store_learning_response_tag(shared_learning_response_tag[32 +: 16]),
        .i_store_learning_response_code(shared_learning_response_code[32 +: 16]),
        .i_store_learning_response_residual(shared_learning_response_residual[32 +: 16]),
        .i_store_learning_response_error(shared_learning_response_error[2]),
        .o_store_learning_write_valid(scnn_store_learning_write_valid),
        .i_store_learning_write_ready(shared_learning_write_ready[2]),
        .o_store_learning_write_addr(scnn_store_learning_write_addr),
        .o_store_learning_write_code(scnn_store_learning_write_code),
        .o_store_learning_write_residual(scnn_store_learning_write_residual),
        .o_store_learning_write_code_enable(scnn_store_learning_write_code_enable),
        .o_store_learning_write_residual_enable(scnn_store_learning_write_residual_enable),
        .o_store_forward_read_valid(scnn_store_forward_read_valid),
        .i_store_forward_read_ready(shared_forward_read_ready[2]),
        .o_store_forward_read_addr(scnn_store_forward_read_addr),
        .o_store_forward_read_tag(scnn_store_forward_read_tag),
        .i_store_forward_response_valid(shared_forward_response_valid[2]),
        .o_store_forward_response_ready(scnn_store_forward_response_ready),
        .i_store_forward_response_tag(shared_forward_response_tag[32 +: 16]),
        .i_store_forward_response_code(shared_forward_response_code[32 +: 16]),
        .i_store_forward_response_error(shared_forward_response_error[2]),
        .i_store_quiescent(shared_backend_quiescent),
        .i_store_protocol_error(shared_backend_error),
        .o_backend_quiescent(),
        .o_backend_error(),
        .o_owner_encoding_error()
    );

    // Signals consumed by the inherited dense-path scoreboards are internal
    // production links in the composition wrapper.
    assign fc_student_trace_valid = dut.link_fc_student_trace_valid;
    assign fc_student_trace_sample = dut.link_fc_student_trace_sample;
    assign fc_student_trace_hidden = dut.link_fc_student_trace_hidden;
    assign fc_student_trace_q016 = dut.link_fc_student_trace_q016;
    assign fc_logit_row_valid = dut.link_fc_logit_row_valid;
    assign fc_logit_row_ready = dut.link_fc_logit_row_ready;
    assign fc_logit_sample = dut.link_fc_logit_sample;
    assign fc_logit_label = dut.link_fc_logit_label;
    assign fc_logit_valid = dut.link_fc_logit_valid;
    assign fc_logit_ready = dut.link_fc_logit_ready;
    assign fc_logit_q6 = dut.link_fc_logit_q6;
    assign srnn_backbone_valid = dut.link_srnn_plane_backbone_valid;
    assign srnn_backbone_path = dut.link_srnn_plane_backbone_path;
    assign srnn_backbone_sample_id = dut.link_srnn_plane_backbone_sample_id;
    assign srnn_backbone_id = dut.link_srnn_plane_backbone_id;
    assign srnn_backbone_q14 = dut.link_srnn_plane_backbone_q14;
    assign srnn_backbone_last = dut.link_srnn_plane_backbone_last;
    assign parameter_load_ready = shared_load_ready[0];

    unified_tp_native_shared_parameter_backend #(
        .DEPTH(256), .PARAM_ADDR_WIDTH(8), .TAG_WIDTH(16),
        .CODE_WIDTH(16), .RESIDUAL_WIDTH(16)
    ) shared_parameter_backend (
        .i_clk(clk), .i_rst_n(rst_n), .i_clear_error(clear_error),
        .i_parameter_owner(parameter_owner),
        .i_topology_learning_enable(3'b100),
        .i_load_valid({scnn_store_load_valid, 1'b0, parameter_load_valid}),
        .o_load_ready(shared_load_ready),
        .i_load_addr({scnn_store_load_addr, 8'b0, parameter_load_addr}),
        .i_load_code({scnn_store_load_code, 16'b0, parameter_load_code}),
        .i_load_residual({scnn_store_load_residual, 32'b0}),
        .i_learning_read_valid({scnn_store_learning_read_valid, 2'b00}),
        .o_learning_read_ready(shared_learning_read_ready),
        .i_learning_read_addr({scnn_store_learning_read_addr, 16'b0}),
        .i_learning_read_tag({scnn_store_learning_read_tag, 32'b0}),
        .o_learning_response_valid(shared_learning_response_valid),
        .i_learning_response_ready(
            {scnn_store_learning_response_ready, 2'b00}),
        .o_learning_response_tag(shared_learning_response_tag),
        .o_learning_response_code(shared_learning_response_code),
        .o_learning_response_residual(shared_learning_response_residual),
        .o_learning_response_error(shared_learning_response_error),
        .i_learning_write_valid({scnn_store_learning_write_valid, 2'b00}),
        .o_learning_write_ready(shared_learning_write_ready),
        .i_learning_write_addr({scnn_store_learning_write_addr, 16'b0}),
        .i_learning_write_code({scnn_store_learning_write_code, 32'b0}),
        .i_learning_write_residual(
            {scnn_store_learning_write_residual, 32'b0}),
        .i_learning_write_code_enable(
            {scnn_store_learning_write_code_enable, 2'b00}),
        .i_learning_write_residual_enable(
            {scnn_store_learning_write_residual_enable, 2'b00}),
        .i_forward_read_valid({scnn_store_forward_read_valid, 2'b00}),
        .o_forward_read_ready(shared_forward_read_ready),
        .i_forward_read_addr({scnn_store_forward_read_addr, 16'b0}),
        .i_forward_read_tag({scnn_store_forward_read_tag, 32'b0}),
        .o_forward_response_valid(shared_forward_response_valid),
        .i_forward_response_ready(
            {scnn_store_forward_response_ready, 2'b00}),
        .o_forward_response_tag(shared_forward_response_tag),
        .o_forward_response_code(shared_forward_response_code),
        .o_forward_response_error(shared_forward_response_error),
        .o_backend_quiescent(shared_backend_quiescent),
        .o_sticky_error(shared_backend_error),
        .o_owner_encoding_error(), .o_router_protocol_error(),
        .o_store_protocol_error()
    );

    initial begin
        repeat (4) tick();
        rst_n = 1'b1;
        tick();

        progress_stage = 1;
        for (int tile = 0; tile < 3; tile++) begin
            clear_work(tile, 4);
            clear_work(tile, 20);
        end

        progress_stage = 2;
        for (int addr = 0; addr < 18; addr++)
            load_parameter(CONV_BASE + addr,
                1 << (SCNN_WEIGHT_FRACTIONAL_BITS - 8));
        for (int addr = 0; addr < FEATURES * CLASSES; addr++)
            load_parameter(CLASSIFIER_BASE + addr, 0);

        progress_stage = 3;
        while (!shared_backend_quiescent) tick();
        parameter_owner = 3'b100;
        tick();
        initialize_system();
        if (!initialized || !classifier_cache_valid ||
                !active_classifier_pair || parameter_version != 5 ||
                model_version != 5 || adapt_version != 7 ||
                !versions_aligned || sticky_error || shared_backend_error)
            $fatal(1, "initialization mismatch init/cache/pair=%0b/%0b/%0b versions=%0d/%0d/%0d error=%0b",
                initialized, classifier_cache_valid, active_classifier_pair,
                parameter_version, model_version, adapt_version, sticky_error);

        progress_stage = 7;
        fork
            request_batch(1'b1, 1);
            begin
                do @(posedge clk); while (!(request_valid && request_ready));
                @(negedge clk);
                if (!dut.u_system.subsystem_busy)
                    $fatal(1, "SCNN transaction not busy after request admission");
                check_learning_refill_block = 1'b1;
                fc_refill_valid = 1'b1;
                fc_cache_read_allow = 1'b1;
                // Withdraw before the next acceptance edge once commit is
                // visible; stage 10 separately tests admission after drain.
                wait (commit_pulse);
                @(negedge clk);
                fc_refill_valid = 1'b0;
                fc_cache_read_allow = 1'b0;
                check_learning_refill_block = 1'b0;
            end
        join
        tick();
        if (learning_refill_block_cycles == 0 ||
                publication_refill_block_cycles == 0 || fc_rows_published != 0)
            $fatal(1, "held FC refill did not cover SCNN learning and publication");
        if (graph_count != (CLASSIFIER_ONLY ? 1 : 2) ||
                target_graph_count != (CLASSIFIER_ONLY ? 0 : 1) ||
                student_graph_count != 1 || commit_count != 1 ||
                classifier_write_count == 0 || model_version != 6 ||
                adapt_version != 8 || parameter_version != 6 || !versions_aligned ||
                !active_classifier_pair || !saw_conv_invalidate ||
                conv_cache_valid || logit_count != 1 ||
                feature_wave_count == 0 ||
                (!CLASSIFIER_ONLY && surrogate_count == 0) ||
                (CLASSIFIER_ONLY && (surrogate_count != 0 ||
                                     conv_write_count != 0)) ||
                sticky_error || shared_backend_error || array_tile_error != 0)
            $fatal(1, "learning mismatch graphs=%0d target/student=%0d/%0d writes=%0d/%0d commit=%0d versions=%0d/%0d/%0d pair=%0b cache=%0b logits=%0d feature/surr=%0d/%0d error=%0b tile=%h",
                graph_count, target_graph_count, student_graph_count,
                conv_write_count, classifier_write_count, commit_count,
                model_version, adapt_version, parameter_version,
                active_classifier_pair, conv_cache_valid, logit_count,
                feature_wave_count, surrogate_count, sticky_error,
                array_tile_error);

        progress_stage = 8;
        for (int tile = 0; tile < 3; tile++) begin
            clear_work(tile, 4);
            clear_work(tile, 20);
        end

        begin
            int writes_before;
            int commits_before;
            int fills_before;
            writes_before = conv_write_count + classifier_write_count;
            commits_before = commit_count;
            fills_before = weight_fill_count;
            request_batch(1'b0, 2);
            tick();
            if (graph_count != (CLASSIFIER_ONLY ? 2 : 3) ||
                    target_graph_count != (CLASSIFIER_ONLY ? 0 : 1) ||
                    student_graph_count != 2 ||
                    conv_write_count + classifier_write_count != writes_before ||
                    commit_count != commits_before ||
                    weight_fill_count <= fills_before ||
                    !conv_cache_valid || conv_cached_version != 6 ||
                    model_version != 6 || parameter_version != 6 || !versions_aligned ||
                    logit_count != 2 || sticky_error || shared_backend_error ||
                    array_tile_error != 0 || array_active_owner != 0 ||
                    !quiescent || array_tile_idle != 16'hffff)
                $fatal(1, "inference mismatch graphs=%0d writes=%0d/%0d commit=%0d fills=%0d/%0d conv_cache=%0b v=%0d logits=%0d owner=%0d quiet=%0b error=%0b tile=%h",
                    graph_count, conv_write_count, classifier_write_count,
                    commit_count, fills_before, weight_fill_count,
                    conv_cache_valid, conv_cached_version, logit_count,
                    array_active_owner, quiescent, sticky_error,
                    array_tile_error);
        end

        if (inference_logits[0] == learned_logits[0] &&
                inference_logits[1] == learned_logits[1] &&
                inference_logits[2] == learned_logits[2])
            $fatal(1, "committed classifier update was not visible to inference");

        if (!quiescent || !shared_backend_quiescent || shared_backend_error ||
                sticky_error || array_tile_error != 0 || array_active_owner != 0 ||
                array_tile_idle != 16'hffff)
            $fatal(1, "final SCNN quiescence mismatch owner=%0d quiet=%0b/%0b error=%0b/%0b tile=%h idle=%h",
                array_active_owner, quiescent, shared_backend_quiescent,
                sticky_error, shared_backend_error, array_tile_error,
                array_tile_idle);

        $display("FINAL_COMPOSITION_SCNN_CLASSIFIER_V1_PASS exact_top=1 sequence=SCNN_FORWARD_LEARN_COMMIT_NEXT_FORWARD classifier_only=%0b target_graphs=%0d weight_frac=%0d shared_output_learner=1 shared_store_client=2 synthetic_pooled_features=1", CLASSIFIER_ONLY, target_graph_count, SCNN_WEIGHT_FRACTIONAL_BITS);
        $finish;
    end
endmodule

`default_nettype wire
