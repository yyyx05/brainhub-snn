`timescale 1ns/1ps
`default_nettype none

module tb_unified_scnn_selected_layers_graph_runtime_v1 #(
    parameter bit CLASSIFIER_ONLY = 1'b0
);
    localparam logic [7:0] PROFILE = 8'h20;
    localparam int CONV_BASE = 0;
    localparam int CLASSIFIER_BASE = 64;

    logic i_clk = 1'b0;
    logic i_rst_n;
    logic i_clear_error;
    logic i_system_parameter_owner;
    logic i_system_learning_enable;
    logic i_system_profile_aligned;
    logic [7:0] i_system_profile_id;
    logic i_request_valid;
    logic o_request_ready;
    logic i_learning_enable;
    logic [1:0] i_batch_count;
    logic i_timestep;
    logic [1:0] i_timestep_count;
    logic [15:0] i_sample_version;
    logic [3:0] i_labels_flat;
    logic o_done;
    logic [7:0] i_profile_id;
    logic [7:0] i_conv_tensor_id;
    logic [7:0] i_classifier_tensor_id;
    logic [1:0] i_total_input_channels;
    logic [1:0] i_total_output_channels;
    logic [7:0] i_conv_parameter_base;
    logic [7:0] i_conv_weight_q16_base;
    logic [7:0] i_classifier_parameter_base;
    logic [3:0] i_classifier_feature_count;
    logic [1:0] i_classifier_class_count;
    logic i_classifier_trace_leak_valid;
    logic signed [15:0] i_classifier_trace_leak_q8_8;
    logic [15:0] i_conv_vote_increment;
    logic [15:0] i_conv_vote_threshold;
    logic [15:0] i_classifier_vote_increment;
    logic [15:0] i_classifier_vote_threshold;
    logic signed [15:0] i_code_min;
    logic signed [15:0] i_code_max;
    logic i_context_load_valid;
    logic o_context_load_ready;
    logic [15:0] i_context_model_version;
    logic [15:0] i_context_adapt_version;
    logic i_forward_start_valid;
    logic o_forward_start_ready;
    logic o_forward_start_fire;
    logic i_forward_done;
    logic o_forward_active;
    logic [15:0] o_forward_model_version;
    logic [15:0] o_forward_adapt_version;
    logic i_restore_complete;
    logic o_restore_ready;
    logic o_recovery_required;
    logic [15:0] i_conv_in_width;
    logic [15:0] i_conv_in_height;
    logic [15:0] i_conv_in_channels;
    logic [7:0] i_conv_padding;
    logic [19:0] i_conv_layer_tag;
    logic [3:0] i_target_feature_count;
    logic signed [15:0] i_target_trace_leak_q8_8;
    logic i_target_trace_state_reset;
    logic o_graph_start_valid;
    logic i_graph_start_ready;
    logic o_graph_target;
    logic o_graph_sample;
    logic o_graph_timestep;
    logic [15:0] o_graph_version;
    logic [1:0] o_graph_label;
    logic o_graph_sample_last;
    logic i_graph_done;
    logic i_graph_error;
    logic o_conv_observation_enable;
    logic o_classifier_observation_enable;
    logic o_observation_target;
    logic o_prescan_feature_request_valid;
    logic i_prescan_feature_request_ready;
    logic [3:0] o_prescan_feature_term_valid;
    logic [39:0] o_prescan_feature_input_channel_flat;
    logic [7:0] o_prescan_feature_kernel_row_flat;
    logic [7:0] o_prescan_feature_kernel_col_flat;
    logic [31:0] o_prescan_feature_spatial_base;
    logic [5:0] o_prescan_feature_spatial_count;
    logic [15:0] o_prescan_feature_output_row;
    logic [15:0] o_prescan_feature_output_col;
    logic i_prescan_feature_response_valid;
    logic o_prescan_feature_response_ready;
    logic [39:0] i_prescan_feature_mask0;
    logic [39:0] i_prescan_feature_mask1;
    logic [39:0] i_prescan_feature_mask2;
    logic [39:0] i_prescan_feature_mask3;
    logic i_prescan_feature_service_error;
    logic i_surrogate_valid;
    logic o_surrogate_source_ready;
    logic i_surrogate_target;
    logic i_surrogate_sample;
    logic i_surrogate_timestep;
    logic [15:0] i_surrogate_version;
    logic [19:0] i_surrogate_layer_tag;
    logic [9:0] i_surrogate_channel;
    logic [31:0] i_surrogate_spatial;
    logic signed [15:0] i_surrogate_q8_8;
    logic i_feature_wave_valid;
    logic o_feature_wave_ready;
    logic [7:0] i_feature_wave_mask;
    logic [2:0] i_feature_wave_base;
    logic [127:0] i_feature_wave_q8_8_flat;
    logic [15:0] i_feature_wave_version;
    logic [7:0] i_feature_wave_frame_index;
    logic i_feature_wave_first;
    logic i_feature_wave_last;
    logic i_feature_wave_sample_reset;
    logic i_feature_wave_sample_last;
    logic [1:0] i_feature_sample_label;
    logic i_logit_wave_valid;
    logic o_logit_wave_ready;
    logic [11:0] i_logit_wave_mask;
    logic [1:0] i_logit_wave_base;
    logic [1:0] i_logit_wave_count;
    logic [191:0] i_logit_wave_q8_8_flat;
    logic [15:0] i_logit_wave_version;
    logic [7:0] i_logit_wave_frame_index;
    logic i_logit_wave_frame_last;
    logic i_logit_wave_sample_last;
    logic o_read_valid;
    logic i_read_ready;
    logic [7:0] o_read_profile_id;
    logic [7:0] o_read_tensor_id;
    logic [7:0] o_read_parameter_addr;
    logic [7:0] o_read_tag;
    logic i_read_response_valid;
    logic o_read_response_ready;
    logic signed [15:0] i_read_response_code;
    logic signed [15:0] i_read_response_residual;
    logic i_read_response_error;
    logic o_result_valid;
    logic i_result_ready;
    logic [7:0] o_result_profile_id;
    logic [7:0] o_result_tensor_id;
    logic [7:0] o_result_parameter_addr;
    logic [7:0] o_result_tag;
    logic signed [15:0] o_result_code;
    logic signed [15:0] o_result_residual;
    logic o_write_code_enable;
    logic o_write_residual_enable;
    logic i_commit_publish_ready;
    logic [15:0] o_model_version;
    logic [15:0] o_adapt_version;
    logic o_capture_complete;
    logic o_learning_start_fire;
    logic o_commit_pulse;
    logic o_commit_code_dirty;
    logic o_commit_optimizer_dirty;
    logic o_detach_pulse;
    logic o_commit_pending;
    logic o_inference_hold;
    logic o_busy;
    logic o_quiescent;
    logic o_sticky_error;

    logic load_valid;
    logic load_ready;
    logic [7:0] load_addr;
    logic store_read_ready;
    logic store_response_valid;
    logic [7:0] store_response_tag;
    logic signed [15:0] store_response_code;
    logic signed [15:0] store_response_residual;
    logic store_response_error;
    logic store_write_ready;
    logic store_protocol_error;
    logic classifier_prime_valid;
    logic classifier_prime_ready;
    logic classifier_prime_done;
    logic conv_context_valid;
    logic conv_context_ready;
    logic conv_context_load_weights;
    logic [9:0] conv_output_channel_base;
    logic [4:0] conv_output_channel_count;
    logic [1:0] conv_input_channel_count;
    logic conv_context_pair;
    logic [7:0] conv_parameter_base;
    logic [15:0] conv_context_version;
    logic [3:0] weight_fill_valid;
    logic [7:0] weight_fill_tile_flat;
    logic [3:0] weight_fill_bank;
    logic [43:0] weight_fill_addr_flat;
    logic [799:0] weight_fill_data_flat;
    logic classifier_cache_valid;
    logic active_classifier_pair;
    logic [15:0] active_parameter_version;
    logic conv_cache_valid;
    logic [9:0] conv_cached_output_base;
    logic [4:0] conv_cached_output_count;
    logic [15:0] conv_cached_version;
    logic conv_cache_invalidate;
    logic parameter_plane_busy;
    logic parameter_plane_error;
    logic saw_conv_invalidate;
    integer weight_fill_count;
    integer graph_count;
    integer feature_request_count;
    integer write_count;
    integer commit_count;
    logic tile_error_seen;
    integer learning_start_count;
    integer conv_write_count;
    integer classifier_write_count;
    integer commit_wait_cycles;
    logic [5:0] graph_target_history;
    logic [5:0] graph_sample_history;

    always #5 i_clk = ~i_clk;

    unified_scnn_selected_layers_graph_runtime_v1 #(
        .PARAM_ADDR_WIDTH(8), .TAG_WIDTH(8), .MAX_BATCH(2),
        .MAX_TIMESTEPS(2), .MAX_INPUT_CHANNELS(2),
        .MAX_OUTPUT_CHANNELS(2), .OBS_INPUT_CHANNELS(2),
        .OBS_OUTPUT_CHANNELS(2), .Q16_ADDR_WIDTH(8),
        .Q64_ADDR_WIDTH(6), .MAX_FEATURES(8), .MAX_CLASSES(3),
        .VERSION_WIDTH(16), .SAMPLE_WIDTH(1), .BATCH_COUNT_WIDTH(2),
        .TIMESTEP_WIDTH(1), .CHANNEL_COUNT_WIDTH(2),
        .FEATURE_WIDTH(3), .FEATURE_COUNT_WIDTH(4),
        .CLASS_WIDTH(2), .CLASS_COUNT_WIDTH(2),
        .INTERNAL_MODULATION_BATCH(2),
        .ENABLE_SCNN_CONV8_LEARNING(!CLASSIFIER_ONLY)
    ) dut (
        .o_external_cfg_valid(), .i_external_cfg_ready(1'b0),
        .o_external_cfg_profile_id(), .o_external_cfg_tensor_id(),
        .o_external_cfg_parameter_base(), .o_external_cfg_batch_count(),
        .o_external_cfg_source_count(), .o_external_cfg_output_count(),
        .o_external_cfg_vote_increment(), .o_external_cfg_vote_threshold(),
        .o_external_cfg_code_min(), .o_external_cfg_code_max(),
        .i_external_cfg_error(1'b0), .i_external_active(1'b0),
        .o_external_release(), .i_external_release_ready(1'b0),
        .o_external_source_trace_valid(),
        .i_external_source_trace_ready(1'b0),
        .o_external_source_trace_sample_id(),
        .o_external_source_trace_index(), .o_external_source_trace_q016(),
        .i_external_source_trace_error(1'b0),
        .o_external_logit_row_valid(), .i_external_logit_row_ready(1'b0),
        .o_external_logit_sample_id(), .o_external_logit_label(),
        .o_external_logit_valid(), .i_external_logit_ready(1'b0),
        .o_external_logit_q6(), .i_external_logit_error(1'b0),
        .i_external_logit_row_done(1'b0), .o_external_observation_done(),
        .i_external_observation_done_ready(1'b0),
        .i_external_window_done(1'b0), .i_external_window_busy(1'b0),
        .i_external_context_error(1'b0),
        .i_external_gradient_overflow(1'b0),
        .i_external_update_config_error(1'b0),
        .i_external_request_error(1'b0), .i_external_store_error(1'b0),
        .i_external_internal_error(1'b0),
        .*
    );

    unified_scnn_selected_layers_parameter_plane_v1 #(
        .PARAM_DEPTH(128), .PARAM_ADDR_WIDTH(8), .TAG_WIDTH(8),
        .MAX_INPUT_CHANNELS(2), .MAX_CLASSIFIER_FEATURES(8),
        .MAX_CLASSES(3), .INPUT_CHANNEL_COUNT_WIDTH(2),
        .CLASSIFIER_FEATURE_COUNT_WIDTH(4), .CLASS_COUNT_WIDTH(2)
    ) parameter_plane (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(1'b0),
        .i_cache_invalidate(1'b0),
        .i_clear_error(i_clear_error),
        .i_parameter_load_valid(load_valid),
        .o_parameter_load_ready(load_ready), .i_parameter_load_addr(load_addr),
        .i_parameter_load_code(16'sd0),
        .i_parameter_load_residual(16'sd0),
        .i_classifier_prime_valid(classifier_prime_valid),
        .o_classifier_prime_ready(classifier_prime_ready),
        .i_classifier_pair(1'b1), .i_initial_version(16'd5),
        .i_classifier_parameter_base(8'(CLASSIFIER_BASE)),
        .i_classifier_feature_count(4'd8),
        .i_classifier_class_count(2'd3),
        .o_classifier_prime_done(classifier_prime_done),
        .i_commit_pending(o_commit_pending),
        .o_commit_publish_ready(i_commit_publish_ready),
        .i_commit_pulse(o_commit_pulse),
        .i_learning_read_valid(o_read_valid),
        .o_learning_read_ready(store_read_ready),
        .i_learning_read_addr(o_read_parameter_addr),
        .i_learning_read_tag(o_read_tag),
        .o_learning_response_valid(store_response_valid),
        .i_learning_response_ready(o_read_response_ready),
        .o_learning_response_tag(store_response_tag),
        .o_learning_response_code(store_response_code),
        .o_learning_response_residual(store_response_residual),
        .o_learning_response_error(store_response_error),
        .i_learning_write_valid(o_result_valid),
        .o_learning_write_ready(store_write_ready),
        .i_learning_write_addr(o_result_parameter_addr),
        .i_learning_write_code(o_result_code),
        .i_learning_write_residual(o_result_residual),
        .i_learning_write_code_enable(o_write_code_enable),
        .i_learning_write_residual_enable(o_write_residual_enable),
        .i_conv_context_valid(conv_context_valid),
        .o_conv_context_ready(conv_context_ready),
        .i_conv_context_load_weights(conv_context_load_weights),
        .i_conv_output_channel_base(conv_output_channel_base),
        .i_conv_output_channel_count(conv_output_channel_count),
        .i_conv_input_channel_count(conv_input_channel_count),
        .i_conv_context_pair(conv_context_pair),
        .i_conv_parameter_base(conv_parameter_base),
        .i_conv_context_version(conv_context_version),
        .i_array_idle(1'b1), .o_weight_fill_valid(weight_fill_valid),
        .i_weight_fill_ready(4'hf),
        .o_weight_fill_tile_flat(weight_fill_tile_flat),
        .o_weight_fill_bank(weight_fill_bank),
        .o_weight_fill_addr_flat(weight_fill_addr_flat),
        .o_weight_fill_data_flat(weight_fill_data_flat),
        .o_classifier_cache_valid(classifier_cache_valid),
        .o_active_classifier_pair(active_classifier_pair),
        .o_active_version(active_parameter_version),
        .o_conv_cache_valid(conv_cache_valid),
        .o_conv_cached_output_base(conv_cached_output_base),
        .o_conv_cached_output_count(conv_cached_output_count),
        .o_conv_cached_version(conv_cached_version),
        .o_conv_cache_invalidate(conv_cache_invalidate),
        .o_busy(parameter_plane_busy), .o_sticky_error(parameter_plane_error)
    );

    assign i_read_ready = store_read_ready;
    assign i_read_response_valid = store_response_valid;
    assign i_read_response_code = store_response_code;
    assign i_read_response_residual = store_response_residual;
    assign i_read_response_error = store_response_error;
    assign i_result_ready = store_write_ready;
    assign i_prescan_feature_request_ready =
        !i_prescan_feature_response_valid ||
        o_prescan_feature_response_ready;
    assign i_prescan_feature_mask0 = '1;
    assign i_prescan_feature_mask1 = '0;
    assign i_prescan_feature_mask2 = '0;
    assign i_prescan_feature_mask3 = '0;

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            i_prescan_feature_response_valid <= 1'b0;
            graph_count <= 0;
            feature_request_count <= 0;
            write_count <= 0;
            commit_count <= 0;
            learning_start_count <= 0;
            commit_wait_cycles <= 0;
            graph_target_history <= '0;
            graph_sample_history <= '0;
            saw_conv_invalidate <= 1'b0;
            weight_fill_count <= 0;
        end else begin
            if (i_prescan_feature_response_valid &&
                    o_prescan_feature_response_ready)
                i_prescan_feature_response_valid <= 1'b0;
            if (o_prescan_feature_request_valid &&
                    i_prescan_feature_request_ready) begin
                i_prescan_feature_response_valid <= 1'b1;
                feature_request_count <= feature_request_count + 1;
            end
            if (o_graph_start_valid && i_graph_start_ready) begin
                graph_target_history[graph_count] <= o_graph_target;
                graph_sample_history[graph_count] <= o_graph_sample;
                graph_count <= graph_count + 1;
            end
            if (o_result_valid && i_result_ready) begin
                write_count <= write_count + 1;
                if (o_result_tensor_id == i_conv_tensor_id) begin
                    if (CLASSIFIER_ONLY)
                        $fatal(1, "classifier-only graph wrote Conv8 parameters");
                    conv_write_count <= conv_write_count + 1;
                end else if (o_result_tensor_id == i_classifier_tensor_id)
                    classifier_write_count <= classifier_write_count + 1;
                else
                    $fatal(1, "unexpected graph write tensor %0d", o_result_tensor_id);
            end
            if (o_learning_start_fire)
                learning_start_count <= learning_start_count + 1;
            if (o_commit_pulse) begin
                commit_count <= commit_count + 1;
            end
            if (conv_cache_invalidate)
                saw_conv_invalidate <= 1'b1;
            for (int region = 0; region < 4; region++)
                if (weight_fill_valid[region])
                    weight_fill_count <= weight_fill_count + 1;
        end
    end

    generate if (!CLASSIFIER_ONLY) begin : g_conv_debug
    always_ff @(posedge i_clk) begin
        if (i_rst_n && dut.input_valid && dut.input_ready &&
                !dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.input_legal)
            $display("ILLEGAL_INPUT target=%0b actual(s/c/p/t/v)=%0d/%0d/%0d/%0d/%0h expected_target0=%0d/%0d/%0d expected_target1=%0d/%0d/%0d",
                dut.input_target, dut.input_sample, dut.input_channel,
                dut.input_spatial, dut.input_timestep, dut.input_version,
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.input_sample_q[0],
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.input_channel_q[0],
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.input_spatial_q[0],
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.input_sample_q[1],
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.input_channel_q[1],
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.input_spatial_q[1]);
        if (i_rst_n && dut.trace_valid && dut.trace_ready &&
                !dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.trace_legal)
            $display("ILLEGAL_TRACE target=%0b actual(s/c/p/t/v)=%0d/%0d/%0d/%0d/%0h expected_target0=%0d/%0d/%0d/%0d expected_target1=%0d/%0d/%0d/%0d",
                dut.trace_target, dut.trace_sample, dut.trace_channel,
                dut.trace_spatial, dut.trace_timestep, dut.trace_version,
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.g_trace_channel_capture_order.sample_q[0],
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.g_trace_channel_capture_order.channel_q[0],
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.g_trace_channel_capture_order.spatial_q[0],
                0,
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.g_trace_channel_capture_order.sample_q[1],
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.g_trace_channel_capture_order.channel_q[1],
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.g_trace_channel_capture_order.spatial_q[1],
                0);
        if (i_rst_n && dut.surrogate_valid && dut.surrogate_ready &&
                !dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.surrogate_legal)
            $display("ILLEGAL_SURROGATE target=%0b actual(s/c/p/t/v)=%0d/%0d/%0d/%0d/%0h expected_target0=%0d/%0d/%0d/%0d expected_target1=%0d/%0d/%0d/%0d",
                dut.surrogate_target, dut.surrogate_sample,
                dut.surrogate_channel, dut.surrogate_spatial,
                dut.surrogate_timestep, dut.surrogate_version,
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.g_wave_capture_order.sample_q[0],
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.g_wave_capture_order.base_q[0],
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.g_wave_capture_order.spatial_q[0],
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.g_wave_capture_order.lane_q[0],
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.g_wave_capture_order.sample_q[1],
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.g_wave_capture_order.base_q[1],
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.g_wave_capture_order.spatial_q[1],
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_observations.g_wave_capture_order.lane_q[1]);
    end

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            tile_error_seen <= 1'b0;
        end else if (!tile_error_seen) begin
            if (dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.state_q == 1 &&
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_student_input_valid &&
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.o_student_input_ready &&
                    !dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.input_coordinates_legal) begin
                tile_error_seen <= 1'b1;
                $display("TILE_ERROR student input sample/index=%0d/%0d limit batch/count=%0d/%0d",
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_student_input_sample,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_student_input_index,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.batch_count_q,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.input_count_q);
            end else if (dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.state_q == 1 &&
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_target_input_valid &&
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.o_target_input_ready &&
                    !dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.target_input_coordinates_legal) begin
                tile_error_seen <= 1'b1;
                $display("TILE_ERROR target input sample/index=%0d/%0d limit batch/count=%0d/%0d",
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_target_input_sample,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_target_input_index,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.batch_count_q,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.input_count_q);
            end else if (dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.state_q == 1 &&
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_student_trace_valid &&
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.o_student_trace_ready &&
                    !dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.student_trace_coordinates_legal) begin
                tile_error_seen <= 1'b1;
                $display("TILE_ERROR student trace sample/index=%0d/%0d limit batch/count=%0d/%0d",
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_student_trace_sample,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_student_trace_index,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.batch_count_q,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.output_count_q);
            end else if (dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.state_q == 1 &&
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_target_trace_valid &&
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.o_target_trace_ready &&
                    !dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.target_trace_coordinates_legal) begin
                tile_error_seen <= 1'b1;
                $display("TILE_ERROR target trace sample/index=%0d/%0d limit batch/count=%0d/%0d",
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_target_trace_sample,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_target_trace_index,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.batch_count_q,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.output_count_q);
            end else if (dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.state_q == 1 &&
                    ((dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_student_surrogate_valid &&
                      dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.o_student_surrogate_ready &&
                      !dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.student_surrogate_coordinates_legal) ||
                     (dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_target_surrogate_valid &&
                      dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.o_target_surrogate_ready &&
                      !dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.target_surrogate_coordinates_legal))) begin
                tile_error_seen <= 1'b1;
                $display("TILE_ERROR surrogate student=%0b samples=%0d/%0d indices=%0d/%0d limit batch/count=%0d/%0d",
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_student_surrogate_valid,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_student_surrogate_sample,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_target_surrogate_sample,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_student_surrogate_index,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_target_surrogate_index,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.batch_count_q,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.output_count_q);
            end else if (dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.state_q == 1 &&
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_modulation_valid &&
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.o_modulation_ready &&
                    !dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.modulation_coordinates_legal) begin
                tile_error_seen <= 1'b1;
                $display("TILE_ERROR modulation row/col=%0d/%0d batch=%0d",
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_modulation_row,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_modulation_col,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.batch_count_q);
            end else if (dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.state_q == 1 &&
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.i_start &&
                    !dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.o_start_ready) begin
                tile_error_seen <= 1'b1;
                $display("TILE_ERROR early start counts input=%0d/%0d trace=%0d/%0d surrogate=%0d/%0d modulation=%0d",
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.student_input_count_q,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.target_input_count_q,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.student_trace_count_q,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.target_trace_count_q,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.student_surrogate_count_q,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.target_surrogate_count_q,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.modulation_count_q);
            end else if (dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.state_q == 6 &&
                    (dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.mapper_error ||
                     !dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.mapper_valid)) begin
                tile_error_seen <= 1'b1;
                $display("TILE_ERROR mapper valid/error/legal=%0b/%0b/%0b total in/out=%0d/%0d mapper ports=%0d/%0d coord out/in/row/col=%0d/%0d/%0d/%0d mapper coords=%0d/%0d/%0d/%0d addr local/param/q16=%0d/%0d/%0d limits=%0d/%0d",
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.mapper_valid,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.mapper_error,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.u_address_mapper.coordinates_legal,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.total_input_channels_q,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.total_output_channels_q,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.u_address_mapper.i_input_channels,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.u_address_mapper.i_output_channels,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.grad_output_q,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.grad_input_q,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.grad_row_q,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.grad_col_q,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.u_address_mapper.i_output_channel,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.u_address_mapper.i_input_channel,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.u_address_mapper.i_kernel_row,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.u_address_mapper.i_kernel_col,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.u_address_mapper.local_addr,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.u_address_mapper.parameter_addr,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.u_address_mapper.weight_q16_addr,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.u_address_mapper.PARAM_LIMIT,
                    dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.u_address_mapper.Q16_LIMIT);
            end else if (dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.o_protocol_error) begin
                tile_error_seen <= 1'b1;
                $display("TILE_ERROR unclassified state=%0d", dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.u_tile_engine.state_q);
            end
        end
    end

    end endgenerate

    task automatic tick;
        @(posedge i_clk);
        #1;
    endtask

    task automatic load_parameters;
        for (int address = 0; address < 36; address++) begin
            @(negedge i_clk);
            while (!load_ready) @(negedge i_clk);
            load_addr = 8'(CONV_BASE + address);
            load_valid = 1'b1;
            tick();
            load_valid = 1'b0;
        end
        for (int address = 0; address < 24; address++) begin
            @(negedge i_clk);
            while (!load_ready) @(negedge i_clk);
            load_addr = 8'(CLASSIFIER_BASE + address);
            load_valid = 1'b1;
            tick();
            load_valid = 1'b0;
        end
    endtask

    task automatic load_versions;
        @(negedge i_clk);
        while (!o_context_load_ready) @(negedge i_clk);
        i_context_load_valid = 1'b1;
        tick();
        i_context_load_valid = 1'b0;
    endtask

    task automatic prime_classifier;
        @(negedge i_clk);
        while (!classifier_prime_ready) @(negedge i_clk);
        classifier_prime_valid = 1'b1;
        tick();
        classifier_prime_valid = 1'b0;
        while (!classifier_prime_done) tick();
    endtask

    task automatic request_conv_context(input int version);
        @(negedge i_clk);
        conv_context_version = 16'(version);
        conv_context_valid = 1'b1;
        while (!conv_context_ready) @(negedge i_clk);
        tick();
        conv_context_valid = 1'b0;
        while (!conv_cache_valid || parameter_plane_busy) tick();
    endtask

    task automatic request_batch(input logic learning);
        @(negedge i_clk);
        while (!o_request_ready) @(negedge i_clk);
        i_learning_enable = learning;
        i_request_valid = 1'b1;
        tick();
        i_request_valid = 1'b0;
    endtask

    task automatic send_surrogates;
        for (int spatial = 0; spatial < 16; spatial++)
            for (int channel = 0; channel < 2; channel++) begin
                @(negedge i_clk);
                i_surrogate_target = o_graph_target;
                i_surrogate_sample = o_graph_sample;
                i_surrogate_timestep = o_graph_timestep;
                i_surrogate_version = o_graph_version;
                i_surrogate_layer_tag = 20'd7;
                i_surrogate_channel = 10'(channel);
                i_surrogate_spatial = 32'(spatial);
                i_surrogate_q8_8 = 16'sh0100;
                while (!o_surrogate_source_ready) @(negedge i_clk);
                i_surrogate_valid = 1'b1;
                tick();
                i_surrogate_valid = 1'b0;
            end
    endtask

    task automatic send_feature_wave;
        @(negedge i_clk);
        i_feature_wave_mask = 8'hff;
        i_feature_wave_base = '0;
        i_feature_wave_q8_8_flat = '0;
        for (int feature = 0; feature < 8; feature++)
            i_feature_wave_q8_8_flat[feature*16 +: 16] =
                16'(16'h0100 + feature + o_graph_sample);
        i_feature_wave_version = o_graph_version;
        i_feature_wave_frame_index = 8'(o_graph_timestep);
        i_feature_wave_first = 1'b1;
        i_feature_wave_last = 1'b1;
        i_feature_wave_sample_reset = 1'b1;
        i_feature_wave_sample_last = 1'b1;
        i_feature_sample_label = o_graph_label;
        while (!o_feature_wave_ready) @(negedge i_clk);
        i_feature_wave_valid = 1'b1;
        tick();
        i_feature_wave_valid = 1'b0;
    endtask

    task automatic send_logit_wave;
        @(negedge i_clk);
        i_logit_wave_mask = 12'h007;
        i_logit_wave_base = '0;
        i_logit_wave_count = 2'd3;
        i_logit_wave_q8_8_flat = '0;
        i_logit_wave_version = o_graph_version;
        i_logit_wave_frame_index = 8'(o_graph_timestep);
        i_logit_wave_frame_last = 1'b1;
        i_logit_wave_sample_last = 1'b1;
        while (!o_logit_wave_ready) @(negedge i_clk);
        i_logit_wave_valid = 1'b1;
        tick();
        i_logit_wave_valid = 1'b0;
    endtask

    task automatic finish_graph;
        @(negedge i_clk);
        i_graph_done = 1'b1;
        tick();
        i_graph_done = 1'b0;
    endtask

    task automatic run_graph(input logic emit_observations);
        @(negedge i_clk);
        while (!o_graph_start_valid) @(negedge i_clk);
        if (!o_graph_sample_last ||
                o_graph_label != i_labels_flat[o_graph_sample*2 +: 2])
            $fatal(1, "graph metadata mismatch");
        i_graph_start_ready = 1'b1;
        tick();
        i_graph_start_ready = 1'b0;
        if (emit_observations) begin
            if (!CLASSIFIER_ONLY)
                send_surrogates();
            send_feature_wave();
            send_logit_wave();
        end
        finish_graph();
    endtask

    initial begin
        i_rst_n = 1'b0;
        conv_write_count = 0;
        classifier_write_count = 0;
        i_clear_error = 1'b0;
        i_system_parameter_owner = 1'b1;
        i_system_learning_enable = 1'b1;
        i_system_profile_aligned = 1'b1;
        i_system_profile_id = PROFILE;
        i_request_valid = 1'b0;
        i_learning_enable = 1'b0;
        i_batch_count = 2;
        i_timestep = 1'b0;
        i_timestep_count = 2'd1;
        i_sample_version = 16'h6100;
        i_labels_flat = {2'd1, 2'd0};
        i_profile_id = PROFILE;
        i_conv_tensor_id = 8'h08;
        i_classifier_tensor_id = 8'h12;
        i_total_input_channels = 2'd2;
        i_total_output_channels = 2'd2;
        i_conv_parameter_base = CONV_BASE;
        i_conv_weight_q16_base = 0;
        i_classifier_parameter_base = CLASSIFIER_BASE;
        i_classifier_feature_count = 8;
        i_classifier_class_count = 3;
        i_classifier_trace_leak_valid = 1'b1;
        i_classifier_trace_leak_q8_8 = 16'sd0;
        i_conv_vote_increment = 1;
        i_conv_vote_threshold = 1;
        i_classifier_vote_increment = 1;
        i_classifier_vote_threshold = 1;
        i_code_min = -16'sd127;
        i_code_max = 16'sd127;
        i_context_load_valid = 1'b0;
        i_context_model_version = 16'd5;
        i_context_adapt_version = 16'd7;
        i_forward_start_valid = 1'b0;
        i_forward_done = 1'b0;
        i_restore_complete = 1'b0;
        i_conv_in_width = 4;
        i_conv_in_height = 4;
        i_conv_in_channels = 2;
        i_conv_padding = 1;
        i_conv_layer_tag = 7;
        i_target_feature_count = 8;
        i_target_trace_leak_q8_8 = 16'sd0;
        i_target_trace_state_reset = 1'b1;
        i_graph_start_ready = 1'b0;
        i_graph_done = 1'b0;
        i_graph_error = 1'b0;
        i_prescan_feature_response_valid = 1'b0;
        i_prescan_feature_service_error = 1'b0;
        i_surrogate_valid = 1'b0;
        i_feature_wave_valid = 1'b0;
        i_logit_wave_valid = 1'b0;
        load_valid = 1'b0;
        load_addr = '0;
        classifier_prime_valid = 1'b0;
        conv_context_valid = 1'b0;
        conv_context_load_weights = 1'b1;
        conv_output_channel_base = 10'd0;
        conv_output_channel_count = 5'd2;
        conv_input_channel_count = 2'd2;
        conv_context_pair = 1'b0;
        conv_parameter_base = 8'(CONV_BASE);
        conv_context_version = 16'd5;

        repeat (4) tick();
        i_rst_n = 1'b1;
        tick();
        load_parameters();
        load_versions();
        prime_classifier();
        request_conv_context(5);
        if (o_model_version != 5 || o_adapt_version != 7 ||
                !classifier_cache_valid || !active_classifier_pair ||
                active_parameter_version != 5 || !conv_cache_valid ||
                conv_cached_version != 5 || weight_fill_count == 0)
            $fatal(1, "initial state mismatch runtime=%0d/%0d classifier valid/pair/version=%0b/%0b/%0d conv valid/version=%0b/%0d fills=%0d plane busy/error=%0b/%0b",
                o_model_version, o_adapt_version, classifier_cache_valid,
                active_classifier_pair, active_parameter_version,
                conv_cache_valid, conv_cached_version, weight_fill_count,
                parameter_plane_busy, parameter_plane_error);

        request_batch(1'b1);
        run_graph(1'b1);
        run_graph(1'b1);
        if (!CLASSIFIER_ONLY) begin
            run_graph(1'b1);
            run_graph(1'b1);
        end
        while (!o_done) tick();
        if (graph_target_history[3:0] !=
                    (CLASSIFIER_ONLY ? 4'b0000 : 4'b0101) ||
                graph_sample_history[3:0] !=
                    (CLASSIFIER_ONLY ? 4'b0010 : 4'b1100) ||
                feature_request_count != (CLASSIFIER_ONLY ? 0 : 32) ||
                write_count == 0 ||
                commit_count != 1 || learning_start_count != 1 ||
                o_model_version != 6 || o_adapt_version != 8 ||
                active_parameter_version != 6 || !saw_conv_invalidate ||
                conv_cache_valid || parameter_plane_error ||
                o_sticky_error || o_recovery_required)
            $fatal(1, "learning graph/runtime integration mismatch graphs=%0d requests=%0d writes=%0d commits=%0d starts=%0d versions=%0d/%0d error=%0b",
                graph_count, feature_request_count, write_count,
                commit_count, learning_start_count, o_model_version,
                o_adapt_version, o_sticky_error);
        if (CLASSIFIER_ONLY &&
            (conv_write_count != 0 || classifier_write_count != write_count ||
             classifier_write_count == 0))
            $fatal(1, "classifier-only graph RMW coverage failed");

        request_conv_context(6);
        if (!conv_cache_valid || conv_cached_version != 6 ||
                active_parameter_version != o_model_version)
            $fatal(1, "new Conv version was not refilled after commit");

        begin
            int writes_before_inference;
            int commits_before_inference;
            int requests_before_inference;
            writes_before_inference = write_count;
            commits_before_inference = commit_count;
            requests_before_inference = feature_request_count;
            request_batch(1'b0);
            run_graph(1'b0);
            run_graph(1'b0);
            while (!o_done) tick();
            if (write_count != writes_before_inference ||
                    commit_count != commits_before_inference ||
                    feature_request_count != requests_before_inference ||
                    o_model_version != 6 || o_adapt_version != 8 ||
                    o_sticky_error)
                $fatal(1, "inference request performed learning work");
        end

        $display("PASS: SCNN graph request classifier_only=%0d performs selected-layer learning, commit/refill, and learning-off inference without reset; conv_writes=%0d classifier_writes=%0d",
            CLASSIFIER_ONLY, conv_write_count, classifier_write_count);
        $finish;
    end

    generate if (!CLASSIFIER_ONLY) begin : g_conv_timeout
    initial begin
        repeat (300000) tick();
        $fatal(1, "SCNN graph runtime integration timeout state_frontend=%0d state_batch=%0d runtime_active=%0b capture=%0b pending=%0b writes=%0d graphs=%0d error=%0b frontend_error=%0b runtime_error=%0b cfg/protocol/abort/boundary=%0b/%0b/%0b/%0b selected_cfg(cv/cl/joint)=%0b/%0b/%0b selected_error=%0b child=%0b observation_version=%0b internal_mod=%0b cv_protocol=%0b cv_parts(store/ctl/lr_cfg/lr_proto/lr_latched)=%0b/%0b/%0b/%0b/%0b lr_parts(tile/tensor/sign/update/request/store/internal/active/rmw)=%0b/%0b/%0b/%0b/%0b/%0b/%0b/%0b/%0b cv_state=%0d tile_cfg(v/r)=%0b/%0b tile_counts(in/out/base_in/base_out)=%0d/%0d/%0d/%0d cl_errors(obs/context/grad/param/proto)=%0b/%0b/%0b/%0b/%0b frontend_parts=%0b/%0b/%0b/%0b graph_target=%0b surrogate_v/r=%0b/%0b feature_v/r=%0b/%0b logit_v/r=%0b/%0b",
            dut.u_frontend.u_prep.state_q,
            dut.u_frontend.u_batch.state_q,
            dut.u_runtime.active_q, o_capture_complete,
            o_commit_pending, write_count, graph_count, o_sticky_error,
            dut.frontend_error, dut.runtime_error,
            dut.u_runtime.path_cfg_error,
            dut.u_runtime.path_protocol_error,
            dut.u_runtime.path_abort_request,
            dut.u_runtime.boundary_error,
            dut.u_runtime.u_selected_layers.cv_o_cfg_error,
            dut.u_runtime.u_selected_layers.cl_o_cfg_error,
            dut.u_runtime.u_selected_layers.joint_cfg_error,
            dut.u_runtime.u_selected_layers.error_q,
            dut.u_runtime.u_selected_layers.child_error,
            dut.u_runtime.u_selected_layers.observation_version_error,
            dut.u_runtime.u_selected_layers.internal_mod_error,
            dut.u_runtime.u_selected_layers.cv_o_protocol_error,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.st_o_protocol_error,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.ctl_error,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.lr_o_cfg_error,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.lr_o_protocol_error,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.learner_cfg_error_q,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.tile_protocol_error,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.gradient_tensor_error,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.gradient_sign_error,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.update_config_error,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.request_error,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.store_error,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.internal_error,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.tile_active !=
                dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.rmw_active,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_learner.rmw_busy,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.u_controller.state_q,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.ctl_tile_cfg_valid,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.lr_o_cfg_ready,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.ctl_input_count,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.ctl_output_count,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.ctl_input_base,
            dut.u_runtime.u_selected_layers.g_conv8_learning.u_cv.ctl_output_base,
            dut.u_runtime.u_selected_layers.cl_o_observation_error,
            dut.u_runtime.u_selected_layers.cl_o_context_error,
            dut.u_runtime.u_selected_layers.cl_o_gradient_overflow,
            dut.u_runtime.u_selected_layers.cl_o_parameter_error,
            dut.u_runtime.u_selected_layers.cl_o_protocol_error,
            dut.u_frontend.coord_error, dut.u_frontend.prep_error,
            dut.u_frontend.prescan_error, dut.u_frontend.router_error,
            o_graph_target, i_surrogate_valid,
            o_surrogate_source_ready, i_feature_wave_valid,
            o_feature_wave_ready, i_logit_wave_valid,
            o_logit_wave_ready);
    end
    end else begin : g_classifier_timeout
        initial begin
            repeat (300000) tick();
            $fatal(1, "classifier-only graph timeout graphs=%0d writes=%0d capture=%0b error=%0b load_ready=%0b prime_ready=%0b prime_done=%0b context_ready=%0b request_ready=%0b plane_error=%0b plane_busy=%0b frontend_busy=%0b runtime_busy=%0b prep=%0d batch=%0d runtime_active=%0b cfg_ready=%0b cfg_error=%0b graph_start=%0b prescan_req=%0b prescan_state=%0d feature_requests=%0d prescan_response=%0b",
                graph_count, write_count, o_capture_complete, o_sticky_error,
                load_ready, classifier_prime_ready, classifier_prime_done,
                o_context_load_ready, o_request_ready, parameter_plane_error,
                parameter_plane_busy, dut.frontend_busy, dut.runtime_busy,
                dut.u_frontend.u_prep.state_q, dut.u_frontend.u_batch.state_q,
                dut.u_runtime.active_q, dut.u_runtime.path_cfg_ready,
                dut.u_runtime.path_cfg_error, o_graph_start_valid,
                o_prescan_feature_request_valid, dut.u_frontend.u_prescan.state_q,
                feature_request_count, i_prescan_feature_response_valid);
        end
    end endgenerate
endmodule

`default_nettype wire
