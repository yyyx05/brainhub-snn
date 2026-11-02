`timescale 1ns/1ps
`default_nettype none

// Bounded real-data classifier transaction. The production H40/P4 array
// executes one DVS-Gesture B4xT4 forward window, the production TP client
// learns and atomically republishes the classifier cache, and the same array
// recomputes the same window from the updated inactive-pair publication.
module tb_unified_scnn_q015_real_b4t4_forward_learn_recompute_v1;
    localparam int BATCH = 4;
    localparam int TIMESTEPS = 4;
    localparam int EFFECTIVE_BATCH = BATCH * TIMESTEPS;
    localparam int FEATURES = 2048;
    localparam int CLASSES = 11;
    localparam int PARAMS = FEATURES * CLASSES;
    localparam int PARAM_AW = $clog2(PARAMS);
    localparam int WEIGHT_DEPTH = 1156;
    localparam int WEIGHT_AW = $clog2(WEIGHT_DEPTH);
    localparam int FEATURE_VALUES = BATCH * TIMESTEPS * FEATURES;
    localparam int CURRENT_VALUES = BATCH * TIMESTEPS * CLASSES;
    localparam logic [19:0] CLASSIFIER_TAG = 20'hc1a53;
    localparam logic [15:0] INITIAL_VERSION = 16'h9150;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic signed [15:0] warm_code_mem [0:PARAMS-1];
    logic signed [15:0] warm_residual_mem [0:PARAMS-1];
    logic signed [15:0] expected_code_mem [0:PARAMS-1];
    logic signed [15:0] expected_residual_mem [0:PARAMS-1];
    logic signed [15:0] feature_mem [0:FEATURE_VALUES-1];
    logic [7:0] label_mem [0:BATCH-1];
    logic signed [15:0] baseline_current_mem [0:CURRENT_VALUES-1];
    logic signed [15:0] baseline_logit_mem [0:CURRENT_VALUES-1];
    logic signed [15:0] updated_current_mem [0:CURRENT_VALUES-1];
    logic signed [15:0] updated_logit_mem [0:CURRENT_VALUES-1];

    logic parameter_load_valid = 1'b0;
    logic parameter_load_ready;
    logic [PARAM_AW-1:0] parameter_load_addr = '0;
    logic signed [15:0] parameter_load_code = '0;
    logic signed [15:0] parameter_load_residual = '0;
    logic cache_prime_valid = 1'b0;
    logic cache_prime_ready;
    logic cache_prime_done;
    logic learning_cfg_valid = 1'b0;
    logic learning_cfg_ready;
    logic learning_active;
    logic learning_enable = 1'b1;
    logic client_feature_valid = 1'b0;
    logic client_feature_ready;
    logic [7:0] client_feature_mask = '0;
    logic [10:0] client_feature_base = '0;
    logic [127:0] client_feature_flat = '0;
    logic [7:0] client_feature_frame = '0;
    logic client_feature_first = 1'b0;
    logic client_feature_last = 1'b0;
    logic client_feature_sample_reset = 1'b0;
    logic client_feature_sample_last = 1'b0;
    logic [3:0] client_sample_label = '0;

    logic [3:0] weight_fill_valid;
    logic [3:0] weight_fill_ready;
    logic [7:0] weight_fill_tile_flat;
    logic [3:0] weight_fill_bank;
    logic [4*WEIGHT_AW-1:0] weight_fill_addr_flat;
    logic [799:0] weight_fill_data_flat;
    logic inference_ready;
    logic inference_hold;
    logic active_pair;
    logic [15:0] active_version;
    logic timestep_learning_done;
    logic batch_learning_done;
    logic [31:0] commit_sequence;
    logic parameter_write_fire;
    logic cache_publish_busy;
    logic cache_valid;
    logic client_busy;
    logic client_error;

    logic scheduler_start_valid = 1'b0;
    logic scheduler_start_ready;
    logic scheduler_feature_valid = 1'b0;
    logic scheduler_feature_ready;
    logic [7:0] scheduler_feature_mask = '0;
    logic [10:0] scheduler_feature_base = '0;
    logic [127:0] scheduler_feature_flat = '0;
    logic scheduler_feature_first = 1'b0;
    logic scheduler_feature_last = 1'b0;
    logic command_valid;
    logic command_ready;
    logic [15:0] command_neuron_count;
    logic command_first;
    logic command_last;
    logic [3:0] command_source_valid;
    logic signed [17:0] command_factor_b;
    logic [5:0] command_spatial_count;
    logic [9:0] command_output_channel_base;
    logic [39:0] command_feature_mask [0:3];
    logic command_active_pair;
    logic [3:0] command_weight_subword [0:3];
    logic [WEIGHT_AW-1:0] command_weight_addr [0:3];
    logic [8:0] command_work_base;
    logic [15:0] command_leak;
    logic [31:0] command_threshold;
    logic command_reset_to_zero;
    logic [15:0] command_trace_decay;
    logic [51:0] command_tag;
    logic scheduler_done;
    logic scheduler_busy;
    logic scheduler_error;

    logic [3:0] observation_valid;
    logic [3:0] observation_ready;
    logic [15:0] observation_tile_flat;
    logic [7:0] observation_mode_flat;
    logic [39:0] observation_lane_mask_flat;
    logic [39:0] observation_range_mask_flat;
    logic [39:0] observation_spike_flat;
    logic [1279:0] observation_pre_state_flat;
    logic [1279:0] observation_trace_next_flat;
    logic [255:0] observation_meta_flat;
    logic array_done;
    logic array_busy;
    logic array_idle;
    logic [15:0] tile_done;
    logic [15:0] tile_idle;
    logic [15:0] tile_error;
    logic array_error;

    logic scalar_valid;
    logic scalar_ready;
    logic [19:0] scalar_layer_tag;
    logic [9:0] scalar_channel;
    logic [31:0] scalar_spatial;
    logic scalar_spike;
    logic signed [15:0] scalar_pre_state;
    logic signed [15:0] scalar_trace_next;
    logic scalar_last;
    logic serializer_busy;
    logic serializer_error;

    logic adapter_cfg_valid = 1'b0;
    logic adapter_cfg_ready;
    logic adapter_release = 1'b0;
    logic adapter_release_ready;
    logic adapter_logit_valid;
    logic adapter_logit_ready;
    logic [11:0] adapter_logit_mask;
    logic [3:0] adapter_logit_base;
    logic [3:0] adapter_logit_count;
    logic [191:0] adapter_logit_flat;
    logic [15:0] adapter_logit_version;
    logic [7:0] adapter_logit_frame;
    logic adapter_logit_frame_last;
    logic adapter_logit_sample_last;
    logic adapter_capture_complete;
    logic adapter_idle;
    logic adapter_error;
    logic client_logit_ready;

    logic learning_phase = 1'b1;
    integer current_sample = 0;
    integer current_timestep = 0;
    logic checking_updated = 1'b0;
    logic [CLASSES-1:0] class_seen = '0;
    integer scalar_count = 0;
    integer logit_count_seen = 0;
    integer parameter_write_count = 0;
    integer batch_done_count = 0;
    integer timestep_done_count = 0;
    integer scheduler_done_count = 0;
    integer total_command_count = 0;
    integer baseline_scalar_total = 0;
    integer updated_scalar_total = 0;
    integer changed_current_count = 0;
    integer changed_logit_count = 0;
    integer class0_pair1_row0_fill_count = 0;
    integer prime_fill_count = 0;
    logic checking_prime_fill = 1'b0;
    longint unsigned cycle_count = 0;

    // Read-only measurement buckets: load, prime, forward/capture,
    // learn/publish wait, updated forward, final checks. They include TB gaps.
    int perf_phase = 0;
    typedef struct packed {
        longint unsigned cycles, array_busy_cycles, command_stall_cycles;
        longint unsigned p4_commands, p4_masked_terms, zero_term_commands;
        longint unsigned feature_waves, feature_values, nonzero_features;
        longint unsigned capture_waves, parameter_loads, cache_fill_words;
        longint unsigned rmw_reads, rmw_results, code_writes, residual_writes;
        longint unsigned publisher_reads, scalar_outputs, logit_rows;
    } perf_counters_t;
    perf_counters_t perf [0:5];
    longint unsigned perf_first_feature_cycle = 0;
    logic perf_frame_open = 1'b0;

    always @(posedge clk) begin : measure_only
        int terms;
        if (!rst_n) begin
            for (int phase = 0; phase < 6; phase++) perf[phase] = '0;
            perf_frame_open = 1'b0;
        end else begin
            perf[perf_phase].cycles++;
            if (array_busy) perf[perf_phase].array_busy_cycles++;
            if (command_valid && !command_ready)
                perf[perf_phase].command_stall_cycles++;
            if (command_valid && command_ready) begin
                terms = 0;
                for (int source = 0; source < 4; source++)
                    terms += $countones(command_feature_mask[source]);
                perf[perf_phase].p4_commands++;
                // Classifier spatial_count=1: masks denote input terms,
                // multiplied across active classes, not all 480 state lanes.
                perf[perf_phase].p4_masked_terms += terms * command_neuron_count;
                if (terms == 0) perf[perf_phase].zero_term_commands++;
            end
            if (scheduler_feature_valid && scheduler_feature_ready) begin
                perf[perf_phase].feature_waves++;
                for (int lane = 0; lane < 8; lane++) begin
                    if (scheduler_feature_mask[lane]) begin
                        perf[perf_phase].feature_values++;
                        if (scheduler_feature_flat[lane*16 +: 16] != 0)
                            perf[perf_phase].nonzero_features++;
                    end
                end
                if (scheduler_feature_first) begin
                    perf_first_feature_cycle = cycle_count;
                    perf_frame_open = 1'b1;
                end
            end
            if (client_feature_valid && client_feature_ready)
                perf[perf_phase].capture_waves++;
            if (parameter_load_valid && parameter_load_ready)
                perf[perf_phase].parameter_loads++;
            for (int region = 0; region < 4; region++)
                if (weight_fill_valid[region] && weight_fill_ready[region])
                    perf[perf_phase].cache_fill_words++;
            if (u_client.bridge_read_valid && u_client.bridge_read_ready)
                perf[perf_phase].rmw_reads++;
            if (u_client.bridge_result_valid && u_client.bridge_result_ready) begin
                perf[perf_phase].rmw_results++;
                if (u_client.bridge_write_code_enable)
                    perf[perf_phase].code_writes++;
                if (u_client.bridge_write_residual_enable)
                    perf[perf_phase].residual_writes++;
            end
            if (u_client.store_forward_read_valid &&
                    u_client.store_forward_read_ready)
                perf[perf_phase].publisher_reads++;
            if (scalar_valid && scalar_ready) perf[perf_phase].scalar_outputs++;
            if (adapter_logit_valid && adapter_logit_ready) begin
                perf[perf_phase].logit_rows++;
                if (!perf_frame_open) $fatal(1, "metric frame has no accepted input");
                $display("SUPPLEMENT_FRAME updated=%0d sample=%0d timestep=%0d input_to_logit_cycles=%0d",
                    checking_updated, current_sample, current_timestep,
                    cycle_count - perf_first_feature_cycle + 1);
                perf_frame_open = 1'b0;
            end
        end
    end

    final begin
        for (int phase = 0; phase < 6; phase++) begin
            $display("SUPPLEMENT_PHASE phase=%0d cycles=%0d array_busy_cycles=%0d command_stall_cycles=%0d p4_commands=%0d p4_masked_terms=%0d zero_term_commands=%0d feature_waves=%0d feature_values=%0d nonzero_features=%0d capture_waves=%0d parameter_loads=%0d cache_fill_words=%0d rmw_reads=%0d rmw_results=%0d code_writes=%0d residual_writes=%0d publisher_reads=%0d scalar_outputs=%0d logit_rows=%0d",
                phase, perf[phase].cycles, perf[phase].array_busy_cycles,
                perf[phase].command_stall_cycles, perf[phase].p4_commands,
                perf[phase].p4_masked_terms, perf[phase].zero_term_commands,
                perf[phase].feature_waves, perf[phase].feature_values,
                perf[phase].nonzero_features, perf[phase].capture_waves,
                perf[phase].parameter_loads, perf[phase].cache_fill_words,
                perf[phase].rmw_reads, perf[phase].rmw_results,
                perf[phase].code_writes, perf[phase].residual_writes,
                perf[phase].publisher_reads, perf[phase].scalar_outputs,
                perf[phase].logit_rows);
        end
    end

    assign adapter_logit_ready = learning_phase ? client_logit_ready : 1'b1;
    assign command_factor_b = command_source_valid == 0 ? 18'sd0 :
        18'(integer'(command_source_valid) * 64);

    unified_scnn_classifier_online_cache_learning_client_v1 #(
        .MAX_BATCH(EFFECTIVE_BATCH), .MAX_TIMESTEPS(TIMESTEPS),
        .CLASSIFIER_CAPTURE_MAX_BATCH(BATCH),
        .MAX_FEATURES(FEATURES), .MAX_CLASSES(CLASSES),
        .PARAM_DEPTH(PARAMS), .PARAM_ADDR_WIDTH(PARAM_AW),
        .TAG_WIDTH(16), .VERSION_WIDTH(16), .FRAME_WIDTH(8),
        .ARRAY_WEIGHT_ADDR_WIDTH(WEIGHT_AW),
        .BATCH_INDEX_WIDTH(2), .LEARNER_BATCH_INDEX_WIDTH(4),
        .BATCH_COUNT_WIDTH(5), .TIMESTEP_INDEX_WIDTH(2),
        .TIMESTEP_COUNT_WIDTH(3), .FEATURE_INDEX_WIDTH(11),
        .FEATURE_COUNT_WIDTH(12), .CLASS_INDEX_WIDTH(4),
        .CLASS_COUNT_WIDTH(4), .ENABLE_COMPUTED_TRACE_CAPTURE(1'b0)
    ) u_client (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(1'b0),
        .i_clear_error(1'b0),
        .i_parameter_load_valid(parameter_load_valid),
        .o_parameter_load_ready(parameter_load_ready),
        .i_parameter_load_addr(parameter_load_addr),
        .i_parameter_load_code(parameter_load_code),
        .i_parameter_load_residual(parameter_load_residual),
        .i_cache_prime_valid(cache_prime_valid),
        .o_cache_prime_ready(cache_prime_ready),
        .i_initial_pair(1'b1), .i_initial_version(INITIAL_VERSION),
        .o_cache_prime_done(cache_prime_done),
        .i_cfg_valid(learning_cfg_valid), .o_cfg_ready(learning_cfg_ready),
        .i_cfg_profile_id(8'h20), .i_cfg_tensor_id(8'h0c),
        .i_cfg_parameter_base('0), .i_cfg_batch_count(5'(BATCH)),
        .i_cfg_timestep_base('0),
        .i_cfg_timestep_count(3'(TIMESTEPS)),
        .i_cfg_trace_state_reset(1'b1),
        .i_cfg_feature_count(12'(FEATURES)),
        .i_cfg_class_count(4'(CLASSES)),
        .i_cfg_trace_leak_valid(1'b1),
        .i_cfg_trace_leak_q8_8(16'sd250),
        .i_cfg_computed_trace_capture_enable(1'b0),
        .i_cfg_vote_increment(16'd0), .i_cfg_vote_threshold(16'h8000),
        .i_cfg_code_min(16'sh8000), .i_cfg_code_max(16'sh7fff),
        .o_active(learning_active), .i_release(1'b0),
        .o_release_ready(), .i_learning_enable(learning_enable),
        .i_feature_trace_wave_valid(client_feature_valid),
        .o_feature_trace_wave_ready(client_feature_ready),
        .i_feature_trace_wave_mask(client_feature_mask),
        .i_feature_trace_wave_base(client_feature_base),
        .i_feature_trace_wave_q8_8_flat(client_feature_flat),
        .i_feature_trace_wave_version(active_version),
        .i_feature_trace_wave_frame_index(client_feature_frame),
        .i_feature_trace_wave_first(client_feature_first),
        .i_feature_trace_wave_last(client_feature_last),
        .i_feature_trace_wave_sample_reset(client_feature_sample_reset),
        .i_feature_trace_wave_sample_last(client_feature_sample_last),
        .i_sample_label(client_sample_label),
        .i_logit_wave_valid(adapter_logit_valid && learning_phase),
        .o_logit_wave_ready(client_logit_ready),
        .i_logit_wave_mask(adapter_logit_mask),
        .i_logit_wave_base(adapter_logit_base),
        .i_logit_wave_count(adapter_logit_count),
        .i_logit_wave_q8_8_flat(adapter_logit_flat),
        .i_logit_wave_version(adapter_logit_version),
        .i_logit_wave_frame_index(adapter_logit_frame),
        .i_logit_wave_frame_last(adapter_logit_frame_last),
        .i_logit_wave_sample_last(adapter_logit_sample_last),
        .i_array_idle(array_idle),
        .o_weight_fill_valid(weight_fill_valid),
        .i_weight_fill_ready(weight_fill_ready),
        .o_weight_fill_tile_flat(weight_fill_tile_flat),
        .o_weight_fill_bank(weight_fill_bank),
        .o_weight_fill_addr_flat(weight_fill_addr_flat),
        .o_weight_fill_data_flat(weight_fill_data_flat),
        .o_inference_ready(inference_ready),
        .o_inference_hold(inference_hold), .o_active_pair(active_pair),
        .o_active_version(active_version),
        .o_timestep_learning_done(timestep_learning_done),
        .o_batch_learning_done(batch_learning_done),
        .o_commit_sequence(commit_sequence),
        .o_parameter_write_fire(parameter_write_fire),
        .o_cache_publish_busy(cache_publish_busy),
        .o_cache_valid(cache_valid), .o_busy(client_busy),
        .o_sticky_error(client_error),
        .o_computed_trace_valid(), .i_computed_trace_ready(1'b0),
        .o_computed_trace_sample_id(), .o_computed_trace_index(),
        .o_computed_trace_timestep(), .o_computed_trace_frame_index(),
        .o_computed_trace_version(), .o_computed_trace_q8_8()
    );

    unified_scnn_h40_classifier_p4_scheduler_v1 #(
        .MAX_FEATURES(FEATURES), .MAX_CLASSES(CLASSES), .LANES(8),
        .WEIGHT_ADDR_WIDTH(WEIGHT_AW), .WORK_ADDR_WIDTH(9),
        .FEATURE_WIDTH(11), .FEATURE_COUNT_WIDTH(12),
        .CLASS_COUNT_WIDTH(4)
    ) u_scheduler (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(1'b0),
        .i_clear_error(1'b0), .i_start_valid(scheduler_start_valid),
        .o_start_ready(scheduler_start_ready),
        .i_feature_count(12'(FEATURES)),
        .i_class_count(4'(CLASSES)), .i_active_pair(active_pair),
        .i_work_base_addr(9'd20), .i_trace_decay(16'd250),
        .i_layer_tag(CLASSIFIER_TAG),
        .i_feature_wave_valid(scheduler_feature_valid),
        .o_feature_wave_ready(scheduler_feature_ready),
        .i_feature_wave_mask(scheduler_feature_mask),
        .i_feature_wave_base(scheduler_feature_base),
        .i_feature_wave_q8_8_flat(scheduler_feature_flat),
        .i_feature_wave_first(scheduler_feature_first),
        .i_feature_wave_last(scheduler_feature_last),
        .o_command_valid(command_valid), .i_command_ready(command_ready),
        .o_command_neuron_count(command_neuron_count),
        .o_command_first(command_first), .o_command_last(command_last),
        .o_command_source_valid(command_source_valid),
        .o_command_spatial_count(command_spatial_count),
        .o_command_output_channel_base(command_output_channel_base),
        .o_command_feature_mask0(command_feature_mask[0]),
        .o_command_feature_mask1(command_feature_mask[1]),
        .o_command_feature_mask2(command_feature_mask[2]),
        .o_command_feature_mask3(command_feature_mask[3]),
        .o_command_active_pair(command_active_pair),
        .o_command_weight_subword0(command_weight_subword[0]),
        .o_command_weight_subword1(command_weight_subword[1]),
        .o_command_weight_subword2(command_weight_subword[2]),
        .o_command_weight_subword3(command_weight_subword[3]),
        .o_command_weight_addr0(command_weight_addr[0]),
        .o_command_weight_addr1(command_weight_addr[1]),
        .o_command_weight_addr2(command_weight_addr[2]),
        .o_command_weight_addr3(command_weight_addr[3]),
        .o_command_work_base_addr(command_work_base),
        .o_command_leak(command_leak),
        .o_command_threshold(command_threshold),
        .o_command_reset_to_zero(command_reset_to_zero),
        .o_command_trace_decay(command_trace_decay),
        .o_command_transaction_tag(command_tag),
        .i_command_done(array_done), .i_child_error(array_error),
        .o_done(scheduler_done), .o_busy(scheduler_busy),
        .o_sticky_error(scheduler_error)
    );

    unified_binary_p4_h40_array12_path_v1 #(
        .SCNN_WEIGHT_FRACTIONAL_BITS(15),
        .WEIGHT_DEPTH(WEIGHT_DEPTH), .WEIGHT_ADDR_WIDTH(WEIGHT_AW)
    ) u_array_path (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(1'b0),
        .i_clear_error(1'b0),
        .i_command_valid(command_valid), .o_command_ready(command_ready),
        .i_mode(2'd2), .i_neuron_count(command_neuron_count),
        .i_first(command_first), .i_last(command_last),
        .i_source_valid(command_source_valid),
        .i_scnn_spatial_count(command_spatial_count),
        .i_scnn_output_channel_base(command_output_channel_base),
        .i_scnn_feature_mask0(command_feature_mask[0]),
        .i_scnn_feature_mask1(command_feature_mask[1]),
        .i_scnn_feature_mask2(command_feature_mask[2]),
        .i_scnn_feature_mask3(command_feature_mask[3]),
        .i_scnn_factor_b(command_factor_b),
        .i_scnn_active_pair(command_active_pair),
        .i_scnn_weight_subword0(command_weight_subword[0]),
        .i_scnn_weight_subword1(command_weight_subword[1]),
        .i_scnn_weight_subword2(command_weight_subword[2]),
        .i_scnn_weight_subword3(command_weight_subword[3]),
        .i_scnn_work_base_addr(command_work_base),
        .i_destination_tile_base(4'd0),
        .i_weight_addr0(command_weight_addr[0]),
        .i_weight_addr1(command_weight_addr[1]),
        .i_weight_addr2(command_weight_addr[2]),
        .i_weight_addr3(command_weight_addr[3]),
        .i_leak(command_leak), .i_threshold(command_threshold),
        .i_reset_to_zero(command_reset_to_zero),
        .i_trace_decay(command_trace_decay),
        .i_transaction_tag(command_tag),
        .i_weight_fill_valid(weight_fill_valid),
        .o_weight_fill_ready(weight_fill_ready),
        .i_weight_fill_tile_flat(weight_fill_tile_flat),
        .i_weight_fill_bank(weight_fill_bank),
        .i_weight_fill_addr_flat(weight_fill_addr_flat),
        .i_weight_fill_data_flat(weight_fill_data_flat),
        .o_observation_valid(observation_valid),
        .i_observation_ready(observation_ready),
        .o_observation_tile_flat(observation_tile_flat),
        .o_observation_mode_flat(observation_mode_flat),
        .o_observation_lane_mask_flat(observation_lane_mask_flat),
        .o_observation_range_mask_flat(observation_range_mask_flat),
        .o_observation_spike_flat(observation_spike_flat),
        .o_observation_pre_state_flat(observation_pre_state_flat),
        .o_observation_trace_next_flat(observation_trace_next_flat),
        .o_observation_meta_flat(observation_meta_flat),
        .i_maintenance_tile('0),
        .i_state_maintenance_write_valid(1'b0),
        .o_state_maintenance_write_ready(),
        .i_state_maintenance_write_addr('0),
        .i_state_maintenance_write_data('0),
        .i_state_maintenance_write_mask('0),
        .i_state_maintenance_read_valid(1'b0),
        .o_state_maintenance_read_ready(),
        .i_state_maintenance_read_addr('0),
        .o_state_maintenance_result_valid(),
        .i_state_maintenance_result_ready(1'b1),
        .o_state_maintenance_result_addr(),
        .o_state_maintenance_read_data(),
        .i_trace_maintenance_write_valid(1'b0),
        .o_trace_maintenance_write_ready(),
        .i_trace_maintenance_write_addr('0),
        .i_trace_maintenance_write_data('0),
        .i_trace_maintenance_write_mask('0),
        .i_trace_maintenance_read_valid(1'b0),
        .o_trace_maintenance_read_ready(),
        .i_trace_maintenance_read_addr('0),
        .o_trace_maintenance_result_valid(),
        .i_trace_maintenance_result_ready(1'b1),
        .o_trace_maintenance_result_addr(),
        .o_trace_maintenance_read_data(),
        .o_busy(array_busy), .o_done(array_done), .o_idle(array_idle),
        .o_tile_done(tile_done), .o_tile_idle(tile_idle),
        .o_tile_sticky_error(tile_error), .o_sticky_error(array_error)
    );

    unified_scnn_h40_tp_observation_serializer_v1 #(
        .PHYSICAL_TILE_COUNT(12)
    ) u_serializer (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(1'b0),
        .i_clear_error(1'b0),
        .i_observation_valid(observation_valid),
        .o_observation_ready(observation_ready),
        .i_observation_tile_flat(observation_tile_flat),
        .i_observation_mode_flat(observation_mode_flat),
        .i_observation_lane_mask_flat(observation_lane_mask_flat),
        .i_observation_range_mask_flat(observation_range_mask_flat),
        .i_observation_spike_flat(observation_spike_flat),
        .i_observation_pre_state_flat(observation_pre_state_flat),
        .i_observation_trace_next_flat(observation_trace_next_flat),
        .i_observation_meta_flat(observation_meta_flat),
        .o_scalar_valid(scalar_valid), .i_scalar_ready(scalar_ready),
        .o_scalar_layer_tag(scalar_layer_tag),
        .o_scalar_channel(scalar_channel),
        .o_scalar_spatial(scalar_spatial), .o_scalar_spike(scalar_spike),
        .o_scalar_pre_state_q8_8(scalar_pre_state),
        .o_scalar_trace_next_q8_8(scalar_trace_next),
        .o_scalar_last(scalar_last), .o_busy(serializer_busy),
        .o_sticky_error(serializer_error)
    );

    unified_scnn_h40_classifier_logit_wave_adapter_v1 #(
        .MAX_CLASSES(CLASSES), .ENGINES(12),
        .ACCUMULATE_SAMPLE_LOGITS(1'b1),
        .VERSION_WIDTH(16), .FRAME_WIDTH(8),
        .CLASS_WIDTH(4), .CLASS_COUNT_WIDTH(4)
    ) u_logit_adapter (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(1'b0),
        .i_clear_error(1'b0), .i_cfg_valid(adapter_cfg_valid),
        .o_cfg_ready(adapter_cfg_ready), .i_cfg_layer_tag(CLASSIFIER_TAG),
        .i_cfg_class_count(4'(CLASSES)), .i_cfg_version(active_version),
        .i_cfg_frame_index(8'(current_timestep)),
        .i_cfg_sample_reset(current_timestep == 0),
        .i_cfg_sample_last(current_timestep == TIMESTEPS-1),
        .i_release(adapter_release),
        .o_release_ready(adapter_release_ready),
        .i_scalar_valid(scalar_valid), .o_scalar_ready(scalar_ready),
        .i_scalar_layer_tag(scalar_layer_tag),
        .i_scalar_class(scalar_channel),
        .i_scalar_spatial(scalar_spatial),
        .i_scalar_logit_q8_8(scalar_pre_state),
        .o_logit_wave_valid(adapter_logit_valid),
        .i_logit_wave_ready(adapter_logit_ready),
        .o_logit_wave_mask(adapter_logit_mask),
        .o_logit_wave_base(adapter_logit_base),
        .o_logit_wave_count(adapter_logit_count),
        .o_logit_wave_q8_8_flat(adapter_logit_flat),
        .o_logit_wave_version(adapter_logit_version),
        .o_logit_wave_frame_index(adapter_logit_frame),
        .o_logit_wave_frame_last(adapter_logit_frame_last),
        .o_logit_wave_sample_last(adapter_logit_sample_last),
        .o_capture_complete(adapter_capture_complete),
        .o_idle(adapter_idle), .o_protocol_error(adapter_error)
    );

    task automatic load_parameters;
        for (int addr = 0; addr < PARAMS; addr++) begin
            @(negedge clk);
            while (!parameter_load_ready) @(negedge clk);
            parameter_load_addr = PARAM_AW'(addr);
            parameter_load_code = warm_code_mem[addr];
            parameter_load_residual = warm_residual_mem[addr];
            parameter_load_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            parameter_load_valid = 1'b0;
        end
    endtask

    task automatic prime_cache;
        int watchdog;
        @(negedge clk);
        while (!cache_prime_ready) @(negedge clk);
        cache_prime_valid = 1'b1;
        @(posedge clk);
        @(negedge clk);
        cache_prime_valid = 1'b0;
        watchdog = 0;
        while (!cache_prime_done && watchdog < 200000) begin
            @(posedge clk);
            watchdog++;
        end
        if (!cache_prime_done || !cache_valid || active_pair != 1'b1 ||
                active_version != INITIAL_VERSION || !inference_ready)
            $fatal(1,
                "initial cache prime failed done=%0b valid=%0b pair=%0b version=%h ready=%0b",
                cache_prime_done, cache_valid, active_pair, active_version,
                inference_ready);
    endtask

    task automatic configure_learning;
        @(negedge clk);
        while (!learning_cfg_ready) @(negedge clk);
        learning_cfg_valid = 1'b1;
        @(posedge clk);
        @(negedge clk);
        learning_cfg_valid = 1'b0;
        while (!learning_active) @(posedge clk);
    endtask

    task automatic send_client_feature_wave(
        input int sample_index,
        input int timestep_index,
        input int wave_index
    );
        int base;
        int row_base;
        base = wave_index * 8;
        row_base = (sample_index * TIMESTEPS + timestep_index) * FEATURES;
        @(negedge clk);
        client_feature_mask = 8'hff;
        client_feature_base = 11'(base);
        client_feature_flat = '0;
        for (int lane = 0; lane < 8; lane++)
            client_feature_flat[lane*16 +: 16] =
                feature_mem[row_base + base + lane];
        client_feature_frame = 8'(timestep_index);
        client_feature_first = wave_index == 0;
        client_feature_last = wave_index == FEATURES/8-1;
        client_feature_sample_reset =
            timestep_index == 0 && wave_index == 0;
        client_feature_sample_last =
            timestep_index == TIMESTEPS-1 && wave_index == FEATURES/8-1;
        client_sample_label = label_mem[sample_index][3:0];
        client_feature_valid = 1'b1;
        do @(posedge clk); while (!client_feature_ready);
        @(negedge clk);
        client_feature_valid = 1'b0;
    endtask

    task automatic send_scheduler_feature_wave(
        input int sample_index,
        input int timestep_index,
        input int wave_index
    );
        int base;
        int row_base;
        base = wave_index * 8;
        row_base = (sample_index * TIMESTEPS + timestep_index) * FEATURES;
        @(negedge clk);
        scheduler_feature_mask = 8'hff;
        scheduler_feature_base = 11'(base);
        scheduler_feature_flat = '0;
        for (int lane = 0; lane < 8; lane++)
            scheduler_feature_flat[lane*16 +: 16] =
                feature_mem[row_base + base + lane];
        scheduler_feature_first = wave_index == 0;
        scheduler_feature_last = wave_index == FEATURES/8-1;
        scheduler_feature_valid = 1'b1;
        do @(posedge clk); while (!scheduler_feature_ready);
        @(negedge clk);
        scheduler_feature_valid = 1'b0;
    endtask

    task automatic run_frame(
        input int sample_index,
        input int timestep_index,
        input logic updated_phase
    );
        int watchdog;
        int expected_scheduler_done;
        current_sample = sample_index;
        current_timestep = timestep_index;
        checking_updated = updated_phase;
        class_seen = '0;
        scalar_count = 0;
        logit_count_seen = 0;
        expected_scheduler_done = scheduler_done_count + 1;

        if (!updated_phase) begin
            for (int wave = 0; wave < FEATURES/8; wave++)
                send_client_feature_wave(sample_index, timestep_index, wave);
        end

        @(negedge clk);
        while (!adapter_cfg_ready) @(negedge clk);
        adapter_cfg_valid = 1'b1;
        @(posedge clk);
        @(negedge clk);
        adapter_cfg_valid = 1'b0;

        while (!scheduler_start_ready || !inference_ready) @(negedge clk);
        scheduler_start_valid = 1'b1;
        @(posedge clk);
        @(negedge clk);
        scheduler_start_valid = 1'b0;
        for (int wave = 0; wave < FEATURES/8; wave++)
            send_scheduler_feature_wave(sample_index, timestep_index, wave);

        watchdog = 0;
        while ((scheduler_done_count < expected_scheduler_done ||
                !adapter_capture_complete) &&
                watchdog < 200000) begin
            @(posedge clk);
            watchdog++;
        end
        @(negedge clk);
        if (watchdog >= 200000 || scalar_count != CLASSES ||
                logit_count_seen != CLASSES ||
                class_seen != {CLASSES{1'b1}} || scheduler_busy ||
                array_busy || serializer_busy)
            $fatal(1,
                "frame incomplete sample=%0d timestep=%0d updated=%0b scalars=%0d logits=%0d seen=%h watchdog=%0d",
                sample_index, timestep_index, updated_phase, scalar_count,
                logit_count_seen, class_seen, watchdog);

        while (!adapter_release_ready) @(negedge clk);
        adapter_release = 1'b1;
        @(posedge clk);
        @(negedge clk);
        adapter_release = 1'b0;
    endtask

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            cycle_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (parameter_write_fire)
                parameter_write_count <= parameter_write_count + 1;
            if (batch_learning_done)
                batch_done_count <= batch_done_count + 1;
            if (timestep_learning_done)
                timestep_done_count <= timestep_done_count + 1;
            if (command_valid && command_ready)
                total_command_count <= total_command_count + 1;
            if (command_valid && command_ready &&
                    command_active_pair != active_pair)
                $fatal(1,
                    "classifier command pair mismatch command=%0b active=%0b",
                    command_active_pair, active_pair);
            if (scheduler_done)
                scheduler_done_count <= scheduler_done_count + 1;
            if (checking_prime_fill) begin
                for (int region = 0; region < 4; region++) begin
                    if (weight_fill_valid[region] &&
                            weight_fill_ready[region]) begin
                        int global_class;
                        int row_index;
                        int feature_index;
                        logic pair_index;
                        logic bank_parity;
                        global_class = region * 4 +
                            weight_fill_tile_flat[region*2 +: 2];
                        pair_index = weight_fill_addr_flat[
                            region*WEIGHT_AW + WEIGHT_AW-1];
                        row_index = weight_fill_addr_flat[
                            region*WEIGHT_AW +: WEIGHT_AW-1];
                        bank_parity = weight_fill_bank[region];
                        if (global_class >= CLASSES || !pair_index ||
                                row_index >= 86)
                            $fatal(1,
                                "prime fill metadata mismatch region=%0d class=%0d pair=%0b parity=%0b row=%0d",
                                region, global_class, pair_index,
                                bank_parity, row_index);
                        for (int subword = 0; subword < 12; subword++) begin
                            logic signed [15:0] expected_weight;
                            feature_index = row_index * 24 + subword * 2 +
                                bank_parity;
                            expected_weight = feature_index < FEATURES ?
                                warm_code_mem[global_class*FEATURES +
                                    feature_index] : 16'sd0;
                            if ($signed(weight_fill_data_flat[
                                    region*200 + subword*16 +: 16]) !==
                                    expected_weight)
                                $fatal(1,
                                    "prime fill mismatch region=%0d class=%0d pair=%0b parity=%0b row=%0d subword=%0d feature=%0d got=%0d expected=%0d",
                                    region, global_class, pair_index,
                                    bank_parity, row_index, subword,
                                    feature_index,
                                    $signed(weight_fill_data_flat[
                                        region*200 + subword*16 +: 16]),
                                    expected_weight);
                        end
                        prime_fill_count <= prime_fill_count + 1;
                    end
                end
            end
            if (weight_fill_valid[0] && weight_fill_ready[0] &&
                    weight_fill_tile_flat[1:0] == 0 &&
                    weight_fill_bank[0] == 0 &&
                    weight_fill_addr_flat[WEIGHT_AW-1:0] ==
                        WEIGHT_AW'(1 << (WEIGHT_AW-1))) begin
                for (int subword = 0; subword < 12; subword++) begin
                    if ($signed(weight_fill_data_flat[subword*16 +: 16]) !==
                            warm_code_mem[subword*2])
                        $fatal(1,
                            "prime fill mismatch subword=%0d got=%0d expected=%0d",
                            subword,
                            $signed(weight_fill_data_flat[subword*16 +: 16]),
                            warm_code_mem[subword*2]);
                end
                class0_pair1_row0_fill_count <=
                    class0_pair1_row0_fill_count + 1;
            end

            if (scalar_valid && scalar_ready) begin
                int index;
                logic signed [15:0] expected_current;
                index = (current_sample * TIMESTEPS + current_timestep) *
                    CLASSES + scalar_channel;
                expected_current = checking_updated ?
                    updated_current_mem[index] : baseline_current_mem[index];
                if (scalar_layer_tag != CLASSIFIER_TAG ||
                        scalar_spatial != 0 || scalar_channel >= CLASSES ||
                        class_seen[scalar_channel])
                    $fatal(1,
                        "classifier scalar metadata mismatch sample=%0d timestep=%0d class=%0d tag=%h spatial=%0d",
                        current_sample, current_timestep, scalar_channel,
                        scalar_layer_tag, scalar_spatial);
                if (scalar_pre_state !== expected_current)
                    $fatal(1,
                        "classifier current mismatch sample=%0d timestep=%0d class=%0d updated=%0b got=%0d expected=%0d",
                        current_sample, current_timestep, scalar_channel,
                        checking_updated, scalar_pre_state, expected_current);
                class_seen[scalar_channel] <= 1'b1;
                scalar_count <= scalar_count + 1;
                if (checking_updated) begin
                    updated_scalar_total <= updated_scalar_total + 1;
                    if (updated_current_mem[index] != baseline_current_mem[index])
                        changed_current_count <= changed_current_count + 1;
                end else begin
                    baseline_scalar_total <= baseline_scalar_total + 1;
                end
            end

            if (adapter_logit_valid && adapter_logit_ready) begin
                int row_base;
                int wave_changes;
                row_base = (current_sample * TIMESTEPS + current_timestep) *
                    CLASSES;
                wave_changes = 0;
                if (adapter_logit_mask != 12'h7ff ||
                        adapter_logit_base != 0 ||
                        adapter_logit_count != CLASSES ||
                        adapter_logit_version != active_version ||
                        adapter_logit_frame != current_timestep ||
                        !adapter_logit_frame_last ||
                        adapter_logit_sample_last !=
                            (current_timestep == TIMESTEPS-1))
                    $fatal(1,
                        "logit wave metadata mismatch sample=%0d timestep=%0d mask=%h base=%0d count=%0d version=%h active=%h",
                        current_sample, current_timestep, adapter_logit_mask,
                        adapter_logit_base, adapter_logit_count,
                        adapter_logit_version, active_version);
                for (int class_index = 0; class_index < CLASSES;
                     class_index++) begin
                    logic signed [15:0] expected_logit;
                    expected_logit = checking_updated ?
                        updated_logit_mem[row_base + class_index] :
                        baseline_logit_mem[row_base + class_index];
                    if ($signed(adapter_logit_flat[class_index*16 +: 16]) !==
                            expected_logit)
                        $fatal(1,
                            "temporal logit mismatch sample=%0d timestep=%0d class=%0d updated=%0b got=%0d expected=%0d",
                            current_sample, current_timestep, class_index,
                            checking_updated,
                            $signed(adapter_logit_flat[class_index*16 +: 16]),
                            expected_logit);
                    if (checking_updated &&
                            updated_logit_mem[row_base + class_index] !=
                            baseline_logit_mem[row_base + class_index])
                        wave_changes++;
                end
                if (checking_updated)
                    changed_logit_count <= changed_logit_count + wave_changes;
                logit_count_seen <= logit_count_seen + CLASSES;
            end

            if (client_error || scheduler_error || array_error ||
                    serializer_error || adapter_error || tile_error != 0)
                $fatal(1,
                    "production path error sample=%0d timestep=%0d client=%0b cfg=%0b capture=%0b context=%0b gradient=%0b parameter=%0b protocol=%0b publisher=%0b store=%0b scheduler=%0b array=%0b serializer=%0b adapter=%0b tile=%h",
                    current_sample, current_timestep,
                    client_error, u_client.bridge_cfg_error,
                    u_client.bridge_capture_error,
                    u_client.bridge_context_error,
                    u_client.bridge_gradient_overflow,
                    u_client.bridge_parameter_error,
                    u_client.bridge_protocol_error,
                    u_client.publisher_error, u_client.store_error,
                    scheduler_error, array_error, serializer_error,
                    adapter_error, tile_error);
        end
    end

    initial begin
        int watchdog;
        int writes_before_recompute;
        string fixture_dir;
        fixture_dir =
            "results/paper_evidence/rtl/scnn_q015_real_b4t4_forward_learn_recompute_20260928/fixture";
        $readmemh($sformatf("%s/warm_code_after_11_windows_q15.hex",
            fixture_dir), warm_code_mem);
        $readmemh($sformatf("%s/warm_residual_after_11_windows_q15.hex",
            fixture_dir), warm_residual_mem);
        $readmemh($sformatf("%s/expected_code_after_commit_q15.hex",
            fixture_dir), expected_code_mem);
        $readmemh($sformatf("%s/expected_residual_after_commit_q15.hex",
            fixture_dir), expected_residual_mem);
        $readmemh(
            "results/paper_evidence/rtl/scnn_q015_dataset_parity/b4t4_update_fixture/features_q8_8.hex",
            feature_mem);
        $readmemh(
            "results/paper_evidence/rtl/scnn_q015_dataset_parity/b4t4_update_fixture/labels.hex",
            label_mem);
        $readmemh($sformatf("%s/baseline_current_q8_8.hex", fixture_dir),
            baseline_current_mem);
        $readmemh($sformatf("%s/baseline_temporal_logits_q8_8.hex",
            fixture_dir), baseline_logit_mem);
        $readmemh($sformatf("%s/updated_current_q8_8.hex", fixture_dir),
            updated_current_mem);
        $readmemh($sformatf("%s/updated_temporal_logits_q8_8.hex",
            fixture_dir), updated_logit_mem);

        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        load_parameters();
        for (int addr = 0; addr < PARAMS; addr++) begin
            if (u_client.u_parameter_store.code_mem[addr] !==
                    warm_code_mem[addr] ||
                    u_client.u_parameter_store.residual_mem[addr] !==
                    warm_residual_mem[addr])
                $fatal(1,
                    "warm store load mismatch addr=%0d code=%0d/%0d residual=%0d/%0d",
                    addr, u_client.u_parameter_store.code_mem[addr],
                    warm_code_mem[addr],
                    u_client.u_parameter_store.residual_mem[addr],
                    warm_residual_mem[addr]);
        end
        checking_prime_fill = 1'b1;
        perf_phase = 1;
        prime_cache();
        checking_prime_fill = 1'b0;
        if (prime_fill_count != CLASSES * 2 * 86)
            $fatal(1, "prime fill count mismatch got=%0d expected=%0d",
                prime_fill_count, CLASSES * 2 * 86);
        for (int row = 0; row < 86; row++) begin
            for (int subword = 0; subword < 12; subword++) begin
                int even_feature;
                int odd_feature;
                logic signed [15:0] expected_even;
                logic signed [15:0] expected_odd;
                even_feature = row * 24 + subword * 2;
                odd_feature = even_feature + 1;
                expected_even = even_feature < FEATURES ?
                    warm_code_mem[even_feature] : 16'sd0;
                expected_odd = odd_feature < FEATURES ?
                    warm_code_mem[odd_feature] : 16'sd0;
                if ($signed(u_array_path.u_array.g_region[0].u_region.
                        g_tile[0].u_tile.g_p4_all_frontend.u_weight_store.
                        g_bank[2].memory[row][subword*16 +: 16]) !==
                        expected_even)
                    $fatal(1,
                        "class0 even cache mismatch row=%0d subword=%0d feature=%0d got=%0d expected=%0d",
                        row, subword, even_feature,
                        $signed(u_array_path.u_array.g_region[0].u_region.
                            g_tile[0].u_tile.g_p4_all_frontend.u_weight_store.
                            g_bank[2].memory[row][subword*16 +: 16]),
                        expected_even);
                if ($signed(u_array_path.u_array.g_region[0].u_region.
                        g_tile[0].u_tile.g_p4_all_frontend.u_weight_store.
                        g_bank[3].memory[row][subword*16 +: 16]) !==
                        expected_odd)
                    $fatal(1,
                        "class0 odd cache mismatch row=%0d subword=%0d feature=%0d got=%0d expected=%0d",
                        row, subword, odd_feature,
                        $signed(u_array_path.u_array.g_region[0].u_region.
                            g_tile[0].u_tile.g_p4_all_frontend.u_weight_store.
                            g_bank[3].memory[row][subword*16 +: 16]),
                        expected_odd);
            end
        end
        if (class0_pair1_row0_fill_count != 1)
            $fatal(1, "missing class0 pair1 row0 cache fill count=%0d",
                class0_pair1_row0_fill_count);
        perf_phase = 2;
        configure_learning();

        for (int sample_index = 0; sample_index < BATCH; sample_index++)
            for (int timestep_index = 0; timestep_index < TIMESTEPS;
                 timestep_index++)
                run_frame(sample_index, timestep_index, 1'b0);

        perf_phase = 3;
        watchdog = 0;
        while ((active_version != INITIAL_VERSION + 1'b1 ||
                cache_publish_busy || !inference_ready) &&
                watchdog < 5000000) begin
            @(posedge clk);
            watchdog++;
        end
        @(negedge clk);
        if (watchdog >= 5000000 || active_pair != 1'b0 ||
                active_version != INITIAL_VERSION + 1'b1 ||
                commit_sequence != 1 || batch_done_count != 1 ||
                parameter_write_count != 300 || !cache_valid ||
                inference_hold || client_busy)
            $fatal(1,
                "learn/commit/publication mismatch watchdog=%0d pair=%0b version=%h sequence=%0d batch_done=%0d writes=%0d hold=%0b busy=%0b",
                watchdog, active_pair, active_version, commit_sequence,
                batch_done_count, parameter_write_count, inference_hold,
                client_busy);

        for (int addr = 0; addr < PARAMS; addr++) begin
            if (u_client.u_parameter_store.code_mem[addr] !==
                    expected_code_mem[addr])
                $fatal(1,
                    "committed code mismatch addr=%0d got=%0d expected=%0d",
                    addr, u_client.u_parameter_store.code_mem[addr],
                    expected_code_mem[addr]);
            if (u_client.u_parameter_store.residual_mem[addr] !==
                    expected_residual_mem[addr])
                $fatal(1,
                    "committed residual mismatch addr=%0d got=%0d expected=%0d",
                    addr, u_client.u_parameter_store.residual_mem[addr],
                    expected_residual_mem[addr]);
        end

        learning_enable = 1'b0;
        learning_phase = 1'b0;
        perf_phase = 4;
        writes_before_recompute = parameter_write_count;
        for (int sample_index = 0; sample_index < BATCH; sample_index++)
            for (int timestep_index = 0; timestep_index < TIMESTEPS;
                 timestep_index++)
                run_frame(sample_index, timestep_index, 1'b1);

        perf_phase = 5;
        repeat (20) @(posedge clk);
        if (parameter_write_count != writes_before_recompute ||
                active_pair != 1'b0 ||
                active_version != INITIAL_VERSION + 1'b1 ||
                commit_sequence != 1 || changed_current_count != 16 ||
                changed_logit_count != 16 ||
                baseline_scalar_total != CURRENT_VALUES ||
                updated_scalar_total != CURRENT_VALUES || client_error)
            $fatal(1,
                "updated forward contract mismatch writes=%0d/%0d pair=%0b version=%h sequence=%0d changed_current=%0d changed_logit=%0d baseline=%0d updated=%0d",
                parameter_write_count, writes_before_recompute, active_pair,
                active_version, commit_sequence, changed_current_count,
                changed_logit_count, baseline_scalar_total,
                updated_scalar_total);

        $display(
            "SCNN_Q015_REAL_B4T4_FORWARD_LEARN_RECOMPUTE_PASS samples=4 timesteps=4 features=2048 classes=11 warm_windows=11 commit_window=12 code_changes=300 changed_currents=%0d changed_logits=%0d pair_flip=1 version_increment=1 p4_commands=%0d",
            changed_current_count, changed_logit_count, total_command_count);
        $finish;
    end

    initial begin
        repeat (12000000) @(posedge clk);
        $fatal(1,
            "real B4xT4 forward/learn/recompute timeout cycles=%0d sample=%0d timestep=%0d version=%h writes=%0d",
            cycle_count, current_sample, current_timestep, active_version,
            parameter_write_count);
    end
endmodule

`default_nettype wire
