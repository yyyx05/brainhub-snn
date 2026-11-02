`timescale 1ns/1ps
`default_nettype none

module tb_scnn_tail_visibility;
    localparam int TEST_TIMESTEPS = 2;
    localparam int WEIGHT_DEPTH = 1156;
    localparam int WEIGHT_AW = $clog2(WEIGHT_DEPTH);
    localparam int MEMORY_DEPTH = 1024;
    localparam int MEMORY_AW = $clog2(MEMORY_DEPTH);
    localparam int WORK_AW = $clog2(MEMORY_DEPTH / 2);
    localparam int FEATURE_DEPTH = 8192;
    localparam int FEATURE_AW = $clog2(FEATURE_DEPTH);
    localparam int CONV1_CHANNELS = 64;
    localparam int CONV1_SPATIAL = 32;
    localparam int CONV1_SCALARS = CONV1_CHANNELS * CONV1_SPATIAL *
        CONV1_SPATIAL;
    localparam int CONV2_CHANNELS = 128;
    localparam int CONV2_SPATIAL = 32;
    localparam int CONV2_SCALARS = CONV2_CHANNELS * CONV2_SPATIAL *
        CONV2_SPATIAL;
    localparam int CONV2_POOLED_SPATIAL = 16;
    localparam int CONV3_CHANNELS = 256;
    localparam int CONV3_SPATIAL = 16;
    localparam int CONV3_SCALARS = CONV3_CHANNELS * CONV3_SPATIAL *
        CONV3_SPATIAL;
    localparam int CONV4_CHANNELS = 256;
    localparam int CONV4_SPATIAL = 16;
    localparam int CONV4_SCALARS = CONV4_CHANNELS * CONV4_SPATIAL *
        CONV4_SPATIAL;
    localparam int CONV4_POOLED_SPATIAL = 8;
    localparam int CONV5_CHANNELS = 512;
    localparam int CONV5_SPATIAL = 8;
    localparam int CONV5_SCALARS = CONV5_CHANNELS * CONV5_SPATIAL *
        CONV5_SPATIAL;
    localparam int CONV6_CHANNELS = 512;
    localparam int CONV6_SPATIAL = 8;
    localparam int CONV6_SCALARS = CONV6_CHANNELS * CONV6_SPATIAL *
        CONV6_SPATIAL;
    localparam int CONV6_POOLED_SPATIAL = 4;
    localparam int CONV7_CHANNELS = 512;
    localparam int CONV7_SPATIAL = 4;
    localparam int CONV7_SCALARS = CONV7_CHANNELS * CONV7_SPATIAL *
        CONV7_SPATIAL;
    localparam int CONV8_CHANNELS = 512;
    localparam int CONV8_SPATIAL = 4;
    localparam int CONV8_SCALARS = CONV8_CHANNELS * CONV8_SPATIAL *
        CONV8_SPATIAL;
    localparam int CONV8_POOLED_SPATIAL = 2;
    localparam int CLASSIFIER_FEATURES = CONV8_CHANNELS *
        CONV8_POOLED_SPATIAL * CONV8_POOLED_SPATIAL;
    localparam int CLASSES = 11;
    localparam int PARAMS = CLASSIFIER_FEATURES * CLASSES;
    localparam int PARAM_AW = $clog2(PARAMS);
    localparam int CONV1_CONTRIBUTIONS = 18;
    localparam int CONV2_CONTRIBUTIONS = 576;
    localparam int CONV3_CONTRIBUTIONS = 1152;
    localparam int CONV4_CONTRIBUTIONS = 2304;
    localparam int CONV5_CONTRIBUTIONS = 2304;
    localparam int CONV6_CONTRIBUTIONS = 4608;
    localparam int CONV7_CONTRIBUTIONS = 4608;
    localparam int CONV8_CONTRIBUTIONS = 4608;
    localparam int VALUE_BITS = 15;
    localparam int CONV1_CHANNEL_WAVES =
        (CONV1_CHANNELS + 11) / 12;
    localparam int CONV1_SPATIAL_GROUPS =
        (CONV1_SPATIAL + 9) / 10;
    localparam int CONV1_CONTEXTS = CONV1_CHANNEL_WAVES * CONV1_SPATIAL;
    localparam int CONV1_REQUESTS = CONV1_CONTEXTS *
        ((CONV1_CONTRIBUTIONS + 3) / 4) * VALUE_BITS;
    localparam int CONV1_OBSERVATIONS = CONV1_CHANNELS * CONV1_SPATIAL *
        CONV1_SPATIAL_GROUPS;
    localparam int CONV2_CONTEXTS =
        ((CONV2_CHANNELS + 11) / 12) * CONV2_SPATIAL;
    localparam int CONV2_REQUESTS = CONV2_CONTEXTS *
        ((CONV2_CONTRIBUTIONS + 3) / 4);
    localparam int CONV2_SPATIAL_GROUPS =
        (CONV2_SPATIAL + 9) / 10;
    localparam int CONV2_OBSERVATIONS = CONV2_CHANNELS * CONV2_SPATIAL *
        CONV2_SPATIAL_GROUPS;
    localparam int CONV3_CONTEXTS = ((CONV3_CHANNELS + 11) / 12) *
        CONV3_SPATIAL;
    localparam int CONV3_REQUESTS = CONV3_CONTEXTS *
        ((CONV3_CONTRIBUTIONS + 3) / 4);
    localparam int CONV3_SPATIAL_GROUPS =
        (CONV3_SPATIAL + 9) / 10;
    localparam int CONV3_OBSERVATIONS = CONV3_CHANNELS * CONV3_SPATIAL *
        CONV3_SPATIAL_GROUPS;
    localparam int CONV4_CONTEXTS = ((CONV4_CHANNELS + 11) / 12) *
        CONV4_SPATIAL;
    localparam int CONV4_REQUESTS = CONV4_CONTEXTS *
        ((CONV4_CONTRIBUTIONS + 3) / 4);
    localparam int CONV4_SPATIAL_GROUPS =
        (CONV4_SPATIAL + 9) / 10;
    localparam int CONV4_OBSERVATIONS = CONV4_CHANNELS * CONV4_SPATIAL *
        CONV4_SPATIAL_GROUPS;
    localparam int CONV5_CONTEXTS = ((CONV5_CHANNELS + 11) / 12) *
        CONV5_SPATIAL;
    localparam int CONV5_REQUESTS = CONV5_CONTEXTS *
        ((CONV5_CONTRIBUTIONS + 3) / 4);
    localparam int CONV5_SPATIAL_GROUPS =
        (CONV5_SPATIAL + 9) / 10;
    localparam int CONV5_OBSERVATIONS = CONV5_CHANNELS * CONV5_SPATIAL *
        CONV5_SPATIAL_GROUPS;
    localparam int CONV6_CONTEXTS = ((CONV6_CHANNELS + 11) / 12) *
        CONV6_SPATIAL;
    localparam int CONV6_REQUESTS = CONV6_CONTEXTS *
        ((CONV6_CONTRIBUTIONS + 3) / 4);
    localparam int CONV6_SPATIAL_GROUPS =
        (CONV6_SPATIAL + 9) / 10;
    localparam int CONV6_OBSERVATIONS = CONV6_CHANNELS * CONV6_SPATIAL *
        CONV6_SPATIAL_GROUPS;
    localparam int CONV7_CONTEXTS = ((CONV7_CHANNELS + 11) / 12) *
        CONV7_SPATIAL;
    localparam int CONV7_REQUESTS = CONV7_CONTEXTS *
        ((CONV7_CONTRIBUTIONS + 3) / 4);
    localparam int CONV7_SPATIAL_GROUPS =
        (CONV7_SPATIAL + 9) / 10;
    localparam int CONV7_OBSERVATIONS = CONV7_CHANNELS * CONV7_SPATIAL *
        CONV7_SPATIAL_GROUPS;
    localparam int CONV8_CONTEXTS = ((CONV8_CHANNELS + 11) / 12) *
        CONV8_SPATIAL;
    localparam int CONV8_REQUESTS = CONV8_CONTEXTS *
        ((CONV8_CONTRIBUTIONS + 3) / 4);
    localparam int CONV8_SPATIAL_GROUPS =
        (CONV8_SPATIAL + 9) / 10;
    localparam int CONV8_OBSERVATIONS = CONV8_CHANNELS * CONV8_SPATIAL *
        CONV8_SPATIAL_GROUPS;
    localparam logic [19:0] CONV1_TAG = 20'hc1001;
    localparam logic [19:0] CONV2_TAG = 20'hc2002;
    localparam logic [19:0] CONV3_TAG = 20'hc3003;
    localparam logic [19:0] CONV4_TAG = 20'hc4004;
    localparam logic [19:0] CONV5_TAG = 20'hc5005;
    localparam logic [19:0] CONV6_TAG = 20'hc6006;
    localparam logic [19:0] CONV7_TAG = 20'hc7007;
    localparam logic [19:0] CONV8_TAG = 20'hc8008;
    localparam logic [19:0] CLASSIFIER_TAG = 20'hc120c;
    localparam logic [15:0] CLASSIFIER_VERSION = 16'h1201;
    localparam int CLASSIFIER_WORK_BASE = 20;
    localparam int CONV2_WRITE_BASE = 0;
    localparam int CONV3_WRITE_BASE = 0;
    localparam int CONV4_WRITE_BASE = 0;
    localparam int CONV5_WRITE_BASE = 0;
    localparam int CONV6_WRITE_BASE = 0;
    localparam int CONV7_WRITE_BASE = 0;
    localparam int CONV8_WRITE_BASE = 0;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic start_valid = 1'b0;
    logic start_ready;
    logic busy;
    logic done;
    logic idle;
    logic sticky_error;
    logic [15:0] tile_done;
    logic [15:0] tile_idle;
    logic [15:0] tile_error;

    logic [15:0] cfg_in_width;
    logic [15:0] cfg_in_height;
    logic [15:0] cfg_in_channels;
    logic [15:0] cfg_out_width;
    logic [15:0] cfg_out_height;
    logic [15:0] cfg_out_channels;
    logic [1:0] cfg_pool_mode;
    logic [7:0] cfg_padding;
    logic cfg_initial_active_pair;
    logic [3:0] cfg_class_count = '0;
    logic [19:0] cfg_layer_tag;
    logic cfg_feature_write_slot;
    logic [FEATURE_AW-1:0] cfg_feature_write_base;
    logic [15:0] cfg_feature_write_row_stride;
    logic [FEATURE_AW-1:0] cfg_feature_write_channel_stride;
    integer layer_phase = 0;
    integer timestep_phase = 0;
    integer progress_stage = 0;

    logic clear_error = 1'b0;
    logic parameter_load_valid = 1'b0;
    logic parameter_load_ready;
    logic [PARAM_AW-1:0] parameter_load_addr = '0;
    logic signed [15:0] parameter_load_code = '0;
    logic cache_prime_valid = 1'b0;
    logic cache_prime_ready;
    logic cache_prime_done;
    logic learning_cfg_valid = 1'b0;
    logic learning_cfg_ready;
    logic logit_wave_fire;
    logic [11:0] logit_wave_mask;
    logic [3:0] logit_wave_base;
    logic [3:0] logit_wave_count;
    logic [191:0] logit_wave_flat;
    logic classifier_pair;
    logic [15:0] classifier_version;
    logic classifier_cache_valid;
    logic [1:0] active_owner;

    logic context_load_valid;
    logic context_load_ready = 1'b0;
    logic context_load_weights;
    logic [9:0] context_output_channel_base;
    logic [9:0] context_output_channel_count;
    logic [31:0] context_spatial_base;
    logic [5:0] context_spatial_count;
    logic context_active_pair;
    logic [WORK_AW-1:0] context_work_base_addr;
    logic context_release_valid;
    logic context_release_ready = 1'b0;
    logic [9:0] release_output_channel_base;
    logic [9:0] release_output_channel_count;
    logic [31:0] release_spatial_base;
    logic [5:0] release_spatial_count;
    logic release_active_pair;
    logic [WORK_AW-1:0] release_work_base_addr;

    logic feature_request_valid;
    logic feature_request_ready;
    logic [3:0] feature_term_valid;
    logic [39:0] feature_input_channel_flat;
    logic [7:0] feature_kernel_row_flat;
    logic [7:0] feature_kernel_col_flat;
    logic [31:0] feature_spatial_base;
    logic [5:0] feature_spatial_count;
    logic [15:0] feature_output_row;
    logic [15:0] feature_output_col;
    logic [3:0] feature_value_bit;
    logic feature_response_valid = 1'b0;
    logic feature_response_ready;
    logic [39:0] feature_mask0 = '0;
    logic [39:0] feature_mask1 = '0;
    logic [39:0] feature_mask2 = '0;
    logic [39:0] feature_mask3 = '0;

    logic feature_update_valid;
    logic feature_update_ready = 1'b1;
    logic feature_update_slot;
    logic [FEATURE_AW-1:0] feature_update_addr;
    logic [63:0] feature_update_data;
    logic [63:0] feature_update_mask;
    logic feature_update_done = 1'b0;

    logic [3:0] weight_fill_valid;
    logic [3:0] weight_fill_ready;
    logic [7:0] weight_fill_tile_flat;
    logic [3:0] weight_fill_bank;
    logic [4*WEIGHT_AW-1:0] weight_fill_addr_flat;
    logic [799:0] weight_fill_data_flat;

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

    logic [3:0] maintenance_tile = '0;
    logic state_write_valid = 1'b0;
    logic state_write_ready;
    logic [MEMORY_AW-1:0] state_write_addr = '0;
    logic [319:0] state_write_data = '0;
    logic [39:0] state_write_mask = '0;
    logic state_read_valid = 1'b0;
    logic state_read_ready;
    logic [MEMORY_AW-1:0] state_read_addr = '0;
    logic state_result_valid;
    logic state_result_ready = 1'b0;
    logic [MEMORY_AW-1:0] state_result_addr;
    logic [319:0] state_read_data;
    logic trace_write_valid = 1'b0;
    logic trace_write_ready;
    logic [MEMORY_AW-1:0] trace_write_addr = '0;
    logic [319:0] trace_write_data = '0;
    logic [39:0] trace_write_mask = '0;
    logic trace_read_valid = 1'b0;
    logic trace_read_ready;
    logic [MEMORY_AW-1:0] trace_read_addr = '0;
    logic trace_result_valid;
    logic trace_result_ready = 1'b0;
    logic [MEMORY_AW-1:0] trace_result_addr;
    logic [319:0] trace_read_data;

    logic [15:0] input_patch [0:2047];
    logic signed [15:0] conv1_weights [0:CONV1_CHANNELS*
        CONV1_CONTRIBUTIONS-1];
    logic signed [15:0] conv2_weights [0:CONV2_CHANNELS*
        CONV2_CONTRIBUTIONS-1];
    logic signed [15:0] conv3_weights [0:CONV3_CHANNELS*
        CONV3_CONTRIBUTIONS-1];
    logic signed [15:0] conv4_weights [0:CONV4_CHANNELS*
        CONV4_CONTRIBUTIONS-1];
    logic signed [15:0] conv5_weights [0:CONV5_CHANNELS*
        CONV5_CONTRIBUTIONS-1];
    logic signed [15:0] conv6_weights [0:CONV6_CHANNELS*
        CONV6_CONTRIBUTIONS-1];
    logic signed [15:0] conv7_weights [0:CONV7_CHANNELS*
        CONV7_CONTRIBUTIONS-1];
    logic signed [15:0] conv8_weights [0:CONV8_CHANNELS*
        CONV8_CONTRIBUTIONS-1];
    logic signed [15:0] conv1_observed_pre [0:CONV1_SCALARS-1];
    logic signed [15:0] conv1_observed_next [0:CONV1_SCALARS-1];
    logic conv1_observed_spike [0:CONV1_SCALARS-1];
    logic conv1_expected_range [0:CONV1_SCALARS-1];
    logic signed [15:0] conv2_observed_pre [0:CONV2_SCALARS-1];
    logic signed [15:0] conv2_observed_next [0:CONV2_SCALARS-1];
    logic conv2_observed_spike [0:CONV2_SCALARS-1];
    logic conv2_expected_range [0:CONV2_SCALARS-1];
    logic conv2_expected_pooled_spike [0:CONV2_CHANNELS*
        CONV2_POOLED_SPATIAL*CONV2_POOLED_SPATIAL-1];
    logic signed [15:0] conv3_observed_pre [0:CONV3_SCALARS-1];
    logic signed [15:0] conv3_observed_next [0:CONV3_SCALARS-1];
    logic conv3_observed_spike [0:CONV3_SCALARS-1];
    logic conv3_expected_range [0:CONV3_SCALARS-1];
    logic signed [15:0] conv4_observed_pre [0:CONV4_SCALARS-1];
    logic signed [15:0] conv4_observed_next [0:CONV4_SCALARS-1];
    logic conv4_observed_spike [0:CONV4_SCALARS-1];
    logic conv4_expected_range [0:CONV4_SCALARS-1];
    logic conv4_expected_pooled_spike [0:CONV4_CHANNELS*
        CONV4_POOLED_SPATIAL*CONV4_POOLED_SPATIAL-1];
    logic signed [15:0] conv5_observed_pre [0:CONV5_SCALARS-1];
    logic signed [15:0] conv5_observed_next [0:CONV5_SCALARS-1];
    logic conv5_observed_spike [0:CONV5_SCALARS-1];
    logic conv5_expected_range [0:CONV5_SCALARS-1];
    logic signed [15:0] conv6_observed_pre [0:CONV6_SCALARS-1];
    logic signed [15:0] conv6_observed_next [0:CONV6_SCALARS-1];
    logic conv6_observed_spike [0:CONV6_SCALARS-1];
    logic conv6_expected_range [0:CONV6_SCALARS-1];
    logic conv6_expected_pooled_spike [0:CONV6_CHANNELS*
        CONV6_POOLED_SPATIAL*CONV6_POOLED_SPATIAL-1];
    logic signed [15:0] conv7_observed_pre [0:CONV7_SCALARS-1];
    logic signed [15:0] conv7_observed_next [0:CONV7_SCALARS-1];
    logic conv7_observed_spike [0:CONV7_SCALARS-1];
    logic conv7_expected_range [0:CONV7_SCALARS-1];
    logic signed [15:0] conv8_observed_pre [0:CONV8_SCALARS-1];
    logic signed [15:0] conv8_observed_next [0:CONV8_SCALARS-1];
    logic conv8_observed_spike [0:CONV8_SCALARS-1];
    logic conv8_expected_range [0:CONV8_SCALARS-1];
    logic conv8_expected_pooled_spike [0:CONV8_CHANNELS*
        CONV8_POOLED_SPATIAL*CONV8_POOLED_SPATIAL-1];
    logic [15:0] conv8_expected_pooled_q8_8 [0:CONV8_CHANNELS*
        CONV8_POOLED_SPATIAL*CONV8_POOLED_SPATIAL-1];
    logic signed [15:0] classifier_weights [0:PARAMS-1];
    logic signed [15:0] classifier_expected_current [0:CLASSES-1];
    logic signed [15:0] classifier_expected_centered [0:CLASSES-1];
    logic [63:0] feature_memory [0:1][0:FEATURE_DEPTH-1];
    // Duplicate protection plus exact scalar counts proves completeness. Keep
    // these as unpacked two-state memories so Verilator does not construct
    // enormous packed-slice comparisons in one C++ translation unit.
    bit conv1_seen [0:52*CONV1_SCALARS-1];
    bit conv2_seen [0:52*CONV2_SCALARS-1];
    bit conv3_seen [0:52*CONV3_SCALARS-1];
    bit conv4_seen [0:52*CONV4_SCALARS-1];
    bit conv5_seen [0:52*CONV5_SCALARS-1];
    bit conv6_seen [0:52*CONV6_SCALARS-1];
    bit conv7_seen [0:52*CONV7_SCALARS-1];
    bit conv8_seen [0:52*CONV8_SCALARS-1];
    bit conv8_commit_seen [0:52*CLASSIFIER_FEATURES-1];

    logic signed [15:0] conv1_state_backing [0:4*CONV1_SCALARS-1];
    logic signed [15:0] conv2_state_backing [0:4*CONV2_SCALARS-1];
    logic signed [15:0] conv3_state_backing [0:4*CONV3_SCALARS-1];
    logic signed [15:0] conv4_state_backing [0:4*CONV4_SCALARS-1];
    logic signed [15:0] conv5_state_backing [0:4*CONV5_SCALARS-1];
    logic signed [15:0] conv6_state_backing [0:4*CONV6_SCALARS-1];
    logic signed [15:0] conv7_state_backing [0:4*CONV7_SCALARS-1];
    logic signed [15:0] conv8_state_backing [0:4*CONV8_SCALARS-1];
    logic signed [15:0] conv1_trace_backing [0:4*CONV1_SCALARS-1];
    logic signed [15:0] conv2_trace_backing [0:4*CONV2_SCALARS-1];
    logic signed [15:0] conv3_trace_backing [0:4*CONV3_SCALARS-1];
    logic signed [15:0] conv4_trace_backing [0:4*CONV4_SCALARS-1];
    logic signed [15:0] conv5_trace_backing [0:4*CONV5_SCALARS-1];
    logic signed [15:0] conv6_trace_backing [0:4*CONV6_SCALARS-1];
    logic signed [15:0] conv7_trace_backing [0:4*CONV7_SCALARS-1];
    logic signed [15:0] conv8_trace_backing [0:4*CONV8_SCALARS-1];

    integer cycle_count = 0;
    // Interface counters are observational; the backing memories remain TB
    // models. A response mask bit is a bit-plane/spike term, not a Q8.8 MAC.
    longint unsigned perf_requests = 0, perf_responses = 0;
    longint unsigned perf_mask_bits = 0, perf_updates = 0;
    longint unsigned perf_weight_fills = 0;
    longint unsigned perf_state_reads = 0, perf_trace_reads = 0;
    longint unsigned perf_state_writes = 0, perf_trace_writes = 0;
    longint unsigned perf_frame_start = 0;
    always @(posedge clk) begin
        if (rst_n) begin
            if (feature_request_valid && feature_request_ready) perf_requests++;
            if (feature_response_valid && feature_response_ready) begin
                perf_responses++;
                perf_mask_bits += $countones(feature_mask0) + $countones(feature_mask1)
                                + $countones(feature_mask2) + $countones(feature_mask3);
            end
            if (feature_update_valid && feature_update_ready) perf_updates++;
            for (int region = 0; region < 4; region++)
                if (weight_fill_valid[region] && weight_fill_ready[region])
                    perf_weight_fills++;
            if (state_read_valid && state_read_ready) perf_state_reads++;
            if (trace_read_valid && trace_read_ready) perf_trace_reads++;
            if (state_write_valid && state_write_ready) perf_state_writes++;
            if (trace_write_valid && trace_write_ready) perf_trace_writes++;
        end
    end
    integer conv1_context_count = 0;
    integer conv2_context_count = 0;
    integer conv3_context_count = 0;
    integer conv4_context_count = 0;
    integer conv5_context_count = 0;
    integer conv6_context_count = 0;
    integer conv7_context_count = 0;
    integer conv8_context_count = 0;
    integer conv1_release_count = 0;
    integer conv2_release_count = 0;
    integer conv3_release_count = 0;
    integer conv4_release_count = 0;
    integer conv5_release_count = 0;
    integer conv6_release_count = 0;
    integer conv7_release_count = 0;
    integer conv8_release_count = 0;
    integer conv1_request_count = 0;
    integer conv2_request_count = 0;
    integer conv3_request_count = 0;
    integer conv4_request_count = 0;
    integer conv5_request_count = 0;
    integer conv6_request_count = 0;
    integer conv7_request_count = 0;
    integer conv8_request_count = 0;
    integer conv1_update_count = 0;
    integer conv2_update_count = 0;
    integer conv3_update_count = 0;
    integer conv4_update_count = 0;
    integer conv5_update_count = 0;
    integer conv6_update_count = 0;
    integer conv7_update_count = 0;
    integer conv8_update_count = 0;
    integer conv1_observation_count = 0;
    integer conv2_observation_count = 0;
    integer conv3_observation_count = 0;
    integer conv4_observation_count = 0;
    integer conv5_observation_count = 0;
    integer conv6_observation_count = 0;
    integer conv7_observation_count = 0;
    integer conv8_observation_count = 0;
    integer conv1_scalar_count = 0;
    integer conv2_scalar_count = 0;
    integer conv3_scalar_count = 0;
    integer conv4_scalar_count = 0;
    integer conv5_scalar_count = 0;
    integer conv6_scalar_count = 0;
    integer conv7_scalar_count = 0;
    integer conv8_scalar_count = 0;
    integer conv8_commit_count = 0;
    integer conv8_commit_value_count = 0;
    integer classifier_context_count = 0;
    integer classifier_logit_wave_count = 0;
    integer cache_prime_count = 0;

    function automatic logic [39:0] make_feature_mask(input int term);
        integer channel;
        integer kernel_row;
        integer kernel_col;
        integer input_row;
        integer input_col;
        integer input_index;
        integer memory_address;
        begin
            make_feature_mask = '0;
            channel = integer'(feature_input_channel_flat[term*10 +: 10]);
            kernel_row = integer'(feature_kernel_row_flat[term*2 +: 2]);
            kernel_col = integer'(feature_kernel_col_flat[term*2 +: 2]);
            for (integer lane = 0; lane < 40; lane++) begin
            input_row = integer'(feature_output_row) + kernel_row -
                integer'(cfg_padding);
            input_col = integer'(feature_output_col) + lane + kernel_col -
                integer'(cfg_padding);
                if (feature_term_valid[term] &&
                        lane < integer'(feature_spatial_count)) begin
                    if (layer_phase == 1 && channel < 2 &&
                            input_row >= 0 && input_row < 32 &&
                            input_col >= 0 && input_col < 32) begin
                        input_index = channel * 1024 + input_row * 32 +
                            input_col;
                        make_feature_mask[lane] =
                            input_patch[input_index][feature_value_bit];
                    end else if (layer_phase == 2 && channel < 64 &&
                            input_row >= 0 && input_row < CONV1_SPATIAL &&
                            input_col >= 0 && input_col < CONV1_SPATIAL) begin
                        memory_address = channel * CONV1_SPATIAL + input_row;
                        make_feature_mask[lane] =
                            feature_memory[0][memory_address][input_col];
                    end else if (layer_phase == 3 && channel < 128 &&
                            input_row >= 0 && input_row < CONV3_SPATIAL &&
                            input_col >= 0 && input_col < CONV3_SPATIAL) begin
                        memory_address = CONV2_WRITE_BASE +
                            channel * CONV3_SPATIAL + input_row;
                        make_feature_mask[lane] =
                            feature_memory[1][memory_address][input_col];
                    end else if (layer_phase == 4 && channel < 256 &&
                            input_row >= 0 && input_row < CONV4_SPATIAL &&
                            input_col >= 0 && input_col < CONV4_SPATIAL) begin
                        memory_address = CONV3_WRITE_BASE +
                            channel * CONV4_SPATIAL + input_row;
                        make_feature_mask[lane] =
                            feature_memory[0][memory_address][input_col];
                    end else if (layer_phase == 5 && channel < 256 &&
                            input_row >= 0 && input_row < CONV5_SPATIAL &&
                            input_col >= 0 && input_col < CONV5_SPATIAL) begin
                        memory_address = CONV4_WRITE_BASE +
                            channel * CONV5_SPATIAL + input_row;
                        make_feature_mask[lane] =
                            feature_memory[1][memory_address][input_col];
                    end else if (layer_phase == 6 && channel < 512 &&
                            input_row >= 0 && input_row < CONV6_SPATIAL &&
                            input_col >= 0 && input_col < CONV6_SPATIAL) begin
                        memory_address = CONV5_WRITE_BASE +
                            channel * CONV6_SPATIAL + input_row;
                        make_feature_mask[lane] =
                            feature_memory[0][memory_address][input_col];
                    end else if (layer_phase == 7 && channel < 512 &&
                            input_row >= 0 && input_row < CONV7_SPATIAL &&
                            input_col >= 0 && input_col < CONV7_SPATIAL) begin
                        memory_address = CONV6_WRITE_BASE +
                            channel * CONV7_SPATIAL + input_row;
                        make_feature_mask[lane] =
                            feature_memory[1][memory_address][input_col];
                    end else if (layer_phase == 8 && channel < 512 &&
                            input_row >= 0 && input_row < CONV8_SPATIAL &&
                            input_col >= 0 && input_col < CONV8_SPATIAL) begin
                        memory_address = CONV7_WRITE_BASE +
                            channel * CONV8_SPATIAL + input_row;
                        make_feature_mask[lane] =
                            feature_memory[0][memory_address][input_col];
                    end
                end
            end
        end
    endfunction

    assign feature_request_ready = !feature_response_valid;
    assign tile_done = '0;
    assign observation_valid = dut.u_core.u_system.u_array.array_observation_valid;
    assign observation_ready = dut.u_core.u_system.u_array.array_observation_ready;
    assign observation_tile_flat = dut.u_core.u_system.u_array.array_observation_tile_flat;
    assign observation_mode_flat = dut.u_core.u_system.u_array.array_observation_mode_flat;
    assign observation_lane_mask_flat =
        dut.u_core.u_system.u_array.array_observation_lane_mask_flat;
    assign observation_range_mask_flat =
        dut.u_core.u_system.u_array.array_observation_range_mask_flat;
    assign observation_spike_flat = dut.u_core.u_system.u_array.array_observation_spike_flat;
    assign observation_pre_state_flat =
        dut.u_core.u_system.u_array.array_observation_pre_state_flat;
    assign observation_trace_next_flat =
        dut.u_core.u_system.u_array.array_observation_trace_next_flat;
    assign observation_meta_flat = dut.u_core.u_system.u_array.array_observation_meta_flat;

    assign scalar_valid = dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
        .serializer_scalar_valid;
    assign scalar_ready = dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
        .serializer_scalar_ready;
    assign scalar_layer_tag = dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
        .serializer_layer_tag;
    assign scalar_channel = dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
        .serializer_channel;
    assign scalar_spatial = dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
        .serializer_spatial;
    assign scalar_spike = dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
        .serializer_spike;
    assign scalar_pre_state = dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
        .serializer_scalar_pre_state;
    assign scalar_trace_next = '0;
    assign scalar_last = 1'b0;
    assign serializer_busy = dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
        .serializer_busy;
    assign serializer_error = dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
        .serializer_error;

    localparam int TOPOLOGY_MASK = 7;
    localparam int SCNN_WEIGHT_FRACTIONAL_BITS = 15;
    localparam bit ACCUMULATE_SCNN_SAMPLE_LOGITS = 1'b1;
    localparam bit ENABLE_SCNN_FRAME_SERVICE = 1'b1;
    localparam int SRNN_M2_PARAM_DEPTH = 120320;
    localparam int SRNN_M2_BATCH_MAX = 64;
    localparam int SRNN_M2_TIMESTEP_MAX = 128;
    localparam int SRNN_M2_BACKBONE_MAX = 450;
    localparam int SRNN_M2_HIDDEN_MAX = 256;
    localparam int SRNN_M2_CLASS_MAX = 20;
    localparam int SRNN_BACKBONE_PARAM_DEPTH = 524288;
    localparam int SRNN_BACKBONE_PARAM_ADDR_WIDTH = $clog2(SRNN_BACKBONE_PARAM_DEPTH);
    localparam int SRNN_FEEDFORWARD_SOURCE_MAX = 700;
    localparam int SRNN_RECURRENT_SOURCE_MAX = 480;
    localparam int WEIGHT_ADDR_WIDTH = $clog2(WEIGHT_DEPTH);
    localparam int MEMORY_ADDR_WIDTH = $clog2(MEMORY_DEPTH);
    localparam int WORK_ADDR_WIDTH = $clog2(MEMORY_DEPTH / 2);
    localparam int FEATURE_WORD_DEPTH = 8192;
    localparam int FEATURE_ADDR_WIDTH = $clog2(FEATURE_WORD_DEPTH);
    localparam int PROFILE_ID_WIDTH = 8;
    localparam int TENSOR_ID_WIDTH = 8;
    localparam int PARAM_DEPTH = 16777216;
    localparam int PARAM_ADDR_WIDTH = 24;
    localparam int TAG_WIDTH = 16;
    localparam int MAX_BATCH = 16;
    localparam int MAX_TIMESTEPS = 12;
    localparam int CLASSIFIER_CAPTURE_MAX_BATCH = MAX_BATCH < 4 ? MAX_BATCH : 4;
    localparam int CLASSIFIER_CAPTURE_MAX_TIMESTEPS = MAX_TIMESTEPS < 4 ? MAX_TIMESTEPS : 4;
    localparam int MAX_INPUT_CHANNELS = 8;
    localparam int MAX_OUTPUT_CHANNELS = 12;
    localparam int OBS_INPUT_CHANNELS = 512;
    localparam int OBS_OUTPUT_CHANNELS = 512;
    localparam int SPATIAL = 4;
    localparam int POOLED_SPATIAL = 2;
    localparam int KERNEL_SIZE = 3;
    localparam int Q16_ADDR_WIDTH = 24;
    localparam int Q64_ADDR_WIDTH = 22;
    localparam int VERSION_WIDTH = 16;
    localparam int MAX_FEATURES = 2048;
    localparam int MAX_CLASSES = 11;
    localparam int LANES = 8;
    localparam int ENGINES = 12;
    localparam int FRAME_WIDTH = 8;
    localparam int SELECTED_LAYER = 7;
    localparam bit ENABLE_SCNN_CONV8_LEARNING = 1'b0;
    localparam int INTERNAL_MODULATION_BATCH = 2;
    localparam int SAMPLE_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH);
    localparam int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1);
    localparam int TIMESTEP_WIDTH = MAX_TIMESTEPS <= 2 ? 1 : $clog2(MAX_TIMESTEPS);
    localparam int TIMESTEP_COUNT_WIDTH = $clog2(MAX_TIMESTEPS + 1);
    localparam int CHANNEL_WIDTH = OBS_INPUT_CHANNELS <= 2 ? 1 : $clog2(OBS_INPUT_CHANNELS);
    localparam int CHANNEL_COUNT_WIDTH = $clog2(OBS_INPUT_CHANNELS + 1);
    localparam int FEATURE_WIDTH = MAX_FEATURES <= 2 ? 1 : $clog2(MAX_FEATURES);
    localparam int FEATURE_COUNT_WIDTH = $clog2(MAX_FEATURES + 1);
    localparam int CLASS_WIDTH = MAX_CLASSES <= 2 ? 1 : $clog2(MAX_CLASSES);
    localparam int CLASS_COUNT_WIDTH = $clog2(MAX_CLASSES + 1);
    localparam bit ENABLE_ONLINE_LEARNING = 1'b1;
    localparam bit ENABLE_SRNN_WH1_LEARNING = 1'b0;
    localparam int CODE_WIDTH = 16;
    localparam int RESIDUAL_WIDTH = 16;
    localparam int FC_MAX_HIDDEN = 480;
    localparam int FC_MAX_OUTPUTS = 16;
    localparam int SRNN_BATCH_MAX = 64;
    localparam int SRNN_TIMESTEP_MAX = 128;
    localparam int SRNN_HIDDEN_MAX = 256;
    localparam int SRNN_BACKBONE_MAX = 450;
    localparam int SRNN_CLASS_MAX = 20;
    logic ft_i_initialize_valid;
    logic ft_o_initialize_ready;
    logic ft_o_initialize_done;
    logic ft_i_classifier_pair;
    logic [VERSION_WIDTH-1:0] ft_i_context_model_version;
    logic [VERSION_WIDTH-1:0] ft_i_context_adapt_version;
    logic ft_i_system_parameter_owner;
    logic ft_i_system_learning_enable;
    logic ft_i_system_profile_aligned;
    logic [PROFILE_ID_WIDTH-1:0] ft_i_system_profile_id;
    logic ft_i_request_valid;
    logic ft_o_request_ready;
    logic ft_i_learning_enable;
    logic [BATCH_COUNT_WIDTH-1:0] ft_i_batch_count;
    logic [TIMESTEP_WIDTH-1:0] ft_i_timestep;
    logic [TIMESTEP_COUNT_WIDTH-1:0] ft_i_timestep_count;
    logic [VERSION_WIDTH-1:0] ft_i_sample_version;
    logic [MAX_BATCH*CLASS_WIDTH-1:0] ft_i_labels_flat;
    logic ft_o_done;
    logic ft_i_fc_srnn_layer_valid;
    logic ft_o_fc_srnn_layer_ready;
    logic [1:0] ft_i_fc_srnn_mode;
    logic ft_i_fc_srnn_weight_int8;
    logic ft_i_fc_srnn_current_shift6;
    logic [15:0] ft_i_fc_srnn_neuron_count;
    logic [15:0] ft_i_fc_srnn_feedforward_group_count;
    logic [15:0] ft_i_fc_srnn_recurrent_group_count;
    logic ft_i_fc_srnn_recurrent_enable;
    logic [15:0] ft_i_fc_srnn_leak;
    logic [31:0] ft_i_fc_srnn_threshold;
    logic ft_i_fc_srnn_reset_to_zero;
    logic [15:0] ft_i_fc_srnn_trace_decay;
    logic [51:0] ft_i_fc_srnn_transaction_tag;
    logic [WORK_ADDR_WIDTH-1:0] ft_i_fc_srnn_work_base_addr;
    logic [3:0] ft_i_fc_srnn_destination_tile_base;
    logic ft_i_fc_srnn_group_valid;
    logic ft_o_fc_srnn_group_ready;
    logic ft_i_fc_srnn_group_recurrent;
    logic [3:0] ft_i_fc_srnn_group_source_valid;
    logic [WEIGHT_ADDR_WIDTH-1:0] ft_i_fc_srnn_group_weight_addr0;
    logic [WEIGHT_ADDR_WIDTH-1:0] ft_i_fc_srnn_group_weight_addr1;
    logic [WEIGHT_ADDR_WIDTH-1:0] ft_i_fc_srnn_group_weight_addr2;
    logic [WEIGHT_ADDR_WIDTH-1:0] ft_i_fc_srnn_group_weight_addr3;
    logic ft_i_srnn_group_streamer_select;
    logic ft_i_srnn_group_streamer_start_valid;
    logic ft_o_srnn_group_streamer_start_ready;
    logic [9:0] ft_i_srnn_group_feedforward_source_count;
    logic [9:0] ft_i_srnn_group_recurrent_source_count;
    logic [9:0] ft_i_srnn_group_feedforward_cache_row_base;
    logic [9:0] ft_i_srnn_group_recurrent_cache_row_base;
    logic [SRNN_FEEDFORWARD_SOURCE_MAX-1:0] ft_i_srnn_group_feedforward_spike_flat;
    logic [SRNN_RECURRENT_SOURCE_MAX-1:0] ft_i_srnn_group_recurrent_spike_flat;
    logic ft_o_srnn_group_streamer_done;
    logic ft_o_srnn_group_streamer_busy;
    logic ft_o_srnn_group_streamer_sticky_error;
    logic [3:0] ft_o_srnn_group_streamer_error_code;
    logic ft_o_fc_srnn_layer_done;
    logic ft_o_fc_srnn_busy;
    logic ft_o_fc_srnn_idle;
    logic ft_o_fc_srnn_sticky_error;
    logic [7:0] ft_o_fc_srnn_error_code;
    logic ft_o_fc_srnn_observation_valid;
    logic ft_i_fc_srnn_observation_ready;
    logic ft_i_fc_srnn_observation_sink_idle;
    logic [1:0] ft_o_fc_srnn_observation_mode;
    logic [51:0] ft_o_fc_srnn_observation_transaction_tag;
    logic [9:0] ft_o_fc_srnn_observation_neuron_id;
    logic ft_o_fc_srnn_observation_spike;
    logic signed [31:0] ft_o_fc_srnn_observation_pre_state;
    logic signed [31:0] ft_o_fc_srnn_observation_trace_next;
    logic ft_o_fc_srnn_observation_last_in_record;
    logic ft_o_fc_srnn_observation_busy;
    logic ft_o_fc_srnn_observation_sticky_error;
    logic ft_i_fc_srnn_learning_cfg_valid;
    logic ft_o_fc_srnn_learning_cfg_ready;
    logic [1:0] ft_i_fc_srnn_learning_cfg_mode;
    logic [51:0] ft_i_fc_srnn_learning_cfg_tag;
    logic [5:0] ft_i_fc_srnn_learning_sample_id;
    logic [9:0] ft_i_fc_srnn_learning_neuron_base;
    logic [9:0] ft_i_fc_srnn_learning_neuron_count;
    logic signed [17:0] ft_i_fc_srnn_learning_fc_threshold;
    logic ft_i_fc_srnn_learning_fc_output_layer;
    logic [3:0] ft_i_fc_srnn_learning_fc_label;
    logic ft_i_fc_srnn_learning_fc_logit_scale_enable;
    logic [23:0] ft_i_fc_srnn_learning_fc_logit_scale_q16;
    logic ft_i_fc_srnn_learning_fc_logit_accumulate;
    logic ft_i_fc_srnn_learning_fc_sample_start;
    logic ft_i_fc_srnn_learning_srnn_path;
    logic ft_o_fc_surrogate_valid;
    logic [5:0] ft_o_fc_surrogate_sample;
    logic [9:0] ft_o_fc_surrogate_hidden;
    logic [16:0] ft_o_fc_surrogate_q016;
    logic ft_o_fc_logit_last;
    logic ft_o_fc_srnn_learning_capture_done;
    logic ft_o_fc_srnn_learning_done;
    logic ft_o_fc_srnn_learning_busy;
    logic ft_o_fc_srnn_learning_idle;
    logic ft_o_fc_srnn_learning_sticky_error;
    logic ft_o_fc_srnn_learning_error_source;
    logic [3:0] ft_o_fc_srnn_learning_error_code;
    logic ft_i_fc_w0_refill_valid;
    logic ft_o_fc_w0_refill_ready;
    logic ft_i_fc_w0_refill_all;
    logic ft_i_fc_cache_refill_weight_int8;
    logic [3:0] ft_i_fc_cache_refill_destination_tile_base;
    logic [FEATURE_COUNT_WIDTH-1:0] ft_i_fc_w0_dirty_feature;
    logic [3:0] ft_i_fc_w0_dirty_tile;
    logic [PARAM_ADDR_WIDTH-1:0] ft_i_fc_w0_parameter_base;
    logic [PARAM_ADDR_WIDTH-1:0] ft_i_fc_w0_hidden_stride;
    logic [9:0] ft_i_fc_cache_row_base;
    logic [FEATURE_COUNT_WIDTH-1:0] ft_i_fc_w0_feature_count;
    logic [9:0] ft_i_fc_w0_hidden_count;
    logic [4:0] ft_i_fc_w0_tile_count;
    logic [VERSION_WIDTH-1:0] ft_i_fc_w0_version;
    logic ft_o_fc_w0_refill_done;
    logic [31:0] ft_o_fc_w0_rows_published;
    logic [VERSION_WIDTH-1:0] ft_o_fc_w0_published_version;
    logic [1:0] ft_o_cache_fill_grant;
    logic ft_o_cache_refill_busy;
    logic ft_o_cache_refill_quiescent;
    logic ft_o_cache_refill_sticky_error;
    logic ft_i_srnn_backbone_refill_valid;
    logic ft_o_srnn_backbone_refill_ready;
    logic ft_i_srnn_backbone_refill_all;
    logic [9:0] ft_i_srnn_backbone_dirty_source;
    logic [3:0] ft_i_srnn_backbone_dirty_tile;
    logic [SRNN_BACKBONE_PARAM_ADDR_WIDTH-1:0] ft_i_srnn_backbone_parameter_base;
    logic [9:0] ft_i_srnn_backbone_cache_row_base;
    logic [9:0] ft_i_srnn_backbone_source_count;
    logic [9:0] ft_i_srnn_backbone_destination_count;
    logic [3:0] ft_i_srnn_backbone_tile_base;
    logic [VERSION_WIDTH-1:0] ft_i_srnn_backbone_version;
    logic ft_o_srnn_backbone_forward_read_valid;
    logic ft_i_srnn_backbone_forward_read_ready;
    logic [SRNN_BACKBONE_PARAM_ADDR_WIDTH-1:0] ft_o_srnn_backbone_forward_read_addr;
    logic [TAG_WIDTH-1:0] ft_o_srnn_backbone_forward_read_tag;
    logic ft_i_srnn_backbone_forward_response_valid;
    logic ft_o_srnn_backbone_forward_response_ready;
    logic [TAG_WIDTH-1:0] ft_i_srnn_backbone_forward_response_tag;
    logic signed [15:0] ft_i_srnn_backbone_forward_response_code;
    logic ft_i_srnn_backbone_forward_response_error;
    logic ft_o_srnn_backbone_refill_done;
    logic [31:0] ft_o_srnn_backbone_rows_published;
    logic [VERSION_WIDTH-1:0] ft_o_srnn_backbone_published_version;
    logic [PROFILE_ID_WIDTH-1:0] ft_i_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] ft_i_conv_tensor_id;
    logic [TENSOR_ID_WIDTH-1:0] ft_i_classifier_tensor_id;
    logic [CHANNEL_COUNT_WIDTH-1:0] ft_i_total_input_channels;
    logic [CHANNEL_COUNT_WIDTH-1:0] ft_i_total_output_channels;
    logic [PARAM_ADDR_WIDTH-1:0] ft_i_conv_parameter_base;
    logic [Q16_ADDR_WIDTH-1:0] ft_i_conv_weight_q16_base;
    logic [PARAM_ADDR_WIDTH-1:0] ft_i_classifier_parameter_base;
    logic [FEATURE_COUNT_WIDTH-1:0] ft_i_classifier_feature_count;
    logic [CLASS_COUNT_WIDTH-1:0] ft_i_classifier_class_count;
    logic ft_i_classifier_trace_leak_valid;
    logic signed [15:0] ft_i_classifier_trace_leak_q8_8;
    logic [15:0] ft_i_conv_vote_increment;
    logic [15:0] ft_i_conv_vote_threshold;
    logic [15:0] ft_i_classifier_vote_increment;
    logic [15:0] ft_i_classifier_vote_threshold;
    logic signed [15:0] ft_i_code_min;
    logic signed [15:0] ft_i_code_max;
    logic ft_i_forward_start_valid;
    logic ft_o_forward_start_ready;
    logic ft_o_forward_start_fire;
    logic ft_i_forward_done;
    logic ft_o_forward_active;
    logic [VERSION_WIDTH-1:0] ft_o_forward_model_version;
    logic [VERSION_WIDTH-1:0] ft_o_forward_adapt_version;
    logic ft_i_restore_complete;
    logic ft_o_restore_ready;
    logic ft_o_recovery_required;
    logic [15:0] ft_i_conv_in_width;
    logic [15:0] ft_i_conv_in_height;
    logic [15:0] ft_i_conv_in_channels;
    logic [15:0] ft_i_conv_out_width;
    logic [15:0] ft_i_conv_out_height;
    logic [15:0] ft_i_conv_out_channels;
    logic [7:0] ft_i_conv_stride;
    logic [7:0] ft_i_conv_padding;
    logic [1:0] ft_i_conv_pool_mode;
    logic [WORK_ADDR_WIDTH-1:0] ft_i_conv_work_base_addr;
    logic [15:0] ft_i_conv_leak;
    logic [31:0] ft_i_conv_threshold;
    logic ft_i_conv_reset_to_zero;
    logic [15:0] ft_i_conv_trace_decay;
    logic [19:0] ft_i_conv_layer_tag;
    logic [FEATURE_COUNT_WIDTH-1:0] ft_i_target_feature_count;
    logic signed [15:0] ft_i_target_trace_leak_q8_8;
    logic ft_i_target_trace_state_reset;
    logic [CHANNEL_WIDTH-1:0] ft_i_classifier_feature_channel_base;
    logic [CHANNEL_COUNT_WIDTH-1:0] ft_i_classifier_feature_channel_count;
    logic [WORK_ADDR_WIDTH-1:0] ft_i_classifier_work_base_addr;
    logic [15:0] ft_i_classifier_trace_decay;
    logic [19:0] ft_i_classifier_layer_tag;
    logic [FRAME_WIDTH-1:0] ft_i_frame_index;
    logic ft_i_graph_descriptor_valid;
    logic ft_o_graph_descriptor_ready;
    logic ft_i_graph_descriptor_last;
    logic ft_o_graph_descriptor_done;
    logic ft_o_graph_frame_active;
    logic ft_o_graph_frame_target;
    logic [SAMPLE_WIDTH-1:0] ft_o_graph_frame_sample;
    logic [TIMESTEP_WIDTH-1:0] ft_o_graph_frame_timestep;
    logic [VERSION_WIDTH-1:0] ft_o_graph_frame_version;
    logic ft_o_state_service_valid;
    logic ft_i_state_service_ready;
    logic ft_o_state_service_release;
    logic ft_o_state_service_is_classifier;
    logic [9:0] ft_o_state_service_output_channel_base;
    logic [9:0] ft_o_state_service_output_channel_count;
    logic [31:0] ft_o_state_service_spatial_base;
    logic [5:0] ft_o_state_service_spatial_count;
    logic ft_o_state_service_active_pair;
    logic [WORK_ADDR_WIDTH-1:0] ft_o_state_service_work_base_addr;
    logic ft_o_feature_request_valid;
    logic ft_i_feature_request_ready;
    logic [3:0] ft_o_feature_term_valid;
    logic [39:0] ft_o_feature_input_channel_flat;
    logic [7:0] ft_o_feature_kernel_row_flat;
    logic [7:0] ft_o_feature_kernel_col_flat;
    logic [31:0] ft_o_feature_spatial_base;
    logic [5:0] ft_o_feature_spatial_count;
    logic [15:0] ft_o_feature_output_row;
    logic [15:0] ft_o_feature_output_col;
    logic [3:0] ft_o_feature_value_bit;
    logic ft_i_feature_response_valid;
    logic ft_o_feature_response_ready;
    logic [39:0] ft_i_feature_mask0;
    logic [39:0] ft_i_feature_mask1;
    logic [39:0] ft_i_feature_mask2;
    logic [39:0] ft_i_feature_mask3;
    logic ft_i_feature_service_error;
    logic ft_i_feature_write_slot;
    logic [FEATURE_ADDR_WIDTH-1:0] ft_i_feature_write_base_addr;
    logic [15:0] ft_i_feature_write_row_stride_words;
    logic [FEATURE_ADDR_WIDTH-1:0] ft_i_feature_write_channel_stride_words;
    logic ft_o_feature_update_valid;
    logic ft_i_feature_update_ready;
    logic ft_o_feature_update_slot;
    logic [FEATURE_ADDR_WIDTH-1:0] ft_o_feature_update_addr;
    logic [63:0] ft_o_feature_update_data;
    logic [63:0] ft_o_feature_update_mask;
    logic ft_i_feature_update_done;
    logic [3:0] ft_i_maintenance_tile;
    logic ft_i_state_maintenance_write_valid;
    logic ft_o_state_maintenance_write_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] ft_i_state_maintenance_write_addr;
    logic [319:0] ft_i_state_maintenance_write_data;
    logic [39:0] ft_i_state_maintenance_write_mask;
    logic ft_i_state_maintenance_read_valid;
    logic ft_o_state_maintenance_read_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] ft_i_state_maintenance_read_addr;
    logic ft_o_state_maintenance_result_valid;
    logic ft_i_state_maintenance_result_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] ft_o_state_maintenance_result_addr;
    logic [319:0] ft_o_state_maintenance_read_data;
    logic ft_i_trace_maintenance_write_valid;
    logic ft_o_trace_maintenance_write_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] ft_i_trace_maintenance_write_addr;
    logic [319:0] ft_i_trace_maintenance_write_data;
    logic [39:0] ft_i_trace_maintenance_write_mask;
    logic ft_i_trace_maintenance_read_valid;
    logic ft_o_trace_maintenance_read_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] ft_i_trace_maintenance_read_addr;
    logic ft_o_trace_maintenance_result_valid;
    logic ft_i_trace_maintenance_result_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] ft_o_trace_maintenance_result_addr;
    logic [319:0] ft_o_trace_maintenance_read_data;
    logic ft_o_classifier_cache_valid;
    logic ft_o_active_classifier_pair;
    logic [VERSION_WIDTH-1:0] ft_o_parameter_version;
    logic ft_o_conv_cache_valid;
    logic [9:0] ft_o_conv_cached_output_base;
    logic [4:0] ft_o_conv_cached_output_count;
    logic [VERSION_WIDTH-1:0] ft_o_conv_cached_version;
    logic ft_o_conv_cache_invalidate;
    logic [VERSION_WIDTH-1:0] ft_o_model_version;
    logic [VERSION_WIDTH-1:0] ft_o_adapt_version;
    logic ft_o_capture_complete;
    logic ft_o_learning_start_fire;
    logic ft_o_commit_pulse;
    logic ft_o_commit_code_dirty;
    logic ft_o_commit_optimizer_dirty;
    logic ft_o_detach_pulse;
    logic ft_o_commit_pending;
    logic ft_o_inference_hold;
    logic ft_o_initialized;
    logic ft_o_versions_aligned;
    logic [1:0] ft_o_array_active_owner;
    logic [15:0] ft_o_array_tile_idle;
    logic [15:0] ft_o_array_tile_sticky_error;
    logic ft_o_context_active;
    logic ft_o_context_is_classifier;
    logic ft_o_busy;
    logic ft_o_quiescent;
    logic ft_o_sticky_error;
    logic ft_i_clk;
    logic ft_i_rst_n;
    logic ft_i_clear_error;
    logic ft_i_srnn_phase_abort;
    logic ft_i_srnn_phase_capture_cfg_valid;
    logic ft_o_srnn_phase_capture_cfg_ready;
    logic [$clog2(SRNN_BACKBONE_MAX + 1)-1:0] ft_i_srnn_phase_record_count;
    logic ft_i_srnn_phase_start_valid;
    logic ft_o_srnn_phase_start_ready;
    logic ft_i_srnn_phase_learning_enable;
    logic ft_i_srnn_phase_target_valid;
    logic ft_o_srnn_phase_target_ready;
    logic [5:0] ft_i_srnn_phase_target_sample_id;
    logic [9:0] ft_i_srnn_phase_target_backbone_id;
    logic signed [31:0] ft_i_srnn_phase_target_q14;
    logic ft_i_srnn_phase_target_last;
    logic ft_i_srnn_phase_target_done;
    logic ft_i_srnn_phase_target_error;
    logic ft_o_srnn_phase_capture_done;
    logic ft_o_srnn_phase_capture_complete;
    logic ft_o_srnn_phase_delivery_done;
    logic ft_o_srnn_phase_busy;
    logic ft_o_srnn_phase_idle;
    logic ft_o_srnn_phase_sticky_error;
    logic [2:0] ft_i_parameter_owner;
    logic [2:0] ft_i_topology_learning_enable;
    logic ft_i_learning_fc_cfg_valid;
    logic ft_o_learning_fc_cfg_ready;
    logic [7:0] ft_i_learning_fc_profile_id;
    logic [7:0] ft_i_learning_fc_w2_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] ft_i_learning_fc_w2_parameter_base;
    logic [4:0] ft_i_learning_fc_batch_count;
    logic [8:0] ft_i_learning_fc_hidden_count;
    logic [4:0] ft_i_learning_fc_output_count;
    logic [15:0] ft_i_learning_fc_vote_increment;
    logic [15:0] ft_i_learning_fc_vote_threshold;
    logic signed [CODE_WIDTH-1:0] ft_i_learning_fc_code_min;
    logic signed [CODE_WIDTH-1:0] ft_i_learning_fc_code_max;
    logic ft_o_learning_fc_cfg_error;
    logic ft_i_learning_fc_release;
    logic ft_o_learning_fc_release_ready;
    logic ft_o_learning_fc_window_commit_valid;
    logic ft_i_learning_fc_window_commit_ready;
    logic [31:0] ft_o_learning_fc_window_commit_sequence;
    logic ft_o_learning_fc_window_busy;
    logic ft_o_learning_fc_error;
    logic ft_i_learning_fc_parameter_load_valid;
    logic ft_o_learning_fc_parameter_load_ready;
    logic [PARAM_ADDR_WIDTH-1:0] ft_i_learning_fc_parameter_load_addr;
    logic signed [CODE_WIDTH-1:0] ft_i_learning_fc_parameter_load_code;
    logic signed [RESIDUAL_WIDTH-1:0] ft_i_learning_fc_parameter_load_residual;
    logic ft_i_learning_srnn_cfg_valid;
    logic ft_o_learning_srnn_cfg_ready;
    logic [7:0] ft_i_learning_srnn_profile_id;
    logic [7:0] ft_i_learning_srnn_wh1_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] ft_i_learning_srnn_wh1_base;
    logic [15:0] ft_i_learning_srnn_wh1_increment;
    logic [15:0] ft_i_learning_srnn_wh1_threshold;
    logic [7:0] ft_i_learning_srnn_wout_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] ft_i_learning_srnn_wout_base;
    logic [15:0] ft_i_learning_srnn_wout_increment;
    logic [15:0] ft_i_learning_srnn_wout_threshold;
    logic signed [CODE_WIDTH-1:0] ft_i_learning_srnn_code_min;
    logic signed [CODE_WIDTH-1:0] ft_i_learning_srnn_code_max;
    logic [7:0] ft_i_learning_srnn_batch_count;
    logic [8:0] ft_i_learning_srnn_hidden_count;
    logic [9:0] ft_i_learning_srnn_backbone_count;
    logic [4:0] ft_i_learning_srnn_class_count;
    logic [PARAM_ADDR_WIDTH-1:0] ft_i_learning_srnn_wout_element_count;
    logic ft_o_learning_srnn_cfg_error;
    logic ft_i_learning_srnn_parameter_load_valid;
    logic ft_o_learning_srnn_parameter_load_ready;
    logic [PARAM_ADDR_WIDTH-1:0] ft_i_learning_srnn_parameter_load_addr;
    logic signed [CODE_WIDTH-1:0] ft_i_learning_srnn_parameter_load_code;
    logic signed [RESIDUAL_WIDTH-1:0] ft_i_learning_srnn_parameter_load_residual;
    logic [$clog2(SRNN_TIMESTEP_MAX)-1:0] ft_i_learning_srnn_timestep_id;
    logic [15:0] ft_i_learning_srnn_sequence_version;
    logic ft_i_learning_srnn_abort_timestep;
    logic ft_o_learning_srnn_abort_ready;
    logic ft_o_learning_srnn_target_trace_start;
    logic ft_i_learning_srnn_target_trace_start_ready;
    logic [$clog2(SRNN_TIMESTEP_MAX)-1:0] ft_o_learning_srnn_target_trace_timestep_id;
    logic ft_i_learning_srnn_target_trace_valid;
    logic ft_o_learning_srnn_target_trace_ready;
    logic [7:0] ft_i_learning_srnn_target_trace_sample_id;
    logic [9:0] ft_i_learning_srnn_target_trace_backbone_id;
    logic signed [31:0] ft_i_learning_srnn_target_trace_q14;
    logic ft_i_learning_srnn_target_trace_last;
    logic ft_i_learning_srnn_target_trace_done;
    logic ft_i_learning_srnn_target_trace_error;
    logic ft_o_learning_srnn_label_replay_start;
    logic ft_i_learning_srnn_label_replay_start_ready;
    logic ft_i_learning_srnn_label_valid;
    logic ft_o_learning_srnn_label_ready;
    logic [7:0] ft_i_learning_srnn_label_sample_id;
    logic [4:0] ft_i_learning_srnn_label_class_id;
    logic ft_i_learning_srnn_label_last;
    logic ft_i_learning_srnn_restore_complete;
    logic ft_o_learning_srnn_restore_ready;
    logic ft_o_learning_srnn_recovery_required;
    logic ft_o_learning_srnn_learning_commit;
    logic ft_o_learning_srnn_detach;
    logic ft_o_learning_srnn_logit_valid;
    logic ft_i_learning_srnn_logit_ready;
    logic [7:0] ft_o_learning_srnn_logit_sample_id;
    logic [4:0] ft_o_learning_srnn_logit_class_id;
    logic signed [63:0] ft_o_learning_srnn_logit_q28;
    logic ft_o_learning_srnn_logit_last;
    logic ft_o_learning_srnn_timestep_done;
    logic ft_o_learning_srnn_timestep_aborted;
    logic ft_o_learning_srnn_busy;
    logic ft_o_learning_srnn_error;
    logic [PARAM_ADDR_WIDTH:0] ft_o_learning_srnn_zero_gradient_count;
    logic [PARAM_ADDR_WIDTH:0] ft_o_learning_srnn_update_count;
    logic ft_o_store_load_valid;
    logic ft_i_store_load_ready;
    logic [PARAM_ADDR_WIDTH-1:0] ft_o_store_load_addr;
    logic signed [CODE_WIDTH-1:0] ft_o_store_load_code;
    logic signed [RESIDUAL_WIDTH-1:0] ft_o_store_load_residual;
    logic ft_o_store_learning_read_valid;
    logic ft_i_store_learning_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] ft_o_store_learning_read_addr;
    logic [TAG_WIDTH-1:0] ft_o_store_learning_read_tag;
    logic ft_i_store_learning_response_valid;
    logic ft_o_store_learning_response_ready;
    logic [TAG_WIDTH-1:0] ft_i_store_learning_response_tag;
    logic signed [CODE_WIDTH-1:0] ft_i_store_learning_response_code;
    logic signed [RESIDUAL_WIDTH-1:0] ft_i_store_learning_response_residual;
    logic ft_i_store_learning_response_error;
    logic ft_o_store_learning_write_valid;
    logic ft_i_store_learning_write_ready;
    logic [PARAM_ADDR_WIDTH-1:0] ft_o_store_learning_write_addr;
    logic signed [CODE_WIDTH-1:0] ft_o_store_learning_write_code;
    logic signed [RESIDUAL_WIDTH-1:0] ft_o_store_learning_write_residual;
    logic ft_o_store_learning_write_code_enable;
    logic ft_o_store_learning_write_residual_enable;
    logic ft_o_store_forward_read_valid;
    logic ft_i_store_forward_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] ft_o_store_forward_read_addr;
    logic [TAG_WIDTH-1:0] ft_o_store_forward_read_tag;
    logic ft_i_store_forward_response_valid;
    logic ft_o_store_forward_response_ready;
    logic [TAG_WIDTH-1:0] ft_i_store_forward_response_tag;
    logic signed [CODE_WIDTH-1:0] ft_i_store_forward_response_code;
    logic ft_i_store_forward_response_error;
    logic ft_i_store_quiescent;
    logic ft_i_store_protocol_error;
    logic ft_o_backend_quiescent;
    logic ft_o_backend_error;
    logic ft_o_owner_encoding_error;
    assign ft_i_initialize_valid = fg_initialize;
    assign ft_i_classifier_pair = 1'b1;
    assign ft_i_context_model_version = CLASSIFIER_VERSION;
    assign ft_i_context_adapt_version = 16'd0;
    assign ft_i_system_parameter_owner = 1'b1;
    assign ft_i_system_learning_enable = fg_learning;
    assign ft_i_system_profile_aligned = 1'b1;
    assign ft_i_system_profile_id = 8'h20;
    assign ft_i_request_valid = fg_request;
    assign ft_i_learning_enable = fg_learning;
    assign ft_i_batch_count = BATCH_COUNT_WIDTH'(tv_query ? 1 : 4);
    assign ft_i_timestep = TIMESTEP_WIDTH'(tv_query ? 0 : tv_window*4);
    assign ft_i_timestep_count = TIMESTEP_COUNT_WIDTH'(4);
    assign ft_i_sample_version = fg_window_version;
    assign ft_i_labels_flat = {{(MAX_BATCH-4)*CLASS_WIDTH{1'b0}}, fg_label[3][CLASS_WIDTH-1:0], fg_label[2][CLASS_WIDTH-1:0], fg_label[1][CLASS_WIDTH-1:0], fg_label[0][CLASS_WIDTH-1:0]};
    assign ft_i_fc_srnn_layer_valid = '0;
    assign ft_i_fc_srnn_mode = '0;
    assign ft_i_fc_srnn_weight_int8 = '0;
    assign ft_i_fc_srnn_current_shift6 = '0;
    assign ft_i_fc_srnn_neuron_count = '0;
    assign ft_i_fc_srnn_feedforward_group_count = '0;
    assign ft_i_fc_srnn_recurrent_group_count = '0;
    assign ft_i_fc_srnn_recurrent_enable = '0;
    assign ft_i_fc_srnn_leak = '0;
    assign ft_i_fc_srnn_threshold = '0;
    assign ft_i_fc_srnn_reset_to_zero = '0;
    assign ft_i_fc_srnn_trace_decay = '0;
    assign ft_i_fc_srnn_transaction_tag = '0;
    assign ft_i_fc_srnn_work_base_addr = '0;
    assign ft_i_fc_srnn_destination_tile_base = '0;
    assign ft_i_fc_srnn_group_valid = '0;
    assign ft_i_fc_srnn_group_recurrent = '0;
    assign ft_i_fc_srnn_group_source_valid = '0;
    assign ft_i_fc_srnn_group_weight_addr0 = '0;
    assign ft_i_fc_srnn_group_weight_addr1 = '0;
    assign ft_i_fc_srnn_group_weight_addr2 = '0;
    assign ft_i_fc_srnn_group_weight_addr3 = '0;
    assign ft_i_srnn_group_streamer_select = '0;
    assign ft_i_srnn_group_streamer_start_valid = '0;
    assign ft_i_srnn_group_feedforward_source_count = '0;
    assign ft_i_srnn_group_recurrent_source_count = '0;
    assign ft_i_srnn_group_feedforward_cache_row_base = '0;
    assign ft_i_srnn_group_recurrent_cache_row_base = '0;
    assign ft_i_srnn_group_feedforward_spike_flat = '0;
    assign ft_i_srnn_group_recurrent_spike_flat = '0;
    assign ft_i_fc_srnn_observation_ready = '0;
    assign ft_i_fc_srnn_observation_sink_idle = '0;
    assign ft_i_fc_srnn_learning_cfg_valid = '0;
    assign ft_i_fc_srnn_learning_cfg_mode = '0;
    assign ft_i_fc_srnn_learning_cfg_tag = '0;
    assign ft_i_fc_srnn_learning_sample_id = '0;
    assign ft_i_fc_srnn_learning_neuron_base = '0;
    assign ft_i_fc_srnn_learning_neuron_count = '0;
    assign ft_i_fc_srnn_learning_fc_threshold = '0;
    assign ft_i_fc_srnn_learning_fc_output_layer = '0;
    assign ft_i_fc_srnn_learning_fc_label = '0;
    assign ft_i_fc_srnn_learning_fc_logit_scale_enable = '0;
    assign ft_i_fc_srnn_learning_fc_logit_scale_q16 = '0;
    assign ft_i_fc_srnn_learning_fc_logit_accumulate = '0;
    assign ft_i_fc_srnn_learning_fc_sample_start = '0;
    assign ft_i_fc_srnn_learning_srnn_path = '0;
    assign ft_i_fc_w0_refill_valid = '0;
    assign ft_i_fc_w0_refill_all = '0;
    assign ft_i_fc_cache_refill_weight_int8 = '0;
    assign ft_i_fc_cache_refill_destination_tile_base = '0;
    assign ft_i_fc_w0_dirty_feature = '0;
    assign ft_i_fc_w0_dirty_tile = '0;
    assign ft_i_fc_w0_parameter_base = '0;
    assign ft_i_fc_w0_hidden_stride = '0;
    assign ft_i_fc_cache_row_base = '0;
    assign ft_i_fc_w0_feature_count = '0;
    assign ft_i_fc_w0_hidden_count = '0;
    assign ft_i_fc_w0_tile_count = '0;
    assign ft_i_fc_w0_version = '0;
    assign ft_i_srnn_backbone_refill_valid = '0;
    assign ft_i_srnn_backbone_refill_all = '0;
    assign ft_i_srnn_backbone_dirty_source = '0;
    assign ft_i_srnn_backbone_dirty_tile = '0;
    assign ft_i_srnn_backbone_parameter_base = '0;
    assign ft_i_srnn_backbone_cache_row_base = '0;
    assign ft_i_srnn_backbone_source_count = '0;
    assign ft_i_srnn_backbone_destination_count = '0;
    assign ft_i_srnn_backbone_tile_base = '0;
    assign ft_i_srnn_backbone_version = '0;
    assign ft_i_srnn_backbone_forward_read_ready = '0;
    assign ft_i_srnn_backbone_forward_response_valid = '0;
    assign ft_i_srnn_backbone_forward_response_tag = '0;
    assign ft_i_srnn_backbone_forward_response_code = '0;
    assign ft_i_srnn_backbone_forward_response_error = '0;
    assign ft_i_profile_id = 8'h20;
    assign ft_i_conv_tensor_id = 8'h08;
    assign ft_i_classifier_tensor_id = 8'h12;
    assign ft_i_total_input_channels = CHANNEL_COUNT_WIDTH'(CONV8_CHANNELS);
    assign ft_i_total_output_channels = CHANNEL_COUNT_WIDTH'(CONV8_CHANNELS);
    assign ft_i_conv_parameter_base = fg_conv_base(layer_phase);
    assign ft_i_conv_weight_q16_base = '0;
    assign ft_i_classifier_parameter_base = PARAM_ADDR_WIDTH'(FG_CLASSIFIER_BASE);
    assign ft_i_classifier_feature_count = FEATURE_COUNT_WIDTH'(CLASSIFIER_FEATURES);
    assign ft_i_classifier_class_count = CLASS_COUNT_WIDTH'(CLASSES);
    assign ft_i_classifier_trace_leak_valid = 1'b1;
    assign ft_i_classifier_trace_leak_q8_8 = 16'sd250;
    assign ft_i_conv_vote_increment = '0;
    assign ft_i_conv_vote_threshold = '0;
    assign ft_i_classifier_vote_increment = 16'd0;
    assign ft_i_classifier_vote_threshold = 16'h8000;
    assign ft_i_code_min = -16'sd32768;
    assign ft_i_code_max = 16'sd32767;
    assign ft_i_forward_start_valid = '0;
    assign ft_i_forward_done = '0;
    assign ft_i_restore_complete = '0;
    assign ft_i_conv_in_width = cfg_in_width;
    assign ft_i_conv_in_height = cfg_in_height;
    assign ft_i_conv_in_channels = cfg_in_channels;
    assign ft_i_conv_out_width = cfg_out_width;
    assign ft_i_conv_out_height = cfg_out_height;
    assign ft_i_conv_out_channels = cfg_out_channels;
    assign ft_i_conv_stride = 8'd1;
    assign ft_i_conv_padding = cfg_padding;
    assign ft_i_conv_pool_mode = cfg_pool_mode;
    assign ft_i_conv_work_base_addr = '0;
    assign ft_i_conv_leak = 16'd135;
    assign ft_i_conv_threshold = 32'd256;
    assign ft_i_conv_reset_to_zero = 1'b0;
    assign ft_i_conv_trace_decay = 16'd128;
    assign ft_i_conv_layer_tag = cfg_layer_tag;
    assign ft_i_target_feature_count = FEATURE_COUNT_WIDTH'(CLASSIFIER_FEATURES);
    assign ft_i_target_trace_leak_q8_8 = 16'sd250;
    assign ft_i_target_trace_state_reset = !tv_query && tv_window == 0;
    assign ft_i_classifier_feature_channel_base = '0;
    assign ft_i_classifier_feature_channel_count = 10'(CONV8_CHANNELS);
    assign ft_i_classifier_work_base_addr = WORK_AW'(CLASSIFIER_WORK_BASE);
    assign ft_i_classifier_trace_decay = 16'd250;
    assign ft_i_classifier_layer_tag = CLASSIFIER_TAG;
    assign ft_i_frame_index = FRAME_WIDTH'(tv_time);
    assign ft_i_graph_descriptor_valid = start_valid;
    assign ft_i_graph_descriptor_last = layer_phase == 8;
    assign ft_i_state_service_ready = context_load_ready || context_release_ready;
    assign ft_i_feature_request_ready = feature_request_ready;
    assign ft_i_feature_response_valid = feature_response_valid;
    assign ft_i_feature_mask0 = feature_mask0;
    assign ft_i_feature_mask1 = feature_mask1;
    assign ft_i_feature_mask2 = feature_mask2;
    assign ft_i_feature_mask3 = feature_mask3;
    assign ft_i_feature_service_error = 1'b0;
    assign ft_i_feature_write_slot = cfg_feature_write_slot;
    assign ft_i_feature_write_base_addr = cfg_feature_write_base;
    assign ft_i_feature_write_row_stride_words = cfg_feature_write_row_stride;
    assign ft_i_feature_write_channel_stride_words = cfg_feature_write_channel_stride;
    assign ft_i_feature_update_ready = feature_update_ready;
    assign ft_i_feature_update_done = feature_update_done;
    assign ft_i_maintenance_tile = maintenance_tile;
    assign ft_i_state_maintenance_write_valid = state_write_valid;
    assign ft_i_state_maintenance_write_addr = state_write_addr;
    assign ft_i_state_maintenance_write_data = state_write_data;
    assign ft_i_state_maintenance_write_mask = state_write_mask;
    assign ft_i_state_maintenance_read_valid = state_read_valid;
    assign ft_i_state_maintenance_read_addr = state_read_addr;
    assign ft_i_state_maintenance_result_ready = state_result_ready;
    assign ft_i_trace_maintenance_write_valid = trace_write_valid;
    assign ft_i_trace_maintenance_write_addr = trace_write_addr;
    assign ft_i_trace_maintenance_write_data = trace_write_data;
    assign ft_i_trace_maintenance_write_mask = trace_write_mask;
    assign ft_i_trace_maintenance_read_valid = trace_read_valid;
    assign ft_i_trace_maintenance_read_addr = trace_read_addr;
    assign ft_i_trace_maintenance_result_ready = trace_result_ready;
    assign ft_i_clk = clk;
    assign ft_i_rst_n = rst_n;
    assign ft_i_clear_error = clear_error;
    assign ft_i_srnn_phase_abort = '0;
    assign ft_i_srnn_phase_capture_cfg_valid = '0;
    assign ft_i_srnn_phase_record_count = '0;
    assign ft_i_srnn_phase_start_valid = '0;
    assign ft_i_srnn_phase_learning_enable = '0;
    assign ft_i_srnn_phase_target_valid = '0;
    assign ft_i_srnn_phase_target_sample_id = '0;
    assign ft_i_srnn_phase_target_backbone_id = '0;
    assign ft_i_srnn_phase_target_q14 = '0;
    assign ft_i_srnn_phase_target_last = '0;
    assign ft_i_srnn_phase_target_done = '0;
    assign ft_i_srnn_phase_target_error = '0;
    assign ft_i_parameter_owner = 3'b100;
    assign ft_i_topology_learning_enable = 3'b100;
    assign ft_i_learning_fc_cfg_valid = '0;
    assign ft_i_learning_fc_profile_id = '0;
    assign ft_i_learning_fc_w2_tensor_id = '0;
    assign ft_i_learning_fc_w2_parameter_base = '0;
    assign ft_i_learning_fc_batch_count = '0;
    assign ft_i_learning_fc_hidden_count = '0;
    assign ft_i_learning_fc_output_count = '0;
    assign ft_i_learning_fc_vote_increment = '0;
    assign ft_i_learning_fc_vote_threshold = '0;
    assign ft_i_learning_fc_code_min = '0;
    assign ft_i_learning_fc_code_max = '0;
    assign ft_i_learning_fc_release = '0;
    assign ft_i_learning_fc_window_commit_ready = '0;
    assign ft_i_learning_fc_parameter_load_valid = '0;
    assign ft_i_learning_fc_parameter_load_addr = '0;
    assign ft_i_learning_fc_parameter_load_code = '0;
    assign ft_i_learning_fc_parameter_load_residual = '0;
    assign ft_i_learning_srnn_cfg_valid = '0;
    assign ft_i_learning_srnn_profile_id = '0;
    assign ft_i_learning_srnn_wh1_tensor_id = '0;
    assign ft_i_learning_srnn_wh1_base = '0;
    assign ft_i_learning_srnn_wh1_increment = '0;
    assign ft_i_learning_srnn_wh1_threshold = '0;
    assign ft_i_learning_srnn_wout_tensor_id = '0;
    assign ft_i_learning_srnn_wout_base = '0;
    assign ft_i_learning_srnn_wout_increment = '0;
    assign ft_i_learning_srnn_wout_threshold = '0;
    assign ft_i_learning_srnn_code_min = '0;
    assign ft_i_learning_srnn_code_max = '0;
    assign ft_i_learning_srnn_batch_count = '0;
    assign ft_i_learning_srnn_hidden_count = '0;
    assign ft_i_learning_srnn_backbone_count = '0;
    assign ft_i_learning_srnn_class_count = '0;
    assign ft_i_learning_srnn_wout_element_count = '0;
    assign ft_i_learning_srnn_parameter_load_valid = '0;
    assign ft_i_learning_srnn_parameter_load_addr = '0;
    assign ft_i_learning_srnn_parameter_load_code = '0;
    assign ft_i_learning_srnn_parameter_load_residual = '0;
    assign ft_i_learning_srnn_timestep_id = '0;
    assign ft_i_learning_srnn_sequence_version = '0;
    assign ft_i_learning_srnn_abort_timestep = '0;
    assign ft_i_learning_srnn_target_trace_start_ready = '0;
    assign ft_i_learning_srnn_target_trace_valid = '0;
    assign ft_i_learning_srnn_target_trace_sample_id = '0;
    assign ft_i_learning_srnn_target_trace_backbone_id = '0;
    assign ft_i_learning_srnn_target_trace_q14 = '0;
    assign ft_i_learning_srnn_target_trace_last = '0;
    assign ft_i_learning_srnn_target_trace_done = '0;
    assign ft_i_learning_srnn_target_trace_error = '0;
    assign ft_i_learning_srnn_label_replay_start_ready = '0;
    assign ft_i_learning_srnn_label_valid = '0;
    assign ft_i_learning_srnn_label_sample_id = '0;
    assign ft_i_learning_srnn_label_class_id = '0;
    assign ft_i_learning_srnn_label_last = '0;
    assign ft_i_learning_srnn_restore_complete = '0;
    assign ft_i_learning_srnn_logit_ready = '0;
    assign ft_i_store_load_ready = 1'b0;
    assign ft_i_store_learning_read_ready = !fg_learning_valid;
    assign ft_i_store_learning_response_valid = fg_learning_valid;
    assign ft_i_store_learning_response_tag = fg_learning_tag;
    assign ft_i_store_learning_response_code = fg_learning_code;
    assign ft_i_store_learning_response_residual = fg_learning_residual;
    assign ft_i_store_learning_response_error = 1'b0;
    assign ft_i_store_learning_write_ready = !fg_learning_valid;
    assign ft_i_store_forward_read_ready = !fg_forward_valid || ft_o_store_forward_response_ready;
    assign ft_i_store_forward_response_valid = fg_forward_valid;
    assign ft_i_store_forward_response_tag = fg_forward_tag;
    assign ft_i_store_forward_response_code = fg_forward_code;
    assign ft_i_store_forward_response_error = 1'b0;
    assign ft_i_store_quiescent = !fg_forward_valid && !fg_learning_valid && !ft_o_store_forward_read_valid && !ft_o_store_learning_read_valid && !ft_o_store_learning_write_valid && !ft_o_store_load_valid;
    assign ft_i_store_protocol_error = 1'b0;
    unified_h40_raw_source_top_v1 #(
        .RAW_SOURCE_POLICY(1),
        .TOPOLOGY_MASK(TOPOLOGY_MASK),
        .M2_CACHE_DEPTH(512),
        .SCNN_WEIGHT_FRACTIONAL_BITS(SCNN_WEIGHT_FRACTIONAL_BITS),
        .ACCUMULATE_SCNN_SAMPLE_LOGITS(ACCUMULATE_SCNN_SAMPLE_LOGITS),
        .ENABLE_SCNN_FRAME_SERVICE(ENABLE_SCNN_FRAME_SERVICE),
        .SRNN_M2_PARAM_DEPTH(SRNN_M2_PARAM_DEPTH),
        .SRNN_M2_BATCH_MAX(SRNN_M2_BATCH_MAX),
        .SRNN_M2_TIMESTEP_MAX(SRNN_M2_TIMESTEP_MAX),
        .SRNN_M2_BACKBONE_MAX(SRNN_M2_BACKBONE_MAX),
        .SRNN_M2_HIDDEN_MAX(SRNN_M2_HIDDEN_MAX),
        .SRNN_M2_CLASS_MAX(SRNN_M2_CLASS_MAX),
        .SRNN_BACKBONE_PARAM_DEPTH(SRNN_BACKBONE_PARAM_DEPTH),
        .SRNN_BACKBONE_PARAM_ADDR_WIDTH(SRNN_BACKBONE_PARAM_ADDR_WIDTH),
        .SRNN_FEEDFORWARD_SOURCE_MAX(SRNN_FEEDFORWARD_SOURCE_MAX),
        .SRNN_RECURRENT_SOURCE_MAX(SRNN_RECURRENT_SOURCE_MAX),
        .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .MEMORY_DEPTH(MEMORY_DEPTH),
        .MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH),
        .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH),
        .FEATURE_WORD_DEPTH(FEATURE_WORD_DEPTH),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_DEPTH(PARAM_DEPTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .MAX_BATCH(MAX_BATCH),
        .MAX_TIMESTEPS(MAX_TIMESTEPS),
        .CLASSIFIER_CAPTURE_MAX_BATCH(CLASSIFIER_CAPTURE_MAX_BATCH),
        .CLASSIFIER_CAPTURE_MAX_TIMESTEPS(CLASSIFIER_CAPTURE_MAX_TIMESTEPS),
        .MAX_INPUT_CHANNELS(MAX_INPUT_CHANNELS),
        .MAX_OUTPUT_CHANNELS(MAX_OUTPUT_CHANNELS),
        .OBS_INPUT_CHANNELS(OBS_INPUT_CHANNELS),
        .OBS_OUTPUT_CHANNELS(OBS_OUTPUT_CHANNELS),
        .SPATIAL(SPATIAL),
        .POOLED_SPATIAL(POOLED_SPATIAL),
        .KERNEL_SIZE(KERNEL_SIZE),
        .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH),
        .MAX_FEATURES(MAX_FEATURES),
        .MAX_CLASSES(MAX_CLASSES),
        .LANES(LANES),
        .ENGINES(ENGINES),
        .FRAME_WIDTH(FRAME_WIDTH),
        .SELECTED_LAYER(SELECTED_LAYER),
        .ENABLE_SCNN_CONV8_LEARNING(ENABLE_SCNN_CONV8_LEARNING),
        .INTERNAL_MODULATION_BATCH(INTERNAL_MODULATION_BATCH),
        .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .TIMESTEP_WIDTH(TIMESTEP_WIDTH),
        .TIMESTEP_COUNT_WIDTH(TIMESTEP_COUNT_WIDTH),
        .CHANNEL_WIDTH(CHANNEL_WIDTH),
        .CHANNEL_COUNT_WIDTH(CHANNEL_COUNT_WIDTH),
        .FEATURE_WIDTH(FEATURE_WIDTH),
        .FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
        .CLASS_WIDTH(CLASS_WIDTH),
        .CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH),
        .ENABLE_ONLINE_LEARNING(ENABLE_ONLINE_LEARNING),
        .ENABLE_SRNN_WH1_LEARNING(ENABLE_SRNN_WH1_LEARNING),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .FC_MAX_HIDDEN(FC_MAX_HIDDEN),
        .FC_MAX_OUTPUTS(FC_MAX_OUTPUTS),
        .SRNN_BATCH_MAX(SRNN_BATCH_MAX),
        .SRNN_TIMESTEP_MAX(SRNN_TIMESTEP_MAX),
        .SRNN_HIDDEN_MAX(SRNN_HIDDEN_MAX),
        .SRNN_BACKBONE_MAX(SRNN_BACKBONE_MAX),
        .SRNN_CLASS_MAX(SRNN_CLASS_MAX)
    ) dut (
        .i_raw_source_active(1'b0),
        .i_raw_count_valid(1'b0),
        .i_raw_counts(32'b0),
        .i_fc_srnn_group_feedforward_end(1'b0),
        .i_initialize_valid(ft_i_initialize_valid),
        .o_initialize_ready(ft_o_initialize_ready),
        .o_initialize_done(ft_o_initialize_done),
        .i_classifier_pair(ft_i_classifier_pair),
        .i_context_model_version(ft_i_context_model_version),
        .i_context_adapt_version(ft_i_context_adapt_version),
        .i_system_parameter_owner(ft_i_system_parameter_owner),
        .i_system_learning_enable(ft_i_system_learning_enable),
        .i_system_profile_aligned(ft_i_system_profile_aligned),
        .i_system_profile_id(ft_i_system_profile_id),
        .i_request_valid(ft_i_request_valid),
        .o_request_ready(ft_o_request_ready),
        .i_learning_enable(ft_i_learning_enable),
        .i_batch_count(ft_i_batch_count),
        .i_timestep(ft_i_timestep),
        .i_timestep_count(ft_i_timestep_count),
        .i_sample_version(ft_i_sample_version),
        .i_labels_flat(ft_i_labels_flat),
        .o_done(ft_o_done),
        .i_fc_srnn_layer_valid(ft_i_fc_srnn_layer_valid),
        .o_fc_srnn_layer_ready(ft_o_fc_srnn_layer_ready),
        .i_fc_srnn_mode(ft_i_fc_srnn_mode),
        .i_fc_srnn_weight_int8(ft_i_fc_srnn_weight_int8),
        .i_fc_srnn_current_shift6(ft_i_fc_srnn_current_shift6),
        .i_fc_srnn_neuron_count(ft_i_fc_srnn_neuron_count),
        .i_fc_srnn_feedforward_group_count(ft_i_fc_srnn_feedforward_group_count),
        .i_fc_srnn_recurrent_group_count(ft_i_fc_srnn_recurrent_group_count),
        .i_fc_srnn_recurrent_enable(ft_i_fc_srnn_recurrent_enable),
        .i_fc_srnn_leak(ft_i_fc_srnn_leak),
        .i_fc_srnn_threshold(ft_i_fc_srnn_threshold),
        .i_fc_srnn_reset_to_zero(ft_i_fc_srnn_reset_to_zero),
        .i_fc_srnn_trace_decay(ft_i_fc_srnn_trace_decay),
        .i_fc_srnn_transaction_tag(ft_i_fc_srnn_transaction_tag),
        .i_fc_srnn_work_base_addr(ft_i_fc_srnn_work_base_addr),
        .i_fc_srnn_destination_tile_base(ft_i_fc_srnn_destination_tile_base),
        .i_fc_srnn_group_valid(ft_i_fc_srnn_group_valid),
        .o_fc_srnn_group_ready(ft_o_fc_srnn_group_ready),
        .i_fc_srnn_group_recurrent(ft_i_fc_srnn_group_recurrent),
        .i_fc_srnn_group_source_valid(ft_i_fc_srnn_group_source_valid),
        .i_fc_srnn_group_weight_addr0(ft_i_fc_srnn_group_weight_addr0),
        .i_fc_srnn_group_weight_addr1(ft_i_fc_srnn_group_weight_addr1),
        .i_fc_srnn_group_weight_addr2(ft_i_fc_srnn_group_weight_addr2),
        .i_fc_srnn_group_weight_addr3(ft_i_fc_srnn_group_weight_addr3),
        .i_srnn_group_streamer_select(ft_i_srnn_group_streamer_select),
        .i_srnn_group_streamer_start_valid(ft_i_srnn_group_streamer_start_valid),
        .o_srnn_group_streamer_start_ready(ft_o_srnn_group_streamer_start_ready),
        .i_srnn_group_feedforward_source_count(ft_i_srnn_group_feedforward_source_count),
        .i_srnn_group_recurrent_source_count(ft_i_srnn_group_recurrent_source_count),
        .i_srnn_group_feedforward_cache_row_base(ft_i_srnn_group_feedforward_cache_row_base),
        .i_srnn_group_recurrent_cache_row_base(ft_i_srnn_group_recurrent_cache_row_base),
        .i_srnn_group_feedforward_spike_flat(ft_i_srnn_group_feedforward_spike_flat),
        .i_srnn_group_recurrent_spike_flat(ft_i_srnn_group_recurrent_spike_flat),
        .o_srnn_group_streamer_done(ft_o_srnn_group_streamer_done),
        .o_srnn_group_streamer_busy(ft_o_srnn_group_streamer_busy),
        .o_srnn_group_streamer_sticky_error(ft_o_srnn_group_streamer_sticky_error),
        .o_srnn_group_streamer_error_code(ft_o_srnn_group_streamer_error_code),
        .o_fc_srnn_layer_done(ft_o_fc_srnn_layer_done),
        .o_fc_srnn_busy(ft_o_fc_srnn_busy),
        .o_fc_srnn_idle(ft_o_fc_srnn_idle),
        .o_fc_srnn_sticky_error(ft_o_fc_srnn_sticky_error),
        .o_fc_srnn_error_code(ft_o_fc_srnn_error_code),
        .o_fc_srnn_observation_valid(ft_o_fc_srnn_observation_valid),
        .i_fc_srnn_observation_ready(ft_i_fc_srnn_observation_ready),
        .i_fc_srnn_observation_sink_idle(ft_i_fc_srnn_observation_sink_idle),
        .o_fc_srnn_observation_mode(ft_o_fc_srnn_observation_mode),
        .o_fc_srnn_observation_transaction_tag(ft_o_fc_srnn_observation_transaction_tag),
        .o_fc_srnn_observation_neuron_id(ft_o_fc_srnn_observation_neuron_id),
        .o_fc_srnn_observation_spike(ft_o_fc_srnn_observation_spike),
        .o_fc_srnn_observation_pre_state(ft_o_fc_srnn_observation_pre_state),
        .o_fc_srnn_observation_trace_next(ft_o_fc_srnn_observation_trace_next),
        .o_fc_srnn_observation_last_in_record(ft_o_fc_srnn_observation_last_in_record),
        .o_fc_srnn_observation_busy(ft_o_fc_srnn_observation_busy),
        .o_fc_srnn_observation_sticky_error(ft_o_fc_srnn_observation_sticky_error),
        .i_fc_srnn_learning_cfg_valid(ft_i_fc_srnn_learning_cfg_valid),
        .o_fc_srnn_learning_cfg_ready(ft_o_fc_srnn_learning_cfg_ready),
        .i_fc_srnn_learning_cfg_mode(ft_i_fc_srnn_learning_cfg_mode),
        .i_fc_srnn_learning_cfg_tag(ft_i_fc_srnn_learning_cfg_tag),
        .i_fc_srnn_learning_sample_id(ft_i_fc_srnn_learning_sample_id),
        .i_fc_srnn_learning_neuron_base(ft_i_fc_srnn_learning_neuron_base),
        .i_fc_srnn_learning_neuron_count(ft_i_fc_srnn_learning_neuron_count),
        .i_fc_srnn_learning_fc_threshold(ft_i_fc_srnn_learning_fc_threshold),
        .i_fc_srnn_learning_fc_output_layer(ft_i_fc_srnn_learning_fc_output_layer),
        .i_fc_srnn_learning_fc_label(ft_i_fc_srnn_learning_fc_label),
        .i_fc_srnn_learning_fc_logit_scale_enable(ft_i_fc_srnn_learning_fc_logit_scale_enable),
        .i_fc_srnn_learning_fc_logit_scale_q16(ft_i_fc_srnn_learning_fc_logit_scale_q16),
        .i_fc_srnn_learning_fc_logit_accumulate(ft_i_fc_srnn_learning_fc_logit_accumulate),
        .i_fc_srnn_learning_fc_sample_start(ft_i_fc_srnn_learning_fc_sample_start),
        .i_fc_srnn_learning_srnn_path(ft_i_fc_srnn_learning_srnn_path),
        .o_fc_surrogate_valid(ft_o_fc_surrogate_valid),
        .o_fc_surrogate_sample(ft_o_fc_surrogate_sample),
        .o_fc_surrogate_hidden(ft_o_fc_surrogate_hidden),
        .o_fc_surrogate_q016(ft_o_fc_surrogate_q016),
        .o_fc_logit_last(ft_o_fc_logit_last),
        .o_fc_srnn_learning_capture_done(ft_o_fc_srnn_learning_capture_done),
        .o_fc_srnn_learning_done(ft_o_fc_srnn_learning_done),
        .o_fc_srnn_learning_busy(ft_o_fc_srnn_learning_busy),
        .o_fc_srnn_learning_idle(ft_o_fc_srnn_learning_idle),
        .o_fc_srnn_learning_sticky_error(ft_o_fc_srnn_learning_sticky_error),
        .o_fc_srnn_learning_error_source(ft_o_fc_srnn_learning_error_source),
        .o_fc_srnn_learning_error_code(ft_o_fc_srnn_learning_error_code),
        .i_fc_w0_refill_valid(ft_i_fc_w0_refill_valid),
        .o_fc_w0_refill_ready(ft_o_fc_w0_refill_ready),
        .i_fc_w0_refill_all(ft_i_fc_w0_refill_all),
        .i_fc_cache_refill_weight_int8(ft_i_fc_cache_refill_weight_int8),
        .i_fc_cache_refill_destination_tile_base(ft_i_fc_cache_refill_destination_tile_base),
        .i_fc_w0_dirty_feature(ft_i_fc_w0_dirty_feature),
        .i_fc_w0_dirty_tile(ft_i_fc_w0_dirty_tile),
        .i_fc_w0_parameter_base(ft_i_fc_w0_parameter_base),
        .i_fc_w0_hidden_stride(ft_i_fc_w0_hidden_stride),
        .i_fc_cache_row_base(ft_i_fc_cache_row_base),
        .i_fc_w0_feature_count(ft_i_fc_w0_feature_count),
        .i_fc_w0_hidden_count(ft_i_fc_w0_hidden_count),
        .i_fc_w0_tile_count(ft_i_fc_w0_tile_count),
        .i_fc_w0_version(ft_i_fc_w0_version),
        .o_fc_w0_refill_done(ft_o_fc_w0_refill_done),
        .o_fc_w0_rows_published(ft_o_fc_w0_rows_published),
        .o_fc_w0_published_version(ft_o_fc_w0_published_version),
        .o_cache_fill_grant(ft_o_cache_fill_grant),
        .o_cache_refill_busy(ft_o_cache_refill_busy),
        .o_cache_refill_quiescent(ft_o_cache_refill_quiescent),
        .o_cache_refill_sticky_error(ft_o_cache_refill_sticky_error),
        .i_srnn_backbone_refill_valid(ft_i_srnn_backbone_refill_valid),
        .o_srnn_backbone_refill_ready(ft_o_srnn_backbone_refill_ready),
        .i_srnn_backbone_refill_all(ft_i_srnn_backbone_refill_all),
        .i_srnn_backbone_dirty_source(ft_i_srnn_backbone_dirty_source),
        .i_srnn_backbone_dirty_tile(ft_i_srnn_backbone_dirty_tile),
        .i_srnn_backbone_parameter_base(ft_i_srnn_backbone_parameter_base),
        .i_srnn_backbone_cache_row_base(ft_i_srnn_backbone_cache_row_base),
        .i_srnn_backbone_source_count(ft_i_srnn_backbone_source_count),
        .i_srnn_backbone_destination_count(ft_i_srnn_backbone_destination_count),
        .i_srnn_backbone_tile_base(ft_i_srnn_backbone_tile_base),
        .i_srnn_backbone_version(ft_i_srnn_backbone_version),
        .o_srnn_backbone_forward_read_valid(ft_o_srnn_backbone_forward_read_valid),
        .i_srnn_backbone_forward_read_ready(ft_i_srnn_backbone_forward_read_ready),
        .o_srnn_backbone_forward_read_addr(ft_o_srnn_backbone_forward_read_addr),
        .o_srnn_backbone_forward_read_tag(ft_o_srnn_backbone_forward_read_tag),
        .i_srnn_backbone_forward_response_valid(ft_i_srnn_backbone_forward_response_valid),
        .o_srnn_backbone_forward_response_ready(ft_o_srnn_backbone_forward_response_ready),
        .i_srnn_backbone_forward_response_tag(ft_i_srnn_backbone_forward_response_tag),
        .i_srnn_backbone_forward_response_code(ft_i_srnn_backbone_forward_response_code),
        .i_srnn_backbone_forward_response_error(ft_i_srnn_backbone_forward_response_error),
        .o_srnn_backbone_refill_done(ft_o_srnn_backbone_refill_done),
        .o_srnn_backbone_rows_published(ft_o_srnn_backbone_rows_published),
        .o_srnn_backbone_published_version(ft_o_srnn_backbone_published_version),
        .i_profile_id(ft_i_profile_id),
        .i_conv_tensor_id(ft_i_conv_tensor_id),
        .i_classifier_tensor_id(ft_i_classifier_tensor_id),
        .i_total_input_channels(ft_i_total_input_channels),
        .i_total_output_channels(ft_i_total_output_channels),
        .i_conv_parameter_base(ft_i_conv_parameter_base),
        .i_conv_weight_q16_base(ft_i_conv_weight_q16_base),
        .i_classifier_parameter_base(ft_i_classifier_parameter_base),
        .i_classifier_feature_count(ft_i_classifier_feature_count),
        .i_classifier_class_count(ft_i_classifier_class_count),
        .i_classifier_trace_leak_valid(ft_i_classifier_trace_leak_valid),
        .i_classifier_trace_leak_q8_8(ft_i_classifier_trace_leak_q8_8),
        .i_conv_vote_increment(ft_i_conv_vote_increment),
        .i_conv_vote_threshold(ft_i_conv_vote_threshold),
        .i_classifier_vote_increment(ft_i_classifier_vote_increment),
        .i_classifier_vote_threshold(ft_i_classifier_vote_threshold),
        .i_code_min(ft_i_code_min),
        .i_code_max(ft_i_code_max),
        .i_forward_start_valid(ft_i_forward_start_valid),
        .o_forward_start_ready(ft_o_forward_start_ready),
        .o_forward_start_fire(ft_o_forward_start_fire),
        .i_forward_done(ft_i_forward_done),
        .o_forward_active(ft_o_forward_active),
        .o_forward_model_version(ft_o_forward_model_version),
        .o_forward_adapt_version(ft_o_forward_adapt_version),
        .i_restore_complete(ft_i_restore_complete),
        .o_restore_ready(ft_o_restore_ready),
        .o_recovery_required(ft_o_recovery_required),
        .i_conv_in_width(ft_i_conv_in_width),
        .i_conv_in_height(ft_i_conv_in_height),
        .i_conv_in_channels(ft_i_conv_in_channels),
        .i_conv_out_width(ft_i_conv_out_width),
        .i_conv_out_height(ft_i_conv_out_height),
        .i_conv_out_channels(ft_i_conv_out_channels),
        .i_conv_stride(ft_i_conv_stride),
        .i_conv_padding(ft_i_conv_padding),
        .i_conv_pool_mode(ft_i_conv_pool_mode),
        .i_conv_work_base_addr(ft_i_conv_work_base_addr),
        .i_conv_leak(ft_i_conv_leak),
        .i_conv_threshold(ft_i_conv_threshold),
        .i_conv_reset_to_zero(ft_i_conv_reset_to_zero),
        .i_conv_trace_decay(ft_i_conv_trace_decay),
        .i_conv_layer_tag(ft_i_conv_layer_tag),
        .i_target_feature_count(ft_i_target_feature_count),
        .i_target_trace_leak_q8_8(ft_i_target_trace_leak_q8_8),
        .i_target_trace_state_reset(ft_i_target_trace_state_reset),
        .i_classifier_feature_channel_base(ft_i_classifier_feature_channel_base),
        .i_classifier_feature_channel_count(ft_i_classifier_feature_channel_count),
        .i_classifier_work_base_addr(ft_i_classifier_work_base_addr),
        .i_classifier_trace_decay(ft_i_classifier_trace_decay),
        .i_classifier_layer_tag(ft_i_classifier_layer_tag),
        .i_frame_index(ft_i_frame_index),
        .i_graph_descriptor_valid(ft_i_graph_descriptor_valid),
        .o_graph_descriptor_ready(ft_o_graph_descriptor_ready),
        .i_graph_descriptor_last(ft_i_graph_descriptor_last),
        .o_graph_descriptor_done(ft_o_graph_descriptor_done),
        .o_graph_frame_active(ft_o_graph_frame_active),
        .o_graph_frame_target(ft_o_graph_frame_target),
        .o_graph_frame_sample(ft_o_graph_frame_sample),
        .o_graph_frame_timestep(ft_o_graph_frame_timestep),
        .o_graph_frame_version(ft_o_graph_frame_version),
        .o_state_service_valid(ft_o_state_service_valid),
        .i_state_service_ready(ft_i_state_service_ready),
        .o_state_service_release(ft_o_state_service_release),
        .o_state_service_is_classifier(ft_o_state_service_is_classifier),
        .o_state_service_output_channel_base(ft_o_state_service_output_channel_base),
        .o_state_service_output_channel_count(ft_o_state_service_output_channel_count),
        .o_state_service_spatial_base(ft_o_state_service_spatial_base),
        .o_state_service_spatial_count(ft_o_state_service_spatial_count),
        .o_state_service_active_pair(ft_o_state_service_active_pair),
        .o_state_service_work_base_addr(ft_o_state_service_work_base_addr),
        .o_feature_request_valid(ft_o_feature_request_valid),
        .i_feature_request_ready(ft_i_feature_request_ready),
        .o_feature_term_valid(ft_o_feature_term_valid),
        .o_feature_input_channel_flat(ft_o_feature_input_channel_flat),
        .o_feature_kernel_row_flat(ft_o_feature_kernel_row_flat),
        .o_feature_kernel_col_flat(ft_o_feature_kernel_col_flat),
        .o_feature_spatial_base(ft_o_feature_spatial_base),
        .o_feature_spatial_count(ft_o_feature_spatial_count),
        .o_feature_output_row(ft_o_feature_output_row),
        .o_feature_output_col(ft_o_feature_output_col),
        .o_feature_value_bit(ft_o_feature_value_bit),
        .i_feature_response_valid(ft_i_feature_response_valid),
        .o_feature_response_ready(ft_o_feature_response_ready),
        .i_feature_mask0(ft_i_feature_mask0),
        .i_feature_mask1(ft_i_feature_mask1),
        .i_feature_mask2(ft_i_feature_mask2),
        .i_feature_mask3(ft_i_feature_mask3),
        .i_feature_service_error(ft_i_feature_service_error),
        .i_feature_write_slot(ft_i_feature_write_slot),
        .i_feature_write_base_addr(ft_i_feature_write_base_addr),
        .i_feature_write_row_stride_words(ft_i_feature_write_row_stride_words),
        .i_feature_write_channel_stride_words(ft_i_feature_write_channel_stride_words),
        .o_feature_update_valid(ft_o_feature_update_valid),
        .i_feature_update_ready(ft_i_feature_update_ready),
        .o_feature_update_slot(ft_o_feature_update_slot),
        .o_feature_update_addr(ft_o_feature_update_addr),
        .o_feature_update_data(ft_o_feature_update_data),
        .o_feature_update_mask(ft_o_feature_update_mask),
        .i_feature_update_done(ft_i_feature_update_done),
        .i_maintenance_tile(ft_i_maintenance_tile),
        .i_state_maintenance_write_valid(ft_i_state_maintenance_write_valid),
        .o_state_maintenance_write_ready(ft_o_state_maintenance_write_ready),
        .i_state_maintenance_write_addr(ft_i_state_maintenance_write_addr),
        .i_state_maintenance_write_data(ft_i_state_maintenance_write_data),
        .i_state_maintenance_write_mask(ft_i_state_maintenance_write_mask),
        .i_state_maintenance_read_valid(ft_i_state_maintenance_read_valid),
        .o_state_maintenance_read_ready(ft_o_state_maintenance_read_ready),
        .i_state_maintenance_read_addr(ft_i_state_maintenance_read_addr),
        .o_state_maintenance_result_valid(ft_o_state_maintenance_result_valid),
        .i_state_maintenance_result_ready(ft_i_state_maintenance_result_ready),
        .o_state_maintenance_result_addr(ft_o_state_maintenance_result_addr),
        .o_state_maintenance_read_data(ft_o_state_maintenance_read_data),
        .i_trace_maintenance_write_valid(ft_i_trace_maintenance_write_valid),
        .o_trace_maintenance_write_ready(ft_o_trace_maintenance_write_ready),
        .i_trace_maintenance_write_addr(ft_i_trace_maintenance_write_addr),
        .i_trace_maintenance_write_data(ft_i_trace_maintenance_write_data),
        .i_trace_maintenance_write_mask(ft_i_trace_maintenance_write_mask),
        .i_trace_maintenance_read_valid(ft_i_trace_maintenance_read_valid),
        .o_trace_maintenance_read_ready(ft_o_trace_maintenance_read_ready),
        .i_trace_maintenance_read_addr(ft_i_trace_maintenance_read_addr),
        .o_trace_maintenance_result_valid(ft_o_trace_maintenance_result_valid),
        .i_trace_maintenance_result_ready(ft_i_trace_maintenance_result_ready),
        .o_trace_maintenance_result_addr(ft_o_trace_maintenance_result_addr),
        .o_trace_maintenance_read_data(ft_o_trace_maintenance_read_data),
        .o_classifier_cache_valid(ft_o_classifier_cache_valid),
        .o_active_classifier_pair(ft_o_active_classifier_pair),
        .o_parameter_version(ft_o_parameter_version),
        .o_conv_cache_valid(ft_o_conv_cache_valid),
        .o_conv_cached_output_base(ft_o_conv_cached_output_base),
        .o_conv_cached_output_count(ft_o_conv_cached_output_count),
        .o_conv_cached_version(ft_o_conv_cached_version),
        .o_conv_cache_invalidate(ft_o_conv_cache_invalidate),
        .o_model_version(ft_o_model_version),
        .o_adapt_version(ft_o_adapt_version),
        .o_capture_complete(ft_o_capture_complete),
        .o_learning_start_fire(ft_o_learning_start_fire),
        .o_commit_pulse(ft_o_commit_pulse),
        .o_commit_code_dirty(ft_o_commit_code_dirty),
        .o_commit_optimizer_dirty(ft_o_commit_optimizer_dirty),
        .o_detach_pulse(ft_o_detach_pulse),
        .o_commit_pending(ft_o_commit_pending),
        .o_inference_hold(ft_o_inference_hold),
        .o_initialized(ft_o_initialized),
        .o_versions_aligned(ft_o_versions_aligned),
        .o_array_active_owner(ft_o_array_active_owner),
        .o_array_tile_idle(ft_o_array_tile_idle),
        .o_array_tile_sticky_error(ft_o_array_tile_sticky_error),
        .o_context_active(ft_o_context_active),
        .o_context_is_classifier(ft_o_context_is_classifier),
        .o_busy(ft_o_busy),
        .o_quiescent(ft_o_quiescent),
        .o_sticky_error(ft_o_sticky_error),
        .i_clk(ft_i_clk),
        .i_rst_n(ft_i_rst_n),
        .i_clear_error(ft_i_clear_error),
        .i_srnn_phase_abort(ft_i_srnn_phase_abort),
        .i_srnn_phase_capture_cfg_valid(ft_i_srnn_phase_capture_cfg_valid),
        .o_srnn_phase_capture_cfg_ready(ft_o_srnn_phase_capture_cfg_ready),
        .i_srnn_phase_record_count(ft_i_srnn_phase_record_count),
        .i_srnn_phase_start_valid(ft_i_srnn_phase_start_valid),
        .o_srnn_phase_start_ready(ft_o_srnn_phase_start_ready),
        .i_srnn_phase_learning_enable(ft_i_srnn_phase_learning_enable),
        .i_srnn_phase_target_valid(ft_i_srnn_phase_target_valid),
        .o_srnn_phase_target_ready(ft_o_srnn_phase_target_ready),
        .i_srnn_phase_target_sample_id(ft_i_srnn_phase_target_sample_id),
        .i_srnn_phase_target_backbone_id(ft_i_srnn_phase_target_backbone_id),
        .i_srnn_phase_target_q14(ft_i_srnn_phase_target_q14),
        .i_srnn_phase_target_last(ft_i_srnn_phase_target_last),
        .i_srnn_phase_target_done(ft_i_srnn_phase_target_done),
        .i_srnn_phase_target_error(ft_i_srnn_phase_target_error),
        .o_srnn_phase_capture_done(ft_o_srnn_phase_capture_done),
        .o_srnn_phase_capture_complete(ft_o_srnn_phase_capture_complete),
        .o_srnn_phase_delivery_done(ft_o_srnn_phase_delivery_done),
        .o_srnn_phase_busy(ft_o_srnn_phase_busy),
        .o_srnn_phase_idle(ft_o_srnn_phase_idle),
        .o_srnn_phase_sticky_error(ft_o_srnn_phase_sticky_error),
        .i_parameter_owner(ft_i_parameter_owner),
        .i_topology_learning_enable(ft_i_topology_learning_enable),
        .i_learning_fc_cfg_valid(ft_i_learning_fc_cfg_valid),
        .o_learning_fc_cfg_ready(ft_o_learning_fc_cfg_ready),
        .i_learning_fc_profile_id(ft_i_learning_fc_profile_id),
        .i_learning_fc_w2_tensor_id(ft_i_learning_fc_w2_tensor_id),
        .i_learning_fc_w2_parameter_base(ft_i_learning_fc_w2_parameter_base),
        .i_learning_fc_batch_count(ft_i_learning_fc_batch_count),
        .i_learning_fc_hidden_count(ft_i_learning_fc_hidden_count),
        .i_learning_fc_output_count(ft_i_learning_fc_output_count),
        .i_learning_fc_vote_increment(ft_i_learning_fc_vote_increment),
        .i_learning_fc_vote_threshold(ft_i_learning_fc_vote_threshold),
        .i_learning_fc_code_min(ft_i_learning_fc_code_min),
        .i_learning_fc_code_max(ft_i_learning_fc_code_max),
        .o_learning_fc_cfg_error(ft_o_learning_fc_cfg_error),
        .i_learning_fc_release(ft_i_learning_fc_release),
        .o_learning_fc_release_ready(ft_o_learning_fc_release_ready),
        .o_learning_fc_window_commit_valid(ft_o_learning_fc_window_commit_valid),
        .i_learning_fc_window_commit_ready(ft_i_learning_fc_window_commit_ready),
        .o_learning_fc_window_commit_sequence(ft_o_learning_fc_window_commit_sequence),
        .o_learning_fc_window_busy(ft_o_learning_fc_window_busy),
        .o_learning_fc_error(ft_o_learning_fc_error),
        .i_learning_fc_parameter_load_valid(ft_i_learning_fc_parameter_load_valid),
        .o_learning_fc_parameter_load_ready(ft_o_learning_fc_parameter_load_ready),
        .i_learning_fc_parameter_load_addr(ft_i_learning_fc_parameter_load_addr),
        .i_learning_fc_parameter_load_code(ft_i_learning_fc_parameter_load_code),
        .i_learning_fc_parameter_load_residual(ft_i_learning_fc_parameter_load_residual),
        .i_learning_srnn_cfg_valid(ft_i_learning_srnn_cfg_valid),
        .o_learning_srnn_cfg_ready(ft_o_learning_srnn_cfg_ready),
        .i_learning_srnn_profile_id(ft_i_learning_srnn_profile_id),
        .i_learning_srnn_wh1_tensor_id(ft_i_learning_srnn_wh1_tensor_id),
        .i_learning_srnn_wh1_base(ft_i_learning_srnn_wh1_base),
        .i_learning_srnn_wh1_increment(ft_i_learning_srnn_wh1_increment),
        .i_learning_srnn_wh1_threshold(ft_i_learning_srnn_wh1_threshold),
        .i_learning_srnn_wout_tensor_id(ft_i_learning_srnn_wout_tensor_id),
        .i_learning_srnn_wout_base(ft_i_learning_srnn_wout_base),
        .i_learning_srnn_wout_increment(ft_i_learning_srnn_wout_increment),
        .i_learning_srnn_wout_threshold(ft_i_learning_srnn_wout_threshold),
        .i_learning_srnn_code_min(ft_i_learning_srnn_code_min),
        .i_learning_srnn_code_max(ft_i_learning_srnn_code_max),
        .i_learning_srnn_batch_count(ft_i_learning_srnn_batch_count),
        .i_learning_srnn_hidden_count(ft_i_learning_srnn_hidden_count),
        .i_learning_srnn_backbone_count(ft_i_learning_srnn_backbone_count),
        .i_learning_srnn_class_count(ft_i_learning_srnn_class_count),
        .i_learning_srnn_wout_element_count(ft_i_learning_srnn_wout_element_count),
        .o_learning_srnn_cfg_error(ft_o_learning_srnn_cfg_error),
        .i_learning_srnn_parameter_load_valid(ft_i_learning_srnn_parameter_load_valid),
        .o_learning_srnn_parameter_load_ready(ft_o_learning_srnn_parameter_load_ready),
        .i_learning_srnn_parameter_load_addr(ft_i_learning_srnn_parameter_load_addr),
        .i_learning_srnn_parameter_load_code(ft_i_learning_srnn_parameter_load_code),
        .i_learning_srnn_parameter_load_residual(ft_i_learning_srnn_parameter_load_residual),
        .i_learning_srnn_timestep_id(ft_i_learning_srnn_timestep_id),
        .i_learning_srnn_sequence_version(ft_i_learning_srnn_sequence_version),
        .i_learning_srnn_abort_timestep(ft_i_learning_srnn_abort_timestep),
        .o_learning_srnn_abort_ready(ft_o_learning_srnn_abort_ready),
        .o_learning_srnn_target_trace_start(ft_o_learning_srnn_target_trace_start),
        .i_learning_srnn_target_trace_start_ready(ft_i_learning_srnn_target_trace_start_ready),
        .o_learning_srnn_target_trace_timestep_id(ft_o_learning_srnn_target_trace_timestep_id),
        .i_learning_srnn_target_trace_valid(ft_i_learning_srnn_target_trace_valid),
        .o_learning_srnn_target_trace_ready(ft_o_learning_srnn_target_trace_ready),
        .i_learning_srnn_target_trace_sample_id(ft_i_learning_srnn_target_trace_sample_id),
        .i_learning_srnn_target_trace_backbone_id(ft_i_learning_srnn_target_trace_backbone_id),
        .i_learning_srnn_target_trace_q14(ft_i_learning_srnn_target_trace_q14),
        .i_learning_srnn_target_trace_last(ft_i_learning_srnn_target_trace_last),
        .i_learning_srnn_target_trace_done(ft_i_learning_srnn_target_trace_done),
        .i_learning_srnn_target_trace_error(ft_i_learning_srnn_target_trace_error),
        .o_learning_srnn_label_replay_start(ft_o_learning_srnn_label_replay_start),
        .i_learning_srnn_label_replay_start_ready(ft_i_learning_srnn_label_replay_start_ready),
        .i_learning_srnn_label_valid(ft_i_learning_srnn_label_valid),
        .o_learning_srnn_label_ready(ft_o_learning_srnn_label_ready),
        .i_learning_srnn_label_sample_id(ft_i_learning_srnn_label_sample_id),
        .i_learning_srnn_label_class_id(ft_i_learning_srnn_label_class_id),
        .i_learning_srnn_label_last(ft_i_learning_srnn_label_last),
        .i_learning_srnn_restore_complete(ft_i_learning_srnn_restore_complete),
        .o_learning_srnn_restore_ready(ft_o_learning_srnn_restore_ready),
        .o_learning_srnn_recovery_required(ft_o_learning_srnn_recovery_required),
        .o_learning_srnn_learning_commit(ft_o_learning_srnn_learning_commit),
        .o_learning_srnn_detach(ft_o_learning_srnn_detach),
        .o_learning_srnn_logit_valid(ft_o_learning_srnn_logit_valid),
        .i_learning_srnn_logit_ready(ft_i_learning_srnn_logit_ready),
        .o_learning_srnn_logit_sample_id(ft_o_learning_srnn_logit_sample_id),
        .o_learning_srnn_logit_class_id(ft_o_learning_srnn_logit_class_id),
        .o_learning_srnn_logit_q28(ft_o_learning_srnn_logit_q28),
        .o_learning_srnn_logit_last(ft_o_learning_srnn_logit_last),
        .o_learning_srnn_timestep_done(ft_o_learning_srnn_timestep_done),
        .o_learning_srnn_timestep_aborted(ft_o_learning_srnn_timestep_aborted),
        .o_learning_srnn_busy(ft_o_learning_srnn_busy),
        .o_learning_srnn_error(ft_o_learning_srnn_error),
        .o_learning_srnn_zero_gradient_count(ft_o_learning_srnn_zero_gradient_count),
        .o_learning_srnn_update_count(ft_o_learning_srnn_update_count),
        .o_store_load_valid(ft_o_store_load_valid),
        .i_store_load_ready(ft_i_store_load_ready),
        .o_store_load_addr(ft_o_store_load_addr),
        .o_store_load_code(ft_o_store_load_code),
        .o_store_load_residual(ft_o_store_load_residual),
        .o_store_learning_read_valid(ft_o_store_learning_read_valid),
        .i_store_learning_read_ready(ft_i_store_learning_read_ready),
        .o_store_learning_read_addr(ft_o_store_learning_read_addr),
        .o_store_learning_read_tag(ft_o_store_learning_read_tag),
        .i_store_learning_response_valid(ft_i_store_learning_response_valid),
        .o_store_learning_response_ready(ft_o_store_learning_response_ready),
        .i_store_learning_response_tag(ft_i_store_learning_response_tag),
        .i_store_learning_response_code(ft_i_store_learning_response_code),
        .i_store_learning_response_residual(ft_i_store_learning_response_residual),
        .i_store_learning_response_error(ft_i_store_learning_response_error),
        .o_store_learning_write_valid(ft_o_store_learning_write_valid),
        .i_store_learning_write_ready(ft_i_store_learning_write_ready),
        .o_store_learning_write_addr(ft_o_store_learning_write_addr),
        .o_store_learning_write_code(ft_o_store_learning_write_code),
        .o_store_learning_write_residual(ft_o_store_learning_write_residual),
        .o_store_learning_write_code_enable(ft_o_store_learning_write_code_enable),
        .o_store_learning_write_residual_enable(ft_o_store_learning_write_residual_enable),
        .o_store_forward_read_valid(ft_o_store_forward_read_valid),
        .i_store_forward_read_ready(ft_i_store_forward_read_ready),
        .o_store_forward_read_addr(ft_o_store_forward_read_addr),
        .o_store_forward_read_tag(ft_o_store_forward_read_tag),
        .i_store_forward_response_valid(ft_i_store_forward_response_valid),
        .o_store_forward_response_ready(ft_o_store_forward_response_ready),
        .i_store_forward_response_tag(ft_i_store_forward_response_tag),
        .i_store_forward_response_code(ft_i_store_forward_response_code),
        .i_store_forward_response_error(ft_i_store_forward_response_error),
        .i_store_quiescent(ft_i_store_quiescent),
        .i_store_protocol_error(ft_i_store_protocol_error),
        .o_backend_quiescent(ft_o_backend_quiescent),
        .o_backend_error(ft_o_backend_error),
        .o_owner_encoding_error(ft_o_owner_encoding_error)
    );
    assign feature_request_valid = ft_o_feature_request_valid;
    assign feature_term_valid = ft_o_feature_term_valid;
    assign feature_input_channel_flat = ft_o_feature_input_channel_flat;
    assign feature_kernel_row_flat = ft_o_feature_kernel_row_flat;
    assign feature_kernel_col_flat = ft_o_feature_kernel_col_flat;
    assign feature_spatial_base = ft_o_feature_spatial_base;
    assign feature_spatial_count = ft_o_feature_spatial_count;
    assign feature_output_row = ft_o_feature_output_row;
    assign feature_output_col = ft_o_feature_output_col;
    assign feature_value_bit = ft_o_feature_value_bit;
    assign feature_response_ready = ft_o_feature_response_ready;
    assign feature_update_valid = ft_o_feature_update_valid;
    assign feature_update_slot = ft_o_feature_update_slot;
    assign feature_update_addr = ft_o_feature_update_addr;
    assign feature_update_data = ft_o_feature_update_data;
    assign feature_update_mask = ft_o_feature_update_mask;
    assign state_write_ready = ft_o_state_maintenance_write_ready;
    assign state_read_ready = ft_o_state_maintenance_read_ready;
    assign state_result_valid = ft_o_state_maintenance_result_valid;
    assign state_result_addr = ft_o_state_maintenance_result_addr;
    assign state_read_data = ft_o_state_maintenance_read_data;
    assign trace_write_ready = ft_o_trace_maintenance_write_ready;
    assign trace_read_ready = ft_o_trace_maintenance_read_ready;
    assign trace_result_valid = ft_o_trace_maintenance_result_valid;
    assign trace_result_addr = ft_o_trace_maintenance_result_addr;
    assign trace_read_data = ft_o_trace_maintenance_read_data;
    assign classifier_cache_valid = ft_o_classifier_cache_valid;
    assign busy = dut.u_core.u_system.u_array.scnn_client_busy;
    assign sticky_error = ft_o_sticky_error;
    assign start_ready = ft_o_graph_descriptor_ready;
    assign done = ft_o_graph_descriptor_done;
    assign idle = dut.u_core.u_system.u_array.scnn_client_idle;
    assign active_owner = ft_o_array_active_owner;
    assign tile_idle = ft_o_array_tile_idle;
    assign tile_error = ft_o_array_tile_sticky_error;
    assign classifier_pair = ft_o_active_classifier_pair;
    assign classifier_version = ft_o_parameter_version;
    assign cache_prime_done = ft_o_initialize_done;
    assign context_load_valid = ft_o_state_service_valid && !ft_o_state_service_release;
    assign context_release_valid = ft_o_state_service_valid && ft_o_state_service_release;
    assign context_load_weights = 1'b0;
    assign logit_wave_fire = dut.u_core.u_system.logit_wave_valid && dut.u_core.u_system.logit_wave_ready;
    assign logit_wave_mask = dut.u_core.u_system.logit_wave_mask;
    assign logit_wave_base = dut.u_core.u_system.logit_wave_base;
    assign logit_wave_count = dut.u_core.u_system.logit_wave_count;
    assign logit_wave_flat = dut.u_core.u_system.logit_wave_flat;
    assign context_output_channel_base = ft_o_state_service_output_channel_base;
    assign context_output_channel_count = ft_o_state_service_output_channel_count;
    assign context_spatial_base = ft_o_state_service_spatial_base;
    assign context_spatial_count = ft_o_state_service_spatial_count;
    assign context_active_pair = ft_o_state_service_active_pair;
    assign context_work_base_addr = ft_o_state_service_work_base_addr;
    assign release_output_channel_base = ft_o_state_service_output_channel_base;
    assign release_output_channel_count = ft_o_state_service_output_channel_count;
    assign release_spatial_base = ft_o_state_service_spatial_base;
    assign release_spatial_count = ft_o_state_service_spatial_count;
    assign release_active_pair = ft_o_state_service_active_pair;
    assign release_work_base_addr = ft_o_state_service_work_base_addr;
    assign weight_fill_valid = dut.u_core.u_system.scnn_weight_fill_valid;
    assign weight_fill_ready = dut.u_core.u_system.scnn_weight_fill_ready;
    assign weight_fill_tile_flat = dut.u_core.u_system.scnn_weight_fill_tile_flat;
    assign weight_fill_bank = dut.u_core.u_system.scnn_weight_fill_bank;
    assign weight_fill_addr_flat = dut.u_core.u_system.scnn_weight_fill_addr_flat;
    assign weight_fill_data_flat = dut.u_core.u_system.scnn_weight_fill_data_flat;












    function automatic int current_layer_spatial_area;
        begin
            case (layer_phase)
                1: current_layer_spatial_area = CONV1_SPATIAL*CONV1_SPATIAL;
                2: current_layer_spatial_area = CONV2_SPATIAL*CONV2_SPATIAL;
                3: current_layer_spatial_area = CONV3_SPATIAL*CONV3_SPATIAL;
                4: current_layer_spatial_area = CONV4_SPATIAL*CONV4_SPATIAL;
                5: current_layer_spatial_area = CONV5_SPATIAL*CONV5_SPATIAL;
                6: current_layer_spatial_area = CONV6_SPATIAL*CONV6_SPATIAL;
                7: current_layer_spatial_area = CONV7_SPATIAL*CONV7_SPATIAL;
                default:
                    current_layer_spatial_area = CONV8_SPATIAL*CONV8_SPATIAL;
            endcase
        end
    endfunction

    function automatic logic signed [15:0] saved_state(input int index);
        begin
            case (layer_phase)
                1: saved_state = conv1_state_backing[tv_sample*CONV1_SCALARS+index];
                2: saved_state = conv2_state_backing[tv_sample*CONV2_SCALARS+index];
                3: saved_state = conv3_state_backing[tv_sample*CONV3_SCALARS+index];
                4: saved_state = conv4_state_backing[tv_sample*CONV4_SCALARS+index];
                5: saved_state = conv5_state_backing[tv_sample*CONV5_SCALARS+index];
                6: saved_state = conv6_state_backing[tv_sample*CONV6_SCALARS+index];
                7: saved_state = conv7_state_backing[tv_sample*CONV7_SCALARS+index];
                default: saved_state = conv8_state_backing[tv_sample*CONV8_SCALARS+index];
            endcase
        end
    endfunction

    function automatic logic signed [15:0] saved_trace(input int index);
        begin
            case (layer_phase)
                1: saved_trace = conv1_trace_backing[tv_sample*CONV1_SCALARS+index];
                2: saved_trace = conv2_trace_backing[tv_sample*CONV2_SCALARS+index];
                3: saved_trace = conv3_trace_backing[tv_sample*CONV3_SCALARS+index];
                4: saved_trace = conv4_trace_backing[tv_sample*CONV4_SCALARS+index];
                5: saved_trace = conv5_trace_backing[tv_sample*CONV5_SCALARS+index];
                6: saved_trace = conv6_trace_backing[tv_sample*CONV6_SCALARS+index];
                7: saved_trace = conv7_trace_backing[tv_sample*CONV7_SCALARS+index];
                default: saved_trace = conv8_trace_backing[tv_sample*CONV8_SCALARS+index];
            endcase
        end
    endfunction

    function automatic logic signed [15:0] expected_next_state(input int index);
        begin
            case (layer_phase)
                1: expected_next_state = conv1_observed_next[index];
                2: expected_next_state = conv2_observed_next[index];
                3: expected_next_state = conv3_observed_next[index];
                4: expected_next_state = conv4_observed_next[index];
                5: expected_next_state = conv5_observed_next[index];
                6: expected_next_state = conv6_observed_next[index];
                7: expected_next_state = conv7_observed_next[index];
                default: expected_next_state = conv8_observed_next[index];
            endcase
        end
    endfunction

    task automatic save_state(input int index, input logic signed [15:0] value);
        begin
            case (layer_phase)
                1: conv1_state_backing[tv_sample*CONV1_SCALARS+index] = value;
                2: conv2_state_backing[tv_sample*CONV2_SCALARS+index] = value;
                3: conv3_state_backing[tv_sample*CONV3_SCALARS+index] = value;
                4: conv4_state_backing[tv_sample*CONV4_SCALARS+index] = value;
                5: conv5_state_backing[tv_sample*CONV5_SCALARS+index] = value;
                6: conv6_state_backing[tv_sample*CONV6_SCALARS+index] = value;
                7: conv7_state_backing[tv_sample*CONV7_SCALARS+index] = value;
                default: conv8_state_backing[tv_sample*CONV8_SCALARS+index] = value;
            endcase
        end
    endtask

    task automatic save_trace(input int index, input logic signed [15:0] value);
        begin
            case (layer_phase)
                1: conv1_trace_backing[tv_sample*CONV1_SCALARS+index] = value;
                2: conv2_trace_backing[tv_sample*CONV2_SCALARS+index] = value;
                3: conv3_trace_backing[tv_sample*CONV3_SCALARS+index] = value;
                4: conv4_trace_backing[tv_sample*CONV4_SCALARS+index] = value;
                5: conv5_trace_backing[tv_sample*CONV5_SCALARS+index] = value;
                6: conv6_trace_backing[tv_sample*CONV6_SCALARS+index] = value;
                7: conv7_trace_backing[tv_sample*CONV7_SCALARS+index] = value;
                default: conv8_trace_backing[tv_sample*CONV8_SCALARS+index] = value;
            endcase
        end
    endtask



    task automatic service_context;
        integer expected_count;
        begin
            if (context_work_base_addr == CLASSIFIER_WORK_BASE) begin
                if (layer_phase != 8 || context_output_channel_base != 0 ||
                        context_output_channel_count != CLASSES ||
                        context_spatial_base != 0 ||
                        context_spatial_count != 1 ||
                        context_active_pair != 1'b1 ||
                        context_load_weights)
                    $fatal(1,
                        "classifier context mismatch phase=%0d base=%0d count=%0d spatial=%0d/%0d pair=%0b load=%0b",
                        layer_phase, context_output_channel_base,
                        context_output_channel_count, context_spatial_base,
                        context_spatial_count, context_active_pair,
                        context_load_weights);
                if (!ft_o_state_service_is_classifier) $fatal(1, "classifier service kind");
                fg_classifier_restore();
                classifier_context_count = classifier_context_count + 1;
            end else begin
            if (ft_o_state_service_is_classifier) $fatal(1, "Conv service kind");
            if (context_work_base_addr != 0)
                $fatal(1, "unexpected work base %0d", context_work_base_addr);
            if (layer_phase == 1) begin
                expected_count = 64 - integer'(context_output_channel_base);
                if (expected_count > 12)
                    expected_count = 12;
                if (context_output_channel_count != expected_count ||
                        context_spatial_count != CONV1_SPATIAL ||
                        context_spatial_base >=
                            CONV1_SPATIAL*CONV1_SPATIAL ||
                        context_spatial_base % CONV1_SPATIAL != 0 ||
                        context_active_pair != 1'b0)
                    $fatal(1,
                        "Conv1 context mismatch base=%0d count=%0d spatial=%0d/%0d pair=%0b",
                        context_output_channel_base,
                        context_output_channel_count, context_spatial_base,
                        context_spatial_count, context_active_pair);
                conv1_context_count = conv1_context_count + 1;
            end else if (layer_phase == 2) begin
                expected_count = CONV2_CHANNELS -
                    integer'(context_output_channel_base);
                if (expected_count > 12)
                    expected_count = 12;
                if (context_output_channel_count != expected_count ||
                        context_spatial_base >=
                            CONV2_SPATIAL*CONV2_SPATIAL ||
                        context_spatial_base % CONV2_SPATIAL != 0 ||
                        context_spatial_count != CONV2_SPATIAL ||
                        context_active_pair != 1'b0)
                    $fatal(1,
                        "Conv2 context mismatch base=%0d count=%0d spatial=%0d/%0d pair=%0b load=%0b",
                        context_output_channel_base,
                        context_output_channel_count, context_spatial_base,
                        context_spatial_count, context_active_pair,
                        context_load_weights);
                conv2_context_count = conv2_context_count + 1;
            end else if (layer_phase == 3) begin
                expected_count = CONV3_CHANNELS -
                    integer'(context_output_channel_base);
                if (expected_count > 12)
                    expected_count = 12;
                if (context_output_channel_count != expected_count ||
                        context_spatial_base >=
                            CONV3_SPATIAL*CONV3_SPATIAL ||
                        context_spatial_base % CONV3_SPATIAL != 0 ||
                        context_spatial_count != CONV3_SPATIAL ||
                        context_active_pair != 1'b0)
                    $fatal(1,
                        "Conv3 context mismatch base=%0d count=%0d spatial=%0d/%0d pair=%0b load=%0b",
                        context_output_channel_base,
                        context_output_channel_count, context_spatial_base,
                        context_spatial_count, context_active_pair,
                        context_load_weights);
                conv3_context_count = conv3_context_count + 1;
            end else if (layer_phase == 4) begin
                expected_count = CONV4_CHANNELS -
                    integer'(context_output_channel_base);
                if (expected_count > 12)
                    expected_count = 12;
                if (context_output_channel_count != expected_count ||
                        context_spatial_base >=
                            CONV4_SPATIAL*CONV4_SPATIAL ||
                        context_spatial_base % CONV4_SPATIAL != 0 ||
                        context_spatial_count != CONV4_SPATIAL ||
                        context_active_pair != 1'b0)
                    $fatal(1,
                        "Conv4 context mismatch base=%0d count=%0d spatial=%0d/%0d pair=%0b load=%0b",
                        context_output_channel_base,
                        context_output_channel_count, context_spatial_base,
                        context_spatial_count, context_active_pair,
                        context_load_weights);
                conv4_context_count = conv4_context_count + 1;
            end else if (layer_phase == 5) begin
                expected_count = CONV5_CHANNELS -
                    integer'(context_output_channel_base);
                if (expected_count > 12)
                    expected_count = 12;
                if (context_output_channel_count != expected_count ||
                        context_spatial_base >=
                            CONV5_SPATIAL*CONV5_SPATIAL ||
                        context_spatial_base % CONV5_SPATIAL != 0 ||
                        context_spatial_count != CONV5_SPATIAL ||
                        context_active_pair != 1'b0)
                    $fatal(1,
                        "Conv5 context mismatch base=%0d count=%0d spatial=%0d/%0d pair=%0b load=%0b",
                        context_output_channel_base,
                        context_output_channel_count, context_spatial_base,
                        context_spatial_count, context_active_pair,
                        context_load_weights);
                conv5_context_count = conv5_context_count + 1;
            end else if (layer_phase == 6) begin
                expected_count = CONV6_CHANNELS -
                    integer'(context_output_channel_base);
                if (expected_count > 12)
                    expected_count = 12;
                if (context_output_channel_count != expected_count ||
                        context_spatial_base >=
                            CONV6_SPATIAL*CONV6_SPATIAL ||
                        context_spatial_base % CONV6_SPATIAL != 0 ||
                        context_spatial_count != CONV6_SPATIAL ||
                        context_active_pair != 1'b0)
                    $fatal(1,
                        "Conv6 context mismatch base=%0d count=%0d spatial=%0d/%0d pair=%0b load=%0b",
                        context_output_channel_base,
                        context_output_channel_count, context_spatial_base,
                        context_spatial_count, context_active_pair,
                        context_load_weights);
                conv6_context_count = conv6_context_count + 1;
            end else if (layer_phase == 7) begin
                expected_count = CONV7_CHANNELS -
                    integer'(context_output_channel_base);
                if (expected_count > 12)
                    expected_count = 12;
                if (context_output_channel_count != expected_count ||
                        context_spatial_base >=
                            CONV7_SPATIAL*CONV7_SPATIAL ||
                        context_spatial_base % CONV7_SPATIAL != 0 ||
                        context_spatial_count != CONV7_SPATIAL ||
                        context_active_pair != 1'b0)
                    $fatal(1,
                        "Conv7 context mismatch base=%0d count=%0d spatial=%0d/%0d pair=%0b load=%0b",
                        context_output_channel_base,
                        context_output_channel_count, context_spatial_base,
                        context_spatial_count, context_active_pair,
                        context_load_weights);
                conv7_context_count = conv7_context_count + 1;
            end else begin
                expected_count = CONV8_CHANNELS -
                    integer'(context_output_channel_base);
                if (expected_count > 12)
                    expected_count = 12;
                if (context_output_channel_count != expected_count ||
                        context_spatial_base >=
                            CONV8_SPATIAL*CONV8_SPATIAL ||
                        context_spatial_base % CONV8_SPATIAL != 0 ||
                        context_spatial_count != CONV8_SPATIAL ||
                        context_active_pair != 1'b0)
                    $fatal(1,
                        "Conv8 context mismatch base=%0d count=%0d spatial=%0d/%0d pair=%0b load=%0b",
                        context_output_channel_base,
                        context_output_channel_count, context_spatial_base,
                        context_spatial_count, context_active_pair,
                        context_load_weights);
                conv8_context_count = conv8_context_count + 1;
            end
            for (int tile = 0;
                    tile < integer'(context_output_channel_count); tile++)
                restore_tile_context(tile);
            end
            @(negedge clk);
            context_load_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            context_load_ready = 1'b0;
        end
    endtask



    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cache_prime_count <= 0;
            classifier_logit_wave_count <= 0;
        end else begin
            if (cache_prime_done)
                cache_prime_count <= cache_prime_count + 1;
            if (logit_wave_fire) begin
                if (layer_phase != 8 || logit_wave_mask != 12'h7ff ||
                        logit_wave_base != 0 ||
                        logit_wave_count != CLASSES)
                    $fatal(1,
                        "classifier logit envelope mismatch phase=%0d mask=%h base=%0d count=%0d",
                        layer_phase, logit_wave_mask, logit_wave_base,
                        logit_wave_count);
                for (int class_id = 0; class_id < CLASSES; class_id++) begin
                    if ($signed(logit_wave_flat[class_id*16 +: 16]) !==
                            classifier_expected_centered[class_id])
                        $fatal(1,
                            "classifier cumulative logit mismatch timestep=%0d class=%0d got=%0d expected=%0d",
                            timestep_phase,
                            class_id,
                            $signed(logit_wave_flat[class_id*16 +: 16]),
                            $signed(classifier_expected_centered[class_id]));
                end
                classifier_logit_wave_count <=
                    classifier_logit_wave_count + 1;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            feature_response_valid <= 1'b0;
            feature_mask0 <= '0;
            feature_mask1 <= '0;
            feature_mask2 <= '0;
            feature_mask3 <= '0;
            feature_update_done <= 1'b0;
            cycle_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            feature_update_done <= 1'b0;
            if (feature_response_valid && feature_response_ready)
                feature_response_valid <= 1'b0;
            if (feature_request_valid && feature_request_ready) begin
                if ((layer_phase == 1 && feature_value_bit >= VALUE_BITS) ||
                        (layer_phase != 1 && feature_value_bit != 0))
                    $fatal(1, "illegal feature bit phase=%0d bit=%0d",
                        layer_phase, feature_value_bit);
                feature_mask0 <= make_feature_mask(0);
                feature_mask1 <= make_feature_mask(1);
                feature_mask2 <= make_feature_mask(2);
                feature_mask3 <= make_feature_mask(3);
                feature_response_valid <= 1'b1;
                if (layer_phase == 1)
                    conv1_request_count <= conv1_request_count + 1;
                else if (layer_phase == 2)
                    conv2_request_count <= conv2_request_count + 1;
                else if (layer_phase == 3)
                    conv3_request_count <= conv3_request_count + 1;
                else if (layer_phase == 4)
                    conv4_request_count <= conv4_request_count + 1;
                else if (layer_phase == 5)
                    conv5_request_count <= conv5_request_count + 1;
                else if (layer_phase == 6)
                    conv6_request_count <= conv6_request_count + 1;
                else if (layer_phase == 7)
                    conv7_request_count <= conv7_request_count + 1;
                else
                    conv8_request_count <= conv8_request_count + 1;
            end
            if (feature_update_valid && feature_update_ready) begin
                if (feature_update_addr >= FEATURE_DEPTH)
                    $fatal(1, "feature update address overflow %0d",
                        feature_update_addr);
                for (int bit_index = 0; bit_index < 64; bit_index++) begin
                    if (feature_update_mask[bit_index])
                        feature_memory[feature_update_slot][feature_update_addr]
                            [bit_index] <=
                            feature_update_data[bit_index];
                end
                feature_update_done <= 1'b1;
                if (layer_phase == 1) begin
                    if (feature_update_slot ||
                            feature_update_mask[63:32] != 0)
                        $fatal(1,
                            "Conv1 writeback envelope mismatch slot=%0b addr=%0d mask=%h",
                            feature_update_slot, feature_update_addr,
                            feature_update_mask);
                    conv1_update_count <= conv1_update_count + 1;
                end else if (layer_phase == 2) begin
                    if (!feature_update_slot ||
                            feature_update_addr < CONV2_WRITE_BASE ||
                            feature_update_addr >=
                                CONV2_WRITE_BASE + CONV2_CHANNELS *
                                    CONV2_POOLED_SPATIAL ||
                            feature_update_mask[63:CONV2_POOLED_SPATIAL] != 0)
                        $fatal(1,
                            "Conv2 writeback envelope mismatch slot=%0b addr=%0d mask=%h",
                            feature_update_slot, feature_update_addr,
                            feature_update_mask);
                    conv2_update_count <= conv2_update_count + 1;
                end else if (layer_phase == 3) begin
                    if (feature_update_slot ||
                            feature_update_addr < CONV3_WRITE_BASE ||
                            feature_update_addr >= CONV3_WRITE_BASE +
                                CONV3_CHANNELS * CONV3_SPATIAL ||
                            feature_update_mask[63:CONV3_SPATIAL] != 0)
                        $fatal(1,
                            "Conv3 writeback envelope mismatch slot=%0b addr=%0d mask=%h",
                            feature_update_slot, feature_update_addr,
                            feature_update_mask);
                    conv3_update_count <= conv3_update_count + 1;
                end else if (layer_phase == 4) begin
                    if (!feature_update_slot ||
                            feature_update_addr < CONV4_WRITE_BASE ||
                            feature_update_addr >= CONV4_WRITE_BASE +
                                CONV4_CHANNELS * CONV4_POOLED_SPATIAL ||
                            feature_update_mask[63:CONV4_POOLED_SPATIAL] != 0)
                        $fatal(1,
                            "Conv4 writeback envelope mismatch slot=%0b addr=%0d mask=%h",
                            feature_update_slot, feature_update_addr,
                            feature_update_mask);
                    conv4_update_count <= conv4_update_count + 1;
                end else if (layer_phase == 5) begin
                    if (feature_update_slot ||
                            feature_update_addr < CONV5_WRITE_BASE ||
                            feature_update_addr >= CONV5_WRITE_BASE +
                                CONV5_CHANNELS * CONV5_SPATIAL ||
                            feature_update_mask[63:CONV5_SPATIAL] != 0)
                        $fatal(1,
                            "Conv5 writeback envelope mismatch slot=%0b addr=%0d mask=%h",
                            feature_update_slot, feature_update_addr,
                            feature_update_mask);
                    conv5_update_count <= conv5_update_count + 1;
                end else if (layer_phase == 6) begin
                    if (!feature_update_slot ||
                            feature_update_addr < CONV6_WRITE_BASE ||
                            feature_update_addr >= CONV6_WRITE_BASE +
                                CONV6_CHANNELS * CONV6_POOLED_SPATIAL ||
                            feature_update_mask[63:CONV6_POOLED_SPATIAL] != 0)
                        $fatal(1,
                            "Conv6 writeback envelope mismatch slot=%0b addr=%0d mask=%h",
                            feature_update_slot, feature_update_addr,
                            feature_update_mask);
                    conv6_update_count <= conv6_update_count + 1;
                end else if (layer_phase == 7) begin
                    if (feature_update_slot ||
                            feature_update_addr < CONV7_WRITE_BASE ||
                            feature_update_addr >= CONV7_WRITE_BASE +
                                CONV7_CHANNELS * CONV7_SPATIAL ||
                            feature_update_mask[63:CONV7_SPATIAL] != 0)
                        $fatal(1,
                            "Conv7 writeback envelope mismatch slot=%0b addr=%0d mask=%h",
                            feature_update_slot, feature_update_addr,
                            feature_update_mask);
                    conv7_update_count <= conv7_update_count + 1;
                end else begin
                    if (!feature_update_slot ||
                            feature_update_addr < CONV8_WRITE_BASE ||
                            feature_update_addr >= CONV8_WRITE_BASE +
                                CONV8_CHANNELS * CONV8_POOLED_SPATIAL ||
                            feature_update_mask[63:CONV8_POOLED_SPATIAL] != 0)
                        $fatal(1,
                            "Conv8 writeback envelope mismatch slot=%0b addr=%0d mask=%h",
                            feature_update_slot, feature_update_addr,
                            feature_update_mask);
                    conv8_update_count <= conv8_update_count + 1;
                end
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst_n) begin
            if (context_release_valid && context_release_ready &&
                    release_work_base_addr == 0) begin
                if (layer_phase == 1)
                    conv1_release_count <= conv1_release_count + 1;
                else if (layer_phase == 2)
                    conv2_release_count <= conv2_release_count + 1;
                else if (layer_phase == 3)
                    conv3_release_count <= conv3_release_count + 1;
                else if (layer_phase == 4)
                    conv4_release_count <= conv4_release_count + 1;
                else if (layer_phase == 5)
                    conv5_release_count <= conv5_release_count + 1;
                else if (layer_phase == 6)
                    conv6_release_count <= conv6_release_count + 1;
                else if (layer_phase == 7)
                    conv7_release_count <= conv7_release_count + 1;
                else
                    conv8_release_count <= conv8_release_count + 1;
            end
            for (int region = 0; region < 4; region++) begin
                if (observation_valid[region] && observation_ready[region] &&
                        dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client.state_q ==
                            2'd1) begin
                    if (observation_mode_flat[region*2 +: 2] != 2'd2 ||
                            observation_range_mask_flat[region*10 +: 10] != 0)
                        if (region == 2) begin
                            $fatal(1,
                                "observation envelope mismatch phase=%0d region=%0d tile=%0d mode=%0d lane=%h range=%h meta=%h raw0=%0d raw1=%0d raw2=%0d bank0=%h bank1=%h",
                                layer_phase, region,
                                observation_tile_flat[region*4 +: 4],
                                observation_mode_flat[region*2 +: 2],
                                observation_lane_mask_flat[region*10 +: 10],
                                observation_range_mask_flat[region*10 +: 10],
                                observation_meta_flat[region*64 +: 64],
                                $signed(dut.u_core.u_system.u_array.u_array.u_array.g_region[2]
                                    .u_region.tile_region_raw_current_flat[
                                        0*480 + 0*48 +: 48]),
                                $signed(dut.u_core.u_system.u_array.u_array.u_array.g_region[2]
                                    .u_region.tile_region_raw_current_flat[
                                        0*480 + 1*48 +: 48]),
                                $signed(dut.u_core.u_system.u_array.u_array.u_array.g_region[2]
                                    .u_region.tile_region_raw_current_flat[
                                        0*480 + 2*48 +: 48]),
                                dut.u_core.u_system.u_array.u_array.u_array.g_region[2].u_region
                                    .g_tile[0].u_tile.g_p4_all_frontend
                                    .u_weight_store.g_context[0].g_enabled.g_original.u_store.g_bank[0].memory[0],
                                dut.u_core.u_system.u_array.u_array.u_array.g_region[2].u_region
                                    .g_tile[0].u_tile.g_p4_all_frontend
                                    .u_weight_store.g_context[0].g_enabled.g_original.u_store.g_bank[1].memory[0]);
                        end else begin
                            $fatal(1,
                                "observation envelope mismatch phase=%0d region=%0d tile=%0d mode=%0d lane=%h range=%h meta=%h",
                                layer_phase, region,
                                observation_tile_flat[region*4 +: 4],
                                observation_mode_flat[region*2 +: 2],
                                observation_lane_mask_flat[region*10 +: 10],
                                observation_range_mask_flat[region*10 +: 10],
                                observation_meta_flat[region*64 +: 64]);
                        end
                    if (layer_phase == 1)
                        conv1_observation_count <=
                            conv1_observation_count + 1;
                    else if (layer_phase == 2)
                        conv2_observation_count <=
                            conv2_observation_count + 1;
                    else if (layer_phase == 3)
                        conv3_observation_count <=
                            conv3_observation_count + 1;
                    else if (layer_phase == 4)
                        conv4_observation_count <=
                            conv4_observation_count + 1;
                    else if (layer_phase == 5)
                        conv5_observation_count <=
                            conv5_observation_count + 1;
                    else if (layer_phase == 6)
                        conv6_observation_count <=
                            conv6_observation_count + 1;
                    else if (layer_phase == 7)
                        conv7_observation_count <=
                            conv7_observation_count + 1;
                    else
                        conv8_observation_count <=
                            conv8_observation_count + 1;
                end
            end
            if (scalar_valid && scalar_ready) begin
                integer expected_index;
                if (scalar_layer_tag == CONV1_TAG) begin
                    expected_index = integer'(scalar_channel) *
                        (CONV1_SPATIAL * CONV1_SPATIAL) +
                        integer'(scalar_spatial);
                    tv_record_scalar(1, expected_index, scalar_pre_state, scalar_spike);
                    if (scalar_channel >= CONV1_CHANNELS ||
                            scalar_spatial >=
                                CONV1_SPATIAL*CONV1_SPATIAL ||
                            conv1_seen[timestep_phase*CONV1_SCALARS +
                                expected_index] ||
                            scalar_spike !== (scalar_pre_state >= 16'sd256))
                        $fatal(1,
                            "Conv1 parity mismatch channel=%0d spatial=%0d got=%0d/%0b expected=%0d/%0b",
                            scalar_channel, scalar_spatial, scalar_pre_state,
                            scalar_spike,
                            $signed(conv1_observed_pre[expected_index]),
                            conv1_observed_spike[expected_index]);
                    conv1_seen[timestep_phase*CONV1_SCALARS +
                        expected_index] <= 1'b1;
                    conv1_scalar_count <= conv1_scalar_count + 1;
                end else if (scalar_layer_tag == CONV2_TAG) begin
                    expected_index = integer'(scalar_channel) *
                        (CONV2_SPATIAL * CONV2_SPATIAL) +
                        integer'(scalar_spatial);
                    tv_record_scalar(2, expected_index, scalar_pre_state, scalar_spike);
                    if (scalar_channel >= CONV2_CHANNELS ||
                            scalar_spatial >=
                                CONV2_SPATIAL*CONV2_SPATIAL ||
                            conv2_seen[timestep_phase*CONV2_SCALARS +
                                expected_index] ||
                            scalar_spike !== (scalar_pre_state >= 16'sd256))
                        $fatal(1,
                            "Conv2 parity mismatch channel=%0d got=%0d/%0b expected=%0d/%0b",
                            scalar_channel, scalar_pre_state, scalar_spike,
                            $signed(conv2_observed_pre[expected_index]),
                            conv2_observed_spike[expected_index]);
                    conv2_seen[timestep_phase*CONV2_SCALARS +
                        expected_index] <= 1'b1;
                    conv2_scalar_count <= conv2_scalar_count + 1;
                end else if (scalar_layer_tag == CONV3_TAG) begin
                    expected_index = integer'(scalar_channel) *
                        (CONV3_SPATIAL * CONV3_SPATIAL) +
                        integer'(scalar_spatial);
                    tv_record_scalar(3, expected_index, scalar_pre_state, scalar_spike);
                    if (scalar_channel >= CONV3_CHANNELS ||
                            scalar_spatial >=
                                CONV3_SPATIAL*CONV3_SPATIAL ||
                            conv3_seen[timestep_phase*CONV3_SCALARS +
                                expected_index] ||
                            scalar_spike !== (scalar_pre_state >= 16'sd256))
                        $fatal(1,
                            "Conv3 parity mismatch channel=%0d got=%0d/%0b expected=%0d/%0b",
                            scalar_channel, scalar_pre_state, scalar_spike,
                            $signed(conv3_observed_pre[expected_index]),
                            conv3_observed_spike[expected_index]);
                    conv3_seen[timestep_phase*CONV3_SCALARS +
                        expected_index] <= 1'b1;
                    conv3_scalar_count <= conv3_scalar_count + 1;
                end else if (scalar_layer_tag == CONV4_TAG) begin
                    expected_index = integer'(scalar_channel) *
                        (CONV4_SPATIAL * CONV4_SPATIAL) +
                        integer'(scalar_spatial);
                    tv_record_scalar(4, expected_index, scalar_pre_state, scalar_spike);
                    if (scalar_channel >= CONV4_CHANNELS ||
                            scalar_spatial >=
                                CONV4_SPATIAL*CONV4_SPATIAL ||
                            conv4_seen[timestep_phase*CONV4_SCALARS +
                                expected_index] ||
                            scalar_spike !== (scalar_pre_state >= 16'sd256))
                        $fatal(1,
                            "Conv4 parity mismatch channel=%0d spatial=%0d got=%0d/%0b expected=%0d/%0b",
                            scalar_channel, scalar_spatial, scalar_pre_state,
                            scalar_spike,
                            $signed(conv4_observed_pre[expected_index]),
                            conv4_observed_spike[expected_index]);
                    conv4_seen[timestep_phase*CONV4_SCALARS +
                        expected_index] <= 1'b1;
                    conv4_scalar_count <= conv4_scalar_count + 1;
                end else if (scalar_layer_tag == CONV5_TAG) begin
                    expected_index = integer'(scalar_channel) *
                        (CONV5_SPATIAL * CONV5_SPATIAL) +
                        integer'(scalar_spatial);
                    tv_record_scalar(5, expected_index, scalar_pre_state, scalar_spike);
                    if (scalar_channel >= CONV5_CHANNELS ||
                            scalar_spatial >=
                                CONV5_SPATIAL*CONV5_SPATIAL ||
                            conv5_seen[timestep_phase*CONV5_SCALARS +
                                expected_index] ||
                            scalar_spike !== (scalar_pre_state >= 16'sd256))
                        $fatal(1,
                            "Conv5 parity mismatch channel=%0d spatial=%0d got=%0d/%0b expected=%0d/%0b",
                            scalar_channel, scalar_spatial, scalar_pre_state,
                            scalar_spike,
                            $signed(conv5_observed_pre[expected_index]),
                            conv5_observed_spike[expected_index]);
                    conv5_seen[timestep_phase*CONV5_SCALARS +
                        expected_index] <= 1'b1;
                    conv5_scalar_count <= conv5_scalar_count + 1;
                end else if (scalar_layer_tag == CONV6_TAG) begin
                    expected_index = integer'(scalar_channel) *
                        (CONV6_SPATIAL * CONV6_SPATIAL) +
                        integer'(scalar_spatial);
                    tv_record_scalar(6, expected_index, scalar_pre_state, scalar_spike);
                    if (scalar_channel >= CONV6_CHANNELS ||
                            scalar_spatial >=
                                CONV6_SPATIAL*CONV6_SPATIAL ||
                            conv6_seen[timestep_phase*CONV6_SCALARS +
                                expected_index] ||
                            scalar_spike !== (scalar_pre_state >= 16'sd256))
                        $fatal(1,
                            "Conv6 parity mismatch channel=%0d spatial=%0d got=%0d/%0b expected=%0d/%0b",
                            scalar_channel, scalar_spatial, scalar_pre_state,
                            scalar_spike,
                            $signed(conv6_observed_pre[expected_index]),
                            conv6_observed_spike[expected_index]);
                    conv6_seen[timestep_phase*CONV6_SCALARS +
                        expected_index] <= 1'b1;
                    conv6_scalar_count <= conv6_scalar_count + 1;
                end else if (scalar_layer_tag == CONV7_TAG) begin
                    expected_index = integer'(scalar_channel) *
                        (CONV7_SPATIAL * CONV7_SPATIAL) +
                        integer'(scalar_spatial);
                    tv_record_scalar(7, expected_index, scalar_pre_state, scalar_spike);
                    if (scalar_channel >= CONV7_CHANNELS ||
                            scalar_spatial >=
                                CONV7_SPATIAL*CONV7_SPATIAL ||
                            conv7_seen[timestep_phase*CONV7_SCALARS +
                                expected_index] ||
                            scalar_spike !== (scalar_pre_state >= 16'sd256))
                        $fatal(1,
                            "Conv7 parity mismatch channel=%0d spatial=%0d got=%0d/%0b expected=%0d/%0b",
                            scalar_channel, scalar_spatial, scalar_pre_state,
                            scalar_spike,
                            $signed(conv7_observed_pre[expected_index]),
                            conv7_observed_spike[expected_index]);
                    conv7_seen[timestep_phase*CONV7_SCALARS +
                        expected_index] <= 1'b1;
                    conv7_scalar_count <= conv7_scalar_count + 1;
                end else if (scalar_layer_tag == CONV8_TAG) begin
                    expected_index = integer'(scalar_channel) *
                        (CONV8_SPATIAL * CONV8_SPATIAL) +
                        integer'(scalar_spatial);
                    tv_record_scalar(8, expected_index, scalar_pre_state, scalar_spike);
                    if (scalar_channel >= CONV8_CHANNELS ||
                            scalar_spatial >=
                                CONV8_SPATIAL*CONV8_SPATIAL ||
                            conv8_seen[timestep_phase*CONV8_SCALARS +
                                expected_index] ||
                            scalar_spike !== (scalar_pre_state >= 16'sd256))
                        $fatal(1,
                            "Conv8 parity mismatch channel=%0d spatial=%0d got=%0d/%0b expected=%0d/%0b",
                            scalar_channel, scalar_spatial, scalar_pre_state,
                            scalar_spike,
                            $signed(conv8_observed_pre[expected_index]),
                            conv8_observed_spike[expected_index]);
                    conv8_seen[timestep_phase*CONV8_SCALARS +
                        expected_index] <= 1'b1;
                    conv8_scalar_count <= conv8_scalar_count + 1;
                end else begin
                    $fatal(1, "unexpected scalar tag %h", scalar_layer_tag);
                end
            end
            if (layer_phase == 8 &&
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                        .conv_feature_commit_valid &&
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                        .conv_feature_commit_ready) begin
                integer commit_index;
                if (dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                            .conv_feature_commit_layer_tag != CONV8_TAG ||
                        dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                            .conv_feature_commit_channel >=
                            CONV8_CHANNELS ||
                        dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                            .conv_feature_commit_spatial_count !=
                            CONV8_POOLED_SPATIAL ||
                        dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                            .conv_feature_commit_spatial_base >=
                            CONV8_POOLED_SPATIAL*CONV8_POOLED_SPATIAL ||
                        dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                            .conv_feature_commit_spatial_base %
                            CONV8_POOLED_SPATIAL != 0)
                    $fatal(1,
                        "Conv8 feature-commit envelope mismatch tag=%h channel=%0d base=%0d count=%0d",
                        dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                            .conv_feature_commit_layer_tag,
                        dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                            .conv_feature_commit_channel,
                        dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                            .conv_feature_commit_spatial_base,
                        dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                            .conv_feature_commit_spatial_count);
                conv8_commit_count <= conv8_commit_count + 1;
                conv8_commit_value_count <= conv8_commit_value_count +
                    CONV8_POOLED_SPATIAL;
                for (int lane = 0; lane < CONV8_POOLED_SPATIAL; lane++) begin
                    commit_index =
                        integer'(dut.u_core.u_system.u_array.g_scnn_graph
                            .u_scnn_graph_client.conv_feature_commit_channel) *
                            CONV8_POOLED_SPATIAL*CONV8_POOLED_SPATIAL +
                        integer'(dut.u_core.u_system.u_array.g_scnn_graph
                            .u_scnn_graph_client
                            .conv_feature_commit_spatial_base) + lane;
                    if (conv8_commit_seen[timestep_phase*CLASSIFIER_FEATURES +
                            commit_index] ||
                            dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                                .conv_feature_commit_spike[lane] !==
                                conv8_expected_pooled_spike[commit_index] ||
                            dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                                .conv_feature_commit_q8_8_flat[
                                lane*16 +: 16] !==
                                conv8_expected_pooled_q8_8[commit_index])
                        $fatal(1,
                            "Conv8 feature-commit parity mismatch index=%0d got=%0b/%0d expected=%0b/%0d",
                            commit_index,
                            dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                                .conv_feature_commit_spike[lane],
                            dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                                .conv_feature_commit_q8_8_flat[
                                lane*16 +: 16],
                            conv8_expected_pooled_spike[commit_index],
                            conv8_expected_pooled_q8_8[commit_index]);
                    conv8_commit_seen[timestep_phase*CLASSIFIER_FEATURES +
                        commit_index] <= 1'b1;
                end
            end
            if (sticky_error || serializer_error || tile_error != 0) begin
                $display("ERROR_TOP timestep=%0d layer=%0d path=%0b serializer=%0b tile=%h busy=%0b idle=%0b",
                    timestep_phase, layer_phase, sticky_error,
                    serializer_error, tile_error, busy, idle);
                $display("FINAL_ERROR backend=%b owner=%b hold=%b", ft_o_backend_error, ft_o_owner_encoding_error, ft_o_inference_hold);
                $display("ERROR_CORE owner=%0b array=%0b scnn=%0b fcobs=%0b fcclient=%0b owner_q=%0d select_scnn=%0b select_maintenance=%0b array_idle=%0b",
                    dut.u_core.u_system.u_array.owner_error, dut.u_core.u_system.u_array.array_error,
                    dut.u_core.u_system.u_array.scnn_client_error,
                    dut.u_core.u_system.u_array.fc_srnn_observation_error,
                    dut.u_core.u_system.u_array.fc_srnn_client_error,
                    dut.u_core.u_system.u_array.u_owner.owner_q, dut.u_core.u_system.u_array.select_scnn,
                    dut.u_core.u_system.u_array.select_maintenance, dut.u_core.u_system.u_array.array_idle);
                $display("ERROR_GRAPH state=%0d local=%0b conv=%0b serializer=%0b fork=%0b classifier=%0b feature=%0b conv_busy=%0b conv_idle=%0b",
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client.state_q,
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client.local_error_q,
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client.conv_error,
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                        .serializer_error,
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client.fork_error,
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                        .classifier_error,
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                        .i_feature_service_error,
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client.conv_busy,
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client.conv_idle);
                $display("ERROR_CONV_CLIENT scheduler=%0b writeback=%0b state=%0d config_legal=%0b writeback_state=%0d array_error=%0b",
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client.u_conv_client
                        .scheduler_error,
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client.u_conv_client
                        .writeback_error,
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client.u_conv_client
                        .u_scheduler.state_q,
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client.u_conv_client
                        .u_scheduler.config_legal,
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client.u_conv_client
                        .u_observation_writeback.state_q,
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client.u_conv_client
                        .i_array_error);
                $display("ERROR_CLASSIFIER local=%0b feature=%0b scheduler=%0b serializer=%0b logit=%0b state=%0d",
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                        .u_classifier_tail.local_error_q,
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                        .u_classifier_tail.feature_error,
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                        .u_classifier_tail.scheduler_error,
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                        .u_classifier_tail.serializer_error,
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                        .u_classifier_tail.logit_error,
                    dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client
                        .u_classifier_tail.state_q);
                $display("ERROR_ARRAY path=%0b walker=%0b active=%0b open=%0b last=%0b command=%0b/%0b context_legal=%0b",
                    dut.u_core.u_system.u_array.u_array.path_sticky_error_q,
                    dut.u_core.u_system.u_array.u_array.walker_sticky_error,
                    dut.u_core.u_system.u_array.u_array.active_q,
                    dut.u_core.u_system.u_array.u_array.transaction_open_q,
                    dut.u_core.u_system.u_array.u_array.command_last_q,
                    dut.u_core.u_system.u_array.u_array.i_command_valid,
                    dut.u_core.u_system.u_array.u_array.o_command_ready,
                    dut.u_core.u_system.u_array.u_array.command_context_legal);
                $fatal(1, "three-layer path error");
            end
        end
    end


    initial begin
        $readmemh(
            "models/scnn_dvsgesture_vgg9_q0_15/bank_01.hex",
            conv1_weights);
        $readmemh(
            "models/scnn_dvsgesture_vgg9_q0_15/bank_02.hex",
            conv2_weights);
        $readmemh(
            "models/scnn_dvsgesture_vgg9_q0_15/bank_03.hex",
            conv3_weights);
        $readmemh(
            "models/scnn_dvsgesture_vgg9_q0_15/bank_04.hex",
            conv4_weights);
        $readmemh(
            "models/scnn_dvsgesture_vgg9_q0_15/bank_05.hex",
            conv5_weights);
        $readmemh(
            "models/scnn_dvsgesture_vgg9_q0_15/bank_06.hex",
            conv6_weights);
        $readmemh(
            "models/scnn_dvsgesture_vgg9_q0_15/bank_07.hex",
            conv7_weights);
        $readmemh(
            "models/scnn_dvsgesture_vgg9_q0_15/bank_08.hex",
            conv8_weights);
        $readmemh(
            "models/scnn_dvsgesture_vgg9_q0_15/bank_12.hex",
            classifier_weights);

        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        progress_stage = 1;
        fg_load_expected();
        fg_initialize_top();
        progress_stage = 2;

        progress_stage = 3;

        progress_stage = 4;
        fg_start_window();
        $display("SCNN_TAIL_INITIALIZATION cycles=%0d weight_fill_words=%0d state_write_words=%0d trace_write_words=%0d",
            cycle_count, perf_weight_fills, perf_state_writes, perf_trace_writes);

        for (timestep_phase = 0; timestep_phase < 52;
                timestep_phase = timestep_phase + 1) begin
        perf_frame_start = cycle_count;
        if (timestep_phase != 0 && timestep_phase % 16 == 0) fg_start_window();
        tv_load_frame();
        for (int c = 0; c < CLASSES; c++) begin
            classifier_expected_current[c] = fg_currents[timestep_phase*CLASSES+c];
            classifier_expected_centered[c] = fg_logits[timestep_phase*CLASSES+c];
        end
        for (int slot = 0; slot < 2; slot++)
            for (int address = 0; address < FEATURE_DEPTH; address++)
                feature_memory[slot][address] = '0;

        for (int channel = 0; channel < 512; channel++)
            for (int row = 0; row < 4; row++)
                for (int col = 0; col < 4; col++)
                    feature_memory[0][CONV7_WRITE_BASE+channel*4+row][col] =
                        tail_input[timestep_phase*8192+channel*16+row*4+col];
        progress_stage = 5+timestep_phase;

        cfg_in_width = CONV7_SPATIAL;
        cfg_in_height = CONV7_SPATIAL;
        cfg_in_channels = CONV7_CHANNELS;
        cfg_out_width = CONV8_SPATIAL;
        cfg_out_height = CONV8_SPATIAL;
        cfg_out_channels = CONV8_CHANNELS;
        cfg_pool_mode = 2'd2;
        cfg_padding = 8'd1;
        cfg_initial_active_pair = 1'b0;
        cfg_layer_tag = CONV8_TAG;
        cfg_feature_write_slot = 1'b1;
        cfg_feature_write_base = FEATURE_AW'(CONV8_WRITE_BASE);
        cfg_feature_write_row_stride = 1;
        cfg_feature_write_channel_stride = CONV8_POOLED_SPATIAL;
        cfg_class_count = 4'(CLASSES);
        layer_phase = 8;
        run_current_layer();

        for (int channel = 0; channel < CONV8_CHANNELS; channel++) begin
            for (int row = 0; row < CONV8_POOLED_SPATIAL; row++) begin
                for (int col = 0; col < CONV8_POOLED_SPATIAL; col++) begin
                    integer expected_index;
                    expected_index = channel * CONV8_POOLED_SPATIAL *
                        CONV8_POOLED_SPATIAL +
                        row * CONV8_POOLED_SPATIAL + col;
                    if (feature_memory[1][CONV8_WRITE_BASE +
                            channel*CONV8_POOLED_SPATIAL + row][col] !==
                            conv8_expected_pooled_spike[expected_index])
                        $fatal(1,
                            "Conv8 pooled writeback mismatch channel=%0d row=%0d col=%0d got=%0b expected=%0b",
                            channel, row, col,
                            feature_memory[1][CONV8_WRITE_BASE +
                                channel*CONV8_POOLED_SPATIAL + row][col],
                            conv8_expected_pooled_spike[expected_index]);
                end
            end
        end

        if (conv8_context_count != (timestep_phase+1)*CONV8_CONTEXTS ||
                conv8_release_count != (timestep_phase+1)*CONV8_CONTEXTS ||
                conv8_request_count != (timestep_phase+1)*CONV8_REQUESTS ||
                conv8_update_count != (timestep_phase+1)*CONV8_CHANNELS*
                    CONV8_POOLED_SPATIAL ||
                conv8_observation_count != (timestep_phase+1)*
                    CONV8_OBSERVATIONS ||
                conv8_scalar_count != (timestep_phase+1)*CONV8_SCALARS ||
                conv8_commit_count != (timestep_phase+1)*CONV8_CHANNELS*
                    CONV8_POOLED_SPATIAL ||
                conv8_commit_value_count != (timestep_phase+1)*
                    CONV8_CHANNELS*CONV8_POOLED_SPATIAL*
                    CONV8_POOLED_SPATIAL ||
                classifier_context_count != timestep_phase+1 ||
                classifier_logit_wave_count != timestep_phase+1 ||
                !classifier_cache_valid || !classifier_pair ||
                (classifier_version != fg_window_version && !(tv_window_end && !tv_query && classifier_version == fg_window_version+1)) ||
                busy || !idle || serializer_busy)
            $fatal(1,
                "Conv8/classifier incomplete context=%0d classifier_context=%0d release=%0d requests=%0d updates=%0d observations=%0d scalars=%0d commits=%0d values=%0d logits=%0d cache=%0b/%0b/%h owner=%0d busy=%0b idle=%0b serializer=%0b",
                conv8_context_count, classifier_context_count,
                conv8_release_count,
                conv8_request_count, conv8_update_count,
                conv8_observation_count, conv8_scalar_count,
                conv8_commit_count, conv8_commit_value_count,
                classifier_logit_wave_count, classifier_cache_valid,
                classifier_pair, classifier_version, active_owner,
                busy, idle, serializer_busy);

        fg_check_frame();
        $display("SCNN_TAIL_TIMESTEP timestep=%0d cycles=%0d classifier_logits=%0d",
            timestep_phase, cycle_count - perf_frame_start, classifier_logit_wave_count);
        $display("SCNN_TEMPORAL_FRAME_PASS frame=%0d conv_membrane_probes_exact=1 full_transfer_consistency=1 state_spill_restore=1 pooled_q8_8_values=%0d classifier_logits=%0d classifier_contexts=%0d one_shared_array=1 requests=%0d+%0d+%0d+%0d+%0d+%0d+%0d+%0d contexts=%0d+%0d+%0d+%0d+%0d+%0d+%0d+%0d scalars=%0d+%0d+%0d+%0d+%0d+%0d+%0d+%0d",
            timestep_phase,
            conv8_commit_value_count,
            classifier_logit_wave_count, classifier_context_count,
            conv1_request_count, conv2_request_count, conv3_request_count,
            conv4_request_count, conv5_request_count, conv6_request_count,
            conv7_request_count, conv8_request_count, conv1_context_count,
            conv2_context_count, conv3_context_count, conv4_context_count,
            conv5_context_count, conv6_context_count, conv7_context_count,
            conv8_context_count, conv1_scalar_count, conv2_scalar_count,
            conv3_scalar_count, conv4_scalar_count, conv5_scalar_count,
            conv6_scalar_count, conv7_scalar_count, conv8_scalar_count);
        end
        fg_finish();
        st_report();
        $display("FINAL_SCNN_WRAPPER_PASS raw_active=0");
        $finish;
    end

    initial begin
        repeat (3) repeat (1000000000) @(posedge clk);
        $fatal(1,
            "Conv8-to-classifier timeout progress=%0d phase=%0d cycles=%0d requests=%0d/%0d/%0d/%0d/%0d/%0d/%0d/%0d scalars=%0d/%0d/%0d/%0d/%0d/%0d/%0d/%0d commits=%0d/%0d",
            progress_stage, layer_phase, cycle_count, conv1_request_count,
            conv2_request_count, conv3_request_count, conv4_request_count,
            conv5_request_count, conv6_request_count, conv7_request_count,
            conv8_request_count,
            conv1_scalar_count, conv2_scalar_count, conv3_scalar_count,
            conv4_scalar_count, conv5_scalar_count, conv6_scalar_count,
            conv7_scalar_count, conv8_scalar_count, conv8_commit_count,
            conv8_commit_value_count);
    end
    localparam int FG_CLASSIFIER_BASE = 9217152;
    localparam int FG_CONV1_BASE = 0;
    localparam int FG_CONV2_BASE = 1152;
    localparam int FG_CONV3_BASE = 74880;
    localparam int FG_CONV4_BASE = 369792;
    localparam int FG_CONV5_BASE = 959616;
    localparam int FG_CONV6_BASE = 2139264;
    localparam int FG_CONV7_BASE = 4498560;
    localparam int FG_CONV8_BASE = 6857856;
    function automatic logic [PARAM_ADDR_WIDTH-1:0] fg_conv_base(input int layer);
        case (layer)
            1: return PARAM_ADDR_WIDTH'(FG_CONV1_BASE);
            2: return PARAM_ADDR_WIDTH'(FG_CONV2_BASE);
            3: return PARAM_ADDR_WIDTH'(FG_CONV3_BASE);
            4: return PARAM_ADDR_WIDTH'(FG_CONV4_BASE);
            5: return PARAM_ADDR_WIDTH'(FG_CONV5_BASE);
            6: return PARAM_ADDR_WIDTH'(FG_CONV6_BASE);
            7: return PARAM_ADDR_WIDTH'(FG_CONV7_BASE);
            8: return PARAM_ADDR_WIDTH'(FG_CONV8_BASE);
            default: return '0;
        endcase
    endfunction
    function automatic int fg_bank(input int address);
        if (address < 0 || address >= FG_CLASSIFIER_BASE + PARAMS)
            return -1;
        if (address < FG_CONV2_BASE) return 0;
        if (address < FG_CONV3_BASE) return 1;
        if (address < FG_CONV4_BASE) return 2;
        if (address < FG_CONV5_BASE) return 3;
        if (address < FG_CONV6_BASE) return 4;
        if (address < FG_CONV7_BASE) return 5;
        if (address < FG_CONV8_BASE) return 6;
        if (address < FG_CLASSIFIER_BASE) return 7;
        return 8;
    endfunction
    function automatic logic signed [15:0] fg_ddr_code(input int address);
        case (fg_bank(address))
            0: return conv1_weights[address-FG_CONV1_BASE];
            1: return conv2_weights[address-FG_CONV2_BASE];
            2: return conv3_weights[address-FG_CONV3_BASE];
            3: return conv4_weights[address-FG_CONV4_BASE];
            4: return conv5_weights[address-FG_CONV5_BASE];
            5: return conv6_weights[address-FG_CONV6_BASE];
            6: return conv7_weights[address-FG_CONV7_BASE];
            7: return conv8_weights[address-FG_CONV8_BASE];
            8: return classifier_weights[address-FG_CLASSIFIER_BASE];
            default: begin $fatal(1, "unmapped DDR read address=%0d", address); return '0; end
        endcase
    endfunction
    task automatic fg_initialize_top;
        @(negedge clk); fg_initialize = 1;
        do @(posedge clk); while (!ft_o_initialize_ready);
        @(negedge clk); fg_initialize = 0;
        while (!ft_o_initialize_done) @(negedge clk);
        @(negedge clk);
        if (!classifier_cache_valid || classifier_pair != 1 ||
            classifier_version != CLASSIFIER_VERSION || !ft_o_versions_aligned)
            $fatal(1, "final-top initialization/cache publication failed");
    endtask
    task automatic fg_write_row(input int tile, input int base, input int row,
                                input logic [319:0] state_data, input logic [319:0] trace_data);
        @(negedge clk);
        maintenance_tile = 4'(tile);
        state_write_addr = MEMORY_AW'(base+row);
        state_write_data = state_data; state_write_mask = '1; state_write_valid = 1;
        do @(posedge clk); while (!state_write_ready);
        @(negedge clk); state_write_valid = 0;
        trace_write_addr = MEMORY_AW'(MEMORY_DEPTH/2+base+row);
        trace_write_data = trace_data; trace_write_mask = '1; trace_write_valid = 1;
        do @(posedge clk); while (!trace_write_ready);
        @(negedge clk); trace_write_valid = 0;
    endtask
    task automatic fg_read_row(input int tile, input int base, input int row,
                               output logic [319:0] state_data, output logic [319:0] trace_data);
        @(negedge clk);
        maintenance_tile = 4'(tile);
        state_read_addr = MEMORY_AW'(base+row); state_read_valid = 1;
        do @(posedge clk); while (!state_read_ready);
        @(negedge clk); state_read_valid = 0;
        while (!state_result_valid) @(negedge clk);
        if (state_result_addr != MEMORY_AW'(base+row)) $fatal(1, "state read address mismatch");
        state_data = state_read_data;
        state_result_ready = 1;
        @(posedge clk); @(negedge clk); state_result_ready = 0;
        trace_read_addr = MEMORY_AW'(MEMORY_DEPTH/2+base+row); trace_read_valid = 1;
        do @(posedge clk); while (!trace_read_ready);
        @(negedge clk); trace_read_valid = 0;
        while (!trace_result_valid) @(negedge clk);
        if (trace_result_addr != MEMORY_AW'(MEMORY_DEPTH/2+base+row)) $fatal(1, "trace read address mismatch");
        trace_data = trace_read_data;
        trace_result_ready = 1;
        @(posedge clk); @(negedge clk); trace_result_ready = 0;
    endtask
    task automatic tv_record_scalar(input int layer, input int index,
        input logic signed [15:0] pre, input logic spike);
        int area, size, channels, post, probe_index;
        case (layer)
            1: begin size = CONV1_SPATIAL; channels = CONV1_CHANNELS; end
            2: begin size = CONV2_SPATIAL; channels = CONV2_CHANNELS; end
            3: begin size = CONV3_SPATIAL; channels = CONV3_CHANNELS; end
            4: begin size = CONV4_SPATIAL; channels = CONV4_CHANNELS; end
            5: begin size = CONV5_SPATIAL; channels = CONV5_CHANNELS; end
            6: begin size = CONV6_SPATIAL; channels = CONV6_CHANNELS; end
            7: begin size = CONV7_SPATIAL; channels = CONV7_CHANNELS; end
            8: begin size = CONV8_SPATIAL; channels = CONV8_CHANNELS; end
            default: $fatal(1, "invalid scalar layer");
        endcase
        area = size*size;
        if (index < 0 || index >= channels*area) $fatal(1, "invalid scalar index");
        if (spike != (pre >= 16'sd256)) $fatal(1, "scalar threshold mismatch");
        post = int'(pre) - (spike ? 256 : 0);
        if (layer != 8 || pre !== tail_pre[timestep_phase*8192+index] ||
            16'(post) !== tail_post[timestep_phase*8192+index] ||
            spike !== tail_spike[timestep_phase*8192+index])
            $fatal(1, "independent full Conv8 scalar mismatch frame=%0d index=%0d",timestep_phase,index);
        tail_scalars++;
        case (layer)
            1: begin conv1_observed_pre[index] = pre; conv1_observed_spike[index] = spike; conv1_observed_next[index] = 16'(post); end
            2: begin conv2_observed_pre[index] = pre; conv2_observed_spike[index] = spike; conv2_observed_next[index] = 16'(post); end
            3: begin conv3_observed_pre[index] = pre; conv3_observed_spike[index] = spike; conv3_observed_next[index] = 16'(post); end
            4: begin conv4_observed_pre[index] = pre; conv4_observed_spike[index] = spike; conv4_observed_next[index] = 16'(post); end
            5: begin conv5_observed_pre[index] = pre; conv5_observed_spike[index] = spike; conv5_observed_next[index] = 16'(post); end
            6: begin conv6_observed_pre[index] = pre; conv6_observed_spike[index] = spike; conv6_observed_next[index] = 16'(post); end
            7: begin conv7_observed_pre[index] = pre; conv7_observed_spike[index] = spike; conv7_observed_next[index] = 16'(post); end
            8: begin conv8_observed_pre[index] = pre; conv8_observed_spike[index] = spike; conv8_observed_next[index] = 16'(post); end
        endcase
        for (int probe = 0; probe < 3; probe++) begin
            case (probe)
                0: probe_index = 0;
                1: probe_index = (channels/2)*area+(size/2)*size+size/2;
                2: probe_index = channels*area-1;
            endcase
            if (index == probe_index) begin
                if (tv_probe_seen[timestep_phase*24+(layer-1)*3+probe] ||
                    post != int'(tv_membrane_probes[timestep_phase*24+(layer-1)*3+probe]))
                    $fatal(1, "independent membrane probe frame=%0d layer=%0d probe=%0d got=%0d expected=%0d",
                        timestep_phase, layer, probe, post, tv_membrane_probes[timestep_phase*24+(layer-1)*3+probe]);
                tv_probe_seen[timestep_phase*24+(layer-1)*3+probe] = 1; tv_probes_checked++;
            end
        end
    endtask
    function automatic logic tv_pooled_occupancy(input int layer, input int index);
        int size, pooled, channel, row, col, source_index; logic value;
        case (layer)
            2: size = CONV2_SPATIAL;
            4: size = CONV4_SPATIAL;
            6: size = CONV6_SPATIAL;
            default: size = CONV8_SPATIAL;
        endcase
        pooled = size/2; channel = index/(pooled*pooled); row = (index/pooled)%pooled; col = index%pooled;
        value = 0;
        for (int y = 0; y < 2; y++) for (int x = 0; x < 2; x++) begin
            source_index = channel*size*size+(row*2+y)*size+col*2+x;
            case (layer)
                2: value |= conv2_observed_spike[source_index];
                4: value |= conv4_observed_spike[source_index];
                6: value |= conv6_observed_spike[source_index];
                8: value |= conv8_observed_spike[source_index];
            endcase
        end
        return value;
    endfunction
    // Explicit Conv7 bits and original banks drive public services.
    // Expected state/trace arrays are read-only; actual spills feed restores.
    wire tv_query = timestep_phase >= 48;
    wire [31:0] tv_window = tv_query ? 3 : 32'(timestep_phase/16);
    wire [31:0] tv_sample = tv_query ? 0 : 32'((timestep_phase%16)/4);
    wire [31:0] tv_time = tv_query ? 32'(timestep_phase-48) : tv_window*4+32'(timestep_phase%4);
    wire tv_window_end = tv_query ? timestep_phase == 51 : timestep_phase%16 == 15;
    logic fg_initialize = 0, fg_request = 0;
    wire fg_learning = !tv_query;
    logic [15:0] fg_window_version = CLASSIFIER_VERSION;
    logic [15:0] fg_label [0:3];
    logic [15:0] tv_raw [0:52*2048-1];
    logic signed [15:0] fg_features [0:52*CLASSIFIER_FEATURES-1];
    logic signed [15:0] fg_currents [0:52*CLASSES-1];
    logic signed [15:0] fg_logits [0:52*CLASSES-1];
    logic signed [31:0] tv_cumulative [0:48*CLASSES-1];
    logic signed [15:0] fg_traces [0:48*CLASSIFIER_FEATURES-1];
    logic signed [15:0] tv_membrane_probes [0:52*24-1];
    logic signed [15:0] tv_query_before_currents [0:4*CLASSES-1];
    logic signed [15:0] tv_query_before_logits [0:4*CLASSES-1];
    logic signed [15:0] fg_codes [0:3*PARAMS-1];
    logic signed [15:0] fg_residuals [0:3*PARAMS-1];
    logic signed [47:0] tv_gradients [0:3*PARAMS-1];
    logic signed [17:0] tv_deltas [0:3*PARAMS-1];
    logic [15:0] fg_rmw_mask [0:3*PARAMS-1];
    logic [15:0] tv_counts [0:11];
    logic signed [15:0] fg_initial_codes [0:PARAMS-1];
    logic signed [15:0] fg_store_residual [0:PARAMS-1];
    logic signed [15:0] tv_before_codes [0:PARAMS-1];
    logic signed [15:0] tv_before_residuals [0:PARAMS-1];
    bit fg_rmw_seen [0:PARAMS-1];
    bit tv_gradient_seen [0:PARAMS-1];
    bit tv_delta_seen [0:PARAMS-1];
    bit tv_probe_seen [0:52*24-1];
    logic signed [47:0] tv_observed_gradient [0:PARAMS-1];
    int tv_contributions [0:PARAMS-1];
    logic [319:0] fg_class_state [0:3][0:CLASSES-1][0:1];
    logic [319:0] fg_class_trace [0:3][0:CLASSES-1][0:1];
    logic fg_forward_valid = 0, fg_learning_valid = 0;
    logic [TAG_WIDTH-1:0] fg_forward_tag = 0, fg_learning_tag = 0;
    logic signed [15:0] fg_forward_code = 0, fg_learning_code = 0, fg_learning_residual = 0;
    int fg_requests = 0, fg_windows = 0, fg_descriptors = 0, fg_commits = 0, fg_detaches = 0;
    int fg_rmw = 0, fg_learning_reads = 0, tv_total_rmw = 0;
    int tv_probes_checked = 0, tv_query_current_changes = 0, tv_query_logit_changes = 0;
    int fg_classifier_restores = 0, fg_classifier_spills = 0;
    int fg_forward_reads [0:8];
    int fg_fresh_reads = 0;
    logic fg_rmw_outstanding = 0;
    logic [PARAM_ADDR_WIDTH-1:0] fg_rmw_address;

    always @(posedge clk) begin : temporal_external_ddr
        int address, index, expected_index, bank;
        if (rst_n) begin
            if (fg_forward_valid && ft_o_store_forward_response_ready) tail_forward_responses++;
            if (fg_learning_valid && ft_o_store_learning_response_ready) tail_learning_responses++;
            if (ft_o_store_learning_write_valid && ft_i_store_learning_write_ready) begin
                tail_code_writes += int'(ft_o_store_learning_write_code_enable);
                tail_residual_writes += int'(ft_o_store_learning_write_residual_enable);
            end
            if (ft_o_conv_cache_invalidate && tv_window_end && !tv_query) begin
                tail_window_invalidated[tv_window] = 1;
                tail_invalidations++;
            end
            if (ft_o_store_load_valid) $fatal(1, "unexpected parameter reload");
            if (fg_forward_valid && ft_o_store_forward_response_ready) fg_forward_valid <= 0;
            if (fg_learning_valid && ft_o_store_learning_response_ready) fg_learning_valid <= 0;
            if (ft_o_store_forward_read_valid && ft_i_store_forward_read_ready) begin
                address = int'(ft_o_store_forward_read_addr); bank = fg_bank(address);
                if (bank < 0) $fatal(1, "unmapped DDR forward address=%0d", address);
                fg_forward_code <= fg_ddr_code(address); fg_forward_tag <= ft_o_store_forward_read_tag;
                fg_forward_valid <= 1; fg_forward_reads[bank]++;
                if (tv_query) fg_fresh_reads++;
            end
            if (ft_o_store_learning_read_valid && ft_i_store_learning_read_ready) begin
                address = int'(ft_o_store_learning_read_addr); index = address-FG_CLASSIFIER_BASE;
                if (!fg_learning || !tv_window_end || fg_rmw_outstanding || index < 0 || index >= PARAMS)
                    $fatal(1, "illegal learner read frame=%0d address=%0d", timestep_phase, address);
                fg_learning_code <= fg_ddr_code(address); fg_learning_residual <= fg_store_residual[index];
                fg_learning_tag <= ft_o_store_learning_read_tag; fg_learning_valid <= 1;
                fg_rmw_outstanding = 1; fg_rmw_address = ft_o_store_learning_read_addr; fg_learning_reads++;
            end
            if (ft_o_store_learning_write_valid && ft_i_store_learning_write_ready) begin
                address = int'(ft_o_store_learning_write_addr); index = address-FG_CLASSIFIER_BASE;
                expected_index = int'(tv_window)*PARAMS+index;
                if (!fg_learning || !tv_window_end || !fg_rmw_outstanding ||
                    ft_o_store_learning_write_addr != fg_rmw_address || index < 0 || index >= PARAMS)
                    $fatal(1, "illegal learner write frame=%0d address=%0d", timestep_phase, address);
                if (fg_rmw_mask[expected_index] != 1 || fg_rmw_seen[index] ||
                    ft_o_store_learning_write_code_enable != (fg_codes[expected_index] != tv_before_codes[index]) ||
                    ft_o_store_learning_write_residual_enable != (fg_residuals[expected_index] != tv_before_residuals[index]) ||
                    ft_o_store_learning_write_code !== fg_codes[expected_index] ||
                    ft_o_store_learning_write_residual !== fg_residuals[expected_index])
                    $fatal(1, "window RMW mismatch window=%0d index=%0d code=%0d/%0d residual=%0d/%0d",
                        tv_window+1, index, ft_o_store_learning_write_code, fg_codes[expected_index],
                        ft_o_store_learning_write_residual, fg_residuals[expected_index]);
                if (ft_o_store_learning_write_code_enable) classifier_weights[index] = ft_o_store_learning_write_code;
                if (ft_o_store_learning_write_residual_enable) fg_store_residual[index] = ft_o_store_learning_write_residual;
                if (tv_window == 2 && index == 18433) begin
                    if (tv_before_codes[index] != 870 || tv_before_residuals[index] != -16155 ||
                        classifier_weights[index] != 869 || fg_store_residual[index] != -4571)
                        $fatal(1, "natural crossing witness mismatch");
                    $display("SCNN_TEMPORAL_CROSSING window=3 index=18433 gradient=2221286714 delta=-21184 code_before=870 code_after=869 residual_before=-16155 residual_after=-4571");
                    $fflush();
                end
                fg_rmw_seen[index] = 1; fg_rmw_outstanding = 0; fg_rmw++; tv_total_rmw++;
            end
        end
    end


    logic tail_input [0:52*8192-1];
    logic signed [15:0] tail_pre [0:52*8192-1];
    logic signed [15:0] tail_post [0:52*8192-1];
    logic tail_spike [0:52*8192-1];
    int tail_scalars, tail_forward_responses, tail_learning_responses;
    int tail_code_writes, tail_residual_writes, tail_invalidations;
    bit tail_window_invalidated [0:2];
    task automatic fg_load_expected;
        string fixture, control, tail_fixture;
        if (!$value$plusargs("TAIL_FIXTURE=%s", tail_fixture))
            $fatal(1, "TAIL_FIXTURE required");
        $readmemh({tail_fixture,"/conv8_input_bits.hex"}, tail_input);
        $readmemh({tail_fixture,"/conv8_pre.hex"}, tail_pre);
        $readmemh({tail_fixture,"/conv8_post.hex"}, tail_post);
        $readmemh({tail_fixture,"/conv8_spike.hex"}, tail_spike);
        if (!$value$plusargs("SCNN_TEMPORAL_FIXTURE=%s", fixture) || !$value$plusargs("SCNN_TEMPORAL_CONTROL=%s", control))
            $fatal(1, "SCNN_TEMPORAL_FIXTURE/CONTROL required");
        $readmemh({fixture,"/labels.hex"}, fg_label);
        $readmemh({fixture,"/initial_codes.hex"}, fg_initial_codes);
        $readmemh({control,"/window_counts.hex"}, tv_counts);
        $readmemh({fixture,"/train_raw.hex"}, tv_raw, 0, 48*2048-1);
        $readmemh({fixture,"/query_raw.hex"}, tv_raw, 48*2048, 52*2048-1);
        $readmemh({fixture,"/train_features.hex"}, fg_features, 0, 48*CLASSIFIER_FEATURES-1);
        $readmemh({fixture,"/query_features.hex"}, fg_features, 48*CLASSIFIER_FEATURES, 52*CLASSIFIER_FEATURES-1);
        $readmemh({fixture,"/train_currents.hex"}, fg_currents, 0, 48*CLASSES-1);
        $readmemh({fixture,"/query_currents.hex"}, fg_currents, 48*CLASSES, 52*CLASSES-1);
        $readmemh({fixture,"/train_centered.hex"}, fg_logits, 0, 48*CLASSES-1);
        $readmemh({fixture,"/query_centered.hex"}, fg_logits, 48*CLASSES, 52*CLASSES-1);
        $readmemh({fixture,"/train_cumulative.hex"}, tv_cumulative);
        $readmemh({fixture,"/train_traces.hex"}, fg_traces);
        $readmemh({fixture,"/train_membrane_post_probes.hex"}, tv_membrane_probes, 0, 48*24-1);
        $readmemh({fixture,"/query_membrane_post_probes.hex"}, tv_membrane_probes, 48*24, 52*24-1);
        $readmemh({fixture,"/query_before_currents.hex"}, tv_query_before_currents);
        $readmemh({fixture,"/query_before_centered.hex"}, tv_query_before_logits);
        for (int window = 0; window < 3; window++) begin
            $readmemh($sformatf("%s/window%02d_codes.hex",fixture,window+1),fg_codes,window*PARAMS,(window+1)*PARAMS-1);
            $readmemh($sformatf("%s/window%02d_residuals.hex",fixture,window+1),fg_residuals,window*PARAMS,(window+1)*PARAMS-1);
            $readmemh($sformatf("%s/window%02d_gradient.hex",fixture,window+1),tv_gradients,window*PARAMS,(window+1)*PARAMS-1);
            $readmemh($sformatf("%s/window%02d_delta.hex",fixture,window+1),tv_deltas,window*PARAMS,(window+1)*PARAMS-1);
            $readmemh($sformatf("%s/window%02d_rmw_mask.hex",fixture,window+1),fg_rmw_mask,window*PARAMS,(window+1)*PARAMS-1);
        end
        for (int index = 0; index < PARAMS; index++) begin
            if (classifier_weights[index] !== fg_initial_codes[index]) $fatal(1, "initial bank mismatch index=%0d", index);
            // Cold start only. No residual fixture ever initializes the DDR.
            fg_store_residual[index] = 0;
        end
    endtask

    task automatic fg_start_window;
        if (timestep_phase != 0) fg_window_version = ft_o_model_version;
        if (!tv_query) begin
            fg_rmw = 0; fg_learning_reads = 0;
            for (int index = 0; index < PARAMS; index++) begin
                fg_rmw_seen[index] = 0; tv_gradient_seen[index] = 0; tv_delta_seen[index] = 0;
                tv_observed_gradient[index] = 0; tv_contributions[index] = 0;
                tv_before_codes[index] = classifier_weights[index]; tv_before_residuals[index] = fg_store_residual[index];
            end
        end
        @(negedge clk); fg_request = 1;
        do @(posedge clk); while (!ft_o_request_ready);
        @(negedge clk); fg_request = 0;
        $display("SCNN_TEMPORAL_WINDOW_START window=%0d query=%0b base=%0d batch=%0d version=%0d cycle=%0d",
            tv_window+1,tv_query,ft_i_timestep,ft_i_batch_count,fg_window_version,cycle_count); $fflush();
    endtask

    task automatic tv_load_frame;
        for (int index = 0; index < 2048; index++) begin
            input_patch[index] = tv_raw[timestep_phase*2048+index];
            conv8_expected_pooled_q8_8[index] = fg_features[timestep_phase*2048+index];
            conv8_expected_pooled_spike[index] = fg_features[timestep_phase*2048+index] != 0;
        end
    endtask

    task automatic restore_tile_context(input int tile);
        int index;
        logic [319:0] restored_state, restored_trace;
        for (int row = 0; row < 2; row++) begin
            restored_state = '0; restored_trace = '0;
            for (int lane = 0; lane < 20; lane++)
                if (row*20+lane < int'(context_spatial_count) && tv_time != 0) begin
                    index = (int'(context_output_channel_base)+tile)*current_layer_spatial_area()+
                        int'(context_spatial_base)+row*20+lane;
                    restored_state[lane*16+:16] = saved_state(index); restored_trace[lane*16+:16] = saved_trace(index);
                end
            fg_write_row(tile,int'(context_work_base_addr),row,restored_state,restored_trace);
        end
    endtask

    task automatic fg_classifier_restore;
        for (int tile = 0; tile < CLASSES; tile++) for (int row = 0; row < 2; row++)
            fg_write_row(tile,CLASSIFIER_WORK_BASE,row,tv_time == 0 ? 320'd0 : fg_class_state[tv_sample][tile][row],
                tv_time == 0 ? 320'd0 : fg_class_trace[tv_sample][tile][row]);
        fg_classifier_restores++;
    endtask

    task automatic service_release;
        int index;
        logic [319:0] state_data, trace_data;
        logic signed [15:0] observed;
        if (ft_o_state_service_is_classifier) begin
            if (layer_phase != 8 || release_work_base_addr != CLASSIFIER_WORK_BASE ||
                release_output_channel_base != 0 || release_output_channel_count != CLASSES ||
                release_spatial_base != 0 || release_spatial_count != 1 || release_active_pair != 1)
                $fatal(1, "classifier release envelope");
            for (int tile = 0; tile < CLASSES; tile++) for (int row = 0; row < 2; row++) begin
                fg_read_row(tile,CLASSIFIER_WORK_BASE,row,state_data,trace_data);
                fg_class_state[tv_sample][tile][row] = state_data; fg_class_trace[tv_sample][tile][row] = trace_data;
            end
            fg_classifier_spills++;
        end else begin
            if (release_work_base_addr != 0 || release_active_pair != 0 || release_output_channel_count == 0 || release_spatial_count == 0)
                $fatal(1, "Conv release envelope");
            for (int tile = 0; tile < int'(release_output_channel_count); tile++) for (int row = 0; row < 2; row++) begin
                fg_read_row(tile,0,row,state_data,trace_data);
                for (int lane = 0; lane < 20; lane++)
                    if (row*20+lane < int'(release_spatial_count)) begin
                        index = (int'(release_output_channel_base)+tile)*current_layer_spatial_area()+int'(release_spatial_base)+row*20+lane;
                        observed = $signed(state_data[lane*16+:16]);
                        if (observed !== expected_next_state(index))
                            $fatal(1, "full state spill consistency frame=%0d sample=%0d layer=%0d index=%0d got=%0d expected=%0d",
                                timestep_phase,tv_sample,layer_phase,index,observed,expected_next_state(index));
                        save_state(index,observed); save_trace(index,$signed(trace_data[lane*16+:16]));
                    end
            end
        end
        @(negedge clk); context_release_ready = 1;
        @(posedge clk); @(negedge clk); context_release_ready = 0;
    endtask

    task automatic run_current_layer;
        longint unsigned started;
        @(negedge clk); while (!start_ready) @(negedge clk);
        if (!ft_o_graph_frame_active || ft_o_graph_frame_target || int'(ft_o_graph_frame_sample) != int'(tv_sample) ||
            int'(ft_o_graph_frame_timestep) != int'(tv_time) || ft_o_graph_frame_version != fg_window_version)
            $fatal(1, "temporal descriptor metadata frame=%0d layer=%0d sample=%0d/%0d time=%0d/%0d version=%h/%h",
                timestep_phase,layer_phase,ft_o_graph_frame_sample,tv_sample,ft_o_graph_frame_timestep,tv_time,
                ft_o_graph_frame_version,fg_window_version);
        started = cycle_count; start_valid = 1;
        @(posedge clk); @(negedge clk); start_valid = 0;
        while (!done) begin
            if (context_load_valid) service_context();
            else if (context_release_valid) service_release();
            else @(negedge clk);
        end
        while (serializer_busy) @(negedge clk);
        @(negedge clk);
        $display("SCNN_TEMPORAL_LAYER frame=%0d sample=%0d timestep=%0d layer=%0d cycles=%0d descriptors=%0d",
            timestep_phase,tv_sample,tv_time,layer_phase,cycle_count-started,fg_descriptors); $fflush();
    endtask

    task automatic fg_check_store;
        int codes, residuals, window, expected_index;
        codes = 0; residuals = 0; window = tv_query ? 2 : int'(tv_window);
        for (int index = 0; index < PARAMS; index++) begin
            expected_index = window*PARAMS+index;
            if (classifier_weights[index] !== fg_codes[expected_index] || fg_store_residual[index] !== fg_residuals[expected_index] ||
                fg_rmw_seen[index] != (fg_rmw_mask[expected_index] != 0) ||
                tv_observed_gradient[index] !== tv_gradients[expected_index] || tv_contributions[index] > 16 ||
                tv_gradient_seen[index] != (tv_gradients[expected_index] != 0) ||
                tv_delta_seen[index] != (tv_deltas[expected_index] != 0))
                $fatal(1, "complete gradient/store/mask check window=%0d index=%0d",window+1,index);
            codes += int'(classifier_weights[index] != tv_before_codes[index]);
            residuals += int'(fg_store_residual[index] != tv_before_residuals[index]);
        end
        if (codes != int'(tv_counts[window*4]) || residuals != int'(tv_counts[window*4+1]) ||
            fg_rmw != int'(tv_counts[window*4+2]) || fg_learning_reads != fg_rmw || fg_rmw_outstanding)
            $fatal(1, "window counts window=%0d codes=%0d residual=%0d rmw=%0d",window+1,codes,residuals,fg_rmw);
    endtask

    task automatic fg_check_frame;
        if (fg_descriptors != timestep_phase+1 || fg_classifier_spills != timestep_phase+1 ||
            tv_probes_checked != (timestep_phase+1)*3)
            $fatal(1, "frame descriptor/spill/probe completeness");
        if (tv_window_end) begin
            while (fg_windows < int'(tv_window)+1) @(negedge clk);
            while (!ft_o_quiescent || !ft_o_backend_quiescent || ft_o_inference_hold) @(negedge clk);
            fg_check_store();
            if (fg_commits != (tv_query ? 3 : int'(tv_window)+1) || fg_detaches != fg_commits ||
                !ft_o_versions_aligned || classifier_version != CLASSIFIER_VERSION+16'(fg_commits) || classifier_pair != 1)
                $fatal(1, "temporal commit/detach/version boundary");
        end
        $display("SCNN_TEMPORAL_FRAME_STATUS frame=%0d sample=%0d timestep=%0d cycles=%0d commits=%0d",
            timestep_phase,tv_sample,tv_time,cycle_count-perf_frame_start,fg_commits); $fflush();
    endtask

// All hierarchy references are read-only scoreboards; data enters public ports.
`define TV_TAIL dut.u_core.u_system.u_array.g_scnn_graph.u_scnn_graph_client.u_classifier_tail
`define TV_PUB dut.u_core.u_system.g_scnn_client.u_train_infer.u_parameter_plane.u_backend.u_classifier_publisher
`define TV_LEARN dut.u_core.u_parameter_plane.gen_shared_fc_scnn_output_learner.u_shared_output_learner.g_output_learning_core.u_shared_output_learning_path
`define TV_SLICE `TV_LEARN.u_learning_slice
`define TV_REPLAY dut.u_core.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.u_cl.u_observation_replay
    bit tv_feature_seen [0:52*CLASSIFIER_FEATURES-1];
    bit tv_current_seen [0:52*CLASSES-1];
    bit tv_trace_seen [0:48*CLASSIFIER_FEATURES-1];
    bit tv_publish_seen [0:3][0:1891];
    int tv_features_checked, tv_currents_checked, tv_traces_checked, tv_publish_words, tv_trace_resets;
    bit tv_descriptor_pending;
    logic [TIMESTEP_WIDTH-1:0] tv_descriptor_time;
    logic [SAMPLE_WIDTH-1:0] tv_descriptor_sample;
    logic [VERSION_WIDTH-1:0] tv_descriptor_version;
    int tv_descriptor_layer;

    always @(posedge clk) begin : temporal_scoreboard
        int feature, index, class_id, row, pair_id, parity, publication, word_index, trace_row, frame;
        logic signed [15:0] expected_weight;
        logic signed [47:0] contribution;
        if (rst_n) begin
            if (ft_o_sticky_error || ft_o_backend_error || ft_o_owner_encoding_error || ft_o_array_tile_sticky_error != 0)
                $fatal(1, "temporal final top error frame=%0d layer=%0d", timestep_phase, layer_phase);
            if (fg_request && ft_o_request_ready) fg_requests++;
            if (ft_o_done) fg_windows++;
            if (start_valid && start_ready) begin
                if (tv_descriptor_pending) $fatal(1, "duplicate descriptor");
                tv_descriptor_pending = 1;
                tv_descriptor_time = ft_o_graph_frame_timestep;
                tv_descriptor_sample = ft_o_graph_frame_sample;
                tv_descriptor_version = ft_o_graph_frame_version;
                tv_descriptor_layer = layer_phase;
            end
            if (tv_descriptor_pending && ((!ft_o_graph_frame_active && !(done && tv_descriptor_layer == 8)) ||
                ft_o_graph_frame_timestep != tv_descriptor_time || ft_o_graph_frame_sample != tv_descriptor_sample ||
                ft_o_graph_frame_version != tv_descriptor_version || layer_phase != tv_descriptor_layer))
                $fatal(1, "frame metadata changed before descriptor drained");
            if (done) begin
                if (!tv_descriptor_pending || ft_o_state_service_valid || state_read_valid || trace_read_valid ||
                    state_result_valid || trace_result_valid || state_write_valid || trace_write_valid)
                    $fatal(1, "descriptor completed with pending state service");
                tv_descriptor_pending = 0; fg_descriptors++;
            end
            if ((state_write_valid || trace_write_valid || state_read_valid || trace_read_valid ||
                 state_result_valid || trace_result_valid) && !ft_o_state_service_valid)
                $fatal(1, "maintenance outside held service");
            if (dut.u_core.u_system.prescan_request_valid || ft_o_graph_frame_target)
                $fatal(1, "classifier-only test entered Conv learning");
            if (`TV_REPLAY.cfg_fire && `TV_REPLAY.i_cfg_trace_state_reset) begin
                if (tv_query || tv_window != 0 || tv_trace_resets != 0)
                    $fatal(1, "continuation cleared TP state");
                tv_trace_resets++;
            end
            if (`TV_TAIL.u_scheduler.i_feature_wave_valid && `TV_TAIL.u_scheduler.o_feature_wave_ready)
                for (int lane = 0; lane < 8; lane++) if (`TV_TAIL.u_scheduler.i_feature_wave_mask[lane]) begin
                    feature = int'(`TV_TAIL.u_scheduler.i_feature_wave_base)+lane;
                    index = timestep_phase*CLASSIFIER_FEATURES+feature;
                    if (layer_phase != 8 || feature >= CLASSIFIER_FEATURES || tv_feature_seen[index] ||
                        $signed(`TV_TAIL.u_scheduler.i_feature_wave_q8_8_flat[lane*16+:16]) !== fg_features[index])
                        $fatal(1, "temporal feature frame=%0d feature=%0d", timestep_phase, feature);
                    tv_feature_seen[index] = 1; tv_features_checked++;
                end
            if (`TV_TAIL.serializer_scalar_valid && `TV_TAIL.serializer_scalar_ready) begin
                class_id = int'(`TV_TAIL.serializer_scalar_channel); index = timestep_phase*CLASSES+class_id;
                if (class_id >= CLASSES || tv_current_seen[index] ||
                    `TV_TAIL.serializer_scalar_layer_tag != CLASSIFIER_TAG || `TV_TAIL.serializer_scalar_spatial != 0 ||
                    `TV_TAIL.serializer_scalar_pre_state !== fg_currents[index])
                    $fatal(1, "temporal current frame=%0d class=%0d got=%0d expected=%0d", timestep_phase,
                        class_id, `TV_TAIL.serializer_scalar_pre_state, fg_currents[index]);
                tv_current_seen[index] = 1; tv_currents_checked++;
                if (tv_query && fg_currents[index] != tv_query_before_currents[(timestep_phase-48)*CLASSES+class_id])
                    tv_query_current_changes++;
            end
            if (logit_wave_fire && tv_query)
                for (int lane = 0; lane < 12; lane++) if (logit_wave_mask[lane]) begin
                    class_id = int'(logit_wave_base)+lane;
                    if (class_id < CLASSES && $signed(logit_wave_flat[lane*16+:16]) !=
                        tv_query_before_logits[(timestep_phase-48)*CLASSES+class_id]) tv_query_logit_changes++;
                end
            if (dut.u_core.link_system_to_plane_scnn_external_source_trace_valid &&
                dut.u_core.link_plane_to_system_scnn_external_source_trace_ready) begin
                feature = int'(dut.u_core.link_system_to_plane_scnn_external_source_trace_index);
                trace_row = int'(dut.u_core.link_system_to_plane_scnn_external_source_trace_sample_id);
                // Learner rows are local-time major; raw frames are sample major.
                frame = int'(tv_window)*16+(trace_row%4)*4+trace_row/4;
                index = frame*CLASSIFIER_FEATURES+feature;
                if (tv_query || !tv_window_end || trace_row >= 16 || feature >= CLASSIFIER_FEATURES ||
                    tv_trace_seen[index] || dut.u_core.link_system_to_plane_scnn_external_source_trace_q016 !==
                    (16'(fg_traces[index]) << 1)) $fatal(1, "temporal trace window=%0d row=%0d feature=%0d",tv_window,trace_row,feature);
                tv_trace_seen[index] = 1; tv_traces_checked++;
            end
            if (`TV_SLICE.i_contribution_valid && `TV_SLICE.o_contribution_ready) begin
                index = int'(`TV_SLICE.i_contribution_local_addr);
                if (tv_query || !tv_window_end || index >= PARAMS) $fatal(1, "unexpected contribution");
                contribution = $signed({1'b0,`TV_SLICE.i_contribution_source_trace}) * $signed(`TV_SLICE.i_contribution_error);
                tv_observed_gradient[index] += contribution; tv_contributions[index]++;
            end
            if (`TV_SLICE.gradient_sign_valid && `TV_SLICE.gradient_sign_ready) begin
                index = int'(`TV_SLICE.gradient_sign_local_addr);
                if (tv_query || index >= PARAMS || tv_gradient_seen[index] ||
                    `TV_SLICE.gradient_accumulator !== tv_gradients[int'(tv_window)*PARAMS+index])
                    $fatal(1, "reduced gradient mismatch window=%0d index=%0d",tv_window,index);
                tv_gradient_seen[index] = 1;
            end
            if (`TV_SLICE.update_path_i.req_valid && `TV_SLICE.update_path_i.req_ready) begin
                index = int'(`TV_SLICE.update_path_i.req_local_parameter_addr);
                if (tv_query || index >= PARAMS || tv_delta_seen[index] ||
                    `TV_SLICE.update_path_i.req_scaled_residual_delta !== tv_deltas[int'(tv_window)*PARAMS+index])
                    $fatal(1, "scaled delta mismatch window=%0d index=%0d",tv_window,index);
                tv_delta_seen[index] = 1;
            end
            for (int region = 0; region < 4; region++)
                if (`TV_PUB.o_weight_fill_valid[region] && `TV_PUB.i_weight_fill_ready[region]) begin
                    class_id = region*4+int'(`TV_PUB.o_weight_fill_tile_flat[region*2+:2]);
                    pair_id = int'(`TV_PUB.o_weight_fill_addr_flat[region*WEIGHT_AW+WEIGHT_AW-1]);
                    row = int'(`TV_PUB.o_weight_fill_addr_flat[region*WEIGHT_AW+:WEIGHT_AW-1]);
                    parity = int'(`TV_PUB.o_weight_fill_bank[region]);
                    publication = cache_prime_count == 0 ? 0 : int'(tv_window)+1;
                    word_index = class_id*172+row*2+parity;
                    if (tv_query || class_id >= CLASSES || row >= 86 || pair_id != 1 || publication > 3 ||
                        tv_publish_seen[publication][word_index] || (publication > 0 &&
                        (!tv_window_end || fg_rmw != int'(tv_counts[int'(tv_window)*4+2]))))
                        $fatal(1, "temporal publication envelope stage=%0d class=%0d row=%0d",publication,class_id,row);
                    for (int word = 0; word < 12; word++) begin
                        feature = row*24+word*2+parity; expected_weight = 0;
                        if (feature < CLASSIFIER_FEATURES) expected_weight = publication == 0 ?
                            fg_initial_codes[class_id*CLASSIFIER_FEATURES+feature] :
                            fg_codes[(publication-1)*PARAMS+class_id*CLASSIFIER_FEATURES+feature];
                        if ($signed(`TV_PUB.o_weight_fill_data_flat[region*200+word*16+:16]) !== expected_weight)
                            $fatal(1, "temporal publication code mismatch");
                    end
                    if (`TV_PUB.o_weight_fill_data_flat[region*200+192+:8] != 0) $fatal(1, "publication padding");
                    tv_publish_seen[publication][word_index] = 1; tv_publish_words++;
                end
            if (ft_o_commit_pulse) begin
                if (tv_query || !tv_window_end || fg_commits != int'(tv_window) ||
                    tv_publish_words != 1892*(int'(tv_window)+2) ||
                    tv_traces_checked != 32768*(int'(tv_window)+1) ||
                    classifier_version != CLASSIFIER_VERSION+16'(tv_window)+1 || classifier_pair != 1)
                    $fatal(1, "temporal commit before complete update/publication");
                fg_commits++;
                $display("SCNN_TEMPORAL_COMMIT window=%0d code_changes=%0d residual_changes=%0d rmw=%0d version=%0d publications=%0d",
                    tv_window+1,tv_counts[int'(tv_window)*4],tv_counts[int'(tv_window)*4+1],fg_rmw,classifier_version,tv_publish_words); $fflush();
            end
            if (ft_o_detach_pulse) fg_detaches++;
            if (tv_query && (fg_initialize || ft_o_store_learning_read_valid || ft_o_store_learning_write_valid || `TV_PUB.o_busy))
                $fatal(1, "query modified parameter state");
        end
    end

    task automatic fg_finish;
        fg_check_store();
        if (fg_requests != 4 || fg_windows != 4 || fg_descriptors != 52 || fg_classifier_restores != 52 ||
            fg_classifier_spills != 52 || tv_features_checked != 106496 || tv_currents_checked != 572 ||
            classifier_logit_wave_count != 52 || tv_traces_checked != 98304 || tv_publish_words != 7568 ||
            fg_commits != 3 || fg_detaches != 3 || cache_prime_count != 1 || tv_descriptor_pending ||
            fg_fresh_reads == 0 || tv_probes_checked != 156 || tv_trace_resets != 1 ||
            tv_query_current_changes != 4 || tv_query_logit_changes != 4 || tv_total_rmw != 60273)
            $fatal(1, "temporal final counts desc=%0d feat=%0d curr=%0d traces=%0d commits=%0d resets=%0d query=%0d/%0d rmw=%0d",
                fg_descriptors,tv_features_checked,tv_currents_checked,tv_traces_checked,fg_commits,tv_trace_resets,
                tv_query_current_changes,tv_query_logit_changes,tv_total_rmw);
        if (tail_scalars != 425984 || tail_code_writes != 662 || tail_residual_writes != 60273 ||
            tail_forward_responses != 122773504 || tail_learning_responses != 60273 ||
            fg_forward_valid || fg_learning_valid || fg_rmw_outstanding ||
            fg_forward_reads[7] != 122683392 || fg_forward_reads[8] != 90112 ||
            fg_fresh_reads != 9437184)
            $fatal(1, "tail scalar/traffic completeness");
        for (int window = 0; window < 3; window++)
            if (!tail_window_invalidated[window]) $fatal(1, "missing post-publication invalidation");
        $display("SCNN_TAIL_SERVICE scalars=%0d forward_responses=%0d learning_responses=%0d code_writes=%0d residual_writes=%0d invalidations=%0d versions=3 query_uses_updated_rtl=1",
            tail_scalars,tail_forward_responses,tail_learning_responses,tail_code_writes,tail_residual_writes,tail_invalidations);
        for (int bank = 0; bank < 9; bank++) begin
            if (bank < 7 && fg_forward_reads[bank] != 0) $fatal(1, "omitted Conv1..7 bank was read");
            $display("SCNN_TEMPORAL_DDR bank=%0d reads=%0d",bank,fg_forward_reads[bank]);
        end
        $display("SCNN_TAIL_VISIBILITY_PASS frames=52 descriptors=%0d features=%0d currents=%0d logits=%0d traces=%0d probes=%0d rmw=%0d commits=%0d publications=%0d code_changes=662 query_currents=%0d query_logits=%0d fresh_reads=%0d cycles=%0d",
            fg_descriptors,tv_features_checked,tv_currents_checked,classifier_logit_wave_count*CLASSES,tv_traces_checked,
            tv_probes_checked,tv_total_rmw,fg_commits,tv_publish_words,tv_query_current_changes,tv_query_logit_changes,fg_fresh_reads,cycle_count);
        $fflush();
    endtask
`undef TV_TAIL
`undef TV_PUB
`undef TV_LEARN
`undef TV_SLICE
`undef TV_REPLAY


    longint unsigned st_reads = 0, st_responses = 0, st_replacements = 0;
    longint unsigned st_bank_reads [0:8] = '{default:0};
    longint unsigned st_bank_responses [0:8] = '{default:0};
    int st_pending_bank = 0;
    bit st_request_held = 0, st_response_held = 0;
    logic [PARAM_ADDR_WIDTH-1:0] st_held_addr;
    logic [TAG_WIDTH-1:0] st_held_request_tag, st_held_response_tag;
    logic signed [15:0] st_held_code;
    always @(posedge clk) begin : turnover_service_accounting
        bit request_fire, response_fire;
        int bank;
        if (rst_n) begin
            request_fire = ft_o_store_forward_read_valid && ft_i_store_forward_read_ready;
            response_fire = fg_forward_valid && ft_o_store_forward_response_ready;
            if (st_request_held && (!ft_o_store_forward_read_valid ||
                ft_o_store_forward_read_addr !== st_held_addr ||
                ft_o_store_forward_read_tag !== st_held_request_tag))
                $fatal(1,"turnover held request changed");
            if (st_response_held && (!fg_forward_valid || fg_forward_tag !== st_held_response_tag ||
                fg_forward_code !== st_held_code)) $fatal(1,"turnover held response changed");
            if (st_reads - st_responses != 64'(fg_forward_valid))
                $fatal(1,"turnover service credit imbalance");
            if (request_fire && fg_forward_valid && !response_fire)
                $fatal(1,"turnover overwrite of outstanding response");
            // Attribute the retiring response before installing the replacement's bank.
            if (response_fire) begin
                st_responses++; st_bank_responses[st_pending_bank]++;
                if (request_fire && tv_query) st_query_replacements++;
                if (request_fire && st_pending_bank==8 && tv_window_end && !tv_query)
                    st_pub_replacements[tv_window]++;
            end
            if (request_fire) begin
                bank = fg_bank(int'(ft_o_store_forward_read_addr));
                if (bank < 0) $fatal(1,"turnover service unmapped address");
                st_reads++; st_bank_reads[bank]++; st_pending_bank = bank;
            end
            if (request_fire && response_fire) st_replacements++;
            st_request_held = ft_o_store_forward_read_valid && !ft_i_store_forward_read_ready;
            st_response_held = fg_forward_valid && !ft_o_store_forward_response_ready;
            st_held_addr = ft_o_store_forward_read_addr;
            st_held_request_tag = ft_o_store_forward_read_tag;
            st_held_response_tag = fg_forward_tag; st_held_code = fg_forward_code;
        end
    end

    longint unsigned st_query_replacements=0;
    longint unsigned st_pub_replacements [0:2]='{default:0};
    task automatic st_report;
        longint unsigned expected_reads;
        if (fg_forward_valid || st_reads!=st_responses || st_reads!=122773504 ||
            st_replacements==0 || st_query_replacements==0 || PARAM_ADDR_WIDTH!=24 || TAG_WIDTH!=16)
            $fatal(1,"turnover tail final credit/width/coverage mismatch");
        for (int bank=0; bank<9; bank++) begin
            expected_reads=bank<7 ? 0 : bank==7 ? 122683392 : 90112;
            if (st_bank_reads[bank]!=expected_reads || st_bank_responses[bank]!=expected_reads)
                $fatal(1,"turnover tail bank mismatch");
            $display("SCNN_TURNOVER_TAIL_BANK bank=%0d reads=%0d responses=%0d",bank,st_bank_reads[bank],st_bank_responses[bank]);
        end
        for (int window=0;window<3;window++) begin
            if (st_pub_replacements[window]==0) $fatal(1,"post-learning publication did not exercise turnover");
            $display("SCNN_TURNOVER_TAIL_PUBLICATION window=%0d replacements=%0d",window+1,st_pub_replacements[window]);
        end
        $display("SCNN_TURNOVER_TAIL_SERVICE_PASS reads=%0d responses=%0d replacements=%0d query_replacements=%0d slots=1 code_bits=16 address_bits=24 tag_bits=16 minimum_latency=1 replaceable=1 m2_cache_depth=512 end_cycle=%0d",
            st_reads,st_responses,st_replacements,st_query_replacements,cycle_count);
        $fflush();
    endtask

endmodule
`default_nettype wire
