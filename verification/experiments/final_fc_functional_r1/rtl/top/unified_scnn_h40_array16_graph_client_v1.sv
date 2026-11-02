`timescale 1ns/1ps
`default_nettype none

// One uninterrupted final-convolution -> classifier graph transaction for the
// shared 16xH40/P4 substrate. The module contains control/reorder state only:
// both stages issue commands to the same external physical array.
module unified_scnn_h40_array16_graph_client_v1 #(
    parameter int WEIGHT_DEPTH = 1156,
    parameter int WEIGHT_ADDR_WIDTH = $clog2(WEIGHT_DEPTH),
    parameter int WORK_ADDR_WIDTH = 9,
    parameter int FEATURE_WORD_DEPTH = 8192,
    parameter int FEATURE_ADDR_WIDTH = $clog2(FEATURE_WORD_DEPTH),
    parameter int PHYSICAL_TILE_COUNT = 16,
    parameter int MAX_FEATURE_CHANNELS = 512,
    parameter int FEATURE_SPATIAL = 4,
    parameter int MAX_CLASSES = 16,
    parameter int FEATURE_LANES = 8,
    parameter int LOGIT_ENGINES = 12,
    parameter bit ACCUMULATE_SAMPLE_LOGITS = 1'b0,
    parameter int VERSION_WIDTH = 16,
    parameter int FRAME_WIDTH = 8,
    parameter int SAMPLE_WIDTH = 4,
    parameter int TIMESTEP_WIDTH = 5,
    parameter int CHANNEL_WIDTH = MAX_FEATURE_CHANNELS <= 2 ? 1 :
        $clog2(MAX_FEATURE_CHANNELS),
    parameter int CHANNEL_COUNT_WIDTH = $clog2(MAX_FEATURE_CHANNELS + 1),
    parameter int FEATURE_WIDTH = MAX_FEATURE_CHANNELS * FEATURE_SPATIAL <= 2 ?
        1 : $clog2(MAX_FEATURE_CHANNELS * FEATURE_SPATIAL),
    parameter int CLASS_WIDTH = MAX_CLASSES <= 2 ? 1 : $clog2(MAX_CLASSES),
    parameter int CLASS_COUNT_WIDTH = $clog2(MAX_CLASSES + 1),
    parameter int LOGIT_CONTEXTS = 1
) (
    input  wire logic                           i_clk,
    input  wire logic                           i_rst_n,
    input  wire logic                           i_abort,
    input  wire logic                           i_clear_error,

    input  wire logic                           i_start_valid,
    output logic                                o_start_ready,
    input  wire logic [15:0]                    i_conv_in_width,
    input  wire logic [15:0]                    i_conv_in_height,
    input  wire logic [15:0]                    i_conv_in_channels,
    input  wire logic [15:0]                    i_conv_out_width,
    input  wire logic [15:0]                    i_conv_out_height,
    input  wire logic [15:0]                    i_conv_out_channels,
    input  wire logic [7:0]                     i_conv_stride,
    input  wire logic [7:0]                     i_conv_padding,
    input  wire logic [1:0]                     i_conv_pool_mode,
    input  wire logic                           i_conv_active_pair,
    input  wire logic [WORK_ADDR_WIDTH-1:0]     i_conv_work_base_addr,
    input  wire logic [15:0]                    i_conv_leak,
    input  wire logic [31:0]                    i_conv_threshold,
    input  wire logic                           i_conv_reset_to_zero,
    input  wire logic [15:0]                    i_conv_trace_decay,
    input  wire logic [19:0]                    i_conv_layer_tag,

    input  wire logic [CHANNEL_WIDTH-1:0]       i_feature_channel_base,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0] i_feature_channel_count,
    input  wire logic [CLASS_COUNT_WIDTH-1:0]   i_class_count,
    input  wire logic                           i_classifier_active_pair,
    input  wire logic [WORK_ADDR_WIDTH-1:0]     i_classifier_work_base_addr,
    input  wire logic [15:0]                    i_classifier_trace_decay,
    input  wire logic [19:0]                    i_classifier_layer_tag,
    input  wire logic [VERSION_WIDTH-1:0]       i_version,
    input  wire logic [FRAME_WIDTH-1:0]         i_frame_index,
    input  wire logic                           i_sample_reset,
    input  wire logic                           i_sample_last,
    input  wire logic [CLASS_WIDTH-1:0]         i_sample_label,
    input  wire logic                           i_learning_enable,
    input  wire logic                           i_conv8_observation_enable,
    input  wire logic                           i_conv8_observation_target,
    input  wire logic [SAMPLE_WIDTH-1:0]        i_sample_index,
    input  wire logic [TIMESTEP_WIDTH-1:0]      i_timestep_index,

    output logic                                o_context_load_valid,
    input  wire logic                           i_context_load_ready,
    output logic                                o_context_load_weights,
    output logic [9:0]                          o_context_output_channel_base,
    output logic [9:0]                          o_context_output_channel_count,
    output logic [31:0]                         o_context_spatial_base,
    output logic [5:0]                          o_context_spatial_count,
    output logic                                o_context_active_pair,
    output logic [WORK_ADDR_WIDTH-1:0]          o_context_work_base_addr,
    output logic                                o_context_release_valid,
    input  wire logic                           i_context_release_ready,
    output logic [9:0]                          o_release_output_channel_base,
    output logic [9:0]                          o_release_output_channel_count,
    output logic [31:0]                         o_release_spatial_base,
    output logic [5:0]                          o_release_spatial_count,
    output logic                                o_release_active_pair,
    output logic [WORK_ADDR_WIDTH-1:0]          o_release_work_base_addr,

    output logic                                o_feature_request_valid,
    input  wire logic                           i_feature_request_ready,
    output logic [3:0]                          o_feature_term_valid,
    output logic [39:0]                         o_feature_input_channel_flat,
    output logic [7:0]                          o_feature_kernel_row_flat,
    output logic [7:0]                          o_feature_kernel_col_flat,
    output logic [31:0]                         o_feature_spatial_base,
    output logic [5:0]                          o_feature_spatial_count,
    output logic [15:0]                         o_feature_output_row,
    output logic [15:0]                         o_feature_output_col,
    output logic [3:0]                          o_feature_value_bit,
    input  wire logic                           i_feature_response_valid,
    output logic                                o_feature_response_ready,
    input  wire logic [39:0]                    i_feature_mask0,
    input  wire logic [39:0]                    i_feature_mask1,
    input  wire logic [39:0]                    i_feature_mask2,
    input  wire logic [39:0]                    i_feature_mask3,
    input  wire logic                           i_feature_service_error,
    input  wire logic                           i_feature_write_slot,
    input  wire logic [FEATURE_ADDR_WIDTH-1:0]  i_feature_write_base_addr,
    input  wire logic [15:0]                    i_feature_write_row_stride_words,
    input  wire logic [FEATURE_ADDR_WIDTH-1:0]  i_feature_write_channel_stride_words,
    output logic                                o_feature_update_valid,
    input  wire logic                           i_feature_update_ready,
    output logic                                o_feature_update_slot,
    output logic [FEATURE_ADDR_WIDTH-1:0]       o_feature_update_addr,
    output logic [63:0]                         o_feature_update_data,
    output logic [63:0]                         o_feature_update_mask,
    input  wire logic                           i_feature_update_done,

    output logic                                o_feature_trace_wave_valid,
    input  wire logic                           i_feature_trace_wave_ready,
    output logic [FEATURE_LANES-1:0]            o_feature_trace_wave_mask,
    output logic [FEATURE_WIDTH-1:0]            o_feature_trace_wave_base,
    output logic [FEATURE_LANES*16-1:0]         o_feature_trace_wave_q8_8_flat,
    output logic [VERSION_WIDTH-1:0]            o_feature_trace_wave_version,
    output logic [FRAME_WIDTH-1:0]              o_feature_trace_wave_frame_index,
    output logic                                o_feature_trace_wave_first,
    output logic                                o_feature_trace_wave_last,
    output logic                                o_feature_trace_wave_sample_reset,
    output logic                                o_feature_trace_wave_sample_last,
    output logic [CLASS_WIDTH-1:0]              o_feature_sample_label,
    output logic                                o_conv8_surrogate_valid,
    input  wire logic                           i_conv8_surrogate_ready,
    output logic                                o_conv8_surrogate_target,
    output logic [SAMPLE_WIDTH-1:0]             o_conv8_surrogate_sample,
    output logic [TIMESTEP_WIDTH-1:0]           o_conv8_surrogate_timestep,
    output logic [VERSION_WIDTH-1:0]            o_conv8_surrogate_version,
    output logic [19:0]                         o_conv8_surrogate_layer_tag,
    output logic [9:0]                          o_conv8_surrogate_channel,
    output logic [31:0]                         o_conv8_surrogate_spatial,
    output logic signed [15:0]                  o_conv8_membrane_pre_q8_8,
    output logic signed [15:0]                  o_conv8_surrogate_q8_8,
    output logic                                o_logit_wave_valid,
    input  wire logic                           i_logit_wave_ready,
    output logic [LOGIT_ENGINES-1:0]            o_logit_wave_mask,
    output logic [CLASS_WIDTH-1:0]              o_logit_wave_base,
    output logic [CLASS_COUNT_WIDTH-1:0]        o_logit_wave_count,
    output logic [LOGIT_ENGINES*16-1:0]         o_logit_wave_q8_8_flat,
    output logic [VERSION_WIDTH-1:0]            o_logit_wave_version,
    output logic [FRAME_WIDTH-1:0]              o_logit_wave_frame_index,
    output logic                                o_logit_wave_frame_last,
    output logic                                o_logit_wave_sample_last,

    output logic                                o_array_command_valid,
    input  wire logic                           i_array_command_ready,
    output logic [15:0]                         o_array_command_neuron_count,
    output logic                                o_array_command_first,
    output logic                                o_array_command_last,
    output logic [3:0]                          o_array_command_source_valid,
    output logic [5:0]                          o_array_command_spatial_count,
    output logic [9:0]                          o_array_command_output_channel_base,
    output logic [39:0]                         o_array_command_feature_mask0,
    output logic [39:0]                         o_array_command_feature_mask1,
    output logic [39:0]                         o_array_command_feature_mask2,
    output logic [39:0]                         o_array_command_feature_mask3,
    output logic signed [17:0]                  o_array_command_factor_b,
    output logic                                o_array_command_active_pair,
    output logic [3:0]                          o_array_command_weight_subword0,
    output logic [3:0]                          o_array_command_weight_subword1,
    output logic [3:0]                          o_array_command_weight_subword2,
    output logic [3:0]                          o_array_command_weight_subword3,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_array_command_weight_addr0,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_array_command_weight_addr1,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_array_command_weight_addr2,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_array_command_weight_addr3,
    output logic [WORK_ADDR_WIDTH-1:0]          o_array_command_work_base_addr,
    output logic [15:0]                         o_array_command_leak,
    output logic [31:0]                         o_array_command_threshold,
    output logic                                o_array_command_reset_to_zero,
    output logic [15:0]                         o_array_command_trace_decay,
    output logic [51:0]                         o_array_command_transaction_tag,
    input  wire logic                           i_array_command_done,
    input  wire logic                           i_array_busy,
    input  wire logic                           i_array_idle,
    input  wire logic                           i_array_error,
    input  wire logic [3:0]                     i_array_observation_valid,
    output logic [3:0]                          o_array_observation_ready,
    input  wire logic [15:0]                    i_array_observation_tile_flat,
    input  wire logic [7:0]                     i_array_observation_mode_flat,
    input  wire logic [39:0]                    i_array_observation_lane_mask_flat,
    input  wire logic [39:0]                    i_array_observation_range_mask_flat,
    input  wire logic [39:0]                    i_array_observation_spike_flat,
    input  wire logic [1279:0]                  i_array_observation_pre_state_flat,
    input  wire logic [1279:0]                  i_array_observation_trace_next_flat,
    input  wire logic [255:0]                   i_array_observation_meta_flat,

    output logic                                o_done,
    output logic                                o_busy,
    output logic                                o_idle,
    output logic                                o_sticky_error
);
    typedef enum logic [1:0] {IDLE, CONV, CLASSIFIER} state_t;
    state_t state_q;
    logic conv_done_seen_q, local_error_q;

    logic conv_start_valid, conv_start_ready;
    logic conv_context_load_valid, conv_context_load_ready;
    logic conv_context_load_weights;
    logic [9:0] conv_context_channel_base, conv_context_channel_count;
    logic [31:0] conv_context_spatial_base;
    logic [5:0] conv_context_spatial_count;
    logic conv_context_active_pair;
    logic [WORK_ADDR_WIDTH-1:0] conv_context_work_base;
    logic conv_context_release_valid, conv_context_release_ready;
    logic [9:0] conv_release_channel_base, conv_release_channel_count;
    logic [31:0] conv_release_spatial_base;
    logic [5:0] conv_release_spatial_count;
    logic conv_release_active_pair;
    logic [WORK_ADDR_WIDTH-1:0] conv_release_work_base;
    logic conv_command_valid, conv_command_ready;
    logic [15:0] conv_command_neuron_count;
    logic conv_command_first, conv_command_last;
    logic [5:0] conv_command_spatial_count;
    logic [9:0] conv_command_channel_base;
    logic [39:0] conv_command_mask0, conv_command_mask1;
    logic [39:0] conv_command_mask2, conv_command_mask3;
    logic conv_command_active_pair;
    logic [3:0] conv_command_subword0, conv_command_subword1;
    logic [3:0] conv_command_subword2, conv_command_subword3;
    logic [WEIGHT_ADDR_WIDTH-1:0] conv_command_addr0, conv_command_addr1;
    logic [WEIGHT_ADDR_WIDTH-1:0] conv_command_addr2, conv_command_addr3;
    logic [WORK_ADDR_WIDTH-1:0] conv_command_work_base;
    logic [15:0] conv_command_leak, conv_command_trace_decay;
    logic [31:0] conv_command_threshold;
    logic signed [17:0] conv_command_factor_b;
    logic conv_command_reset_to_zero;
    logic [51:0] conv_command_tag;
    logic [3:0] conv_array_observation_ready;
    logic [3:0] conv_observation_valid, conv_observation_ready;
    logic [15:0] conv_observation_tile_flat;
    logic [7:0] conv_observation_mode_flat;
    logic [39:0] conv_observation_lane_mask_flat;
    logic [39:0] conv_observation_range_mask_flat;
    logic [39:0] conv_observation_spike_flat;
    logic [1279:0] conv_observation_pre_state_flat;
    logic [1279:0] conv_observation_trace_next_flat;
    logic [255:0] conv_observation_meta_flat;
    logic conv_busy, conv_done, conv_idle, conv_error;
    logic conv_feature_commit_valid, conv_feature_commit_ready;
    logic classifier_feature_commit_ready;
    logic [19:0] conv_feature_commit_layer_tag;
    logic [9:0] conv_feature_commit_channel;
    logic [31:0] conv_feature_commit_spatial_base;
    logic [5:0] conv_feature_commit_spatial_count;
    logic [39:0] conv_feature_commit_spike;
    logic [639:0] conv_feature_commit_q8_8_flat;

    logic serializer_scalar_valid, serializer_scalar_ready;
    logic [19:0] serializer_layer_tag;
    logic [9:0] serializer_channel;
    logic [31:0] serializer_spatial;
    logic signed [15:0] serializer_scalar_pre_state;
    logic serializer_spike, serializer_busy, serializer_error;
    logic [3:0] serializer_observation_ready;
    logic fork_scalar_ready;
    logic fork_busy, fork_error;

    logic classifier_start_valid, classifier_start_ready;
    logic classifier_context_load_valid, classifier_context_load_ready;
    logic classifier_context_load_weights;
    logic [9:0] classifier_context_channel_base;
    logic [9:0] classifier_context_channel_count;
    logic [31:0] classifier_context_spatial_base;
    logic [5:0] classifier_context_spatial_count;
    logic classifier_context_active_pair;
    logic [WORK_ADDR_WIDTH-1:0] classifier_context_work_base;
    logic classifier_context_release_valid, classifier_context_release_ready;
    logic [9:0] classifier_release_channel_base;
    logic [9:0] classifier_release_channel_count;
    logic [31:0] classifier_release_spatial_base;
    logic [5:0] classifier_release_spatial_count;
    logic classifier_release_active_pair;
    logic [WORK_ADDR_WIDTH-1:0] classifier_release_work_base;
    logic classifier_command_valid, classifier_command_ready;
    logic [15:0] classifier_command_neuron_count;
    logic classifier_command_first, classifier_command_last;
    logic [3:0] classifier_command_source_valid;
    logic [5:0] classifier_command_spatial_count;
    logic [9:0] classifier_command_channel_base;
    logic [39:0] classifier_command_mask0, classifier_command_mask1;
    logic [39:0] classifier_command_mask2, classifier_command_mask3;
    logic classifier_command_active_pair;
    logic [3:0] classifier_command_subword0, classifier_command_subword1;
    logic [3:0] classifier_command_subword2, classifier_command_subword3;
    logic [WEIGHT_ADDR_WIDTH-1:0] classifier_command_addr0;
    logic [WEIGHT_ADDR_WIDTH-1:0] classifier_command_addr1;
    logic [WEIGHT_ADDR_WIDTH-1:0] classifier_command_addr2;
    logic [WEIGHT_ADDR_WIDTH-1:0] classifier_command_addr3;
    logic [WORK_ADDR_WIDTH-1:0] classifier_command_work_base;
    logic [15:0] classifier_command_leak, classifier_command_trace_decay;
    logic [31:0] classifier_command_threshold;
    logic classifier_command_reset_to_zero;
    logic [51:0] classifier_command_tag;
    logic [3:0] classifier_array_observation_ready;
    logic classifier_done, classifier_busy, classifier_idle, classifier_error;

    logic start_fire;
    logic classifier_requested, classifier_requested_q;

    // A zero class count is a runtime conv-only graph transaction.  This lets
    // intermediate SCNN layers reuse the same physical array without starting
    // or reserving the classifier tail; a nonzero count preserves the original
    // final-convolution -> classifier transaction.
    assign classifier_requested = |i_class_count;
    assign o_start_ready = state_q == IDLE && conv_start_ready &&
        (!classifier_requested || classifier_start_ready) && !o_sticky_error;
    assign start_fire = i_start_valid && o_start_ready;
    assign conv_start_valid = i_start_valid && state_q == IDLE &&
        (!classifier_requested || classifier_start_ready) && !o_sticky_error;
    assign classifier_start_valid = i_start_valid && state_q == IDLE &&
        classifier_requested && conv_start_ready && !o_sticky_error;

    // The feature-commit tap is consumed by the classifier tail only for the
    // final layer.  Intermediate conv-only layers already write their result
    // through the external feature-update interface, so acknowledge the tap
    // locally to avoid introducing an inactive-tail backpressure dependency.
    assign conv_feature_commit_ready = classifier_requested_q ?
        classifier_feature_commit_ready : 1'b1;

    assign conv_context_load_ready = state_q == CONV &&
        i_context_load_ready;
    assign classifier_context_load_ready = state_q == CLASSIFIER &&
        i_context_load_ready;
    assign conv_context_release_ready = state_q == CONV &&
        i_context_release_ready;
    assign classifier_context_release_ready = state_q == CLASSIFIER &&
        i_context_release_ready;
    assign conv_command_ready = state_q == CONV && i_array_command_ready;
    assign classifier_command_ready = state_q == CLASSIFIER &&
        i_array_command_ready;
    assign conv_observation_ready = serializer_observation_ready;

    always_comb begin
        o_context_load_valid = 1'b0;
        o_context_load_weights = 1'b0;
        o_context_output_channel_base = '0;
        o_context_output_channel_count = '0;
        o_context_spatial_base = '0;
        o_context_spatial_count = '0;
        o_context_active_pair = 1'b0;
        o_context_work_base_addr = '0;
        o_context_release_valid = 1'b0;
        o_release_output_channel_base = '0;
        o_release_output_channel_count = '0;
        o_release_spatial_base = '0;
        o_release_spatial_count = '0;
        o_release_active_pair = 1'b0;
        o_release_work_base_addr = '0;
        o_array_command_valid = 1'b0;
        o_array_command_neuron_count = '0;
        o_array_command_first = 1'b0;
        o_array_command_last = 1'b0;
        o_array_command_source_valid = '0;
        o_array_command_spatial_count = '0;
        o_array_command_output_channel_base = '0;
        o_array_command_feature_mask0 = '0;
        o_array_command_feature_mask1 = '0;
        o_array_command_feature_mask2 = '0;
        o_array_command_feature_mask3 = '0;
        o_array_command_factor_b = 18'sd256;
        o_array_command_active_pair = 1'b0;
        o_array_command_weight_subword0 = '0;
        o_array_command_weight_subword1 = '0;
        o_array_command_weight_subword2 = '0;
        o_array_command_weight_subword3 = '0;
        o_array_command_weight_addr0 = '0;
        o_array_command_weight_addr1 = '0;
        o_array_command_weight_addr2 = '0;
        o_array_command_weight_addr3 = '0;
        o_array_command_work_base_addr = '0;
        o_array_command_leak = '0;
        o_array_command_threshold = '0;
        o_array_command_reset_to_zero = 1'b0;
        o_array_command_trace_decay = '0;
        o_array_command_transaction_tag = '0;
        o_array_observation_ready = '0;

        if (state_q == CONV) begin
            o_context_load_valid = conv_context_load_valid;
            o_context_load_weights = conv_context_load_weights;
            o_context_output_channel_base = conv_context_channel_base;
            o_context_output_channel_count = conv_context_channel_count;
            o_context_spatial_base = conv_context_spatial_base;
            o_context_spatial_count = conv_context_spatial_count;
            o_context_active_pair = conv_context_active_pair;
            o_context_work_base_addr = conv_context_work_base;
            o_context_release_valid = conv_context_release_valid;
            o_release_output_channel_base = conv_release_channel_base;
            o_release_output_channel_count = conv_release_channel_count;
            o_release_spatial_base = conv_release_spatial_base;
            o_release_spatial_count = conv_release_spatial_count;
            o_release_active_pair = conv_release_active_pair;
            o_release_work_base_addr = conv_release_work_base;
            o_array_command_valid = conv_command_valid;
            o_array_command_neuron_count = conv_command_neuron_count;
            o_array_command_first = conv_command_first;
            o_array_command_last = conv_command_last;
            o_array_command_spatial_count = conv_command_spatial_count;
            o_array_command_output_channel_base = conv_command_channel_base;
            o_array_command_feature_mask0 = conv_command_mask0;
            o_array_command_feature_mask1 = conv_command_mask1;
            o_array_command_feature_mask2 = conv_command_mask2;
            o_array_command_feature_mask3 = conv_command_mask3;
            o_array_command_factor_b = conv_command_factor_b;
            o_array_command_active_pair = conv_command_active_pair;
            o_array_command_weight_subword0 = conv_command_subword0;
            o_array_command_weight_subword1 = conv_command_subword1;
            o_array_command_weight_subword2 = conv_command_subword2;
            o_array_command_weight_subword3 = conv_command_subword3;
            o_array_command_weight_addr0 = conv_command_addr0;
            o_array_command_weight_addr1 = conv_command_addr1;
            o_array_command_weight_addr2 = conv_command_addr2;
            o_array_command_weight_addr3 = conv_command_addr3;
            o_array_command_work_base_addr = conv_command_work_base;
            o_array_command_leak = conv_command_leak;
            o_array_command_threshold = conv_command_threshold;
            o_array_command_reset_to_zero = conv_command_reset_to_zero;
            o_array_command_trace_decay = conv_command_trace_decay;
            o_array_command_transaction_tag = conv_command_tag;
            o_array_observation_ready = conv_array_observation_ready;
        end else if (state_q == CLASSIFIER) begin
            o_context_load_valid = classifier_context_load_valid;
            o_context_load_weights = classifier_context_load_weights;
            o_context_output_channel_base = classifier_context_channel_base;
            o_context_output_channel_count = classifier_context_channel_count;
            o_context_spatial_base = classifier_context_spatial_base;
            o_context_spatial_count = classifier_context_spatial_count;
            o_context_active_pair = classifier_context_active_pair;
            o_context_work_base_addr = classifier_context_work_base;
            o_context_release_valid = classifier_context_release_valid;
            o_release_output_channel_base = classifier_release_channel_base;
            o_release_output_channel_count = classifier_release_channel_count;
            o_release_spatial_base = classifier_release_spatial_base;
            o_release_spatial_count = classifier_release_spatial_count;
            o_release_active_pair = classifier_release_active_pair;
            o_release_work_base_addr = classifier_release_work_base;
            o_array_command_valid = classifier_command_valid;
            o_array_command_neuron_count = classifier_command_neuron_count;
            o_array_command_first = classifier_command_first;
            o_array_command_last = classifier_command_last;
            o_array_command_source_valid = classifier_command_source_valid;
            o_array_command_spatial_count = classifier_command_spatial_count;
            o_array_command_output_channel_base = classifier_command_channel_base;
            o_array_command_feature_mask0 = classifier_command_mask0;
            o_array_command_feature_mask1 = classifier_command_mask1;
            o_array_command_feature_mask2 = classifier_command_mask2;
            o_array_command_feature_mask3 = classifier_command_mask3;
            o_array_command_factor_b = classifier_command_source_valid == 0 ?
                18'sd256 :
                18'(integer'(classifier_command_source_valid) * 64);
            o_array_command_active_pair = classifier_command_active_pair;
            o_array_command_weight_subword0 = classifier_command_subword0;
            o_array_command_weight_subword1 = classifier_command_subword1;
            o_array_command_weight_subword2 = classifier_command_subword2;
            o_array_command_weight_subword3 = classifier_command_subword3;
            o_array_command_weight_addr0 = classifier_command_addr0;
            o_array_command_weight_addr1 = classifier_command_addr1;
            o_array_command_weight_addr2 = classifier_command_addr2;
            o_array_command_weight_addr3 = classifier_command_addr3;
            o_array_command_work_base_addr = classifier_command_work_base;
            o_array_command_leak = classifier_command_leak;
            o_array_command_threshold = classifier_command_threshold;
            o_array_command_reset_to_zero = classifier_command_reset_to_zero;
            o_array_command_trace_decay = classifier_command_trace_decay;
            o_array_command_transaction_tag = classifier_command_tag;
            o_array_observation_ready = classifier_array_observation_ready;
        end
    end

    assign o_busy = state_q != IDLE || conv_busy || serializer_busy ||
        fork_busy || classifier_busy;
    assign o_idle = state_q == IDLE && conv_idle && !serializer_busy &&
        !fork_busy && classifier_idle;
    assign o_sticky_error = local_error_q || conv_error ||
        serializer_error || fork_error || classifier_error ||
        i_feature_service_error || i_array_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= IDLE;
            conv_done_seen_q <= 1'b0;
            classifier_requested_q <= 1'b0;
            local_error_q <= 1'b0;
            o_done <= 1'b0;
        end else if (i_abort) begin
            state_q <= IDLE;
            conv_done_seen_q <= 1'b0;
            classifier_requested_q <= 1'b0;
            local_error_q <= 1'b0;
            o_done <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_clear_error && state_q == IDLE)
                local_error_q <= 1'b0;
            if (state_q != IDLE && o_sticky_error) begin
                state_q <= IDLE;
                local_error_q <= 1'b1;
            end else begin
                case (state_q)
                    IDLE: begin
                        conv_done_seen_q <= 1'b0;
                        if (start_fire) begin
                            classifier_requested_q <= classifier_requested;
                            state_q <= CONV;
                        end
                    end
                    CONV: begin
                        if (conv_done)
                            conv_done_seen_q <= 1'b1;
                        if ((conv_done_seen_q || conv_done) && conv_idle &&
                            !serializer_busy && !fork_busy) begin
                            if (classifier_requested_q) begin
                                state_q <= CLASSIFIER;
                            end else begin
                                state_q <= IDLE;
                                classifier_requested_q <= 1'b0;
                                o_done <= 1'b1;
                            end
                        end
                    end
                    CLASSIFIER: begin
                        if (classifier_done) begin
                            state_q <= IDLE;
                            classifier_requested_q <= 1'b0;
                            o_done <= 1'b1;
                        end
                    end
                    default: begin
                        state_q <= IDLE;
                        local_error_q <= 1'b1;
                    end
                endcase
            end
        end
    end

    unified_scnn_h40_array16_layer_client_v1 #(
        .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH),
        .FEATURE_WORD_DEPTH(FEATURE_WORD_DEPTH),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .PHYSICAL_TILE_COUNT(PHYSICAL_TILE_COUNT),
        .ENABLE_FEATURE_COMMIT_TAP(1'b1),
        // The classifier cache owns the opposite pair for the whole graph.
        // Conv contexts therefore refill and reuse one fixed pair serially.
        .HOLD_ACTIVE_PAIR(1'b1)
    ) u_conv_client (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error), .i_start_valid(conv_start_valid),
        .o_start_ready(conv_start_ready),
        .i_in_width(i_conv_in_width), .i_in_height(i_conv_in_height),
        .i_in_channels(i_conv_in_channels),
        .i_out_width(i_conv_out_width), .i_out_height(i_conv_out_height),
        .i_out_channels(i_conv_out_channels), .i_stride(i_conv_stride),
        .i_padding(i_conv_padding), .i_pool_mode(i_conv_pool_mode),
        .i_initial_active_pair(i_conv_active_pair),
        .i_work_base_addr(i_conv_work_base_addr), .i_leak(i_conv_leak),
        .i_threshold(i_conv_threshold),
        .i_reset_to_zero(i_conv_reset_to_zero),
        .i_trace_decay(i_conv_trace_decay), .i_layer_tag(i_conv_layer_tag),
        .o_context_load_valid(conv_context_load_valid),
        .i_context_load_ready(conv_context_load_ready),
        .o_context_load_weights(conv_context_load_weights),
        .o_context_output_channel_base(conv_context_channel_base),
        .o_context_output_channel_count(conv_context_channel_count),
        .o_context_spatial_base(conv_context_spatial_base),
        .o_context_spatial_count(conv_context_spatial_count),
        .o_context_active_pair(conv_context_active_pair),
        .o_context_work_base_addr(conv_context_work_base),
        .o_feature_request_valid(o_feature_request_valid),
        .i_feature_request_ready(i_feature_request_ready),
        .o_feature_term_valid(o_feature_term_valid),
        .o_feature_input_channel_flat(o_feature_input_channel_flat),
        .o_feature_kernel_row_flat(o_feature_kernel_row_flat),
        .o_feature_kernel_col_flat(o_feature_kernel_col_flat),
        .o_feature_spatial_base(o_feature_spatial_base),
        .o_feature_spatial_count(o_feature_spatial_count),
        .o_feature_output_row(o_feature_output_row),
        .o_feature_output_col(o_feature_output_col),
        .o_feature_value_bit(o_feature_value_bit),
        .i_feature_response_valid(i_feature_response_valid),
        .o_feature_response_ready(o_feature_response_ready),
        .i_feature_mask0(i_feature_mask0), .i_feature_mask1(i_feature_mask1),
        .i_feature_mask2(i_feature_mask2), .i_feature_mask3(i_feature_mask3),
        .i_feature_service_error(i_feature_service_error),
        .i_feature_write_slot(i_feature_write_slot),
        .i_feature_write_base_addr(i_feature_write_base_addr),
        .i_feature_write_row_stride_words(i_feature_write_row_stride_words),
        .i_feature_write_channel_stride_words(
            i_feature_write_channel_stride_words),
        .o_feature_update_valid(o_feature_update_valid),
        .i_feature_update_ready(i_feature_update_ready),
        .o_feature_update_slot(o_feature_update_slot),
        .o_feature_update_addr(o_feature_update_addr),
        .o_feature_update_data(o_feature_update_data),
        .o_feature_update_mask(o_feature_update_mask),
        .i_feature_update_done(i_feature_update_done),
        .o_feature_commit_valid(conv_feature_commit_valid),
        .i_feature_commit_ready(conv_feature_commit_ready),
        .o_feature_commit_layer_tag(conv_feature_commit_layer_tag),
        .o_feature_commit_channel(conv_feature_commit_channel),
        .o_feature_commit_spatial_base(conv_feature_commit_spatial_base),
        .o_feature_commit_spatial_count(conv_feature_commit_spatial_count),
        .o_feature_commit_spike(conv_feature_commit_spike),
        .o_feature_commit_q8_8_flat(conv_feature_commit_q8_8_flat),
        .o_context_release_valid(conv_context_release_valid),
        .i_context_release_ready(conv_context_release_ready),
        .o_release_output_channel_base(conv_release_channel_base),
        .o_release_output_channel_count(conv_release_channel_count),
        .o_release_spatial_base(conv_release_spatial_base),
        .o_release_spatial_count(conv_release_spatial_count),
        .o_release_active_pair(conv_release_active_pair),
        .o_release_work_base_addr(conv_release_work_base),
        .o_array_command_valid(conv_command_valid),
        .i_array_command_ready(conv_command_ready),
        .o_array_command_neuron_count(conv_command_neuron_count),
        .o_array_command_first(conv_command_first),
        .o_array_command_last(conv_command_last),
        .o_array_command_spatial_count(conv_command_spatial_count),
        .o_array_command_output_channel_base(conv_command_channel_base),
        .o_array_command_feature_mask0(conv_command_mask0),
        .o_array_command_feature_mask1(conv_command_mask1),
        .o_array_command_feature_mask2(conv_command_mask2),
        .o_array_command_feature_mask3(conv_command_mask3),
        .o_array_command_factor_b(conv_command_factor_b),
        .o_array_command_active_pair(conv_command_active_pair),
        .o_array_command_weight_subword0(conv_command_subword0),
        .o_array_command_weight_subword1(conv_command_subword1),
        .o_array_command_weight_subword2(conv_command_subword2),
        .o_array_command_weight_subword3(conv_command_subword3),
        .o_array_command_weight_addr0(conv_command_addr0),
        .o_array_command_weight_addr1(conv_command_addr1),
        .o_array_command_weight_addr2(conv_command_addr2),
        .o_array_command_weight_addr3(conv_command_addr3),
        .o_array_command_work_base_addr(conv_command_work_base),
        .o_array_command_leak(conv_command_leak),
        .o_array_command_threshold(conv_command_threshold),
        .o_array_command_reset_to_zero(conv_command_reset_to_zero),
        .o_array_command_trace_decay(conv_command_trace_decay),
        .o_array_command_transaction_tag(conv_command_tag),
        .i_array_command_done(i_array_command_done && state_q == CONV),
        .i_array_busy(i_array_busy && state_q == CONV),
        .i_array_idle(i_array_idle),
        .i_array_error(i_array_error && state_q == CONV),
        .i_array_observation_valid(state_q == CONV ?
            i_array_observation_valid : 4'b0000),
        .o_array_observation_ready(conv_array_observation_ready),
        .i_array_observation_tile_flat(i_array_observation_tile_flat),
        .i_array_observation_mode_flat(i_array_observation_mode_flat),
        .i_array_observation_lane_mask_flat(
            i_array_observation_lane_mask_flat),
        .i_array_observation_range_mask_flat(
            i_array_observation_range_mask_flat),
        .i_array_observation_spike_flat(i_array_observation_spike_flat),
        .i_array_observation_pre_state_flat(
            i_array_observation_pre_state_flat),
        .i_array_observation_trace_next_flat(
            i_array_observation_trace_next_flat),
        .i_array_observation_meta_flat(i_array_observation_meta_flat),
        .o_observation_valid(conv_observation_valid),
        .i_observation_ready(conv_observation_ready),
        .o_observation_tile_flat(conv_observation_tile_flat),
        .o_observation_mode_flat(conv_observation_mode_flat),
        .o_observation_lane_mask_flat(conv_observation_lane_mask_flat),
        .o_observation_range_mask_flat(conv_observation_range_mask_flat),
        .o_observation_spike_flat(conv_observation_spike_flat),
        .o_observation_pre_state_flat(conv_observation_pre_state_flat),
        .o_observation_trace_next_flat(conv_observation_trace_next_flat),
        .o_observation_meta_flat(conv_observation_meta_flat),
        .o_busy(conv_busy), .o_done(conv_done), .o_idle(conv_idle),
        .o_sticky_error(conv_error)
    );

    unified_scnn_h40_tp_observation_serializer_v1 #(
        .PHYSICAL_TILE_COUNT(PHYSICAL_TILE_COUNT)
    ) u_conv_serializer (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_observation_valid(conv_observation_valid),
        .o_observation_ready(serializer_observation_ready),
        .i_observation_tile_flat(conv_observation_tile_flat),
        .i_observation_mode_flat(conv_observation_mode_flat),
        .i_observation_lane_mask_flat(conv_observation_lane_mask_flat),
        .i_observation_range_mask_flat(conv_observation_range_mask_flat),
        .i_observation_spike_flat(conv_observation_spike_flat),
        .i_observation_pre_state_flat(conv_observation_pre_state_flat),
        .i_observation_trace_next_flat(conv_observation_trace_next_flat),
        .i_observation_meta_flat(conv_observation_meta_flat),
        .o_scalar_valid(serializer_scalar_valid),
        .i_scalar_ready(serializer_scalar_ready),
        .o_scalar_layer_tag(serializer_layer_tag),
        .o_scalar_channel(serializer_channel),
        .o_scalar_spatial(serializer_spatial),
        .o_scalar_spike(serializer_spike),
        .o_scalar_pre_state_q8_8(serializer_scalar_pre_state),
        .o_scalar_trace_next_q8_8(),
        .o_scalar_last(), .o_busy(serializer_busy),
        .o_sticky_error(serializer_error)
    );

    assign serializer_scalar_ready = fork_scalar_ready;

    unified_scnn_h40_conv8_scalar_observation_fork_v1 #(
        .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .TIMESTEP_WIDTH(TIMESTEP_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH)
    ) u_conv8_observation_fork (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_observation_enable(i_conv8_observation_enable),
        .i_target(i_conv8_observation_target),
        .i_sample(i_sample_index), .i_timestep(i_timestep_index),
        .i_version(i_version),
        .i_threshold_q8_8($signed(i_conv_threshold[15:0])),
        .i_scalar_valid(serializer_scalar_valid),
        .o_scalar_ready(fork_scalar_ready),
        .i_scalar_layer_tag(serializer_layer_tag),
        .i_scalar_channel(serializer_channel),
        .i_scalar_spatial(serializer_spatial),
        .i_scalar_spike(serializer_spike),
        .i_scalar_pre_state_q8_8(serializer_scalar_pre_state),
        .o_forward_valid(), .i_forward_ready(1'b1),
        .o_forward_layer_tag(), .o_forward_channel(),
        .o_forward_spatial(), .o_forward_spike(),
        .o_observation_valid(o_conv8_surrogate_valid),
        .i_observation_ready(i_conv8_surrogate_ready),
        .o_observation_target(o_conv8_surrogate_target),
        .o_observation_sample(o_conv8_surrogate_sample),
        .o_observation_timestep(o_conv8_surrogate_timestep),
        .o_observation_version(o_conv8_surrogate_version),
        .o_observation_layer_tag(o_conv8_surrogate_layer_tag),
        .o_observation_channel(o_conv8_surrogate_channel),
        .o_observation_spatial(o_conv8_surrogate_spatial),
        .o_observation_pre_state_q8_8(o_conv8_membrane_pre_q8_8),
        .o_observation_surrogate_q8_8(o_conv8_surrogate_q8_8),
        .o_busy(fork_busy), .o_sticky_error(fork_error)
    );

    unified_scnn_h40_classifier_tail_client_v1 #(
        .MAX_FEATURE_CHANNELS(MAX_FEATURE_CHANNELS),
        .FEATURE_SPATIAL(FEATURE_SPATIAL), .MAX_CLASSES(MAX_CLASSES),
        .FEATURE_LANES(FEATURE_LANES), .LOGIT_ENGINES(LOGIT_ENGINES),
        .LOGIT_CONTEXTS(LOGIT_CONTEXTS), .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .ACCUMULATE_SAMPLE_LOGITS(ACCUMULATE_SAMPLE_LOGITS),
        .PHYSICAL_TILE_COUNT(PHYSICAL_TILE_COUNT),
        .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH), .VERSION_WIDTH(VERSION_WIDTH),
        .FRAME_WIDTH(FRAME_WIDTH), .CHANNEL_WIDTH(CHANNEL_WIDTH),
        .CHANNEL_COUNT_WIDTH(CHANNEL_COUNT_WIDTH),
        .FEATURE_WIDTH(FEATURE_WIDTH), .CLASS_WIDTH(CLASS_WIDTH),
        .CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH)
    ) u_classifier_tail (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error), .i_start_valid(classifier_start_valid),
        .o_start_ready(classifier_start_ready),
        .i_feature_layer_tag(i_conv_layer_tag),
        .i_feature_channel_base(i_feature_channel_base),
        .i_feature_channel_count(i_feature_channel_count),
        .i_class_count(i_class_count),
        .i_classifier_active_pair(i_classifier_active_pair),
        .i_classifier_work_base_addr(i_classifier_work_base_addr),
        .i_classifier_trace_decay(i_classifier_trace_decay),
        .i_classifier_layer_tag(i_classifier_layer_tag),
        .i_version(i_version), .i_frame_index(i_frame_index),
        .i_sample_index(i_sample_index),
        .i_sample_reset(i_sample_reset), .i_sample_last(i_sample_last),
        .i_sample_label(i_sample_label),
        .i_learning_enable(i_learning_enable), .i_conv_done(conv_done),
        .i_conv_error(conv_error || serializer_error),
        .i_conv_feature_commit_valid(conv_feature_commit_valid),
        .o_conv_feature_commit_ready(classifier_feature_commit_ready),
        .i_conv_feature_commit_layer_tag(conv_feature_commit_layer_tag),
        .i_conv_feature_commit_channel(conv_feature_commit_channel),
        .i_conv_feature_commit_spatial_base(
            conv_feature_commit_spatial_base),
        .i_conv_feature_commit_spatial_count(
            conv_feature_commit_spatial_count),
        .i_conv_feature_commit_spike(conv_feature_commit_spike),
        .i_conv_feature_commit_q8_8_flat(conv_feature_commit_q8_8_flat),
        .o_feature_trace_wave_valid(o_feature_trace_wave_valid),
        .i_feature_trace_wave_ready(i_feature_trace_wave_ready),
        .o_feature_trace_wave_mask(o_feature_trace_wave_mask),
        .o_feature_trace_wave_base(o_feature_trace_wave_base),
        .o_feature_trace_wave_q8_8_flat(o_feature_trace_wave_q8_8_flat),
        .o_feature_trace_wave_version(o_feature_trace_wave_version),
        .o_feature_trace_wave_frame_index(
            o_feature_trace_wave_frame_index),
        .o_feature_trace_wave_first(o_feature_trace_wave_first),
        .o_feature_trace_wave_last(o_feature_trace_wave_last),
        .o_feature_trace_wave_sample_reset(
            o_feature_trace_wave_sample_reset),
        .o_feature_trace_wave_sample_last(o_feature_trace_wave_sample_last),
        .o_feature_sample_label(o_feature_sample_label),
        .o_context_load_valid(classifier_context_load_valid),
        .i_context_load_ready(classifier_context_load_ready),
        .o_context_load_weights(classifier_context_load_weights),
        .o_context_output_channel_base(classifier_context_channel_base),
        .o_context_output_channel_count(classifier_context_channel_count),
        .o_context_spatial_base(classifier_context_spatial_base),
        .o_context_spatial_count(classifier_context_spatial_count),
        .o_context_active_pair(classifier_context_active_pair),
        .o_context_work_base_addr(classifier_context_work_base),
        .o_context_release_valid(classifier_context_release_valid),
        .i_context_release_ready(classifier_context_release_ready),
        .o_release_output_channel_base(classifier_release_channel_base),
        .o_release_output_channel_count(classifier_release_channel_count),
        .o_release_spatial_base(classifier_release_spatial_base),
        .o_release_spatial_count(classifier_release_spatial_count),
        .o_release_active_pair(classifier_release_active_pair),
        .o_release_work_base_addr(classifier_release_work_base),
        .o_array_command_valid(classifier_command_valid),
        .i_array_command_ready(classifier_command_ready),
        .o_array_command_neuron_count(classifier_command_neuron_count),
        .o_array_command_first(classifier_command_first),
        .o_array_command_last(classifier_command_last),
        .o_array_command_source_valid(classifier_command_source_valid),
        .o_array_command_spatial_count(classifier_command_spatial_count),
        .o_array_command_output_channel_base(classifier_command_channel_base),
        .o_array_command_feature_mask0(classifier_command_mask0),
        .o_array_command_feature_mask1(classifier_command_mask1),
        .o_array_command_feature_mask2(classifier_command_mask2),
        .o_array_command_feature_mask3(classifier_command_mask3),
        .o_array_command_active_pair(classifier_command_active_pair),
        .o_array_command_weight_subword0(classifier_command_subword0),
        .o_array_command_weight_subword1(classifier_command_subword1),
        .o_array_command_weight_subword2(classifier_command_subword2),
        .o_array_command_weight_subword3(classifier_command_subword3),
        .o_array_command_weight_addr0(classifier_command_addr0),
        .o_array_command_weight_addr1(classifier_command_addr1),
        .o_array_command_weight_addr2(classifier_command_addr2),
        .o_array_command_weight_addr3(classifier_command_addr3),
        .o_array_command_work_base_addr(classifier_command_work_base),
        .o_array_command_leak(classifier_command_leak),
        .o_array_command_threshold(classifier_command_threshold),
        .o_array_command_reset_to_zero(classifier_command_reset_to_zero),
        .o_array_command_trace_decay(classifier_command_trace_decay),
        .o_array_command_transaction_tag(classifier_command_tag),
        .i_array_command_done(i_array_command_done &&
            state_q == CLASSIFIER), .i_array_idle(i_array_idle),
        .i_array_error(i_array_error && state_q == CLASSIFIER),
        .i_array_observation_valid(state_q == CLASSIFIER ?
            i_array_observation_valid : 4'b0000),
        .o_array_observation_ready(classifier_array_observation_ready),
        .i_array_observation_tile_flat(i_array_observation_tile_flat),
        .i_array_observation_mode_flat(i_array_observation_mode_flat),
        .i_array_observation_lane_mask_flat(
            i_array_observation_lane_mask_flat),
        .i_array_observation_range_mask_flat(
            i_array_observation_range_mask_flat),
        .i_array_observation_spike_flat(i_array_observation_spike_flat),
        .i_array_observation_pre_state_flat(
            i_array_observation_pre_state_flat),
        .i_array_observation_trace_next_flat(
            i_array_observation_trace_next_flat),
        .i_array_observation_meta_flat(i_array_observation_meta_flat),
        .o_logit_wave_valid(o_logit_wave_valid),
        .i_logit_wave_ready(i_logit_wave_ready),
        .o_logit_wave_mask(o_logit_wave_mask),
        .o_logit_wave_base(o_logit_wave_base),
        .o_logit_wave_count(o_logit_wave_count),
        .o_logit_wave_q8_8_flat(o_logit_wave_q8_8_flat),
        .o_logit_wave_version(o_logit_wave_version),
        .o_logit_wave_frame_index(o_logit_wave_frame_index),
        .o_logit_wave_frame_last(o_logit_wave_frame_last),
        .o_logit_wave_sample_last(o_logit_wave_sample_last),
        .o_done(classifier_done), .o_busy(classifier_busy),
        .o_idle(classifier_idle), .o_sticky_error(classifier_error)
    );

    initial begin
        if (FEATURE_SPATIAL < 1 || FEATURE_SPATIAL > 40)
            $error("SCNN graph classifier spatial shape must fit H40");
    end
endmodule

`default_nettype wire
