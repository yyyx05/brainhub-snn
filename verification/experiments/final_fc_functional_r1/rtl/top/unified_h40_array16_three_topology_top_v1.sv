`timescale 1ns/1ps
`default_nettype none

// Product-oriented FC/SRNN/SCNN inference substrate. FC and SRNN issue direct
// transactions; the SCNN client expands a layer into the same P4 command
// format. Exactly one native 12xH40 array owns all MAC, LIF, state, trace and
// weight storage resources. Four-lane region ports are retained at this
// compatibility boundary; region slot 3 is physically absent and inert.
module unified_h40_array16_three_topology_top_v1 #(
    parameter int M2_CACHE_DEPTH = 578,
    parameter int SCNN_WEIGHT_FRACTIONAL_BITS = 8,
    parameter int WEIGHT_DEPTH = 1156,
    parameter int WEIGHT_ADDR_WIDTH = $clog2(WEIGHT_DEPTH),
    parameter int MEMORY_DEPTH = 1024,
    parameter int MEMORY_ADDR_WIDTH = $clog2(MEMORY_DEPTH),
    parameter int WORK_ADDR_WIDTH = $clog2(MEMORY_DEPTH / 2),
    parameter int FEATURE_WORD_DEPTH = 8192,
    parameter int FEATURE_ADDR_WIDTH = $clog2(FEATURE_WORD_DEPTH),
    parameter bit CUT_RECORD_READY = 1'b1,
    parameter bit ENABLE_FC_SRNN_LAYER_CLIENT = 1'b0,
    parameter bit ENABLE_FC_SRNN_TP_OBSERVATION_STREAM = 1'b0,
    parameter bit ENABLE_SCNN_CLIENT = 1'b1,
    parameter bit ENABLE_SCNN_CLASSIFIER_GRAPH = 1'b0,
    parameter bit ENABLE_SCNN_STATE_SERVICE = 1'b0,
    parameter int SCNN_LOGIT_CONTEXTS = 1,
    parameter bit ACCUMULATE_SCNN_SAMPLE_LOGITS = 1'b0,
    parameter int MAX_SCNN_FEATURE_CHANNELS = 512,
    parameter int SCNN_CLASSIFIER_FEATURE_SPATIAL = 4,
    parameter int MAX_SCNN_CLASSES = 16,
    parameter int SCNN_FEATURE_LANES = 8,
    parameter int SCNN_LOGIT_ENGINES = 12,
    parameter int SCNN_SAMPLE_WIDTH = 4,
    parameter int SCNN_TIMESTEP_WIDTH = 5,
    parameter int SCNN_CHANNEL_WIDTH = MAX_SCNN_FEATURE_CHANNELS <= 2 ? 1 :
        $clog2(MAX_SCNN_FEATURE_CHANNELS),
    parameter int SCNN_CHANNEL_COUNT_WIDTH =
        $clog2(MAX_SCNN_FEATURE_CHANNELS + 1),
    parameter int SCNN_FEATURE_WIDTH =
        MAX_SCNN_FEATURE_CHANNELS * SCNN_CLASSIFIER_FEATURE_SPATIAL <= 2 ?
        1 : $clog2(MAX_SCNN_FEATURE_CHANNELS *
                   SCNN_CLASSIFIER_FEATURE_SPATIAL),
    parameter int SCNN_CLASS_WIDTH = MAX_SCNN_CLASSES <= 2 ? 1 :
        $clog2(MAX_SCNN_CLASSES),
    parameter int SCNN_CLASS_COUNT_WIDTH = $clog2(MAX_SCNN_CLASSES + 1)
) (
    input  wire logic                           i_clk,
    input  wire logic                           i_rst_n,
    input  wire logic                           i_abort,
    input  wire logic                           i_clear_error,

    input  wire logic                           i_direct_command_valid,
    output logic                                o_direct_command_ready,
    input  wire logic [1:0]                     i_direct_mode,
    input  wire logic [15:0]                    i_direct_neuron_count,
    input  wire logic                           i_direct_first,
    input  wire logic                           i_direct_last,
    input  wire logic [3:0]                     i_direct_source_valid,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]   i_direct_weight_addr0,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]   i_direct_weight_addr1,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]   i_direct_weight_addr2,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]   i_direct_weight_addr3,
    input  wire logic [15:0]                    i_direct_leak,
    input  wire logic [31:0]                    i_direct_threshold,
    input  wire logic                           i_direct_reset_to_zero,
    input  wire logic [15:0]                    i_direct_trace_decay,
    input  wire logic [51:0]                    i_direct_transaction_tag,
    input  wire logic [WORK_ADDR_WIDTH-1:0]     i_direct_work_base_addr,
    input  wire logic [3:0]                     i_direct_destination_tile_base,
    output logic                                o_direct_command_done,
    output logic                                o_direct_transaction_done,
    output logic [3:0]                          o_direct_observation_valid,
    input  wire logic [3:0]                     i_direct_observation_ready,
    output logic [15:0]                         o_direct_observation_tile_flat,
    output logic [7:0]                          o_direct_observation_mode_flat,
    output logic [39:0]                         o_direct_observation_lane_mask_flat,
    output logic [39:0]                         o_direct_observation_range_mask_flat,
    output logic [39:0]                         o_direct_observation_spike_flat,
    output logic [1279:0]                       o_direct_observation_pre_state_flat,
    output logic [1279:0]                       o_direct_observation_trace_next_flat,
    output logic [255:0]                        o_direct_observation_meta_flat,

    input  wire logic                           i_fc_srnn_layer_valid,
    input wire logic i_fc_srnn_streamed_feedforward,
    input wire logic i_fc_srnn_group_feedforward_end,
    output logic                                o_fc_srnn_layer_ready,
    input  wire logic [1:0]                     i_fc_srnn_mode,
    input  wire logic                           i_fc_srnn_weight_int8,
    input  wire logic                           i_fc_srnn_current_shift6,
    input  wire logic [15:0]                    i_fc_srnn_neuron_count,
    input  wire logic [15:0]                    i_fc_srnn_feedforward_group_count,
    input  wire logic [15:0]                    i_fc_srnn_recurrent_group_count,
    input  wire logic                           i_fc_srnn_recurrent_enable,
    input  wire logic [15:0]                    i_fc_srnn_leak,
    input  wire logic [31:0]                    i_fc_srnn_threshold,
    input  wire logic                           i_fc_srnn_reset_to_zero,
    input  wire logic [15:0]                    i_fc_srnn_trace_decay,
    input  wire logic [51:0]                    i_fc_srnn_transaction_tag,
    input  wire logic [WORK_ADDR_WIDTH-1:0]     i_fc_srnn_work_base_addr,
    input  wire logic [3:0]                     i_fc_srnn_destination_tile_base,
    input  wire logic                           i_fc_srnn_group_valid,
    output logic                                o_fc_srnn_group_ready,
    input  wire logic                           i_fc_srnn_group_recurrent,
    input  wire logic [3:0]                     i_fc_srnn_group_source_valid,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]   i_fc_srnn_group_weight_addr0,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]   i_fc_srnn_group_weight_addr1,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]   i_fc_srnn_group_weight_addr2,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]   i_fc_srnn_group_weight_addr3,
    output logic                                o_fc_srnn_layer_done,
    output logic                                o_fc_srnn_busy,
    output logic                                o_fc_srnn_idle,
    output logic                                o_fc_srnn_sticky_error,
    output logic [7:0]                          o_fc_srnn_error_code,
    output logic                                o_fc_srnn_observation_valid,
    input  wire logic                           i_fc_srnn_observation_ready,
    input  wire logic                           i_fc_srnn_observation_sink_idle,
    output logic [1:0]                          o_fc_srnn_observation_mode,
    output logic [51:0]                         o_fc_srnn_observation_transaction_tag,
    output logic [9:0]                          o_fc_srnn_observation_neuron_id,
    output logic                                o_fc_srnn_observation_spike,
    output logic signed [31:0]                  o_fc_srnn_observation_pre_state,
    output logic signed [31:0]                  o_fc_srnn_observation_trace_next,
    output logic                                o_fc_srnn_observation_last_in_record,
    output logic                                o_fc_srnn_observation_busy,
    output logic                                o_fc_srnn_observation_sticky_error,

    input  wire logic                           i_scnn_start_valid,
    output logic                                o_scnn_start_ready,
    input  wire logic [15:0]                    i_scnn_in_width,
    input  wire logic [15:0]                    i_scnn_in_height,
    input  wire logic [15:0]                    i_scnn_in_channels,
    input  wire logic [15:0]                    i_scnn_out_width,
    input  wire logic [15:0]                    i_scnn_out_height,
    input  wire logic [15:0]                    i_scnn_out_channels,
    input  wire logic [7:0]                     i_scnn_stride,
    input  wire logic [7:0]                     i_scnn_padding,
    input  wire logic [1:0]                     i_scnn_pool_mode,
    input  wire logic                           i_scnn_initial_active_pair,
    input  wire logic [WORK_ADDR_WIDTH-1:0]     i_scnn_work_base_addr,
    input  wire logic [15:0]                    i_scnn_leak,
    input  wire logic [31:0]                    i_scnn_threshold,
    input  wire logic                           i_scnn_reset_to_zero,
    input  wire logic [15:0]                    i_scnn_trace_decay,
    input  wire logic [19:0]                    i_scnn_layer_tag,
    input  wire logic [SCNN_CHANNEL_WIDTH-1:0]  i_scnn_classifier_feature_channel_base,
    input  wire logic [SCNN_CHANNEL_COUNT_WIDTH-1:0] i_scnn_classifier_feature_channel_count,
    input  wire logic [SCNN_CLASS_COUNT_WIDTH-1:0] i_scnn_classifier_class_count,
    input  wire logic                           i_scnn_classifier_active_pair,
    input  wire logic [WORK_ADDR_WIDTH-1:0]     i_scnn_classifier_work_base_addr,
    input  wire logic [15:0]                    i_scnn_classifier_trace_decay,
    input  wire logic [19:0]                    i_scnn_classifier_layer_tag,
    input  wire logic [15:0]                    i_scnn_version,
    input  wire logic [7:0]                     i_scnn_frame_index,
    input  wire logic                           i_scnn_sample_reset,
    input  wire logic                           i_scnn_sample_last,
    input  wire logic [SCNN_CLASS_WIDTH-1:0]    i_scnn_sample_label,
    input  wire logic                           i_scnn_learning_enable,
    input  wire logic                           i_scnn_conv8_observation_enable,
    input  wire logic                           i_scnn_conv8_observation_target,
    input  wire logic [SCNN_SAMPLE_WIDTH-1:0]   i_scnn_sample_index,
    input  wire logic [SCNN_TIMESTEP_WIDTH-1:0] i_scnn_timestep_index,

    output logic                                o_scnn_context_load_valid,
    input  wire logic                           i_scnn_context_load_ready,
    output logic                                o_scnn_context_load_weights,
    output logic [9:0]                          o_scnn_context_output_channel_base,
    output logic [9:0]                          o_scnn_context_output_channel_count,
    output logic [31:0]                         o_scnn_context_spatial_base,
    output logic [5:0]                          o_scnn_context_spatial_count,
    output logic                                o_scnn_context_active_pair,
    output logic [WORK_ADDR_WIDTH-1:0]          o_scnn_context_work_base_addr,
    output logic                                o_scnn_feature_request_valid,
    input  wire logic                           i_scnn_feature_request_ready,
    output logic [3:0]                          o_scnn_feature_term_valid,
    output logic [39:0]                         o_scnn_feature_input_channel_flat,
    output logic [7:0]                          o_scnn_feature_kernel_row_flat,
    output logic [7:0]                          o_scnn_feature_kernel_col_flat,
    output logic [31:0]                         o_scnn_feature_spatial_base,
    output logic [5:0]                          o_scnn_feature_spatial_count,
    output logic [15:0]                         o_scnn_feature_output_row,
    output logic [15:0]                         o_scnn_feature_output_col,
    output logic [3:0]                          o_scnn_feature_value_bit,
    input  wire logic                           i_scnn_feature_response_valid,
    output logic                                o_scnn_feature_response_ready,
    input  wire logic [39:0]                    i_scnn_feature_mask0,
    input  wire logic [39:0]                    i_scnn_feature_mask1,
    input  wire logic [39:0]                    i_scnn_feature_mask2,
    input  wire logic [39:0]                    i_scnn_feature_mask3,
    input  wire logic                           i_scnn_feature_service_error,
    input  wire logic                           i_scnn_feature_write_slot,
    input  wire logic [FEATURE_ADDR_WIDTH-1:0]  i_scnn_feature_write_base_addr,
    input  wire logic [15:0]                    i_scnn_feature_write_row_stride_words,
    input  wire logic [FEATURE_ADDR_WIDTH-1:0]  i_scnn_feature_write_channel_stride_words,
    output logic                                o_scnn_feature_update_valid,
    input  wire logic                           i_scnn_feature_update_ready,
    output logic                                o_scnn_feature_update_slot,
    output logic [FEATURE_ADDR_WIDTH-1:0]       o_scnn_feature_update_addr,
    output logic [63:0]                         o_scnn_feature_update_data,
    output logic [63:0]                         o_scnn_feature_update_mask,
    input  wire logic                           i_scnn_feature_update_done,
    output logic                                o_scnn_context_release_valid,
    input  wire logic                           i_scnn_context_release_ready,
    output logic [9:0]                          o_scnn_release_output_channel_base,
    output logic [9:0]                          o_scnn_release_output_channel_count,
    output logic [31:0]                         o_scnn_release_spatial_base,
    output logic [5:0]                          o_scnn_release_spatial_count,
    output logic                                o_scnn_release_active_pair,
    output logic [WORK_ADDR_WIDTH-1:0]          o_scnn_release_work_base_addr,
    output logic [3:0]                          o_scnn_observation_valid,
    input  wire logic [3:0]                     i_scnn_observation_ready,
    input  wire logic                           i_scnn_observation_sink_idle,
    output logic [15:0]                         o_scnn_observation_tile_flat,
    output logic [7:0]                          o_scnn_observation_mode_flat,
    output logic [39:0]                         o_scnn_observation_lane_mask_flat,
    output logic [39:0]                         o_scnn_observation_range_mask_flat,
    output logic [39:0]                         o_scnn_observation_spike_flat,
    output logic [1279:0]                       o_scnn_observation_pre_state_flat,
    output logic [1279:0]                       o_scnn_observation_trace_next_flat,
    output logic [255:0]                        o_scnn_observation_meta_flat,
    output logic                                o_scnn_done,
    output logic                                o_scnn_feature_trace_wave_valid,
    input  wire logic                           i_scnn_feature_trace_wave_ready,
    output logic [SCNN_FEATURE_LANES-1:0]       o_scnn_feature_trace_wave_mask,
    output logic [SCNN_FEATURE_WIDTH-1:0]       o_scnn_feature_trace_wave_base,
    output logic [SCNN_FEATURE_LANES*16-1:0]    o_scnn_feature_trace_wave_q8_8_flat,
    output logic [15:0]                         o_scnn_feature_trace_wave_version,
    output logic [7:0]                          o_scnn_feature_trace_wave_frame_index,
    output logic                                o_scnn_feature_trace_wave_first,
    output logic                                o_scnn_feature_trace_wave_last,
    output logic                                o_scnn_feature_trace_wave_sample_reset,
    output logic                                o_scnn_feature_trace_wave_sample_last,
    output logic [SCNN_CLASS_WIDTH-1:0]         o_scnn_feature_sample_label,
    output logic                                o_scnn_conv8_surrogate_valid,
    input  wire logic                           i_scnn_conv8_surrogate_ready,
    output logic                                o_scnn_conv8_surrogate_target,
    output logic [SCNN_SAMPLE_WIDTH-1:0]        o_scnn_conv8_surrogate_sample,
    output logic [SCNN_TIMESTEP_WIDTH-1:0]      o_scnn_conv8_surrogate_timestep,
    output logic [15:0]                         o_scnn_conv8_surrogate_version,
    output logic [19:0]                         o_scnn_conv8_surrogate_layer_tag,
    output logic [9:0]                          o_scnn_conv8_surrogate_channel,
    output logic [31:0]                         o_scnn_conv8_surrogate_spatial,
    output logic signed [15:0]                  o_scnn_conv8_membrane_pre_q8_8,
    output logic signed [15:0]                  o_scnn_conv8_surrogate_q8_8,
    output logic                                o_scnn_logit_wave_valid,
    input  wire logic                           i_scnn_logit_wave_ready,
    output logic [SCNN_LOGIT_ENGINES-1:0]       o_scnn_logit_wave_mask,
    output logic [SCNN_CLASS_WIDTH-1:0]         o_scnn_logit_wave_base,
    output logic [SCNN_CLASS_COUNT_WIDTH-1:0]   o_scnn_logit_wave_count,
    output logic [SCNN_LOGIT_ENGINES*16-1:0]    o_scnn_logit_wave_q8_8_flat,
    output logic [15:0]                         o_scnn_logit_wave_version,
    output logic [7:0]                          o_scnn_logit_wave_frame_index,
    output logic                                o_scnn_logit_wave_frame_last,
    output logic                                o_scnn_logit_wave_sample_last,

    input  wire logic                          i_weight_fill_context,
    input  wire logic [3:0]                     i_weight_fill_valid,
    output logic [3:0]                          o_weight_fill_ready,
    output logic                                o_scnn_context_refill_safe,
    input  wire logic [7:0]                     i_weight_fill_tile_flat,
    input  wire logic [3:0]                     i_weight_fill_bank,
    input  wire logic [4*WEIGHT_ADDR_WIDTH-1:0] i_weight_fill_addr_flat,
    input  wire logic [799:0]                   i_weight_fill_data_flat,
    input  wire logic                           i_maintenance_start_valid,
    output logic                                o_maintenance_start_ready,
    input  wire logic                           i_maintenance_release_valid,
    output logic                                o_maintenance_release_ready,
    output logic                                o_maintenance_done,
    input  wire logic [3:0]                     i_maintenance_tile,
    input  wire logic                           i_state_maintenance_write_valid,
    output logic                                o_state_maintenance_write_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]   i_state_maintenance_write_addr,
    input  wire logic [319:0]                   i_state_maintenance_write_data,
    input  wire logic [39:0]                    i_state_maintenance_write_mask,
    input  wire logic                           i_state_maintenance_read_valid,
    output logic                                o_state_maintenance_read_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]   i_state_maintenance_read_addr,
    output logic                                o_state_maintenance_result_valid,
    input  wire logic                           i_state_maintenance_result_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]        o_state_maintenance_result_addr,
    output logic [319:0]                        o_state_maintenance_read_data,
    input  wire logic                           i_trace_maintenance_write_valid,
    output logic                                o_trace_maintenance_write_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]   i_trace_maintenance_write_addr,
    input  wire logic [319:0]                   i_trace_maintenance_write_data,
    input  wire logic [39:0]                    i_trace_maintenance_write_mask,
    input  wire logic                           i_trace_maintenance_read_valid,
    output logic                                o_trace_maintenance_read_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]   i_trace_maintenance_read_addr,
    output logic                                o_trace_maintenance_result_valid,
    input  wire logic                           i_trace_maintenance_result_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]        o_trace_maintenance_result_addr,
    output logic [319:0]                        o_trace_maintenance_read_data,

    output logic [1:0]                          o_active_owner,
    output logic                                o_busy,
    output logic                                o_idle,
    output logic [15:0]                         o_tile_done,
    output logic [15:0]                         o_tile_idle,
    output logic [15:0]                         o_tile_sticky_error,
    output logic                                o_sticky_error
);
    localparam int PHYSICAL_TILE_COUNT = 12;

    logic select_direct;
    logic select_scnn;
    logic select_maintenance;
    logic owner_select_scnn;
    logic owner_busy;
    logic owner_idle;
    logic owner_error;
    logic owner_scnn_start_ready;
    logic owner_scnn_client_start_valid;
    logic owner_scnn_done;
    logic scnn_client_start_valid;
    logic scnn_client_start_ready;
    logic scnn_client_done;
    logic scnn_client_busy;
    logic scnn_client_idle;
    logic scnn_client_error;

    logic direct_command_valid;
    logic direct_command_ready;
    logic [1:0] direct_mode;
    logic direct_fc_weight_int8;
    logic direct_fc_current_shift6;
    logic [15:0] direct_neuron_count;
    logic direct_first;
    logic direct_last;
    logic [3:0] direct_source_valid;
    logic [WEIGHT_ADDR_WIDTH-1:0] direct_weight_addr0;
    logic [WEIGHT_ADDR_WIDTH-1:0] direct_weight_addr1;
    logic [WEIGHT_ADDR_WIDTH-1:0] direct_weight_addr2;
    logic [WEIGHT_ADDR_WIDTH-1:0] direct_weight_addr3;
    logic [15:0] direct_leak;
    logic [31:0] direct_threshold;
    logic direct_reset_to_zero;
    logic [15:0] direct_trace_decay;
    logic [51:0] direct_transaction_tag;
    logic [WORK_ADDR_WIDTH-1:0] direct_work_base_addr;
    logic [3:0] direct_destination_tile_base;
    logic direct_transaction_done;
    logic fc_srnn_client_busy;
    logic fc_srnn_client_idle;
    logic fc_srnn_client_error;
    logic fc_srnn_owner_request_valid;
    logic fc_srnn_owner_request_ready;

    logic scnn_command_valid;
    logic scnn_command_ready;
    logic [15:0] scnn_command_neuron_count;
    logic scnn_command_first;
    logic scnn_command_last;
    logic [3:0] scnn_command_source_valid;
    logic [5:0] scnn_command_spatial_count;
    logic [9:0] scnn_command_output_channel_base;
    logic [39:0] scnn_command_feature_mask0;
    logic [39:0] scnn_command_feature_mask1;
    logic [39:0] scnn_command_feature_mask2;
    logic [39:0] scnn_command_feature_mask3;
    logic signed [17:0] scnn_command_factor_b;
    logic scnn_command_active_pair;
    logic [3:0] scnn_command_weight_subword0;
    logic [3:0] scnn_command_weight_subword1;
    logic [3:0] scnn_command_weight_subword2;
    logic [3:0] scnn_command_weight_subword3;
    logic [WEIGHT_ADDR_WIDTH-1:0] scnn_command_weight_addr0;
    logic [WEIGHT_ADDR_WIDTH-1:0] scnn_command_weight_addr1;
    logic [WEIGHT_ADDR_WIDTH-1:0] scnn_command_weight_addr2;
    logic [WEIGHT_ADDR_WIDTH-1:0] scnn_command_weight_addr3;
    logic [WORK_ADDR_WIDTH-1:0] scnn_command_work_base_addr;
    logic [15:0] scnn_command_leak;
    logic [31:0] scnn_command_threshold;
    logic scnn_command_reset_to_zero;
    logic [15:0] scnn_command_trace_decay;
    logic [51:0] scnn_command_transaction_tag;

    logic array_command_valid;
    logic array_command_ready;
    logic [1:0] array_command_mode;
    logic array_command_fc_weight_int8;
    logic array_command_fc_current_shift6;
    logic [15:0] array_command_neuron_count;
    logic array_command_first;
    logic array_command_last;
    logic [3:0] array_command_source_valid;
    logic [5:0] array_command_spatial_count;
    logic [9:0] array_command_output_channel_base;
    logic [39:0] array_command_feature_mask0;
    logic [39:0] array_command_feature_mask1;
    logic [39:0] array_command_feature_mask2;
    logic [39:0] array_command_feature_mask3;
    logic signed [17:0] array_command_factor_b;
    logic array_command_active_pair;
    logic [3:0] array_command_weight_subword0;
    logic [3:0] array_command_weight_subword1;
    logic [3:0] array_command_weight_subword2;
    logic [3:0] array_command_weight_subword3;
    logic [WORK_ADDR_WIDTH-1:0] array_command_work_base_addr;
    logic [3:0] array_command_destination_tile_base;
    logic [WEIGHT_ADDR_WIDTH-1:0] array_command_weight_addr0;
    logic [WEIGHT_ADDR_WIDTH-1:0] array_command_weight_addr1;
    logic [WEIGHT_ADDR_WIDTH-1:0] array_command_weight_addr2;
    logic [WEIGHT_ADDR_WIDTH-1:0] array_command_weight_addr3;
    logic [15:0] array_command_leak;
    logic [31:0] array_command_threshold;
    logic array_command_reset_to_zero;
    logic [15:0] array_command_trace_decay;
    logic [51:0] array_command_transaction_tag;
    logic array_busy;
    logic array_done;
    logic array_idle;
    logic array_error;
    logic [3:0] maintenance_weight_fill_valid;
    logic [3:0] maintenance_weight_fill_ready;
    logic scnn_context_refill_safe;
    logic scnn_context_state_safe;
    logic maintenance_state_write_valid;
    logic maintenance_state_write_ready;
    logic maintenance_state_read_valid;
    logic maintenance_state_read_ready;
    logic maintenance_state_result_valid;
    logic maintenance_state_result_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] maintenance_state_result_addr;
    logic [319:0] maintenance_state_read_data;
    logic maintenance_trace_write_valid;
    logic maintenance_trace_write_ready;
    logic maintenance_trace_read_valid;
    logic maintenance_trace_read_ready;
    logic maintenance_trace_result_valid;
    logic maintenance_trace_result_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] maintenance_trace_result_addr;
    logic [319:0] maintenance_trace_read_data;
    logic maintenance_drained;
    logic [3:0] array_observation_valid;
    logic [3:0] array_observation_ready;
    logic [15:0] array_observation_tile_flat;
    logic [7:0] array_observation_mode_flat;
    logic [39:0] array_observation_lane_mask_flat;
    logic [39:0] array_observation_range_mask_flat;
    logic [39:0] array_observation_spike_flat;
    logic [1279:0] array_observation_pre_state_flat;
    logic [1279:0] array_observation_trace_next_flat;
    logic [255:0] array_observation_meta_flat;
    logic [3:0] scnn_array_observation_ready;
    logic [3:0] fc_srnn_array_observation_ready;
    logic fc_srnn_observation_busy;
    logic fc_srnn_observation_error;
    logic fc_srnn_direct_sink_idle;

    generate if (ENABLE_FC_SRNN_TP_OBSERVATION_STREAM) begin :
            g_fc_srnn_observation_stream
        unified_fc_srnn_h40_tp_observation_serializer_v1
            u_fc_srnn_observation_serializer (
            .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
            .i_clear_error(i_clear_error),
            .i_observation_valid(select_direct ?
                array_observation_valid : 4'b0000),
            .o_observation_ready(fc_srnn_array_observation_ready),
            .i_observation_tile_flat(array_observation_tile_flat),
            .i_observation_mode_flat(array_observation_mode_flat),
            .i_observation_lane_mask_flat(
                array_observation_lane_mask_flat),
            .i_observation_range_mask_flat(
                array_observation_range_mask_flat),
            .i_observation_spike_flat(array_observation_spike_flat),
            .i_observation_pre_state_flat(
                array_observation_pre_state_flat),
            .i_observation_trace_next_flat(
                array_observation_trace_next_flat),
            .i_observation_meta_flat(array_observation_meta_flat),
            .o_scalar_valid(o_fc_srnn_observation_valid),
            .i_scalar_ready(i_fc_srnn_observation_ready),
            .o_scalar_mode(o_fc_srnn_observation_mode),
            .o_scalar_transaction_tag(
                o_fc_srnn_observation_transaction_tag),
            .o_scalar_neuron_id(o_fc_srnn_observation_neuron_id),
            .o_scalar_spike(o_fc_srnn_observation_spike),
            .o_scalar_pre_state(o_fc_srnn_observation_pre_state),
            .o_scalar_trace_next(o_fc_srnn_observation_trace_next),
            .o_scalar_last_in_record(
                o_fc_srnn_observation_last_in_record),
            .o_busy(fc_srnn_observation_busy),
            .o_sticky_error(fc_srnn_observation_error)
        );
        assign o_fc_srnn_observation_busy = fc_srnn_observation_busy;
        assign o_fc_srnn_observation_sticky_error =
            fc_srnn_observation_error;
        assign fc_srnn_direct_sink_idle = !fc_srnn_observation_busy &&
            i_fc_srnn_observation_sink_idle;
    end else begin : g_no_fc_srnn_observation_stream
        assign fc_srnn_array_observation_ready =
            i_direct_observation_ready;
        assign fc_srnn_observation_busy = 1'b0;
        assign fc_srnn_observation_error = 1'b0;
        assign o_fc_srnn_observation_valid = 1'b0;
        assign o_fc_srnn_observation_mode = '0;
        assign o_fc_srnn_observation_transaction_tag = '0;
        assign o_fc_srnn_observation_neuron_id = '0;
        assign o_fc_srnn_observation_spike = 1'b0;
        assign o_fc_srnn_observation_pre_state = '0;
        assign o_fc_srnn_observation_trace_next = '0;
        assign o_fc_srnn_observation_last_in_record = 1'b0;
        assign o_fc_srnn_observation_busy = 1'b0;
        assign o_fc_srnn_observation_sticky_error = 1'b0;
        assign fc_srnn_direct_sink_idle = 1'b1;
    end endgenerate

    generate if (ENABLE_FC_SRNN_LAYER_CLIENT) begin : g_fc_srnn_layer
        assign o_direct_command_ready = 1'b0;

        unified_fc_srnn_h40_p4_layer_client_v1 #(
            .WEIGHT_DEPTH(WEIGHT_DEPTH),
            .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
            .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH)
        ) u_fc_srnn_client (
            .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
            .i_clear_error(i_clear_error),
            .i_streamed_feedforward(i_fc_srnn_streamed_feedforward),
            .i_group_feedforward_end(i_fc_srnn_group_feedforward_end),
            .i_layer_valid(i_fc_srnn_layer_valid),
            .o_layer_ready(o_fc_srnn_layer_ready),
            .o_owner_request_valid(fc_srnn_owner_request_valid),
            .i_owner_request_ready(fc_srnn_owner_request_ready),
            .i_mode(i_fc_srnn_mode),
            .i_fc_weight_int8(i_fc_srnn_weight_int8),
            .i_fc_current_shift6(i_fc_srnn_current_shift6),
            .i_neuron_count(i_fc_srnn_neuron_count),
            .i_feedforward_group_count(
                i_fc_srnn_feedforward_group_count),
            .i_recurrent_group_count(i_fc_srnn_recurrent_group_count),
            .i_recurrent_enable(i_fc_srnn_recurrent_enable),
            .i_leak(i_fc_srnn_leak),
            .i_threshold(i_fc_srnn_threshold),
            .i_reset_to_zero(i_fc_srnn_reset_to_zero),
            .i_trace_decay(i_fc_srnn_trace_decay),
            .i_transaction_tag(i_fc_srnn_transaction_tag),
            .i_work_base_addr(i_fc_srnn_work_base_addr),
            .i_destination_tile_base(i_fc_srnn_destination_tile_base),
            .i_group_valid(i_fc_srnn_group_valid),
            .o_group_ready(o_fc_srnn_group_ready),
            .i_group_recurrent(i_fc_srnn_group_recurrent),
            .i_group_source_valid(i_fc_srnn_group_source_valid),
            .i_group_weight_addr0(i_fc_srnn_group_weight_addr0),
            .i_group_weight_addr1(i_fc_srnn_group_weight_addr1),
            .i_group_weight_addr2(i_fc_srnn_group_weight_addr2),
            .i_group_weight_addr3(i_fc_srnn_group_weight_addr3),
            .o_array_command_valid(direct_command_valid),
            .i_array_command_ready(direct_command_ready),
            .o_array_command_mode(direct_mode),
            .o_array_command_fc_weight_int8(direct_fc_weight_int8),
            .o_array_command_fc_current_shift6(direct_fc_current_shift6),
            .o_array_command_neuron_count(direct_neuron_count),
            .o_array_command_first(direct_first),
            .o_array_command_last(direct_last),
            .o_array_command_source_valid(direct_source_valid),
            .o_array_command_weight_addr0(direct_weight_addr0),
            .o_array_command_weight_addr1(direct_weight_addr1),
            .o_array_command_weight_addr2(direct_weight_addr2),
            .o_array_command_weight_addr3(direct_weight_addr3),
            .o_array_command_leak(direct_leak),
            .o_array_command_threshold(direct_threshold),
            .o_array_command_reset_to_zero(direct_reset_to_zero),
            .o_array_command_trace_decay(direct_trace_decay),
            .o_array_command_transaction_tag(direct_transaction_tag),
            .o_array_command_work_base_addr(direct_work_base_addr),
            .o_array_command_destination_tile_base(
                direct_destination_tile_base),
            .i_array_transaction_done(direct_transaction_done),
            .i_array_sticky_error(array_error),
            .o_layer_done(o_fc_srnn_layer_done),
            .o_busy(fc_srnn_client_busy),
            .o_idle(fc_srnn_client_idle),
            .o_sticky_error(fc_srnn_client_error),
            .o_error_code(o_fc_srnn_error_code)
        );
        assign o_fc_srnn_busy = fc_srnn_client_busy;
        assign o_fc_srnn_idle = fc_srnn_client_idle;
        assign o_fc_srnn_sticky_error = fc_srnn_client_error;
    end else begin : g_direct_debug
        assign direct_command_valid = i_direct_command_valid;
        assign direct_mode = i_direct_mode;
        assign direct_fc_weight_int8 = 1'b0;
        assign direct_fc_current_shift6 = 1'b0;
        assign direct_neuron_count = i_direct_neuron_count;
        assign direct_first = i_direct_first;
        assign direct_last = i_direct_last;
        assign direct_source_valid = i_direct_source_valid;
        assign direct_weight_addr0 = i_direct_weight_addr0;
        assign direct_weight_addr1 = i_direct_weight_addr1;
        assign direct_weight_addr2 = i_direct_weight_addr2;
        assign direct_weight_addr3 = i_direct_weight_addr3;
        assign direct_leak = i_direct_leak;
        assign direct_threshold = i_direct_threshold;
        assign direct_reset_to_zero = i_direct_reset_to_zero;
        assign direct_trace_decay = i_direct_trace_decay;
        assign direct_transaction_tag = i_direct_transaction_tag;
        assign direct_work_base_addr = i_direct_work_base_addr;
        assign direct_destination_tile_base =
            i_direct_destination_tile_base;
        assign o_direct_command_ready = direct_command_ready;
        assign o_fc_srnn_layer_ready = 1'b0;
        assign o_fc_srnn_group_ready = 1'b0;
        assign o_fc_srnn_layer_done = 1'b0;
        assign o_fc_srnn_busy = 1'b0;
        assign o_fc_srnn_idle = 1'b1;
        assign o_fc_srnn_sticky_error = 1'b0;
        assign o_fc_srnn_error_code = 8'h00;
        assign fc_srnn_client_busy = 1'b0;
        assign fc_srnn_client_idle = 1'b1;
        assign fc_srnn_client_error = 1'b0;
        assign fc_srnn_owner_request_valid = 1'b0;
    end endgenerate

    always_comb begin
        array_command_valid = 1'b0;
        array_command_mode = direct_mode;
        array_command_fc_weight_int8 = direct_fc_weight_int8;
        array_command_fc_current_shift6 = direct_fc_current_shift6;
        array_command_neuron_count = direct_neuron_count;
        array_command_first = direct_first;
        array_command_last = direct_last;
        array_command_source_valid = direct_source_valid;
        array_command_spatial_count = '0;
        array_command_output_channel_base = '0;
        array_command_feature_mask0 = '0;
        array_command_feature_mask1 = '0;
        array_command_feature_mask2 = '0;
        array_command_feature_mask3 = '0;
        array_command_factor_b = 18'sd256;
        array_command_active_pair = 1'b0;
        array_command_weight_subword0 = '0;
        array_command_weight_subword1 = '0;
        array_command_weight_subword2 = '0;
        array_command_weight_subword3 = '0;
        array_command_work_base_addr = direct_work_base_addr;
        array_command_destination_tile_base = direct_destination_tile_base;
        array_command_weight_addr0 = direct_weight_addr0;
        array_command_weight_addr1 = direct_weight_addr1;
        array_command_weight_addr2 = direct_weight_addr2;
        array_command_weight_addr3 = direct_weight_addr3;
        array_command_leak = direct_leak;
        array_command_threshold = direct_threshold;
        array_command_reset_to_zero = direct_reset_to_zero;
        array_command_trace_decay = direct_trace_decay;
        array_command_transaction_tag = direct_transaction_tag;

        if (select_direct) begin
            array_command_valid = direct_command_valid;
        end else if (select_scnn) begin
            array_command_valid = scnn_command_valid;
            array_command_mode = 2'd2;
            array_command_fc_weight_int8 = 1'b0;
            array_command_fc_current_shift6 = 1'b0;
            array_command_neuron_count = scnn_command_neuron_count;
            array_command_first = scnn_command_first;
            array_command_last = scnn_command_last;
            array_command_source_valid = scnn_command_source_valid;
            array_command_spatial_count = scnn_command_spatial_count;
            array_command_output_channel_base =
                scnn_command_output_channel_base;
            array_command_feature_mask0 = scnn_command_feature_mask0;
            array_command_feature_mask1 = scnn_command_feature_mask1;
            array_command_feature_mask2 = scnn_command_feature_mask2;
            array_command_feature_mask3 = scnn_command_feature_mask3;
            array_command_factor_b = scnn_command_factor_b;
            array_command_active_pair = scnn_command_active_pair;
            array_command_weight_subword0 = scnn_command_weight_subword0;
            array_command_weight_subword1 = scnn_command_weight_subword1;
            array_command_weight_subword2 = scnn_command_weight_subword2;
            array_command_weight_subword3 = scnn_command_weight_subword3;
            array_command_work_base_addr = scnn_command_work_base_addr;
            array_command_destination_tile_base = 4'd0;
            array_command_weight_addr0 = scnn_command_weight_addr0;
            array_command_weight_addr1 = scnn_command_weight_addr1;
            array_command_weight_addr2 = scnn_command_weight_addr2;
            array_command_weight_addr3 = scnn_command_weight_addr3;
            array_command_leak = scnn_command_leak;
            array_command_threshold = scnn_command_threshold;
            array_command_reset_to_zero = scnn_command_reset_to_zero;
            array_command_trace_decay = scnn_command_trace_decay;
            array_command_transaction_tag = scnn_command_transaction_tag;
        end

        scnn_command_ready = ENABLE_SCNN_CLIENT && select_scnn &&
            array_command_ready;
        array_observation_ready = select_scnn ?
            scnn_array_observation_ready :
            (select_direct ? fc_srnn_array_observation_ready : 4'b0000);
    end

    assign o_direct_command_done = array_done && select_direct;
    assign o_direct_observation_valid = select_direct ?
        array_observation_valid : 4'b0000;
    assign o_direct_observation_tile_flat = array_observation_tile_flat;
    assign o_direct_observation_mode_flat = array_observation_mode_flat;
    assign o_direct_observation_lane_mask_flat =
        array_observation_lane_mask_flat;
    assign o_direct_observation_range_mask_flat =
        array_observation_range_mask_flat;
    assign o_direct_observation_spike_flat = array_observation_spike_flat;
    assign o_direct_observation_pre_state_flat =
        array_observation_pre_state_flat;
    assign o_direct_observation_trace_next_flat =
        array_observation_trace_next_flat;
    assign o_direct_observation_meta_flat = array_observation_meta_flat;
    assign select_scnn = ENABLE_SCNN_CLIENT ? owner_select_scnn : 1'b0;
    assign scnn_client_start_valid = ENABLE_SCNN_CLIENT ?
        owner_scnn_client_start_valid : 1'b0;
    assign o_scnn_start_ready = ENABLE_SCNN_CLIENT ?
        owner_scnn_start_ready : 1'b0;
    assign o_scnn_done = ENABLE_SCNN_CLIENT ? owner_scnn_done : 1'b0;
    assign o_busy = owner_busy || array_busy || scnn_client_busy ||
        fc_srnn_client_busy || fc_srnn_observation_busy;
    assign o_idle = owner_idle && array_idle && !scnn_client_busy &&
        fc_srnn_client_idle && !fc_srnn_observation_busy;
    assign o_sticky_error = owner_error || array_error ||
                            scnn_client_error || fc_srnn_client_error ||
                            fc_srnn_observation_error;
    assign o_direct_transaction_done = direct_transaction_done;

    // A Conv output-channel wave discovers its working set only after the SCNN
    // graph client owns the array. Permit refill while the client is stopped at
    // a Conv context barrier and every tile is idle. Weight writes are limited
    // to the load barrier; state/trace transfer is legal at load and release so
    // a logical context can be restored and retired without a second PE array.
    assign scnn_context_refill_safe = ENABLE_SCNN_CLIENT &&
        ENABLE_SCNN_CLASSIFIER_GRAPH && select_scnn &&
        o_scnn_context_load_valid &&
        o_scnn_context_active_pair != i_scnn_classifier_active_pair &&
        array_idle && !scnn_command_valid;
    assign o_scnn_context_refill_safe = scnn_context_refill_safe;
    // A state/trace read makes array_idle fall while the response traverses
    // tile, region and array pending stages.  Keep the context-side response
    // path open once either raw result is present; otherwise a request accepted
    // at the idle barrier can deadlock with its result hidden by that barrier.
    // New requests still require array_idle, so this does not permit overlap
    // with array commands or a second maintenance transaction.
    assign scnn_context_state_safe = ENABLE_SCNN_CLIENT &&
        ENABLE_SCNN_CLASSIFIER_GRAPH && select_scnn && !scnn_command_valid &&
        (array_idle || maintenance_state_result_valid ||
         maintenance_trace_result_valid) &&
        ((o_scnn_context_load_valid &&
          (ENABLE_SCNN_STATE_SERVICE ||
           o_scnn_context_active_pair != i_scnn_classifier_active_pair)) ||
         (o_scnn_context_release_valid &&
          (ENABLE_SCNN_STATE_SERVICE ||
           o_scnn_release_active_pair != i_scnn_classifier_active_pair)));
    assign maintenance_weight_fill_valid = i_weight_fill_valid &
        {4{select_maintenance || scnn_context_refill_safe}};
    assign o_weight_fill_ready = maintenance_weight_fill_ready &
        {4{select_maintenance || scnn_context_refill_safe}};
    assign maintenance_state_write_valid =
        (select_maintenance || scnn_context_state_safe) &&
        i_state_maintenance_write_valid;
    assign o_state_maintenance_write_ready =
        (select_maintenance || scnn_context_state_safe) &&
        maintenance_state_write_ready;
    assign maintenance_state_read_valid =
        (select_maintenance || scnn_context_state_safe) &&
        i_state_maintenance_read_valid;
    assign o_state_maintenance_read_ready =
        (select_maintenance || scnn_context_state_safe) &&
        maintenance_state_read_ready;
    assign o_state_maintenance_result_valid =
        (select_maintenance || scnn_context_state_safe) &&
        maintenance_state_result_valid;
    assign maintenance_state_result_ready =
        (select_maintenance || scnn_context_state_safe) &&
        i_state_maintenance_result_ready;
    assign o_state_maintenance_result_addr = maintenance_state_result_addr;
    assign o_state_maintenance_read_data = maintenance_state_read_data;
    assign maintenance_trace_write_valid =
        (select_maintenance || scnn_context_state_safe) &&
        i_trace_maintenance_write_valid;
    assign o_trace_maintenance_write_ready =
        (select_maintenance || scnn_context_state_safe) &&
        maintenance_trace_write_ready;
    assign maintenance_trace_read_valid =
        (select_maintenance || scnn_context_state_safe) &&
        i_trace_maintenance_read_valid;
    assign o_trace_maintenance_read_ready =
        (select_maintenance || scnn_context_state_safe) &&
        maintenance_trace_read_ready;
    assign o_trace_maintenance_result_valid =
        (select_maintenance || scnn_context_state_safe) &&
        maintenance_trace_result_valid;
    assign maintenance_trace_result_ready =
        (select_maintenance || scnn_context_state_safe) &&
        i_trace_maintenance_result_ready;
    assign o_trace_maintenance_result_addr = maintenance_trace_result_addr;
    assign o_trace_maintenance_read_data = maintenance_trace_read_data;
    assign maintenance_drained = array_idle &&
        !(|i_weight_fill_valid) &&
        !i_state_maintenance_write_valid &&
        !i_state_maintenance_read_valid &&
        !i_trace_maintenance_write_valid &&
        !i_trace_maintenance_read_valid &&
        !maintenance_state_result_valid &&
        !maintenance_trace_result_valid;

    unified_h40_array16_topology_owner_v1 u_owner (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_maintenance_start_valid(i_maintenance_start_valid),
        .o_maintenance_start_ready(o_maintenance_start_ready),
        .i_maintenance_release_valid(i_maintenance_release_valid),
        .o_maintenance_release_ready(o_maintenance_release_ready),
        .i_maintenance_idle(maintenance_drained),
        .i_direct_start_valid(fc_srnn_owner_request_valid),
        .o_direct_start_ready(fc_srnn_owner_request_ready),
        .i_direct_start_mode(i_fc_srnn_mode),
        .i_direct_command_valid(direct_command_valid),
        .o_direct_command_ready(direct_command_ready),
        .i_direct_mode(direct_mode), .i_direct_first(direct_first),
        .i_direct_last(direct_last),
        .i_scnn_start_valid(ENABLE_SCNN_CLIENT && i_scnn_start_valid),
        .o_scnn_start_ready(owner_scnn_start_ready),
        .o_scnn_client_start_valid(owner_scnn_client_start_valid),
        .i_scnn_client_start_ready(scnn_client_start_ready),
        .i_scnn_client_done(scnn_client_done),
        .i_scnn_observation_sink_idle(!ENABLE_SCNN_CLIENT ? 1'b1 :
            (ENABLE_SCNN_CLASSIFIER_GRAPH ? scnn_client_idle :
             i_scnn_observation_sink_idle)),
        .i_array_command_ready(array_command_ready),
        .i_array_command_done(array_done), .i_array_idle(array_idle),
        .i_direct_observation_sink_idle(fc_srnn_direct_sink_idle),
        .o_select_direct(select_direct),
        .o_select_scnn(owner_select_scnn),
        .o_select_maintenance(select_maintenance),
        .o_direct_done(direct_transaction_done),
        .o_scnn_done(owner_scnn_done),
        .o_maintenance_done(o_maintenance_done),
        .o_active_owner(o_active_owner),
        .o_busy(owner_busy), .o_idle(owner_idle),
        .o_sticky_error(owner_error)
    );

    generate if (!ENABLE_SCNN_CLIENT) begin : g_no_scnn_client
    assign scnn_client_start_ready = 1'b0;
    assign scnn_client_done = 1'b0;
    assign scnn_client_busy = 1'b0;
    assign scnn_client_idle = 1'b1;
    assign scnn_client_error = 1'b0;
    assign scnn_command_valid = 1'b0;
    assign scnn_command_neuron_count = '0;
    assign scnn_command_first = 1'b0;
    assign scnn_command_last = 1'b0;
    assign scnn_command_source_valid = '0;
    assign scnn_command_spatial_count = '0;
    assign scnn_command_output_channel_base = '0;
    assign scnn_command_feature_mask0 = '0;
    assign scnn_command_feature_mask1 = '0;
    assign scnn_command_feature_mask2 = '0;
    assign scnn_command_feature_mask3 = '0;
    assign scnn_command_factor_b = '0;
    assign scnn_command_active_pair = 1'b0;
    assign scnn_command_weight_subword0 = '0;
    assign scnn_command_weight_subword1 = '0;
    assign scnn_command_weight_subword2 = '0;
    assign scnn_command_weight_subword3 = '0;
    assign scnn_command_weight_addr0 = '0;
    assign scnn_command_weight_addr1 = '0;
    assign scnn_command_weight_addr2 = '0;
    assign scnn_command_weight_addr3 = '0;
    assign scnn_command_work_base_addr = '0;
    assign scnn_command_leak = '0;
    assign scnn_command_threshold = '0;
    assign scnn_command_reset_to_zero = 1'b0;
    assign scnn_command_trace_decay = '0;
    assign scnn_command_transaction_tag = '0;
    assign scnn_array_observation_ready = '0;

    assign o_scnn_context_load_valid = 1'b0;
    assign o_scnn_context_load_weights = 1'b0;
    assign o_scnn_context_output_channel_base = '0;
    assign o_scnn_context_output_channel_count = '0;
    assign o_scnn_context_spatial_base = '0;
    assign o_scnn_context_spatial_count = '0;
    assign o_scnn_context_active_pair = 1'b0;
    assign o_scnn_context_work_base_addr = '0;
    assign o_scnn_feature_request_valid = 1'b0;
    assign o_scnn_feature_term_valid = '0;
    assign o_scnn_feature_input_channel_flat = '0;
    assign o_scnn_feature_kernel_row_flat = '0;
    assign o_scnn_feature_kernel_col_flat = '0;
    assign o_scnn_feature_spatial_base = '0;
    assign o_scnn_feature_spatial_count = '0;
    assign o_scnn_feature_output_row = '0;
    assign o_scnn_feature_output_col = '0;
    assign o_scnn_feature_value_bit = '0;
    assign o_scnn_feature_response_ready = 1'b0;
    assign o_scnn_feature_update_valid = 1'b0;
    assign o_scnn_feature_update_slot = 1'b0;
    assign o_scnn_feature_update_addr = '0;
    assign o_scnn_feature_update_data = '0;
    assign o_scnn_feature_update_mask = '0;
    assign o_scnn_context_release_valid = 1'b0;
    assign o_scnn_release_output_channel_base = '0;
    assign o_scnn_release_output_channel_count = '0;
    assign o_scnn_release_spatial_base = '0;
    assign o_scnn_release_spatial_count = '0;
    assign o_scnn_release_active_pair = 1'b0;
    assign o_scnn_release_work_base_addr = '0;
    assign o_scnn_observation_valid = '0;
    assign o_scnn_observation_tile_flat = '0;
    assign o_scnn_observation_mode_flat = '0;
    assign o_scnn_observation_lane_mask_flat = '0;
    assign o_scnn_observation_range_mask_flat = '0;
    assign o_scnn_observation_spike_flat = '0;
    assign o_scnn_observation_pre_state_flat = '0;
    assign o_scnn_observation_trace_next_flat = '0;
    assign o_scnn_observation_meta_flat = '0;
    assign o_scnn_feature_trace_wave_valid = 1'b0;
    assign o_scnn_feature_trace_wave_mask = '0;
    assign o_scnn_feature_trace_wave_base = '0;
    assign o_scnn_feature_trace_wave_q8_8_flat = '0;
    assign o_scnn_feature_trace_wave_version = '0;
    assign o_scnn_feature_trace_wave_frame_index = '0;
    assign o_scnn_feature_trace_wave_first = 1'b0;
    assign o_scnn_feature_trace_wave_last = 1'b0;
    assign o_scnn_feature_trace_wave_sample_reset = 1'b0;
    assign o_scnn_feature_trace_wave_sample_last = 1'b0;
    assign o_scnn_feature_sample_label = '0;
    assign o_scnn_conv8_surrogate_valid = 1'b0;
    assign o_scnn_conv8_surrogate_target = 1'b0;
    assign o_scnn_conv8_surrogate_sample = '0;
    assign o_scnn_conv8_surrogate_timestep = '0;
    assign o_scnn_conv8_surrogate_version = '0;
    assign o_scnn_conv8_surrogate_layer_tag = '0;
    assign o_scnn_conv8_surrogate_channel = '0;
    assign o_scnn_conv8_surrogate_spatial = '0;
    assign o_scnn_conv8_membrane_pre_q8_8 = '0;
    assign o_scnn_conv8_surrogate_q8_8 = '0;
    assign o_scnn_logit_wave_valid = 1'b0;
    assign o_scnn_logit_wave_mask = '0;
    assign o_scnn_logit_wave_base = '0;
    assign o_scnn_logit_wave_count = '0;
    assign o_scnn_logit_wave_q8_8_flat = '0;
    assign o_scnn_logit_wave_version = '0;
    assign o_scnn_logit_wave_frame_index = '0;
    assign o_scnn_logit_wave_frame_last = 1'b0;
    assign o_scnn_logit_wave_sample_last = 1'b0;
    end else if (!ENABLE_SCNN_CLASSIFIER_GRAPH) begin : g_scnn_layer
    assign scnn_command_source_valid = 4'b0000;
    assign o_scnn_feature_trace_wave_valid = 1'b0;
    assign o_scnn_feature_trace_wave_mask = '0;
    assign o_scnn_feature_trace_wave_base = '0;
    assign o_scnn_feature_trace_wave_q8_8_flat = '0;
    assign o_scnn_feature_trace_wave_version = '0;
    assign o_scnn_feature_trace_wave_frame_index = '0;
    assign o_scnn_feature_trace_wave_first = 1'b0;
    assign o_scnn_feature_trace_wave_last = 1'b0;
    assign o_scnn_feature_trace_wave_sample_reset = 1'b0;
    assign o_scnn_feature_trace_wave_sample_last = 1'b0;
    assign o_scnn_feature_sample_label = '0;
    assign o_scnn_conv8_surrogate_valid = 1'b0;
    assign o_scnn_conv8_surrogate_target = 1'b0;
    assign o_scnn_conv8_surrogate_sample = '0;
    assign o_scnn_conv8_surrogate_timestep = '0;
    assign o_scnn_conv8_surrogate_version = '0;
    assign o_scnn_conv8_surrogate_layer_tag = '0;
    assign o_scnn_conv8_surrogate_channel = '0;
    assign o_scnn_conv8_surrogate_spatial = '0;
    assign o_scnn_conv8_membrane_pre_q8_8 = '0;
    assign o_scnn_conv8_surrogate_q8_8 = '0;
    assign o_scnn_logit_wave_valid = 1'b0;
    assign o_scnn_logit_wave_mask = '0;
    assign o_scnn_logit_wave_base = '0;
    assign o_scnn_logit_wave_count = '0;
    assign o_scnn_logit_wave_q8_8_flat = '0;
    assign o_scnn_logit_wave_version = '0;
    assign o_scnn_logit_wave_frame_index = '0;
    assign o_scnn_logit_wave_frame_last = 1'b0;
    assign o_scnn_logit_wave_sample_last = 1'b0;

    unified_scnn_h40_array16_layer_client_v1 #(
        .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH),
        .FEATURE_WORD_DEPTH(FEATURE_WORD_DEPTH),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .PHYSICAL_TILE_COUNT(PHYSICAL_TILE_COUNT)
    ) u_scnn_client (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_start_valid(scnn_client_start_valid),
        .o_start_ready(scnn_client_start_ready),
        .i_in_width(i_scnn_in_width), .i_in_height(i_scnn_in_height),
        .i_in_channels(i_scnn_in_channels),
        .i_out_width(i_scnn_out_width),
        .i_out_height(i_scnn_out_height),
        .i_out_channels(i_scnn_out_channels),
        .i_stride(i_scnn_stride), .i_padding(i_scnn_padding),
        .i_pool_mode(i_scnn_pool_mode),
        .i_initial_active_pair(i_scnn_initial_active_pair),
        .i_work_base_addr(i_scnn_work_base_addr),
        .i_leak(i_scnn_leak), .i_threshold(i_scnn_threshold),
        .i_reset_to_zero(i_scnn_reset_to_zero),
        .i_trace_decay(i_scnn_trace_decay),
        .i_layer_tag(i_scnn_layer_tag),
        .o_context_load_valid(o_scnn_context_load_valid),
        .i_context_load_ready(i_scnn_context_load_ready),
        .o_context_load_weights(o_scnn_context_load_weights),
        .o_context_output_channel_base(
            o_scnn_context_output_channel_base),
        .o_context_output_channel_count(
            o_scnn_context_output_channel_count),
        .o_context_spatial_base(o_scnn_context_spatial_base),
        .o_context_spatial_count(o_scnn_context_spatial_count),
        .o_context_active_pair(o_scnn_context_active_pair),
        .o_context_work_base_addr(o_scnn_context_work_base_addr),
        .o_feature_request_valid(o_scnn_feature_request_valid),
        .i_feature_request_ready(i_scnn_feature_request_ready),
        .o_feature_term_valid(o_scnn_feature_term_valid),
        .o_feature_input_channel_flat(
            o_scnn_feature_input_channel_flat),
        .o_feature_kernel_row_flat(o_scnn_feature_kernel_row_flat),
        .o_feature_kernel_col_flat(o_scnn_feature_kernel_col_flat),
        .o_feature_spatial_base(o_scnn_feature_spatial_base),
        .o_feature_spatial_count(o_scnn_feature_spatial_count),
        .o_feature_output_row(o_scnn_feature_output_row),
        .o_feature_output_col(o_scnn_feature_output_col),
        .o_feature_value_bit(o_scnn_feature_value_bit),
        .i_feature_response_valid(i_scnn_feature_response_valid),
        .o_feature_response_ready(o_scnn_feature_response_ready),
        .i_feature_mask0(i_scnn_feature_mask0),
        .i_feature_mask1(i_scnn_feature_mask1),
        .i_feature_mask2(i_scnn_feature_mask2),
        .i_feature_mask3(i_scnn_feature_mask3),
        .i_feature_service_error(i_scnn_feature_service_error),
        .i_feature_write_slot(i_scnn_feature_write_slot),
        .i_feature_write_base_addr(i_scnn_feature_write_base_addr),
        .i_feature_write_row_stride_words(
            i_scnn_feature_write_row_stride_words),
        .i_feature_write_channel_stride_words(
            i_scnn_feature_write_channel_stride_words),
        .o_feature_update_valid(o_scnn_feature_update_valid),
        .i_feature_update_ready(i_scnn_feature_update_ready),
        .o_feature_update_slot(o_scnn_feature_update_slot),
        .o_feature_update_addr(o_scnn_feature_update_addr),
        .o_feature_update_data(o_scnn_feature_update_data),
        .o_feature_update_mask(o_scnn_feature_update_mask),
        .i_feature_update_done(i_scnn_feature_update_done),
        .o_feature_commit_valid(),
        .i_feature_commit_ready(1'b1),
        .o_feature_commit_layer_tag(),
        .o_feature_commit_channel(),
        .o_feature_commit_spatial_base(),
        .o_feature_commit_spatial_count(),
        .o_feature_commit_spike(),
        .o_feature_commit_q8_8_flat(),
        .o_context_release_valid(o_scnn_context_release_valid),
        .i_context_release_ready(i_scnn_context_release_ready),
        .o_release_output_channel_base(
            o_scnn_release_output_channel_base),
        .o_release_output_channel_count(
            o_scnn_release_output_channel_count),
        .o_release_spatial_base(o_scnn_release_spatial_base),
        .o_release_spatial_count(o_scnn_release_spatial_count),
        .o_release_active_pair(o_scnn_release_active_pair),
        .o_release_work_base_addr(o_scnn_release_work_base_addr),
        .o_array_command_valid(scnn_command_valid),
        .i_array_command_ready(scnn_command_ready),
        .o_array_command_neuron_count(scnn_command_neuron_count),
        .o_array_command_first(scnn_command_first),
        .o_array_command_last(scnn_command_last),
        .o_array_command_spatial_count(scnn_command_spatial_count),
        .o_array_command_output_channel_base(
            scnn_command_output_channel_base),
        .o_array_command_feature_mask0(scnn_command_feature_mask0),
        .o_array_command_feature_mask1(scnn_command_feature_mask1),
        .o_array_command_feature_mask2(scnn_command_feature_mask2),
        .o_array_command_feature_mask3(scnn_command_feature_mask3),
        .o_array_command_factor_b(scnn_command_factor_b),
        .o_array_command_active_pair(scnn_command_active_pair),
        .o_array_command_weight_subword0(scnn_command_weight_subword0),
        .o_array_command_weight_subword1(scnn_command_weight_subword1),
        .o_array_command_weight_subword2(scnn_command_weight_subword2),
        .o_array_command_weight_subword3(scnn_command_weight_subword3),
        .o_array_command_weight_addr0(scnn_command_weight_addr0),
        .o_array_command_weight_addr1(scnn_command_weight_addr1),
        .o_array_command_weight_addr2(scnn_command_weight_addr2),
        .o_array_command_weight_addr3(scnn_command_weight_addr3),
        .o_array_command_work_base_addr(scnn_command_work_base_addr),
        .o_array_command_leak(scnn_command_leak),
        .o_array_command_threshold(scnn_command_threshold),
        .o_array_command_reset_to_zero(scnn_command_reset_to_zero),
        .o_array_command_trace_decay(scnn_command_trace_decay),
        .o_array_command_transaction_tag(scnn_command_transaction_tag),
        .i_array_command_done(array_done),
        .i_array_busy(array_busy), .i_array_idle(array_idle),
        .i_array_error(array_error),
        .i_array_observation_valid(array_observation_valid &
            {4{select_scnn}}),
        .o_array_observation_ready(scnn_array_observation_ready),
        .i_array_observation_tile_flat(array_observation_tile_flat),
        .i_array_observation_mode_flat(array_observation_mode_flat),
        .i_array_observation_lane_mask_flat(
            array_observation_lane_mask_flat),
        .i_array_observation_range_mask_flat(
            array_observation_range_mask_flat),
        .i_array_observation_spike_flat(array_observation_spike_flat),
        .i_array_observation_pre_state_flat(
            array_observation_pre_state_flat),
        .i_array_observation_trace_next_flat(
            array_observation_trace_next_flat),
        .i_array_observation_meta_flat(array_observation_meta_flat),
        .o_observation_valid(o_scnn_observation_valid),
        .i_observation_ready(i_scnn_observation_ready),
        .o_observation_tile_flat(o_scnn_observation_tile_flat),
        .o_observation_mode_flat(o_scnn_observation_mode_flat),
        .o_observation_lane_mask_flat(
            o_scnn_observation_lane_mask_flat),
        .o_observation_range_mask_flat(
            o_scnn_observation_range_mask_flat),
        .o_observation_spike_flat(o_scnn_observation_spike_flat),
        .o_observation_pre_state_flat(
            o_scnn_observation_pre_state_flat),
        .o_observation_trace_next_flat(
            o_scnn_observation_trace_next_flat),
        .o_observation_meta_flat(o_scnn_observation_meta_flat),
        .o_busy(scnn_client_busy), .o_done(scnn_client_done),
        .o_idle(scnn_client_idle), .o_sticky_error(scnn_client_error)
    );
    end else begin : g_scnn_graph
    assign o_scnn_observation_valid = 4'b0000;
    assign o_scnn_observation_tile_flat = '0;
    assign o_scnn_observation_mode_flat = '0;
    assign o_scnn_observation_lane_mask_flat = '0;
    assign o_scnn_observation_range_mask_flat = '0;
    assign o_scnn_observation_spike_flat = '0;
    assign o_scnn_observation_pre_state_flat = '0;
    assign o_scnn_observation_trace_next_flat = '0;
    assign o_scnn_observation_meta_flat = '0;

    unified_scnn_h40_array16_graph_client_v1 #(
        .LOGIT_CONTEXTS(SCNN_LOGIT_CONTEXTS),
        .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH),
        .FEATURE_WORD_DEPTH(FEATURE_WORD_DEPTH),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .PHYSICAL_TILE_COUNT(PHYSICAL_TILE_COUNT),
        .MAX_FEATURE_CHANNELS(MAX_SCNN_FEATURE_CHANNELS),
        .FEATURE_SPATIAL(SCNN_CLASSIFIER_FEATURE_SPATIAL),
        .MAX_CLASSES(MAX_SCNN_CLASSES),
        .FEATURE_LANES(SCNN_FEATURE_LANES),
        .LOGIT_ENGINES(SCNN_LOGIT_ENGINES),
        .ACCUMULATE_SAMPLE_LOGITS(ACCUMULATE_SCNN_SAMPLE_LOGITS),
        .SAMPLE_WIDTH(SCNN_SAMPLE_WIDTH),
        .TIMESTEP_WIDTH(SCNN_TIMESTEP_WIDTH),
        .CHANNEL_WIDTH(SCNN_CHANNEL_WIDTH),
        .CHANNEL_COUNT_WIDTH(SCNN_CHANNEL_COUNT_WIDTH),
        .FEATURE_WIDTH(SCNN_FEATURE_WIDTH),
        .CLASS_WIDTH(SCNN_CLASS_WIDTH),
        .CLASS_COUNT_WIDTH(SCNN_CLASS_COUNT_WIDTH)
    ) u_scnn_graph_client (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_start_valid(scnn_client_start_valid),
        .o_start_ready(scnn_client_start_ready),
        .i_conv_in_width(i_scnn_in_width),
        .i_conv_in_height(i_scnn_in_height),
        .i_conv_in_channels(i_scnn_in_channels),
        .i_conv_out_width(i_scnn_out_width),
        .i_conv_out_height(i_scnn_out_height),
        .i_conv_out_channels(i_scnn_out_channels),
        .i_conv_stride(i_scnn_stride),
        .i_conv_padding(i_scnn_padding),
        .i_conv_pool_mode(i_scnn_pool_mode),
        .i_conv_active_pair(i_scnn_initial_active_pair),
        .i_conv_work_base_addr(i_scnn_work_base_addr),
        .i_conv_leak(i_scnn_leak),
        .i_conv_threshold(i_scnn_threshold),
        .i_conv_reset_to_zero(i_scnn_reset_to_zero),
        .i_conv_trace_decay(i_scnn_trace_decay),
        .i_conv_layer_tag(i_scnn_layer_tag),
        .i_feature_channel_base(
            i_scnn_classifier_feature_channel_base),
        .i_feature_channel_count(
            i_scnn_classifier_feature_channel_count),
        .i_class_count(i_scnn_classifier_class_count),
        .i_classifier_active_pair(i_scnn_classifier_active_pair),
        .i_classifier_work_base_addr(
            i_scnn_classifier_work_base_addr),
        .i_classifier_trace_decay(i_scnn_classifier_trace_decay),
        .i_classifier_layer_tag(i_scnn_classifier_layer_tag),
        .i_version(i_scnn_version), .i_frame_index(i_scnn_frame_index),
        .i_sample_reset(i_scnn_sample_reset),
        .i_sample_last(i_scnn_sample_last),
        .i_sample_label(i_scnn_sample_label),
        .i_learning_enable(i_scnn_learning_enable),
        .i_conv8_observation_enable(i_scnn_conv8_observation_enable),
        .i_conv8_observation_target(i_scnn_conv8_observation_target),
        .i_sample_index(i_scnn_sample_index),
        .i_timestep_index(i_scnn_timestep_index),
        .o_context_load_valid(o_scnn_context_load_valid),
        .i_context_load_ready(i_scnn_context_load_ready),
        .o_context_load_weights(o_scnn_context_load_weights),
        .o_context_output_channel_base(
            o_scnn_context_output_channel_base),
        .o_context_output_channel_count(
            o_scnn_context_output_channel_count),
        .o_context_spatial_base(o_scnn_context_spatial_base),
        .o_context_spatial_count(o_scnn_context_spatial_count),
        .o_context_active_pair(o_scnn_context_active_pair),
        .o_context_work_base_addr(o_scnn_context_work_base_addr),
        .o_context_release_valid(o_scnn_context_release_valid),
        .i_context_release_ready(i_scnn_context_release_ready),
        .o_release_output_channel_base(
            o_scnn_release_output_channel_base),
        .o_release_output_channel_count(
            o_scnn_release_output_channel_count),
        .o_release_spatial_base(o_scnn_release_spatial_base),
        .o_release_spatial_count(o_scnn_release_spatial_count),
        .o_release_active_pair(o_scnn_release_active_pair),
        .o_release_work_base_addr(o_scnn_release_work_base_addr),
        .o_feature_request_valid(o_scnn_feature_request_valid),
        .i_feature_request_ready(i_scnn_feature_request_ready),
        .o_feature_term_valid(o_scnn_feature_term_valid),
        .o_feature_input_channel_flat(
            o_scnn_feature_input_channel_flat),
        .o_feature_kernel_row_flat(o_scnn_feature_kernel_row_flat),
        .o_feature_kernel_col_flat(o_scnn_feature_kernel_col_flat),
        .o_feature_spatial_base(o_scnn_feature_spatial_base),
        .o_feature_spatial_count(o_scnn_feature_spatial_count),
        .o_feature_output_row(o_scnn_feature_output_row),
        .o_feature_output_col(o_scnn_feature_output_col),
        .o_feature_value_bit(o_scnn_feature_value_bit),
        .i_feature_response_valid(i_scnn_feature_response_valid),
        .o_feature_response_ready(o_scnn_feature_response_ready),
        .i_feature_mask0(i_scnn_feature_mask0),
        .i_feature_mask1(i_scnn_feature_mask1),
        .i_feature_mask2(i_scnn_feature_mask2),
        .i_feature_mask3(i_scnn_feature_mask3),
        .i_feature_service_error(i_scnn_feature_service_error),
        .i_feature_write_slot(i_scnn_feature_write_slot),
        .i_feature_write_base_addr(i_scnn_feature_write_base_addr),
        .i_feature_write_row_stride_words(
            i_scnn_feature_write_row_stride_words),
        .i_feature_write_channel_stride_words(
            i_scnn_feature_write_channel_stride_words),
        .o_feature_update_valid(o_scnn_feature_update_valid),
        .i_feature_update_ready(i_scnn_feature_update_ready),
        .o_feature_update_slot(o_scnn_feature_update_slot),
        .o_feature_update_addr(o_scnn_feature_update_addr),
        .o_feature_update_data(o_scnn_feature_update_data),
        .o_feature_update_mask(o_scnn_feature_update_mask),
        .i_feature_update_done(i_scnn_feature_update_done),
        .o_feature_trace_wave_valid(o_scnn_feature_trace_wave_valid),
        .i_feature_trace_wave_ready(i_scnn_feature_trace_wave_ready),
        .o_feature_trace_wave_mask(o_scnn_feature_trace_wave_mask),
        .o_feature_trace_wave_base(o_scnn_feature_trace_wave_base),
        .o_feature_trace_wave_q8_8_flat(
            o_scnn_feature_trace_wave_q8_8_flat),
        .o_feature_trace_wave_version(
            o_scnn_feature_trace_wave_version),
        .o_feature_trace_wave_frame_index(
            o_scnn_feature_trace_wave_frame_index),
        .o_feature_trace_wave_first(o_scnn_feature_trace_wave_first),
        .o_feature_trace_wave_last(o_scnn_feature_trace_wave_last),
        .o_feature_trace_wave_sample_reset(
            o_scnn_feature_trace_wave_sample_reset),
        .o_feature_trace_wave_sample_last(
            o_scnn_feature_trace_wave_sample_last),
        .o_feature_sample_label(o_scnn_feature_sample_label),
        .o_conv8_surrogate_valid(o_scnn_conv8_surrogate_valid),
        .i_conv8_surrogate_ready(i_scnn_conv8_surrogate_ready),
        .o_conv8_surrogate_target(o_scnn_conv8_surrogate_target),
        .o_conv8_surrogate_sample(o_scnn_conv8_surrogate_sample),
        .o_conv8_surrogate_timestep(o_scnn_conv8_surrogate_timestep),
        .o_conv8_surrogate_version(o_scnn_conv8_surrogate_version),
        .o_conv8_surrogate_layer_tag(o_scnn_conv8_surrogate_layer_tag),
        .o_conv8_surrogate_channel(o_scnn_conv8_surrogate_channel),
        .o_conv8_surrogate_spatial(o_scnn_conv8_surrogate_spatial),
        .o_conv8_membrane_pre_q8_8(o_scnn_conv8_membrane_pre_q8_8),
        .o_conv8_surrogate_q8_8(o_scnn_conv8_surrogate_q8_8),
        .o_logit_wave_valid(o_scnn_logit_wave_valid),
        .i_logit_wave_ready(i_scnn_logit_wave_ready),
        .o_logit_wave_mask(o_scnn_logit_wave_mask),
        .o_logit_wave_base(o_scnn_logit_wave_base),
        .o_logit_wave_count(o_scnn_logit_wave_count),
        .o_logit_wave_q8_8_flat(o_scnn_logit_wave_q8_8_flat),
        .o_logit_wave_version(o_scnn_logit_wave_version),
        .o_logit_wave_frame_index(o_scnn_logit_wave_frame_index),
        .o_logit_wave_frame_last(o_scnn_logit_wave_frame_last),
        .o_logit_wave_sample_last(o_scnn_logit_wave_sample_last),
        .o_array_command_valid(scnn_command_valid),
        .i_array_command_ready(scnn_command_ready),
        .o_array_command_neuron_count(scnn_command_neuron_count),
        .o_array_command_first(scnn_command_first),
        .o_array_command_last(scnn_command_last),
        .o_array_command_source_valid(scnn_command_source_valid),
        .o_array_command_spatial_count(scnn_command_spatial_count),
        .o_array_command_output_channel_base(
            scnn_command_output_channel_base),
        .o_array_command_feature_mask0(scnn_command_feature_mask0),
        .o_array_command_feature_mask1(scnn_command_feature_mask1),
        .o_array_command_feature_mask2(scnn_command_feature_mask2),
        .o_array_command_feature_mask3(scnn_command_feature_mask3),
        .o_array_command_factor_b(scnn_command_factor_b),
        .o_array_command_active_pair(scnn_command_active_pair),
        .o_array_command_weight_subword0(scnn_command_weight_subword0),
        .o_array_command_weight_subword1(scnn_command_weight_subword1),
        .o_array_command_weight_subword2(scnn_command_weight_subword2),
        .o_array_command_weight_subword3(scnn_command_weight_subword3),
        .o_array_command_weight_addr0(scnn_command_weight_addr0),
        .o_array_command_weight_addr1(scnn_command_weight_addr1),
        .o_array_command_weight_addr2(scnn_command_weight_addr2),
        .o_array_command_weight_addr3(scnn_command_weight_addr3),
        .o_array_command_work_base_addr(scnn_command_work_base_addr),
        .o_array_command_leak(scnn_command_leak),
        .o_array_command_threshold(scnn_command_threshold),
        .o_array_command_reset_to_zero(scnn_command_reset_to_zero),
        .o_array_command_trace_decay(scnn_command_trace_decay),
        .o_array_command_transaction_tag(scnn_command_transaction_tag),
        .i_array_command_done(array_done), .i_array_busy(array_busy),
        .i_array_idle(array_idle), .i_array_error(array_error),
        .i_array_observation_valid(array_observation_valid &
            {4{select_scnn}}),
        .o_array_observation_ready(scnn_array_observation_ready),
        .i_array_observation_tile_flat(array_observation_tile_flat),
        .i_array_observation_mode_flat(array_observation_mode_flat),
        .i_array_observation_lane_mask_flat(
            array_observation_lane_mask_flat),
        .i_array_observation_range_mask_flat(
            array_observation_range_mask_flat),
        .i_array_observation_spike_flat(array_observation_spike_flat),
        .i_array_observation_pre_state_flat(
            array_observation_pre_state_flat),
        .i_array_observation_trace_next_flat(
            array_observation_trace_next_flat),
        .i_array_observation_meta_flat(array_observation_meta_flat),
        .o_done(scnn_client_done), .o_busy(scnn_client_busy),
        .o_idle(scnn_client_idle), .o_sticky_error(scnn_client_error)
    );
    end endgenerate

    unified_binary_p4_h40_array12_path_v1 #(
        .M2_CACHE_DEPTH(M2_CACHE_DEPTH),
        .SCNN_WEIGHT_FRACTIONAL_BITS(SCNN_WEIGHT_FRACTIONAL_BITS),
        .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .MEMORY_DEPTH(MEMORY_DEPTH),
        .MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH),
        .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH),
        .CUT_RECORD_READY(CUT_RECORD_READY)
    ) u_array (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_command_valid(array_command_valid),
        .o_command_ready(array_command_ready),
        .i_mode(array_command_mode),
        .i_fc_weight_int8(array_command_fc_weight_int8),
        .i_fc_current_shift6(array_command_fc_current_shift6),
        .i_neuron_count(array_command_neuron_count),
        .i_first(array_command_first), .i_last(array_command_last),
        .i_source_valid(array_command_source_valid),
        .i_scnn_spatial_count(array_command_spatial_count),
        .i_scnn_output_channel_base(array_command_output_channel_base),
        .i_scnn_feature_mask0(array_command_feature_mask0),
        .i_scnn_feature_mask1(array_command_feature_mask1),
        .i_scnn_feature_mask2(array_command_feature_mask2),
        .i_scnn_feature_mask3(array_command_feature_mask3),
        .i_scnn_factor_b(array_command_factor_b),
        .i_scnn_active_pair(array_command_active_pair),
        .i_scnn_weight_subword0(array_command_weight_subword0),
        .i_scnn_weight_subword1(array_command_weight_subword1),
        .i_scnn_weight_subword2(array_command_weight_subword2),
        .i_scnn_weight_subword3(array_command_weight_subword3),
        .i_scnn_work_base_addr(array_command_work_base_addr),
        .i_destination_tile_base(array_command_destination_tile_base),
        .i_weight_addr0(array_command_weight_addr0),
        .i_weight_addr1(array_command_weight_addr1),
        .i_weight_addr2(array_command_weight_addr2),
        .i_weight_addr3(array_command_weight_addr3),
        .i_leak(array_command_leak),
        .i_threshold(array_command_threshold),
        .i_reset_to_zero(array_command_reset_to_zero),
        .i_trace_decay(array_command_trace_decay),
        .i_transaction_tag(array_command_transaction_tag),
        .i_weight_fill_context(i_weight_fill_context),
        .i_weight_fill_valid(maintenance_weight_fill_valid),
        .o_weight_fill_ready(maintenance_weight_fill_ready),
        .i_weight_fill_tile_flat(i_weight_fill_tile_flat),
        .i_weight_fill_bank(i_weight_fill_bank),
        .i_weight_fill_addr_flat(i_weight_fill_addr_flat),
        .i_weight_fill_data_flat(i_weight_fill_data_flat),
        .o_observation_valid(array_observation_valid),
        .i_observation_ready(array_observation_ready),
        .o_observation_tile_flat(array_observation_tile_flat),
        .o_observation_mode_flat(array_observation_mode_flat),
        .o_observation_lane_mask_flat(array_observation_lane_mask_flat),
        .o_observation_range_mask_flat(array_observation_range_mask_flat),
        .o_observation_spike_flat(array_observation_spike_flat),
        .o_observation_pre_state_flat(array_observation_pre_state_flat),
        .o_observation_trace_next_flat(
            array_observation_trace_next_flat),
        .o_observation_meta_flat(array_observation_meta_flat),
        .i_maintenance_tile(i_maintenance_tile),
        .i_state_maintenance_write_valid(maintenance_state_write_valid),
        .o_state_maintenance_write_ready(maintenance_state_write_ready),
        .i_state_maintenance_write_addr(i_state_maintenance_write_addr),
        .i_state_maintenance_write_data(i_state_maintenance_write_data),
        .i_state_maintenance_write_mask(i_state_maintenance_write_mask),
        .i_state_maintenance_read_valid(maintenance_state_read_valid),
        .o_state_maintenance_read_ready(maintenance_state_read_ready),
        .i_state_maintenance_read_addr(i_state_maintenance_read_addr),
        .o_state_maintenance_result_valid(maintenance_state_result_valid),
        .i_state_maintenance_result_ready(maintenance_state_result_ready),
        .o_state_maintenance_result_addr(maintenance_state_result_addr),
        .o_state_maintenance_read_data(maintenance_state_read_data),
        .i_trace_maintenance_write_valid(maintenance_trace_write_valid),
        .o_trace_maintenance_write_ready(maintenance_trace_write_ready),
        .i_trace_maintenance_write_addr(i_trace_maintenance_write_addr),
        .i_trace_maintenance_write_data(i_trace_maintenance_write_data),
        .i_trace_maintenance_write_mask(i_trace_maintenance_write_mask),
        .i_trace_maintenance_read_valid(maintenance_trace_read_valid),
        .o_trace_maintenance_read_ready(maintenance_trace_read_ready),
        .i_trace_maintenance_read_addr(i_trace_maintenance_read_addr),
        .o_trace_maintenance_result_valid(maintenance_trace_result_valid),
        .i_trace_maintenance_result_ready(maintenance_trace_result_ready),
        .o_trace_maintenance_result_addr(maintenance_trace_result_addr),
        .o_trace_maintenance_read_data(maintenance_trace_read_data),
        .o_busy(array_busy), .o_done(array_done), .o_idle(array_idle),
        .o_tile_done(o_tile_done), .o_tile_idle(o_tile_idle),
        .o_tile_sticky_error(o_tile_sticky_error),
        .o_sticky_error(array_error)
    );
endmodule

`default_nettype wire
