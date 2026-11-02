`timescale 1ns/1ps
`default_nettype none

// SCNN selected-layer train/infer system on the single shared 12xH40/P4
// substrate. Conv and classifier forward work, state and traces stay in the
// array; TP learning owns only observations, persistent parameters and cache
// refill control. No duplicate inference datapath is instantiated here.
module unified_scnn_h40_array16_selected_layers_train_infer_system_v1 #(
    parameter int M2_CACHE_DEPTH = 578,
    parameter int SCNN_WEIGHT_FRACTIONAL_BITS = 8,
    parameter bit ACCUMULATE_SCNN_SAMPLE_LOGITS = 1'b0,
    parameter bit ENABLE_SCNN_FRAME_SERVICE = 1'b0,
    parameter bit ENABLE_FC_CLIENT = 1'b1,
    parameter bit ENABLE_SRNN_CLIENT = 1'b1,
    parameter bit ENABLE_SCNN_CLIENT = 1'b1,
    parameter bit USE_INTERNAL_SCNN_PARAMETER_STORE = 1'b1,
    parameter bit ENABLE_SRNN_M2_H40_CLIENT = 1'b0,
    parameter int SRNN_M2_PARAM_DEPTH = 120320,
    parameter int SRNN_M2_BATCH_MAX = 64,
    parameter int SRNN_M2_TIMESTEP_MAX = 128,
    parameter int SRNN_M2_BACKBONE_MAX = 450,
    parameter int SRNN_M2_HIDDEN_MAX = 256,
    parameter int SRNN_M2_CLASS_MAX = 20,
    parameter int SRNN_BACKBONE_PARAM_DEPTH = 524288,
    parameter int SRNN_BACKBONE_PARAM_ADDR_WIDTH =
        $clog2(SRNN_BACKBONE_PARAM_DEPTH),
    parameter bit ENABLE_SRNN_GROUP_STREAMER = 1'b0,
    parameter int SRNN_FEEDFORWARD_SOURCE_MAX = 700,
    parameter int SRNN_RECURRENT_SOURCE_MAX = 480,
    parameter int WEIGHT_DEPTH = 1156,
    parameter int WEIGHT_ADDR_WIDTH = $clog2(WEIGHT_DEPTH),
    parameter int MEMORY_DEPTH = 1024,
    parameter int MEMORY_ADDR_WIDTH = $clog2(MEMORY_DEPTH),
    parameter int WORK_ADDR_WIDTH = $clog2(MEMORY_DEPTH / 2),
    parameter int FEATURE_WORD_DEPTH = 8192,
    parameter int FEATURE_ADDR_WIDTH = $clog2(FEATURE_WORD_DEPTH),
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_DEPTH = 32768,
    parameter int PARAM_ADDR_WIDTH = $clog2(PARAM_DEPTH),
    parameter int TAG_WIDTH = 16,
    parameter int MAX_BATCH = 10,
    parameter int MAX_TIMESTEPS = 20,
    // The shared learner accepts B*T flattened samples, while the SCNN
    // capture RAM only needs the locked B4xT4 adaptation window.
    parameter int CLASSIFIER_CAPTURE_MAX_BATCH =
        MAX_BATCH < 4 ? MAX_BATCH : 4,
    parameter int CLASSIFIER_CAPTURE_MAX_TIMESTEPS =
        MAX_TIMESTEPS < 4 ? MAX_TIMESTEPS : 4,
    parameter int MAX_INPUT_CHANNELS = 8,
    parameter int MAX_OUTPUT_CHANNELS = 12,
    parameter int OBS_INPUT_CHANNELS = 512,
    parameter int OBS_OUTPUT_CHANNELS = 512,
    parameter int SPATIAL = 4,
    parameter int POOLED_SPATIAL = 2,
    parameter int KERNEL_SIZE = 3,
    parameter int Q16_ADDR_WIDTH = 24,
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int VERSION_WIDTH = 16,
    parameter int MAX_FEATURES = 2048,
    parameter int MAX_CLASSES = 11,
    parameter int LANES = 8,
    parameter int ENGINES = 12,
    parameter int FRAME_WIDTH = 8,
    parameter int SELECTED_LAYER = 7,
    parameter bit USE_EXTERNAL_OUTPUT_LEARNER = 1'b0,
    parameter bit ENABLE_SCNN_CONV8_LEARNING = 1'b1,
    parameter int INTERNAL_MODULATION_BATCH = 2,
    parameter int SAMPLE_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int TIMESTEP_WIDTH = MAX_TIMESTEPS <= 2 ? 1 :
        $clog2(MAX_TIMESTEPS),
    parameter int TIMESTEP_COUNT_WIDTH = $clog2(MAX_TIMESTEPS + 1),
    parameter int CHANNEL_WIDTH = OBS_INPUT_CHANNELS <= 2 ? 1 :
        $clog2(OBS_INPUT_CHANNELS),
    parameter int CHANNEL_COUNT_WIDTH = $clog2(OBS_INPUT_CHANNELS + 1),
    parameter int FEATURE_WIDTH = MAX_FEATURES <= 2 ? 1 :
        $clog2(MAX_FEATURES),
    parameter int FEATURE_COUNT_WIDTH = $clog2(MAX_FEATURES + 1),
    parameter int CLASS_WIDTH = MAX_CLASSES <= 2 ? 1 : $clog2(MAX_CLASSES),
    parameter int CLASS_COUNT_WIDTH = $clog2(MAX_CLASSES + 1)
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_clear_error,

    input  wire logic                         i_parameter_load_valid,
    output logic                              o_parameter_load_ready,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]  i_parameter_load_addr,
    input  wire logic signed [15:0]           i_parameter_load_code,
    input  wire logic signed [15:0]           i_parameter_load_residual,
    output logic                              o_scnn_store_load_valid,
    input  wire logic                         i_scnn_store_load_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]       o_scnn_store_load_addr,
    output logic signed [15:0]                o_scnn_store_load_code,
    output logic signed [15:0]                o_scnn_store_load_residual,
    output logic                              o_scnn_store_learning_read_valid,
    input  wire logic                         i_scnn_store_learning_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]       o_scnn_store_learning_read_addr,
    output logic [TAG_WIDTH-1:0]              o_scnn_store_learning_read_tag,
    input  wire logic                         i_scnn_store_learning_response_valid,
    output logic                              o_scnn_store_learning_response_ready,
    input  wire logic [TAG_WIDTH-1:0]         i_scnn_store_learning_response_tag,
    input  wire logic signed [15:0]           i_scnn_store_learning_response_code,
    input  wire logic signed [15:0]           i_scnn_store_learning_response_residual,
    input  wire logic                         i_scnn_store_learning_response_error,
    output logic                              o_scnn_store_learning_write_valid,
    input  wire logic                         i_scnn_store_learning_write_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]       o_scnn_store_learning_write_addr,
    output logic signed [15:0]                o_scnn_store_learning_write_code,
    output logic signed [15:0]                o_scnn_store_learning_write_residual,
    output logic                              o_scnn_store_learning_write_code_enable,
    output logic                              o_scnn_store_learning_write_residual_enable,
    output logic                              o_scnn_store_forward_read_valid,
    input  wire logic                         i_scnn_store_forward_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]       o_scnn_store_forward_read_addr,
    output logic [TAG_WIDTH-1:0]              o_scnn_store_forward_read_tag,
    input  wire logic                         i_scnn_store_forward_response_valid,
    output logic                              o_scnn_store_forward_response_ready,
    input  wire logic [TAG_WIDTH-1:0]         i_scnn_store_forward_response_tag,
    input  wire logic signed [15:0]           i_scnn_store_forward_response_code,
    input  wire logic                         i_scnn_store_forward_response_error,
    input  wire logic                         i_scnn_store_protocol_error,
    input  wire logic                         i_initialize_valid,
    output logic                              o_initialize_ready,
    output logic                              o_initialize_done,
    input  wire logic                         i_classifier_pair,
    input  wire logic [VERSION_WIDTH-1:0]     i_context_model_version,
    input  wire logic [VERSION_WIDTH-1:0]     i_context_adapt_version,

    input  wire logic                         i_system_parameter_owner,
    input  wire logic                         i_system_learning_enable,
    input  wire logic                         i_system_profile_aligned,
    input  wire logic [PROFILE_ID_WIDTH-1:0]  i_system_profile_id,
    input  wire logic                         i_request_valid,
    output logic                              o_request_ready,
    input  wire logic                         i_learning_enable,
    input  wire logic [BATCH_COUNT_WIDTH-1:0] i_batch_count,
    input  wire logic [TIMESTEP_WIDTH-1:0]    i_timestep,
    input  wire logic [TIMESTEP_COUNT_WIDTH-1:0]
                                                i_timestep_count,
    input  wire logic [VERSION_WIDTH-1:0]     i_sample_version,
    input  wire logic [MAX_BATCH*CLASS_WIDTH-1:0] i_labels_flat,
    output logic                              o_done,

    // FC and SRNN share the same 12xH40/P4 array used by the SCNN graph.
    // The recurrent suffix is accepted only in SRNN mode by the common layer
    // client; observations are serialized for the topology learning adapter.
    input  wire logic                         i_fc_srnn_layer_valid,
    input wire logic i_fc_srnn_streamed_feedforward,
    input wire logic i_fc_srnn_group_feedforward_end,
    output logic                              o_fc_srnn_layer_ready,
    input  wire logic [1:0]                   i_fc_srnn_mode,
    input  wire logic                         i_fc_srnn_weight_int8,
    input  wire logic                         i_fc_srnn_current_shift6,
    input  wire logic [15:0]                  i_fc_srnn_neuron_count,
    input  wire logic [15:0]                  i_fc_srnn_feedforward_group_count,
    input  wire logic [15:0]                  i_fc_srnn_recurrent_group_count,
    input  wire logic                         i_fc_srnn_recurrent_enable,
    input  wire logic [15:0]                  i_fc_srnn_leak,
    input  wire logic [31:0]                  i_fc_srnn_threshold,
    input  wire logic                         i_fc_srnn_reset_to_zero,
    input  wire logic [15:0]                  i_fc_srnn_trace_decay,
    input  wire logic [51:0]                  i_fc_srnn_transaction_tag,
    input  wire logic [WORK_ADDR_WIDTH-1:0]   i_fc_srnn_work_base_addr,
    input  wire logic [3:0]                   i_fc_srnn_destination_tile_base,
    input  wire logic                         i_fc_srnn_group_valid,
    output logic                              o_fc_srnn_group_ready,
    input  wire logic                         i_fc_srnn_group_recurrent,
    input  wire logic [3:0]                   i_fc_srnn_group_source_valid,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0] i_fc_srnn_group_weight_addr0,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0] i_fc_srnn_group_weight_addr1,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0] i_fc_srnn_group_weight_addr2,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0] i_fc_srnn_group_weight_addr3,
    input  wire logic                         i_srnn_group_streamer_select,
    input  wire logic                         i_srnn_group_streamer_start_valid,
    output logic                              o_srnn_group_streamer_start_ready,
    input  wire logic [9:0]                   i_srnn_group_feedforward_source_count,
    input  wire logic [9:0]                   i_srnn_group_recurrent_source_count,
    input  wire logic [9:0]                   i_srnn_group_feedforward_cache_row_base,
    input  wire logic [9:0]                   i_srnn_group_recurrent_cache_row_base,
    input  wire logic [SRNN_FEEDFORWARD_SOURCE_MAX-1:0]
                                                i_srnn_group_feedforward_spike_flat,
    input  wire logic [SRNN_RECURRENT_SOURCE_MAX-1:0]
                                                i_srnn_group_recurrent_spike_flat,
    output logic                              o_srnn_group_streamer_done,
    output logic                              o_srnn_group_streamer_busy,
    output logic                              o_srnn_group_streamer_sticky_error,
    output logic [3:0]                        o_srnn_group_streamer_error_code,
    output logic                              o_fc_srnn_layer_done,
    output logic                              o_fc_srnn_busy,
    output logic                              o_fc_srnn_idle,
    output logic                              o_fc_srnn_sticky_error,
    output logic [7:0]                        o_fc_srnn_error_code,
    output logic                              o_fc_srnn_observation_valid,
    input  wire logic                         i_fc_srnn_observation_ready,
    input  wire logic                         i_fc_srnn_observation_sink_idle,
    output logic [1:0]                        o_fc_srnn_observation_mode,
    output logic [51:0]                       o_fc_srnn_observation_transaction_tag,
    output logic [9:0]                        o_fc_srnn_observation_neuron_id,
    output logic                              o_fc_srnn_observation_spike,
    output logic signed [31:0]                o_fc_srnn_observation_pre_state,
    output logic signed [31:0]                o_fc_srnn_observation_trace_next,
    output logic                              o_fc_srnn_observation_last_in_record,
    output logic                              o_fc_srnn_observation_busy,
    output logic                              o_fc_srnn_observation_sticky_error,

    // Production FC/SRNN learning ingress. It captures the physical H40
    // completion stream, restores neuron-major order and routes the result to
    // the existing topology-specific TP frontends.
    input  wire logic                         i_fc_srnn_learning_cfg_valid,
    output logic                              o_fc_srnn_learning_cfg_ready,
    input  wire logic [1:0]                   i_fc_srnn_learning_cfg_mode,
    input  wire logic [51:0]                  i_fc_srnn_learning_cfg_tag,
    input  wire logic [5:0]                   i_fc_srnn_learning_sample_id,
    input  wire logic [9:0]                   i_fc_srnn_learning_neuron_base,
    input  wire logic [9:0]                   i_fc_srnn_learning_neuron_count,
    input  wire logic signed [17:0]            i_fc_srnn_learning_fc_threshold,
    input  wire logic                         i_fc_srnn_learning_fc_output_layer,
    input  wire logic [3:0]                   i_fc_srnn_learning_fc_label,
    input  wire logic                         i_fc_srnn_learning_fc_logit_scale_enable,
    input  wire logic [23:0]                  i_fc_srnn_learning_fc_logit_scale_q16,
    input  wire logic                         i_fc_srnn_learning_fc_logit_accumulate,
    input  wire logic                         i_fc_srnn_learning_fc_sample_start,
    input  wire logic                         i_fc_srnn_learning_srnn_path,
    output logic                              o_fc_student_trace_valid,
    input  wire logic                         i_fc_student_trace_ready,
    output logic [5:0]                        o_fc_student_trace_sample,
    output logic [9:0]                        o_fc_student_trace_hidden,
    output logic [16:0]                       o_fc_student_trace_q016,
    output logic                              o_fc_surrogate_valid,
    input  wire logic                         i_fc_surrogate_ready,
    output logic [5:0]                        o_fc_surrogate_sample,
    output logic [9:0]                        o_fc_surrogate_hidden,
    output logic [16:0]                       o_fc_surrogate_q016,
    output logic                              o_fc_logit_row_valid,
    input  wire logic                         i_fc_logit_row_ready,
    output logic [5:0]                        o_fc_logit_sample,
    output logic [3:0]                        o_fc_logit_label,
    output logic                              o_fc_logit_valid,
    input  wire logic                         i_fc_logit_ready,
    output logic signed [15:0]                o_fc_logit_q6,
    output logic                              o_fc_logit_last,
    output logic                              o_srnn_backbone_valid,
    input  wire logic                         i_srnn_backbone_ready,
    output logic                              o_srnn_backbone_path,
    output logic [5:0]                        o_srnn_backbone_sample_id,
    output logic [9:0]                        o_srnn_backbone_id,
    output logic signed [31:0]                o_srnn_backbone_q14,
    output logic                              o_srnn_backbone_last,
    output logic                              o_fc_srnn_learning_capture_done,
    output logic                              o_fc_srnn_learning_done,
    output logic                              o_fc_srnn_learning_busy,
    output logic                              o_fc_srnn_learning_idle,
    output logic                              o_fc_srnn_learning_sticky_error,
    output logic                              o_fc_srnn_learning_error_source,
    output logic [3:0]                        o_fc_srnn_learning_error_code,

    input  wire logic                         i_srnn_h40_cfg_valid,
    output logic                              o_srnn_h40_cfg_ready,
    output logic                              o_srnn_h40_cfg_error,
    input  wire logic [7:0]                   i_srnn_h40_cfg_batch_count,
    input  wire logic [9:0]                   i_srnn_h40_cfg_backbone_count,
    input  wire logic [8:0]                   i_srnn_h40_cfg_hidden_count,
    input  wire logic [4:0]                   i_srnn_h40_cfg_class_count,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]
                                                i_srnn_h40_cfg_wh1_parameter_base,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]
                                                i_srnn_h40_cfg_wout_parameter_base,
    input  wire logic                         i_srnn_h40_prepare_valid,
    output logic                              o_srnn_h40_prepare_ready,
    input  wire logic [VERSION_WIDTH-1:0]     i_srnn_h40_prepare_version,
    output logic                              o_srnn_h40_prepare_done,
    output logic                              o_srnn_h40_cache_valid,
    input  wire logic                         i_srnn_h40_cache_invalidate,
    input  wire logic                         i_srnn_h40_wh1_cache_invalidate,
    output logic                              o_srnn_h40_forward_read_valid,
    input  wire logic                         i_srnn_h40_forward_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]       o_srnn_h40_forward_read_addr,
    output logic [TAG_WIDTH-1:0]              o_srnn_h40_forward_read_tag,
    input  wire logic                         i_srnn_h40_forward_response_valid,
    output logic                              o_srnn_h40_forward_response_ready,
    input  wire logic [TAG_WIDTH-1:0]         i_srnn_h40_forward_response_tag,
    input  wire logic signed [15:0]           i_srnn_h40_forward_response_code,
    input  wire logic                         i_srnn_h40_forward_response_error,
    input  wire logic                         i_srnn_h40_timestep_start,
    output logic                              o_srnn_h40_timestep_start_ready,
    input  wire logic [$clog2(SRNN_M2_TIMESTEP_MAX)-1:0]
                                                i_srnn_h40_timestep_id,
    input  wire logic [VERSION_WIDTH-1:0]     i_srnn_h40_sequence_version,
    input  wire logic                         i_srnn_h40_include_target_path,
    input  wire logic                         i_srnn_h40_observation_valid,
    output logic                              o_srnn_h40_observation_ready,
    input  wire logic                         i_srnn_h40_observation_path,
    input  wire logic [7:0]                   i_srnn_h40_observation_sample_id,
    input  wire logic [9:0]                   i_srnn_h40_observation_backbone_id,
    input  wire logic signed [31:0]           i_srnn_h40_observation_q14,
    input  wire logic                         i_srnn_h40_observation_last,
    input  wire logic                         i_srnn_h40_observation_done,
    output logic                              o_srnn_h40_hidden_valid,
    input  wire logic                         i_srnn_h40_hidden_ready,
    output logic                              o_srnn_h40_hidden_path,
    output logic [7:0]                        o_srnn_h40_hidden_sample_id,
    output logic [8:0]                        o_srnn_h40_hidden_id,
    output logic signed [31:0]                o_srnn_h40_hidden_membrane_pre_q14,
    output logic signed [31:0]                o_srnn_h40_hidden_membrane_next_q14,
    output logic                              o_srnn_h40_hidden_spike,
    output logic signed [31:0]                o_srnn_h40_hidden_trace_q14,
    output logic signed [31:0]                o_srnn_h40_hidden_output_feature_q14,
    output logic                              o_srnn_h40_hidden_last,
    output logic                              o_srnn_h40_logit_valid,
    input  wire logic                         i_srnn_h40_logit_ready,
    output logic [7:0]                        o_srnn_h40_logit_sample_id,
    output logic [4:0]                        o_srnn_h40_logit_class_id,
    output logic signed [63:0]                o_srnn_h40_logit_q28,
    output logic                              o_srnn_h40_logit_last,
    output logic [$clog2(SRNN_M2_TIMESTEP_MAX)-1:0]
                                                o_srnn_h40_timestep_id,
    output logic                              o_srnn_h40_timestep_done,
    output logic                              o_srnn_h40_busy,
    output logic                              o_srnn_h40_sticky_error,
    output logic                              o_srnn_h40_protocol_error,
    output logic                              o_srnn_h40_overflow,

    // FC W0 cache coherence. The scalar source is the shared TP parameter
    // backend; this system only packs/refills the one physical H40 cache.
    input  wire logic                         i_fc_w0_refill_valid,
    output logic                              o_fc_w0_refill_ready,
    input  wire logic                         i_fc_w0_refill_all,
    input  wire logic                         i_fc_cache_refill_weight_int8,
    input  wire logic [3:0]                   i_fc_cache_refill_destination_tile_base,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0]
                                                i_fc_w0_dirty_feature,
    input  wire logic [3:0]                   i_fc_w0_dirty_tile,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]  i_fc_w0_parameter_base,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]  i_fc_w0_hidden_stride,
    input  wire logic [9:0]                   i_fc_cache_row_base,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0]
                                                i_fc_w0_feature_count,
    input  wire logic [9:0]                   i_fc_w0_hidden_count,
    input  wire logic [4:0]                   i_fc_w0_tile_count,
    input  wire logic [VERSION_WIDTH-1:0]     i_fc_w0_version,
    output logic                              o_fc_cache_forward_read_valid,
    input  wire logic                         i_fc_cache_forward_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]       o_fc_cache_forward_read_addr,
    output logic [TAG_WIDTH-1:0]              o_fc_cache_forward_read_tag,
    input  wire logic                         i_fc_cache_forward_response_valid,
    output logic                              o_fc_cache_forward_response_ready,
    input  wire logic [TAG_WIDTH-1:0]         i_fc_cache_forward_response_tag,
    input  wire logic signed [15:0]           i_fc_cache_forward_response_code,
    input  wire logic                         i_fc_cache_forward_response_error,
    output logic                              o_fc_w0_refill_done,
    output logic [31:0]                       o_fc_w0_rows_published,
    output logic [VERSION_WIDTH-1:0]          o_fc_w0_published_version,
    output logic [1:0]                        o_cache_fill_grant,
    output logic                              o_cache_refill_busy,
    output logic                              o_cache_refill_quiescent,
    output logic                              o_cache_refill_sticky_error,

    // Frozen SRNN backbone cache coherence. The destination-major INT8
    // matrix is packed into the SRNN bank-pair layout of the same H40 cache;
    // it does not instantiate a second weight memory or compute array.
    input  wire logic                         i_srnn_backbone_refill_valid,
    output logic                              o_srnn_backbone_refill_ready,
    input  wire logic                         i_srnn_backbone_refill_all,
    input  wire logic [9:0]                   i_srnn_backbone_dirty_source,
    input  wire logic [3:0]                   i_srnn_backbone_dirty_tile,
    input  wire logic [SRNN_BACKBONE_PARAM_ADDR_WIDTH-1:0]
                                                i_srnn_backbone_parameter_base,
    input  wire logic [9:0]                   i_srnn_backbone_cache_row_base,
    input  wire logic [9:0]                   i_srnn_backbone_source_count,
    input  wire logic [9:0]                   i_srnn_backbone_destination_count,
    input  wire logic [3:0]                   i_srnn_backbone_tile_base,
    input  wire logic [VERSION_WIDTH-1:0]     i_srnn_backbone_version,
    output logic                              o_srnn_backbone_forward_read_valid,
    input  wire logic                         i_srnn_backbone_forward_read_ready,
    output logic [SRNN_BACKBONE_PARAM_ADDR_WIDTH-1:0]
                                                o_srnn_backbone_forward_read_addr,
    output logic [TAG_WIDTH-1:0]              o_srnn_backbone_forward_read_tag,
    input  wire logic                         i_srnn_backbone_forward_response_valid,
    output logic                              o_srnn_backbone_forward_response_ready,
    input  wire logic [TAG_WIDTH-1:0]
                                                i_srnn_backbone_forward_response_tag,
    input  wire logic signed [15:0]
                                                i_srnn_backbone_forward_response_code,
    input  wire logic                         i_srnn_backbone_forward_response_error,
    output logic                              o_srnn_backbone_refill_done,
    output logic [31:0]                       o_srnn_backbone_rows_published,
    output logic [VERSION_WIDTH-1:0]
                                                o_srnn_backbone_published_version,

    input  wire logic [PROFILE_ID_WIDTH-1:0]  i_profile_id,
    input  wire logic [TENSOR_ID_WIDTH-1:0]   i_conv_tensor_id,
    input  wire logic [TENSOR_ID_WIDTH-1:0]   i_classifier_tensor_id,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0] i_total_input_channels,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0] i_total_output_channels,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]  i_conv_parameter_base,
    input  wire logic [Q16_ADDR_WIDTH-1:0]    i_conv_weight_q16_base,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]  i_classifier_parameter_base,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0] i_classifier_feature_count,
    input  wire logic [CLASS_COUNT_WIDTH-1:0] i_classifier_class_count,
    input  wire logic                         i_classifier_trace_leak_valid,
    input  wire logic signed [15:0]           i_classifier_trace_leak_q8_8,
    input  wire logic [15:0]                  i_conv_vote_increment,
    input  wire logic [15:0]                  i_conv_vote_threshold,
    input  wire logic [15:0]                  i_classifier_vote_increment,
    input  wire logic [15:0]                  i_classifier_vote_threshold,
    input  wire logic signed [15:0]           i_code_min,
    input  wire logic signed [15:0]           i_code_max,

    input  wire logic                         i_forward_start_valid,
    output logic                              o_forward_start_ready,
    output logic                              o_forward_start_fire,
    input  wire logic                         i_forward_done,
    output logic                              o_forward_active,
    output logic [VERSION_WIDTH-1:0]          o_forward_model_version,
    output logic [VERSION_WIDTH-1:0]          o_forward_adapt_version,
    input  wire logic                         i_restore_complete,
    output logic                              o_restore_ready,
    output logic                              o_recovery_required,

    input  wire logic [15:0]                  i_conv_in_width,
    input  wire logic [15:0]                  i_conv_in_height,
    input  wire logic [15:0]                  i_conv_in_channels,
    input  wire logic [15:0]                  i_conv_out_width,
    input  wire logic [15:0]                  i_conv_out_height,
    input  wire logic [15:0]                  i_conv_out_channels,
    input  wire logic [7:0]                   i_conv_stride,
    input  wire logic [7:0]                   i_conv_padding,
    input  wire logic [1:0]                   i_conv_pool_mode,
    input  wire logic [WORK_ADDR_WIDTH-1:0]   i_conv_work_base_addr,
    input  wire logic [15:0]                  i_conv_leak,
    input  wire logic [31:0]                  i_conv_threshold,
    input  wire logic                         i_conv_reset_to_zero,
    input  wire logic [15:0]                  i_conv_trace_decay,
    input  wire logic [19:0]                  i_conv_layer_tag,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0] i_target_feature_count,
    input  wire logic signed [15:0]           i_target_trace_leak_q8_8,
    input  wire logic                         i_target_trace_state_reset,
    input  wire logic [CHANNEL_WIDTH-1:0]     i_classifier_feature_channel_base,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0] i_classifier_feature_channel_count,
    input  wire logic [WORK_ADDR_WIDTH-1:0]   i_classifier_work_base_addr,
    input  wire logic [15:0]                  i_classifier_trace_decay,
    input  wire logic [19:0]                  i_classifier_layer_tag,
    input  wire logic [FRAME_WIDTH-1:0]       i_frame_index,

    // Each accepted descriptor snapshots the existing Conv/layout inputs.
    // A final descriptor also executes the classifier and retires one frame.
    input  wire logic                         i_graph_descriptor_valid,
    output logic                              o_graph_descriptor_ready,
    input  wire logic                         i_graph_descriptor_last,
    output logic                              o_graph_descriptor_done,
    output logic                              o_graph_frame_active,
    output logic                              o_graph_frame_target,
    output logic [SAMPLE_WIDTH-1:0]           o_graph_frame_sample,
    output logic [TIMESTEP_WIDTH-1:0]         o_graph_frame_timestep,
    output logic [VERSION_WIDTH-1:0]          o_graph_frame_version,
    output logic                              o_state_service_valid,
    input  wire logic                         i_state_service_ready,
    output logic                              o_state_service_release,
    output logic                              o_state_service_is_classifier,
    output logic [9:0]                        o_state_service_output_channel_base,
    output logic [9:0]                        o_state_service_output_channel_count,
    output logic [31:0]                       o_state_service_spatial_base,
    output logic [5:0]                        o_state_service_spatial_count,
    output logic                              o_state_service_active_pair,
    output logic [WORK_ADDR_WIDTH-1:0]        o_state_service_work_base_addr,

    output logic                              o_feature_request_valid,
    input  wire logic                         i_feature_request_ready,
    output logic [3:0]                        o_feature_term_valid,
    output logic [39:0]                       o_feature_input_channel_flat,
    output logic [7:0]                        o_feature_kernel_row_flat,
    output logic [7:0]                        o_feature_kernel_col_flat,
    output logic [31:0]                       o_feature_spatial_base,
    output logic [5:0]                        o_feature_spatial_count,
    output logic [15:0]                       o_feature_output_row,
    output logic [15:0]                       o_feature_output_col,
    output logic [3:0]                        o_feature_value_bit,
    input  wire logic                         i_feature_response_valid,
    output logic                              o_feature_response_ready,
    input  wire logic [39:0]                  i_feature_mask0,
    input  wire logic [39:0]                  i_feature_mask1,
    input  wire logic [39:0]                  i_feature_mask2,
    input  wire logic [39:0]                  i_feature_mask3,
    input  wire logic                         i_feature_service_error,
    input  wire logic                         i_feature_write_slot,
    input  wire logic [FEATURE_ADDR_WIDTH-1:0] i_feature_write_base_addr,
    input  wire logic [15:0]                  i_feature_write_row_stride_words,
    input  wire logic [FEATURE_ADDR_WIDTH-1:0]
                                                i_feature_write_channel_stride_words,
    output logic                              o_feature_update_valid,
    input  wire logic                         i_feature_update_ready,
    output logic                              o_feature_update_slot,
    output logic [FEATURE_ADDR_WIDTH-1:0]     o_feature_update_addr,
    output logic [63:0]                       o_feature_update_data,
    output logic [63:0]                       o_feature_update_mask,
    input  wire logic                         i_feature_update_done,

    input  wire logic [3:0]                   i_maintenance_tile,
    input  wire logic                         i_state_maintenance_write_valid,
    output logic                              o_state_maintenance_write_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0] i_state_maintenance_write_addr,
    input  wire logic [319:0]                 i_state_maintenance_write_data,
    input  wire logic [39:0]                  i_state_maintenance_write_mask,
    input  wire logic                         i_state_maintenance_read_valid,
    output logic                              o_state_maintenance_read_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0] i_state_maintenance_read_addr,
    output logic                              o_state_maintenance_result_valid,
    input  wire logic                         i_state_maintenance_result_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]      o_state_maintenance_result_addr,
    output logic [319:0]                      o_state_maintenance_read_data,
    input  wire logic                         i_trace_maintenance_write_valid,
    output logic                              o_trace_maintenance_write_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0] i_trace_maintenance_write_addr,
    input  wire logic [319:0]                 i_trace_maintenance_write_data,
    input  wire logic [39:0]                  i_trace_maintenance_write_mask,
    input  wire logic                         i_trace_maintenance_read_valid,
    output logic                              o_trace_maintenance_read_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0] i_trace_maintenance_read_addr,
    output logic                              o_trace_maintenance_result_valid,
    input  wire logic                         i_trace_maintenance_result_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]      o_trace_maintenance_result_addr,
    output logic [319:0]                      o_trace_maintenance_read_data,

    output logic                              o_classifier_cache_valid,
    output logic                              o_active_classifier_pair,
    output logic [VERSION_WIDTH-1:0]          o_parameter_version,
    output logic                              o_conv_cache_valid,
    output logic [9:0]                        o_conv_cached_output_base,
    output logic [4:0]                        o_conv_cached_output_count,
    output logic [VERSION_WIDTH-1:0]          o_conv_cached_version,
    output logic                              o_conv_cache_invalidate,
    output logic [VERSION_WIDTH-1:0]          o_model_version,
    output logic [VERSION_WIDTH-1:0]          o_adapt_version,
    output logic                              o_external_cfg_valid,
    input  wire logic                         i_external_cfg_ready,
    output logic [PROFILE_ID_WIDTH-1:0]       o_external_cfg_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]        o_external_cfg_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]       o_external_cfg_parameter_base,
    output logic [BATCH_COUNT_WIDTH-1:0]      o_external_cfg_batch_count,
    output logic [FEATURE_COUNT_WIDTH-1:0]    o_external_cfg_source_count,
    output logic [CLASS_COUNT_WIDTH-1:0]      o_external_cfg_output_count,
    output logic [15:0]                       o_external_cfg_vote_increment,
    output logic [15:0]                       o_external_cfg_vote_threshold,
    output logic signed [15:0]                o_external_cfg_code_min,
    output logic signed [15:0]                o_external_cfg_code_max,
    input  wire logic                         i_external_cfg_error,
    input  wire logic                         i_external_active,
    output logic                              o_external_release,
    input  wire logic                         i_external_release_ready,
    output logic                              o_external_source_trace_valid,
    input  wire logic                         i_external_source_trace_ready,
    output logic [SAMPLE_WIDTH-1:0]           o_external_source_trace_sample_id,
    output logic [FEATURE_WIDTH-1:0]          o_external_source_trace_index,
    output logic [15:0]                       o_external_source_trace_q016,
    input  wire logic                         i_external_source_trace_error,
    output logic                              o_external_logit_row_valid,
    input  wire logic                         i_external_logit_row_ready,
    output logic [SAMPLE_WIDTH-1:0]           o_external_logit_sample_id,
    output logic [CLASS_WIDTH-1:0]            o_external_logit_label,
    output logic                              o_external_logit_valid,
    input  wire logic                         i_external_logit_ready,
    output logic signed [15:0]                o_external_logit_q6,
    input  wire logic                         i_external_logit_error,
    input  wire logic                         i_external_logit_row_done,
    output logic                              o_external_observation_done,
    input  wire logic                         i_external_observation_done_ready,
    input  wire logic                         i_external_window_done,
    input  wire logic                         i_external_window_busy,
    input  wire logic                         i_external_context_error,
    input  wire logic                         i_external_gradient_overflow,
    input  wire logic                         i_external_update_config_error,
    input  wire logic                         i_external_request_error,
    input  wire logic                         i_external_store_error,
    input  wire logic                         i_external_internal_error,
    output logic                              o_capture_complete,
    output logic                              o_learning_start_fire,
    output logic                              o_commit_pulse,
    output logic                              o_commit_code_dirty,
    output logic                              o_commit_optimizer_dirty,
    output logic                              o_detach_pulse,
    output logic                              o_commit_pending,
    output logic                              o_inference_hold,
    output logic                              o_initialized,
    output logic                              o_versions_aligned,
    output logic [1:0]                        o_array_active_owner,
    output logic [15:0]                       o_array_tile_idle,
    output logic [15:0]                       o_array_tile_sticky_error,
    output logic                              o_context_active,
    output logic                              o_context_is_classifier,
    output logic                              o_busy,
    output logic                              o_quiescent,
    output logic                              o_sticky_error
);
    localparam int PHYSICAL_TILE_COUNT = 12;
    localparam bit SRNN_M2_ENABLED =
        ENABLE_SRNN_CLIENT && ENABLE_SRNN_M2_H40_CLIENT;
    localparam bit SRNN_STREAM_ENABLED =
        ENABLE_SRNN_CLIENT && ENABLE_SRNN_GROUP_STREAMER;
    logic graph_start_valid;
    logic graph_start_ready;
    logic graph_target;
    logic [SAMPLE_WIDTH-1:0] graph_sample;
    logic [TIMESTEP_WIDTH-1:0] graph_timestep;
    logic [FRAME_WIDTH-1:0] scnn_frame_index;
    logic [VERSION_WIDTH-1:0] graph_version;
    logic [CLASS_WIDTH-1:0] graph_label;
    logic graph_sample_last;
    logic graph_done;
    logic layer_start_valid;
    logic layer_start_ready;
    logic layer_done;
    logic frame_service_busy;
    logic frame_service_error;
    logic layer_is_tail;
    logic state_read_pending_q;
    logic trace_read_pending_q;
    logic state_service_drained;
    typedef struct packed {
        logic [PARAM_ADDR_WIDTH-1:0] parameter_base;
        logic [15:0] in_width, in_height, in_channels;
        logic [15:0] out_width, out_height, out_channels;
        logic [7:0] stride, padding;
        logic [1:0] pool_mode;
        logic [WORK_ADDR_WIDTH-1:0] work_base;
        logic [15:0] leak;
        logic [31:0] threshold;
        logic reset_to_zero;
        logic [15:0] trace_decay;
        logic [19:0] layer_tag;
        logic feature_write_slot;
        logic [FEATURE_ADDR_WIDTH-1:0] feature_write_base;
        logic [15:0] feature_write_row_stride;
        logic [FEATURE_ADDR_WIDTH-1:0] feature_write_channel_stride;
    } frame_descriptor_t;
    frame_descriptor_t descriptor_input;
    frame_descriptor_t descriptor_active;
    logic conv_observation_enable;
    logic classifier_observation_enable;
    logic observation_target;

    assign descriptor_input = {i_conv_parameter_base,
        i_conv_in_width, i_conv_in_height, i_conv_in_channels,
        i_conv_out_width, i_conv_out_height, i_conv_out_channels,
        i_conv_stride, i_conv_padding, i_conv_pool_mode, i_conv_work_base_addr,
        i_conv_leak, i_conv_threshold, i_conv_reset_to_zero, i_conv_trace_decay,
        i_conv_layer_tag, i_feature_write_slot, i_feature_write_base_addr,
        i_feature_write_row_stride_words, i_feature_write_channel_stride_words};
    assign o_graph_frame_active = frame_service_busy;
    assign o_graph_frame_target = graph_target;
    assign o_graph_frame_sample = graph_sample;
    assign o_graph_frame_timestep = graph_timestep;
    assign o_graph_frame_version = graph_version;
    if (ENABLE_SCNN_CLIENT && ENABLE_SCNN_FRAME_SERVICE) begin : g_frame_service
        unified_scnn_frame_descriptor_sequencer_v1 #(
            .DESCRIPTOR_WIDTH($bits(frame_descriptor_t))
        ) u_frame_sequence (
            .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
            .i_frame_start_valid(graph_start_valid),
            .o_frame_start_ready(graph_start_ready), .o_frame_done(graph_done),
            .i_descriptor_valid(i_graph_descriptor_valid),
            .o_descriptor_ready(o_graph_descriptor_ready),
            .i_descriptor(descriptor_input), .i_descriptor_last(i_graph_descriptor_last),
            .o_descriptor_done(o_graph_descriptor_done),
            .o_descriptor(descriptor_active), .o_tail_active(layer_is_tail),
            .o_layer_start_valid(layer_start_valid), .i_layer_start_ready(layer_start_ready),
            .i_layer_done(layer_done), .o_busy(frame_service_busy),
            .o_sticky_error(frame_service_error)
        );
    end else begin : g_legacy_frame
        assign graph_start_ready = layer_start_ready;
        assign graph_done = layer_done;
        assign layer_start_valid = graph_start_valid;
        assign descriptor_active = descriptor_input;
        assign layer_is_tail = 1'b1;
        assign o_graph_descriptor_ready = 1'b0;
        assign o_graph_descriptor_done = 1'b0;
        assign frame_service_busy = 1'b0;
        assign frame_service_error = 1'b0;
    end

    logic prescan_request_valid;
    logic prescan_request_ready;
    logic [3:0] prescan_term_valid;
    logic [39:0] prescan_input_channel_flat;
    logic [7:0] prescan_kernel_row_flat;
    logic [7:0] prescan_kernel_col_flat;
    logic [31:0] prescan_spatial_base;
    logic [5:0] prescan_spatial_count;
    logic [15:0] prescan_output_row;
    logic [15:0] prescan_output_col;
    logic prescan_response_valid;
    logic prescan_response_ready;

    logic core_request_valid;
    logic core_request_ready;
    logic [3:0] core_term_valid;
    logic [39:0] core_input_channel_flat;
    logic [7:0] core_kernel_row_flat;
    logic [7:0] core_kernel_col_flat;
    logic [31:0] core_spatial_base;
    logic [5:0] core_spatial_count;
    logic [15:0] core_output_row;
    logic [15:0] core_output_col;
    logic [3:0] core_value_bit;
    logic core_response_valid;
    logic core_response_ready;
    logic [39:0] response_mask0;
    logic [39:0] response_mask1;
    logic [39:0] response_mask2;
    logic [39:0] response_mask3;
    logic response_error;
    logic feature_arbiter_busy;
    logic feature_arbiter_error;

    logic array_context_load_valid;
    logic array_context_load_ready;
    logic array_context_load_weights;
    logic [9:0] array_context_output_base;
    logic [9:0] array_context_output_count;
    logic [31:0] array_context_spatial_base;
    logic [5:0] array_context_spatial_count;
    logic array_context_pair;
    logic [WORK_ADDR_WIDTH-1:0] array_context_work_base;
    logic array_context_release_valid;
    logic array_context_release_ready;
    logic [9:0] array_release_output_base;
    logic [9:0] array_release_output_count;
    logic [31:0] array_release_spatial_base;
    logic [5:0] array_release_spatial_count;
    logic array_release_pair;
    logic [WORK_ADDR_WIDTH-1:0] array_release_work_base;

    logic conv_context_valid;
    logic conv_context_ready;
    logic conv_context_load_weights;
    logic [9:0] conv_context_output_base;
    logic [4:0] conv_context_output_count;
    logic [CHANNEL_COUNT_WIDTH-1:0] conv_context_input_count;
    logic conv_context_pair;
    logic [PARAM_ADDR_WIDTH-1:0] conv_context_parameter_base;
    logic [VERSION_WIDTH-1:0] conv_context_version;
    logic context_bridge_error;

    logic surrogate_valid;
    logic surrogate_ready;
    logic surrogate_target;
    logic [SAMPLE_WIDTH-1:0] surrogate_sample;
    logic [TIMESTEP_WIDTH-1:0] surrogate_timestep;
    logic [VERSION_WIDTH-1:0] surrogate_version;
    logic [19:0] surrogate_layer_tag;
    logic [9:0] surrogate_channel;
    logic [31:0] surrogate_spatial;
    logic signed [15:0] surrogate_q8_8;

    logic feature_wave_valid;
    logic feature_wave_ready;
    logic [LANES-1:0] feature_wave_mask;
    logic [FEATURE_WIDTH-1:0] feature_wave_base;
    logic [LANES*16-1:0] feature_wave_flat;
    logic [VERSION_WIDTH-1:0] feature_wave_version;
    logic [FRAME_WIDTH-1:0] feature_wave_frame;
    logic feature_wave_first;
    logic feature_wave_last;
    logic feature_wave_sample_reset;
    logic feature_wave_sample_last;
    logic [CLASS_WIDTH-1:0] feature_sample_label;

    logic logit_wave_valid;
    logic logit_wave_ready;
    logic [ENGINES-1:0] logit_wave_mask;
    logic [CLASS_WIDTH-1:0] logit_wave_base;
    logic [CLASS_COUNT_WIDTH-1:0] logit_wave_count;
    logic [ENGINES*16-1:0] logit_wave_flat;
    logic [VERSION_WIDTH-1:0] logit_wave_version;
    logic [FRAME_WIDTH-1:0] logit_wave_frame;
    logic logit_wave_frame_last;
    logic logit_wave_sample_last;

    logic [3:0] scnn_weight_fill_valid;
    logic [3:0] scnn_weight_fill_ready;
    logic [7:0] scnn_weight_fill_tile_flat;
    logic [3:0] scnn_weight_fill_bank;
    logic [4*WEIGHT_ADDR_WIDTH-1:0] scnn_weight_fill_addr_flat;
    logic [799:0] scnn_weight_fill_data_flat;
    logic scnn_weight_fill_busy;
    logic [3:0] m2_weight_fill_valid;
    logic [3:0] m2_weight_fill_ready;
    logic [7:0] m2_weight_fill_tile_flat;
    logic [3:0] m2_weight_fill_bank;
    logic [4*WEIGHT_ADDR_WIDTH-1:0] m2_weight_fill_addr_flat;
    logic [799:0] m2_weight_fill_data_flat;
    logic [3:0] topology_weight_fill_valid;
    logic [3:0] topology_weight_fill_ready;
    logic [7:0] topology_weight_fill_tile_flat;
    logic [3:0] topology_weight_fill_bank;
    logic [4*WEIGHT_ADDR_WIDTH-1:0] topology_weight_fill_addr_flat;
    logic [799:0] topology_weight_fill_data_flat;
    logic topology_weight_fill_busy;
    logic topology_weight_fill_quiescent;
    logic topology_weight_fill_error;
    logic [3:0] srnn_backbone_weight_fill_valid;
    logic [3:0] srnn_backbone_weight_fill_ready;
    logic [7:0] srnn_backbone_weight_fill_tile_flat;
    logic [3:0] srnn_backbone_weight_fill_bank;
    logic [4*WEIGHT_ADDR_WIDTH-1:0]
        srnn_backbone_weight_fill_addr_flat;
    logic [799:0] srnn_backbone_weight_fill_data_flat;
    logic srnn_backbone_refill_busy;
    logic srnn_backbone_refill_error;
    logic [3:0] weight_fill_valid;
    logic [3:0] weight_fill_ready;
    logic [7:0] weight_fill_tile_flat;
    logic [3:0] weight_fill_bank;
    logic [4*WEIGHT_ADDR_WIDTH-1:0] weight_fill_addr_flat;
    logic [799:0] weight_fill_data_flat;
    logic m2_cache_evicted;
    logic weight_fill_context;
    logic [1:0] topology_weight_fill_grant;
    logic scnn_cache_evicted;
    logic fc_refill_busy;
    logic scnn_admission_allowed;
    logic scnn_initialize_ready;
    logic scnn_request_ready;
    logic scnn_forward_start_ready;
    logic m2_prepare_allowed;
    logic m2_prepare_ready;
    logic foreign_refill_allowed;
    logic fc_refill_ready;
    logic backbone_refill_ready;
    logic array_refill_safe;
    logic array_busy;
    logic array_idle;
    logic array_error;
    logic subsystem_busy;
    logic subsystem_quiescent;
    logic subsystem_error;
    logic maintenance_lease_q;
    logic maintenance_owner_m2_q;
    logic maintenance_request;
    logic maintenance_start_valid;
    logic maintenance_start_ready;
    logic maintenance_release_valid;
    logic maintenance_release_ready;
    logic array_maintenance_start_valid;
    logic array_maintenance_start_ready;
    logic array_maintenance_release_valid;
    logic array_maintenance_release_ready;
    logic [3:0] array_maintenance_tile;
    logic array_state_write_valid;
    logic array_state_write_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] array_state_write_addr;
    logic [319:0] array_state_write_data;
    logic [39:0] array_state_write_mask;
    logic array_trace_write_valid;
    logic array_trace_write_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] array_trace_write_addr;
    logic [319:0] array_trace_write_data;
    logic [39:0] array_trace_write_mask;
    logic array_state_read_ready;
    logic array_trace_read_ready;
    logic parameter_access_safe;
    logic fc_srnn_learning_scalar_ready;
    logic [3:0] fc_srnn_destination_tile_base;
    logic [3:0] fc_cache_refill_destination_tile_base;

    logic m2_array_owner;
    logic m2_engine_busy;
    logic m2_start_allowed;
    logic m2_timestep_start_ready;
    logic m2_maintenance_start_valid;
    logic m2_maintenance_start_ready;
    logic m2_maintenance_release_valid;
    logic m2_maintenance_release_ready;
    logic [3:0] m2_maintenance_tile;
    logic m2_state_write_valid;
    logic m2_state_write_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] m2_state_write_addr;
    logic [319:0] m2_state_write_data;
    logic [39:0] m2_state_write_mask;
    logic m2_trace_write_valid;
    logic m2_trace_write_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] m2_trace_write_addr;
    logic [319:0] m2_trace_write_data;
    logic [39:0] m2_trace_write_mask;

    logic m2_layer_valid;
    logic m2_layer_ready;
    logic [1:0] m2_layer_mode;
    logic [15:0] m2_layer_neuron_count;
    logic [15:0] m2_layer_feedforward_group_count;
    logic [15:0] m2_layer_recurrent_group_count;
    logic m2_layer_recurrent_enable;
    logic [15:0] m2_layer_leak;
    logic [31:0] m2_layer_threshold;
    logic m2_layer_reset_to_zero;
    logic [15:0] m2_layer_trace_decay;
    logic [51:0] m2_layer_transaction_tag;
    logic [WORK_ADDR_WIDTH-1:0] m2_layer_work_base_addr;
    logic [3:0] m2_layer_destination_tile_base;
    logic m2_group_valid;
    logic m2_group_ready;
    logic m2_group_recurrent;
    logic [3:0] m2_group_source_valid;
    logic [WEIGHT_ADDR_WIDTH-1:0] m2_group_weight_addr0;
    logic [WEIGHT_ADDR_WIDTH-1:0] m2_group_weight_addr1;
    logic [WEIGHT_ADDR_WIDTH-1:0] m2_group_weight_addr2;
    logic [WEIGHT_ADDR_WIDTH-1:0] m2_group_weight_addr3;
    logic srnn_group_streamer_selected;
    logic srnn_streamer_start_ready;
    logic srnn_streamer_done;
    logic srnn_streamer_busy;
    logic srnn_streamer_sticky_error;
    logic [3:0] srnn_streamer_error_code;
    logic srnn_streamer_layer_valid;
    logic srnn_streamer_layer_ready;
    logic [15:0] srnn_streamer_feedforward_group_count;
    logic [15:0] srnn_streamer_recurrent_group_count;
    logic srnn_streamer_recurrent_enable;
    logic [15:0] srnn_streamer_neuron_count;
    logic [15:0] srnn_streamer_leak;
    logic [31:0] srnn_streamer_threshold;
    logic srnn_streamer_reset_to_zero;
    logic [15:0] srnn_streamer_trace_decay;
    logic [51:0] srnn_streamer_transaction_tag;
    logic [WORK_ADDR_WIDTH-1:0] srnn_streamer_work_base_addr;
    logic [3:0] srnn_streamer_destination_tile_base;
    logic srnn_streamer_group_valid;
    logic srnn_streamer_group_ready;
    logic srnn_streamer_group_recurrent;
    logic [3:0] srnn_streamer_group_source_valid;
    logic [WEIGHT_ADDR_WIDTH-1:0] srnn_streamer_group_weight_addr0;
    logic [WEIGHT_ADDR_WIDTH-1:0] srnn_streamer_group_weight_addr1;
    logic [WEIGHT_ADDR_WIDTH-1:0] srnn_streamer_group_weight_addr2;
    logic [WEIGHT_ADDR_WIDTH-1:0] srnn_streamer_group_weight_addr3;
    logic array_fc_srnn_layer_ready;
    logic array_fc_srnn_group_ready;
    logic array_fc_srnn_layer_done;
    logic array_fc_srnn_busy;
    logic array_fc_srnn_idle;
    logic array_fc_srnn_sticky_error;
    logic [7:0] array_fc_srnn_error_code;
    logic array_fc_srnn_observation_valid;
    logic array_fc_srnn_observation_ready;
    logic m2_array_observation_ready;
    logic [1:0] array_fc_srnn_observation_mode;
    logic [51:0] array_fc_srnn_observation_transaction_tag;
    logic [9:0] array_fc_srnn_observation_neuron_id;
    logic array_fc_srnn_observation_spike;
    logic signed [31:0] array_fc_srnn_observation_pre_state;
    logic signed [31:0] array_fc_srnn_observation_trace_next;
    logic array_fc_srnn_observation_last_in_record;
    logic array_fc_srnn_observation_busy;
    logic array_fc_srnn_observation_sticky_error;

    // Preserve tile zero for omitted legacy inputs and for each unknown bit.
    for (genvar bit_index = 0; bit_index < 4; bit_index++) begin : g_fc_tile_base
        assign fc_srnn_destination_tile_base[bit_index] =
            i_fc_srnn_destination_tile_base[bit_index] === 1'b1;
        assign fc_cache_refill_destination_tile_base[bit_index] =
            i_fc_cache_refill_destination_tile_base[bit_index] === 1'b1;
    end

    assign maintenance_request = ((|weight_fill_valid) && !array_refill_safe) ||
        state_read_pending_q || trace_read_pending_q ||
        i_state_maintenance_write_valid || i_state_maintenance_read_valid ||
        o_state_maintenance_result_valid ||
        i_trace_maintenance_write_valid || i_trace_maintenance_read_valid ||
        o_trace_maintenance_result_valid;
    // A read is pending before its result-valid becomes visible. Do not release
    // the graph barrier in that gap, or while an external requester holds valid.
    assign state_service_drained = !maintenance_request && !maintenance_lease_q &&
        !state_read_pending_q && !trace_read_pending_q &&
        (&o_array_tile_idle[PHYSICAL_TILE_COUNT-1:0]);
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_read_pending_q <= 1'b0;
            trace_read_pending_q <= 1'b0;
        end else begin
            case ({i_state_maintenance_read_valid && o_state_maintenance_read_ready,
                   o_state_maintenance_result_valid && i_state_maintenance_result_ready})
                2'b10: state_read_pending_q <= 1'b1;
                2'b01: state_read_pending_q <= 1'b0;
                default: ;
            endcase
            case ({i_trace_maintenance_read_valid && o_trace_maintenance_read_ready,
                   o_trace_maintenance_result_valid && i_trace_maintenance_result_ready})
                2'b10: trace_read_pending_q <= 1'b1;
                2'b01: trace_read_pending_q <= 1'b0;
                default: ;
            endcase
        end
    end
    assign maintenance_start_valid = maintenance_request &&
        !maintenance_lease_q && !m2_maintenance_start_valid;
    assign maintenance_start_ready = !maintenance_lease_q &&
        !m2_maintenance_start_valid && array_maintenance_start_ready;
    assign maintenance_release_valid = maintenance_lease_q &&
        !maintenance_owner_m2_q && !maintenance_request;
    assign maintenance_release_ready = maintenance_lease_q &&
        !maintenance_owner_m2_q && array_maintenance_release_ready;
    assign array_maintenance_start_valid = !maintenance_lease_q &&
        (m2_maintenance_start_valid || maintenance_start_valid);
    assign m2_maintenance_start_ready = SRNN_M2_ENABLED &&
        !maintenance_lease_q && array_maintenance_start_ready;
    assign array_maintenance_release_valid = maintenance_lease_q &&
        (maintenance_owner_m2_q ? m2_maintenance_release_valid :
            maintenance_release_valid);
    assign m2_maintenance_release_ready = maintenance_lease_q &&
        maintenance_owner_m2_q && array_maintenance_release_ready;
    assign array_maintenance_tile = maintenance_owner_m2_q ?
        m2_maintenance_tile : i_maintenance_tile;
    assign array_state_write_valid = maintenance_owner_m2_q ?
        m2_state_write_valid : i_state_maintenance_write_valid;
    assign array_state_write_addr = maintenance_owner_m2_q ?
        m2_state_write_addr : i_state_maintenance_write_addr;
    assign array_state_write_data = maintenance_owner_m2_q ?
        m2_state_write_data : i_state_maintenance_write_data;
    assign array_state_write_mask = maintenance_owner_m2_q ?
        m2_state_write_mask : i_state_maintenance_write_mask;
    assign m2_state_write_ready = maintenance_owner_m2_q &&
        array_state_write_ready;
    assign o_state_maintenance_write_ready = !maintenance_owner_m2_q &&
        array_state_write_ready;
    assign array_trace_write_valid = maintenance_owner_m2_q ?
        m2_trace_write_valid : i_trace_maintenance_write_valid;
    assign array_trace_write_addr = maintenance_owner_m2_q ?
        m2_trace_write_addr : i_trace_maintenance_write_addr;
    assign array_trace_write_data = maintenance_owner_m2_q ?
        m2_trace_write_data : i_trace_maintenance_write_data;
    assign array_trace_write_mask = maintenance_owner_m2_q ?
        m2_trace_write_mask : i_trace_maintenance_write_mask;
    assign m2_trace_write_ready = maintenance_owner_m2_q &&
        array_trace_write_ready;
    assign o_trace_maintenance_write_ready = !maintenance_owner_m2_q &&
        array_trace_write_ready;
    assign o_state_maintenance_read_ready = !maintenance_owner_m2_q &&
        array_state_read_ready;
    assign o_trace_maintenance_read_ready = !maintenance_owner_m2_q &&
        array_trace_read_ready;
    assign parameter_access_safe = array_idle || maintenance_lease_q ||
        array_refill_safe;
    // M2 is the second SRNN phase and has the lowest admission priority.
    // Prevent a same-cycle handshake from racing an FC descriptor, streamed
    // backbone launch or SCNN graph launch before m2_engine_busy can retain
    // the array mux on the following cycle.
    assign m2_start_allowed = SRNN_M2_ENABLED && array_idle &&
        !array_fc_srnn_busy && !maintenance_lease_q && !subsystem_busy &&
        !graph_start_valid && !i_request_valid &&
        !i_fc_srnn_layer_valid && !srnn_group_streamer_selected &&
        !i_srnn_group_streamer_start_valid;
    assign o_srnn_h40_timestep_start_ready = m2_start_allowed &&
        m2_timestep_start_ready;
    // The engine raises busy from registered prepare/adapter state before it
    // can issue array work.  Using the incoming start/prepare handshakes here
    // would feed array error gating back into the engine's combinational ready
    // path and create a false ownership loop.
    assign m2_array_owner = SRNN_M2_ENABLED && m2_engine_busy;

    // A different client can replace these rows without changing M2's model
    // version. Only accepted physical writes evict; our own refill must not.
    assign weight_fill_context = o_cache_fill_grant == 2'd2 && topology_weight_fill_grant == 2'd2;
    assign m2_cache_evicted = 1'b0;
    assign scnn_cache_evicted = |(weight_fill_valid & weight_fill_ready) &&
        !(|(scnn_weight_fill_valid & scnn_weight_fill_ready));

    // Qualify admission only. The refill-safe signal must stay valid while
    // the admitted publisher owns the maintenance lease and drains writes.
    assign scnn_admission_allowed = !fc_refill_busy &&
        !srnn_backbone_refill_busy && !m2_engine_busy &&
        !(ENABLE_FC_CLIENT && i_fc_w0_refill_valid) &&
        !(ENABLE_SRNN_CLIENT && i_srnn_backbone_refill_valid);
    assign foreign_refill_allowed = !subsystem_busy && !m2_engine_busy &&
        !o_context_active;
    assign o_fc_w0_refill_ready = fc_refill_ready && foreign_refill_allowed;
    assign m2_prepare_allowed = !subsystem_busy && !o_context_active && !fc_refill_busy &&
        !srnn_backbone_refill_busy && !scnn_weight_fill_busy &&
        !(ENABLE_FC_CLIENT && i_fc_w0_refill_valid) &&
        !(ENABLE_SRNN_CLIENT && i_srnn_backbone_refill_valid) &&
        !(ENABLE_SCNN_CLIENT && (i_initialize_valid || i_request_valid ||
                                i_forward_start_valid));

    assign o_busy = subsystem_busy || array_busy || feature_arbiter_busy ||
        frame_service_busy || maintenance_lease_q || o_context_active || o_fc_srnn_learning_busy ||
        o_cache_refill_busy || m2_engine_busy;
    assign o_quiescent = subsystem_quiescent && array_idle &&
        !frame_service_busy && !feature_arbiter_busy && !maintenance_lease_q && !o_context_active &&
        o_fc_srnn_learning_idle && o_cache_refill_quiescent &&
        !m2_engine_busy;
    assign o_sticky_error = subsystem_error || array_error ||
        frame_service_error || feature_arbiter_error || context_bridge_error ||
        o_fc_srnn_learning_sticky_error || o_cache_refill_sticky_error ||
        topology_weight_fill_error || srnn_backbone_refill_error ||
        o_srnn_h40_sticky_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            maintenance_lease_q <= 1'b0;
            maintenance_owner_m2_q <= 1'b0;
        end else begin
            if (array_maintenance_start_valid &&
                    array_maintenance_start_ready) begin
                maintenance_lease_q <= 1'b1;
                maintenance_owner_m2_q <= m2_maintenance_start_valid;
            end else if (array_maintenance_release_valid &&
                    array_maintenance_release_ready) begin
                maintenance_lease_q <= 1'b0;
                maintenance_owner_m2_q <= 1'b0;
            end
        end
    end

    unified_array16_three_client_weight_fill_arbiter_v1 #(
        .ARRAY_WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH)
    ) u_topology_weight_fill_arbiter (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(1'b0),
        .i_clear_error(i_clear_error),
        .i_client0_busy(scnn_weight_fill_busy),
        .i_client0_fill_valid(scnn_weight_fill_valid),
        .o_client0_fill_ready(scnn_weight_fill_ready),
        .i_client0_fill_tile_flat(scnn_weight_fill_tile_flat),
        .i_client0_fill_bank(scnn_weight_fill_bank),
        .i_client0_fill_addr_flat(scnn_weight_fill_addr_flat),
        .i_client0_fill_data_flat(scnn_weight_fill_data_flat),
        .i_client1_busy(m2_engine_busy),
        .i_client1_fill_valid(m2_weight_fill_valid),
        .o_client1_fill_ready(m2_weight_fill_ready),
        .i_client1_fill_tile_flat(m2_weight_fill_tile_flat),
        .i_client1_fill_bank(m2_weight_fill_bank),
        .i_client1_fill_addr_flat(m2_weight_fill_addr_flat),
        .i_client1_fill_data_flat(m2_weight_fill_data_flat),
        .i_client2_busy(srnn_backbone_refill_busy),
        .i_client2_fill_valid(srnn_backbone_weight_fill_valid),
        .o_client2_fill_ready(srnn_backbone_weight_fill_ready),
        .i_client2_fill_tile_flat(srnn_backbone_weight_fill_tile_flat),
        .i_client2_fill_bank(srnn_backbone_weight_fill_bank),
        .i_client2_fill_addr_flat(srnn_backbone_weight_fill_addr_flat),
        .i_client2_fill_data_flat(srnn_backbone_weight_fill_data_flat),
        .o_fill_valid(topology_weight_fill_valid),
        .i_fill_ready(topology_weight_fill_ready),
        .o_fill_tile_flat(topology_weight_fill_tile_flat),
        .o_fill_bank(topology_weight_fill_bank),
        .o_fill_addr_flat(topology_weight_fill_addr_flat),
        .o_fill_data_flat(topology_weight_fill_data_flat),
        .o_grant(topology_weight_fill_grant), .o_busy(topology_weight_fill_busy),
        .o_quiescent(topology_weight_fill_quiescent),
        .o_sticky_error(topology_weight_fill_error)
    );

    generate if (ENABLE_SRNN_CLIENT) begin : g_srnn_backbone_refill
    assign o_srnn_backbone_refill_ready = backbone_refill_ready && foreign_refill_allowed;
    unified_srnn_h40_int8_cache_refill_scheduler_v1 #(
        .PARAM_DEPTH(SRNN_BACKBONE_PARAM_DEPTH),
        .PARAM_ADDR_WIDTH(SRNN_BACKBONE_PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH), .VERSION_WIDTH(VERSION_WIDTH),
        .SOURCE_COUNT_WIDTH(10), .DESTINATION_COUNT_WIDTH(10),
        .ARRAY_WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .ARRAY_TILES(PHYSICAL_TILE_COUNT)
    ) u_srnn_backbone_cache_refill (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(1'b0),
        .i_clear_error(i_clear_error),
        .i_refill_valid(i_srnn_backbone_refill_valid && foreign_refill_allowed),
        .o_refill_ready(backbone_refill_ready),
        .i_refill_all(i_srnn_backbone_refill_all),
        .i_dirty_source(i_srnn_backbone_dirty_source),
        .i_dirty_destination_tile(i_srnn_backbone_dirty_tile),
        .i_parameter_base(i_srnn_backbone_parameter_base),
        .i_cache_row_base(i_srnn_backbone_cache_row_base),
        .i_source_count(i_srnn_backbone_source_count),
        .i_destination_count(i_srnn_backbone_destination_count),
        .i_destination_tile_base(i_srnn_backbone_tile_base),
        .i_version(i_srnn_backbone_version),
        .i_array_idle(parameter_access_safe),
        .o_forward_read_valid(o_srnn_backbone_forward_read_valid),
        .i_forward_read_ready(i_srnn_backbone_forward_read_ready),
        .o_forward_read_addr(o_srnn_backbone_forward_read_addr),
        .o_forward_read_tag(o_srnn_backbone_forward_read_tag),
        .i_forward_response_valid(i_srnn_backbone_forward_response_valid),
        .o_forward_response_ready(o_srnn_backbone_forward_response_ready),
        .i_forward_response_tag(i_srnn_backbone_forward_response_tag),
        .i_forward_response_code(i_srnn_backbone_forward_response_code),
        .i_forward_response_error(i_srnn_backbone_forward_response_error),
        .o_weight_fill_valid(srnn_backbone_weight_fill_valid),
        .i_weight_fill_ready(srnn_backbone_weight_fill_ready),
        .o_weight_fill_tile_flat(srnn_backbone_weight_fill_tile_flat),
        .o_weight_fill_bank(srnn_backbone_weight_fill_bank),
        .o_weight_fill_addr_flat(srnn_backbone_weight_fill_addr_flat),
        .o_weight_fill_data_flat(srnn_backbone_weight_fill_data_flat),
        .o_refill_done(o_srnn_backbone_refill_done),
        .o_rows_published(o_srnn_backbone_rows_published),
        .o_published_version(o_srnn_backbone_published_version),
        .o_busy(srnn_backbone_refill_busy),
        .o_sticky_error(srnn_backbone_refill_error)
    );
    end else begin : g_no_srnn_backbone_refill
        assign o_srnn_backbone_refill_ready = 1'b0;
        assign o_srnn_backbone_forward_read_valid = 1'b0;
        assign o_srnn_backbone_forward_read_addr = '0;
        assign o_srnn_backbone_forward_read_tag = '0;
        assign o_srnn_backbone_forward_response_ready = 1'b0;
        assign srnn_backbone_weight_fill_valid = '0;
        assign srnn_backbone_weight_fill_tile_flat = '0;
        assign srnn_backbone_weight_fill_bank = '0;
        assign srnn_backbone_weight_fill_addr_flat = '0;
        assign srnn_backbone_weight_fill_data_flat = '0;
        assign o_srnn_backbone_refill_done = 1'b0;
        assign o_srnn_backbone_rows_published = '0;
        assign o_srnn_backbone_published_version = '0;
        assign srnn_backbone_refill_busy = 1'b0;
        assign srnn_backbone_refill_error = 1'b0;
    end endgenerate

    unified_array16_shared_weight_cache_refill_plane_v1 #(
        .PARAM_DEPTH(PARAM_DEPTH), .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH), .VERSION_WIDTH(VERSION_WIDTH),
        .ENABLE_FC_REFILL(ENABLE_FC_CLIENT),
        .FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
        .HIDDEN_COUNT_WIDTH(10),
        .ARRAY_WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .ARRAY_TILES(PHYSICAL_TILE_COUNT)
    ) u_shared_weight_cache_refill (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(1'b0),
        .i_clear_error(i_clear_error),
        .i_fc_refill_valid(i_fc_w0_refill_valid && foreign_refill_allowed),
        .o_fc_refill_ready(fc_refill_ready),
        .i_fc_refill_all(i_fc_w0_refill_all),
        .i_fc_weight_int8(i_fc_cache_refill_weight_int8),
        .i_fc_destination_tile_base(fc_cache_refill_destination_tile_base),
        .i_fc_dirty_feature(i_fc_w0_dirty_feature),
        .i_fc_dirty_tile(i_fc_w0_dirty_tile),
        .i_fc_parameter_base(i_fc_w0_parameter_base),
        .i_fc_hidden_stride(i_fc_w0_hidden_stride),
        .i_fc_cache_row_base(i_fc_cache_row_base),
        .i_fc_feature_count(i_fc_w0_feature_count),
        .i_fc_hidden_count(i_fc_w0_hidden_count),
        .i_fc_tile_count(i_fc_w0_tile_count),
        .i_fc_version(i_fc_w0_version),
        .i_array_idle(parameter_access_safe),
        .o_forward_read_valid(o_fc_cache_forward_read_valid),
        .i_forward_read_ready(i_fc_cache_forward_read_ready),
        .o_forward_read_addr(o_fc_cache_forward_read_addr),
        .o_forward_read_tag(o_fc_cache_forward_read_tag),
        .i_forward_response_valid(i_fc_cache_forward_response_valid),
        .o_forward_response_ready(o_fc_cache_forward_response_ready),
        .i_forward_response_tag(i_fc_cache_forward_response_tag),
        .i_forward_response_code(i_fc_cache_forward_response_code),
        .i_forward_response_error(i_fc_cache_forward_response_error),
        .i_topology_fill_busy(topology_weight_fill_busy),
        .i_topology_fill_valid(topology_weight_fill_valid),
        .o_topology_fill_ready(topology_weight_fill_ready),
        .i_topology_fill_tile_flat(topology_weight_fill_tile_flat),
        .i_topology_fill_bank(topology_weight_fill_bank),
        .i_topology_fill_addr_flat(topology_weight_fill_addr_flat),
        .i_topology_fill_data_flat(topology_weight_fill_data_flat),
        .o_weight_fill_valid(weight_fill_valid),
        .i_weight_fill_ready(weight_fill_ready),
        .o_weight_fill_tile_flat(weight_fill_tile_flat),
        .o_weight_fill_bank(weight_fill_bank),
        .o_weight_fill_addr_flat(weight_fill_addr_flat),
        .o_weight_fill_data_flat(weight_fill_data_flat),
        .o_fc_refill_done(o_fc_w0_refill_done),
        .o_fc_refill_busy(fc_refill_busy),
        .o_fc_rows_published(o_fc_w0_rows_published),
        .o_fc_published_version(o_fc_w0_published_version),
        .o_fill_grant(o_cache_fill_grant),
        .o_busy(o_cache_refill_busy),
        .o_quiescent(o_cache_refill_quiescent),
        .o_sticky_error(o_cache_refill_sticky_error)
    );

    generate if (ENABLE_SCNN_CLIENT) begin : g_scnn_client
    assign o_initialize_ready = scnn_initialize_ready && scnn_admission_allowed;
    assign o_request_ready = scnn_request_ready && scnn_admission_allowed;
    assign o_forward_start_ready = scnn_forward_start_ready && scnn_admission_allowed;
    unified_scnn_two_client_feature_service_arbiter_v1 u_feature_arbiter (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_c0_request_valid(prescan_request_valid),
        .o_c0_request_ready(prescan_request_ready),
        .i_c0_term_valid(prescan_term_valid),
        .i_c0_input_channel_flat(prescan_input_channel_flat),
        .i_c0_kernel_row_flat(prescan_kernel_row_flat),
        .i_c0_kernel_col_flat(prescan_kernel_col_flat),
        .i_c0_spatial_base(prescan_spatial_base),
        .i_c0_spatial_count(prescan_spatial_count),
        .i_c0_output_row(prescan_output_row),
        .i_c0_output_col(prescan_output_col),
        .i_c0_value_bit(4'd0),
        .o_c0_response_valid(prescan_response_valid),
        .i_c0_response_ready(prescan_response_ready),
        .i_c1_request_valid(core_request_valid),
        .o_c1_request_ready(core_request_ready),
        .i_c1_term_valid(core_term_valid),
        .i_c1_input_channel_flat(core_input_channel_flat),
        .i_c1_kernel_row_flat(core_kernel_row_flat),
        .i_c1_kernel_col_flat(core_kernel_col_flat),
        .i_c1_spatial_base(core_spatial_base),
        .i_c1_spatial_count(core_spatial_count),
        .i_c1_output_row(core_output_row),
        .i_c1_output_col(core_output_col),
        .i_c1_value_bit(core_value_bit),
        .o_c1_response_valid(core_response_valid),
        .i_c1_response_ready(core_response_ready),
        .o_service_request_valid(o_feature_request_valid),
        .i_service_request_ready(i_feature_request_ready),
        .o_service_term_valid(o_feature_term_valid),
        .o_service_input_channel_flat(o_feature_input_channel_flat),
        .o_service_kernel_row_flat(o_feature_kernel_row_flat),
        .o_service_kernel_col_flat(o_feature_kernel_col_flat),
        .o_service_spatial_base(o_feature_spatial_base),
        .o_service_spatial_count(o_feature_spatial_count),
        .o_service_output_row(o_feature_output_row),
        .o_service_output_col(o_feature_output_col),
        .o_service_value_bit(o_feature_value_bit),
        .i_service_response_valid(i_feature_response_valid),
        .o_service_response_ready(o_feature_response_ready),
        .i_service_mask0(i_feature_mask0), .i_service_mask1(i_feature_mask1),
        .i_service_mask2(i_feature_mask2), .i_service_mask3(i_feature_mask3),
        .i_service_error(i_feature_service_error),
        .o_response_mask0(response_mask0), .o_response_mask1(response_mask1),
        .o_response_mask2(response_mask2), .o_response_mask3(response_mask3),
        .o_response_error(response_error), .o_busy(feature_arbiter_busy),
        .o_protocol_error(feature_arbiter_error)
    );

    unified_scnn_selected_layers_context_bridge_v1 #(
        .ENABLE_FRAME_SERVICE(ENABLE_SCNN_FRAME_SERVICE),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .VERSION_WIDTH(VERSION_WIDTH),
        .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH),
        .MAX_INPUT_CHANNELS(OBS_INPUT_CHANNELS),
        .INPUT_CHANNEL_COUNT_WIDTH(CHANNEL_COUNT_WIDTH),
        .CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH)
    ) u_context_bridge (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(1'b0),
        .i_clear_error(i_clear_error),
        .i_context_load_valid(array_context_load_valid),
        .o_context_load_ready(array_context_load_ready),
        .i_context_load_weights(array_context_load_weights),
        .i_context_output_channel_base(array_context_output_base),
        .i_context_output_channel_count(array_context_output_count),
        .i_context_spatial_base(array_context_spatial_base),
        .i_context_spatial_count(array_context_spatial_count),
        .i_context_active_pair(array_context_pair),
        .i_context_work_base_addr(array_context_work_base),
        .i_context_release_valid(array_context_release_valid),
        .o_context_release_ready(array_context_release_ready),
        .i_release_output_channel_base(array_release_output_base),
        .i_release_output_channel_count(array_release_output_count),
        .i_release_spatial_base(array_release_spatial_base),
        .i_release_spatial_count(array_release_spatial_count),
        .i_release_active_pair(array_release_pair),
        .i_release_work_base_addr(array_release_work_base),
        // Cache residency tracks the published parameter image.  The graph
        // version remains the independent sample/observation identity.
        .i_graph_version(o_parameter_version),
        .i_classifier_cache_valid(o_classifier_cache_valid),
        .i_classifier_pair(o_active_classifier_pair),
        .i_classifier_version(o_parameter_version),
        .i_classifier_class_count(i_classifier_class_count),
        .i_classifier_work_base_addr(i_classifier_work_base_addr),
        .i_conv_input_channel_count(ENABLE_SCNN_FRAME_SERVICE ?
            CHANNEL_COUNT_WIDTH'(descriptor_active.in_channels) : i_total_input_channels),
        .i_conv_parameter_base(descriptor_active.parameter_base),
        .o_conv_context_valid(conv_context_valid),
        .i_conv_context_ready(conv_context_ready),
        .o_conv_context_load_weights(conv_context_load_weights),
        .o_conv_output_channel_base(conv_context_output_base),
        .o_conv_output_channel_count(conv_context_output_count),
        .o_conv_input_channel_count(conv_context_input_count),
        .o_conv_context_pair(conv_context_pair),
        .o_conv_parameter_base(conv_context_parameter_base),
        .o_conv_context_version(conv_context_version),
        .o_context_active(o_context_active),
        .o_context_is_classifier(o_context_is_classifier),
        .o_state_service_valid(o_state_service_valid),
        .i_state_service_ready(i_state_service_ready),
        .i_state_service_drained(state_service_drained),
        .o_state_service_release(o_state_service_release),
        .o_state_service_is_classifier(o_state_service_is_classifier),
        .o_state_service_output_channel_base(o_state_service_output_channel_base),
        .o_state_service_output_channel_count(o_state_service_output_channel_count),
        .o_state_service_spatial_base(o_state_service_spatial_base),
        .o_state_service_spatial_count(o_state_service_spatial_count),
        .o_state_service_active_pair(o_state_service_active_pair),
        .o_state_service_work_base_addr(o_state_service_work_base_addr),
        .o_sticky_error(context_bridge_error)
    );

    unified_scnn_selected_layers_train_infer_subsystem_v1 #(
        .USE_CONTEXT_PARAMETER_BASE(ENABLE_SCNN_FRAME_SERVICE),
        .ALLOW_UNORDERED_SURROGATE_CAPTURE(1'b1),
        .USE_INTERNAL_PARAMETER_STORE(
            USE_INTERNAL_SCNN_PARAMETER_STORE),
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH), .PARAM_DEPTH(PARAM_DEPTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .MAX_BATCH(MAX_BATCH), .MAX_TIMESTEPS(MAX_TIMESTEPS),
        .CLASSIFIER_CAPTURE_MAX_BATCH(CLASSIFIER_CAPTURE_MAX_BATCH),
        .CLASSIFIER_CAPTURE_MAX_TIMESTEPS(
            CLASSIFIER_CAPTURE_MAX_TIMESTEPS),
        .MAX_INPUT_CHANNELS(MAX_INPUT_CHANNELS),
        .MAX_OUTPUT_CHANNELS(MAX_OUTPUT_CHANNELS),
        .OBS_INPUT_CHANNELS(OBS_INPUT_CHANNELS),
        .OBS_OUTPUT_CHANNELS(OBS_OUTPUT_CHANNELS),
        // Each H40 context covers one output row; tiles serialize channels
        // inside that row before the scheduler advances to the next row.
        .CONV_SURROGATE_WAVE_CHANNELS(0),
        .CONV_SURROGATE_SPATIAL_BLOCK_SIZE(SPATIAL), .SPATIAL(SPATIAL),
        .POOLED_SPATIAL(POOLED_SPATIAL), .KERNEL_SIZE(KERNEL_SIZE),
        .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH), .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .MAX_FEATURES(MAX_FEATURES),
        .MAX_CLASSES(MAX_CLASSES), .LANES(LANES), .ENGINES(ENGINES),
        .FRAME_WIDTH(FRAME_WIDTH),
        .ARRAY_WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .SAMPLE_WIDTH(SAMPLE_WIDTH), .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .TIMESTEP_WIDTH(TIMESTEP_WIDTH),
        .TIMESTEP_COUNT_WIDTH(TIMESTEP_COUNT_WIDTH),
        .CHANNEL_COUNT_WIDTH(CHANNEL_COUNT_WIDTH),
        .FEATURE_WIDTH(FEATURE_WIDTH),
        .FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
        .CLASS_WIDTH(CLASS_WIDTH), .CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH),
        .SELECTED_LAYER(SELECTED_LAYER),
        .USE_EXTERNAL_OUTPUT_LEARNER(USE_EXTERNAL_OUTPUT_LEARNER),
        .ENABLE_SCNN_CONV8_LEARNING(ENABLE_SCNN_CONV8_LEARNING),
        .INTERNAL_MODULATION_BATCH(INTERNAL_MODULATION_BATCH)
    ) u_train_infer (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_parameter_load_valid(i_parameter_load_valid),
        .o_parameter_load_ready(o_parameter_load_ready),
        .i_parameter_load_addr(i_parameter_load_addr),
        .i_parameter_load_code(i_parameter_load_code),
        .i_parameter_load_residual(i_parameter_load_residual),
        .o_store_load_valid(o_scnn_store_load_valid),
        .i_store_load_ready(i_scnn_store_load_ready),
        .o_store_load_addr(o_scnn_store_load_addr),
        .o_store_load_code(o_scnn_store_load_code),
        .o_store_load_residual(o_scnn_store_load_residual),
        .o_store_learning_read_valid(o_scnn_store_learning_read_valid),
        .i_store_learning_read_ready(i_scnn_store_learning_read_ready),
        .o_store_learning_read_addr(o_scnn_store_learning_read_addr),
        .o_store_learning_read_tag(o_scnn_store_learning_read_tag),
        .i_store_learning_response_valid(
            i_scnn_store_learning_response_valid),
        .o_store_learning_response_ready(
            o_scnn_store_learning_response_ready),
        .i_store_learning_response_tag(i_scnn_store_learning_response_tag),
        .i_store_learning_response_code(i_scnn_store_learning_response_code),
        .i_store_learning_response_residual(
            i_scnn_store_learning_response_residual),
        .i_store_learning_response_error(
            i_scnn_store_learning_response_error),
        .o_store_learning_write_valid(o_scnn_store_learning_write_valid),
        .i_store_learning_write_ready(i_scnn_store_learning_write_ready),
        .o_store_learning_write_addr(o_scnn_store_learning_write_addr),
        .o_store_learning_write_code(o_scnn_store_learning_write_code),
        .o_store_learning_write_residual(
            o_scnn_store_learning_write_residual),
        .o_store_learning_write_code_enable(
            o_scnn_store_learning_write_code_enable),
        .o_store_learning_write_residual_enable(
            o_scnn_store_learning_write_residual_enable),
        .o_store_forward_read_valid(o_scnn_store_forward_read_valid),
        .i_store_forward_read_ready(i_scnn_store_forward_read_ready),
        .o_store_forward_read_addr(o_scnn_store_forward_read_addr),
        .o_store_forward_read_tag(o_scnn_store_forward_read_tag),
        .i_store_forward_response_valid(
            i_scnn_store_forward_response_valid),
        .o_store_forward_response_ready(
            o_scnn_store_forward_response_ready),
        .i_store_forward_response_tag(i_scnn_store_forward_response_tag),
        .i_store_forward_response_code(i_scnn_store_forward_response_code),
        .i_store_forward_response_error(i_scnn_store_forward_response_error),
        .i_store_protocol_error(i_scnn_store_protocol_error),
        .i_cache_invalidate(scnn_cache_evicted),
        .i_initialize_valid(i_initialize_valid && scnn_admission_allowed),
        .o_initialize_ready(scnn_initialize_ready),
        .o_initialize_done(o_initialize_done),
        .i_classifier_pair(i_classifier_pair),
        .i_context_model_version(i_context_model_version),
        .i_context_adapt_version(i_context_adapt_version),
        .i_system_parameter_owner(i_system_parameter_owner),
        .i_system_learning_enable(i_system_learning_enable),
        .i_system_profile_aligned(i_system_profile_aligned),
        .i_system_profile_id(i_system_profile_id),
        .i_request_valid(i_request_valid && scnn_admission_allowed),
        .o_request_ready(scnn_request_ready),
        .i_learning_enable(i_learning_enable), .i_batch_count(i_batch_count),
        .i_timestep(i_timestep), .i_timestep_count(i_timestep_count),
        .i_sample_version(i_sample_version),
        .i_labels_flat(i_labels_flat), .o_done(o_done),
        .i_profile_id(i_profile_id), .i_conv_tensor_id(i_conv_tensor_id),
        .i_classifier_tensor_id(i_classifier_tensor_id),
        .i_total_input_channels(i_total_input_channels),
        .i_total_output_channels(i_total_output_channels),
        .i_conv_parameter_base(i_conv_parameter_base),
        .i_conv_context_parameter_base(conv_context_parameter_base),
        .i_conv_weight_q16_base(i_conv_weight_q16_base),
        .i_classifier_parameter_base(i_classifier_parameter_base),
        .i_classifier_feature_count(i_classifier_feature_count),
        .i_classifier_class_count(i_classifier_class_count),
        .i_classifier_trace_leak_valid(i_classifier_trace_leak_valid),
        .i_classifier_trace_leak_q8_8(i_classifier_trace_leak_q8_8),
        .i_conv_vote_increment(i_conv_vote_increment),
        .i_conv_vote_threshold(i_conv_vote_threshold),
        .i_classifier_vote_increment(i_classifier_vote_increment),
        .i_classifier_vote_threshold(i_classifier_vote_threshold),
        .i_code_min(i_code_min), .i_code_max(i_code_max),
        .i_forward_start_valid(i_forward_start_valid && scnn_admission_allowed),
        .o_forward_start_ready(scnn_forward_start_ready),
        .o_forward_start_fire(o_forward_start_fire),
        .i_forward_done(i_forward_done), .o_forward_active(o_forward_active),
        .o_forward_model_version(o_forward_model_version),
        .o_forward_adapt_version(o_forward_adapt_version),
        .i_restore_complete(i_restore_complete),
        .o_restore_ready(o_restore_ready),
        .o_recovery_required(o_recovery_required),
        .i_conv_in_width(i_conv_in_width), .i_conv_in_height(i_conv_in_height),
        .i_conv_in_channels(i_conv_in_channels),
        .i_conv_padding(i_conv_padding), .i_conv_layer_tag(i_conv_layer_tag),
        .i_target_feature_count(i_target_feature_count),
        .i_target_trace_leak_q8_8(i_target_trace_leak_q8_8),
        .i_target_trace_state_reset(i_target_trace_state_reset),
        .o_graph_start_valid(graph_start_valid),
        .i_graph_start_ready(graph_start_ready), .o_graph_target(graph_target),
        .o_graph_sample(graph_sample), .o_graph_timestep(graph_timestep),
        .o_graph_version(graph_version), .o_graph_label(graph_label),
        .o_graph_sample_last(graph_sample_last), .i_graph_done(graph_done),
        .i_graph_error(array_error),
        .o_conv_observation_enable(conv_observation_enable),
        .o_classifier_observation_enable(classifier_observation_enable),
        .o_observation_target(observation_target),
        .o_prescan_feature_request_valid(prescan_request_valid),
        .i_prescan_feature_request_ready(prescan_request_ready),
        .o_prescan_feature_term_valid(prescan_term_valid),
        .o_prescan_feature_input_channel_flat(prescan_input_channel_flat),
        .o_prescan_feature_kernel_row_flat(prescan_kernel_row_flat),
        .o_prescan_feature_kernel_col_flat(prescan_kernel_col_flat),
        .o_prescan_feature_spatial_base(prescan_spatial_base),
        .o_prescan_feature_spatial_count(prescan_spatial_count),
        .o_prescan_feature_output_row(prescan_output_row),
        .o_prescan_feature_output_col(prescan_output_col),
        .i_prescan_feature_response_valid(prescan_response_valid),
        .o_prescan_feature_response_ready(prescan_response_ready),
        .i_prescan_feature_mask0(response_mask0),
        .i_prescan_feature_mask1(response_mask1),
        .i_prescan_feature_mask2(response_mask2),
        .i_prescan_feature_mask3(response_mask3),
        .i_prescan_feature_service_error(response_error),
        .i_surrogate_valid(surrogate_valid),
        .o_surrogate_source_ready(surrogate_ready),
        .i_surrogate_target(surrogate_target),
        .i_surrogate_sample(surrogate_sample),
        .i_surrogate_timestep(surrogate_timestep),
        .i_surrogate_version(surrogate_version),
        .i_surrogate_layer_tag(surrogate_layer_tag),
        .i_surrogate_channel(surrogate_channel),
        .i_surrogate_spatial(surrogate_spatial),
        .i_surrogate_q8_8(surrogate_q8_8),
        .i_feature_wave_valid(feature_wave_valid),
        .o_feature_wave_ready(feature_wave_ready),
        .i_feature_wave_mask(feature_wave_mask),
        .i_feature_wave_base(feature_wave_base),
        .i_feature_wave_q8_8_flat(feature_wave_flat),
        .i_feature_wave_version(feature_wave_version),
        .i_feature_wave_frame_index(feature_wave_frame),
        .i_feature_wave_first(feature_wave_first),
        .i_feature_wave_last(feature_wave_last),
        .i_feature_wave_sample_reset(feature_wave_sample_reset),
        .i_feature_wave_sample_last(feature_wave_sample_last),
        .i_feature_sample_label(feature_sample_label),
        .i_logit_wave_valid(logit_wave_valid),
        .o_logit_wave_ready(logit_wave_ready),
        .i_logit_wave_mask(logit_wave_mask), .i_logit_wave_base(logit_wave_base),
        .i_logit_wave_count(logit_wave_count),
        .i_logit_wave_q8_8_flat(logit_wave_flat),
        .i_logit_wave_version(logit_wave_version),
        .i_logit_wave_frame_index(logit_wave_frame),
        .i_logit_wave_frame_last(logit_wave_frame_last),
        .i_logit_wave_sample_last(logit_wave_sample_last),
        .i_conv_context_valid(conv_context_valid),
        .o_conv_context_ready(conv_context_ready),
        .i_conv_context_load_weights(conv_context_load_weights),
        .i_conv_output_channel_base(conv_context_output_base),
        .i_conv_output_channel_count(conv_context_output_count),
        .i_conv_input_channel_count(conv_context_input_count),
        .i_conv_context_pair(conv_context_pair),
        .i_conv_context_version(conv_context_version),
        .i_array_idle(parameter_access_safe),
        .o_weight_fill_valid(scnn_weight_fill_valid),
        .i_weight_fill_ready(scnn_weight_fill_ready),
        .o_weight_fill_tile_flat(scnn_weight_fill_tile_flat),
        .o_weight_fill_bank(scnn_weight_fill_bank),
        .o_weight_fill_addr_flat(scnn_weight_fill_addr_flat),
        .o_weight_fill_data_flat(scnn_weight_fill_data_flat),
        .o_weight_fill_busy(scnn_weight_fill_busy),
        .o_classifier_cache_valid(o_classifier_cache_valid),
        .o_active_classifier_pair(o_active_classifier_pair),
        .o_parameter_version(o_parameter_version),
        .o_conv_cache_valid(o_conv_cache_valid),
        .o_conv_cached_output_base(o_conv_cached_output_base),
        .o_conv_cached_output_count(o_conv_cached_output_count),
        .o_conv_cached_version(o_conv_cached_version),
        .o_conv_cache_invalidate(o_conv_cache_invalidate),
        .o_model_version(o_model_version), .o_adapt_version(o_adapt_version),
        .o_external_cfg_valid(o_external_cfg_valid),
        .i_external_cfg_ready(i_external_cfg_ready),
        .o_external_cfg_profile_id(o_external_cfg_profile_id),
        .o_external_cfg_tensor_id(o_external_cfg_tensor_id),
        .o_external_cfg_parameter_base(o_external_cfg_parameter_base),
        .o_external_cfg_batch_count(o_external_cfg_batch_count),
        .o_external_cfg_source_count(o_external_cfg_source_count),
        .o_external_cfg_output_count(o_external_cfg_output_count),
        .o_external_cfg_vote_increment(o_external_cfg_vote_increment),
        .o_external_cfg_vote_threshold(o_external_cfg_vote_threshold),
        .o_external_cfg_code_min(o_external_cfg_code_min),
        .o_external_cfg_code_max(o_external_cfg_code_max),
        .i_external_cfg_error(i_external_cfg_error),
        .i_external_active(i_external_active),
        .o_external_release(o_external_release),
        .i_external_release_ready(i_external_release_ready),
        .o_external_source_trace_valid(o_external_source_trace_valid),
        .i_external_source_trace_ready(i_external_source_trace_ready),
        .o_external_source_trace_sample_id(o_external_source_trace_sample_id),
        .o_external_source_trace_index(o_external_source_trace_index),
        .o_external_source_trace_q016(o_external_source_trace_q016),
        .i_external_source_trace_error(i_external_source_trace_error),
        .o_external_logit_row_valid(o_external_logit_row_valid),
        .i_external_logit_row_ready(i_external_logit_row_ready),
        .o_external_logit_sample_id(o_external_logit_sample_id),
        .o_external_logit_label(o_external_logit_label),
        .o_external_logit_valid(o_external_logit_valid),
        .i_external_logit_ready(i_external_logit_ready),
        .o_external_logit_q6(o_external_logit_q6),
        .i_external_logit_error(i_external_logit_error),
        .i_external_logit_row_done(i_external_logit_row_done),
        .o_external_observation_done(o_external_observation_done),
        .i_external_observation_done_ready(i_external_observation_done_ready),
        .i_external_window_done(i_external_window_done),
        .i_external_window_busy(i_external_window_busy),
        .i_external_context_error(i_external_context_error),
        .i_external_gradient_overflow(i_external_gradient_overflow),
        .i_external_update_config_error(i_external_update_config_error),
        .i_external_request_error(i_external_request_error),
        .i_external_store_error(i_external_store_error),
        .i_external_internal_error(i_external_internal_error),
        .o_capture_complete(o_capture_complete),
        .o_learning_start_fire(o_learning_start_fire),
        .o_commit_pulse(o_commit_pulse),
        .o_commit_code_dirty(o_commit_code_dirty),
        .o_commit_optimizer_dirty(o_commit_optimizer_dirty),
        .o_detach_pulse(o_detach_pulse), .o_commit_pending(o_commit_pending),
        .o_inference_hold(o_inference_hold), .o_initialized(o_initialized),
        .o_versions_aligned(o_versions_aligned), .o_busy(subsystem_busy),
        .o_quiescent(subsystem_quiescent), .o_sticky_error(subsystem_error)
    );
    end else begin : g_no_scnn_client
        assign prescan_request_ready = 1'b0;
        assign prescan_response_valid = 1'b0;
        assign core_request_ready = 1'b0;
        assign core_response_valid = 1'b0;
        assign o_feature_request_valid = 1'b0;
        assign o_feature_term_valid = '0;
        assign o_feature_input_channel_flat = '0;
        assign o_feature_kernel_row_flat = '0;
        assign o_feature_kernel_col_flat = '0;
        assign o_feature_spatial_base = '0;
        assign o_feature_spatial_count = '0;
        assign o_feature_output_row = '0;
        assign o_feature_output_col = '0;
        assign o_feature_value_bit = '0;
        assign o_feature_response_ready = 1'b0;
        assign response_mask0 = '0;
        assign response_mask1 = '0;
        assign response_mask2 = '0;
        assign response_mask3 = '0;
        assign response_error = 1'b0;
        assign feature_arbiter_busy = 1'b0;
        assign feature_arbiter_error = 1'b0;

        assign array_context_load_ready = 1'b0;
        assign array_context_release_ready = 1'b0;
        assign conv_context_valid = 1'b0;
        assign conv_context_load_weights = 1'b0;
        assign conv_context_output_base = '0;
        assign conv_context_output_count = '0;
        assign conv_context_input_count = '0;
        assign conv_context_pair = 1'b0;
        assign conv_context_parameter_base = '0;
        assign conv_context_version = '0;
        assign o_context_active = 1'b0;
        assign o_context_is_classifier = 1'b0;
        assign context_bridge_error = 1'b0;
        assign o_state_service_valid = 1'b0;
        assign o_state_service_release = 1'b0;
        assign o_state_service_is_classifier = 1'b0;
        assign o_state_service_output_channel_base = '0;
        assign o_state_service_output_channel_count = '0;
        assign o_state_service_spatial_base = '0;
        assign o_state_service_spatial_count = '0;
        assign o_state_service_active_pair = 1'b0;
        assign o_state_service_work_base_addr = '0;

        assign o_parameter_load_ready = 1'b0;
        assign o_scnn_store_load_valid = 1'b0;
        assign o_scnn_store_load_addr = '0;
        assign o_scnn_store_load_code = '0;
        assign o_scnn_store_load_residual = '0;
        assign o_scnn_store_learning_read_valid = 1'b0;
        assign o_scnn_store_learning_read_addr = '0;
        assign o_scnn_store_learning_read_tag = '0;
        assign o_scnn_store_learning_response_ready = 1'b0;
        assign o_scnn_store_learning_write_valid = 1'b0;
        assign o_scnn_store_learning_write_addr = '0;
        assign o_scnn_store_learning_write_code = '0;
        assign o_scnn_store_learning_write_residual = '0;
        assign o_scnn_store_learning_write_code_enable = 1'b0;
        assign o_scnn_store_learning_write_residual_enable = 1'b0;
        assign o_scnn_store_forward_read_valid = 1'b0;
        assign o_scnn_store_forward_read_addr = '0;
        assign o_scnn_store_forward_read_tag = '0;
        assign o_scnn_store_forward_response_ready = 1'b0;
        assign o_initialize_ready = 1'b0;
        assign o_initialize_done = 1'b0;
        assign o_request_ready = 1'b0;
        assign o_done = 1'b0;
        assign o_forward_start_ready = 1'b0;
        assign o_forward_start_fire = 1'b0;
        assign o_forward_active = 1'b0;
        assign o_forward_model_version = '0;
        assign o_forward_adapt_version = '0;
        assign o_restore_ready = 1'b0;
        assign o_recovery_required = 1'b0;

        assign graph_start_valid = 1'b0;
        assign graph_target = 1'b0;
        assign graph_sample = '0;
        assign graph_timestep = '0;
        assign graph_version = '0;
        assign graph_label = '0;
        assign graph_sample_last = 1'b0;
        assign conv_observation_enable = 1'b0;
        assign classifier_observation_enable = 1'b0;
        assign observation_target = 1'b0;
        assign prescan_request_valid = 1'b0;
        assign prescan_term_valid = '0;
        assign prescan_input_channel_flat = '0;
        assign prescan_kernel_row_flat = '0;
        assign prescan_kernel_col_flat = '0;
        assign prescan_spatial_base = '0;
        assign prescan_spatial_count = '0;
        assign prescan_output_row = '0;
        assign prescan_output_col = '0;
        assign prescan_response_ready = 1'b0;
        assign surrogate_ready = 1'b0;
        assign feature_wave_ready = 1'b0;
        assign logit_wave_ready = 1'b0;
        assign conv_context_ready = 1'b0;

        assign scnn_weight_fill_valid = '0;
        assign scnn_weight_fill_tile_flat = '0;
        assign scnn_weight_fill_bank = '0;
        assign scnn_weight_fill_addr_flat = '0;
        assign scnn_weight_fill_data_flat = '0;
        assign scnn_weight_fill_busy = 1'b0;
        assign o_classifier_cache_valid = 1'b0;
        assign o_active_classifier_pair = 1'b0;
        assign o_parameter_version = '0;
        assign o_conv_cache_valid = 1'b0;
        assign o_conv_cached_output_base = '0;
        assign o_conv_cached_output_count = '0;
        assign o_conv_cached_version = '0;
        assign o_conv_cache_invalidate = 1'b0;
        assign o_model_version = '0;
        assign o_adapt_version = '0;

        assign o_external_cfg_valid = 1'b0;
        assign o_external_cfg_profile_id = '0;
        assign o_external_cfg_tensor_id = '0;
        assign o_external_cfg_parameter_base = '0;
        assign o_external_cfg_batch_count = '0;
        assign o_external_cfg_source_count = '0;
        assign o_external_cfg_output_count = '0;
        assign o_external_cfg_vote_increment = '0;
        assign o_external_cfg_vote_threshold = '0;
        assign o_external_cfg_code_min = '0;
        assign o_external_cfg_code_max = '0;
        assign o_external_release = 1'b0;
        assign o_external_source_trace_valid = 1'b0;
        assign o_external_source_trace_sample_id = '0;
        assign o_external_source_trace_index = '0;
        assign o_external_source_trace_q016 = '0;
        assign o_external_logit_row_valid = 1'b0;
        assign o_external_logit_sample_id = '0;
        assign o_external_logit_label = '0;
        assign o_external_logit_valid = 1'b0;
        assign o_external_logit_q6 = '0;
        assign o_external_observation_done = 1'b0;
        assign o_capture_complete = 1'b0;
        assign o_learning_start_fire = 1'b0;
        assign o_commit_pulse = 1'b0;
        assign o_commit_code_dirty = 1'b0;
        assign o_commit_optimizer_dirty = 1'b0;
        assign o_detach_pulse = 1'b0;
        assign o_commit_pending = 1'b0;
        assign o_inference_hold = 1'b0;
        assign o_initialized = 1'b0;
        assign o_versions_aligned = 1'b0;
        assign subsystem_busy = 1'b0;
        assign subsystem_quiescent = 1'b1;
        assign subsystem_error = 1'b0;
    end endgenerate

    generate
    if (SRNN_M2_ENABLED) begin : g_srnn_m2_h40_client
        assign o_srnn_h40_prepare_ready = m2_prepare_ready && m2_prepare_allowed;
        unified_srnn_m2_h40_forward_engine_v1 #(
            .M2_CACHE_DEPTH(M2_CACHE_DEPTH),
            .PARAM_DEPTH(SRNN_M2_PARAM_DEPTH),
            .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
            .BATCH_MAX(SRNN_M2_BATCH_MAX),
            .TIMESTEP_MAX(SRNN_M2_TIMESTEP_MAX),
            .BACKBONE_MAX(SRNN_M2_BACKBONE_MAX),
            .HIDDEN_MAX(SRNN_M2_HIDDEN_MAX),
            .CLASS_MAX(SRNN_M2_CLASS_MAX),
            .WEIGHT_DEPTH(WEIGHT_DEPTH),
            .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
            .MEMORY_DEPTH(MEMORY_DEPTH),
            .MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH),
            .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH),
            .PHYSICAL_TILE_COUNT(PHYSICAL_TILE_COUNT), .OUTPUT_TILE(7)
        ) u_srnn_m2_h40_forward (
            .i_clk(i_clk), .i_rst_n(i_rst_n),
            .i_clear_error(i_clear_error), .i_abort(1'b0),
            .i_cfg_valid(i_srnn_h40_cfg_valid),
            .o_cfg_ready(o_srnn_h40_cfg_ready),
            .i_cfg_batch_count(i_srnn_h40_cfg_batch_count),
            .i_cfg_backbone_count(i_srnn_h40_cfg_backbone_count),
            .i_cfg_hidden_count(i_srnn_h40_cfg_hidden_count),
            .i_cfg_class_count(i_srnn_h40_cfg_class_count),
            .i_cfg_wh1_parameter_base(
                i_srnn_h40_cfg_wh1_parameter_base),
            .i_cfg_wout_parameter_base(
                i_srnn_h40_cfg_wout_parameter_base),
            .i_cfg_wh1_cache_base(10'd0), .i_cfg_wout_cache_base(10'd0),
            .i_cfg_hidden_work_base('0), .i_cfg_output_work_base('0),
            .o_cfg_error(o_srnn_h40_cfg_error),
            .i_prepare_valid(i_srnn_h40_prepare_valid && m2_prepare_allowed),
            .o_prepare_ready(m2_prepare_ready),
            .i_prepare_version(i_srnn_h40_prepare_version),
            .i_cache_invalidate(i_srnn_h40_cache_invalidate || m2_cache_evicted),
            .i_wh1_cache_invalidate(i_srnn_h40_wh1_cache_invalidate),
            .o_prepare_done(o_srnn_h40_prepare_done),
            .o_cache_valid(o_srnn_h40_cache_valid), .o_cache_version(),
            .i_timestep_start(i_srnn_h40_timestep_start &&
                m2_start_allowed),
            .o_timestep_start_ready(m2_timestep_start_ready),
            .i_timestep_id(i_srnn_h40_timestep_id),
            .i_sequence_version(i_srnn_h40_sequence_version),
            .i_include_target_path(i_srnn_h40_include_target_path),
            .i_observation_valid(i_srnn_h40_observation_valid),
            .o_observation_ready(o_srnn_h40_observation_ready),
            .i_observation_path(i_srnn_h40_observation_path),
            .i_observation_sample_id(i_srnn_h40_observation_sample_id),
            .i_observation_backbone_id(
                i_srnn_h40_observation_backbone_id),
            .i_observation_q14(i_srnn_h40_observation_q14),
            .i_observation_last(i_srnn_h40_observation_last),
            .i_observation_done(i_srnn_h40_observation_done),
            .o_forward_read_valid(o_srnn_h40_forward_read_valid),
            .i_forward_read_ready(i_srnn_h40_forward_read_ready),
            .o_forward_read_addr(o_srnn_h40_forward_read_addr),
            .o_forward_read_tag(o_srnn_h40_forward_read_tag),
            .i_forward_response_valid(i_srnn_h40_forward_response_valid),
            .o_forward_response_ready(o_srnn_h40_forward_response_ready),
            .i_forward_response_tag(i_srnn_h40_forward_response_tag),
            .i_forward_response_code(i_srnn_h40_forward_response_code),
            .i_forward_response_error(i_srnn_h40_forward_response_error),
            // Reuse the same refill-safety definition as the FC cache client.
            // A cache write can make the physical array report non-idle even
            // though the maintenance lease keeps compute mutually exclusive.
            .i_array_idle(parameter_access_safe),
            .o_weight_fill_valid(m2_weight_fill_valid),
            .i_weight_fill_ready(m2_weight_fill_ready),
            .o_weight_fill_tile_flat(m2_weight_fill_tile_flat),
            .o_weight_fill_bank(m2_weight_fill_bank),
            .o_weight_fill_addr_flat(m2_weight_fill_addr_flat),
            .o_weight_fill_data_flat(m2_weight_fill_data_flat),
            .o_maintenance_start_valid(m2_maintenance_start_valid),
            .i_maintenance_start_ready(m2_maintenance_start_ready),
            .o_maintenance_release_valid(m2_maintenance_release_valid),
            .i_maintenance_release_ready(m2_maintenance_release_ready),
            .o_maintenance_tile(m2_maintenance_tile),
            .o_state_write_valid(m2_state_write_valid),
            .i_state_write_ready(m2_state_write_ready),
            .o_state_write_addr(m2_state_write_addr),
            .o_state_write_data(m2_state_write_data),
            .o_state_write_mask(m2_state_write_mask),
            .o_trace_write_valid(m2_trace_write_valid),
            .i_trace_write_ready(m2_trace_write_ready),
            .o_trace_write_addr(m2_trace_write_addr),
            .o_trace_write_data(m2_trace_write_data),
            .o_trace_write_mask(m2_trace_write_mask),
            .o_layer_valid(m2_layer_valid), .i_layer_ready(m2_layer_ready),
            .o_layer_mode(m2_layer_mode),
            .o_layer_neuron_count(m2_layer_neuron_count),
            .o_layer_feedforward_group_count(
                m2_layer_feedforward_group_count),
            .o_layer_recurrent_group_count(m2_layer_recurrent_group_count),
            .o_layer_recurrent_enable(m2_layer_recurrent_enable),
            .o_layer_leak(m2_layer_leak),
            .o_layer_threshold(m2_layer_threshold),
            .o_layer_reset_to_zero(m2_layer_reset_to_zero),
            .o_layer_trace_decay(m2_layer_trace_decay),
            .o_layer_transaction_tag(m2_layer_transaction_tag),
            .o_layer_work_base_addr(m2_layer_work_base_addr),
            .o_layer_destination_tile_base(m2_layer_destination_tile_base),
            .o_group_valid(m2_group_valid), .i_group_ready(m2_group_ready),
            .o_group_recurrent(m2_group_recurrent),
            .o_group_source_valid(m2_group_source_valid),
            .o_group_weight_addr0(m2_group_weight_addr0),
            .o_group_weight_addr1(m2_group_weight_addr1),
            .o_group_weight_addr2(m2_group_weight_addr2),
            .o_group_weight_addr3(m2_group_weight_addr3),
            .i_layer_done(array_fc_srnn_layer_done && m2_array_owner),
            .i_layer_sticky_error(array_fc_srnn_sticky_error &&
                m2_array_owner),
            .i_array_observation_valid(array_fc_srnn_observation_valid &&
                m2_array_owner),
            .o_array_observation_ready(m2_array_observation_ready),
            .i_array_observation_mode(array_fc_srnn_observation_mode),
            .i_array_observation_transaction_tag(
                array_fc_srnn_observation_transaction_tag),
            .i_array_observation_neuron_id(
                array_fc_srnn_observation_neuron_id),
            .i_array_observation_spike(array_fc_srnn_observation_spike),
            .i_array_observation_pre_state(
                array_fc_srnn_observation_pre_state),
            .i_array_observation_trace_next(
                array_fc_srnn_observation_trace_next),
            .i_array_observation_last_in_record(
                array_fc_srnn_observation_last_in_record),
            .o_hidden_valid(o_srnn_h40_hidden_valid),
            .i_hidden_ready(i_srnn_h40_hidden_ready),
            .o_hidden_path(o_srnn_h40_hidden_path),
            .o_hidden_sample_id(o_srnn_h40_hidden_sample_id),
            .o_hidden_id(o_srnn_h40_hidden_id),
            .o_hidden_membrane_pre_q14(
                o_srnn_h40_hidden_membrane_pre_q14),
            .o_hidden_membrane_next_q14(
                o_srnn_h40_hidden_membrane_next_q14),
            .o_hidden_spike(o_srnn_h40_hidden_spike),
            .o_hidden_trace_q14(o_srnn_h40_hidden_trace_q14),
            .o_hidden_output_feature_q14(
                o_srnn_h40_hidden_output_feature_q14),
            .o_hidden_last(o_srnn_h40_hidden_last),
            .o_logit_valid(o_srnn_h40_logit_valid),
            .i_logit_ready(i_srnn_h40_logit_ready),
            .o_logit_sample_id(o_srnn_h40_logit_sample_id),
            .o_logit_class_id(o_srnn_h40_logit_class_id),
            .o_logit_q28(o_srnn_h40_logit_q28),
            .o_logit_last(o_srnn_h40_logit_last),
            .o_timestep_id(o_srnn_h40_timestep_id),
            .o_timestep_done(o_srnn_h40_timestep_done),
            .o_busy(m2_engine_busy),
            .o_sticky_error(o_srnn_h40_sticky_error),
            .o_protocol_error(o_srnn_h40_protocol_error),
            .o_overflow(o_srnn_h40_overflow)
        );
    end else begin : g_no_srnn_m2_h40_client
        assign o_srnn_h40_cfg_ready = 1'b0;
        assign o_srnn_h40_cfg_error = 1'b0;
        assign o_srnn_h40_prepare_ready = 1'b0;
        assign o_srnn_h40_prepare_done = 1'b0;
        assign o_srnn_h40_cache_valid = 1'b0;
        assign o_srnn_h40_forward_read_valid = 1'b0;
        assign o_srnn_h40_forward_read_addr = '0;
        assign o_srnn_h40_forward_read_tag = '0;
        assign o_srnn_h40_forward_response_ready = 1'b0;
        assign m2_timestep_start_ready = 1'b0;
        assign o_srnn_h40_observation_ready = 1'b0;
        assign o_srnn_h40_hidden_valid = 1'b0;
        assign o_srnn_h40_hidden_path = 1'b0;
        assign o_srnn_h40_hidden_sample_id = '0;
        assign o_srnn_h40_hidden_id = '0;
        assign o_srnn_h40_hidden_membrane_pre_q14 = '0;
        assign o_srnn_h40_hidden_membrane_next_q14 = '0;
        assign o_srnn_h40_hidden_spike = 1'b0;
        assign o_srnn_h40_hidden_trace_q14 = '0;
        assign o_srnn_h40_hidden_output_feature_q14 = '0;
        assign o_srnn_h40_hidden_last = 1'b0;
        assign o_srnn_h40_logit_valid = 1'b0;
        assign o_srnn_h40_logit_sample_id = '0;
        assign o_srnn_h40_logit_class_id = '0;
        assign o_srnn_h40_logit_q28 = '0;
        assign o_srnn_h40_logit_last = 1'b0;
        assign o_srnn_h40_timestep_id = '0;
        assign o_srnn_h40_timestep_done = 1'b0;
        assign m2_engine_busy = 1'b0;
        assign o_srnn_h40_sticky_error = 1'b0;
        assign o_srnn_h40_protocol_error = 1'b0;
        assign o_srnn_h40_overflow = 1'b0;
        assign m2_weight_fill_valid = '0;
        assign m2_weight_fill_tile_flat = '0;
        assign m2_weight_fill_bank = '0;
        assign m2_weight_fill_addr_flat = '0;
        assign m2_weight_fill_data_flat = '0;
        assign m2_maintenance_start_valid = 1'b0;
        assign m2_maintenance_release_valid = 1'b0;
        assign m2_maintenance_tile = '0;
        assign m2_state_write_valid = 1'b0;
        assign m2_state_write_addr = '0;
        assign m2_state_write_data = '0;
        assign m2_state_write_mask = '0;
        assign m2_trace_write_valid = 1'b0;
        assign m2_trace_write_addr = '0;
        assign m2_trace_write_data = '0;
        assign m2_trace_write_mask = '0;
        assign m2_layer_valid = 1'b0;
        assign m2_layer_mode = '0;
        assign m2_layer_neuron_count = '0;
        assign m2_layer_feedforward_group_count = '0;
        assign m2_layer_recurrent_group_count = '0;
        assign m2_layer_recurrent_enable = 1'b0;
        assign m2_layer_leak = '0;
        assign m2_layer_threshold = '0;
        assign m2_layer_reset_to_zero = 1'b0;
        assign m2_layer_trace_decay = '0;
        assign m2_layer_transaction_tag = '0;
        assign m2_layer_work_base_addr = '0;
        assign m2_layer_destination_tile_base = '0;
        assign m2_group_valid = 1'b0;
        assign m2_group_recurrent = 1'b0;
        assign m2_group_source_valid = '0;
        assign m2_group_weight_addr0 = '0;
        assign m2_group_weight_addr1 = '0;
        assign m2_group_weight_addr2 = '0;
        assign m2_group_weight_addr3 = '0;
        assign m2_array_observation_ready = 1'b0;
    end
    endgenerate

    assign o_srnn_h40_busy = m2_engine_busy;

    generate if (ENABLE_FC_CLIENT || ENABLE_SRNN_CLIENT) begin : g_fc_srnn_ingress
    unified_fc_srnn_h40_tp_observation_ingress_v1 #(
        .MAX_NEURONS(PHYSICAL_TILE_COUNT * 40),
        .SAMPLE_ID_WIDTH(6), .NEURON_ID_WIDTH(10),
        .COUNT_WIDTH(10), .FC_VMEM_WIDTH(18), .FC_VMEM_FRAC_BITS(10)
    ) u_fc_srnn_learning_ingress (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(1'b0),
        .i_clear_error(i_clear_error),
        .i_cfg_valid(i_fc_srnn_learning_cfg_valid),
        .o_cfg_ready(o_fc_srnn_learning_cfg_ready),
        .i_cfg_mode(i_fc_srnn_learning_cfg_mode),
        .i_cfg_transaction_tag(i_fc_srnn_learning_cfg_tag),
        .i_cfg_sample_id(i_fc_srnn_learning_sample_id),
        .i_cfg_neuron_base(i_fc_srnn_learning_neuron_base),
        .i_cfg_neuron_count(i_fc_srnn_learning_neuron_count),
        .i_cfg_fc_threshold(i_fc_srnn_learning_fc_threshold),
        .i_cfg_fc_output_layer(i_fc_srnn_learning_fc_output_layer),
        .i_cfg_fc_label(i_fc_srnn_learning_fc_label),
        .i_cfg_fc_logit_scale_enable(i_fc_srnn_learning_fc_logit_scale_enable),
        .i_cfg_fc_logit_scale_q16(i_fc_srnn_learning_fc_logit_scale_q16),
        .i_cfg_fc_logit_accumulate(i_fc_srnn_learning_fc_logit_accumulate),
        .i_cfg_fc_sample_start(i_fc_srnn_learning_fc_sample_start),
        .i_cfg_srnn_path(i_fc_srnn_learning_srnn_path),
        .i_scalar_valid(array_fc_srnn_observation_valid && !m2_array_owner),
        .o_scalar_ready(fc_srnn_learning_scalar_ready),
        .i_scalar_mode(array_fc_srnn_observation_mode),
        .i_scalar_transaction_tag(
            array_fc_srnn_observation_transaction_tag),
        .i_scalar_neuron_id(array_fc_srnn_observation_neuron_id),
        .i_scalar_spike(array_fc_srnn_observation_spike),
        .i_scalar_pre_state(array_fc_srnn_observation_pre_state),
        .i_scalar_trace_next(array_fc_srnn_observation_trace_next),
        .o_fc_student_trace_valid(o_fc_student_trace_valid),
        .i_fc_student_trace_ready(i_fc_student_trace_ready),
        .o_fc_student_trace_sample(o_fc_student_trace_sample),
        .o_fc_student_trace_hidden(o_fc_student_trace_hidden),
        .o_fc_student_trace_q016(o_fc_student_trace_q016),
        .o_fc_surrogate_valid(o_fc_surrogate_valid),
        .i_fc_surrogate_ready(i_fc_surrogate_ready),
        .o_fc_surrogate_sample(o_fc_surrogate_sample),
        .o_fc_surrogate_hidden(o_fc_surrogate_hidden),
        .o_fc_surrogate_q016(o_fc_surrogate_q016),
        .o_fc_logit_row_valid(o_fc_logit_row_valid),
        .i_fc_logit_row_ready(i_fc_logit_row_ready),
        .o_fc_logit_sample(o_fc_logit_sample),
        .o_fc_logit_label(o_fc_logit_label),
        .o_fc_logit_valid(o_fc_logit_valid),
        .i_fc_logit_ready(i_fc_logit_ready),
        .o_fc_logit_q6(o_fc_logit_q6),
        .o_fc_logit_last(o_fc_logit_last),
        .o_srnn_backbone_valid(o_srnn_backbone_valid),
        .i_srnn_backbone_ready(i_srnn_backbone_ready),
        .o_srnn_backbone_path(o_srnn_backbone_path),
        .o_srnn_backbone_sample_id(o_srnn_backbone_sample_id),
        .o_srnn_backbone_id(o_srnn_backbone_id),
        .o_srnn_backbone_q14(o_srnn_backbone_q14),
        .o_srnn_backbone_last(o_srnn_backbone_last),
        .o_capture_done(o_fc_srnn_learning_capture_done),
        .o_done(o_fc_srnn_learning_done),
        .o_busy(o_fc_srnn_learning_busy),
        .o_idle(o_fc_srnn_learning_idle),
        .o_sticky_error(o_fc_srnn_learning_sticky_error),
        .o_error_source(o_fc_srnn_learning_error_source),
        .o_error_code(o_fc_srnn_learning_error_code)
    );
    end else begin : g_no_fc_srnn_ingress
        assign o_fc_srnn_learning_cfg_ready = 1'b0;
        assign fc_srnn_learning_scalar_ready = 1'b0;
        assign o_fc_student_trace_valid = 1'b0;
        assign o_fc_student_trace_sample = '0;
        assign o_fc_student_trace_hidden = '0;
        assign o_fc_student_trace_q016 = '0;
        assign o_fc_surrogate_valid = 1'b0;
        assign o_fc_surrogate_sample = '0;
        assign o_fc_surrogate_hidden = '0;
        assign o_fc_surrogate_q016 = '0;
        assign o_fc_logit_row_valid = 1'b0;
        assign o_fc_logit_sample = '0;
        assign o_fc_logit_label = '0;
        assign o_fc_logit_valid = 1'b0;
        assign o_fc_logit_q6 = '0;
        assign o_fc_logit_last = 1'b0;
        assign o_srnn_backbone_valid = 1'b0;
        assign o_srnn_backbone_path = 1'b0;
        assign o_srnn_backbone_sample_id = '0;
        assign o_srnn_backbone_id = '0;
        assign o_srnn_backbone_q14 = '0;
        assign o_srnn_backbone_last = 1'b0;
        assign o_fc_srnn_learning_capture_done = 1'b0;
        assign o_fc_srnn_learning_done = 1'b0;
        assign o_fc_srnn_learning_busy = 1'b0;
        assign o_fc_srnn_learning_idle = 1'b1;
        assign o_fc_srnn_learning_sticky_error = 1'b0;
        assign o_fc_srnn_learning_error_source = '0;
        assign o_fc_srnn_learning_error_code = '0;
    end endgenerate

    // Once a streamed backbone transaction is admitted, retain array
    // ownership until the streamer has observed layer completion.  The
    // profile selector is an admission qualifier, not a cycle-by-cycle mux
    // control that software must hold throughout all 288 source groups.
    assign srnn_group_streamer_selected = SRNN_STREAM_ENABLED &&
        (i_srnn_group_streamer_select || srnn_streamer_busy);
    assign o_srnn_group_streamer_start_ready =
        srnn_group_streamer_selected && srnn_streamer_start_ready;
    assign o_srnn_group_streamer_done = srnn_streamer_done;
    assign o_srnn_group_streamer_busy = srnn_streamer_busy;
    assign o_srnn_group_streamer_sticky_error =
        srnn_streamer_sticky_error;
    assign o_srnn_group_streamer_error_code = srnn_streamer_error_code;

    generate
        if (SRNN_STREAM_ENABLED) begin : g_srnn_group_streamer
            unified_srnn_h40_p4_group_streamer_v1 #(
                .FEEDFORWARD_MAX(SRNN_FEEDFORWARD_SOURCE_MAX),
                .RECURRENT_MAX(SRNN_RECURRENT_SOURCE_MAX),
                .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
                .WEIGHT_ROW_DEPTH(WEIGHT_DEPTH / 2),
                .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH), .ARRAY_TILES(12)
            ) u_srnn_group_streamer (
                .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(1'b0),
                .i_clear_error(i_clear_error),
                .i_start_valid(i_srnn_group_streamer_start_valid &&
                    i_srnn_group_streamer_select),
                .o_start_ready(srnn_streamer_start_ready),
                .i_feedforward_source_count(
                    i_srnn_group_feedforward_source_count),
                .i_recurrent_source_count(
                    i_srnn_group_recurrent_source_count),
                .i_feedforward_cache_row_base(
                    i_srnn_group_feedforward_cache_row_base),
                .i_recurrent_cache_row_base(
                    i_srnn_group_recurrent_cache_row_base),
                .i_recurrent_enable(i_fc_srnn_recurrent_enable),
                .i_feedforward_spike_flat(
                    i_srnn_group_feedforward_spike_flat),
                .i_recurrent_spike_flat(i_srnn_group_recurrent_spike_flat),
                .i_neuron_count(i_fc_srnn_neuron_count),
                .i_leak(i_fc_srnn_leak),
                .i_threshold(i_fc_srnn_threshold),
                .i_reset_to_zero(i_fc_srnn_reset_to_zero),
                .i_trace_decay(i_fc_srnn_trace_decay),
                .i_transaction_tag(i_fc_srnn_transaction_tag),
                .i_work_base_addr(i_fc_srnn_work_base_addr),
                .i_destination_tile_base(4'd0),
                .o_layer_valid(srnn_streamer_layer_valid),
                .i_layer_ready(srnn_streamer_layer_ready),
                .o_feedforward_group_count(
                    srnn_streamer_feedforward_group_count),
                .o_recurrent_group_count(
                    srnn_streamer_recurrent_group_count),
                .o_recurrent_enable(srnn_streamer_recurrent_enable),
                .o_neuron_count(srnn_streamer_neuron_count),
                .o_leak(srnn_streamer_leak),
                .o_threshold(srnn_streamer_threshold),
                .o_reset_to_zero(srnn_streamer_reset_to_zero),
                .o_trace_decay(srnn_streamer_trace_decay),
                .o_transaction_tag(srnn_streamer_transaction_tag),
                .o_work_base_addr(srnn_streamer_work_base_addr),
                .o_destination_tile_base(
                    srnn_streamer_destination_tile_base),
                .o_group_valid(srnn_streamer_group_valid),
                .i_group_ready(srnn_streamer_group_ready),
                .o_group_recurrent(srnn_streamer_group_recurrent),
                .o_group_source_valid(srnn_streamer_group_source_valid),
                .o_group_weight_addr0(srnn_streamer_group_weight_addr0),
                .o_group_weight_addr1(srnn_streamer_group_weight_addr1),
                .o_group_weight_addr2(srnn_streamer_group_weight_addr2),
                .o_group_weight_addr3(srnn_streamer_group_weight_addr3),
                .i_layer_done(array_fc_srnn_layer_done &&
                    !m2_array_owner && srnn_group_streamer_selected),
                .o_done(srnn_streamer_done), .o_busy(srnn_streamer_busy),
                .o_idle(), .o_sticky_error(srnn_streamer_sticky_error),
                .o_error_code(srnn_streamer_error_code)
            );
        end else begin : g_no_srnn_group_streamer
            assign srnn_streamer_start_ready = 1'b0;
            assign srnn_streamer_done = 1'b0;
            assign srnn_streamer_busy = 1'b0;
            assign srnn_streamer_sticky_error = 1'b0;
            assign srnn_streamer_error_code = '0;
            assign srnn_streamer_layer_valid = 1'b0;
            assign srnn_streamer_feedforward_group_count = '0;
            assign srnn_streamer_recurrent_group_count = '0;
            assign srnn_streamer_recurrent_enable = 1'b0;
            assign srnn_streamer_neuron_count = '0;
            assign srnn_streamer_leak = '0;
            assign srnn_streamer_threshold = '0;
            assign srnn_streamer_reset_to_zero = 1'b0;
            assign srnn_streamer_trace_decay = '0;
            assign srnn_streamer_transaction_tag = '0;
            assign srnn_streamer_work_base_addr = '0;
            assign srnn_streamer_destination_tile_base = '0;
            assign srnn_streamer_group_valid = 1'b0;
            assign srnn_streamer_group_recurrent = 1'b0;
            assign srnn_streamer_group_source_valid = '0;
            assign srnn_streamer_group_weight_addr0 = '0;
            assign srnn_streamer_group_weight_addr1 = '0;
            assign srnn_streamer_group_weight_addr2 = '0;
            assign srnn_streamer_group_weight_addr3 = '0;
        end
    endgenerate

    assign m2_layer_ready = m2_array_owner && array_fc_srnn_layer_ready;
    assign m2_group_ready = m2_array_owner && array_fc_srnn_group_ready;
    assign srnn_streamer_layer_ready = !m2_array_owner &&
        srnn_group_streamer_selected && array_fc_srnn_layer_ready;
    assign srnn_streamer_group_ready = !m2_array_owner &&
        srnn_group_streamer_selected && array_fc_srnn_group_ready;
    assign o_fc_srnn_layer_ready = !m2_array_owner &&
        !srnn_group_streamer_selected &&
        array_fc_srnn_layer_ready;
    assign o_fc_srnn_group_ready = !m2_array_owner &&
        !srnn_group_streamer_selected &&
        array_fc_srnn_group_ready;
    assign o_fc_srnn_layer_done = !m2_array_owner &&
        !srnn_group_streamer_selected &&
        array_fc_srnn_layer_done;
    assign o_fc_srnn_busy = !m2_array_owner &&
        !srnn_group_streamer_selected && array_fc_srnn_busy;
    assign o_fc_srnn_idle = !m2_array_owner &&
        !srnn_group_streamer_selected && array_fc_srnn_idle;
    assign o_fc_srnn_sticky_error = !m2_array_owner &&
        !srnn_group_streamer_selected &&
        array_fc_srnn_sticky_error;
    assign o_fc_srnn_error_code = !m2_array_owner &&
        !srnn_group_streamer_selected ?
        array_fc_srnn_error_code : '0;
    assign array_fc_srnn_observation_ready = m2_array_owner ?
        m2_array_observation_ready : fc_srnn_learning_scalar_ready;
    assign o_fc_srnn_observation_valid = !m2_array_owner &&
        array_fc_srnn_observation_valid;
    assign o_fc_srnn_observation_mode = array_fc_srnn_observation_mode;
    assign o_fc_srnn_observation_transaction_tag =
        array_fc_srnn_observation_transaction_tag;
    assign o_fc_srnn_observation_neuron_id =
        array_fc_srnn_observation_neuron_id;
    assign o_fc_srnn_observation_spike = array_fc_srnn_observation_spike;
    assign o_fc_srnn_observation_pre_state =
        array_fc_srnn_observation_pre_state;
    assign o_fc_srnn_observation_trace_next =
        array_fc_srnn_observation_trace_next;
    assign o_fc_srnn_observation_last_in_record =
        array_fc_srnn_observation_last_in_record;
    assign o_fc_srnn_observation_busy = !m2_array_owner &&
        array_fc_srnn_observation_busy;
    assign o_fc_srnn_observation_sticky_error = !m2_array_owner &&
        array_fc_srnn_observation_sticky_error;

    unified_h40_array16_three_topology_top_v1 #(
        .M2_CACHE_DEPTH(M2_CACHE_DEPTH),
        .SCNN_WEIGHT_FRACTIONAL_BITS(SCNN_WEIGHT_FRACTIONAL_BITS),
        .ACCUMULATE_SCNN_SAMPLE_LOGITS(ACCUMULATE_SCNN_SAMPLE_LOGITS),
        .WEIGHT_DEPTH(WEIGHT_DEPTH), .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .MEMORY_DEPTH(MEMORY_DEPTH), .MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH),
        .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH),
         .FEATURE_WORD_DEPTH(FEATURE_WORD_DEPTH),
         .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
         .ENABLE_FC_SRNN_LAYER_CLIENT(
             ENABLE_FC_CLIENT || ENABLE_SRNN_CLIENT),
         .ENABLE_FC_SRNN_TP_OBSERVATION_STREAM(
             ENABLE_FC_CLIENT || ENABLE_SRNN_CLIENT),
         .ENABLE_SCNN_CLIENT(ENABLE_SCNN_CLIENT),
         .ENABLE_SCNN_CLASSIFIER_GRAPH(1'b1),
        .ENABLE_SCNN_STATE_SERVICE(ENABLE_SCNN_FRAME_SERVICE),
        .SCNN_LOGIT_CONTEXTS(MAX_BATCH),
        .MAX_SCNN_FEATURE_CHANNELS(OBS_OUTPUT_CHANNELS),
        .SCNN_CLASSIFIER_FEATURE_SPATIAL(POOLED_SPATIAL * POOLED_SPATIAL),
        .MAX_SCNN_CLASSES(MAX_CLASSES), .SCNN_FEATURE_LANES(LANES),
        .SCNN_LOGIT_ENGINES(ENGINES), .SCNN_SAMPLE_WIDTH(SAMPLE_WIDTH),
        .SCNN_TIMESTEP_WIDTH(TIMESTEP_WIDTH),
        .SCNN_CHANNEL_WIDTH(CHANNEL_WIDTH),
        .SCNN_CHANNEL_COUNT_WIDTH(CHANNEL_COUNT_WIDTH),
        .SCNN_FEATURE_WIDTH(FEATURE_WIDTH), .SCNN_CLASS_WIDTH(CLASS_WIDTH),
        .SCNN_CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH)
    ) u_array (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(1'b0),
        .i_clear_error(i_clear_error),
        .i_direct_command_valid(1'b0), .i_direct_mode('0),
        .i_direct_neuron_count('0), .i_direct_first(1'b0),
        .i_direct_last(1'b0), .i_direct_source_valid('0),
        .i_direct_weight_addr0('0), .i_direct_weight_addr1('0),
        .i_direct_weight_addr2('0), .i_direct_weight_addr3('0),
        .i_direct_leak('0), .i_direct_threshold('0),
        .i_direct_reset_to_zero(1'b0), .i_direct_trace_decay('0),
         .i_direct_transaction_tag('0), .i_direct_work_base_addr('0),
         .i_direct_destination_tile_base(4'd0),
         .i_direct_observation_ready('1),
         .i_fc_srnn_streamed_feedforward((m2_array_owner || srnn_group_streamer_selected) ? 1'b0 : i_fc_srnn_streamed_feedforward),
         .i_fc_srnn_group_feedforward_end((m2_array_owner || srnn_group_streamer_selected) ? 1'b0 : i_fc_srnn_group_feedforward_end),
         .i_fc_srnn_layer_valid(m2_array_owner ? m2_layer_valid :
             srnn_group_streamer_selected ? srnn_streamer_layer_valid :
                 i_fc_srnn_layer_valid),
         .o_fc_srnn_layer_ready(array_fc_srnn_layer_ready),
         .i_fc_srnn_mode(m2_array_owner ? m2_layer_mode :
             srnn_group_streamer_selected ? 2'd1 : i_fc_srnn_mode),
         .i_fc_srnn_weight_int8((m2_array_owner ||
             srnn_group_streamer_selected) ? 1'b0 : i_fc_srnn_weight_int8),
         .i_fc_srnn_current_shift6((m2_array_owner ||
             srnn_group_streamer_selected) ? 1'b0 : i_fc_srnn_current_shift6),
         .i_fc_srnn_neuron_count(m2_array_owner ? m2_layer_neuron_count :
             srnn_group_streamer_selected ? srnn_streamer_neuron_count :
                 i_fc_srnn_neuron_count),
         .i_fc_srnn_feedforward_group_count(
             m2_array_owner ? m2_layer_feedforward_group_count :
                 srnn_group_streamer_selected ?
                     srnn_streamer_feedforward_group_count :
                     i_fc_srnn_feedforward_group_count),
         .i_fc_srnn_recurrent_group_count(m2_array_owner ?
             m2_layer_recurrent_group_count :
                 srnn_group_streamer_selected ?
                     srnn_streamer_recurrent_group_count :
                     i_fc_srnn_recurrent_group_count),
         .i_fc_srnn_recurrent_enable(m2_array_owner ?
             m2_layer_recurrent_enable : srnn_group_streamer_selected ?
                 srnn_streamer_recurrent_enable : i_fc_srnn_recurrent_enable),
         .i_fc_srnn_leak(m2_array_owner ? m2_layer_leak :
             srnn_group_streamer_selected ? srnn_streamer_leak :
                 i_fc_srnn_leak),
         .i_fc_srnn_threshold(m2_array_owner ? m2_layer_threshold :
             srnn_group_streamer_selected ? srnn_streamer_threshold :
                 i_fc_srnn_threshold),
         .i_fc_srnn_reset_to_zero(m2_array_owner ?
             m2_layer_reset_to_zero : srnn_group_streamer_selected ?
                 srnn_streamer_reset_to_zero : i_fc_srnn_reset_to_zero),
         .i_fc_srnn_trace_decay(m2_array_owner ? m2_layer_trace_decay :
             srnn_group_streamer_selected ? srnn_streamer_trace_decay :
                 i_fc_srnn_trace_decay),
         .i_fc_srnn_transaction_tag(m2_array_owner ?
             m2_layer_transaction_tag : srnn_group_streamer_selected ?
                 srnn_streamer_transaction_tag : i_fc_srnn_transaction_tag),
         .i_fc_srnn_work_base_addr(m2_array_owner ?
             m2_layer_work_base_addr : srnn_group_streamer_selected ?
                 srnn_streamer_work_base_addr : i_fc_srnn_work_base_addr),
         .i_fc_srnn_destination_tile_base(m2_array_owner ?
             m2_layer_destination_tile_base : srnn_group_streamer_selected ?
                 srnn_streamer_destination_tile_base : fc_srnn_destination_tile_base),
         .i_fc_srnn_group_valid(m2_array_owner ? m2_group_valid :
             srnn_group_streamer_selected ? srnn_streamer_group_valid :
                 i_fc_srnn_group_valid),
         .o_fc_srnn_group_ready(array_fc_srnn_group_ready),
         .i_fc_srnn_group_recurrent(m2_array_owner ? m2_group_recurrent :
             srnn_group_streamer_selected ? srnn_streamer_group_recurrent :
                 i_fc_srnn_group_recurrent),
         .i_fc_srnn_group_source_valid(m2_array_owner ?
             m2_group_source_valid : srnn_group_streamer_selected ?
                 srnn_streamer_group_source_valid :
                 i_fc_srnn_group_source_valid),
         .i_fc_srnn_group_weight_addr0(m2_array_owner ?
             m2_group_weight_addr0 : srnn_group_streamer_selected ?
                 srnn_streamer_group_weight_addr0 :
                 i_fc_srnn_group_weight_addr0),
         .i_fc_srnn_group_weight_addr1(m2_array_owner ?
             m2_group_weight_addr1 : srnn_group_streamer_selected ?
                 srnn_streamer_group_weight_addr1 :
                 i_fc_srnn_group_weight_addr1),
         .i_fc_srnn_group_weight_addr2(m2_array_owner ?
             m2_group_weight_addr2 : srnn_group_streamer_selected ?
                 srnn_streamer_group_weight_addr2 :
                 i_fc_srnn_group_weight_addr2),
         .i_fc_srnn_group_weight_addr3(m2_array_owner ?
             m2_group_weight_addr3 : srnn_group_streamer_selected ?
                 srnn_streamer_group_weight_addr3 :
                 i_fc_srnn_group_weight_addr3),
         .o_fc_srnn_layer_done(array_fc_srnn_layer_done),
         .o_fc_srnn_busy(array_fc_srnn_busy),
         .o_fc_srnn_idle(array_fc_srnn_idle),
         .o_fc_srnn_sticky_error(array_fc_srnn_sticky_error),
         .o_fc_srnn_error_code(array_fc_srnn_error_code),
         .o_fc_srnn_observation_valid(array_fc_srnn_observation_valid),
         .i_fc_srnn_observation_ready(array_fc_srnn_observation_ready),
         .i_fc_srnn_observation_sink_idle(m2_array_owner ? 1'b1 :
             o_fc_srnn_learning_idle),
         .o_fc_srnn_observation_mode(array_fc_srnn_observation_mode),
         .o_fc_srnn_observation_transaction_tag(
             array_fc_srnn_observation_transaction_tag),
         .o_fc_srnn_observation_neuron_id(
             array_fc_srnn_observation_neuron_id),
         .o_fc_srnn_observation_spike(array_fc_srnn_observation_spike),
         .o_fc_srnn_observation_pre_state(
             array_fc_srnn_observation_pre_state),
         .o_fc_srnn_observation_trace_next(
             array_fc_srnn_observation_trace_next),
         .o_fc_srnn_observation_last_in_record(
             array_fc_srnn_observation_last_in_record),
         .o_fc_srnn_observation_busy(array_fc_srnn_observation_busy),
         .o_fc_srnn_observation_sticky_error(
             array_fc_srnn_observation_sticky_error),
        .i_scnn_start_valid(layer_start_valid),
        .o_scnn_start_ready(layer_start_ready),
        .i_scnn_in_width(descriptor_active.in_width),
        .i_scnn_in_height(descriptor_active.in_height),
        .i_scnn_in_channels(descriptor_active.in_channels),
        .i_scnn_out_width(descriptor_active.out_width),
        .i_scnn_out_height(descriptor_active.out_height),
        .i_scnn_out_channels(descriptor_active.out_channels),
        .i_scnn_stride(descriptor_active.stride), .i_scnn_padding(descriptor_active.padding),
        .i_scnn_pool_mode(descriptor_active.pool_mode),
        .i_scnn_initial_active_pair(~o_active_classifier_pair),
        .i_scnn_work_base_addr(descriptor_active.work_base),
        .i_scnn_leak(descriptor_active.leak), .i_scnn_threshold(descriptor_active.threshold),
        .i_scnn_reset_to_zero(descriptor_active.reset_to_zero),
        .i_scnn_trace_decay(descriptor_active.trace_decay),
        .i_scnn_layer_tag(descriptor_active.layer_tag),
        .i_scnn_classifier_feature_channel_base(
            i_classifier_feature_channel_base),
        .i_scnn_classifier_feature_channel_count(
            i_classifier_feature_channel_count),
        .i_scnn_classifier_class_count(layer_is_tail ? i_classifier_class_count : '0),
        .i_scnn_classifier_active_pair(o_active_classifier_pair),
        .i_scnn_classifier_work_base_addr(i_classifier_work_base_addr),
        .i_scnn_classifier_trace_decay(i_classifier_trace_decay),
        .i_scnn_classifier_layer_tag(i_classifier_layer_tag),
        .i_scnn_version(graph_version), .i_scnn_frame_index(scnn_frame_index),
        .i_scnn_sample_reset(graph_timestep == '0),
        .i_scnn_sample_last(graph_sample_last),
        .i_scnn_sample_label(graph_label),
        // Both target and student learning passes export the pooled feature
        // wave. Target consumes it in the temporal-trace branch; student also
        // forwards it to the classifier observation path.
        .i_scnn_learning_enable(layer_is_tail && (conv_observation_enable ||
            classifier_observation_enable)),
        .i_scnn_conv8_observation_enable(layer_is_tail && conv_observation_enable),
        .i_scnn_conv8_observation_target(observation_target),
        .i_scnn_sample_index(graph_sample),
        .i_scnn_timestep_index(graph_timestep),
        .o_scnn_context_load_valid(array_context_load_valid),
        .i_scnn_context_load_ready(array_context_load_ready),
        .o_scnn_context_load_weights(array_context_load_weights),
        .o_scnn_context_output_channel_base(array_context_output_base),
        .o_scnn_context_output_channel_count(array_context_output_count),
        .o_scnn_context_spatial_base(array_context_spatial_base),
        .o_scnn_context_spatial_count(array_context_spatial_count),
        .o_scnn_context_active_pair(array_context_pair),
        .o_scnn_context_work_base_addr(array_context_work_base),
        .o_scnn_feature_request_valid(core_request_valid),
        .i_scnn_feature_request_ready(core_request_ready),
        .o_scnn_feature_term_valid(core_term_valid),
        .o_scnn_feature_input_channel_flat(core_input_channel_flat),
        .o_scnn_feature_kernel_row_flat(core_kernel_row_flat),
        .o_scnn_feature_kernel_col_flat(core_kernel_col_flat),
        .o_scnn_feature_spatial_base(core_spatial_base),
        .o_scnn_feature_spatial_count(core_spatial_count),
        .o_scnn_feature_output_row(core_output_row),
        .o_scnn_feature_output_col(core_output_col),
        .o_scnn_feature_value_bit(core_value_bit),
        .i_scnn_feature_response_valid(core_response_valid),
        .o_scnn_feature_response_ready(core_response_ready),
        .i_scnn_feature_mask0(response_mask0),
        .i_scnn_feature_mask1(response_mask1),
        .i_scnn_feature_mask2(response_mask2),
        .i_scnn_feature_mask3(response_mask3),
        .i_scnn_feature_service_error(response_error),
        .i_scnn_feature_write_slot(descriptor_active.feature_write_slot),
        .i_scnn_feature_write_base_addr(descriptor_active.feature_write_base),
        .i_scnn_feature_write_row_stride_words(
            descriptor_active.feature_write_row_stride),
        .i_scnn_feature_write_channel_stride_words(
            descriptor_active.feature_write_channel_stride),
        .o_scnn_feature_update_valid(o_feature_update_valid),
        .i_scnn_feature_update_ready(i_feature_update_ready),
        .o_scnn_feature_update_slot(o_feature_update_slot),
        .o_scnn_feature_update_addr(o_feature_update_addr),
        .o_scnn_feature_update_data(o_feature_update_data),
        .o_scnn_feature_update_mask(o_feature_update_mask),
        .i_scnn_feature_update_done(i_feature_update_done),
        .o_scnn_context_release_valid(array_context_release_valid),
        .i_scnn_context_release_ready(array_context_release_ready),
        .o_scnn_release_output_channel_base(array_release_output_base),
        .o_scnn_release_output_channel_count(array_release_output_count),
        .o_scnn_release_spatial_base(array_release_spatial_base),
        .o_scnn_release_spatial_count(array_release_spatial_count),
        .o_scnn_release_active_pair(array_release_pair),
        .o_scnn_release_work_base_addr(array_release_work_base),
        .i_scnn_observation_ready('1),
        .i_scnn_observation_sink_idle(1'b1), .o_scnn_done(layer_done),
        .o_scnn_feature_trace_wave_valid(feature_wave_valid),
        .i_scnn_feature_trace_wave_ready(feature_wave_ready),
        .o_scnn_feature_trace_wave_mask(feature_wave_mask),
        .o_scnn_feature_trace_wave_base(feature_wave_base),
        .o_scnn_feature_trace_wave_q8_8_flat(feature_wave_flat),
        .o_scnn_feature_trace_wave_version(feature_wave_version),
        .o_scnn_feature_trace_wave_frame_index(feature_wave_frame),
        .o_scnn_feature_trace_wave_first(feature_wave_first),
        .o_scnn_feature_trace_wave_last(feature_wave_last),
        .o_scnn_feature_trace_wave_sample_reset(feature_wave_sample_reset),
        .o_scnn_feature_trace_wave_sample_last(feature_wave_sample_last),
        .o_scnn_feature_sample_label(feature_sample_label),
        .o_scnn_conv8_surrogate_valid(surrogate_valid),
        .i_scnn_conv8_surrogate_ready(surrogate_ready),
        .o_scnn_conv8_surrogate_target(surrogate_target),
        .o_scnn_conv8_surrogate_sample(surrogate_sample),
        .o_scnn_conv8_surrogate_timestep(surrogate_timestep),
        .o_scnn_conv8_surrogate_version(surrogate_version),
        .o_scnn_conv8_surrogate_layer_tag(surrogate_layer_tag),
        .o_scnn_conv8_surrogate_channel(surrogate_channel),
        .o_scnn_conv8_surrogate_spatial(surrogate_spatial),
        .o_scnn_conv8_surrogate_q8_8(surrogate_q8_8),
        .o_scnn_logit_wave_valid(logit_wave_valid),
        .i_scnn_logit_wave_ready(logit_wave_ready),
        .o_scnn_logit_wave_mask(logit_wave_mask),
        .o_scnn_logit_wave_base(logit_wave_base),
        .o_scnn_logit_wave_count(logit_wave_count),
        .o_scnn_logit_wave_q8_8_flat(logit_wave_flat),
        .o_scnn_logit_wave_version(logit_wave_version),
        .o_scnn_logit_wave_frame_index(logit_wave_frame),
        .o_scnn_logit_wave_frame_last(logit_wave_frame_last),
        .o_scnn_logit_wave_sample_last(logit_wave_sample_last),
        .i_weight_fill_context(weight_fill_context),
        .i_weight_fill_valid(weight_fill_valid),
        .o_weight_fill_ready(weight_fill_ready),
        .o_scnn_context_refill_safe(array_refill_safe),
        .i_weight_fill_tile_flat(weight_fill_tile_flat),
        .i_weight_fill_bank(weight_fill_bank),
        .i_weight_fill_addr_flat(weight_fill_addr_flat),
        .i_weight_fill_data_flat(weight_fill_data_flat),
        .i_maintenance_start_valid(array_maintenance_start_valid),
        .o_maintenance_start_ready(array_maintenance_start_ready),
        .i_maintenance_release_valid(array_maintenance_release_valid),
        .o_maintenance_release_ready(array_maintenance_release_ready),
        .i_maintenance_tile(array_maintenance_tile),
        .i_state_maintenance_write_valid(array_state_write_valid),
        .o_state_maintenance_write_ready(array_state_write_ready),
        .i_state_maintenance_write_addr(array_state_write_addr),
        .i_state_maintenance_write_data(array_state_write_data),
        .i_state_maintenance_write_mask(array_state_write_mask),
        .i_state_maintenance_read_valid(i_state_maintenance_read_valid &&
            !maintenance_owner_m2_q),
        .o_state_maintenance_read_ready(array_state_read_ready),
        .i_state_maintenance_read_addr(i_state_maintenance_read_addr),
        .o_state_maintenance_result_valid(o_state_maintenance_result_valid),
        .i_state_maintenance_result_ready(i_state_maintenance_result_ready),
        .o_state_maintenance_result_addr(o_state_maintenance_result_addr),
        .o_state_maintenance_read_data(o_state_maintenance_read_data),
        .i_trace_maintenance_write_valid(array_trace_write_valid),
        .o_trace_maintenance_write_ready(array_trace_write_ready),
        .i_trace_maintenance_write_addr(array_trace_write_addr),
        .i_trace_maintenance_write_data(array_trace_write_data),
        .i_trace_maintenance_write_mask(array_trace_write_mask),
        .i_trace_maintenance_read_valid(i_trace_maintenance_read_valid &&
            !maintenance_owner_m2_q),
        .o_trace_maintenance_read_ready(array_trace_read_ready),
        .i_trace_maintenance_read_addr(i_trace_maintenance_read_addr),
        .o_trace_maintenance_result_valid(o_trace_maintenance_result_valid),
        .i_trace_maintenance_result_ready(i_trace_maintenance_result_ready),
        .o_trace_maintenance_result_addr(o_trace_maintenance_result_addr),
        .o_trace_maintenance_read_data(o_trace_maintenance_read_data),
        .o_active_owner(o_array_active_owner), .o_busy(array_busy),
        .o_idle(array_idle), .o_tile_idle(o_array_tile_idle),
        .o_tile_sticky_error(o_array_tile_sticky_error),
        .o_sticky_error(array_error)
    );

    // Learning observations are indexed by the graph timestep inside the
    // current BxT transaction. Inference retains the caller's frame index.
    assign scnn_frame_index = (conv_observation_enable ||
        classifier_observation_enable) ? FRAME_WIDTH'(graph_timestep) :
        i_frame_index;

    initial begin
        if (LANES != 8 || ENGINES != 12 || MAX_CLASSES > 16 ||
                POOLED_SPATIAL * POOLED_SPATIAL > 40)
            $error("SCNN selected-layer array geometry is unsupported");
    end
endmodule

`default_nettype wire
