`timescale 1ns/1ps
`default_nettype none
module tb_srnn_profile_measurement;
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
    localparam int WEIGHT_DEPTH = 1156;
    localparam int WEIGHT_ADDR_WIDTH = $clog2(WEIGHT_DEPTH);
    localparam int MEMORY_DEPTH = 1024;
    localparam int MEMORY_ADDR_WIDTH = $clog2(MEMORY_DEPTH);
    localparam int WORK_ADDR_WIDTH = $clog2(MEMORY_DEPTH / 2);
    localparam int FEATURE_WORD_DEPTH = 8192;
    localparam int FEATURE_ADDR_WIDTH = $clog2(FEATURE_WORD_DEPTH);
    localparam int PROFILE_ID_WIDTH = 8;
    localparam int TENSOR_ID_WIDTH = 8;
    localparam int PARAM_DEPTH = 371520;
    localparam int PARAM_ADDR_WIDTH = 24;
    localparam int TAG_WIDTH = 16;
    localparam int MAX_BATCH = 16;
    localparam int MAX_TIMESTEPS = 20;
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
    logic i_initialize_valid = '0;
    logic o_initialize_ready;
    logic o_initialize_done;
    logic i_classifier_pair = '0;
    logic [VERSION_WIDTH-1:0] i_context_model_version = '0;
    logic [VERSION_WIDTH-1:0] i_context_adapt_version = '0;
    logic i_system_parameter_owner = '0;
    logic i_system_learning_enable = '0;
    logic i_system_profile_aligned = '0;
    logic [PROFILE_ID_WIDTH-1:0] i_system_profile_id = '0;
    logic i_request_valid = '0;
    logic o_request_ready;
    logic i_learning_enable = '0;
    logic [BATCH_COUNT_WIDTH-1:0] i_batch_count = '0;
    logic [TIMESTEP_WIDTH-1:0] i_timestep = '0;
    logic [TIMESTEP_COUNT_WIDTH-1:0] i_timestep_count = '0;
    logic [VERSION_WIDTH-1:0] i_sample_version = '0;
    logic [MAX_BATCH*CLASS_WIDTH-1:0] i_labels_flat = '0;
    logic o_done;
    logic i_fc_srnn_layer_valid = '0;
    logic o_fc_srnn_layer_ready;
    logic [1:0] i_fc_srnn_mode = '0;
    logic i_fc_srnn_weight_int8 = '0;
    logic i_fc_srnn_current_shift6 = '0;
    logic [15:0] i_fc_srnn_neuron_count = '0;
    logic [15:0] i_fc_srnn_feedforward_group_count = '0;
    logic [15:0] i_fc_srnn_recurrent_group_count = '0;
    logic i_fc_srnn_recurrent_enable = '0;
    logic [15:0] i_fc_srnn_leak = '0;
    logic [31:0] i_fc_srnn_threshold = '0;
    logic i_fc_srnn_reset_to_zero = '0;
    logic [15:0] i_fc_srnn_trace_decay = '0;
    logic [51:0] i_fc_srnn_transaction_tag = '0;
    logic [WORK_ADDR_WIDTH-1:0] i_fc_srnn_work_base_addr = '0;
    logic [3:0] i_fc_srnn_destination_tile_base = '0;
    logic i_fc_srnn_group_valid;
    logic o_fc_srnn_group_ready;
    logic i_fc_srnn_group_recurrent;
    logic [3:0] i_fc_srnn_group_source_valid;
    logic [WEIGHT_ADDR_WIDTH-1:0] i_fc_srnn_group_weight_addr0;
    logic [WEIGHT_ADDR_WIDTH-1:0] i_fc_srnn_group_weight_addr1;
    logic [WEIGHT_ADDR_WIDTH-1:0] i_fc_srnn_group_weight_addr2;
    logic [WEIGHT_ADDR_WIDTH-1:0] i_fc_srnn_group_weight_addr3;
    logic i_srnn_group_streamer_select = '0;
    logic i_srnn_group_streamer_start_valid = '0;
    logic o_srnn_group_streamer_start_ready;
    logic [9:0] i_srnn_group_feedforward_source_count = '0;
    logic [9:0] i_srnn_group_recurrent_source_count = '0;
    logic [9:0] i_srnn_group_feedforward_cache_row_base = '0;
    logic [9:0] i_srnn_group_recurrent_cache_row_base = '0;
    logic [SRNN_FEEDFORWARD_SOURCE_MAX-1:0] i_srnn_group_feedforward_spike_flat = '0;
    logic [SRNN_RECURRENT_SOURCE_MAX-1:0] i_srnn_group_recurrent_spike_flat = '0;
    logic o_srnn_group_streamer_done;
    logic o_srnn_group_streamer_busy;
    logic o_srnn_group_streamer_sticky_error;
    logic [3:0] o_srnn_group_streamer_error_code;
    logic o_fc_srnn_layer_done;
    logic o_fc_srnn_busy;
    logic o_fc_srnn_idle;
    logic o_fc_srnn_sticky_error;
    logic [7:0] o_fc_srnn_error_code;
    logic o_fc_srnn_observation_valid;
    logic i_fc_srnn_observation_ready = '0;
    logic i_fc_srnn_observation_sink_idle = '0;
    logic [1:0] o_fc_srnn_observation_mode;
    logic [51:0] o_fc_srnn_observation_transaction_tag;
    logic [9:0] o_fc_srnn_observation_neuron_id;
    logic o_fc_srnn_observation_spike;
    logic signed [31:0] o_fc_srnn_observation_pre_state;
    logic signed [31:0] o_fc_srnn_observation_trace_next;
    logic o_fc_srnn_observation_last_in_record;
    logic o_fc_srnn_observation_busy;
    logic o_fc_srnn_observation_sticky_error;
    logic i_fc_srnn_learning_cfg_valid = '0;
    logic o_fc_srnn_learning_cfg_ready;
    logic [1:0] i_fc_srnn_learning_cfg_mode = '0;
    logic [51:0] i_fc_srnn_learning_cfg_tag = '0;
    logic [5:0] i_fc_srnn_learning_sample_id = '0;
    logic [9:0] i_fc_srnn_learning_neuron_base = '0;
    logic [9:0] i_fc_srnn_learning_neuron_count = '0;
    logic signed [17:0] i_fc_srnn_learning_fc_threshold = '0;
    logic i_fc_srnn_learning_fc_output_layer = '0;
    logic [3:0] i_fc_srnn_learning_fc_label = '0;
    logic i_fc_srnn_learning_fc_logit_scale_enable = '0;
    logic [23:0] i_fc_srnn_learning_fc_logit_scale_q16 = '0;
    logic i_fc_srnn_learning_fc_logit_accumulate = '0;
    logic i_fc_srnn_learning_fc_sample_start = '0;
    logic i_fc_srnn_learning_srnn_path = '0;
    logic o_fc_surrogate_valid;
    logic [5:0] o_fc_surrogate_sample;
    logic [9:0] o_fc_surrogate_hidden;
    logic [16:0] o_fc_surrogate_q016;
    logic o_fc_logit_last;
    logic o_fc_srnn_learning_capture_done;
    logic o_fc_srnn_learning_done;
    logic o_fc_srnn_learning_busy;
    logic o_fc_srnn_learning_idle;
    logic o_fc_srnn_learning_sticky_error;
    logic o_fc_srnn_learning_error_source;
    logic [3:0] o_fc_srnn_learning_error_code;
    logic i_fc_w0_refill_valid = '0;
    logic o_fc_w0_refill_ready;
    logic i_fc_w0_refill_all = '0;
    logic i_fc_cache_refill_weight_int8 = '0;
    logic [3:0] i_fc_cache_refill_destination_tile_base = '0;
    logic [FEATURE_COUNT_WIDTH-1:0] i_fc_w0_dirty_feature = '0;
    logic [3:0] i_fc_w0_dirty_tile = '0;
    logic [PARAM_ADDR_WIDTH-1:0] i_fc_w0_parameter_base = '0;
    logic [PARAM_ADDR_WIDTH-1:0] i_fc_w0_hidden_stride = '0;
    logic [9:0] i_fc_cache_row_base = '0;
    logic [FEATURE_COUNT_WIDTH-1:0] i_fc_w0_feature_count = '0;
    logic [9:0] i_fc_w0_hidden_count = '0;
    logic [4:0] i_fc_w0_tile_count = '0;
    logic [VERSION_WIDTH-1:0] i_fc_w0_version = '0;
    logic o_fc_w0_refill_done;
    logic [31:0] o_fc_w0_rows_published;
    logic [VERSION_WIDTH-1:0] o_fc_w0_published_version;
    logic [1:0] o_cache_fill_grant;
    logic o_cache_refill_busy;
    logic o_cache_refill_quiescent;
    logic o_cache_refill_sticky_error;
    logic i_srnn_backbone_refill_valid = '0;
    logic o_srnn_backbone_refill_ready;
    logic i_srnn_backbone_refill_all = '0;
    logic [9:0] i_srnn_backbone_dirty_source = '0;
    logic [3:0] i_srnn_backbone_dirty_tile = '0;
    logic [SRNN_BACKBONE_PARAM_ADDR_WIDTH-1:0] i_srnn_backbone_parameter_base = '0;
    logic [9:0] i_srnn_backbone_cache_row_base = '0;
    logic [9:0] i_srnn_backbone_source_count = '0;
    logic [9:0] i_srnn_backbone_destination_count = '0;
    logic [3:0] i_srnn_backbone_tile_base = '0;
    logic [VERSION_WIDTH-1:0] i_srnn_backbone_version = '0;
    logic o_srnn_backbone_forward_read_valid;
    logic i_srnn_backbone_forward_read_ready = '0;
    logic [SRNN_BACKBONE_PARAM_ADDR_WIDTH-1:0] o_srnn_backbone_forward_read_addr;
    logic [TAG_WIDTH-1:0] o_srnn_backbone_forward_read_tag;
    logic i_srnn_backbone_forward_response_valid = '0;
    logic o_srnn_backbone_forward_response_ready;
    logic [TAG_WIDTH-1:0] i_srnn_backbone_forward_response_tag = '0;
    logic signed [15:0] i_srnn_backbone_forward_response_code = '0;
    logic i_srnn_backbone_forward_response_error = '0;
    logic o_srnn_backbone_refill_done;
    logic [31:0] o_srnn_backbone_rows_published;
    logic [VERSION_WIDTH-1:0] o_srnn_backbone_published_version;
    logic [PROFILE_ID_WIDTH-1:0] i_profile_id = '0;
    logic [TENSOR_ID_WIDTH-1:0] i_conv_tensor_id = '0;
    logic [TENSOR_ID_WIDTH-1:0] i_classifier_tensor_id = '0;
    logic [CHANNEL_COUNT_WIDTH-1:0] i_total_input_channels = '0;
    logic [CHANNEL_COUNT_WIDTH-1:0] i_total_output_channels = '0;
    logic [PARAM_ADDR_WIDTH-1:0] i_conv_parameter_base = '0;
    logic [Q16_ADDR_WIDTH-1:0] i_conv_weight_q16_base = '0;
    logic [PARAM_ADDR_WIDTH-1:0] i_classifier_parameter_base = '0;
    logic [FEATURE_COUNT_WIDTH-1:0] i_classifier_feature_count = '0;
    logic [CLASS_COUNT_WIDTH-1:0] i_classifier_class_count = '0;
    logic i_classifier_trace_leak_valid = '0;
    logic signed [15:0] i_classifier_trace_leak_q8_8 = '0;
    logic [15:0] i_conv_vote_increment = '0;
    logic [15:0] i_conv_vote_threshold = '0;
    logic [15:0] i_classifier_vote_increment = '0;
    logic [15:0] i_classifier_vote_threshold = '0;
    logic signed [15:0] i_code_min = '0;
    logic signed [15:0] i_code_max = '0;
    logic i_forward_start_valid = '0;
    logic o_forward_start_ready;
    logic o_forward_start_fire;
    logic i_forward_done = '0;
    logic o_forward_active;
    logic [VERSION_WIDTH-1:0] o_forward_model_version;
    logic [VERSION_WIDTH-1:0] o_forward_adapt_version;
    logic i_restore_complete = '0;
    logic o_restore_ready;
    logic o_recovery_required;
    logic [15:0] i_conv_in_width = '0;
    logic [15:0] i_conv_in_height = '0;
    logic [15:0] i_conv_in_channels = '0;
    logic [15:0] i_conv_out_width = '0;
    logic [15:0] i_conv_out_height = '0;
    logic [15:0] i_conv_out_channels = '0;
    logic [7:0] i_conv_stride = '0;
    logic [7:0] i_conv_padding = '0;
    logic [1:0] i_conv_pool_mode = '0;
    logic [WORK_ADDR_WIDTH-1:0] i_conv_work_base_addr = '0;
    logic [15:0] i_conv_leak = '0;
    logic [31:0] i_conv_threshold = '0;
    logic i_conv_reset_to_zero = '0;
    logic [15:0] i_conv_trace_decay = '0;
    logic [19:0] i_conv_layer_tag = '0;
    logic [FEATURE_COUNT_WIDTH-1:0] i_target_feature_count = '0;
    logic signed [15:0] i_target_trace_leak_q8_8 = '0;
    logic i_target_trace_state_reset = '0;
    logic [CHANNEL_WIDTH-1:0] i_classifier_feature_channel_base = '0;
    logic [CHANNEL_COUNT_WIDTH-1:0] i_classifier_feature_channel_count = '0;
    logic [WORK_ADDR_WIDTH-1:0] i_classifier_work_base_addr = '0;
    logic [15:0] i_classifier_trace_decay = '0;
    logic [19:0] i_classifier_layer_tag = '0;
    logic [FRAME_WIDTH-1:0] i_frame_index = '0;
    logic i_graph_descriptor_valid = '0;
    logic o_graph_descriptor_ready;
    logic i_graph_descriptor_last = '0;
    logic o_graph_descriptor_done;
    logic o_graph_frame_active;
    logic o_graph_frame_target;
    logic [SAMPLE_WIDTH-1:0] o_graph_frame_sample;
    logic [TIMESTEP_WIDTH-1:0] o_graph_frame_timestep;
    logic [VERSION_WIDTH-1:0] o_graph_frame_version;
    logic o_state_service_valid;
    logic i_state_service_ready = '0;
    logic o_state_service_release;
    logic o_state_service_is_classifier;
    logic [9:0] o_state_service_output_channel_base;
    logic [9:0] o_state_service_output_channel_count;
    logic [31:0] o_state_service_spatial_base;
    logic [5:0] o_state_service_spatial_count;
    logic o_state_service_active_pair;
    logic [WORK_ADDR_WIDTH-1:0] o_state_service_work_base_addr;
    logic o_feature_request_valid;
    logic i_feature_request_ready = '0;
    logic [3:0] o_feature_term_valid;
    logic [39:0] o_feature_input_channel_flat;
    logic [7:0] o_feature_kernel_row_flat;
    logic [7:0] o_feature_kernel_col_flat;
    logic [31:0] o_feature_spatial_base;
    logic [5:0] o_feature_spatial_count;
    logic [15:0] o_feature_output_row;
    logic [15:0] o_feature_output_col;
    logic [3:0] o_feature_value_bit;
    logic i_feature_response_valid = '0;
    logic o_feature_response_ready;
    logic [39:0] i_feature_mask0 = '0;
    logic [39:0] i_feature_mask1 = '0;
    logic [39:0] i_feature_mask2 = '0;
    logic [39:0] i_feature_mask3 = '0;
    logic i_feature_service_error = '0;
    logic i_feature_write_slot = '0;
    logic [FEATURE_ADDR_WIDTH-1:0] i_feature_write_base_addr = '0;
    logic [15:0] i_feature_write_row_stride_words = '0;
    logic [FEATURE_ADDR_WIDTH-1:0] i_feature_write_channel_stride_words = '0;
    logic o_feature_update_valid;
    logic i_feature_update_ready = '0;
    logic o_feature_update_slot;
    logic [FEATURE_ADDR_WIDTH-1:0] o_feature_update_addr;
    logic [63:0] o_feature_update_data;
    logic [63:0] o_feature_update_mask;
    logic i_feature_update_done = '0;
    logic [3:0] i_maintenance_tile = '0;
    logic i_state_maintenance_write_valid = '0;
    logic o_state_maintenance_write_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] i_state_maintenance_write_addr = '0;
    logic [319:0] i_state_maintenance_write_data = '0;
    logic [39:0] i_state_maintenance_write_mask = '0;
    logic i_state_maintenance_read_valid = '0;
    logic o_state_maintenance_read_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] i_state_maintenance_read_addr = '0;
    logic o_state_maintenance_result_valid;
    logic i_state_maintenance_result_ready = '0;
    logic [MEMORY_ADDR_WIDTH-1:0] o_state_maintenance_result_addr;
    logic [319:0] o_state_maintenance_read_data;
    logic i_trace_maintenance_write_valid = '0;
    logic o_trace_maintenance_write_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] i_trace_maintenance_write_addr = '0;
    logic [319:0] i_trace_maintenance_write_data = '0;
    logic [39:0] i_trace_maintenance_write_mask = '0;
    logic i_trace_maintenance_read_valid = '0;
    logic o_trace_maintenance_read_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] i_trace_maintenance_read_addr = '0;
    logic o_trace_maintenance_result_valid;
    logic i_trace_maintenance_result_ready = '0;
    logic [MEMORY_ADDR_WIDTH-1:0] o_trace_maintenance_result_addr;
    logic [319:0] o_trace_maintenance_read_data;
    logic o_classifier_cache_valid;
    logic o_active_classifier_pair;
    logic [VERSION_WIDTH-1:0] o_parameter_version;
    logic o_conv_cache_valid;
    logic [9:0] o_conv_cached_output_base;
    logic [4:0] o_conv_cached_output_count;
    logic [VERSION_WIDTH-1:0] o_conv_cached_version;
    logic o_conv_cache_invalidate;
    logic [VERSION_WIDTH-1:0] o_model_version;
    logic [VERSION_WIDTH-1:0] o_adapt_version;
    logic o_capture_complete;
    logic o_learning_start_fire;
    logic o_commit_pulse;
    logic o_commit_code_dirty;
    logic o_commit_optimizer_dirty;
    logic o_detach_pulse;
    logic o_commit_pending;
    logic o_inference_hold;
    logic o_initialized;
    logic o_versions_aligned;
    logic [1:0] o_array_active_owner;
    logic [15:0] o_array_tile_idle;
    logic [15:0] o_array_tile_sticky_error;
    logic o_context_active;
    logic o_context_is_classifier;
    logic o_busy;
    logic o_quiescent;
    logic o_sticky_error;
    logic i_clk = '0;
    logic i_rst_n = '0;
    logic i_clear_error = '0;
    logic i_srnn_phase_abort = '0;
    logic i_srnn_phase_capture_cfg_valid = '0;
    logic o_srnn_phase_capture_cfg_ready;
    logic [$clog2(SRNN_BACKBONE_MAX + 1)-1:0] i_srnn_phase_record_count = '0;
    logic i_srnn_phase_start_valid = '0;
    logic o_srnn_phase_start_ready;
    logic i_srnn_phase_learning_enable = '0;
    logic i_srnn_phase_target_valid = '0;
    logic o_srnn_phase_target_ready;
    logic [5:0] i_srnn_phase_target_sample_id = '0;
    logic [9:0] i_srnn_phase_target_backbone_id = '0;
    logic signed [31:0] i_srnn_phase_target_q14 = '0;
    logic i_srnn_phase_target_last = '0;
    logic i_srnn_phase_target_done = '0;
    logic i_srnn_phase_target_error = '0;
    logic o_srnn_phase_capture_done;
    logic o_srnn_phase_capture_complete;
    logic o_srnn_phase_delivery_done;
    logic o_srnn_phase_busy;
    logic o_srnn_phase_idle;
    logic o_srnn_phase_sticky_error;
    logic [2:0] i_parameter_owner = '0;
    logic [2:0] i_topology_learning_enable = '0;
    logic i_learning_fc_cfg_valid = '0;
    logic o_learning_fc_cfg_ready;
    logic [7:0] i_learning_fc_profile_id = '0;
    logic [7:0] i_learning_fc_w2_tensor_id = '0;
    logic [PARAM_ADDR_WIDTH-1:0] i_learning_fc_w2_parameter_base = '0;
    logic [4:0] i_learning_fc_batch_count = '0;
    logic [8:0] i_learning_fc_hidden_count = '0;
    logic [4:0] i_learning_fc_output_count = '0;
    logic [15:0] i_learning_fc_vote_increment = '0;
    logic [15:0] i_learning_fc_vote_threshold = '0;
    logic signed [CODE_WIDTH-1:0] i_learning_fc_code_min = '0;
    logic signed [CODE_WIDTH-1:0] i_learning_fc_code_max = '0;
    logic o_learning_fc_cfg_error;
    logic i_learning_fc_release = '0;
    logic o_learning_fc_release_ready;
    logic o_learning_fc_window_commit_valid;
    logic i_learning_fc_window_commit_ready = '0;
    logic [31:0] o_learning_fc_window_commit_sequence;
    logic o_learning_fc_window_busy;
    logic o_learning_fc_error;
    logic i_learning_fc_parameter_load_valid = '0;
    logic o_learning_fc_parameter_load_ready;
    logic [PARAM_ADDR_WIDTH-1:0] i_learning_fc_parameter_load_addr = '0;
    logic signed [CODE_WIDTH-1:0] i_learning_fc_parameter_load_code = '0;
    logic signed [RESIDUAL_WIDTH-1:0] i_learning_fc_parameter_load_residual = '0;
    logic i_learning_srnn_cfg_valid = '0;
    logic o_learning_srnn_cfg_ready;
    logic [7:0] i_learning_srnn_profile_id = '0;
    logic [7:0] i_learning_srnn_wh1_tensor_id = '0;
    logic [PARAM_ADDR_WIDTH-1:0] i_learning_srnn_wh1_base = '0;
    logic [15:0] i_learning_srnn_wh1_increment = '0;
    logic [15:0] i_learning_srnn_wh1_threshold = '0;
    logic [7:0] i_learning_srnn_wout_tensor_id = '0;
    logic [PARAM_ADDR_WIDTH-1:0] i_learning_srnn_wout_base = '0;
    logic [15:0] i_learning_srnn_wout_increment = '0;
    logic [15:0] i_learning_srnn_wout_threshold = '0;
    logic signed [CODE_WIDTH-1:0] i_learning_srnn_code_min = '0;
    logic signed [CODE_WIDTH-1:0] i_learning_srnn_code_max = '0;
    logic [7:0] i_learning_srnn_batch_count = '0;
    logic [8:0] i_learning_srnn_hidden_count = '0;
    logic [9:0] i_learning_srnn_backbone_count = '0;
    logic [4:0] i_learning_srnn_class_count = '0;
    logic [PARAM_ADDR_WIDTH-1:0] i_learning_srnn_wout_element_count = '0;
    logic o_learning_srnn_cfg_error;
    logic i_learning_srnn_parameter_load_valid = '0;
    logic o_learning_srnn_parameter_load_ready;
    logic [PARAM_ADDR_WIDTH-1:0] i_learning_srnn_parameter_load_addr = '0;
    logic signed [CODE_WIDTH-1:0] i_learning_srnn_parameter_load_code = '0;
    logic signed [RESIDUAL_WIDTH-1:0] i_learning_srnn_parameter_load_residual = '0;
    logic [$clog2(SRNN_TIMESTEP_MAX)-1:0] i_learning_srnn_timestep_id = '0;
    logic [15:0] i_learning_srnn_sequence_version = '0;
    logic i_learning_srnn_abort_timestep = '0;
    logic o_learning_srnn_abort_ready;
    logic o_learning_srnn_target_trace_start;
    logic i_learning_srnn_target_trace_start_ready = '0;
    logic [$clog2(SRNN_TIMESTEP_MAX)-1:0] o_learning_srnn_target_trace_timestep_id;
    logic i_learning_srnn_target_trace_valid = '0;
    logic o_learning_srnn_target_trace_ready;
    logic [7:0] i_learning_srnn_target_trace_sample_id = '0;
    logic [9:0] i_learning_srnn_target_trace_backbone_id = '0;
    logic signed [31:0] i_learning_srnn_target_trace_q14 = '0;
    logic i_learning_srnn_target_trace_last = '0;
    logic i_learning_srnn_target_trace_done = '0;
    logic i_learning_srnn_target_trace_error = '0;
    logic o_learning_srnn_label_replay_start;
    logic i_learning_srnn_label_replay_start_ready = '0;
    logic i_learning_srnn_label_valid = '0;
    logic o_learning_srnn_label_ready;
    logic [7:0] i_learning_srnn_label_sample_id = '0;
    logic [4:0] i_learning_srnn_label_class_id = '0;
    logic i_learning_srnn_label_last = '0;
    logic i_learning_srnn_restore_complete = '0;
    logic o_learning_srnn_restore_ready;
    logic o_learning_srnn_recovery_required;
    logic o_learning_srnn_learning_commit;
    logic o_learning_srnn_detach;
    logic o_learning_srnn_logit_valid;
    logic i_learning_srnn_logit_ready = '0;
    logic [7:0] o_learning_srnn_logit_sample_id;
    logic [4:0] o_learning_srnn_logit_class_id;
    logic signed [63:0] o_learning_srnn_logit_q28;
    logic o_learning_srnn_logit_last;
    logic o_learning_srnn_timestep_done;
    logic o_learning_srnn_timestep_aborted;
    logic o_learning_srnn_busy;
    logic o_learning_srnn_error;
    logic [PARAM_ADDR_WIDTH:0] o_learning_srnn_zero_gradient_count;
    logic [PARAM_ADDR_WIDTH:0] o_learning_srnn_update_count;
    logic o_store_load_valid;
    logic i_store_load_ready;
    logic [PARAM_ADDR_WIDTH-1:0] o_store_load_addr;
    logic signed [CODE_WIDTH-1:0] o_store_load_code;
    logic signed [RESIDUAL_WIDTH-1:0] o_store_load_residual;
    logic o_store_learning_read_valid;
    logic i_store_learning_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] o_store_learning_read_addr;
    logic [TAG_WIDTH-1:0] o_store_learning_read_tag;
    logic i_store_learning_response_valid;
    logic o_store_learning_response_ready;
    logic [TAG_WIDTH-1:0] i_store_learning_response_tag;
    logic signed [CODE_WIDTH-1:0] i_store_learning_response_code;
    logic signed [RESIDUAL_WIDTH-1:0] i_store_learning_response_residual;
    logic i_store_learning_response_error;
    logic o_store_learning_write_valid;
    logic i_store_learning_write_ready;
    logic [PARAM_ADDR_WIDTH-1:0] o_store_learning_write_addr;
    logic signed [CODE_WIDTH-1:0] o_store_learning_write_code;
    logic signed [RESIDUAL_WIDTH-1:0] o_store_learning_write_residual;
    logic o_store_learning_write_code_enable;
    logic o_store_learning_write_residual_enable;
    logic o_store_forward_read_valid;
    logic i_store_forward_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] o_store_forward_read_addr;
    logic [TAG_WIDTH-1:0] o_store_forward_read_tag;
    logic i_store_forward_response_valid;
    logic o_store_forward_response_ready;
    logic [TAG_WIDTH-1:0] i_store_forward_response_tag;
    logic signed [CODE_WIDTH-1:0] i_store_forward_response_code;
    logic i_store_forward_response_error;
    logic i_store_quiescent;
    logic i_store_protocol_error;
    logic o_backend_quiescent;
    logic o_backend_error;
    logic o_owner_encoding_error;
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
    ) u_final (
        .i_raw_source_active(feed_active),
        .i_raw_count_valid(count_valid),
        .i_raw_counts(count_word),
        .o_raw_count_ready(count_ready),
        .i_initialize_valid(i_initialize_valid),
        .o_initialize_ready(o_initialize_ready),
        .o_initialize_done(o_initialize_done),
        .i_classifier_pair(i_classifier_pair),
        .i_context_model_version(i_context_model_version),
        .i_context_adapt_version(i_context_adapt_version),
        .i_system_parameter_owner(i_system_parameter_owner),
        .i_system_learning_enable(i_system_learning_enable),
        .i_system_profile_aligned(i_system_profile_aligned),
        .i_system_profile_id(i_system_profile_id),
        .i_request_valid(i_request_valid),
        .o_request_ready(o_request_ready),
        .i_learning_enable(i_learning_enable),
        .i_batch_count(i_batch_count),
        .i_timestep(i_timestep),
        .i_timestep_count(i_timestep_count),
        .i_sample_version(i_sample_version),
        .i_labels_flat(i_labels_flat),
        .o_done(o_done),
        .i_fc_srnn_streamed_feedforward(1'b1),
        .i_fc_srnn_group_feedforward_end(1'b0),
        .i_fc_srnn_layer_valid(i_fc_srnn_layer_valid),
        .o_fc_srnn_layer_ready(o_fc_srnn_layer_ready),
        .i_fc_srnn_mode(i_fc_srnn_mode),
        .i_fc_srnn_weight_int8(i_fc_srnn_weight_int8),
        .i_fc_srnn_current_shift6(i_fc_srnn_current_shift6),
        .i_fc_srnn_neuron_count(i_fc_srnn_neuron_count),
        .i_fc_srnn_feedforward_group_count(i_fc_srnn_feedforward_group_count),
        .i_fc_srnn_recurrent_group_count(i_fc_srnn_recurrent_group_count),
        .i_fc_srnn_recurrent_enable(i_fc_srnn_recurrent_enable),
        .i_fc_srnn_leak(i_fc_srnn_leak),
        .i_fc_srnn_threshold(i_fc_srnn_threshold),
        .i_fc_srnn_reset_to_zero(i_fc_srnn_reset_to_zero),
        .i_fc_srnn_trace_decay(i_fc_srnn_trace_decay),
        .i_fc_srnn_transaction_tag(i_fc_srnn_transaction_tag),
        .i_fc_srnn_work_base_addr(i_fc_srnn_work_base_addr),
        .i_fc_srnn_destination_tile_base(i_fc_srnn_destination_tile_base),
        .i_fc_srnn_group_valid(manual_group_valid),
        .o_fc_srnn_group_ready(o_fc_srnn_group_ready),
        .i_fc_srnn_group_recurrent(manual_group_recurrent),
        .i_fc_srnn_group_source_valid(manual_group_source_valid),
        .i_fc_srnn_group_weight_addr0(manual_group_weight_addr0),
        .i_fc_srnn_group_weight_addr1(manual_group_weight_addr1),
        .i_fc_srnn_group_weight_addr2(manual_group_weight_addr2),
        .i_fc_srnn_group_weight_addr3(manual_group_weight_addr3),
        .i_srnn_group_streamer_select(i_srnn_group_streamer_select),
        .i_srnn_group_streamer_start_valid(i_srnn_group_streamer_start_valid),
        .o_srnn_group_streamer_start_ready(o_srnn_group_streamer_start_ready),
        .i_srnn_group_feedforward_source_count(i_srnn_group_feedforward_source_count),
        .i_srnn_group_recurrent_source_count(i_srnn_group_recurrent_source_count),
        .i_srnn_group_feedforward_cache_row_base(i_srnn_group_feedforward_cache_row_base),
        .i_srnn_group_recurrent_cache_row_base(i_srnn_group_recurrent_cache_row_base),
        .i_srnn_group_feedforward_spike_flat(i_srnn_group_feedforward_spike_flat),
        .i_srnn_group_recurrent_spike_flat(i_srnn_group_recurrent_spike_flat),
        .o_srnn_group_streamer_done(o_srnn_group_streamer_done),
        .o_srnn_group_streamer_busy(o_srnn_group_streamer_busy),
        .o_srnn_group_streamer_sticky_error(o_srnn_group_streamer_sticky_error),
        .o_srnn_group_streamer_error_code(o_srnn_group_streamer_error_code),
        .o_fc_srnn_layer_done(o_fc_srnn_layer_done),
        .o_fc_srnn_busy(o_fc_srnn_busy),
        .o_fc_srnn_idle(o_fc_srnn_idle),
        .o_fc_srnn_sticky_error(o_fc_srnn_sticky_error),
        .o_fc_srnn_error_code(o_fc_srnn_error_code),
        .o_fc_srnn_observation_valid(o_fc_srnn_observation_valid),
        .i_fc_srnn_observation_ready(i_fc_srnn_observation_ready),
        .i_fc_srnn_observation_sink_idle(i_fc_srnn_observation_sink_idle),
        .o_fc_srnn_observation_mode(o_fc_srnn_observation_mode),
        .o_fc_srnn_observation_transaction_tag(o_fc_srnn_observation_transaction_tag),
        .o_fc_srnn_observation_neuron_id(o_fc_srnn_observation_neuron_id),
        .o_fc_srnn_observation_spike(o_fc_srnn_observation_spike),
        .o_fc_srnn_observation_pre_state(o_fc_srnn_observation_pre_state),
        .o_fc_srnn_observation_trace_next(o_fc_srnn_observation_trace_next),
        .o_fc_srnn_observation_last_in_record(o_fc_srnn_observation_last_in_record),
        .o_fc_srnn_observation_busy(o_fc_srnn_observation_busy),
        .o_fc_srnn_observation_sticky_error(o_fc_srnn_observation_sticky_error),
        .i_fc_srnn_learning_cfg_valid(i_fc_srnn_learning_cfg_valid),
        .o_fc_srnn_learning_cfg_ready(o_fc_srnn_learning_cfg_ready),
        .i_fc_srnn_learning_cfg_mode(i_fc_srnn_learning_cfg_mode),
        .i_fc_srnn_learning_cfg_tag(i_fc_srnn_learning_cfg_tag),
        .i_fc_srnn_learning_sample_id(i_fc_srnn_learning_sample_id),
        .i_fc_srnn_learning_neuron_base(i_fc_srnn_learning_neuron_base),
        .i_fc_srnn_learning_neuron_count(i_fc_srnn_learning_neuron_count),
        .i_fc_srnn_learning_fc_threshold(i_fc_srnn_learning_fc_threshold),
        .i_fc_srnn_learning_fc_output_layer(i_fc_srnn_learning_fc_output_layer),
        .i_fc_srnn_learning_fc_label(i_fc_srnn_learning_fc_label),
        .i_fc_srnn_learning_fc_logit_scale_enable(i_fc_srnn_learning_fc_logit_scale_enable),
        .i_fc_srnn_learning_fc_logit_scale_q16(i_fc_srnn_learning_fc_logit_scale_q16),
        .i_fc_srnn_learning_fc_logit_accumulate(i_fc_srnn_learning_fc_logit_accumulate),
        .i_fc_srnn_learning_fc_sample_start(i_fc_srnn_learning_fc_sample_start),
        .i_fc_srnn_learning_srnn_path(i_fc_srnn_learning_srnn_path),
        .o_fc_surrogate_valid(o_fc_surrogate_valid),
        .o_fc_surrogate_sample(o_fc_surrogate_sample),
        .o_fc_surrogate_hidden(o_fc_surrogate_hidden),
        .o_fc_surrogate_q016(o_fc_surrogate_q016),
        .o_fc_logit_last(o_fc_logit_last),
        .o_fc_srnn_learning_capture_done(o_fc_srnn_learning_capture_done),
        .o_fc_srnn_learning_done(o_fc_srnn_learning_done),
        .o_fc_srnn_learning_busy(o_fc_srnn_learning_busy),
        .o_fc_srnn_learning_idle(o_fc_srnn_learning_idle),
        .o_fc_srnn_learning_sticky_error(o_fc_srnn_learning_sticky_error),
        .o_fc_srnn_learning_error_source(o_fc_srnn_learning_error_source),
        .o_fc_srnn_learning_error_code(o_fc_srnn_learning_error_code),
        .i_fc_w0_refill_valid(i_fc_w0_refill_valid),
        .o_fc_w0_refill_ready(o_fc_w0_refill_ready),
        .i_fc_w0_refill_all(i_fc_w0_refill_all),
        .i_fc_cache_refill_weight_int8(i_fc_cache_refill_weight_int8),
        .i_fc_cache_refill_destination_tile_base(i_fc_cache_refill_destination_tile_base),
        .i_fc_w0_dirty_feature(i_fc_w0_dirty_feature),
        .i_fc_w0_dirty_tile(i_fc_w0_dirty_tile),
        .i_fc_w0_parameter_base(i_fc_w0_parameter_base),
        .i_fc_w0_hidden_stride(i_fc_w0_hidden_stride),
        .i_fc_cache_row_base(i_fc_cache_row_base),
        .i_fc_w0_feature_count(i_fc_w0_feature_count),
        .i_fc_w0_hidden_count(i_fc_w0_hidden_count),
        .i_fc_w0_tile_count(i_fc_w0_tile_count),
        .i_fc_w0_version(i_fc_w0_version),
        .o_fc_w0_refill_done(o_fc_w0_refill_done),
        .o_fc_w0_rows_published(o_fc_w0_rows_published),
        .o_fc_w0_published_version(o_fc_w0_published_version),
        .o_cache_fill_grant(o_cache_fill_grant),
        .o_cache_refill_busy(o_cache_refill_busy),
        .o_cache_refill_quiescent(o_cache_refill_quiescent),
        .o_cache_refill_sticky_error(o_cache_refill_sticky_error),
        .i_srnn_backbone_refill_valid(i_srnn_backbone_refill_valid),
        .o_srnn_backbone_refill_ready(o_srnn_backbone_refill_ready),
        .i_srnn_backbone_refill_all(i_srnn_backbone_refill_all),
        .i_srnn_backbone_dirty_source(i_srnn_backbone_dirty_source),
        .i_srnn_backbone_dirty_tile(i_srnn_backbone_dirty_tile),
        .i_srnn_backbone_parameter_base(i_srnn_backbone_parameter_base),
        .i_srnn_backbone_cache_row_base(i_srnn_backbone_cache_row_base),
        .i_srnn_backbone_source_count(i_srnn_backbone_source_count),
        .i_srnn_backbone_destination_count(i_srnn_backbone_destination_count),
        .i_srnn_backbone_tile_base(i_srnn_backbone_tile_base),
        .i_srnn_backbone_version(i_srnn_backbone_version),
        .o_srnn_backbone_forward_read_valid(o_srnn_backbone_forward_read_valid),
        .i_srnn_backbone_forward_read_ready(i_srnn_backbone_forward_read_ready),
        .o_srnn_backbone_forward_read_addr(o_srnn_backbone_forward_read_addr),
        .o_srnn_backbone_forward_read_tag(o_srnn_backbone_forward_read_tag),
        .i_srnn_backbone_forward_response_valid(i_srnn_backbone_forward_response_valid),
        .o_srnn_backbone_forward_response_ready(o_srnn_backbone_forward_response_ready),
        .i_srnn_backbone_forward_response_tag(i_srnn_backbone_forward_response_tag),
        .i_srnn_backbone_forward_response_code(i_srnn_backbone_forward_response_code),
        .i_srnn_backbone_forward_response_error(i_srnn_backbone_forward_response_error),
        .o_srnn_backbone_refill_done(o_srnn_backbone_refill_done),
        .o_srnn_backbone_rows_published(o_srnn_backbone_rows_published),
        .o_srnn_backbone_published_version(o_srnn_backbone_published_version),
        .i_profile_id(i_profile_id),
        .i_conv_tensor_id(i_conv_tensor_id),
        .i_classifier_tensor_id(i_classifier_tensor_id),
        .i_total_input_channels(i_total_input_channels),
        .i_total_output_channels(i_total_output_channels),
        .i_conv_parameter_base(i_conv_parameter_base),
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
        .i_code_min(i_code_min),
        .i_code_max(i_code_max),
        .i_forward_start_valid(i_forward_start_valid),
        .o_forward_start_ready(o_forward_start_ready),
        .o_forward_start_fire(o_forward_start_fire),
        .i_forward_done(i_forward_done),
        .o_forward_active(o_forward_active),
        .o_forward_model_version(o_forward_model_version),
        .o_forward_adapt_version(o_forward_adapt_version),
        .i_restore_complete(i_restore_complete),
        .o_restore_ready(o_restore_ready),
        .o_recovery_required(o_recovery_required),
        .i_conv_in_width(i_conv_in_width),
        .i_conv_in_height(i_conv_in_height),
        .i_conv_in_channels(i_conv_in_channels),
        .i_conv_out_width(i_conv_out_width),
        .i_conv_out_height(i_conv_out_height),
        .i_conv_out_channels(i_conv_out_channels),
        .i_conv_stride(i_conv_stride),
        .i_conv_padding(i_conv_padding),
        .i_conv_pool_mode(i_conv_pool_mode),
        .i_conv_work_base_addr(i_conv_work_base_addr),
        .i_conv_leak(i_conv_leak),
        .i_conv_threshold(i_conv_threshold),
        .i_conv_reset_to_zero(i_conv_reset_to_zero),
        .i_conv_trace_decay(i_conv_trace_decay),
        .i_conv_layer_tag(i_conv_layer_tag),
        .i_target_feature_count(i_target_feature_count),
        .i_target_trace_leak_q8_8(i_target_trace_leak_q8_8),
        .i_target_trace_state_reset(i_target_trace_state_reset),
        .i_classifier_feature_channel_base(i_classifier_feature_channel_base),
        .i_classifier_feature_channel_count(i_classifier_feature_channel_count),
        .i_classifier_work_base_addr(i_classifier_work_base_addr),
        .i_classifier_trace_decay(i_classifier_trace_decay),
        .i_classifier_layer_tag(i_classifier_layer_tag),
        .i_frame_index(i_frame_index),
        .i_graph_descriptor_valid(i_graph_descriptor_valid),
        .o_graph_descriptor_ready(o_graph_descriptor_ready),
        .i_graph_descriptor_last(i_graph_descriptor_last),
        .o_graph_descriptor_done(o_graph_descriptor_done),
        .o_graph_frame_active(o_graph_frame_active),
        .o_graph_frame_target(o_graph_frame_target),
        .o_graph_frame_sample(o_graph_frame_sample),
        .o_graph_frame_timestep(o_graph_frame_timestep),
        .o_graph_frame_version(o_graph_frame_version),
        .o_state_service_valid(o_state_service_valid),
        .i_state_service_ready(i_state_service_ready),
        .o_state_service_release(o_state_service_release),
        .o_state_service_is_classifier(o_state_service_is_classifier),
        .o_state_service_output_channel_base(o_state_service_output_channel_base),
        .o_state_service_output_channel_count(o_state_service_output_channel_count),
        .o_state_service_spatial_base(o_state_service_spatial_base),
        .o_state_service_spatial_count(o_state_service_spatial_count),
        .o_state_service_active_pair(o_state_service_active_pair),
        .o_state_service_work_base_addr(o_state_service_work_base_addr),
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
        .i_feature_mask0(i_feature_mask0),
        .i_feature_mask1(i_feature_mask1),
        .i_feature_mask2(i_feature_mask2),
        .i_feature_mask3(i_feature_mask3),
        .i_feature_service_error(i_feature_service_error),
        .i_feature_write_slot(i_feature_write_slot),
        .i_feature_write_base_addr(i_feature_write_base_addr),
        .i_feature_write_row_stride_words(i_feature_write_row_stride_words),
        .i_feature_write_channel_stride_words(i_feature_write_channel_stride_words),
        .o_feature_update_valid(o_feature_update_valid),
        .i_feature_update_ready(i_feature_update_ready),
        .o_feature_update_slot(o_feature_update_slot),
        .o_feature_update_addr(o_feature_update_addr),
        .o_feature_update_data(o_feature_update_data),
        .o_feature_update_mask(o_feature_update_mask),
        .i_feature_update_done(i_feature_update_done),
        .i_maintenance_tile(i_maintenance_tile),
        .i_state_maintenance_write_valid(i_state_maintenance_write_valid),
        .o_state_maintenance_write_ready(o_state_maintenance_write_ready),
        .i_state_maintenance_write_addr(i_state_maintenance_write_addr),
        .i_state_maintenance_write_data(i_state_maintenance_write_data),
        .i_state_maintenance_write_mask(i_state_maintenance_write_mask),
        .i_state_maintenance_read_valid(i_state_maintenance_read_valid),
        .o_state_maintenance_read_ready(o_state_maintenance_read_ready),
        .i_state_maintenance_read_addr(i_state_maintenance_read_addr),
        .o_state_maintenance_result_valid(o_state_maintenance_result_valid),
        .i_state_maintenance_result_ready(i_state_maintenance_result_ready),
        .o_state_maintenance_result_addr(o_state_maintenance_result_addr),
        .o_state_maintenance_read_data(o_state_maintenance_read_data),
        .i_trace_maintenance_write_valid(i_trace_maintenance_write_valid),
        .o_trace_maintenance_write_ready(o_trace_maintenance_write_ready),
        .i_trace_maintenance_write_addr(i_trace_maintenance_write_addr),
        .i_trace_maintenance_write_data(i_trace_maintenance_write_data),
        .i_trace_maintenance_write_mask(i_trace_maintenance_write_mask),
        .i_trace_maintenance_read_valid(i_trace_maintenance_read_valid),
        .o_trace_maintenance_read_ready(o_trace_maintenance_read_ready),
        .i_trace_maintenance_read_addr(i_trace_maintenance_read_addr),
        .o_trace_maintenance_result_valid(o_trace_maintenance_result_valid),
        .i_trace_maintenance_result_ready(i_trace_maintenance_result_ready),
        .o_trace_maintenance_result_addr(o_trace_maintenance_result_addr),
        .o_trace_maintenance_read_data(o_trace_maintenance_read_data),
        .o_classifier_cache_valid(o_classifier_cache_valid),
        .o_active_classifier_pair(o_active_classifier_pair),
        .o_parameter_version(o_parameter_version),
        .o_conv_cache_valid(o_conv_cache_valid),
        .o_conv_cached_output_base(o_conv_cached_output_base),
        .o_conv_cached_output_count(o_conv_cached_output_count),
        .o_conv_cached_version(o_conv_cached_version),
        .o_conv_cache_invalidate(o_conv_cache_invalidate),
        .o_model_version(o_model_version),
        .o_adapt_version(o_adapt_version),
        .o_capture_complete(o_capture_complete),
        .o_learning_start_fire(o_learning_start_fire),
        .o_commit_pulse(o_commit_pulse),
        .o_commit_code_dirty(o_commit_code_dirty),
        .o_commit_optimizer_dirty(o_commit_optimizer_dirty),
        .o_detach_pulse(o_detach_pulse),
        .o_commit_pending(o_commit_pending),
        .o_inference_hold(o_inference_hold),
        .o_initialized(o_initialized),
        .o_versions_aligned(o_versions_aligned),
        .o_array_active_owner(o_array_active_owner),
        .o_array_tile_idle(o_array_tile_idle),
        .o_array_tile_sticky_error(o_array_tile_sticky_error),
        .o_context_active(o_context_active),
        .o_context_is_classifier(o_context_is_classifier),
        .o_busy(o_busy),
        .o_quiescent(o_quiescent),
        .o_sticky_error(o_sticky_error),
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_srnn_phase_abort(i_srnn_phase_abort),
        .i_srnn_phase_capture_cfg_valid(i_srnn_phase_capture_cfg_valid),
        .o_srnn_phase_capture_cfg_ready(o_srnn_phase_capture_cfg_ready),
        .i_srnn_phase_record_count(i_srnn_phase_record_count),
        .i_srnn_phase_start_valid(i_srnn_phase_start_valid),
        .o_srnn_phase_start_ready(o_srnn_phase_start_ready),
        .i_srnn_phase_learning_enable(i_srnn_phase_learning_enable),
        .i_srnn_phase_target_valid(i_srnn_phase_target_valid),
        .o_srnn_phase_target_ready(o_srnn_phase_target_ready),
        .i_srnn_phase_target_sample_id(i_srnn_phase_target_sample_id),
        .i_srnn_phase_target_backbone_id(i_srnn_phase_target_backbone_id),
        .i_srnn_phase_target_q14(i_srnn_phase_target_q14),
        .i_srnn_phase_target_last(i_srnn_phase_target_last),
        .i_srnn_phase_target_done(i_srnn_phase_target_done),
        .i_srnn_phase_target_error(i_srnn_phase_target_error),
        .o_srnn_phase_capture_done(o_srnn_phase_capture_done),
        .o_srnn_phase_capture_complete(o_srnn_phase_capture_complete),
        .o_srnn_phase_delivery_done(o_srnn_phase_delivery_done),
        .o_srnn_phase_busy(o_srnn_phase_busy),
        .o_srnn_phase_idle(o_srnn_phase_idle),
        .o_srnn_phase_sticky_error(o_srnn_phase_sticky_error),
        .i_parameter_owner(i_parameter_owner),
        .i_topology_learning_enable(i_topology_learning_enable),
        .i_learning_fc_cfg_valid(i_learning_fc_cfg_valid),
        .o_learning_fc_cfg_ready(o_learning_fc_cfg_ready),
        .i_learning_fc_profile_id(i_learning_fc_profile_id),
        .i_learning_fc_w2_tensor_id(i_learning_fc_w2_tensor_id),
        .i_learning_fc_w2_parameter_base(i_learning_fc_w2_parameter_base),
        .i_learning_fc_batch_count(i_learning_fc_batch_count),
        .i_learning_fc_hidden_count(i_learning_fc_hidden_count),
        .i_learning_fc_output_count(i_learning_fc_output_count),
        .i_learning_fc_vote_increment(i_learning_fc_vote_increment),
        .i_learning_fc_vote_threshold(i_learning_fc_vote_threshold),
        .i_learning_fc_code_min(i_learning_fc_code_min),
        .i_learning_fc_code_max(i_learning_fc_code_max),
        .o_learning_fc_cfg_error(o_learning_fc_cfg_error),
        .i_learning_fc_release(i_learning_fc_release),
        .o_learning_fc_release_ready(o_learning_fc_release_ready),
        .o_learning_fc_window_commit_valid(o_learning_fc_window_commit_valid),
        .i_learning_fc_window_commit_ready(i_learning_fc_window_commit_ready),
        .o_learning_fc_window_commit_sequence(o_learning_fc_window_commit_sequence),
        .o_learning_fc_window_busy(o_learning_fc_window_busy),
        .o_learning_fc_error(o_learning_fc_error),
        .i_learning_fc_parameter_load_valid(i_learning_fc_parameter_load_valid),
        .o_learning_fc_parameter_load_ready(o_learning_fc_parameter_load_ready),
        .i_learning_fc_parameter_load_addr(i_learning_fc_parameter_load_addr),
        .i_learning_fc_parameter_load_code(i_learning_fc_parameter_load_code),
        .i_learning_fc_parameter_load_residual(i_learning_fc_parameter_load_residual),
        .i_learning_srnn_cfg_valid(i_learning_srnn_cfg_valid),
        .o_learning_srnn_cfg_ready(o_learning_srnn_cfg_ready),
        .i_learning_srnn_profile_id(i_learning_srnn_profile_id),
        .i_learning_srnn_wh1_tensor_id(i_learning_srnn_wh1_tensor_id),
        .i_learning_srnn_wh1_base(i_learning_srnn_wh1_base),
        .i_learning_srnn_wh1_increment(i_learning_srnn_wh1_increment),
        .i_learning_srnn_wh1_threshold(i_learning_srnn_wh1_threshold),
        .i_learning_srnn_wout_tensor_id(i_learning_srnn_wout_tensor_id),
        .i_learning_srnn_wout_base(i_learning_srnn_wout_base),
        .i_learning_srnn_wout_increment(i_learning_srnn_wout_increment),
        .i_learning_srnn_wout_threshold(i_learning_srnn_wout_threshold),
        .i_learning_srnn_code_min(i_learning_srnn_code_min),
        .i_learning_srnn_code_max(i_learning_srnn_code_max),
        .i_learning_srnn_batch_count(i_learning_srnn_batch_count),
        .i_learning_srnn_hidden_count(i_learning_srnn_hidden_count),
        .i_learning_srnn_backbone_count(i_learning_srnn_backbone_count),
        .i_learning_srnn_class_count(i_learning_srnn_class_count),
        .i_learning_srnn_wout_element_count(i_learning_srnn_wout_element_count),
        .o_learning_srnn_cfg_error(o_learning_srnn_cfg_error),
        .i_learning_srnn_parameter_load_valid(i_learning_srnn_parameter_load_valid),
        .o_learning_srnn_parameter_load_ready(o_learning_srnn_parameter_load_ready),
        .i_learning_srnn_parameter_load_addr(i_learning_srnn_parameter_load_addr),
        .i_learning_srnn_parameter_load_code(i_learning_srnn_parameter_load_code),
        .i_learning_srnn_parameter_load_residual(i_learning_srnn_parameter_load_residual),
        .i_learning_srnn_timestep_id(i_learning_srnn_timestep_id),
        .i_learning_srnn_sequence_version(i_learning_srnn_sequence_version),
        .i_learning_srnn_abort_timestep(i_learning_srnn_abort_timestep),
        .o_learning_srnn_abort_ready(o_learning_srnn_abort_ready),
        .o_learning_srnn_target_trace_start(o_learning_srnn_target_trace_start),
        .i_learning_srnn_target_trace_start_ready(i_learning_srnn_target_trace_start_ready),
        .o_learning_srnn_target_trace_timestep_id(o_learning_srnn_target_trace_timestep_id),
        .i_learning_srnn_target_trace_valid(i_learning_srnn_target_trace_valid),
        .o_learning_srnn_target_trace_ready(o_learning_srnn_target_trace_ready),
        .i_learning_srnn_target_trace_sample_id(i_learning_srnn_target_trace_sample_id),
        .i_learning_srnn_target_trace_backbone_id(i_learning_srnn_target_trace_backbone_id),
        .i_learning_srnn_target_trace_q14(i_learning_srnn_target_trace_q14),
        .i_learning_srnn_target_trace_last(i_learning_srnn_target_trace_last),
        .i_learning_srnn_target_trace_done(i_learning_srnn_target_trace_done),
        .i_learning_srnn_target_trace_error(i_learning_srnn_target_trace_error),
        .o_learning_srnn_label_replay_start(o_learning_srnn_label_replay_start),
        .i_learning_srnn_label_replay_start_ready(i_learning_srnn_label_replay_start_ready),
        .i_learning_srnn_label_valid(i_learning_srnn_label_valid),
        .o_learning_srnn_label_ready(o_learning_srnn_label_ready),
        .i_learning_srnn_label_sample_id(i_learning_srnn_label_sample_id),
        .i_learning_srnn_label_class_id(i_learning_srnn_label_class_id),
        .i_learning_srnn_label_last(i_learning_srnn_label_last),
        .i_learning_srnn_restore_complete(i_learning_srnn_restore_complete),
        .o_learning_srnn_restore_ready(o_learning_srnn_restore_ready),
        .o_learning_srnn_recovery_required(o_learning_srnn_recovery_required),
        .o_learning_srnn_learning_commit(o_learning_srnn_learning_commit),
        .o_learning_srnn_detach(o_learning_srnn_detach),
        .o_learning_srnn_logit_valid(o_learning_srnn_logit_valid),
        .i_learning_srnn_logit_ready(i_learning_srnn_logit_ready),
        .o_learning_srnn_logit_sample_id(o_learning_srnn_logit_sample_id),
        .o_learning_srnn_logit_class_id(o_learning_srnn_logit_class_id),
        .o_learning_srnn_logit_q28(o_learning_srnn_logit_q28),
        .o_learning_srnn_logit_last(o_learning_srnn_logit_last),
        .o_learning_srnn_timestep_done(o_learning_srnn_timestep_done),
        .o_learning_srnn_timestep_aborted(o_learning_srnn_timestep_aborted),
        .o_learning_srnn_busy(o_learning_srnn_busy),
        .o_learning_srnn_error(o_learning_srnn_error),
        .o_learning_srnn_zero_gradient_count(o_learning_srnn_zero_gradient_count),
        .o_learning_srnn_update_count(o_learning_srnn_update_count),
        .o_store_load_valid(o_store_load_valid),
        .i_store_load_ready(i_store_load_ready),
        .o_store_load_addr(o_store_load_addr),
        .o_store_load_code(o_store_load_code),
        .o_store_load_residual(o_store_load_residual),
        .o_store_learning_read_valid(o_store_learning_read_valid),
        .i_store_learning_read_ready(i_store_learning_read_ready),
        .o_store_learning_read_addr(o_store_learning_read_addr),
        .o_store_learning_read_tag(o_store_learning_read_tag),
        .i_store_learning_response_valid(i_store_learning_response_valid),
        .o_store_learning_response_ready(o_store_learning_response_ready),
        .i_store_learning_response_tag(i_store_learning_response_tag),
        .i_store_learning_response_code(i_store_learning_response_code),
        .i_store_learning_response_residual(i_store_learning_response_residual),
        .i_store_learning_response_error(i_store_learning_response_error),
        .o_store_learning_write_valid(o_store_learning_write_valid),
        .i_store_learning_write_ready(i_store_learning_write_ready),
        .o_store_learning_write_addr(o_store_learning_write_addr),
        .o_store_learning_write_code(o_store_learning_write_code),
        .o_store_learning_write_residual(o_store_learning_write_residual),
        .o_store_learning_write_code_enable(o_store_learning_write_code_enable),
        .o_store_learning_write_residual_enable(o_store_learning_write_residual_enable),
        .o_store_forward_read_valid(o_store_forward_read_valid),
        .i_store_forward_read_ready(i_store_forward_read_ready),
        .o_store_forward_read_addr(o_store_forward_read_addr),
        .o_store_forward_read_tag(o_store_forward_read_tag),
        .i_store_forward_response_valid(i_store_forward_response_valid),
        .o_store_forward_response_ready(o_store_forward_response_ready),
        .i_store_forward_response_tag(i_store_forward_response_tag),
        .i_store_forward_response_code(i_store_forward_response_code),
        .i_store_forward_response_error(i_store_forward_response_error),
        .i_store_quiescent(i_store_quiescent),
        .i_store_protocol_error(i_store_protocol_error),
        .o_backend_quiescent(o_backend_quiescent),
        .o_backend_error(o_backend_error),
        .o_owner_encoding_error(o_owner_encoding_error)
    );
    unified_tp_shared_forward_rmw_store #(.DEPTH(PARAM_DEPTH), .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH)) u_store (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_load_valid(o_store_load_valid),
        .o_load_ready(i_store_load_ready),
        .i_load_addr(o_store_load_addr),
        .i_load_code(o_store_load_code),
        .i_load_residual(o_store_load_residual),
        .i_learning_read_valid(o_store_learning_read_valid),
        .o_learning_read_ready(i_store_learning_read_ready),
        .i_learning_read_addr(o_store_learning_read_addr),
        .i_learning_read_tag(o_store_learning_read_tag),
        .o_learning_response_valid(i_store_learning_response_valid),
        .i_learning_response_ready(o_store_learning_response_ready),
        .o_learning_response_tag(i_store_learning_response_tag),
        .o_learning_response_code(i_store_learning_response_code),
        .o_learning_response_residual(i_store_learning_response_residual),
        .o_learning_response_error(i_store_learning_response_error),
        .i_learning_write_valid(o_store_learning_write_valid),
        .o_learning_write_ready(i_store_learning_write_ready),
        .i_learning_write_addr(o_store_learning_write_addr),
        .i_learning_write_code(o_store_learning_write_code),
        .i_learning_write_residual(o_store_learning_write_residual),
        .i_learning_write_code_enable(o_store_learning_write_code_enable),
        .i_learning_write_residual_enable(o_store_learning_write_residual_enable),
        .i_forward_read_valid(o_store_forward_read_valid),
        .o_forward_read_ready(i_store_forward_read_ready),
        .i_forward_read_addr(o_store_forward_read_addr),
        .i_forward_read_tag(o_store_forward_read_tag),
        .o_forward_response_valid(i_store_forward_response_valid),
        .i_forward_response_ready(o_store_forward_response_ready),
        .o_forward_response_tag(i_store_forward_response_tag),
        .o_forward_response_code(i_store_forward_response_code),
        .o_forward_response_error(i_store_forward_response_error),
        .o_protocol_error(i_store_protocol_error)
    );
    assign i_store_quiescent = !o_store_load_valid && !o_store_learning_read_valid &&
        !o_store_learning_write_valid && !o_store_forward_read_valid &&
        !i_store_learning_response_valid && !i_store_forward_response_valid;
    localparam int H = 450, T = 88, BONE_BASE = 256;
    logic signed [15:0] w0[0:314999], recurrent[0:202499], model_mem[0:120319];
    logic signed [31:0] bone_pre[0:T*480-1];
    logic [31:0] bone_spike[0:T*480-1];


    logic signed [31:0] m_pre[0:T*256-1], m_state[0:T*256-1], m_trace[0:T*256-1], m_feature[0:T*256-1];
    logic [31:0] m_spike[0:T*256-1];
    logic signed [63:0] m_logit[0:T*20-1], m_gradient[0:T*5120-1];
    logic signed [15:0] m_before[0:T*5120-1], m_after[0:T*5120-1], m_residual[0:T*5120-1];
    logic [7:0] labels[0:0];
    logic [H-1:0] previous_spikes = '0, next_spikes;
    integer timestep = 0, learning = 0, bone_seen = 0, hidden_seen = 0, logit_seen = 0;
    integer layers = 0, captures = 0, replays = 0, done_count = 0, commits = 0, detaches = 0;
    integer code_writes = 0, residual_writes = 0, gradient_seen = 0, store_checks = 0, param_reads = 0;
    longint cycles = 0, initialized_cycle;
    string raw, fixture, mode, fixture_root;
    integer sequence_index=0, completed_steps=0;
    integer native_steps[0:7] = '{70,63,70,81,80,55,81,88};
    always #5 i_clk = !i_clk;

    `define M2 u_final.u_core.u_parameter_plane.g_srnn_client.u_srnn_m2
    `define ENG u_final.u_core.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward
    `define LINK(name) u_final.u_core.link_system_to_plane_srnn_h40_``name
    `define RED `M2.parameter_store.g_wout_only_learning.learning_engine

    assign i_srnn_backbone_forward_read_ready = !i_srnn_backbone_forward_response_valid || o_srnn_backbone_forward_response_ready;
    always @(posedge i_clk) begin
        cycles <= cycles + 1;
        if (cycles > 367000000) $fatal(1,"SAME_DUT_WATCHDOG t=%0d layer=%0d hidden=%0d logits=%0d phase=%0d coordinator=%0d prep=%0d cache=%0b",
            timestep,layers,hidden_seen,logit_seen,o_srnn_phase_busy,`M2.state_q,`ENG.prep_state_q,`ENG.o_cache_valid);
        if (!i_rst_n) i_srnn_backbone_forward_response_valid <= 0;
        else begin
            if (i_srnn_backbone_forward_response_valid && o_srnn_backbone_forward_response_ready)
                i_srnn_backbone_forward_response_valid <= 0;
            if (o_srnn_backbone_forward_read_valid && i_srnn_backbone_forward_read_ready) begin
                param_reads++;
                i_srnn_backbone_forward_response_valid <= 1;
                i_srnn_backbone_forward_response_tag <= o_srnn_backbone_forward_read_tag;
                if (o_srnn_backbone_forward_read_addr < 315000)
                    i_srnn_backbone_forward_response_code <= w0[o_srnn_backbone_forward_read_addr];
                else if (o_srnn_backbone_forward_read_addr < 517500)
                    i_srnn_backbone_forward_response_code <= recurrent[int'(o_srnn_backbone_forward_read_addr)-315000];
                else $fatal(1,"backbone address range");
            end
            if (o_sticky_error || o_backend_error || o_srnn_phase_sticky_error || o_learning_srnn_error ||
                o_learning_srnn_cfg_error || o_fc_srnn_sticky_error || o_fc_srnn_learning_sticky_error ||
                o_fc_srnn_observation_sticky_error || o_array_tile_sticky_error != 0)
                $fatal(1,"SAME_DUT_ERROR t=%0d global=%b backend=%b phase=%b m2=%b layer=%b/%h ingress=%b/%h tiles=%h coord=%0d prep=%0d",
                    timestep,o_sticky_error,o_backend_error,o_srnn_phase_sticky_error,o_learning_srnn_error,
                    o_fc_srnn_sticky_error,o_fc_srnn_error_code,o_fc_srnn_learning_sticky_error,
                    o_fc_srnn_learning_error_code,o_array_tile_sticky_error,`M2.state_q,`ENG.prep_state_q);
            if (o_fc_srnn_layer_done) layers++;
            if (o_fc_srnn_learning_capture_done) captures++;
            if (o_fc_srnn_learning_done) replays++;
            if (o_learning_srnn_timestep_done) done_count++;
            if (o_learning_srnn_learning_commit) commits++;
            if (o_learning_srnn_detach) detaches++;
            if (o_fc_srnn_observation_valid && i_fc_srnn_observation_ready) begin
                int h;
                h = int'(o_fc_srnn_observation_neuron_id);
                if (h >= H || o_fc_srnn_observation_mode != 1 ||
                    o_fc_srnn_observation_pre_state !== bone_pre[timestep*480+h] ||
                    o_fc_srnn_observation_spike !== bone_spike[timestep*480+h][0])
                    $fatal(1,"BONE_MISMATCH t=%0d h=%0d pre=%0d/%0d spike=%b/%b",timestep,h,
                        o_fc_srnn_observation_pre_state,bone_pre[timestep*480+h],
                        o_fc_srnn_observation_spike,bone_spike[timestep*480+h][0]);
                next_spikes[h] = o_fc_srnn_observation_spike;
                bone_seen++;
            end
            if (`LINK(hidden_valid) && u_final.u_core.link_plane_to_system_srnn_h40_hidden_ready) begin
                int idx;
                idx = timestep*256 + int'(`LINK(hidden_id));
                if (`LINK(hidden_sample_id) != 0 || `LINK(hidden_path) != 0 ||
                    `LINK(hidden_id) != hidden_seen%256 ||
                    `LINK(hidden_membrane_pre_q14) !== m_pre[idx] ||
                    `LINK(hidden_membrane_next_q14) !== m_state[idx] ||
                    `LINK(hidden_spike) !== m_spike[idx][0] ||
                    `LINK(hidden_trace_q14) !== m_trace[idx] ||
                    `LINK(hidden_output_feature_q14) !== m_feature[idx])
                    $fatal(1,"M2_HIDDEN_MISMATCH t=%0d h=%0d pre=%0d/%0d state=%0d/%0d spike=%b/%b trace=%0d/%0d",
                        timestep,`LINK(hidden_id),`LINK(hidden_membrane_pre_q14),m_pre[idx],
                        `LINK(hidden_membrane_next_q14),m_state[idx],`LINK(hidden_spike),m_spike[idx][0],
                        `LINK(hidden_trace_q14),m_trace[idx]);
                hidden_seen++;
            end
            if (o_learning_srnn_logit_valid && i_learning_srnn_logit_ready) begin
                if (o_learning_srnn_logit_sample_id != 0 || o_learning_srnn_logit_class_id != logit_seen%20 ||
                    o_learning_srnn_logit_q28 !== m_logit[logit_seen])
                    $fatal(1,"M2_LOGIT_MISMATCH t=%0d c=%0d got=%0d expected=%0d",timestep,
                        o_learning_srnn_logit_class_id,o_learning_srnn_logit_q28,m_logit[logit_seen]);
                logit_seen++;
            end
            if (`RED.reducer_gradient_valid && `RED.reducer_gradient_ready) begin
                if (`RED.reducer_gradient_addr != gradient_seen%5120 ||
                    `RED.reducer_gradient_q40 !== m_gradient[gradient_seen])
                    $fatal(1,"gradient mismatch index=%0d",gradient_seen);
                gradient_seen++;
            end
            if (o_store_learning_write_valid && i_store_learning_write_ready) begin
                if (!learning || o_store_learning_write_addr < 115200 || o_store_learning_write_addr >= 120320)
                    $fatal(1,"unexpected/frozen parameter write");
                if (o_store_learning_write_code_enable) code_writes++;
                if (o_store_learning_write_residual_enable) residual_writes++;
            end
            if (u_final.u_core.link_plane_to_system_srnn_h40_prepare_valid && `LINK(prepare_ready)) begin
                $display("M2_PREPARE t=%0d cache=%b version=%0d request=%0d",timestep,`ENG.o_cache_valid,
                    `ENG.o_cache_version,u_final.u_core.link_plane_to_system_srnn_h40_prepare_version);
                $fflush();
            end
        end
    end

    initial begin
        wait(i_rst_n);
        forever begin
            @(posedge i_clk);
            if (o_learning_srnn_label_replay_start) begin
                @(negedge i_clk); i_learning_srnn_label_valid = 1;
                do @(posedge i_clk); while (!o_learning_srnn_label_ready);
                @(negedge i_clk); i_learning_srnn_label_valid = 0;
            end
        end
    end

    integer streaming=0;
    logic [7:0] raw_counts[0:T*700-1];
    logic feed_active=0, count_valid=0;
    logic [31:0] count_word=0;
    wire count_ready, producer_valid, producer_first, producer_last;
    wire [3:0] producer_mask;
    wire [9:0] producer_row;
    wire [2:0] producer_phase;
    logic manual_group_valid=0, manual_group_recurrent=0;
    logic [3:0] manual_group_source_valid=0;
    logic [10:0] manual_group_weight_addr0=0, manual_group_weight_addr1=0;
    logic [10:0] manual_group_weight_addr2=0, manual_group_weight_addr3=0;
    longint source_words=0, source_groups=0, source_tails=0;
    longint source_count_stalls=0, source_group_stalls=0;
    assign producer_valid = u_final.u_raw_source.producer_valid;
    assign producer_last = u_final.u_raw_source.producer_last;
    assign i_fc_srnn_group_valid = u_final.raw_group_valid;
    assign i_fc_srnn_group_recurrent = u_final.raw_group_recurrent;
    assign i_fc_srnn_group_source_valid = u_final.raw_group_source_valid;
    assign i_fc_srnn_group_weight_addr0 = u_final.raw_group_weight_addr0;
    assign i_fc_srnn_group_weight_addr1 = u_final.raw_group_weight_addr1;
    assign i_fc_srnn_group_weight_addr2 = u_final.raw_group_weight_addr2;
    assign i_fc_srnn_group_weight_addr3 = u_final.raw_group_weight_addr3;
    always @(posedge i_clk) if(i_rst_n) begin
        if(count_valid && count_ready)source_words<=source_words+1;
        if(count_valid && !count_ready)source_count_stalls<=source_count_stalls+1;
        if(feed_active && producer_valid) begin
            if(o_fc_srnn_group_ready)begin
                source_groups<=source_groups+1;
                if(producer_last)source_tails<=source_tails+1;
            end else source_group_stalls<=source_group_stalls+1;
        end
    end
    task automatic feed_raw_frame;
        longint start_cycle, old_words, old_groups, old_tails;
        start_cycle=cycles;old_words=source_words;old_groups=source_groups;old_tails=source_tails;
        @(negedge i_clk);feed_active=1;count_valid=1;
        for(integer b=0;b<175;b++)begin
            for(integer l=0;l<4;l++)count_word[l*8+:8]=raw_counts[timestep*700+b*4+l];
            do @(posedge i_clk);while(!count_ready);
            @(negedge i_clk);
        end
        count_valid=0;
        while(source_tails==old_tails)@(negedge i_clk);
        feed_active=0;
        $display("SOURCE_CORE_FRAME sequence=%0d timestep=%0d cycles=%0d words=%0d groups=%0d tails=%0d",
            sequence_index,timestep,cycles-start_cycle,source_words-old_words,source_groups-old_groups,source_tails-old_tails);
    endtask

    task automatic refill(input bit rec);
        @(negedge i_clk);
        i_srnn_backbone_parameter_base = rec ? 315000 : 0;
        i_srnn_backbone_source_count = rec ? 450 : 700;
        i_srnn_backbone_cache_row_base = rec ? 350 : 0;
        i_srnn_backbone_refill_valid = 1;
        do @(posedge i_clk); while (!o_srnn_backbone_refill_ready);
        @(negedge i_clk); i_srnn_backbone_refill_valid = 0;
        while (!o_srnn_backbone_refill_done) @(negedge i_clk);
        @(negedge i_clk);
    endtask
    task automatic issue(input bit rec, input logic [3:0] mask, input int a0,a1,a2,a3);
        @(negedge i_clk);
        manual_group_recurrent = rec; manual_group_source_valid = mask;
        manual_group_weight_addr0 = 11'(a0); manual_group_weight_addr1 = 11'(a1);
        manual_group_weight_addr2 = 11'(a2); manual_group_weight_addr3 = 11'(a3);
        manual_group_valid = 1;
        do @(posedge i_clk); while (!o_fc_srnn_group_ready);
        @(negedge i_clk); manual_group_valid = 0;
    endtask

    initial begin
        if (!$value$plusargs("RAW=%s",raw) || !$value$plusargs("FIXTURE=%s",fixture) ||
            !$value$plusargs("LEARNING=%d",learning)) $fatal(1,"missing arguments");
        if (!$value$plusargs("STREAMING=%d",streaming)) $fatal(1,"missing source policy");
        if (streaming!=1) $fatal(1,"static wrapper requires streaming=1");
        fixture_root = fixture;
        raw = {fixture_root,"/sequence_0"}; fixture = raw;
        mode = learning ? "/on_" : "/off_";
        $readmemh({raw,"/w0_refill_dest_major.mem"},w0); $readmemh({raw,"/r_refill_dest_major.mem"},recurrent);
        $readmemh({raw,"/backbone_pre.mem"},bone_pre); $readmemh({raw,"/backbone_spike.mem"},bone_spike);
        $readmemh({raw,"/raw700_counts.mem"},raw_counts);
        $readmemh("tb/data/srnn_m2_v28_wh1_initial_q14.mem",model_mem,0,115199);
        $readmemh("tb/data/srnn_m2_v28_wout_initial_q14.mem",model_mem,115200,120319);
        $readmemh({fixture,"/labels.mem"},labels);
        $readmemh({fixture,mode,"pre.mem"},m_pre); $readmemh({fixture,mode,"membrane.mem"},m_state);
        $readmemh({fixture,mode,"spike.mem"},m_spike); $readmemh({fixture,mode,"trace.mem"},m_trace);
        $readmemh({fixture,mode,"feature.mem"},m_feature); $readmemh({fixture,mode,"logit.mem"},m_logit);
        $readmemh({fixture,mode,"wout_before.mem"},m_before); $readmemh({fixture,mode,"wout_after.mem"},m_after);
        $readmemh({fixture,mode,"residual.mem"},m_residual); $readmemh({fixture,mode,"gradient.mem"},m_gradient);
        i_parameter_owner = 3'b010; i_topology_learning_enable = learning ? 3'b010 : 3'b000;
        i_learning_srnn_profile_id = 7; i_learning_srnn_wh1_tensor_id = 16;
        i_learning_srnn_wout_tensor_id = 17; i_learning_srnn_wout_base = 115200;
        i_learning_srnn_wh1_increment = 31; i_learning_srnn_wh1_threshold = 65;
        i_learning_srnn_wout_increment = 31; i_learning_srnn_wout_threshold = 65;
        i_learning_srnn_code_min = -32768; i_learning_srnn_code_max = 32767;
        i_learning_srnn_batch_count = 1; i_learning_srnn_hidden_count = 256;
        i_learning_srnn_backbone_count = 450; i_learning_srnn_class_count = 20;
        i_learning_srnn_wout_element_count = 5120; i_learning_srnn_sequence_version = 1;
        i_learning_srnn_label_replay_start_ready = 1; i_learning_srnn_label_last = 1;
        i_learning_srnn_label_class_id = labels[0][4:0]; i_learning_srnn_logit_ready = 1;
        i_srnn_phase_learning_enable = 1'(learning); i_srnn_phase_record_count = H;
        i_srnn_backbone_refill_all = 1; i_srnn_backbone_destination_count = H; i_srnn_backbone_version = 1;
        i_state_maintenance_write_mask = '1; i_trace_maintenance_write_mask = '1;
        i_state_maintenance_result_ready = 1; i_trace_maintenance_result_ready = 1;
        i_fc_srnn_observation_ready = 1; i_fc_srnn_observation_sink_idle = 1;
        i_fc_srnn_mode = 1; i_fc_srnn_learning_cfg_mode = 1;
        i_fc_srnn_neuron_count = H; i_fc_srnn_learning_neuron_count = H;
        i_fc_srnn_recurrent_group_count = 113; i_fc_srnn_recurrent_enable = 1;
        i_fc_srnn_leak = 27853; i_fc_srnn_threshold = 32; i_fc_srnn_trace_decay = 27853;
        i_fc_srnn_work_base_addr = WORK_ADDR_WIDTH'(BONE_BASE);
        repeat(6) @(negedge i_clk); i_rst_n = 1; profile_enter("config");
        @(negedge i_clk); i_learning_srnn_cfg_valid = 1;
        do @(posedge i_clk); while (!o_learning_srnn_cfg_ready);
        @(negedge i_clk); i_learning_srnn_cfg_valid = 0;
        profile_enter("parameter_load");
        for (int a=0; a<120320; a++) begin
            @(negedge i_clk); i_learning_srnn_parameter_load_valid = 1;
            i_learning_srnn_parameter_load_addr = PARAM_ADDR_WIDTH'(a);
            i_learning_srnn_parameter_load_code = model_mem[a];
            do @(posedge i_clk); while (!o_learning_srnn_parameter_load_ready);
        end
        @(negedge i_clk); i_learning_srnn_parameter_load_valid = 0;
        profile_enter("initial_state");
        for (int tile=0; tile<12; tile++) for (int row=0; row<4; row++) begin
            @(negedge i_clk); i_maintenance_tile = 4'(tile);
            i_state_maintenance_write_addr = 10'(BONE_BASE+row);
            i_trace_maintenance_write_addr = 10'(512+BONE_BASE+row);
            i_state_maintenance_write_valid = 1; i_trace_maintenance_write_valid = 1;
            do @(posedge i_clk); while (!o_state_maintenance_write_ready || !o_trace_maintenance_write_ready);
            @(negedge i_clk); i_state_maintenance_write_valid = 0; i_trace_maintenance_write_valid = 0;
        end
        for (sequence_index=0; sequence_index<8; sequence_index++) begin
            longint setup_begin;
            setup_begin=cycles;
            if (sequence_index != 0) begin
                raw=$sformatf("%s/sequence_%0d",fixture_root,sequence_index); fixture=raw;
        $readmemh({raw,"/backbone_pre.mem"},bone_pre); $readmemh({raw,"/backbone_spike.mem"},bone_spike);
        $readmemh({raw,"/raw700_counts.mem"},raw_counts);
        $readmemh({fixture,"/labels.mem"},labels);
        $readmemh({fixture,mode,"pre.mem"},m_pre); $readmemh({fixture,mode,"membrane.mem"},m_state);
        $readmemh({fixture,mode,"spike.mem"},m_spike); $readmemh({fixture,mode,"trace.mem"},m_trace);
        $readmemh({fixture,mode,"feature.mem"},m_feature); $readmemh({fixture,mode,"logit.mem"},m_logit);
        $readmemh({fixture,mode,"wout_before.mem"},m_before); $readmemh({fixture,mode,"wout_after.mem"},m_after);
        $readmemh({fixture,mode,"residual.mem"},m_residual); $readmemh({fixture,mode,"gradient.mem"},m_gradient);
                i_learning_srnn_label_class_id=labels[0][4:0];
                for (int tile=0; tile<12; tile++) for (int row=0; row<4; row++) begin
                    @(negedge i_clk); i_maintenance_tile=4'(tile);
                    i_state_maintenance_write_addr=10'(BONE_BASE+row);
                    i_trace_maintenance_write_addr=10'(512+BONE_BASE+row);
                    i_state_maintenance_write_valid=1; i_trace_maintenance_write_valid=1;
                    do @(posedge i_clk); while (!o_state_maintenance_write_ready || !o_trace_maintenance_write_ready);
                    @(negedge i_clk); i_state_maintenance_write_valid=0; i_trace_maintenance_write_valid=0;
                end
                previous_spikes='0; next_spikes='0;
                bone_seen=0; hidden_seen=0; logit_seen=0; layers=0; captures=0; replays=0;
                done_count=0; commits=0; detaches=0; code_writes=0; residual_writes=0;
                gradient_seen=0; store_checks=0;
                i_learning_srnn_sequence_version=sequence_index+1;
            end
        initialized_cycle = cycles; profile_enter("boundary");
        $display("REPRESENTATIVE_BEGIN sequence=%0d cycle=%0d setup_begin=%0d",sequence_index,cycles,setup_begin);
        for (timestep=0; timestep<native_steps[sequence_index]; timestep++) begin
            // M2 uses the same cache: restore the backbone working set each step.
            profile_enter("backbone_refill_ff"); if (sequence_index == 0 && timestep == 0) refill(0);
            profile_enter("backbone_refill_r"); if (sequence_index == 0 && timestep == 0) refill(1);
            profile_enter("backbone"); next_spikes = 0;
            @(negedge i_clk); i_srnn_phase_capture_cfg_valid = 1;
            do @(posedge i_clk); while (!o_srnn_phase_capture_cfg_ready);
            @(negedge i_clk); i_srnn_phase_capture_cfg_valid = 0;
            i_fc_srnn_transaction_tag = 52'(timestep); i_fc_srnn_learning_cfg_tag = 52'(timestep);
            i_fc_srnn_learning_cfg_valid = 1;
            do @(posedge i_clk); while (!o_fc_srnn_learning_cfg_ready);
            @(negedge i_clk); i_fc_srnn_learning_cfg_valid = 0;
            i_fc_srnn_feedforward_group_count = 0; i_fc_srnn_layer_valid = 1;
            do @(posedge i_clk); while (!o_fc_srnn_layer_ready);
            @(negedge i_clk); i_fc_srnn_layer_valid = 0;
            feed_raw_frame();
            for (int g=0; g<113; g++) begin
                logic [3:0] mask;
                mask = 0;
                for (int lane=0; lane<4; lane++) if(g*4+lane<H) mask[lane]=previous_spikes[g*4+lane];
                issue(1,mask,350+g*2,350+g*2,350+g*2+1,350+g*2+1);
            end
            while(layers!=timestep+1 || captures!=timestep+1 || replays!=timestep+1 || !o_srnn_phase_capture_complete)
                @(negedge i_clk);
            previous_spikes = next_spikes;
            profile_enter("m2");
            @(negedge i_clk); i_learning_srnn_timestep_id = $clog2(SRNN_TIMESTEP_MAX)'(timestep);
            i_srnn_phase_start_valid = 1;
            do @(posedge i_clk); while(!o_srnn_phase_start_ready);
            @(negedge i_clk); i_srnn_phase_start_valid = 0;
            while(done_count!=timestep+1) @(negedge i_clk);
            profile_enter("tb_store_check");
            for(int a=0; a<120320; a++) begin
                if (u_store.code_mem[a] !== (a<115200 ? model_mem[a] : m_after[timestep*5120+a-115200]) ||
                    u_store.residual_mem[a] !== (a<115200 ? 16'sd0 : m_residual[timestep*5120+a-115200]))
                    $fatal(1,"store mismatch t=%0d a=%0d",timestep,a);
                store_checks++;
            end
            profile_enter("boundary");
            $display("SRNN_SAME_DUT_STEP t=%0d learning=%0d bone=%0d hidden=%0d logits=%0d code=%0d residual=%0d commits=%0d cycles=%0d",
                timestep,learning,bone_seen,hidden_seen,logit_seen,code_writes,residual_writes,commits,cycles-initialized_cycle);
            $fflush();
        end
        if(bone_seen!=native_steps[sequence_index]*H || hidden_seen!=native_steps[sequence_index]*256 || logit_seen!=native_steps[sequence_index]*20 ||
           gradient_seen!=(learning ? native_steps[sequence_index]*5120 : 0) || commits!=(learning ? native_steps[sequence_index] : 0) || detaches!=(learning ? native_steps[sequence_index] : 0))
            $fatal(1,"incomplete coverage bone=%0d hidden=%0d logits=%0d gradient=%0d commits=%0d detach=%0d learning=%0d",
                bone_seen,hidden_seen,logit_seen,gradient_seen,commits,detaches,learning);
        $display("SRNN_SAME_DUT_PASS batch=1 steps=%0d learning=%0d bone=%0d hidden=%0d logits=%0d gradient=%0d store=%0d code=%0d residual=%0d commits=%0d detach=%0d cycles=%0d",
            native_steps[sequence_index],learning,bone_seen,hidden_seen,logit_seen,gradient_seen,store_checks,code_writes,residual_writes,commits,detaches,cycles-initialized_cycle);
        if (resident_backbone_loads != 2 || resident_wh1_loads != 1 ||
            resident_wout_loads < 1 || resident_wout_loads > completed_steps+native_steps[sequence_index] ||
            (!learning && resident_wout_loads != 1))
            $fatal(1,"residency count mismatch backbone=%0d wh1=%0d wout=%0d",
                resident_backbone_loads,resident_wh1_loads,resident_wout_loads);
        $display("RESIDENCY_PASS backbone_loads=%0d wh1_loads=%0d wout_loads=%0d wout_reloads=%0d",
            resident_backbone_loads,resident_wh1_loads,resident_wout_loads,resident_wout_loads-1);
        profile_finish();
        $display("RESIDENT_NATIVE_SEQUENCE sequence=%0d steps=%0d learning=%0d cycles=%0d state_setup_cycles=%0d bone=%0d hidden=%0d logits=%0d store=%0d code=%0d residual=%0d commits=%0d backbone_loads=%0d wh1_loads=%0d wout_loads=%0d",
            sequence_index,native_steps[sequence_index],learning,cycles-initialized_cycle,initialized_cycle-setup_begin,
            bone_seen,hidden_seen,logit_seen,store_checks,code_writes,residual_writes,commits,
            resident_backbone_loads,resident_wh1_loads,resident_wout_loads);
        completed_steps+=native_steps[sequence_index];
        end
        $display("RESIDENT_NATIVE_PAIR_PASS sequences=8 learning=%0d steps=%0d",learning,completed_steps);
        $display("REPRESENTATIVE_TOTAL learning=%0d cycles=%0d reset_cycles=%0d parameter_loads=%0d store_loads=%0d backbone_reads=%0d backbone_responses=%0d",
            learning,cycles,profile_count[86],profile_count[0],profile_count[2],profile_count[6],profile_count[8]);
        if (source_words!=102900 || source_groups!=33761 || source_tails!=588)
            $fatal(1,"source coverage");
        $display("SOURCE_CORE_PASS streaming=%0d learning=%0d words=%0d groups=%0d tails=%0d count_stalls=%0d group_stalls=%0d",
            streaming,learning,source_words,source_groups,source_tails,source_count_stalls,source_group_stalls);
        if (m2_turn_credit != 0 || m2_turn_reads != m2_turn_responses)
            $fatal(1,"M2 refill credit not drained");
        $display("M2_REFILL_TURNOVER_PASS reads=%0d responses=%0d replacements=%0d max_credit=%0d final_credit=%0d",
            m2_turn_reads,m2_turn_responses,m2_turn_pairs,m2_turn_max_credit,m2_turn_credit);
        $display("FINAL_RAW_WRAPPER_PASS policy=1 sequences=8 steps=588");
        $display("GRADIENT_ELASTIC_CORE_PASS inputs=%0d outputs=%0d finalize_inputs=%0d held_inputs=%0d double_holds=%0d",
            elastic_inputs, elastic_outputs, elastic_finalize_inputs, elastic_held_inputs, elastic_double_holds);
        $finish;
    end
    // All counters sample actual pre-edge valid/ready or busy signals.
    longint profile_phase_start = 0;
    string profile_phase = "reset";
    integer profile_phase_timestep = -1;
    longint profile_count[0:103];
    bit profile_forward_active = 0, profile_learning_active = 0;
    bit profile_prepare_active = 0;
    longint profile_forward_start, profile_learning_start;
    longint profile_prepare_start;
    initial foreach (profile_count[k]) profile_count[k] = 0;

    task automatic profile_enter(input string next_phase);
        $display("SRNN_PROFILE_INTERVAL kind=stage phase=%s timestep=%0d start=%0d end=%0d cycles=%0d",
            profile_phase, profile_phase_timestep, profile_phase_start, cycles, cycles-profile_phase_start);
        profile_phase = next_phase;
        profile_phase_timestep = (next_phase == "config" || next_phase == "parameter_load" ||
                                 next_phase == "initial_state") ? -1 : timestep;
        profile_phase_start = cycles;
    endtask

    always @(posedge i_clk) begin
        if (!i_rst_n) profile_count[86]++;
        else begin
            if ((i_learning_srnn_parameter_load_valid) && (o_learning_srnn_parameter_load_ready)) profile_count[0]++;
            if ((i_learning_srnn_parameter_load_valid) && !(o_learning_srnn_parameter_load_ready)) profile_count[1]++;
            if ((o_store_load_valid) && (i_store_load_ready)) profile_count[2]++;
            if ((o_store_load_valid) && !(i_store_load_ready)) profile_count[3]++;
            if ((i_srnn_backbone_refill_valid) && (o_srnn_backbone_refill_ready)) profile_count[4]++;
            if ((i_srnn_backbone_refill_valid) && !(o_srnn_backbone_refill_ready)) profile_count[5]++;
            if ((o_srnn_backbone_forward_read_valid) && (i_srnn_backbone_forward_read_ready)) profile_count[6]++;
            if ((o_srnn_backbone_forward_read_valid) && !(i_srnn_backbone_forward_read_ready)) profile_count[7]++;
            if ((i_srnn_backbone_forward_response_valid) && (o_srnn_backbone_forward_response_ready)) profile_count[8]++;
            if ((i_srnn_backbone_forward_response_valid) && !(o_srnn_backbone_forward_response_ready)) profile_count[9]++;
            if ((i_fc_srnn_layer_valid) && (o_fc_srnn_layer_ready)) profile_count[10]++;
            if ((i_fc_srnn_layer_valid) && !(o_fc_srnn_layer_ready)) profile_count[11]++;
            if ((i_fc_srnn_group_valid) && (o_fc_srnn_group_ready)) profile_count[12]++;
            if ((i_fc_srnn_group_valid) && !(o_fc_srnn_group_ready)) profile_count[13]++;
            if ((o_fc_srnn_observation_valid) && (i_fc_srnn_observation_ready)) profile_count[14]++;
            if ((o_fc_srnn_observation_valid) && !(i_fc_srnn_observation_ready)) profile_count[15]++;
            if ((i_srnn_phase_capture_cfg_valid) && (o_srnn_phase_capture_cfg_ready)) profile_count[16]++;
            if ((i_srnn_phase_capture_cfg_valid) && !(o_srnn_phase_capture_cfg_ready)) profile_count[17]++;
            if ((i_srnn_phase_start_valid) && (o_srnn_phase_start_ready)) profile_count[18]++;
            if ((i_srnn_phase_start_valid) && !(o_srnn_phase_start_ready)) profile_count[19]++;
            if ((u_final.u_core.link_srnn_actual_valid) && (u_final.u_core.link_srnn_actual_ready)) profile_count[20]++;
            if ((u_final.u_core.link_srnn_actual_valid) && !(u_final.u_core.link_srnn_actual_ready)) profile_count[21]++;
            if ((u_final.u_core.link_srnn_plane_backbone_valid) && (u_final.u_core.link_srnn_plane_backbone_ready)) profile_count[22]++;
            if ((u_final.u_core.link_srnn_plane_backbone_valid) && !(u_final.u_core.link_srnn_plane_backbone_ready)) profile_count[23]++;
            if ((u_final.u_core.link_plane_to_system_srnn_h40_prepare_valid) && (u_final.u_core.link_system_to_plane_srnn_h40_prepare_ready)) profile_count[24]++;
            if ((u_final.u_core.link_plane_to_system_srnn_h40_prepare_valid) && !(u_final.u_core.link_system_to_plane_srnn_h40_prepare_ready)) profile_count[25]++;
            if ((u_final.u_core.link_system_to_plane_srnn_h40_forward_read_valid) && (u_final.u_core.link_plane_to_system_srnn_h40_forward_read_ready)) profile_count[26]++;
            if ((u_final.u_core.link_system_to_plane_srnn_h40_forward_read_valid) && !(u_final.u_core.link_plane_to_system_srnn_h40_forward_read_ready)) profile_count[27]++;
            if ((u_final.u_core.link_plane_to_system_srnn_h40_forward_response_valid) && (u_final.u_core.link_system_to_plane_srnn_h40_forward_response_ready)) profile_count[28]++;
            if ((u_final.u_core.link_plane_to_system_srnn_h40_forward_response_valid) && !(u_final.u_core.link_system_to_plane_srnn_h40_forward_response_ready)) profile_count[29]++;
            if ((u_final.u_core.link_plane_to_system_srnn_h40_timestep_start) && (u_final.u_core.link_system_to_plane_srnn_h40_timestep_start_ready)) profile_count[30]++;
            if ((u_final.u_core.link_plane_to_system_srnn_h40_timestep_start) && !(u_final.u_core.link_system_to_plane_srnn_h40_timestep_start_ready)) profile_count[31]++;
            if ((u_final.u_core.link_system_to_plane_srnn_h40_hidden_valid) && (u_final.u_core.link_plane_to_system_srnn_h40_hidden_ready)) profile_count[32]++;
            if ((u_final.u_core.link_system_to_plane_srnn_h40_hidden_valid) && !(u_final.u_core.link_plane_to_system_srnn_h40_hidden_ready)) profile_count[33]++;
            if ((o_learning_srnn_logit_valid) && (i_learning_srnn_logit_ready)) profile_count[34]++;
            if ((o_learning_srnn_logit_valid) && !(i_learning_srnn_logit_ready)) profile_count[35]++;
            if ((i_learning_srnn_label_valid) && (o_learning_srnn_label_ready)) profile_count[36]++;
            if ((i_learning_srnn_label_valid) && !(o_learning_srnn_label_ready)) profile_count[37]++;
            if ((o_store_forward_read_valid) && (i_store_forward_read_ready)) profile_count[38]++;
            if ((o_store_forward_read_valid) && !(i_store_forward_read_ready)) profile_count[39]++;
            if ((i_store_forward_response_valid) && (o_store_forward_response_ready)) profile_count[40]++;
            if ((i_store_forward_response_valid) && !(o_store_forward_response_ready)) profile_count[41]++;
            if ((o_store_learning_read_valid) && (i_store_learning_read_ready)) profile_count[42]++;
            if ((o_store_learning_read_valid) && !(i_store_learning_read_ready)) profile_count[43]++;
            if ((i_store_learning_response_valid) && (o_store_learning_response_ready)) profile_count[44]++;
            if ((i_store_learning_response_valid) && !(o_store_learning_response_ready)) profile_count[45]++;
            if ((o_store_learning_write_valid) && (i_store_learning_write_ready)) profile_count[46]++;
            if ((o_store_learning_write_valid) && !(i_store_learning_write_ready)) profile_count[47]++;
            if ((o_store_learning_write_valid && o_store_learning_write_code_enable) && (i_store_learning_write_ready)) profile_count[48]++;
            if ((o_store_learning_write_valid && o_store_learning_write_code_enable) && !(i_store_learning_write_ready)) profile_count[49]++;
            if ((o_store_learning_write_valid && o_store_learning_write_residual_enable) && (i_store_learning_write_ready)) profile_count[50]++;
            if ((o_store_learning_write_valid && o_store_learning_write_residual_enable) && !(i_store_learning_write_ready)) profile_count[51]++;
            if ((`RED.reducer_gradient_valid) && (`RED.reducer_gradient_ready)) profile_count[52]++;
            if ((`RED.reducer_gradient_valid) && !(`RED.reducer_gradient_ready)) profile_count[53]++;
            if ((`RED.i_start_window) && (`RED.o_start_ready)) profile_count[54]++;
            if ((`RED.i_start_window) && !(`RED.o_start_ready)) profile_count[55]++;
            if ((`RED.o_commit_valid) && (`RED.i_commit_ready)) profile_count[56]++;
            if ((`RED.o_commit_valid) && !(`RED.i_commit_ready)) profile_count[57]++;
            if ((i_state_maintenance_write_valid) && (o_state_maintenance_write_ready)) profile_count[58]++;
            if ((i_state_maintenance_write_valid) && !(o_state_maintenance_write_ready)) profile_count[59]++;
            if ((i_trace_maintenance_write_valid) && (o_trace_maintenance_write_ready)) profile_count[60]++;
            if ((i_trace_maintenance_write_valid) && !(o_trace_maintenance_write_ready)) profile_count[61]++;
            if ((i_state_maintenance_read_valid) && (o_state_maintenance_read_ready)) profile_count[62]++;
            if ((i_state_maintenance_read_valid) && !(o_state_maintenance_read_ready)) profile_count[63]++;
            if ((i_trace_maintenance_read_valid) && (o_trace_maintenance_read_ready)) profile_count[64]++;
            if ((i_trace_maintenance_read_valid) && !(o_trace_maintenance_read_ready)) profile_count[65]++;
            if ((o_state_maintenance_result_valid) && (i_state_maintenance_result_ready)) profile_count[66]++;
            if ((o_state_maintenance_result_valid) && !(i_state_maintenance_result_ready)) profile_count[67]++;
            if ((o_trace_maintenance_result_valid) && (i_trace_maintenance_result_ready)) profile_count[68]++;
            if ((o_trace_maintenance_result_valid) && !(i_trace_maintenance_result_ready)) profile_count[69]++;
            if ((`ENG.o_maintenance_start_valid) && (`ENG.i_maintenance_start_ready)) profile_count[70]++;
            if ((`ENG.o_maintenance_start_valid) && !(`ENG.i_maintenance_start_ready)) profile_count[71]++;
            if ((`ENG.o_maintenance_release_valid) && (`ENG.i_maintenance_release_ready)) profile_count[72]++;
            if ((`ENG.o_maintenance_release_valid) && !(`ENG.i_maintenance_release_ready)) profile_count[73]++;
            if ((`ENG.o_state_write_valid) && (`ENG.i_state_write_ready)) profile_count[74]++;
            if ((`ENG.o_state_write_valid) && !(`ENG.i_state_write_ready)) profile_count[75]++;
            if ((`ENG.o_trace_write_valid) && (`ENG.i_trace_write_ready)) profile_count[76]++;
            if ((`ENG.o_trace_write_valid) && !(`ENG.i_trace_write_ready)) profile_count[77]++;
            if ((`ENG.o_weight_fill_valid[0]) && (`ENG.i_weight_fill_ready[0])) profile_count[78]++;
            if ((`ENG.o_weight_fill_valid[0]) && !(`ENG.i_weight_fill_ready[0])) profile_count[79]++;
            if ((`ENG.o_weight_fill_valid[1]) && (`ENG.i_weight_fill_ready[1])) profile_count[80]++;
            if ((`ENG.o_weight_fill_valid[1]) && !(`ENG.i_weight_fill_ready[1])) profile_count[81]++;
            if ((`ENG.o_weight_fill_valid[2]) && (`ENG.i_weight_fill_ready[2])) profile_count[82]++;
            if ((`ENG.o_weight_fill_valid[2]) && !(`ENG.i_weight_fill_ready[2])) profile_count[83]++;
            if ((`ENG.o_weight_fill_valid[3]) && (`ENG.i_weight_fill_ready[3])) profile_count[84]++;
            if ((`ENG.o_weight_fill_valid[3]) && !(`ENG.i_weight_fill_ready[3])) profile_count[85]++;
            if (o_srnn_backbone_refill_done) profile_count[87]++;
            if (o_fc_srnn_layer_done) profile_count[88]++;
            if (o_fc_srnn_learning_capture_done) profile_count[89]++;
            if (o_fc_srnn_learning_done) profile_count[90]++;
            if (u_final.u_core.link_system_to_plane_srnn_h40_timestep_done) profile_count[91]++;
            if (u_final.u_core.link_system_to_plane_srnn_h40_prepare_done) profile_count[92]++;
            if (o_learning_srnn_timestep_done) profile_count[93]++;
            if (o_learning_srnn_learning_commit) profile_count[94]++;
            if (o_learning_srnn_detach) profile_count[95]++;
            if (`RED.o_busy) profile_count[96]++;
            if (`RED.o_busy && !`ENG.o_busy && profile_phase == "m2") profile_count[97]++;
            if (`ENG.refill_busy) profile_count[98]++;
            if (`ENG.clear_busy) profile_count[99]++;
            if (`ENG.adapter_busy) profile_count[100]++;
            if (`ENG.o_busy) profile_count[101]++;
            if (o_srnn_phase_busy) profile_count[102]++;
            if (o_fc_srnn_busy) profile_count[103]++;
            if (u_final.u_core.link_plane_to_system_srnn_h40_prepare_valid &&
                u_final.u_core.link_system_to_plane_srnn_h40_prepare_ready) begin
                if (profile_prepare_active) $fatal(1,"overlapping M2 prepare interval");
                profile_prepare_active = 1;
                profile_prepare_start = cycles;
            end
            if (u_final.u_core.link_system_to_plane_srnn_h40_prepare_done) begin
                if (!profile_prepare_active) $fatal(1,"M2 prepare done without accepted start");
                $display("SRNN_PROFILE_INTERVAL kind=hardware phase=m2_prepare timestep=%0d start=%0d end=%0d cycles=%0d",
                    timestep,profile_prepare_start,cycles+1,cycles+1-profile_prepare_start);
                profile_prepare_active = 0;
            end
            if (u_final.u_core.link_plane_to_system_srnn_h40_timestep_start &&
                u_final.u_core.link_system_to_plane_srnn_h40_timestep_start_ready) begin
                if (profile_forward_active) $fatal(1,"overlapping M2 forward interval");
                profile_forward_active = 1;
                profile_forward_start = cycles;
            end
            if (u_final.u_core.link_system_to_plane_srnn_h40_timestep_done) begin
                if (!profile_forward_active) $fatal(1,"M2 forward done without accepted start");
                $display("SRNN_PROFILE_INTERVAL kind=hardware phase=m2_forward timestep=%0d start=%0d end=%0d cycles=%0d",
                    timestep,profile_forward_start,cycles+1,cycles+1-profile_forward_start);
                profile_forward_active = 0;
            end
            if ((`RED.o_busy && !`ENG.o_busy && profile_phase == "m2") && !profile_learning_active) begin
                profile_learning_active = 1;
                profile_learning_start = cycles;
            end
            if (!(`RED.o_busy && !`ENG.o_busy && profile_phase == "m2") && profile_learning_active) begin
                $display("SRNN_PROFILE_INTERVAL kind=hardware phase=pure_learning_busy timestep=%0d start=%0d end=%0d cycles=%0d",
                    timestep,profile_learning_start,cycles,cycles-profile_learning_start);
                profile_learning_active = 0;
            end
            if (o_learning_srnn_learning_commit)
                $display("SRNN_PROFILE_EVENT name=commit timestep=%0d cycle=%0d",timestep,cycles);
        end
    end

    task automatic profile_finish;
        profile_enter("finished");
        if (profile_forward_active || profile_learning_active || profile_prepare_active)
            $fatal(1,"profile interval still active at numerical completion");
        $display("SRNN_PROFILE_COUNT name=parameter_load_accepted value=%0d",profile_count[0]);
        $display("SRNN_PROFILE_COUNT name=parameter_load_stalled_cycles value=%0d",profile_count[1]);
        $display("SRNN_PROFILE_COUNT name=store_load_accepted value=%0d",profile_count[2]);
        $display("SRNN_PROFILE_COUNT name=store_load_stalled_cycles value=%0d",profile_count[3]);
        $display("SRNN_PROFILE_COUNT name=backbone_refill_request_accepted value=%0d",profile_count[4]);
        $display("SRNN_PROFILE_COUNT name=backbone_refill_request_stalled_cycles value=%0d",profile_count[5]);
        $display("SRNN_PROFILE_COUNT name=backbone_parameter_read_accepted value=%0d",profile_count[6]);
        $display("SRNN_PROFILE_COUNT name=backbone_parameter_read_stalled_cycles value=%0d",profile_count[7]);
        $display("SRNN_PROFILE_COUNT name=backbone_parameter_response_accepted value=%0d",profile_count[8]);
        $display("SRNN_PROFILE_COUNT name=backbone_parameter_response_stalled_cycles value=%0d",profile_count[9]);
        $display("SRNN_PROFILE_COUNT name=backbone_layer_accepted value=%0d",profile_count[10]);
        $display("SRNN_PROFILE_COUNT name=backbone_layer_stalled_cycles value=%0d",profile_count[11]);
        $display("SRNN_PROFILE_COUNT name=backbone_group_accepted value=%0d",profile_count[12]);
        $display("SRNN_PROFILE_COUNT name=backbone_group_stalled_cycles value=%0d",profile_count[13]);
        $display("SRNN_PROFILE_COUNT name=backbone_observation_accepted value=%0d",profile_count[14]);
        $display("SRNN_PROFILE_COUNT name=backbone_observation_stalled_cycles value=%0d",profile_count[15]);
        $display("SRNN_PROFILE_COUNT name=phase_capture_config_accepted value=%0d",profile_count[16]);
        $display("SRNN_PROFILE_COUNT name=phase_capture_config_stalled_cycles value=%0d",profile_count[17]);
        $display("SRNN_PROFILE_COUNT name=phase_start_accepted value=%0d",profile_count[18]);
        $display("SRNN_PROFILE_COUNT name=phase_start_stalled_cycles value=%0d",profile_count[19]);
        $display("SRNN_PROFILE_COUNT name=phase_actual_capture_accepted value=%0d",profile_count[20]);
        $display("SRNN_PROFILE_COUNT name=phase_actual_capture_stalled_cycles value=%0d",profile_count[21]);
        $display("SRNN_PROFILE_COUNT name=phase_backbone_replay_accepted value=%0d",profile_count[22]);
        $display("SRNN_PROFILE_COUNT name=phase_backbone_replay_stalled_cycles value=%0d",profile_count[23]);
        $display("SRNN_PROFILE_COUNT name=m2_prepare_accepted value=%0d",profile_count[24]);
        $display("SRNN_PROFILE_COUNT name=m2_prepare_stalled_cycles value=%0d",profile_count[25]);
        $display("SRNN_PROFILE_COUNT name=m2_refill_read_accepted value=%0d",profile_count[26]);
        $display("SRNN_PROFILE_COUNT name=m2_refill_read_stalled_cycles value=%0d",profile_count[27]);
        $display("SRNN_PROFILE_COUNT name=m2_refill_response_accepted value=%0d",profile_count[28]);
        $display("SRNN_PROFILE_COUNT name=m2_refill_response_stalled_cycles value=%0d",profile_count[29]);
        $display("SRNN_PROFILE_COUNT name=m2_timestep_start_accepted value=%0d",profile_count[30]);
        $display("SRNN_PROFILE_COUNT name=m2_timestep_start_stalled_cycles value=%0d",profile_count[31]);
        $display("SRNN_PROFILE_COUNT name=m2_hidden_accepted value=%0d",profile_count[32]);
        $display("SRNN_PROFILE_COUNT name=m2_hidden_stalled_cycles value=%0d",profile_count[33]);
        $display("SRNN_PROFILE_COUNT name=m2_logit_accepted value=%0d",profile_count[34]);
        $display("SRNN_PROFILE_COUNT name=m2_logit_stalled_cycles value=%0d",profile_count[35]);
        $display("SRNN_PROFILE_COUNT name=label_accepted value=%0d",profile_count[36]);
        $display("SRNN_PROFILE_COUNT name=label_stalled_cycles value=%0d",profile_count[37]);
        $display("SRNN_PROFILE_COUNT name=store_forward_read_accepted value=%0d",profile_count[38]);
        $display("SRNN_PROFILE_COUNT name=store_forward_read_stalled_cycles value=%0d",profile_count[39]);
        $display("SRNN_PROFILE_COUNT name=store_forward_response_accepted value=%0d",profile_count[40]);
        $display("SRNN_PROFILE_COUNT name=store_forward_response_stalled_cycles value=%0d",profile_count[41]);
        $display("SRNN_PROFILE_COUNT name=store_learning_read_accepted value=%0d",profile_count[42]);
        $display("SRNN_PROFILE_COUNT name=store_learning_read_stalled_cycles value=%0d",profile_count[43]);
        $display("SRNN_PROFILE_COUNT name=store_learning_response_accepted value=%0d",profile_count[44]);
        $display("SRNN_PROFILE_COUNT name=store_learning_response_stalled_cycles value=%0d",profile_count[45]);
        $display("SRNN_PROFILE_COUNT name=store_learning_write_accepted value=%0d",profile_count[46]);
        $display("SRNN_PROFILE_COUNT name=store_learning_write_stalled_cycles value=%0d",profile_count[47]);
        $display("SRNN_PROFILE_COUNT name=code_write_accepted value=%0d",profile_count[48]);
        $display("SRNN_PROFILE_COUNT name=code_write_stalled_cycles value=%0d",profile_count[49]);
        $display("SRNN_PROFILE_COUNT name=residual_write_accepted value=%0d",profile_count[50]);
        $display("SRNN_PROFILE_COUNT name=residual_write_stalled_cycles value=%0d",profile_count[51]);
        $display("SRNN_PROFILE_COUNT name=gradient_accepted value=%0d",profile_count[52]);
        $display("SRNN_PROFILE_COUNT name=gradient_stalled_cycles value=%0d",profile_count[53]);
        $display("SRNN_PROFILE_COUNT name=learner_start_accepted value=%0d",profile_count[54]);
        $display("SRNN_PROFILE_COUNT name=learner_start_stalled_cycles value=%0d",profile_count[55]);
        $display("SRNN_PROFILE_COUNT name=learner_commit_accepted value=%0d",profile_count[56]);
        $display("SRNN_PROFILE_COUNT name=learner_commit_stalled_cycles value=%0d",profile_count[57]);
        $display("SRNN_PROFILE_COUNT name=external_state_write_accepted value=%0d",profile_count[58]);
        $display("SRNN_PROFILE_COUNT name=external_state_write_stalled_cycles value=%0d",profile_count[59]);
        $display("SRNN_PROFILE_COUNT name=external_trace_write_accepted value=%0d",profile_count[60]);
        $display("SRNN_PROFILE_COUNT name=external_trace_write_stalled_cycles value=%0d",profile_count[61]);
        $display("SRNN_PROFILE_COUNT name=external_state_read_accepted value=%0d",profile_count[62]);
        $display("SRNN_PROFILE_COUNT name=external_state_read_stalled_cycles value=%0d",profile_count[63]);
        $display("SRNN_PROFILE_COUNT name=external_trace_read_accepted value=%0d",profile_count[64]);
        $display("SRNN_PROFILE_COUNT name=external_trace_read_stalled_cycles value=%0d",profile_count[65]);
        $display("SRNN_PROFILE_COUNT name=external_state_result_accepted value=%0d",profile_count[66]);
        $display("SRNN_PROFILE_COUNT name=external_state_result_stalled_cycles value=%0d",profile_count[67]);
        $display("SRNN_PROFILE_COUNT name=external_trace_result_accepted value=%0d",profile_count[68]);
        $display("SRNN_PROFILE_COUNT name=external_trace_result_stalled_cycles value=%0d",profile_count[69]);
        $display("SRNN_PROFILE_COUNT name=m2_maintenance_acquire_accepted value=%0d",profile_count[70]);
        $display("SRNN_PROFILE_COUNT name=m2_maintenance_acquire_stalled_cycles value=%0d",profile_count[71]);
        $display("SRNN_PROFILE_COUNT name=m2_maintenance_release_accepted value=%0d",profile_count[72]);
        $display("SRNN_PROFILE_COUNT name=m2_maintenance_release_stalled_cycles value=%0d",profile_count[73]);
        $display("SRNN_PROFILE_COUNT name=m2_state_write_accepted value=%0d",profile_count[74]);
        $display("SRNN_PROFILE_COUNT name=m2_state_write_stalled_cycles value=%0d",profile_count[75]);
        $display("SRNN_PROFILE_COUNT name=m2_trace_write_accepted value=%0d",profile_count[76]);
        $display("SRNN_PROFILE_COUNT name=m2_trace_write_stalled_cycles value=%0d",profile_count[77]);
        $display("SRNN_PROFILE_COUNT name=m2_cache_bank0_accepted value=%0d",profile_count[78]);
        $display("SRNN_PROFILE_COUNT name=m2_cache_bank0_stalled_cycles value=%0d",profile_count[79]);
        $display("SRNN_PROFILE_COUNT name=m2_cache_bank1_accepted value=%0d",profile_count[80]);
        $display("SRNN_PROFILE_COUNT name=m2_cache_bank1_stalled_cycles value=%0d",profile_count[81]);
        $display("SRNN_PROFILE_COUNT name=m2_cache_bank2_accepted value=%0d",profile_count[82]);
        $display("SRNN_PROFILE_COUNT name=m2_cache_bank2_stalled_cycles value=%0d",profile_count[83]);
        $display("SRNN_PROFILE_COUNT name=m2_cache_bank3_accepted value=%0d",profile_count[84]);
        $display("SRNN_PROFILE_COUNT name=m2_cache_bank3_stalled_cycles value=%0d",profile_count[85]);
        $display("SRNN_PROFILE_COUNT name=reset_cycles value=%0d",profile_count[86]);
        $display("SRNN_PROFILE_COUNT name=backbone_refill_done value=%0d",profile_count[87]);
        $display("SRNN_PROFILE_COUNT name=backbone_layer_done value=%0d",profile_count[88]);
        $display("SRNN_PROFILE_COUNT name=capture_done value=%0d",profile_count[89]);
        $display("SRNN_PROFILE_COUNT name=backbone_replay_done value=%0d",profile_count[90]);
        $display("SRNN_PROFILE_COUNT name=m2_forward_done value=%0d",profile_count[91]);
        $display("SRNN_PROFILE_COUNT name=m2_prepare_done value=%0d",profile_count[92]);
        $display("SRNN_PROFILE_COUNT name=timestep_done value=%0d",profile_count[93]);
        $display("SRNN_PROFILE_COUNT name=commit_pulse value=%0d",profile_count[94]);
        $display("SRNN_PROFILE_COUNT name=detach_pulse value=%0d",profile_count[95]);
        $display("SRNN_PROFILE_COUNT name=learner_busy_cycles value=%0d",profile_count[96]);
        $display("SRNN_PROFILE_COUNT name=pure_learning_busy_cycles value=%0d",profile_count[97]);
        $display("SRNN_PROFILE_COUNT name=m2_refill_busy_cycles value=%0d",profile_count[98]);
        $display("SRNN_PROFILE_COUNT name=m2_state_clear_busy_cycles value=%0d",profile_count[99]);
        $display("SRNN_PROFILE_COUNT name=m2_adapter_busy_cycles value=%0d",profile_count[100]);
        $display("SRNN_PROFILE_COUNT name=m2_engine_busy_cycles value=%0d",profile_count[101]);
        $display("SRNN_PROFILE_COUNT name=phase_busy_cycles value=%0d",profile_count[102]);
        $display("SRNN_PROFILE_COUNT name=backbone_busy_cycles value=%0d",profile_count[103]);
        $display("SRNN_PROFILE_COMPLETE batch=1 steps=%0d learning=%0d initialized=%0d end=%0d store_checks=%0d",
            native_steps[sequence_index],learning,initialized_cycle,cycles,store_checks);
        $fflush();
    endtask


    integer resident_wh1_loads = 0, resident_wout_loads = 0, resident_backbone_loads = 0;
    always @(posedge i_clk) if (i_rst_n) begin
        if (i_srnn_backbone_refill_valid && o_srnn_backbone_refill_ready) resident_backbone_loads++;
        if (`ENG.refill_fire) begin
            if (`ENG.prep_state_q == 1) resident_wh1_loads++;
            if (`ENG.prep_state_q == 3) resident_wout_loads++;
        end
    end

    `undef RED
    `undef LINK
    `undef ENG
    `undef M2

    longint m2_turn_reads = 0, m2_turn_responses = 0, m2_turn_pairs = 0;
    integer m2_turn_credit = 0, m2_turn_max_credit = 0;
    wire m2_turn_read = u_final.u_core.link_system_to_plane_srnn_h40_forward_read_valid &&
        u_final.u_core.link_plane_to_system_srnn_h40_forward_read_ready;
    wire m2_turn_response = u_final.u_core.link_plane_to_system_srnn_h40_forward_response_valid &&
        u_final.u_core.link_system_to_plane_srnn_h40_forward_response_ready;
    always @(posedge i_clk) begin
        if (!i_rst_n) begin
            m2_turn_reads = 0;
            m2_turn_responses = 0;
            m2_turn_pairs = 0;
            m2_turn_credit = 0;
            m2_turn_max_credit = 0;
        end else begin
            if (m2_turn_response && m2_turn_credit != 1)
                $fatal(1,"M2 refill response without prior credit");
            if (m2_turn_read && m2_turn_credit != 0 && !m2_turn_response)
                $fatal(1,"M2 refill second outstanding read");
            if (m2_turn_response) begin
                m2_turn_responses++;
                m2_turn_credit--;
            end
            if (m2_turn_read) begin
                m2_turn_reads++;
                m2_turn_credit++;
            end
            if (m2_turn_read && m2_turn_response) m2_turn_pairs++;
            if (m2_turn_credit > m2_turn_max_credit) m2_turn_max_credit = m2_turn_credit;
            if (m2_turn_credit < 0 || m2_turn_credit > 1)
                $fatal(1,"M2 refill credit range");
        end
    end


    longint elastic_inputs = 0, elastic_outputs = 0;
    longint elastic_finalize_inputs = 0, elastic_held_inputs = 0, elastic_double_holds = 0;
    always @(posedge i_clk) if (i_rst_n) begin
        if (u_final.u_core.u_parameter_plane.g_srnn_client.u_srnn_m2.parameter_store.g_wout_only_learning.learning_engine.u_wout_reducer.input_fire) elastic_inputs++;
        if (u_final.u_core.u_parameter_plane.g_srnn_client.u_srnn_m2.parameter_store.g_wout_only_learning.learning_engine.u_wout_reducer.output_fire) elastic_outputs++;
        if (u_final.u_core.u_parameter_plane.g_srnn_client.u_srnn_m2.parameter_store.g_wout_only_learning.learning_engine.u_wout_reducer.input_fire && u_final.u_core.u_parameter_plane.g_srnn_client.u_srnn_m2.parameter_store.g_wout_only_learning.learning_engine.u_wout_reducer.finalize_q) elastic_finalize_inputs++;
        if (u_final.u_core.u_parameter_plane.g_srnn_client.u_srnn_m2.parameter_store.g_wout_only_learning.learning_engine.u_wout_reducer.input_fire && u_final.u_core.u_parameter_plane.g_srnn_client.u_srnn_m2.parameter_store.g_wout_only_learning.learning_engine.u_wout_reducer.pending_q && !u_final.u_core.u_parameter_plane.g_srnn_client.u_srnn_m2.parameter_store.g_wout_only_learning.learning_engine.u_wout_reducer.i_gradient_ready)
            elastic_held_inputs++;
        if (u_final.u_core.u_parameter_plane.g_srnn_client.u_srnn_m2.parameter_store.g_wout_only_learning.learning_engine.u_wout_reducer.finalize_q && u_final.u_core.u_parameter_plane.g_srnn_client.u_srnn_m2.parameter_store.g_wout_only_learning.learning_engine.u_wout_reducer.pending_q && !u_final.u_core.u_parameter_plane.g_srnn_client.u_srnn_m2.parameter_store.g_wout_only_learning.learning_engine.u_wout_reducer.i_gradient_ready)
            elastic_double_holds++;
    end

endmodule
`default_nettype wire
