`timescale 1ns/1ps
`default_nettype none
module tb_fc_profile_measurement;
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
    logic i_fc_srnn_group_valid = '0;
    logic o_fc_srnn_group_ready;
    logic i_fc_srnn_group_recurrent = '0;
    logic [3:0] i_fc_srnn_group_source_valid = '0;
    logic [WEIGHT_ADDR_WIDTH-1:0] i_fc_srnn_group_weight_addr0 = '0;
    logic [WEIGHT_ADDR_WIDTH-1:0] i_fc_srnn_group_weight_addr1 = '0;
    logic [WEIGHT_ADDR_WIDTH-1:0] i_fc_srnn_group_weight_addr2 = '0;
    logic [WEIGHT_ADDR_WIDTH-1:0] i_fc_srnn_group_weight_addr3 = '0;
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
        .i_raw_source_active(1'b0),
        .i_raw_count_valid(1'b0),
        .i_raw_counts(32'b0),
        .i_fc_srnn_group_feedforward_end(1'b0),
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
        .i_fc_srnn_group_valid(i_fc_srnn_group_valid),
        .o_fc_srnn_group_ready(o_fc_srnn_group_ready),
        .i_fc_srnn_group_recurrent(i_fc_srnn_group_recurrent),
        .i_fc_srnn_group_source_valid(i_fc_srnn_group_source_valid),
        .i_fc_srnn_group_weight_addr0(i_fc_srnn_group_weight_addr0),
        .i_fc_srnn_group_weight_addr1(i_fc_srnn_group_weight_addr1),
        .i_fc_srnn_group_weight_addr2(i_fc_srnn_group_weight_addr2),
        .i_fc_srnn_group_weight_addr3(i_fc_srnn_group_weight_addr3),
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
    // Read-only pre-NBA edge samples. The original body remains the sole driver.
    `define FC_MEASURE_PATH u_final.u_core.u_parameter_plane.gen_shared_fc_scnn_output_learner.u_shared_output_learner.g_output_learning_core.u_shared_output_learning_path
    integer fc_measure_stage = 0;
    bit fc_measure_complete = 0;
    longint fc_measure_clock = 0;
    longint fc_measure_cycles[8] = '{default:0};
    longint fc_measure_window_busy[8] = '{default:0};
    longint fc_measure_work_busy[8] = '{default:0};
    longint fc_measure_work_overlap[8] = '{default:0};
    longint fc_measure_work_only[8] = '{default:0};
    longint fc_measure_work_other[8] = '{default:0};
    longint fc_measure_forward[8] = '{default:0};
    longint fc_measure_refill[8] = '{default:0};
    longint fc_measure_bus[8][20] = '{default:'{default:0}};
    longint fc_measure_state_bytes[8] = '{default:0};
    longint fc_measure_trace_bytes[8] = '{default:0};
    longint fc_measure_commit_present[8] = '{default:0};
    longint fc_measure_commit_stall[8] = '{default:0};
    longint fc_measure_observation = -1, fc_measure_done = -1;
    longint fc_measure_commit_first = -1, fc_measure_work_first = -1;
    longint fc_measure_last_write = -1, fc_measure_update_start = -1;
    longint fc_measure_last_forward = -1;
    longint fc_measure_window_work = 0, fc_measure_window_overlap = 0;
    longint fc_measure_window_other = 0;
    integer fc_measure_commits = 0;
    wire fc_measure_work = learning && (
        `FC_MEASURE_PATH.frontend_busy ||
        `FC_MEASURE_PATH.u_learning_slice.gradient_busy ||
        `FC_MEASURE_PATH.u_learning_slice.update_busy);
    wire fc_measure_in_forward = phase == 1 || phase == 2;
    wire fc_measure_runtime = fc_measure_stage >= 4 && fc_measure_stage <= 6;
    wire fc_measure_other = !fc_measure_runtime || fc_measure_stage == 6 || o_cache_refill_busy;

    always @(posedge i_clk) begin : fc_measure_sample
        integer st;
        if (i_rst_n && !fc_measure_complete) begin
            st = fc_measure_stage;
            fc_measure_clock++;
            fc_measure_cycles[st]++;
            if (o_learning_fc_window_busy) fc_measure_window_busy[st]++;
            if (fc_measure_in_forward) fc_measure_forward[st]++;
            if (o_cache_refill_busy) fc_measure_refill[st]++;
            if (fc_measure_work) begin
                fc_measure_work_busy[st]++;
                fc_measure_window_work++;
                if (fc_measure_work_first < 0) fc_measure_work_first = fc_measure_clock;
                if (fc_measure_in_forward) begin
                    fc_measure_work_overlap[st]++;
                    fc_measure_window_overlap++;
                end else if (fc_measure_other) begin
                    fc_measure_work_other[st]++;
                    fc_measure_window_other++;
                end else fc_measure_work_only[st]++;
            end
            if (`FC_MEASURE_PATH.observation_fire) fc_measure_observation = fc_measure_clock;
            if (`FC_MEASURE_PATH.learning_window_emit) fc_measure_update_start = fc_measure_clock;
            if (`FC_MEASURE_PATH.learning_window_done) fc_measure_done = fc_measure_clock;
            if (phase == 2 && sample_index == B-1 && o_fc_srnn_learning_done)
                fc_measure_last_forward = fc_measure_clock;
            if (o_learning_fc_window_commit_valid) begin
                fc_measure_commit_present[st]++;
                if (fc_measure_commit_first < 0) fc_measure_commit_first = fc_measure_clock;
                if (!i_learning_fc_window_commit_ready) fc_measure_commit_stall[st]++;
            end
            if (o_learning_fc_window_commit_valid && i_learning_fc_window_commit_ready) begin
                $display("FC_PROFILE_WINDOW index=%0d observation=%0d update_start=%0d done=%0d commit_first=%0d commit=%0d work_first=%0d last_forward=%0d last_write=%0d work_busy=%0d work_overlap_forward=%0d work_other=%0d",
                    fc_measure_commits, fc_measure_observation, fc_measure_update_start,
                    fc_measure_done, fc_measure_commit_first, fc_measure_clock,
                    fc_measure_work_first, fc_measure_last_forward, fc_measure_last_write,
                    fc_measure_window_work, fc_measure_window_overlap, fc_measure_window_other);
                fc_measure_commits++;
                fc_measure_observation = -1;
                fc_measure_update_start = -1;
                fc_measure_done = -1;
                fc_measure_commit_first = -1;
                fc_measure_work_first = -1;
                fc_measure_last_write = -1;
                fc_measure_last_forward = -1;
                fc_measure_window_work = 0;
                fc_measure_window_overlap = 0;
                fc_measure_window_other = 0;
            end
            if (i_learning_fc_parameter_load_valid && o_learning_fc_parameter_load_ready) fc_measure_bus[st][0]++;
            if (o_store_load_valid && i_store_load_ready) fc_measure_bus[st][1]++;
            if (o_store_forward_read_valid && i_store_forward_read_ready) fc_measure_bus[st][2]++;
            if (i_store_forward_response_valid && o_store_forward_response_ready) fc_measure_bus[st][3]++;
            if (o_store_learning_read_valid && i_store_learning_read_ready) fc_measure_bus[st][4]++;
            if (i_store_learning_response_valid && o_store_learning_response_ready) fc_measure_bus[st][5]++;
            if (o_store_learning_write_valid && i_store_learning_write_ready) begin
                fc_measure_bus[st][6]++;
                if (o_store_learning_write_code_enable) fc_measure_bus[st][7]++;
                if (o_store_learning_write_residual_enable) fc_measure_bus[st][8]++;
                fc_measure_last_write = fc_measure_clock;
            end
            if (i_state_maintenance_write_valid && o_state_maintenance_write_ready) begin
                fc_measure_bus[st][9]++;
                fc_measure_state_bytes[st] += $countones(i_state_maintenance_write_mask);
            end
            if (i_trace_maintenance_write_valid && o_trace_maintenance_write_ready) begin
                fc_measure_bus[st][10]++;
                fc_measure_trace_bytes[st] += $countones(i_trace_maintenance_write_mask);
            end
            if (i_state_maintenance_read_valid && o_state_maintenance_read_ready) fc_measure_bus[st][11]++;
            if (o_state_maintenance_result_valid && i_state_maintenance_result_ready) fc_measure_bus[st][12]++;
            if (i_trace_maintenance_read_valid && o_trace_maintenance_read_ready) fc_measure_bus[st][13]++;
            if (o_trace_maintenance_result_valid && i_trace_maintenance_result_ready) fc_measure_bus[st][14]++;
            if (i_fc_srnn_group_valid && o_fc_srnn_group_ready) begin
                fc_measure_bus[st][15]++;
                fc_measure_bus[st][16] += $countones(i_fc_srnn_group_source_valid);
            end
            if (o_fc_srnn_observation_valid && i_fc_srnn_observation_ready) fc_measure_bus[st][17]++;
            if (learning && u_final.u_core.link_fc_student_trace_valid && u_final.u_core.link_fc_student_trace_ready) fc_measure_bus[st][18]++;
            if (u_final.u_core.link_fc_logit_valid && (!learning || u_final.u_core.link_fc_logit_ready)) fc_measure_bus[st][19]++;
        end
    end

    final begin
        if (fc_measure_complete) begin
            for (int st = 0; st < 8; st++) begin
                $display("FC_PROFILE_STAGE id=%0d cycles=%0d window_busy=%0d work_busy=%0d work_overlap_forward=%0d work_only=%0d work_other=%0d forward_scope=%0d refill_busy=%0d commit_present=%0d commit_stall=%0d parameter_input=%0d store_load=%0d forward_request=%0d forward_response=%0d learning_request=%0d learning_response=%0d learning_write=%0d code_write=%0d residual_write=%0d state_write=%0d trace_write=%0d state_read_request=%0d state_read_response=%0d trace_read_request=%0d trace_read_response=%0d source_group=%0d source_events=%0d observation=%0d student_trace=%0d logit=%0d state_write_mask_bytes=%0d trace_write_mask_bytes=%0d",
                    st, fc_measure_cycles[st], fc_measure_window_busy[st], fc_measure_work_busy[st],
                    fc_measure_work_overlap[st], fc_measure_work_only[st], fc_measure_work_other[st],
                    fc_measure_forward[st], fc_measure_refill[st], fc_measure_commit_present[st], fc_measure_commit_stall[st],
                    fc_measure_bus[st][0], fc_measure_bus[st][1], fc_measure_bus[st][2], fc_measure_bus[st][3],
                    fc_measure_bus[st][4], fc_measure_bus[st][5], fc_measure_bus[st][6], fc_measure_bus[st][7],
                    fc_measure_bus[st][8], fc_measure_bus[st][9], fc_measure_bus[st][10], fc_measure_bus[st][11],
                    fc_measure_bus[st][12], fc_measure_bus[st][13], fc_measure_bus[st][14], fc_measure_bus[st][15],
                    fc_measure_bus[st][16], fc_measure_bus[st][17], fc_measure_bus[st][18], fc_measure_bus[st][19],
                    fc_measure_state_bytes[st], fc_measure_trace_bytes[st]);
            end
        end
    end
    `undef FC_MEASURE_PATH

    localparam int B = 16;
    localparam int H = 160;
    localparam int C = 10;
    localparam int F = 2312;
    localparam int W2_BASE = F*H;
    logic [15:0] input_counts [0:100*B*F-1];
    logic signed [15:0] parameters [0:371519];
    logic [3:0] labels [0:B-1];
    logic signed [31:0] expected_pre [0:100*B*H-1];
    logic signed [31:0] expected_trace [0:100*B*H-1];
    logic expected_spike [0:100*B*H-1];
    logic signed [31:0] expected_dot [0:100*B*C-1];
    logic signed [15:0] expected_q6 [0:100*B*C-1];
    logic signed [15:0] expected_w2 [0:100*H*C-1];
    logic signed [15:0] expected_residual [0:100*H*C-1];
    logic [H-1:0] hidden_spikes;
    integer steps, learning, timestep, sample_index, phase = 0;
    integer batches = 1, batch_index = 0, window_index = 0;
    integer batch_steps [0:9];
    integer total_steps = 0;
    integer hidden_checks = 0, dot_checks = 0, q6_checks = 0;
    integer commits = 0, layer_done_count = 0, capture_done_count = 0;
    integer replay_done_count = 0;
    integer input_events = 0, hidden_events = 0, groups = 0;
    integer learning_reads = 0, code_writes = 0, residual_writes = 0, forward_reads = 0;
    integer scale_q16 = 47303;
    longint cycles = 0, start_cycle, load_cycles, init_cycles;
    longint w0_cycles = 0, w2_cycles = 0, learning_cycles = 0, refill_cycles = 0;
    string fixture, batch_fixture;

    always #5 i_clk = ~i_clk;
    always @(posedge i_clk) begin
        cycles <= cycles+1;
        if (cycles > 100000000) $fatal(1, "FC final watchdog phase=%0d t=%0d s=%0d", phase,timestep,sample_index);
        if (i_rst_n) begin
            if (o_sticky_error || o_backend_error || i_store_protocol_error || o_learning_fc_error ||
                o_cache_refill_sticky_error || o_fc_srnn_sticky_error || o_fc_srnn_learning_sticky_error ||
                o_fc_srnn_observation_sticky_error || o_array_tile_sticky_error != 0)
                $fatal(1, "FC final error phase=%0d t=%0d s=%0d global=%0b backend=%0b store=%0b fc=%0b cache=%0b layer=%0b/%h ingress=%0b/%h tile=%h",
                    phase,timestep,sample_index,o_sticky_error,o_backend_error,i_store_protocol_error,
                    o_learning_fc_error,o_cache_refill_sticky_error,o_fc_srnn_sticky_error,o_fc_srnn_error_code,
                    o_fc_srnn_learning_sticky_error,o_fc_srnn_learning_error_code,o_array_tile_sticky_error);
            if (o_fc_srnn_layer_done) layer_done_count++;
            if (o_fc_srnn_learning_capture_done) capture_done_count++;
            if (o_fc_srnn_learning_done) replay_done_count++;
            if (o_learning_fc_window_commit_valid && i_learning_fc_window_commit_ready) commits++;
            if (o_store_forward_read_valid && i_store_forward_read_ready) forward_reads++;
            if (o_store_learning_read_valid && i_store_learning_read_ready) learning_reads++;
            if (o_store_learning_write_valid && i_store_learning_write_ready) begin
                if (o_store_learning_write_addr < W2_BASE || o_store_learning_write_addr >= W2_BASE+H*C)
                    $fatal(1, "FC frozen/out-of-range parameter write %0d", o_store_learning_write_addr);
                if (!learning) $fatal(1, "FC learning-off parameter write");
                if (o_store_learning_write_code_enable) code_writes++;
                if (o_store_learning_write_residual_enable) residual_writes++;
                if (o_store_learning_write_code_enable && o_store_learning_write_code !=
                    expected_w2[timestep*H*C+int'(o_store_learning_write_addr)-W2_BASE])
                    $fatal(1, "FC updated code mismatch t=%0d addr=%0d got=%0d exp=%0d", timestep,
                        o_store_learning_write_addr,o_store_learning_write_code,
                        expected_w2[timestep*H*C+int'(o_store_learning_write_addr)-W2_BASE]);
                if (o_store_learning_write_residual_enable && o_store_learning_write_residual !=
                    expected_residual[timestep*H*C+int'(o_store_learning_write_addr)-W2_BASE])
                    $fatal(1, "FC residual mismatch t=%0d addr=%0d got=%0d exp=%0d", timestep,
                        o_store_learning_write_addr,o_store_learning_write_residual,
                        expected_residual[timestep*H*C+int'(o_store_learning_write_addr)-W2_BASE]);
            end
            if (o_fc_srnn_observation_valid && i_fc_srnn_observation_ready) begin
                if (phase == 1) begin
                    if (int'(o_fc_srnn_observation_neuron_id) >= H) $fatal(1, "hidden id range");
                    if (o_fc_srnn_observation_pre_state != expected_pre[(timestep*B+sample_index)*H+int'(o_fc_srnn_observation_neuron_id)] ||
                        o_fc_srnn_observation_trace_next != expected_trace[(timestep*B+sample_index)*H+int'(o_fc_srnn_observation_neuron_id)] ||
                        o_fc_srnn_observation_spike != expected_spike[(timestep*B+sample_index)*H+int'(o_fc_srnn_observation_neuron_id)])
                        $fatal(1, "hidden mismatch t=%0d s=%0d h=%0d pre=%0d/%0d trace=%0d/%0d spike=%0b/%0b", timestep,sample_index,
                            o_fc_srnn_observation_neuron_id,o_fc_srnn_observation_pre_state,
                            expected_pre[(timestep*B+sample_index)*H+int'(o_fc_srnn_observation_neuron_id)],
                            o_fc_srnn_observation_trace_next,expected_trace[(timestep*B+sample_index)*H+int'(o_fc_srnn_observation_neuron_id)],
                            o_fc_srnn_observation_spike,expected_spike[(timestep*B+sample_index)*H+int'(o_fc_srnn_observation_neuron_id)]);
                    hidden_spikes[o_fc_srnn_observation_neuron_id] = o_fc_srnn_observation_spike;
                    hidden_checks++;
                end else if (phase == 2) begin
                    if (o_fc_srnn_observation_neuron_id >= C)
                        $fatal(1, "output id range %0d", o_fc_srnn_observation_neuron_id);
                    if (o_fc_srnn_observation_pre_state != expected_dot[(timestep*B+sample_index)*C+int'(o_fc_srnn_observation_neuron_id)])
                        $fatal(1, "dot mismatch t=%0d s=%0d c=%0d got=%0d exp=%0d", timestep,sample_index,
                            o_fc_srnn_observation_neuron_id,o_fc_srnn_observation_pre_state,
                            expected_dot[(timestep*B+sample_index)*C+int'(o_fc_srnn_observation_neuron_id)]);
                    dot_checks++;
                end else $fatal(1, "observation outside forward");
            end
            // Read-only tap of the production router output, before the learner.
            if (u_final.u_core.link_fc_logit_valid && ((!learning) || u_final.u_core.link_fc_logit_ready)) begin
                if (u_final.u_core.link_fc_logit_q6 != expected_q6[(timestep*B+sample_index)*C+(q6_checks%C)])
                    $fatal(1, "Q6 mismatch t=%0d s=%0d c=%0d got=%0d exp=%0d", timestep,sample_index,q6_checks%C,
                        u_final.u_core.link_fc_logit_q6,expected_q6[(timestep*B+sample_index)*C+(q6_checks%C)]);
                q6_checks++;
            end
        end
    end

    task automatic load_parameters;
        for (int addr = 0; addr < 371520; addr++) begin
            @(negedge i_clk);
            i_learning_fc_parameter_load_valid = 1;
            i_learning_fc_parameter_load_addr = 24'(addr);
            i_learning_fc_parameter_load_code = parameters[addr];
            do @(posedge i_clk); while (!o_learning_fc_parameter_load_ready);
            @(negedge i_clk); i_learning_fc_parameter_load_valid = 0;
        end
    endtask

    task automatic refill(input bit output_layer);
        longint start;
        start = cycles;
        @(negedge i_clk);
        i_fc_cache_refill_weight_int8 = output_layer;
        i_fc_cache_refill_destination_tile_base = output_layer ? 4 : 0;
        i_fc_w0_parameter_base = output_layer ? W2_BASE : 0;
        i_fc_w0_hidden_stride = output_layer ? C : H;
        i_fc_w0_feature_count = output_layer ? H : F;
        i_fc_w0_hidden_count = output_layer ? C : H;
        i_fc_w0_tile_count = output_layer ? 1 : 4;
        i_fc_w0_refill_valid = 1;
        do @(posedge i_clk); while (!o_fc_w0_refill_ready);
        @(negedge i_clk); i_fc_w0_refill_valid = 0;
        while (!o_fc_w0_refill_done) @(negedge i_clk);
        @(negedge i_clk);
        refill_cycles += cycles-start;
    endtask

    task automatic clear_states;
        for (int s = 0; s < B; s++) begin
            for (int tile = 0; tile < 5; tile++) begin
                for (int row = 0; row < 4; row++) begin
                    @(negedge i_clk);
                    i_maintenance_tile = 4'(tile);
                    i_state_maintenance_write_addr = 10'(s*16+row);
                    i_trace_maintenance_write_addr = 10'(512+s*16+row);
                    i_state_maintenance_write_valid = 1;
                    i_trace_maintenance_write_valid = 1;
                    do @(posedge i_clk); while (!o_state_maintenance_write_ready || !o_trace_maintenance_write_ready);
                    @(negedge i_clk);
                    i_state_maintenance_write_valid = 0;
                    i_trace_maintenance_write_valid = 0;
                end
            end
        end
    endtask

    function automatic int rounds(input bit output_layer, input int row);
        int n;
        n = 0;
        if (output_layer) return 1;
        for (int lane = 0; lane < 4; lane++)
            if (int'(input_counts[(timestep*B+sample_index)*F+row*4+lane]) > n)
                n = int'(input_counts[(timestep*B+sample_index)*F+row*4+lane]);
        return n;
    endfunction

    function automatic logic [3:0] mask_for(input bit output_layer, input int row, input int rep);
        logic [3:0] mask;
        if (output_layer) return hidden_spikes[row*4 +: 4];
        for (int lane = 0; lane < 4; lane++)
            mask[lane] = int'(input_counts[(timestep*B+sample_index)*F+row*4+lane]) > rep;
        return mask;
    endfunction

    task automatic issue(input int row, input logic [3:0] mask);
        @(negedge i_clk);
        i_fc_srnn_group_weight_addr0 = 11'(row);
        i_fc_srnn_group_weight_addr1 = 11'(row);
        i_fc_srnn_group_weight_addr2 = 11'(row);
        i_fc_srnn_group_weight_addr3 = 11'(row);
        i_fc_srnn_group_source_valid = mask;
        i_fc_srnn_group_valid = 1;
        do @(posedge i_clk); while (!o_fc_srnn_group_ready);
        groups++;
        if (phase == 1) input_events += $countones(mask);
        else hidden_events += $countones(mask);
        @(negedge i_clk); i_fc_srnn_group_valid = 0;
    endtask

    task automatic forward_layer(input bit output_layer);
        int n, limit, done_before, capture_before, replay_before;
        longint start;
        logic [3:0] mask;
        phase = output_layer ? 2 : 1;
        start = cycles;
        if (!output_layer) hidden_spikes = 0;
        limit = output_layer ? 40 : 578;
        n = 0;
        for (int row = 0; row < limit; row++)
            for (int rep = 0; rep < rounds(output_layer, row); rep++)
                if (mask_for(output_layer, row, rep) != 0) n++;
        @(negedge i_clk);
        i_fc_srnn_transaction_tag = 52'((timestep*B+sample_index)*2+output_layer);
        i_fc_srnn_learning_cfg_tag = i_fc_srnn_transaction_tag;
        i_fc_srnn_learning_sample_id = 6'(sample_index);
        // Observations use layer-local IDs, independent of the physical tile.
        i_fc_srnn_learning_neuron_base = 0;
        i_fc_srnn_learning_neuron_count = output_layer ? C : H;
        i_fc_srnn_learning_fc_output_layer = output_layer;
        i_fc_srnn_learning_fc_logit_scale_enable = output_layer;
        i_fc_srnn_learning_fc_logit_accumulate = output_layer;
        i_fc_srnn_learning_fc_label = labels[sample_index];
        i_fc_srnn_learning_fc_sample_start = timestep == 0;
        i_fc_srnn_learning_cfg_valid = 1;
        do @(posedge i_clk); while (!o_fc_srnn_learning_cfg_ready);
        @(negedge i_clk); i_fc_srnn_learning_cfg_valid = 0;
        i_fc_srnn_weight_int8 = output_layer;
        i_fc_srnn_current_shift6 = !output_layer;
        i_fc_srnn_neuron_count = output_layer ? C : H;
        i_fc_srnn_destination_tile_base = output_layer ? 4 : 0;
        i_fc_srnn_leak = output_layer ? 0 : 32113;
        i_fc_srnn_threshold = output_layer ? 131071 : 1024;
        i_fc_srnn_work_base_addr = WORK_ADDR_WIDTH'(sample_index*16);
        i_fc_srnn_feedforward_group_count = 16'(n == 0 ? 1 : n);
        done_before = layer_done_count;
        capture_before = capture_done_count;
        replay_before = replay_done_count;
        i_fc_srnn_layer_valid = 1;
        do @(posedge i_clk); while (!o_fc_srnn_layer_ready);
        @(negedge i_clk); i_fc_srnn_layer_valid = 0;
        if (n == 0) issue(0, 0);
        else for (int row = 0; row < limit; row++) begin
            for (int rep = 0; rep < rounds(output_layer, row); rep++) begin
                mask = mask_for(output_layer, row, rep);
                if (mask != 0) issue(row, mask);
            end
        end
        while (layer_done_count != done_before+1 || capture_done_count != capture_before+1 ||
               replay_done_count != replay_before+1) @(negedge i_clk);
        @(negedge i_clk);
        if (output_layer) w2_cycles += cycles-start;
        else w0_cycles += cycles-start;
        phase = 0;
    endtask

    task automatic read_batch;
        batch_fixture = batches == 1 ? fixture : $sformatf("%s/batch_%0d", fixture, batch_index);
        $readmemh({batch_fixture,"/input_counts.mem"}, input_counts);
        $readmemh({batch_fixture,"/labels.mem"}, labels);
        $readmemh({batch_fixture,"/pre.mem"}, expected_pre);
        $readmemh({batch_fixture,"/trace.mem"}, expected_trace);
        $readmemh({batch_fixture,"/spike.mem"}, expected_spike);
        $readmemh({batch_fixture,"/dot.mem"}, expected_dot);
        $readmemh({batch_fixture,"/q6.mem"}, expected_q6);
        $readmemh({batch_fixture,"/w2_after.mem"}, expected_w2);
        $readmemh({batch_fixture,"/residual_after.mem"}, expected_residual);
    endtask

    initial begin
        if (!$value$plusargs("FIXTURE=%s", fixture)) $fatal(1, "fixture required");
        if (!$value$plusargs("STEPS=%d", steps)) steps = 2;
        if (!$value$plusargs("LEARNING=%d", learning)) learning = 1;
        if (!$value$plusargs("BATCHES=%d", batches)) batches = 1;
        if (steps < 1 || steps > 99 || (learning != 0 && learning != 1)) $fatal(1, "invalid run configuration");
        if (batches < 1 || batches > 10) $fatal(1, "invalid batch count");
        if (batches > 1) $readmemh({fixture,"/batch_steps.mem"}, batch_steps);
        else batch_steps[0] = steps;
        for (int b = 0; b < batches; b++) begin
            if (batch_steps[b] < 1 || batch_steps[b] > 99) $fatal(1, "invalid batch steps");
            total_steps += batch_steps[b];
        end
        read_batch();
        $readmemh({batch_fixture,"/parameters.mem"}, parameters);
        i_parameter_owner = 3'b001;
        i_topology_learning_enable = learning ? 3'b001 : 3'b000;
        i_fc_w0_refill_all = 1;
        i_fc_w0_version = 1;
        i_state_maintenance_write_mask = '1;
        i_trace_maintenance_write_mask = '1;
        i_state_maintenance_result_ready = 1;
        i_trace_maintenance_result_ready = 1;
        i_fc_srnn_observation_ready = 1;
        i_fc_srnn_observation_sink_idle = 1;
        i_fc_srnn_trace_decay = 64225;
        i_fc_srnn_learning_fc_threshold = 1024;
        i_fc_srnn_learning_fc_logit_scale_q16 = 24'(scale_q16);
        i_learning_fc_profile_id = 8'h11;
        i_learning_fc_w2_tensor_id = 8'h12;
        i_learning_fc_w2_parameter_base = W2_BASE;
        i_learning_fc_batch_count = B;
        i_learning_fc_hidden_count = H;
        i_learning_fc_output_count = C;
        i_learning_fc_vote_increment = 11;
        i_learning_fc_vote_threshold = 16'h8000;
        i_learning_fc_code_min = -127;
        i_learning_fc_code_max = 127;
        i_learning_fc_window_commit_ready = 1;
        repeat (6) @(negedge i_clk);
        i_rst_n = 1;
        start_cycle = cycles;
        fc_measure_stage = 1;
        load_parameters();
        fc_measure_stage = 2;
        refill(0); refill(1);
        load_cycles = cycles-start_cycle;
        start_cycle = cycles;
        fc_measure_stage = 3;
        clear_states();
        init_cycles = cycles-start_cycle;
        fc_measure_stage = 7;
        if (learning) begin
            @(negedge i_clk); i_learning_fc_cfg_valid = 1;
            do @(posedge i_clk); while (!o_learning_fc_cfg_ready);
            @(negedge i_clk); i_learning_fc_cfg_valid = 0;
        end
        $display("FC_FINAL_COUNTS_LOAD cycles=%0d init_cycles=%0d forward_reads=%0d",load_cycles,init_cycles,forward_reads);
        start_cycle = cycles;
        fc_measure_stage = 4;
        for (batch_index = 0; batch_index < batches; batch_index++) begin
            steps = batch_steps[batch_index];
            if (batch_index != 0) begin
                // Model state resets for new samples; parameter/residual store persists.
                read_batch();
                clear_states();
            end
        for (timestep = 0; timestep < steps; timestep++) begin
            for (sample_index = 0; sample_index < B; sample_index++) begin
                forward_layer(0);
                forward_layer(1);
            end
            if (learning) begin
                longint learn_start;
                fc_measure_stage = 5;
                learn_start = cycles;
                while (commits != window_index+1 || o_learning_fc_window_busy) @(negedge i_clk);
                @(negedge i_clk);
                learning_cycles += cycles-learn_start;
                fc_measure_stage = 4;
            end
            for (int addr = 0; addr < H*C; addr++) begin
                if (u_store.code_mem[W2_BASE+addr] != expected_w2[timestep*H*C+addr] ||
                    u_store.residual_mem[W2_BASE+addr] != expected_residual[timestep*H*C+addr])
                    $fatal(1,"FC committed store mismatch t=%0d addr=%0d code=%0d/%0d residual=%0d/%0d",timestep,addr,
                        u_store.code_mem[W2_BASE+addr],expected_w2[timestep*H*C+addr],
                        u_store.residual_mem[W2_BASE+addr],expected_residual[timestep*H*C+addr]);
            end
            if (learning) begin
                fc_measure_stage = 6;
                refill(1);
                fc_measure_stage = 4;
            end
            $display("FC_FINAL_COUNTS_STEP batch_index=%0d t=%0d cycles=%0d commits=%0d code_writes=%0d residual_writes=%0d",
                batch_index,timestep,cycles-start_cycle,commits,code_writes,residual_writes);
            window_index++;
        end
        end
        for (int addr = 0; addr < W2_BASE; addr++)
            if (u_store.code_mem[addr] != parameters[addr] || u_store.residual_mem[addr] != 0)
                $fatal(1, "frozen W0 mismatch addr=%0d",addr);
        if (hidden_checks != total_steps*B*H || dot_checks != total_steps*B*C || q6_checks != total_steps*B*C ||
            commits != (learning ? total_steps : 0)) $fatal(1,"incomplete FC final coverage h=%0d dot=%0d q6=%0d commits=%0d",hidden_checks,dot_checks,q6_checks,commits);
        $display("FC_FINAL_COUNTS_PASS learning=%0d batch=%0d batches=%0d steps=%0d cycles=%0d hidden_checks=%0d dot_checks=%0d q6_checks=%0d commits=%0d w0_cycles=%0d w2_cycles=%0d learning_cycles=%0d refill_cycles=%0d input_events=%0d hidden_events=%0d groups=%0d forward_reads=%0d learning_reads=%0d code_writes=%0d residual_writes=%0d",
            learning,B,batches,total_steps,cycles-start_cycle,hidden_checks,dot_checks,q6_checks,commits,w0_cycles,w2_cycles,learning_cycles,refill_cycles,
            input_events,hidden_events,groups,forward_reads,learning_reads,code_writes,residual_writes);
        fc_measure_complete = 1;
        $display("FC_PROFILE_MEASUREMENT_PASS learning=%0d steps=%0d", learning, total_steps);
        $display("FINAL_FC_WRAPPER_PASS raw_active=0");
        $finish;
    end

endmodule
`default_nettype wire
