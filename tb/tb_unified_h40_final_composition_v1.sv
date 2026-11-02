`timescale 1ns/1ps
`default_nettype none

module tb_unified_h40_final_composition_v1;
    localparam bit ENABLE_SRNN_WH1_LEARNING = 1'b0;
    localparam int PARAM_ADDR_WIDTH = 24;
    localparam int PARAM_DEPTH = 64;
    localparam int WH1_BASE = 0;
    localparam int WOUT_BASE = 16;
    localparam int FC_W0_BASE = 32;
    localparam int FC_W2_BASE = 48;
    localparam logic [51:0] FC_HIDDEN_TAG = 52'hfc00000000011;
    localparam logic [51:0] FC_OUTPUT_TAG = 52'hfc00000000012;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic clear_error = 1'b0;
    logic [4:0] debug_adapter_state_q = '0;

    logic srnn_cfg_valid = 1'b0;
    logic srnn_cfg_ready;
    logic srnn_cfg_error;
    logic srnn_load_valid = 1'b0;
    logic srnn_load_ready;
    logic [PARAM_ADDR_WIDTH-1:0] srnn_load_addr = '0;
    logic signed [15:0] srnn_load_code = '0;
    logic srnn_start;
    logic srnn_start_ready;
    logic srnn_learning_enable = 1'b0;
    logic [2:0] topology_learning_enable = 3'b000;
    logic [2:0] parameter_owner = 3'b010;

    logic fc_cfg_valid = 1'b0;
    logic fc_cfg_ready;
    logic fc_cfg_error;
    logic fc_release = 1'b0;
    logic fc_release_ready;
    logic fc_load_valid = 1'b0;
    logic fc_load_ready;
    logic [PARAM_ADDR_WIDTH-1:0] fc_load_addr = '0;
    logic signed [15:0] fc_load_code = '0;
    logic fc_window_commit_valid;
    logic [31:0] fc_window_commit_sequence;
    logic fc_window_busy;
    logic fc_error;

    logic fc_srnn_layer_valid = 1'b0;
    logic fc_srnn_layer_ready;
    logic [15:0] fc_srnn_neuron_count = '0;
    logic [15:0] fc_srnn_feedforward_group_count = 16'd1;
    logic [51:0] fc_srnn_transaction_tag = '0;
    logic fc_srnn_group_valid = 1'b0;
    logic fc_srnn_group_ready;
    logic [3:0] fc_srnn_group_source_valid = 4'b1111;
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
    logic fc_srnn_observation_last;
    logic fc_srnn_observation_error;

    logic fc_learning_cfg_valid = 1'b0;
    logic fc_learning_cfg_ready;
    logic [5:0] fc_learning_sample_id = '0;
    logic [9:0] fc_learning_neuron_count = '0;
    logic fc_learning_output_layer = 1'b0;
    logic [3:0] fc_learning_label = '0;
    logic fc_student_trace_valid;
    logic fc_student_trace_ready;
    logic [5:0] fc_student_trace_sample;
    logic [9:0] fc_student_trace_hidden;
    logic [16:0] fc_student_trace_q016;
    logic fc_logit_row_valid;
    logic fc_logit_row_ready;
    logic [5:0] fc_logit_sample;
    logic [3:0] fc_logit_label;
    logic fc_logit_valid;
    logic fc_logit_ready;
    logic signed [15:0] fc_logit_q6;
    logic fc_logit_last;
    logic fc_capture_done;
    logic fc_learning_busy;
    logic fc_learning_idle;
    logic fc_learning_error;

    logic fc_refill_valid = 1'b0;
    logic fc_refill_ready;
    logic [PARAM_ADDR_WIDTH-1:0] fc_refill_base = '0;
    logic [PARAM_ADDR_WIDTH-1:0] fc_refill_stride = '0;
    logic [4:0] fc_refill_feature_count = '0;
    logic [9:0] fc_refill_neuron_count = '0;
    logic [4:0] fc_refill_tile_count = '0;
    logic [15:0] fc_refill_version = '0;
    logic fc_refill_done;
    logic [31:0] fc_refill_rows;
    logic [15:0] fc_refill_published_version;
    logic fc_cache_forward_read_valid;
    logic fc_cache_forward_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] fc_cache_forward_read_addr;
    logic [15:0] fc_cache_forward_read_tag;
    logic fc_cache_forward_response_valid;
    logic fc_cache_forward_response_ready;
    logic [15:0] fc_cache_forward_response_tag;
    logic signed [15:0] fc_cache_forward_response_code;
    logic fc_cache_forward_response_error;
    logic srnn_backbone_refill_valid = 1'b0;
    logic srnn_backbone_refill_ready;
    logic [PARAM_ADDR_WIDTH-1:0] srnn_backbone_refill_base = '0;
    logic [9:0] srnn_backbone_refill_source_count = '0;
    logic [9:0] srnn_backbone_refill_destination_count = '0;
    logic [15:0] srnn_backbone_refill_version = '0;
    logic srnn_backbone_refill_done;
    logic [31:0] srnn_backbone_refill_rows;
    logic [15:0] srnn_backbone_published_version;
    logic srnn_backbone_cache_forward_read_valid;
    logic srnn_backbone_cache_forward_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0]
        srnn_backbone_cache_forward_read_addr;
    logic [15:0] srnn_backbone_cache_forward_read_tag;
    logic srnn_backbone_cache_forward_response_valid;
    logic srnn_backbone_cache_forward_response_ready;
    logic [15:0] srnn_backbone_cache_forward_response_tag;
    logic signed [15:0] srnn_backbone_cache_forward_response_code;
    logic srnn_backbone_cache_forward_response_error;
    logic plane_fc_forward_read_valid;
    logic plane_fc_forward_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] plane_fc_forward_read_addr;
    logic [15:0] plane_fc_forward_read_tag;
    logic plane_fc_forward_response_valid;
    logic plane_fc_forward_response_ready;
    logic [15:0] plane_fc_forward_response_tag;
    logic signed [15:0] plane_fc_forward_response_code;
    logic plane_fc_forward_response_error;
    logic srnn_backbone_refill_response_valid_q = 1'b0;
    logic [15:0] srnn_backbone_refill_response_tag_q = '0;

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
    logic srnn_backbone_valid = 1'b0;
    logic srnn_backbone_ready;
    logic srnn_backbone_path = 1'b0;
    logic [9:0] srnn_backbone_id = '0;
    logic signed [31:0] srnn_backbone_q14 = '0;
    logic srnn_backbone_last = 1'b0;
    logic srnn_backbone_done = 1'b0;
    logic srnn_target_fixture_ready;
    logic c17_srnn_backbone_valid;
    logic c17_srnn_backbone_ready;
    logic c17_srnn_backbone_path;
    logic [5:0] c17_srnn_backbone_sample_id;
    logic [9:0] c17_srnn_backbone_id;
    logic signed [31:0] c17_srnn_backbone_q14;
    logic c17_srnn_backbone_last;
    logic c17_srnn_backbone_done;
    logic srnn_actual_tape_cfg_valid = 1'b0;
    logic srnn_actual_tape_cfg_ready;
    logic srnn_actual_tape_capture_done;
    logic srnn_actual_tape_capture_complete;
    logic srnn_actual_tape_replay_start = 1'b0;
    logic srnn_actual_tape_replay_start_ready;
    logic srnn_actual_delivery_done;
    logic srnn_actual_tape_busy;
    logic srnn_actual_tape_error;
    logic plane_srnn_backbone_valid;
    logic plane_srnn_backbone_path;
    logic [5:0] plane_srnn_backbone_sample_id;
    logic [9:0] plane_srnn_backbone_id;
    logic signed [31:0] plane_srnn_backbone_q14;
    logic plane_srnn_backbone_last;
    logic plane_srnn_backbone_done;
    logic plane_srnn_backbone_error;
    logic srnn_target_trace_start;
    logic srnn_target_trace_valid = 1'b0;
    logic srnn_target_trace_ready;
    logic [7:0] srnn_target_trace_sample = '0;
    logic [9:0] srnn_target_trace_backbone = '0;
    logic signed [31:0] srnn_target_trace_q14 = '0;
    logic srnn_target_trace_last = 1'b0;
    logic srnn_target_trace_done = 1'b0;
    logic srnn_label_replay_start;
    logic srnn_label_valid = 1'b0;
    logic srnn_label_ready;
    logic [7:0] srnn_label_sample = '0;
    logic [4:0] srnn_label_class = '0;
    logic srnn_label_last = 1'b0;
    logic srnn_learning_commit;
    logic srnn_detach;
    logic srnn_aborted;
    logic [PARAM_ADDR_WIDTH:0] srnn_zero_gradients;
    logic [PARAM_ADDR_WIDTH:0] srnn_updates;
    logic srnn_logit_valid;
    logic [7:0] srnn_logit_sample;
    logic [4:0] srnn_logit_class;
    logic signed [63:0] srnn_logit_q28;
    logic srnn_logit_last;
    logic srnn_done;
    logic srnn_busy;
    logic srnn_error;

    logic h40_cfg_valid;
    logic h40_cfg_ready;
    logic h40_cfg_error;
    logic [7:0] h40_cfg_batch_count;
    logic [9:0] h40_cfg_backbone_count;
    logic [8:0] h40_cfg_hidden_count;
    logic [4:0] h40_cfg_class_count;
    logic [PARAM_ADDR_WIDTH-1:0] h40_cfg_wh1_base;
    logic [PARAM_ADDR_WIDTH-1:0] h40_cfg_wout_base;
    logic h40_prepare_valid;
    logic h40_prepare_ready;
    logic [15:0] h40_prepare_version;
    logic h40_prepare_done;
    logic h40_cache_valid;
    logic h40_cache_invalidate;
    logic h40_forward_read_valid;
    logic h40_forward_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] h40_forward_read_addr;
    logic [15:0] h40_forward_read_tag;
    logic h40_forward_response_valid;
    logic h40_forward_response_ready;
    logic [15:0] h40_forward_response_tag;
    logic signed [15:0] h40_forward_response_code;
    logic h40_forward_response_error;
    logic h40_timestep_start;
    logic h40_timestep_start_ready;
    logic h40_start_timestep_id;
    logic [15:0] h40_sequence_version;
    logic h40_include_target_path;
    logic h40_observation_valid;
    logic h40_observation_ready;
    logic h40_observation_path;
    logic [7:0] h40_observation_sample;
    logic [9:0] h40_observation_backbone;
    logic signed [31:0] h40_observation_q14;
    logic h40_observation_last;
    logic h40_observation_done;
    logic h40_hidden_valid;
    logic h40_hidden_ready;
    logic h40_hidden_path;
    logic [7:0] h40_hidden_sample;
    logic [8:0] h40_hidden_id;
    logic signed [31:0] h40_hidden_pre;
    logic signed [31:0] h40_hidden_next;
    logic h40_hidden_spike;
    logic signed [31:0] h40_hidden_trace;
    logic signed [31:0] h40_hidden_feature;
    logic h40_hidden_last;
    logic h40_logit_valid;
    logic h40_logit_ready;
    logic [7:0] h40_logit_sample;
    logic [4:0] h40_logit_class;
    logic signed [63:0] h40_logit_q28;
    logic h40_logit_last;
    logic h40_result_timestep_id;
    logic h40_timestep_done;
    logic h40_busy;
    logic h40_sticky_error;
    logic h40_protocol_error;
    logic h40_overflow;

    // The production C17 SCNN learner and the FC/SRNN learning plane share
    // client 2 of the same parameter router even though this focused test
    // drives only the FC and SRNN transactions.
    logic scnn_store_load_valid;
    logic scnn_store_load_ready;
    logic [PARAM_ADDR_WIDTH-1:0] scnn_store_load_addr;
    logic signed [15:0] scnn_store_load_code;
    logic signed [15:0] scnn_store_load_residual;
    logic scnn_store_learning_read_valid;
    logic scnn_store_learning_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] scnn_store_learning_read_addr;
    logic [15:0] scnn_store_learning_read_tag;
    logic scnn_store_learning_response_valid;
    logic scnn_store_learning_response_ready;
    logic [15:0] scnn_store_learning_response_tag;
    logic signed [15:0] scnn_store_learning_response_code;
    logic signed [15:0] scnn_store_learning_response_residual;
    logic scnn_store_learning_response_error;
    logic scnn_store_learning_write_valid;
    logic scnn_store_learning_write_ready;
    logic [PARAM_ADDR_WIDTH-1:0] scnn_store_learning_write_addr;
    logic signed [15:0] scnn_store_learning_write_code;
    logic signed [15:0] scnn_store_learning_write_residual;
    logic scnn_store_learning_write_code_enable;
    logic scnn_store_learning_write_residual_enable;
    logic scnn_store_forward_read_valid;
    logic scnn_store_forward_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] scnn_store_forward_read_addr;
    logic [15:0] scnn_store_forward_read_tag;
    logic scnn_store_forward_response_valid;
    logic scnn_store_forward_response_ready;
    logic [15:0] scnn_store_forward_response_tag;
    logic signed [15:0] scnn_store_forward_response_code;
    logic scnn_store_forward_response_error;
    logic scnn_store_protocol_error;

    logic store_load_valid;
    logic store_load_ready;
    logic [PARAM_ADDR_WIDTH-1:0] store_load_addr;
    logic signed [15:0] store_load_code;
    logic signed [15:0] store_load_residual;
    logic store_learning_read_valid;
    logic store_learning_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] store_learning_read_addr;
    logic [15:0] store_learning_read_tag;
    logic store_learning_response_valid;
    logic store_learning_response_ready;
    logic [15:0] store_learning_response_tag;
    logic signed [15:0] store_learning_response_code;
    logic signed [15:0] store_learning_response_residual;
    logic store_learning_response_error;
    logic store_learning_write_valid;
    logic store_learning_write_ready;
    logic [PARAM_ADDR_WIDTH-1:0] store_learning_write_addr;
    logic signed [15:0] store_learning_write_code;
    logic signed [15:0] store_learning_write_residual;
    logic store_learning_write_code_enable;
    logic store_learning_write_residual_enable;
    logic store_forward_read_valid;
    logic store_forward_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] store_forward_read_addr;
    logic [15:0] store_forward_read_tag;
    logic store_forward_response_valid;
    logic store_forward_response_ready;
    logic [15:0] store_forward_response_tag;
    logic signed [15:0] store_forward_response_code;
    logic store_forward_response_error;
    logic store_protocol_error;
    logic backend_error;
    logic [11:0] array_tile_error;

    int hidden_seen = 0;
    int logit_seen = 0;
    int forward_reads = 0;
    int learning_writes = 0;
    int learning_code_writes = 0;
    int learning_residual_writes = 0;
    int srnn_wh1_writes = 0;
    int srnn_wout_writes = 0;
    int commit_count = 0;
    int detach_count = 0;
    int cache_invalidations = 0;
    int cache_invalidations_before_learning = 0;
    int fc_commit_count = 0;
    int fc_learning_write_base = 0;
    int fc_learning_code_write_base = 0;
    int fc_learning_residual_write_base = 0;
    int fc_refill_forward_reads = 0;
    int fc_trace_count = 0;
    int fc_logit_count = 0;
    int fc_capture_done_count = 0;
    int fc_layer_done_count = 0;
    int fc_phase = 0;
    logic signed [31:0] fc_baseline_output [0:1];
    logic signed [31:0] fc_postlearn_output [0:1];
    logic [3:0] fc_hidden_spike_mask = '0;
    logic [1:0] fc_srnn_mode = 2'd0;
    logic [1:0] fc_learning_cfg_mode = 2'd0;
    logic fc_learning_srnn_path = 1'b0;
    int phase = 0;
    int phase_hidden_base = 0;
    int phase_logit_base = 0;
    logic signed [63:0] baseline_logits [0:1];
    logic signed [63:0] postlearn_logits [0:1];

    always #5 clk = ~clk;

    // FC W0 refill uses the shared trainable parameter service. The frozen
    // SRNN backbone has a separate immutable-model scalar service because its
    // INT8 bank-pair layout differs from the FC INT5 row layout.
    // FC cache/parameter links are internal to u_final.
    assign srnn_backbone_cache_forward_read_ready =
        !srnn_backbone_refill_response_valid_q;
    assign srnn_backbone_cache_forward_response_valid =
        srnn_backbone_refill_response_valid_q;
    assign srnn_backbone_cache_forward_response_tag =
        srnn_backbone_refill_response_tag_q;
    assign srnn_backbone_cache_forward_response_code = 16'sd10;
    assign srnn_backbone_cache_forward_response_error = 1'b0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            srnn_backbone_refill_response_valid_q <= 1'b0;
            srnn_backbone_refill_response_tag_q <= '0;
        end else begin
            if (srnn_backbone_refill_response_valid_q &&
                    srnn_backbone_cache_forward_response_ready)
                srnn_backbone_refill_response_valid_q <= 1'b0;
            if (srnn_backbone_cache_forward_read_valid &&
                    srnn_backbone_cache_forward_read_ready) begin
                srnn_backbone_refill_response_valid_q <= 1'b1;
                srnn_backbone_refill_response_tag_q <=
                    srnn_backbone_cache_forward_read_tag;
            end
        end
    end

    // Production phase bridge: capture the physical actual path, start the
    // M2 plane and replay only after the frozen backbone releases the array.
    // In learning mode it then admits the bounded target fixture.
        // Generated compatibility instantiation: the original directed
    // integration stimulus now reaches the production composition top.
    unified_h40_final_composition_top_v1 #(
        .SRNN_M2_PARAM_DEPTH(PARAM_DEPTH),
        .SRNN_BACKBONE_PARAM_DEPTH(PARAM_DEPTH),
        .SRNN_BACKBONE_PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .SRNN_M2_BATCH_MAX(2),
        .SRNN_M2_TIMESTEP_MAX(2),
        .SRNN_M2_BACKBONE_MAX(4),
        .SRNN_M2_HIDDEN_MAX(4),
        .SRNN_M2_CLASS_MAX(4),
        .PARAM_DEPTH(PARAM_DEPTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
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
        .INTERNAL_MODULATION_BATCH(1)
    ) u_final (
        .i_initialize_valid(1'b0),
        .i_classifier_pair(1'b0),
        .i_context_model_version('0),
        .i_context_adapt_version('0),
        .i_system_parameter_owner(1'b0),
        .i_system_learning_enable(1'b0),
        .i_system_profile_aligned(1'b0),
        .i_system_profile_id('0),
        .i_request_valid(1'b0),
        .i_learning_enable(1'b0),
        .i_batch_count(1'b0),
        .i_timestep(1'b0),
        .i_timestep_count(1'b1),
        .i_sample_version('0),
        .i_labels_flat('0),
        .i_fc_srnn_layer_valid(fc_srnn_layer_valid),
        .o_fc_srnn_layer_ready(fc_srnn_layer_ready),
        .i_fc_srnn_mode(fc_srnn_mode),
        .i_fc_srnn_neuron_count(fc_srnn_neuron_count),
        .i_fc_srnn_feedforward_group_count(fc_srnn_feedforward_group_count),
        .i_fc_srnn_recurrent_group_count('0),
        .i_fc_srnn_recurrent_enable(1'b0),
        .i_fc_srnn_leak('0),
        .i_fc_srnn_threshold(32'd8),
        .i_fc_srnn_reset_to_zero(1'b1),
        .i_fc_srnn_trace_decay('0),
        .i_fc_srnn_transaction_tag(fc_srnn_transaction_tag),
        .i_fc_srnn_work_base_addr('0),
        .i_fc_srnn_group_valid(fc_srnn_group_valid),
        .o_fc_srnn_group_ready(fc_srnn_group_ready),
        .i_fc_srnn_group_recurrent(1'b0),
        .i_fc_srnn_group_source_valid(fc_srnn_group_source_valid),
        .i_fc_srnn_group_weight_addr0(11'd0),
        .i_fc_srnn_group_weight_addr1(11'd0),
        .i_fc_srnn_group_weight_addr2(11'd0),
        .i_fc_srnn_group_weight_addr3(11'd0),
        .o_fc_srnn_layer_done(fc_srnn_layer_done),
        .o_fc_srnn_busy(fc_srnn_busy),
        .o_fc_srnn_idle(fc_srnn_idle),
        .o_fc_srnn_sticky_error(fc_srnn_error),
        .o_fc_srnn_error_code(fc_srnn_error_code),
        .o_fc_srnn_observation_valid(fc_srnn_observation_valid),
        .i_fc_srnn_observation_ready(1'b1),
        .i_fc_srnn_observation_sink_idle(fc_learning_idle),
        .o_fc_srnn_observation_mode(fc_srnn_observation_mode),
        .o_fc_srnn_observation_transaction_tag(fc_srnn_observation_tag),
        .o_fc_srnn_observation_neuron_id(fc_srnn_observation_neuron),
        .o_fc_srnn_observation_spike(fc_srnn_observation_spike),
        .o_fc_srnn_observation_pre_state(fc_srnn_observation_pre_state),
        .o_fc_srnn_observation_last_in_record(fc_srnn_observation_last),
        .o_fc_srnn_observation_sticky_error(fc_srnn_observation_error),
        .i_fc_srnn_learning_cfg_valid(fc_learning_cfg_valid),
        .o_fc_srnn_learning_cfg_ready(fc_learning_cfg_ready),
        .i_fc_srnn_learning_cfg_mode(fc_learning_cfg_mode),
        .i_fc_srnn_learning_cfg_tag(fc_srnn_transaction_tag),
        .i_fc_srnn_learning_sample_id(fc_learning_sample_id),
        .i_fc_srnn_learning_neuron_base('0),
        .i_fc_srnn_learning_neuron_count(fc_learning_neuron_count),
        .i_fc_srnn_learning_fc_threshold(18'sd8),
        .i_fc_srnn_learning_fc_output_layer(fc_learning_output_layer),
        .i_fc_srnn_learning_fc_label(fc_learning_label),
        .i_fc_srnn_learning_srnn_path(fc_learning_srnn_path),
        .o_fc_logit_last(fc_logit_last),
        .o_fc_srnn_learning_capture_done(fc_capture_done),
        .o_fc_srnn_learning_done(c17_srnn_backbone_done),
        .o_fc_srnn_learning_busy(fc_learning_busy),
        .o_fc_srnn_learning_idle(fc_learning_idle),
        .o_fc_srnn_learning_sticky_error(fc_learning_error),
        .i_fc_w0_refill_valid(fc_refill_valid),
        .o_fc_w0_refill_ready(fc_refill_ready),
        .i_fc_w0_refill_all(1'b1),
        .i_fc_w0_dirty_feature('0),
        .i_fc_w0_dirty_tile('0),
        .i_fc_w0_parameter_base(fc_refill_base),
        .i_fc_w0_hidden_stride(fc_refill_stride),
        .i_fc_cache_row_base('0),
        .i_fc_w0_feature_count(fc_refill_feature_count),
        .i_fc_w0_hidden_count(fc_refill_neuron_count),
        .i_fc_w0_tile_count(fc_refill_tile_count),
        .i_fc_w0_version(fc_refill_version),
        .o_fc_w0_refill_done(fc_refill_done),
        .o_fc_w0_rows_published(fc_refill_rows),
        .o_fc_w0_published_version(fc_refill_published_version),
        .i_srnn_backbone_refill_valid(srnn_backbone_refill_valid),
        .o_srnn_backbone_refill_ready(srnn_backbone_refill_ready),
        .i_srnn_backbone_refill_all(1'b1),
        .i_srnn_backbone_dirty_source('0),
        .i_srnn_backbone_dirty_tile('0),
        .i_srnn_backbone_parameter_base(srnn_backbone_refill_base),
        .i_srnn_backbone_cache_row_base('0),
        .i_srnn_backbone_source_count(srnn_backbone_refill_source_count),
        .i_srnn_backbone_destination_count(srnn_backbone_refill_destination_count),
        .i_srnn_backbone_tile_base('0),
        .i_srnn_backbone_version(srnn_backbone_refill_version),
        .o_srnn_backbone_forward_read_valid(srnn_backbone_cache_forward_read_valid),
        .i_srnn_backbone_forward_read_ready(srnn_backbone_cache_forward_read_ready),
        .o_srnn_backbone_forward_read_addr(srnn_backbone_cache_forward_read_addr),
        .o_srnn_backbone_forward_read_tag(srnn_backbone_cache_forward_read_tag),
        .i_srnn_backbone_forward_response_valid(srnn_backbone_cache_forward_response_valid),
        .o_srnn_backbone_forward_response_ready(srnn_backbone_cache_forward_response_ready),
        .i_srnn_backbone_forward_response_tag(srnn_backbone_cache_forward_response_tag),
        .i_srnn_backbone_forward_response_code(srnn_backbone_cache_forward_response_code),
        .i_srnn_backbone_forward_response_error(srnn_backbone_cache_forward_response_error),
        .o_srnn_backbone_refill_done(srnn_backbone_refill_done),
        .o_srnn_backbone_rows_published(srnn_backbone_refill_rows),
        .o_srnn_backbone_published_version(srnn_backbone_published_version),
        .i_forward_start_valid(1'b0),
        .i_forward_done(1'b0),
        .i_restore_complete(1'b0),
        .i_feature_request_ready(1'b0),
        .i_feature_response_valid(1'b0),
        .i_feature_mask0('0),
        .i_feature_mask1('0),
        .i_feature_mask2('0),
        .i_feature_mask3('0),
        .i_feature_service_error(1'b0),
        .i_feature_write_slot(1'b0),
        .i_feature_write_base_addr('0),
        .i_feature_write_row_stride_words('0),
        .i_feature_write_channel_stride_words('0),
        .i_feature_update_ready(1'b0),
        .i_feature_update_done(1'b0),
        .i_maintenance_tile(maintenance_tile),
        .i_state_maintenance_write_valid(state_write_valid),
        .o_state_maintenance_write_ready(state_write_ready),
        .i_state_maintenance_write_addr(state_write_addr),
        .i_state_maintenance_write_data(state_write_data),
        .i_state_maintenance_write_mask(state_write_mask),
        .i_state_maintenance_read_valid(1'b0),
        .i_state_maintenance_read_addr('0),
        .i_state_maintenance_result_ready(1'b1),
        .i_trace_maintenance_write_valid(trace_write_valid),
        .o_trace_maintenance_write_ready(trace_write_ready),
        .i_trace_maintenance_write_addr(trace_write_addr),
        .i_trace_maintenance_write_data(trace_write_data),
        .i_trace_maintenance_write_mask(trace_write_mask),
        .i_trace_maintenance_read_valid(1'b0),
        .i_trace_maintenance_read_addr('0),
        .i_trace_maintenance_result_ready(1'b1),
        .o_array_tile_sticky_error(array_tile_error),
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_clear_error(clear_error),
        .i_srnn_phase_abort(1'b0),
        .i_srnn_phase_capture_cfg_valid(srnn_actual_tape_cfg_valid),
        .o_srnn_phase_capture_cfg_ready(srnn_actual_tape_cfg_ready),
        .i_srnn_phase_record_count(10'd2),
        .i_srnn_phase_start_valid(srnn_actual_tape_replay_start),
        .o_srnn_phase_start_ready(srnn_actual_tape_replay_start_ready),
        .i_srnn_phase_learning_enable(srnn_learning_enable),
        .i_srnn_phase_target_valid(srnn_backbone_valid),
        .o_srnn_phase_target_ready(srnn_target_fixture_ready),
        .i_srnn_phase_target_sample_id(6'd0),
        .i_srnn_phase_target_backbone_id(srnn_backbone_id),
        .i_srnn_phase_target_q14(srnn_backbone_q14),
        .i_srnn_phase_target_last(srnn_backbone_last),
        .i_srnn_phase_target_done(srnn_backbone_done),
        .i_srnn_phase_target_error(1'b0),
        .o_srnn_phase_capture_done(srnn_actual_tape_capture_done),
        .o_srnn_phase_capture_complete(srnn_actual_tape_capture_complete),
        .o_srnn_phase_delivery_done(srnn_actual_delivery_done),
        .o_srnn_phase_busy(srnn_actual_tape_busy),
        .o_srnn_phase_idle(),
        .o_srnn_phase_sticky_error(srnn_actual_tape_error),
        .i_parameter_owner(parameter_owner),
        .i_topology_learning_enable(topology_learning_enable),
        .i_learning_fc_cfg_valid(fc_cfg_valid),
        .o_learning_fc_cfg_ready(fc_cfg_ready),
        .i_learning_fc_profile_id(8'h11),
        .i_learning_fc_w2_tensor_id(8'h12),
        .i_learning_fc_w2_parameter_base(PARAM_ADDR_WIDTH'(FC_W2_BASE)),
        .i_learning_fc_batch_count(5'd2),
        .i_learning_fc_hidden_count(9'd4),
        .i_learning_fc_output_count(5'd2),
        .i_learning_fc_vote_increment(16'd11),
        .i_learning_fc_vote_threshold(16'h8000),
        .i_learning_fc_code_min(-16'sd16),
        .i_learning_fc_code_max(16'sd15),
        .o_learning_fc_cfg_error(fc_cfg_error),
        .i_learning_fc_release(fc_release),
        .o_learning_fc_release_ready(fc_release_ready),
        .o_learning_fc_window_commit_valid(fc_window_commit_valid),
        .i_learning_fc_window_commit_ready(1'b1),
        .o_learning_fc_window_commit_sequence(fc_window_commit_sequence),
        .o_learning_fc_window_busy(fc_window_busy),
        .o_learning_fc_error(fc_error),
        .i_learning_fc_parameter_load_valid(fc_load_valid),
        .o_learning_fc_parameter_load_ready(fc_load_ready),
        .i_learning_fc_parameter_load_addr(fc_load_addr),
        .i_learning_fc_parameter_load_code(fc_load_code),
        .i_learning_fc_parameter_load_residual('0),
        .i_learning_srnn_cfg_valid(srnn_cfg_valid),
        .o_learning_srnn_cfg_ready(srnn_cfg_ready),
        .i_learning_srnn_profile_id(8'h21),
        .i_learning_srnn_wh1_tensor_id(8'h31),
        .i_learning_srnn_wh1_base(PARAM_ADDR_WIDTH'(WH1_BASE)),
        .i_learning_srnn_wh1_increment(16'd1),
        .i_learning_srnn_wh1_threshold(16'd1),
        .i_learning_srnn_wout_tensor_id(8'h32),
        .i_learning_srnn_wout_base(PARAM_ADDR_WIDTH'(WOUT_BASE)),
        .i_learning_srnn_wout_increment(16'd1),
        .i_learning_srnn_wout_threshold(16'd1),
        .i_learning_srnn_code_min(-16'sd16384),
        .i_learning_srnn_code_max(16'sd16384),
        .i_learning_srnn_batch_count(8'd1),
        .i_learning_srnn_hidden_count(9'd4),
        .i_learning_srnn_backbone_count(10'd2),
        .i_learning_srnn_class_count(5'd2),
        .i_learning_srnn_wout_element_count(PARAM_ADDR_WIDTH'(8)),
        .o_learning_srnn_cfg_error(srnn_cfg_error),
        .i_learning_srnn_parameter_load_valid(srnn_load_valid),
        .o_learning_srnn_parameter_load_ready(srnn_load_ready),
        .i_learning_srnn_parameter_load_addr(srnn_load_addr),
        .i_learning_srnn_parameter_load_code(srnn_load_code),
        .i_learning_srnn_parameter_load_residual('0),
        .i_learning_srnn_timestep_id(1'b0),
        .i_learning_srnn_sequence_version(16'd1),
        .i_learning_srnn_abort_timestep(1'b0),
        .o_learning_srnn_target_trace_start(srnn_target_trace_start),
        .i_learning_srnn_target_trace_start_ready(1'b1),
        .i_learning_srnn_target_trace_valid(srnn_target_trace_valid),
        .o_learning_srnn_target_trace_ready(srnn_target_trace_ready),
        .i_learning_srnn_target_trace_sample_id(srnn_target_trace_sample),
        .i_learning_srnn_target_trace_backbone_id(srnn_target_trace_backbone),
        .i_learning_srnn_target_trace_q14(srnn_target_trace_q14),
        .i_learning_srnn_target_trace_last(srnn_target_trace_last),
        .i_learning_srnn_target_trace_done(srnn_target_trace_done),
        .i_learning_srnn_target_trace_error(1'b0),
        .o_learning_srnn_label_replay_start(srnn_label_replay_start),
        .i_learning_srnn_label_replay_start_ready(1'b1),
        .i_learning_srnn_label_valid(srnn_label_valid),
        .o_learning_srnn_label_ready(srnn_label_ready),
        .i_learning_srnn_label_sample_id(srnn_label_sample),
        .i_learning_srnn_label_class_id(srnn_label_class),
        .i_learning_srnn_label_last(srnn_label_last),
        .i_learning_srnn_restore_complete(1'b0),
        .o_learning_srnn_learning_commit(srnn_learning_commit),
        .o_learning_srnn_detach(srnn_detach),
        .o_learning_srnn_logit_valid(srnn_logit_valid),
        .i_learning_srnn_logit_ready(1'b1),
        .o_learning_srnn_logit_sample_id(srnn_logit_sample),
        .o_learning_srnn_logit_class_id(srnn_logit_class),
        .o_learning_srnn_logit_q28(srnn_logit_q28),
        .o_learning_srnn_logit_last(srnn_logit_last),
        .o_learning_srnn_timestep_done(srnn_done),
        .o_learning_srnn_timestep_aborted(srnn_aborted),
        .o_learning_srnn_busy(srnn_busy),
        .o_learning_srnn_error(srnn_error),
        .o_learning_srnn_zero_gradient_count(srnn_zero_gradients),
        .o_learning_srnn_update_count(srnn_updates),
        .o_store_load_valid(store_load_valid),
        .i_store_load_ready(store_load_ready),
        .o_store_load_addr(store_load_addr),
        .o_store_load_code(store_load_code),
        .o_store_load_residual(store_load_residual),
        .o_store_learning_read_valid(store_learning_read_valid),
        .i_store_learning_read_ready(store_learning_read_ready),
        .o_store_learning_read_addr(store_learning_read_addr),
        .o_store_learning_read_tag(store_learning_read_tag),
        .i_store_learning_response_valid(store_learning_response_valid),
        .o_store_learning_response_ready(store_learning_response_ready),
        .i_store_learning_response_tag(store_learning_response_tag),
        .i_store_learning_response_code(store_learning_response_code),
        .i_store_learning_response_residual(store_learning_response_residual),
        .i_store_learning_response_error(store_learning_response_error),
        .o_store_learning_write_valid(store_learning_write_valid),
        .i_store_learning_write_ready(store_learning_write_ready),
        .o_store_learning_write_addr(store_learning_write_addr),
        .o_store_learning_write_code(store_learning_write_code),
        .o_store_learning_write_residual(store_learning_write_residual),
        .o_store_learning_write_code_enable(store_learning_write_code_enable),
        .o_store_learning_write_residual_enable(store_learning_write_residual_enable),
        .o_store_forward_read_valid(store_forward_read_valid),
        .i_store_forward_read_ready(store_forward_read_ready),
        .o_store_forward_read_addr(store_forward_read_addr),
        .o_store_forward_read_tag(store_forward_read_tag),
        .i_store_forward_response_valid(store_forward_response_valid),
        .o_store_forward_response_ready(store_forward_response_ready),
        .i_store_forward_response_tag(store_forward_response_tag),
        .i_store_forward_response_code(store_forward_response_code),
        .i_store_forward_response_error(store_forward_response_error),
        .i_store_quiescent(1'b1),
        .i_store_protocol_error(store_protocol_error),
        .o_backend_error(backend_error)
    );

    // Compatibility observation aliases for the original scoreboards.
    assign srnn_start = u_final.link_srnn_plane_timestep_start;
    assign srnn_start_ready = u_final.link_srnn_plane_timestep_start_ready;
    assign srnn_backbone_ready = u_final.link_srnn_plane_backbone_ready;
    assign fc_student_trace_valid = u_final.link_fc_student_trace_valid;
    assign fc_student_trace_ready = u_final.link_fc_student_trace_ready;
    assign fc_student_trace_sample = u_final.link_fc_student_trace_sample;
    assign fc_student_trace_hidden = u_final.link_fc_student_trace_hidden;
    assign fc_student_trace_q016 = u_final.link_fc_student_trace_q016;
    assign fc_logit_row_valid = u_final.link_fc_logit_row_valid;
    assign fc_logit_row_ready = u_final.link_fc_logit_row_ready;
    assign fc_logit_sample = u_final.link_fc_logit_sample;
    assign fc_logit_label = u_final.link_fc_logit_label;
    assign fc_logit_valid = u_final.link_fc_logit_valid;
    assign fc_logit_ready = u_final.link_fc_logit_ready;
    assign fc_logit_q6 = u_final.link_fc_logit_q6;
    assign fc_cache_forward_read_valid = u_final.link_fc_forward_read_valid;
    assign fc_cache_forward_read_addr = u_final.link_fc_forward_read_addr;
    assign fc_cache_forward_read_tag = u_final.link_fc_forward_read_tag;
    assign fc_cache_forward_read_ready = u_final.link_fc_forward_read_ready;
    assign fc_cache_forward_response_valid = u_final.link_fc_forward_response_valid;
    assign fc_cache_forward_response_ready = u_final.link_fc_forward_response_ready;
    assign fc_cache_forward_response_tag = u_final.link_fc_forward_response_tag;
    assign fc_cache_forward_response_code = u_final.link_fc_forward_response_code;
    assign fc_cache_forward_response_error = u_final.link_fc_forward_response_error;
    assign plane_fc_forward_read_valid = u_final.link_fc_forward_read_valid;
    assign plane_fc_forward_read_ready = u_final.link_fc_forward_read_ready;
    assign plane_fc_forward_read_addr = u_final.link_fc_forward_read_addr;
    assign plane_fc_forward_read_tag = u_final.link_fc_forward_read_tag;
    assign plane_fc_forward_response_valid = u_final.link_fc_forward_response_valid;
    assign plane_fc_forward_response_ready = u_final.link_fc_forward_response_ready;
    assign plane_fc_forward_response_tag = u_final.link_fc_forward_response_tag;
    assign plane_fc_forward_response_code = u_final.link_fc_forward_response_code;
    assign plane_fc_forward_response_error = u_final.link_fc_forward_response_error;
    assign h40_cfg_valid = u_final.link_plane_to_system_srnn_h40_cfg_valid;
    assign h40_cfg_ready = u_final.link_system_to_plane_srnn_h40_cfg_ready;
    assign h40_cfg_error = u_final.link_system_to_plane_srnn_h40_cfg_error;
    assign h40_cfg_batch_count = u_final.link_plane_to_system_srnn_h40_cfg_batch_count;
    assign h40_cfg_backbone_count = u_final.link_plane_to_system_srnn_h40_cfg_backbone_count;
    assign h40_cfg_hidden_count = u_final.link_plane_to_system_srnn_h40_cfg_hidden_count;
    assign h40_cfg_class_count = u_final.link_plane_to_system_srnn_h40_cfg_class_count;
    assign h40_cfg_wh1_base = u_final.link_plane_to_system_srnn_h40_cfg_wh1_parameter_base;
    assign h40_cfg_wout_base = u_final.link_plane_to_system_srnn_h40_cfg_wout_parameter_base;
    assign h40_prepare_valid = u_final.link_plane_to_system_srnn_h40_prepare_valid;
    assign h40_prepare_ready = u_final.link_system_to_plane_srnn_h40_prepare_ready;
    assign h40_prepare_version = u_final.link_plane_to_system_srnn_h40_prepare_version;
    assign h40_prepare_done = u_final.link_system_to_plane_srnn_h40_prepare_done;
    assign h40_cache_valid = u_final.link_system_to_plane_srnn_h40_cache_valid;
    assign h40_cache_invalidate = u_final.link_plane_to_system_srnn_h40_cache_invalidate;
    assign h40_forward_read_valid = u_final.link_system_to_plane_srnn_h40_forward_read_valid;
    assign h40_forward_read_ready = u_final.link_plane_to_system_srnn_h40_forward_read_ready;
    assign h40_forward_read_addr = u_final.link_system_to_plane_srnn_h40_forward_read_addr;
    assign h40_forward_read_tag = u_final.link_system_to_plane_srnn_h40_forward_read_tag;
    assign h40_forward_response_valid = u_final.link_plane_to_system_srnn_h40_forward_response_valid;
    assign h40_forward_response_ready = u_final.link_system_to_plane_srnn_h40_forward_response_ready;
    assign h40_forward_response_tag = u_final.link_plane_to_system_srnn_h40_forward_response_tag;
    assign h40_forward_response_code = u_final.link_plane_to_system_srnn_h40_forward_response_code;
    assign h40_forward_response_error = u_final.link_plane_to_system_srnn_h40_forward_response_error;
    assign h40_timestep_start = u_final.link_plane_to_system_srnn_h40_timestep_start;
    assign h40_timestep_start_ready = u_final.link_system_to_plane_srnn_h40_timestep_start_ready;
    assign h40_start_timestep_id = u_final.link_plane_to_system_srnn_h40_timestep_id;
    assign h40_sequence_version = u_final.link_plane_to_system_srnn_h40_sequence_version;
    assign h40_include_target_path = u_final.link_plane_to_system_srnn_h40_include_target_path;
    assign h40_observation_valid = u_final.link_plane_to_system_srnn_h40_observation_valid;
    assign h40_observation_ready = u_final.link_system_to_plane_srnn_h40_observation_ready;
    assign h40_observation_path = u_final.link_plane_to_system_srnn_h40_observation_path;
    assign h40_observation_sample = u_final.link_plane_to_system_srnn_h40_observation_sample_id;
    assign h40_observation_backbone = u_final.link_plane_to_system_srnn_h40_observation_backbone_id;
    assign h40_observation_q14 = u_final.link_plane_to_system_srnn_h40_observation_q14;
    assign h40_observation_last = u_final.link_plane_to_system_srnn_h40_observation_last;
    assign h40_observation_done = u_final.link_plane_to_system_srnn_h40_observation_done;
    assign h40_hidden_valid = u_final.link_system_to_plane_srnn_h40_hidden_valid;
    assign h40_hidden_ready = u_final.link_plane_to_system_srnn_h40_hidden_ready;
    assign h40_hidden_path = u_final.link_system_to_plane_srnn_h40_hidden_path;
    assign h40_hidden_sample = u_final.link_system_to_plane_srnn_h40_hidden_sample_id;
    assign h40_hidden_id = u_final.link_system_to_plane_srnn_h40_hidden_id;
    assign h40_hidden_pre = u_final.link_system_to_plane_srnn_h40_hidden_membrane_pre_q14;
    assign h40_hidden_next = u_final.link_system_to_plane_srnn_h40_hidden_membrane_next_q14;
    assign h40_hidden_spike = u_final.link_system_to_plane_srnn_h40_hidden_spike;
    assign h40_hidden_trace = u_final.link_system_to_plane_srnn_h40_hidden_trace_q14;
    assign h40_hidden_feature = u_final.link_system_to_plane_srnn_h40_hidden_output_feature_q14;
    assign h40_hidden_last = u_final.link_system_to_plane_srnn_h40_hidden_last;
    assign h40_logit_valid = u_final.link_system_to_plane_srnn_h40_logit_valid;
    assign h40_logit_ready = u_final.link_plane_to_system_srnn_h40_logit_ready;
    assign h40_logit_sample = u_final.link_system_to_plane_srnn_h40_logit_sample_id;
    assign h40_logit_class = u_final.link_system_to_plane_srnn_h40_logit_class_id;
    assign h40_logit_q28 = u_final.link_system_to_plane_srnn_h40_logit_q28;
    assign h40_logit_last = u_final.link_system_to_plane_srnn_h40_logit_last;
    assign h40_result_timestep_id = u_final.link_system_to_plane_srnn_h40_timestep_id;
    assign h40_timestep_done = u_final.link_system_to_plane_srnn_h40_timestep_done;
    assign h40_busy = u_final.link_system_to_plane_srnn_h40_busy;
    assign h40_sticky_error = u_final.link_system_to_plane_srnn_h40_sticky_error;
    assign h40_protocol_error = u_final.link_system_to_plane_srnn_h40_protocol_error;
    assign h40_overflow = u_final.link_system_to_plane_srnn_h40_overflow;

    unified_tp_shared_forward_rmw_store #(
        .DEPTH(PARAM_DEPTH), .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH)
    ) u_store (
        .i_clk(clk), .i_rst_n(rst_n), .i_clear_error(clear_error),
        .i_load_valid(store_load_valid), .o_load_ready(store_load_ready),
        .i_load_addr(store_load_addr), .i_load_code(store_load_code),
        .i_load_residual(store_load_residual),
        .i_learning_read_valid(store_learning_read_valid),
        .o_learning_read_ready(store_learning_read_ready),
        .i_learning_read_addr(store_learning_read_addr),
        .i_learning_read_tag(store_learning_read_tag),
        .o_learning_response_valid(store_learning_response_valid),
        .i_learning_response_ready(store_learning_response_ready),
        .o_learning_response_tag(store_learning_response_tag),
        .o_learning_response_code(store_learning_response_code),
        .o_learning_response_residual(store_learning_response_residual),
        .o_learning_response_error(store_learning_response_error),
        .i_learning_write_valid(store_learning_write_valid),
        .o_learning_write_ready(store_learning_write_ready),
        .i_learning_write_addr(store_learning_write_addr),
        .i_learning_write_code(store_learning_write_code),
        .i_learning_write_residual(store_learning_write_residual),
        .i_learning_write_code_enable(store_learning_write_code_enable),
        .i_learning_write_residual_enable(store_learning_write_residual_enable),
        .i_forward_read_valid(store_forward_read_valid),
        .o_forward_read_ready(store_forward_read_ready),
        .i_forward_read_addr(store_forward_read_addr),
        .i_forward_read_tag(store_forward_read_tag),
        .o_forward_response_valid(store_forward_response_valid),
        .i_forward_response_ready(store_forward_response_ready),
        .o_forward_response_tag(store_forward_response_tag),
        .o_forward_response_code(store_forward_response_code),
        .o_forward_response_error(store_forward_response_error),
        .o_protocol_error(store_protocol_error)
    );

    function automatic logic signed [15:0] hidden_weight(
        input int source, input int neuron
    );
        if (source == 0) begin
            case (neuron)
                0: hidden_weight = 16'sd5000;
                1: hidden_weight = -16'sd1000;
                2: hidden_weight = 16'sd3000;
                default: hidden_weight = 16'sd1000;
            endcase
        end else begin
            case (neuron)
                0: hidden_weight = 16'sd5000;
                1: hidden_weight = 16'sd9000;
                2: hidden_weight = -16'sd3000;
                default: hidden_weight = 16'sd8000;
            endcase
        end
    endfunction

    function automatic logic signed [15:0] output_weight(
        input int source, input int neuron
    );
        case (source)
            0: output_weight = neuron == 0 ? 16'sd100 : -16'sd50;
            1: output_weight = neuron == 0 ? 16'sd200 : 16'sd300;
            2: output_weight = neuron == 0 ? -16'sd100 : 16'sd75;
            default: output_weight = neuron == 0 ? 16'sd25 : 16'sd125;
        endcase
    endfunction

    function automatic logic signed [15:0] fc_hidden_weight(
        input int source, input int neuron
    );
        case (source)
            0: case (neuron)
                0: fc_hidden_weight = 16'sd10;
                1: fc_hidden_weight = 16'sd2;
                2: fc_hidden_weight = -16'sd1;
                default: fc_hidden_weight = 16'sd4;
            endcase
            1: case (neuron)
                0: fc_hidden_weight = 16'sd1;
                1: fc_hidden_weight = 16'sd10;
                2: fc_hidden_weight = 16'sd3;
                default: fc_hidden_weight = -16'sd2;
            endcase
            2: case (neuron)
                0: fc_hidden_weight = 16'sd4;
                1: fc_hidden_weight = -16'sd1;
                2: fc_hidden_weight = 16'sd10;
                default: fc_hidden_weight = 16'sd2;
            endcase
            default: case (neuron)
                0: fc_hidden_weight = -16'sd2;
                1: fc_hidden_weight = 16'sd3;
                2: fc_hidden_weight = 16'sd1;
                default: fc_hidden_weight = 16'sd10;
            endcase
        endcase
    endfunction

    function automatic logic signed [15:0] fc_output_weight(
        input int source, input int neuron
    );
        case (source)
            0: fc_output_weight = neuron == 0 ? 16'sd4 : 16'sd1;
            1: fc_output_weight = neuron == 0 ? 16'sd2 : 16'sd3;
            2: fc_output_weight = neuron == 0 ? 16'sd1 : -16'sd1;
            default: fc_output_weight = neuron == 0 ? 16'sd4 : 16'sd2;
        endcase
    endfunction

    task automatic load_parameter(
        input int address, input logic signed [15:0] code
    );
        @(negedge clk);
        srnn_load_addr = PARAM_ADDR_WIDTH'(address);
        srnn_load_code = code;
        srnn_load_valid = 1'b1;
        do @(posedge clk); while (!srnn_load_ready);
        @(negedge clk);
        srnn_load_valid = 1'b0;
    endtask

    task automatic refill_srnn_backbone_cache(
        input int parameter_base,
        input int source_count,
        input int destination_count,
        input int version
    );
        int expected_rows;
        int wait_cycles;
        @(negedge clk);
        srnn_backbone_refill_base = PARAM_ADDR_WIDTH'(parameter_base);
        srnn_backbone_refill_source_count = 10'(source_count);
        srnn_backbone_refill_destination_count = 10'(destination_count);
        srnn_backbone_refill_version = 16'(version);
        srnn_backbone_refill_valid = 1'b1;
        do @(posedge clk); while (!srnn_backbone_refill_ready);
        @(negedge clk);
        srnn_backbone_refill_valid = 1'b0;
        wait_cycles = 0;
        while (!srnn_backbone_refill_done && wait_cycles < 8000) begin
            @(posedge clk);
            wait_cycles++;
        end
        expected_rows = source_count *
            ((destination_count / 40) * 2 +
             (destination_count % 40 == 0 ? 0 :
              (destination_count % 40 <= 20 ? 1 : 2)));
        if (!srnn_backbone_refill_done ||
                srnn_backbone_refill_rows != 32'(expected_rows) ||
                srnn_backbone_published_version != 16'(version) ||
                u_final.u_system.g_srnn_backbone_refill.u_srnn_backbone_cache_refill.o_sticky_error) begin
            $fatal(1,
                "SRNN backbone cache refill failed rows=%0d/%0d version=%0d/%0d error=%0b",
                srnn_backbone_refill_rows, expected_rows,
                srnn_backbone_published_version, version,
                u_final.u_system.g_srnn_backbone_refill.u_srnn_backbone_cache_refill.o_sticky_error);
        end
        @(negedge clk);
    endtask

    task automatic load_fc_parameter(
        input int address, input logic signed [15:0] code
    );
        @(negedge clk);
        fc_load_addr = PARAM_ADDR_WIDTH'(address);
        fc_load_code = code;
        fc_load_valid = 1'b1;
        do @(posedge clk); while (!fc_load_ready);
        @(negedge clk);
        fc_load_valid = 1'b0;
    endtask

    task automatic clear_fc_state;
        @(negedge clk);
        maintenance_tile = 4'd0;
        state_write_addr = '0;
        state_write_data = '0;
        state_write_mask = '1;
        trace_write_addr = 10'h200;
        trace_write_data = '0;
        trace_write_mask = '1;
        state_write_valid = 1'b1;
        trace_write_valid = 1'b1;
        do @(posedge clk); while (!state_write_ready || !trace_write_ready);
        @(negedge clk);
        state_write_valid = 1'b0;
        trace_write_valid = 1'b0;
    endtask

    task automatic refill_fc_cache(
        input int parameter_base,
        input int stride,
        input int feature_count,
        input int neuron_count,
        input int version
    );
        int wait_cycles;
        @(negedge clk);
        fc_refill_base = PARAM_ADDR_WIDTH'(parameter_base);
        fc_refill_stride = PARAM_ADDR_WIDTH'(stride);
        fc_refill_feature_count = 5'(feature_count);
        fc_refill_neuron_count = 10'(neuron_count);
        fc_refill_tile_count = 5'd1;
        fc_refill_version = 16'(version);
        fc_refill_valid = 1'b1;
        do @(posedge clk); while (!fc_refill_ready);
        @(negedge clk);
        fc_refill_valid = 1'b0;
        wait_cycles = 0;
        while (!fc_refill_done && wait_cycles < 4000) begin
            @(posedge clk);
            wait_cycles++;
        end
        if (!fc_refill_done || fc_refill_rows != 32'(feature_count) ||
                fc_refill_published_version != 16'(version) ||
                u_final.u_system.o_cache_refill_sticky_error) begin
            $display("FC_REFILL_DEBUG outer_ready/done/rows/version/error=%0b/%0b/%0d/%0d/%0b busy/q=%0b/%0b owner=%b learn=%b array_idle=%0b lease=%0b maint=%0b store(fv/fr/rv/rr/err)=%0b/%0b/%0b/%0b/%0b backend=%0b scheduler(state/legal/row_busy/row_error/feature/tile)=%0d/%0b/%0b/%0b/%0d/%0d publisher(state/error)=%0d/%0b",
                fc_refill_ready, fc_refill_done, fc_refill_rows,
                fc_refill_published_version,
                u_final.u_system.o_cache_refill_sticky_error,
                u_final.u_system.o_cache_refill_busy,
                u_final.u_system.o_cache_refill_quiescent,
                parameter_owner, topology_learning_enable,
                u_final.u_system.array_idle, u_final.u_system.maintenance_lease_q,
                u_final.u_system.maintenance_request,
                store_forward_read_valid, store_forward_read_ready,
                store_forward_response_valid, store_forward_response_ready,
                store_forward_response_error, backend_error,
                u_final.u_system.u_shared_weight_cache_refill.g_fc_refill.u_fc_refill.state_q,
                u_final.u_system.u_shared_weight_cache_refill.g_fc_refill.u_fc_refill.request_legal,
                u_final.u_system.u_shared_weight_cache_refill.g_fc_refill.u_fc_refill.row_busy,
                u_final.u_system.u_shared_weight_cache_refill.g_fc_refill.u_fc_refill.row_error,
                u_final.u_system.u_shared_weight_cache_refill.g_fc_refill.u_fc_refill.feature_q,
                u_final.u_system.u_shared_weight_cache_refill.g_fc_refill.u_fc_refill.tile_q,
                u_final.u_system.u_shared_weight_cache_refill.g_fc_refill.u_fc_refill.u_row_publisher.state_q,
                u_final.u_system.u_shared_weight_cache_refill.g_fc_refill.u_fc_refill.u_row_publisher.o_sticky_error);
            $fatal(1, "FC cache refill failed base=%0d rows=%0d version=%0d/%0d",
                parameter_base, fc_refill_rows,
                fc_refill_published_version, version);
        end
        @(negedge clk);
    endtask

    task automatic configure_fc_capture(
        input int sample_id,
        input int neuron_count,
        input logic output_layer,
        input logic [3:0] label
    );
        int wait_cycles;
        @(negedge clk);
        fc_srnn_transaction_tag = output_layer ? FC_OUTPUT_TAG : FC_HIDDEN_TAG;
        fc_learning_cfg_mode = 2'd0;
        fc_learning_srnn_path = 1'b0;
        fc_learning_sample_id = 6'(sample_id);
        fc_learning_neuron_count = 10'(neuron_count);
        fc_learning_output_layer = output_layer;
        fc_learning_label = label;
        fc_learning_cfg_valid = 1'b1;
        wait_cycles = 0;
        while (!fc_learning_cfg_ready && wait_cycles < 1000) begin
            @(posedge clk);
            wait_cycles++;
        end
        if (!fc_learning_cfg_ready)
            $fatal(1, "FC observation config timeout sample=%0d output=%0b",
                sample_id, output_layer);
        @(negedge clk);
        fc_learning_cfg_valid = 1'b0;
    endtask

    task automatic configure_srnn_backbone_capture;
        int wait_cycles;
        begin
            @(negedge clk);
            fc_srnn_transaction_tag = 52'h5a00000000001;
            fc_learning_sample_id = 6'd0;
            fc_learning_neuron_count = 10'd2;
            fc_learning_cfg_mode = 2'd1;
            fc_learning_srnn_path = 1'b0;
            fc_learning_output_layer = 1'b0;
            fc_learning_label = '0;
            fc_learning_cfg_valid = 1'b1;
            wait_cycles = 0;
            while (!fc_learning_cfg_ready && wait_cycles < 1000) begin
                @(posedge clk);
                wait_cycles++;
            end
            if (!fc_learning_cfg_ready)
                $fatal(1, "SRNN physical backbone capture config timeout");
            @(negedge clk);
            fc_learning_cfg_valid = 1'b0;
        end
    endtask

    task automatic refill_and_capture_srnn_backbone;
        int done_base;
        int capture_base;
        int wait_cycles;
        begin
            wait (srnn_actual_tape_cfg_ready);
            @(negedge clk);
            srnn_actual_tape_cfg_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            srnn_actual_tape_cfg_valid = 1'b0;
            refill_srnn_backbone_cache(0, 2, 2, 16'h5a00 + phase);
            clear_fc_state();
            configure_srnn_backbone_capture();
            done_base = fc_layer_done_count;
            capture_base = fc_capture_done_count;
            @(negedge clk);
            fc_srnn_mode = 2'd1;
            fc_srnn_neuron_count = 16'd2;
            fc_srnn_feedforward_group_count = 16'd1;
            fc_srnn_group_source_valid = 4'b0011;
            fc_srnn_layer_valid = 1'b1;
            wait_cycles = 0;
            while (!fc_srnn_layer_ready && wait_cycles < 1000) begin
                @(posedge clk);
                wait_cycles++;
            end
            if (!fc_srnn_layer_ready)
                $fatal(1, "SRNN physical backbone layer admission timeout");
            @(negedge clk);
            fc_srnn_layer_valid = 1'b0;
            fc_srnn_group_valid = 1'b1;
            wait_cycles = 0;
            while (!fc_srnn_group_ready && wait_cycles < 1000) begin
                @(posedge clk);
                wait_cycles++;
            end
            if (!fc_srnn_group_ready)
                $fatal(1, "SRNN physical backbone group admission timeout");
            @(negedge clk);
            fc_srnn_group_valid = 1'b0;
            wait_cycles = 0;
            while ((!srnn_actual_tape_capture_complete ||
                    fc_layer_done_count == done_base) &&
                    wait_cycles < 10000) begin
                @(posedge clk);
                wait_cycles++;
            end
            if (!srnn_actual_tape_capture_complete ||
                    fc_layer_done_count != done_base + 1 ||
                    fc_capture_done_count != capture_base + 1 ||
                    srnn_actual_tape_error || fc_srnn_error ||
                    fc_srnn_observation_error || fc_learning_error)
                $fatal(1, "SRNN physical backbone/tape capture failed layer=%0d/%0d ingress=%0d/%0d tape=%0b error=%0b/%0b/%0b/%0b",
                    fc_layer_done_count, done_base,
                    fc_capture_done_count, capture_base + 1,
                    srnn_actual_tape_capture_complete,
                    srnn_actual_tape_error,
                    fc_srnn_error, fc_srnn_observation_error,
                    fc_learning_error);
            fc_srnn_mode = 2'd0;
            @(negedge clk);
        end
    endtask

    task automatic run_fc_layer(
        input int sample_id,
        input int neuron_count,
        input logic output_layer,
        input logic [3:0] label,
        input logic [3:0] source_mask
    );
        int done_base;
        int capture_base;
        int wait_cycles;
        begin
            if (!output_layer)
                fc_hidden_spike_mask = '0;
            fc_srnn_mode = 2'd0;
            configure_fc_capture(sample_id, neuron_count, output_layer, label);
            done_base = fc_layer_done_count;
            capture_base = fc_capture_done_count;
            @(negedge clk);
            fc_srnn_neuron_count = 16'(neuron_count);
            fc_srnn_group_source_valid = source_mask;
            fc_srnn_layer_valid = 1'b1;
            wait_cycles = 0;
            while (!fc_srnn_layer_ready && wait_cycles < 1000) begin
                @(posedge clk);
                wait_cycles++;
            end
            if (!fc_srnn_layer_ready)
                $fatal(1, "FC layer admission timeout sample=%0d output=%0b",
                    sample_id, output_layer);
            @(negedge clk);
            fc_srnn_layer_valid = 1'b0;
            fc_srnn_group_valid = 1'b1;
            wait_cycles = 0;
            while (!fc_srnn_group_ready && wait_cycles < 1000) begin
                @(posedge clk);
                wait_cycles++;
            end
            if (!fc_srnn_group_ready)
                $fatal(1, "FC group admission timeout sample=%0d output=%0b",
                    sample_id, output_layer);
            @(negedge clk);
            fc_srnn_group_valid = 1'b0;
            wait_cycles = 0;
            while ((fc_layer_done_count == done_base ||
                    fc_capture_done_count == capture_base) &&
                    wait_cycles < 10000) begin
                @(posedge clk);
                wait_cycles++;
            end
            if (fc_layer_done_count != done_base + 1 ||
                    fc_capture_done_count != capture_base + 1 ||
                    fc_srnn_error || fc_srnn_observation_error ||
                    fc_learning_error)
                $fatal(1, "FC physical layer failed sample=%0d output=%0b done=%0d/%0d capture=%0d/%0d error=%0b/%0b/%0b code=%h",
                    sample_id, output_layer,
                    fc_layer_done_count, done_base + 1,
                    fc_capture_done_count, capture_base + 1,
                    fc_srnn_error, fc_srnn_observation_error,
                    fc_learning_error, fc_srnn_error_code);
            @(negedge clk);
        end
    endtask

    task automatic send_observation(
        input bit path,
        input int backbone,
        input logic signed [31:0] q14,
        input bit last
    );
        @(negedge clk);
        srnn_backbone_path = path;
        srnn_backbone_id = 10'(backbone);
        srnn_backbone_q14 = q14;
        srnn_backbone_last = last;
        srnn_backbone_valid = 1'b1;
        do @(posedge clk); while (!srnn_target_fixture_ready);
        @(negedge clk);
        srnn_backbone_valid = 1'b0;
    endtask

    task automatic send_target_trace(input int backbone, input bit last);
        @(negedge clk);
        srnn_target_trace_sample = '0;
        srnn_target_trace_backbone = 10'(backbone);
        srnn_target_trace_q14 = backbone == 0 ? 32'sd0 : 32'sd16384;
        srnn_target_trace_last = last;
        srnn_target_trace_valid = 1'b1;
        do @(posedge clk); while (!srnn_target_trace_ready);
        @(negedge clk);
        srnn_target_trace_valid = 1'b0;
    endtask

    task automatic run_timestep(input bit learning);
        refill_and_capture_srnn_backbone();
        wait (srnn_actual_tape_replay_start_ready);
        @(negedge clk);
        srnn_actual_tape_replay_start = 1'b1;
        @(posedge clk);
        @(negedge clk);
        srnn_actual_tape_replay_start = 1'b0;

        if (learning) begin
            fork
                begin
                    wait (srnn_label_replay_start);
                    @(negedge clk);
                    srnn_label_sample = '0;
                    srnn_label_class = 5'd1;
                    srnn_label_last = 1'b1;
                    srnn_label_valid = 1'b1;
                    do @(posedge clk); while (!srnn_label_ready);
                    @(negedge clk);
                    srnn_label_valid = 1'b0;
                end
                begin
                    if (ENABLE_SRNN_WH1_LEARNING) begin
                        wait (srnn_target_trace_start);
                        send_target_trace(0, 1'b0);
                        send_target_trace(1, 1'b1);
                        @(negedge clk);
                        srnn_target_trace_done = 1'b1;
                        @(posedge clk);
                        @(negedge clk);
                        srnn_target_trace_done = 1'b0;
                        srnn_target_trace_last = 1'b0;
                    end
                end
                begin
                    if (ENABLE_SRNN_WH1_LEARNING) begin
                        wait (srnn_target_fixture_ready);
                        send_observation(1'b1, 0, 32'sd0, 1'b0);
                        send_observation(1'b1, 1, 32'sd16384, 1'b1);
                        @(negedge clk);
                        srnn_backbone_done = 1'b1;
                        @(posedge clk);
                        @(negedge clk);
                        srnn_backbone_done = 1'b0;
                        srnn_backbone_path = 1'b0;
                        srnn_backbone_last = 1'b0;
                    end
                end
            join
        end
        wait (srnn_done);
        @(negedge clk);
    endtask

    always_ff @(posedge clk) begin
        debug_adapter_state_q <=
            u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_adapter.state_q;
        if (rst_n && debug_adapter_state_q != 5'd15 &&
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_adapter.state_q == 5'd15) begin
            $display("SRNN_ADAPTER_ERROR_EDGE prev=%0d layer_error=%0b obs_valid/fire/legal=%0b/%0b/%0b path=%0b/%0b sample=%0d/%0d backbone=%0d/%0d last=%0b/%0b done=%0b array_valid/fire/legal=%0b/%0b/%0b mode=%0d tag=%h/%h neuron=%0d received=%0b include_target=%0b",
                debug_adapter_state_q,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.i_layer_sticky_error,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.i_observation_valid,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_adapter.observation_fire,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_adapter.current_input_legal,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.i_observation_path,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_adapter.expected_path_q,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.i_observation_sample_id,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_adapter.expected_sample_q,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.i_observation_backbone_id,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_adapter.expected_backbone_q,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.i_observation_last,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_adapter.expected_input_last,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.i_observation_done,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.i_array_observation_valid,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_adapter.array_observation_fire,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_adapter.array_observation_legal,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.i_array_observation_mode,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.i_array_observation_transaction_tag,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_adapter.active_tag_q,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.i_array_observation_neuron_id,
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_adapter.received_neuron_q[int'(u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.i_array_observation_neuron_id)],
                u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_adapter.include_target_q);
        end
        if (rst_n && store_forward_read_valid && store_forward_read_ready)
            forward_reads <= forward_reads + 1;
        if (rst_n && parameter_owner == 3'b001 &&
                store_forward_read_valid && store_forward_read_ready)
            fc_refill_forward_reads <= fc_refill_forward_reads + 1;
        if (rst_n && store_learning_write_valid && store_learning_write_ready) begin
            learning_writes <= learning_writes + 1;
            if (store_learning_write_code_enable)
                learning_code_writes <= learning_code_writes + 1;
            if (store_learning_write_residual_enable)
                learning_residual_writes <= learning_residual_writes + 1;
            if (parameter_owner == 3'b010 &&
                    store_learning_write_addr < PARAM_ADDR_WIDTH'(WOUT_BASE))
                srnn_wh1_writes <= srnn_wh1_writes + 1;
            if (parameter_owner == 3'b010 &&
                    store_learning_write_addr >= PARAM_ADDR_WIDTH'(WOUT_BASE) &&
                    store_learning_write_addr < PARAM_ADDR_WIDTH'(FC_W0_BASE))
                srnn_wout_writes <= srnn_wout_writes + 1;
        end
        if (rst_n && fc_window_commit_valid)
            fc_commit_count <= fc_commit_count + 1;
        if (rst_n && fc_srnn_layer_done)
            fc_layer_done_count <= fc_layer_done_count + 1;
        if (rst_n && fc_capture_done)
            fc_capture_done_count <= fc_capture_done_count + 1;
        if (rst_n && fc_student_trace_valid && fc_student_trace_ready)
            fc_trace_count <= fc_trace_count + 1;
        if (rst_n && fc_logit_valid && fc_logit_ready)
            fc_logit_count <= fc_logit_count + 1;
        if (rst_n && fc_srnn_observation_valid &&
                fc_srnn_observation_tag == FC_HIDDEN_TAG &&
                fc_srnn_observation_neuron < 10'd4)
            fc_hidden_spike_mask[fc_srnn_observation_neuron[1:0]] <=
                fc_srnn_observation_spike;
        if (rst_n && fc_srnn_observation_valid &&
                fc_srnn_observation_tag == FC_OUTPUT_TAG &&
                fc_srnn_observation_neuron < 10'd2) begin
            if (fc_phase == 1)
                fc_baseline_output[fc_srnn_observation_neuron[0]] <=
                    fc_srnn_observation_pre_state;
            if (fc_phase == 2)
                fc_postlearn_output[fc_srnn_observation_neuron[0]] <=
                    fc_srnn_observation_pre_state;
        end
        if (rst_n && srnn_learning_commit)
            commit_count <= commit_count + 1;
        if (rst_n && srnn_detach)
            detach_count <= detach_count + 1;
        if (rst_n && h40_cache_invalidate)
            cache_invalidations <= cache_invalidations + 1;

        if (rst_n && h40_hidden_valid && h40_hidden_ready) begin
            logic signed [31:0] expected_pre;
            logic expected_spike;
            int phase_hidden_index;
            phase_hidden_index = hidden_seen - phase_hidden_base;
            case (phase_hidden_index)
                0: expected_pre = 32'sd10000;
                1: expected_pre = 32'sd8000;
                2: expected_pre = 32'sd0;
                default: expected_pre = 32'sd9000;
            endcase
            expected_spike = expected_pre >= 32'sd8192;
            if (phase == 0 && (h40_hidden_path || h40_hidden_sample != 0 ||
                    h40_hidden_id != 9'(phase_hidden_index) ||
                    h40_hidden_pre != expected_pre ||
                    h40_hidden_next != expected_pre -
                        (expected_spike ? 32'sd8192 : 32'sd0) ||
                    h40_hidden_spike != expected_spike ||
                    h40_hidden_trace !=
                        (expected_spike ? 32'sd16384 : 32'sd0) ||
                    h40_hidden_feature !=
                        (expected_spike ? 32'sd16384 : 32'sd0) ||
                    h40_hidden_last != (phase_hidden_index == 3)))
                $fatal(1, "shared C17 H40 hidden mismatch index=%0d path=%0b sample=%0d id=%0d pre=%0d next=%0d spike=%0b trace=%0d feature=%0d last=%0b expected_pre=%0d",
                    phase_hidden_index, h40_hidden_path,
                    h40_hidden_sample, h40_hidden_id, h40_hidden_pre,
                    h40_hidden_next, h40_hidden_spike, h40_hidden_trace,
                    h40_hidden_feature, h40_hidden_last, expected_pre);
            if (phase == 2 && (h40_hidden_path || h40_hidden_sample != 0 ||
                    h40_hidden_id != 9'(phase_hidden_index) ||
                    h40_hidden_last != (phase_hidden_index == 3)))
                $fatal(1, "post-learning hidden metadata mismatch index=%0d",
                    phase_hidden_index);
            hidden_seen <= hidden_seen + 1;
        end

        if (rst_n && srnn_logit_valid) begin
            logic signed [63:0] expected_logit;
            int phase_logit_index;
            phase_logit_index = logit_seen - phase_logit_base;
            expected_logit = phase_logit_index == 0 ?
                (64'sd125 <<< 14) : (64'sd75 <<< 14);
            if (phase == 0 && (srnn_logit_sample != 0 ||
                    srnn_logit_class != 5'(phase_logit_index) ||
                    srnn_logit_q28 != expected_logit ||
                    srnn_logit_last != (phase_logit_index == 1)))
                $fatal(1, "shared C17 H40 logit mismatch index=%0d",
                    phase_logit_index);
            if (phase == 0 && phase_logit_index < 2)
                baseline_logits[phase_logit_index] <= srnn_logit_q28;
            if (phase == 2 && phase_logit_index < 2) begin
                if (srnn_logit_sample != 0 ||
                        srnn_logit_class != 5'(phase_logit_index) ||
                        srnn_logit_last != (phase_logit_index == 1))
                    $fatal(1, "post-learning logit metadata mismatch index=%0d",
                        phase_logit_index);
                postlearn_logits[phase_logit_index] <= srnn_logit_q28;
            end
            logit_seen <= logit_seen + 1;
        end
    end

    initial begin
        int fc_wait;
        logic [3:0] fc_sample0_hidden_mask;
        logic [3:0] fc_sample1_hidden_mask;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        wait (srnn_cfg_ready);
        @(negedge clk);
        srnn_cfg_valid = 1'b1;
        @(posedge clk);
        @(negedge clk);
        srnn_cfg_valid = 1'b0;
        if (srnn_cfg_error || h40_cfg_error)
            $fatal(1, "shared SRNN configuration failed");

        for (int neuron = 0; neuron < 4; neuron++) begin
            for (int source = 0; source < 2; source++)
                load_parameter(WH1_BASE + neuron*2 + source,
                    hidden_weight(source, neuron));
        end
        for (int neuron = 0; neuron < 2; neuron++) begin
            for (int source = 0; source < 4; source++)
                load_parameter(WOUT_BASE + neuron*4 + source,
                    output_weight(source, neuron));
        end

        run_timestep(1'b0);
        if (hidden_seen != 4 || logit_seen != 2 || forward_reads != 16 ||
                srnn_busy || srnn_error || h40_sticky_error ||
                h40_protocol_error || h40_overflow || backend_error ||
                store_protocol_error || array_tile_error != 0)
            $fatal(1, "shared C17 H40 baseline inference failed");

        phase = 1;
        phase_hidden_base = hidden_seen;
        phase_logit_base = logit_seen;
        cache_invalidations_before_learning = cache_invalidations;
        topology_learning_enable = 3'b010;
        srnn_learning_enable = 1'b1;
        run_timestep(1'b1);
        if (commit_count != 1 || detach_count != 1 || srnn_updates == 0 ||
                learning_writes == 0 ||
                cache_invalidations <= cache_invalidations_before_learning ||
                srnn_busy || srnn_error || srnn_aborted ||
                h40_sticky_error || h40_protocol_error || h40_overflow ||
                backend_error || store_protocol_error || array_tile_error != 0)
            $fatal(1, "shared C17 H40 learning/commit phase failed commits=%0d detach=%0d updates=%0d writes=%0d invalidations=%0d/%0d",
                commit_count, detach_count, srnn_updates, learning_writes,
                cache_invalidations, cache_invalidations_before_learning);
        if (!ENABLE_SRNN_WH1_LEARNING &&
                (srnn_wh1_writes != 0 || srnn_wout_writes == 0))
            $fatal(1, "SRNN Wout-only production mode wrote the wrong tensor wh1=%0d wout=%0d",
                srnn_wh1_writes, srnn_wout_writes);

        phase = 2;
        phase_hidden_base = hidden_seen;
        phase_logit_base = logit_seen;
        topology_learning_enable = 3'b000;
        srnn_learning_enable = 1'b0;
        run_timestep(1'b0);
        if (hidden_seen - phase_hidden_base != 4 ||
                logit_seen - phase_logit_base != 2 ||
                // Each of the three backbone phases evicts the M2 cache.
                forward_reads != 48 || commit_count != 1 ||
                detach_count != 1 || srnn_busy || srnn_error || srnn_aborted ||
                h40_sticky_error || h40_protocol_error || h40_overflow ||
                backend_error || store_protocol_error || array_tile_error != 0)
            $fatal(1, "post-learning forward visibility phase failed reads=%0d hidden=%0d logits=%0d",
                forward_reads, hidden_seen - phase_hidden_base,
                logit_seen - phase_logit_base);
        if (postlearn_logits[0] == baseline_logits[0] &&
                postlearn_logits[1] == baseline_logits[1])
            $fatal(1, "committed SRNN update was not visible to the next physical forward");

        // Switch from SRNN to FC without reset. W0 and W2 are successively
        // packed into the same tile-0 H40 cache; there is no FC-only array.
        parameter_owner = 3'b001;
        topology_learning_enable = 3'b001;
        fc_learning_write_base = learning_writes;
        fc_learning_code_write_base = learning_code_writes;
        fc_learning_residual_write_base = learning_residual_writes;
        repeat (2) @(posedge clk);
        wait (fc_cfg_ready);
        @(negedge clk);
        fc_cfg_valid = 1'b1;
        @(posedge clk);
        @(negedge clk);
        fc_cfg_valid = 1'b0;
        if (fc_cfg_error || fc_error)
            $fatal(1, "FC W2 configuration failed");

        for (int source = 0; source < 4; source++) begin
            for (int neuron = 0; neuron < 4; neuron++)
                load_fc_parameter(FC_W0_BASE + source*4 + neuron,
                    fc_hidden_weight(source, neuron));
            for (int neuron = 0; neuron < 2; neuron++)
                load_fc_parameter(FC_W2_BASE + source*2 + neuron,
                    fc_output_weight(source, neuron));
        end

        refill_fc_cache(FC_W0_BASE, 4, 4, 4, 1);
        clear_fc_state();
        run_fc_layer(0, 4, 1'b0, 4'd1, 4'b1111);
        fc_sample0_hidden_mask = fc_hidden_spike_mask;
        if (fc_sample0_hidden_mask == 0)
            $fatal(1, "FC sample0 hidden layer produced no physical spikes");

        refill_fc_cache(FC_W2_BASE, 2, 4, 2, 2);
        clear_fc_state();
        fc_phase = 1;
        run_fc_layer(0, 2, 1'b1, 4'd1, fc_sample0_hidden_mask);
        fc_phase = 0;

        refill_fc_cache(FC_W0_BASE, 4, 4, 4, 3);
        clear_fc_state();
        run_fc_layer(1, 4, 1'b0, 4'd0, 4'b0101);
        fc_sample1_hidden_mask = fc_hidden_spike_mask;
        if (fc_sample1_hidden_mask == 0 ||
                fc_sample1_hidden_mask == fc_sample0_hidden_mask)
            $fatal(1, "FC physical hidden patterns did not distinguish samples mask0=%b mask1=%b",
                fc_sample0_hidden_mask, fc_sample1_hidden_mask);

        refill_fc_cache(FC_W2_BASE, 2, 4, 2, 4);
        clear_fc_state();
        run_fc_layer(1, 2, 1'b1, 4'd0, fc_sample1_hidden_mask);

        fc_wait = 0;
        while (fc_commit_count == 0 && fc_wait < 20000) begin
            @(posedge clk);
            fc_wait++;
        end
        if (fc_commit_count != 1 ||
                learning_writes == fc_learning_write_base ||
                learning_residual_writes == fc_learning_residual_write_base ||
                fc_trace_count != 8 || fc_logit_count != 4 ||
                fc_window_busy || fc_error || backend_error ||
                store_protocol_error || array_tile_error != 0)
            $fatal(1, "FC physical magnitude learning closure failed commit=%0d writes=%0d/%0d residual=%0d/%0d traces=%0d logits=%0d busy/error=%0b/%0b",
                fc_commit_count, learning_writes, fc_learning_write_base,
                learning_residual_writes, fc_learning_residual_write_base,
                fc_trace_count, fc_logit_count, fc_window_busy, fc_error);

        topology_learning_enable = 3'b000;
        refill_fc_cache(FC_W2_BASE, 2, 4, 2, 5);
        clear_fc_state();
        fc_phase = 2;
        run_fc_layer(0, 2, 1'b1, 4'd1, fc_sample0_hidden_mask);
        fc_phase = 0;
        if (learning_code_writes > fc_learning_code_write_base &&
                fc_postlearn_output[0] == fc_baseline_output[0] &&
                fc_postlearn_output[1] == fc_baseline_output[1])
            $fatal(1, "FC magnitude code update was not visible to the next physical H40 forward");
        if (learning_code_writes == fc_learning_code_write_base &&
                (fc_postlearn_output[0] != fc_baseline_output[0] ||
                 fc_postlearn_output[1] != fc_baseline_output[1]))
            $fatal(1, "FC quantized forward changed without a code write");
        if (fc_refill_forward_reads != 56 || fc_commit_count != 1 ||
                fc_srnn_busy || fc_srnn_error || fc_srnn_observation_error ||
                fc_learning_error || u_final.u_system.o_cache_refill_sticky_error ||
                backend_error || store_protocol_error || array_tile_error != 0)
            $fatal(1, "FC post-learning physical visibility failed reads=%0d commit=%0d",
                fc_refill_forward_reads, fc_commit_count);

        $display("PASS: C17 FC W2 and SRNN M2 share one scalar store and one production 12xH40 array without reset, with the production C17 SCNN client wired to router client2; SRNN baseline->learn->refill->changed forward and FC UQ6.10 multiplier-11 magnitude W2 learn/commit/refill all closed with residual-preserving visibility; srnn_wh1_enabled=%0b wh1_writes=%0d wout_writes=%0d total_reads=%0d total_writes=%0d total_code_writes=%0d total_residual_writes=%0d fc_code_writes=%0d fc_residual_writes=%0d fc_reads=%0d fc_commits=%0d fc_traces=%0d fc_logits=%0d",
            ENABLE_SRNN_WH1_LEARNING, srnn_wh1_writes, srnn_wout_writes,
            forward_reads, learning_writes, learning_code_writes,
            learning_residual_writes,
            learning_code_writes - fc_learning_code_write_base,
            learning_residual_writes - fc_learning_residual_write_base,
            fc_refill_forward_reads, fc_commit_count, fc_trace_count,
            fc_logit_count);
        $finish;
    end

    initial begin
        repeat (20000) @(posedge clk);
        $display("debug prep=%0d refill_state=%0d row_state=%0d dispatch_active=%0b dispatch_complete=%0b refill_busy=%0b refill_error=%0b row_busy=%0b row_error=%0b",
            u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.prep_state_q,
            u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_refill.state_q,
            u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_refill.u_row_publisher.state_q,
            u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_refill.u_row_publisher.u_refill_dispatcher.active_q,
            u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_refill.u_row_publisher.u_refill_dispatcher.complete_valid_q,
            u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.refill_busy,
            u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.refill_error,
            u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_refill.row_busy,
            u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_refill.row_error);
        $display("debug array_idle=%0b lease=%0b owner_m2=%0b maint_req=%0b fill=%b/%b m2_fill=%b/%b topo_grant=%0d shared_grant=%0d array_owner=%0b sticky=%0b/%0b/%0b",
            u_final.u_system.array_idle, u_final.u_system.maintenance_lease_q,
            u_final.u_system.maintenance_owner_m2_q, u_final.u_system.maintenance_request,
            u_final.u_system.weight_fill_valid, u_final.u_system.weight_fill_ready,
            u_final.u_system.m2_weight_fill_valid, u_final.u_system.m2_weight_fill_ready,
            u_final.u_system.u_topology_weight_fill_arbiter.grant_q,
            u_final.u_system.u_shared_weight_cache_refill.u_fill_arbiter.grant_q,
            u_final.u_system.m2_array_owner,
            u_final.u_system.topology_weight_fill_error,
            u_final.u_system.o_cache_refill_sticky_error,
            h40_sticky_error);
        $display("debug coord_state=%0d learn=%0b pair_ready=%0b label_start=%0b label=%0b/%0b target_start=%0b target=%0b/%0b target_done=%0b obs=%0b/%0b obs_done=%0b done=%0b",
            u_final.u_parameter_plane.g_srnn_client.u_srnn_m2.state_q,
            u_final.u_parameter_plane.g_srnn_client.u_srnn_m2.learning_enable_q,
            u_final.u_parameter_plane.g_srnn_client.u_srnn_m2.learning_pair_fire,
            srnn_label_replay_start, srnn_label_valid, srnn_label_ready,
            srnn_target_trace_start, srnn_target_trace_valid,
            srnn_target_trace_ready, srnn_target_trace_done,
            srnn_backbone_valid, srnn_backbone_ready,
            srnn_backbone_done, srnn_done);
        $display("debug adapter_state=%0d clear_req=%0b clear_done=%0b clear_state=%0d clear_busy=%0b layer=%0b/%0b group=%0b/%0b arr_obs=%0b/%0b adapter_busy=%0b",
            u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_adapter.state_q,
            u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.state_clear_required,
            u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.state_clear_complete,
            u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.u_state_clear.state_q,
            u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.clear_busy,
            u_final.u_system.m2_layer_valid, u_final.u_system.m2_layer_ready,
            u_final.u_system.m2_group_valid, u_final.u_system.m2_group_ready,
            u_final.u_system.array_fc_srnn_observation_valid,
            u_final.u_system.m2_array_observation_ready,
            u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward.adapter_busy);
        $display("debug owner_state=%0d direct_last=%0b direct_done=%0b select=%0b client_state=%0d issued=%0d/%0d serializer_busy=%0b sink_idle=%0b array_core_idle=%0b direct_done_pulse=%0b",
            u_final.u_system.u_array.u_owner.owner_q,
            u_final.u_system.u_array.u_owner.direct_last_inflight_q,
            u_final.u_system.u_array.u_owner.direct_done_seen_q,
            u_final.u_system.u_array.select_direct,
            u_final.u_system.u_array.g_fc_srnn_layer.u_fc_srnn_client.state_q,
            u_final.u_system.u_array.g_fc_srnn_layer.u_fc_srnn_client.issued_group_count_q,
            u_final.u_system.u_array.g_fc_srnn_layer.u_fc_srnn_client.total_group_count_q,
            u_final.u_system.u_array.fc_srnn_observation_busy,
            u_final.u_system.u_array.fc_srnn_direct_sink_idle,
            u_final.u_system.u_array.array_idle,
            u_final.u_system.u_array.direct_transaction_done);
        $fatal(1, "timeout cfg=%0b start=%0b prepare=%0b cache=%0b busy=%0b hidden=%0d logits=%0d reads=%0d",
            srnn_cfg_ready, srnn_start_ready, h40_prepare_valid,
            h40_cache_valid, h40_busy, hidden_seen, logit_seen, forward_reads);
    end
endmodule

`default_nettype wire
