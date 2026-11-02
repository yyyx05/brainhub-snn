`timescale 1ns/1ps
`default_nettype none

module unified_tp_actual_learning_subsystem_v1_shared_fc_tb;
    localparam int LANES = 2;
    localparam int HIDDEN = 2;
    localparam int FEATURES = 2;
    localparam int OUTPUTS = 10;
    localparam int BATCH = 16;

    logic clk;
    logic rst_n = 1'b0;

    logic profile_request_valid = 1'b0;
    logic [6:0] profile_request_code = '0;
    logic profile_learning_enable = 1'b0;
    logic profile_switch_done;
    logic [6:0] active_profile_code;
    logic [2:0] parameter_owner;
    logic system_error;
    logic [7:0] control_error_code;

    logic run_valid = 1'b0;
    logic run_ready;
    logic run_done;
    logic run_busy;

    logic core_start;
    logic core_sample_done = 1'b0;
    logic core_model_busy_q = 1'b0;

    logic fc_cfg_valid = 1'b0;
    logic fc_cfg_ready;
    logic fc_configured;
    logic fc_cfg_error;
    logic fc_window_valid = 1'b0;
    logic fc_window_ready;
    logic [LANES-1:0] fc_record_valid = '0;
    logic [15:0] fc_window_sample_id = '0;
    logic [8:0] fc_window_timestep = '0;
    logic [LANES-1:0] fc_boundary_mask = '0;
    logic fc_window_sample_last = 1'b0;
    logic [LANES*12-1:0] fc_addr_flat = '0;
    logic [LANES*8-1:0] fc_data_flat = '0;
    logic [3:0] fc_snapshot_label = '0;
    logic [HIDDEN-1:0] fc_snapshot_spikes = '0;
    logic [HIDDEN*16-1:0] fc_snapshot_trace = '0;
    logic [HIDDEN*18-1:0] fc_snapshot_vmem = '0;
    logic [OUTPUTS*24-1:0] fc_snapshot_logits = '0;

    logic target_hidden_valid = 1'b0;
    logic target_hidden_ready;
    logic [3:0] target_trace_sample = '0;
    logic [8:0] target_hidden_index = '0;
    logic [16:0] target_hidden_trace = '0;
    logic target_input_valid = 1'b0;
    logic target_input_ready;
    logic [3:0] target_input_index = '0;
    logic [19:0] target_input_trace = '0;
    logic target_done = 1'b0;
    logic target_done_ready;

    logic fc_event_window_valid;
    logic [LANES-1:0] fc_event_record_valid;
    logic [15:0] fc_event_window_sample_id;
    logic [8:0] fc_event_window_timestep;
    logic [LANES-1:0] fc_event_boundary_mask;
    logic fc_event_window_sample_last;
    logic [LANES*12-1:0] fc_event_addr_flat;
    logic [LANES*8-1:0] fc_event_data_flat;
    logic fc_learning_commit;
    logic fc_learning_busy;
    logic fc_cache_forward_read_valid;
    logic fc_cache_forward_read_ready;
    logic [7:0] fc_cache_forward_read_addr;
    logic [7:0] fc_cache_forward_read_tag;
    logic fc_cache_forward_response_valid;
    logic fc_cache_forward_response_ready;
    logic [7:0] fc_cache_forward_response_tag;
    logic signed [15:0] fc_cache_forward_response_code;
    logic fc_cache_forward_response_error;
    logic fc_w0_refresh_valid;
    logic fc_w0_refresh_ready;
    logic fc_w0_refresh_feature;
    logic [3:0] fc_w0_refresh_tile;
    logic fc_w0_refresh_done;
    logic fc_w0_refill_busy;
    logic fc_w0_refill_error;
    logic [31:0] fc_w0_rows_published;
    logic [3:0] shared_cache_fill_valid;
    logic [3:0] shared_cache_fill_ready;
    logic [7:0] shared_cache_fill_tile_flat;
    logic [3:0] shared_cache_fill_bank;
    logic [43:0] shared_cache_fill_addr_flat;
    logic [799:0] shared_cache_fill_data_flat;
    logic [1:0] cache_fill_grant;
    logic cache_fill_arbiter_quiescent;
    logic physical_cache_fill_ready;
    logic physical_cache_read_valid = 1'b0;
    logic physical_cache_read_ready;
    logic physical_cache_result_valid;
    logic [999:0] physical_cache_term0_flat;
    logic [999:0] physical_cache_term1_flat;
    logic [999:0] physical_cache_term2_flat;
    logic [999:0] physical_cache_term3_flat;

    logic legacy_read_valid;
    logic legacy_response_ready;
    logic legacy_write_valid;

    logic store_learning_read_valid;
    logic store_learning_read_ready;
    logic [7:0] store_learning_read_addr;
    logic [7:0] store_learning_read_tag;
    logic store_learning_response_valid = 1'b0;
    logic store_learning_response_ready;
    logic [7:0] store_learning_response_tag = '0;
    logic signed [15:0] store_learning_response_code = '0;
    logic signed [15:0] store_learning_response_residual = '0;
    logic store_learning_write_valid;
    logic [7:0] store_learning_write_addr;
    logic signed [15:0] store_learning_write_code;
    logic signed [15:0] store_learning_write_residual;
    logic store_learning_write_code_enable;
    logic store_learning_write_residual_enable;
    logic store_forward_read_valid;
    logic store_forward_read_ready;
    logic [7:0] store_forward_read_addr;
    logic [7:0] store_forward_read_tag;
    logic store_forward_response_valid = 1'b0;
    logic store_forward_response_ready;
    logic [7:0] store_forward_response_tag = '0;
    logic signed [15:0] store_forward_response_code = '0;
    logic store_quiescent;

    logic store_pending_q = 1'b0;
    logic store_pending_is_forward_q = 1'b0;
    logic [7:0] store_pending_addr_q = '0;
    logic [7:0] store_pending_tag_q = '0;
    logic signed [15:0] code_mem [0:255];
    logic signed [15:0] residual_mem [0:255];

    integer store_read_count = 0;
    integer store_write_count = 0;
    integer retired_sample_count = 0;
    integer core_start_count = 0;
    integer commit_count = 0;
    integer changed_write_count = 0;
    integer fc_w0_refresh_count = 0;
    integer store_forward_read_count = 0;
    integer last_changed_w0_addr = -1;
    integer last_changed_w0_code = 0;
    logic saw_w0 = 1'b0;
    logic saw_w2 = 1'b0;
    logic saw_changed_w0 = 1'b0;
    logic saw_changed_w2 = 1'b0;
    logic rmw_scoreboard_valid_q = 1'b0;
    logic rmw_response_seen_q = 1'b0;
    logic [7:0] rmw_scoreboard_addr_q = '0;
    logic [7:0] rmw_scoreboard_tag_q = '0;
    logic signed [15:0] rmw_old_code_q = '0;
    logic signed [15:0] rmw_old_residual_q = '0;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    assign store_learning_read_ready = !store_pending_q &&
        !store_learning_response_valid && !store_forward_response_valid;
    assign store_forward_read_ready = store_learning_read_ready;
    assign store_quiescent = !store_pending_q &&
        !store_learning_response_valid && !store_forward_response_valid;
    assign shared_cache_fill_ready = {3'b111, physical_cache_fill_ready};

    unified_tp_actual_learning_subsystem_v1 #(
        .FC_LEARN_W0(1'b1),
        .FC_USE_SHARED_PARAMETER_BACKEND(1'b1),
        .LANES(LANES), .ENGINES(2),
        .FC_MAX_HIDDEN(HIDDEN), .FC_FEATURES(FEATURES),
        .SRNN_HIDDEN(2), .SRNN_BACKBONE(2), .SRNN_CLASSES(2),
        .SCNN_CHANNELS(2), .SCNN_FEATURES(4), .SCNN_CLASSES(2),
        .PARAM_ADDR_WIDTH(8), .TAG_WIDTH(8),
        .FC_W0_BASE(8'd0), .FC_W2_BASE(8'd32),
        .SCNN_CONV8_BASE(8'd96), .SCNN_CLASSIFIER_BASE(8'd132),
        .SRNN_WH1_BASE(8'd64), .SRNN_WOUT_BASE(8'd68)
    ) dut (
        .i_clk(clk), .i_rst_n(rst_n), .i_clear_error(1'b0), .i_abort(1'b0),
        .i_profile_request_valid(profile_request_valid),
        .i_profile_request_code(profile_request_code),
        .i_profile_learning_enable(profile_learning_enable),
        .o_profile_switch_done(profile_switch_done),
        .o_active_profile_code(active_profile_code),
        .o_parameter_owner(parameter_owner),
        .o_system_error(system_error),
        .o_control_error_code(control_error_code),
        .i_run_valid(run_valid), .o_run_ready(run_ready),
        .i_run_profile_code(7'h02), .o_run_done(run_done),
        .o_run_busy(run_busy),
        .o_core_start(core_start), .i_core_start_ready(1'b1),
        .i_core_sample_done(core_sample_done),
        .i_core_global_idle(!core_model_busy_q), .i_core_error(1'b0),

        .i_fc_cfg_valid(fc_cfg_valid), .o_fc_cfg_ready(fc_cfg_ready),
        .i_fc_hidden_count(9'(HIDDEN)), .o_fc_configured(fc_configured),
        .o_fc_cfg_error(fc_cfg_error),
        .i_fc_window_valid(fc_window_valid),
        .o_fc_window_ready(fc_window_ready),
        .i_fc_record_valid(fc_record_valid),
        .i_fc_window_sample_id(fc_window_sample_id),
        .i_fc_window_timestep(fc_window_timestep),
        .i_fc_boundary_mask(fc_boundary_mask),
        .i_fc_window_sample_last(fc_window_sample_last),
        .i_fc_addr_flat(fc_addr_flat), .i_fc_data_flat(fc_data_flat),
        .i_fc_snapshot_label(fc_snapshot_label),
        .i_fc_snapshot_spikes_flat(fc_snapshot_spikes),
        .i_fc_snapshot_trace_flat(fc_snapshot_trace),
        .i_fc_snapshot_vmem_flat(fc_snapshot_vmem),
        .i_fc_snapshot_logits_flat(fc_snapshot_logits),
        .i_fc_target_hidden_trace_valid(target_hidden_valid),
        .o_fc_target_hidden_trace_ready(target_hidden_ready),
        .i_fc_target_trace_sample(target_trace_sample),
        .i_fc_target_hidden_index(target_hidden_index),
        .i_fc_target_hidden_trace_q016(target_hidden_trace),
        .i_fc_target_input_trace_valid(target_input_valid),
        .o_fc_target_input_trace_ready(target_input_ready),
        .i_fc_target_input_index(target_input_index),
        .i_fc_target_input_trace_q016(target_input_trace),
        .i_fc_target_done(target_done),
        .o_fc_target_done_ready(target_done_ready),
        .o_fc_event_window_valid(fc_event_window_valid),
        .i_fc_event_window_ready(1'b1),
        .o_fc_event_record_valid(fc_event_record_valid),
        .o_fc_event_window_sample_id(fc_event_window_sample_id),
        .o_fc_event_window_timestep(fc_event_window_timestep),
        .o_fc_event_boundary_mask(fc_event_boundary_mask),
        .o_fc_event_window_sample_last(fc_event_window_sample_last),
        .o_fc_event_addr_flat(fc_event_addr_flat),
        .o_fc_event_data_flat(fc_event_data_flat),
        .o_fc_native_read_valid(legacy_read_valid),
        .i_fc_native_read_ready(1'b0),
        .i_fc_native_response_valid(1'b0),
        .o_fc_native_response_ready(legacy_response_ready),
        .i_fc_native_response_tag('0),
        .i_fc_native_response_code('0),
        .i_fc_native_response_residual('0),
        .i_fc_native_response_error(1'b0),
        .o_fc_native_write_valid(legacy_write_valid),
        .i_fc_native_write_ready(1'b0),
        .i_fc_native_parameter_quiescent(1'b0),
        .o_fc_learning_commit(fc_learning_commit),
        .o_fc_learning_busy(fc_learning_busy),
        .i_fc_cache_forward_read_valid(fc_cache_forward_read_valid),
        .o_fc_cache_forward_read_ready(fc_cache_forward_read_ready),
        .i_fc_cache_forward_read_addr(fc_cache_forward_read_addr),
        .i_fc_cache_forward_read_tag(fc_cache_forward_read_tag),
        .o_fc_cache_forward_response_valid(
            fc_cache_forward_response_valid),
        .i_fc_cache_forward_response_ready(fc_cache_forward_response_ready),
        .o_fc_cache_forward_response_tag(fc_cache_forward_response_tag),
        .o_fc_cache_forward_response_code(fc_cache_forward_response_code),
        .o_fc_cache_forward_response_error(
            fc_cache_forward_response_error),
        .o_fc_w0_refresh_valid(fc_w0_refresh_valid),
        .i_fc_w0_refresh_ready(fc_w0_refresh_ready),
        .o_fc_w0_refresh_feature(fc_w0_refresh_feature),
        .o_fc_w0_refresh_tile(fc_w0_refresh_tile),
        .i_fc_w0_refresh_done(fc_w0_refresh_done),
        .i_fc_w0_refresh_error(fc_w0_refill_error),

        .i_srnn_cfg_valid(1'b0),
        .i_srnn_cfg_commit_version('0),
        .i_srnn_parameter_load_valid(1'b0),
        .i_srnn_parameter_load_addr('0),
        .i_srnn_parameter_load_code('0),
        .i_srnn_parameter_load_residual('0),
        .i_srnn_target_label_load_start(1'b0),
        .i_srnn_target_label_valid(1'b0),
        .i_srnn_target_label_sample(1'b0),
        .i_srnn_target_label_class('0),
        .i_srnn_target_label_last(1'b0),
        .i_srnn_target_spike_word_write_valid(1'b0),
        .i_srnn_target_spike_word_class('0),
        .i_srnn_target_spike_word_timestep(1'b0),
        .i_srnn_target_spike_word_id('0),
        .i_srnn_target_spike_word_data('0),
        .i_srnn_target_trace_write_valid(1'b0),
        .i_srnn_target_trace_class('0),
        .i_srnn_target_trace_timestep(1'b0),
        .i_srnn_target_trace_backbone('0),
        .i_srnn_target_trace_q14('0),
        .i_srnn_backbone_sample_start_valid(1'b0),
        .i_srnn_commit_valid(1'b0),
        .i_srnn_commit_version('0),
        .i_srnn_commit_sample_last(1'b0),
        .i_srnn_commit_spikes('0),

        .i_scnn_configured(1'b1),
        .i_scnn_target_start_ready(1'b0),
        .i_scnn_target_done(1'b0),
        .i_scnn_target_busy(1'b0),
        .i_scnn_target_error(1'b0),
        .i_scnn_selected_conv_done(1'b0),
        .i_scnn_sample_context_valid(1'b0),
        .i_scnn_sample_context_target(1'b0),
        .i_scnn_sample_context_sample(1'b0),
        .i_scnn_sample_context_version('0),
        .i_scnn_input_valid(1'b0),
        .i_scnn_input_target(1'b0),
        .i_scnn_input_sample(1'b0),
        .i_scnn_input_timestep(1'b0),
        .i_scnn_input_sample_version('0),
        .i_scnn_input_channel('0),
        .i_scnn_input_spatial('0),
        .i_scnn_input_q8_8('0),
        .i_scnn_trace_valid(1'b0),
        .i_scnn_trace_target(1'b0),
        .i_scnn_trace_sample(1'b0),
        .i_scnn_trace_timestep(1'b0),
        .i_scnn_trace_sample_version('0),
        .i_scnn_trace_channel('0),
        .i_scnn_trace_spatial('0),
        .i_scnn_trace_q8_8('0),
        .i_scnn_surrogate_valid(1'b0),
        .i_scnn_surrogate_target(1'b0),
        .i_scnn_surrogate_sample(1'b0),
        .i_scnn_surrogate_timestep(1'b0),
        .i_scnn_surrogate_sample_version('0),
        .i_scnn_surrogate_channel('0),
        .i_scnn_surrogate_spatial('0),
        .i_scnn_surrogate_q8_8('0),
        .i_scnn_feature_wave_valid(1'b0),
        .i_scnn_feature_wave_mask('0),
        .i_scnn_feature_wave_base('0),
        .i_scnn_feature_wave_q8_8_flat('0),
        .i_scnn_feature_wave_version('0),
        .i_scnn_feature_wave_frame('0),
        .i_scnn_feature_wave_first(1'b0),
        .i_scnn_feature_wave_last(1'b0),
        .i_scnn_feature_wave_sample_reset(1'b0),
        .i_scnn_feature_wave_sample_last(1'b0),
        .i_scnn_sample_label('0),
        .i_scnn_logit_wave_valid(1'b0),
        .i_scnn_logit_wave_mask('0),
        .i_scnn_logit_wave_base('0),
        .i_scnn_logit_wave_count('0),
        .i_scnn_logit_wave_q8_8_flat('0),
        .i_scnn_logit_wave_version('0),
        .i_scnn_logit_wave_frame('0),
        .i_scnn_logit_wave_frame_last(1'b0),
        .i_scnn_logit_wave_sample_last(1'b0),
        .i_scnn_parameter_load_valid(1'b0),
        .i_scnn_parameter_load_addr('0),
        .i_scnn_parameter_load_code('0),
        .i_scnn_parameter_load_residual('0),
        .i_scnn_forward_valid(1'b0),
        .i_scnn_forward_addr('0), .i_scnn_forward_tag('0),
        .i_scnn_forward_response_ready(1'b0),

        .i_store_load_ready(1'b1),
        .o_store_learning_read_valid(store_learning_read_valid),
        .i_store_learning_read_ready(store_learning_read_ready),
        .o_store_learning_read_addr(store_learning_read_addr),
        .o_store_learning_read_tag(store_learning_read_tag),
        .i_store_learning_response_valid(store_learning_response_valid),
        .o_store_learning_response_ready(store_learning_response_ready),
        .i_store_learning_response_tag(store_learning_response_tag),
        .i_store_learning_response_code(store_learning_response_code),
        .i_store_learning_response_residual(store_learning_response_residual),
        .i_store_learning_response_error(1'b0),
        .o_store_learning_write_valid(store_learning_write_valid),
        .i_store_learning_write_ready(1'b1),
        .o_store_learning_write_addr(store_learning_write_addr),
        .o_store_learning_write_code(store_learning_write_code),
        .o_store_learning_write_residual(store_learning_write_residual),
        .o_store_learning_write_code_enable(store_learning_write_code_enable),
        .o_store_learning_write_residual_enable(
            store_learning_write_residual_enable),
        .o_store_forward_read_valid(store_forward_read_valid),
        .i_store_forward_read_ready(store_forward_read_ready),
        .o_store_forward_read_addr(store_forward_read_addr),
        .o_store_forward_read_tag(store_forward_read_tag),
        .i_store_forward_response_valid(store_forward_response_valid),
        .o_store_forward_response_ready(store_forward_response_ready),
        .i_store_forward_response_tag(store_forward_response_tag),
        .i_store_forward_response_code(store_forward_response_code),
        .i_store_forward_response_error(1'b0),
        .i_store_quiescent(store_quiescent),
        .i_store_protocol_error(1'b0)
    );

    unified_array16_shared_weight_cache_refill_plane_v1 #(
        .PARAM_DEPTH(256), .PARAM_ADDR_WIDTH(8), .TAG_WIDTH(8),
        .VERSION_WIDTH(16), .FEATURE_COUNT_WIDTH(2),
        .HIDDEN_COUNT_WIDTH(7), .ARRAY_WEIGHT_ADDR_WIDTH(11)
    ) u_cache_refill_plane (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(1'b0),
        .i_clear_error(1'b0),
        .i_fc_refill_valid(fc_w0_refresh_valid),
        .o_fc_refill_ready(fc_w0_refresh_ready),
        .i_fc_refill_all(1'b0),
        .i_fc_dirty_feature({1'b0, fc_w0_refresh_feature}),
        .i_fc_dirty_tile(fc_w0_refresh_tile),
        .i_fc_parameter_base(8'd0),
        .i_fc_hidden_stride(8'(HIDDEN)),
        .i_fc_cache_row_base(10'd0),
        .i_fc_feature_count(2'(FEATURES)),
        .i_fc_hidden_count(7'(HIDDEN)), .i_fc_tile_count(5'd1),
        .i_fc_version(16'd1), .i_array_idle(!core_model_busy_q),
        .o_forward_read_valid(fc_cache_forward_read_valid),
        .i_forward_read_ready(fc_cache_forward_read_ready),
        .o_forward_read_addr(fc_cache_forward_read_addr),
        .o_forward_read_tag(fc_cache_forward_read_tag),
        .i_forward_response_valid(fc_cache_forward_response_valid),
        .o_forward_response_ready(fc_cache_forward_response_ready),
        .i_forward_response_tag(fc_cache_forward_response_tag),
        .i_forward_response_code(fc_cache_forward_response_code),
        .i_forward_response_error(fc_cache_forward_response_error),
        .i_topology_fill_busy(1'b0),
        .i_topology_fill_valid('0), .o_topology_fill_ready(),
        .i_topology_fill_tile_flat('0), .i_topology_fill_bank('0),
        .i_topology_fill_addr_flat('0), .i_topology_fill_data_flat('0),
        .o_weight_fill_valid(shared_cache_fill_valid),
        .i_weight_fill_ready(shared_cache_fill_ready),
        .o_weight_fill_tile_flat(shared_cache_fill_tile_flat),
        .o_weight_fill_bank(shared_cache_fill_bank),
        .o_weight_fill_addr_flat(shared_cache_fill_addr_flat),
        .o_weight_fill_data_flat(shared_cache_fill_data_flat),
        .o_fc_refill_done(fc_w0_refresh_done),
        .o_fc_refill_busy(),
        .o_fc_rows_published(fc_w0_rows_published),
        .o_fc_published_version(), .o_fill_grant(cache_fill_grant),
        .o_busy(fc_w0_refill_busy),
        .o_quiescent(cache_fill_arbiter_quiescent),
        .o_sticky_error(fc_w0_refill_error)
    );

    unified_dual_role_weight_store_p4all_candidate_v1 u_fc_tile0_cache (
        .i_clk(clk), .i_rst_n(rst_n),
        .i_fill_valid(shared_cache_fill_valid[0] &&
            shared_cache_fill_tile_flat[0 +: 2] == 2'd0),
        .o_fill_ready(physical_cache_fill_ready),
        .i_fill_bank({shared_cache_fill_addr_flat[10],
            shared_cache_fill_bank[0]}),
        .i_fill_addr(shared_cache_fill_addr_flat[0 +: 10]),
        .i_fill_data(shared_cache_fill_data_flat[0 +: 200]),
        .i_read_valid(physical_cache_read_valid),
        .o_read_ready(physical_cache_read_ready), .i_mode(2'd0),
        .i_scnn_active_pair(1'b0),
        .i_read_addr0(10'd0), .i_read_addr1(10'd0),
        .i_read_addr2(10'd0), .i_read_addr3(10'd0),
        .i_read_subword0('0), .i_read_subword1('0),
        .i_read_subword2('0), .i_read_subword3('0),
        .o_read_result_valid(physical_cache_result_valid),
        .i_read_result_ready(1'b1),
        .o_term0_flat(physical_cache_term0_flat),
        .o_term1_flat(physical_cache_term1_flat),
        .o_term2_flat(physical_cache_term2_flat),
        .o_term3_flat(physical_cache_term3_flat)
    );

    task automatic check(input logic condition, input string message);
        if (!condition)
            $fatal(1, "%s", message);
    endtask

    task automatic configure_fc;
        int timeout;
        begin
            timeout = 0;
            while (!fc_cfg_ready && timeout < 100) begin
                @(negedge clk);
                timeout++;
            end
            check(timeout < 100, "FC configuration ready timeout");
            fc_cfg_valid = 1'b1;
            @(negedge clk);
            fc_cfg_valid = 1'b0;
            timeout = 0;
            while (!fc_configured && !fc_cfg_error && timeout < 100) begin
                @(negedge clk);
                timeout++;
            end
            check(fc_configured && !fc_cfg_error,
                "FC shared-backend configuration failed");
        end
    endtask

    task automatic select_fc_profile;
        int timeout;
        begin
            profile_request_code = 7'h02;
            profile_learning_enable = 1'b1;
            profile_request_valid = 1'b1;
            timeout = 0;
            while (!profile_switch_done && timeout < 100) begin
                @(negedge clk);
                timeout++;
            end
            check(timeout < 100 && active_profile_code == 7'h02 &&
                parameter_owner == 3'b001 && dut.o_profile_aligned &&
                dut.o_start_allowed,
                "FC shared-backend profile acquisition failed");
        end
    endtask

    task automatic start_fc_run;
        int timeout;
        begin
            run_valid = 1'b1;
            timeout = 0;
            while (!run_ready && timeout < 100) begin
                @(negedge clk);
                timeout++;
            end
            check(timeout < 100, "FC run admission timeout");
            @(negedge clk);
            run_valid = 1'b0;
            timeout = 0;
            while (!run_busy && timeout < 100) begin
                @(negedge clk);
                timeout++;
            end
            if (timeout >= 100) begin
                $display("FC_RUN_DEBUG ready=%b busy=%b aligned=%b start_allowed=%b active_profile=%02x owner=%03b client_active=%03b client_quiescent=%03b fc_configured=%b fc_release_ready=%b fc_execution_quiescent=%b backend_quiescent=%b core_quiescent=%b core_busy=%b system_error=%b/%02x dispatch_state=%0d",
                    run_ready, run_busy, dut.o_profile_aligned,
                    dut.o_start_allowed, active_profile_code,
                    parameter_owner, dut.client_active,
                    dut.client_quiescent, fc_configured,
                    dut.fc_release_ready, dut.fc_execution_quiescent,
                    dut.backend_quiescent, dut.u_fabric.o_core_quiescent,
                    dut.u_fabric.o_core_busy, system_error,
                    control_error_code,
                    dut.u_fabric.u_runtime.u_runtime.u_control.
                        u_dispatch.state_q);
            end
            check(timeout < 100, "FC run did not become active");
        end
    endtask

    task automatic send_target_hidden(
        input int sample_index,
        input int hidden_index,
        input logic [16:0] value
    );
        begin
            while (!target_hidden_ready) @(negedge clk);
            target_trace_sample = 4'(sample_index);
            target_hidden_index = 9'(hidden_index);
            target_hidden_trace = value;
            target_hidden_valid = 1'b1;
            @(negedge clk);
            target_hidden_valid = 1'b0;
        end
    endtask

    task automatic send_target_input(
        input int sample_index,
        input int input_index,
        input logic [19:0] value
    );
        begin
            while (!target_input_ready) @(negedge clk);
            target_trace_sample = 4'(sample_index);
            target_input_index = 4'(input_index);
            target_input_trace = value;
            target_input_valid = 1'b1;
            @(negedge clk);
            target_input_valid = 1'b0;
        end
    endtask

    task automatic send_sample(input int sample_index);
        int expected_retired;
        begin
            fc_record_valid = 2'b11;
            fc_window_sample_id = 16'(sample_index);
            fc_window_timestep = 9'd0;
            fc_boundary_mask = 2'b01;
            fc_window_sample_last = 1'b1;
            fc_addr_flat = '0;
            fc_addr_flat[0 +: 12] = 12'd0;
            fc_addr_flat[12 +: 12] = 12'd1;
            fc_data_flat = {8'd2, 8'd1};
            fc_snapshot_label = 4'(sample_index % OUTPUTS);
            fc_snapshot_spikes = sample_index[0] ? 2'b10 : 2'b01;
            fc_snapshot_trace[0 +: 16] = sample_index[0] ?
                16'h5000 : 16'h1000;
            fc_snapshot_trace[16 +: 16] = sample_index[0] ?
                16'h7000 : 16'h3000;
            fc_snapshot_vmem[0 +: 18] = 18'sd1024;
            fc_snapshot_vmem[18 +: 18] = 18'sd1024;
            fc_snapshot_logits = '0;
            fc_snapshot_logits[(sample_index % OUTPUTS)*24 +: 24] =
                24'sd128;
            expected_retired = retired_sample_count + 1;
            fc_window_valid = 1'b1;
            while (!fc_window_ready) @(negedge clk);
            @(posedge clk);
            @(negedge clk);
            fc_window_valid = 1'b0;
            while (core_start_count < sample_index + 1) @(negedge clk);
            repeat (2) @(negedge clk);
            core_sample_done = 1'b1;
            @(negedge clk);
            core_sample_done = 1'b0;
            while (retired_sample_count < expected_retired) @(negedge clk);
        end
    endtask

    task automatic check_updated_physical_cache;
        integer feature;
        integer lane;
        integer bank;
        integer observed;
        integer tail_observed;
        integer timeout;
        begin
            check(last_changed_w0_addr >= 0,
                "no changed FC W0 code was available for cache readback");
            feature = last_changed_w0_addr / HIDDEN;
            lane = last_changed_w0_addr % HIDDEN;
            bank = feature & 3;

            @(negedge clk);
            physical_cache_read_valid = 1'b1;
            timeout = 0;
            while (!physical_cache_read_ready && timeout < 20) begin
                @(negedge clk);
                timeout++;
            end
            check(timeout < 20, "physical FC cache read request timeout");
            @(posedge clk);
            #1;
            physical_cache_read_valid = 1'b0;
            check(physical_cache_result_valid,
                "physical FC cache did not return the refreshed row");

            unique case (bank)
                0: begin
                    observed = $signed(
                        physical_cache_term0_flat[lane*25 +: 25]);
                    tail_observed = $signed(
                        physical_cache_term0_flat[HIDDEN*25 +: 25]);
                end
                1: begin
                    observed = $signed(
                        physical_cache_term1_flat[lane*25 +: 25]);
                    tail_observed = $signed(
                        physical_cache_term1_flat[HIDDEN*25 +: 25]);
                end
                2: begin
                    observed = $signed(
                        physical_cache_term2_flat[lane*25 +: 25]);
                    tail_observed = $signed(
                        physical_cache_term2_flat[HIDDEN*25 +: 25]);
                end
                default: begin
                    observed = $signed(
                        physical_cache_term3_flat[lane*25 +: 25]);
                    tail_observed = $signed(
                        physical_cache_term3_flat[HIDDEN*25 +: 25]);
                end
            endcase
            check(observed == last_changed_w0_code,
                $sformatf("physical FC cache mismatch addr=%0d got=%0d expected=%0d",
                    last_changed_w0_addr, observed, last_changed_w0_code));
            check(tail_observed == 0,
                "physical FC cache row did not zero an inactive H40 lane");
            @(negedge clk);
        end
    endtask

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            core_model_busy_q <= 1'b0;
            store_pending_q <= 1'b0;
            store_pending_is_forward_q <= 1'b0;
            store_pending_addr_q <= '0;
            store_pending_tag_q <= '0;
            store_learning_response_valid <= 1'b0;
            store_learning_response_tag <= '0;
            store_learning_response_code <= '0;
            store_learning_response_residual <= '0;
            store_forward_response_valid <= 1'b0;
            store_forward_response_tag <= '0;
            store_forward_response_code <= '0;
            rmw_scoreboard_valid_q <= 1'b0;
            rmw_response_seen_q <= 1'b0;
            rmw_scoreboard_addr_q <= '0;
            rmw_scoreboard_tag_q <= '0;
            rmw_old_code_q <= '0;
            rmw_old_residual_q <= '0;
        end else begin
            if (fc_w0_refresh_valid && fc_w0_refresh_ready)
                fc_w0_refresh_count <= fc_w0_refresh_count + 1;
            if (core_start)
                core_model_busy_q <= 1'b1;
            if (core_sample_done)
                core_model_busy_q <= 1'b0;
            if (run_done && (fc_w0_refill_busy || fc_w0_refresh_valid))
                $fatal(1, "FC run retired before W0 cache refresh drained");
            if (store_learning_response_valid &&
                store_learning_response_ready) begin
                if (!rmw_scoreboard_valid_q ||
                    store_learning_response_tag != rmw_scoreboard_tag_q ||
                    store_learning_response_code != rmw_old_code_q ||
                    store_learning_response_residual != rmw_old_residual_q)
                    $fatal(1, "shared FC response scoreboard mismatch");
                rmw_response_seen_q <= 1'b1;
                store_learning_response_valid <= 1'b0;
            end
            if (store_forward_response_valid &&
                    store_forward_response_ready)
                store_forward_response_valid <= 1'b0;
            if (store_pending_q && !store_learning_response_valid &&
                    !store_forward_response_valid) begin
                store_pending_q <= 1'b0;
                if (store_pending_is_forward_q) begin
                    store_forward_response_valid <= 1'b1;
                    store_forward_response_tag <= store_pending_tag_q;
                    store_forward_response_code <=
                        code_mem[store_pending_addr_q];
                end else begin
                    store_learning_response_valid <= 1'b1;
                    store_learning_response_tag <= store_pending_tag_q;
                    store_learning_response_code <=
                        code_mem[store_pending_addr_q];
                    store_learning_response_residual <=
                        residual_mem[store_pending_addr_q];
                end
            end
            if (store_learning_read_valid && store_forward_read_valid &&
                    store_learning_read_ready && store_forward_read_ready)
                $fatal(1, "shared store accepted simultaneous learning/forward reads");
            if (store_learning_read_valid && store_learning_read_ready) begin
                if (rmw_scoreboard_valid_q)
                    $fatal(1, "shared FC accepted overlapping RMW reads");
                store_pending_q <= 1'b1;
                store_pending_is_forward_q <= 1'b0;
                store_pending_addr_q <= store_learning_read_addr;
                store_pending_tag_q <= store_learning_read_tag;
                rmw_scoreboard_valid_q <= 1'b1;
                rmw_response_seen_q <= 1'b0;
                rmw_scoreboard_addr_q <= store_learning_read_addr;
                rmw_scoreboard_tag_q <= store_learning_read_tag;
                rmw_old_code_q <= code_mem[store_learning_read_addr];
                rmw_old_residual_q <= residual_mem[store_learning_read_addr];
            end else if (store_forward_read_valid &&
                    store_forward_read_ready) begin
                store_pending_q <= 1'b1;
                store_pending_is_forward_q <= 1'b1;
                store_pending_addr_q <= store_forward_read_addr;
                store_pending_tag_q <= store_forward_read_tag;
                store_forward_read_count <= store_forward_read_count + 1;
            end
            if (store_learning_write_valid) begin
                if (!rmw_scoreboard_valid_q ||
                    !(rmw_response_seen_q ||
                      (store_learning_response_valid &&
                       store_learning_response_ready)) ||
                    store_learning_write_addr != rmw_scoreboard_addr_q)
                    $fatal(1, "shared FC write did not match its read transaction");
                if (store_learning_write_code_enable)
                    code_mem[store_learning_write_addr] <=
                        store_learning_write_code;
                if (store_learning_write_residual_enable)
                    residual_mem[store_learning_write_addr] <=
                        store_learning_write_residual;
                rmw_scoreboard_valid_q <= 1'b0;
                rmw_response_seen_q <= 1'b0;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst_n) begin
            if (legacy_read_valid || legacy_response_ready ||
                legacy_write_valid)
                $fatal(1, "legacy FC native port was active in shared mode");
            if (store_learning_read_valid && store_learning_read_ready) begin
                if (parameter_owner != 3'b001)
                    $fatal(1, "shared FC read occurred without client0 owner");
                store_read_count <= store_read_count + 1;
                if (store_learning_read_addr < 8'd4)
                    saw_w0 <= 1'b1;
                if (store_learning_read_addr >= 8'd32 &&
                    store_learning_read_addr < 8'd52)
                    saw_w2 <= 1'b1;
            end
            if (store_learning_write_valid) begin
                if (parameter_owner != 3'b001)
                    $fatal(1, "shared FC write occurred without client0 owner");
                store_write_count <= store_write_count + 1;
                if ((store_learning_write_code_enable &&
                     store_learning_write_code != rmw_old_code_q) ||
                    (store_learning_write_residual_enable &&
                     store_learning_write_residual != rmw_old_residual_q)) begin
                    changed_write_count <= changed_write_count + 1;
                    if (store_learning_write_addr < 8'd4)
                        saw_changed_w0 <= 1'b1;
                    if (store_learning_write_addr >= 8'd32 &&
                        store_learning_write_addr < 8'd52)
                        saw_changed_w2 <= 1'b1;
                end
                if (store_learning_write_addr < 8'd4 &&
                        store_learning_write_code_enable) begin
                    last_changed_w0_addr <= store_learning_write_addr;
                    last_changed_w0_code <= $signed(
                        store_learning_write_code);
                end
            end
            if (core_start)
                core_start_count <= core_start_count + 1;
            if (dut.fc_runtime_sample_retired)
                retired_sample_count <= retired_sample_count + 1;
            if (fc_learning_commit)
                commit_count <= commit_count + 1;
            if (system_error) begin
                $display("FC_SHARED_ERROR runtime=%b cfg=%b observation=%b context=%b arithmetic=%b parameter=%b bridge=%b adapter=%b capture=%b coordinator=%b subsystem=%b command_state=%0d engine_state=%0d window_valid=%b ready=%b core_start=%b core_done=%b",
                    dut.fc_runtime_error, dut.u_fc.o_cfg_error,
                    dut.u_fc.o_observation_error,
                    dut.u_fc.o_context_error,
                    dut.u_fc.o_arithmetic_overflow,
                    dut.u_fc.o_parameter_error,
                    dut.u_fc.bridge_protocol_error,
                    dut.u_fc.adapter_protocol_error,
                    dut.u_fc.u_learning_bridge.capture_protocol_error,
                    dut.u_fc.u_learning_bridge.coordinator_protocol_error,
                    dut.u_fc.u_learning_bridge.subsystem_protocol_error,
                    dut.u_fc.u_command_adapter.state_q,
                    dut.u_fc.u_learning_bridge.u_online_subsystem.u_engine.state_q,
                    fc_window_valid, fc_window_ready, core_start,
                    core_sample_done);
                $display("FC_SHARED_PARAMETER_DETAIL w0_cfg=%b w0_req=%b w0_store=%b w0_internal=%b w2_cfg=%b w2_req=%b w2_store=%b w2_internal=%b rmw_protocol=%b backend=%b router=%b store_pending=%b response_valid=%b scoreboard_valid=%b response_seen=%b",
                    dut.u_fc.u_learning_bridge.u_online_subsystem.u_engine.
                        w0_update_config_error,
                    dut.u_fc.u_learning_bridge.u_online_subsystem.u_engine.
                        w0_request_error,
                    dut.u_fc.u_learning_bridge.u_online_subsystem.u_engine.
                        w0_store_error,
                    dut.u_fc.u_learning_bridge.u_online_subsystem.u_engine.
                        w0_internal_error,
                    dut.u_fc.u_learning_bridge.u_online_subsystem.u_engine.
                        w2_update_config_error,
                    dut.u_fc.u_learning_bridge.u_online_subsystem.u_engine.
                        w2_request_error,
                    dut.u_fc.u_learning_bridge.u_online_subsystem.u_engine.
                        w2_store_error,
                    dut.u_fc.u_learning_bridge.u_online_subsystem.u_engine.
                        w2_internal_error,
                    dut.u_fc.u_learning_bridge.u_online_subsystem.u_engine.
                        rmw_protocol_error,
                    dut.backend_error, dut.router_protocol_error,
                    store_pending_q, store_learning_response_valid,
                    rmw_scoreboard_valid_q, rmw_response_seen_q);
                $fatal(1, "shared FC subsystem error code=0x%02x",
                    control_error_code);
            end
        end
    end

    initial begin
        for (int address = 0; address < 256; address++) begin
            code_mem[address] = 16'(address % 7 - 3);
            // vote_threshold is one in this integration, so zero is the only
            // legal canonical residual entering a direct TP update.
            residual_mem[address] = 16'sd0;
        end

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        configure_fc();
        select_fc_profile();
        start_fc_run();
        $display("FC_SHARED_STAGE run_active busy=%b target_hidden_ready=%b target_input_ready=%b window_ready=%b",
            run_busy, target_hidden_ready, target_input_ready, fc_window_ready);

        for (int sample = 0; sample < BATCH; sample++) begin
            send_target_hidden(sample, 0,
                sample[0] ? 17'h00000 : 17'h10000);
            send_target_hidden(sample, 1,
                sample[0] ? 17'h10000 : 17'h00000);
            for (int input_index = 0; input_index < OUTPUTS;
                 input_index++) begin
                send_target_input(sample, input_index,
                    input_index == (sample % OUTPUTS) ?
                    20'h10000 : 20'h00000);
            end
        end
        $display("FC_SHARED_STAGE target_payload_loaded hidden=%0d input=%0d target_done_ready=%b",
            BATCH * HIDDEN, BATCH * OUTPUTS, target_done_ready);
        target_done = 1'b1;
        fork
            begin
                while (!target_done_ready) @(negedge clk);
                @(negedge clk);
                target_done = 1'b0;
            end
            begin
                for (int sample = 0; sample < BATCH; sample++)
                    send_sample(sample);
            end
        join

        for (int timeout = 0;
             timeout < 500000 && (!run_done || commit_count == 0);
             timeout++)
            @(posedge clk);

        check(run_done && !run_busy,
            "FC shared-backend run did not retire");
        check(commit_count == 1,
            "FC shared-backend batch did not commit exactly once");
        check(retired_sample_count == BATCH && core_start_count == BATCH,
            "FC shared-backend sample lifecycle count mismatch");
        check(store_read_count > 0 &&
              store_read_count == store_write_count,
            "FC shared-backend RMW coverage incomplete");
        check(saw_w0 && saw_w2,
            "FC shared backend did not observe both W0 and W2 address windows");
        check(changed_write_count > 0 && saw_changed_w0 && saw_changed_w2 &&
              !rmw_scoreboard_valid_q,
            "FC shared backend did not persist matched W0/W2 RMW updates");
        check(store_quiescent && !fc_learning_busy,
            "FC shared backend did not drain after commit");
        check(fc_w0_refresh_count > 0 &&
              !fc_w0_refill_busy && !fc_w0_refresh_valid &&
              !fc_w0_refill_error && fc_w0_rows_published == 1 &&
              cache_fill_arbiter_quiescent && cache_fill_grant == 2'b00,
            "FC shared backend did not publish and drain dirty W0 rows");
        check(store_forward_read_count >= HIDDEN,
            "FC W0 refill did not read the authoritative parameter store");
        check_updated_physical_cache();

        profile_request_valid = 1'b0;

        $display("TP_ACTUAL_LEARNING_SUBSYSTEM_SHARED_FC_PASS samples=16 owner=client0 reads=%0d writes=%0d changed=%0d refreshes=%0d forward_reads=%0d w0=1 w2=1 response_scoreboard=1 persistent_updates=1 fill_arbiter=1 physical_cache_forward_updated=1 cache_flush_before_run_done=1 legacy_native_silent=1", store_read_count, store_write_count, changed_write_count, fc_w0_refresh_count, store_forward_read_count);
        $finish;
    end

    initial begin
        repeat (1000000) @(posedge clk);
        $display("FC_SHARED_TIMEOUT ready=%b busy=%b done=%b target_ready=%b/%b/%b window_ready=%b core_start=%b core_done=%b retired=%0d starts=%0d reads=%0d writes=%0d commits=%0d owner=%03b fc_learning_busy=%b engine_state=%0d runtime_dispatch=%0d system_error=%b/%02x",
            run_ready, run_busy, run_done, target_hidden_ready,
            target_input_ready, target_done_ready, fc_window_ready,
            core_start, core_sample_done, retired_sample_count,
            core_start_count, store_read_count, store_write_count,
            commit_count, parameter_owner, fc_learning_busy,
            dut.u_fc.u_learning_bridge.u_online_subsystem.u_engine.state_q,
            dut.u_fabric.u_runtime.u_runtime.u_control.u_dispatch.state_q,
            system_error, control_error_code);
        $fatal(1, "FC shared-backend integration timeout");
    end
endmodule

`default_nettype wire
