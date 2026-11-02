`timescale 1ns/1ps
`default_nettype none

module tb_srnn_formal_m2_numeric #(parameter bit WITH_COORDINATOR = 1'b0);
    localparam int AW = 11;
    localparam int MEMORY_DEPTH = 1024;
    localparam int MW = 10;
    localparam int WW = 9;
    localparam int PARAM_DEPTH = 120320;
    localparam int PARAM_ADDR_WIDTH = 17;
    localparam int SAMPLE_ID_WIDTH = 8;
    bit inject_timestep_id_error = 0;
    bit inject_done_timestep_id_error = 0;
    localparam int HIDDEN_PARAMETER_BASE = 0;
    localparam int OUTPUT_PARAMETER_BASE = 115200;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic abort_all = 1'b0;
    logic clear_error = 1'b0;

    logic cfg_valid = 1'b0, cfg_ready, cfg_error;
    logic timestep_start = 1'b0, timestep_start_ready;
    logic [6:0] timestep_id = '0;
    logic [7:0] sequence_version = 8'd1;
    logic prepare_valid = 1'b0, prepare_ready, prepare_done;
    logic cache_valid;
    logic [7:0] cache_version;

    logic observation_valid = 1'b0, observation_ready;
    logic observation_path = 1'b0;
    logic [SAMPLE_ID_WIDTH-1:0] observation_sample = '0;
    logic [8:0] observation_backbone = '0;
    logic signed [31:0] observation_q14 = '0;
    logic observation_last = 1'b0, observation_done = 1'b0;

    logic layer_valid, layer_ready;
    logic [1:0] layer_mode;
    logic [15:0] layer_neurons, layer_ff_groups, layer_rec_groups;
    logic layer_rec_enable;
    logic [15:0] layer_leak, layer_trace_decay;
    logic [31:0] layer_threshold;
    logic layer_reset_to_zero;
    logic [51:0] layer_tag;
    logic [WW-1:0] layer_work_base;
    logic [3:0] layer_destination_base;
    logic group_valid, group_ready, group_recurrent;
    logic [3:0] group_source_valid;
    logic [AW-1:0] group_addr0, group_addr1, group_addr2, group_addr3;
    logic layer_done, layer_busy, layer_idle, layer_error;
    logic [7:0] layer_error_code;

    logic owner_request_valid, owner_request_ready;
    logic owner_select_direct, owner_select_maintenance, owner_direct_done;
    logic owner_busy, owner_idle, owner_error;
    logic command_valid, command_ready, physical_command_ready;
    logic [1:0] command_mode;
    logic [15:0] command_neurons;
    logic command_first, command_last;
    logic [3:0] command_source_valid;
    logic [AW-1:0] command_addr0, command_addr1;
    logic [AW-1:0] command_addr2, command_addr3;
    logic [15:0] command_leak, command_trace_decay;
    logic [31:0] command_threshold;
    logic command_reset_to_zero;
    logic [51:0] command_tag;
    logic [WW-1:0] command_work_base;
    logic [3:0] command_destination_base;
    logic array_done, array_busy, array_idle, array_error;
    logic [15:0] tile_done, tile_idle, tile_error;

    logic [3:0] weight_fill_valid, weight_fill_ready;
    logic [7:0] weight_fill_tile_flat;
    logic [3:0] weight_fill_bank;
    logic [4*AW-1:0] weight_fill_addr_flat;
    logic [799:0] weight_fill_data_flat;
    logic refill_read_valid, refill_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] refill_read_addr;
    logic [7:0] refill_read_tag;
    logic refill_response_valid, refill_response_ready;
    logic [7:0] refill_response_tag;
    logic signed [15:0] refill_response_code;
    logic signed [15:0] model_mem [0:PARAM_DEPTH-1];
    logic refill_response_valid_q;
    logic [7:0] refill_response_tag_q;
    logic signed [15:0] refill_response_code_q;
    logic [3:0] raw_observation_valid, raw_observation_ready;
    logic [15:0] raw_observation_tile_flat;
    logic [7:0] raw_observation_mode_flat;
    logic [39:0] raw_observation_lane_mask_flat;
    logic [39:0] raw_observation_range_mask_flat;
    logic [39:0] raw_observation_spike_flat;
    logic [1279:0] raw_observation_pre_state_flat;
    logic [1279:0] raw_observation_trace_next_flat;
    logic [255:0] raw_observation_meta_flat;

    logic clear_start_valid, clear_start_ready;
    logic clear_release_valid, clear_release_ready;
    logic state_write_valid, state_write_ready;
    logic [MW-1:0] state_write_addr;
    logic [319:0] state_write_data;
    logic [39:0] state_write_mask;
    logic trace_write_valid, trace_write_ready;
    logic [MW-1:0] trace_write_addr;
    logic [319:0] trace_write_data;
    logic [39:0] trace_write_mask;
    logic [3:0] maintenance_tile;

    logic scalar_valid, scalar_ready;
    logic [1:0] scalar_mode;
    logic [51:0] scalar_tag;
    logic [9:0] scalar_neuron;
    logic scalar_spike;
    logic signed [31:0] scalar_pre, scalar_trace;
    logic scalar_last_record, scalar_busy, scalar_error;

    logic hidden_valid, hidden_ready, hidden_path;
    logic [SAMPLE_ID_WIDTH-1:0] hidden_sample;
    logic [8:0] hidden_id;
    logic signed [31:0] hidden_pre, hidden_next, hidden_trace;
    logic hidden_spike;
    logic signed [31:0] hidden_feature;
    logic hidden_last;
    logic logit_valid, logit_ready;
    logic [SAMPLE_ID_WIDTH-1:0] logit_sample;
    logic [4:0] logit_class;
    logic signed [63:0] logit_q28;
    logic logit_last;
    logic [6:0] observed_timestep;
    logic timestep_done, engine_busy, engine_error, protocol_error, overflow;
    integer hidden_seen = 0;
    integer logit_seen = 0;

    always #5 clk = ~clk;

    generate if (!WITH_COORDINATOR) begin : g_readonly_memory
    assign hidden_ready = 1'b1;
    assign logit_ready = 1'b1;
    assign refill_read_ready = !refill_response_valid_q ||
        refill_response_ready;
    assign refill_response_valid = refill_response_valid_q;
    assign refill_response_tag = refill_response_tag_q;
    assign refill_response_code = refill_response_code_q;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            refill_response_valid_q <= 1'b0;
            refill_response_tag_q <= '0;
            refill_response_code_q <= '0;
        end else begin
            if (refill_response_valid_q && refill_response_ready)
                refill_response_valid_q <= 1'b0;
            if (refill_read_valid && refill_read_ready) begin
                refill_response_valid_q <= 1'b1;
                refill_response_tag_q <= refill_read_tag;
                refill_response_code_q <= model_mem[refill_read_addr];
            end
        end
    end

    end endgenerate

    unified_srnn_m2_h40_forward_engine_v1 #(
        .PARAM_DEPTH(PARAM_DEPTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(8),
        .BATCH_MAX(2), .TIMESTEP_MAX(128), .BACKBONE_MAX(450),
        .HIDDEN_MAX(256), .CLASS_MAX(20), .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .TIMESTEP_ID_WIDTH(7), .BACKBONE_INDEX_WIDTH(9),
        .HIDDEN_INDEX_WIDTH(9), .CLASS_INDEX_WIDTH(5),
        .VERSION_WIDTH(8), .WEIGHT_DEPTH(1156),
        .WEIGHT_ADDR_WIDTH(AW), .MEMORY_DEPTH(MEMORY_DEPTH),
        .MEMORY_ADDR_WIDTH(MW), .WORK_ADDR_WIDTH(WW),
        .PHYSICAL_TILE_COUNT(12),
        .OUTPUT_TILE(7)
    ) u_engine (
        .i_clk(clk), .i_rst_n(rst_n), .i_clear_error(clear_error),
        .i_abort(abort_all),
        .i_cfg_valid(WITH_COORDINATOR ? c_engine_cfg_valid : cfg_valid), .o_cfg_ready(cfg_ready),
        .i_cfg_batch_count(SAMPLE_ID_WIDTH'(2)), .i_cfg_backbone_count(9'd450),
        .i_cfg_hidden_count(9'd256), .i_cfg_class_count(5'd20),
        .i_cfg_wh1_parameter_base(PARAM_ADDR_WIDTH'(HIDDEN_PARAMETER_BASE)),
        .i_cfg_wout_parameter_base(PARAM_ADDR_WIDTH'(OUTPUT_PARAMETER_BASE)),
        .i_cfg_wh1_cache_base(11'd0),
        .i_cfg_wout_cache_base(11'd0),
        .i_cfg_hidden_work_base(WW'(0)),
        .i_cfg_output_work_base(WW'(32)), .o_cfg_error(cfg_error),
        .i_prepare_valid(WITH_COORDINATOR ? c_prepare_valid : prepare_valid), .o_prepare_ready(prepare_ready),
        .i_prepare_version(WITH_COORDINATOR ? c_prepare_version : sequence_version), .i_cache_invalidate(WITH_COORDINATOR && c_cache_invalidate),
        .o_prepare_done(prepare_done), .o_cache_valid(cache_valid),
        .o_cache_version(cache_version),
        .i_timestep_start(WITH_COORDINATOR ? c_engine_start : timestep_start),
        .o_timestep_start_ready(timestep_start_ready),
        .i_timestep_id(WITH_COORDINATOR ? c_engine_timestep : timestep_id),
        .i_sequence_version(WITH_COORDINATOR ? c_engine_sequence : sequence_version),
        .i_include_target_path(WITH_COORDINATOR && c_include_target),
        .i_observation_valid(WITH_COORDINATOR ? c_obs_valid : observation_valid),
        .o_observation_ready(observation_ready),
        .i_observation_path(WITH_COORDINATOR ? c_obs_path : observation_path),
        .i_observation_sample_id(WITH_COORDINATOR ? c_obs_sample : observation_sample),
        .i_observation_backbone_id(WITH_COORDINATOR ? c_obs_backbone : observation_backbone),
        .i_observation_q14(WITH_COORDINATOR ? c_obs_q14 : observation_q14),
        .i_observation_last(WITH_COORDINATOR ? c_obs_last : observation_last),
        .i_observation_done(WITH_COORDINATOR ? c_obs_done : observation_done),
        .o_forward_read_valid(refill_read_valid),
        .i_forward_read_ready(refill_read_ready),
        .o_forward_read_addr(refill_read_addr),
        .o_forward_read_tag(refill_read_tag),
        .i_forward_response_valid(refill_response_valid),
        .o_forward_response_ready(refill_response_ready),
        .i_forward_response_tag(refill_response_tag),
        .i_forward_response_code(refill_response_code),
        .i_forward_response_error(1'b0),
        .i_array_idle(array_idle && owner_idle),
        .o_weight_fill_valid(weight_fill_valid),
        .i_weight_fill_ready(weight_fill_ready),
        .o_weight_fill_tile_flat(weight_fill_tile_flat),
        .o_weight_fill_bank(weight_fill_bank),
        .o_weight_fill_addr_flat(weight_fill_addr_flat),
        .o_weight_fill_data_flat(weight_fill_data_flat),
        .o_maintenance_start_valid(clear_start_valid),
        .i_maintenance_start_ready(clear_start_ready),
        .o_maintenance_release_valid(clear_release_valid),
        .i_maintenance_release_ready(clear_release_ready),
        .o_maintenance_tile(maintenance_tile),
        .o_state_write_valid(state_write_valid),
        .i_state_write_ready(state_write_ready && owner_select_maintenance),
        .o_state_write_addr(state_write_addr),
        .o_state_write_data(state_write_data),
        .o_state_write_mask(state_write_mask),
        .o_trace_write_valid(trace_write_valid),
        .i_trace_write_ready(trace_write_ready && owner_select_maintenance),
        .o_trace_write_addr(trace_write_addr),
        .o_trace_write_data(trace_write_data),
        .o_trace_write_mask(trace_write_mask),
        .o_layer_valid(layer_valid), .i_layer_ready(layer_ready),
        .o_layer_mode(layer_mode),
        .o_layer_neuron_count(layer_neurons),
        .o_layer_feedforward_group_count(layer_ff_groups),
        .o_layer_recurrent_group_count(layer_rec_groups),
        .o_layer_recurrent_enable(layer_rec_enable),
        .o_layer_leak(layer_leak),
        .o_layer_threshold(layer_threshold),
        .o_layer_reset_to_zero(layer_reset_to_zero),
        .o_layer_trace_decay(layer_trace_decay),
        .o_layer_transaction_tag(layer_tag),
        .o_layer_work_base_addr(layer_work_base),
        .o_layer_destination_tile_base(layer_destination_base),
        .o_group_valid(group_valid), .i_group_ready(group_ready),
        .o_group_recurrent(group_recurrent),
        .o_group_source_valid(group_source_valid),
        .o_group_weight_addr0(group_addr0),
        .o_group_weight_addr1(group_addr1),
        .o_group_weight_addr2(group_addr2),
        .o_group_weight_addr3(group_addr3),
        .i_layer_done(layer_done),
        .i_layer_sticky_error(layer_error),
        .i_array_observation_valid(scalar_valid),
        .o_array_observation_ready(scalar_ready),
        .i_array_observation_mode(scalar_mode),
        .i_array_observation_transaction_tag(scalar_tag),
        .i_array_observation_neuron_id(scalar_neuron),
        .i_array_observation_spike(scalar_spike),
        .i_array_observation_pre_state(scalar_pre),
        .i_array_observation_trace_next(scalar_trace),
        .i_array_observation_last_in_record(scalar_last_record),
        .o_hidden_valid(hidden_valid), .i_hidden_ready(hidden_ready),
        .o_hidden_path(hidden_path),
        .o_hidden_sample_id(hidden_sample), .o_hidden_id(hidden_id),
        .o_hidden_membrane_pre_q14(hidden_pre),
        .o_hidden_membrane_next_q14(hidden_next),
        .o_hidden_spike(hidden_spike),
        .o_hidden_trace_q14(hidden_trace),
        .o_hidden_output_feature_q14(hidden_feature),
        .o_hidden_last(hidden_last),
        .o_logit_valid(logit_valid), .i_logit_ready(logit_ready),
        .o_logit_sample_id(logit_sample),
        .o_logit_class_id(logit_class), .o_logit_q28(logit_q28),
        .o_logit_last(logit_last), .o_timestep_id(observed_timestep),
        .o_timestep_done(timestep_done), .o_busy(engine_busy),
        .o_sticky_error(engine_error),
        .o_protocol_error(protocol_error), .o_overflow(overflow)
    );

    unified_fc_srnn_h40_p4_layer_client_v1 #(
        .ARRAY_TILES(12), .WEIGHT_DEPTH(1156),
        .WEIGHT_ADDR_WIDTH(AW), .WORK_ADDR_WIDTH(WW)
    ) u_layer_client (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(abort_all),
        .i_clear_error(clear_error), .i_layer_valid(layer_valid),
        .o_layer_ready(layer_ready),
        .o_owner_request_valid(owner_request_valid),
        .i_owner_request_ready(owner_request_ready), .i_mode(layer_mode),
        .i_neuron_count(layer_neurons),
        .i_feedforward_group_count(layer_ff_groups),
        .i_recurrent_group_count(layer_rec_groups),
        .i_recurrent_enable(layer_rec_enable), .i_leak(layer_leak),
        .i_threshold(layer_threshold),
        .i_reset_to_zero(layer_reset_to_zero),
        .i_trace_decay(layer_trace_decay),
        .i_transaction_tag(layer_tag),
        .i_work_base_addr(layer_work_base),
        .i_destination_tile_base(layer_destination_base),
        .i_group_valid(group_valid), .o_group_ready(group_ready),
        .i_group_recurrent(group_recurrent),
        .i_group_source_valid(group_source_valid),
        .i_group_weight_addr0(group_addr0),
        .i_group_weight_addr1(group_addr1),
        .i_group_weight_addr2(group_addr2),
        .i_group_weight_addr3(group_addr3),
        .o_array_command_valid(command_valid),
        .i_array_command_ready(command_ready),
        .o_array_command_mode(command_mode),
        .o_array_command_neuron_count(command_neurons),
        .o_array_command_first(command_first),
        .o_array_command_last(command_last),
        .o_array_command_source_valid(command_source_valid),
        .o_array_command_weight_addr0(command_addr0),
        .o_array_command_weight_addr1(command_addr1),
        .o_array_command_weight_addr2(command_addr2),
        .o_array_command_weight_addr3(command_addr3),
        .o_array_command_leak(command_leak),
        .o_array_command_threshold(command_threshold),
        .o_array_command_reset_to_zero(command_reset_to_zero),
        .o_array_command_trace_decay(command_trace_decay),
        .o_array_command_transaction_tag(command_tag),
        .o_array_command_work_base_addr(command_work_base),
        .o_array_command_destination_tile_base(command_destination_base),
        .i_array_transaction_done(owner_direct_done),
        .i_array_sticky_error(array_error), .o_layer_done(layer_done),
        .o_busy(layer_busy), .o_idle(layer_idle),
        .o_sticky_error(layer_error), .o_error_code(layer_error_code)
    );

    unified_h40_array16_topology_owner_v1 u_owner (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(abort_all),
        .i_clear_error(clear_error),
        .i_maintenance_start_valid(clear_start_valid),
        .o_maintenance_start_ready(clear_start_ready),
        .i_maintenance_release_valid(clear_release_valid),
        .o_maintenance_release_ready(clear_release_ready),
        .i_maintenance_idle(array_idle && !state_write_valid &&
            !trace_write_valid),
        .i_direct_start_valid(owner_request_valid),
        .o_direct_start_ready(owner_request_ready),
        .i_direct_start_mode(layer_mode),
        .i_direct_command_valid(command_valid),
        .o_direct_command_ready(command_ready),
        .i_direct_mode(command_mode), .i_direct_first(command_first),
        .i_direct_last(command_last),
        .i_scnn_start_valid(1'b0), .o_scnn_start_ready(),
        .o_scnn_client_start_valid(), .i_scnn_client_start_ready(1'b0),
        .i_scnn_client_done(1'b0), .i_scnn_observation_sink_idle(1'b1),
        .i_array_command_ready(physical_command_ready),
        .i_array_command_done(array_done), .i_array_idle(array_idle),
        .i_direct_observation_sink_idle(!scalar_busy),
        .o_select_direct(owner_select_direct), .o_select_scnn(),
        .o_select_maintenance(owner_select_maintenance),
        .o_direct_done(owner_direct_done),
        .o_scnn_done(), .o_maintenance_done(), .o_active_owner(),
        .o_busy(owner_busy), .o_idle(owner_idle),
        .o_sticky_error(owner_error)
    );

    unified_binary_p4_h40_array12_path_v1 #(
        .WEIGHT_DEPTH(1156), .WEIGHT_ADDR_WIDTH(AW),
        .MEMORY_DEPTH(MEMORY_DEPTH), .MEMORY_ADDR_WIDTH(MW),
        .WORK_ADDR_WIDTH(WW), .CUT_RECORD_READY(1'b1)
    ) u_array (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(abort_all),
        .i_clear_error(clear_error),
        .i_command_valid(command_valid && owner_select_direct),
        .o_command_ready(physical_command_ready), .i_mode(command_mode),
        .i_neuron_count(command_neurons), .i_first(command_first),
        .i_last(command_last), .i_source_valid(command_source_valid),
        .i_scnn_spatial_count('0),
        .i_scnn_output_channel_base('0),
        .i_scnn_feature_mask0('0), .i_scnn_feature_mask1('0),
        .i_scnn_feature_mask2('0), .i_scnn_feature_mask3('0),
        .i_scnn_factor_b(18'sd256),
        .i_scnn_active_pair(1'b0), .i_scnn_weight_subword0('0),
        .i_scnn_weight_subword1('0), .i_scnn_weight_subword2('0),
        .i_scnn_weight_subword3('0),
        .i_scnn_work_base_addr(command_work_base),
        .i_destination_tile_base(command_destination_base),
        .i_weight_addr0(command_addr0), .i_weight_addr1(command_addr1),
        .i_weight_addr2(command_addr2), .i_weight_addr3(command_addr3),
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
        .o_observation_valid(raw_observation_valid),
        .i_observation_ready(raw_observation_ready &
            {4{owner_select_direct}}),
        .o_observation_tile_flat(raw_observation_tile_flat),
        .o_observation_mode_flat(raw_observation_mode_flat),
        .o_observation_lane_mask_flat(raw_observation_lane_mask_flat),
        .o_observation_range_mask_flat(raw_observation_range_mask_flat),
        .o_observation_spike_flat(raw_observation_spike_flat),
        .o_observation_pre_state_flat(raw_observation_pre_state_flat),
        .o_observation_trace_next_flat(raw_observation_trace_next_flat),
        .o_observation_meta_flat(raw_observation_meta_flat),
        .i_maintenance_tile(maintenance_tile),
        .i_state_maintenance_write_valid(state_write_valid &&
            owner_select_maintenance),
        .o_state_maintenance_write_ready(state_write_ready),
        .i_state_maintenance_write_addr(state_write_addr),
        .i_state_maintenance_write_data(state_write_data),
        .i_state_maintenance_write_mask(state_write_mask),
        .i_state_maintenance_read_valid(1'b0),
        .o_state_maintenance_read_ready(),
        .i_state_maintenance_read_addr('0),
        .o_state_maintenance_result_valid(),
        .i_state_maintenance_result_ready(1'b1),
        .o_state_maintenance_result_addr(),
        .o_state_maintenance_read_data(),
        .i_trace_maintenance_write_valid(trace_write_valid &&
            owner_select_maintenance),
        .o_trace_maintenance_write_ready(trace_write_ready),
        .i_trace_maintenance_write_addr(trace_write_addr),
        .i_trace_maintenance_write_data(trace_write_data),
        .i_trace_maintenance_write_mask(trace_write_mask),
        .i_trace_maintenance_read_valid(1'b0),
        .o_trace_maintenance_read_ready(),
        .i_trace_maintenance_read_addr('0),
        .o_trace_maintenance_result_valid(),
        .i_trace_maintenance_result_ready(1'b1),
        .o_trace_maintenance_result_addr(),
        .o_trace_maintenance_read_data(), .o_busy(array_busy),
        .o_done(array_done), .o_idle(array_idle),
        .o_tile_done(tile_done), .o_tile_idle(tile_idle),
        .o_tile_sticky_error(tile_error), .o_sticky_error(array_error)
    );

    unified_fc_srnn_h40_tp_observation_serializer_v1 u_serializer (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(abort_all),
        .i_clear_error(clear_error),
        .i_observation_valid(raw_observation_valid &
            {4{owner_select_direct}}),
        .o_observation_ready(raw_observation_ready),
        .i_observation_tile_flat(raw_observation_tile_flat),
        .i_observation_mode_flat(raw_observation_mode_flat),
        .i_observation_lane_mask_flat(raw_observation_lane_mask_flat),
        .i_observation_range_mask_flat(raw_observation_range_mask_flat),
        .i_observation_spike_flat(raw_observation_spike_flat),
        .i_observation_pre_state_flat(raw_observation_pre_state_flat),
        .i_observation_trace_next_flat(raw_observation_trace_next_flat),
        .i_observation_meta_flat(raw_observation_meta_flat),
        .o_scalar_valid(scalar_valid), .i_scalar_ready(scalar_ready),
        .o_scalar_mode(scalar_mode),
        .o_scalar_transaction_tag(scalar_tag),
        .o_scalar_neuron_id(scalar_neuron),
        .o_scalar_spike(scalar_spike),
        .o_scalar_pre_state(scalar_pre),
        .o_scalar_trace_next(scalar_trace),
        .o_scalar_last_in_record(scalar_last_record),
        .o_busy(scalar_busy), .o_sticky_error(scalar_error)
    );


    // Read-only TB backing memory; this stage does not instantiate the learner
    // or shared RMW store and must not be reported as final-top online closure.
    string fixture, expectation_prefix;
    logic learning_enable = 1'b0;
    `include "tb/include/srnn_formal_m2_online_body.svh"
    integer steps = 2;
    integer cycles = 0, parameter_reads = 0, commands = 0, source_spikes = 0;
    integer formal_pre_differences = 0;
    integer prepare_cycles = 0, run_start_cycles = 0;
    logic signed [31:0] input_mem [0:128*2*450-1];
    logic signed [31:0] expected_pre [0:128*2*256-1];
    logic signed [31:0] expected_next [0:128*2*256-1];
    logic signed [31:0] expected_spike [0:128*2*256-1];
    logic signed [31:0] expected_trace [0:128*2*256-1];
    logic signed [31:0] expected_feature [0:128*2*256-1];
    logic signed [31:0] formal_pre [0:128*2*256-1];
    logic signed [63:0] expected_logit [0:128*2*20-1];

    task automatic send_observation(input int sample_id, input int source_id);
        @(negedge clk);
        observation_path = 1'b0;
        observation_sample = SAMPLE_ID_WIDTH'(sample_id);
        observation_backbone = 9'(source_id);
        observation_q14 = input_mem[(int'(timestep_id)*2+sample_id)*450+source_id];
        observation_last = sample_id == 1 && source_id == 449;
        observation_valid = 1'b1;
        do @(posedge clk); while (!(WITH_COORDINATOR ? c_external_ready : observation_ready));
        @(negedge clk);
        observation_valid = 1'b0;
    endtask

    always @(posedge clk) begin
        if (rst_n) begin
            cycles <= cycles + 1;
            if (refill_read_valid && refill_read_ready)
                parameter_reads <= parameter_reads + 1;
            if (command_valid && command_ready)
                commands <= commands + 1;
            if (observation_valid && (WITH_COORDINATOR ? c_external_ready : observation_ready) && observation_q14 != 0)
                source_spikes <= source_spikes + 1;
            if (hidden_valid && hidden_ready) begin
                if (hidden_path != 0 ||
                    observed_timestep != hidden_seen / 512 ||
                    hidden_sample != (hidden_seen % 512) / 256 ||
                    hidden_id != hidden_seen % 256 ||
                    hidden_pre != expected_pre[hidden_seen] ||
                    hidden_next != expected_next[hidden_seen] ||
                    hidden_spike != expected_spike[hidden_seen] ||
                    hidden_trace != expected_trace[hidden_seen] ||
                    hidden_feature != expected_feature[hidden_seen] ||
                    hidden_last != (hidden_seen % 512 == 511))
                    $fatal(1, "M2_HIDDEN_MISMATCH index=%0d t=%0d sample=%0d hidden=%0d pre=%0d expected_pre=%0d next=%0d expected_next=%0d trace=%0d expected_trace=%0d",
                        hidden_seen, observed_timestep, hidden_sample, hidden_id,
                        hidden_pre, expected_pre[hidden_seen], hidden_next, expected_next[hidden_seen],
                        hidden_trace, expected_trace[hidden_seen]);
                if (hidden_pre != formal_pre[hidden_seen]) begin
                    if (formal_pre_differences == 0)
                        $display("SRNN_FORMAL_Q14_FIRST_DIFF index=%0d t=%0d sample=%0d hidden=%0d production=%0d formal=%0d",
                            hidden_seen, observed_timestep, hidden_sample, hidden_id,
                            hidden_pre, formal_pre[hidden_seen]);
                    formal_pre_differences <= formal_pre_differences + 1;
                end
                hidden_seen <= hidden_seen + 1;
            end
            if (logit_valid && logit_ready) begin
                if (observed_timestep != logit_seen / 40 ||
                    logit_sample != (logit_seen % 40) / 20 ||
                    logit_class != logit_seen % 20 ||
                    logit_q28 != expected_logit[logit_seen] ||
                    logit_last != (logit_seen % 40 == 39))
                    $fatal(1, "M2_LOGIT_MISMATCH index=%0d actual=%0d expected=%0d",
                           logit_seen, logit_q28, expected_logit[logit_seen]);
                logit_seen <= logit_seen + 1;
            end
            if (engine_error || protocol_error || overflow || layer_error || array_error ||
                scalar_error || owner_error || tile_error != 0)
                $fatal(1, "M2_ERROR engine=%0b protocol=%0b overflow=%0b layer=%0b code=%0h array=%0b scalar=%0b owner=%0b tiles=%0h prep=%0d adapter=%0d reads=%0d refill_error=%0d clear_error=%0d local_error=%0d",
                       engine_error, protocol_error, overflow, layer_error, layer_error_code,
                       array_error, scalar_error, owner_error, tile_error,
                       u_engine.prep_state_q, u_engine.u_adapter.state_q, parameter_reads,
                       u_engine.refill_error, u_engine.clear_error, u_engine.local_error_q);
        end
    end

    initial begin
        if (!$value$plusargs("FIXTURE=%s", fixture))
            $fatal(1, "FIXTURE is required");
        if ($value$plusargs("STEPS=%d", steps)) begin end
        if (steps < 2 || steps > 128) $fatal(1, "STEPS outside 2..128");
        $readmemh("tb/data/srnn_m2_v28_wh1_initial_q14.mem", model_mem, 0, 115199);
        $readmemh("tb/data/srnn_m2_v28_wout_initial_q14.mem", model_mem, 115200, 120319);
        $readmemh({fixture, "/input_q14.mem"}, input_mem, 0, steps*900-1);
        if ($value$plusargs("LEARNING=%d", learning_enable)) begin end
        inject_timestep_id_error = $test$plusargs("INJECT_TIMESTEP_ID_ERROR");
        inject_done_timestep_id_error = $test$plusargs("INJECT_DONE_TIMESTEP_ID_ERROR");
        if (learning_enable && !WITH_COORDINATOR) $fatal(1, "LEARNING requires WITH_COORDINATOR");
        expectation_prefix = learning_enable ? "/expected_on_" : "/expected_";
        $readmemh({fixture, expectation_prefix, "pre.mem"}, expected_pre, 0, steps*512-1);
        $readmemh({fixture, expectation_prefix, "membrane.mem"}, expected_next, 0, steps*512-1);
        $readmemh({fixture, expectation_prefix, "spike.mem"}, expected_spike, 0, steps*512-1);
        $readmemh({fixture, expectation_prefix, "trace.mem"}, expected_trace, 0, steps*512-1);
        $readmemh({fixture, expectation_prefix, "feature.mem"}, expected_feature, 0, steps*512-1);
        $readmemh({fixture, "/formal_pre.mem"}, formal_pre, 0, steps*512-1);
        $readmemh({fixture, expectation_prefix, "logit.mem"}, expected_logit, 0, steps*40-1);
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);
        cfg_valid = 1'b1;
        do @(posedge clk); while (!(WITH_COORDINATOR ? c_cfg_ready : cfg_ready));
        @(negedge clk);
        cfg_valid = 1'b0;
        if (cfg_error || (WITH_COORDINATOR && c_cfg_error)) $fatal(1, "configuration failed");
        if (WITH_COORDINATOR) begin
`ifdef SRNN_FORMAL_M2_ONLINE
            g_online.load_real_parameters();
`endif
        end else begin
        wait (prepare_ready);
        @(negedge clk);
        prepare_valid = 1'b1;
        @(posedge clk);
        @(negedge clk);
        prepare_valid = 1'b0;
        wait (prepare_done);
        @(negedge clk);
        prepare_cycles = cycles;
        if (!cache_valid || cache_version != sequence_version)
            $fatal(1, "cache preparation failed");
        $display("SRNN_FORMAL_M2_PREPARED cycles=%0d parameter_reads=%0d", cycles, parameter_reads);
        end
        run_start_cycles = cycles;
        for (int t = 0; t < steps; t++) begin
            wait (WITH_COORDINATOR ? c_start_ready : timestep_start_ready);
            @(negedge clk);
            timestep_id = 7'(t);
            timestep_start = 1'b1;
            @(posedge clk);
            @(negedge clk);
            timestep_start = 1'b0;
            for (int sample_id = 0; sample_id < 2; sample_id++)
                for (int source_id = 0; source_id < 450; source_id++)
                    send_observation(sample_id, source_id);
            @(negedge clk);
            observation_done = 1'b1;
            @(posedge clk);
            @(negedge clk);
            observation_done = 1'b0;
            observation_last = 1'b0;
            wait (WITH_COORDINATOR ? c_done : timestep_done);
            @(negedge clk);
`ifdef SRNN_FORMAL_M2_ONLINE
            if (WITH_COORDINATOR) g_online.check_store_after(t);
`endif
            $display("SRNN_FORMAL_M2_STEP timestep=%0d cycles=%0d hidden=%0d logits=%0d",
                     t, cycles, hidden_seen, logit_seen);
        end
        if (hidden_seen != steps*512 || logit_seen != steps*40 || (!WITH_COORDINATOR && parameter_reads != 120320))
            $fatal(1, "incomplete real-model checks hidden=%0d logits=%0d reads=%0d",
                   hidden_seen, logit_seen, parameter_reads);
        if (WITH_COORDINATOR) begin
            if (learning_enable && (commit_count != steps || detach_count != steps || gradient_checks != steps*5120))
                $fatal(1, "online incomplete commits=%0d detach=%0d gradients=%0d", commit_count, detach_count, gradient_checks);
            if (!learning_enable && (commit_count != 0 || rmw_reads != 0 || code_writes != 0 || residual_writes != 0))
                $fatal(1, "learning-off mutated store");
            $display("SRNN_FORMAL_M2_ONLINE_PASS batch=2 backbone=450 hidden=256 classes=20 steps=%0d learning=%0d parameter_reads=%0d total_cycles=%0d hidden_checks=%0d logit_checks=%0d gradient_checks=%0d commits=%0d detach=%0d rmw_reads=%0d code_writes=%0d residual_writes=%0d changed_readbacks=%0d frozen_wh1_checks=%0d store_element_checks=%0d model_version=%0d adapt_version=%0d formal_pre_differences=%0d start_cycle=%0d end_cycle=%0d sample_id_width=8 first_code_step=%0d first_changed_read_step=%0d first_logit_change_step=%0d changed_logit_checks=%0d prepare_stale_id_cycles=%0d",
                steps, learning_enable, parameter_reads, cycles-run_start_cycles, hidden_seen, logit_seen,
                gradient_checks, commit_count, detach_count, rmw_reads, code_writes, residual_writes,
                changed_readbacks, frozen_wh1_checks, store_element_checks, c_model_version,
                c_adapt_version, formal_pre_differences, run_start_cycles, cycles,
                first_code_step, first_changed_read_step, first_logit_change_step, changed_logit_checks, prepare_stale_id_cycles);
        end else
        $display("SRNN_FORMAL_M2_FORWARD_PASS batch=2 backbone=450 hidden=256 classes=20 steps=%0d parameter_reads=%0d prepare_cycles=%0d forward_cycles=%0d hidden_checks=%0d logit_checks=%0d commands=%0d source_spikes=%0d formal_pre_differences=%0d",
            steps, parameter_reads, prepare_cycles, cycles-run_start_cycles,
            hidden_seen, logit_seen, commands, source_spikes, formal_pre_differences);
        $finish;
    end
    initial begin
        repeat (20000000) @(posedge clk);
        $fatal(1, "timeout prep=%0d state=%0d hidden=%0d logits=%0d",
               u_engine.prep_state_q, u_engine.u_adapter.state_q, hidden_seen, logit_seen);
    end
endmodule

// Arithmetic-only minimal reproducer; it is not a reduced-size model run.
module tb_srnn_formal_m2_lif_probe;
    logic clk = 0, rst_n = 0, valid = 0, ready, out_valid;
    logic [9:0] spike;
    logic [319:0] pre_state, next_state;
    always #5 clk = ~clk;
    unified_dual_role_lif10_arithmetic_candidate_v3 u_lif (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(1'b0),
        .i_valid(valid), .o_ready(ready), .i_mode(2'd3),
        .i_state_flat(320'd16384), .i_current_flat(320'd0),
        .i_leak(16'd13926), .i_threshold(32'd8192), .i_reset_to_zero(1'b0),
        .o_valid(out_valid), .i_ready(1'b1), .o_spike(spike),
        .o_pre_state_flat(pre_state), .o_next_state_flat(next_state)
    );
    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
        valid = 1;
        do @(posedge clk); while (!ready);
        @(negedge clk);
        valid = 0;
        wait (out_valid);
        @(negedge clk);
        if (pre_state != 320'd13926 || spike != 1 || next_state != 320'd5734)
            $fatal(1, "LIF current-source expectation changed: pre=%0d spike=%0d", pre_state, spike);
        $display("SRNN_M2_LIF_Q14_PASS state=16384 current=0 leak=13926 production_pre=%0d formal_pre=13926 production_spike=%0d formal_spike=1",
                 pre_state[31:0], spike[0]);
        $finish;
    end
    initial begin
        repeat (100) @(posedge clk);
        $fatal(1, "LIF probe timeout");
    end
endmodule
`default_nettype wire

