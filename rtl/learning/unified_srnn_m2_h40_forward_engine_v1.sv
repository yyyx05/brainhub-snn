`timescale 1ns/1ps
`default_nettype none

// SRNN M2 topology adapter for the shared H40 physical array.  Parameters
// remain authoritative in the common scalar store; this engine publishes the
// current model version into the array's Q2.14 working-set cache, clears only
// sequence-local state, and translates M2 observations into shared layer work.
module unified_srnn_m2_h40_forward_engine_v1 #(
    parameter int PARAM_DEPTH = 120320,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int BATCH_MAX = 64,
    parameter int TIMESTEP_MAX = 128,
    parameter int BACKBONE_MAX = 450,
    parameter int HIDDEN_MAX = 256,
    parameter int CLASS_MAX = 20,
    parameter int SAMPLE_ID_WIDTH = 8,
    parameter int TIMESTEP_ID_WIDTH = $clog2(TIMESTEP_MAX),
    parameter int BACKBONE_INDEX_WIDTH = 10,
    parameter int HIDDEN_INDEX_WIDTH = 9,
    parameter int CLASS_INDEX_WIDTH = 5,
    parameter int VERSION_WIDTH = 16,
    parameter int WEIGHT_DEPTH = 1156,
    parameter int WEIGHT_ADDR_WIDTH = $clog2(WEIGHT_DEPTH),
    parameter int MEMORY_DEPTH = 1024,
    parameter int MEMORY_ADDR_WIDTH = $clog2(MEMORY_DEPTH),
    parameter int WORK_ADDR_WIDTH = MEMORY_ADDR_WIDTH - 1,
    parameter int PHYSICAL_TILE_COUNT = 12,
    parameter int OUTPUT_TILE = 7
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,
    input  logic                                      i_abort,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_cfg_batch_count,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_cfg_backbone_count,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]             i_cfg_hidden_count,
    input  logic [CLASS_INDEX_WIDTH-1:0]              i_cfg_class_count,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_wh1_parameter_base,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_wout_parameter_base,
    input  logic [9:0]                                i_cfg_wh1_cache_base,
    input  logic [9:0]                                i_cfg_wout_cache_base,
    input  logic [WORK_ADDR_WIDTH-1:0]                i_cfg_hidden_work_base,
    input  logic [WORK_ADDR_WIDTH-1:0]                i_cfg_output_work_base,
    output logic                                      o_cfg_error,

    input  logic                                      i_prepare_valid,
    output logic                                      o_prepare_ready,
    input  logic [VERSION_WIDTH-1:0]                  i_prepare_version,
    input  logic                                      i_cache_invalidate,
    output logic                                      o_prepare_done,
    output logic                                      o_cache_valid,
    output logic [VERSION_WIDTH-1:0]                  o_cache_version,

    input  logic                                      i_timestep_start,
    output logic                                      o_timestep_start_ready,
    input  logic [TIMESTEP_ID_WIDTH-1:0]              i_timestep_id,
    input  logic [VERSION_WIDTH-1:0]                  i_sequence_version,
    input  logic                                      i_include_target_path,

    input  logic                                      i_observation_valid,
    output logic                                      o_observation_ready,
    input  logic                                      i_observation_path,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_observation_sample_id,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_observation_backbone_id,
    input  logic signed [31:0]                        i_observation_q14,
    input  logic                                      i_observation_last,
    input  logic                                      i_observation_done,

    output logic                                      o_forward_read_valid,
    input  logic                                      i_forward_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_forward_read_addr,
    output logic [TAG_WIDTH-1:0]                      o_forward_read_tag,
    input  logic                                      i_forward_response_valid,
    output logic                                      o_forward_response_ready,
    input  logic [TAG_WIDTH-1:0]                      i_forward_response_tag,
    input  logic signed [15:0]                        i_forward_response_code,
    input  logic                                      i_forward_response_error,

    input  logic                                      i_array_idle,
    output logic [3:0]                                o_weight_fill_valid,
    input  logic [3:0]                                i_weight_fill_ready,
    output logic [7:0]                                o_weight_fill_tile_flat,
    output logic [3:0]                                o_weight_fill_bank,
    output logic [4*WEIGHT_ADDR_WIDTH-1:0]            o_weight_fill_addr_flat,
    output logic [799:0]                              o_weight_fill_data_flat,

    output logic                                      o_maintenance_start_valid,
    input  logic                                      i_maintenance_start_ready,
    output logic                                      o_maintenance_release_valid,
    input  logic                                      i_maintenance_release_ready,
    output logic [3:0]                                o_maintenance_tile,
    output logic                                      o_state_write_valid,
    input  logic                                      i_state_write_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]              o_state_write_addr,
    output logic [319:0]                              o_state_write_data,
    output logic [39:0]                               o_state_write_mask,
    output logic                                      o_trace_write_valid,
    input  logic                                      i_trace_write_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]              o_trace_write_addr,
    output logic [319:0]                              o_trace_write_data,
    output logic [39:0]                               o_trace_write_mask,

    output logic                                      o_layer_valid,
    input  logic                                      i_layer_ready,
    output logic [1:0]                                o_layer_mode,
    output logic [15:0]                               o_layer_neuron_count,
    output logic [15:0]                               o_layer_feedforward_group_count,
    output logic [15:0]                               o_layer_recurrent_group_count,
    output logic                                      o_layer_recurrent_enable,
    output logic [15:0]                               o_layer_leak,
    output logic [31:0]                               o_layer_threshold,
    output logic                                      o_layer_reset_to_zero,
    output logic [15:0]                               o_layer_trace_decay,
    output logic [51:0]                               o_layer_transaction_tag,
    output logic [WORK_ADDR_WIDTH-1:0]                o_layer_work_base_addr,
    output logic [3:0]                                o_layer_destination_tile_base,
    output logic                                      o_group_valid,
    input  logic                                      i_group_ready,
    output logic                                      o_group_recurrent,
    output logic [3:0]                                o_group_source_valid,
    output logic [WEIGHT_ADDR_WIDTH-1:0]              o_group_weight_addr0,
    output logic [WEIGHT_ADDR_WIDTH-1:0]              o_group_weight_addr1,
    output logic [WEIGHT_ADDR_WIDTH-1:0]              o_group_weight_addr2,
    output logic [WEIGHT_ADDR_WIDTH-1:0]              o_group_weight_addr3,
    input  logic                                      i_layer_done,
    input  logic                                      i_layer_sticky_error,

    input  logic                                      i_array_observation_valid,
    output logic                                      o_array_observation_ready,
    input  logic [1:0]                                i_array_observation_mode,
    input  logic [51:0]                               i_array_observation_transaction_tag,
    input  logic [9:0]                                i_array_observation_neuron_id,
    input  logic                                      i_array_observation_spike,
    input  logic signed [31:0]                        i_array_observation_pre_state,
    input  logic signed [31:0]                        i_array_observation_trace_next,
    input  logic                                      i_array_observation_last_in_record,

    output logic                                      o_hidden_valid,
    input  logic                                      i_hidden_ready,
    output logic                                      o_hidden_path,
    output logic [SAMPLE_ID_WIDTH-1:0]                o_hidden_sample_id,
    output logic [HIDDEN_INDEX_WIDTH-1:0]             o_hidden_id,
    output logic signed [31:0]                        o_hidden_membrane_pre_q14,
    output logic signed [31:0]                        o_hidden_membrane_next_q14,
    output logic                                      o_hidden_spike,
    output logic signed [31:0]                        o_hidden_trace_q14,
    output logic signed [31:0]                        o_hidden_output_feature_q14,
    output logic                                      o_hidden_last,
    output logic                                      o_logit_valid,
    input  logic                                      i_logit_ready,
    output logic [SAMPLE_ID_WIDTH-1:0]                o_logit_sample_id,
    output logic [CLASS_INDEX_WIDTH-1:0]              o_logit_class_id,
    output logic signed [63:0]                        o_logit_q28,
    output logic                                      o_logit_last,
    output logic [TIMESTEP_ID_WIDTH-1:0]              o_timestep_id,
    output logic                                      o_timestep_done,
    output logic                                      o_busy,
    output logic                                      o_sticky_error,
    output logic                                      o_protocol_error,
    output logic                                      o_overflow
);
    localparam int REFILL_SOURCE_WIDTH =
        BACKBONE_INDEX_WIDTH > HIDDEN_INDEX_WIDTH ?
            BACKBONE_INDEX_WIDTH : HIDDEN_INDEX_WIDTH;

    typedef enum logic [2:0] {
        PREP_IDLE,
        PREP_WH1_ISSUE,
        PREP_WH1_WAIT,
        PREP_WOUT_ISSUE,
        PREP_WOUT_WAIT
    } prep_state_t;

    prep_state_t prep_state_q;
    logic configured_q;
    logic local_error_q;
    logic [PARAM_ADDR_WIDTH-1:0] wh1_parameter_base_q;
    logic [PARAM_ADDR_WIDTH-1:0] wout_parameter_base_q;
    logic [9:0] wh1_cache_base_q;
    logic [9:0] wout_cache_base_q;
    logic [BACKBONE_INDEX_WIDTH-1:0] backbone_count_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] hidden_count_q;
    logic [CLASS_INDEX_WIDTH-1:0] class_count_q;
    logic [VERSION_WIDTH-1:0] prepare_version_q;

    logic adapter_cfg_ready;
    logic adapter_cfg_error;
    logic adapter_start_ready;
    logic adapter_busy;
    logic state_clear_required;
    logic state_clear_complete;
    logic [PHYSICAL_TILE_COUNT-1:0] clear_tile_mask;
    logic [WORK_ADDR_WIDTH-1:0] clear_hidden_base;
    logic [WORK_ADDR_WIDTH:0] clear_hidden_rows;
    logic [WORK_ADDR_WIDTH-1:0] clear_output_base;
    logic [2:0] clear_output_rows;
    logic clear_busy;
    logic clear_error;

    logic refill_valid;
    logic refill_ready;
    logic refill_done;
    logic refill_busy;
    logic refill_error;
    logic [VERSION_WIDTH-1:0] refill_version;
    logic refill_fire;
    logic cfg_fire;
    logic prepare_fire;
    logic adapter_start;

    assign o_cfg_ready = prep_state_q == PREP_IDLE && !refill_busy &&
        !clear_busy && !adapter_busy && adapter_cfg_ready && !local_error_q;
    assign cfg_fire = i_cfg_valid && o_cfg_ready;
    assign o_cfg_error = adapter_cfg_error;

    assign o_prepare_ready = prep_state_q == PREP_IDLE && configured_q &&
        !adapter_busy && !clear_busy && !o_sticky_error && i_array_idle &&
        !i_cache_invalidate;
    assign prepare_fire = i_prepare_valid && o_prepare_ready;
    assign refill_valid = prep_state_q == PREP_WH1_ISSUE ||
        prep_state_q == PREP_WOUT_ISSUE;
    assign refill_fire = refill_valid && refill_ready;

    assign adapter_start = i_timestep_start && o_timestep_start_ready;
    assign o_timestep_start_ready = prep_state_q == PREP_IDLE &&
        o_cache_valid && !i_cache_invalidate && adapter_start_ready &&
        !o_sticky_error;
    assign o_busy = prep_state_q != PREP_IDLE || refill_busy || clear_busy ||
        adapter_busy;
    assign o_sticky_error = local_error_q || refill_error || clear_error ||
        i_layer_sticky_error;

    unified_srnn_m2_h40_q14_cache_refill_scheduler_v1 #(
        .PARAM_DEPTH(PARAM_DEPTH), .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH), .VERSION_WIDTH(VERSION_WIDTH),
        .SOURCE_COUNT_WIDTH(REFILL_SOURCE_WIDTH),
        .DESTINATION_COUNT_WIDTH(HIDDEN_INDEX_WIDTH),
        .ARRAY_WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .ARRAY_TILES(PHYSICAL_TILE_COUNT)
    ) u_refill (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error), .i_refill_valid(refill_valid),
        .o_refill_ready(refill_ready), .i_refill_all(1'b1),
        .i_dirty_source('0), .i_dirty_destination_tile('0),
        .i_parameter_base(prep_state_q == PREP_WH1_ISSUE ||
            prep_state_q == PREP_WH1_WAIT ? wh1_parameter_base_q :
            wout_parameter_base_q),
        .i_cache_row_base(prep_state_q == PREP_WH1_ISSUE ||
            prep_state_q == PREP_WH1_WAIT ? wh1_cache_base_q :
            wout_cache_base_q),
        .i_source_count(prep_state_q == PREP_WH1_ISSUE ||
            prep_state_q == PREP_WH1_WAIT ?
            REFILL_SOURCE_WIDTH'(backbone_count_q) :
            REFILL_SOURCE_WIDTH'(hidden_count_q)),
        .i_destination_count(prep_state_q == PREP_WH1_ISSUE ||
            prep_state_q == PREP_WH1_WAIT ? hidden_count_q :
            HIDDEN_INDEX_WIDTH'(class_count_q)),
        .i_destination_tile_base(prep_state_q == PREP_WH1_ISSUE ||
            prep_state_q == PREP_WH1_WAIT ? 4'd0 : 4'(OUTPUT_TILE)),
        .i_version(prepare_version_q), .i_array_idle(i_array_idle),
        .o_forward_read_valid(o_forward_read_valid),
        .i_forward_read_ready(i_forward_read_ready),
        .o_forward_read_addr(o_forward_read_addr),
        .o_forward_read_tag(o_forward_read_tag),
        .i_forward_response_valid(i_forward_response_valid),
        .o_forward_response_ready(o_forward_response_ready),
        .i_forward_response_tag(i_forward_response_tag),
        .i_forward_response_code(i_forward_response_code),
        .i_forward_response_error(i_forward_response_error),
        .o_weight_fill_valid(o_weight_fill_valid),
        .i_weight_fill_ready(i_weight_fill_ready),
        .o_weight_fill_tile_flat(o_weight_fill_tile_flat),
        .o_weight_fill_bank(o_weight_fill_bank),
        .o_weight_fill_addr_flat(o_weight_fill_addr_flat),
        .o_weight_fill_data_flat(o_weight_fill_data_flat),
        .o_refill_done(refill_done), .o_rows_published(),
        .o_published_version(refill_version), .o_busy(refill_busy),
        .o_sticky_error(refill_error)
    );

    unified_srnn_m2_h40_state_clear_sequencer_v1 #(
        .ARRAY_TILES(PHYSICAL_TILE_COUNT), .MEMORY_DEPTH(MEMORY_DEPTH),
        .TILE_INDEX_WIDTH(4), .MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH)
    ) u_state_clear (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error), .i_clear_valid(state_clear_required),
        .o_clear_ready(), .i_tile_mask(clear_tile_mask),
        .i_output_tile(4'(OUTPUT_TILE)), .i_hidden_base(clear_hidden_base),
        .i_hidden_row_count(MEMORY_ADDR_WIDTH'(clear_hidden_rows)),
        .i_output_base(clear_output_base),
        .i_output_row_count(MEMORY_ADDR_WIDTH'(clear_output_rows)),
        .o_maintenance_start_valid(o_maintenance_start_valid),
        .i_maintenance_start_ready(i_maintenance_start_ready),
        .o_maintenance_release_valid(o_maintenance_release_valid),
        .i_maintenance_release_ready(i_maintenance_release_ready),
        .o_maintenance_tile(o_maintenance_tile),
        .o_state_write_valid(o_state_write_valid),
        .i_state_write_ready(i_state_write_ready),
        .o_state_write_addr(o_state_write_addr),
        .o_state_write_data(o_state_write_data),
        .o_state_write_mask(o_state_write_mask),
        .o_trace_write_valid(o_trace_write_valid),
        .i_trace_write_ready(i_trace_write_ready),
        .o_trace_write_addr(o_trace_write_addr),
        .o_trace_write_data(o_trace_write_data),
        .o_trace_write_mask(o_trace_write_mask),
        .o_done(state_clear_complete), .o_busy(clear_busy),
        .o_sticky_error(clear_error)
    );

    unified_srnn_m2_h40_forward_observation_adapter_v1 #(
        .BATCH_MAX(BATCH_MAX), .TIMESTEP_MAX(TIMESTEP_MAX),
        .BACKBONE_MAX(BACKBONE_MAX), .HIDDEN_MAX(HIDDEN_MAX),
        .CLASS_MAX(CLASS_MAX), .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .TIMESTEP_ID_WIDTH(TIMESTEP_ID_WIDTH),
        .BACKBONE_INDEX_WIDTH(BACKBONE_INDEX_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .CLASS_INDEX_WIDTH(CLASS_INDEX_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH),
        .HALF_STATE_DEPTH(MEMORY_DEPTH/2),
        .PHYSICAL_TILE_COUNT(PHYSICAL_TILE_COUNT),
        .OUTPUT_TILE(OUTPUT_TILE)
    ) u_adapter (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_abort_timestep(i_abort), .i_cfg_valid(cfg_fire),
        .o_cfg_ready(adapter_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_backbone_count(i_cfg_backbone_count),
        .i_cfg_hidden_count(i_cfg_hidden_count),
        .i_cfg_class_count(i_cfg_class_count),
        .i_cfg_wh1_cache_base(WEIGHT_ADDR_WIDTH'(i_cfg_wh1_cache_base)),
        .i_cfg_wout_cache_base(WEIGHT_ADDR_WIDTH'(i_cfg_wout_cache_base)),
        .i_cfg_hidden_work_base(i_cfg_hidden_work_base),
        .i_cfg_output_work_base(i_cfg_output_work_base),
        .o_cfg_error(adapter_cfg_error), .i_timestep_start(adapter_start),
        .o_timestep_start_ready(adapter_start_ready),
        .i_timestep_id(i_timestep_id),
        .i_sequence_version(i_sequence_version),
        .i_include_target_path(i_include_target_path),
        .o_state_clear_required(state_clear_required),
        .i_state_clear_complete(state_clear_complete),
        .o_state_clear_sequence_version(),
        .o_state_clear_tile_mask(clear_tile_mask),
        .o_state_clear_hidden_work_base(clear_hidden_base),
        .o_state_clear_hidden_row_count(clear_hidden_rows),
        .o_state_clear_output_work_base(clear_output_base),
        .o_state_clear_output_row_count(clear_output_rows),
        .i_observation_valid(i_observation_valid),
        .o_observation_ready(o_observation_ready),
        .i_observation_path(i_observation_path),
        .i_observation_sample_id(i_observation_sample_id),
        .i_observation_backbone_id(i_observation_backbone_id),
        .i_observation_q14(i_observation_q14),
        .i_observation_last(i_observation_last),
        .i_observation_done(i_observation_done),
        .o_layer_valid(o_layer_valid), .i_layer_ready(i_layer_ready),
        .o_layer_mode(o_layer_mode),
        .o_layer_neuron_count(o_layer_neuron_count),
        .o_layer_feedforward_group_count(o_layer_feedforward_group_count),
        .o_layer_recurrent_group_count(o_layer_recurrent_group_count),
        .o_layer_recurrent_enable(o_layer_recurrent_enable),
        .o_layer_leak(o_layer_leak),
        .o_layer_threshold(o_layer_threshold),
        .o_layer_reset_to_zero(o_layer_reset_to_zero),
        .o_layer_trace_decay(o_layer_trace_decay),
        .o_layer_transaction_tag(o_layer_transaction_tag),
        .o_layer_work_base_addr(o_layer_work_base_addr),
        .o_layer_destination_tile_base(o_layer_destination_tile_base),
        .o_group_valid(o_group_valid), .i_group_ready(i_group_ready),
        .o_group_recurrent(o_group_recurrent),
        .o_group_source_valid(o_group_source_valid),
        .o_group_weight_addr0(o_group_weight_addr0),
        .o_group_weight_addr1(o_group_weight_addr1),
        .o_group_weight_addr2(o_group_weight_addr2),
        .o_group_weight_addr3(o_group_weight_addr3),
        .i_layer_done(i_layer_done),
        .i_layer_sticky_error(i_layer_sticky_error),
        .i_array_observation_valid(i_array_observation_valid),
        .o_array_observation_ready(o_array_observation_ready),
        .i_array_observation_mode(i_array_observation_mode),
        .i_array_observation_transaction_tag(
            i_array_observation_transaction_tag),
        .i_array_observation_neuron_id(i_array_observation_neuron_id),
        .i_array_observation_spike(i_array_observation_spike),
        .i_array_observation_pre_state(i_array_observation_pre_state),
        .i_array_observation_trace_next(i_array_observation_trace_next),
        .i_array_observation_last_in_record(
            i_array_observation_last_in_record),
        .o_hidden_valid(o_hidden_valid), .i_hidden_ready(i_hidden_ready),
        .o_hidden_path(o_hidden_path),
        .o_hidden_sample_id(o_hidden_sample_id), .o_hidden_id(o_hidden_id),
        .o_hidden_membrane_pre_q14(o_hidden_membrane_pre_q14),
        .o_hidden_membrane_next_q14(o_hidden_membrane_next_q14),
        .o_hidden_spike(o_hidden_spike),
        .o_hidden_trace_q14(o_hidden_trace_q14),
        .o_hidden_output_feature_q14(o_hidden_output_feature_q14),
        .o_hidden_last(o_hidden_last), .o_logit_valid(o_logit_valid),
        .i_logit_ready(i_logit_ready),
        .o_logit_sample_id(o_logit_sample_id),
        .o_logit_class_id(o_logit_class_id), .o_logit_q28(o_logit_q28),
        .o_logit_last(o_logit_last), .o_timestep_id(o_timestep_id),
        .o_timestep_done(o_timestep_done), .o_busy(adapter_busy),
        .o_protocol_error(o_protocol_error), .o_overflow(o_overflow)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            prep_state_q <= PREP_IDLE;
            configured_q <= 1'b0;
            local_error_q <= 1'b0;
            wh1_parameter_base_q <= '0;
            wout_parameter_base_q <= '0;
            wh1_cache_base_q <= '0;
            wout_cache_base_q <= '0;
            backbone_count_q <= '0;
            hidden_count_q <= '0;
            class_count_q <= '0;
            prepare_version_q <= '0;
            o_prepare_done <= 1'b0;
            o_cache_valid <= 1'b0;
            o_cache_version <= '0;
        end else begin
            o_prepare_done <= 1'b0;

            if (i_clear_error && prep_state_q == PREP_IDLE && !o_busy)
                local_error_q <= 1'b0;
            if (refill_error || clear_error || i_layer_sticky_error)
                local_error_q <= 1'b1;

            if (cfg_fire) begin
                configured_q <= 1'b1;
                wh1_parameter_base_q <= i_cfg_wh1_parameter_base;
                wout_parameter_base_q <= i_cfg_wout_parameter_base;
                wh1_cache_base_q <= i_cfg_wh1_cache_base;
                wout_cache_base_q <= i_cfg_wout_cache_base;
                backbone_count_q <= i_cfg_backbone_count;
                hidden_count_q <= i_cfg_hidden_count;
                class_count_q <= i_cfg_class_count;
                o_cache_valid <= 1'b0;
            end

            if (i_cache_invalidate) begin
                o_cache_valid <= 1'b0;
                if (prep_state_q != PREP_IDLE)
                    local_error_q <= 1'b1;
            end

            if (i_abort) begin
                if (prep_state_q != PREP_IDLE)
                    o_cache_valid <= 1'b0;
                prep_state_q <= PREP_IDLE;
            end else begin
                unique case (prep_state_q)
                    PREP_IDLE: begin
                        if (prepare_fire) begin
                            if (o_cache_valid &&
                                    o_cache_version == i_prepare_version) begin
                                o_prepare_done <= 1'b1;
                            end else begin
                                prepare_version_q <= i_prepare_version;
                                o_cache_valid <= 1'b0;
                                prep_state_q <= PREP_WH1_ISSUE;
                            end
                        end
                    end
                    PREP_WH1_ISSUE: begin
                        if (refill_fire)
                            prep_state_q <= PREP_WH1_WAIT;
                    end
                    PREP_WH1_WAIT: begin
                        if (refill_done) begin
                            if (refill_version != prepare_version_q)
                                local_error_q <= 1'b1;
                            prep_state_q <= PREP_WOUT_ISSUE;
                        end
                    end
                    PREP_WOUT_ISSUE: begin
                        if (refill_fire)
                            prep_state_q <= PREP_WOUT_WAIT;
                    end
                    PREP_WOUT_WAIT: begin
                        if (refill_done) begin
                            if (refill_version != prepare_version_q) begin
                                local_error_q <= 1'b1;
                            end else begin
                                o_cache_valid <= 1'b1;
                                o_cache_version <= prepare_version_q;
                                o_prepare_done <= 1'b1;
                            end
                            prep_state_q <= PREP_IDLE;
                        end
                    end
                    default: begin
                        prep_state_q <= PREP_IDLE;
                        local_error_q <= 1'b1;
                        o_cache_valid <= 1'b0;
                    end
                endcase
            end
        end
    end

endmodule

`default_nettype wire
