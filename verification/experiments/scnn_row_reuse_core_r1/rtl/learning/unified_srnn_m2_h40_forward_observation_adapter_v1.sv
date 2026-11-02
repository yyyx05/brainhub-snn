`timescale 1ns/1ps
`default_nettype none

// Maps the trainable SHD M2 head onto the common H40 array.  The physical
// array may return tile records out of neuron order, so one layer-sized
// observation scratchpad restores the deterministic hidden/class ordering
// expected by the TP learning frontend.  Weights remain exclusively in the
// shared parameter store and H40 cache; this block contains only transient
// activations, cumulative features/logits and control state.
module unified_srnn_m2_h40_forward_observation_adapter_v1 #(
    parameter int M2_CACHE_DEPTH = 578,
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
    parameter int WORK_ADDR_WIDTH = 9,
    parameter int HALF_STATE_DEPTH = 1 << WORK_ADDR_WIDTH,
    parameter int PHYSICAL_TILE_COUNT = 12,
    parameter int OUTPUT_TILE = 7,
    parameter logic signed [31:0] HIDDEN_THRESHOLD_Q14 = 32'sd8192,
    parameter logic [15:0] HIDDEN_LEAK_Q14 = 16'd13926
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,
    input  logic                                      i_abort_timestep,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_cfg_batch_count,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_cfg_backbone_count,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]             i_cfg_hidden_count,
    input  logic [CLASS_INDEX_WIDTH-1:0]              i_cfg_class_count,
    input  logic [WEIGHT_ADDR_WIDTH-1:0]              i_cfg_wh1_cache_base,
    input  logic [WEIGHT_ADDR_WIDTH-1:0]              i_cfg_wout_cache_base,
    input  logic [WORK_ADDR_WIDTH-1:0]                i_cfg_hidden_work_base,
    input  logic [WORK_ADDR_WIDTH-1:0]                i_cfg_output_work_base,
    output logic                                      o_cfg_error,

    input  logic                                      i_timestep_start,
    output logic                                      o_timestep_start_ready,
    input  logic [TIMESTEP_ID_WIDTH-1:0]              i_timestep_id,
    input  logic [VERSION_WIDTH-1:0]                  i_sequence_version,
    input  logic                                      i_include_target_path,

    // A new sequence must not inherit resident membrane/trace state from an
    // older sequence or topology.  The outer maintenance service clears the
    // described logical rows and acknowledges completion before execution.
    output logic                                      o_state_clear_required,
    input  logic                                      i_state_clear_complete,
    output logic [VERSION_WIDTH-1:0]                  o_state_clear_sequence_version,
    output logic [PHYSICAL_TILE_COUNT-1:0]            o_state_clear_tile_mask,
    output logic [WORK_ADDR_WIDTH-1:0]                o_state_clear_hidden_work_base,
    output logic [WORK_ADDR_WIDTH:0]                  o_state_clear_hidden_row_count,
    output logic [WORK_ADDR_WIDTH-1:0]                o_state_clear_output_work_base,
    output logic [2:0]                                o_state_clear_output_row_count,

    // Sample-major backbone stream: all actual samples, followed by all
    // target samples when target-path execution is enabled.
    input  logic                                      i_observation_valid,
    output logic                                      o_observation_ready,
    input  logic                                      i_observation_path,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_observation_sample_id,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_observation_backbone_id,
    input  logic signed [31:0]                        i_observation_q14,
    input  logic                                      i_observation_last,
    input  logic                                      i_observation_done,

    // Descriptor/group side of the existing FC/SRNN H40 layer client.
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

    // Scalar observations from the shared H40 observation serializer.
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
    output logic                                      o_protocol_error,
    output logic                                      o_overflow
);
    localparam logic [1:0] MODE_M2_Q14 = 2'd3;
    localparam int FEATURE_CAPACITY = BATCH_MAX * HIDDEN_MAX;
    localparam int LOGIT_CAPACITY = BATCH_MAX * CLASS_MAX;
    localparam int FEATURE_ADDR_WIDTH = $clog2(FEATURE_CAPACITY);
    localparam int LOGIT_ADDR_WIDTH = $clog2(LOGIT_CAPACITY);
    localparam int FEATURE_STATE_WIDTH = VERSION_WIDTH + 32;
    localparam int LOGIT_STATE_WIDTH = VERSION_WIDTH + 64;
    localparam int SCRATCH_WIDTH = 65;

    typedef enum logic [4:0] {
        STATE_IDLE,
        STATE_CLEAR_WAIT,
        STATE_H_FIRST_WAIT,
        STATE_H_LAYER_START,
        STATE_H_GROUP_STREAM,
        STATE_H_COLLECT,
        STATE_H_READ,
        STATE_H_EMIT,
        STATE_INPUT_DONE_WAIT,
        STATE_W_LAYER_START,
        STATE_W_SPIKE_READ,
        STATE_W_GROUP_STREAM,
        STATE_W_COLLECT,
        STATE_W_READ,
        STATE_W_EMIT,
        STATE_ERROR
    } state_t;

    (* ram_style = "block" *) logic [SCRATCH_WIDTH-1:0]
        observation_scratch [0:HIDDEN_MAX-1];
    (* ram_style = "block" *) logic [FEATURE_STATE_WIDTH-1:0]
        feature_memory [0:FEATURE_CAPACITY-1];
    (* ram_style = "block" *) logic [LOGIT_STATE_WIDTH-1:0]
        logit_memory [0:LOGIT_CAPACITY-1];
    (* ram_style = "block" *) logic
        actual_spike_memory [0:FEATURE_CAPACITY-1];

    state_t state_q;
    logic configured_q;
    logic [SAMPLE_ID_WIDTH-1:0] batch_count_q;
    logic [BACKBONE_INDEX_WIDTH-1:0] backbone_count_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] hidden_count_q;
    logic [CLASS_INDEX_WIDTH-1:0] class_count_q;
    logic [WEIGHT_ADDR_WIDTH-1:0] wh1_cache_base_q;
    logic [WEIGHT_ADDR_WIDTH-1:0] wout_cache_base_q;
    logic [WORK_ADDR_WIDTH-1:0] hidden_work_base_q;
    logic [WORK_ADDR_WIDTH-1:0] output_work_base_q;

    logic [TIMESTEP_ID_WIDTH-1:0] timestep_q;
    logic [VERSION_WIDTH-1:0] sequence_version_q;
    logic include_target_q;
    logic prepared_sequence_valid_q;
    logic [VERSION_WIDTH-1:0] prepared_sequence_q;
    logic expected_path_q;
    logic [SAMPLE_ID_WIDTH-1:0] expected_sample_q;
    logic [BACKBONE_INDEX_WIDTH-1:0] expected_backbone_q;
    logic buffered_source_valid_q;
    logic observation_done_q;

    logic [51:0] active_tag_q;
    logic [15:0] received_count_q;
    logic [HIDDEN_MAX-1:0] received_neuron_q;
    logic observations_complete_q;
    logic layer_done_seen_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] emit_hidden_q;
    logic [CLASS_INDEX_WIDTH-1:0] emit_class_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] wout_hidden_q;
    logic actual_spike_read_q;
    logic [SCRATCH_WIDTH-1:0] scratch_read_q;
    logic [FEATURE_STATE_WIDTH-1:0] feature_read_q;
    logic [LOGIT_STATE_WIDTH-1:0] logit_read_q;

    logic cfg_legal;
    logic start_fire;
    logic observation_fire;
    logic layer_fire;
    logic group_fire;
    logic array_observation_fire;
    logic hidden_fire;
    logic logit_fire;
    logic current_input_legal;
    logic expected_input_last;
    logic array_observation_legal;
    logic current_group_source_valid;
    logic final_hidden_group;
    logic final_hidden_emit;
    logic final_class_emit;
    logic final_sample;
    logic final_path;
    logic [WORK_ADDR_WIDTH:0] hidden_clear_rows;
    logic [WORK_ADDR_WIDTH:0] hidden_work_last;
    logic [WORK_ADDR_WIDTH:0] output_work_last;
    logic [FEATURE_ADDR_WIDTH-1:0] feature_address;
    logic [LOGIT_ADDR_WIDTH-1:0] logit_address;
    logic [FEATURE_STATE_WIDTH-1:0] feature_state_current;
    logic [LOGIT_STATE_WIDTH-1:0] logit_state_current;
    logic signed [63:0] feature_previous_wide;
    logic signed [63:0] feature_next_wide;
    logic signed [31:0] feature_next_clamped;
    logic feature_overflow;
    logic signed [63:0] membrane_next_wide;
    logic signed [31:0] membrane_next_clamped;
    logic signed [63:0] current_logit_q28;
    logic signed [63:0] previous_logit_q28;
    logic signed [64:0] logit_sum_wide;
    logic signed [63:0] logit_next_q28;
    logic logit_overflow;

    function automatic logic [51:0] make_tag(
        input logic [3:0] stage,
        input logic path,
        input logic [SAMPLE_ID_WIDTH-1:0] sample_id,
        input logic [TIMESTEP_ID_WIDTH-1:0] timestep_id,
        input logic [VERSION_WIDTH-1:0] sequence_version
    );
        logic [51:0] value;
        begin
            value = '0;
            value[51:48] = stage;
            value[47] = path;
            value[46:39] = 8'(sample_id);
            value[38:32] = 7'(timestep_id);
            value[31:16] = 16'(sequence_version);
            make_tag = value;
        end
    endfunction

    function automatic logic signed [31:0] clamp_s32(
        input logic signed [63:0] value
    );
        begin
            if (value > 64'sd2147483647)
                clamp_s32 = 32'sh7fffffff;
            else if (value < -64'sd2147483648)
                clamp_s32 = 32'sh80000000;
            else
                clamp_s32 = value[31:0];
        end
    endfunction

    always_comb begin
        hidden_clear_rows = (WORK_ADDR_WIDTH+1)'(
            int'(i_cfg_batch_count) * 8);
        hidden_work_last = {1'b0, i_cfg_hidden_work_base} +
            hidden_clear_rows;
        output_work_last = {1'b0, i_cfg_output_work_base} +
            (WORK_ADDR_WIDTH+1)'((int'(i_cfg_class_count) + 9) / 10);
        cfg_legal = i_cfg_batch_count != '0 &&
            int'(i_cfg_batch_count) <= BATCH_MAX &&
            i_cfg_backbone_count != '0 &&
            int'(i_cfg_backbone_count) <= BACKBONE_MAX &&
            i_cfg_hidden_count != '0 &&
            int'(i_cfg_hidden_count) <= HIDDEN_MAX &&
            int'(i_cfg_hidden_count) <= OUTPUT_TILE * 40 &&
            i_cfg_class_count != '0 &&
            int'(i_cfg_class_count) <= CLASS_MAX &&
            OUTPUT_TILE < PHYSICAL_TILE_COUNT &&
            int'(i_cfg_wh1_cache_base) + int'(i_cfg_backbone_count) <=
                M2_CACHE_DEPTH &&
            int'(i_cfg_wout_cache_base) + int'(i_cfg_hidden_count) <=
                M2_CACHE_DEPTH &&
            hidden_work_last <= (WORK_ADDR_WIDTH+1)'(HALF_STATE_DEPTH) &&
            output_work_last <= (WORK_ADDR_WIDTH+1)'(HALF_STATE_DEPTH);

        o_cfg_ready = state_q == STATE_IDLE;
        o_timestep_start_ready = state_q == STATE_IDLE && configured_q &&
            !i_layer_sticky_error && i_sequence_version != '0;
        start_fire = i_timestep_start && o_timestep_start_ready;

        o_state_clear_required = state_q == STATE_CLEAR_WAIT;
        o_state_clear_sequence_version = sequence_version_q;
        o_state_clear_tile_mask = '0;
        for (int tile = 0; tile < PHYSICAL_TILE_COUNT; tile++) begin
            if (tile < ((int'(hidden_count_q) + 39) / 40) ||
                tile == OUTPUT_TILE)
                o_state_clear_tile_mask[tile] = 1'b1;
        end
        o_state_clear_hidden_work_base = hidden_work_base_q;
        o_state_clear_hidden_row_count = (WORK_ADDR_WIDTH+1)'(
            int'(batch_count_q) * 8);
        o_state_clear_output_work_base = output_work_base_q;
        o_state_clear_output_row_count = 3'(
            (int'(class_count_q) + 9) / 10);

        final_hidden_group = expected_backbone_q ==
            (backbone_count_q - 1'b1);
        final_hidden_emit = emit_hidden_q == (hidden_count_q - 1'b1);
        final_class_emit = emit_class_q == (class_count_q - 1'b1);
        final_sample = expected_sample_q == (batch_count_q - 1'b1);
        final_path = expected_path_q || !include_target_q;
        expected_input_last = final_path && final_sample &&
            final_hidden_group;
        current_input_legal =
            (i_observation_path == expected_path_q) &&
            (i_observation_sample_id == expected_sample_q) &&
            (i_observation_backbone_id == expected_backbone_q) &&
            (i_observation_q14 == 32'sd0 ||
             i_observation_q14 == 32'sd16384) &&
            (i_observation_last == expected_input_last);

        o_observation_ready = 1'b0;
        if (state_q == STATE_H_FIRST_WAIT)
            o_observation_ready = 1'b1;
        else if (state_q == STATE_H_GROUP_STREAM &&
                 expected_backbone_q != '0)
            o_observation_ready = current_input_legal ?
                i_group_ready : 1'b1;
        observation_fire = i_observation_valid && o_observation_ready;

        o_layer_valid = state_q == STATE_H_LAYER_START ||
            state_q == STATE_W_LAYER_START;
        o_layer_mode = MODE_M2_Q14;
        o_layer_neuron_count = state_q == STATE_W_LAYER_START ?
            16'(class_count_q) : 16'(hidden_count_q);
        o_layer_feedforward_group_count =
            state_q == STATE_W_LAYER_START ?
                16'(hidden_count_q) : 16'(backbone_count_q);
        o_layer_recurrent_group_count = '0;
        o_layer_recurrent_enable = 1'b0;
        o_layer_leak = state_q == STATE_W_LAYER_START ?
            16'd0 : HIDDEN_LEAK_Q14;
        o_layer_threshold = state_q == STATE_W_LAYER_START ?
            32'h80000000 : HIDDEN_THRESHOLD_Q14;
        o_layer_reset_to_zero = state_q == STATE_W_LAYER_START;
        o_layer_trace_decay = state_q == STATE_W_LAYER_START ?
            16'd0 : HIDDEN_LEAK_Q14;
        o_layer_transaction_tag = state_q == STATE_W_LAYER_START ?
            make_tag(4'h2, 1'b0, expected_sample_q, timestep_q,
                     sequence_version_q) :
            make_tag(4'h1, expected_path_q, expected_sample_q, timestep_q,
                     sequence_version_q);
        o_layer_work_base_addr = state_q == STATE_W_LAYER_START ?
            output_work_base_q : WORK_ADDR_WIDTH'(
                int'(hidden_work_base_q) +
                ((expected_path_q ? int'(batch_count_q) : 0) +
                 int'(expected_sample_q)) * 4);
        o_layer_destination_tile_base = state_q == STATE_W_LAYER_START ?
            4'(OUTPUT_TILE) : 4'd0;
        layer_fire = o_layer_valid && i_layer_ready;

        current_group_source_valid = expected_backbone_q == '0 ?
            buffered_source_valid_q : (i_observation_q14 != 32'sd0);
        o_group_valid = 1'b0;
        if (state_q == STATE_H_GROUP_STREAM) begin
            if (expected_backbone_q == '0)
                o_group_valid = 1'b1;
            else
                o_group_valid = i_observation_valid && current_input_legal;
        end else if (state_q == STATE_W_GROUP_STREAM) begin
            o_group_valid = 1'b1;
        end
        o_group_recurrent = 1'b0;
        o_group_source_valid = state_q == STATE_W_GROUP_STREAM ?
            {3'b000, actual_spike_read_q} :
            {3'b000, current_group_source_valid};
        o_group_weight_addr0 = state_q == STATE_W_GROUP_STREAM ?
            WEIGHT_ADDR_WIDTH'(int'(wout_cache_base_q) +
                               int'(wout_hidden_q)) :
            WEIGHT_ADDR_WIDTH'(int'(wh1_cache_base_q) +
                               int'(expected_backbone_q));
        o_group_weight_addr1 = '0;
        o_group_weight_addr2 = '0;
        o_group_weight_addr3 = '0;
        group_fire = o_group_valid && i_group_ready;

        array_observation_legal =
            i_array_observation_mode == MODE_M2_Q14 &&
            i_array_observation_transaction_tag == active_tag_q &&
            ((state_q == STATE_H_COLLECT &&
              int'(i_array_observation_neuron_id) <
                  int'(hidden_count_q)) ||
             (state_q == STATE_W_COLLECT &&
              int'(i_array_observation_neuron_id) <
                  int'(class_count_q))) &&
            !received_neuron_q[int'(i_array_observation_neuron_id)];
        o_array_observation_ready =
            (state_q == STATE_H_COLLECT || state_q == STATE_W_COLLECT) &&
            !observations_complete_q;
        array_observation_fire = i_array_observation_valid &&
            o_array_observation_ready;

        feature_address = FEATURE_ADDR_WIDTH'(
            int'(expected_sample_q) * HIDDEN_MAX + int'(emit_hidden_q));
        logit_address = LOGIT_ADDR_WIDTH'(
            int'(expected_sample_q) * CLASS_MAX + int'(emit_class_q));
        feature_state_current = feature_read_q;
        logit_state_current = logit_read_q;
        feature_previous_wide =
            feature_state_current[FEATURE_STATE_WIDTH-1:32] ==
                sequence_version_q ?
            {{32{feature_state_current[31]}},
              feature_state_current[31:0]} : 64'sd0;
        feature_next_wide = feature_previous_wide +
            ((!expected_path_q && scratch_read_q[64]) ?
                64'sd16384 : 64'sd0);
        feature_next_clamped = clamp_s32(feature_next_wide);
        feature_overflow = feature_next_wide > 64'sd2147483647 ||
            feature_next_wide < -64'sd2147483648;
        membrane_next_wide =
            {{32{scratch_read_q[63]}}, scratch_read_q[63:32]} -
            (scratch_read_q[64] ?
                {{32{HIDDEN_THRESHOLD_Q14[31]}},
                  HIDDEN_THRESHOLD_Q14} : 64'sd0);
        membrane_next_clamped = clamp_s32(membrane_next_wide);

        o_hidden_valid = state_q == STATE_H_EMIT;
        o_hidden_path = expected_path_q;
        o_hidden_sample_id = expected_sample_q;
        o_hidden_id = emit_hidden_q;
        o_hidden_spike = scratch_read_q[64];
        o_hidden_membrane_pre_q14 = $signed(scratch_read_q[63:32]);
        o_hidden_membrane_next_q14 = membrane_next_clamped;
        o_hidden_trace_q14 = $signed(scratch_read_q[31:0]);
        o_hidden_output_feature_q14 = expected_path_q ?
            32'sd0 : feature_next_clamped;
        o_hidden_last = o_hidden_valid && final_hidden_emit &&
            final_sample && final_path;
        hidden_fire = o_hidden_valid && i_hidden_ready;

        current_logit_q28 = $signed({
            {18{scratch_read_q[63]}}, scratch_read_q[63:32], 14'b0});
        previous_logit_q28 =
            logit_state_current[LOGIT_STATE_WIDTH-1:64] ==
                sequence_version_q ?
            $signed(logit_state_current[63:0]) : 64'sd0;
        logit_sum_wide = {previous_logit_q28[63], previous_logit_q28} +
            {current_logit_q28[63], current_logit_q28};
        logit_next_q28 = logit_sum_wide[63:0];
        logit_overflow = logit_sum_wide[64] != logit_sum_wide[63];

        o_logit_valid = state_q == STATE_W_EMIT;
        o_logit_sample_id = expected_sample_q;
        o_logit_class_id = emit_class_q;
        o_logit_q28 = logit_next_q28;
        o_logit_last = o_logit_valid && final_class_emit && final_sample;
        logit_fire = o_logit_valid && i_logit_ready;

        o_timestep_id = timestep_q;
        o_busy = state_q != STATE_IDLE;
    end

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            configured_q <= 1'b0;
            batch_count_q <= '0;
            backbone_count_q <= '0;
            hidden_count_q <= '0;
            class_count_q <= '0;
            wh1_cache_base_q <= '0;
            wout_cache_base_q <= '0;
            hidden_work_base_q <= '0;
            output_work_base_q <= '0;
            timestep_q <= '0;
            sequence_version_q <= '0;
            include_target_q <= 1'b0;
            prepared_sequence_valid_q <= 1'b0;
            prepared_sequence_q <= '0;
            expected_path_q <= 1'b0;
            expected_sample_q <= '0;
            expected_backbone_q <= '0;
            buffered_source_valid_q <= 1'b0;
            observation_done_q <= 1'b0;
            active_tag_q <= '0;
            received_count_q <= '0;
            received_neuron_q <= '0;
            observations_complete_q <= 1'b0;
            layer_done_seen_q <= 1'b0;
            emit_hidden_q <= '0;
            emit_class_q <= '0;
            wout_hidden_q <= '0;
            actual_spike_read_q <= 1'b0;
            scratch_read_q <= '0;
            feature_read_q <= '0;
            logit_read_q <= '0;
            o_cfg_error <= 1'b0;
            o_timestep_done <= 1'b0;
            o_protocol_error <= 1'b0;
            o_overflow <= 1'b0;
        end else begin
            o_cfg_error <= 1'b0;
            o_timestep_done <= 1'b0;

            if (i_clear_error) begin
                o_protocol_error <= 1'b0;
                o_overflow <= 1'b0;
                if (state_q == STATE_ERROR)
                    state_q <= STATE_IDLE;
            end

            if (i_cfg_valid && o_cfg_ready) begin
                if (!cfg_legal) begin
                    configured_q <= 1'b0;
                    prepared_sequence_valid_q <= 1'b0;
                    o_cfg_error <= 1'b1;
                end else begin
                    configured_q <= 1'b1;
                    batch_count_q <= i_cfg_batch_count;
                    backbone_count_q <= i_cfg_backbone_count;
                    hidden_count_q <= i_cfg_hidden_count;
                    class_count_q <= i_cfg_class_count;
                    wh1_cache_base_q <= i_cfg_wh1_cache_base;
                    wout_cache_base_q <= i_cfg_wout_cache_base;
                    hidden_work_base_q <= i_cfg_hidden_work_base;
                    output_work_base_q <= i_cfg_output_work_base;
                    prepared_sequence_valid_q <= 1'b0;
                    o_protocol_error <= 1'b0;
                    o_overflow <= 1'b0;
                end
            end

            if (i_abort_timestep && state_q != STATE_IDLE) begin
                state_q <= STATE_IDLE;
                prepared_sequence_valid_q <= 1'b0;
                observation_done_q <= 1'b0;
                received_neuron_q <= '0;
                observations_complete_q <= 1'b0;
                layer_done_seen_q <= 1'b0;
            end else begin
                if (i_observation_done && state_q != STATE_IDLE &&
                    state_q != STATE_ERROR)
                    observation_done_q <= 1'b1;

                if (i_layer_sticky_error && state_q != STATE_IDLE) begin
                    o_protocol_error <= 1'b1;
                    prepared_sequence_valid_q <= 1'b0;
                    state_q <= STATE_ERROR;
                end else begin
                    if (start_fire) begin
                        timestep_q <= i_timestep_id;
                        sequence_version_q <= i_sequence_version;
                        include_target_q <= i_include_target_path;
                        expected_path_q <= 1'b0;
                        expected_sample_q <= '0;
                        expected_backbone_q <= '0;
                        observation_done_q <= 1'b0;
                        emit_hidden_q <= '0;
                        emit_class_q <= '0;
                        wout_hidden_q <= '0;
                        o_overflow <= 1'b0;
                        if (!prepared_sequence_valid_q ||
                            prepared_sequence_q != i_sequence_version)
                            state_q <= STATE_CLEAR_WAIT;
                        else
                            state_q <= STATE_H_FIRST_WAIT;
                    end else if (i_timestep_start && state_q == STATE_IDLE &&
                                 i_sequence_version == '0) begin
                        o_protocol_error <= 1'b1;
                        state_q <= STATE_ERROR;
                    end

                    unique case (state_q)
                    STATE_IDLE: begin
                    end

                    STATE_CLEAR_WAIT: begin
                        if (i_state_clear_complete) begin
                            prepared_sequence_valid_q <= 1'b1;
                            prepared_sequence_q <= sequence_version_q;
                            state_q <= STATE_H_FIRST_WAIT;
                        end
                    end

                    STATE_H_FIRST_WAIT: begin
                        if (observation_fire) begin
                            if (!current_input_legal) begin
                                o_protocol_error <= 1'b1;
                                prepared_sequence_valid_q <= 1'b0;
                                state_q <= STATE_ERROR;
                            end else begin
                                buffered_source_valid_q <=
                                    i_observation_q14 != 32'sd0;
                                active_tag_q <= make_tag(
                                    4'h1, expected_path_q,
                                    expected_sample_q, timestep_q,
                                    sequence_version_q);
                                state_q <= STATE_H_LAYER_START;
                            end
                        end
                    end

                    STATE_H_LAYER_START: begin
                        if (layer_fire) begin
                            expected_backbone_q <= '0;
                            received_count_q <= '0;
                            received_neuron_q <= '0;
                            observations_complete_q <= 1'b0;
                            layer_done_seen_q <= 1'b0;
                            state_q <= STATE_H_GROUP_STREAM;
                        end
                    end

                    STATE_H_GROUP_STREAM: begin
                        if (expected_backbone_q != '0 &&
                            observation_fire && !current_input_legal) begin
                            o_protocol_error <= 1'b1;
                            prepared_sequence_valid_q <= 1'b0;
                            state_q <= STATE_ERROR;
                        end else if (group_fire) begin
                            if (final_hidden_group) begin
                                state_q <= STATE_H_COLLECT;
                            end else begin
                                expected_backbone_q <=
                                    expected_backbone_q + 1'b1;
                            end
                        end
                    end

                    STATE_H_COLLECT: begin
                        if (i_layer_done) begin
                            layer_done_seen_q <= 1'b1;
                            if (observations_complete_q) begin
                                emit_hidden_q <= '0;
                                state_q <= STATE_H_READ;
                            end
                        end
                        if (array_observation_fire) begin
                            if (!array_observation_legal) begin
                                o_protocol_error <= 1'b1;
                                prepared_sequence_valid_q <= 1'b0;
                                state_q <= STATE_ERROR;
                            end else begin
                                observation_scratch[
                                    int'(i_array_observation_neuron_id)] <= {
                                    i_array_observation_spike,
                                    i_array_observation_pre_state,
                                    i_array_observation_trace_next
                                };
                                received_neuron_q[
                                    int'(i_array_observation_neuron_id)] <=
                                    1'b1;
                                received_count_q <= received_count_q + 1'b1;
                                if (received_count_q + 1'b1 ==
                                    16'(hidden_count_q)) begin
                                    observations_complete_q <= 1'b1;
                                    if (layer_done_seen_q || i_layer_done) begin
                                        emit_hidden_q <= '0;
                                        state_q <= STATE_H_READ;
                                    end
                                end
                            end
                        end
                    end

                    STATE_H_READ: begin
                        scratch_read_q <= observation_scratch[
                            int'(emit_hidden_q)];
                        feature_read_q <= feature_memory[
                            int'(feature_address)];
                        state_q <= STATE_H_EMIT;
                    end

                    STATE_H_EMIT: begin
                        if (hidden_fire) begin
                            if (!expected_path_q) begin
                                feature_memory[int'(feature_address)] <= {
                                    sequence_version_q,
                                    feature_next_clamped
                                };
                                actual_spike_memory[int'(feature_address)] <=
                                    scratch_read_q[64];
                            end
                            if (feature_overflow ||
                                membrane_next_wide > 64'sd2147483647 ||
                                membrane_next_wide < -64'sd2147483648)
                                o_overflow <= 1'b1;

                            if (final_hidden_emit) begin
                                expected_backbone_q <= '0;
                                if (final_path && final_sample) begin
                                    expected_path_q <= 1'b0;
                                    expected_sample_q <= '0;
                                    wout_hidden_q <= '0;
                                    emit_class_q <= '0;
                                    if (observation_done_q ||
                                        i_observation_done)
                                        state_q <= STATE_W_LAYER_START;
                                    else
                                        state_q <= STATE_INPUT_DONE_WAIT;
                                end else begin
                                    if (final_sample) begin
                                        expected_path_q <= 1'b1;
                                        expected_sample_q <= '0;
                                    end else begin
                                        expected_sample_q <=
                                            expected_sample_q + 1'b1;
                                    end
                                    state_q <= STATE_H_FIRST_WAIT;
                                end
                            end else begin
                                emit_hidden_q <= emit_hidden_q + 1'b1;
                                state_q <= STATE_H_READ;
                            end
                        end
                    end

                    STATE_INPUT_DONE_WAIT: begin
                        if (observation_done_q || i_observation_done) begin
                            expected_path_q <= 1'b0;
                            expected_sample_q <= '0;
                            wout_hidden_q <= '0;
                            emit_class_q <= '0;
                            state_q <= STATE_W_LAYER_START;
                        end
                    end

                    STATE_W_LAYER_START: begin
                        if (layer_fire) begin
                            active_tag_q <= make_tag(
                                4'h2, 1'b0, expected_sample_q,
                                timestep_q, sequence_version_q);
                            wout_hidden_q <= '0;
                            received_count_q <= '0;
                            received_neuron_q <= '0;
                            observations_complete_q <= 1'b0;
                            layer_done_seen_q <= 1'b0;
                            state_q <= STATE_W_SPIKE_READ;
                        end
                    end

                    STATE_W_SPIKE_READ: begin
                        actual_spike_read_q <= actual_spike_memory[
                            int'(expected_sample_q) * HIDDEN_MAX +
                            int'(wout_hidden_q)];
                        state_q <= STATE_W_GROUP_STREAM;
                    end

                    STATE_W_GROUP_STREAM: begin
                        if (group_fire) begin
                            if (wout_hidden_q == hidden_count_q - 1'b1)
                                state_q <= STATE_W_COLLECT;
                            else begin
                                wout_hidden_q <= wout_hidden_q + 1'b1;
                                state_q <= STATE_W_SPIKE_READ;
                            end
                        end
                    end

                    STATE_W_COLLECT: begin
                        if (i_layer_done) begin
                            layer_done_seen_q <= 1'b1;
                            if (observations_complete_q) begin
                                emit_class_q <= '0;
                                state_q <= STATE_W_READ;
                            end
                        end
                        if (array_observation_fire) begin
                            if (!array_observation_legal) begin
                                o_protocol_error <= 1'b1;
                                prepared_sequence_valid_q <= 1'b0;
                                state_q <= STATE_ERROR;
                            end else begin
                                observation_scratch[
                                    int'(i_array_observation_neuron_id)] <= {
                                    i_array_observation_spike,
                                    i_array_observation_pre_state,
                                    i_array_observation_trace_next
                                };
                                received_neuron_q[
                                    int'(i_array_observation_neuron_id)] <=
                                    1'b1;
                                received_count_q <= received_count_q + 1'b1;
                                if (received_count_q + 1'b1 ==
                                    16'(class_count_q)) begin
                                    observations_complete_q <= 1'b1;
                                    if (layer_done_seen_q || i_layer_done) begin
                                        emit_class_q <= '0;
                                        state_q <= STATE_W_READ;
                                    end
                                end
                            end
                        end
                    end

                    STATE_W_READ: begin
                        scratch_read_q <= observation_scratch[
                            int'(emit_class_q)];
                        logit_read_q <= logit_memory[int'(logit_address)];
                        state_q <= STATE_W_EMIT;
                    end

                    STATE_W_EMIT: begin
                        if (logit_fire) begin
                            logit_memory[int'(logit_address)] <= {
                                sequence_version_q,
                                logit_next_q28
                            };
                            if (logit_overflow)
                                o_overflow <= 1'b1;
                            if (final_class_emit) begin
                                emit_class_q <= '0;
                                if (final_sample) begin
                                    expected_sample_q <= '0;
                                    o_timestep_done <= 1'b1;
                                    state_q <= STATE_IDLE;
                                end else begin
                                    expected_sample_q <=
                                        expected_sample_q + 1'b1;
                                    state_q <= STATE_W_LAYER_START;
                                end
                            end else begin
                                emit_class_q <= emit_class_q + 1'b1;
                                state_q <= STATE_W_READ;
                            end
                        end
                    end

                    STATE_ERROR: begin
                    end

                    default: begin
                        o_protocol_error <= 1'b1;
                        prepared_sequence_valid_q <= 1'b0;
                        state_q <= STATE_ERROR;
                    end
                    endcase
                end
            end
        end
    end

    initial begin
        if (BATCH_MAX < 1 || BACKBONE_MAX < 1 || HIDDEN_MAX < 1 ||
            CLASS_MAX < 1 || WEIGHT_DEPTH < 1 || WORK_ADDR_WIDTH < 1 ||
            PHYSICAL_TILE_COUNT < 1 || OUTPUT_TILE < 1 ||
            OUTPUT_TILE >= PHYSICAL_TILE_COUNT || HIDDEN_MAX > OUTPUT_TILE*40)
            $error("SRNN M2 H40 adapter has invalid geometry");
    end
endmodule

`default_nettype wire
