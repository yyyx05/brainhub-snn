`timescale 1ns/1ps
`default_nettype none

// Routes observations from one time-shared target/student graph engine into
// the selected-layer TP runtime. Target pooled features become scalar Conv
// traces; student pooled features/logits go to the shared classifier replay.
module unified_scnn_selected_layers_graph_observation_router_v1 #(
    parameter int MAX_BATCH = 10,
    parameter int MAX_INPUT_CHANNELS = 512,
    parameter int MAX_OUTPUT_CHANNELS = 512,
    parameter int MAX_FEATURES = 2048,
    parameter int LANES = 8,
    parameter int ENGINES = 12,
    parameter int VERSION_WIDTH = 16,
    parameter int FRAME_WIDTH = 8,
    parameter int SAMPLE_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int TIMESTEP_WIDTH = 5,
    parameter int FEATURE_WIDTH = MAX_FEATURES <= 2 ? 1 :
        $clog2(MAX_FEATURES),
    parameter int FEATURE_COUNT_WIDTH = $clog2(MAX_FEATURES + 1),
    parameter int CLASS_WIDTH = 4,
    parameter int CLASS_COUNT_WIDTH = 5,
    parameter int SELECTED_LAYER = 7
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_abort,
    input  wire logic                         i_clear_error,
    input  wire logic                         i_enable,
    input  wire logic                         i_active_target,
    input  wire logic                         i_active_student,
    input  wire logic                         i_conv_observation_enable,
    input  wire logic                         i_classifier_observation_enable,
    input  wire logic [SAMPLE_WIDTH-1:0]       i_active_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]     i_active_timestep,
    input  wire logic [VERSION_WIDTH-1:0]      i_active_version,

    input  wire logic                         i_target_trace_cfg_valid,
    output logic                              o_target_trace_cfg_ready,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0]
                                                i_target_feature_count,
    input  wire logic signed [15:0]            i_target_trace_leak_q8_8,
    input  wire logic                         i_target_trace_state_reset,
    input  wire logic                         i_target_sample_last,
    output logic                              o_target_trace_done,

    input  wire logic                         i_prescan_valid,
    output logic                              o_prescan_ready,
    input  wire logic                         i_prescan_target,
    input  wire logic [SAMPLE_WIDTH-1:0]       i_prescan_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]     i_prescan_timestep,
    input  wire logic [VERSION_WIDTH-1:0]      i_prescan_version,
    input  wire logic [19:0]                  i_prescan_layer_tag,
    input  wire logic [9:0]                   i_prescan_channel,
    input  wire logic [31:0]                  i_prescan_spatial,
    input  wire logic signed [15:0]            i_prescan_q8_8,

    input  wire logic                         i_surrogate_valid,
    output logic                              o_surrogate_source_ready,
    input  wire logic                         i_surrogate_target,
    input  wire logic [SAMPLE_WIDTH-1:0]       i_surrogate_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]     i_surrogate_timestep,
    input  wire logic [VERSION_WIDTH-1:0]      i_surrogate_version,
    input  wire logic [19:0]                  i_surrogate_layer_tag,
    input  wire logic [9:0]                   i_surrogate_channel,
    input  wire logic [31:0]                  i_surrogate_spatial,
    input  wire logic signed [15:0]            i_surrogate_q8_8,

    input  wire logic                         i_feature_wave_valid,
    output logic                              o_feature_wave_ready,
    input  wire logic [LANES-1:0]             i_feature_wave_mask,
    input  wire logic [FEATURE_WIDTH-1:0]      i_feature_wave_base,
    input  wire logic [LANES*16-1:0]          i_feature_wave_q8_8_flat,
    input  wire logic [VERSION_WIDTH-1:0]      i_feature_wave_version,
    input  wire logic [FRAME_WIDTH-1:0]        i_feature_wave_frame_index,
    input  wire logic                         i_feature_wave_first,
    input  wire logic                         i_feature_wave_last,
    input  wire logic                         i_feature_wave_sample_reset,
    input  wire logic                         i_feature_wave_sample_last,
    input  wire logic [CLASS_WIDTH-1:0]        i_feature_sample_label,

    input  wire logic                         i_logit_wave_valid,
    output logic                              o_logit_wave_ready,
    input  wire logic [ENGINES-1:0]           i_logit_wave_mask,
    input  wire logic [CLASS_WIDTH-1:0]        i_logit_wave_base,
    input  wire logic [CLASS_COUNT_WIDTH-1:0]  i_logit_wave_count,
    input  wire logic [ENGINES*16-1:0]        i_logit_wave_q8_8_flat,
    input  wire logic [VERSION_WIDTH-1:0]      i_logit_wave_version,
    input  wire logic [FRAME_WIDTH-1:0]        i_logit_wave_frame_index,
    input  wire logic                         i_logit_wave_frame_last,
    input  wire logic                         i_logit_wave_sample_last,

    output logic                              o_input_valid,
    input  wire logic                         i_input_ready,
    output logic                              o_input_target,
    output logic [SAMPLE_WIDTH-1:0]            o_input_sample,
    output logic [TIMESTEP_WIDTH-1:0]          o_input_timestep,
    output logic [VERSION_WIDTH-1:0]           o_input_version,
    output logic [8:0]                        o_input_channel,
    output logic [3:0]                        o_input_spatial,
    output logic signed [15:0]                o_input_q8_8,

    output logic                              o_trace_valid,
    input  wire logic                         i_trace_ready,
    output logic                              o_trace_target,
    output logic [SAMPLE_WIDTH-1:0]            o_trace_sample,
    output logic [TIMESTEP_WIDTH-1:0]          o_trace_timestep,
    output logic [VERSION_WIDTH-1:0]           o_trace_version,
    output logic [8:0]                        o_trace_channel,
    output logic [1:0]                        o_trace_spatial,
    output logic signed [15:0]                o_trace_q8_8,

    output logic                              o_surrogate_valid,
    input  wire logic                         i_surrogate_ready,
    output logic                              o_surrogate_target,
    output logic [SAMPLE_WIDTH-1:0]            o_surrogate_sample,
    output logic [TIMESTEP_WIDTH-1:0]          o_surrogate_timestep,
    output logic [VERSION_WIDTH-1:0]           o_surrogate_version,
    output logic [8:0]                        o_surrogate_channel,
    output logic [3:0]                        o_surrogate_spatial,
    output logic signed [15:0]                o_surrogate_q8_8,

    output logic                              o_student_feature_wave_valid,
    input  wire logic                         i_student_feature_wave_ready,
    output logic [LANES-1:0]                 o_student_feature_wave_mask,
    output logic [FEATURE_WIDTH-1:0]          o_student_feature_wave_base,
    output logic [LANES*16-1:0]              o_student_feature_wave_q8_8_flat,
    output logic [VERSION_WIDTH-1:0]          o_student_feature_wave_version,
    output logic [FRAME_WIDTH-1:0]            o_student_feature_wave_frame_index,
    output logic                              o_student_feature_wave_first,
    output logic                              o_student_feature_wave_last,
    output logic                              o_student_feature_wave_sample_reset,
    output logic                              o_student_feature_wave_sample_last,
    output logic [CLASS_WIDTH-1:0]            o_student_sample_label,
    output logic                              o_student_logit_wave_valid,
    input  wire logic                         i_student_logit_wave_ready,
    output logic [ENGINES-1:0]               o_student_logit_wave_mask,
    output logic [CLASS_WIDTH-1:0]            o_student_logit_wave_base,
    output logic [CLASS_COUNT_WIDTH-1:0]      o_student_logit_wave_count,
    output logic [ENGINES*16-1:0]            o_student_logit_wave_q8_8_flat,
    output logic [VERSION_WIDTH-1:0]          o_student_logit_wave_version,
    output logic [FRAME_WIDTH-1:0]            o_student_logit_wave_frame_index,
    output logic                              o_student_logit_wave_frame_last,
    output logic                              o_student_logit_wave_sample_last,

    output logic                              o_busy,
    output logic                              o_sticky_error
);
    logic target_cfg_ready;
    logic target_wave_ready;
    logic target_trace_valid;
    logic target_trace_ready;
    logic [SAMPLE_WIDTH-1:0] target_trace_sample;
    logic [TIMESTEP_WIDTH-1:0] target_trace_timestep;
    logic [VERSION_WIDTH-1:0] target_trace_version;
    logic [8:0] target_trace_channel;
    logic [1:0] target_trace_spatial;
    logic signed [15:0] target_trace_data;
    logic target_trace_last;
    logic target_trace_busy;
    logic target_trace_error;
    logic binding_error;
    logic local_error_q;
    logic phase_target;
    logic phase_student;
    logic prescan_metadata_legal;
    logic surrogate_metadata_legal;

    assign phase_target = i_enable && i_active_target &&
        !i_active_student;
    assign phase_student = i_enable && i_active_student &&
        !i_active_target;
    assign o_target_trace_cfg_ready = phase_target &&
        i_conv_observation_enable && target_cfg_ready;

    assign prescan_metadata_legal = i_prescan_target == i_active_target &&
        i_prescan_sample == i_active_sample &&
        i_prescan_timestep == i_active_timestep &&
        i_prescan_version == i_active_version &&
        i_prescan_layer_tag == 20'(SELECTED_LAYER);
    assign surrogate_metadata_legal =
        i_surrogate_target == i_active_target &&
        i_surrogate_sample == i_active_sample &&
        i_surrogate_timestep == i_active_timestep &&
        i_surrogate_version == i_active_version &&
        i_surrogate_layer_tag == 20'(SELECTED_LAYER);

    assign o_feature_wave_ready = phase_target ? target_wave_ready :
        phase_student && i_classifier_observation_enable ?
            i_student_feature_wave_ready : 1'b1;
    assign o_student_feature_wave_valid = i_feature_wave_valid &&
        phase_student && i_classifier_observation_enable && !o_sticky_error;
    assign o_student_feature_wave_mask = i_feature_wave_mask;
    assign o_student_feature_wave_base = i_feature_wave_base;
    assign o_student_feature_wave_q8_8_flat = i_feature_wave_q8_8_flat;
    assign o_student_feature_wave_version = i_feature_wave_version;
    assign o_student_feature_wave_frame_index = i_feature_wave_frame_index;
    assign o_student_feature_wave_first = i_feature_wave_first;
    assign o_student_feature_wave_last = i_feature_wave_last;
    assign o_student_feature_wave_sample_reset = i_feature_wave_sample_reset;
    assign o_student_feature_wave_sample_last = i_feature_wave_sample_last;
    assign o_student_sample_label = i_feature_sample_label;

    // Target logits and learning-disabled student observations are consumed
    // and discarded. They remain legal graph outputs; the observation-enable
    // bits decide whether they enter TP rather than whether the graph may
    // produce them.
    assign o_logit_wave_ready = phase_student &&
        i_classifier_observation_enable ? i_student_logit_wave_ready : 1'b1;
    assign o_student_logit_wave_valid = i_logit_wave_valid &&
        phase_student && i_classifier_observation_enable && !o_sticky_error;
    assign o_student_logit_wave_mask = i_logit_wave_mask;
    assign o_student_logit_wave_base = i_logit_wave_base;
    assign o_student_logit_wave_count = i_logit_wave_count;
    assign o_student_logit_wave_q8_8_flat = i_logit_wave_q8_8_flat;
    assign o_student_logit_wave_version = i_logit_wave_version;
    assign o_student_logit_wave_frame_index = i_logit_wave_frame_index;
    assign o_student_logit_wave_frame_last = i_logit_wave_frame_last;
    assign o_student_logit_wave_sample_last = i_logit_wave_sample_last;

    unified_scnn_h40_target_feature_wave_trace_v1 #(
        .MAX_BATCH(MAX_BATCH), .MAX_FEATURES(MAX_FEATURES), .LANES(LANES),
        .VERSION_WIDTH(VERSION_WIDTH), .FRAME_WIDTH(FRAME_WIDTH),
        .SAMPLE_WIDTH(SAMPLE_WIDTH), .TIMESTEP_WIDTH(TIMESTEP_WIDTH),
        .FEATURE_WIDTH(FEATURE_WIDTH),
        .FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH)
    ) u_target_trace (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_cfg_valid(i_target_trace_cfg_valid && phase_target &&
            i_conv_observation_enable), .o_cfg_ready(target_cfg_ready),
        .i_cfg_sample(i_active_sample),
        .i_cfg_timestep(i_active_timestep),
        .i_cfg_version(i_active_version),
        .i_cfg_frame_index(FRAME_WIDTH'(i_active_timestep)),
        .i_cfg_feature_count(i_target_feature_count),
        .i_cfg_trace_leak_q8_8(i_target_trace_leak_q8_8),
        .i_cfg_trace_state_reset(i_target_trace_state_reset),
        .i_cfg_sample_last(i_target_sample_last),
        .i_wave_valid(i_feature_wave_valid && phase_target &&
            i_conv_observation_enable), .o_wave_ready(target_wave_ready),
        .i_wave_mask(i_feature_wave_mask),
        .i_wave_base(i_feature_wave_base),
        .i_wave_q8_8_flat(i_feature_wave_q8_8_flat),
        .i_wave_version(i_feature_wave_version),
        .i_wave_frame_index(i_feature_wave_frame_index),
        .i_wave_first(i_feature_wave_first),
        .i_wave_last(i_feature_wave_last),
        .i_wave_sample_reset(i_feature_wave_sample_reset),
        .i_wave_sample_last(i_feature_wave_sample_last),
        .o_trace_valid(target_trace_valid),
        .i_trace_ready(target_trace_ready),
        .o_trace_sample(target_trace_sample),
        .o_trace_timestep(target_trace_timestep),
        .o_trace_version(target_trace_version),
        .o_trace_channel(target_trace_channel),
        .o_trace_spatial(target_trace_spatial),
        .o_trace_q8_8(target_trace_data),
        .o_trace_last(target_trace_last),
        .o_done(o_target_trace_done), .o_busy(target_trace_busy),
        .o_sticky_error(target_trace_error)
    );

    unified_scnn_tp_actual_graph_observation_binding #(
        .MAX_BATCH(MAX_BATCH),
        .MAX_INPUT_CHANNELS(MAX_INPUT_CHANNELS),
        .MAX_OUTPUT_CHANNELS(MAX_OUTPUT_CHANNELS),
        .INPUT_SPATIAL_COUNT(16), .SURROGATE_SPATIAL_COUNT(16),
        .VERSION_WIDTH(VERSION_WIDTH), .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .TIMESTEP_WIDTH(TIMESTEP_WIDTH)
    ) u_observation_binding (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_enable(i_enable && i_conv_observation_enable),
        .i_active_target(i_active_target),
        .i_active_student(i_active_student),
        .i_target_sample(i_active_sample),
        .i_target_timestep(i_active_timestep),
        .i_target_version(i_active_version),
        .i_student_sample(i_active_sample),
        .i_student_timestep(i_active_timestep),
        .i_student_version(i_active_version),
        .i_prescan_valid(i_prescan_valid && prescan_metadata_legal),
        .o_prescan_ready(o_prescan_ready),
        .i_prescan_channel({6'b0, i_prescan_channel}),
        .i_prescan_spatial(i_prescan_spatial[15:0]),
        .i_prescan_layer(i_prescan_layer_tag[3:0]),
        .i_prescan_version(i_prescan_version),
        .i_prescan_data_q8_8(i_prescan_q8_8),
        .i_lif_valid(i_surrogate_valid && surrogate_metadata_legal),
        .o_lif_ready(o_surrogate_source_ready),
        .i_lif_layer(i_surrogate_layer_tag[3:0]),
        .i_lif_channel({6'b0, i_surrogate_channel}),
        .i_lif_spatial(i_surrogate_spatial[15:0]),
        .i_lif_version(i_surrogate_version),
        .i_lif_surrogate_q8_8(i_surrogate_q8_8),
        .i_trace_valid(target_trace_valid && phase_target),
        .o_trace_source_ready(target_trace_ready),
        .i_trace_sample(target_trace_sample),
        .i_trace_timestep(target_trace_timestep),
        .i_trace_version(target_trace_version),
        .i_trace_channel(target_trace_channel),
        .i_trace_spatial(target_trace_spatial),
        .i_trace_q8_8(target_trace_data),
        .o_input_valid(o_input_valid), .i_input_ready(i_input_ready),
        .o_input_target(o_input_target), .o_input_sample(o_input_sample),
        .o_input_timestep(o_input_timestep),
        .o_input_version(o_input_version),
        .o_input_channel(o_input_channel),
        .o_input_spatial(o_input_spatial), .o_input_q8_8(o_input_q8_8),
        .o_surrogate_valid(o_surrogate_valid),
        .i_surrogate_ready(i_surrogate_ready),
        .o_surrogate_target(o_surrogate_target),
        .o_surrogate_sample(o_surrogate_sample),
        .o_surrogate_timestep(o_surrogate_timestep),
        .o_surrogate_version(o_surrogate_version),
        .o_surrogate_channel(o_surrogate_channel),
        .o_surrogate_spatial(o_surrogate_spatial),
        .o_surrogate_q8_8(o_surrogate_q8_8),
        .o_trace_valid(o_trace_valid), .i_trace_ready(i_trace_ready),
        .o_trace_target(o_trace_target), .o_trace_sample(o_trace_sample),
        .o_trace_timestep(o_trace_timestep),
        .o_trace_version(o_trace_version),
        .o_trace_channel(o_trace_channel),
        .o_trace_spatial(o_trace_spatial), .o_trace_q8_8(o_trace_q8_8),
        .o_protocol_error(binding_error)
    );

    assign o_busy = target_trace_busy;
    assign o_sticky_error = local_error_q || target_trace_error ||
        binding_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            local_error_q <= 1'b0;
        else begin
            if (i_clear_error && !o_busy)
                local_error_q <= 1'b0;
            if (i_enable && (i_active_target && i_active_student ||
                i_target_trace_cfg_valid && !phase_target ||
                i_prescan_valid && !prescan_metadata_legal ||
                i_surrogate_valid && !surrogate_metadata_legal ||
                i_feature_wave_valid && !phase_target && !phase_student ||
                i_logit_wave_valid && !phase_target && !phase_student))
                local_error_q <= 1'b1;
        end
    end

    /* verilator lint_off UNUSEDSIGNAL */
    logic unused_target_trace_last;
    always_comb unused_target_trace_last = target_trace_last;
    /* verilator lint_on UNUSEDSIGNAL */
endmodule

`default_nettype wire
