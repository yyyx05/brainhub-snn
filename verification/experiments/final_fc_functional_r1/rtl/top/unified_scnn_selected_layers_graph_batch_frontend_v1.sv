`timescale 1ns/1ps
`default_nettype none

// Batch-aware observation front end for the selected Conv + classifier TP
// path. It sequences target/student passes, prescans the real feature store,
// and routes graph observations without owning any arithmetic or parameters.
module unified_scnn_selected_layers_graph_batch_frontend_v1 #(
    parameter int MAX_BATCH = 10,
    parameter int MAX_TIMESTEPS = 20,
    parameter int MAX_WINDOW_BATCH = MAX_BATCH,
    parameter int MAX_WINDOW_TIMESTEPS = MAX_TIMESTEPS,
    parameter int MAX_FLATTENED_SAMPLES = MAX_BATCH,
    parameter int MAX_INPUT_CHANNELS = 512,
    parameter int MAX_OUTPUT_CHANNELS = 512,
    parameter int MAX_FEATURES = 2048,
    parameter int MAX_CLASSES = 11,
    parameter int LANES = 8,
    parameter int ENGINES = 12,
    parameter int VERSION_WIDTH = 16,
    parameter int FRAME_WIDTH = 8,
    parameter int SAMPLE_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int TIMESTEP_WIDTH = 5,
    parameter int TIMESTEP_COUNT_WIDTH = $clog2(MAX_TIMESTEPS + 1),
    parameter int FEATURE_WIDTH = MAX_FEATURES <= 2 ? 1 :
        $clog2(MAX_FEATURES),
    parameter int FEATURE_COUNT_WIDTH = $clog2(MAX_FEATURES + 1),
    parameter int CLASS_WIDTH = MAX_CLASSES <= 2 ? 1 :
        $clog2(MAX_CLASSES),
    parameter int CLASS_COUNT_WIDTH = $clog2(MAX_CLASSES + 1),
    parameter int SELECTED_LAYER = 7,
    parameter bit ENABLE_TARGET_GRAPH = 1'b1
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_abort,
    input  wire logic                         i_clear_error,

    input  wire logic                         i_request_valid,
    output logic                              o_request_ready,
    input  wire logic                         i_learning_enable,
    input  wire logic [BATCH_COUNT_WIDTH-1:0] i_batch_count,
    input  wire logic [TIMESTEP_WIDTH-1:0]     i_timestep,
    input  wire logic [TIMESTEP_COUNT_WIDTH-1:0]
                                                i_timestep_count,
    input  wire logic [VERSION_WIDTH-1:0]      i_version,
    input  wire logic [MAX_BATCH*CLASS_WIDTH-1:0]
                                                i_labels_flat,
    output logic                              o_done,

    output logic                              o_runtime_run_valid,
    input  wire logic                         i_runtime_run_ready,
    output logic [TIMESTEP_WIDTH-1:0]          o_runtime_timestep,
    output logic [TIMESTEP_COUNT_WIDTH-1:0]    o_runtime_timestep_count,
    output logic [BATCH_COUNT_WIDTH-1:0]       o_runtime_batch_count,
    output logic [VERSION_WIDTH-1:0]           o_runtime_version,
    input  wire logic                         i_runtime_run_done,
    output logic                              o_sample_context_valid,
    input  wire logic                         i_sample_context_ready,
    output logic                              o_sample_context_target,
    output logic [SAMPLE_WIDTH-1:0]            o_sample_context_sample,
    output logic [VERSION_WIDTH-1:0]           o_sample_context_version,

    input  wire logic [15:0]                  i_conv_in_width,
    input  wire logic [15:0]                  i_conv_in_height,
    input  wire logic [15:0]                  i_conv_in_channels,
    input  wire logic [7:0]                   i_conv_padding,
    input  wire logic [19:0]                  i_conv_layer_tag,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0]
                                                i_target_feature_count,
    input  wire logic signed [15:0]            i_target_trace_leak_q8_8,
    input  wire logic                         i_target_trace_state_reset,

    output logic                              o_graph_start_valid,
    input  wire logic                         i_graph_start_ready,
    output logic                              o_graph_target,
    output logic [SAMPLE_WIDTH-1:0]            o_graph_sample,
    output logic [TIMESTEP_WIDTH-1:0]          o_graph_timestep,
    output logic [VERSION_WIDTH-1:0]           o_graph_version,
    output logic [CLASS_WIDTH-1:0]             o_graph_label,
    output logic                              o_graph_sample_last,
    input  wire logic                         i_graph_done,
    input  wire logic                         i_graph_error,
    output logic                              o_conv_observation_enable,
    output logic                              o_classifier_observation_enable,
    output logic                              o_observation_target,

    output logic                              o_prescan_feature_request_valid,
    input  wire logic                         i_prescan_feature_request_ready,
    output logic [3:0]                        o_prescan_feature_term_valid,
    output logic [39:0]                       o_prescan_feature_input_channel_flat,
    output logic [7:0]                        o_prescan_feature_kernel_row_flat,
    output logic [7:0]                        o_prescan_feature_kernel_col_flat,
    output logic [31:0]                       o_prescan_feature_spatial_base,
    output logic [5:0]                        o_prescan_feature_spatial_count,
    output logic [15:0]                       o_prescan_feature_output_row,
    output logic [15:0]                       o_prescan_feature_output_col,
    input  wire logic                         i_prescan_feature_response_valid,
    output logic                              o_prescan_feature_response_ready,
    input  wire logic [39:0]                  i_prescan_feature_mask0,
    input  wire logic [39:0]                  i_prescan_feature_mask1,
    input  wire logic [39:0]                  i_prescan_feature_mask2,
    input  wire logic [39:0]                  i_prescan_feature_mask3,
    input  wire logic                         i_prescan_feature_service_error,

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
    logic coord_graph_valid;
    logic coord_graph_ready;
    logic coord_graph_target;
    logic [SAMPLE_WIDTH-1:0] coord_graph_sample;
    logic [TIMESTEP_WIDTH-1:0] coord_graph_timestep;
    logic [VERSION_WIDTH-1:0] coord_graph_version;
    logic [CLASS_WIDTH-1:0] coord_graph_label;
    logic coord_graph_sample_last;
    logic prep_done;
    logic coord_busy;
    logic coord_error;
    logic prep_busy;
    logic prep_error;
    logic prescan_start_valid;
    logic prescan_start_ready;
    logic prescan_target;
    logic [SAMPLE_WIDTH-1:0] prescan_sample;
    logic [TIMESTEP_WIDTH-1:0] prescan_timestep;
    logic [VERSION_WIDTH-1:0] prescan_version;
    logic prescan_done;
    logic prescan_busy;
    logic prescan_error;
    logic prescan_observation_valid;
    logic prescan_observation_ready;
    logic prescan_observation_target;
    logic [SAMPLE_WIDTH-1:0] prescan_observation_sample;
    logic [TIMESTEP_WIDTH-1:0] prescan_observation_timestep;
    logic [VERSION_WIDTH-1:0] prescan_observation_version;
    logic [19:0] prescan_observation_layer_tag;
    logic [9:0] prescan_observation_channel;
    logic [31:0] prescan_observation_spatial;
    logic signed [15:0] prescan_observation_q8_8;
    logic target_trace_cfg_valid;
    logic target_trace_cfg_ready;
    logic target_trace_done;
    logic joined_graph_done;
    logic target_join_active;
    logic graph_done_seen_q;
    logic target_trace_done_seen_q;
    logic router_busy;
    logic router_error;
    logic [15:0] conv_in_width_q;
    logic [15:0] conv_in_height_q;
    logic [15:0] conv_in_channels_q;
    logic [7:0] conv_padding_q;
    logic [19:0] conv_layer_tag_q;
    logic [FEATURE_COUNT_WIDTH-1:0] target_feature_count_q;
    logic signed [15:0] target_trace_leak_q;
    logic target_trace_state_reset_q;

    unified_scnn_selected_layers_graph_batch_coordinator_v1 #(
        .MAX_BATCH(MAX_BATCH), .MAX_TIMESTEPS(MAX_TIMESTEPS),
        .MAX_WINDOW_BATCH(MAX_WINDOW_BATCH),
        .MAX_WINDOW_TIMESTEPS(MAX_WINDOW_TIMESTEPS),
        .MAX_FLATTENED_SAMPLES(MAX_FLATTENED_SAMPLES),
        .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .ENABLE_TARGET_GRAPH(ENABLE_TARGET_GRAPH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .TIMESTEP_WIDTH(TIMESTEP_WIDTH),
        .TIMESTEP_COUNT_WIDTH(TIMESTEP_COUNT_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH),
        .CLASS_WIDTH(CLASS_WIDTH)
    ) u_batch (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_request_valid(i_request_valid), .o_request_ready(o_request_ready),
        .i_learning_enable(i_learning_enable), .i_batch_count(i_batch_count),
        .i_timestep(i_timestep), .i_timestep_count(i_timestep_count),
        .i_version(i_version),
        .i_labels_flat(i_labels_flat), .o_done(o_done),
        .o_runtime_run_valid(o_runtime_run_valid),
        .i_runtime_run_ready(i_runtime_run_ready),
        .o_runtime_timestep(o_runtime_timestep),
        .o_runtime_timestep_count(o_runtime_timestep_count),
        .o_runtime_batch_count(o_runtime_batch_count),
        .o_runtime_version(o_runtime_version),
        .i_runtime_run_done(i_runtime_run_done),
        .o_sample_context_valid(o_sample_context_valid),
        .i_sample_context_ready(i_sample_context_ready),
        .o_sample_context_target(o_sample_context_target),
        .o_sample_context_sample(o_sample_context_sample),
        .o_sample_context_version(o_sample_context_version),
        .o_graph_start_valid(coord_graph_valid),
        .i_graph_start_ready(coord_graph_ready),
        .o_graph_target(coord_graph_target),
        .o_graph_sample(coord_graph_sample),
        .o_graph_timestep(coord_graph_timestep),
        .o_graph_version(coord_graph_version),
        .o_graph_label(coord_graph_label),
        .o_graph_sample_last(coord_graph_sample_last),
        .i_graph_done(prep_done),
        .o_conv_observation_enable(o_conv_observation_enable),
        .o_classifier_observation_enable(o_classifier_observation_enable),
        .o_observation_target(o_observation_target),
        .o_busy(coord_busy), .o_sticky_error(coord_error)
    );

    unified_scnn_selected_layers_graph_launch_prep_v1 #(
        .SAMPLE_WIDTH(SAMPLE_WIDTH), .TIMESTEP_WIDTH(TIMESTEP_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .CLASS_WIDTH(CLASS_WIDTH)
    ) u_prep (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_pass_valid(coord_graph_valid), .o_pass_ready(coord_graph_ready),
        .i_prescan_enable(o_conv_observation_enable),
        .i_pass_target(coord_graph_target),
        .i_pass_sample(coord_graph_sample),
        .i_pass_timestep(coord_graph_timestep),
        .i_pass_version(coord_graph_version),
        .i_pass_label(coord_graph_label), .o_pass_done(prep_done),
        .o_prescan_start_valid(prescan_start_valid),
        .i_prescan_start_ready(prescan_start_ready),
        .o_prescan_target(prescan_target),
        .o_prescan_sample(prescan_sample),
        .o_prescan_timestep(prescan_timestep),
        .o_prescan_version(prescan_version), .i_prescan_done(prescan_done),
        .i_prescan_error(prescan_error),
        .o_target_trace_cfg_valid(target_trace_cfg_valid),
        .i_target_trace_cfg_ready(target_trace_cfg_ready),
        .o_graph_start_valid(o_graph_start_valid),
        .i_graph_start_ready(i_graph_start_ready),
        .o_graph_target(o_graph_target), .o_graph_sample(o_graph_sample),
        .o_graph_timestep(o_graph_timestep),
        .o_graph_version(o_graph_version), .o_graph_label(o_graph_label),
        .i_graph_done(joined_graph_done), .i_graph_error(i_graph_error),
        .o_busy(prep_busy), .o_sticky_error(prep_error)
    );

    assign o_graph_sample_last = coord_graph_sample_last;
    assign target_join_active = o_conv_observation_enable &&
        o_observation_target;
    assign joined_graph_done = target_join_active ?
        ((graph_done_seen_q || i_graph_done) &&
         (target_trace_done_seen_q || target_trace_done)) : i_graph_done;

    unified_scnn_h40_tp_input_prescan_v1 #(
        .MAX_CHANNELS(MAX_INPUT_CHANNELS), .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .TIMESTEP_WIDTH(TIMESTEP_WIDTH), .VERSION_WIDTH(VERSION_WIDTH)
    ) u_prescan (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error), .i_start_valid(prescan_start_valid),
        .o_start_ready(prescan_start_ready), .i_width(conv_in_width_q),
        .i_height(conv_in_height_q), .i_channels(conv_in_channels_q),
        .i_padding(conv_padding_q), .i_target(prescan_target),
        .i_sample(prescan_sample), .i_timestep(prescan_timestep),
        .i_version(prescan_version), .i_layer_tag(conv_layer_tag_q),
        .o_feature_request_valid(o_prescan_feature_request_valid),
        .i_feature_request_ready(i_prescan_feature_request_ready),
        .o_feature_term_valid(o_prescan_feature_term_valid),
        .o_feature_input_channel_flat(
            o_prescan_feature_input_channel_flat),
        .o_feature_kernel_row_flat(o_prescan_feature_kernel_row_flat),
        .o_feature_kernel_col_flat(o_prescan_feature_kernel_col_flat),
        .o_feature_spatial_base(o_prescan_feature_spatial_base),
        .o_feature_spatial_count(o_prescan_feature_spatial_count),
        .o_feature_output_row(o_prescan_feature_output_row),
        .o_feature_output_col(o_prescan_feature_output_col),
        .i_feature_response_valid(i_prescan_feature_response_valid),
        .o_feature_response_ready(o_prescan_feature_response_ready),
        .i_feature_mask0(i_prescan_feature_mask0),
        .i_feature_mask1(i_prescan_feature_mask1),
        .i_feature_mask2(i_prescan_feature_mask2),
        .i_feature_mask3(i_prescan_feature_mask3),
        .i_feature_service_error(i_prescan_feature_service_error),
        .o_observation_valid(prescan_observation_valid),
        .i_observation_ready(prescan_observation_ready),
        .o_observation_target(prescan_observation_target),
        .o_observation_sample(prescan_observation_sample),
        .o_observation_timestep(prescan_observation_timestep),
        .o_observation_version(prescan_observation_version),
        .o_observation_layer_tag(prescan_observation_layer_tag),
        .o_observation_channel(prescan_observation_channel),
        .o_observation_spatial(prescan_observation_spatial),
        .o_observation_q8_8(prescan_observation_q8_8),
        .o_observation_last(), .o_done(prescan_done),
        .o_busy(prescan_busy), .o_sticky_error(prescan_error)
    );

    unified_scnn_selected_layers_graph_observation_router_v1 #(
        .MAX_BATCH(MAX_BATCH), .MAX_INPUT_CHANNELS(MAX_INPUT_CHANNELS),
        .MAX_OUTPUT_CHANNELS(MAX_OUTPUT_CHANNELS),
        .MAX_FEATURES(MAX_FEATURES), .LANES(LANES), .ENGINES(ENGINES),
        .VERSION_WIDTH(VERSION_WIDTH), .FRAME_WIDTH(FRAME_WIDTH),
        .SAMPLE_WIDTH(SAMPLE_WIDTH), .TIMESTEP_WIDTH(TIMESTEP_WIDTH),
        .FEATURE_WIDTH(FEATURE_WIDTH),
        .FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
        .CLASS_WIDTH(CLASS_WIDTH), .CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH),
        .SELECTED_LAYER(SELECTED_LAYER)
    ) u_router (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error), .i_enable(1'b1),
        // Graph phase and TP observation capture are independent. Inference
        // still executes a student graph even when both observation enables
        // are low, so derive the phase from the admitted launch lifetime.
        .i_active_target((coord_graph_valid || prep_busy) &&
            o_graph_target),
        .i_active_student((coord_graph_valid || prep_busy) &&
            !o_graph_target),
        .i_conv_observation_enable(o_conv_observation_enable),
        .i_classifier_observation_enable(o_classifier_observation_enable),
        .i_active_sample(o_graph_sample),
        .i_active_timestep(o_graph_timestep),
        .i_active_version(o_graph_version),
        .i_target_trace_cfg_valid(target_trace_cfg_valid),
        .o_target_trace_cfg_ready(target_trace_cfg_ready),
        .i_target_feature_count(target_feature_count_q),
        .i_target_trace_leak_q8_8(target_trace_leak_q),
        .i_target_trace_state_reset(target_trace_state_reset_q),
        .i_target_sample_last(1'b1),
        .o_target_trace_done(target_trace_done),
        .i_prescan_valid(prescan_observation_valid),
        .o_prescan_ready(prescan_observation_ready),
        .i_prescan_target(prescan_observation_target),
        .i_prescan_sample(prescan_observation_sample),
        .i_prescan_timestep(prescan_observation_timestep),
        .i_prescan_version(prescan_observation_version),
        .i_prescan_layer_tag(prescan_observation_layer_tag),
        .i_prescan_channel(prescan_observation_channel),
        .i_prescan_spatial(prescan_observation_spatial),
        .i_prescan_q8_8(prescan_observation_q8_8),
        .i_surrogate_valid(i_surrogate_valid),
        .o_surrogate_source_ready(o_surrogate_source_ready),
        .i_surrogate_target(i_surrogate_target),
        .i_surrogate_sample(i_surrogate_sample),
        .i_surrogate_timestep(i_surrogate_timestep),
        .i_surrogate_version(i_surrogate_version),
        .i_surrogate_layer_tag(i_surrogate_layer_tag),
        .i_surrogate_channel(i_surrogate_channel),
        .i_surrogate_spatial(i_surrogate_spatial),
        .i_surrogate_q8_8(i_surrogate_q8_8),
        .i_feature_wave_valid(i_feature_wave_valid),
        .o_feature_wave_ready(o_feature_wave_ready),
        .i_feature_wave_mask(i_feature_wave_mask),
        .i_feature_wave_base(i_feature_wave_base),
        .i_feature_wave_q8_8_flat(i_feature_wave_q8_8_flat),
        .i_feature_wave_version(i_feature_wave_version),
        .i_feature_wave_frame_index(i_feature_wave_frame_index),
        .i_feature_wave_first(i_feature_wave_first),
        .i_feature_wave_last(i_feature_wave_last),
        .i_feature_wave_sample_reset(i_feature_wave_sample_reset),
        .i_feature_wave_sample_last(i_feature_wave_sample_last),
        .i_feature_sample_label(i_feature_sample_label),
        .i_logit_wave_valid(i_logit_wave_valid),
        .o_logit_wave_ready(o_logit_wave_ready),
        .i_logit_wave_mask(i_logit_wave_mask),
        .i_logit_wave_base(i_logit_wave_base),
        .i_logit_wave_count(i_logit_wave_count),
        .i_logit_wave_q8_8_flat(i_logit_wave_q8_8_flat),
        .i_logit_wave_version(i_logit_wave_version),
        .i_logit_wave_frame_index(i_logit_wave_frame_index),
        .i_logit_wave_frame_last(i_logit_wave_frame_last),
        .i_logit_wave_sample_last(i_logit_wave_sample_last),
        .o_input_valid(o_input_valid), .i_input_ready(i_input_ready),
        .o_input_target(o_input_target), .o_input_sample(o_input_sample),
        .o_input_timestep(o_input_timestep),
        .o_input_version(o_input_version),
        .o_input_channel(o_input_channel),
        .o_input_spatial(o_input_spatial), .o_input_q8_8(o_input_q8_8),
        .o_trace_valid(o_trace_valid), .i_trace_ready(i_trace_ready),
        .o_trace_target(o_trace_target), .o_trace_sample(o_trace_sample),
        .o_trace_timestep(o_trace_timestep),
        .o_trace_version(o_trace_version),
        .o_trace_channel(o_trace_channel),
        .o_trace_spatial(o_trace_spatial), .o_trace_q8_8(o_trace_q8_8),
        .o_surrogate_valid(o_surrogate_valid),
        .i_surrogate_ready(i_surrogate_ready),
        .o_surrogate_target(o_surrogate_target),
        .o_surrogate_sample(o_surrogate_sample),
        .o_surrogate_timestep(o_surrogate_timestep),
        .o_surrogate_version(o_surrogate_version),
        .o_surrogate_channel(o_surrogate_channel),
        .o_surrogate_spatial(o_surrogate_spatial),
        .o_surrogate_q8_8(o_surrogate_q8_8),
        .o_student_feature_wave_valid(o_student_feature_wave_valid),
        .i_student_feature_wave_ready(i_student_feature_wave_ready),
        .o_student_feature_wave_mask(o_student_feature_wave_mask),
        .o_student_feature_wave_base(o_student_feature_wave_base),
        .o_student_feature_wave_q8_8_flat(
            o_student_feature_wave_q8_8_flat),
        .o_student_feature_wave_version(o_student_feature_wave_version),
        .o_student_feature_wave_frame_index(
            o_student_feature_wave_frame_index),
        .o_student_feature_wave_first(o_student_feature_wave_first),
        .o_student_feature_wave_last(o_student_feature_wave_last),
        .o_student_feature_wave_sample_reset(
            o_student_feature_wave_sample_reset),
        .o_student_feature_wave_sample_last(
            o_student_feature_wave_sample_last),
        .o_student_sample_label(o_student_sample_label),
        .o_student_logit_wave_valid(o_student_logit_wave_valid),
        .i_student_logit_wave_ready(i_student_logit_wave_ready),
        .o_student_logit_wave_mask(o_student_logit_wave_mask),
        .o_student_logit_wave_base(o_student_logit_wave_base),
        .o_student_logit_wave_count(o_student_logit_wave_count),
        .o_student_logit_wave_q8_8_flat(o_student_logit_wave_q8_8_flat),
        .o_student_logit_wave_version(o_student_logit_wave_version),
        .o_student_logit_wave_frame_index(
            o_student_logit_wave_frame_index),
        .o_student_logit_wave_frame_last(o_student_logit_wave_frame_last),
        .o_student_logit_wave_sample_last(o_student_logit_wave_sample_last),
        .o_busy(router_busy), .o_sticky_error(router_error)
    );

    assign o_busy = coord_busy || prep_busy || prescan_busy || router_busy;
    assign o_sticky_error = coord_error || prep_error || prescan_error ||
        router_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            conv_in_width_q <= '0;
            conv_in_height_q <= '0;
            conv_in_channels_q <= '0;
            conv_padding_q <= '0;
            conv_layer_tag_q <= '0;
            target_feature_count_q <= '0;
            target_trace_leak_q <= '0;
            target_trace_state_reset_q <= 1'b0;
            graph_done_seen_q <= 1'b0;
            target_trace_done_seen_q <= 1'b0;
        end else if (i_abort) begin
            graph_done_seen_q <= 1'b0;
            target_trace_done_seen_q <= 1'b0;
        end else if (i_request_valid && o_request_ready) begin
            conv_in_width_q <= i_conv_in_width;
            conv_in_height_q <= i_conv_in_height;
            conv_in_channels_q <= i_conv_in_channels;
            conv_padding_q <= i_conv_padding;
            conv_layer_tag_q <= i_conv_layer_tag;
            target_feature_count_q <= i_target_feature_count;
            target_trace_leak_q <= i_target_trace_leak_q8_8;
            target_trace_state_reset_q <= i_target_trace_state_reset;
            graph_done_seen_q <= 1'b0;
            target_trace_done_seen_q <= 1'b0;
        end else begin
            if (o_graph_start_valid && i_graph_start_ready) begin
                graph_done_seen_q <= 1'b0;
                target_trace_done_seen_q <= 1'b0;
            end
            if (target_join_active && i_graph_done)
                graph_done_seen_q <= 1'b1;
            if (target_join_active && target_trace_done)
                target_trace_done_seen_q <= 1'b1;
            if (joined_graph_done) begin
                graph_done_seen_q <= 1'b0;
                target_trace_done_seen_q <= 1'b0;
            end
        end
    end

    initial begin
        if (MAX_INPUT_CHANNELS > 512 || MAX_OUTPUT_CHANNELS > 512)
            $error("SCNN graph observation channels exceed encoded metadata");
    end
endmodule

`default_nettype wire
