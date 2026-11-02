`timescale 1ns/1ps
`default_nettype none

// Routes observations from the currently locked target/student graph layer
// into the common selected-layer learner streams. Numeric state remains in
// the final core and the two existing observation adapters.
module unified_scnn_tp_actual_graph_observation_binding #(
    parameter int MAX_BATCH = 2,
    parameter int MAX_INPUT_CHANNELS = 512,
    parameter int MAX_OUTPUT_CHANNELS = 512,
    parameter int INPUT_SPATIAL_COUNT = 16,
    parameter int SURROGATE_SPATIAL_COUNT = 16,
    parameter int VERSION_WIDTH = 16,
    parameter int SAMPLE_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int TIMESTEP_WIDTH = 1
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_clear_error,
    input  wire logic                         i_enable,
    input  wire logic                         i_active_target,
    input  wire logic                         i_active_student,
    input  wire logic [SAMPLE_WIDTH-1:0]      i_target_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]    i_target_timestep,
    input  wire logic [VERSION_WIDTH-1:0]     i_target_version,
    input  wire logic [SAMPLE_WIDTH-1:0]      i_student_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]    i_student_timestep,
    input  wire logic [VERSION_WIDTH-1:0]     i_student_version,

    input  wire logic                         i_prescan_valid,
    output logic                         o_prescan_ready,
    input  wire logic [15:0]                  i_prescan_channel,
    input  wire logic [15:0]                  i_prescan_spatial,
    input  wire logic [3:0]                   i_prescan_layer,
    input  wire logic [VERSION_WIDTH-1:0]     i_prescan_version,
    input  wire logic signed [15:0]           i_prescan_data_q8_8,

    input  wire logic                         i_lif_valid,
    output logic                         o_lif_ready,
    input  wire logic [3:0]                   i_lif_layer,
    input  wire logic [15:0]                  i_lif_channel,
    input  wire logic [15:0]                  i_lif_spatial,
    input  wire logic [VERSION_WIDTH-1:0]     i_lif_version,
    input  wire logic [15:0]                  i_lif_surrogate_q8_8,

    input  wire logic                         i_trace_valid,
    output logic                         o_trace_source_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]      i_trace_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]    i_trace_timestep,
    input  wire logic [VERSION_WIDTH-1:0]     i_trace_version,
    input  wire logic [8:0]                   i_trace_channel,
    input  wire logic [1:0]                   i_trace_spatial,
    input  wire logic signed [15:0]           i_trace_q8_8,

    output logic                         o_input_valid,
    input  wire logic                         i_input_ready,
    output logic                         o_input_target,
    output logic [SAMPLE_WIDTH-1:0]      o_input_sample,
    output logic [TIMESTEP_WIDTH-1:0]    o_input_timestep,
    output logic [VERSION_WIDTH-1:0]     o_input_version,
    output logic [8:0]                   o_input_channel,
    output logic [3:0]                   o_input_spatial,
    output logic signed [15:0]           o_input_q8_8,

    output logic                         o_surrogate_valid,
    input  wire logic                         i_surrogate_ready,
    output logic                         o_surrogate_target,
    output logic [SAMPLE_WIDTH-1:0]      o_surrogate_sample,
    output logic [TIMESTEP_WIDTH-1:0]    o_surrogate_timestep,
    output logic [VERSION_WIDTH-1:0]     o_surrogate_version,
    output logic [8:0]                   o_surrogate_channel,
    output logic [3:0]                   o_surrogate_spatial,
    output logic signed [15:0]           o_surrogate_q8_8,

    output logic                         o_trace_valid,
    input  wire logic                         i_trace_ready,
    output logic                         o_trace_target,
    output logic [SAMPLE_WIDTH-1:0]      o_trace_sample,
    output logic [TIMESTEP_WIDTH-1:0]    o_trace_timestep,
    output logic [VERSION_WIDTH-1:0]     o_trace_version,
    output logic [8:0]                   o_trace_channel,
    output logic [1:0]                   o_trace_spatial,
    output logic signed [15:0]           o_trace_q8_8,
    output logic                         o_protocol_error
);
    logic target_prescan_ready, student_prescan_ready;
    logic target_lif_ready, student_lif_ready;
    logic target_trace_source_ready, student_trace_source_ready;
    logic target_input_valid, student_input_valid;
    logic target_input_target, student_input_target;
    logic [SAMPLE_WIDTH-1:0] target_input_sample, student_input_sample;
    logic [TIMESTEP_WIDTH-1:0]
        target_input_timestep, student_input_timestep;
    logic [VERSION_WIDTH-1:0] target_input_version, student_input_version;
    logic [8:0] target_input_channel, student_input_channel;
    logic [3:0] target_input_spatial, student_input_spatial;
    logic signed [15:0] target_input_data, student_input_data;
    logic target_surrogate_valid, student_surrogate_valid;
    logic target_surrogate_target, student_surrogate_target;
    logic [SAMPLE_WIDTH-1:0]
        target_surrogate_sample, student_surrogate_sample;
    logic [TIMESTEP_WIDTH-1:0]
        target_surrogate_timestep, student_surrogate_timestep;
    logic [VERSION_WIDTH-1:0]
        target_surrogate_version, student_surrogate_version;
    logic [8:0] target_surrogate_channel, student_surrogate_channel;
    logic [3:0] target_surrogate_spatial, student_surrogate_spatial;
    logic signed [15:0] target_surrogate_data, student_surrogate_data;
    logic target_trace_valid, student_trace_valid;
    logic target_trace_target, student_trace_target;
    logic [SAMPLE_WIDTH-1:0] target_trace_sample, student_trace_sample;
    logic [TIMESTEP_WIDTH-1:0]
        target_trace_timestep, student_trace_timestep;
    logic [VERSION_WIDTH-1:0] target_trace_version, student_trace_version;
    logic [8:0] target_trace_channel, student_trace_channel;
    logic [1:0] target_trace_spatial, student_trace_spatial;
    logic signed [15:0] target_trace_data, student_trace_data;
    logic target_error, student_error, owner_error_q;

    assign o_prescan_ready = i_enable && i_active_target ?
        target_prescan_ready : i_enable && i_active_student ?
        student_prescan_ready : 1'b0;
    assign o_lif_ready = i_enable && i_active_target ? target_lif_ready :
        i_enable && i_active_student ? student_lif_ready : 1'b0;
    assign o_trace_source_ready = i_enable && i_active_target ?
        target_trace_source_ready : i_active_student ?
        (i_enable && student_trace_source_ready) : 1'b0;

    assign o_input_valid = i_active_target ? target_input_valid :
        i_active_student ? student_input_valid : 1'b0;
    assign o_input_target = i_active_target ? target_input_target :
        student_input_target;
    assign o_input_sample = i_active_target ? target_input_sample :
        student_input_sample;
    assign o_input_timestep = i_active_target ? target_input_timestep :
        student_input_timestep;
    assign o_input_version = i_active_target ? target_input_version :
        student_input_version;
    assign o_input_channel = i_active_target ? target_input_channel :
        student_input_channel;
    assign o_input_spatial = i_active_target ? target_input_spatial :
        student_input_spatial;
    assign o_input_q8_8 = i_active_target ? target_input_data :
        student_input_data;

    assign o_surrogate_valid = i_active_target ? target_surrogate_valid :
        i_active_student ? student_surrogate_valid : 1'b0;
    assign o_surrogate_target = i_active_target ? target_surrogate_target :
        student_surrogate_target;
    assign o_surrogate_sample = i_active_target ? target_surrogate_sample :
        student_surrogate_sample;
    assign o_surrogate_timestep = i_active_target ?
        target_surrogate_timestep : student_surrogate_timestep;
    assign o_surrogate_version = i_active_target ?
        target_surrogate_version : student_surrogate_version;
    assign o_surrogate_channel = i_active_target ?
        target_surrogate_channel : student_surrogate_channel;
    assign o_surrogate_spatial = i_active_target ?
        target_surrogate_spatial : student_surrogate_spatial;
    assign o_surrogate_q8_8 = i_active_target ? target_surrogate_data :
        student_surrogate_data;

    assign o_trace_valid = i_active_target ? target_trace_valid :
        i_active_student ? student_trace_valid : 1'b0;
    assign o_trace_target = i_active_target ? target_trace_target :
        student_trace_target;
    assign o_trace_sample = i_active_target ? target_trace_sample :
        student_trace_sample;
    assign o_trace_timestep = i_active_target ? target_trace_timestep :
        student_trace_timestep;
    assign o_trace_version = i_active_target ? target_trace_version :
        student_trace_version;
    assign o_trace_channel = i_active_target ? target_trace_channel :
        student_trace_channel;
    assign o_trace_spatial = i_active_target ? target_trace_spatial :
        student_trace_spatial;
    assign o_trace_q8_8 = i_active_target ? target_trace_data :
        student_trace_data;
    assign o_protocol_error = owner_error_q || target_error || student_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            owner_error_q <= 1'b0;
        else begin
            if (i_clear_error)
                owner_error_q <= 1'b0;
            if (i_enable && ((i_active_target && i_active_student) ||
                ((i_prescan_valid || i_lif_valid || i_trace_valid) &&
                 !(i_active_target ^ i_active_student))))
                owner_error_q <= 1'b1;
        end
    end

    unified_scnn_tp_target_selected_observation_adapter_t1 #(
        .MAX_BATCH(MAX_BATCH),
        .MAX_INPUT_CHANNELS(MAX_INPUT_CHANNELS),
        .MAX_OUTPUT_CHANNELS(MAX_OUTPUT_CHANNELS),
        .INPUT_SPATIAL_COUNT(INPUT_SPATIAL_COUNT),
        .SURROGATE_SPATIAL_COUNT(SURROGATE_SPATIAL_COUNT),
        .VERSION_WIDTH(VERSION_WIDTH), .TIMESTEP_WIDTH(TIMESTEP_WIDTH),
        .PATH_TARGET(1'b1)
    ) u_target (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_sample(i_target_sample), .i_timestep(i_target_timestep),
        .i_version(i_target_version),
        .i_prescan_valid(i_enable && i_prescan_valid && i_active_target),
        .o_prescan_ready(target_prescan_ready),
        .i_prescan_channel(i_prescan_channel),
        .i_prescan_spatial(i_prescan_spatial),
        .i_prescan_layer(i_prescan_layer),
        .i_prescan_version(i_prescan_version),
        .i_prescan_data_q8_8(i_prescan_data_q8_8),
        .i_lif_valid(i_enable && i_lif_valid && i_active_target),
        .o_lif_ready(target_lif_ready), .i_lif_layer(i_lif_layer),
        .i_lif_channel(i_lif_channel), .i_lif_spatial(i_lif_spatial),
        .i_lif_version(i_lif_version),
        .i_lif_surrogate_q8_8(i_lif_surrogate_q8_8),
        .i_trace_valid(i_enable && i_trace_valid && i_active_target),
        .o_trace_source_ready(target_trace_source_ready),
        .i_trace_sample(i_trace_sample),
        .i_trace_timestep(i_trace_timestep),
        .i_trace_version(i_trace_version),
        .i_trace_channel(i_trace_channel),
        .i_trace_spatial(i_trace_spatial), .i_trace_q8_8(i_trace_q8_8),
        .o_input_valid(target_input_valid), .i_input_ready(i_input_ready),
        .o_input_target(target_input_target),
        .o_input_sample(target_input_sample),
        .o_input_timestep(target_input_timestep),
        .o_input_version(target_input_version),
        .o_input_channel(target_input_channel),
        .o_input_spatial(target_input_spatial),
        .o_input_q8_8(target_input_data),
        .o_surrogate_valid(target_surrogate_valid),
        .i_surrogate_ready(i_surrogate_ready),
        .o_surrogate_target(target_surrogate_target),
        .o_surrogate_sample(target_surrogate_sample),
        .o_surrogate_timestep(target_surrogate_timestep),
        .o_surrogate_version(target_surrogate_version),
        .o_surrogate_channel(target_surrogate_channel),
        .o_surrogate_spatial(target_surrogate_spatial),
        .o_surrogate_q8_8(target_surrogate_data),
        .o_trace_valid(target_trace_valid), .i_trace_ready(i_trace_ready),
        .o_trace_target(target_trace_target),
        .o_trace_sample(target_trace_sample),
        .o_trace_timestep(target_trace_timestep),
        .o_trace_version(target_trace_version),
        .o_trace_channel(target_trace_channel),
        .o_trace_spatial(target_trace_spatial),
        .o_trace_q8_8(target_trace_data), .o_protocol_error(target_error)
    );

    unified_scnn_tp_target_selected_observation_adapter_t1 #(
        .MAX_BATCH(MAX_BATCH),
        .MAX_INPUT_CHANNELS(MAX_INPUT_CHANNELS),
        .MAX_OUTPUT_CHANNELS(MAX_OUTPUT_CHANNELS),
        .INPUT_SPATIAL_COUNT(INPUT_SPATIAL_COUNT),
        .SURROGATE_SPATIAL_COUNT(SURROGATE_SPATIAL_COUNT),
        .VERSION_WIDTH(VERSION_WIDTH), .TIMESTEP_WIDTH(TIMESTEP_WIDTH),
        .PATH_TARGET(1'b0)
    ) u_student (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_sample(i_student_sample), .i_timestep(i_student_timestep),
        .i_version(i_student_version),
        .i_prescan_valid(i_enable && i_prescan_valid && i_active_student),
        .o_prescan_ready(student_prescan_ready),
        .i_prescan_channel(i_prescan_channel),
        .i_prescan_spatial(i_prescan_spatial),
        .i_prescan_layer(i_prescan_layer),
        .i_prescan_version(i_prescan_version),
        .i_prescan_data_q8_8(i_prescan_data_q8_8),
        .i_lif_valid(i_enable && i_lif_valid && i_active_student),
        .o_lif_ready(student_lif_ready), .i_lif_layer(i_lif_layer),
        .i_lif_channel(i_lif_channel), .i_lif_spatial(i_lif_spatial),
        .i_lif_version(i_lif_version),
        .i_lif_surrogate_q8_8(i_lif_surrogate_q8_8),
        .i_trace_valid(i_enable && i_trace_valid && i_active_student),
        .o_trace_source_ready(student_trace_source_ready),
        .i_trace_sample(i_trace_sample),
        .i_trace_timestep(i_trace_timestep),
        .i_trace_version(i_trace_version),
        .i_trace_channel(i_trace_channel),
        .i_trace_spatial(i_trace_spatial), .i_trace_q8_8(i_trace_q8_8),
        .o_input_valid(student_input_valid), .i_input_ready(i_input_ready),
        .o_input_target(student_input_target),
        .o_input_sample(student_input_sample),
        .o_input_timestep(student_input_timestep),
        .o_input_version(student_input_version),
        .o_input_channel(student_input_channel),
        .o_input_spatial(student_input_spatial),
        .o_input_q8_8(student_input_data),
        .o_surrogate_valid(student_surrogate_valid),
        .i_surrogate_ready(i_surrogate_ready),
        .o_surrogate_target(student_surrogate_target),
        .o_surrogate_sample(student_surrogate_sample),
        .o_surrogate_timestep(student_surrogate_timestep),
        .o_surrogate_version(student_surrogate_version),
        .o_surrogate_channel(student_surrogate_channel),
        .o_surrogate_spatial(student_surrogate_spatial),
        .o_surrogate_q8_8(student_surrogate_data),
        .o_trace_valid(student_trace_valid), .i_trace_ready(i_trace_ready),
        .o_trace_target(student_trace_target),
        .o_trace_sample(student_trace_sample),
        .o_trace_timestep(student_trace_timestep),
        .o_trace_version(student_trace_version),
        .o_trace_channel(student_trace_channel),
        .o_trace_spatial(student_trace_spatial),
        .o_trace_q8_8(student_trace_data), .o_protocol_error(student_error)
    );
endmodule

`default_nettype wire
