`default_nettype none

// Converts one real forward branch's three observation boundaries into the
// path-tagged streams consumed by unified_scnn_tp_selected_layers_t1_path.
// Conv7 pre-pool LIF observations are drained because Conv7 pooled inputs are
// supplied by the prescan; only Conv8 pre-pool surrogate values are forwarded.
module unified_scnn_tp_target_selected_observation_adapter_t1 #(
    parameter int MAX_BATCH = 10,
    parameter int MAX_INPUT_CHANNELS = 512,
    parameter int MAX_OUTPUT_CHANNELS = 512,
    parameter int INPUT_SPATIAL_COUNT = 16,
    parameter int SURROGATE_SPATIAL_COUNT = 16,
    parameter int VERSION_WIDTH = 16,
    parameter int SAMPLE_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int TIMESTEP_WIDTH = 1,
    parameter bit PATH_TARGET = 1'b1
) (
    input  wire logic                                      i_clk,
    input  wire logic                                      i_rst_n,
    input  wire logic                                      i_clear_error,
    input  wire logic [SAMPLE_WIDTH-1:0]                   i_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]                 i_timestep,
    input  wire logic [VERSION_WIDTH-1:0]                  i_version,

    input  wire logic                                      i_prescan_valid,
    output logic                                      o_prescan_ready,
    input  wire logic [15:0]                               i_prescan_channel,
    input  wire logic [15:0]                               i_prescan_spatial,
    input  wire logic [3:0]                                i_prescan_layer,
    input  wire logic [VERSION_WIDTH-1:0]                  i_prescan_version,
    input  wire logic signed [15:0]                        i_prescan_data_q8_8,

    input  wire logic                                      i_lif_valid,
    output logic                                      o_lif_ready,
    input  wire logic [3:0]                                i_lif_layer,
    input  wire logic [15:0]                               i_lif_channel,
    input  wire logic [15:0]                               i_lif_spatial,
    input  wire logic [VERSION_WIDTH-1:0]                  i_lif_version,
    input  wire logic [15:0]                               i_lif_surrogate_q8_8,

    input  wire logic                                      i_trace_valid,
    output logic                                      o_trace_source_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]                   i_trace_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]                 i_trace_timestep,
    input  wire logic [VERSION_WIDTH-1:0]                  i_trace_version,
    input  wire logic [8:0]                                i_trace_channel,
    input  wire logic [1:0]                                i_trace_spatial,
    input  wire logic signed [15:0]                        i_trace_q8_8,

    output logic                                      o_input_valid,
    input  wire logic                                      i_input_ready,
    output logic                                      o_input_target,
    output logic [SAMPLE_WIDTH-1:0]                   o_input_sample,
    output logic [TIMESTEP_WIDTH-1:0]                 o_input_timestep,
    output logic [VERSION_WIDTH-1:0]                  o_input_version,
    output logic [8:0]                                o_input_channel,
    output logic [3:0]                                o_input_spatial,
    output logic signed [15:0]                        o_input_q8_8,

    output logic                                      o_surrogate_valid,
    input  wire logic                                      i_surrogate_ready,
    output logic                                      o_surrogate_target,
    output logic [SAMPLE_WIDTH-1:0]                   o_surrogate_sample,
    output logic [TIMESTEP_WIDTH-1:0]                 o_surrogate_timestep,
    output logic [VERSION_WIDTH-1:0]                  o_surrogate_version,
    output logic [8:0]                                o_surrogate_channel,
    output logic [3:0]                                o_surrogate_spatial,
    output logic signed [15:0]                        o_surrogate_q8_8,

    output logic                                      o_trace_valid,
    input  wire logic                                      i_trace_ready,
    output logic                                      o_trace_target,
    output logic [SAMPLE_WIDTH-1:0]                   o_trace_sample,
    output logic [TIMESTEP_WIDTH-1:0]                 o_trace_timestep,
    output logic [VERSION_WIDTH-1:0]                  o_trace_version,
    output logic [8:0]                                o_trace_channel,
    output logic [1:0]                                o_trace_spatial,
    output logic signed [15:0]                        o_trace_q8_8,
    output logic                                      o_protocol_error
);
    logic prescan_legal;
    logic lif_selected;
    logic lif_legal;
    logic trace_legal;

    assign prescan_legal = i_prescan_layer == 7 &&
        i_prescan_channel < MAX_INPUT_CHANNELS &&
        i_prescan_spatial < INPUT_SPATIAL_COUNT &&
        i_prescan_version == i_version;
    assign lif_selected = i_lif_layer == 7;
    assign lif_legal = i_lif_channel < MAX_OUTPUT_CHANNELS &&
        i_lif_spatial < SURROGATE_SPATIAL_COUNT &&
        i_lif_version == i_version;
    assign trace_legal = i_trace_sample == i_sample &&
        i_trace_timestep == i_timestep && i_trace_version == i_version &&
        i_trace_channel < MAX_OUTPUT_CHANNELS;

    assign o_input_valid = i_prescan_valid && prescan_legal &&
        !o_protocol_error;
    assign o_prescan_ready = i_input_ready && prescan_legal &&
        !o_protocol_error;
    assign o_input_target = PATH_TARGET;
    assign o_input_sample = i_sample;
    assign o_input_timestep = i_timestep;
    assign o_input_version = i_prescan_version;
    assign o_input_channel = i_prescan_channel[8:0];
    assign o_input_spatial = i_prescan_spatial[3:0];
    assign o_input_q8_8 = i_prescan_data_q8_8;

    assign o_surrogate_valid = i_lif_valid && lif_selected && lif_legal &&
        !o_protocol_error;
    assign o_lif_ready = !lif_selected ||
        (i_surrogate_ready && lif_legal && !o_protocol_error);
    assign o_surrogate_target = PATH_TARGET;
    assign o_surrogate_sample = i_sample;
    assign o_surrogate_timestep = i_timestep;
    assign o_surrogate_version = i_lif_version;
    assign o_surrogate_channel = i_lif_channel[8:0];
    assign o_surrogate_spatial = i_lif_spatial[3:0];
    assign o_surrogate_q8_8 = $signed(i_lif_surrogate_q8_8);

    assign o_trace_valid = i_trace_valid && trace_legal &&
        !o_protocol_error;
    assign o_trace_source_ready = i_trace_ready && trace_legal &&
        !o_protocol_error;
    assign o_trace_target = PATH_TARGET;
    assign o_trace_sample = i_trace_sample;
    assign o_trace_timestep = i_trace_timestep;
    assign o_trace_version = i_trace_version;
    assign o_trace_channel = i_trace_channel;
    assign o_trace_spatial = i_trace_spatial;
    assign o_trace_q8_8 = i_trace_q8_8;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            o_protocol_error <= 1'b0;
        end else begin
            if (i_clear_error)
                o_protocol_error <= 1'b0;
            if ((i_prescan_valid && !prescan_legal) ||
                (i_lif_valid && lif_selected && !lif_legal) ||
                (i_trace_valid && !trace_legal))
                o_protocol_error <= 1'b1;
        end
    end
endmodule

`default_nettype wire
