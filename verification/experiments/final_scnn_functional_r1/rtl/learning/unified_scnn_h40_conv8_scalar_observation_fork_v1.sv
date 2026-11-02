`timescale 1ns/1ps
`default_nettype none

// Atomically forwards one serialized Conv8 result to the classifier and,
// when enabled, to the selected-layer TP observer. The observer adds only the
// shared type-1 surrogate lookup; it does not duplicate convolution, LIF, or
// state storage. The upstream token retires after both enabled consumers have
// accepted it exactly once.
module unified_scnn_h40_conv8_scalar_observation_fork_v1 #(
    parameter int SAMPLE_WIDTH = 4,
    parameter int TIMESTEP_WIDTH = 5,
    parameter int VERSION_WIDTH = 16,
    parameter SURROGATE_ROM_FILE =
        "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem"
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_abort,
    input  wire logic                         i_clear_error,
    input  wire logic                         i_observation_enable,
    input  wire logic                         i_target,
    input  wire logic [SAMPLE_WIDTH-1:0]      i_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]    i_timestep,
    input  wire logic [VERSION_WIDTH-1:0]     i_version,
    input  wire logic signed [15:0]           i_threshold_q8_8,

    input  wire logic                         i_scalar_valid,
    output logic                              o_scalar_ready,
    input  wire logic [19:0]                  i_scalar_layer_tag,
    input  wire logic [9:0]                   i_scalar_channel,
    input  wire logic [31:0]                  i_scalar_spatial,
    input  wire logic                         i_scalar_spike,
    input  wire logic signed [15:0]           i_scalar_pre_state_q8_8,

    output logic                              o_forward_valid,
    input  wire logic                         i_forward_ready,
    output logic [19:0]                       o_forward_layer_tag,
    output logic [9:0]                        o_forward_channel,
    output logic [31:0]                       o_forward_spatial,
    output logic                              o_forward_spike,

    output logic                              o_observation_valid,
    input  wire logic                         i_observation_ready,
    output logic                              o_observation_target,
    output logic [SAMPLE_WIDTH-1:0]           o_observation_sample,
    output logic [TIMESTEP_WIDTH-1:0]         o_observation_timestep,
    output logic [VERSION_WIDTH-1:0]          o_observation_version,
    output logic [19:0]                       o_observation_layer_tag,
    output logic [9:0]                        o_observation_channel,
    output logic [31:0]                       o_observation_spatial,
    output logic signed [15:0]                o_observation_pre_state_q8_8,
    output logic signed [15:0]                o_observation_surrogate_q8_8,

    output logic                              o_busy,
    output logic                              o_sticky_error
);
    logic forward_accepted_q;
    logic observer_issued_q;
    logic forward_fire;
    logic observer_issue_fire;
    logic token_fire;

    logic metadata_valid_q;
    logic metadata_ready;
    logic surrogate_valid;
    logic surrogate_ready;
    logic [15:0] surrogate_value;

    logic observation_target_q;
    logic [SAMPLE_WIDTH-1:0] observation_sample_q;
    logic [TIMESTEP_WIDTH-1:0] observation_timestep_q;
    logic [VERSION_WIDTH-1:0] observation_version_q;
    logic [19:0] observation_layer_tag_q;
    logic [9:0] observation_channel_q;
    logic [31:0] observation_spatial_q;
    logic signed [15:0] observation_pre_state_q;

    assign o_forward_valid = i_scalar_valid && !forward_accepted_q &&
        !o_sticky_error;
    assign o_forward_layer_tag = i_scalar_layer_tag;
    assign o_forward_channel = i_scalar_channel;
    assign o_forward_spatial = i_scalar_spatial;
    assign o_forward_spike = i_scalar_spike;
    assign forward_fire = o_forward_valid && i_forward_ready;

    assign metadata_ready = !metadata_valid_q || i_observation_ready;
    assign observer_issue_fire = i_scalar_valid &&
        i_observation_enable && !observer_issued_q && metadata_ready &&
        surrogate_ready && !o_sticky_error;
    assign o_scalar_ready = !o_sticky_error &&
        (forward_accepted_q || i_forward_ready) &&
        (observer_issued_q || !i_observation_enable ||
         (metadata_ready && surrogate_ready));
    assign token_fire = i_scalar_valid && o_scalar_ready;

    assign o_observation_valid = metadata_valid_q && surrogate_valid &&
        !o_sticky_error;
    assign o_observation_target = observation_target_q;
    assign o_observation_sample = observation_sample_q;
    assign o_observation_timestep = observation_timestep_q;
    assign o_observation_version = observation_version_q;
    assign o_observation_layer_tag = observation_layer_tag_q;
    assign o_observation_channel = observation_channel_q;
    assign o_observation_spatial = observation_spatial_q;
    assign o_observation_pre_state_q8_8 = observation_pre_state_q;
    assign o_observation_surrogate_q8_8 = $signed(surrogate_value);
    assign o_busy = i_scalar_valid || forward_accepted_q ||
        observer_issued_q || metadata_valid_q || surrogate_valid;

    unified_scnn_tp_surrogate_type1_q8_8 #(
        .ROM_FILE(SURROGATE_ROM_FILE)
    ) u_surrogate (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_valid(observer_issue_fire), .o_ready(surrogate_ready),
        .i_membrane_pre_q8_8(i_scalar_pre_state_q8_8),
        .i_threshold_q8_8(i_threshold_q8_8),
        .o_valid(surrogate_valid), .i_ready(i_observation_ready),
        .o_surrogate_q8_8(surrogate_value)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            forward_accepted_q <= 1'b0;
            observer_issued_q <= 1'b0;
            metadata_valid_q <= 1'b0;
            observation_target_q <= 1'b0;
            observation_sample_q <= '0;
            observation_timestep_q <= '0;
            observation_version_q <= '0;
            observation_layer_tag_q <= '0;
            observation_channel_q <= '0;
            observation_spatial_q <= '0;
            observation_pre_state_q <= '0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            forward_accepted_q <= 1'b0;
            observer_issued_q <= 1'b0;
            metadata_valid_q <= 1'b0;
            o_sticky_error <= 1'b0;
        end else begin
            if (i_clear_error && !o_busy)
                o_sticky_error <= 1'b0;
            if (metadata_valid_q != surrogate_valid)
                o_sticky_error <= 1'b1;

            if (token_fire) begin
                forward_accepted_q <= 1'b0;
                observer_issued_q <= 1'b0;
            end else begin
                if (forward_fire)
                    forward_accepted_q <= 1'b1;
                if (observer_issue_fire)
                    observer_issued_q <= 1'b1;
            end

            if (metadata_ready) begin
                metadata_valid_q <= observer_issue_fire;
                if (observer_issue_fire) begin
                    observation_target_q <= i_target;
                    observation_sample_q <= i_sample;
                    observation_timestep_q <= i_timestep;
                    observation_version_q <= i_version;
                    observation_layer_tag_q <= i_scalar_layer_tag;
                    observation_channel_q <= i_scalar_channel;
                    observation_spatial_q <= i_scalar_spatial;
                    observation_pre_state_q <= i_scalar_pre_state_q8_8;
                end
            end
        end
    end
endmodule

`default_nettype wire
