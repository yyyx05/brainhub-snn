`timescale 1ns/1ps
`default_nettype none

// Emits every selected convolution input exactly once in CHW order while
// borrowing the existing binary feature-window service.  Setting the sole
// active kernel coordinate equal to padding makes the service return the raw
// input row, so no second feature-map store is required.
module unified_scnn_h40_tp_input_prescan_v1 #(
    parameter int MAX_CHANNELS = 512,
    parameter int MAX_WIDTH = 256,
    parameter int SAMPLE_WIDTH = 4,
    parameter int TIMESTEP_WIDTH = 5,
    parameter int VERSION_WIDTH = 16
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_abort,
    input  wire logic                         i_clear_error,

    input  wire logic                         i_start_valid,
    output logic                              o_start_ready,
    input  wire logic [15:0]                  i_width,
    input  wire logic [15:0]                  i_height,
    input  wire logic [15:0]                  i_channels,
    input  wire logic [7:0]                   i_padding,
    input  wire logic                         i_target,
    input  wire logic [SAMPLE_WIDTH-1:0]      i_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]    i_timestep,
    input  wire logic [VERSION_WIDTH-1:0]     i_version,
    input  wire logic [19:0]                  i_layer_tag,

    output logic                              o_feature_request_valid,
    input  wire logic                         i_feature_request_ready,
    output logic [3:0]                        o_feature_term_valid,
    output logic [39:0]                       o_feature_input_channel_flat,
    output logic [7:0]                        o_feature_kernel_row_flat,
    output logic [7:0]                        o_feature_kernel_col_flat,
    output logic [31:0]                       o_feature_spatial_base,
    output logic [5:0]                        o_feature_spatial_count,
    output logic [15:0]                       o_feature_output_row,
    output logic [15:0]                       o_feature_output_col,
    input  wire logic                         i_feature_response_valid,
    output logic                              o_feature_response_ready,
    input  wire logic [39:0]                  i_feature_mask0,
    input  wire logic [39:0]                  i_feature_mask1,
    input  wire logic [39:0]                  i_feature_mask2,
    input  wire logic [39:0]                  i_feature_mask3,
    input  wire logic                         i_feature_service_error,

    output logic                              o_observation_valid,
    input  wire logic                         i_observation_ready,
    output logic                              o_observation_target,
    output logic [SAMPLE_WIDTH-1:0]           o_observation_sample,
    output logic [TIMESTEP_WIDTH-1:0]         o_observation_timestep,
    output logic [VERSION_WIDTH-1:0]          o_observation_version,
    output logic [19:0]                       o_observation_layer_tag,
    output logic [9:0]                        o_observation_channel,
    output logic [31:0]                       o_observation_spatial,
    output logic signed [15:0]                o_observation_q8_8,
    output logic                              o_observation_last,
    output logic                              o_done,
    output logic                              o_busy,
    output logic                              o_sticky_error
);
    typedef enum logic [1:0] {IDLE, REQUEST, RESPONSE, SERIALIZE} state_t;
    state_t state_q;

    logic [15:0] width_q, height_q, channels_q;
    logic [7:0] padding_q;
    logic target_q;
    logic [SAMPLE_WIDTH-1:0] sample_q;
    logic [TIMESTEP_WIDTH-1:0] timestep_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic [19:0] layer_tag_q;
    logic [15:0] channel_q, row_q, column_base_q;
    logic [5:0] wave_count_q, lane_q;
    logic [39:0] response_mask_q;
    logic start_legal;
    logic final_lane, final_wave, final_row, final_channel;
    integer unsigned remaining_columns;

    always_comb begin
        start_legal = i_width != 0 && i_width <= 16'(MAX_WIDTH) &&
            i_height != 0 && i_channels != 0 &&
            i_channels <= 16'(MAX_CHANNELS) && i_padding <= 8'd2;

        remaining_columns = integer'(width_q) - integer'(column_base_q);
        o_start_ready = state_q == IDLE && !o_sticky_error;
        o_feature_request_valid = state_q == REQUEST;
        o_feature_term_valid = 4'b0001;
        o_feature_input_channel_flat = '0;
        o_feature_input_channel_flat[9:0] = channel_q[9:0];
        o_feature_kernel_row_flat = '0;
        o_feature_kernel_col_flat = '0;
        o_feature_kernel_row_flat[1:0] = padding_q[1:0];
        o_feature_kernel_col_flat[1:0] = padding_q[1:0];
        o_feature_spatial_base = 32'(integer'(row_q) * integer'(width_q) +
            integer'(column_base_q));
        o_feature_spatial_count = remaining_columns > 40 ?
            6'd40 : 6'(remaining_columns);
        o_feature_output_row = row_q;
        o_feature_output_col = column_base_q;
        o_feature_response_ready = state_q == RESPONSE;

        o_observation_valid = state_q == SERIALIZE;
        o_observation_target = target_q;
        o_observation_sample = sample_q;
        o_observation_timestep = timestep_q;
        o_observation_version = version_q;
        o_observation_layer_tag = layer_tag_q;
        o_observation_channel = channel_q[9:0];
        o_observation_spatial = 32'(integer'(row_q) * integer'(width_q) +
            integer'(column_base_q) + integer'(lane_q));
        o_observation_q8_8 = response_mask_q[lane_q] ? 16'sh0100 : 16'sh0000;
        final_lane = lane_q + 1'b1 == wave_count_q;
        final_wave = integer'(column_base_q) + integer'(wave_count_q) ==
            integer'(width_q);
        final_row = row_q + 1'b1 == height_q;
        final_channel = channel_q + 1'b1 == channels_q;
        o_observation_last = final_lane && final_wave && final_row &&
            final_channel;
        o_busy = state_q != IDLE;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= IDLE;
            width_q <= '0;
            height_q <= '0;
            channels_q <= '0;
            padding_q <= '0;
            target_q <= 1'b0;
            sample_q <= '0;
            timestep_q <= '0;
            version_q <= '0;
            layer_tag_q <= '0;
            channel_q <= '0;
            row_q <= '0;
            column_base_q <= '0;
            wave_count_q <= '0;
            lane_q <= '0;
            response_mask_q <= '0;
            o_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= IDLE;
            o_done <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_clear_error && state_q == IDLE)
                o_sticky_error <= 1'b0;
            if (i_feature_service_error && state_q != IDLE) begin
                state_q <= IDLE;
                o_sticky_error <= 1'b1;
            end else begin
                case (state_q)
                    IDLE: begin
                        if (i_start_valid && o_start_ready) begin
                            if (!start_legal) begin
                                o_sticky_error <= 1'b1;
                            end else begin
                                width_q <= i_width;
                                height_q <= i_height;
                                channels_q <= i_channels;
                                padding_q <= i_padding;
                                target_q <= i_target;
                                sample_q <= i_sample;
                                timestep_q <= i_timestep;
                                version_q <= i_version;
                                layer_tag_q <= i_layer_tag;
                                channel_q <= '0;
                                row_q <= '0;
                                column_base_q <= '0;
                                state_q <= REQUEST;
                            end
                        end
                    end
                    REQUEST: begin
                        if (o_feature_request_valid &&
                                i_feature_request_ready)
                            state_q <= RESPONSE;
                    end
                    RESPONSE: begin
                        if (i_feature_response_valid &&
                                o_feature_response_ready) begin
                            if (i_feature_mask1 != 0 ||
                                    i_feature_mask2 != 0 ||
                                    i_feature_mask3 != 0) begin
                                state_q <= IDLE;
                                o_sticky_error <= 1'b1;
                            end else begin
                                response_mask_q <= i_feature_mask0;
                                wave_count_q <= o_feature_spatial_count;
                                lane_q <= '0;
                                state_q <= SERIALIZE;
                            end
                        end
                    end
                    SERIALIZE: begin
                        if (o_observation_valid && i_observation_ready) begin
                            if (!final_lane) begin
                                lane_q <= lane_q + 1'b1;
                            end else if (!final_wave) begin
                                column_base_q <= column_base_q +
                                    16'(wave_count_q);
                                state_q <= REQUEST;
                            end else if (!final_row) begin
                                row_q <= row_q + 1'b1;
                                column_base_q <= '0;
                                state_q <= REQUEST;
                            end else if (!final_channel) begin
                                channel_q <= channel_q + 1'b1;
                                row_q <= '0;
                                column_base_q <= '0;
                                state_q <= REQUEST;
                            end else begin
                                state_q <= IDLE;
                                o_done <= 1'b1;
                            end
                        end
                    end
                    default: begin
                        state_q <= IDLE;
                        o_sticky_error <= 1'b1;
                    end
                endcase
            end
        end
    end

    initial begin
        if (MAX_CHANNELS < 1 || MAX_CHANNELS > 1024 || MAX_WIDTH < 1)
            $error("SCNN input prescan parameter mismatch");
    end
endmodule

`default_nettype wire
