`timescale 1ns/1ps
`default_nettype none

// Converts one committed recurrent-backbone spike vector at a time into the
// path-major Q14 stream consumed by the SRNN M2 W_h1 learning frontend.  The
// source may expose a padded physical vector (H480); only the configured H450
// logical prefix is emitted and every padded bit must be zero.
module unified_srnn_m2_backbone_observation_adapter #(
    parameter int BATCH_MAX = 64,
    parameter int BACKBONE_MAX = 450,
    parameter int SOURCE_VECTOR_MAX = 480,
    parameter int SAMPLE_ID_WIDTH = 8,
    parameter int BACKBONE_INDEX_WIDTH = 10
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_cfg_batch_count,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_cfg_backbone_count,
    output logic                                      o_cfg_error,

    input  logic                                      i_window_start,
    output logic                                      o_window_start_ready,
    input  logic                                      i_window_include_target,
    input  logic                                      i_abort_window,

    input  logic                                      i_observation_valid,
    output logic                                      o_observation_ready,
    input  logic                                      i_observation_path,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_observation_sample_id,
    input  logic [SOURCE_VECTOR_MAX-1:0]              i_observation_spikes,

    output logic                                      o_backbone_valid,
    input  logic                                      i_backbone_ready,
    output logic                                      o_backbone_path,
    output logic [SAMPLE_ID_WIDTH-1:0]                o_backbone_sample_id,
    output logic [BACKBONE_INDEX_WIDTH-1:0]           o_backbone_id,
    output logic signed [31:0]                        o_backbone_q14,
    output logic                                      o_backbone_last,

    output logic                                      o_busy,
    output logic                                      o_done,
    output logic                                      o_protocol_error
);

    localparam int EMIT_INDEX_WIDTH = $clog2(SOURCE_VECTOR_MAX);

    logic configured_q;
    logic window_open_q;
    logic vector_valid_q;
    logic [SAMPLE_ID_WIDTH-1:0] batch_count_q;
    logic [BACKBONE_INDEX_WIDTH-1:0] backbone_count_q;
    logic include_target_q;
    logic expected_path_q;
    logic [SAMPLE_ID_WIDTH-1:0] expected_sample_q;
    logic [EMIT_INDEX_WIDTH-1:0] emit_id_q;
    logic [SOURCE_VECTOR_MAX-1:0] vector_q;

    logic cfg_legal;
    logic observation_fire;
    logic observation_legal;
    logic observation_tail_nonzero;
    logic backbone_fire;
    logic vector_last;
    logic window_last;

    always_comb begin
        cfg_legal = (i_cfg_batch_count != '0) &&
            (int'(i_cfg_batch_count) <= BATCH_MAX) &&
            (i_cfg_backbone_count != '0) &&
            (int'(i_cfg_backbone_count) <= BACKBONE_MAX) &&
            (int'(i_cfg_backbone_count) <= SOURCE_VECTOR_MAX);

        o_cfg_ready = !window_open_q && !vector_valid_q;
        o_window_start_ready = configured_q && !window_open_q &&
            !vector_valid_q && !o_protocol_error && !i_cfg_valid;
        o_observation_ready = window_open_q && !vector_valid_q &&
            !o_protocol_error;
        observation_fire = i_observation_valid && o_observation_ready;
        observation_tail_nonzero =
            |(i_observation_spikes >> backbone_count_q);
        observation_legal =
            (i_observation_path == expected_path_q) &&
            (i_observation_sample_id == expected_sample_q) &&
            !observation_tail_nonzero;

        vector_last = emit_id_q ==
            EMIT_INDEX_WIDTH'(backbone_count_q - 1'b1);
        window_last = (expected_path_q || !include_target_q) &&
            (expected_sample_q == (batch_count_q - 1'b1)) && vector_last;
        o_backbone_valid = vector_valid_q && !o_protocol_error;
        o_backbone_path = expected_path_q;
        o_backbone_sample_id = expected_sample_q;
        o_backbone_id = BACKBONE_INDEX_WIDTH'(emit_id_q);
        o_backbone_q14 = vector_q[emit_id_q] ? 32'sd16384 : 32'sd0;
        o_backbone_last = o_backbone_valid && window_last;
        backbone_fire = o_backbone_valid && i_backbone_ready;

        o_busy = window_open_q || vector_valid_q;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            configured_q <= 1'b0;
            window_open_q <= 1'b0;
            vector_valid_q <= 1'b0;
            batch_count_q <= '0;
            backbone_count_q <= '0;
            include_target_q <= 1'b0;
            expected_path_q <= 1'b0;
            expected_sample_q <= '0;
            emit_id_q <= '0;
            vector_q <= '0;
            o_cfg_error <= 1'b0;
            o_done <= 1'b0;
            o_protocol_error <= 1'b0;
        end else begin
            o_cfg_error <= 1'b0;
            o_done <= 1'b0;

            if (i_clear_error)
                o_protocol_error <= 1'b0;

            if (i_abort_window) begin
                window_open_q <= 1'b0;
                vector_valid_q <= 1'b0;
                include_target_q <= 1'b0;
                expected_path_q <= 1'b0;
                expected_sample_q <= '0;
                emit_id_q <= '0;
            end

            if (i_cfg_valid && o_cfg_ready) begin
                if (!cfg_legal) begin
                    configured_q <= 1'b0;
                    o_cfg_error <= 1'b1;
                end else begin
                    configured_q <= 1'b1;
                    batch_count_q <= i_cfg_batch_count;
                    backbone_count_q <= i_cfg_backbone_count;
                    o_protocol_error <= 1'b0;
                end
            end

            if (i_window_start && o_window_start_ready) begin
                window_open_q <= 1'b1;
                vector_valid_q <= 1'b0;
                include_target_q <= i_window_include_target;
                expected_path_q <= 1'b0;
                expected_sample_q <= '0;
                emit_id_q <= '0;
            end

            if (observation_fire) begin
                if (!observation_legal) begin
                    o_protocol_error <= 1'b1;
                end else begin
                    vector_q <= i_observation_spikes;
                    vector_valid_q <= 1'b1;
                    emit_id_q <= '0;
                end
            end

            if (backbone_fire) begin
                if (vector_last) begin
                    vector_valid_q <= 1'b0;
                    emit_id_q <= '0;
                    if (window_last) begin
                        window_open_q <= 1'b0;
                        expected_path_q <= 1'b0;
                        expected_sample_q <= '0;
                        o_done <= 1'b1;
                    end else if (expected_sample_q ==
                        (batch_count_q - 1'b1)) begin
                        expected_path_q <= 1'b1;
                        expected_sample_q <= '0;
                    end else begin
                        expected_sample_q <= expected_sample_q + 1'b1;
                    end
                end else begin
                    emit_id_q <= emit_id_q + 1'b1;
                end
            end
        end
    end

endmodule

`default_nettype wire
