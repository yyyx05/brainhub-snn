`timescale 1ns/1ps
`default_nettype none

// Captures the committed, post-pooling feature frame and replays it as strict
// channel-major Q8.8 waves. Each graph commit covers one contiguous slice; the
// scheduler guarantees that every frame position is committed exactly once,
// even though pooled rows can interleave channels. Exploiting that internal
// uniqueness contract removes the 2048-entry duplicate bitmap: three-bit
// quarter-spike levels are striped across LANES small memories.
module unified_scnn_h40_classifier_feature_commit_adapter_v1 #(
    parameter int MAX_CHANNELS = 512,
    parameter int SPATIAL = 4,
    parameter int LANES = 8,
    parameter int VERSION_WIDTH = 16,
    parameter int FRAME_WIDTH = 8,
    parameter int CHANNEL_WIDTH = MAX_CHANNELS <= 2 ? 1 : $clog2(MAX_CHANNELS),
    parameter int CHANNEL_COUNT_WIDTH = $clog2(MAX_CHANNELS + 1),
    parameter int FEATURE_WIDTH = MAX_CHANNELS * SPATIAL <= 2 ? 1 :
        $clog2(MAX_CHANNELS * SPATIAL),
    parameter int CLASS_WIDTH = 4
) (
    input  wire logic                           i_clk,
    input  wire logic                           i_rst_n,
    input  wire logic                           i_abort,
    input  wire logic                           i_clear_error,

    input  wire logic                           i_cfg_valid,
    output logic                                o_cfg_ready,
    input  wire logic [19:0]                    i_cfg_layer_tag,
    input  wire logic [CHANNEL_WIDTH-1:0]       i_cfg_channel_base,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0] i_cfg_channel_count,
    input  wire logic [VERSION_WIDTH-1:0]       i_cfg_version,
    input  wire logic [FRAME_WIDTH-1:0]         i_cfg_frame_index,
    input  wire logic                           i_cfg_sample_reset,
    input  wire logic                           i_cfg_sample_last,
    input  wire logic [CLASS_WIDTH-1:0]         i_cfg_sample_label,
    input  wire logic                           i_release,
    output logic                                o_release_ready,

    input  wire logic                           i_commit_valid,
    output logic                                o_commit_ready,
    input  wire logic [19:0]                    i_commit_layer_tag,
    input  wire logic [9:0]                     i_commit_channel,
    input  wire logic [31:0]                    i_commit_spatial_base,
    input  wire logic [5:0]                     i_commit_spatial_count,
    input  wire logic [39:0]                    i_commit_spike,
    input  wire logic [639:0]                   i_commit_q8_8_flat,

    output logic                                o_feature_wave_valid,
    input  wire logic                           i_feature_wave_ready,
    output logic [LANES-1:0]                    o_feature_wave_mask,
    output logic [FEATURE_WIDTH-1:0]            o_feature_wave_base,
    output logic [LANES*16-1:0]                 o_feature_wave_q8_8_flat,
    output logic [VERSION_WIDTH-1:0]            o_feature_wave_version,
    output logic [FRAME_WIDTH-1:0]              o_feature_wave_frame_index,
    output logic                                o_feature_wave_first,
    output logic                                o_feature_wave_last,
    output logic                                o_feature_wave_sample_reset,
    output logic                                o_feature_wave_sample_last,
    output logic [CLASS_WIDTH-1:0]              o_sample_label,

    output logic                                o_capture_complete,
    output logic                                o_active,
    output logic                                o_idle,
    output logic                                o_protocol_error
);
    localparam int FRAME_FEATURES = MAX_CHANNELS * SPATIAL;
    localparam int FRAME_COUNT_WIDTH = $clog2(FRAME_FEATURES + 1);
    localparam int FEATURE_BANK_DEPTH = (FRAME_FEATURES + LANES - 1) / LANES;
    localparam int FEATURE_BANK_ADDR_WIDTH = FEATURE_BANK_DEPTH <= 2 ? 1 :
        $clog2(FEATURE_BANK_DEPTH);

    typedef enum logic [1:0] {IDLE, COLLECT, EMIT, COMPLETE} state_t;
    state_t state_q;

    logic [19:0] layer_tag_q;
    logic [CHANNEL_WIDTH-1:0] channel_base_q;
    logic [CHANNEL_COUNT_WIDTH-1:0] channel_count_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic [FRAME_WIDTH-1:0] frame_index_q;
    logic sample_reset_q;
    logic sample_last_q;
    logic [CLASS_WIDTH-1:0] sample_label_q;
    (* ram_style = "distributed" *)
    logic [2:0] feature_level_bank_q [0:LANES-1]
        [0:FEATURE_BANK_DEPTH-1];
    logic [LANES-1:0] feature_bank_write_en;
    logic [FEATURE_BANK_ADDR_WIDTH-1:0]
        feature_bank_write_addr [0:LANES-1];
    logic [2:0] feature_bank_write_data [0:LANES-1];
    logic [FEATURE_BANK_ADDR_WIDTH-1:0] feature_bank_read_addr;
    logic [2:0] feature_bank_read_data [0:LANES-1];
    logic [FRAME_COUNT_WIDTH-1:0] feature_count_q;
    logic [FRAME_COUNT_WIDTH-1:0] collected_count_q;
    logic [FEATURE_WIDTH-1:0] emit_offset_q;

    logic cfg_legal;
    logic cfg_fire;
    logic commit_fire;
    logic commit_legal;
    logic commit_payload_legal;
    logic feature_fire;
    logic final_wave;
    integer unsigned commit_local_start;
    integer unsigned commit_local_end;
    integer unsigned commit_start_bank;
    integer unsigned commit_bank_offset;

    always_comb begin
        cfg_legal = i_cfg_channel_count != 0 &&
            integer'(i_cfg_channel_base) + integer'(i_cfg_channel_count) <=
                MAX_CHANNELS;
        o_cfg_ready = state_q == IDLE && !o_protocol_error;
        cfg_fire = i_cfg_valid && o_cfg_ready;

        commit_local_start =
            (integer'(i_commit_channel) - integer'(channel_base_q)) *
                SPATIAL + integer'(i_commit_spatial_base);
        commit_local_end = commit_local_start +
            integer'(i_commit_spatial_count);
        commit_legal = i_commit_layer_tag == layer_tag_q &&
            integer'(i_commit_channel) >= integer'(channel_base_q) &&
            integer'(i_commit_channel) < integer'(channel_base_q) +
                integer'(channel_count_q) &&
            i_commit_spatial_count != 0 &&
            integer'(i_commit_spatial_count) <= SPATIAL &&
            integer'(i_commit_spatial_base) +
                integer'(i_commit_spatial_count) <= SPATIAL &&
            commit_local_end <= integer'(feature_count_q);
        commit_payload_legal = 1'b1;
        for (integer lane = 0; lane < 40; lane++)
            if (lane < integer'(i_commit_spatial_count)) begin
                if ($signed(i_commit_q8_8_flat[lane*16 +: 16]) < 0 ||
                    $unsigned(i_commit_q8_8_flat[lane*16 +: 16]) > 256 ||
                    i_commit_q8_8_flat[lane*16 +: 6] != 0 ||
                    ((i_commit_q8_8_flat[lane*16 +: 16] != 0) !=
                     i_commit_spike[lane]))
                    commit_payload_legal = 1'b0;
            end
        o_commit_ready = state_q == COLLECT && !o_protocol_error;
        commit_fire = i_commit_valid && o_commit_ready;

        o_feature_wave_valid = state_q == EMIT && !o_protocol_error;
        o_feature_wave_mask = '0;
        o_feature_wave_base = emit_offset_q;
        o_feature_wave_q8_8_flat = '0;
        feature_bank_read_addr = FEATURE_BANK_ADDR_WIDTH'(
            integer'(emit_offset_q) / LANES);
        for (integer lane = 0; lane < LANES; lane++) begin
            if (integer'(emit_offset_q) + lane <
                integer'(feature_count_q)) begin
                o_feature_wave_mask[lane] = 1'b1;
                o_feature_wave_q8_8_flat[lane*16 +: 16] =
                    {7'd0, feature_bank_read_data[lane], 6'd0};
            end
        end
        final_wave = integer'(emit_offset_q) + LANES >=
            integer'(feature_count_q);
        o_feature_wave_version = version_q;
        o_feature_wave_frame_index = frame_index_q;
        o_feature_wave_first = emit_offset_q == 0;
        o_feature_wave_last = final_wave;
        o_feature_wave_sample_reset = sample_reset_q &&
            o_feature_wave_first;
        o_feature_wave_sample_last = sample_last_q && final_wave;
        o_sample_label = sample_label_q;
        feature_fire = o_feature_wave_valid && i_feature_wave_ready;

        o_release_ready = state_q == COMPLETE;
        o_capture_complete = state_q == COMPLETE;
        o_active = state_q != IDLE;
        o_idle = state_q == IDLE || state_q == COMPLETE ||
            (state_q == COLLECT && collected_count_q == 0);

        feature_bank_write_en = '0;
        for (integer bank = 0; bank < LANES; bank++) begin
            feature_bank_write_addr[bank] = '0;
            feature_bank_write_data[bank] = '0;
        end
        commit_start_bank = commit_local_start % LANES;
        for (integer bank = 0; bank < LANES; bank++) begin
            commit_bank_offset = (bank + LANES - commit_start_bank) % LANES;
            if (commit_fire && commit_legal && commit_payload_legal &&
                    commit_bank_offset < integer'(i_commit_spatial_count)) begin
                feature_bank_write_en[bank] = 1'b1;
                feature_bank_write_addr[bank] = FEATURE_BANK_ADDR_WIDTH'(
                    (commit_local_start + commit_bank_offset) / LANES);
                feature_bank_write_data[bank] =
                    i_commit_q8_8_flat[commit_bank_offset*16 + 6 +: 3];
            end
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= IDLE;
            layer_tag_q <= '0;
            channel_base_q <= '0;
            channel_count_q <= '0;
            version_q <= '0;
            frame_index_q <= '0;
            sample_reset_q <= 1'b0;
            sample_last_q <= 1'b0;
            sample_label_q <= '0;
            feature_count_q <= '0;
            collected_count_q <= '0;
            emit_offset_q <= '0;
            o_protocol_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= IDLE;
            collected_count_q <= '0;
            emit_offset_q <= '0;
        end else begin
            if (i_clear_error && state_q == IDLE)
                o_protocol_error <= 1'b0;

            if (cfg_fire) begin
                if (!cfg_legal) begin
                    o_protocol_error <= 1'b1;
                end else begin
                    state_q <= COLLECT;
                    layer_tag_q <= i_cfg_layer_tag;
                    channel_base_q <= i_cfg_channel_base;
                    channel_count_q <= i_cfg_channel_count;
                    version_q <= i_cfg_version;
                    frame_index_q <= i_cfg_frame_index;
                    sample_reset_q <= i_cfg_sample_reset;
                    sample_last_q <= i_cfg_sample_last;
                    sample_label_q <= i_cfg_sample_label;
                    feature_count_q <= FRAME_COUNT_WIDTH'(
                        integer'(i_cfg_channel_count) * SPATIAL);
                    collected_count_q <= '0;
                    emit_offset_q <= '0;
                end
            end else if (commit_fire) begin
                if (!commit_legal || !commit_payload_legal) begin
                    state_q <= IDLE;
                    o_protocol_error <= 1'b1;
                end else begin
                    collected_count_q <= collected_count_q +
                        FRAME_COUNT_WIDTH'(i_commit_spatial_count);
                    if (collected_count_q +
                            FRAME_COUNT_WIDTH'(i_commit_spatial_count) ==
                            feature_count_q) begin
                        emit_offset_q <= '0;
                        state_q <= EMIT;
                    end
                end
            end else if (feature_fire) begin
                if (final_wave)
                    state_q <= COMPLETE;
                else
                    emit_offset_q <= emit_offset_q + FEATURE_WIDTH'(LANES);
            end else if (i_release && o_release_ready) begin
                state_q <= IDLE;
            end
        end
    end

    generate
        for (genvar bank = 0; bank < LANES; bank++) begin : g_feature_bank
            always_ff @(posedge i_clk) begin
                if (feature_bank_write_en[bank])
                    feature_level_bank_q[bank][
                        feature_bank_write_addr[bank]] <=
                        feature_bank_write_data[bank];
            end
            assign feature_bank_read_data[bank] =
                feature_level_bank_q[bank][feature_bank_read_addr];
        end
    endgenerate

    initial begin
        if (SPATIAL < 1 || SPATIAL > LANES || LANES < 1 ||
            MAX_CHANNELS < 1 || FRAME_FEATURES > 65536)
            $error("SCNN classifier feature-commit adapter parameter mismatch");
    end
endmodule

`default_nettype wire
