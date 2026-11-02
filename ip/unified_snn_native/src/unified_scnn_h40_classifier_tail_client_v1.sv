`timescale 1ns/1ps
`default_nettype none

// Classifier tail for the shared H40/P4 substrate. It captures the committed
// post-pooling feature frame, replays it through the same array as a Q8.8 linear
// classifier, and converts unordered physical observations to TP logit waves.
// No MAC, LIF, state or weight array is instantiated here.
module unified_scnn_h40_classifier_tail_client_v1 #(
    parameter int MAX_FEATURE_CHANNELS = 512,
    parameter int FEATURE_SPATIAL = 4,
    parameter int MAX_CLASSES = 16,
    parameter int FEATURE_LANES = 8,
    parameter int LOGIT_ENGINES = 12,
    parameter bit ACCUMULATE_SAMPLE_LOGITS = 1'b0,
    parameter int PHYSICAL_TILE_COUNT = 16,
    parameter int WEIGHT_ADDR_WIDTH = 10,
    parameter int WORK_ADDR_WIDTH = 9,
    parameter int VERSION_WIDTH = 16,
    parameter int FRAME_WIDTH = 8,
    parameter int CHANNEL_WIDTH = MAX_FEATURE_CHANNELS <= 2 ? 1 :
        $clog2(MAX_FEATURE_CHANNELS),
    parameter int CHANNEL_COUNT_WIDTH = $clog2(MAX_FEATURE_CHANNELS + 1),
    parameter int FEATURE_WIDTH = MAX_FEATURE_CHANNELS * FEATURE_SPATIAL <= 2 ?
        1 : $clog2(MAX_FEATURE_CHANNELS * FEATURE_SPATIAL),
    parameter int FEATURE_COUNT_WIDTH =
        $clog2(MAX_FEATURE_CHANNELS * FEATURE_SPATIAL + 1),
    parameter int CLASS_WIDTH = MAX_CLASSES <= 2 ? 1 : $clog2(MAX_CLASSES),
    parameter int CLASS_COUNT_WIDTH = $clog2(MAX_CLASSES + 1),
    parameter int LOGIT_CONTEXTS = 1,
    parameter int SAMPLE_WIDTH = LOGIT_CONTEXTS <= 2 ? 1 : $clog2(LOGIT_CONTEXTS)
) (
    input  wire logic                           i_clk,
    input  wire logic                           i_rst_n,
    input  wire logic                           i_abort,
    input  wire logic                           i_clear_error,

    input  wire logic                           i_start_valid,
    output logic                                o_start_ready,
    input  wire logic [19:0]                    i_feature_layer_tag,
    input  wire logic [CHANNEL_WIDTH-1:0]       i_feature_channel_base,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0] i_feature_channel_count,
    input  wire logic [CLASS_COUNT_WIDTH-1:0]   i_class_count,
    input  wire logic                           i_classifier_active_pair,
    input  wire logic [WORK_ADDR_WIDTH-1:0]     i_classifier_work_base_addr,
    input  wire logic [15:0]                    i_classifier_trace_decay,
    input  wire logic [19:0]                    i_classifier_layer_tag,
    input  wire logic [VERSION_WIDTH-1:0]       i_version,
    input  wire logic [FRAME_WIDTH-1:0]         i_frame_index,
    input  wire logic [SAMPLE_WIDTH-1:0]        i_sample_index,
    input  wire logic                           i_sample_reset,
    input  wire logic                           i_sample_last,
    input  wire logic [CLASS_WIDTH-1:0]         i_sample_label,
    input  wire logic                           i_learning_enable,

    input  wire logic                           i_conv_done,
    input  wire logic                           i_conv_error,
    input  wire logic                           i_conv_feature_commit_valid,
    output logic                                o_conv_feature_commit_ready,
    input  wire logic [19:0]                    i_conv_feature_commit_layer_tag,
    input  wire logic [9:0]                     i_conv_feature_commit_channel,
    input  wire logic [31:0]                    i_conv_feature_commit_spatial_base,
    input  wire logic [5:0]                     i_conv_feature_commit_spatial_count,
    input  wire logic [39:0]                    i_conv_feature_commit_spike,
    input  wire logic [639:0]                   i_conv_feature_commit_q8_8_flat,

    output logic                                o_feature_trace_wave_valid,
    input  wire logic                           i_feature_trace_wave_ready,
    output logic [FEATURE_LANES-1:0]            o_feature_trace_wave_mask,
    output logic [FEATURE_WIDTH-1:0]            o_feature_trace_wave_base,
    output logic [FEATURE_LANES*16-1:0]         o_feature_trace_wave_q8_8_flat,
    output logic [VERSION_WIDTH-1:0]            o_feature_trace_wave_version,
    output logic [FRAME_WIDTH-1:0]              o_feature_trace_wave_frame_index,
    output logic                                o_feature_trace_wave_first,
    output logic                                o_feature_trace_wave_last,
    output logic                                o_feature_trace_wave_sample_reset,
    output logic                                o_feature_trace_wave_sample_last,
    output logic [CLASS_WIDTH-1:0]              o_feature_sample_label,

    output logic                                o_context_load_valid,
    input  wire logic                           i_context_load_ready,
    output logic                                o_context_load_weights,
    output logic [9:0]                          o_context_output_channel_base,
    output logic [9:0]                          o_context_output_channel_count,
    output logic [31:0]                         o_context_spatial_base,
    output logic [5:0]                          o_context_spatial_count,
    output logic                                o_context_active_pair,
    output logic [WORK_ADDR_WIDTH-1:0]          o_context_work_base_addr,
    output logic                                o_context_release_valid,
    input  wire logic                           i_context_release_ready,
    output logic [9:0]                          o_release_output_channel_base,
    output logic [9:0]                          o_release_output_channel_count,
    output logic [31:0]                         o_release_spatial_base,
    output logic [5:0]                          o_release_spatial_count,
    output logic                                o_release_active_pair,
    output logic [WORK_ADDR_WIDTH-1:0]          o_release_work_base_addr,

    output logic                                o_array_command_valid,
    input  wire logic                           i_array_command_ready,
    output logic [15:0]                         o_array_command_neuron_count,
    output logic                                o_array_command_first,
    output logic                                o_array_command_last,
    output logic [3:0]                          o_array_command_source_valid,
    output logic [5:0]                          o_array_command_spatial_count,
    output logic [9:0]                          o_array_command_output_channel_base,
    output logic [39:0]                         o_array_command_feature_mask0,
    output logic [39:0]                         o_array_command_feature_mask1,
    output logic [39:0]                         o_array_command_feature_mask2,
    output logic [39:0]                         o_array_command_feature_mask3,
    output logic                                o_array_command_active_pair,
    output logic [3:0]                          o_array_command_weight_subword0,
    output logic [3:0]                          o_array_command_weight_subword1,
    output logic [3:0]                          o_array_command_weight_subword2,
    output logic [3:0]                          o_array_command_weight_subword3,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_array_command_weight_addr0,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_array_command_weight_addr1,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_array_command_weight_addr2,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_array_command_weight_addr3,
    output logic [WORK_ADDR_WIDTH-1:0]          o_array_command_work_base_addr,
    output logic [15:0]                         o_array_command_leak,
    output logic [31:0]                         o_array_command_threshold,
    output logic                                o_array_command_reset_to_zero,
    output logic [15:0]                         o_array_command_trace_decay,
    output logic [51:0]                         o_array_command_transaction_tag,
    input  wire logic                           i_array_command_done,
    input  wire logic                           i_array_idle,
    input  wire logic                           i_array_error,

    input  wire logic [3:0]                     i_array_observation_valid,
    output logic [3:0]                          o_array_observation_ready,
    input  wire logic [15:0]                    i_array_observation_tile_flat,
    input  wire logic [7:0]                     i_array_observation_mode_flat,
    input  wire logic [39:0]                    i_array_observation_lane_mask_flat,
    input  wire logic [39:0]                    i_array_observation_range_mask_flat,
    input  wire logic [39:0]                    i_array_observation_spike_flat,
    input  wire logic [1279:0]                  i_array_observation_pre_state_flat,
    input  wire logic [1279:0]                  i_array_observation_trace_next_flat,
    input  wire logic [255:0]                   i_array_observation_meta_flat,

    output logic                                o_logit_wave_valid,
    input  wire logic                           i_logit_wave_ready,
    output logic [LOGIT_ENGINES-1:0]            o_logit_wave_mask,
    output logic [CLASS_WIDTH-1:0]              o_logit_wave_base,
    output logic [CLASS_COUNT_WIDTH-1:0]        o_logit_wave_count,
    output logic [LOGIT_ENGINES*16-1:0]         o_logit_wave_q8_8_flat,
    output logic [VERSION_WIDTH-1:0]            o_logit_wave_version,
    output logic [FRAME_WIDTH-1:0]              o_logit_wave_frame_index,
    output logic                                o_logit_wave_frame_last,
    output logic                                o_logit_wave_sample_last,

    output logic                                o_done,
    output logic                                o_busy,
    output logic                                o_idle,
    output logic                                o_sticky_error
);
    typedef enum logic [2:0] {
        IDLE, CAPTURE, CONTEXT_LOAD, SCHEDULER_START, RUN, RELEASE
    } state_t;
    state_t state_q;

    logic [FEATURE_COUNT_WIDTH-1:0] feature_count_q;
    logic [CLASS_COUNT_WIDTH-1:0] class_count_q;
    logic active_pair_q;
    logic [WORK_ADDR_WIDTH-1:0] work_base_addr_q;
    logic [15:0] trace_decay_q;
    logic [19:0] classifier_layer_tag_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic [FRAME_WIDTH-1:0] frame_index_q;
    logic sample_last_q;
    logic learning_enable_q;
    logic conv_done_seen_q;
    logic scheduler_done_seen_q;
    logic local_error_q;

    logic start_fire;
    logic config_legal;
    logic feature_cfg_ready;
    logic feature_wave_valid;
    logic feature_wave_ready;
    logic [FEATURE_LANES-1:0] feature_wave_mask;
    logic [FEATURE_WIDTH-1:0] feature_wave_base;
    logic [FEATURE_LANES*16-1:0] feature_wave_flat;
    logic [VERSION_WIDTH-1:0] feature_wave_version;
    logic [FRAME_WIDTH-1:0] feature_wave_frame;
    logic feature_wave_first, feature_wave_last;
    logic feature_wave_sample_reset, feature_wave_sample_last;
    logic [CLASS_WIDTH-1:0] feature_sample_label;
    logic feature_capture_complete, feature_release_ready;
    logic feature_commit_ready;
    logic feature_error;

    logic scheduler_start_valid, scheduler_start_ready;
    logic scheduler_feature_ready;
    logic scheduler_done, scheduler_busy, scheduler_error;

    logic serializer_scalar_valid, serializer_scalar_ready;
    logic [19:0] serializer_scalar_layer_tag;
    logic [9:0] serializer_scalar_channel;
    logic [31:0] serializer_scalar_spatial;
    logic serializer_scalar_spike;
    logic signed [15:0] serializer_scalar_pre_state;
    logic signed [15:0] serializer_scalar_trace_next;
    logic serializer_busy, serializer_error;
    logic [3:0] serializer_observation_ready;

    logic logit_cfg_ready, logit_capture_complete, logit_release_ready;
    logic logit_scalar_ready;
    logic logit_error;
    logic release_fire;
    logic unused_feature_active;
    logic unused_feature_idle;
    logic unused_serializer_scalar_last;
    logic unused_logit_idle;

    assign config_legal = i_feature_channel_count != 0 &&
        integer'(i_feature_channel_base) +
            integer'(i_feature_channel_count) <= MAX_FEATURE_CHANNELS &&
        integer'(i_feature_channel_count) * FEATURE_SPATIAL <=
            MAX_FEATURE_CHANNELS * FEATURE_SPATIAL &&
        i_class_count != 0 && integer'(i_class_count) <= MAX_CLASSES &&
        integer'(i_sample_label) < integer'(i_class_count) &&
        (LOGIT_CONTEXTS == 1 || integer'(i_sample_index) < LOGIT_CONTEXTS);
    assign o_start_ready = state_q == IDLE && feature_cfg_ready &&
        logit_cfg_ready && !o_sticky_error;
    assign start_fire = i_start_valid && o_start_ready;

    assign o_conv_feature_commit_ready = state_q == CAPTURE &&
        feature_commit_ready;

    assign o_feature_trace_wave_valid = feature_wave_valid &&
        learning_enable_q && scheduler_feature_ready;
    assign feature_wave_ready = scheduler_feature_ready &&
        (!learning_enable_q || i_feature_trace_wave_ready);
    assign o_feature_trace_wave_mask = feature_wave_mask;
    assign o_feature_trace_wave_base = feature_wave_base;
    assign o_feature_trace_wave_q8_8_flat = feature_wave_flat;
    assign o_feature_trace_wave_version = feature_wave_version;
    assign o_feature_trace_wave_frame_index = feature_wave_frame;
    assign o_feature_trace_wave_first = feature_wave_first;
    assign o_feature_trace_wave_last = feature_wave_last;
    assign o_feature_trace_wave_sample_reset = feature_wave_sample_reset;
    assign o_feature_trace_wave_sample_last = feature_wave_sample_last;
    assign o_feature_sample_label = feature_sample_label;

    assign o_context_load_valid = state_q == CONTEXT_LOAD;
    assign o_context_load_weights = 1'b1;
    assign o_context_output_channel_base = 10'd0;
    assign o_context_output_channel_count = 10'(class_count_q);
    assign o_context_spatial_base = 32'd0;
    assign o_context_spatial_count = 6'd1;
    assign o_context_active_pair = active_pair_q;
    assign o_context_work_base_addr = work_base_addr_q;

    assign release_fire = state_q == RELEASE && feature_release_ready &&
        logit_release_ready && i_context_release_ready;
    assign o_context_release_valid = state_q == RELEASE &&
        feature_release_ready && logit_release_ready;
    assign o_release_output_channel_base = 10'd0;
    assign o_release_output_channel_count = 10'(class_count_q);
    assign o_release_spatial_base = 32'd0;
    assign o_release_spatial_count = 6'd1;
    assign o_release_active_pair = active_pair_q;
    assign o_release_work_base_addr = work_base_addr_q;

    assign scheduler_start_valid = state_q == SCHEDULER_START;
    assign o_array_observation_ready = state_q == RUN ?
        serializer_observation_ready : 4'b0000;
    assign serializer_scalar_ready = state_q == RUN ?
        logit_scalar_ready : 1'b0;

    assign o_busy = state_q != IDLE || scheduler_busy || serializer_busy;
    assign o_idle = state_q == IDLE && !scheduler_busy && !serializer_busy;
    assign o_sticky_error = local_error_q || feature_error ||
        scheduler_error || serializer_error || logit_error || i_array_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= IDLE;
            feature_count_q <= '0;
            class_count_q <= '0;
            active_pair_q <= 1'b0;
            work_base_addr_q <= '0;
            trace_decay_q <= '0;
            classifier_layer_tag_q <= '0;
            version_q <= '0;
            frame_index_q <= '0;
            sample_last_q <= 1'b0;
            learning_enable_q <= 1'b0;
            conv_done_seen_q <= 1'b0;
            scheduler_done_seen_q <= 1'b0;
            local_error_q <= 1'b0;
            o_done <= 1'b0;
        end else if (i_abort) begin
            state_q <= IDLE;
            conv_done_seen_q <= 1'b0;
            scheduler_done_seen_q <= 1'b0;
            local_error_q <= 1'b0;
            o_done <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_clear_error && state_q == IDLE)
                local_error_q <= 1'b0;

            if (state_q != IDLE &&
                (i_conv_error || i_array_error || feature_error ||
                 scheduler_error || serializer_error || logit_error)) begin
                state_q <= IDLE;
                local_error_q <= 1'b1;
            end else begin
                case (state_q)
                    IDLE: begin
                        conv_done_seen_q <= 1'b0;
                        scheduler_done_seen_q <= 1'b0;
                        if (start_fire) begin
                            if (!config_legal) begin
                                local_error_q <= 1'b1;
                            end else begin
                                feature_count_q <= FEATURE_COUNT_WIDTH'(
                                    integer'(i_feature_channel_count) *
                                        FEATURE_SPATIAL);
                                class_count_q <= i_class_count;
                                active_pair_q <= i_classifier_active_pair;
                                work_base_addr_q <=
                                    i_classifier_work_base_addr;
                                trace_decay_q <= i_classifier_trace_decay;
                                classifier_layer_tag_q <=
                                    i_classifier_layer_tag;
                                version_q <= i_version;
                                frame_index_q <= i_frame_index;
                                sample_last_q <= i_sample_last;
                                learning_enable_q <= i_learning_enable;
                                state_q <= CAPTURE;
                            end
                        end
                    end
                    CAPTURE: begin
                        if (i_conv_done)
                            conv_done_seen_q <= 1'b1;
                        if ((conv_done_seen_q || i_conv_done) &&
                            feature_wave_valid)
                            state_q <= CONTEXT_LOAD;
                    end
                    CONTEXT_LOAD: begin
                        if (i_context_load_ready)
                            state_q <= SCHEDULER_START;
                    end
                    SCHEDULER_START: begin
                        if (scheduler_start_ready)
                            state_q <= RUN;
                    end
                    RUN: begin
                        if (scheduler_done)
                            scheduler_done_seen_q <= 1'b1;
                        if ((scheduler_done_seen_q || scheduler_done) &&
                            feature_capture_complete &&
                            logit_capture_complete && i_array_idle)
                            state_q <= RELEASE;
                    end
                    RELEASE: begin
                        if (release_fire) begin
                            state_q <= IDLE;
                            o_done <= 1'b1;
                        end
                    end
                    default: begin
                        state_q <= IDLE;
                        local_error_q <= 1'b1;
                    end
                endcase
            end
        end
    end

    unified_scnn_h40_classifier_feature_commit_adapter_v1 #(
        .MAX_CHANNELS(MAX_FEATURE_CHANNELS), .SPATIAL(FEATURE_SPATIAL),
        .LANES(FEATURE_LANES),
        .VERSION_WIDTH(VERSION_WIDTH), .FRAME_WIDTH(FRAME_WIDTH),
        .CHANNEL_WIDTH(CHANNEL_WIDTH),
        .CHANNEL_COUNT_WIDTH(CHANNEL_COUNT_WIDTH),
        .FEATURE_WIDTH(FEATURE_WIDTH), .CLASS_WIDTH(CLASS_WIDTH)
    ) u_feature_adapter (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_cfg_valid(start_fire && config_legal),
        .o_cfg_ready(feature_cfg_ready),
        .i_cfg_layer_tag(i_feature_layer_tag),
        .i_cfg_channel_base(i_feature_channel_base),
        .i_cfg_channel_count(i_feature_channel_count),
        .i_cfg_version(i_version), .i_cfg_frame_index(i_frame_index),
        .i_cfg_sample_reset(i_sample_reset),
        .i_cfg_sample_last(i_sample_last),
        .i_cfg_sample_label(i_sample_label),
        .i_release(release_fire), .o_release_ready(feature_release_ready),
        .i_commit_valid(i_conv_feature_commit_valid && state_q == CAPTURE),
        .o_commit_ready(feature_commit_ready),
        .i_commit_layer_tag(i_conv_feature_commit_layer_tag),
        .i_commit_channel(i_conv_feature_commit_channel),
        .i_commit_spatial_base(i_conv_feature_commit_spatial_base),
        .i_commit_spatial_count(i_conv_feature_commit_spatial_count),
        .i_commit_spike(i_conv_feature_commit_spike),
        .i_commit_q8_8_flat(i_conv_feature_commit_q8_8_flat),
        .o_feature_wave_valid(feature_wave_valid),
        .i_feature_wave_ready(feature_wave_ready),
        .o_feature_wave_mask(feature_wave_mask),
        .o_feature_wave_base(feature_wave_base),
        .o_feature_wave_q8_8_flat(feature_wave_flat),
        .o_feature_wave_version(feature_wave_version),
        .o_feature_wave_frame_index(feature_wave_frame),
        .o_feature_wave_first(feature_wave_first),
        .o_feature_wave_last(feature_wave_last),
        .o_feature_wave_sample_reset(feature_wave_sample_reset),
        .o_feature_wave_sample_last(feature_wave_sample_last),
        .o_sample_label(feature_sample_label),
        .o_capture_complete(feature_capture_complete),
        .o_active(unused_feature_active),
        .o_idle(unused_feature_idle), .o_protocol_error(feature_error)
    );

    unified_scnn_h40_classifier_p4_scheduler_v1 #(
        .MAX_FEATURES(MAX_FEATURE_CHANNELS * FEATURE_SPATIAL),
        .MAX_CLASSES(MAX_CLASSES), .LANES(FEATURE_LANES),
        .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH),
        .FEATURE_WIDTH(FEATURE_WIDTH),
        .FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
        .CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH)
    ) u_scheduler (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_start_valid(scheduler_start_valid),
        .o_start_ready(scheduler_start_ready),
        .i_feature_count(feature_count_q), .i_class_count(class_count_q),
        .i_active_pair(active_pair_q), .i_work_base_addr(work_base_addr_q),
        .i_trace_decay(trace_decay_q), .i_layer_tag(classifier_layer_tag_q),
        .i_feature_wave_valid(feature_wave_valid &&
            (!learning_enable_q || i_feature_trace_wave_ready)),
        .o_feature_wave_ready(scheduler_feature_ready),
        .i_feature_wave_mask(feature_wave_mask),
        .i_feature_wave_base(feature_wave_base),
        .i_feature_wave_q8_8_flat(feature_wave_flat),
        .i_feature_wave_first(feature_wave_first),
        .i_feature_wave_last(feature_wave_last),
        .o_command_valid(o_array_command_valid),
        .i_command_ready(i_array_command_ready),
        .o_command_neuron_count(o_array_command_neuron_count),
        .o_command_first(o_array_command_first),
        .o_command_last(o_array_command_last),
        .o_command_source_valid(o_array_command_source_valid),
        .o_command_spatial_count(o_array_command_spatial_count),
        .o_command_output_channel_base(
            o_array_command_output_channel_base),
        .o_command_feature_mask0(o_array_command_feature_mask0),
        .o_command_feature_mask1(o_array_command_feature_mask1),
        .o_command_feature_mask2(o_array_command_feature_mask2),
        .o_command_feature_mask3(o_array_command_feature_mask3),
        .o_command_active_pair(o_array_command_active_pair),
        .o_command_weight_subword0(o_array_command_weight_subword0),
        .o_command_weight_subword1(o_array_command_weight_subword1),
        .o_command_weight_subword2(o_array_command_weight_subword2),
        .o_command_weight_subword3(o_array_command_weight_subword3),
        .o_command_weight_addr0(o_array_command_weight_addr0),
        .o_command_weight_addr1(o_array_command_weight_addr1),
        .o_command_weight_addr2(o_array_command_weight_addr2),
        .o_command_weight_addr3(o_array_command_weight_addr3),
        .o_command_work_base_addr(o_array_command_work_base_addr),
        .o_command_leak(o_array_command_leak),
        .o_command_threshold(o_array_command_threshold),
        .o_command_reset_to_zero(o_array_command_reset_to_zero),
        .o_command_trace_decay(o_array_command_trace_decay),
        .o_command_transaction_tag(o_array_command_transaction_tag),
        .i_command_done(i_array_command_done),
        .i_child_error(i_array_error), .o_done(scheduler_done),
        .o_busy(scheduler_busy), .o_sticky_error(scheduler_error)
    );

    unified_scnn_h40_tp_observation_serializer_v1 #(
        .PHYSICAL_TILE_COUNT(PHYSICAL_TILE_COUNT)
    ) u_serializer (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_observation_valid(state_q == RUN ?
            i_array_observation_valid : 4'b0000),
        .o_observation_ready(serializer_observation_ready),
        .i_observation_tile_flat(i_array_observation_tile_flat),
        .i_observation_mode_flat(i_array_observation_mode_flat),
        .i_observation_lane_mask_flat(i_array_observation_lane_mask_flat),
        .i_observation_range_mask_flat(i_array_observation_range_mask_flat),
        .i_observation_spike_flat(i_array_observation_spike_flat),
        .i_observation_pre_state_flat(i_array_observation_pre_state_flat),
        .i_observation_trace_next_flat(i_array_observation_trace_next_flat),
        .i_observation_meta_flat(i_array_observation_meta_flat),
        .o_scalar_valid(serializer_scalar_valid),
        .i_scalar_ready(serializer_scalar_ready),
        .o_scalar_layer_tag(serializer_scalar_layer_tag),
        .o_scalar_channel(serializer_scalar_channel),
        .o_scalar_spatial(serializer_scalar_spatial),
        .o_scalar_spike(serializer_scalar_spike),
        .o_scalar_pre_state_q8_8(serializer_scalar_pre_state),
        .o_scalar_trace_next_q8_8(serializer_scalar_trace_next),
        .o_scalar_last(unused_serializer_scalar_last),
        .o_busy(serializer_busy),
        .o_sticky_error(serializer_error)
    );

    unified_scnn_h40_classifier_logit_wave_adapter_v1 #(
        .MAX_CLASSES(MAX_CLASSES), .ENGINES(LOGIT_ENGINES),
        .MAX_SAMPLES(LOGIT_CONTEXTS), .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .ACCUMULATE_SAMPLE_LOGITS(ACCUMULATE_SAMPLE_LOGITS),
        .VERSION_WIDTH(VERSION_WIDTH), .FRAME_WIDTH(FRAME_WIDTH),
        .CLASS_WIDTH(CLASS_WIDTH),
        .CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH)
    ) u_logit_adapter (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_cfg_valid(start_fire && config_legal),
        .o_cfg_ready(logit_cfg_ready),
        .i_cfg_layer_tag(i_classifier_layer_tag),
        .i_cfg_class_count(i_class_count), .i_cfg_version(i_version),
        .i_cfg_frame_index(i_frame_index),
        .i_cfg_sample_index(i_sample_index),
        .i_cfg_sample_reset(i_sample_reset),
        .i_cfg_sample_last(i_sample_last),
        .i_release(release_fire), .o_release_ready(logit_release_ready),
        .i_scalar_valid(serializer_scalar_valid),
        .o_scalar_ready(logit_scalar_ready),
        .i_scalar_layer_tag(serializer_scalar_layer_tag),
        .i_scalar_class(serializer_scalar_channel),
        .i_scalar_spatial(serializer_scalar_spatial),
        .i_scalar_logit_q8_8(serializer_scalar_pre_state),
        .o_logit_wave_valid(o_logit_wave_valid),
        .i_logit_wave_ready(i_logit_wave_ready),
        .o_logit_wave_mask(o_logit_wave_mask),
        .o_logit_wave_base(o_logit_wave_base),
        .o_logit_wave_count(o_logit_wave_count),
        .o_logit_wave_q8_8_flat(o_logit_wave_q8_8_flat),
        .o_logit_wave_version(o_logit_wave_version),
        .o_logit_wave_frame_index(o_logit_wave_frame_index),
        .o_logit_wave_frame_last(o_logit_wave_frame_last),
        .o_logit_wave_sample_last(o_logit_wave_sample_last),
        .o_capture_complete(logit_capture_complete),
        .o_idle(unused_logit_idle),
        .o_protocol_error(logit_error)
    );

    initial begin
        if (FEATURE_LANES != 8 || LOGIT_ENGINES < 1 || MAX_CLASSES < 1 ||
            MAX_FEATURE_CHANNELS * FEATURE_SPATIAL > 13872)
            $error("SCNN H40 classifier tail parameter mismatch");
    end
endmodule

`default_nettype wire
