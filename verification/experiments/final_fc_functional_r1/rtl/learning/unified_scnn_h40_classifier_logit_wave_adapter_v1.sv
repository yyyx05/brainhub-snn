`timescale 1ns/1ps
`default_nettype none

// Reorders scalar H40 classifier observations into class-major logit waves.
// Optional sample accumulation restores the software LI output leak=1 contract
// without replicating the classifier MACs. Row-max centering preserves softmax
// and argmax while keeping the 16-bit wave interface in range.
module unified_scnn_h40_classifier_logit_wave_adapter_v1 #(
    parameter int MAX_CLASSES = 16,
    parameter int ENGINES = 12,
    parameter bit ACCUMULATE_SAMPLE_LOGITS = 1'b0,
    parameter int VERSION_WIDTH = 16,
    parameter int FRAME_WIDTH = 8,
    parameter int CLASS_WIDTH = MAX_CLASSES <= 2 ? 1 : $clog2(MAX_CLASSES),
    parameter int CLASS_COUNT_WIDTH = $clog2(MAX_CLASSES + 1),
    parameter int MAX_SAMPLES = 1,
    parameter int SAMPLE_WIDTH = MAX_SAMPLES <= 2 ? 1 : $clog2(MAX_SAMPLES)
) (
    input  wire logic                           i_clk,
    input  wire logic                           i_rst_n,
    input  wire logic                           i_abort,
    input  wire logic                           i_clear_error,

    input  wire logic                           i_cfg_valid,
    output logic                                o_cfg_ready,
    input  wire logic [19:0]                    i_cfg_layer_tag,
    input  wire logic [CLASS_COUNT_WIDTH-1:0]   i_cfg_class_count,
    input  wire logic [VERSION_WIDTH-1:0]       i_cfg_version,
    input  wire logic [FRAME_WIDTH-1:0]         i_cfg_frame_index,
    input  wire logic [SAMPLE_WIDTH-1:0]        i_cfg_sample_index,
    input  wire logic                           i_cfg_sample_reset,
    input  wire logic                           i_cfg_sample_last,
    input  wire logic                           i_release,
    output logic                                o_release_ready,

    input  wire logic                           i_scalar_valid,
    output logic                                o_scalar_ready,
    input  wire logic [19:0]                    i_scalar_layer_tag,
    input  wire logic [9:0]                     i_scalar_class,
    input  wire logic [31:0]                    i_scalar_spatial,
    input  wire logic signed [15:0]             i_scalar_logit_q8_8,

    output logic                                o_logit_wave_valid,
    input  wire logic                           i_logit_wave_ready,
    output logic [ENGINES-1:0]                  o_logit_wave_mask,
    output logic [CLASS_WIDTH-1:0]              o_logit_wave_base,
    output logic [CLASS_COUNT_WIDTH-1:0]        o_logit_wave_count,
    output logic [ENGINES*16-1:0]               o_logit_wave_q8_8_flat,
    output logic [VERSION_WIDTH-1:0]            o_logit_wave_version,
    output logic [FRAME_WIDTH-1:0]              o_logit_wave_frame_index,
    output logic                                o_logit_wave_frame_last,
    output logic                                o_logit_wave_sample_last,

    output logic                                o_capture_complete,
    output logic                                o_idle,
    output logic                                o_protocol_error
);
    localparam int SAMPLE_INDEX_WIDTH = MAX_SAMPLES <= 2 ? 1 : $clog2(MAX_SAMPLES);
    typedef enum logic [1:0] {IDLE, COLLECT, EMIT, COMPLETE} state_t;
    state_t state_q;

    logic [19:0] layer_tag_q;
    logic [CLASS_COUNT_WIDTH-1:0] class_count_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic [FRAME_WIDTH-1:0] frame_index_q;
    logic sample_last_q;
    logic signed [15:0] logit_q [0:MAX_CLASSES-1];
    logic signed [31:0] cumulative_logit_q [0:MAX_SAMPLES-1][0:MAX_CLASSES-1];
    logic [SAMPLE_INDEX_WIDTH-1:0] sample_index_q;
    logic [SAMPLE_WIDTH-1:0] cfg_sample_index;
    logic signed [31:0] frame_max_q;
    logic frame_max_valid_q;
    logic signed [31:0] scalar_cumulative_next;
    logic [MAX_CLASSES-1:0] seen_q;
    logic [CLASS_COUNT_WIDTH-1:0] captured_count_q;
    logic [CLASS_WIDTH-1:0] emit_base_q;

    logic cfg_fire;
    logic scalar_fire;
    logic scalar_legal;
    logic scalar_duplicate;
    logic wave_fire;
    logic final_wave;

    assign o_cfg_ready = state_q == IDLE && !o_protocol_error;
    assign cfg_fire = i_cfg_valid && o_cfg_ready;
    // Legacy single-context callers may leave the sample port unconnected.
    assign cfg_sample_index = MAX_SAMPLES == 1 ? '0 : i_cfg_sample_index;
    assign scalar_cumulative_next =
        cumulative_logit_q[sample_index_q][CLASS_WIDTH'(i_scalar_class)] +
        {{16{i_scalar_logit_q8_8[15]}}, i_scalar_logit_q8_8};

    function automatic logic signed [15:0] saturate32_to16(
        input logic signed [31:0] value
    );
        if (value > 32'sd32767)
            return 16'sh7fff;
        if (value < -32'sd32768)
            return 16'sh8000;
        return value[15:0];
    endfunction

    always_comb begin
        scalar_legal = i_scalar_layer_tag == layer_tag_q &&
            i_scalar_spatial == 0 &&
            integer'(i_scalar_class) < integer'(class_count_q);
        scalar_duplicate = scalar_legal &&
            seen_q[CLASS_WIDTH'(i_scalar_class)];
        o_scalar_ready = state_q == COLLECT && !o_protocol_error;
        scalar_fire = i_scalar_valid && o_scalar_ready;

        o_logit_wave_valid = state_q == EMIT && !o_protocol_error;
        o_logit_wave_mask = '0;
        o_logit_wave_base = emit_base_q;
        o_logit_wave_count = class_count_q;
        o_logit_wave_q8_8_flat = '0;
        for (integer engine = 0; engine < ENGINES; engine++) begin
            if (integer'(emit_base_q) + engine <
                integer'(class_count_q)) begin
                o_logit_wave_mask[engine] = 1'b1;
                if (ACCUMULATE_SAMPLE_LOGITS)
                    o_logit_wave_q8_8_flat[engine*16 +: 16] =
                        saturate32_to16(
                            cumulative_logit_q[sample_index_q]
                                [integer'(emit_base_q) + engine] -
                            frame_max_q);
                else
                    o_logit_wave_q8_8_flat[engine*16 +: 16] =
                        logit_q[integer'(emit_base_q) + engine];
            end
        end
        final_wave = integer'(emit_base_q) + ENGINES >=
            integer'(class_count_q);
        o_logit_wave_version = version_q;
        o_logit_wave_frame_index = frame_index_q;
        o_logit_wave_frame_last = final_wave;
        o_logit_wave_sample_last = sample_last_q && final_wave;
        wave_fire = o_logit_wave_valid && i_logit_wave_ready;

        o_release_ready = state_q == COMPLETE;
        o_capture_complete = state_q == COMPLETE;
        o_idle = state_q == IDLE || state_q == COMPLETE ||
            (state_q == COLLECT && captured_count_q == 0);
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= IDLE;
            layer_tag_q <= '0;
            class_count_q <= '0;
            version_q <= '0;
            frame_index_q <= '0;
            sample_last_q <= 1'b0;
            sample_index_q <= '0;
            seen_q <= '0;
            captured_count_q <= '0;
            emit_base_q <= '0;
            frame_max_q <= '0;
            frame_max_valid_q <= 1'b0;
            o_protocol_error <= 1'b0;
            for (integer class_index = 0; class_index < MAX_CLASSES;
                 class_index++) begin
                logit_q[class_index] <= '0;
            end
            for (integer sample_index = 0; sample_index < MAX_SAMPLES;
                 sample_index++)
                for (integer class_index = 0; class_index < MAX_CLASSES;
                     class_index++)
                    cumulative_logit_q[sample_index][class_index] <= '0;
        end else if (i_abort) begin
            state_q <= IDLE;
            seen_q <= '0;
            captured_count_q <= '0;
            emit_base_q <= '0;
            frame_max_valid_q <= 1'b0;
            for (integer sample_index = 0; sample_index < MAX_SAMPLES;
                 sample_index++)
                for (integer class_index = 0; class_index < MAX_CLASSES;
                     class_index++)
                    cumulative_logit_q[sample_index][class_index] <= '0;
        end else begin
            if (i_clear_error && state_q == IDLE)
                o_protocol_error <= 1'b0;

            if (cfg_fire) begin
                if (i_cfg_class_count == 0 ||
                    integer'(i_cfg_class_count) > MAX_CLASSES ||
                    (MAX_SAMPLES > 1 &&
                     integer'(cfg_sample_index) >= MAX_SAMPLES)) begin
                    o_protocol_error <= 1'b1;
                end else begin
                    state_q <= COLLECT;
                    layer_tag_q <= i_cfg_layer_tag;
                    class_count_q <= i_cfg_class_count;
                    version_q <= i_cfg_version;
                    frame_index_q <= i_cfg_frame_index;
                    sample_last_q <= i_cfg_sample_last;
                    sample_index_q <= SAMPLE_INDEX_WIDTH'(cfg_sample_index);
                    seen_q <= '0;
                    captured_count_q <= '0;
                    emit_base_q <= '0;
                    frame_max_q <= '0;
                    frame_max_valid_q <= 1'b0;
                    if (ACCUMULATE_SAMPLE_LOGITS && i_cfg_sample_reset)
                        for (integer class_index = 0;
                             class_index < MAX_CLASSES; class_index++)
                            cumulative_logit_q[SAMPLE_INDEX_WIDTH'(cfg_sample_index)]
                                [class_index] <= '0;
                end
            end else if (scalar_fire) begin
                if (!scalar_legal || scalar_duplicate) begin
                    state_q <= IDLE;
                    o_protocol_error <= 1'b1;
                end else begin
                    logit_q[CLASS_WIDTH'(i_scalar_class)] <=
                        i_scalar_logit_q8_8;
                    if (ACCUMULATE_SAMPLE_LOGITS) begin
                        cumulative_logit_q[sample_index_q]
                            [CLASS_WIDTH'(i_scalar_class)] <=
                            scalar_cumulative_next;
                        if (!frame_max_valid_q ||
                            scalar_cumulative_next > frame_max_q) begin
                            frame_max_q <= scalar_cumulative_next;
                            frame_max_valid_q <= 1'b1;
                        end
                    end
                    seen_q[CLASS_WIDTH'(i_scalar_class)] <= 1'b1;
                    captured_count_q <= captured_count_q + 1'b1;
                    if (captured_count_q + 1'b1 == class_count_q) begin
                        emit_base_q <= '0;
                        state_q <= EMIT;
                    end
                end
            end else if (wave_fire) begin
                if (final_wave)
                    state_q <= COMPLETE;
                else
                    emit_base_q <= emit_base_q + CLASS_WIDTH'(ENGINES);
            end else if (i_release && o_release_ready) begin
                state_q <= IDLE;
            end
        end
    end

    initial begin
        // Extra physical engines are legal; the wave mask disables lanes
        // beyond the configured logical class count.
        if (MAX_CLASSES < 1 || ENGINES < 1 || MAX_SAMPLES < 1 ||
            SAMPLE_WIDTH < 1 || SAMPLE_WIDTH < $clog2(MAX_SAMPLES))
            $error("SCNN H40 classifier logit adapter parameter mismatch");
    end
endmodule

`default_nettype wire
