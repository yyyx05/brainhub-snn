`timescale 1ns/1ps
`default_nettype none

// Converts target-pass pooled feature waves into scalar temporal traces for
// the selected Conv learner. The student path reuses the classifier replay's
// trace engine; this block therefore owns target state only.
module unified_scnn_h40_target_feature_wave_trace_v1 #(
    parameter int MAX_BATCH = 10,
    parameter int MAX_FEATURES = 2048,
    parameter int LANES = 8,
    parameter int VERSION_WIDTH = 16,
    parameter int FRAME_WIDTH = 8,
    parameter int SAMPLE_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int TIMESTEP_WIDTH = 5,
    parameter int FEATURE_WIDTH = MAX_FEATURES <= 2 ? 1 :
        $clog2(MAX_FEATURES),
    parameter int FEATURE_COUNT_WIDTH = $clog2(MAX_FEATURES + 1),
    parameter int LANE_WIDTH = LANES <= 2 ? 1 : $clog2(LANES),
    parameter int TRACE_COUNT = MAX_BATCH * MAX_FEATURES,
    parameter int TRACE_ADDR_WIDTH = TRACE_COUNT <= 2 ? 1 :
        $clog2(TRACE_COUNT)
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_abort,
    input  wire logic                         i_clear_error,

    input  wire logic                         i_cfg_valid,
    output logic                              o_cfg_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]       i_cfg_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]     i_cfg_timestep,
    input  wire logic [VERSION_WIDTH-1:0]      i_cfg_version,
    input  wire logic [FRAME_WIDTH-1:0]        i_cfg_frame_index,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0] i_cfg_feature_count,
    input  wire logic signed [15:0]            i_cfg_trace_leak_q8_8,
    input  wire logic                         i_cfg_trace_state_reset,
    input  wire logic                         i_cfg_sample_last,

    input  wire logic                         i_wave_valid,
    output logic                              o_wave_ready,
    input  wire logic [LANES-1:0]             i_wave_mask,
    input  wire logic [FEATURE_WIDTH-1:0]      i_wave_base,
    input  wire logic [LANES*16-1:0]          i_wave_q8_8_flat,
    input  wire logic [VERSION_WIDTH-1:0]      i_wave_version,
    input  wire logic [FRAME_WIDTH-1:0]        i_wave_frame_index,
    input  wire logic                         i_wave_first,
    input  wire logic                         i_wave_last,
    input  wire logic                         i_wave_sample_reset,
    input  wire logic                         i_wave_sample_last,

    output logic                              o_trace_valid,
    input  wire logic                         i_trace_ready,
    output logic [SAMPLE_WIDTH-1:0]            o_trace_sample,
    output logic [TIMESTEP_WIDTH-1:0]          o_trace_timestep,
    output logic [VERSION_WIDTH-1:0]           o_trace_version,
    output logic [8:0]                        o_trace_channel,
    output logic [1:0]                        o_trace_spatial,
    output logic signed [15:0]                o_trace_q8_8,
    output logic                              o_trace_last,
    output logic                              o_done,
    output logic                              o_busy,
    output logic                              o_sticky_error
);
    typedef enum logic [1:0] {
        TRACE_IDLE, TRACE_WAVE, TRACE_ISSUE, TRACE_RESULT
    } trace_state_t;

    trace_state_t state_q;
    logic [SAMPLE_WIDTH-1:0] sample_q;
    logic [TIMESTEP_WIDTH-1:0] timestep_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic [FRAME_WIDTH-1:0] frame_q;
    logic [FEATURE_COUNT_WIDTH-1:0] feature_count_q;
    logic signed [15:0] trace_leak_q;
    logic sample_reset_q;
    logic sample_last_q;
    logic [FEATURE_WIDTH-1:0] wave_base_q;
    logic [LANES*16-1:0] wave_payload_q;
    logic [LANE_WIDTH-1:0] lane_q;

    logic [LANES-1:0] expected_mask;
    logic wave_legal;
    logic cfg_legal;
    logic cfg_fire;
    logic wave_fire;
    logic trace_input_valid;
    logic trace_input_ready;
    logic trace_result_valid;
    logic [TRACE_ADDR_WIDTH-1:0] trace_result_addr;
    logic signed [15:0] trace_result;
    logic trace_idle;
    logic trace_reset_error;
    logic trace_address_error;
    logic [FEATURE_WIDTH:0] current_feature;
    logic [TRACE_ADDR_WIDTH-1:0] trace_input_addr;
    logic trace_result_fire;

    assign current_feature = {1'b0, wave_base_q} + lane_q;
    assign trace_input_addr = TRACE_ADDR_WIDTH'(
        int'(sample_q) * MAX_FEATURES + int'(current_feature));

    always_comb begin
        expected_mask = '0;
        for (int lane = 0; lane < LANES; lane++)
            expected_mask[lane] =
                int'(i_wave_base) + lane < int'(feature_count_q);
    end

    assign cfg_legal = int'(i_cfg_sample) < MAX_BATCH &&
        i_cfg_feature_count != 0 &&
        int'(i_cfg_feature_count) <= MAX_FEATURES && i_cfg_version != 0;
    assign wave_legal = i_wave_mask == expected_mask &&
        i_wave_base == wave_base_q && i_wave_version == version_q &&
        i_wave_frame_index == frame_q &&
        i_wave_first == (wave_base_q == 0) &&
        i_wave_last ==
            (int'(wave_base_q) + LANES >= int'(feature_count_q)) &&
        i_wave_sample_reset == (sample_reset_q && wave_base_q == 0) &&
        i_wave_sample_last == (sample_last_q &&
            (int'(wave_base_q) + LANES >= int'(feature_count_q)));

    assign o_cfg_ready = state_q == TRACE_IDLE && trace_idle &&
        !o_sticky_error;
    assign cfg_fire = i_cfg_valid && o_cfg_ready;
    assign o_wave_ready = state_q == TRACE_WAVE && !o_sticky_error;
    assign wave_fire = i_wave_valid && o_wave_ready;
    assign trace_input_valid = state_q == TRACE_ISSUE &&
        !o_sticky_error;
    assign o_trace_valid = state_q == TRACE_RESULT && trace_result_valid &&
        trace_result_addr == trace_input_addr && !o_sticky_error;
    assign trace_result_fire = o_trace_valid && i_trace_ready;
    assign o_trace_sample = sample_q;
    assign o_trace_timestep = timestep_q;
    assign o_trace_version = version_q;
    assign o_trace_channel = 9'(current_feature >> 2);
    assign o_trace_spatial = current_feature[1:0];
    assign o_trace_q8_8 = trace_result;
    assign o_trace_last = int'(current_feature) + 1 ==
        int'(feature_count_q);
    assign o_busy = state_q != TRACE_IDLE || !trace_idle;

    unified_scnn_q8_8_trace_state_engine #(
        .NUM_NEURONS(TRACE_COUNT), .ADDR_WIDTH(TRACE_ADDR_WIDTH)
    ) u_trace_state (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_sample_reset(cfg_fire && i_cfg_trace_state_reset),
        .i_valid(trace_input_valid), .o_ready(trace_input_ready),
        .i_addr(trace_input_addr),
        .i_spike_q8_8($signed(
            wave_payload_q[int'(lane_q)*16 +: 16])),
        .i_leak_q8_8(trace_leak_q), .o_valid(trace_result_valid),
        .i_ready(state_q == TRACE_RESULT && i_trace_ready),
        .o_addr(trace_result_addr), .o_trace_q8_8(trace_result),
        .o_idle(trace_idle),
        .o_reset_while_busy_error(trace_reset_error),
        .o_address_error(trace_address_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= TRACE_IDLE;
            sample_q <= '0;
            timestep_q <= '0;
            version_q <= '0;
            frame_q <= '0;
            feature_count_q <= '0;
            trace_leak_q <= '0;
            sample_reset_q <= 1'b0;
            sample_last_q <= 1'b0;
            wave_base_q <= '0;
            wave_payload_q <= '0;
            lane_q <= '0;
            o_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= TRACE_IDLE;
            o_done <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_clear_error && state_q == TRACE_IDLE && trace_idle)
                o_sticky_error <= 1'b0;
            if (trace_reset_error || trace_address_error)
                o_sticky_error <= 1'b1;

            case (state_q)
                TRACE_IDLE: begin
                    if (i_cfg_valid && !cfg_legal)
                        o_sticky_error <= 1'b1;
                    if (cfg_fire && cfg_legal) begin
                        sample_q <= i_cfg_sample;
                        timestep_q <= i_cfg_timestep;
                        version_q <= i_cfg_version;
                        frame_q <= i_cfg_frame_index;
                        feature_count_q <= i_cfg_feature_count;
                        trace_leak_q <= i_cfg_trace_leak_q8_8;
                        sample_reset_q <= i_cfg_trace_state_reset;
                        sample_last_q <= i_cfg_sample_last;
                        wave_base_q <= '0;
                        lane_q <= '0;
                        state_q <= TRACE_WAVE;
                    end
                end

                TRACE_WAVE: begin
                    if (wave_fire) begin
                        if (!wave_legal) begin
                            o_sticky_error <= 1'b1;
                            state_q <= TRACE_IDLE;
                        end else begin
                            wave_payload_q <= i_wave_q8_8_flat;
                            lane_q <= '0;
                            state_q <= TRACE_ISSUE;
                        end
                    end
                end

                TRACE_ISSUE: begin
                    if (trace_input_valid && trace_input_ready)
                        state_q <= TRACE_RESULT;
                end

                TRACE_RESULT: begin
                    if (trace_result_valid &&
                            trace_result_addr != trace_input_addr) begin
                        o_sticky_error <= 1'b1;
                        state_q <= TRACE_IDLE;
                    end else if (trace_result_fire) begin
                        if (o_trace_last) begin
                            state_q <= TRACE_IDLE;
                            o_done <= 1'b1;
                        end else if (int'(lane_q) + 1 == LANES) begin
                            wave_base_q <= wave_base_q + FEATURE_WIDTH'(LANES);
                            lane_q <= '0;
                            state_q <= TRACE_WAVE;
                        end else begin
                            lane_q <= lane_q + 1'b1;
                            state_q <= TRACE_ISSUE;
                        end
                    end
                end

                default: begin
                    state_q <= TRACE_IDLE;
                    o_sticky_error <= 1'b1;
                end
            endcase
        end
    end

    initial begin
        if (MAX_BATCH < 1 || MAX_FEATURES < 1 || LANES < 1 ||
                LANES > MAX_FEATURES)
            $error("SCNN target feature-wave trace parameter mismatch");
    end
endmodule

`default_nettype wire
