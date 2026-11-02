`timescale 1ns/1ps
`default_nettype none

// Serializes the frozen 700->450 backbone and the M2 learning head while both
// reuse the same H40 array. Actual backbone observations are captured first,
// then replayed only after the backbone transaction has retired. In learning
// mode a caller-provided target trajectory follows the actual trajectory.
module unified_srnn_backbone_m2_phase_bridge_v1 #(
    parameter int MAX_RECORDS = 480,
    parameter int SAMPLE_ID_WIDTH = 6,
    parameter int BACKBONE_ID_WIDTH = $clog2(MAX_RECORDS),
    parameter int COUNT_WIDTH = $clog2(MAX_RECORDS + 1)
) (
    input  wire logic                           i_clk,
    input  wire logic                           i_rst_n,
    input  wire logic                           i_abort,
    input  wire logic                           i_clear_error,

    input  wire logic                           i_capture_cfg_valid,
    output logic                                o_capture_cfg_ready,
    input  wire logic [COUNT_WIDTH-1:0]         i_capture_record_count,
    input  wire logic                           i_actual_valid,
    output logic                                o_actual_ready,
    input  wire logic                           i_actual_path,
    input  wire logic [SAMPLE_ID_WIDTH-1:0]     i_actual_sample_id,
    input  wire logic [BACKBONE_ID_WIDTH-1:0]   i_actual_backbone_id,
    input  wire logic signed [31:0]             i_actual_q14,
    input  wire logic                           i_actual_last,
    output logic                                o_capture_done,
    output logic                                o_capture_complete,

    input  wire logic                           i_sequence_start_valid,
    output logic                                o_sequence_start_ready,
    input  wire logic                           i_include_target_path,
    output logic                                o_plane_timestep_start,
    input  wire logic                           i_plane_timestep_start_ready,

    input  wire logic                           i_target_valid,
    output logic                                o_target_ready,
    input  wire logic [SAMPLE_ID_WIDTH-1:0]     i_target_sample_id,
    input  wire logic [BACKBONE_ID_WIDTH-1:0]   i_target_backbone_id,
    input  wire logic signed [31:0]             i_target_q14,
    input  wire logic                           i_target_last,
    input  wire logic                           i_target_done,
    input  wire logic                           i_target_error,

    output logic                                o_plane_backbone_valid,
    input  wire logic                           i_plane_backbone_ready,
    output logic                                o_plane_backbone_path,
    output logic [SAMPLE_ID_WIDTH-1:0]          o_plane_backbone_sample_id,
    output logic [BACKBONE_ID_WIDTH-1:0]        o_plane_backbone_id,
    output logic signed [31:0]                  o_plane_backbone_q14,
    output logic                                o_plane_backbone_last,
    output logic                                o_plane_backbone_done,
    output logic                                o_plane_backbone_error,

    output logic                                o_delivery_done,
    output logic                                o_busy,
    output logic                                o_idle,
    output logic                                o_sticky_error
);
    typedef enum logic [2:0] {
        PHASE_IDLE,
        PHASE_CAPTURE,
        PHASE_READY,
        PHASE_ACTUAL,
        PHASE_TARGET,
        PHASE_ACTUAL_DONE
    } phase_t;

    phase_t phase_q;
    logic include_target_q;
    logic tape_cfg_ready;
    logic tape_capture_ready;
    logic tape_capture_done;
    logic tape_capture_complete;
    logic tape_replay_start;
    logic tape_replay_start_ready;
    logic tape_replay_valid;
    logic tape_replay_ready;
    logic [SAMPLE_ID_WIDTH-1:0] tape_replay_sample_id;
    logic [BACKBONE_ID_WIDTH-1:0] tape_replay_id;
    logic signed [31:0] tape_replay_q14;
    logic tape_replay_last;
    logic tape_replay_done;
    logic tape_error;
    logic start_fire;

    assign o_capture_cfg_ready = phase_q == PHASE_IDLE && tape_cfg_ready;
    assign o_actual_ready = phase_q == PHASE_CAPTURE && tape_capture_ready;
    assign o_capture_done = tape_capture_done;
    assign o_capture_complete = phase_q == PHASE_READY ||
        phase_q == PHASE_ACTUAL || phase_q == PHASE_TARGET;

    assign o_sequence_start_ready = phase_q == PHASE_READY &&
        tape_replay_start_ready && i_plane_timestep_start_ready &&
        !o_sticky_error;
    assign start_fire = i_sequence_start_valid && o_sequence_start_ready;
    assign tape_replay_start = start_fire;
    assign o_plane_timestep_start = start_fire;

    assign tape_replay_ready = phase_q == PHASE_ACTUAL &&
        i_plane_backbone_ready;
    assign o_target_ready = phase_q == PHASE_TARGET &&
        i_plane_backbone_ready;

    always_comb begin
        o_plane_backbone_valid = 1'b0;
        o_plane_backbone_path = 1'b0;
        o_plane_backbone_sample_id = '0;
        o_plane_backbone_id = '0;
        o_plane_backbone_q14 = '0;
        o_plane_backbone_last = 1'b0;
        o_plane_backbone_done = 1'b0;
        o_plane_backbone_error = tape_error;

        if (phase_q == PHASE_ACTUAL) begin
            o_plane_backbone_valid = tape_replay_valid;
            o_plane_backbone_path = 1'b0;
            o_plane_backbone_sample_id = tape_replay_sample_id;
            o_plane_backbone_id = tape_replay_id;
            o_plane_backbone_q14 = tape_replay_q14;
            // During learning the target trajectory closes the combined
            // observation stream, so actual-path end markers are suppressed.
            o_plane_backbone_last = !include_target_q &&
                tape_replay_last;
        end else if (phase_q == PHASE_TARGET) begin
            o_plane_backbone_valid = i_target_valid;
            o_plane_backbone_path = 1'b1;
            o_plane_backbone_sample_id = i_target_sample_id;
            o_plane_backbone_id = i_target_backbone_id;
            o_plane_backbone_q14 = i_target_q14;
            o_plane_backbone_last = i_target_last;
            o_plane_backbone_done = i_target_done;
            o_plane_backbone_error = tape_error || i_target_error;
        end else if (phase_q == PHASE_ACTUAL_DONE) begin
            o_plane_backbone_done = 1'b1;
        end
    end

    assign o_busy = phase_q != PHASE_IDLE;
    assign o_idle = phase_q == PHASE_IDLE;
    assign o_sticky_error = tape_error;

    unified_srnn_m2_actual_scalar_tape_v1 #(
        .MAX_RECORDS(MAX_RECORDS),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .BACKBONE_ID_WIDTH(BACKBONE_ID_WIDTH),
        .COUNT_WIDTH(COUNT_WIDTH)
    ) u_actual_tape (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_cfg_valid(i_capture_cfg_valid && phase_q == PHASE_IDLE),
        .o_cfg_ready(tape_cfg_ready),
        .i_cfg_record_count(i_capture_record_count),
        .i_capture_valid(i_actual_valid && phase_q == PHASE_CAPTURE),
        .o_capture_ready(tape_capture_ready),
        .i_capture_path(i_actual_path),
        .i_capture_sample_id(i_actual_sample_id),
        .i_capture_id(i_actual_backbone_id),
        .i_capture_q14(i_actual_q14), .i_capture_last(i_actual_last),
        .o_capture_done(tape_capture_done),
        .o_capture_complete(tape_capture_complete),
        .i_replay_start(tape_replay_start),
        .o_replay_start_ready(tape_replay_start_ready),
        .o_replay_valid(tape_replay_valid),
        .i_replay_ready(tape_replay_ready),
        .o_replay_sample_id(tape_replay_sample_id),
        .o_replay_id(tape_replay_id), .o_replay_q14(tape_replay_q14),
        .o_replay_last(tape_replay_last),
        .o_replay_done(tape_replay_done), .o_busy(), .o_idle(),
        .o_sticky_error(tape_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            phase_q <= PHASE_IDLE;
            include_target_q <= 1'b0;
            o_delivery_done <= 1'b0;
        end else if (i_abort) begin
            phase_q <= PHASE_IDLE;
            include_target_q <= 1'b0;
            o_delivery_done <= 1'b0;
        end else begin
            o_delivery_done <= 1'b0;
            unique case (phase_q)
                PHASE_IDLE: begin
                    if (i_capture_cfg_valid && o_capture_cfg_ready)
                        phase_q <= PHASE_CAPTURE;
                end
                PHASE_CAPTURE: begin
                    if (tape_capture_done)
                        phase_q <= PHASE_READY;
                end
                PHASE_READY: begin
                    if (start_fire) begin
                        include_target_q <= i_include_target_path;
                        phase_q <= PHASE_ACTUAL;
                    end
                end
                PHASE_ACTUAL: begin
                    if (tape_replay_done) begin
                        if (include_target_q)
                            phase_q <= PHASE_TARGET;
                        else
                            phase_q <= PHASE_ACTUAL_DONE;
                    end
                end
                PHASE_TARGET: begin
                    if (i_target_done) begin
                        o_delivery_done <= 1'b1;
                        phase_q <= PHASE_IDLE;
                    end
                end
                PHASE_ACTUAL_DONE: begin
                    o_delivery_done <= 1'b1;
                    phase_q <= PHASE_IDLE;
                end
                default: phase_q <= PHASE_IDLE;
            endcase
        end
    end

    initial begin
        if (MAX_RECORDS < 2 || SAMPLE_ID_WIDTH < 1 ||
                BACKBONE_ID_WIDTH < 1 || COUNT_WIDTH < 2)
            $error("SRNN backbone/M2 phase bridge has invalid geometry");
    end
endmodule

`default_nettype wire
