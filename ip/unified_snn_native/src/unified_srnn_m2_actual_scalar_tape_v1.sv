`timescale 1ns/1ps
`default_nettype none

// Decouples the frozen SRNN backbone from the M2 consumer while both reuse the
// same physical H40 array.  The backbone first retires into this ordered tape;
// M2 can then acquire the array and replay the captured Q14 observations.
// Payload RAM is reset-free.  State and counters are the validity authority.
module unified_srnn_m2_actual_scalar_tape_v1 #(
    parameter int MAX_RECORDS = 480,
    parameter int SAMPLE_ID_WIDTH = 6,
    parameter int BACKBONE_ID_WIDTH = $clog2(MAX_RECORDS),
    parameter int COUNT_WIDTH = $clog2(MAX_RECORDS + 1)
) (
    input  wire logic                           i_clk,
    input  wire logic                           i_rst_n,
    input  wire logic                           i_abort,
    input  wire logic                           i_clear_error,

    input  wire logic                           i_cfg_valid,
    output logic                                o_cfg_ready,
    input  wire logic [COUNT_WIDTH-1:0]         i_cfg_record_count,

    input  wire logic                           i_capture_valid,
    output logic                                o_capture_ready,
    input  wire logic                           i_capture_path,
    input  wire logic [SAMPLE_ID_WIDTH-1:0]     i_capture_sample_id,
    input  wire logic [BACKBONE_ID_WIDTH-1:0]   i_capture_id,
    input  wire logic signed [31:0]             i_capture_q14,
    input  wire logic                           i_capture_last,
    output logic                                o_capture_done,
    output logic                                o_capture_complete,

    input  wire logic                           i_replay_start,
    output logic                                o_replay_start_ready,
    output logic                                o_replay_valid,
    input  wire logic                           i_replay_ready,
    output logic [SAMPLE_ID_WIDTH-1:0]          o_replay_sample_id,
    output logic [BACKBONE_ID_WIDTH-1:0]        o_replay_id,
    output logic signed [31:0]                  o_replay_q14,
    output logic                                o_replay_last,
    output logic                                o_replay_done,

    output logic                                o_busy,
    output logic                                o_idle,
    output logic                                o_sticky_error
);
    localparam int PAYLOAD_WIDTH = SAMPLE_ID_WIDTH +
        BACKBONE_ID_WIDTH + 32 + 1;

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_CAPTURE,
        STATE_READY,
        STATE_REPLAY,
        STATE_ERROR
    } state_t;

    (* ram_style = "block" *) logic [PAYLOAD_WIDTH-1:0]
        payload_memory [0:MAX_RECORDS-1];

    state_t state_q;
    logic [COUNT_WIDTH-1:0] record_count_q;
    logic [COUNT_WIDTH-1:0] capture_index_q;
    logic [COUNT_WIDTH-1:0] replay_index_q;
    logic [PAYLOAD_WIDTH-1:0] replay_memory_data_q;
    logic [PAYLOAD_WIDTH-1:0] replay_payload_q;
    logic replay_read_pending_q;
    logic replay_valid_q;

    logic cfg_legal;
    logic cfg_fire;
    logic capture_fire;
    logic capture_metadata_legal;
    logic replay_start_fire;
    logic replay_fire;
    logic replay_read_request;

    always_comb begin
        cfg_legal = i_cfg_record_count != 0 &&
            i_cfg_record_count <= COUNT_WIDTH'(MAX_RECORDS);
        o_cfg_ready = state_q == STATE_IDLE && !o_sticky_error;
        cfg_fire = i_cfg_valid && o_cfg_ready;

        o_capture_ready = state_q == STATE_CAPTURE && !o_sticky_error;
        capture_fire = i_capture_valid && o_capture_ready;
        capture_metadata_legal = !i_capture_path &&
            i_capture_id == BACKBONE_ID_WIDTH'(capture_index_q) &&
            i_capture_last ==
                (capture_index_q + 1'b1 == record_count_q);
        o_capture_complete = state_q == STATE_READY ||
            state_q == STATE_REPLAY;

        o_replay_start_ready = state_q == STATE_READY && !o_sticky_error;
        replay_start_fire = i_replay_start && o_replay_start_ready;
        o_replay_valid = replay_valid_q && !o_sticky_error;
        o_replay_sample_id = replay_payload_q[
            PAYLOAD_WIDTH-1 -: SAMPLE_ID_WIDTH];
        o_replay_id = replay_payload_q[
            PAYLOAD_WIDTH-SAMPLE_ID_WIDTH-1 -: BACKBONE_ID_WIDTH];
        o_replay_q14 = $signed(replay_payload_q[32:1]);
        o_replay_last = replay_payload_q[0];
        replay_fire = o_replay_valid && i_replay_ready;
        replay_read_request = state_q == STATE_REPLAY &&
            !replay_read_pending_q && !replay_valid_q;

        o_busy = state_q != STATE_IDLE;
        o_idle = state_q == STATE_IDLE;
    end

    // Keep the inferred RAM out of reset logic.
    always_ff @(posedge i_clk) begin
        if (i_rst_n && !i_abort && capture_fire &&
                capture_metadata_legal) begin
            payload_memory[capture_index_q] <= {
                i_capture_sample_id,
                i_capture_id,
                i_capture_q14,
                i_capture_last
            };
        end else if (i_rst_n && !i_abort && replay_read_request) begin
            replay_memory_data_q <= payload_memory[replay_index_q];
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            record_count_q <= '0;
            capture_index_q <= '0;
            replay_index_q <= '0;
            replay_payload_q <= '0;
            replay_read_pending_q <= 1'b0;
            replay_valid_q <= 1'b0;
            o_capture_done <= 1'b0;
            o_replay_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            capture_index_q <= '0;
            replay_index_q <= '0;
            replay_read_pending_q <= 1'b0;
            replay_valid_q <= 1'b0;
            o_capture_done <= 1'b0;
            o_replay_done <= 1'b0;
        end else begin
            o_capture_done <= 1'b0;
            o_replay_done <= 1'b0;

            if (i_clear_error && state_q == STATE_ERROR) begin
                state_q <= STATE_IDLE;
                o_sticky_error <= 1'b0;
            end else begin
                unique case (state_q)
                    STATE_IDLE: begin
                        if (cfg_fire) begin
                            if (!cfg_legal) begin
                                state_q <= STATE_ERROR;
                                o_sticky_error <= 1'b1;
                            end else begin
                                state_q <= STATE_CAPTURE;
                                record_count_q <= i_cfg_record_count;
                                capture_index_q <= '0;
                            end
                        end
                    end

                    STATE_CAPTURE: begin
                        if (capture_fire) begin
                            if (!capture_metadata_legal) begin
                                state_q <= STATE_ERROR;
                                o_sticky_error <= 1'b1;
                            end else if (capture_index_q + 1'b1 ==
                                    record_count_q) begin
                                state_q <= STATE_READY;
                                o_capture_done <= 1'b1;
                            end else begin
                                capture_index_q <= capture_index_q + 1'b1;
                            end
                        end
                    end

                    STATE_READY: begin
                        if (replay_start_fire) begin
                            state_q <= STATE_REPLAY;
                            replay_index_q <= '0;
                            replay_read_pending_q <= 1'b0;
                            replay_valid_q <= 1'b0;
                        end
                    end

                    STATE_REPLAY: begin
                        if (replay_read_request) begin
                            replay_read_pending_q <= 1'b1;
                        end else if (replay_read_pending_q) begin
                            replay_payload_q <= replay_memory_data_q;
                            replay_read_pending_q <= 1'b0;
                            replay_valid_q <= 1'b1;
                        end

                        if (replay_fire) begin
                            replay_valid_q <= 1'b0;
                            if (replay_index_q + 1'b1 == record_count_q) begin
                                state_q <= STATE_IDLE;
                                o_replay_done <= 1'b1;
                            end else begin
                                replay_index_q <= replay_index_q + 1'b1;
                            end
                        end
                    end

                    default: begin
                        if (i_clear_error) begin
                            state_q <= STATE_IDLE;
                            o_sticky_error <= 1'b0;
                        end
                    end
                endcase
            end
        end
    end

    initial begin
        if (MAX_RECORDS < 2 || BACKBONE_ID_WIDTH < 1 || COUNT_WIDTH < 2)
            $error("SRNN actual scalar tape has invalid geometry");
    end
endmodule

`default_nettype wire
