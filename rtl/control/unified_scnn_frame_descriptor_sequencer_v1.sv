`timescale 1ns/1ps
`default_nettype none

// One batch-coordinator frame contains one or more serialized layer descriptors.
// Only the final descriptor retires the frame; descriptors stay stable through
// context transfer, compute and output draining inside the graph client.
module unified_scnn_frame_descriptor_sequencer_v1 #(
    parameter int DESCRIPTOR_WIDTH = 1
) (
    input  wire logic i_clk,
    input  wire logic i_rst_n,
    input  wire logic i_clear_error,
    input  wire logic i_frame_start_valid,
    output logic o_frame_start_ready,
    output logic o_frame_done,
    input  wire logic i_descriptor_valid,
    output logic o_descriptor_ready,
    input  wire logic [DESCRIPTOR_WIDTH-1:0] i_descriptor,
    input  wire logic i_descriptor_last,
    output logic o_descriptor_done,
    output logic [DESCRIPTOR_WIDTH-1:0] o_descriptor,
    output logic o_tail_active,
    output logic o_layer_start_valid,
    input  wire logic i_layer_start_ready,
    input  wire logic i_layer_done,
    output logic o_busy,
    output logic o_sticky_error
);
    typedef enum logic [1:0] {IDLE, DESCRIPTOR, START, WAIT_DONE} state_t;
    state_t state_q;
    logic last_q;

    assign o_frame_start_ready = state_q == IDLE && !o_sticky_error;
    assign o_descriptor_ready = state_q == DESCRIPTOR && !o_sticky_error;
    assign o_layer_start_valid = state_q == START && !o_sticky_error;
    assign o_tail_active = last_q && (state_q == START || state_q == WAIT_DONE);
    assign o_busy = state_q != IDLE;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= IDLE;
            last_q <= 1'b0;
            o_descriptor <= '0;
            o_frame_done <= 1'b0;
            o_descriptor_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else begin
            o_frame_done <= 1'b0;
            o_descriptor_done <= 1'b0;
            if (i_clear_error && state_q == IDLE)
                o_sticky_error <= 1'b0;
            if (i_layer_done && state_q != WAIT_DONE)
                o_sticky_error <= 1'b1;
            case (state_q)
                IDLE: if (i_frame_start_valid && o_frame_start_ready)
                    state_q <= DESCRIPTOR;
                DESCRIPTOR: if (i_descriptor_valid && o_descriptor_ready) begin
                    o_descriptor <= i_descriptor;
                    last_q <= i_descriptor_last;
                    state_q <= START;
                end
                START: if (o_layer_start_valid && i_layer_start_ready)
                    state_q <= WAIT_DONE;
                WAIT_DONE: if (i_layer_done) begin
                    o_descriptor_done <= 1'b1;
                    if (last_q) begin
                        o_frame_done <= 1'b1;
                        state_q <= IDLE;
                    end else begin
                        state_q <= DESCRIPTOR;
                    end
                end
                default: begin
                    state_q <= IDLE;
                    o_sticky_error <= 1'b1;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
