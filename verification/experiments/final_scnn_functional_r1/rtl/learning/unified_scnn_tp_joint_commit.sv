`timescale 1ns/1ps
`default_nettype none

// Join held Conv8 window_done and classifier window_commit_valid. The shared
// version boundary remains the only version authority. Begin must precede any
// selected learner writes; done means all selected timesteps and RMW drained.
// Direct connection to the current Conv8 replay path is T=1 only. For T>1 a
// parent must release intermediate Conv8 timesteps and join only its final one.
// This block controls visibility, not rollback: recovery_ack requires actual
// store restoration (when dirty) and child cleanup by the parent.
module unified_scnn_tp_joint_commit (
    input wire logic i_clk, i_rst_n,
    input wire logic i_start_valid,
    output logic o_start_ready,
    input wire logic [1:0] i_selected_banks, // bit 0 Conv8, bit 1 classifier
    input wire logic [7:0] i_profile_id,
    input wire logic [31:0] i_sequence,
    output logic o_learning_start_valid,
    input wire logic i_learning_start_ready,
    output logic o_start_fire,
    input wire logic i_conv8_done, i_conv8_release_ready,
    output logic o_conv8_release,
    input wire logic i_classifier_commit_valid,
    input wire logic [7:0] i_classifier_profile_id,
    input wire logic [31:0] i_classifier_sequence,
    output logic o_classifier_commit_ready,
    input wire logic i_rmw_quiescent,
    input wire logic i_abort, i_child_error,
    output logic o_abort_request,
    input wire logic i_children_quiescent,
    output logic o_abort_complete,
    input wire logic i_recovery_ack,
    output logic o_commit_valid,
    input wire logic i_commit_ready,
    output logic [7:0] o_commit_profile_id,
    output logic [31:0] o_commit_sequence,
    output logic o_commit_fire,
    output logic o_inference_hold,
    output logic o_owner_quiescent,
    output logic o_fault,
    output logic o_config_error
);
    typedef enum logic [1:0] {IDLE, WAIT_BANKS, ABORT_DRAIN, FAULT_LOCK} state_t;
    state_t state_q;
    logic [1:0] banks_q;
    logic [7:0] profile_q;
    logic [31:0] sequence_q;
    logic metadata_error, fault_now, banks_done;
    logic drain_seen_q;

    assign metadata_error = state_q == WAIT_BANKS && banks_q[1] &&
        i_classifier_commit_valid &&
        (i_classifier_profile_id != profile_q || i_classifier_sequence != sequence_q);
    assign fault_now = i_abort || i_child_error || metadata_error;
    assign o_learning_start_valid = state_q == IDLE && i_start_valid &&
        i_selected_banks != 0 && !fault_now;
    assign o_start_ready = state_q == IDLE && i_learning_start_ready &&
        i_selected_banks != 0 && !fault_now;
    assign o_start_fire = i_start_valid && o_start_ready;
    assign o_config_error = state_q == IDLE && i_start_valid && i_selected_banks == 0;
    assign banks_done = (!banks_q[0] || (i_conv8_done && i_conv8_release_ready)) &&
        (!banks_q[1] || i_classifier_commit_valid);
    assign o_commit_valid = state_q == WAIT_BANKS && banks_done &&
        i_rmw_quiescent && !fault_now;
    assign o_commit_fire = o_commit_valid && i_commit_ready;
    // Neither child can rearm until the common visibility commit is accepted.
    assign o_conv8_release = o_commit_fire && banks_q[0];
    assign o_classifier_commit_ready = o_commit_fire && banks_q[1];
    assign o_commit_profile_id = profile_q;
    assign o_commit_sequence = sequence_q;
    assign o_abort_request = state_q == ABORT_DRAIN ||
        (state_q == WAIT_BANKS && fault_now);
    assign o_abort_complete = state_q == ABORT_DRAIN &&
        drain_seen_q && i_children_quiescent && i_rmw_quiescent;
    assign o_inference_hold = state_q != IDLE || o_learning_start_valid;
    assign o_owner_quiescent = state_q == IDLE && !o_learning_start_valid &&
        i_children_quiescent && i_rmw_quiescent && !i_child_error;
    assign o_fault = state_q inside {ABORT_DRAIN, FAULT_LOCK} || metadata_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= IDLE;
            banks_q <= '0;
            profile_q <= '0;
            sequence_q <= '0;
            drain_seen_q <= 1'b0;
        end else begin
            drain_seen_q <= state_q == ABORT_DRAIN &&
                i_children_quiescent && i_rmw_quiescent;
            case (state_q)
                IDLE: if (o_start_fire) begin
                    banks_q <= i_selected_banks;
                    profile_q <= i_profile_id;
                    sequence_q <= i_sequence;
                    state_q <= WAIT_BANKS;
                end
                WAIT_BANKS: begin
                    if (fault_now) state_q <= ABORT_DRAIN;
                    else if (o_commit_fire) state_q <= IDLE;
                end
                ABORT_DRAIN: if (o_abort_complete) state_q <= FAULT_LOCK;
                FAULT_LOCK: if (i_recovery_ack && i_children_quiescent &&
                        i_rmw_quiescent && !i_abort && !i_child_error)
                    state_q <= IDLE;
                default: state_q <= FAULT_LOCK;
            endcase
        end
    end
endmodule
`default_nettype wire
