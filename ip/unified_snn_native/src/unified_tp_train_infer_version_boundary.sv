`timescale 1ns/1ps
`default_nettype none

// Shared visibility boundary between serial forward and online-learning
// windows. Parameter writes may occur in place, but their versions become
// visible only when the complete learning window commits. A dirty abort keeps
// the old versions and blocks the next forward until the backing store restores
// the last committed state.
module unified_tp_train_infer_version_boundary #(
    parameter int VERSION_WIDTH = 16
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,

    input  wire logic                         i_context_load_valid,
    output logic                         o_context_load_ready,
    input  wire logic [VERSION_WIDTH-1:0]     i_context_model_version,
    input  wire logic [VERSION_WIDTH-1:0]     i_context_adapt_version,

    input  wire logic                         i_forward_start_valid,
    output logic                         o_forward_start_ready,
    output logic                         o_forward_start_fire,
    input  wire logic                         i_forward_done,
    output logic                         o_forward_active,
    output logic [VERSION_WIDTH-1:0]     o_forward_model_version,
    output logic [VERSION_WIDTH-1:0]     o_forward_adapt_version,

    input  wire logic                         i_learning_start_valid,
    output logic                         o_learning_start_ready,
    output logic                         o_learning_start_fire,
    output logic                         o_learning_active,
    output logic [VERSION_WIDTH-1:0]     o_learning_base_model_version,
    output logic [VERSION_WIDTH-1:0]     o_learning_base_adapt_version,

    input  wire logic                         i_forward_code_write_fire,
    input  wire logic                         i_optimizer_state_write_fire,
    input  wire logic                         i_learning_commit_valid,
    output logic                         o_learning_commit_ready,
    input  wire logic                         i_learning_abort_complete,

    input  wire logic                         i_restore_complete,
    output logic                         o_restore_ready,
    output logic                         o_recovery_required,

    output logic [VERSION_WIDTH-1:0]     o_model_version,
    output logic [VERSION_WIDTH-1:0]     o_adapt_version,
    output logic                         o_commit_pulse,
    output logic                         o_commit_code_dirty,
    output logic                         o_commit_optimizer_dirty,
    output logic                         o_detach_pulse,

    input  wire logic                         i_clear_error,
    output logic                         o_protocol_error
);
    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_FORWARD,
        STATE_LEARNING,
        STATE_RECOVERY
    } state_t;

    state_t state_q;
    logic [VERSION_WIDTH-1:0] model_version_q;
    logic [VERSION_WIDTH-1:0] adapt_version_q;
    logic [VERSION_WIDTH-1:0] forward_model_version_q;
    logic [VERSION_WIDTH-1:0] forward_adapt_version_q;
    logic [VERSION_WIDTH-1:0] learning_base_model_version_q;
    logic [VERSION_WIDTH-1:0] learning_base_adapt_version_q;
    logic code_dirty_q;
    logic optimizer_dirty_q;
    logic protocol_error_q;
    logic context_load_fire;
    logic learning_commit_fire;
    logic commit_code_dirty;
    logic commit_optimizer_dirty;
    logic abort_dirty;

    always_comb begin
        o_context_load_ready = (state_q == STATE_IDLE) && !protocol_error_q;
        context_load_fire = i_context_load_valid && o_context_load_ready;

        o_forward_start_ready = (state_q == STATE_IDLE) &&
            !protocol_error_q && !i_context_load_valid;
        o_forward_start_fire = i_forward_start_valid &&
            o_forward_start_ready;
        o_forward_active = state_q == STATE_FORWARD;

        o_learning_start_ready = (state_q == STATE_IDLE) &&
            !protocol_error_q && !i_context_load_valid &&
            !i_forward_start_valid;
        o_learning_start_fire = i_learning_start_valid &&
            o_learning_start_ready;
        o_learning_active = state_q == STATE_LEARNING;

        o_learning_commit_ready = (state_q == STATE_LEARNING) &&
            !protocol_error_q && !i_learning_abort_complete;
        learning_commit_fire = i_learning_commit_valid &&
            o_learning_commit_ready;

        commit_code_dirty = code_dirty_q || i_forward_code_write_fire;
        commit_optimizer_dirty = optimizer_dirty_q ||
            i_optimizer_state_write_fire;
        abort_dirty = commit_code_dirty || commit_optimizer_dirty;

        o_restore_ready = state_q == STATE_RECOVERY;
        o_recovery_required = state_q == STATE_RECOVERY;
        o_model_version = model_version_q;
        o_adapt_version = adapt_version_q;
        o_forward_model_version = forward_model_version_q;
        o_forward_adapt_version = forward_adapt_version_q;
        o_learning_base_model_version = learning_base_model_version_q;
        o_learning_base_adapt_version = learning_base_adapt_version_q;
        o_protocol_error = protocol_error_q;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            model_version_q <= '0;
            adapt_version_q <= '0;
            forward_model_version_q <= '0;
            forward_adapt_version_q <= '0;
            learning_base_model_version_q <= '0;
            learning_base_adapt_version_q <= '0;
            code_dirty_q <= 1'b0;
            optimizer_dirty_q <= 1'b0;
            protocol_error_q <= 1'b0;
            o_commit_pulse <= 1'b0;
            o_commit_code_dirty <= 1'b0;
            o_commit_optimizer_dirty <= 1'b0;
            o_detach_pulse <= 1'b0;
        end else begin
            o_commit_pulse <= 1'b0;
            o_commit_code_dirty <= 1'b0;
            o_commit_optimizer_dirty <= 1'b0;
            o_detach_pulse <= 1'b0;

            if (i_clear_error &&
                ((state_q == STATE_IDLE) || (state_q == STATE_RECOVERY)))
                protocol_error_q <= 1'b0;

            case (state_q)
                STATE_IDLE: begin
                    code_dirty_q <= 1'b0;
                    optimizer_dirty_q <= 1'b0;
                    if (context_load_fire) begin
                        model_version_q <= i_context_model_version;
                        adapt_version_q <= i_context_adapt_version;
                    end else if (o_forward_start_fire) begin
                        forward_model_version_q <= model_version_q;
                        forward_adapt_version_q <= adapt_version_q;
                        state_q <= STATE_FORWARD;
                    end else if (o_learning_start_fire) begin
                        learning_base_model_version_q <= model_version_q;
                        learning_base_adapt_version_q <= adapt_version_q;
                        state_q <= STATE_LEARNING;
                    end
                end

                STATE_FORWARD: begin
                    if (i_forward_done)
                        state_q <= STATE_IDLE;
                end

                STATE_LEARNING: begin
                    if (i_forward_code_write_fire)
                        code_dirty_q <= 1'b1;
                    if (i_optimizer_state_write_fire)
                        optimizer_dirty_q <= 1'b1;

                    if (i_learning_abort_complete) begin
                        if (abort_dirty)
                            state_q <= STATE_RECOVERY;
                        else
                            state_q <= STATE_IDLE;
                        code_dirty_q <= 1'b0;
                        optimizer_dirty_q <= 1'b0;
                    end else if (learning_commit_fire) begin
                        if (commit_code_dirty)
                            model_version_q <= model_version_q + 1'b1;
                        if (commit_code_dirty || commit_optimizer_dirty)
                            adapt_version_q <= adapt_version_q + 1'b1;
                        o_commit_pulse <= 1'b1;
                        o_commit_code_dirty <= commit_code_dirty;
                        o_commit_optimizer_dirty <= commit_optimizer_dirty;
                        o_detach_pulse <= 1'b1;
                        code_dirty_q <= 1'b0;
                        optimizer_dirty_q <= 1'b0;
                        state_q <= STATE_IDLE;
                    end
                end

                STATE_RECOVERY: begin
                    if (i_restore_complete)
                        state_q <= STATE_IDLE;
                end

                default: state_q <= STATE_IDLE;
            endcase

            if (i_forward_done && (state_q != STATE_FORWARD))
                protocol_error_q <= 1'b1;
            if ((i_forward_code_write_fire ||
                 i_optimizer_state_write_fire) &&
                (state_q != STATE_LEARNING))
                protocol_error_q <= 1'b1;
            if (i_learning_commit_valid &&
                (state_q != STATE_LEARNING))
                protocol_error_q <= 1'b1;
            if (i_learning_abort_complete &&
                (state_q != STATE_LEARNING))
                protocol_error_q <= 1'b1;
            if (i_restore_complete && (state_q != STATE_RECOVERY))
                protocol_error_q <= 1'b1;
        end
    end
endmodule

`default_nettype wire
