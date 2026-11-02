`timescale 1ns/1ps
`default_nettype none

// Turns one coordinator pass request into prescan -> graph execution. Target
// passes atomically configure their temporal-trace context with prescan start.
module unified_scnn_selected_layers_graph_launch_prep_v1 #(
    parameter int SAMPLE_WIDTH = 4,
    parameter int TIMESTEP_WIDTH = 5,
    parameter int VERSION_WIDTH = 16,
    parameter int CLASS_WIDTH = 4
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_abort,
    input  wire logic                         i_clear_error,

    input  wire logic                         i_pass_valid,
    output logic                              o_pass_ready,
    input  wire logic                         i_prescan_enable,
    input  wire logic                         i_pass_target,
    input  wire logic [SAMPLE_WIDTH-1:0]       i_pass_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]     i_pass_timestep,
    input  wire logic [VERSION_WIDTH-1:0]      i_pass_version,
    input  wire logic [CLASS_WIDTH-1:0]        i_pass_label,
    output logic                              o_pass_done,

    output logic                              o_prescan_start_valid,
    input  wire logic                         i_prescan_start_ready,
    output logic                              o_prescan_target,
    output logic [SAMPLE_WIDTH-1:0]            o_prescan_sample,
    output logic [TIMESTEP_WIDTH-1:0]          o_prescan_timestep,
    output logic [VERSION_WIDTH-1:0]           o_prescan_version,
    input  wire logic                         i_prescan_done,
    input  wire logic                         i_prescan_error,

    output logic                              o_target_trace_cfg_valid,
    input  wire logic                         i_target_trace_cfg_ready,

    output logic                              o_graph_start_valid,
    input  wire logic                         i_graph_start_ready,
    output logic                              o_graph_target,
    output logic [SAMPLE_WIDTH-1:0]            o_graph_sample,
    output logic [TIMESTEP_WIDTH-1:0]          o_graph_timestep,
    output logic [VERSION_WIDTH-1:0]           o_graph_version,
    output logic [CLASS_WIDTH-1:0]             o_graph_label,
    input  wire logic                         i_graph_done,
    input  wire logic                         i_graph_error,

    output logic                              o_busy,
    output logic                              o_sticky_error
);
    typedef enum logic [1:0] {
        PREP_IDLE, PREP_SCAN, PREP_GRAPH, PREP_WAIT
    } prep_state_t;

    prep_state_t state_q;
    logic target_q;
    logic [SAMPLE_WIDTH-1:0] sample_q;
    logic [TIMESTEP_WIDTH-1:0] timestep_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic [CLASS_WIDTH-1:0] label_q;
    logic pass_fire;

    assign o_pass_ready = state_q == PREP_IDLE &&
        (!i_prescan_enable || (i_prescan_start_ready &&
         (!i_pass_target || i_target_trace_cfg_ready))) && !o_sticky_error;
    assign pass_fire = i_pass_valid && o_pass_ready;
    assign o_prescan_start_valid = state_q == PREP_IDLE && i_pass_valid &&
        i_prescan_enable &&
        (!i_pass_target || i_target_trace_cfg_ready) && !o_sticky_error;
    assign o_target_trace_cfg_valid = state_q == PREP_IDLE &&
        i_pass_valid && i_prescan_enable && i_pass_target &&
        i_prescan_start_ready &&
        !o_sticky_error;
    assign o_graph_start_valid = state_q == PREP_GRAPH && !o_sticky_error;

    assign o_prescan_target = state_q == PREP_IDLE ? i_pass_target : target_q;
    assign o_prescan_sample = state_q == PREP_IDLE ? i_pass_sample : sample_q;
    assign o_prescan_timestep = state_q == PREP_IDLE ?
        i_pass_timestep : timestep_q;
    assign o_prescan_version = state_q == PREP_IDLE ?
        i_pass_version : version_q;
    // The target-trace configuration fires with pass admission. Bypass the
    // incoming metadata on that edge, then hold the captured context through
    // prescan and graph execution.
    assign o_graph_target = state_q == PREP_IDLE ? i_pass_target : target_q;
    assign o_graph_sample = state_q == PREP_IDLE ? i_pass_sample : sample_q;
    assign o_graph_timestep = state_q == PREP_IDLE ?
        i_pass_timestep : timestep_q;
    assign o_graph_version = state_q == PREP_IDLE ?
        i_pass_version : version_q;
    assign o_graph_label = state_q == PREP_IDLE ? i_pass_label : label_q;
    assign o_busy = state_q != PREP_IDLE;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= PREP_IDLE;
            target_q <= 1'b0;
            sample_q <= '0;
            timestep_q <= '0;
            version_q <= '0;
            label_q <= '0;
            o_pass_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= PREP_IDLE;
            o_pass_done <= 1'b0;
        end else begin
            o_pass_done <= 1'b0;
            if (i_clear_error && state_q == PREP_IDLE)
                o_sticky_error <= 1'b0;
            if (i_prescan_done && state_q != PREP_SCAN ||
                    i_graph_done && state_q != PREP_WAIT)
                o_sticky_error <= 1'b1;

            case (state_q)
                PREP_IDLE: begin
                    if (pass_fire) begin
                        target_q <= i_pass_target;
                        sample_q <= i_pass_sample;
                        timestep_q <= i_pass_timestep;
                        version_q <= i_pass_version;
                        label_q <= i_pass_label;
                        state_q <= i_prescan_enable ? PREP_SCAN : PREP_GRAPH;
                    end
                end

                PREP_SCAN: begin
                    if (i_prescan_error) begin
                        state_q <= PREP_IDLE;
                        o_sticky_error <= 1'b1;
                    end else if (i_prescan_done) begin
                        state_q <= PREP_GRAPH;
                    end
                end

                PREP_GRAPH: begin
                    if (i_graph_error) begin
                        state_q <= PREP_IDLE;
                        o_sticky_error <= 1'b1;
                    end else if (o_graph_start_valid && i_graph_start_ready) begin
                        state_q <= PREP_WAIT;
                    end
                end

                PREP_WAIT: begin
                    if (i_graph_error) begin
                        state_q <= PREP_IDLE;
                        o_sticky_error <= 1'b1;
                    end else if (i_graph_done) begin
                        state_q <= PREP_IDLE;
                        o_pass_done <= 1'b1;
                    end
                end

                default: begin
                    state_q <= PREP_IDLE;
                    o_sticky_error <= 1'b1;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
