`timescale 1ns/1ps
`default_nettype none

// FC W2 online-learning frontend: inference traces plus labelled logit rows are
// converted into sparse Q0.16 trace*error contributions for the shared W2
// gradient accumulator and parameter-update backend.
module unified_fc_tp_w2_online_frontend #(
    parameter int MAX_BATCH = 16,
    parameter int MAX_HIDDEN = 400,
    parameter int MAX_OUTPUTS = 16,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter bit OUTPUT_MAJOR_ADDRESS = 1'b0,
    parameter int PARAMETER_HIDDEN_STRIDE = 0,
    parameter int PARAMETER_OUTPUT_STRIDE = 0,
    parameter int BATCH_INDEX_WIDTH = (MAX_BATCH <= 2) ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int HIDDEN_INDEX_WIDTH = (MAX_HIDDEN <= 2) ? 1 : $clog2(MAX_HIDDEN),
    parameter int HIDDEN_COUNT_WIDTH = $clog2(MAX_HIDDEN + 1),
    parameter int OUTPUT_INDEX_WIDTH = (MAX_OUTPUTS <= 2) ? 1 : $clog2(MAX_OUTPUTS),
    parameter int OUTPUT_COUNT_WIDTH = $clog2(MAX_OUTPUTS + 1)
) (
    input  wire logic                                  i_clk,
    input  wire logic                                  i_rst_n,
    input  wire logic                                  i_cfg_valid,
    output logic                                  o_cfg_ready,
    input  wire logic [BATCH_COUNT_WIDTH-1:0]           i_cfg_batch_count,
    input  wire logic [HIDDEN_COUNT_WIDTH-1:0]          i_cfg_hidden_count,
    input  wire logic [OUTPUT_COUNT_WIDTH-1:0]          i_cfg_output_count,
    input  wire logic                                  i_cfg_output_major,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]           i_cfg_hidden_stride,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]           i_cfg_output_stride,
    output logic                                  o_cfg_error,
    output logic                                  o_active,
    input  wire logic                                  i_release,
    output logic                                  o_release_ready,
    input  wire logic                                  i_hidden_trace_valid,
    output logic                                  o_hidden_trace_ready,
    input  wire logic [BATCH_INDEX_WIDTH-1:0]           i_hidden_trace_sample_id,
    input  wire logic [HIDDEN_INDEX_WIDTH-1:0]          i_hidden_trace_hidden_index,
    input  wire logic [15:0]                           i_hidden_trace_q016,
    output logic                                  o_hidden_trace_error,
    input  wire logic                                  i_logit_row_valid,
    output logic                                  o_logit_row_ready,
    input  wire logic [BATCH_INDEX_WIDTH-1:0]           i_logit_sample_id,
    input  wire logic [OUTPUT_INDEX_WIDTH-1:0]          i_logit_label,
    input  wire logic                                  i_logit_valid,
    output logic                                  o_logit_ready,
    input  wire logic signed [15:0]                    i_logit_q6,
    output logic                                  o_logit_error,
    output logic                                  o_logit_row_done,
    input  wire logic                                  i_observation_done,
    output logic                                  o_observation_done_ready,
    input  wire logic                                  i_emit_start,
    output logic                                  o_emit_ready,
    output logic                                  o_contribution_valid,
    input  wire logic                                  i_contribution_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]            o_contribution_local_addr,
    output logic [15:0]                           o_contribution_source_trace,
    output logic signed [17:0]                    o_contribution_error_q016,
    output logic                                  o_emit_done,
    output logic                                  o_busy
);

    logic scheduler_cfg_ready;
    logic scheduler_cfg_error;
    logic scheduler_active;
    logic scheduler_release_ready;
    logic scheduler_output_error_ready;
    logic scheduler_output_error_error;
    logic output_row_ready;
    logic output_config_error;
    logic output_error_valid;
    logic [BATCH_INDEX_WIDTH-1:0] output_error_sample_id;
    logic [OUTPUT_INDEX_WIDTH-1:0] output_error_index;
    logic signed [17:0] output_error_q016;
    logic output_error_last;
    logic output_busy;
    logic scheduler_busy;
    logic cfg_fire;
    logic shared_release;
    logic [OUTPUT_COUNT_WIDTH-1:0] output_count_q;

    always_comb begin
        o_cfg_ready = scheduler_cfg_ready && !output_busy;
        cfg_fire = i_cfg_valid && o_cfg_ready;
        o_cfg_error = scheduler_cfg_error;
        o_active = scheduler_active;
        o_release_ready = scheduler_release_ready && !output_busy;
        shared_release = i_release && o_release_ready;
        o_logit_row_ready = scheduler_active && output_row_ready &&
            scheduler_output_error_ready;
        o_logit_error = output_config_error || scheduler_output_error_error ||
            (output_error_valid &&
             (output_error_last !=
              (OUTPUT_COUNT_WIDTH'({1'b0, output_error_index} + 1'b1) >=
               output_count_q)));
        o_busy = output_busy || scheduler_busy;
    end

    unified_fc_tp_output_error_frontend #(
        .SAMPLE_ID_WIDTH(BATCH_INDEX_WIDTH),
        .MAX_OUTPUTS(MAX_OUTPUTS),
        .OUTPUT_INDEX_WIDTH(OUTPUT_INDEX_WIDTH),
        .OUTPUT_COUNT_WIDTH(OUTPUT_COUNT_WIDTH)
    ) output_error_frontend_i (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_row_valid(i_logit_row_valid && o_logit_row_ready),
        .o_row_ready(output_row_ready),
        .i_sample_id(i_logit_sample_id),
        .i_output_count(output_count_q),
        .i_label(i_logit_label),
        .o_config_error(output_config_error),
        .i_logit_valid(i_logit_valid),
        .o_logit_ready(o_logit_ready),
        .i_logit_q6(i_logit_q6),
        .o_error_valid(output_error_valid),
        .i_error_ready(scheduler_output_error_ready),
        .o_error_sample_id(output_error_sample_id),
        .o_error_output_index(output_error_index),
        .o_error_q016(output_error_q016),
        .o_error_last(output_error_last),
        .o_done(o_logit_row_done),
        .o_busy(output_busy)
    );

    unified_fc_tp_w2_contribution_scheduler #(
        .MAX_BATCH(MAX_BATCH),
        .MAX_HIDDEN(MAX_HIDDEN),
        .MAX_OUTPUTS(MAX_OUTPUTS),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .ERROR_WIDTH(18),
        .OUTPUT_MAJOR_ADDRESS(OUTPUT_MAJOR_ADDRESS),
        .PARAMETER_HIDDEN_STRIDE(PARAMETER_HIDDEN_STRIDE),
        .PARAMETER_OUTPUT_STRIDE(PARAMETER_OUTPUT_STRIDE),
        .BATCH_INDEX_WIDTH(BATCH_INDEX_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .HIDDEN_COUNT_WIDTH(HIDDEN_COUNT_WIDTH),
        .OUTPUT_INDEX_WIDTH(OUTPUT_INDEX_WIDTH),
        .OUTPUT_COUNT_WIDTH(OUTPUT_COUNT_WIDTH)
    ) contribution_scheduler_i (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(cfg_fire),
        .o_cfg_ready(scheduler_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_hidden_count(i_cfg_hidden_count),
        .i_cfg_output_count(i_cfg_output_count),
        .i_cfg_output_major(i_cfg_output_major),
        .i_cfg_hidden_stride(i_cfg_hidden_stride),
        .i_cfg_output_stride(i_cfg_output_stride),
        .o_cfg_error(scheduler_cfg_error),
        .o_active(scheduler_active),
        .i_release(shared_release),
        .o_release_ready(scheduler_release_ready),
        .i_hidden_trace_valid(i_hidden_trace_valid),
        .o_hidden_trace_ready(o_hidden_trace_ready),
        .i_hidden_trace_sample_id(i_hidden_trace_sample_id),
        .i_hidden_trace_hidden_index(i_hidden_trace_hidden_index),
        .i_hidden_trace_q016(i_hidden_trace_q016),
        .o_hidden_trace_error(o_hidden_trace_error),
        .i_output_error_valid(output_error_valid),
        .o_output_error_ready(scheduler_output_error_ready),
        .i_output_error_sample_id(output_error_sample_id),
        .i_output_error_index(output_error_index),
        .i_output_error_q016(output_error_q016),
        .o_output_error_error(scheduler_output_error_error),
        .i_observation_done(i_observation_done),
        .o_observation_done_ready(o_observation_done_ready),
        .i_emit_start(i_emit_start),
        .o_emit_ready(o_emit_ready),
        .o_contribution_valid(o_contribution_valid),
        .i_contribution_ready(i_contribution_ready),
        .o_contribution_local_addr(o_contribution_local_addr),
        .o_contribution_source_trace(o_contribution_source_trace),
        .o_contribution_error_q016(o_contribution_error_q016),
        .o_emit_done(o_emit_done),
        .o_busy(scheduler_busy)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            output_count_q <= '0;
        end else begin
            if (cfg_fire)
                output_count_q <= i_cfg_output_count;
            if (shared_release)
                output_count_q <= '0;
        end
    end

endmodule

`default_nettype wire
