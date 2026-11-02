`timescale 1ns/1ps
`default_nettype none

// FC classification adapter around the shared row softmax. It converts one
// Q6 logit row into signed Q0.16 (softmax - one_hot(label)) error tokens.
module unified_fc_tp_output_error_frontend #(
    parameter int SAMPLE_ID_WIDTH = 4,
    parameter int MAX_OUTPUTS = 16,
    parameter int OUTPUT_INDEX_WIDTH = (MAX_OUTPUTS <= 2) ? 1 : $clog2(MAX_OUTPUTS),
    parameter int OUTPUT_COUNT_WIDTH = $clog2(MAX_OUTPUTS + 1)
) (
    input  wire logic                                  i_clk,
    input  wire logic                                  i_rst_n,
    input  wire logic                                  i_row_valid,
    output logic                                  o_row_ready,
    input  wire logic [SAMPLE_ID_WIDTH-1:0]             i_sample_id,
    input  wire logic [OUTPUT_COUNT_WIDTH-1:0]          i_output_count,
    input  wire logic [OUTPUT_INDEX_WIDTH-1:0]          i_label,
    output logic                                  o_config_error,
    input  wire logic                                  i_logit_valid,
    output logic                                  o_logit_ready,
    input  wire logic signed [15:0]                    i_logit_q6,
    output logic                                  o_error_valid,
    input  wire logic                                  i_error_ready,
    output logic [SAMPLE_ID_WIDTH-1:0]             o_error_sample_id,
    output logic [OUTPUT_INDEX_WIDTH-1:0]          o_error_output_index,
    output logic signed [17:0]                    o_error_q016,
    output logic                                  o_error_last,
    output logic                                  o_done,
    output logic                                  o_busy
);

    logic softmax_start_valid;
    logic softmax_start_ready;
    logic softmax_config_error;
    logic softmax_probability_valid;
    logic [OUTPUT_INDEX_WIDTH-1:0] softmax_probability_index;
    logic [16:0] softmax_probability_q016;
    logic softmax_probability_last;
    logic softmax_done;
    logic softmax_busy;
    logic softmax_logit_ready;
    logic cfg_legal;
    logic row_fire;
    logic cfg_reject_q;
    logic [SAMPLE_ID_WIDTH-1:0] sample_id_q;
    logic [OUTPUT_INDEX_WIDTH-1:0] label_q;
    logic signed [17:0] probability_extended;

    always_comb begin
        cfg_legal = (i_output_count != '0) &&
            (i_output_count <= OUTPUT_COUNT_WIDTH'(MAX_OUTPUTS)) &&
            (OUTPUT_COUNT_WIDTH'({1'b0, i_label}) < i_output_count);
        o_row_ready = softmax_start_ready;
        row_fire = i_row_valid && o_row_ready;
        softmax_start_valid = row_fire && cfg_legal;
        o_config_error = cfg_reject_q || softmax_config_error;
        o_logit_ready = softmax_busy ? softmax_logit_ready : 1'b0;
        o_error_valid = softmax_probability_valid;
        o_error_sample_id = sample_id_q;
        o_error_output_index = softmax_probability_index;
        probability_extended = $signed({1'b0, softmax_probability_q016});
        o_error_q016 = probability_extended;
        if (softmax_probability_index == label_q)
            o_error_q016 = probability_extended - 18'sd65536;
        o_error_last = softmax_probability_last;
        o_done = softmax_done;
        o_busy = softmax_busy;
    end

    unified_tp_row_softmax_q6 #(
        .INPUT_WIDTH(16),
        .MAX_ITEMS(MAX_OUTPUTS),
        .ITEM_INDEX_WIDTH(OUTPUT_INDEX_WIDTH),
        .ITEM_COUNT_WIDTH(OUTPUT_COUNT_WIDTH)
    ) softmax_i (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start_valid(softmax_start_valid),
        .o_start_ready(softmax_start_ready),
        .i_row_length(i_output_count),
        .o_config_error(softmax_config_error),
        .i_value_valid(i_logit_valid),
        .o_value_ready(softmax_logit_ready),
        .i_value_q6(i_logit_q6),
        .o_probability_valid(softmax_probability_valid),
        .i_probability_ready(i_error_ready),
        .o_probability_index(softmax_probability_index),
        .o_probability_q016(softmax_probability_q016),
        .o_probability_last(softmax_probability_last),
        .o_done(softmax_done),
        .o_busy(softmax_busy)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            cfg_reject_q <= 1'b0;
            sample_id_q <= '0;
            label_q <= '0;
        end else begin
            cfg_reject_q <= row_fire && !cfg_legal;
            if (row_fire && cfg_legal) begin
                sample_id_q <= i_sample_id;
                label_q <= i_label;
            end
        end
    end

endmodule

`default_nettype wire
