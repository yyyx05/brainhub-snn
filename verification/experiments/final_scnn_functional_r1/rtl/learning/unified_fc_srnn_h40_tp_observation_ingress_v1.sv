`timescale 1ns/1ps
`default_nettype none

// Shared FC/SRNN learning ingress. Physical completion records are captured
// once, replayed in neuron order, then interpreted by the selected topology.
// Configuration is committed atomically to both stages.
module unified_fc_srnn_h40_tp_observation_ingress_v1 #(
    parameter int MAX_NEURONS = 640,
    parameter int SAMPLE_ID_WIDTH = 6,
    parameter int NEURON_ID_WIDTH = $clog2(MAX_NEURONS),
    parameter int COUNT_WIDTH = $clog2(MAX_NEURONS + 1),
    parameter int FC_VMEM_WIDTH = 18,
    parameter int FC_VMEM_FRAC_BITS = 10,
    parameter int FC_LOGIT_FRAC_BITS = 6,
    parameter int MAX_FC_OUTPUTS = 16
) (
    input  wire logic                            i_clk,
    input  wire logic                            i_rst_n,
    input  wire logic                            i_abort,
    input  wire logic                            i_clear_error,

    input  wire logic                            i_cfg_valid,
    output logic                                 o_cfg_ready,
    input  wire logic [1:0]                      i_cfg_mode,
    input  wire logic [51:0]                     i_cfg_transaction_tag,
    input  wire logic [SAMPLE_ID_WIDTH-1:0]      i_cfg_sample_id,
    input  wire logic [NEURON_ID_WIDTH-1:0]      i_cfg_neuron_base,
    input  wire logic [COUNT_WIDTH-1:0]          i_cfg_neuron_count,
    input  wire logic signed [FC_VMEM_WIDTH-1:0] i_cfg_fc_threshold,
    input  wire logic                            i_cfg_fc_output_layer,
    input  wire logic [3:0]                      i_cfg_fc_label,
    input  wire logic                            i_cfg_fc_logit_scale_enable,
    input  wire logic [23:0]                     i_cfg_fc_logit_scale_q16,
    input  wire logic                            i_cfg_fc_logit_accumulate,
    input  wire logic                            i_cfg_fc_sample_start,
    input  wire logic                            i_cfg_srnn_path,

    input  wire logic                            i_scalar_valid,
    output logic                                 o_scalar_ready,
    input  wire logic [1:0]                      i_scalar_mode,
    input  wire logic [51:0]                     i_scalar_transaction_tag,
    input  wire logic [NEURON_ID_WIDTH-1:0]      i_scalar_neuron_id,
    input  wire logic                            i_scalar_spike,
    input  wire logic signed [31:0]              i_scalar_pre_state,
    input  wire logic signed [31:0]              i_scalar_trace_next,

    output logic                                 o_fc_student_trace_valid,
    input  wire logic                            i_fc_student_trace_ready,
    output logic [SAMPLE_ID_WIDTH-1:0]           o_fc_student_trace_sample,
    output logic [NEURON_ID_WIDTH-1:0]           o_fc_student_trace_hidden,
    output logic [16:0]                          o_fc_student_trace_q016,

    output logic                                 o_fc_surrogate_valid,
    input  wire logic                            i_fc_surrogate_ready,
    output logic [SAMPLE_ID_WIDTH-1:0]           o_fc_surrogate_sample,
    output logic [NEURON_ID_WIDTH-1:0]           o_fc_surrogate_hidden,
    output logic [16:0]                          o_fc_surrogate_q016,

    output logic                                 o_fc_logit_row_valid,
    input  wire logic                            i_fc_logit_row_ready,
    output logic [SAMPLE_ID_WIDTH-1:0]           o_fc_logit_sample,
    output logic [3:0]                           o_fc_logit_label,
    output logic                                 o_fc_logit_valid,
    input  wire logic                            i_fc_logit_ready,
    output logic signed [15:0]                   o_fc_logit_q6,
    output logic                                 o_fc_logit_last,

    output logic                                 o_srnn_backbone_valid,
    input  wire logic                            i_srnn_backbone_ready,
    output logic                                 o_srnn_backbone_path,
    output logic [SAMPLE_ID_WIDTH-1:0]           o_srnn_backbone_sample_id,
    output logic [NEURON_ID_WIDTH-1:0]           o_srnn_backbone_id,
    output logic signed [31:0]                   o_srnn_backbone_q14,
    output logic                                 o_srnn_backbone_last,

    output logic                                 o_capture_done,
    output logic                                 o_done,
    output logic                                 o_busy,
    output logic                                 o_idle,
    output logic                                 o_sticky_error,
    output logic                                 o_error_source,
    output logic [3:0]                           o_error_code
);
    localparam logic ERROR_SOURCE_REORDER = 1'b0;
    localparam logic ERROR_SOURCE_ROUTER = 1'b1;

    logic reorder_cfg_ready;
    logic router_cfg_ready;
    logic cfg_fire;
    logic sub_abort;
    logic sub_clear_error;

    logic replay_valid;
    logic replay_ready;
    logic [1:0] replay_mode;
    logic [51:0] replay_transaction_tag;
    logic [COUNT_WIDTH-1:0] replay_local_index;
    logic [NEURON_ID_WIDTH-1:0] replay_neuron_id;
    logic replay_spike;
    logic signed [31:0] replay_pre_state;
    logic signed [31:0] replay_trace_next;
    logic replay_last;
    logic reorder_replay_done;
    logic reorder_busy;
    logic reorder_idle;
    logic reorder_error;
    logic [3:0] reorder_error_code;
    logic router_done;
    logic router_busy;
    logic router_idle;
    logic router_error;
    logic [3:0] router_error_code;

    logic sticky_error_q;
    logic error_source_q;
    logic [3:0] error_code_q;
    logic fault_flush_q;

    always_comb begin
        o_cfg_ready = reorder_cfg_ready && router_cfg_ready &&
            !sticky_error_q;
        cfg_fire = i_cfg_valid && o_cfg_ready;

        sub_abort = i_abort || fault_flush_q;
        sub_clear_error = i_clear_error ||
            (sticky_error_q && reorder_idle && router_idle);

        o_busy = reorder_busy || router_busy || fault_flush_q;
        o_idle = !o_busy;
        o_done = router_done;
        o_sticky_error = sticky_error_q;
        o_error_source = error_source_q;
        o_error_code = error_code_q;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            sticky_error_q <= 1'b0;
            error_source_q <= ERROR_SOURCE_REORDER;
            error_code_q <= '0;
            fault_flush_q <= 1'b0;
        end else if (i_abort) begin
            sticky_error_q <= 1'b0;
            error_source_q <= ERROR_SOURCE_REORDER;
            error_code_q <= '0;
            fault_flush_q <= 1'b0;
        end else begin
            fault_flush_q <= 1'b0;

            if (!sticky_error_q && (reorder_error || router_error)) begin
                sticky_error_q <= 1'b1;
                fault_flush_q <= 1'b1;
                if (reorder_error) begin
                    error_source_q <= ERROR_SOURCE_REORDER;
                    error_code_q <= reorder_error_code;
                end else begin
                    error_source_q <= ERROR_SOURCE_ROUTER;
                    error_code_q <= router_error_code;
                end
            end

            if (i_clear_error && reorder_idle && router_idle &&
                !reorder_error && !router_error) begin
                sticky_error_q <= 1'b0;
                error_source_q <= ERROR_SOURCE_REORDER;
                error_code_q <= '0;
            end
        end
    end

    unified_fc_srnn_h40_tp_observation_reorder_v1 #(
        .MAX_NEURONS(MAX_NEURONS),
        .NEURON_ID_WIDTH(NEURON_ID_WIDTH),
        .NEURON_COUNT_WIDTH(COUNT_WIDTH)
    ) u_reorder (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_abort(sub_abort),
        .i_clear_error(sub_clear_error),
        .i_cfg_valid(cfg_fire),
        .o_cfg_ready(reorder_cfg_ready),
        .i_cfg_mode(i_cfg_mode),
        .i_cfg_transaction_tag(i_cfg_transaction_tag),
        .i_cfg_neuron_base(i_cfg_neuron_base),
        .i_cfg_neuron_count(i_cfg_neuron_count),
        .i_scalar_valid(i_scalar_valid),
        .o_scalar_ready(o_scalar_ready),
        .i_scalar_mode(i_scalar_mode),
        .i_scalar_transaction_tag(i_scalar_transaction_tag),
        .i_scalar_neuron_id(i_scalar_neuron_id),
        .i_scalar_spike(i_scalar_spike),
        .i_scalar_pre_state(i_scalar_pre_state),
        .i_scalar_trace_next(i_scalar_trace_next),
        .o_replay_valid(replay_valid),
        .i_replay_ready(replay_ready),
        .o_replay_mode(replay_mode),
        .o_replay_transaction_tag(replay_transaction_tag),
        .o_replay_local_index(replay_local_index),
        .o_replay_neuron_id(replay_neuron_id),
        .o_replay_spike(replay_spike),
        .o_replay_pre_state(replay_pre_state),
        .o_replay_trace_next(replay_trace_next),
        .o_replay_last(replay_last),
        .o_capture_done(o_capture_done),
        .o_replay_done(reorder_replay_done),
        .o_busy(reorder_busy),
        .o_idle(reorder_idle),
        .o_sticky_error(reorder_error),
        .o_error_code(reorder_error_code)
    );

    unified_fc_srnn_h40_tp_replay_router_v1 #(
        .MAX_NEURONS(MAX_NEURONS),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .INDEX_WIDTH(NEURON_ID_WIDTH),
        .COUNT_WIDTH(COUNT_WIDTH),
        .FC_VMEM_WIDTH(FC_VMEM_WIDTH),
        .FC_VMEM_FRAC_BITS(FC_VMEM_FRAC_BITS),
        .FC_LOGIT_FRAC_BITS(FC_LOGIT_FRAC_BITS),
        .MAX_FC_OUTPUTS(MAX_FC_OUTPUTS)
    ) u_router (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_abort(sub_abort),
        .i_clear_error(sub_clear_error),
        .i_cfg_valid(cfg_fire),
        .o_cfg_ready(router_cfg_ready),
        .i_cfg_mode(i_cfg_mode),
        .i_cfg_sample_id(i_cfg_sample_id),
        .i_cfg_neuron_count(i_cfg_neuron_count),
        .i_cfg_fc_threshold(i_cfg_fc_threshold),
        .i_cfg_fc_output_layer(i_cfg_fc_output_layer),
        .i_cfg_fc_label(i_cfg_fc_label),
        .i_cfg_fc_logit_scale_enable(i_cfg_fc_logit_scale_enable),
        .i_cfg_fc_logit_scale_q16(i_cfg_fc_logit_scale_q16),
        .i_cfg_fc_logit_accumulate(i_cfg_fc_logit_accumulate),
        .i_cfg_fc_sample_start(i_cfg_fc_sample_start),
        .i_cfg_srnn_path(i_cfg_srnn_path),
        .i_replay_valid(replay_valid),
        .o_replay_ready(replay_ready),
        .i_replay_mode(replay_mode),
        .i_replay_local_index(replay_local_index),
        .i_replay_spike(replay_spike),
        .i_replay_pre_state(replay_pre_state),
        .i_replay_trace_next(replay_trace_next),
        .i_replay_last(replay_last),
        .o_fc_student_trace_valid(o_fc_student_trace_valid),
        .i_fc_student_trace_ready(i_fc_student_trace_ready),
        .o_fc_student_trace_sample(o_fc_student_trace_sample),
        .o_fc_student_trace_hidden(o_fc_student_trace_hidden),
        .o_fc_student_trace_q016(o_fc_student_trace_q016),
        .o_fc_surrogate_valid(o_fc_surrogate_valid),
        .i_fc_surrogate_ready(i_fc_surrogate_ready),
        .o_fc_surrogate_sample(o_fc_surrogate_sample),
        .o_fc_surrogate_hidden(o_fc_surrogate_hidden),
        .o_fc_surrogate_q016(o_fc_surrogate_q016),
        .o_fc_logit_row_valid(o_fc_logit_row_valid),
        .i_fc_logit_row_ready(i_fc_logit_row_ready),
        .o_fc_logit_sample(o_fc_logit_sample),
        .o_fc_logit_label(o_fc_logit_label),
        .o_fc_logit_valid(o_fc_logit_valid),
        .i_fc_logit_ready(i_fc_logit_ready),
        .o_fc_logit_q6(o_fc_logit_q6),
        .o_fc_logit_last(o_fc_logit_last),
        .o_srnn_backbone_valid(o_srnn_backbone_valid),
        .i_srnn_backbone_ready(i_srnn_backbone_ready),
        .o_srnn_backbone_path(o_srnn_backbone_path),
        .o_srnn_backbone_sample_id(o_srnn_backbone_sample_id),
        .o_srnn_backbone_id(o_srnn_backbone_id),
        .o_srnn_backbone_q14(o_srnn_backbone_q14),
        .o_srnn_backbone_last(o_srnn_backbone_last),
        .o_done(router_done),
        .o_busy(router_busy),
        .o_idle(router_idle),
        .o_sticky_error(router_error),
        .o_error_code(router_error_code)
    );

    logic unused_replay_metadata;
    assign unused_replay_metadata = ^{replay_transaction_tag,
        replay_neuron_id, reorder_replay_done};

    initial begin
        if (MAX_NEURONS < 2)
            $error("FC/SRNN observation ingress has invalid geometry");
    end
endmodule

`default_nettype wire
