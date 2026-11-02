`timescale 1ns/1ps
`default_nettype none

// Converts deterministic H40 observation replay into the existing FC and
// SRNN learning interfaces. FC uses the true pre-reset membrane value from
// the physical tile, so no post-reset reconstruction is required. SRNN emits
// the backbone spike directly as Q2.14 and avoids a scalar-vector-scalar copy.
module unified_fc_srnn_h40_tp_replay_router_v1 #(
    parameter int MAX_NEURONS = 640,
    parameter int SAMPLE_ID_WIDTH = 6,
    parameter int INDEX_WIDTH = $clog2(MAX_NEURONS),
    parameter int COUNT_WIDTH = $clog2(MAX_NEURONS + 1),
    parameter int FC_VMEM_WIDTH = 18,
    parameter int FC_VMEM_FRAC_BITS = 10,
    parameter int FC_LOGIT_FRAC_BITS = 6,
    parameter int MAX_FC_OUTPUTS = 16
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_abort,
    input  wire logic                         i_clear_error,

    input  wire logic                         i_cfg_valid,
    output logic                              o_cfg_ready,
    input  wire logic [1:0]                   i_cfg_mode,
    input  wire logic [SAMPLE_ID_WIDTH-1:0]   i_cfg_sample_id,
    input  wire logic [COUNT_WIDTH-1:0]       i_cfg_neuron_count,
    input  wire logic signed [FC_VMEM_WIDTH-1:0] i_cfg_fc_threshold,
    input  wire logic                         i_cfg_fc_output_layer,
    input  wire logic [3:0]                   i_cfg_fc_label,
    input  wire logic                         i_cfg_srnn_path,
    input  wire logic                         i_cfg_fc_logit_scale_enable,
    input  wire logic [23:0]                  i_cfg_fc_logit_scale_q16,
    input  wire logic                         i_cfg_fc_logit_accumulate,
    input  wire logic                         i_cfg_fc_sample_start,

    input  wire logic                         i_replay_valid,
    output logic                              o_replay_ready,
    input  wire logic [1:0]                   i_replay_mode,
    input  wire logic [COUNT_WIDTH-1:0]       i_replay_local_index,
    input  wire logic                         i_replay_spike,
    input  wire logic signed [31:0]           i_replay_pre_state,
    input  wire logic signed [31:0]           i_replay_trace_next,
    input  wire logic                         i_replay_last,

    output logic                              o_fc_student_trace_valid,
    input  wire logic                         i_fc_student_trace_ready,
    output logic [SAMPLE_ID_WIDTH-1:0]        o_fc_student_trace_sample,
    output logic [INDEX_WIDTH-1:0]            o_fc_student_trace_hidden,
    output logic [16:0]                       o_fc_student_trace_q016,

    output logic                              o_fc_surrogate_valid,
    input  wire logic                         i_fc_surrogate_ready,
    output logic [SAMPLE_ID_WIDTH-1:0]        o_fc_surrogate_sample,
    output logic [INDEX_WIDTH-1:0]            o_fc_surrogate_hidden,
    output logic [16:0]                       o_fc_surrogate_q016,

    output logic                              o_fc_logit_row_valid,
    input  wire logic                         i_fc_logit_row_ready,
    output logic [SAMPLE_ID_WIDTH-1:0]        o_fc_logit_sample,
    output logic [3:0]                        o_fc_logit_label,
    output logic                              o_fc_logit_valid,
    input  wire logic                         i_fc_logit_ready,
    output logic signed [15:0]                o_fc_logit_q6,
    output logic                              o_fc_logit_last,

    output logic                              o_srnn_backbone_valid,
    input  wire logic                         i_srnn_backbone_ready,
    output logic                              o_srnn_backbone_path,
    output logic [SAMPLE_ID_WIDTH-1:0]        o_srnn_backbone_sample_id,
    output logic [INDEX_WIDTH-1:0]            o_srnn_backbone_id,
    output logic signed [31:0]                o_srnn_backbone_q14,
    output logic                              o_srnn_backbone_last,

    output logic                              o_done,
    output logic                              o_busy,
    output logic                              o_idle,
    output logic                              o_sticky_error,
    output logic [3:0]                        o_error_code
);
    localparam logic [1:0] MODE_FC = 2'd0;
    localparam logic [1:0] MODE_SRNN = 2'd1;
    localparam logic [3:0] ERROR_NONE = 4'h0;
    localparam logic [3:0] ERROR_CONFIG = 4'h1;
    localparam logic [3:0] ERROR_METADATA = 4'h2;
    localparam logic [3:0] ERROR_FC_NUMERIC = 4'h3;
    localparam int SURROGATE_INDEX_SHIFT = FC_VMEM_FRAC_BITS - 4;
    localparam int LOGIT_SHIFT = FC_VMEM_FRAC_BITS - FC_LOGIT_FRAC_BITS;
    localparam int LOGIT_RAW_WIDTH = 24;
    localparam int LOGIT_PRODUCT_WIDTH = LOGIT_RAW_WIDTH + 25;
    localparam int FC_SAMPLES = 1 << SAMPLE_ID_WIDTH;
    localparam int FC_ACCUM_DEPTH = FC_SAMPLES * MAX_FC_OUTPUTS;
    localparam int FC_ACCUM_ADDR_WIDTH = $clog2(FC_ACCUM_DEPTH);

    logic configured_q;
    logic [1:0] mode_q;
    logic [SAMPLE_ID_WIDTH-1:0] sample_id_q;
    logic [COUNT_WIDTH-1:0] neuron_count_q;
    logic signed [FC_VMEM_WIDTH-1:0] fc_threshold_q;
    logic fc_output_layer_q;
    logic [3:0] fc_label_q;
    logic fc_logit_header_sent_q;
    logic fc_logit_scale_enable_q;
    logic [23:0] fc_logit_scale_q16_q;
    logic fc_logit_pending_q;
    logic signed [15:0] fc_logit_q6_q;
    logic fc_logit_last_q;
    logic fc_logit_accumulate_q;
    logic fc_sample_start_q;
    logic [FC_SAMPLES-1:0] fc_sample_valid_q;
    logic [COUNT_WIDTH-1:0] fc_sample_count_q [0:FC_SAMPLES-1];
    logic signed [LOGIT_RAW_WIDTH-1:0] fc_accum_mem [0:FC_ACCUM_DEPTH-1];
    logic signed [LOGIT_RAW_WIDTH-1:0] fc_accum_read_q;
    logic signed [LOGIT_RAW_WIDTH-1:0] fc_accum_dot_q;
    logic [FC_ACCUM_ADDR_WIDTH-1:0] fc_accum_addr_q;
    logic fc_accum_last_q;
    logic fc_accum_read_pending_q;
    logic [FC_ACCUM_ADDR_WIDTH-1:0] fc_accum_read_addr;
    logic signed [LOGIT_RAW_WIDTH-1:0] fc_accum_sum;
    logic fc_accum_capture;
    logic fc_accum_advance;
    logic srnn_path_q;
    logic [COUNT_WIDTH-1:0] expected_index_q;
    logic final_seen_q;

    logic fc_pending_q;
    logic fc_trace_pending_q;
    logic fc_surrogate_pending_q;
    logic [INDEX_WIDTH-1:0] fc_hidden_q;
    logic [16:0] fc_trace_q;
    logic [16:0] fc_surrogate_q;

    logic cfg_legal;
    logic cfg_logit_scale_enable;
    logic cfg_logit_accumulate;
    logic cfg_sample_start;
    logic cfg_fire;
    logic expected_last;
    logic metadata_legal;
    logic fc_numeric_legal;
    logic replay_legal;
    logic replay_fire;
    logic fc_trace_fire;
    logic fc_surrogate_fire;
    logic fc_logit_row_fire;
    logic fc_logit_fire;
    logic srnn_fire;
    logic signed [FC_VMEM_WIDTH:0] surrogate_delta;
    logic [FC_VMEM_WIDTH:0] surrogate_magnitude;
    logic [FC_VMEM_WIDTH:0] surrogate_index_rounded;
    logic [5:0] surrogate_index;

    function automatic logic [16:0] surrogate_lut_q016(
        input logic [5:0] index
    );
        begin
            case (index)
                6'd0:  surrogate_lut_q016 = 17'd65536;
                6'd1:  surrogate_lut_q016 = 17'd63103;
                6'd2:  surrogate_lut_q016 = 17'd56780;
                6'd3:  surrogate_lut_q016 = 17'd48654;
                6'd4:  surrogate_lut_q016 = 17'd40533;
                6'd5:  surrogate_lut_q016 = 17'd33372;
                6'd6:  surrogate_lut_q016 = 17'd27445;
                6'd7:  surrogate_lut_q016 = 17'd22684;
                6'd8:  surrogate_lut_q016 = 17'd18901;
                6'd9:  surrogate_lut_q016 = 17'd15896;
                6'd10: surrogate_lut_q016 = 17'd13498;
                6'd11: surrogate_lut_q016 = 17'd11569;
                6'd12: surrogate_lut_q016 = 17'd10003;
                6'd13: surrogate_lut_q016 = 17'd8720;
                6'd14: surrogate_lut_q016 = 17'd7659;
                6'd15: surrogate_lut_q016 = 17'd6774;
                6'd16: surrogate_lut_q016 = 17'd6029;
                6'd17: surrogate_lut_q016 = 17'd5398;
                6'd18: surrogate_lut_q016 = 17'd4858;
                6'd19: surrogate_lut_q016 = 17'd4393;
                6'd20: surrogate_lut_q016 = 17'd3991;
                6'd21: surrogate_lut_q016 = 17'd3640;
                6'd22: surrogate_lut_q016 = 17'd3334;
                6'd23: surrogate_lut_q016 = 17'd3063;
                6'd24: surrogate_lut_q016 = 17'd2824;
                6'd25: surrogate_lut_q016 = 17'd2611;
                6'd26: surrogate_lut_q016 = 17'd2422;
                6'd27: surrogate_lut_q016 = 17'd2252;
                6'd28: surrogate_lut_q016 = 17'd2099;
                6'd29: surrogate_lut_q016 = 17'd1961;
                6'd30: surrogate_lut_q016 = 17'd1836;
                6'd31: surrogate_lut_q016 = 17'd1722;
                6'd32: surrogate_lut_q016 = 17'd1619;
                6'd33: surrogate_lut_q016 = 17'd1525;
                6'd34: surrogate_lut_q016 = 17'd1438;
                6'd35: surrogate_lut_q016 = 17'd1359;
                6'd36: surrogate_lut_q016 = 17'd1286;
                6'd37: surrogate_lut_q016 = 17'd1219;
                6'd38: surrogate_lut_q016 = 17'd1156;
                6'd39: surrogate_lut_q016 = 17'd1099;
                6'd40: surrogate_lut_q016 = 17'd1045;
                6'd41: surrogate_lut_q016 = 17'd996;
                6'd42: surrogate_lut_q016 = 17'd950;
                6'd43: surrogate_lut_q016 = 17'd907;
                6'd44: surrogate_lut_q016 = 17'd866;
                6'd45: surrogate_lut_q016 = 17'd829;
                6'd46: surrogate_lut_q016 = 17'd794;
                6'd47: surrogate_lut_q016 = 17'd761;
                6'd48: surrogate_lut_q016 = 17'd730;
                6'd49: surrogate_lut_q016 = 17'd700;
                6'd50: surrogate_lut_q016 = 17'd673;
                6'd51: surrogate_lut_q016 = 17'd647;
                6'd52: surrogate_lut_q016 = 17'd623;
                6'd53: surrogate_lut_q016 = 17'd600;
                6'd54: surrogate_lut_q016 = 17'd578;
                6'd55: surrogate_lut_q016 = 17'd557;
                6'd56: surrogate_lut_q016 = 17'd538;
                6'd57: surrogate_lut_q016 = 17'd519;
                6'd58: surrogate_lut_q016 = 17'd501;
                6'd59: surrogate_lut_q016 = 17'd485;
                6'd60: surrogate_lut_q016 = 17'd469;
                6'd61: surrogate_lut_q016 = 17'd454;
                6'd62: surrogate_lut_q016 = 17'd439;
                default: surrogate_lut_q016 = 17'd426;
            endcase
        end
    endfunction

    // Same signed-magnitude RNE/sat16 rule as the FC student numeric adapter.
    // Convert either a sign-extended step dot or a wrapped raw24 cumulative logit.
    function automatic logic signed [15:0] scale_logit_q6(
        input logic signed [LOGIT_RAW_WIDTH-1:0] raw_value,
        input logic [23:0] scale_value
    );
        logic signed [LOGIT_PRODUCT_WIDTH-1:0] product;
        logic [LOGIT_PRODUCT_WIDTH-1:0] magnitude;
        logic [LOGIT_PRODUCT_WIDTH-1:0] integer_part;
        logic [15:0] remainder;
        logic round_increment;
        logic [LOGIT_PRODUCT_WIDTH:0] rounded_magnitude;
        logic signed [LOGIT_PRODUCT_WIDTH+1:0] rounded_signed;
        begin
            product = raw_value * $signed({1'b0, scale_value});
            magnitude = product[LOGIT_PRODUCT_WIDTH-1] ?
                $unsigned(-product) : $unsigned(product);
            integer_part = magnitude >> 16;
            remainder = magnitude[15:0];
            round_increment = (remainder > 16'h8000) ||
                ((remainder == 16'h8000) && integer_part[0]);
            rounded_magnitude = {1'b0, integer_part} +
                {{LOGIT_PRODUCT_WIDTH{1'b0}}, round_increment};
            rounded_signed = product[LOGIT_PRODUCT_WIDTH-1] ?
                -$signed({1'b0, rounded_magnitude}) :
                $signed({1'b0, rounded_magnitude});
            if (rounded_signed > 32767)
                scale_logit_q6 = 16'sh7fff;
            else if (rounded_signed < -32768)
                scale_logit_q6 = 16'sh8000;
            else
                scale_logit_q6 = rounded_signed[15:0];
        end
    endfunction

    always_comb begin
        // Omitted/X/Z optional enables retain the legacy conversion.
        cfg_logit_scale_enable = i_cfg_fc_logit_scale_enable === 1'b1;
        cfg_logit_accumulate = i_cfg_fc_logit_accumulate === 1'b1;
        cfg_sample_start = i_cfg_fc_sample_start === 1'b1;
        cfg_legal = (i_cfg_mode == MODE_FC || i_cfg_mode == MODE_SRNN) &&
            i_cfg_neuron_count != '0 &&
            integer'(i_cfg_neuron_count) <= MAX_NEURONS &&
            (!i_cfg_fc_output_layer ||
             (i_cfg_mode == MODE_FC &&
              integer'(i_cfg_neuron_count) <= MAX_FC_OUTPUTS)) &&
            (!cfg_logit_scale_enable ||
             (i_cfg_mode == MODE_FC && i_cfg_fc_output_layer &&
              i_cfg_fc_logit_scale_q16 != 24'd0)) &&
            (!cfg_logit_accumulate ||
             (i_cfg_mode == MODE_FC && i_cfg_fc_output_layer &&
              cfg_logit_scale_enable &&
              (cfg_sample_start ||
               (fc_sample_valid_q[i_cfg_sample_id] &&
                fc_sample_count_q[i_cfg_sample_id] == i_cfg_neuron_count))));
        o_cfg_ready = !configured_q && !fc_pending_q &&
            !fc_logit_pending_q && !fc_accum_read_pending_q &&
            !o_sticky_error && !i_abort && !i_clear_error;
        cfg_fire = i_cfg_valid && o_cfg_ready;

        expected_last = i_replay_local_index + 1'b1 == neuron_count_q;
        metadata_legal = i_replay_mode == mode_q &&
            i_replay_local_index == expected_index_q &&
            i_replay_last == expected_last;
        fc_numeric_legal =
            i_replay_pre_state[31:FC_VMEM_WIDTH] ==
                {(32-FC_VMEM_WIDTH){i_replay_pre_state[FC_VMEM_WIDTH-1]}} &&
            i_replay_trace_next[31:16] == '0;
        replay_legal = metadata_legal &&
            (mode_q != MODE_FC || fc_numeric_legal);

        o_replay_ready = 1'b0;
        if (configured_q && !o_sticky_error && !i_abort &&
            !i_clear_error && !final_seen_q) begin
            if (i_replay_valid && !replay_legal)
                o_replay_ready = 1'b1;
            else if (mode_q == MODE_FC && fc_output_layer_q)
                o_replay_ready = fc_logit_header_sent_q &&
                    (fc_logit_accumulate_q ?
                     (!fc_accum_read_pending_q && !fc_logit_pending_q) :
                     fc_logit_scale_enable_q ?
                     (!fc_logit_pending_q || i_fc_logit_ready) :
                     i_fc_logit_ready);
            else if (mode_q == MODE_FC)
                o_replay_ready = !fc_pending_q;
            else
                o_replay_ready = i_srnn_backbone_ready;
        end
        replay_fire = i_replay_valid && o_replay_ready;
        fc_accum_capture = replay_fire && replay_legal &&
            mode_q == MODE_FC && fc_output_layer_q && fc_logit_accumulate_q;
        fc_accum_read_addr = FC_ACCUM_ADDR_WIDTH'(
            integer'(sample_id_q) * MAX_FC_OUTPUTS + integer'(i_replay_local_index));
        // Assignment truncates the raw sum to signed24 before scaling.
        fc_accum_sum = (fc_sample_start_q ? 24'sd0 : fc_accum_read_q) +
            fc_accum_dot_q;
        fc_accum_advance = fc_accum_read_pending_q &&
            (!fc_logit_pending_q || i_fc_logit_ready) &&
            !(replay_fire && !replay_legal) && !i_abort && !i_clear_error;

        surrogate_delta =
            $signed({i_replay_pre_state[FC_VMEM_WIDTH-1],
                     i_replay_pre_state[FC_VMEM_WIDTH-1:0]}) -
            $signed({fc_threshold_q[FC_VMEM_WIDTH-1], fc_threshold_q});
        surrogate_magnitude = surrogate_delta[FC_VMEM_WIDTH] ?
            $unsigned(-surrogate_delta) : $unsigned(surrogate_delta);
        surrogate_index_rounded =
            (surrogate_magnitude + (1 << (SURROGATE_INDEX_SHIFT-1))) >>
            SURROGATE_INDEX_SHIFT;
        surrogate_index = surrogate_index_rounded > 63 ? 6'd63 :
            surrogate_index_rounded[5:0];

        o_fc_student_trace_valid = fc_pending_q && fc_trace_pending_q &&
            !i_abort && !i_clear_error;
        o_fc_student_trace_sample = sample_id_q;
        o_fc_student_trace_hidden = fc_hidden_q;
        o_fc_student_trace_q016 = fc_trace_q;
        fc_trace_fire = o_fc_student_trace_valid &&
            i_fc_student_trace_ready;

        o_fc_surrogate_valid = fc_pending_q && fc_surrogate_pending_q &&
            !i_abort && !i_clear_error;
        o_fc_surrogate_sample = sample_id_q;
        o_fc_surrogate_hidden = fc_hidden_q;
        o_fc_surrogate_q016 = fc_surrogate_q;
        fc_surrogate_fire = o_fc_surrogate_valid && i_fc_surrogate_ready;

        o_fc_logit_row_valid = configured_q && mode_q == MODE_FC &&
            fc_output_layer_q && i_replay_valid && replay_legal &&
            !fc_logit_header_sent_q && !o_sticky_error && !i_abort &&
            !i_clear_error;
        o_fc_logit_sample = sample_id_q;
        o_fc_logit_label = fc_label_q;
        fc_logit_row_fire = o_fc_logit_row_valid && i_fc_logit_row_ready;
        o_fc_logit_valid = configured_q && mode_q == MODE_FC &&
            fc_output_layer_q && fc_logit_header_sent_q &&
            !o_sticky_error && !i_abort && !i_clear_error &&
            (fc_logit_scale_enable_q ? fc_logit_pending_q :
             (i_replay_valid && replay_legal));
        o_fc_logit_q6 = fc_logit_scale_enable_q ? fc_logit_q6_q :
            16'($signed(i_replay_pre_state) >>> LOGIT_SHIFT);
        o_fc_logit_last = o_fc_logit_valid &&
            (fc_logit_scale_enable_q ? fc_logit_last_q : i_replay_last);
        fc_logit_fire = o_fc_logit_valid && i_fc_logit_ready;

        o_srnn_backbone_valid = configured_q && mode_q == MODE_SRNN &&
            i_replay_valid && replay_legal && !i_abort && !i_clear_error;
        o_srnn_backbone_path = srnn_path_q;
        o_srnn_backbone_sample_id = sample_id_q;
        o_srnn_backbone_id = i_replay_local_index[INDEX_WIDTH-1:0];
        o_srnn_backbone_q14 = i_replay_spike ? 32'sd16384 : 32'sd0;
        o_srnn_backbone_last = o_srnn_backbone_valid && i_replay_last;
        srnn_fire = o_srnn_backbone_valid && i_srnn_backbone_ready;

        o_busy = configured_q || fc_pending_q || fc_logit_pending_q ||
            fc_accum_read_pending_q;
        o_idle = !o_busy;
    end

    // No bulk RAM reset. Row validity protects unread/partially written state.
    // Accumulating inputs serialize the read, write/scale and output phases.
    always_ff @(posedge i_clk) begin
        if (fc_accum_capture) begin
            fc_accum_read_q <= fc_accum_mem[fc_accum_read_addr];
            fc_accum_dot_q <= LOGIT_RAW_WIDTH'(
                $signed(i_replay_pre_state[FC_VMEM_WIDTH-1:0]));
            fc_accum_addr_q <= fc_accum_read_addr;
            fc_accum_last_q <= i_replay_last;
        end else if (fc_accum_advance) begin
            fc_accum_mem[fc_accum_addr_q] <= fc_accum_sum;
        end
        if (cfg_fire && cfg_legal && cfg_logit_accumulate)
            fc_sample_count_q[i_cfg_sample_id] <= i_cfg_neuron_count;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            configured_q <= 1'b0;
            mode_q <= MODE_FC;
            sample_id_q <= '0;
            neuron_count_q <= '0;
            fc_threshold_q <= '0;
            fc_output_layer_q <= 1'b0;
            fc_label_q <= '0;
            fc_logit_header_sent_q <= 1'b0;
            fc_logit_scale_enable_q <= 1'b0;
            fc_logit_scale_q16_q <= '0;
            fc_logit_pending_q <= 1'b0;
            fc_logit_q6_q <= '0;
            fc_logit_last_q <= 1'b0;
            fc_logit_accumulate_q <= 1'b0;
            fc_sample_start_q <= 1'b0;
            fc_sample_valid_q <= '0;
            fc_accum_read_pending_q <= 1'b0;
            srnn_path_q <= 1'b0;
            expected_index_q <= '0;
            final_seen_q <= 1'b0;
            fc_pending_q <= 1'b0;
            fc_trace_pending_q <= 1'b0;
            fc_surrogate_pending_q <= 1'b0;
            fc_hidden_q <= '0;
            fc_trace_q <= '0;
            fc_surrogate_q <= '0;
            o_done <= 1'b0;
            o_sticky_error <= 1'b0;
            o_error_code <= ERROR_NONE;
        end else if (i_abort || i_clear_error) begin
            configured_q <= 1'b0;
            expected_index_q <= '0;
            final_seen_q <= 1'b0;
            fc_pending_q <= 1'b0;
            fc_trace_pending_q <= 1'b0;
            fc_surrogate_pending_q <= 1'b0;
            fc_logit_header_sent_q <= 1'b0;
            fc_logit_pending_q <= 1'b0;
            fc_logit_last_q <= 1'b0;
            fc_accum_read_pending_q <= 1'b0;
            if (configured_q && fc_logit_accumulate_q)
                fc_sample_valid_q[sample_id_q] <= 1'b0;
            o_done <= 1'b0;
            o_sticky_error <= 1'b0;
            o_error_code <= ERROR_NONE;
        end else begin
            o_done <= 1'b0;

            if (cfg_fire) begin
                if (!cfg_legal) begin
                    o_sticky_error <= 1'b1;
                    o_error_code <= ERROR_CONFIG;
                end else begin
                    configured_q <= 1'b1;
                    mode_q <= i_cfg_mode;
                    sample_id_q <= i_cfg_sample_id;
                    neuron_count_q <= i_cfg_neuron_count;
                    fc_threshold_q <= i_cfg_fc_threshold;
                    fc_output_layer_q <= i_cfg_fc_output_layer;
                    fc_label_q <= i_cfg_fc_label;
                    fc_logit_header_sent_q <= 1'b0;
                    fc_logit_scale_enable_q <= cfg_logit_scale_enable;
                    fc_logit_scale_q16_q <= i_cfg_fc_logit_scale_q16;
                    fc_logit_accumulate_q <= cfg_logit_accumulate;
                    fc_sample_start_q <= cfg_sample_start;
                    if (cfg_logit_accumulate)
                        fc_sample_valid_q[i_cfg_sample_id] <= 1'b0;
                    srnn_path_q <= i_cfg_srnn_path;
                    expected_index_q <= '0;
                    final_seen_q <= 1'b0;
                end
            end

            if (fc_accum_advance) begin
                fc_accum_read_pending_q <= 1'b0;
                fc_logit_pending_q <= 1'b1;
                fc_logit_q6_q <= scale_logit_q6(fc_accum_sum, fc_logit_scale_q16_q);
                fc_logit_last_q <= fc_accum_last_q;
            end

            if (fc_logit_fire && fc_logit_scale_enable_q) begin
                fc_logit_pending_q <= 1'b0;
                if (fc_logit_last_q) begin
                    configured_q <= 1'b0;
                    final_seen_q <= 1'b0;
                    fc_logit_header_sent_q <= 1'b0;
                    o_done <= 1'b1;
                    if (fc_logit_accumulate_q)
                        fc_sample_valid_q[sample_id_q] <= 1'b1;
                end
            end

            if (replay_fire && !replay_legal) begin
                configured_q <= 1'b0;
                final_seen_q <= 1'b0;
                fc_logit_pending_q <= 1'b0;
                fc_logit_header_sent_q <= 1'b0;
                fc_logit_last_q <= 1'b0;
                fc_accum_read_pending_q <= 1'b0;
                o_sticky_error <= 1'b1;
                o_error_code <= metadata_legal ?
                    ERROR_FC_NUMERIC : ERROR_METADATA;
            end else if (replay_fire && mode_q == MODE_FC &&
                         fc_output_layer_q) begin
                expected_index_q <= expected_index_q + 1'b1;
                if (fc_logit_scale_enable_q) begin
                    if (fc_logit_accumulate_q) begin
                        fc_accum_read_pending_q <= 1'b1;
                    end else begin
                        // Elastic one-entry output register: replace only on drain.
                        fc_logit_pending_q <= 1'b1;
                        fc_logit_q6_q <= scale_logit_q6(LOGIT_RAW_WIDTH'(
                            $signed(i_replay_pre_state[FC_VMEM_WIDTH-1:0])),
                            fc_logit_scale_q16_q);
                        fc_logit_last_q <= i_replay_last;
                    end
                    if (i_replay_last)
                        final_seen_q <= 1'b1;
                end else if (i_replay_last) begin
                    configured_q <= 1'b0;
                    fc_logit_header_sent_q <= 1'b0;
                    o_done <= 1'b1;
                end
            end else if (replay_fire && mode_q == MODE_FC) begin
                fc_pending_q <= 1'b1;
                fc_trace_pending_q <= 1'b1;
                fc_surrogate_pending_q <= 1'b1;
                fc_hidden_q <= i_replay_local_index[INDEX_WIDTH-1:0];
                fc_trace_q <= {1'b0, i_replay_trace_next[15:0]};
                fc_surrogate_q <= surrogate_lut_q016(surrogate_index);
                expected_index_q <= expected_index_q + 1'b1;
                if (i_replay_last)
                    final_seen_q <= 1'b1;
            end else if (srnn_fire) begin
                expected_index_q <= expected_index_q + 1'b1;
                if (i_replay_last) begin
                    configured_q <= 1'b0;
                    final_seen_q <= 1'b0;
                    o_done <= 1'b1;
                end
            end

            if (fc_logit_row_fire)
                fc_logit_header_sent_q <= 1'b1;

            if (fc_pending_q) begin
                if (fc_trace_fire)
                    fc_trace_pending_q <= 1'b0;
                if (fc_surrogate_fire)
                    fc_surrogate_pending_q <= 1'b0;
                if ((!fc_trace_pending_q || fc_trace_fire) &&
                    (!fc_surrogate_pending_q || fc_surrogate_fire))
                    fc_pending_q <= 1'b0;
            end

            if (configured_q && mode_q == MODE_FC && !fc_output_layer_q &&
                final_seen_q && !fc_pending_q) begin
                configured_q <= 1'b0;
                final_seen_q <= 1'b0;
                o_done <= 1'b1;
            end
        end
    end

    initial begin
        if (MAX_NEURONS < 1 || FC_VMEM_WIDTH >= 32 ||
            FC_VMEM_FRAC_BITS <= 4 || FC_LOGIT_FRAC_BITS < 0 ||
            FC_LOGIT_FRAC_BITS > FC_VMEM_FRAC_BITS || MAX_FC_OUTPUTS < 2)
            $error("FC/SRNN H40 replay router has invalid geometry");
    end
endmodule

`default_nettype wire
