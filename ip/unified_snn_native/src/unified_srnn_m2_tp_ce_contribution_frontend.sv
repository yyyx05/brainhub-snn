`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off DECLFILENAME */
// One-write/one-read observation RAM.  The explicit XPM path prevents the
// mutually exclusive FSM reads from being implemented as replicated LUTRAM.
module unified_srnn_m2_observation_ram #(
    parameter int DATA_WIDTH = 32,
    parameter int DEPTH = 1024,
    parameter int ADDR_WIDTH = $clog2(DEPTH)
) (
    input  logic                      i_clk,
    input  logic                      i_write_enable,
    input  logic [ADDR_WIDTH-1:0]     i_write_addr,
    input  logic [DATA_WIDTH-1:0]     i_write_data,
    input  logic                      i_read_enable,
    input  logic [ADDR_WIDTH-1:0]     i_read_addr,
    output logic [DATA_WIDTH-1:0]     o_read_data
);

`ifdef UNIFIED_USE_XPM_MEMORY
    xpm_memory_sdpram #(
        .MEMORY_SIZE(DEPTH * DATA_WIDTH),
        .MEMORY_PRIMITIVE("block"),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("none"),
        .MEMORY_INIT_PARAM(""),
        .USE_MEM_INIT(0),
        .USE_MEM_INIT_MMI(0),
        .WAKEUP_TIME("disable_sleep"),
        .AUTO_SLEEP_TIME(0),
        .MESSAGE_CONTROL(0),
        .USE_EMBEDDED_CONSTRAINT(0),
        .MEMORY_OPTIMIZATION("true"),
        .CASCADE_HEIGHT(0),
        .SIM_ASSERT_CHK(0),
        .WRITE_PROTECT(1),
        .IGNORE_INIT_SYNTH(0),
        .WRITE_DATA_WIDTH_A(DATA_WIDTH),
        .BYTE_WRITE_WIDTH_A(DATA_WIDTH),
        .ADDR_WIDTH_A(ADDR_WIDTH),
        .RST_MODE_A("SYNC"),
        .READ_DATA_WIDTH_B(DATA_WIDTH),
        .ADDR_WIDTH_B(ADDR_WIDTH),
        .READ_RESET_VALUE_B("0"),
        .READ_LATENCY_B(1),
        .WRITE_MODE_B("read_first"),
        .RST_MODE_B("SYNC")
    ) u_memory (
        .sleep(1'b0),
        .clka(i_clk),
        .ena(i_write_enable),
        .wea(i_write_enable),
        .addra(i_write_addr),
        .dina(i_write_data),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0),
        .clkb(i_clk),
        .rstb(1'b0),
        .enb(i_read_enable),
        .regceb(1'b1),
        .addrb(i_read_addr),
        .doutb(o_read_data),
        .sbiterrb(),
        .dbiterrb()
    );
`else
    (* ram_style = "block" *)
    logic [DATA_WIDTH-1:0] memory [0:DEPTH-1];

    always_ff @(posedge i_clk) begin
        if (i_write_enable)
            memory[i_write_addr] <= i_write_data;
        if (i_read_enable)
            o_read_data <= memory[i_read_addr];
    end
`endif

endmodule
/* verilator lint_on DECLFILENAME */

// Buffers one M2 timestep observation window, reuses one scalar metric and one
// Q12/Q24 softmax, then emits the exact contribution order expected by the
// existing WH1 and Wout reducers.  It owns no parameter or optimizer state.
module unified_srnn_m2_tp_ce_contribution_frontend #(
    parameter bit ENABLE_WH1_LEARNING = 1'b1,
    parameter int BATCH_MAX = 64,
    parameter int HIDDEN_MAX = 256,
    parameter int BACKBONE_MAX = 450,
    parameter int CLASS_MAX = 20,
    parameter int SAMPLE_ID_WIDTH = 8,
    parameter int HIDDEN_INDEX_WIDTH = 9,
    parameter int BACKBONE_INDEX_WIDTH = 10,
    parameter int CLASS_INDEX_WIDTH = 5,
    parameter int PARAM_ADDR_WIDTH = 24
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_cfg_batch_count,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]             i_cfg_hidden_count,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_cfg_backbone_count,
    input  logic [CLASS_INDEX_WIDTH-1:0]              i_cfg_class_count,
    output logic                                      o_cfg_error,

    input  logic                                      i_start_window,
    output logic                                      o_start_ready,
    input  logic                                      i_abort_window,

    input  logic                                      i_hidden_valid,
    output logic                                      o_hidden_ready,
    input  logic                                      i_hidden_path,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_hidden_sample_id,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]             i_hidden_id,
    input  logic signed [31:0]                        i_hidden_membrane_pre_q14,
    input  logic signed [31:0]                        i_hidden_trace_q14,
    input  logic                                      i_hidden_last,

    input  logic                                      i_target_trace_valid,
    output logic                                      o_target_trace_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_target_trace_sample_id,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_target_trace_backbone_id,
    input  logic signed [31:0]                        i_target_trace_q14,
    input  logic                                      i_target_trace_last,

    input  logic                                      i_logit_valid,
    output logic                                      o_logit_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_logit_sample_id,
    input  logic [CLASS_INDEX_WIDTH-1:0]              i_logit_class_id,
    input  logic signed [63:0]                        i_logit_q28,
    input  logic                                      i_logit_last,

    input  logic                                      i_label_valid,
    output logic                                      o_label_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_label_sample_id,
    input  logic [CLASS_INDEX_WIDTH-1:0]              i_label_class_id,
    input  logic                                      i_label_last,

    output logic                                      o_wh1_contribution_valid,
    input  logic                                      i_wh1_contribution_ready,
    output logic                                      o_wh1_contribution_path,
    output logic [SAMPLE_ID_WIDTH-1:0]                o_wh1_contribution_sample_id,
    output logic [HIDDEN_INDEX_WIDTH-1:0]             o_wh1_contribution_hidden_id,
    output logic [SAMPLE_ID_WIDTH-1:0]                o_wh1_contribution_column_id,
    output logic signed [63:0]                        o_wh1_error_q40,
    output logic signed [31:0]                        o_wh1_opposite_trace_q14,
    output logic [24:0]                               o_wh1_surrogate_uq24,
    output logic                                      o_wh1_contribution_last,

    output logic                                      o_wout_contribution_valid,
    input  logic                                      i_wout_contribution_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_wout_contribution_addr,
    output logic [SAMPLE_ID_WIDTH-1:0]                o_wout_contribution_sample_id,
    output logic signed [63:0]                        o_wout_error_q40,
    output logic signed [31:0]                        o_wout_hidden_trace_q14,
    output logic                                      o_wout_contribution_last,

    output logic                                      o_busy,
    output logic                                      o_done,
    output logic                                      o_aborted,
    output logic                                      o_protocol_error,
    output logic                                      o_overflow
);

    localparam int HIDDEN_DEPTH = BATCH_MAX * HIDDEN_MAX;
    localparam int TARGET_DEPTH = BATCH_MAX * BACKBONE_MAX;
    localparam int LOGIT_DEPTH = BATCH_MAX * CLASS_MAX;
    localparam int PAIR_DEPTH = BATCH_MAX * BATCH_MAX;
    localparam int ROW_ITEM_MAX =
        (BATCH_MAX > CLASS_MAX) ? BATCH_MAX : CLASS_MAX;
    localparam int ROW_ITEM_WIDTH =
        (ROW_ITEM_MAX <= 2) ? 1 : $clog2(ROW_ITEM_MAX);
    localparam int ROW_COUNT_WIDTH = $clog2(ROW_ITEM_MAX + 1);
    localparam int HIDDEN_ADDR_WIDTH = $clog2(HIDDEN_DEPTH);
    localparam int TARGET_ADDR_WIDTH = $clog2(TARGET_DEPTH);
    localparam int LOGIT_ADDR_WIDTH = $clog2(LOGIT_DEPTH);
    localparam int PAIR_ADDR_WIDTH = $clog2(PAIR_DEPTH);
    localparam int SQRT_ROOT_WIDTH = 64;
    localparam int SQRT_ITER_WIDTH = 6;
    localparam logic [23:0] PI2_Q20 = 24'd10349030;

    typedef enum logic [4:0] {
        STATE_IDLE,
        STATE_COLLECT,
        STATE_ROW_START,
        STATE_METRIC_READ_A,
        STATE_METRIC_READ_B,
        STATE_METRIC_ACCUMULATE,
        STATE_METRIC_FINALIZE,
        STATE_SQRT_STEP,
        STATE_OUTPUT_LOGIT_READ,
        STATE_OUTPUT_LOGIT_STORE,
        STATE_SOFTMAX_START,
        STATE_SOFTMAX_LOAD,
        STATE_SOFTMAX_CAPTURE,
        STATE_ROW_ADVANCE,
        STATE_WH_SURROGATE_READ,
        STATE_WH_SURROGATE_SQUARE,
        STATE_WH_SURROGATE_PRODUCT,
        STATE_WH_SURROGATE_PREP,
        STATE_WH_SURROGATE_DIVIDE,
        STATE_WH_READ,
        STATE_WH_EMIT,
        STATE_WOUT_READ,
        STATE_WOUT_EMIT,
        STATE_ERROR
    } state_t;

    typedef enum logic [1:0] {
        PHASE_STUDENT,
        PHASE_TARGET,
        PHASE_OUTPUT
    } phase_t;

    (* ram_style = "distributed" *) logic [CLASS_INDEX_WIDTH-1:0]
        label_mem [0:BATCH_MAX-1];
    (* ram_style = "distributed" *) logic signed [31:0]
        row_metric_mem [0:ROW_ITEM_MAX-1];
    (* ram_style = "block" *) logic signed [63:0]
        pair_error_mem [0:PAIR_DEPTH-1];
    (* ram_style = "block" *) logic signed [63:0]
        output_error_mem [0:LOGIT_DEPTH-1];

    state_t state_q;
    phase_t phase_q;
    logic configured_q;
    logic [SAMPLE_ID_WIDTH-1:0] batch_count_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] hidden_count_q;
    logic [BACKBONE_INDEX_WIDTH-1:0] backbone_count_q;
    logic [CLASS_INDEX_WIDTH-1:0] class_count_q;

    logic hidden_complete_q;
    logic target_trace_complete_q;
    logic logit_complete_q;
    logic label_complete_q;
    logic hidden_expected_path_q;
    logic [SAMPLE_ID_WIDTH-1:0] hidden_expected_sample_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] hidden_expected_id_q;
    logic [SAMPLE_ID_WIDTH-1:0] target_expected_sample_q;
    logic [BACKBONE_INDEX_WIDTH-1:0] target_expected_id_q;
    logic [SAMPLE_ID_WIDTH-1:0] logit_expected_sample_q;
    logic [CLASS_INDEX_WIDTH-1:0] logit_expected_class_q;
    logic [SAMPLE_ID_WIDTH-1:0] label_expected_sample_q;

    logic [SAMPLE_ID_WIDTH-1:0] row_q;
    logic [SAMPLE_ID_WIDTH-1:0] column_q;
    logic [BACKBONE_INDEX_WIDTH-1:0] vector_index_q;
    logic signed [127:0] metric_accumulator_q;
    logic signed [31:0] metric_operand_a_q;
    logic [127:0] sqrt_radicand_q;
    logic [SQRT_ROOT_WIDTH+1:0] sqrt_remainder_q;
    logic [SQRT_ROOT_WIDTH-1:0] sqrt_root_q;
    logic [SQRT_ITER_WIDTH-1:0] sqrt_iteration_q;

    logic [ROW_ITEM_WIDTH-1:0] softmax_load_index_q;
    logic softmax_start_valid;
    logic softmax_start_ready;
    logic [ROW_COUNT_WIDTH-1:0] softmax_row_length;
    logic softmax_config_error;
    logic softmax_value_valid;
    logic softmax_value_ready;
    logic signed [31:0] softmax_value_q12;
    logic softmax_probability_valid;
    logic softmax_probability_ready;
    logic [ROW_ITEM_WIDTH-1:0] softmax_probability_index;
    logic [24:0] softmax_probability_q024;
    logic softmax_probability_last;
    logic softmax_done;
    logic softmax_busy;
    logic softmax_overflow;

    logic wh_path_q;
    logic [SAMPLE_ID_WIDTH-1:0] wh_sample_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] wh_hidden_q;
    logic [SAMPLE_ID_WIDTH-1:0] wh_column_q;
    logic [24:0] wh_surrogate_q;
    logic signed [63:0] wh_error_q;
    logic [127:0] surrogate_denominator_q;
    logic [127:0] surrogate_remainder_q;
    logic [52:0] surrogate_quotient_q;
    logic [5:0] surrogate_divide_bit_q;

    logic [PARAM_ADDR_WIDTH-1:0] wout_addr_q;
    logic [CLASS_INDEX_WIDTH-1:0] wout_class_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] wout_hidden_q;
    logic [SAMPLE_ID_WIDTH-1:0] wout_sample_q;
    logic signed [63:0] wout_error_q;

    logic cfg_legal;
    logic hidden_fire;
    logic target_trace_fire;
    logic logit_fire;
    logic label_fire;
    logic hidden_expected_last;
    logic target_expected_last;
    logic logit_expected_last;
    logic label_expected_last;
    logic hidden_legal;
    logic target_trace_legal;
    logic logit_legal;
    logic label_legal;
    logic collection_complete;
    logic [HIDDEN_ADDR_WIDTH-1:0] hidden_write_addr;
    logic [TARGET_ADDR_WIDTH-1:0] target_write_addr;
    logic [LOGIT_ADDR_WIDTH-1:0] logit_write_addr;

    logic collect_protocol_error;
    logic actual_hidden_write_enable;
    logic target_hidden_write_enable;
    logic target_backbone_write_enable;
    logic logit_write_enable;
    logic actual_hidden_read_enable;
    logic target_hidden_read_enable;
    logic actual_membrane_read_enable;
    logic target_membrane_read_enable;
    logic target_backbone_read_enable;
    logic logit_read_enable;
    logic [HIDDEN_ADDR_WIDTH-1:0] actual_hidden_read_addr;
    logic [HIDDEN_ADDR_WIDTH-1:0] target_hidden_read_addr;
    logic [HIDDEN_ADDR_WIDTH-1:0] actual_membrane_read_addr;
    logic [HIDDEN_ADDR_WIDTH-1:0] target_membrane_read_addr;
    logic [TARGET_ADDR_WIDTH-1:0] target_backbone_read_addr;
    logic [LOGIT_ADDR_WIDTH-1:0] logit_read_addr;
    logic [31:0] actual_hidden_read_data;
    logic [31:0] target_hidden_read_data;
    logic [31:0] actual_membrane_read_data;
    logic [31:0] target_membrane_read_data;
    logic [31:0] target_backbone_read_data;
    logic [63:0] logit_read_data;
    logic signed [31:0] metric_operand_b_active;
    logic signed [31:0] membrane_read_active;

    logic final_row;
    logic final_column;
    logic final_vector;
    logic signed [63:0] metric_product;
    logic signed [32:0] metric_difference;
    logic signed [65:0] metric_square_signed;
    logic signed [127:0] metric_contribution;
    logic signed [127:0] metric_accumulator_next;
    logic signed [127:0] student_metric_rounded;
    logic metric_overflow;

    logic [SQRT_ROOT_WIDTH+3:0] sqrt_remainder_shifted;
    logic [SQRT_ROOT_WIDTH+3:0] sqrt_trial;
    logic [SQRT_ROOT_WIDTH+1:0] sqrt_remainder_next;
    logic [SQRT_ROOT_WIDTH-1:0] sqrt_root_next;
    logic sqrt_last;
    logic [SQRT_ROOT_WIDTH-1:0] distance_q12_magnitude;
    logic signed [127:0] distance_metric_wide;
    logic signed [127:0] output_logit_metric_rounded;

    logic softmax_start_fire;
    logic softmax_value_fire;
    logic softmax_probability_fire;
    logic [PAIR_ADDR_WIDTH-1:0] pair_probability_addr;
    logic [LOGIT_ADDR_WIDTH-1:0] output_probability_addr;
    logic signed [63:0] probability_q40;
    logic signed [63:0] output_probability_signed;
    logic signed [63:0] output_label_q24;

    logic signed [63:0] surrogate_delta;
    logic signed [127:0] surrogate_square;
    logic [127:0] surrogate_square_q;
    logic [151:0] surrogate_product;
    logic [151:0] surrogate_product_q;
    logic signed [191:0] surrogate_product_extended;
    logic signed [191:0] surrogate_term_rounded;
    logic [127:0] surrogate_denominator_next;
    logic surrogate_denominator_overflow;
    logic [128:0] surrogate_remainder_shifted;
    logic [128:0] surrogate_denominator_extended;
    logic [127:0] surrogate_remainder_next;
    logic [52:0] surrogate_quotient_next;
    logic [128:0] surrogate_twice_remainder;
    logic surrogate_round_increment;
    logic [53:0] surrogate_rounded_quotient;
    logic [24:0] surrogate_factor_next;

    logic [PAIR_ADDR_WIDTH-1:0] wh_pair_addr;
    logic wh_fire;
    logic wout_fire;
    logic wh_last;
    logic wout_last;

    function automatic logic signed [127:0] round_shift_even_128(
        input logic signed [127:0] value,
        input integer shift
    );
        logic negative;
        logic [127:0] magnitude;
        logic [127:0] quotient;
        logic [127:0] mask;
        logic [127:0] remainder;
        logic [127:0] half;
        begin
            negative = value[127];
            magnitude = negative ? -value : value;
            quotient = magnitude >> shift;
            mask = ({128{1'b1}} >> (128-shift));
            remainder = magnitude & mask;
            half = 128'(1) << (shift-1);
            if ((remainder > half) ||
                ((remainder == half) && quotient[0]))
                quotient = quotient + 1'b1;
            round_shift_even_128 = negative ? -$signed(quotient) :
                $signed(quotient);
        end
    endfunction

    function automatic logic signed [191:0] round_shift_even_192(
        input logic signed [191:0] value,
        input integer shift
    );
        logic negative;
        logic [191:0] magnitude;
        logic [191:0] quotient;
        logic [191:0] mask;
        logic [191:0] remainder;
        logic [191:0] half;
        begin
            negative = value[191];
            magnitude = negative ? -value : value;
            quotient = magnitude >> shift;
            mask = ({192{1'b1}} >> (192-shift));
            remainder = magnitude & mask;
            half = 192'(1) << (shift-1);
            if ((remainder > half) ||
                ((remainder == half) && quotient[0]))
                quotient = quotient + 1'b1;
            round_shift_even_192 = negative ? -$signed(quotient) :
                $signed(quotient);
        end
    endfunction

    always_comb begin
        cfg_legal = (i_cfg_batch_count != '0) &&
            (i_cfg_batch_count <= SAMPLE_ID_WIDTH'(BATCH_MAX)) &&
            (i_cfg_hidden_count != '0) &&
            (i_cfg_hidden_count <= HIDDEN_INDEX_WIDTH'(HIDDEN_MAX)) &&
            (i_cfg_backbone_count != '0) &&
            (i_cfg_backbone_count <= BACKBONE_INDEX_WIDTH'(BACKBONE_MAX)) &&
            (i_cfg_class_count != '0) &&
            (i_cfg_class_count <= CLASS_INDEX_WIDTH'(CLASS_MAX));
        o_cfg_ready = state_q == STATE_IDLE;
        o_start_ready = configured_q && (state_q == STATE_IDLE) &&
            !i_cfg_valid;
        o_busy = state_q != STATE_IDLE;

        hidden_expected_last = (!ENABLE_WH1_LEARNING ||
                hidden_expected_path_q) &&
            (hidden_expected_sample_q == (batch_count_q - 1'b1)) &&
            (hidden_expected_id_q == (hidden_count_q - 1'b1));
        target_expected_last =
            (target_expected_sample_q == (batch_count_q - 1'b1)) &&
            (target_expected_id_q == (backbone_count_q - 1'b1));
        logit_expected_last =
            (logit_expected_sample_q == (batch_count_q - 1'b1)) &&
            (logit_expected_class_q == (class_count_q - 1'b1));
        label_expected_last =
            label_expected_sample_q == (batch_count_q - 1'b1);
        hidden_legal = (i_hidden_path == hidden_expected_path_q) &&
            (i_hidden_sample_id == hidden_expected_sample_q) &&
            (i_hidden_id == hidden_expected_id_q) &&
            (i_hidden_last == hidden_expected_last);
        target_trace_legal =
            (i_target_trace_sample_id == target_expected_sample_q) &&
            (i_target_trace_backbone_id == target_expected_id_q) &&
            (i_target_trace_last == target_expected_last);
        logit_legal = (i_logit_sample_id == logit_expected_sample_q) &&
            (i_logit_class_id == logit_expected_class_q) &&
            (i_logit_last == logit_expected_last);
        label_legal = (i_label_sample_id == label_expected_sample_q) &&
            (i_label_class_id < class_count_q) &&
            (i_label_last == label_expected_last);

        o_hidden_ready = (state_q == STATE_COLLECT) && !hidden_complete_q;
        o_target_trace_ready = ENABLE_WH1_LEARNING &&
            (state_q == STATE_COLLECT) &&
            !target_trace_complete_q;
        o_logit_ready = (state_q == STATE_COLLECT) && !logit_complete_q;
        o_label_ready = (state_q == STATE_COLLECT) && !label_complete_q;
        hidden_fire = i_hidden_valid && o_hidden_ready;
        target_trace_fire = i_target_trace_valid && o_target_trace_ready;
        logit_fire = i_logit_valid && o_logit_ready;
        label_fire = i_label_valid && o_label_ready;
        collection_complete =
            (hidden_complete_q || (hidden_fire && hidden_expected_last)) &&
            (!ENABLE_WH1_LEARNING || target_trace_complete_q ||
                (target_trace_fire && target_expected_last)) &&
            (logit_complete_q || (logit_fire && logit_expected_last)) &&
            (label_complete_q || (label_fire && label_expected_last));
        collect_protocol_error =
            (hidden_fire && !hidden_legal) ||
            (target_trace_fire && !target_trace_legal) ||
            (logit_fire && !logit_legal) ||
            (label_fire && !label_legal);

        hidden_write_addr = HIDDEN_ADDR_WIDTH'(
            int'(i_hidden_sample_id) * HIDDEN_MAX + int'(i_hidden_id));
        target_write_addr = TARGET_ADDR_WIDTH'(
            int'(i_target_trace_sample_id) * BACKBONE_MAX +
            int'(i_target_trace_backbone_id));
        logit_write_addr = LOGIT_ADDR_WIDTH'(
            int'(i_logit_sample_id) * CLASS_MAX + int'(i_logit_class_id));

        actual_hidden_write_enable = hidden_fire && hidden_legal &&
            !hidden_expected_path_q && !collect_protocol_error &&
            !i_abort_window;
        target_hidden_write_enable = ENABLE_WH1_LEARNING &&
            hidden_fire && hidden_legal &&
            hidden_expected_path_q && !collect_protocol_error &&
            !i_abort_window;
        target_backbone_write_enable = ENABLE_WH1_LEARNING &&
            target_trace_fire &&
            target_trace_legal && !collect_protocol_error && !i_abort_window;
        logit_write_enable = logit_fire && logit_legal &&
            !collect_protocol_error && !i_abort_window;

        actual_hidden_read_enable = 1'b0;
        target_hidden_read_enable = 1'b0;
        actual_membrane_read_enable = 1'b0;
        target_membrane_read_enable = 1'b0;
        target_backbone_read_enable = 1'b0;
        logit_read_enable = 1'b0;
        actual_hidden_read_addr = '0;
        target_hidden_read_addr = '0;
        actual_membrane_read_addr = '0;
        target_membrane_read_addr = '0;
        target_backbone_read_addr = '0;
        logit_read_addr = '0;

        unique case (state_q)
            STATE_METRIC_READ_A: begin
                if (phase_q == PHASE_STUDENT) begin
                    actual_hidden_read_enable = 1'b1;
                    actual_hidden_read_addr = HIDDEN_ADDR_WIDTH'(
                        int'(row_q) * HIDDEN_MAX + int'(vector_index_q));
                end else begin
                    target_backbone_read_enable = 1'b1;
                    target_backbone_read_addr = TARGET_ADDR_WIDTH'(
                        int'(row_q) * BACKBONE_MAX + int'(vector_index_q));
                end
            end
            STATE_METRIC_READ_B: begin
                if (phase_q == PHASE_STUDENT) begin
                    target_hidden_read_enable = 1'b1;
                    target_hidden_read_addr = HIDDEN_ADDR_WIDTH'(
                        int'(column_q) * HIDDEN_MAX + int'(vector_index_q));
                end else begin
                    target_backbone_read_enable = 1'b1;
                    target_backbone_read_addr = TARGET_ADDR_WIDTH'(
                        int'(column_q) * BACKBONE_MAX + int'(vector_index_q));
                end
            end
            STATE_OUTPUT_LOGIT_READ: begin
                logit_read_enable = 1'b1;
                logit_read_addr = LOGIT_ADDR_WIDTH'(
                    int'(row_q) * CLASS_MAX + int'(column_q));
            end
            STATE_WH_SURROGATE_READ: begin
                if (wh_path_q) begin
                    target_membrane_read_enable = 1'b1;
                    target_membrane_read_addr = HIDDEN_ADDR_WIDTH'(
                        int'(wh_sample_q) * HIDDEN_MAX + int'(wh_hidden_q));
                end else begin
                    actual_membrane_read_enable = 1'b1;
                    actual_membrane_read_addr = HIDDEN_ADDR_WIDTH'(
                        int'(wh_sample_q) * HIDDEN_MAX + int'(wh_hidden_q));
                end
            end
            STATE_WH_READ: begin
                if (wh_path_q) begin
                    actual_hidden_read_enable = 1'b1;
                    actual_hidden_read_addr = HIDDEN_ADDR_WIDTH'(
                        int'(wh_column_q) * HIDDEN_MAX + int'(wh_hidden_q));
                end else begin
                    target_hidden_read_enable = 1'b1;
                    target_hidden_read_addr = HIDDEN_ADDR_WIDTH'(
                        int'(wh_column_q) * HIDDEN_MAX + int'(wh_hidden_q));
                end
            end
            STATE_WOUT_READ: begin
                actual_hidden_read_enable = 1'b1;
                actual_hidden_read_addr = HIDDEN_ADDR_WIDTH'(
                    int'(wout_sample_q) * HIDDEN_MAX + int'(wout_hidden_q));
            end
            default: begin
            end
        endcase

        metric_operand_b_active = phase_q == PHASE_STUDENT ?
            $signed(target_hidden_read_data) :
            $signed(target_backbone_read_data);
        membrane_read_active = wh_path_q ?
            $signed(target_membrane_read_data) :
            $signed(actual_membrane_read_data);

        final_row = row_q == (batch_count_q - 1'b1);
        final_column = column_q == (batch_count_q - 1'b1);
        final_vector = phase_q == PHASE_STUDENT ?
            (vector_index_q == BACKBONE_INDEX_WIDTH'(
                hidden_count_q - 1'b1)) :
            (vector_index_q == (backbone_count_q - 1'b1));
        metric_product = $signed(metric_operand_a_q) *
            $signed(metric_operand_b_active);
        metric_difference = $signed({metric_operand_a_q[31],
            metric_operand_a_q}) - $signed({metric_operand_b_active[31],
            metric_operand_b_active});
        metric_square_signed = metric_difference * metric_difference;
        if (phase_q == PHASE_STUDENT)
            metric_contribution = {{64{metric_product[63]}}, metric_product};
        else
            metric_contribution = {{62{1'b0}},
                metric_square_signed[65:0]};
        metric_accumulator_next = metric_accumulator_q + metric_contribution;
        student_metric_rounded = round_shift_even_128(
            metric_accumulator_q, 16);
        metric_overflow = student_metric_rounded[127:32] !=
            {96{student_metric_rounded[31]}};

        sqrt_remainder_shifted = {
            sqrt_remainder_q, sqrt_radicand_q[127 -: 2]};
        sqrt_trial = {{2{1'b0}}, sqrt_root_q, 2'b01};
        sqrt_root_next = sqrt_root_q << 1;
        if (sqrt_remainder_shifted >= sqrt_trial) begin
            sqrt_remainder_next = (SQRT_ROOT_WIDTH+2)'(
                sqrt_remainder_shifted - sqrt_trial);
            sqrt_root_next[0] = 1'b1;
        end else begin
            sqrt_remainder_next =
                sqrt_remainder_shifted[SQRT_ROOT_WIDTH+1:0];
        end
        sqrt_last = sqrt_iteration_q == SQRT_ITER_WIDTH'(63);
        distance_q12_magnitude = (sqrt_root_next + 64'd2) >> 2;
        distance_metric_wide = -$signed({64'd0,
            distance_q12_magnitude});
        output_logit_metric_rounded = round_shift_even_128(
            {{64{logit_read_data[63]}}, logit_read_data}, 16);

        softmax_start_valid = state_q == STATE_SOFTMAX_START;
        softmax_row_length = phase_q == PHASE_OUTPUT ?
            ROW_COUNT_WIDTH'(class_count_q) :
            ROW_COUNT_WIDTH'(batch_count_q);
        softmax_start_fire = softmax_start_valid && softmax_start_ready;
        softmax_value_valid = state_q == STATE_SOFTMAX_LOAD;
        softmax_value_q12 = row_metric_mem[softmax_load_index_q];
        softmax_value_fire = softmax_value_valid && softmax_value_ready;
        softmax_probability_ready = state_q == STATE_SOFTMAX_CAPTURE;
        softmax_probability_fire = softmax_probability_valid &&
            softmax_probability_ready;
        pair_probability_addr = PAIR_ADDR_WIDTH'(
            int'(row_q) * BATCH_MAX + int'(softmax_probability_index));
        output_probability_addr = LOGIT_ADDR_WIDTH'(
            int'(row_q) * CLASS_MAX + int'(softmax_probability_index));
        probability_q40 = $signed({29'd0,
            softmax_probability_q024, 10'd0});
        output_probability_signed = $signed({39'd0,
            softmax_probability_q024});
        output_label_q24 =
            (CLASS_INDEX_WIDTH'(softmax_probability_index) ==
                label_mem[int'(row_q)]) ?
            64'sd16777216 : 64'sd0;

        surrogate_delta = $signed({{32{membrane_read_active[31]}},
            membrane_read_active}) - 64'sd8192;
        surrogate_square = surrogate_delta * surrogate_delta;
        surrogate_product = surrogate_square_q * PI2_Q20;
        surrogate_product_extended = $signed({40'd0, surrogate_product_q});
        surrogate_term_rounded = round_shift_even_192(
            surrogate_product_extended, 20);
        surrogate_denominator_next =
            surrogate_term_rounded[127:0] + 128'(1 << 28);
        surrogate_denominator_overflow =
            surrogate_term_rounded[191:128] != 64'd0;

        surrogate_remainder_shifted = {surrogate_remainder_q,
            (surrogate_divide_bit_q == 6'd52)};
        surrogate_denominator_extended = {1'b0,
            surrogate_denominator_q};
        surrogate_quotient_next = surrogate_quotient_q;
        if (surrogate_remainder_shifted >=
            surrogate_denominator_extended) begin
            surrogate_remainder_next = 128'(
                surrogate_remainder_shifted -
                surrogate_denominator_extended);
            surrogate_quotient_next[surrogate_divide_bit_q] = 1'b1;
        end else begin
            surrogate_remainder_next =
                surrogate_remainder_shifted[127:0];
            surrogate_quotient_next[surrogate_divide_bit_q] = 1'b0;
        end
        surrogate_twice_remainder = {surrogate_remainder_next, 1'b0};
        surrogate_round_increment =
            (surrogate_twice_remainder > surrogate_denominator_extended) ||
            ((surrogate_twice_remainder == surrogate_denominator_extended) &&
                surrogate_quotient_next[0]);
        surrogate_rounded_quotient = {1'b0, surrogate_quotient_next} +
            surrogate_round_increment;
        surrogate_factor_next =
            (surrogate_rounded_quotient > 54'(1 << 24)) ?
            25'(1 << 24) : surrogate_rounded_quotient[24:0];

        wh_pair_addr = wh_path_q ?
            PAIR_ADDR_WIDTH'(int'(wh_column_q) * BATCH_MAX +
                int'(wh_sample_q)) :
            PAIR_ADDR_WIDTH'(int'(wh_sample_q) * BATCH_MAX +
                int'(wh_column_q));
        wh_last = wh_path_q &&
            (wh_sample_q == (batch_count_q - 1'b1)) &&
            (wh_hidden_q == (hidden_count_q - 1'b1)) &&
            (wh_column_q == (batch_count_q - 1'b1));
        o_wh1_contribution_valid = ENABLE_WH1_LEARNING &&
            (state_q == STATE_WH_EMIT);
        o_wh1_contribution_path = wh_path_q;
        o_wh1_contribution_sample_id = wh_sample_q;
        o_wh1_contribution_hidden_id = wh_hidden_q;
        o_wh1_contribution_column_id = wh_column_q;
        o_wh1_error_q40 = wh_error_q;
        o_wh1_opposite_trace_q14 = wh_path_q ?
            $signed(actual_hidden_read_data) :
            $signed(target_hidden_read_data);
        o_wh1_surrogate_uq24 = wh_surrogate_q;
        o_wh1_contribution_last = wh_last;
        wh_fire = o_wh1_contribution_valid && i_wh1_contribution_ready;

        wout_last =
            (wout_addr_q == PARAM_ADDR_WIDTH'(
                int'(class_count_q) * int'(hidden_count_q) - 1)) &&
            (wout_sample_q == (batch_count_q - 1'b1));
        o_wout_contribution_valid = state_q == STATE_WOUT_EMIT;
        o_wout_contribution_addr = wout_addr_q;
        o_wout_contribution_sample_id = wout_sample_q;
        o_wout_error_q40 = wout_error_q;
        o_wout_hidden_trace_q14 = $signed(actual_hidden_read_data);
        o_wout_contribution_last = wout_last;
        wout_fire = o_wout_contribution_valid && i_wout_contribution_ready;
    end

    unified_srnn_m2_observation_ram #(
        .DATA_WIDTH(32), .DEPTH(HIDDEN_DEPTH),
        .ADDR_WIDTH(HIDDEN_ADDR_WIDTH)
    ) u_actual_hidden_trace_mem (
        .i_clk(i_clk),
        .i_write_enable(actual_hidden_write_enable),
        .i_write_addr(hidden_write_addr),
        .i_write_data(i_hidden_trace_q14),
        .i_read_enable(actual_hidden_read_enable),
        .i_read_addr(actual_hidden_read_addr),
        .o_read_data(actual_hidden_read_data)
    );

    unified_srnn_m2_observation_ram #(
        .DATA_WIDTH(32), .DEPTH(HIDDEN_DEPTH),
        .ADDR_WIDTH(HIDDEN_ADDR_WIDTH)
    ) u_target_hidden_trace_mem (
        .i_clk(i_clk),
        .i_write_enable(target_hidden_write_enable),
        .i_write_addr(hidden_write_addr),
        .i_write_data(i_hidden_trace_q14),
        .i_read_enable(target_hidden_read_enable),
        .i_read_addr(target_hidden_read_addr),
        .o_read_data(target_hidden_read_data)
    );

    unified_srnn_m2_observation_ram #(
        .DATA_WIDTH(32), .DEPTH(HIDDEN_DEPTH),
        .ADDR_WIDTH(HIDDEN_ADDR_WIDTH)
    ) u_actual_membrane_pre_mem (
        .i_clk(i_clk),
        .i_write_enable(actual_hidden_write_enable),
        .i_write_addr(hidden_write_addr),
        .i_write_data(i_hidden_membrane_pre_q14),
        .i_read_enable(actual_membrane_read_enable),
        .i_read_addr(actual_membrane_read_addr),
        .o_read_data(actual_membrane_read_data)
    );

    unified_srnn_m2_observation_ram #(
        .DATA_WIDTH(32), .DEPTH(HIDDEN_DEPTH),
        .ADDR_WIDTH(HIDDEN_ADDR_WIDTH)
    ) u_target_membrane_pre_mem (
        .i_clk(i_clk),
        .i_write_enable(target_hidden_write_enable),
        .i_write_addr(hidden_write_addr),
        .i_write_data(i_hidden_membrane_pre_q14),
        .i_read_enable(target_membrane_read_enable),
        .i_read_addr(target_membrane_read_addr),
        .o_read_data(target_membrane_read_data)
    );

    unified_srnn_m2_observation_ram #(
        .DATA_WIDTH(32), .DEPTH(TARGET_DEPTH),
        .ADDR_WIDTH(TARGET_ADDR_WIDTH)
    ) u_target_backbone_trace_mem (
        .i_clk(i_clk),
        .i_write_enable(target_backbone_write_enable),
        .i_write_addr(target_write_addr),
        .i_write_data(i_target_trace_q14),
        .i_read_enable(target_backbone_read_enable),
        .i_read_addr(target_backbone_read_addr),
        .o_read_data(target_backbone_read_data)
    );

    unified_srnn_m2_observation_ram #(
        .DATA_WIDTH(64), .DEPTH(LOGIT_DEPTH),
        .ADDR_WIDTH(LOGIT_ADDR_WIDTH)
    ) u_logit_mem (
        .i_clk(i_clk),
        .i_write_enable(logit_write_enable),
        .i_write_addr(logit_write_addr),
        .i_write_data(i_logit_q28),
        .i_read_enable(logit_read_enable),
        .i_read_addr(logit_read_addr),
        .o_read_data(logit_read_data)
    );

    unified_tp_row_softmax_q12_q24 #(
        .MAX_ITEMS(ROW_ITEM_MAX),
        .ITEM_INDEX_WIDTH(ROW_ITEM_WIDTH),
        .ITEM_COUNT_WIDTH(ROW_COUNT_WIDTH)
    ) shared_softmax (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start_valid(softmax_start_valid),
        .o_start_ready(softmax_start_ready),
        .i_row_length(softmax_row_length),
        .o_config_error(softmax_config_error),
        .i_value_valid(softmax_value_valid),
        .o_value_ready(softmax_value_ready),
        .i_value_q12(softmax_value_q12),
        .o_probability_valid(softmax_probability_valid),
        .i_probability_ready(softmax_probability_ready),
        .o_probability_index(softmax_probability_index),
        .o_probability_q024(softmax_probability_q024),
        .o_probability_last(softmax_probability_last),
        .o_done(softmax_done),
        .o_busy(softmax_busy),
        .o_arithmetic_overflow(softmax_overflow)
    );

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            phase_q <= PHASE_STUDENT;
            configured_q <= 1'b0;
            batch_count_q <= '0;
            hidden_count_q <= '0;
            backbone_count_q <= '0;
            class_count_q <= '0;
            hidden_complete_q <= 1'b0;
            target_trace_complete_q <= 1'b0;
            logit_complete_q <= 1'b0;
            label_complete_q <= 1'b0;
            hidden_expected_path_q <= 1'b0;
            hidden_expected_sample_q <= '0;
            hidden_expected_id_q <= '0;
            target_expected_sample_q <= '0;
            target_expected_id_q <= '0;
            logit_expected_sample_q <= '0;
            logit_expected_class_q <= '0;
            label_expected_sample_q <= '0;
            row_q <= '0;
            column_q <= '0;
            vector_index_q <= '0;
            metric_accumulator_q <= '0;
            metric_operand_a_q <= '0;
            sqrt_radicand_q <= '0;
            sqrt_remainder_q <= '0;
            sqrt_root_q <= '0;
            sqrt_iteration_q <= '0;
            softmax_load_index_q <= '0;
            wh_path_q <= 1'b0;
            wh_sample_q <= '0;
            wh_hidden_q <= '0;
            wh_column_q <= '0;
            wh_surrogate_q <= '0;
            wh_error_q <= '0;
            surrogate_denominator_q <= '0;
            surrogate_square_q <= '0;
            surrogate_product_q <= '0;
            surrogate_remainder_q <= '0;
            surrogate_quotient_q <= '0;
            surrogate_divide_bit_q <= '0;
            wout_addr_q <= '0;
            wout_class_q <= '0;
            wout_hidden_q <= '0;
            wout_sample_q <= '0;
            wout_error_q <= '0;
            o_cfg_error <= 1'b0;
            o_done <= 1'b0;
            o_aborted <= 1'b0;
            o_protocol_error <= 1'b0;
            o_overflow <= 1'b0;
        end else begin
            o_cfg_error <= 1'b0;
            o_done <= 1'b0;
            o_aborted <= 1'b0;

            if (i_clear_error) begin
                o_protocol_error <= 1'b0;
                o_overflow <= 1'b0;
                if (state_q == STATE_ERROR)
                    state_q <= STATE_IDLE;
            end

            if (i_cfg_valid && o_cfg_ready) begin
                if (!cfg_legal) begin
                    configured_q <= 1'b0;
                    o_cfg_error <= 1'b1;
                end else begin
                    configured_q <= 1'b1;
                    batch_count_q <= i_cfg_batch_count;
                    hidden_count_q <= i_cfg_hidden_count;
                    backbone_count_q <= i_cfg_backbone_count;
                    class_count_q <= i_cfg_class_count;
                end
            end

            if (i_abort_window && (state_q != STATE_IDLE)) begin
                state_q <= STATE_IDLE;
                o_aborted <= 1'b1;
            end else begin
                unique case (state_q)
                    STATE_IDLE: begin
                        if (i_start_window && o_start_ready) begin
                            hidden_complete_q <= 1'b0;
                            target_trace_complete_q <= 1'b0;
                            logit_complete_q <= 1'b0;
                            label_complete_q <= 1'b0;
                            hidden_expected_path_q <= 1'b0;
                            hidden_expected_sample_q <= '0;
                            hidden_expected_id_q <= '0;
                            target_expected_sample_q <= '0;
                            target_expected_id_q <= '0;
                            logit_expected_sample_q <= '0;
                            logit_expected_class_q <= '0;
                            label_expected_sample_q <= '0;
                            o_protocol_error <= 1'b0;
                            o_overflow <= 1'b0;
                            state_q <= STATE_COLLECT;
                        end
                    end

                    STATE_COLLECT: begin
                        if ((hidden_fire && !hidden_legal) ||
                            (target_trace_fire && !target_trace_legal) ||
                            (logit_fire && !logit_legal) ||
                            (label_fire && !label_legal)) begin
                            o_protocol_error <= 1'b1;
                            state_q <= STATE_ERROR;
                        end else begin
                            if (hidden_fire) begin
                                if (hidden_expected_last) begin
                                    hidden_complete_q <= 1'b1;
                                end else if (hidden_expected_id_q ==
                                        (hidden_count_q - 1'b1)) begin
                                    hidden_expected_id_q <= '0;
                                    if (hidden_expected_sample_q ==
                                            (batch_count_q - 1'b1)) begin
                                        hidden_expected_sample_q <= '0;
                                        hidden_expected_path_q <= 1'b1;
                                    end else begin
                                        hidden_expected_sample_q <=
                                            hidden_expected_sample_q + 1'b1;
                                    end
                                end else begin
                                    hidden_expected_id_q <=
                                        hidden_expected_id_q + 1'b1;
                                end
                            end

                            if (target_trace_fire) begin
                                if (target_expected_last) begin
                                    target_trace_complete_q <= 1'b1;
                                end else if (target_expected_id_q ==
                                        (backbone_count_q - 1'b1)) begin
                                    target_expected_id_q <= '0;
                                    target_expected_sample_q <=
                                        target_expected_sample_q + 1'b1;
                                end else begin
                                    target_expected_id_q <=
                                        target_expected_id_q + 1'b1;
                                end
                            end

                            if (logit_fire) begin
                                if (logit_expected_last) begin
                                    logit_complete_q <= 1'b1;
                                end else if (logit_expected_class_q ==
                                        (class_count_q - 1'b1)) begin
                                    logit_expected_class_q <= '0;
                                    logit_expected_sample_q <=
                                        logit_expected_sample_q + 1'b1;
                                end else begin
                                    logit_expected_class_q <=
                                        logit_expected_class_q + 1'b1;
                                end
                            end

                            if (label_fire) begin
                                label_mem[int'(i_label_sample_id)] <=
                                    i_label_class_id;
                                if (label_expected_last)
                                    label_complete_q <= 1'b1;
                                else
                                    label_expected_sample_q <=
                                        label_expected_sample_q + 1'b1;
                            end

                            if (collection_complete) begin
                                phase_q <= ENABLE_WH1_LEARNING ?
                                    PHASE_STUDENT : PHASE_OUTPUT;
                                row_q <= '0;
                                column_q <= '0;
                                state_q <= STATE_ROW_START;
                            end
                        end
                    end

                    STATE_ROW_START: begin
                        column_q <= '0;
                        vector_index_q <= '0;
                        metric_accumulator_q <= '0;
                        if (phase_q == PHASE_OUTPUT)
                            state_q <= STATE_OUTPUT_LOGIT_READ;
                        else
                            state_q <= STATE_METRIC_READ_A;
                    end

                    STATE_METRIC_READ_A: begin
                        state_q <= STATE_METRIC_READ_B;
                    end

                    STATE_METRIC_READ_B: begin
                        metric_operand_a_q <= phase_q == PHASE_STUDENT ?
                            $signed(actual_hidden_read_data) :
                            $signed(target_backbone_read_data);
                        state_q <= STATE_METRIC_ACCUMULATE;
                    end

                    STATE_METRIC_ACCUMULATE: begin
                        if (final_vector) begin
                            metric_accumulator_q <= metric_accumulator_next;
                            state_q <= STATE_METRIC_FINALIZE;
                        end else begin
                            metric_accumulator_q <= metric_accumulator_next;
                            vector_index_q <= vector_index_q + 1'b1;
                            state_q <= STATE_METRIC_READ_A;
                        end
                    end

                    // Register the final multiply-accumulate result before the
                    // wide rounding/overflow or square-root preparation logic.
                    STATE_METRIC_FINALIZE: begin
                        if (phase_q == PHASE_STUDENT) begin
                                row_metric_mem[int'(column_q)] <=
                                    student_metric_rounded[31:0];
                                if (metric_overflow)
                                    o_overflow <= 1'b1;
                                if (final_column)
                                    state_q <= STATE_SOFTMAX_START;
                                else begin
                                    column_q <= column_q + 1'b1;
                                    vector_index_q <= '0;
                                    metric_accumulator_q <= '0;
                                    state_q <= STATE_METRIC_READ_A;
                                end
                        end else begin
                            sqrt_radicand_q <= metric_accumulator_q[127:0];
                            sqrt_remainder_q <= '0;
                            sqrt_root_q <= '0;
                            sqrt_iteration_q <= '0;
                            state_q <= STATE_SQRT_STEP;
                        end
                    end

                    STATE_SQRT_STEP: begin
                        sqrt_remainder_q <= sqrt_remainder_next;
                        sqrt_root_q <= sqrt_root_next;
                        sqrt_radicand_q <= sqrt_radicand_q << 2;
                        if (sqrt_last) begin
                            row_metric_mem[int'(column_q)] <=
                                distance_metric_wide[31:0];
                            if (distance_metric_wide[127:32] !=
                                {96{distance_metric_wide[31]}})
                                o_overflow <= 1'b1;
                            if (final_column)
                                state_q <= STATE_SOFTMAX_START;
                            else begin
                                column_q <= column_q + 1'b1;
                                vector_index_q <= '0;
                                metric_accumulator_q <= '0;
                                state_q <= STATE_METRIC_READ_A;
                            end
                        end else begin
                            sqrt_iteration_q <= sqrt_iteration_q + 1'b1;
                        end
                    end

                    STATE_OUTPUT_LOGIT_READ: begin
                        state_q <= STATE_OUTPUT_LOGIT_STORE;
                    end

                    STATE_OUTPUT_LOGIT_STORE: begin
                        row_metric_mem[int'(column_q)] <=
                            output_logit_metric_rounded[31:0];
                        if (column_q == SAMPLE_ID_WIDTH'(
                                class_count_q - 1'b1)) begin
                            state_q <= STATE_SOFTMAX_START;
                        end else begin
                            column_q <= column_q + 1'b1;
                            state_q <= STATE_OUTPUT_LOGIT_READ;
                        end
                    end

                    STATE_SOFTMAX_START: begin
                        if (softmax_start_fire) begin
                            softmax_load_index_q <= '0;
                            state_q <= STATE_SOFTMAX_LOAD;
                        end
                    end

                    STATE_SOFTMAX_LOAD: begin
                        if (softmax_value_fire) begin
                            if ((phase_q == PHASE_OUTPUT &&
                                    softmax_load_index_q ==
                                        ROW_ITEM_WIDTH'(
                                            class_count_q - 1'b1)) ||
                                (phase_q != PHASE_OUTPUT &&
                                    softmax_load_index_q ==
                                        ROW_ITEM_WIDTH'(
                                            batch_count_q - 1'b1))) begin
                                state_q <= STATE_SOFTMAX_CAPTURE;
                            end else begin
                                softmax_load_index_q <=
                                    softmax_load_index_q + 1'b1;
                            end
                        end
                    end

                    STATE_SOFTMAX_CAPTURE: begin
                        if (softmax_config_error || softmax_overflow) begin
                            if (softmax_overflow)
                                o_overflow <= 1'b1;
                            else
                                o_protocol_error <= 1'b1;
                            state_q <= STATE_ERROR;
                        end else if (softmax_probability_fire) begin
                            if (phase_q == PHASE_STUDENT) begin
                                pair_error_mem[pair_probability_addr] <=
                                    probability_q40;
                            end else if (phase_q == PHASE_TARGET) begin
                                pair_error_mem[pair_probability_addr] <=
                                    pair_error_mem[pair_probability_addr] -
                                    probability_q40;
                            end else begin
                                output_error_mem[output_probability_addr] <=
                                    (output_probability_signed -
                                        output_label_q24) <<< 16;
                            end
                            if (softmax_probability_last)
                                state_q <= STATE_ROW_ADVANCE;
                        end
                    end

                    STATE_ROW_ADVANCE: begin
                        if (final_row) begin
                            row_q <= '0;
                            if (phase_q == PHASE_STUDENT) begin
                                phase_q <= PHASE_TARGET;
                                state_q <= STATE_ROW_START;
                            end else if (phase_q == PHASE_TARGET) begin
                                phase_q <= PHASE_OUTPUT;
                                state_q <= STATE_ROW_START;
                            end else if (ENABLE_WH1_LEARNING) begin
                                wh_path_q <= 1'b0;
                                wh_sample_q <= '0;
                                wh_hidden_q <= '0;
                                wh_column_q <= '0;
                                state_q <= STATE_WH_SURROGATE_READ;
                            end else begin
                                wout_addr_q <= '0;
                                wout_class_q <= '0;
                                wout_hidden_q <= '0;
                                wout_sample_q <= '0;
                                state_q <= STATE_WOUT_READ;
                            end
                        end else begin
                            row_q <= row_q + 1'b1;
                            state_q <= STATE_ROW_START;
                        end
                    end

                    STATE_WH_SURROGATE_READ: begin
                        state_q <= STATE_WH_SURROGATE_SQUARE;
                    end

                    STATE_WH_SURROGATE_SQUARE: begin
                        surrogate_square_q <= surrogate_square;
                        state_q <= STATE_WH_SURROGATE_PRODUCT;
                    end

                    STATE_WH_SURROGATE_PRODUCT: begin
                        surrogate_product_q <= surrogate_product;
                        state_q <= STATE_WH_SURROGATE_PREP;
                    end

                    STATE_WH_SURROGATE_PREP: begin
                        surrogate_denominator_q <= surrogate_denominator_next;
                        surrogate_remainder_q <= '0;
                        surrogate_quotient_q <= '0;
                        surrogate_divide_bit_q <= 6'd52;
                        if (surrogate_denominator_overflow) begin
                            o_overflow <= 1'b1;
                            state_q <= STATE_ERROR;
                        end else begin
                            state_q <= STATE_WH_SURROGATE_DIVIDE;
                        end
                    end

                    STATE_WH_SURROGATE_DIVIDE: begin
                        surrogate_remainder_q <= surrogate_remainder_next;
                        surrogate_quotient_q <= surrogate_quotient_next;
                        if (surrogate_divide_bit_q == 0) begin
                            wh_surrogate_q <= surrogate_factor_next;
                            wh_column_q <= '0;
                            state_q <= STATE_WH_READ;
                        end else begin
                            surrogate_divide_bit_q <=
                                surrogate_divide_bit_q - 1'b1;
                        end
                    end

                    STATE_WH_READ: begin
                        wh_error_q <= pair_error_mem[wh_pair_addr];
                        state_q <= STATE_WH_EMIT;
                    end

                    STATE_WH_EMIT: begin
                        if (wh_fire) begin
                            if (wh_last) begin
                                wout_addr_q <= '0;
                                wout_class_q <= '0;
                                wout_hidden_q <= '0;
                                wout_sample_q <= '0;
                                state_q <= STATE_WOUT_READ;
                            end else if (wh_column_q ==
                                    (batch_count_q - 1'b1)) begin
                                wh_column_q <= '0;
                                if (wh_hidden_q ==
                                        (hidden_count_q - 1'b1)) begin
                                    wh_hidden_q <= '0;
                                    if (wh_sample_q ==
                                            (batch_count_q - 1'b1)) begin
                                        wh_sample_q <= '0;
                                        wh_path_q <= 1'b1;
                                    end else begin
                                        wh_sample_q <= wh_sample_q + 1'b1;
                                    end
                                end else begin
                                    wh_hidden_q <= wh_hidden_q + 1'b1;
                                end
                                state_q <= STATE_WH_SURROGATE_READ;
                            end else begin
                                wh_column_q <= wh_column_q + 1'b1;
                                state_q <= STATE_WH_READ;
                            end
                        end
                    end

                    STATE_WOUT_READ: begin
                        wout_error_q <= output_error_mem[
                            int'(wout_sample_q) * CLASS_MAX +
                            int'(wout_class_q)];
                        state_q <= STATE_WOUT_EMIT;
                    end

                    STATE_WOUT_EMIT: begin
                        if (wout_fire) begin
                            if (wout_last) begin
                                state_q <= STATE_IDLE;
                                o_done <= 1'b1;
                            end else if (wout_sample_q ==
                                    (batch_count_q - 1'b1)) begin
                                wout_sample_q <= '0;
                                wout_addr_q <= wout_addr_q + 1'b1;
                                if (wout_hidden_q ==
                                        (hidden_count_q - 1'b1)) begin
                                    wout_hidden_q <= '0;
                                    wout_class_q <= wout_class_q + 1'b1;
                                end else begin
                                    wout_hidden_q <= wout_hidden_q + 1'b1;
                                end
                                state_q <= STATE_WOUT_READ;
                            end else begin
                                wout_sample_q <= wout_sample_q + 1'b1;
                                state_q <= STATE_WOUT_READ;
                            end
                        end
                    end

                    STATE_ERROR: begin
                    end

                    default: state_q <= STATE_ERROR;
                endcase
            end

            if ((softmax_config_error || softmax_overflow) &&
                (state_q != STATE_IDLE)) begin
                if (softmax_overflow)
                    o_overflow <= 1'b1;
                else
                    o_protocol_error <= 1'b1;
                state_q <= STATE_ERROR;
            end
        end
    end

    logic _unused_softmax_done;
    logic _unused_softmax_busy;
    always_comb begin
        _unused_softmax_done = softmax_done;
        _unused_softmax_busy = softmax_busy;
    end

endmodule

`default_nettype wire
