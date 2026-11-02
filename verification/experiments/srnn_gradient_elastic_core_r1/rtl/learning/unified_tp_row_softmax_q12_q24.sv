`timescale 1ns/1ps
`default_nettype none

// One sequential SRNN softmax shared by TP pair rows and output CE rows.
// Inputs are signed Q12.  The max-subtracted exponent table is Q20 with a
// 1/16 step, and the emitted probabilities are unsigned Q0.24.
module unified_tp_row_softmax_q12_q24 #(
    parameter int INPUT_WIDTH = 32,
    parameter int MAX_ITEMS = 64,
    parameter int ITEM_INDEX_WIDTH = (MAX_ITEMS <= 2) ? 1 : $clog2(MAX_ITEMS),
    parameter int ITEM_COUNT_WIDTH = $clog2(MAX_ITEMS + 1)
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_start_valid,
    output logic                                  o_start_ready,
    input  logic [ITEM_COUNT_WIDTH-1:0]           i_row_length,
    output logic                                  o_config_error,
    input  logic                                  i_value_valid,
    output logic                                  o_value_ready,
    input  logic signed [INPUT_WIDTH-1:0]         i_value_q12,
    output logic                                  o_probability_valid,
    input  logic                                  i_probability_ready,
    output logic [ITEM_INDEX_WIDTH-1:0]           o_probability_index,
    output logic [24:0]                           o_probability_q024,
    output logic                                  o_probability_last,
    output logic                                  o_done,
    output logic                                  o_busy,
    output logic                                  o_arithmetic_overflow
);

    localparam int EXP_WIDTH = 21;
    localparam int PROB_FRAC = 24;
    localparam int SUM_WIDTH = EXP_WIDTH + $clog2(MAX_ITEMS);
    localparam int NUMERATOR_WIDTH = EXP_WIDTH + PROB_FRAC;
    localparam int DIV_INDEX_WIDTH = $clog2(NUMERATOR_WIDTH);

    localparam logic [EXP_WIDTH-1:0] EXP_LUT [0:128] = '{
        21'd1048576, 21'd985046, 21'd925365, 21'd869300,
        21'd816632, 21'd767155, 21'd720675, 21'd677012,
        21'd635993, 21'd597461, 21'd561262, 21'd527257,
        21'd495312, 21'd465303, 21'd437112, 21'd410628,
        21'd385750, 21'd362378, 21'd340423, 21'd319798,
        21'd300422, 21'd282220, 21'd265122, 21'd249059,
        21'd233969, 21'd219793, 21'd206477, 21'd193967,
        21'd182215, 21'd171175, 21'd160804, 21'd151062,
        21'd141909, 21'd133311, 21'd125235, 21'd117647,
        21'd110519, 21'd103823, 21'd97533, 21'd91624,
        21'd86072, 21'd80857, 21'd75959, 21'd71356,
        21'd67033, 21'd62972, 21'd59157, 21'd55572,
        21'd52206, 21'd49043, 21'd46071, 21'd43280,
        21'd40658, 21'd38194, 21'd35880, 21'd33706,
        21'd31664, 21'd29746, 21'd27944, 21'd26251,
        21'd24660, 21'd23166, 21'd21763, 21'd20444,
        21'd19205, 21'd18042, 21'd16949, 21'd15922,
        21'd14957, 21'd14051, 21'd13200, 21'd12400,
        21'd11649, 21'd10943, 21'd10280, 21'd9657,
        21'd9072, 21'd8522, 21'd8006, 21'd7521,
        21'd7065, 21'd6637, 21'd6235, 21'd5857,
        21'd5502, 21'd5169, 21'd4856, 21'd4562,
        21'd4285, 21'd4026, 21'd3782, 21'd3553,
        21'd3337, 21'd3135, 21'd2945, 21'd2767,
        21'd2599, 21'd2442, 21'd2294, 21'd2155,
        21'd2024, 21'd1902, 21'd1786, 21'd1678,
        21'd1576, 21'd1481, 21'd1391, 21'd1307,
        21'd1228, 21'd1153, 21'd1083, 21'd1018,
        21'd956, 21'd898, 21'd844, 21'd793,
        21'd745, 21'd700, 21'd657, 21'd617,
        21'd580, 21'd545, 21'd512, 21'd481,
        21'd452, 21'd424, 21'd399, 21'd374,
        21'd352
    };

    typedef enum logic [3:0] {
        STATE_IDLE,
        STATE_LOAD,
        STATE_EXP_READ,
        STATE_EXP_CALC,
        STATE_DIV_READ,
        STATE_DIV_INIT,
        STATE_DIV_STEP,
        STATE_OUTPUT
    } state_t;

    state_t state_q;
    logic signed [INPUT_WIDTH-1:0] row_mem [0:MAX_ITEMS-1];
    logic [EXP_WIDTH-1:0] exp_mem [0:MAX_ITEMS-1];
    logic [ITEM_COUNT_WIDTH-1:0] row_length_q;
    logic [ITEM_COUNT_WIDTH-1:0] load_count_q;
    logic signed [INPUT_WIDTH-1:0] max_value_q;
    logic [ITEM_INDEX_WIDTH-1:0] exp_index_q;
    logic signed [INPUT_WIDTH-1:0] exp_value_q;
    logic [SUM_WIDTH-1:0] exp_sum_q;
    logic [ITEM_INDEX_WIDTH-1:0] output_index_q;
    logic [EXP_WIDTH-1:0] output_exp_q;
    logic [NUMERATOR_WIDTH-1:0] dividend_q;
    logic [NUMERATOR_WIDTH-1:0] quotient_q;
    logic [SUM_WIDTH-1:0] remainder_q;
    logic [SUM_WIDTH-1:0] denominator_q;
    logic [DIV_INDEX_WIDTH-1:0] divide_bit_q;
    logic [24:0] probability_q;

    logic start_fire;
    logic value_fire;
    logic probability_fire;
    logic load_last;
    logic exp_last;
    logic output_last;
    logic signed [INPUT_WIDTH:0] max_value_extended;
    logic signed [INPUT_WIDTH:0] exp_value_extended;
    logic [INPUT_WIDTH:0] exp_delta;
    logic [INPUT_WIDTH:0] lut_quotient;
    logic [7:0] lut_remainder;
    logic lut_increment;
    logic [INPUT_WIDTH:0] rounded_lut_index;
    logic [7:0] clipped_lut_index;
    logic [EXP_WIDTH-1:0] exp_lookup;
    logic [SUM_WIDTH-1:0] exp_sum_next;
    logic [NUMERATOR_WIDTH-1:0] numerator_value;
    logic [SUM_WIDTH:0] remainder_shifted;
    logic [SUM_WIDTH:0] denominator_extended;
    logic [SUM_WIDTH-1:0] remainder_next;
    logic [NUMERATOR_WIDTH-1:0] quotient_next;
    logic [SUM_WIDTH:0] twice_remainder;
    logic division_round_increment;
    logic [NUMERATOR_WIDTH:0] rounded_quotient;
    logic probability_overflow;

    always_comb begin
        o_start_ready = state_q == STATE_IDLE;
        start_fire = i_start_valid && o_start_ready;
        o_value_ready = state_q == STATE_LOAD;
        value_fire = i_value_valid && o_value_ready;
        o_probability_valid = state_q == STATE_OUTPUT;
        probability_fire = o_probability_valid && i_probability_ready;
        o_probability_index = output_index_q;
        o_probability_q024 = probability_q;
        output_last = ({1'b0, output_index_q} + 1'b1) >= row_length_q;
        o_probability_last = output_last;
        o_busy = state_q != STATE_IDLE;

        load_last = (load_count_q + 1'b1) >= row_length_q;
        exp_last = ({1'b0, exp_index_q} + 1'b1) >= row_length_q;
        max_value_extended = {max_value_q[INPUT_WIDTH-1], max_value_q};
        exp_value_extended = {exp_value_q[INPUT_WIDTH-1], exp_value_q};
        exp_delta = $unsigned(max_value_extended - exp_value_extended);
        lut_quotient = exp_delta >> 8;
        lut_remainder = exp_delta[7:0];
        lut_increment = (lut_remainder > 8'd128) ||
            ((lut_remainder == 8'd128) && lut_quotient[0]);
        rounded_lut_index = lut_quotient + lut_increment;
        clipped_lut_index = (rounded_lut_index > (INPUT_WIDTH+1)'(128)) ?
            8'd128 : rounded_lut_index[7:0];
        exp_lookup = EXP_LUT[clipped_lut_index];
        exp_sum_next = exp_sum_q + SUM_WIDTH'(exp_lookup);

        numerator_value = NUMERATOR_WIDTH'(output_exp_q) << PROB_FRAC;
        remainder_shifted = {remainder_q, dividend_q[divide_bit_q]};
        denominator_extended = {1'b0, denominator_q};
        quotient_next = quotient_q;
        if (remainder_shifted >= denominator_extended) begin
            remainder_next = SUM_WIDTH'(
                remainder_shifted - denominator_extended);
            quotient_next[divide_bit_q] = 1'b1;
        end else begin
            remainder_next = remainder_shifted[SUM_WIDTH-1:0];
            quotient_next[divide_bit_q] = 1'b0;
        end
        twice_remainder = {remainder_next, 1'b0};
        division_round_increment =
            (twice_remainder > denominator_extended) ||
            ((twice_remainder == denominator_extended) && quotient_next[0]);
        rounded_quotient = {1'b0, quotient_next} +
            division_round_increment;
        probability_overflow =
            rounded_quotient > (NUMERATOR_WIDTH+1)'(1 << PROB_FRAC);
    end

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            row_length_q <= '0;
            load_count_q <= '0;
            max_value_q <= '0;
            exp_index_q <= '0;
            exp_value_q <= '0;
            exp_sum_q <= '0;
            output_index_q <= '0;
            output_exp_q <= '0;
            dividend_q <= '0;
            quotient_q <= '0;
            remainder_q <= '0;
            denominator_q <= '0;
            divide_bit_q <= '0;
            probability_q <= '0;
            o_config_error <= 1'b0;
            o_done <= 1'b0;
            o_arithmetic_overflow <= 1'b0;
        end else begin
            o_config_error <= 1'b0;
            o_done <= 1'b0;
            o_arithmetic_overflow <= 1'b0;

            unique case (state_q)
                STATE_IDLE: begin
                    if (start_fire) begin
                        if ((i_row_length == '0) ||
                            (i_row_length > ITEM_COUNT_WIDTH'(MAX_ITEMS))) begin
                            o_config_error <= 1'b1;
                        end else begin
                            row_length_q <= i_row_length;
                            load_count_q <= '0;
                            max_value_q <= {1'b1, {(INPUT_WIDTH-1){1'b0}}};
                            state_q <= STATE_LOAD;
                        end
                    end
                end

                STATE_LOAD: begin
                    if (value_fire) begin
                        row_mem[load_count_q[ITEM_INDEX_WIDTH-1:0]] <=
                            i_value_q12;
                        if ($signed(i_value_q12) > $signed(max_value_q))
                            max_value_q <= i_value_q12;
                        if (load_last) begin
                            exp_index_q <= '0;
                            exp_sum_q <= '0;
                            state_q <= STATE_EXP_READ;
                        end else begin
                            load_count_q <= load_count_q + 1'b1;
                        end
                    end
                end

                STATE_EXP_READ: begin
                    exp_value_q <= row_mem[exp_index_q];
                    state_q <= STATE_EXP_CALC;
                end

                STATE_EXP_CALC: begin
                    exp_mem[exp_index_q] <= exp_lookup;
                    exp_sum_q <= exp_sum_next;
                    if (exp_last) begin
                        output_index_q <= '0;
                        state_q <= STATE_DIV_READ;
                    end else begin
                        exp_index_q <= exp_index_q + 1'b1;
                        state_q <= STATE_EXP_READ;
                    end
                end

                STATE_DIV_READ: begin
                    output_exp_q <= exp_mem[output_index_q];
                    state_q <= STATE_DIV_INIT;
                end

                STATE_DIV_INIT: begin
                    dividend_q <= numerator_value;
                    quotient_q <= '0;
                    remainder_q <= '0;
                    denominator_q <= exp_sum_q;
                    divide_bit_q <= DIV_INDEX_WIDTH'(NUMERATOR_WIDTH - 1);
                    state_q <= STATE_DIV_STEP;
                end

                STATE_DIV_STEP: begin
                    quotient_q <= quotient_next;
                    remainder_q <= remainder_next;
                    if (divide_bit_q == 0) begin
                        probability_q <= rounded_quotient[24:0];
                        if (probability_overflow)
                            o_arithmetic_overflow <= 1'b1;
                        state_q <= STATE_OUTPUT;
                    end else begin
                        divide_bit_q <= divide_bit_q - 1'b1;
                    end
                end

                STATE_OUTPUT: begin
                    if (probability_fire) begin
                        if (output_last) begin
                            o_done <= 1'b1;
                            state_q <= STATE_IDLE;
                        end else begin
                            output_index_q <= output_index_q + 1'b1;
                            state_q <= STATE_DIV_READ;
                        end
                    end
                end

                default: state_q <= STATE_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
