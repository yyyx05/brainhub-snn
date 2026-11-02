`timescale 1ns/1ps
`default_nettype none

// Shared max-subtracted row softmax selected by the FC B16 timestep 0..7
// screen. Inputs are signed Q6, exp/probability use Q0.16, and normalization
// uses a sequential restoring divider to avoid a wide combinational divider.
module unified_tp_row_softmax_q6 #(
    parameter int INPUT_WIDTH = 16,
    parameter int MAX_ITEMS = 16,
    parameter int ITEM_INDEX_WIDTH = (MAX_ITEMS <= 2) ? 1 : $clog2(MAX_ITEMS),
    parameter int ITEM_COUNT_WIDTH = $clog2(MAX_ITEMS + 1)
) (
    input  wire logic                                  i_clk,
    input  wire logic                                  i_rst_n,
    input  wire logic                                  i_start_valid,
    output logic                                  o_start_ready,
    input  wire logic [ITEM_COUNT_WIDTH-1:0]            i_row_length,
    output logic                                  o_config_error,
    input  wire logic                                  i_value_valid,
    output logic                                  o_value_ready,
    input  wire logic signed [INPUT_WIDTH-1:0]          i_value_q6,
    output logic                                  o_probability_valid,
    input  wire logic                                  i_probability_ready,
    output logic [ITEM_INDEX_WIDTH-1:0]            o_probability_index,
    output logic [16:0]                           o_probability_q016,
    output logic                                  o_probability_last,
    output logic                                  o_done,
    output logic                                  o_busy
);

    localparam int PROB_WIDTH = 17;
    localparam int PROB_FRAC = 16;
    localparam int SUM_WIDTH = PROB_WIDTH + $clog2(MAX_ITEMS);
    localparam int NUMERATOR_WIDTH = PROB_WIDTH + PROB_FRAC;
    localparam int DIV_INDEX_WIDTH = $clog2(NUMERATOR_WIDTH);

    localparam logic [16:0] EXP_LUT [0:64] = '{
        17'd65536, 17'd57835, 17'd51039, 17'd45042, 17'd39750,
        17'd35079, 17'd30957, 17'd27319, 17'd24109, 17'd21276,
        17'd18776, 17'd16570, 17'd14623, 17'd12905, 17'd11388,
        17'd10050, 17'd8869, 17'd7827, 17'd6907, 17'd6096,
        17'd5380, 17'd4747, 17'd4190, 17'd3697, 17'd3263,
        17'd2879, 17'd2541, 17'd2243, 17'd1979, 17'd1746,
        17'd1541, 17'd1360, 17'd1200, 17'd1059, 17'd935,
        17'd825, 17'd728, 17'd642, 17'd567, 17'd500,
        17'd442, 17'd390, 17'd344, 17'd303, 17'd268,
        17'd236, 17'd209, 17'd184, 17'd162, 17'd143,
        17'd127, 17'd112, 17'd99, 17'd87, 17'd77,
        17'd68, 17'd60, 17'd53, 17'd47, 17'd41,
        17'd36, 17'd32, 17'd28, 17'd25, 17'd22
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
    logic [16:0] exp_mem [0:MAX_ITEMS-1];
    logic [ITEM_COUNT_WIDTH-1:0] row_length_q;
    logic [ITEM_COUNT_WIDTH-1:0] load_count_q;
    logic signed [INPUT_WIDTH-1:0] max_value_q;
    logic [ITEM_INDEX_WIDTH-1:0] exp_index_q;
    logic signed [INPUT_WIDTH-1:0] exp_value_q;
    logic [SUM_WIDTH-1:0] exp_sum_q;
    logic [ITEM_INDEX_WIDTH-1:0] output_index_q;
    logic [16:0] output_exp_q;
    logic [NUMERATOR_WIDTH-1:0] dividend_q;
    logic [NUMERATOR_WIDTH-1:0] quotient_q;
    logic [SUM_WIDTH-2:0] remainder_q;
    logic [SUM_WIDTH-1:0] denominator_q;
    logic [DIV_INDEX_WIDTH-1:0] divide_bit_q;
    logic [16:0] probability_q;

    logic start_fire;
    logic value_fire;
    logic probability_fire;
    logic load_last;
    logic exp_last;
    logic output_last;
    logic [INPUT_WIDTH:0] exp_delta_q;
    logic signed [INPUT_WIDTH:0] max_value_extended;
    logic signed [INPUT_WIDTH:0] exp_value_extended;
    logic [INPUT_WIDTH:0] rounded_lut_index;
    logic [6:0] clipped_lut_index;
    logic [16:0] exp_lookup;
    logic [SUM_WIDTH-1:0] exp_sum_next;
    logic [NUMERATOR_WIDTH-1:0] numerator_value;
    logic [SUM_WIDTH-1:0] remainder_shifted;
    logic [SUM_WIDTH-2:0] remainder_next;
    logic [NUMERATOR_WIDTH-1:0] quotient_next;

    always_comb begin
        o_start_ready = state_q == STATE_IDLE;
        start_fire = i_start_valid && o_start_ready;
        o_value_ready = state_q == STATE_LOAD;
        value_fire = i_value_valid && o_value_ready;
        o_probability_valid = state_q == STATE_OUTPUT;
        probability_fire = o_probability_valid && i_probability_ready;
        o_probability_index = output_index_q;
        o_probability_q016 = probability_q;
        output_last = ITEM_COUNT_WIDTH'(
            {1'b0, output_index_q} + 1'b1) >= row_length_q;
        o_probability_last = output_last;
        o_busy = state_q != STATE_IDLE;

        load_last = (load_count_q + 1'b1) >= row_length_q;
        exp_last = ITEM_COUNT_WIDTH'(
            {1'b0, exp_index_q} + 1'b1) >= row_length_q;
        max_value_extended = {max_value_q[INPUT_WIDTH-1], max_value_q};
        exp_value_extended = {exp_value_q[INPUT_WIDTH-1], exp_value_q};
        exp_delta_q = $unsigned(max_value_extended - exp_value_extended);
        rounded_lut_index = (exp_delta_q + (INPUT_WIDTH+1)'(4)) >> 3;
        clipped_lut_index = (rounded_lut_index > (INPUT_WIDTH+1)'(64)) ?
            7'd64 : rounded_lut_index[6:0];
        exp_lookup = EXP_LUT[clipped_lut_index];
        exp_sum_next = exp_sum_q + SUM_WIDTH'(exp_lookup);
        numerator_value = (NUMERATOR_WIDTH'(output_exp_q) << PROB_FRAC) +
            NUMERATOR_WIDTH'(exp_sum_q >> 1);

        remainder_shifted = {remainder_q, dividend_q[divide_bit_q]};
        quotient_next = quotient_q;
        if (remainder_shifted >= denominator_q) begin
            remainder_next = (SUM_WIDTH-1)'(remainder_shifted - denominator_q);
            quotient_next[divide_bit_q] = 1'b1;
        end else begin
            remainder_next = remainder_shifted[SUM_WIDTH-2:0];
            quotient_next[divide_bit_q] = 1'b0;
        end
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
        end else begin
            o_config_error <= 1'b0;
            o_done <= 1'b0;

            case (state_q)
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
                        row_mem[load_count_q[ITEM_INDEX_WIDTH-1:0]] <= i_value_q6;
                        if ($signed(i_value_q6) > $signed(max_value_q))
                            max_value_q <= i_value_q6;
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
                        probability_q <= quotient_next[16:0];
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
