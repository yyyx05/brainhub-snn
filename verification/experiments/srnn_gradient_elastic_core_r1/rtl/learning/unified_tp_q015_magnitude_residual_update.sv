`timescale 1ns/1ps
`default_nettype none

// Applies one already-scaled signed residual delta to a Q0.15 parameter. The
// binary residual scale makes normalization a shift/mask operation and permits
// one request to cross multiple weight-code LSBs.
module unified_tp_q015_magnitude_residual_update #(
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int DELTA_WIDTH = 18,
    parameter int RESIDUAL_FRAC_BITS = 15
) (
    input  wire logic                            i_clk,
    input  wire logic                            i_rst_n,
    input  wire logic                            i_valid,
    output logic                                 o_ready,
    input  wire logic signed [CODE_WIDTH-1:0]    i_code,
    input  wire logic signed [RESIDUAL_WIDTH-1:0] i_residual,
    input  wire logic signed [DELTA_WIDTH-1:0]   i_scaled_delta,
    input  wire logic signed [CODE_WIDTH-1:0]    i_code_min,
    input  wire logic signed [CODE_WIDTH-1:0]    i_code_max,
    output logic                                 o_valid,
    input  wire logic                            i_ready,
    output logic signed [CODE_WIDTH-1:0]         o_code,
    output logic signed [RESIDUAL_WIDTH-1:0]     o_residual,
    output logic                                 o_code_changed,
    output logic                                 o_saturated,
    output logic                                 o_config_error
);

    localparam int BASE_WIDTH =
        (CODE_WIDTH > DELTA_WIDTH) ? CODE_WIDTH : DELTA_WIDTH;
    localparam int WORK_WIDTH =
        ((BASE_WIDTH > RESIDUAL_WIDTH) ? BASE_WIDTH : RESIDUAL_WIDTH) + 2;
    localparam logic signed [WORK_WIDTH-1:0] RESIDUAL_SCALE =
        WORK_WIDTH'(1 << RESIDUAL_FRAC_BITS);

    logic signed [WORK_WIDTH-1:0] code_ext;
    logic signed [WORK_WIDTH-1:0] code_min_ext;
    logic signed [WORK_WIDTH-1:0] code_max_ext;
    logic signed [WORK_WIDTH-1:0] residual_ext;
    logic signed [WORK_WIDTH-1:0] delta_ext;
    logic signed [WORK_WIDTH-1:0] total;
    logic [WORK_WIDTH-1:0] total_abs;
    logic [WORK_WIDTH-1:0] code_delta_abs;
    logic [RESIDUAL_FRAC_BITS-1:0] residual_abs;
    logic signed [WORK_WIDTH-1:0] code_delta;
    logic signed [RESIDUAL_WIDTH-1:0] normalized_residual;
    logic signed [WORK_WIDTH-1:0] code_candidate;

    logic signed [CODE_WIDTH-1:0] code_next;
    logic signed [RESIDUAL_WIDTH-1:0] residual_next;
    logic code_changed_next;
    logic saturated_next;
    logic config_error_next;

    logic valid_q;
    logic signed [CODE_WIDTH-1:0] code_q;
    logic signed [RESIDUAL_WIDTH-1:0] residual_q;
    logic code_changed_q;
    logic saturated_q;
    logic config_error_q;

    assign o_ready = !valid_q || i_ready;
    assign o_valid = valid_q;
    assign o_code = code_q;
    assign o_residual = residual_q;
    assign o_code_changed = code_changed_q;
    assign o_saturated = saturated_q;
    assign o_config_error = config_error_q;

    always_comb begin
        code_ext = {{(WORK_WIDTH-CODE_WIDTH){i_code[CODE_WIDTH-1]}}, i_code};
        code_min_ext = {{(WORK_WIDTH-CODE_WIDTH){i_code_min[CODE_WIDTH-1]}},
                        i_code_min};
        code_max_ext = {{(WORK_WIDTH-CODE_WIDTH){i_code_max[CODE_WIDTH-1]}},
                        i_code_max};
        residual_ext = {
            {(WORK_WIDTH-RESIDUAL_WIDTH){i_residual[RESIDUAL_WIDTH-1]}},
            i_residual
        };
        delta_ext = {{(WORK_WIDTH-DELTA_WIDTH){i_scaled_delta[DELTA_WIDTH-1]}},
                     i_scaled_delta};
        total = residual_ext + delta_ext;
        total_abs = total[WORK_WIDTH-1] ? $unsigned(-total) : $unsigned(total);
        code_delta_abs = total_abs >> RESIDUAL_FRAC_BITS;
        residual_abs = total_abs[RESIDUAL_FRAC_BITS-1:0];
        code_delta = total[WORK_WIDTH-1] ?
            -$signed(code_delta_abs) : $signed(code_delta_abs);
        normalized_residual = total[WORK_WIDTH-1] ?
            -$signed({{(RESIDUAL_WIDTH-RESIDUAL_FRAC_BITS){1'b0}}, residual_abs}) :
             $signed({{(RESIDUAL_WIDTH-RESIDUAL_FRAC_BITS){1'b0}}, residual_abs});
        code_candidate = code_ext + code_delta;

        code_next = i_code;
        residual_next = i_residual;
        code_changed_next = 1'b0;
        saturated_next = 1'b0;
        config_error_next = 1'b0;

        if ((code_min_ext > code_max_ext) ||
            (residual_ext >= RESIDUAL_SCALE) ||
            (residual_ext <= -RESIDUAL_SCALE)) begin
            config_error_next = 1'b1;
        end else if (code_candidate > code_max_ext) begin
            code_next = i_code_max;
            residual_next = '0;
            saturated_next = 1'b1;
        end else if (code_candidate < code_min_ext) begin
            code_next = i_code_min;
            residual_next = '0;
            saturated_next = 1'b1;
        end else begin
            code_next = code_candidate[CODE_WIDTH-1:0];
            residual_next = normalized_residual;
        end
        code_changed_next = code_next != i_code;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            valid_q <= 1'b0;
            code_q <= '0;
            residual_q <= '0;
            code_changed_q <= 1'b0;
            saturated_q <= 1'b0;
            config_error_q <= 1'b0;
        end else begin
            if (valid_q && i_ready)
                valid_q <= 1'b0;
            if (i_valid && o_ready) begin
                valid_q <= 1'b1;
                code_q <= code_next;
                residual_q <= residual_next;
                code_changed_q <= code_changed_next;
                saturated_q <= saturated_next;
                config_error_q <= config_error_next;
            end
        end
    end

endmodule

`default_nettype wire
