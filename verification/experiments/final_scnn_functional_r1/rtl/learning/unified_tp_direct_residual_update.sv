`timescale 1ns/1ps
`default_nettype none

module unified_tp_direct_residual_update #(
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16
) (
    input  wire logic                               i_clk,
    input  wire logic                               i_rst_n,

    input  wire logic                               i_valid,
    output logic                               o_ready,
    input  wire logic signed [CODE_WIDTH-1:0]       i_code,
    input  wire logic signed [RESIDUAL_WIDTH-1:0]   i_residual,
    input  wire logic signed [1:0]                  i_update_direction,
    input  wire logic        [VOTE_WIDTH-1:0]       i_vote_increment,
    input  wire logic        [VOTE_WIDTH-1:0]       i_vote_threshold,
    input  wire logic signed [CODE_WIDTH-1:0]       i_code_min,
    input  wire logic signed [CODE_WIDTH-1:0]       i_code_max,

    output logic                               o_valid,
    input  wire logic                               i_ready,
    output logic signed [CODE_WIDTH-1:0]       o_code,
    output logic signed [RESIDUAL_WIDTH-1:0]   o_residual,
    output logic                               o_code_changed,
    output logic                               o_saturated,
    output logic                               o_config_error
);

    localparam logic signed [1:0] DIR_INVALID = 2'sb10;

    localparam int BASE_WIDTH =
        (CODE_WIDTH > RESIDUAL_WIDTH) ? CODE_WIDTH : RESIDUAL_WIDTH;
    localparam int WORK_WIDTH =
        ((BASE_WIDTH > VOTE_WIDTH) ? BASE_WIDTH : VOTE_WIDTH) + 3;

    logic signed [WORK_WIDTH-1:0] code_ext;
    logic signed [WORK_WIDTH-1:0] code_min_ext;
    logic signed [WORK_WIDTH-1:0] code_max_ext;
    logic signed [WORK_WIDTH-1:0] residual_ext;
    logic signed [WORK_WIDTH-1:0] increment_ext;
    logic signed [WORK_WIDTH-1:0] threshold_ext;
    logic signed [WORK_WIDTH-1:0] total_residual;
    logic signed [WORK_WIDTH-1:0] normalized_residual;
    logic signed [WORK_WIDTH-1:0] code_delta;
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
        code_min_ext = {{(WORK_WIDTH-CODE_WIDTH){i_code_min[CODE_WIDTH-1]}}, i_code_min};
        code_max_ext = {{(WORK_WIDTH-CODE_WIDTH){i_code_max[CODE_WIDTH-1]}}, i_code_max};
        residual_ext = {
            {(WORK_WIDTH-RESIDUAL_WIDTH){i_residual[RESIDUAL_WIDTH-1]}},
            i_residual
        };
        increment_ext = $signed({{(WORK_WIDTH-VOTE_WIDTH){1'b0}}, i_vote_increment});
        threshold_ext = $signed({{(WORK_WIDTH-VOTE_WIDTH){1'b0}}, i_vote_threshold});

        total_residual = residual_ext;
        normalized_residual = residual_ext;
        code_delta = '0;
        code_candidate = code_ext;
        code_next = i_code;
        residual_next = i_residual;
        code_changed_next = 1'b0;
        saturated_next = 1'b0;
        config_error_next = 1'b0;

        if ((i_vote_threshold == '0) ||
            (increment_ext >= (threshold_ext + threshold_ext)) ||
            (i_update_direction == DIR_INVALID) ||
            (code_min_ext > code_max_ext) ||
            (residual_ext >= threshold_ext) ||
            (residual_ext <= -threshold_ext)) begin
            config_error_next = 1'b1;
        end else begin
            case (i_update_direction)
                -2'sd1: total_residual = residual_ext - increment_ext;
                 2'sd0: total_residual = residual_ext;
                 2'sd1: total_residual = residual_ext + increment_ext;
                default: total_residual = residual_ext;
            endcase
            normalized_residual = total_residual;

            if (normalized_residual >= threshold_ext) begin
                normalized_residual = normalized_residual - threshold_ext;
                code_delta = code_delta + 1;
                if (normalized_residual >= threshold_ext) begin
                    normalized_residual = normalized_residual - threshold_ext;
                    code_delta = code_delta + 1;
                end
            end else if (normalized_residual <= -threshold_ext) begin
                normalized_residual = normalized_residual + threshold_ext;
                code_delta = code_delta - 1;
                if (normalized_residual <= -threshold_ext) begin
                    normalized_residual = normalized_residual + threshold_ext;
                    code_delta = code_delta - 1;
                end
            end

            code_candidate = code_ext + code_delta;
            if (code_candidate > code_max_ext) begin
                code_next = i_code_max;
                residual_next = '0;
                saturated_next = 1'b1;
            end else if (code_candidate < code_min_ext) begin
                code_next = i_code_min;
                residual_next = '0;
                saturated_next = 1'b1;
            end else begin
                code_next = code_candidate[CODE_WIDTH-1:0];
                residual_next = normalized_residual[RESIDUAL_WIDTH-1:0];
            end
            code_changed_next = (code_next != i_code);
        end
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
            if (valid_q && i_ready) begin
                valid_q <= 1'b0;
            end
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
