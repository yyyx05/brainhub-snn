`timescale 1ns/1ps
`default_nettype none
// One elastic input register in each arm makes input arithmetic reg-to-reg.
/* verilator lint_off DECLFILENAME */
module direct_registered44 (
    input wire i_clk,i_rst_n,i_valid,
    output wire o_ready,
    input wire signed [43:0] i_gradient,
    input wire signed [15:0] i_code,i_residual,
    output wire o_valid,
    input wire i_ready,
    output wire signed [15:0] o_code,o_residual,
    output wire o_code_changed,o_saturated,o_config_error
);
    logic valid_q;
    logic signed [43:0] gradient_q;
    logic signed [15:0] code_q,residual_q;
    wire core_ready;
    assign o_ready=!valid_q || core_ready;
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if(!i_rst_n) begin valid_q<=0;gradient_q<=0;code_q<=0;residual_q<=0;end
        else if(o_ready) begin
            valid_q<=i_valid;
            if(i_valid) begin gradient_q<=i_gradient;code_q<=i_code;residual_q<=i_residual;end
        end
    end
    direct_update_wrapper core_i(.i_clk(i_clk),.i_rst_n(i_rst_n),.i_valid(valid_q),.o_ready(core_ready),
        .i_gradient({{20{gradient_q[43]}},gradient_q}),.i_code(code_q),.i_residual(residual_q),
        .o_valid(o_valid),.i_ready(i_ready),.o_code(o_code),.o_residual(o_residual),
        .o_code_changed(o_code_changed),.o_saturated(o_saturated),.o_config_error(o_config_error));
endmodule

module sgd_registered44 (
    input wire i_clk,i_rst_n,i_valid,
    output wire o_ready,
    input wire signed [43:0] i_gradient,
    input wire signed [15:0] i_code,
    input wire signed [31:0] i_residual,
    output wire o_valid,
    input wire i_ready,
    output wire signed [15:0] o_code,
    output wire signed [31:0] o_residual,
    output wire o_code_changed,o_saturated,o_config_error
);
    logic valid_q;
    logic signed [43:0] gradient_q;
    logic signed [15:0] code_q;
    logic signed [31:0] residual_q;
    wire core_ready;
    assign o_ready=!valid_q || core_ready;
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if(!i_rst_n) begin valid_q<=0;gradient_q<=0;code_q<=0;residual_q<=0;end
        else if(o_ready) begin
            valid_q<=i_valid;
            if(i_valid) begin gradient_q<=i_gradient;code_q<=i_code;residual_q<=i_residual;end
        end
    end
    bounded_sgd_update core_i(.i_clk(i_clk),.i_rst_n(i_rst_n),.i_valid(valid_q),.o_ready(core_ready),
        .i_gradient(gradient_q),.i_code(code_q),.i_residual(residual_q),
        .o_valid(o_valid),.i_ready(i_ready),.o_code(o_code),.o_residual(o_residual),
        .o_code_changed(o_code_changed),.o_saturated(o_saturated),.o_config_error(o_config_error));
endmodule
/* verilator lint_on DECLFILENAME */
`default_nettype wire
