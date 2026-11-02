`timescale 1ns/1ps
`default_nettype none
// For legal carried SGD residues, exact delta zero leaves both stored fields
// unchanged. The fixed binary64 coefficient/RNE zero interval is [-66666,66666].
module replay_gradient_filter # (parameter bit SGD=1'b0) (
    input wire i_valid,
    input wire signed [63:0] i_gradient,
    input wire i_service_ready,
    output wire o_ready, o_valid, o_skip
);
    wire zero_delta=SGD ? ((i_gradient>= -64'sd66666) && (i_gradient<=64'sd66666))
                        : (i_gradient==0);
    assign o_ready=i_service_ready;
    assign o_valid=i_valid && !zero_delta;
    assign o_skip=i_valid && zero_delta;
endmodule
`default_nettype wire
