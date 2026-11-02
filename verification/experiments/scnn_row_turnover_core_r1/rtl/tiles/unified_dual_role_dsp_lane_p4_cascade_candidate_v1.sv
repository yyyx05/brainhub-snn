`timescale 1ns/1ps
`default_nettype none

// Two vertically cascaded DSP48E1s accumulate four independently enabled
// topology terms into one 48-bit state. The first DSP adds pair 0/1 to the
// previous result; the second adds pair 2/3 through PCIN and registers P.
module unified_dual_role_dsp_lane_p4_cascade_candidate_v1 (
    input  wire logic                    i_clk,
    input  wire logic                    i_fire,
    input  wire logic                    i_first,
    input  wire logic                    i_enable0,
    input  wire logic                    i_enable1,
    input  wire logic                    i_enable2,
    input  wire logic                    i_enable3,
    input  wire logic signed [24:0]      i_term0,
    input  wire logic signed [24:0]      i_term1,
    input  wire logic signed [24:0]      i_term2,
    input  wire logic signed [24:0]      i_term3,
    input  wire logic signed [17:0]      i_factor,
    output logic signed [47:0]           o_result
);
`ifdef SYNTHESIS
    logic [47:0] cascade_value;
    logic [47:0] result_value;
    logic [4:0] inmode0;
    logic [4:0] inmode1;
    logic [6:0] opmode0;

    assign inmode0 = {2'b00, i_enable1, i_enable0, 1'b0};
    assign inmode1 = {2'b00, i_enable3, i_enable2, 1'b0};
    assign opmode0 = i_first ? 7'b0000101 : 7'b0110101;
    assign o_result = $signed(result_value);

    DSP48E1 #(
        .A_INPUT("DIRECT"), .B_INPUT("DIRECT"),
        .USE_DPORT("TRUE"), .USE_MULT("MULTIPLY"),
        .USE_SIMD("ONE48"),
        .AREG(0), .ACASCREG(0), .BREG(0), .BCASCREG(0),
        .DREG(0), .ADREG(0), .MREG(0), .PREG(0), .CREG(0),
        .INMODEREG(0), .OPMODEREG(0), .ALUMODEREG(0),
        .CARRYINREG(0), .CARRYINSELREG(0),
        .IS_INMODE_INVERTED(5'b00010)
    ) u_pair01 (
        .CLK(i_clk),
        .A({{5{i_term0[24]}}, i_term0}), .B(i_factor),
        .C(result_value), .D(i_term1),
        .INMODE(inmode0), .OPMODE(opmode0), .ALUMODE(4'b0000),
        .CARRYINSEL(3'b000), .CARRYIN(1'b0),
        .CEA1(1'b0), .CEA2(1'b0), .CEB1(1'b0), .CEB2(1'b0),
        .CEC(1'b0), .CED(1'b0), .CEAD(1'b0), .CEM(1'b0),
        .CEINMODE(1'b0), .CECTRL(1'b0), .CEALUMODE(1'b0),
        .CECARRYIN(1'b0), .CEP(1'b0),
        .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0),
        .RSTM(1'b0), .RSTP(1'b0), .RSTINMODE(1'b0),
        .RSTCTRL(1'b0), .RSTALUMODE(1'b0),
        .RSTALLCARRYIN(1'b0),
        .ACIN(30'b0), .BCIN(18'b0), .PCIN(48'b0),
        .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
        .P(), .ACOUT(), .BCOUT(), .PCOUT(cascade_value),
        .CARRYOUT(), .CARRYCASCOUT(), .MULTSIGNOUT(),
        .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(),
        .PATTERNBDETECT()
    );

    DSP48E1 #(
        .A_INPUT("DIRECT"), .B_INPUT("DIRECT"),
        .USE_DPORT("TRUE"), .USE_MULT("MULTIPLY"),
        .USE_SIMD("ONE48"),
        .AREG(0), .ACASCREG(0), .BREG(0), .BCASCREG(0),
        .DREG(0), .ADREG(0), .MREG(0), .PREG(1), .CREG(0),
        .INMODEREG(0), .OPMODEREG(0), .ALUMODEREG(0),
        .CARRYINREG(0), .CARRYINSELREG(0),
        .IS_INMODE_INVERTED(5'b00010)
    ) u_pair23 (
        .CLK(i_clk),
        .A({{5{i_term2[24]}}, i_term2}), .B(i_factor),
        .C(48'b0), .D(i_term3),
        .INMODE(inmode1), .OPMODE(7'b0010101),
        .ALUMODE(4'b0000), .CARRYINSEL(3'b000),
        .CARRYIN(1'b0),
        .CEA1(1'b0), .CEA2(1'b0), .CEB1(1'b0), .CEB2(1'b0),
        .CEC(1'b0), .CED(1'b0), .CEAD(1'b0), .CEM(1'b0),
        .CEINMODE(1'b0), .CECTRL(1'b0), .CEALUMODE(1'b0),
        .CECARRYIN(1'b0), .CEP(i_fire),
        .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0),
        .RSTM(1'b0), .RSTP(1'b0), .RSTINMODE(1'b0),
        .RSTCTRL(1'b0), .RSTALUMODE(1'b0),
        .RSTALLCARRYIN(1'b0),
        .ACIN(30'b0), .BCIN(18'b0), .PCIN(cascade_value),
        .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
        .P(result_value), .ACOUT(), .BCOUT(), .PCOUT(),
        .CARRYOUT(), .CARRYCASCOUT(), .MULTSIGNOUT(),
        .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(),
        .PATTERNBDETECT()
    );
`else
    logic signed [47:0] result_q;
    logic signed [25:0] pair01;
    logic signed [25:0] pair23;
    logic signed [26:0] four_term_sum;
    logic signed [44:0] contribution;

    always_comb begin
        pair01 = (i_enable0 ? $signed({i_term0[24], i_term0}) : 26'sd0) +
            (i_enable1 ? $signed({i_term1[24], i_term1}) : 26'sd0);
        pair23 = (i_enable2 ? $signed({i_term2[24], i_term2}) : 26'sd0) +
            (i_enable3 ? $signed({i_term3[24], i_term3}) : 26'sd0);
        four_term_sum = $signed({pair01[25], pair01}) +
            $signed({pair23[25], pair23});
        contribution = four_term_sum * i_factor;
    end

    // The physical DSP48E1 pre-adder is 25 bits. All supported INT5/INT8/INT16
    // modes fit this range; flag a future descriptor/precision extension that
    // would silently overflow either pair before multiplication.
    always_ff @(posedge i_clk) begin
        if (i_fire) begin
            assert (pair01[25] == pair01[24])
                else $error("P4 cascade pair 0/1 exceeds DSP48E1 pre-adder range");
            assert (pair23[25] == pair23[24])
                else $error("P4 cascade pair 2/3 exceeds DSP48E1 pre-adder range");
        end
    end

    always_ff @(posedge i_clk) begin
        if (i_fire)
            result_q <= i_first ?
                {{3{contribution[44]}}, contribution} :
                result_q + {{3{contribution[44]}}, contribution};
    end

    assign o_result = result_q;
`endif
endmodule

`default_nettype wire
