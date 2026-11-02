`timescale 1ns/1ps
`default_nettype none

// Parameterized ready/valid pipeline for control and metadata that must stay
// cycle-aligned with an independently implemented elastic arithmetic path.
// Payload registers are reset-free; valid bits own their contents.
module unified_elastic_pipeline #(
    parameter int DATA_WIDTH = 1,
    parameter int STAGES = 1
) (
    input  wire logic                  i_clk,
    input  wire logic                  i_rst_n,
    input  wire logic                  i_abort,
    input  wire logic                  i_valid,
    output logic                       o_ready,
    input  wire logic [DATA_WIDTH-1:0] i_data,
    output logic                       o_valid,
    input  wire logic                  i_ready,
    output logic [DATA_WIDTH-1:0]      o_data
);
    logic [STAGES-1:0] valid_q;
    logic [STAGES-1:0] ready;
    logic [DATA_WIDTH-1:0] data_q [0:STAGES-1];

    always_comb begin
        ready[STAGES-1] = !valid_q[STAGES-1] || i_ready;
        for (int stage = STAGES - 2; stage >= 0; stage--)
            ready[stage] = !valid_q[stage] || ready[stage + 1];
    end

    assign o_ready = !i_abort && ready[0];
    assign o_valid = !i_abort && valid_q[STAGES-1];
    assign o_data = data_q[STAGES-1];

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n || i_abort) begin
            valid_q <= '0;
        end else begin
            for (int stage = STAGES - 1; stage >= 0; stage--) begin
                if (ready[stage]) begin
                    if (stage == 0) begin
                        valid_q[stage] <= i_valid && o_ready;
                    end else begin
                        valid_q[stage] <= valid_q[stage - 1];
                    end
                end
            end
        end
    end

    // The valid pipeline is the sole ownership authority. Keeping payload out
    // of the reset/abort process avoids a high-fanout control on every data bit
    // and prevents set/reset inference when the first payload bit is constant.
    always_ff @(posedge i_clk) begin
        for (int stage = STAGES - 1; stage >= 0; stage--) begin
            if (ready[stage]) begin
                if (stage == 0) begin
                    if (i_valid && o_ready)
                        data_q[stage] <= i_data;
                end else if (valid_q[stage - 1]) begin
                    data_q[stage] <= data_q[stage - 1];
                end
            end
        end
    end

    initial begin
        if (DATA_WIDTH < 1 || STAGES < 1)
            $error("elastic pipeline requires positive width and depth");
    end
endmodule

`default_nettype wire
