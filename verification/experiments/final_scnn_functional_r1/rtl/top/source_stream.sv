`timescale 1ns/1ps
`default_nettype none
module source_stream (
    input wire clk, rst_n,
    input wire count_valid,
    output logic count_ready,
    input wire [31:0] counts,
    output logic group_valid,
    input wire group_ready,
    output logic first, last,
    output logic [3:0] mask,
    output logic [9:0] row,
    input wire result_fire,
    output logic [2:0] phase
);
    localparam logic [2:0] LOAD=0, EMIT=2, FINISH=3, DRAIN=4;
    logic [7:0] block_q, repeat_q, maximum_q, incoming_max;
    logic [31:0] block_counts_q;
    logic first_q;
    always_comb begin
        incoming_max=0;
        for (int lane=0; lane<4; lane++)
            if (counts[lane*8+:8]>incoming_max)
                incoming_max=counts[lane*8+:8];
        count_ready=phase==LOAD;
        group_valid=phase==EMIT || phase==FINISH;
        first=first_q;
        last=phase==FINISH;
        row={1'b0,block_q,1'b0};
        mask=0;
        if (phase==EMIT)
            for (int lane=0; lane<4; lane++)
                mask[lane]=block_counts_q[lane*8+:8]>repeat_q;
    end
    // Ingress is backpressured while repeating a nonzero block. No full-frame
    // buffer, replay, input reordering, or unaccounted host packing is used.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase<=LOAD; block_q<=0; repeat_q<=0; maximum_q<=0;
            block_counts_q<=0; first_q<=1;
        end else case (phase)
            LOAD: if (count_valid && count_ready) begin
                if (incoming_max!=0) begin
                    block_counts_q<=counts; maximum_q<=incoming_max;
                    repeat_q<=0; phase<=EMIT;
                end else if (block_q==174) phase<=FINISH;
                else block_q<=block_q+1'b1;
            end
            EMIT: if (group_valid && group_ready) begin
                first_q<=0;
                if (repeat_q==maximum_q-1'b1) begin
                    if (block_q==174) phase<=FINISH;
                    else begin block_q<=block_q+1'b1; phase<=LOAD; end
                end else repeat_q<=repeat_q+1'b1;
            end
            FINISH: if (group_valid && group_ready) begin
                first_q<=0; phase<=DRAIN;
            end
            DRAIN: if (result_fire) begin
                phase<=LOAD; block_q<=0; first_q<=1;
            end
            default: phase<=LOAD;
        endcase
    end
endmodule

module source_scan_experiment (
    input wire clk, rst_n, sparse, streaming,
    input wire count_valid,
    output logic count_ready,
    input wire [31:0] counts,
    output logic group_valid,
    input wire group_ready,
    output logic first, last,
    output logic [3:0] mask,
    output logic [9:0] row,
    input wire result_fire,
    output logic [2:0] phase
);
    wire [1:0] ready_i, valid_i, first_i, last_i;
    wire [1:0][3:0] mask_i;
    wire [1:0][9:0] row_i;
    wire [1:0][2:0] phase_i;
    source_scan buffered_i(.clk(clk),.rst_n(rst_n),.sparse(sparse),
        .count_valid(count_valid && !streaming),.count_ready(ready_i[0]),.counts(counts),
        .group_valid(valid_i[0]),.group_ready(group_ready && !streaming),
        .first(first_i[0]),.last(last_i[0]),.mask(mask_i[0]),.row(row_i[0]),
        .result_fire(result_fire && !streaming),.phase(phase_i[0]));
    source_stream streaming_i(.clk(clk),.rst_n(rst_n),
        .count_valid(count_valid && streaming),.count_ready(ready_i[1]),.counts(counts),
        .group_valid(valid_i[1]),.group_ready(group_ready && streaming),
        .first(first_i[1]),.last(last_i[1]),.mask(mask_i[1]),.row(row_i[1]),
        .result_fire(result_fire && streaming),.phase(phase_i[1]));
    assign count_ready=ready_i[streaming];
    assign group_valid=valid_i[streaming];
    assign first=first_i[streaming];
    assign last=last_i[streaming];
    assign mask=mask_i[streaming];
    assign row=row_i[streaming];
    assign phase=phase_i[streaming];
endmodule

module scan_cost_top #(parameter integer POLICY=0) (
    input wire clk, rst_n,
    input wire count_valid,
    output logic count_ready,
    input wire [31:0] counts,
    output logic group_valid,
    input wire group_ready,
    output logic first, last,
    output logic [3:0] mask,
    output logic [9:0] row,
    input wire result_fire,
    output logic [2:0] phase
);
    generate if (POLICY==2) begin: g_stream
        source_stream producer(.*);
    end else begin: g_buffered
        source_scan producer(.sparse(POLICY==1),.*);
    end endgenerate
endmodule
`default_nettype wire
