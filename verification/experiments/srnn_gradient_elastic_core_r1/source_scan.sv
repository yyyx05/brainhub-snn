`timescale 1ns/1ps
`default_nettype none

// Experimental count-preserving four-source scan. Both policies pay the same
// 175-word ingress and block inspection; only empty-block emission differs.
module source_scan (
    input wire clk, rst_n, sparse,
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
    localparam logic [2:0] LOAD=0, SCAN=1, EMIT=2, FINISH=3, DRAIN=4;
    logic [31:0] frame [0:174];
    logic [7:0] load_q, block_q, repeat_q, maximum_q;
    logic [31:0] block_counts_q;
    logic any_q, sparse_q, first_q;
    logic [7:0] scan_max;
    always_comb begin
        scan_max=0;
        for (int lane=0; lane<4; lane++)
            if (frame[block_q][lane*8+:8]>scan_max)
                scan_max=frame[block_q][lane*8+:8];
        count_ready=phase==LOAD;
        group_valid=phase==EMIT || phase==FINISH;
        first=first_q;
        last=phase==FINISH;
        mask=0;
        row=10'(block_q)*10'd2;
        if (phase==EMIT)
            for (int lane=0; lane<4; lane++)
                mask[lane]=block_counts_q[lane*8+:8]>repeat_q;
    end
    always_ff @(posedge clk) begin
        if (rst_n && phase==LOAD && count_valid && count_ready) frame[load_q]<=counts;
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase<=LOAD; load_q<=0; block_q<=0; repeat_q<=0;
            maximum_q<=0; block_counts_q<=0; any_q<=0;
            sparse_q<=0; first_q<=1;
        end else case (phase)
            LOAD: if (count_valid && count_ready) begin
                any_q<=any_q || (|counts);
                if (load_q==0) sparse_q<=sparse;
                if (load_q==174) begin
                    phase<=(any_q || (|counts)) ? SCAN : FINISH;
                    block_q<=0;
                end else load_q<=load_q+1'b1;
            end
            SCAN: begin
                block_counts_q<=frame[block_q]; repeat_q<=0;
                maximum_q<=scan_max==0 ? 8'd1 : scan_max;
                if (sparse_q && scan_max==0) begin
                    if (block_q==174) phase<=FINISH;
                    else block_q<=block_q+1'b1;
                end else phase<=EMIT;
            end
            EMIT: if (group_valid && group_ready) begin
                first_q<=0;
                if (repeat_q==maximum_q-1'b1) begin
                    if (block_q==174) phase<=FINISH;
                    else begin block_q<=block_q+1'b1; phase<=SCAN; end
                end else repeat_q<=repeat_q+1'b1;
            end
            FINISH: if (group_valid && group_ready) begin
                first_q<=0; phase<=DRAIN;
            end
            DRAIN: if (result_fire) begin
                phase<=LOAD; load_q<=0; any_q<=0; first_q<=1;
            end
            default: phase<=LOAD;
        endcase
    end
endmodule
`default_nettype wire
