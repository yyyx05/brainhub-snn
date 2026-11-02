`timescale 1ns/1ps
`default_nettype none

// Isolated M2-only experiment. Context0 and the accepted full core are unchanged.
module m2_compact_store #(
    parameter int DEPTH = 578
) (
    input wire logic i_clk, i_rst_n,
    input wire logic i_fill_valid,
    output logic o_fill_ready,
    input wire logic [1:0] i_fill_bank,
    input wire logic [9:0] i_fill_addr,
    input wire logic [199:0] i_fill_data,
    input wire logic i_read_valid,
    output logic o_read_ready,
    input wire logic [9:0] i_read_addr,
    output logic o_read_result_valid,
    input wire logic i_read_result_ready,
    output logic [999:0] o_term0_flat
);
    localparam int AW = $clog2(DEPTH);
    logic read_fire, fill_fire;
    logic [639:0] codes;
    assign o_read_ready = !o_read_result_valid || i_read_result_ready;
    assign o_fill_ready = !i_read_valid;
    assign read_fire = i_read_valid && o_read_ready;
    assign fill_fire = i_fill_valid && o_fill_ready;

    for (genvar bank = 0; bank < 4; bank++) begin : g_bank
        localparam int W = bank == 3 ? 64 : 192;
        logic [W-1:0] word_q;
        wire write_fire = fill_fire && i_fill_bank == 2'(bank);
`ifdef UNIFIED_USE_XPM_MEMORY
        xpm_memory_sdpram #(
            .MEMORY_SIZE(DEPTH * W), .MEMORY_PRIMITIVE("block"),
            .CLOCKING_MODE("common_clock"), .ECC_MODE("no_ecc"),
            .MEMORY_INIT_FILE("none"), .MEMORY_INIT_PARAM(""),
            .USE_MEM_INIT(0), .USE_MEM_INIT_MMI(0),
            .WAKEUP_TIME("disable_sleep"), .AUTO_SLEEP_TIME(0),
            .MESSAGE_CONTROL(0), .USE_EMBEDDED_CONSTRAINT(0),
            .MEMORY_OPTIMIZATION("true"), .CASCADE_HEIGHT(0),
            .SIM_ASSERT_CHK(0), .WRITE_PROTECT(1), .IGNORE_INIT_SYNTH(0),
            .WRITE_DATA_WIDTH_A(W), .BYTE_WRITE_WIDTH_A(W),
            .ADDR_WIDTH_A(AW), .READ_DATA_WIDTH_B(W), .ADDR_WIDTH_B(AW),
            .READ_RESET_VALUE_B("0"), .READ_LATENCY_B(1),
            .WRITE_MODE_B("read_first"), .RST_MODE_B("SYNC")
        ) u_memory (
            .sleep(1'b0), .clka(i_clk), .ena(write_fire), .wea(write_fire),
            .addra(i_fill_addr[AW-1:0]), .dina(i_fill_data[W-1:0]),
            .injectsbiterra(1'b0), .injectdbiterra(1'b0),
            .clkb(i_clk), .rstb(1'b0), .enb(read_fire), .regceb(1'b1),
            .addrb(i_read_addr[AW-1:0]), .doutb(word_q),
            .sbiterrb(), .dbiterrb()
        );
`else
        (* ram_style = "block" *) logic [W-1:0] memory [0:DEPTH-1];
        always_ff @(posedge i_clk) begin
            if (write_fire) memory[i_fill_addr] <= i_fill_data[W-1:0];
            if (read_fire) word_q <= memory[i_read_addr];
        end
`endif
        assign codes[bank*192 +: W] = word_q;
    end
    for (genvar lane = 0; lane < 40; lane++) begin : g_lane
        assign o_term0_flat[lane*25 +: 25] =
            {{9{codes[lane*16+15]}}, codes[lane*16 +: 16]};
    end
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) o_read_result_valid <= 1'b0;
        else if (read_fire) o_read_result_valid <= 1'b1;
        else if (i_read_result_ready) o_read_result_valid <= 1'b0;
    end
`ifndef SYNTHESIS
    always_ff @(posedge i_clk) begin
        if (i_rst_n && read_fire && int'(i_read_addr) >= DEPTH)
            $fatal(1, "M2 compact read outside declared capacity");
        if (i_rst_n && fill_fire && int'(i_fill_addr) >= DEPTH)
            $fatal(1, "M2 compact fill outside declared capacity");
    end
`endif
endmodule
`default_nettype wire
