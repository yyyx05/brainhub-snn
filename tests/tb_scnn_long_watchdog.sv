`timescale 1ns/1ps
module tb_scnn_long_watchdog;
    bit clk = 0;
    bit short_done = 0;
    always #5 clk = ~clk;
    initial begin
        repeat (3) repeat (1000000000) @(posedge clk);
        $fatal(1, "long watchdog terminated prematurely");
    end
    initial begin
        repeat (2) repeat (8) @(posedge clk);
        short_done = 1;
    end
    initial begin
        repeat (32) @(negedge clk);
        if (!short_done) $fatal(1, "short watchdog did not terminate");
        $display("SCNN_LONG_WATCHDOG_PASS elapsed=%0t", $time);
        $finish;
    end
endmodule
