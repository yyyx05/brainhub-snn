`timescale 1ns/1ps
module tb_supplement_h40_activity_monitor;
    logic clk = 0, rst_n = 0, aborting = 0, idle = 1;
    logic accumulator_valid = 0, accumulator_ready = 0;
    logic [39:0] lane_mask = 3, enable0 = 3, enable1 = 1, enable2 = 0, enable3 = 0;
    logic [1:0] mode = 0;
    logic issue_valid = 0, issue_ready = 0, region_valid = 0, region_ready = 0;
    logic observation_valid = 0, observation_ready = 0, fill_valid = 0, fill_ready = 0;
    supplement_h40_activity_monitor u_monitor (.*);
    always #5 clk = !clk;
    initial begin
        @(negedge clk); rst_n = 1;
        @(negedge clk); accumulator_valid = 1; accumulator_ready = 1; issue_valid = 1;
        @(negedge clk); accumulator_ready = 0;
        @(negedge clk); issue_valid = 0; region_valid = 1;
        @(negedge clk); region_valid = 0; observation_valid = 1;
        @(negedge clk); observation_valid = 0; fill_valid = 1; fill_ready = 1;
        @(negedge clk); fill_valid = 0; idle = 0;
        @(negedge clk); aborting = 1;
        @(negedge clk);
        assert(u_monitor.cycles == 8 && u_monitor.contributions == 1 &&
            u_monitor.lane_fires == 2 && u_monitor.enabled_terms == 3 &&
            u_monitor.bucket_compute == 1 && u_monitor.bucket_issue_wait == 1 &&
            u_monitor.bucket_region_wait == 1 && u_monitor.bucket_observation_wait == 1 &&
            u_monitor.bucket_fill == 1 && u_monitor.bucket_internal == 1 &&
            u_monitor.bucket_abort == 1 && u_monitor.bucket_idle == 1)
            else $fatal(1, "passive activity counters or bucket priority incorrect");
        $display("H40_ACTIVITY_UNIT_PASS");
        $finish;
    end
endmodule
