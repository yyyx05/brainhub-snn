`timescale 1ns/1ps
`default_nettype none

// Interface-only diagnostic. Real B2T2 tensors, full Wout size, TB backing RAM.
module tb_srnn_formal_m2_context_probe;
    localparam int ELEMENTS = 5120;
    localparam int BASE = 115200;
    logic clk = 0, rst_n = 0;
    logic cfg_valid = 0, cfg_ready, cfg_error;
    logic start = 0, start_ready;
    logic contribution_valid = 0, contribution_ready, contribution_last = 0;
    logic [16:0] contribution_addr = 0;
    logic [1:0] contribution_sample = 0;
    logic signed [63:0] contribution_error = 0;
    logic signed [31:0] contribution_trace = 0;
    logic read_valid, read_ready, response_valid = 0, response_ready;
    logic [16:0] read_addr, result_addr;
    logic signed [15:0] response_code = 0, response_residual = 0;
    logic result_valid, result_code_enable, result_residual_enable;
    logic signed [15:0] result_code, result_residual;
    logic commit_valid, done, busy, error, aborted;
    logic [31:0] commit_sequence;
    logic [17:0] zero_count, update_count;
    logic signed [63:0] errors [0:79], gradients [0:10239];
    logic signed [31:0] traces [0:1023];
    logic signed [15:0] code_mem [0:ELEMENTS-1], residual_mem [0:ELEMENTS-1];
    logic signed [15:0] expected_code [0:10239], expected_residual [0:10239];
    integer cycles = 0, configs = 0, starts = 0, contributions = 0;
    integer gradient_checks = 0, commits = 0, done_count = 0;
    integer reads = 0, code_writes = 0, residual_writes = 0, store_checks = 0;
    integer held_start_accepts = 0, active_window = 0;
    bit measuring_held_start = 0;
    string fixture;

    always #5 clk = ~clk;
    assign read_ready = !response_valid || response_ready;

    unified_srnn_tp_wout_direct_online_learning_engine #(
        .PARAM_ADDR_WIDTH(17), .SAMPLE_ID_WIDTH(2)
    ) dut (
        .i_clk(clk), .i_rst_n(rst_n), .i_cfg_valid(cfg_valid), .o_cfg_ready(cfg_ready),
        .i_cfg_profile_id(8'd2), .i_cfg_wout_tensor_id(8'd17),
        .i_cfg_wout_base(17'(BASE)), .i_cfg_wout_increment(16'd31),
        .i_cfg_wout_threshold(16'd65), .i_cfg_code_min(16'sh8000),
        .i_cfg_code_max(16'sh7fff), .i_cfg_batch_count(2'd2),
        .i_cfg_wout_element_count(17'(ELEMENTS)), .o_cfg_error(cfg_error),
        .i_start_window(start), .o_start_ready(start_ready), .i_abort_window(1'b0),
        .i_wout_contribution_valid(contribution_valid),
        .o_wout_contribution_ready(contribution_ready),
        .i_wout_contribution_addr(contribution_addr),
        .i_wout_contribution_sample_id(contribution_sample),
        .i_wout_error_q40(contribution_error), .i_wout_hidden_trace_q14(contribution_trace),
        .i_wout_contribution_last(contribution_last),
        .o_read_valid(read_valid), .i_read_ready(read_ready),
        .o_read_profile_id(), .o_read_tensor_id(), .o_read_parameter_addr(read_addr),
        .o_read_tag(), .i_read_response_valid(response_valid),
        .o_read_response_ready(response_ready), .i_read_response_code(response_code),
        .i_read_response_residual(response_residual), .i_read_response_error(1'b0),
        .o_result_valid(result_valid), .i_result_ready(1'b1),
        .o_result_profile_id(), .o_result_tensor_id(), .o_result_parameter_addr(result_addr),
        .o_result_tag(), .o_result_code(result_code), .o_result_residual(result_residual),
        .o_write_code_enable(result_code_enable), .o_write_residual_enable(result_residual_enable),
        .o_write_code_fire(), .o_write_residual_fire(), .o_dirty(), .o_saturated(),
        .o_commit_valid(commit_valid), .i_commit_ready(1'b1), .o_commit_profile_id(),
        .o_commit_window_sequence(commit_sequence), .o_busy(busy), .o_done(done),
        .o_window_aborted(aborted), .o_error(error),
        .o_zero_gradient_count(zero_count), .o_update_count(update_count)
    );

    always @(posedge clk) begin
        if (rst_n) begin
            cycles <= cycles + 1;
            if (cycles > 150000 || error || cfg_error || aborted)
                $fatal(1, "CONTEXT_PROBE failure cycles=%0d state=%0d error=%0b", cycles, dut.state_q, error);
            if (cfg_valid && cfg_ready) configs <= configs + 1;
            if (start && start_ready) begin
                starts <= starts + 1;
                if (measuring_held_start) held_start_accepts <= held_start_accepts + 1;
            end
            if (contribution_valid && contribution_ready) contributions <= contributions + 1;
            if (dut.reducer_gradient_valid && dut.reducer_gradient_ready) begin
                if (dut.reducer_gradient_addr != 17'(gradient_checks % ELEMENTS) ||
                    dut.reducer_gradient_q40 !== gradients[gradient_checks])
                    $fatal(1, "CONTEXT_PROBE gradient index=%0d", gradient_checks);
                gradient_checks <= gradient_checks + 1;
            end
            if (response_valid && response_ready) response_valid <= 0;
            if (read_valid && read_ready) begin
                if (read_addr < BASE || read_addr >= BASE + ELEMENTS)
                    $fatal(1, "CONTEXT_PROBE read escaped Wout");
                response_code <= code_mem[int'(read_addr)-BASE];
                response_residual <= residual_mem[int'(read_addr)-BASE];
                response_valid <= 1;
                reads <= reads + 1;
            end
            if (result_valid) begin
                if (result_addr < BASE || result_addr >= BASE + ELEMENTS)
                    $fatal(1, "CONTEXT_PROBE result escaped Wout");
                if (result_code_enable) begin
                    code_mem[int'(result_addr)-BASE] <= result_code;
                    code_writes <= code_writes + 1;
                end
                if (result_residual_enable) begin
                    residual_mem[int'(result_addr)-BASE] <= result_residual;
                    residual_writes <= residual_writes + 1;
                end
            end
            if (commit_valid) begin
                if (commit_sequence != 32'(commits)) $fatal(1, "CONTEXT_PROBE commit sequence");
                commits <= commits + 1;
            end
            if (done) done_count <= done_count + 1;
        end
    end

    task automatic configure;
        @(negedge clk);
        cfg_valid = 1;
        do @(posedge clk); while (!cfg_ready);
        @(negedge clk);
        cfg_valid = 0;
    endtask

    task automatic run_window(input integer timestep);
        active_window = timestep;
        @(negedge clk);
        start = 1;
        do @(posedge clk); while (!start_ready);
        @(negedge clk);
        start = 0;
        for (integer address = 0; address < ELEMENTS; address++) begin
            for (integer sample_id = 0; sample_id < 2; sample_id++) begin
                contribution_valid = 1;
                contribution_addr = 17'(address);
                contribution_sample = 2'(sample_id);
                contribution_error = errors[timestep*40 + sample_id*20 + address/256];
                contribution_trace = traces[timestep*512 + sample_id*256 + address%256];
                contribution_last = address == ELEMENTS-1 && sample_id == 1;
                do @(posedge clk); while (!contribution_ready);
                @(negedge clk);
            end
        end
        contribution_valid = 0;
        contribution_last = 0;
        wait (done_count == timestep + 1);
        @(negedge clk);
        for (integer address = 0; address < ELEMENTS; address++) begin
            if (code_mem[address] !== expected_code[timestep*ELEMENTS+address] ||
                residual_mem[address] !== expected_residual[timestep*ELEMENTS+address])
                $fatal(1, "CONTEXT_PROBE store window=%0d address=%0d", timestep, address);
            store_checks++;
        end
        $display("M2_CONTEXT_WINDOW timestep=%0d gradients=%0d commits=%0d zero_count=%0d update_count=%0d state=%0d configured=%0b update_active=%0b cfg_ready=%0b start_ready=%0b",
            timestep, gradient_checks, commits, zero_count, update_count, dut.state_q,
            dut.configured_q, dut.update_active, cfg_ready, start_ready);
        $fflush();
    endtask

    initial begin
        if (!$value$plusargs("FIXTURE=%s", fixture)) $fatal(1, "CONTEXT_PROBE requires fixture");
        $readmemh({fixture, "/expected_on_error.mem"}, errors);
        $readmemh({fixture, "/expected_on_trace.mem"}, traces);
        $readmemh({fixture, "/expected_on_gradient.mem"}, gradients);
        $readmemh({fixture, "/expected_on_wout_after.mem"}, expected_code);
        $readmemh({fixture, "/expected_on_residual.mem"}, expected_residual);
        $readmemh("tb/data/srnn_m2_v28_wout_initial_q14.mem", code_mem);
        for (integer address = 0; address < ELEMENTS; address++) residual_mem[address] = 0;
        repeat (4) @(negedge clk);
        rst_n = 1;
        configure();
        run_window(0);
        if (dut.state_q != 0 || !dut.configured_q || !dut.reducer_start_ready ||
            dut.update_active || !cfg_ready || start_ready || busy)
            $fatal(1, "CONTEXT_PROBE unexpected post-commit context");
        measuring_held_start = 1;
        start = 1;
        repeat (64) @(negedge clk);
        start = 0;
        measuring_held_start = 0;
        if (held_start_accepts != 0 || starts != 1 || commits != 1 || dut.update_active)
            $fatal(1, "CONTEXT_PROBE start unexpectedly reacquired context");
        $display("M2_CONTEXT_NO_CFG held_start_cycles=64 accepted=%0d configured=%0b reducer_ready=%0b update_active=%0b cfg_ready=%0b",
            held_start_accepts, dut.configured_q, dut.reducer_start_ready, dut.update_active, cfg_ready);
        $fflush();
        configure();
        @(negedge clk);
        if (!dut.update_active || !start_ready)
            $fatal(1, "CONTEXT_PROBE cfg did not reacquire context active=%0b ready=%0b cfg=%0b", dut.update_active, start_ready, cfg_valid);
        run_window(1);
        if (configs != 2 || starts != 2 || commits != 2 || gradient_checks != 10240 ||
            contributions != 20480 || reads != 0 || code_writes != 0 || residual_writes != 0)
            $fatal(1, "CONTEXT_PROBE accounting differs from real zero-gradient B2T2 prefix");
        $display("SRNN_M2_CONTEXT_PROBE_PASS windows=2 held_start_cycles=64 held_start_accepts=%0d configs=%0d starts=%0d gradients=%0d commits=%0d contributions=%0d store_checks=%0d reads=%0d code_writes=%0d residual_writes=%0d cycles=%0d",
            held_start_accepts, configs, starts, gradient_checks, commits, contributions,
            store_checks, reads, code_writes, residual_writes, cycles);
        $finish;
    end
endmodule
`default_nettype wire
