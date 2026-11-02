`timescale 1ns/1ps
`default_nettype none

// Control regression at the production store/learner boundary, not a forward model.
module tb_srnn_formal_m2_rearm_probe;
    localparam int BASE = 115200;
    localparam int ELEMENTS = 5120;
    logic clk = 0, rst_n = 0;
    logic cfg_valid = 0, cfg_ready, cfg_error, allow_cfg_error = 0;
    logic [7:0] profile = 2;
    logic [15:0] increment = 31, threshold = 65;
    logic load_valid = 0, load_ready;
    logic [16:0] load_addr = 0;
    logic signed [15:0] load_code = 0, load_residual = 0;
    logic forward_start = 0, forward_ready, forward_close = 0, forward_close_ready;
    logic learning_start = 0, learning_ready, learning_abort = 0;
    logic restore_complete = 0, restore_ready, recovery;
    logic contribution_valid = 0, contribution_ready, contribution_last = 0;
    logic [16:0] contribution_addr = 0;
    logic [7:0] contribution_sample = 0;
    logic signed [63:0] contribution_error = 0;
    logic signed [31:0] contribution_trace = 0;
    logic commit, detach, done, aborted, error;
    logic [31:0] commit_sequence;
    logic [7:0] model_version, adapt_version;
    logic signed [15:0] wh1 [0:BASE-1], initial_wout [0:ELEMENTS-1];
    logic signed [15:0] expected_code [0:8*ELEMENTS-1], expected_residual [0:8*ELEMENTS-1];
    logic signed [63:0] errors [0:8*40-1], gradients [0:8*ELEMENTS-1];
    logic signed [31:0] traces [0:8*512-1];
    integer cycles = 0, commits = 0, detaches = 0, completions = 0, aborts = 0;
    integer external_configs = 0, auto_configs = 0, residual_writes = 0;
    integer gradient_checks = 0, window_gradients = 0, active_timestep = 0;
    integer store_checks = 0;
    logic fault_armed = 0;
    integer read_faults = 0;
    logic backend_load_valid, backend_load_ready;
    logic [16:0] backend_load_addr;
    logic signed [15:0] backend_load_code, backend_load_residual;
    logic read_valid, read_ready, response_valid, response_ready, response_error;
    logic [16:0] read_addr;
    logic [15:0] read_tag, response_tag;
    logic signed [15:0] response_code, response_residual;
    logic write_valid, write_ready, write_code_enable, write_residual_enable;
    logic [16:0] write_addr;
    logic signed [15:0] write_code, write_residual;
    logic forward_read_valid, forward_read_ready, forward_response_valid, forward_response_ready;
    logic forward_response_error, store_error;
    logic [16:0] forward_read_addr;
    logic [15:0] forward_read_tag, forward_response_tag;
    logic signed [15:0] forward_response_code;
    string fixture;

    always #5 clk = ~clk;
    unified_srnn_m2_forward_learning_store_slice #(
        .USE_INTERNAL_STORE(0), .USE_RANDOM_ACCESS_FORWARD(1), .ENABLE_WH1_LEARNING(0),
        .STORE_DEPTH(120320), .BATCH_MAX(2), .HIDDEN_MAX(256), .BACKBONE_MAX(450),
        .SAMPLE_ID_WIDTH(8), .HIDDEN_INDEX_WIDTH(9), .BACKBONE_INDEX_WIDTH(9),
        .PARAM_ADDR_WIDTH(17), .TAG_WIDTH(16), .VERSION_WIDTH(8)
    ) dut (
        .i_clk(clk), .i_rst_n(rst_n), .i_clear_error(1'b0),
        .i_cfg_valid(cfg_valid), .o_cfg_ready(cfg_ready), .o_cfg_error(cfg_error),
        .i_cfg_profile_id(profile), .i_cfg_wh1_tensor_id(8'd16), .i_cfg_wh1_base(17'd0),
        .i_cfg_wh1_increment(16'd31), .i_cfg_wh1_threshold(16'd65),
        .i_cfg_wout_tensor_id(8'd17), .i_cfg_wout_base(17'(BASE)),
        .i_cfg_wout_increment(increment), .i_cfg_wout_threshold(threshold),
        .i_cfg_code_min(16'sh8000), .i_cfg_code_max(16'sh7fff), .i_cfg_batch_count(8'd2),
        .i_cfg_hidden_count(9'd256), .i_cfg_backbone_count(9'd450), .i_cfg_wout_element_count(17'(ELEMENTS)),
        .i_load_valid(load_valid), .o_load_ready(load_ready), .i_load_addr(load_addr),
        .i_load_code(load_code), .i_load_residual(load_residual),
        .i_forward_start(forward_start), .o_forward_start_ready(forward_ready),
        .i_forward_code_ready(1'b1), .i_forward_consumer_done_valid(forward_close),
        .o_forward_consumer_done_ready(forward_close_ready),
        .i_cache_read_valid(1'b0), .i_cache_read_addr('0), .i_cache_read_tag('0), .i_cache_response_ready(1'b1),
        .i_learning_start(learning_start), .o_learning_start_ready(learning_ready),
        .i_learning_abort(learning_abort), .i_restore_complete(restore_complete),
        .i_wh1_tp_contribution_valid(1'b0), .i_wh1_tp_contribution_path(1'b0),
        .i_wh1_tp_contribution_sample_id('0), .i_wh1_tp_contribution_hidden_id('0),
        .i_wh1_tp_contribution_column_id('0), .i_wh1_tp_error_q40('0),
        .i_wh1_tp_opposite_trace_q14('0), .i_wh1_tp_surrogate_uq24('0), .i_wh1_tp_contribution_last(1'b0),
        .i_wh1_backbone_valid(1'b0), .i_wh1_backbone_path(1'b0), .i_wh1_backbone_sample_id('0),
        .i_wh1_backbone_id('0), .i_wh1_backbone_q14('0), .i_wh1_backbone_last(1'b0),
        .i_wout_contribution_valid(contribution_valid), .o_wout_contribution_ready(contribution_ready),
        .i_wout_contribution_addr(contribution_addr), .i_wout_contribution_sample_id(contribution_sample),
        .i_wout_error_q40(contribution_error), .i_wout_hidden_trace_q14(contribution_trace),
        .i_wout_contribution_last(contribution_last), .o_learning_commit_pulse(commit),
        .o_detach_pulse(detach), .o_model_version(model_version), .o_adapt_version(adapt_version),
        .o_learning_commit_window_sequence(commit_sequence), .o_restore_ready(restore_ready),
        .o_recovery_required(recovery), .o_learning_done(done), .o_learning_aborted(aborted), .o_error(error),
        .o_store_load_valid(backend_load_valid), .i_store_load_ready(backend_load_ready),
        .o_store_load_addr(backend_load_addr), .o_store_load_code(backend_load_code),
        .o_store_load_residual(backend_load_residual),
        .o_store_learning_read_valid(read_valid), .i_store_learning_read_ready(read_ready),
        .o_store_learning_read_addr(read_addr), .o_store_learning_read_tag(read_tag),
        .i_store_learning_response_valid(response_valid), .o_store_learning_response_ready(response_ready),
        .i_store_learning_response_tag(response_tag), .i_store_learning_response_code(response_code),
        .i_store_learning_response_residual(response_residual),
        .i_store_learning_response_error(response_error || (fault_armed && response_valid)),
        .o_store_learning_write_valid(write_valid), .i_store_learning_write_ready(write_ready),
        .o_store_learning_write_addr(write_addr), .o_store_learning_write_code(write_code),
        .o_store_learning_write_residual(write_residual),
        .o_store_learning_write_code_enable(write_code_enable),
        .o_store_learning_write_residual_enable(write_residual_enable),
        .o_store_forward_read_valid(forward_read_valid), .i_store_forward_read_ready(forward_read_ready),
        .o_store_forward_read_addr(forward_read_addr), .o_store_forward_read_tag(forward_read_tag),
        .i_store_forward_response_valid(forward_response_valid), .o_store_forward_response_ready(forward_response_ready),
        .i_store_forward_response_tag(forward_response_tag), .i_store_forward_response_code(forward_response_code),
        .i_store_forward_response_error(forward_response_error), .i_store_protocol_error(store_error)
    );

    unified_tp_shared_forward_rmw_store #(.DEPTH(120320), .PARAM_ADDR_WIDTH(17), .TAG_WIDTH(16)) shared_store (
        .i_clk(clk), .i_rst_n(rst_n), .i_clear_error(1'b0),
        .i_load_valid(backend_load_valid), .o_load_ready(backend_load_ready), .i_load_addr(backend_load_addr),
        .i_load_code(backend_load_code), .i_load_residual(backend_load_residual),
        .i_learning_read_valid(read_valid), .o_learning_read_ready(read_ready),
        .i_learning_read_addr(read_addr), .i_learning_read_tag(read_tag),
        .o_learning_response_valid(response_valid), .i_learning_response_ready(response_ready),
        .o_learning_response_tag(response_tag), .o_learning_response_code(response_code),
        .o_learning_response_residual(response_residual), .o_learning_response_error(response_error),
        .i_learning_write_valid(write_valid), .o_learning_write_ready(write_ready), .i_learning_write_addr(write_addr),
        .i_learning_write_code(write_code), .i_learning_write_residual(write_residual),
        .i_learning_write_code_enable(write_code_enable), .i_learning_write_residual_enable(write_residual_enable),
        .i_forward_read_valid(forward_read_valid), .o_forward_read_ready(forward_read_ready),
        .i_forward_read_addr(forward_read_addr), .i_forward_read_tag(forward_read_tag),
        .o_forward_response_valid(forward_response_valid), .i_forward_response_ready(forward_response_ready),
        .o_forward_response_tag(forward_response_tag), .o_forward_response_code(forward_response_code),
        .o_forward_response_error(forward_response_error), .o_protocol_error(store_error)
    );

    always @(posedge clk) begin
        if (rst_n) begin
            cycles <= cycles + 1;
            if (cycles > 1000000 || (error && !allow_cfg_error))
                $fatal(1, "REARM_PROBE failure cycle=%0d error=%0b forward=%0d learner=%0d", cycles, error,
                    dut.forward_state_q, dut.g_wout_only_learning.learning_engine.state_q);
            if (cfg_valid && cfg_ready) external_configs <= external_configs + 1;
            if (dut.g_wout_only_learning.auto_cfg_fire) auto_configs <= auto_configs + 1;
            if (cfg_valid && cfg_ready && dut.g_wout_only_learning.auto_cfg_valid)
                $fatal(1, "REARM_PROBE automatic cfg raced accepted external cfg");
            if (dut.g_wout_only_learning.auto_cfg_valid && !dut.boundary_learning_active)
                $fatal(1, "REARM_PROBE reacquired without accepted learning request");
            if (recovery && dut.g_wout_only_learning.auto_cfg_valid)
                $fatal(1, "REARM_PROBE reacquired during dirty recovery");
            if (learning_start && learning_ready) window_gradients <= 0;
            if (dut.g_wout_only_learning.learning_engine.gradient_fire) begin
                if (dut.g_wout_only_learning.learning_engine.reducer_gradient_addr != 17'(window_gradients) ||
                    dut.g_wout_only_learning.learning_engine.reducer_gradient_q40 !==
                    gradients[active_timestep*ELEMENTS+window_gradients])
                    $fatal(1, "REARM_PROBE gradient t=%0d address=%0d", active_timestep, window_gradients);
                window_gradients <= window_gradients + 1;
                gradient_checks <= gradient_checks + 1;
            end
            if (dut.engine_write_residual_fire) residual_writes <= residual_writes + 1;
            if (fault_armed && response_valid && response_ready) begin
                fault_armed <= 0;
                read_faults <= read_faults + 1;
            end
            if (commit) begin
                if (commit_sequence != 32'(commits)) $fatal(1, "REARM_PROBE commit sequence reset");
                commits <= commits + 1;
            end
            if (detach) detaches <= detaches + 1;
            if (done) completions <= completions + 1;
            if (aborted) aborts <= aborts + 1;
        end
    end

    task automatic configure;
        @(negedge clk);
        cfg_valid = 1;
        do @(posedge clk); while (!cfg_ready);
        @(negedge clk);
        cfg_valid = 0;
    endtask

    task automatic load_element(input integer address, input logic signed [15:0] code_value,
                                input logic signed [15:0] residual_value);
        load_valid = 1;
        load_addr = 17'(address);
        load_code = code_value;
        load_residual = residual_value;
        do @(posedge clk); while (!load_ready);
        @(negedge clk);
        load_valid = 0;
    endtask

    task automatic open_forward;
        @(negedge clk);
        forward_start = 1;
        do @(posedge clk); while (!forward_ready);
        @(negedge clk);
        forward_start = 0;
    endtask

    task automatic close_forward;
        forward_close = 1;
        do @(posedge clk); while (!forward_close_ready);
        @(negedge clk);
        forward_close = 0;
    endtask

    task automatic start_learning(input integer timestep);
        active_timestep = timestep;
        @(negedge clk);
        learning_start = 1;
        do @(posedge clk); while (!learning_ready);
        @(negedge clk);
        learning_start = 0;
    endtask

    task automatic send_window(input integer timestep, input integer stop_mode);
        integer writes_before;
        writes_before = residual_writes;
        for (integer address = 0; address < ELEMENTS; address++) begin
            for (integer sample_id = 0; sample_id < 2; sample_id++) begin
                contribution_valid = 1;
                contribution_addr = 17'(address);
                contribution_sample = 8'(sample_id);
                contribution_error = errors[timestep*40+sample_id*20+address/256];
                contribution_trace = traces[timestep*512+sample_id*256+address%256];
                contribution_last = address == ELEMENTS-1 && sample_id == 1;
                do @(posedge clk); while (!contribution_ready && !(stop_mode == 2 && read_faults != 0));
                @(negedge clk);
                if ((stop_mode == 1 && residual_writes > writes_before) ||
                    (stop_mode == 2 && read_faults != 0)) begin
                    contribution_valid = 0;
                    contribution_last = 0;
                    return;
                end
            end
        end
        contribution_valid = 0;
        contribution_last = 0;
        if (stop_mode != 0) $fatal(1, "REARM_PROBE missing requested write/fault in real prefix");
    endtask

    task automatic check_store(input integer timestep);
        for (integer address = 0; address < BASE; address++) begin
            if (shared_store.code_mem[address] !== wh1[address] ||
                shared_store.residual_mem[address] !== 0)
                $fatal(1, "REARM_PROBE frozen Wh1 changed");
        end
        for (integer address = 0; address < ELEMENTS; address++) begin
            if (shared_store.code_mem[BASE+address] !== expected_code[timestep*ELEMENTS+address] ||
                shared_store.residual_mem[BASE+address] !== expected_residual[timestep*ELEMENTS+address])
                $fatal(1, "REARM_PROBE store t=%0d address=%0d", timestep, address);
            store_checks++;
        end
    endtask

    task automatic complete_window(input integer timestep);
        integer before_done;
        before_done = completions;
        start_learning(timestep);
        send_window(timestep, 0);
        wait (completions == before_done + 1);
        @(negedge clk);
        check_store(timestep);
        if (dut.g_wout_only_learning.learning_engine.update_active)
            $fatal(1, "REARM_PROBE context retained after commit");
        $display("M2_REARM_WINDOW timestep=%0d commits=%0d sequence=%0d auto_configs=%0d model=%0d adapt=%0d",
            timestep, commits, commit_sequence, auto_configs, model_version, adapt_version);
        $fflush();
    endtask

    task automatic abort_window;
        integer before_done;
        before_done = completions;
        learning_abort = 1;
        wait (completions == before_done + 1);
        @(negedge clk);
        learning_abort = 0;
        repeat (2) @(negedge clk);
        if (dut.g_wout_only_learning.learning_engine.update_active)
            $fatal(1, "REARM_PROBE context retained after abort");
    endtask

    initial begin : stimulus
        integer auto_before;
        integer before_abort_done;
        bit queued_cfg_accepted;
        logic [7:0] model_before, adapt_before;
        if (!$value$plusargs("FIXTURE=%s", fixture)) $fatal(1, "REARM_PROBE requires fixture");
        $readmemh({fixture, "/expected_on_error.mem"}, errors);
        $readmemh({fixture, "/expected_on_trace.mem"}, traces);
        $readmemh({fixture, "/expected_on_gradient.mem"}, gradients);
        $readmemh({fixture, "/expected_on_wout_after.mem"}, expected_code);
        $readmemh({fixture, "/expected_on_residual.mem"}, expected_residual);
        $readmemh("tb/data/srnn_m2_v28_wh1_initial_q14.mem", wh1);
        $readmemh("tb/data/srnn_m2_v28_wout_initial_q14.mem", initial_wout);
        repeat (4) @(negedge clk);
        rst_n = 1;
        allow_cfg_error = 1;
        threshold = 0;
        configure();
        if (!cfg_error || dut.g_wout_only_learning.saved_cfg_valid_q)
            $fatal(1, "REARM_PROBE invalid accepted cfg cached");
        threshold = 65;
        configure();
        repeat (2) @(negedge clk);
        allow_cfg_error = 0;
        for (integer address = 0; address < BASE; address++) load_element(address, wh1[address], 0);
        for (integer address = 0; address < ELEMENTS; address++) load_element(BASE+address, initial_wout[address], 0);
        open_forward(); close_forward();
        repeat (64) @(negedge clk);
        if (!cfg_ready || auto_configs != 0 || dut.g_wout_only_learning.learning_engine.update_active)
            $fatal(1, "REARM_PROBE initial cfg/off acquired a lease");
        configure();
        open_forward(); close_forward(); complete_window(0);
        auto_before = auto_configs;
        open_forward(); close_forward();
        repeat (64) begin
            @(negedge clk);
            if (auto_configs != auto_before || dut.g_wout_only_learning.learning_engine.update_active)
                $fatal(1, "REARM_PROBE off-forward reacquired a lease");
        end
        // Unaccepted bus changes must not replace the saved descriptor.
        profile = 8'he1; threshold = 0; increment = 16'd400;
        open_forward(); close_forward(); complete_window(1);
        if (dut.g_wout_only_learning.saved_cfg_q.profile_id != 2 ||
            dut.g_wout_only_learning.saved_cfg_q.increment != 31)
            $fatal(1, "REARM_PROBE unaccepted cfg captured");

        open_forward(); close_forward();
        repeat (2) @(negedge clk);
        auto_before = auto_configs;
        profile = 3; threshold = 65; increment = 31; cfg_valid = 1; learning_start = 1;
        do @(posedge clk); while (!cfg_ready);
        if (learning_ready) $fatal(1, "REARM_PROBE cfg and learning accepted together");
        @(negedge clk);
        cfg_valid = 0; learning_start = 0;
        if (dut.g_wout_only_learning.saved_cfg_q.profile_id != 3 ||
            auto_configs != auto_before || dut.g_wout_only_learning.rearm_pending_q)
            $fatal(1, "REARM_PROBE external cfg did not replace pending replay");
        open_forward(); close_forward(); complete_window(2);

        open_forward(); close_forward(); start_learning(3);
        // A cfg queued after start acceptance must not deadlock rearm/abort.
        before_abort_done = completions;
        cfg_valid = 1;
        learning_abort = 1;
        while (completions != before_abort_done + 1 || cfg_valid) begin
            @(posedge clk);
            queued_cfg_accepted = cfg_valid && cfg_ready;
            @(negedge clk);
            if (queued_cfg_accepted) cfg_valid = 0;
            if (completions == before_abort_done + 1) learning_abort = 0;
        end
        repeat (2) @(negedge clk);
        if (dut.g_wout_only_learning.learning_engine.update_active)
            $fatal(1, "REARM_PROBE queued cfg retained a lease after abort");
        if (recovery || commits != 3 || detaches != 3) $fatal(1, "REARM_PROBE clean abort boundary");
        open_forward(); close_forward(); complete_window(3);

        model_before = model_version; adapt_before = adapt_version;
        open_forward(); close_forward(); start_learning(4);
        send_window(4, 1); abort_window();
        if (!recovery || !restore_ready || commits != 4 || detaches != 4 ||
            model_version != model_before || adapt_version != adapt_before)
            $fatal(1, "REARM_PROBE dirty abort changed committed boundary");
        auto_before = auto_configs;
        forward_start = 1;
        repeat (64) begin
            @(negedge clk);
            if (forward_ready || auto_configs != auto_before || !recovery)
                $fatal(1, "REARM_PROBE dirty recovery bypassed");
        end
        forward_start = 0;
        for (integer address = 0; address < ELEMENTS; address++)
            load_element(BASE+address, expected_code[3*ELEMENTS+address], expected_residual[3*ELEMENTS+address]);
        check_store(3);
        restore_complete = 1;
        @(negedge clk);
        restore_complete = 0;
        if (model_version != model_before || adapt_version != adapt_before)
            $fatal(1, "REARM_PROBE restore changed versions");
        open_forward(); close_forward(); complete_window(4);

        model_before = model_version; adapt_before = adapt_version;
        before_abort_done = completions;
        allow_cfg_error = 1;
        fault_armed = 1;
        open_forward(); close_forward(); start_learning(5);
        send_window(5, 2);
        wait (completions == before_abort_done + 1);
        repeat (2) @(negedge clk);
        if (!error || recovery || read_faults != 1 ||
            dut.g_wout_only_learning.learning_engine.update_active ||
            model_version != model_before || adapt_version != adapt_before)
            $fatal(1, "REARM_PROBE first-read fault did not clean-abort/release");
        check_store(4);
        configure();
        repeat (2) @(negedge clk);
        if (error || dut.g_wout_only_learning.learning_engine.update_active)
            $fatal(1, "REARM_PROBE accepted legal cfg failed to clear fault without lease");
        allow_cfg_error = 0;
        open_forward(); close_forward(); complete_window(5);
        if (commits != 6 || detaches != 6 || aborts != 3 || auto_configs != 9 || external_configs != 6)
            $fatal(1, "REARM_PROBE incomplete coverage commits=%0d aborts=%0d auto=%0d external=%0d",
                commits, aborts, auto_configs, external_configs);
        $display("SRNN_M2_REARM_PROBE_PASS windows=6 commits=%0d detach=%0d aborts=%0d auto_configs=%0d external_configs=%0d invalid_cfg=1 cfg_competition=1 cfg_pending_during_rearm=1 initial_cfg_off_cfg=1 learning_off_cfg=1 clean_abort=1 dirty_abort=1 restore=1 first_read_fault=1 legal_cfg_error_clear=1 recovery_blocked_cycles=64 idle_no_rearm_cycles=64 gradient_checks=%0d store_checks=%0d model=%0d adapt=%0d cycles=%0d",
            commits, detaches, aborts, auto_configs, external_configs, gradient_checks,
            store_checks, model_version, adapt_version, cycles);
        $finish;
    end
endmodule
`default_nettype wire
