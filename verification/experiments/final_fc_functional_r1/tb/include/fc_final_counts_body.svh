    localparam int B = 16;
    localparam int H = 160;
    localparam int C = 10;
    localparam int F = 2312;
    localparam int W2_BASE = F*H;
    logic [15:0] input_counts [0:100*B*F-1];
    logic signed [15:0] parameters [0:371519];
    logic [3:0] labels [0:B-1];
    logic signed [31:0] expected_pre [0:100*B*H-1];
    logic signed [31:0] expected_trace [0:100*B*H-1];
    logic expected_spike [0:100*B*H-1];
    logic signed [31:0] expected_dot [0:100*B*C-1];
    logic signed [15:0] expected_q6 [0:100*B*C-1];
    logic signed [15:0] expected_w2 [0:100*H*C-1];
    logic signed [15:0] expected_residual [0:100*H*C-1];
    logic [H-1:0] hidden_spikes;
    integer steps, learning, timestep, sample_index, phase = 0;
    integer batches = 1, batch_index = 0, window_index = 0;
    integer batch_steps [0:9];
    integer total_steps = 0;
    integer hidden_checks = 0, dot_checks = 0, q6_checks = 0;
    integer commits = 0, layer_done_count = 0, capture_done_count = 0;
    integer replay_done_count = 0;
    integer input_events = 0, hidden_events = 0, groups = 0;
    integer learning_reads = 0, code_writes = 0, residual_writes = 0, forward_reads = 0;
    integer scale_q16 = 47303;
    longint cycles = 0, start_cycle, load_cycles, init_cycles;
    longint w0_cycles = 0, w2_cycles = 0, learning_cycles = 0, refill_cycles = 0;
    string fixture, batch_fixture;

    always #5 i_clk = ~i_clk;
    always @(posedge i_clk) begin
        cycles <= cycles+1;
        if (cycles > 100000000) $fatal(1, "FC final watchdog phase=%0d t=%0d s=%0d", phase,timestep,sample_index);
        if (i_rst_n) begin
            if (o_sticky_error || o_backend_error || i_store_protocol_error || o_learning_fc_error ||
                o_cache_refill_sticky_error || o_fc_srnn_sticky_error || o_fc_srnn_learning_sticky_error ||
                o_fc_srnn_observation_sticky_error || o_array_tile_sticky_error != 0)
                $fatal(1, "FC final error phase=%0d t=%0d s=%0d global=%0b backend=%0b store=%0b fc=%0b cache=%0b layer=%0b/%h ingress=%0b/%h tile=%h",
                    phase,timestep,sample_index,o_sticky_error,o_backend_error,i_store_protocol_error,
                    o_learning_fc_error,o_cache_refill_sticky_error,o_fc_srnn_sticky_error,o_fc_srnn_error_code,
                    o_fc_srnn_learning_sticky_error,o_fc_srnn_learning_error_code,o_array_tile_sticky_error);
            if (o_fc_srnn_layer_done) layer_done_count++;
            if (o_fc_srnn_learning_capture_done) capture_done_count++;
            if (o_fc_srnn_learning_done) replay_done_count++;
            if (o_learning_fc_window_commit_valid && i_learning_fc_window_commit_ready) commits++;
            if (o_store_forward_read_valid && i_store_forward_read_ready) forward_reads++;
            if (o_store_learning_read_valid && i_store_learning_read_ready) learning_reads++;
            if (o_store_learning_write_valid && i_store_learning_write_ready) begin
                if (o_store_learning_write_addr < W2_BASE || o_store_learning_write_addr >= W2_BASE+H*C)
                    $fatal(1, "FC frozen/out-of-range parameter write %0d", o_store_learning_write_addr);
                if (!learning) $fatal(1, "FC learning-off parameter write");
                if (o_store_learning_write_code_enable) code_writes++;
                if (o_store_learning_write_residual_enable) residual_writes++;
                if (o_store_learning_write_code_enable && o_store_learning_write_code !=
                    expected_w2[timestep*H*C+int'(o_store_learning_write_addr)-W2_BASE])
                    $fatal(1, "FC updated code mismatch t=%0d addr=%0d got=%0d exp=%0d", timestep,
                        o_store_learning_write_addr,o_store_learning_write_code,
                        expected_w2[timestep*H*C+int'(o_store_learning_write_addr)-W2_BASE]);
                if (o_store_learning_write_residual_enable && o_store_learning_write_residual !=
                    expected_residual[timestep*H*C+int'(o_store_learning_write_addr)-W2_BASE])
                    $fatal(1, "FC residual mismatch t=%0d addr=%0d got=%0d exp=%0d", timestep,
                        o_store_learning_write_addr,o_store_learning_write_residual,
                        expected_residual[timestep*H*C+int'(o_store_learning_write_addr)-W2_BASE]);
            end
            if (o_fc_srnn_observation_valid && i_fc_srnn_observation_ready) begin
                if (phase == 1) begin
                    if (int'(o_fc_srnn_observation_neuron_id) >= H) $fatal(1, "hidden id range");
                    if (o_fc_srnn_observation_pre_state != expected_pre[(timestep*B+sample_index)*H+int'(o_fc_srnn_observation_neuron_id)] ||
                        o_fc_srnn_observation_trace_next != expected_trace[(timestep*B+sample_index)*H+int'(o_fc_srnn_observation_neuron_id)] ||
                        o_fc_srnn_observation_spike != expected_spike[(timestep*B+sample_index)*H+int'(o_fc_srnn_observation_neuron_id)])
                        $fatal(1, "hidden mismatch t=%0d s=%0d h=%0d pre=%0d/%0d trace=%0d/%0d spike=%0b/%0b", timestep,sample_index,
                            o_fc_srnn_observation_neuron_id,o_fc_srnn_observation_pre_state,
                            expected_pre[(timestep*B+sample_index)*H+int'(o_fc_srnn_observation_neuron_id)],
                            o_fc_srnn_observation_trace_next,expected_trace[(timestep*B+sample_index)*H+int'(o_fc_srnn_observation_neuron_id)],
                            o_fc_srnn_observation_spike,expected_spike[(timestep*B+sample_index)*H+int'(o_fc_srnn_observation_neuron_id)]);
                    hidden_spikes[o_fc_srnn_observation_neuron_id] = o_fc_srnn_observation_spike;
                    hidden_checks++;
                end else if (phase == 2) begin
                    if (o_fc_srnn_observation_neuron_id >= C)
                        $fatal(1, "output id range %0d", o_fc_srnn_observation_neuron_id);
                    if (o_fc_srnn_observation_pre_state != expected_dot[(timestep*B+sample_index)*C+int'(o_fc_srnn_observation_neuron_id)])
                        $fatal(1, "dot mismatch t=%0d s=%0d c=%0d got=%0d exp=%0d", timestep,sample_index,
                            o_fc_srnn_observation_neuron_id,o_fc_srnn_observation_pre_state,
                            expected_dot[(timestep*B+sample_index)*C+int'(o_fc_srnn_observation_neuron_id)]);
                    dot_checks++;
                end else $fatal(1, "observation outside forward");
            end
            // Read-only tap of the production router output, before the learner.
            if (u_final.link_fc_logit_valid && ((!learning) || u_final.link_fc_logit_ready)) begin
                if (u_final.link_fc_logit_q6 != expected_q6[(timestep*B+sample_index)*C+(q6_checks%C)])
                    $fatal(1, "Q6 mismatch t=%0d s=%0d c=%0d got=%0d exp=%0d", timestep,sample_index,q6_checks%C,
                        u_final.link_fc_logit_q6,expected_q6[(timestep*B+sample_index)*C+(q6_checks%C)]);
                q6_checks++;
            end
        end
    end

    task automatic load_parameters;
        for (int addr = 0; addr < 371520; addr++) begin
            @(negedge i_clk);
            i_learning_fc_parameter_load_valid = 1;
            i_learning_fc_parameter_load_addr = 24'(addr);
            i_learning_fc_parameter_load_code = parameters[addr];
            do @(posedge i_clk); while (!o_learning_fc_parameter_load_ready);
            @(negedge i_clk); i_learning_fc_parameter_load_valid = 0;
        end
    endtask

    task automatic refill(input bit output_layer);
        longint start;
        start = cycles;
        @(negedge i_clk);
        i_fc_cache_refill_weight_int8 = output_layer;
        i_fc_cache_refill_destination_tile_base = output_layer ? 4 : 0;
        i_fc_w0_parameter_base = output_layer ? W2_BASE : 0;
        i_fc_w0_hidden_stride = output_layer ? C : H;
        i_fc_w0_feature_count = output_layer ? H : F;
        i_fc_w0_hidden_count = output_layer ? C : H;
        i_fc_w0_tile_count = output_layer ? 1 : 4;
        i_fc_w0_refill_valid = 1;
        do @(posedge i_clk); while (!o_fc_w0_refill_ready);
        @(negedge i_clk); i_fc_w0_refill_valid = 0;
        while (!o_fc_w0_refill_done) @(negedge i_clk);
        @(negedge i_clk);
        refill_cycles += cycles-start;
    endtask

    task automatic clear_states;
        for (int s = 0; s < B; s++) begin
            for (int tile = 0; tile < 5; tile++) begin
                for (int row = 0; row < 4; row++) begin
                    @(negedge i_clk);
                    i_maintenance_tile = 4'(tile);
                    i_state_maintenance_write_addr = 10'(s*16+row);
                    i_trace_maintenance_write_addr = 10'(512+s*16+row);
                    i_state_maintenance_write_valid = 1;
                    i_trace_maintenance_write_valid = 1;
                    do @(posedge i_clk); while (!o_state_maintenance_write_ready || !o_trace_maintenance_write_ready);
                    @(negedge i_clk);
                    i_state_maintenance_write_valid = 0;
                    i_trace_maintenance_write_valid = 0;
                end
            end
        end
    endtask

    function automatic int rounds(input bit output_layer, input int row);
        int n;
        n = 0;
        if (output_layer) return 1;
        for (int lane = 0; lane < 4; lane++)
            if (int'(input_counts[(timestep*B+sample_index)*F+row*4+lane]) > n)
                n = int'(input_counts[(timestep*B+sample_index)*F+row*4+lane]);
        return n;
    endfunction

    function automatic logic [3:0] mask_for(input bit output_layer, input int row, input int rep);
        logic [3:0] mask;
        if (output_layer) return hidden_spikes[row*4 +: 4];
        for (int lane = 0; lane < 4; lane++)
            mask[lane] = int'(input_counts[(timestep*B+sample_index)*F+row*4+lane]) > rep;
        return mask;
    endfunction

    task automatic issue(input int row, input logic [3:0] mask);
        @(negedge i_clk);
        i_fc_srnn_group_weight_addr0 = 11'(row);
        i_fc_srnn_group_weight_addr1 = 11'(row);
        i_fc_srnn_group_weight_addr2 = 11'(row);
        i_fc_srnn_group_weight_addr3 = 11'(row);
        i_fc_srnn_group_source_valid = mask;
        i_fc_srnn_group_valid = 1;
        do @(posedge i_clk); while (!o_fc_srnn_group_ready);
        groups++;
        if (phase == 1) input_events += $countones(mask);
        else hidden_events += $countones(mask);
        @(negedge i_clk); i_fc_srnn_group_valid = 0;
    endtask

    task automatic forward_layer(input bit output_layer);
        int n, limit, done_before, capture_before, replay_before;
        longint start;
        logic [3:0] mask;
        phase = output_layer ? 2 : 1;
        start = cycles;
        if (!output_layer) hidden_spikes = 0;
        limit = output_layer ? 40 : 578;
        n = 0;
        for (int row = 0; row < limit; row++)
            for (int rep = 0; rep < rounds(output_layer, row); rep++)
                if (mask_for(output_layer, row, rep) != 0) n++;
        @(negedge i_clk);
        i_fc_srnn_transaction_tag = 52'((timestep*B+sample_index)*2+output_layer);
        i_fc_srnn_learning_cfg_tag = i_fc_srnn_transaction_tag;
        i_fc_srnn_learning_sample_id = 6'(sample_index);
        // Observations use layer-local IDs, independent of the physical tile.
        i_fc_srnn_learning_neuron_base = 0;
        i_fc_srnn_learning_neuron_count = output_layer ? C : H;
        i_fc_srnn_learning_fc_output_layer = output_layer;
        i_fc_srnn_learning_fc_logit_scale_enable = output_layer;
        i_fc_srnn_learning_fc_logit_accumulate = output_layer;
        i_fc_srnn_learning_fc_label = labels[sample_index];
        i_fc_srnn_learning_fc_sample_start = timestep == 0;
        i_fc_srnn_learning_cfg_valid = 1;
        do @(posedge i_clk); while (!o_fc_srnn_learning_cfg_ready);
        @(negedge i_clk); i_fc_srnn_learning_cfg_valid = 0;
        i_fc_srnn_weight_int8 = output_layer;
        i_fc_srnn_current_shift6 = !output_layer;
        i_fc_srnn_neuron_count = output_layer ? C : H;
        i_fc_srnn_destination_tile_base = output_layer ? 4 : 0;
        i_fc_srnn_leak = output_layer ? 0 : 32113;
        i_fc_srnn_threshold = output_layer ? 131071 : 1024;
        i_fc_srnn_work_base_addr = WORK_ADDR_WIDTH'(sample_index*16);
        i_fc_srnn_feedforward_group_count = 16'(n == 0 ? 1 : n);
        done_before = layer_done_count;
        capture_before = capture_done_count;
        replay_before = replay_done_count;
        i_fc_srnn_layer_valid = 1;
        do @(posedge i_clk); while (!o_fc_srnn_layer_ready);
        @(negedge i_clk); i_fc_srnn_layer_valid = 0;
        if (n == 0) issue(0, 0);
        else for (int row = 0; row < limit; row++) begin
            for (int rep = 0; rep < rounds(output_layer, row); rep++) begin
                mask = mask_for(output_layer, row, rep);
                if (mask != 0) issue(row, mask);
            end
        end
        while (layer_done_count != done_before+1 || capture_done_count != capture_before+1 ||
               replay_done_count != replay_before+1) @(negedge i_clk);
        @(negedge i_clk);
        if (output_layer) w2_cycles += cycles-start;
        else w0_cycles += cycles-start;
        phase = 0;
    endtask

    task automatic read_batch;
        batch_fixture = batches == 1 ? fixture : $sformatf("%s/batch_%0d", fixture, batch_index);
        $readmemh({batch_fixture,"/input_counts.mem"}, input_counts);
        $readmemh({batch_fixture,"/labels.mem"}, labels);
        $readmemh({batch_fixture,"/pre.mem"}, expected_pre);
        $readmemh({batch_fixture,"/trace.mem"}, expected_trace);
        $readmemh({batch_fixture,"/spike.mem"}, expected_spike);
        $readmemh({batch_fixture,"/dot.mem"}, expected_dot);
        $readmemh({batch_fixture,"/q6.mem"}, expected_q6);
        $readmemh({batch_fixture,"/w2_after.mem"}, expected_w2);
        $readmemh({batch_fixture,"/residual_after.mem"}, expected_residual);
    endtask

    initial begin
        if (!$value$plusargs("FIXTURE=%s", fixture)) $fatal(1, "fixture required");
        if (!$value$plusargs("STEPS=%d", steps)) steps = 2;
        if (!$value$plusargs("LEARNING=%d", learning)) learning = 1;
        if (!$value$plusargs("BATCHES=%d", batches)) batches = 1;
        if (steps < 1 || steps > 99 || (learning != 0 && learning != 1)) $fatal(1, "invalid run configuration");
        if (batches < 1 || batches > 10) $fatal(1, "invalid batch count");
        if (batches > 1) $readmemh({fixture,"/batch_steps.mem"}, batch_steps);
        else batch_steps[0] = steps;
        for (int b = 0; b < batches; b++) begin
            if (batch_steps[b] < 1 || batch_steps[b] > 99) $fatal(1, "invalid batch steps");
            total_steps += batch_steps[b];
        end
        read_batch();
        $readmemh({batch_fixture,"/parameters.mem"}, parameters);
        i_parameter_owner = 3'b001;
        i_topology_learning_enable = learning ? 3'b001 : 3'b000;
        i_fc_w0_refill_all = 1;
        i_fc_w0_version = 1;
        i_state_maintenance_write_mask = '1;
        i_trace_maintenance_write_mask = '1;
        i_state_maintenance_result_ready = 1;
        i_trace_maintenance_result_ready = 1;
        i_fc_srnn_observation_ready = 1;
        i_fc_srnn_observation_sink_idle = 1;
        i_fc_srnn_trace_decay = 64225;
        i_fc_srnn_learning_fc_threshold = 1024;
        i_fc_srnn_learning_fc_logit_scale_q16 = 24'(scale_q16);
        i_learning_fc_profile_id = 8'h11;
        i_learning_fc_w2_tensor_id = 8'h12;
        i_learning_fc_w2_parameter_base = W2_BASE;
        i_learning_fc_batch_count = B;
        i_learning_fc_hidden_count = H;
        i_learning_fc_output_count = C;
        i_learning_fc_vote_increment = 11;
        i_learning_fc_vote_threshold = 16'h8000;
        i_learning_fc_code_min = -127;
        i_learning_fc_code_max = 127;
        i_learning_fc_window_commit_ready = 1;
        repeat (6) @(negedge i_clk);
        i_rst_n = 1;
        start_cycle = cycles;
        load_parameters();
        refill(0); refill(1);
        load_cycles = cycles-start_cycle;
        start_cycle = cycles;
        clear_states();
        init_cycles = cycles-start_cycle;
        if (learning) begin
            @(negedge i_clk); i_learning_fc_cfg_valid = 1;
            do @(posedge i_clk); while (!o_learning_fc_cfg_ready);
            @(negedge i_clk); i_learning_fc_cfg_valid = 0;
        end
        $display("FC_FINAL_COUNTS_LOAD cycles=%0d init_cycles=%0d forward_reads=%0d",load_cycles,init_cycles,forward_reads);
        start_cycle = cycles;
        for (batch_index = 0; batch_index < batches; batch_index++) begin
            steps = batch_steps[batch_index];
            if (batch_index != 0) begin
                // Model state resets for new samples; parameter/residual store persists.
                read_batch();
                clear_states();
            end
        for (timestep = 0; timestep < steps; timestep++) begin
            for (sample_index = 0; sample_index < B; sample_index++) begin
                forward_layer(0);
                forward_layer(1);
            end
            if (learning) begin
                longint learn_start;
                learn_start = cycles;
                while (commits != window_index+1 || o_learning_fc_window_busy) @(negedge i_clk);
                @(negedge i_clk);
                learning_cycles += cycles-learn_start;
            end
            for (int addr = 0; addr < H*C; addr++) begin
                if (u_store.code_mem[W2_BASE+addr] != expected_w2[timestep*H*C+addr] ||
                    u_store.residual_mem[W2_BASE+addr] != expected_residual[timestep*H*C+addr])
                    $fatal(1,"FC committed store mismatch t=%0d addr=%0d code=%0d/%0d residual=%0d/%0d",timestep,addr,
                        u_store.code_mem[W2_BASE+addr],expected_w2[timestep*H*C+addr],
                        u_store.residual_mem[W2_BASE+addr],expected_residual[timestep*H*C+addr]);
            end
            if (learning) refill(1);
            $display("FC_FINAL_COUNTS_STEP batch_index=%0d t=%0d cycles=%0d commits=%0d code_writes=%0d residual_writes=%0d",
                batch_index,timestep,cycles-start_cycle,commits,code_writes,residual_writes);
            window_index++;
        end
        end
        for (int addr = 0; addr < W2_BASE; addr++)
            if (u_store.code_mem[addr] != parameters[addr] || u_store.residual_mem[addr] != 0)
                $fatal(1, "frozen W0 mismatch addr=%0d",addr);
        if (hidden_checks != total_steps*B*H || dot_checks != total_steps*B*C || q6_checks != total_steps*B*C ||
            commits != (learning ? total_steps : 0)) $fatal(1,"incomplete FC final coverage h=%0d dot=%0d q6=%0d commits=%0d",hidden_checks,dot_checks,q6_checks,commits);
        $display("FC_FINAL_COUNTS_PASS learning=%0d batch=%0d batches=%0d steps=%0d cycles=%0d hidden_checks=%0d dot_checks=%0d q6_checks=%0d commits=%0d w0_cycles=%0d w2_cycles=%0d learning_cycles=%0d refill_cycles=%0d input_events=%0d hidden_events=%0d groups=%0d forward_reads=%0d learning_reads=%0d code_writes=%0d residual_writes=%0d",
            learning,B,batches,total_steps,cycles-start_cycle,hidden_checks,dot_checks,q6_checks,commits,w0_cycles,w2_cycles,learning_cycles,refill_cycles,
            input_events,hidden_events,groups,forward_reads,learning_reads,code_writes,residual_writes);
        $finish;
    end
