    localparam int H = 450, T = 8, BONE_BASE = 256;
    logic signed [15:0] w0[0:314999], recurrent[0:202499], model_mem[0:120319];
    logic signed [31:0] bone_pre[0:T*2*480-1];
    logic [31:0] bone_spike[0:T*2*480-1];
    logic [63:0] ff_groups[0:8191];
    logic [31:0] ff_offsets[0:T*2];
    logic [15:0] ff_counts[0:T*2-1];
    logic signed [31:0] m_pre[0:T*256-1], m_state[0:T*256-1], m_trace[0:T*256-1], m_feature[0:T*256-1];
    logic [31:0] m_spike[0:T*256-1];
    logic signed [63:0] m_logit[0:T*20-1], m_gradient[0:T*5120-1];
    logic signed [15:0] m_before[0:T*5120-1], m_after[0:T*5120-1], m_residual[0:T*5120-1];
    logic [7:0] labels[0:0];
    logic [H-1:0] previous_spikes = '0, next_spikes;
    integer timestep = 0, learning = 0, bone_seen = 0, hidden_seen = 0, logit_seen = 0;
    integer layers = 0, captures = 0, replays = 0, done_count = 0, commits = 0, detaches = 0;
    integer code_writes = 0, residual_writes = 0, gradient_seen = 0, store_checks = 0, param_reads = 0;
    longint cycles = 0, initialized_cycle;
    string raw, fixture, mode;
    always #5 i_clk = !i_clk;

    `define M2 u_final.u_parameter_plane.g_srnn_client.u_srnn_m2
    `define ENG u_final.u_system.g_srnn_m2_h40_client.u_srnn_m2_h40_forward
    `define LINK(name) u_final.link_system_to_plane_srnn_h40_``name
    `define RED `M2.parameter_store.g_wout_only_learning.learning_engine

    assign i_srnn_backbone_forward_read_ready = !i_srnn_backbone_forward_response_valid || o_srnn_backbone_forward_response_ready;
    always @(posedge i_clk) begin
        cycles <= cycles + 1;
        if (cycles > 24000000) $fatal(1,"SAME_DUT_WATCHDOG t=%0d layer=%0d hidden=%0d logits=%0d phase=%0d coordinator=%0d prep=%0d cache=%0b",
            timestep,layers,hidden_seen,logit_seen,o_srnn_phase_busy,`M2.state_q,`ENG.prep_state_q,`ENG.o_cache_valid);
        if (!i_rst_n) i_srnn_backbone_forward_response_valid <= 0;
        else begin
            if (i_srnn_backbone_forward_response_valid && o_srnn_backbone_forward_response_ready)
                i_srnn_backbone_forward_response_valid <= 0;
            if (o_srnn_backbone_forward_read_valid && i_srnn_backbone_forward_read_ready) begin
                param_reads++;
                i_srnn_backbone_forward_response_valid <= 1;
                i_srnn_backbone_forward_response_tag <= o_srnn_backbone_forward_read_tag;
                if (o_srnn_backbone_forward_read_addr < 315000)
                    i_srnn_backbone_forward_response_code <= w0[o_srnn_backbone_forward_read_addr];
                else if (o_srnn_backbone_forward_read_addr < 517500)
                    i_srnn_backbone_forward_response_code <= recurrent[int'(o_srnn_backbone_forward_read_addr)-315000];
                else $fatal(1,"backbone address range");
            end
            if (o_sticky_error || o_backend_error || o_srnn_phase_sticky_error || o_learning_srnn_error ||
                o_learning_srnn_cfg_error || o_fc_srnn_sticky_error || o_fc_srnn_learning_sticky_error ||
                o_fc_srnn_observation_sticky_error || o_array_tile_sticky_error != 0)
                $fatal(1,"SAME_DUT_ERROR t=%0d global=%b backend=%b phase=%b m2=%b layer=%b/%h ingress=%b/%h tiles=%h coord=%0d prep=%0d",
                    timestep,o_sticky_error,o_backend_error,o_srnn_phase_sticky_error,o_learning_srnn_error,
                    o_fc_srnn_sticky_error,o_fc_srnn_error_code,o_fc_srnn_learning_sticky_error,
                    o_fc_srnn_learning_error_code,o_array_tile_sticky_error,`M2.state_q,`ENG.prep_state_q);
            if (o_fc_srnn_layer_done) layers++;
            if (o_fc_srnn_learning_capture_done) captures++;
            if (o_fc_srnn_learning_done) replays++;
            if (o_learning_srnn_timestep_done) done_count++;
            if (o_learning_srnn_learning_commit) commits++;
            if (o_learning_srnn_detach) detaches++;
            if (o_fc_srnn_observation_valid && i_fc_srnn_observation_ready) begin
                int h;
                h = int'(o_fc_srnn_observation_neuron_id);
                if (h >= H || o_fc_srnn_observation_mode != 1 ||
                    o_fc_srnn_observation_pre_state !== bone_pre[timestep*960+h] ||
                    o_fc_srnn_observation_spike !== bone_spike[timestep*960+h][0])
                    $fatal(1,"BONE_MISMATCH t=%0d h=%0d pre=%0d/%0d spike=%b/%b",timestep,h,
                        o_fc_srnn_observation_pre_state,bone_pre[timestep*960+h],
                        o_fc_srnn_observation_spike,bone_spike[timestep*960+h][0]);
                next_spikes[h] = o_fc_srnn_observation_spike;
                bone_seen++;
            end
            if (`LINK(hidden_valid) && u_final.link_plane_to_system_srnn_h40_hidden_ready) begin
                int idx;
                idx = timestep*256 + int'(`LINK(hidden_id));
                if (`LINK(hidden_sample_id) != 0 || `LINK(hidden_path) != 0 ||
                    `LINK(hidden_id) != hidden_seen%256 ||
                    `LINK(hidden_membrane_pre_q14) !== m_pre[idx] ||
                    `LINK(hidden_membrane_next_q14) !== m_state[idx] ||
                    `LINK(hidden_spike) !== m_spike[idx][0] ||
                    `LINK(hidden_trace_q14) !== m_trace[idx] ||
                    `LINK(hidden_output_feature_q14) !== m_feature[idx])
                    $fatal(1,"M2_HIDDEN_MISMATCH t=%0d h=%0d pre=%0d/%0d state=%0d/%0d spike=%b/%b trace=%0d/%0d",
                        timestep,`LINK(hidden_id),`LINK(hidden_membrane_pre_q14),m_pre[idx],
                        `LINK(hidden_membrane_next_q14),m_state[idx],`LINK(hidden_spike),m_spike[idx][0],
                        `LINK(hidden_trace_q14),m_trace[idx]);
                hidden_seen++;
            end
            if (o_learning_srnn_logit_valid && i_learning_srnn_logit_ready) begin
                if (o_learning_srnn_logit_sample_id != 0 || o_learning_srnn_logit_class_id != logit_seen%20 ||
                    o_learning_srnn_logit_q28 !== m_logit[logit_seen])
                    $fatal(1,"M2_LOGIT_MISMATCH t=%0d c=%0d got=%0d expected=%0d",timestep,
                        o_learning_srnn_logit_class_id,o_learning_srnn_logit_q28,m_logit[logit_seen]);
                logit_seen++;
            end
            if (`RED.reducer_gradient_valid && `RED.reducer_gradient_ready) begin
                if (`RED.reducer_gradient_addr != gradient_seen%5120 ||
                    `RED.reducer_gradient_q40 !== m_gradient[gradient_seen])
                    $fatal(1,"gradient mismatch index=%0d",gradient_seen);
                gradient_seen++;
            end
            if (o_store_learning_write_valid && i_store_learning_write_ready) begin
                if (!learning || o_store_learning_write_addr < 115200 || o_store_learning_write_addr >= 120320)
                    $fatal(1,"unexpected/frozen parameter write");
                if (o_store_learning_write_code_enable) code_writes++;
                if (o_store_learning_write_residual_enable) residual_writes++;
            end
            if (u_final.link_plane_to_system_srnn_h40_prepare_valid && `LINK(prepare_ready)) begin
                $display("M2_PREPARE t=%0d cache=%b version=%0d request=%0d",timestep,`ENG.o_cache_valid,
                    `ENG.o_cache_version,u_final.link_plane_to_system_srnn_h40_prepare_version);
                $fflush();
            end
        end
    end

    initial begin
        wait(i_rst_n);
        forever begin
            @(posedge i_clk);
            if (o_learning_srnn_label_replay_start) begin
                @(negedge i_clk); i_learning_srnn_label_valid = 1;
                do @(posedge i_clk); while (!o_learning_srnn_label_ready);
                @(negedge i_clk); i_learning_srnn_label_valid = 0;
            end
        end
    end

    task automatic refill(input bit rec);
        @(negedge i_clk);
        i_srnn_backbone_parameter_base = rec ? 315000 : 0;
        i_srnn_backbone_source_count = rec ? 450 : 700;
        i_srnn_backbone_cache_row_base = rec ? 350 : 0;
        i_srnn_backbone_refill_valid = 1;
        do @(posedge i_clk); while (!o_srnn_backbone_refill_ready);
        @(negedge i_clk); i_srnn_backbone_refill_valid = 0;
        while (!o_srnn_backbone_refill_done) @(negedge i_clk);
        @(negedge i_clk);
    endtask
    task automatic issue(input bit rec, input logic [3:0] mask, input int a0,a1,a2,a3);
        @(negedge i_clk);
        i_fc_srnn_group_recurrent = rec; i_fc_srnn_group_source_valid = mask;
        i_fc_srnn_group_weight_addr0 = 11'(a0); i_fc_srnn_group_weight_addr1 = 11'(a1);
        i_fc_srnn_group_weight_addr2 = 11'(a2); i_fc_srnn_group_weight_addr3 = 11'(a3);
        i_fc_srnn_group_valid = 1;
        do @(posedge i_clk); while (!o_fc_srnn_group_ready);
        @(negedge i_clk); i_fc_srnn_group_valid = 0;
    endtask

    initial begin
        if (!$value$plusargs("RAW=%s",raw) || !$value$plusargs("FIXTURE=%s",fixture) ||
            !$value$plusargs("LEARNING=%d",learning)) $fatal(1,"missing arguments");
        mode = learning ? "/on_" : "/off_";
        $readmemh({raw,"/w0_refill_dest_major.mem"},w0); $readmemh({raw,"/r_refill_dest_major.mem"},recurrent);
        $readmemh({raw,"/backbone_pre.mem"},bone_pre); $readmemh({raw,"/backbone_spike.mem"},bone_spike);
        $readmemh({raw,"/feedforward_p4.mem"},ff_groups); $readmemh({raw,"/feedforward_offsets.mem"},ff_offsets);
        $readmemh({raw,"/feedforward_group_counts.mem"},ff_counts);
        $readmemh("tb/data/srnn_m2_v28_wh1_initial_q14.mem",model_mem,0,115199);
        $readmemh("tb/data/srnn_m2_v28_wout_initial_q14.mem",model_mem,115200,120319);
        $readmemh({fixture,"/labels.mem"},labels);
        $readmemh({fixture,mode,"pre.mem"},m_pre); $readmemh({fixture,mode,"membrane.mem"},m_state);
        $readmemh({fixture,mode,"spike.mem"},m_spike); $readmemh({fixture,mode,"trace.mem"},m_trace);
        $readmemh({fixture,mode,"feature.mem"},m_feature); $readmemh({fixture,mode,"logit.mem"},m_logit);
        $readmemh({fixture,mode,"wout_before.mem"},m_before); $readmemh({fixture,mode,"wout_after.mem"},m_after);
        $readmemh({fixture,mode,"residual.mem"},m_residual); $readmemh({fixture,mode,"gradient.mem"},m_gradient);
        i_parameter_owner = 3'b010; i_topology_learning_enable = learning ? 3'b010 : 3'b000;
        i_learning_srnn_profile_id = 7; i_learning_srnn_wh1_tensor_id = 16;
        i_learning_srnn_wout_tensor_id = 17; i_learning_srnn_wout_base = 115200;
        i_learning_srnn_wh1_increment = 31; i_learning_srnn_wh1_threshold = 65;
        i_learning_srnn_wout_increment = 31; i_learning_srnn_wout_threshold = 65;
        i_learning_srnn_code_min = -32768; i_learning_srnn_code_max = 32767;
        i_learning_srnn_batch_count = 1; i_learning_srnn_hidden_count = 256;
        i_learning_srnn_backbone_count = 450; i_learning_srnn_class_count = 20;
        i_learning_srnn_wout_element_count = 5120; i_learning_srnn_sequence_version = 1;
        i_learning_srnn_label_replay_start_ready = 1; i_learning_srnn_label_last = 1;
        i_learning_srnn_label_class_id = labels[0][4:0]; i_learning_srnn_logit_ready = 1;
        i_srnn_phase_learning_enable = 1'(learning); i_srnn_phase_record_count = H;
        i_srnn_backbone_refill_all = 1; i_srnn_backbone_destination_count = H; i_srnn_backbone_version = 1;
        i_state_maintenance_write_mask = '1; i_trace_maintenance_write_mask = '1;
        i_state_maintenance_result_ready = 1; i_trace_maintenance_result_ready = 1;
        i_fc_srnn_observation_ready = 1; i_fc_srnn_observation_sink_idle = 1;
        i_fc_srnn_mode = 1; i_fc_srnn_learning_cfg_mode = 1;
        i_fc_srnn_neuron_count = H; i_fc_srnn_learning_neuron_count = H;
        i_fc_srnn_recurrent_group_count = 113; i_fc_srnn_recurrent_enable = 1;
        i_fc_srnn_leak = 27853; i_fc_srnn_threshold = 32; i_fc_srnn_trace_decay = 27853;
        i_fc_srnn_work_base_addr = WORK_ADDR_WIDTH'(BONE_BASE);
        repeat(6) @(negedge i_clk); i_rst_n = 1;
        @(negedge i_clk); i_learning_srnn_cfg_valid = 1;
        do @(posedge i_clk); while (!o_learning_srnn_cfg_ready);
        @(negedge i_clk); i_learning_srnn_cfg_valid = 0;
        for (int a=0; a<120320; a++) begin
            @(negedge i_clk); i_learning_srnn_parameter_load_valid = 1;
            i_learning_srnn_parameter_load_addr = PARAM_ADDR_WIDTH'(a);
            i_learning_srnn_parameter_load_code = model_mem[a];
            do @(posedge i_clk); while (!o_learning_srnn_parameter_load_ready);
        end
        @(negedge i_clk); i_learning_srnn_parameter_load_valid = 0;
        for (int tile=0; tile<12; tile++) for (int row=0; row<4; row++) begin
            @(negedge i_clk); i_maintenance_tile = 4'(tile);
            i_state_maintenance_write_addr = 10'(BONE_BASE+row);
            i_trace_maintenance_write_addr = 10'(512+BONE_BASE+row);
            i_state_maintenance_write_valid = 1; i_trace_maintenance_write_valid = 1;
            do @(posedge i_clk); while (!o_state_maintenance_write_ready || !o_trace_maintenance_write_ready);
            @(negedge i_clk); i_state_maintenance_write_valid = 0; i_trace_maintenance_write_valid = 0;
        end
        initialized_cycle = cycles;
        for (timestep=0; timestep<T; timestep++) begin
            // M2 uses the same cache: restore the backbone working set each step.
            refill(0); refill(1); next_spikes = 0;
            @(negedge i_clk); i_srnn_phase_capture_cfg_valid = 1;
            do @(posedge i_clk); while (!o_srnn_phase_capture_cfg_ready);
            @(negedge i_clk); i_srnn_phase_capture_cfg_valid = 0;
            i_fc_srnn_transaction_tag = 52'(timestep); i_fc_srnn_learning_cfg_tag = 52'(timestep);
            i_fc_srnn_learning_cfg_valid = 1;
            do @(posedge i_clk); while (!o_fc_srnn_learning_cfg_ready);
            @(negedge i_clk); i_fc_srnn_learning_cfg_valid = 0;
            i_fc_srnn_feedforward_group_count = ff_counts[timestep*2]; i_fc_srnn_layer_valid = 1;
            do @(posedge i_clk); while (!o_fc_srnn_layer_ready);
            @(negedge i_clk); i_fc_srnn_layer_valid = 0;
            for (int k=int'(ff_offsets[timestep*2]); k<int'(ff_offsets[timestep*2+1]); k++)
                issue(0,ff_groups[k][3:0],int'(ff_groups[k][18:8]),int'(ff_groups[k][29:19]),int'(ff_groups[k][40:30]),int'(ff_groups[k][51:41]));
            for (int g=0; g<113; g++) begin
                logic [3:0] mask;
                mask = 0;
                for (int lane=0; lane<4; lane++) if(g*4+lane<H) mask[lane]=previous_spikes[g*4+lane];
                issue(1,mask,350+g*2,350+g*2,350+g*2+1,350+g*2+1);
            end
            while(layers!=timestep+1 || captures!=timestep+1 || replays!=timestep+1 || !o_srnn_phase_capture_complete)
                @(negedge i_clk);
            previous_spikes = next_spikes;
            @(negedge i_clk); i_learning_srnn_timestep_id = $clog2(SRNN_TIMESTEP_MAX)'(timestep);
            i_srnn_phase_start_valid = 1;
            do @(posedge i_clk); while(!o_srnn_phase_start_ready);
            @(negedge i_clk); i_srnn_phase_start_valid = 0;
            while(done_count!=timestep+1) @(negedge i_clk);
            for(int a=0; a<120320; a++) begin
                if (u_store.code_mem[a] !== (a<115200 ? model_mem[a] : m_after[timestep*5120+a-115200]) ||
                    u_store.residual_mem[a] !== (a<115200 ? 16'sd0 : m_residual[timestep*5120+a-115200]))
                    $fatal(1,"store mismatch t=%0d a=%0d",timestep,a);
                store_checks++;
            end
            $display("SRNN_SAME_DUT_STEP t=%0d learning=%0d bone=%0d hidden=%0d logits=%0d code=%0d residual=%0d commits=%0d cycles=%0d",
                timestep,learning,bone_seen,hidden_seen,logit_seen,code_writes,residual_writes,commits,cycles-initialized_cycle);
            $fflush();
        end
        if(bone_seen!=T*H || hidden_seen!=T*256 || logit_seen!=T*20 ||
           gradient_seen!=(learning ? T*5120 : 0) || commits!=(learning ? T : 0) || detaches!=(learning ? T : 0))
            $fatal(1,"incomplete coverage bone=%0d hidden=%0d logits=%0d gradient=%0d commits=%0d detach=%0d learning=%0d",
                bone_seen,hidden_seen,logit_seen,gradient_seen,commits,detaches,learning);
        $display("SRNN_SAME_DUT_PASS batch=1 steps=8 learning=%0d bone=%0d hidden=%0d logits=%0d gradient=%0d store=%0d code=%0d residual=%0d commits=%0d detach=%0d cycles=%0d",
            learning,bone_seen,hidden_seen,logit_seen,gradient_seen,store_checks,code_writes,residual_writes,commits,detaches,cycles-initialized_cycle);
        $finish;
    end
    `undef RED
    `undef LINK
    `undef ENG
    `undef M2
