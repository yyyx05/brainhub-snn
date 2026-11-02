    localparam int H = 450, B = 2, T = 8;
    logic signed [15:0] w0 [0:700*H-1], recurrent [0:H*H-1];
    logic signed [31:0] expected_pre [0:T*B*480-1];
    logic [31:0] expected_spike [0:T*B*480-1];
    logic [63:0] ff_groups [0:8191];
    logic [31:0] ff_offsets [0:T*B];
    logic [15:0] ff_counts [0:T*B-1];
    logic [H-1:0] previous_spikes [0:B-1];
    logic [H-1:0] next_spikes;
    integer timestep = 0, sample_index = 0, observations = 0, captures = 0, layers = 0, replays = 0;
    integer read_count = 0, group_count = 0, fd;
    longint cycles = 0, initialized_cycle;
    string fixture, dump;
    always #5 i_clk = !i_clk;

    assign i_srnn_backbone_forward_read_ready = !i_srnn_backbone_forward_response_valid || o_srnn_backbone_forward_response_ready;
    always @(posedge i_clk) begin
        cycles <= cycles + 1;
        if (cycles > 8000000) $fatal(1, "SRNN raw700 watchdog t=%0d s=%0d layers=%0d obs=%0d capture=%0d replay=%0d", timestep,sample_index,layers,observations,captures,replays);
        if (!i_rst_n) i_srnn_backbone_forward_response_valid <= 0;
        else begin
            if (i_srnn_backbone_forward_response_valid && o_srnn_backbone_forward_response_ready)
                i_srnn_backbone_forward_response_valid <= 0;
            if (o_srnn_backbone_forward_read_valid && i_srnn_backbone_forward_read_ready) begin
                read_count++;
                i_srnn_backbone_forward_response_valid <= 1;
                i_srnn_backbone_forward_response_tag <= o_srnn_backbone_forward_read_tag;
                if (o_srnn_backbone_forward_read_addr < 700*H)
                    i_srnn_backbone_forward_response_code <= w0[o_srnn_backbone_forward_read_addr];
                else if (o_srnn_backbone_forward_read_addr < (700+H)*H)
                    i_srnn_backbone_forward_response_code <= recurrent[int'(o_srnn_backbone_forward_read_addr)-700*H];
                else $fatal(1, "SRNN raw700 parameter read out of range");
            end
            if (o_sticky_error || o_backend_error || o_srnn_phase_sticky_error || o_fc_srnn_sticky_error ||
                o_fc_srnn_learning_sticky_error || o_fc_srnn_observation_sticky_error || o_array_tile_sticky_error != 0)
                $fatal(1,"SRNN raw700 errors global=%0b phase=%0b layer=%0b/%h ingress=%0b/%h tiles=%h",
                    o_sticky_error,o_srnn_phase_sticky_error,o_fc_srnn_sticky_error,o_fc_srnn_error_code,
                    o_fc_srnn_learning_sticky_error,o_fc_srnn_learning_error_code,o_array_tile_sticky_error);
            if (o_fc_srnn_layer_done) layers++;
            if (o_fc_srnn_learning_capture_done) captures++;
            if (o_fc_srnn_learning_done) replays++;
            if (o_fc_srnn_observation_valid && i_fc_srnn_observation_ready) begin
                if (o_fc_srnn_observation_neuron_id >= H || o_fc_srnn_observation_mode != 1)
                    $fatal(1,"SRNN raw700 observation identity");
                if (o_fc_srnn_observation_pre_state != expected_pre[(timestep*B+sample_index)*480+int'(o_fc_srnn_observation_neuron_id)] ||
                    o_fc_srnn_observation_spike != expected_spike[(timestep*B+sample_index)*480+int'(o_fc_srnn_observation_neuron_id)][0])
                    $fatal(1,"SRNN raw700 numeric mismatch t=%0d s=%0d h=%0d pre=%0d/%0d spike=%0d/%0d",
                        timestep,sample_index,o_fc_srnn_observation_neuron_id,o_fc_srnn_observation_pre_state,
                        expected_pre[(timestep*B+sample_index)*480+int'(o_fc_srnn_observation_neuron_id)],
                        o_fc_srnn_observation_spike,expected_spike[(timestep*B+sample_index)*480+int'(o_fc_srnn_observation_neuron_id)]);
                next_spikes[o_fc_srnn_observation_neuron_id] = o_fc_srnn_observation_spike;
                $fwrite(fd,"%0d,%0d,%0d,%0d,%0d\n",timestep,sample_index,o_fc_srnn_observation_neuron_id,
                    o_fc_srnn_observation_pre_state,o_fc_srnn_observation_spike);
                observations++;
            end
            if (o_store_learning_write_valid) $fatal(1,"frozen raw700 unexpectedly writes learning store");
        end
    end

    task automatic refill(input bit rec);
        @(negedge i_clk);
        i_srnn_backbone_parameter_base = rec ? 700*H : 0;
        i_srnn_backbone_source_count = rec ? H : 700;
        i_srnn_backbone_cache_row_base = rec ? 350 : 0;
        i_srnn_backbone_refill_valid = 1;
        do @(posedge i_clk); while (!o_srnn_backbone_refill_ready);
        @(negedge i_clk); i_srnn_backbone_refill_valid = 0;
        while (!o_srnn_backbone_refill_done) @(negedge i_clk);
        @(negedge i_clk);
    endtask

    task automatic issue(input bit rec, input logic [3:0] mask, input int a0, a1, a2, a3);
        @(negedge i_clk);
        i_fc_srnn_group_recurrent = rec;
        i_fc_srnn_group_source_valid = mask;
        i_fc_srnn_group_weight_addr0 = 11'(a0); i_fc_srnn_group_weight_addr1 = 11'(a1);
        i_fc_srnn_group_weight_addr2 = 11'(a2); i_fc_srnn_group_weight_addr3 = 11'(a3);
        i_fc_srnn_group_valid = 1;
        do @(posedge i_clk); while (!o_fc_srnn_group_ready);
        group_count++;
        @(negedge i_clk); i_fc_srnn_group_valid = 0;
    endtask

    initial begin
        if (!$value$plusargs("FIXTURE=%s", fixture) || !$value$plusargs("DUMP=%s", dump))
            $fatal(1,"FIXTURE and DUMP required");
        $readmemh({fixture,"/w0_refill_dest_major.mem"},w0);
        $readmemh({fixture,"/r_refill_dest_major.mem"},recurrent);
        $readmemh({fixture,"/backbone_pre.mem"},expected_pre);
        $readmemh({fixture,"/backbone_spike.mem"},expected_spike);
        $readmemh({fixture,"/feedforward_p4.mem"},ff_groups);
        $readmemh({fixture,"/feedforward_offsets.mem"},ff_offsets);
        $readmemh({fixture,"/feedforward_group_counts.mem"},ff_counts);
        fd = $fopen(dump,"w");
        if (!fd) $fatal(1,"cannot open observations");
        $fwrite(fd,"timestep,sample,neuron,pre,spike\n");
        previous_spikes[0] = 0; previous_spikes[1] = 0;
        i_parameter_owner = 3'b010;
        i_srnn_backbone_refill_all = 1;
        i_srnn_backbone_destination_count = H;
        i_srnn_backbone_version = 1;
        i_state_maintenance_write_mask = '1;
        i_trace_maintenance_write_mask = '1;
        i_state_maintenance_result_ready = 1; i_trace_maintenance_result_ready = 1;
        i_fc_srnn_observation_ready = 1; i_fc_srnn_observation_sink_idle = 1;
        i_fc_srnn_mode = 1; i_fc_srnn_learning_cfg_mode = 1;
        i_fc_srnn_neuron_count = H; i_fc_srnn_learning_neuron_count = H;
        i_fc_srnn_recurrent_group_count = 113; i_fc_srnn_recurrent_enable = 1;
        i_fc_srnn_leak = 27853; i_fc_srnn_threshold = 32; i_fc_srnn_trace_decay = 27853;
        i_srnn_phase_record_count = H;
        repeat (6) @(negedge i_clk);
        i_rst_n = 1;
        refill(0); refill(1);
        for (int s = 0; s < B; s++) for (int tile = 0; tile < 12; tile++) for (int row = 0; row < 4; row++) begin
            @(negedge i_clk);
            i_maintenance_tile = 4'(tile);
            i_state_maintenance_write_addr = 10'(s*16+row);
            i_trace_maintenance_write_addr = 10'(512+s*16+row);
            i_state_maintenance_write_valid = 1; i_trace_maintenance_write_valid = 1;
            do @(posedge i_clk); while (!o_state_maintenance_write_ready || !o_trace_maintenance_write_ready);
            @(negedge i_clk); i_state_maintenance_write_valid = 0; i_trace_maintenance_write_valid = 0;
        end
        initialized_cycle = cycles;
        $display("SRNN_RAW700_LOAD cycles=%0d parameter_reads=%0d",cycles,read_count);
        for (timestep = 0; timestep < T; timestep++) for (sample_index = 0; sample_index < B; sample_index++) begin
            int record_index;
            record_index = timestep*B+sample_index;
            next_spikes = 0;
            @(negedge i_clk); i_srnn_phase_capture_cfg_valid = 1;
            do @(posedge i_clk); while (!o_srnn_phase_capture_cfg_ready);
            @(negedge i_clk); i_srnn_phase_capture_cfg_valid = 0;
            i_fc_srnn_transaction_tag = 52'(record_index);
            i_fc_srnn_learning_cfg_tag = 52'(record_index);
            i_fc_srnn_learning_sample_id = 6'(sample_index);
            i_fc_srnn_learning_cfg_valid = 1;
            do @(posedge i_clk); while (!o_fc_srnn_learning_cfg_ready);
            @(negedge i_clk); i_fc_srnn_learning_cfg_valid = 0;
            i_fc_srnn_work_base_addr = WORK_ADDR_WIDTH'(sample_index*16);
            i_fc_srnn_feedforward_group_count = ff_counts[record_index];
            i_fc_srnn_layer_valid = 1;
            do @(posedge i_clk); while (!o_fc_srnn_layer_ready);
            @(negedge i_clk); i_fc_srnn_layer_valid = 0;
            for (int k = int'(ff_offsets[record_index]); k < int'(ff_offsets[record_index+1]); k++)
                issue(0,ff_groups[k][3:0],int'(ff_groups[k][18:8]),int'(ff_groups[k][29:19]),
                    int'(ff_groups[k][40:30]),int'(ff_groups[k][51:41]));
            for (int g = 0; g < 113; g++) begin
                logic [3:0] mask;
                mask = 0;
                for (int lane = 0; lane < 4; lane++) if (g*4+lane < H) mask[lane] = previous_spikes[sample_index][g*4+lane];
                issue(1,mask,350+g*2,350+g*2,350+g*2+1,350+g*2+1);
            end
            while (layers != record_index+1 || captures != record_index+1 || replays != record_index+1 || !o_srnn_phase_capture_complete)
                @(negedge i_clk);
            previous_spikes[sample_index] = next_spikes;
            // This first stage validates the real backbone and capture only.
            // No M2 output is substituted; clear just the captured record.
            @(negedge i_clk); i_srnn_phase_abort = 1;
            @(negedge i_clk); i_srnn_phase_abort = 0;
            $display("SRNN_RAW700_STEP t=%0d sample=%0d observations=%0d cycles=%0d",timestep,sample_index,observations,cycles-initialized_cycle);
        end
        if (observations != T*B*H || read_count != (700+H)*H) $fatal(1,"raw700 incomplete coverage");
        $fclose(fd);
        $display("SRNN_RAW700_BACKBONE_PASS batch=2 steps=8 neurons=450 pre_checks=%0d spike_checks=%0d parameter_reads=%0d groups=%0d cycles=%0d",
            observations,observations,read_count,group_count,cycles-initialized_cycle);
        $finish;
    end
