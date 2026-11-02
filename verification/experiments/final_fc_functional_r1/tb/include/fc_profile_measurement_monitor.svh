    // Read-only pre-NBA edge samples. The original body remains the sole driver.
    `define FC_MEASURE_PATH u_final.u_parameter_plane.gen_shared_fc_scnn_output_learner.u_shared_output_learner.g_output_learning_core.u_shared_output_learning_path
    integer fc_measure_stage = 0;
    bit fc_measure_complete = 0;
    longint fc_measure_clock = 0;
    longint fc_measure_cycles[8] = '{default:0};
    longint fc_measure_window_busy[8] = '{default:0};
    longint fc_measure_work_busy[8] = '{default:0};
    longint fc_measure_work_overlap[8] = '{default:0};
    longint fc_measure_work_only[8] = '{default:0};
    longint fc_measure_work_other[8] = '{default:0};
    longint fc_measure_forward[8] = '{default:0};
    longint fc_measure_refill[8] = '{default:0};
    longint fc_measure_bus[8][20] = '{default:'{default:0}};
    longint fc_measure_state_bytes[8] = '{default:0};
    longint fc_measure_trace_bytes[8] = '{default:0};
    longint fc_measure_commit_present[8] = '{default:0};
    longint fc_measure_commit_stall[8] = '{default:0};
    longint fc_measure_observation = -1, fc_measure_done = -1;
    longint fc_measure_commit_first = -1, fc_measure_work_first = -1;
    longint fc_measure_last_write = -1, fc_measure_update_start = -1;
    longint fc_measure_last_forward = -1;
    longint fc_measure_window_work = 0, fc_measure_window_overlap = 0;
    longint fc_measure_window_other = 0;
    integer fc_measure_commits = 0;
    wire fc_measure_work = learning && (
        `FC_MEASURE_PATH.frontend_busy ||
        `FC_MEASURE_PATH.u_learning_slice.gradient_busy ||
        `FC_MEASURE_PATH.u_learning_slice.update_busy);
    wire fc_measure_in_forward = phase == 1 || phase == 2;
    wire fc_measure_runtime = fc_measure_stage >= 4 && fc_measure_stage <= 6;
    wire fc_measure_other = !fc_measure_runtime || fc_measure_stage == 6 || o_cache_refill_busy;

    always @(posedge i_clk) begin : fc_measure_sample
        integer st;
        if (i_rst_n && !fc_measure_complete) begin
            st = fc_measure_stage;
            fc_measure_clock++;
            fc_measure_cycles[st]++;
            if (o_learning_fc_window_busy) fc_measure_window_busy[st]++;
            if (fc_measure_in_forward) fc_measure_forward[st]++;
            if (o_cache_refill_busy) fc_measure_refill[st]++;
            if (fc_measure_work) begin
                fc_measure_work_busy[st]++;
                fc_measure_window_work++;
                if (fc_measure_work_first < 0) fc_measure_work_first = fc_measure_clock;
                if (fc_measure_in_forward) begin
                    fc_measure_work_overlap[st]++;
                    fc_measure_window_overlap++;
                end else if (fc_measure_other) begin
                    fc_measure_work_other[st]++;
                    fc_measure_window_other++;
                end else fc_measure_work_only[st]++;
            end
            if (`FC_MEASURE_PATH.observation_fire) fc_measure_observation = fc_measure_clock;
            if (`FC_MEASURE_PATH.learning_window_emit) fc_measure_update_start = fc_measure_clock;
            if (`FC_MEASURE_PATH.learning_window_done) fc_measure_done = fc_measure_clock;
            if (phase == 2 && sample_index == B-1 && o_fc_srnn_learning_done)
                fc_measure_last_forward = fc_measure_clock;
            if (o_learning_fc_window_commit_valid) begin
                fc_measure_commit_present[st]++;
                if (fc_measure_commit_first < 0) fc_measure_commit_first = fc_measure_clock;
                if (!i_learning_fc_window_commit_ready) fc_measure_commit_stall[st]++;
            end
            if (o_learning_fc_window_commit_valid && i_learning_fc_window_commit_ready) begin
                $display("FC_PROFILE_WINDOW index=%0d observation=%0d update_start=%0d done=%0d commit_first=%0d commit=%0d work_first=%0d last_forward=%0d last_write=%0d work_busy=%0d work_overlap_forward=%0d work_other=%0d",
                    fc_measure_commits, fc_measure_observation, fc_measure_update_start,
                    fc_measure_done, fc_measure_commit_first, fc_measure_clock,
                    fc_measure_work_first, fc_measure_last_forward, fc_measure_last_write,
                    fc_measure_window_work, fc_measure_window_overlap, fc_measure_window_other);
                fc_measure_commits++;
                fc_measure_observation = -1;
                fc_measure_update_start = -1;
                fc_measure_done = -1;
                fc_measure_commit_first = -1;
                fc_measure_work_first = -1;
                fc_measure_last_write = -1;
                fc_measure_last_forward = -1;
                fc_measure_window_work = 0;
                fc_measure_window_overlap = 0;
                fc_measure_window_other = 0;
            end
            if (i_learning_fc_parameter_load_valid && o_learning_fc_parameter_load_ready) fc_measure_bus[st][0]++;
            if (o_store_load_valid && i_store_load_ready) fc_measure_bus[st][1]++;
            if (o_store_forward_read_valid && i_store_forward_read_ready) fc_measure_bus[st][2]++;
            if (i_store_forward_response_valid && o_store_forward_response_ready) fc_measure_bus[st][3]++;
            if (o_store_learning_read_valid && i_store_learning_read_ready) fc_measure_bus[st][4]++;
            if (i_store_learning_response_valid && o_store_learning_response_ready) fc_measure_bus[st][5]++;
            if (o_store_learning_write_valid && i_store_learning_write_ready) begin
                fc_measure_bus[st][6]++;
                if (o_store_learning_write_code_enable) fc_measure_bus[st][7]++;
                if (o_store_learning_write_residual_enable) fc_measure_bus[st][8]++;
                fc_measure_last_write = fc_measure_clock;
            end
            if (i_state_maintenance_write_valid && o_state_maintenance_write_ready) begin
                fc_measure_bus[st][9]++;
                fc_measure_state_bytes[st] += $countones(i_state_maintenance_write_mask);
            end
            if (i_trace_maintenance_write_valid && o_trace_maintenance_write_ready) begin
                fc_measure_bus[st][10]++;
                fc_measure_trace_bytes[st] += $countones(i_trace_maintenance_write_mask);
            end
            if (i_state_maintenance_read_valid && o_state_maintenance_read_ready) fc_measure_bus[st][11]++;
            if (o_state_maintenance_result_valid && i_state_maintenance_result_ready) fc_measure_bus[st][12]++;
            if (i_trace_maintenance_read_valid && o_trace_maintenance_read_ready) fc_measure_bus[st][13]++;
            if (o_trace_maintenance_result_valid && i_trace_maintenance_result_ready) fc_measure_bus[st][14]++;
            if (i_fc_srnn_group_valid && o_fc_srnn_group_ready) begin
                fc_measure_bus[st][15]++;
                fc_measure_bus[st][16] += $countones(i_fc_srnn_group_source_valid);
            end
            if (o_fc_srnn_observation_valid && i_fc_srnn_observation_ready) fc_measure_bus[st][17]++;
            if (learning && u_final.link_fc_student_trace_valid && u_final.link_fc_student_trace_ready) fc_measure_bus[st][18]++;
            if (u_final.link_fc_logit_valid && (!learning || u_final.link_fc_logit_ready)) fc_measure_bus[st][19]++;
        end
    end

    final begin
        if (fc_measure_complete) begin
            for (int st = 0; st < 8; st++) begin
                $display("FC_PROFILE_STAGE id=%0d cycles=%0d window_busy=%0d work_busy=%0d work_overlap_forward=%0d work_only=%0d work_other=%0d forward_scope=%0d refill_busy=%0d commit_present=%0d commit_stall=%0d parameter_input=%0d store_load=%0d forward_request=%0d forward_response=%0d learning_request=%0d learning_response=%0d learning_write=%0d code_write=%0d residual_write=%0d state_write=%0d trace_write=%0d state_read_request=%0d state_read_response=%0d trace_read_request=%0d trace_read_response=%0d source_group=%0d source_events=%0d observation=%0d student_trace=%0d logit=%0d state_write_mask_bytes=%0d trace_write_mask_bytes=%0d",
                    st, fc_measure_cycles[st], fc_measure_window_busy[st], fc_measure_work_busy[st],
                    fc_measure_work_overlap[st], fc_measure_work_only[st], fc_measure_work_other[st],
                    fc_measure_forward[st], fc_measure_refill[st], fc_measure_commit_present[st], fc_measure_commit_stall[st],
                    fc_measure_bus[st][0], fc_measure_bus[st][1], fc_measure_bus[st][2], fc_measure_bus[st][3],
                    fc_measure_bus[st][4], fc_measure_bus[st][5], fc_measure_bus[st][6], fc_measure_bus[st][7],
                    fc_measure_bus[st][8], fc_measure_bus[st][9], fc_measure_bus[st][10], fc_measure_bus[st][11],
                    fc_measure_bus[st][12], fc_measure_bus[st][13], fc_measure_bus[st][14], fc_measure_bus[st][15],
                    fc_measure_bus[st][16], fc_measure_bus[st][17], fc_measure_bus[st][18], fc_measure_bus[st][19],
                    fc_measure_state_bytes[st], fc_measure_trace_bytes[st]);
            end
        end
    end
    `undef FC_MEASURE_PATH
