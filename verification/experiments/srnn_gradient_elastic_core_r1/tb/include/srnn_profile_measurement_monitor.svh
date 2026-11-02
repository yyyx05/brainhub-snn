    // All counters sample actual pre-edge valid/ready or busy signals.
    longint profile_phase_start = 0;
    string profile_phase = "reset";
    integer profile_phase_timestep = -1;
    longint profile_count[0:@COUNT_LAST@];
    bit profile_forward_active = 0, profile_learning_active = 0;
    bit profile_prepare_active = 0;
    longint profile_forward_start, profile_learning_start;
    longint profile_prepare_start;
    initial foreach (profile_count[k]) profile_count[k] = 0;

    task automatic profile_enter(input string next_phase);
        $display("SRNN_PROFILE_INTERVAL kind=stage phase=%s timestep=%0d start=%0d end=%0d cycles=%0d",
            profile_phase, profile_phase_timestep, profile_phase_start, cycles, cycles-profile_phase_start);
        profile_phase = next_phase;
        profile_phase_timestep = (next_phase == "config" || next_phase == "parameter_load" ||
                                 next_phase == "initial_state") ? -1 : timestep;
        profile_phase_start = cycles;
    endtask

    always @(posedge i_clk) begin
        if (!i_rst_n) profile_count[@RESET_COUNTER@]++;
        else begin
@COUNTER_UPDATES@
            if (u_final.link_plane_to_system_srnn_h40_prepare_valid &&
                u_final.link_system_to_plane_srnn_h40_prepare_ready) begin
                if (profile_prepare_active) $fatal(1,"overlapping M2 prepare interval");
                profile_prepare_active = 1;
                profile_prepare_start = cycles;
            end
            if (u_final.link_system_to_plane_srnn_h40_prepare_done) begin
                if (!profile_prepare_active) $fatal(1,"M2 prepare done without accepted start");
                $display("SRNN_PROFILE_INTERVAL kind=hardware phase=m2_prepare timestep=%0d start=%0d end=%0d cycles=%0d",
                    timestep,profile_prepare_start,cycles+1,cycles+1-profile_prepare_start);
                profile_prepare_active = 0;
            end
            if (u_final.link_plane_to_system_srnn_h40_timestep_start &&
                u_final.link_system_to_plane_srnn_h40_timestep_start_ready) begin
                if (profile_forward_active) $fatal(1,"overlapping M2 forward interval");
                profile_forward_active = 1;
                profile_forward_start = cycles;
            end
            if (u_final.link_system_to_plane_srnn_h40_timestep_done) begin
                if (!profile_forward_active) $fatal(1,"M2 forward done without accepted start");
                $display("SRNN_PROFILE_INTERVAL kind=hardware phase=m2_forward timestep=%0d start=%0d end=%0d cycles=%0d",
                    timestep,profile_forward_start,cycles+1,cycles+1-profile_forward_start);
                profile_forward_active = 0;
            end
            if ((@PURE_BUSY@) && !profile_learning_active) begin
                profile_learning_active = 1;
                profile_learning_start = cycles;
            end
            if (!(@PURE_BUSY@) && profile_learning_active) begin
                $display("SRNN_PROFILE_INTERVAL kind=hardware phase=pure_learning_busy timestep=%0d start=%0d end=%0d cycles=%0d",
                    timestep,profile_learning_start,cycles,cycles-profile_learning_start);
                profile_learning_active = 0;
            end
            if (o_learning_srnn_learning_commit)
                $display("SRNN_PROFILE_EVENT name=commit timestep=%0d cycle=%0d",timestep,cycles);
        end
    end

    task automatic profile_finish;
        profile_enter("finished");
        if (profile_forward_active || profile_learning_active || profile_prepare_active)
            $fatal(1,"profile interval still active at numerical completion");
@COUNTER_DUMP@
        $display("SRNN_PROFILE_COMPLETE batch=1 steps=%0d learning=%0d initialized=%0d end=%0d store_checks=%0d",
            T,learning,initialized_cycle,cycles,store_checks);
        $fflush();
    endtask
