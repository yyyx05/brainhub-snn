`timescale 1ns/1ps
`default_nettype none

// Passive export from the existing real Conv1..8/classifier T2 test. All
// stimulus, intermediate memories, state spill/restore and scoreboards remain
// owned by that test. Nothing is driven back into its DUT from this wrapper.
module tb_supplement_scnn_graph_capture;
    tb_unified_scnn_q015_real_conv1_to_classifier_t2_same_array_v1 graph();
    integer capture_fd;
    string capture_path;
    bit feature_seen [0:1][0:2047];
    bit logit_seen [0:1][0:10];
    integer feature_count = 0;
    integer logit_count = 0;
    // The public feature-trace port is disabled in learning-off mode. Observe
    // the inference scheduler's real accepted input, including zero values.
    wire feature_fire = graph.dut.u_core.g_scnn_graph.u_scnn_graph_client
        .u_classifier_tail.u_scheduler.i_feature_wave_valid &&
        graph.dut.u_core.g_scnn_graph.u_scnn_graph_client
        .u_classifier_tail.u_scheduler.o_feature_wave_ready;
    wire [7:0] feature_mask = graph.dut.u_core.g_scnn_graph.u_scnn_graph_client
        .u_classifier_tail.u_scheduler.i_feature_wave_mask;
    wire [10:0] feature_base = graph.dut.u_core.g_scnn_graph.u_scnn_graph_client
        .u_classifier_tail.u_scheduler.i_feature_wave_base;
    wire [127:0] feature_values = graph.dut.u_core.g_scnn_graph.u_scnn_graph_client
        .u_classifier_tail.u_scheduler.i_feature_wave_q8_8_flat;

    initial begin
        if (!$value$plusargs("SCNN_GRAPH_CAPTURE=%s", capture_path))
            $fatal(1, "SCNN_GRAPH_CAPTURE output path is required");
        capture_fd = $fopen(capture_path, "w");
        if (!capture_fd) $fatal(1, "cannot open graph capture: %s", capture_path);
        $fwrite(capture_fd, "kind,timestep,index,value\n");
    end

    always @(posedge graph.clk) begin : capture_only
        integer timestep;
        integer index;
        if (graph.rst_n) begin
            timestep = graph.timestep_phase;
            if (feature_fire) begin
                if (timestep < 0 || timestep > 1 || graph.layer_phase != 8)
                    $fatal(1, "unexpected classifier feature phase");
                for (int lane = 0; lane < 8; lane++) begin
                    if (feature_mask[lane]) begin
                        index = int'(feature_base) + lane;
                        if (index >= 2048 || feature_seen[timestep][index])
                            $fatal(1, "duplicate/out-of-range feature t=%0d i=%0d", timestep, index);
                        feature_seen[timestep][index] = 1'b1;
                        feature_count++;
                        $fwrite(capture_fd, "feature,%0d,%0d,%0d\n", timestep, index,
                            int'($signed(feature_values[lane*16 +: 16])));
                    end
                end
            end
            if (graph.dut.o_logit_wave_fire) begin
                if (timestep < 0 || timestep > 1 || graph.layer_phase != 8)
                    $fatal(1, "unexpected classifier logit phase");
                for (int lane = 0; lane < 12; lane++) begin
                    if (graph.dut.o_logit_wave_mask[lane]) begin
                        index = int'(graph.dut.o_logit_wave_base) + lane;
                        if (index >= 11 || logit_seen[timestep][index])
                            $fatal(1, "duplicate/out-of-range logit t=%0d i=%0d", timestep, index);
                        logit_seen[timestep][index] = 1'b1;
                        logit_count++;
                        $fwrite(capture_fd, "logit,%0d,%0d,%0d\n", timestep, index,
                            int'($signed(graph.dut.o_logit_wave_q8_8_flat[lane*16 +: 16])));
                    end
                end
            end
        end
    end

    final begin
        $display("SCNN_GRAPH_CAPTURE_COUNTS features=%0d logits=%0d", feature_count, logit_count);
        if (capture_fd) $fclose(capture_fd);
    end
endmodule
`default_nettype wire
