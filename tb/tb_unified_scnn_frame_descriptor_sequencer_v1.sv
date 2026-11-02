`timescale 1ns/1ps
module tb_unified_scnn_frame_descriptor_sequencer_v1;
    logic clk = 0;
    always #5 clk = !clk;
    logic rst_n = 0;
    logic clear_error = 0;
    logic request = 0, request_ready, batch_done;
    logic frame_valid, frame_ready, frame_done;
    logic sample_id;
    logic [1:0] timestep;
    logic [7:0] version;
    logic descriptor_valid = 0, descriptor_ready, descriptor_last = 0;
    logic [15:0] descriptor = 0, active_descriptor;
    logic descriptor_done, tail, start, start_ready = 0, layer_done = 0;
    logic busy, error, batch_error;
    int frames = 0, descriptors = 0;
    unified_scnn_selected_layers_graph_batch_coordinator_v1 #(
        .MAX_BATCH(2), .MAX_TIMESTEPS(2), .MAX_FLATTENED_SAMPLES(4),
        .ENABLE_TARGET_GRAPH(0), .TIMESTEP_WIDTH(2), .VERSION_WIDTH(8)
    ) batcher (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(1'b0), .i_clear_error(clear_error),
        .i_request_valid(request), .o_request_ready(request_ready),
        .i_learning_enable(1'b0), .i_batch_count(2'd2), .i_timestep(2'd0),
        .i_timestep_count(2'd2), .i_version(8'h31), .i_labels_flat(8'h21),
        .o_done(batch_done), .i_runtime_run_ready(1'b1), .i_runtime_run_done(1'b0),
        .i_sample_context_ready(1'b1),
        .o_graph_start_valid(frame_valid), .i_graph_start_ready(frame_ready),
        .o_graph_sample(sample_id), .o_graph_timestep(timestep), .o_graph_version(version),
        .i_graph_done(frame_done), .o_sticky_error(batch_error)
    );
    unified_scnn_frame_descriptor_sequencer_v1 #(.DESCRIPTOR_WIDTH(16)) dut (
        .i_clk(clk), .i_rst_n(rst_n), .i_clear_error(clear_error),
        .i_frame_start_valid(frame_valid), .o_frame_start_ready(frame_ready),
        .o_frame_done(frame_done), .i_descriptor_valid(descriptor_valid),
        .o_descriptor_ready(descriptor_ready), .i_descriptor(descriptor),
        .i_descriptor_last(descriptor_last), .o_descriptor_done(descriptor_done),
        .o_descriptor(active_descriptor), .o_tail_active(tail),
        .o_layer_start_valid(start), .i_layer_start_ready(start_ready),
        .i_layer_done(layer_done), .o_busy(busy), .o_sticky_error(error)
    );
    task automatic tick;
        @(posedge clk); #1;
    endtask
    task automatic reset_all;
        @(negedge clk); rst_n = 0; request = 0; descriptor_valid = 0;
        layer_done = 0; start_ready = 0;
        tick(); tick(); @(negedge clk); rst_n = 1; tick();
        if (busy || error || frame_done || descriptor_done || batch_error)
            $fatal(1, "reset did not clear the transaction");
    endtask
    task automatic open_batch;
        @(negedge clk); request = 1;
        while (!request_ready) tick();
        tick(); @(negedge clk); request = 0;
        while (!descriptor_ready) tick();
    endtask
    initial begin
        reset_all(); open_batch();
        for (int frame = 0; frame < 4; frame++) begin
            for (int layer = 0; layer < 8; layer++) begin
                while (!descriptor_ready) tick();
                if (int'(sample_id) != frame / 2 || int'(timestep) != frame % 2 || version != 8'h31)
                    $fatal(1, "coordinator advanced before frame completion");
                @(negedge clk); descriptor = 16'(frame*8+layer);
                descriptor_last = layer == 7; descriptor_valid = 1;
                tick(); @(negedge clk); descriptor_valid = 0; descriptor = 16'hffff;
                repeat (3) begin
                    tick();
                    if (!start || active_descriptor != 16'(frame*8+layer) || tail != (layer == 7))
                        $fatal(1, "descriptor changed under start backpressure");
                end
                @(negedge clk); start_ready = 1; tick();
                @(negedge clk); start_ready = 0;
                repeat (4) begin
                    tick();
                    if (descriptor_ready || frame_done || descriptor_done || start ||
                        int'(sample_id) != frame/2 || int'(timestep) != frame%2)
                        $fatal(1, "premature descriptor/frame retirement");
                end
                @(negedge clk); layer_done = 1; tick();
                if (!descriptor_done || frame_done != (layer == 7))
                    $fatal(1, "incorrect final-descriptor completion");
                descriptors++;
                if (frame_done) frames++;
                @(negedge clk); layer_done = 0; tick();
            end
        end
        while (!batch_done) tick();
        if (frames != 4 || descriptors != 32 || error || batch_error)
            $fatal(1, "B2T2 frame counts or errors");
        // Reset while waiting for a descriptor, stalled start, and layer done.
        for (int stage = 0; stage < 3; stage++) begin
            reset_all(); open_batch();
            if (stage > 0) begin
                @(negedge clk); descriptor_valid = 1; descriptor_last = 1;
                tick(); @(negedge clk); descriptor_valid = 0;
            end
            if (stage > 1) begin
                start_ready = 1; tick(); @(negedge clk); start_ready = 0;
            end
            reset_all();
        end
        @(negedge clk); layer_done = 1; tick();
        if (!error) $fatal(1, "unexpected completion not rejected");
        @(negedge clk); layer_done = 0; clear_error = 1; tick();
        if (error) $fatal(1, "idle clear_error failed");
        $display("PASS: SCNN_FRAME_DESCRIPTOR_B2T2 frames=4 descriptors=32 reset_states=3");
        $finish;
    end
    initial begin
        repeat (10000) tick();
        $fatal(1, "timeout");
    end
endmodule
