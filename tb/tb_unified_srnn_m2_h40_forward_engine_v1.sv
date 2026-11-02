`timescale 1ns/1ps
`default_nettype none

module tb_unified_srnn_m2_h40_forward_engine_v1;
    localparam int PARAM_DEPTH = 128;
    localparam int PARAM_ADDR_WIDTH = 8;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic clear_error = 1'b0;
    logic abort_all = 1'b0;
    logic cfg_valid = 1'b0;
    logic cfg_ready;
    logic cfg_error;
    logic prepare_valid = 1'b0;
    logic prepare_ready;
    logic [7:0] prepare_version = '0;
    logic cache_invalidate = 1'b0;
    logic prepare_done;
    logic cache_valid;
    logic [7:0] cache_version;
    logic timestep_start_ready;

    logic read_valid;
    logic read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] read_addr;
    logic [7:0] read_tag;
    logic response_valid;
    logic response_ready;
    logic [7:0] response_tag;
    logic signed [15:0] response_code;
    logic response_valid_q;
    logic [7:0] response_tag_q;
    logic signed [15:0] response_code_q;
    logic signed [15:0] model_mem [0:PARAM_DEPTH-1];

    logic [3:0] fill_valid;
    logic [7:0] fill_tile_flat;
    logic [3:0] fill_bank;
    logic [43:0] fill_addr_flat;
    logic [799:0] fill_data_flat;
    logic busy;
    logic sticky_error;
    logic protocol_error;
    logic overflow;
    integer read_count = 0;
    integer fill_count = 0;

    always #5 clk = ~clk;

    assign read_ready = !response_valid_q || response_ready;
    assign response_valid = response_valid_q;
    assign response_tag = response_tag_q;
    assign response_code = response_code_q;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            response_valid_q <= 1'b0;
            response_tag_q <= '0;
            response_code_q <= '0;
            read_count <= 0;
            fill_count <= 0;
        end else begin
            if (response_valid_q && response_ready)
                response_valid_q <= 1'b0;
            if (read_valid && read_ready) begin
                response_valid_q <= 1'b1;
                response_tag_q <= read_tag;
                response_code_q <= model_mem[read_addr[6:0]];
                read_count <= read_count + 1;
            end
            if (|fill_valid)
                fill_count <= fill_count + 1;
        end
    end

    unified_srnn_m2_h40_forward_engine_v1 #(
        .PARAM_DEPTH(PARAM_DEPTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(8),
        .BATCH_MAX(1), .TIMESTEP_MAX(4), .BACKBONE_MAX(4),
        .HIDDEN_MAX(4), .CLASS_MAX(2), .SAMPLE_ID_WIDTH(1),
        .TIMESTEP_ID_WIDTH(2), .BACKBONE_INDEX_WIDTH(2),
        .HIDDEN_INDEX_WIDTH(3), .CLASS_INDEX_WIDTH(2),
        .VERSION_WIDTH(8), .WEIGHT_DEPTH(1156),
        .WEIGHT_ADDR_WIDTH(11), .MEMORY_DEPTH(1024),
        .MEMORY_ADDR_WIDTH(10), .WORK_ADDR_WIDTH(9),
        .PHYSICAL_TILE_COUNT(12), .OUTPUT_TILE(7)
    ) dut (
        .i_clk(clk), .i_rst_n(rst_n), .i_clear_error(clear_error),
        .i_abort(abort_all), .i_cfg_valid(cfg_valid),
        .o_cfg_ready(cfg_ready), .i_cfg_batch_count(1'd1),
        .i_cfg_backbone_count(2'd2), .i_cfg_hidden_count(3'd4),
        .i_cfg_class_count(2'd2),
        .i_cfg_wh1_parameter_base(PARAM_ADDR_WIDTH'(5)),
        .i_cfg_wout_parameter_base(PARAM_ADDR_WIDTH'(32)),
        .i_cfg_wh1_cache_base(10'd10),
        .i_cfg_wout_cache_base(10'd40),
        .i_cfg_hidden_work_base(9'd0),
        .i_cfg_output_work_base(9'd20), .o_cfg_error(cfg_error),
        .i_prepare_valid(prepare_valid), .o_prepare_ready(prepare_ready),
        .i_prepare_version(prepare_version),
        .i_cache_invalidate(cache_invalidate),
        .o_prepare_done(prepare_done), .o_cache_valid(cache_valid),
        .o_cache_version(cache_version), .i_timestep_start(1'b0),
        .o_timestep_start_ready(timestep_start_ready),
        .i_timestep_id('0), .i_sequence_version(8'd1),
        .i_include_target_path(1'b1), .i_observation_valid(1'b0),
        .o_observation_ready(), .i_observation_path(1'b0),
        .i_observation_sample_id('0), .i_observation_backbone_id('0),
        .i_observation_q14('0), .i_observation_last(1'b0),
        .i_observation_done(1'b0), .o_forward_read_valid(read_valid),
        .i_forward_read_ready(read_ready), .o_forward_read_addr(read_addr),
        .o_forward_read_tag(read_tag),
        .i_forward_response_valid(response_valid),
        .o_forward_response_ready(response_ready),
        .i_forward_response_tag(response_tag),
        .i_forward_response_code(response_code),
        .i_forward_response_error(1'b0), .i_array_idle(1'b1),
        .o_weight_fill_valid(fill_valid), .i_weight_fill_ready(4'b1111),
        .o_weight_fill_tile_flat(fill_tile_flat),
        .o_weight_fill_bank(fill_bank),
        .o_weight_fill_addr_flat(fill_addr_flat),
        .o_weight_fill_data_flat(fill_data_flat),
        .o_maintenance_start_valid(),
        .i_maintenance_start_ready(1'b0),
        .o_maintenance_release_valid(),
        .i_maintenance_release_ready(1'b0), .o_maintenance_tile(),
        .o_state_write_valid(), .i_state_write_ready(1'b0),
        .o_state_write_addr(), .o_state_write_data(),
        .o_state_write_mask(), .o_trace_write_valid(),
        .i_trace_write_ready(1'b0), .o_trace_write_addr(),
        .o_trace_write_data(), .o_trace_write_mask(),
        .o_layer_valid(), .i_layer_ready(1'b0), .o_layer_mode(),
        .o_layer_neuron_count(), .o_layer_feedforward_group_count(),
        .o_layer_recurrent_group_count(), .o_layer_recurrent_enable(),
        .o_layer_leak(), .o_layer_threshold(), .o_layer_reset_to_zero(),
        .o_layer_trace_decay(), .o_layer_transaction_tag(),
        .o_layer_work_base_addr(), .o_layer_destination_tile_base(),
        .o_group_valid(), .i_group_ready(1'b0), .o_group_recurrent(),
        .o_group_source_valid(), .o_group_weight_addr0(),
        .o_group_weight_addr1(), .o_group_weight_addr2(),
        .o_group_weight_addr3(), .i_layer_done(1'b0),
        .i_layer_sticky_error(1'b0),
        .i_array_observation_valid(1'b0),
        .o_array_observation_ready(), .i_array_observation_mode('0),
        .i_array_observation_transaction_tag('0),
        .i_array_observation_neuron_id('0),
        .i_array_observation_spike(1'b0),
        .i_array_observation_pre_state('0),
        .i_array_observation_trace_next('0),
        .i_array_observation_last_in_record(1'b0),
        .o_hidden_valid(), .i_hidden_ready(1'b1), .o_hidden_path(),
        .o_hidden_sample_id(), .o_hidden_id(),
        .o_hidden_membrane_pre_q14(), .o_hidden_membrane_next_q14(),
        .o_hidden_spike(), .o_hidden_trace_q14(),
        .o_hidden_output_feature_q14(), .o_hidden_last(),
        .o_logit_valid(), .i_logit_ready(1'b1), .o_logit_sample_id(),
        .o_logit_class_id(), .o_logit_q28(), .o_logit_last(),
        .o_timestep_id(), .o_timestep_done(), .o_busy(busy),
        .o_sticky_error(sticky_error), .o_protocol_error(protocol_error),
        .o_overflow(overflow)
    );

    task automatic run_prepare(input logic [7:0] version);
        begin
            wait (prepare_ready);
            @(negedge clk);
            prepare_version = version;
            prepare_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            prepare_valid = 1'b0;
            wait (prepare_done);
            if (!cache_valid || cache_version != version)
                $fatal(1, "cache publication mismatch version=%0d", version);
        end
    endtask

    initial begin
        for (int address = 0; address < PARAM_DEPTH; address++)
            model_mem[address] = 16'(address * 17 - 300);

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);
        wait (cfg_ready);
        @(negedge clk);
        cfg_valid = 1'b1;
        @(posedge clk);
        @(negedge clk);
        cfg_valid = 1'b0;
        if (cfg_error)
            $fatal(1, "engine configuration failed");

        run_prepare(8'd5);
        if (read_count != 16 || fill_count != 6 ||
                !timestep_start_ready || busy)
            $fatal(1, "initial prepare mismatch reads=%0d fills=%0d",
                read_count, fill_count);

        run_prepare(8'd5);
        if (read_count != 16 || fill_count != 6)
            $fatal(1, "same-version prepare did not bypass refill");

        @(negedge clk);
        cache_invalidate = 1'b1;
        @(posedge clk);
        @(negedge clk);
        cache_invalidate = 1'b0;
        if (cache_valid || timestep_start_ready)
            $fatal(1, "invalidated cache remained executable");

        run_prepare(8'd6);
        if (read_count != 32 || fill_count != 12 || sticky_error ||
                protocol_error || overflow || busy)
            $fatal(1, "version refresh mismatch reads=%0d fills=%0d",
                read_count, fill_count);

        // A residency eviction has priority over a same-version cache hit.
        @(negedge clk);
        prepare_version = 8'd6;
        prepare_valid = 1'b1;
        cache_invalidate = 1'b1;
        #1;
        if (prepare_ready || timestep_start_ready)
            $fatal(1, "eviction admitted stale prepare or timestep");
        repeat (2) @(posedge clk);
        @(negedge clk);
        if (prepare_done || cache_valid || read_count != 32)
            $fatal(1, "eviction acknowledged stale cache hit");
        cache_invalidate = 1'b0;
        #1;
        if (!prepare_ready)
            $fatal(1, "prepare did not recover after eviction");
        @(posedge clk);
        @(negedge clk);
        prepare_valid = 1'b0;
        wait (prepare_done);
        if (!cache_valid || cache_version != 8'd6 || read_count != 48 ||
                fill_count != 18 || sticky_error || protocol_error || overflow)
            $fatal(1, "same-version eviction did not refill exactly once");

        $display("PASS: M2 H40 forward engine publishes two matrices, bypasses valid same-version refill and handles eviction/prepare collision");
        $finish;
    end

    initial begin
        repeat (5000) @(posedge clk);
        $fatal(1, "timeout");
    end
endmodule

`default_nettype wire
