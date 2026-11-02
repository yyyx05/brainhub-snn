`timescale 1ns/1ps
`default_nettype none

module tb_unified_array16_shared_weight_cache_refill_plane_v1;
    localparam int PARAM_DEPTH = 128;
    localparam int PARAM_ADDR_WIDTH = 7;
    localparam int TAG_WIDTH = 8;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic fc_refill_valid;
    logic fc_refill_ready;
    logic fc_dirty_feature;
    logic fc_forward_read_valid;
    logic fc_forward_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] fc_forward_read_addr;
    logic [TAG_WIDTH-1:0] fc_forward_read_tag;
    logic fc_forward_response_valid;
    logic fc_forward_response_ready;
    logic [TAG_WIDTH-1:0] fc_forward_response_tag;
    logic signed [15:0] fc_forward_response_code;
    logic fc_refill_done;
    logic [31:0] fc_rows_published;

    logic scnn_context_valid;
    logic scnn_context_ready;
    logic scnn_forward_read_valid;
    logic scnn_forward_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] scnn_forward_read_addr;
    logic [TAG_WIDTH-1:0] scnn_forward_read_tag;
    logic scnn_forward_response_valid;
    logic scnn_forward_response_ready;
    logic [TAG_WIDTH-1:0] scnn_forward_response_tag;
    logic signed [15:0] scnn_forward_response_code;
    logic [3:0] scnn_fill_valid;
    logic [3:0] scnn_fill_ready;
    logic [7:0] scnn_fill_tile_flat;
    logic [3:0] scnn_fill_bank;
    logic [43:0] scnn_fill_addr_flat;
    logic [799:0] scnn_fill_data_flat;
    logic scnn_busy;
    logic scnn_refill_done;
    logic scnn_error;

    logic store_read_valid;
    logic store_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] store_read_addr;
    logic [TAG_WIDTH-1:0] store_read_tag;
    logic store_response_valid_q;
    logic store_response_ready;
    logic [TAG_WIDTH-1:0] store_response_tag_q;
    logic signed [15:0] store_response_code_q;
    logic store_pending_q;
    logic [PARAM_ADDR_WIDTH-1:0] store_pending_addr_q;
    logic [TAG_WIDTH-1:0] store_pending_tag_q;
    logic forward_arbiter_busy;
    logic forward_arbiter_error;
    logic signed [15:0] parameter_mem [0:PARAM_DEPTH-1];

    logic [3:0] shared_fill_valid;
    logic [3:0] shared_fill_ready;
    logic [7:0] shared_fill_tile_flat;
    logic [3:0] shared_fill_bank;
    logic [43:0] shared_fill_addr_flat;
    logic [799:0] shared_fill_data_flat;
    logic [1:0] fill_grant;
    logic refill_plane_busy;
    logic refill_plane_quiescent;
    logic refill_plane_error;

    logic physical_fill_ready;
    logic physical_read_valid;
    logic physical_read_ready;
    logic [1:0] physical_mode;
    logic physical_scnn_pair;
    logic [9:0] physical_read_addr0;
    logic [9:0] physical_read_addr1;
    logic [9:0] physical_read_addr2;
    logic [9:0] physical_read_addr3;
    logic [3:0] physical_read_subword0;
    logic [3:0] physical_read_subword1;
    logic [3:0] physical_read_subword2;
    logic [3:0] physical_read_subword3;
    logic physical_result_valid;
    logic physical_result_ready;
    logic [999:0] physical_term0_flat;
    logic [999:0] physical_term1_flat;
    logic [999:0] physical_term2_flat;
    logic [999:0] physical_term3_flat;

    int fc_fill_count;
    int scnn_fill_count;
    int fc_read_count;
    int scnn_read_count;

    assign shared_fill_ready = {3'b111, physical_fill_ready};
    assign store_read_ready = !store_pending_q && !store_response_valid_q;

    unified_array16_shared_weight_cache_refill_plane_v1 #(
        .PARAM_DEPTH(PARAM_DEPTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .VERSION_WIDTH(16), .FEATURE_COUNT_WIDTH(2),
        .HIDDEN_COUNT_WIDTH(7), .ARRAY_WEIGHT_ADDR_WIDTH(11)
    ) u_refill_plane (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(1'b0),
        .i_clear_error(1'b0),
        .i_fc_refill_valid(fc_refill_valid),
        .o_fc_refill_ready(fc_refill_ready), .i_fc_refill_all(1'b0),
        .i_fc_dirty_feature({1'b0, fc_dirty_feature}),
        .i_fc_dirty_tile(4'd0), .i_fc_parameter_base(7'd0),
        .i_fc_hidden_stride(7'd40), .i_fc_cache_row_base(10'd0),
        .i_fc_feature_count(2'd2),
        .i_fc_hidden_count(7'd40), .i_fc_tile_count(5'd1),
        .i_fc_version(16'd3), .i_array_idle(1'b1),
        .o_forward_read_valid(fc_forward_read_valid),
        .i_forward_read_ready(fc_forward_read_ready),
        .o_forward_read_addr(fc_forward_read_addr),
        .o_forward_read_tag(fc_forward_read_tag),
        .i_forward_response_valid(fc_forward_response_valid),
        .o_forward_response_ready(fc_forward_response_ready),
        .i_forward_response_tag(fc_forward_response_tag),
        .i_forward_response_code(fc_forward_response_code),
        .i_forward_response_error(1'b0),
        .i_topology_fill_busy(scnn_busy),
        .i_topology_fill_valid(scnn_fill_valid),
        .o_topology_fill_ready(scnn_fill_ready),
        .i_topology_fill_tile_flat(scnn_fill_tile_flat),
        .i_topology_fill_bank(scnn_fill_bank),
        .i_topology_fill_addr_flat(scnn_fill_addr_flat),
        .i_topology_fill_data_flat(scnn_fill_data_flat),
        .o_weight_fill_valid(shared_fill_valid),
        .i_weight_fill_ready(shared_fill_ready),
        .o_weight_fill_tile_flat(shared_fill_tile_flat),
        .o_weight_fill_bank(shared_fill_bank),
        .o_weight_fill_addr_flat(shared_fill_addr_flat),
        .o_weight_fill_data_flat(shared_fill_data_flat),
        .o_fc_refill_done(fc_refill_done),
        .o_fc_refill_busy(),
        .o_fc_rows_published(fc_rows_published),
        .o_fc_published_version(), .o_fill_grant(fill_grant),
        .o_busy(refill_plane_busy), .o_quiescent(refill_plane_quiescent),
        .o_sticky_error(refill_plane_error)
    );

    unified_scnn_h40_conv_context_refill_v1 #(
        .PARAM_DEPTH(PARAM_DEPTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .MAX_INPUT_CHANNELS(2), .MAX_CONTRIBUTIONS(18),
        .ARRAY_WEIGHT_ADDR_WIDTH(11),
        .INPUT_CHANNEL_COUNT_WIDTH(2), .CONTRIBUTION_COUNT_WIDTH(5)
    ) u_scnn_refill (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(1'b0),
        .i_clear_error(1'b0),
        .i_context_valid(scnn_context_valid),
        .o_context_ready(scnn_context_ready),
        .i_context_load_weights(1'b1),
        .i_output_channel_base(10'd0),
        .i_output_channel_count(5'd1), .i_input_channel_count(2'd1),
        .i_context_pair(1'b1), .i_reserved_pair_valid(1'b1),
        .i_reserved_pair(1'b0), .i_conv_parameter_base(7'd80),
        .i_version(16'd5), .i_refill_safe(1'b1),
        .o_forward_read_valid(scnn_forward_read_valid),
        .i_forward_read_ready(scnn_forward_read_ready),
        .o_forward_read_addr(scnn_forward_read_addr),
        .o_forward_read_tag(scnn_forward_read_tag),
        .i_forward_response_valid(scnn_forward_response_valid),
        .o_forward_response_ready(scnn_forward_response_ready),
        .i_forward_response_tag(scnn_forward_response_tag),
        .i_forward_response_code(scnn_forward_response_code),
        .i_forward_response_error(1'b0),
        .o_weight_fill_valid(scnn_fill_valid),
        .i_weight_fill_ready(scnn_fill_ready),
        .o_weight_fill_tile_flat(scnn_fill_tile_flat),
        .o_weight_fill_bank(scnn_fill_bank),
        .o_weight_fill_addr_flat(scnn_fill_addr_flat),
        .o_weight_fill_data_flat(scnn_fill_data_flat),
        .o_busy(scnn_busy), .o_refill_done(scnn_refill_done),
        .o_sticky_error(scnn_error)
    );

    unified_tp_two_client_forward_read_arbiter #(
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(16)
    ) u_forward_arbiter (
        .i_clk(clk), .i_rst_n(rst_n), .i_clear_error(1'b0),
        .i_c0_valid(fc_forward_read_valid),
        .o_c0_ready(fc_forward_read_ready),
        .i_c0_addr(fc_forward_read_addr), .i_c0_tag(fc_forward_read_tag),
        .o_c0_response_valid(fc_forward_response_valid),
        .i_c0_response_ready(fc_forward_response_ready),
        .o_c0_response_tag(fc_forward_response_tag),
        .o_c0_response_code(fc_forward_response_code),
        .o_c0_response_error(),
        .i_c1_valid(scnn_forward_read_valid),
        .o_c1_ready(scnn_forward_read_ready),
        .i_c1_addr(scnn_forward_read_addr),
        .i_c1_tag(scnn_forward_read_tag),
        .o_c1_response_valid(scnn_forward_response_valid),
        .i_c1_response_ready(scnn_forward_response_ready),
        .o_c1_response_tag(scnn_forward_response_tag),
        .o_c1_response_code(scnn_forward_response_code),
        .o_c1_response_error(),
        .o_store_valid(store_read_valid), .i_store_ready(store_read_ready),
        .o_store_addr(store_read_addr), .o_store_tag(store_read_tag),
        .i_store_response_valid(store_response_valid_q),
        .o_store_response_ready(store_response_ready),
        .i_store_response_tag(store_response_tag_q),
        .i_store_response_code(store_response_code_q),
        .i_store_response_error(1'b0),
        .o_busy(forward_arbiter_busy),
        .o_protocol_error(forward_arbiter_error)
    );

    unified_dual_role_weight_store_p4all_candidate_v1 u_physical_cache (
        .i_clk(clk), .i_rst_n(rst_n),
        .i_fill_valid(shared_fill_valid[0] &&
            shared_fill_tile_flat[0 +: 2] == 2'd0),
        .o_fill_ready(physical_fill_ready),
        .i_fill_bank({shared_fill_addr_flat[10], shared_fill_bank[0]}),
        .i_fill_addr(shared_fill_addr_flat[0 +: 10]),
        .i_fill_data(shared_fill_data_flat[0 +: 200]),
        .i_read_valid(physical_read_valid),
        .o_read_ready(physical_read_ready), .i_mode(physical_mode),
        .i_scnn_active_pair(physical_scnn_pair),
        .i_read_addr0(physical_read_addr0),
        .i_read_addr1(physical_read_addr1),
        .i_read_addr2(physical_read_addr2),
        .i_read_addr3(physical_read_addr3),
        .i_read_subword0(physical_read_subword0),
        .i_read_subword1(physical_read_subword1),
        .i_read_subword2(physical_read_subword2),
        .i_read_subword3(physical_read_subword3),
        .o_read_result_valid(physical_result_valid),
        .i_read_result_ready(physical_result_ready),
        .o_term0_flat(physical_term0_flat),
        .o_term1_flat(physical_term1_flat),
        .o_term2_flat(physical_term2_flat),
        .o_term3_flat(physical_term3_flat)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            store_pending_q <= 1'b0;
            store_pending_addr_q <= '0;
            store_pending_tag_q <= '0;
            store_response_valid_q <= 1'b0;
            store_response_tag_q <= '0;
            store_response_code_q <= '0;
        end else begin
            if (store_response_valid_q && store_response_ready)
                store_response_valid_q <= 1'b0;
            if (store_pending_q) begin
                store_pending_q <= 1'b0;
                store_response_valid_q <= 1'b1;
                store_response_tag_q <= store_pending_tag_q;
                store_response_code_q <= parameter_mem[store_pending_addr_q];
            end
            if (store_read_valid && store_read_ready) begin
                store_pending_q <= 1'b1;
                store_pending_addr_q <= store_read_addr;
                store_pending_tag_q <= store_read_tag;
                if (fc_forward_read_valid)
                    fc_read_count <= fc_read_count + 1;
                else
                    scnn_read_count <= scnn_read_count + 1;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst_n && |(shared_fill_valid & shared_fill_ready)) begin
            if (shared_fill_valid[3:1] != 0 ||
                    shared_fill_tile_flat[0 +: 2] != 2'd0)
                $fatal(1, "test issued a fill outside physical tile 0");
            unique case (fill_grant)
                2'b01: fc_fill_count <= fc_fill_count + 1;
                2'b10: scnn_fill_count <= scnn_fill_count + 1;
                default: $fatal(1, "accepted fill without one transaction owner");
            endcase
        end
    end

    task automatic run_fc_refill(input logic feature);
        @(negedge clk);
        fc_dirty_feature = feature;
        fc_refill_valid = 1'b1;
        do @(posedge clk); while (!fc_refill_ready);
        @(negedge clk);
        fc_refill_valid = 1'b0;
        do @(posedge clk); while (!fc_refill_done);
        @(posedge clk);
        if (fc_rows_published != 1)
            $fatal(1, "targeted FC refill did not publish exactly one row");
    endtask

    task automatic run_scnn_refill;
        @(negedge clk);
        scnn_context_valid = 1'b1;
        do @(posedge clk); while (!scnn_context_ready);
        @(negedge clk);
        scnn_context_valid = 1'b0;
        @(posedge clk);
        if (!scnn_refill_done && scnn_busy)
            $fatal(1, "SCNN refill did not retire cleanly");
    endtask

    task automatic start_physical_read(input logic [1:0] mode,
            input logic scnn_pair);
        @(negedge clk);
        physical_mode = mode;
        physical_scnn_pair = scnn_pair;
        physical_read_valid = 1'b1;
        do @(posedge clk); while (!physical_read_ready);
        @(negedge clk);
        physical_read_valid = 1'b0;
        wait (physical_result_valid);
        #1;
    endtask

    task automatic finish_physical_read;
        @(negedge clk);
        physical_result_ready = 1'b1;
        @(negedge clk);
        physical_result_ready = 1'b0;
    endtask

    initial begin
        fc_refill_valid = 1'b0;
        fc_dirty_feature = 1'b0;
        scnn_context_valid = 1'b0;
        physical_read_valid = 1'b0;
        physical_result_ready = 1'b0;
        physical_mode = 2'd0;
        physical_scnn_pair = 1'b0;
        physical_read_addr0 = 10'd0;
        physical_read_addr1 = 10'd0;
        physical_read_addr2 = 10'd0;
        physical_read_addr3 = 10'd0;
        physical_read_subword0 = 4'd0;
        physical_read_subword1 = 4'd0;
        physical_read_subword2 = 4'd0;
        physical_read_subword3 = 4'd0;
        fc_fill_count = 0;
        scnn_fill_count = 0;
        fc_read_count = 0;
        scnn_read_count = 0;
        for (int i = 0; i < PARAM_DEPTH; i++)
            parameter_mem[i] = 16'sd0;
        for (int i = 0; i < 80; i++)
            parameter_mem[i] = $signed((i % 31) - 15);
        for (int i = 80; i < 98; i++)
            parameter_mem[i] = $signed(100 + i);

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        run_fc_refill(1'b0);
        start_physical_read(2'd0, 1'b0);
        if ($signed(physical_term0_flat[0 +: 25]) != parameter_mem[0] ||
                $signed(physical_term0_flat[39*25 +: 25]) !=
                    parameter_mem[39])
            $fatal(1, "FC feature-0 row was not visible in physical bank 0");
        finish_physical_read();

        run_scnn_refill();
        start_physical_read(2'd2, 1'b1);
        if ($signed(physical_term0_flat[0 +: 25]) != parameter_mem[80] ||
                $signed(physical_term1_flat[0 +: 25]) != parameter_mem[81])
            $fatal(1, "SCNN pair-1 row was not visible in physical banks 2/3");
        finish_physical_read();

        run_fc_refill(1'b1);
        start_physical_read(2'd0, 1'b0);
        if ($signed(physical_term1_flat[0 +: 25]) != parameter_mem[40] ||
                $signed(physical_term1_flat[39*25 +: 25]) !=
                    parameter_mem[79])
            $fatal(1, "FC feature-1 row was not visible after SCNN refill");
        finish_physical_read();

        if (fc_fill_count != 2 || scnn_fill_count != 2 ||
                fc_read_count != 80 || scnn_read_count != 9)
            $fatal(1, "transaction counts mismatch fills=%0d/%0d reads=%0d/%0d",
                fc_fill_count, scnn_fill_count, fc_read_count,
                scnn_read_count);
        if (refill_plane_busy || !refill_plane_quiescent ||
                fill_grant != 2'b00 || forward_arbiter_busy ||
                refill_plane_error || scnn_error || forward_arbiter_error)
            $fatal(1, "shared refill plane did not return to clean idle");

        $display("ARRAY16_SHARED_WEIGHT_CACHE_REFILL_PLANE_PASS sequence=FC_SCNN_FC physical_cache=one fc_rows=2 scnn_rows=2 authoritative_store=shared no_reset=1");
        $finish;
    end

    initial begin
        repeat (5000) @(posedge clk);
        $fatal(1, "timeout");
    end
endmodule

`default_nettype wire
