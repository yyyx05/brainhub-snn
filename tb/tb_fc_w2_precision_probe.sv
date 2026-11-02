`timescale 1ns/1ps
`default_nettype none

module tb_fc_w2_precision_probe;
    logic clk = 0;
    always #5 clk = !clk;
    logic rst_n = 0;
    logic use_generic = 0;
    logic launch_fc = 0, launch_generic = 0;
    logic signed [15:0] test_code = 0;
    logic signed [15:0] cases [0:31];
    string cases_path;
    int case_count;
    int cycles = 0, read_count = 0, response_count = 0, fill_count = 0;
    int rejected = 0, accepted = 0, controls = 0;

    wire fc_ready, fc_done, fc_busy, fc_error;
    wire [31:0] fc_rows;
    wire [15:0] fc_version;
    wire generic_ready, generic_done, generic_busy, generic_error;
    wire [1:0] read_valid, response_ready;
    wire [7:0] read_addr [0:1];
    wire [15:0] read_tag [0:1];
    logic [1:0] response_valid = 0;
    logic [15:0] response_tag [0:1];
    wire [3:0] fill_valid [0:1];
    wire [3:0] fill_ready [0:1];
    wire [7:0] fill_tile [0:1];
    wire [3:0] fill_bank [0:1];
    wire [43:0] fill_addr [0:1];
    wire [799:0] fill_data [0:1];
    wire [3:0] selected_valid = fill_valid[use_generic];
    wire [3:0] selected_bank = fill_bank[use_generic];
    wire [43:0] selected_addr = fill_addr[use_generic];
    wire [799:0] selected_data = fill_data[use_generic];
    wire store_fill_ready;
    wire allow_fill = (cycles % 5) != 2;
    wire [3:0] selected_ready = {3'b000, store_fill_ready && allow_fill};
    assign fill_ready[0] = use_generic ? 4'b0 : selected_ready;
    assign fill_ready[1] = use_generic ? selected_ready : 4'b0;

    logic store_read_valid = 0;
    wire store_read_ready, store_result_valid;
    logic [1:0] store_mode = 0;
    wire [999:0] term0, term1, term2, term3;
    logic stalled_q = 0;
    logic [199:0] stalled_data_q;
    logic [10:0] stalled_addr_q;

    unified_fc_h40_w0_cache_refill_scheduler_v1 #(
        .PARAM_DEPTH(64), .PARAM_ADDR_WIDTH(8), .ARRAY_TILES(12)
    ) fc_dut (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(1'b0), .i_clear_error(1'b0),
        .i_refill_valid(launch_fc), .o_refill_ready(fc_ready), .i_refill_all(1'b1),
        .i_dirty_feature(12'd0), .i_dirty_tile(4'd0), .i_parameter_base(8'd0),
        .i_hidden_stride(8'd10), .i_cache_row_base(10'd0),
        .i_feature_count(12'd1), .i_hidden_count(10'd10), .i_tile_count(5'd1),
        .i_version(16'd7), .i_array_idle(1'b1),
        .o_forward_read_valid(read_valid[0]), .i_forward_read_ready(1'b1),
        .o_forward_read_addr(read_addr[0]), .o_forward_read_tag(read_tag[0]),
        .i_forward_response_valid(response_valid[0]),
        .o_forward_response_ready(response_ready[0]),
        .i_forward_response_tag(response_tag[0]), .i_forward_response_code(test_code),
        .i_forward_response_error(1'b0), .o_weight_fill_valid(fill_valid[0]),
        .i_weight_fill_ready(fill_ready[0]), .o_weight_fill_tile_flat(fill_tile[0]),
        .o_weight_fill_bank(fill_bank[0]), .o_weight_fill_addr_flat(fill_addr[0]),
        .o_weight_fill_data_flat(fill_data[0]), .o_refill_done(fc_done),
        .o_rows_published(fc_rows), .o_published_version(fc_version),
        .o_busy(fc_busy), .o_sticky_error(fc_error)
    );

    // Legal INT8 leaf control, deliberately not represented as a final FC route.
    unified_h40_scalar_parameter_row_publisher_v1 #(
        .PARAM_DEPTH(64), .PARAM_ADDR_WIDTH(8)
    ) int8_control (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(1'b0), .i_clear_error(1'b0),
        .i_command_valid(launch_generic), .o_command_ready(generic_ready),
        .i_target_tile(4'd0), .i_target_bank(2'd0), .i_target_row(10'd0),
        .i_scalar_base(8'd0), .i_scalar_stride(8'd1), .i_scalar_count(6'd10),
        .i_element_format(2'd1), .i_version(16'd7), .i_array_idle(1'b1),
        .o_forward_read_valid(read_valid[1]), .i_forward_read_ready(1'b1),
        .o_forward_read_addr(read_addr[1]), .o_forward_read_tag(read_tag[1]),
        .i_forward_response_valid(response_valid[1]),
        .o_forward_response_ready(response_ready[1]),
        .i_forward_response_tag(response_tag[1]), .i_forward_response_code(test_code),
        .i_forward_response_error(1'b0), .o_weight_fill_valid(fill_valid[1]),
        .i_weight_fill_ready(fill_ready[1]), .o_weight_fill_tile_flat(fill_tile[1]),
        .o_weight_fill_bank(fill_bank[1]), .o_weight_fill_addr_flat(fill_addr[1]),
        .o_weight_fill_data_flat(fill_data[1]), .o_done(generic_done),
        .o_published_tile(), .o_published_bank(), .o_published_row(),
        .o_published_version(), .o_busy(generic_busy), .o_sticky_error(generic_error)
    );

    // Same bank/address recombination as the final stationary tile, no mock decoder.
    unified_dual_role_weight_store_p4all_candidate_v1 store_dut (
        .i_clk(clk), .i_rst_n(rst_n),
        .i_fill_valid(selected_valid[0] && allow_fill), .o_fill_ready(store_fill_ready),
        .i_fill_bank({selected_addr[10], selected_bank[0]}),
        .i_fill_addr(selected_addr[9:0]), .i_fill_data(selected_data[199:0]),
        .i_read_valid(store_read_valid), .o_read_ready(store_read_ready),
        .i_mode(store_mode), .i_scnn_active_pair(1'b0),
        .i_read_addr0(10'd0), .i_read_addr1(10'd0),
        .i_read_addr2(10'd0), .i_read_addr3(10'd0),
        .i_read_subword0(4'd0), .i_read_subword1(4'd0),
        .i_read_subword2(4'd0), .i_read_subword3(4'd0),
        .o_read_result_valid(store_result_valid), .i_read_result_ready(1'b1),
        .o_term0_flat(term0), .o_term1_flat(term1),
        .o_term2_flat(term2), .o_term3_flat(term3)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            response_valid <= 0;
            response_tag[0] <= 0;
            response_tag[1] <= 0;
            cycles <= 0;
            read_count <= 0;
            response_count <= 0;
            fill_count <= 0;
            stalled_q <= 0;
        end else begin
            cycles <= cycles + 1;
            if (cycles > 500) $fatal(1, "bounded watchdog");
            for (int port_id = 0; port_id < 2; port_id++) begin
                if (response_valid[port_id] && response_ready[port_id]) begin
                    response_valid[port_id] <= 0;
                    response_count <= response_count + 1;
                end
                if (read_valid[port_id]) begin
                    if (port_id != int'(use_generic) ||
                        read_addr[port_id] != 8'(read_count) ||
                        read_tag[port_id] != 16'(read_count))
                        $fatal(1, "read address/tag/owner reference mismatch");
                    response_valid[port_id] <= 1;
                    response_tag[port_id] <= read_tag[port_id];
                    read_count <= read_count + 1;
                end
            end
            if (stalled_q && (!selected_valid[0] ||
                selected_data[199:0] != stalled_data_q ||
                selected_addr[10:0] != stalled_addr_q))
                $fatal(1, "published row changed under backpressure");
            stalled_q <= selected_valid[0] && !selected_ready[0];
            stalled_data_q <= selected_data[199:0];
            stalled_addr_q <= selected_addr[10:0];
            if (|(selected_valid & selected_ready)) begin
                logic [199:0] expected_row;
                expected_row = 0;
                for (int lane = 0; lane < 10; lane++) begin
                    if (use_generic) expected_row[lane*8 +: 8] = test_code[7:0];
                    else expected_row[lane*5 +: 5] = test_code[4:0];
                end
                if (selected_valid != 4'b0001 || selected_addr != 0 ||
                    selected_bank != 0 || fill_tile[use_generic] != 0 ||
                    selected_data[199:0] != expected_row)
                    $fatal(1, "packed-row reference mismatch");
                fill_count <= fill_count + 1;
            end
        end
    end

    task automatic reset_case(input bit generic_route, input logic signed [15:0] code);
        @(negedge clk);
        rst_n = 0;
        launch_fc = 0;
        launch_generic = 0;
        store_read_valid = 0;
        use_generic = generic_route;
        test_code = code;
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
    endtask

    task automatic check_decode(input logic [1:0] mode, input bit packed_int8,
                                output int first_lane, output bit all_equal);
        logic [199:0] reference_row;
        int expected;
        reference_row = 0;
        for (int lane = 0; lane < 10; lane++) begin
            if (packed_int8) reference_row[lane*8 +: 8] = test_code[7:0];
            else reference_row[lane*5 +: 5] = test_code[4:0];
        end
        @(negedge clk);
        store_mode = mode;
        store_read_valid = 1;
        if (!store_read_ready) $fatal(1, "store unexpectedly not ready");
        @(negedge clk);
        store_read_valid = 0;
        if (!store_result_valid) $fatal(1, "missing read result");
        all_equal = 1;
        first_lane = int'($signed(term0[24:0]));
        for (int lane = 0; lane < 10; lane++) begin
            if (mode == 0) expected = int'($signed(reference_row[lane*5 +: 5]));
            else expected = int'($signed(reference_row[lane*8 +: 8]));
            if (int'($signed(term0[lane*25 +: 25])) != expected)
                $fatal(1, "decode reference mismatch mode=%0d lane=%0d", mode, lane);
            if (expected != int'(test_code)) all_equal = 0;
        end
        @(negedge clk);
    endtask

    initial begin
        bit in_int5, all_equal;
        int first_lane;
        if (!$value$plusargs("CASES=%s", cases_path) ||
            !$value$plusargs("CASE_COUNT=%d", case_count) ||
            case_count < 1 || case_count > 32)
            $fatal(1, "missing bounded fixture arguments");
        $readmemh(cases_path, cases, 0, case_count-1);
        for (int case_id = 0; case_id < case_count; case_id++) begin
            in_int5 = cases[case_id] >= -16 && cases[case_id] <= 15;
            reset_case(0, cases[case_id]);
            if (!fc_ready) $fatal(1, "scheduler not ready");
            launch_fc = 1;
            @(negedge clk);
            launch_fc = 0;
            while (!fc_done && !fc_error && cycles < 200) @(negedge clk);
            if (in_int5) begin
                if (!fc_done || fc_error || fc_rows != 1 || fc_version != 7 ||
                    read_count != 10 || response_count != 10 || fill_count != 1)
                    $fatal(1, "legal FC code failed code=%0d", test_code);
                check_decode(0, 0, first_lane, all_equal);
                if (!all_equal) $fatal(1, "FC legal numeric mismatch");
                accepted++;
                $display("FC_W2_PRECISION_CASE case_id=%0d route=fc_scheduler code=%0d result=ACCEPT reads=%0d responses=%0d fills=%0d rows=%0d sticky=%0b busy=%0b",
                    case_id, test_code, read_count, response_count, fill_count,
                    fc_rows, fc_error, fc_busy);
            end else begin
                repeat (8) @(negedge clk);
                if (!fc_error || fc_done || fc_rows != 0 || fill_count != 0 ||
                    read_count != 1 || response_count != 1)
                    $fatal(1, "out-of-range code not rejected atomically code=%0d", test_code);
                rejected++;
                $display("FC_W2_PRECISION_CASE case_id=%0d route=fc_scheduler code=%0d result=REJECT reads=%0d responses=%0d fills=%0d rows=%0d sticky=%0b busy=%0b",
                    case_id, test_code, read_count, response_count, fill_count,
                    fc_rows, fc_error, fc_busy);
            end

            reset_case(1, cases[case_id]);
            if (!generic_ready) $fatal(1, "INT8 publisher not ready");
            launch_generic = 1;
            @(negedge clk);
            launch_generic = 0;
            while (!generic_done && !generic_error && cycles < 200) @(negedge clk);
            if (!generic_done || generic_error || fill_count != 1 ||
                read_count != 10 || response_count != 10)
                $fatal(1, "INT8 control failed code=%0d", test_code);
            check_decode(1, 1, first_lane, all_equal);
            if (!all_equal) $fatal(1, "INT8 SRNN decode mismatch");
            check_decode(0, 1, first_lane, all_equal);
            controls++;
            $display("FC_W2_PRECISION_CASE case_id=%0d route=generic_int8 code=%0d result=INT8_PACK_SRNN_DECODE_PASS fills=%0d fc_lane0=%0d fc_all10_equal=%0b",
                case_id, test_code, fill_count, first_lane, all_equal);
        end
        if (rejected == 0 || accepted == 0) $fatal(1, "vacuous probe");
        $display("FC_W2_PRECISION_PROBE_PASS cases=%0d fc_accepted=%0d fc_rejected=%0d int8_controls=%0d problem_reproduced=1 final_top_simulated=0",
            case_count, accepted, rejected, controls);
        $finish;
    end
endmodule

`default_nettype wire
