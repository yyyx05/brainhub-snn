`timescale 1ns/1ps
`default_nettype none

module tb_fc_w2_int8_cache_repair;
    logic clk = 0;
    always #5 clk = !clk;
    logic rst_n = 0, abort_refill = 0, clear_error = 0;
    logic launch = 0, refill_int8 = 0, refill_all = 1;
    logic [11:0] features = 0, dirty_feature = 0;
    logic [9:0] destinations = 0, cache_base = 0;
    logic [4:0] tiles = 1;
    logic [3:0] dirty_tile = 0;
    logic [3:0] destination_tile_base = 0;
    logic [17:0] parameter_base = 0, parameter_stride = 0;
    wire ready, done, busy, error_sticky;
    wire [31:0] rows;
    wire [15:0] version;
    wire request_valid, response_ready;
    wire [17:0] request_addr;
    wire [15:0] request_tag;
    logic response_valid = 0;
    logic [15:0] response_tag = 0;
    logic signed [15:0] response_code = 0;
    logic signed [15:0] parameters [0:4095];
    string fixture_path;
    wire [3:0] fill_valid, fill_ready, fill_bank;
    wire [7:0] fill_tile;
    wire [43:0] fill_addr;
    wire [799:0] fill_data;
    wire store_fill_ready, store_read_ready, store_result_valid;
    logic store_read_valid = 0, store_result_ready = 1, store_int8 = 0;
    logic [9:0] store_row = 0;
    wire [999:0] terms [0:3];
    logic mac_valid = 0, mac_first = 0, mac_last = 0, mac_result_ready = 0;
    wire mac_ready, mac_result_valid;
    logic [39:0] mac_enable [0:3];
    logic [719:0] mac_factor = 0;
    wire [39:0] mac_result_mask;
    wire [1919:0] mac_result;
    int dot_checks = 0, mac_commands = 0;
    int cycles = 0, reads = 0, responses = 0, fills = 0, checks = 0;
    int response_delay = 0, refill_cases = 0, rejected_cases = 0;
    int ref_destinations, ref_feature, ref_cache_base;
    int ref_destination_base = 0, ref_tile = 0, ref_tiles = 1;
    int ref_parameter_base, ref_stride, ref_read_feature, ref_read_tile, ref_read_lane;
    int base4_fills = 0, endpoint_fills = 0, local_dirty_checks = 0;
    bit ref_int8;
    logic was_stalled = 0;
    logic [799:0] held_data;
    logic [43:0] held_addr;
    logic [7:0] held_tile;
    logic [3:0] held_valid, held_bank;
    wire allow_fill = cycles % 5 != 2;
    wire request_ready = response_delay == 0 && !response_valid && cycles % 7 != 3;
    wire [1:0] store_region = 2'((ref_destination_base + ref_tile) / 4);
    wire [10:0] store_fill_addr = fill_addr[store_region*11 +: 11];
    // One existing store models the selected physical destination; the monitor
    // independently verifies the complete region/tile routing on every row.
    assign fill_ready = {4{allow_fill && store_fill_ready}};

    unified_array16_shared_weight_cache_refill_plane_v1 #(
        .PARAM_DEPTH(262144), .PARAM_ADDR_WIDTH(18), .ARRAY_TILES(12)
    ) refill_plane (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(abort_refill), .i_clear_error(clear_error),
        .i_fc_refill_valid(launch), .o_fc_refill_ready(ready), .i_fc_refill_all(refill_all),
        .i_fc_weight_int8(refill_int8), .i_fc_dirty_feature(dirty_feature),
        .i_fc_dirty_tile(dirty_tile), .i_fc_parameter_base(parameter_base),
        .i_fc_hidden_stride(parameter_stride), .i_fc_cache_row_base(cache_base),
        .i_fc_feature_count(features), .i_fc_hidden_count(destinations), .i_fc_tile_count(tiles),
        .i_fc_destination_tile_base(destination_tile_base),
        .i_fc_version(16'd9), .i_array_idle(1'b1),
        .o_forward_read_valid(request_valid), .i_forward_read_ready(request_ready),
        .o_forward_read_addr(request_addr), .o_forward_read_tag(request_tag),
        .i_forward_response_valid(response_valid), .o_forward_response_ready(response_ready),
        .i_forward_response_tag(response_tag), .i_forward_response_code(response_code),
        .i_forward_response_error(1'b0), .o_weight_fill_valid(fill_valid),
        .i_weight_fill_ready(fill_ready), .o_weight_fill_tile_flat(fill_tile),
        .o_weight_fill_bank(fill_bank), .o_weight_fill_addr_flat(fill_addr),
        .o_weight_fill_data_flat(fill_data), .o_fc_refill_done(done), .o_fc_refill_busy(), .o_fc_rows_published(rows),
        .o_fc_published_version(version), .o_busy(busy), .o_sticky_error(error_sticky),
        .i_topology_fill_busy(1'b0), .i_topology_fill_valid(4'd0),
        .o_topology_fill_ready(), .i_topology_fill_tile_flat(8'd0),
        .i_topology_fill_bank(4'd0), .i_topology_fill_addr_flat(44'd0),
        .i_topology_fill_data_flat(800'd0), .o_fill_grant(), .o_quiescent()
    );

    unified_dual_role_h40_tile_p4all_cascade_accum_candidate_v1 accumulator (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(abort_refill),
        .i_valid(mac_valid), .o_ready(mac_ready), .i_first(mac_first), .i_last(mac_last),
        .i_lane_mask(40'h00000003ff), .i_enable0(mac_enable[0]), .i_enable1(mac_enable[1]),
        .i_enable2(mac_enable[2]), .i_enable3(mac_enable[3]),
        .i_term0_flat(terms[0]), .i_term1_flat(terms[1]),
        .i_term2_flat(terms[2]), .i_term3_flat(terms[3]), .i_factor_b_flat(mac_factor),
        .o_result_valid(mac_result_valid), .i_result_ready(mac_result_ready),
        .o_result_lane_mask(mac_result_mask), .o_result_flat(mac_result)
    );

    unified_dual_role_weight_store_p4all_candidate_v1 store (
        .i_clk(clk), .i_rst_n(rst_n), .i_fill_valid(fill_valid[store_region] && allow_fill),
        .o_fill_ready(store_fill_ready), .i_fill_bank({store_fill_addr[10], fill_bank[store_region]}),
        .i_fill_addr(store_fill_addr[9:0]), .i_fill_data(fill_data[store_region*200 +: 200]),
        .i_read_valid(store_read_valid), .o_read_ready(store_read_ready),
        .i_mode(2'd0), .i_fc_weight_int8(store_int8), .i_scnn_active_pair(1'b0),
        .i_read_addr0(store_row), .i_read_addr1(store_row),
        .i_read_addr2(store_row), .i_read_addr3(store_row),
        .i_read_subword0(4'd0), .i_read_subword1(4'd0),
        .i_read_subword2(4'd0), .i_read_subword3(4'd0),
        .o_read_result_valid(store_result_valid), .i_read_result_ready(store_result_ready),
        .o_term0_flat(terms[0]), .o_term1_flat(terms[1]),
        .o_term2_flat(terms[2]), .o_term3_flat(terms[3])
    );

    always @(posedge clk) begin
        if (!rst_n || abort_refill) begin
            response_valid <= 0;
            response_delay <= 0;
            was_stalled <= 0;
        end else begin
            cycles <= cycles + 1;
            if (cycles > 50000) $fatal(1, "watchdog");
            if (response_delay > 0) begin
                response_delay <= response_delay - 1;
                if (response_delay == 1) response_valid <= 1;
            end
            if (response_valid && response_ready) begin
                response_valid <= 0;
                responses <= responses + 1;
            end
            if (request_valid && request_ready) begin
                if (request_addr >= 4096) $fatal(1, "out of fixture range");
                if (int'(request_addr) != ref_parameter_base + ref_read_feature * ref_stride +
                    ref_read_tile * 40 + ref_read_lane)
                    $fatal(1, "physical base contaminated scalar address got=%0d", request_addr);
                if (ref_read_lane + 1 == ((ref_destinations - ref_read_tile * 40 < 40) ?
                    ref_destinations - ref_read_tile * 40 : 40)) begin
                    ref_read_lane <= 0;
                    if (ref_read_tile + 1 == ref_tiles) begin
                        ref_read_tile <= 0;
                        ref_read_feature <= ref_read_feature + 1;
                    end else ref_read_tile <= ref_read_tile + 1;
                end else ref_read_lane <= ref_read_lane + 1;
                response_tag <= request_tag;
                response_code <= parameters[request_addr[11:0]];
                response_delay <= 2;
                reads <= reads + 1;
            end
            if (was_stalled && (fill_valid != held_valid || fill_data != held_data ||
                fill_addr != held_addr || fill_tile != held_tile || fill_bank != held_bank))
                $fatal(1, "row unstable under backpressure");
            was_stalled <= (|fill_valid) && !(|(fill_valid & fill_ready));
            held_data <= fill_data;
            held_addr <= fill_addr;
            held_tile <= fill_tile;
            held_valid <= fill_valid;
            held_bank <= fill_bank;
            if (|(fill_valid & fill_ready)) begin
                logic [199:0] expected;
                int physical_tile, region, scalar_addr;
                expected = 0;
                physical_tile = ref_destination_base + ref_tile;
                region = physical_tile / 4;
                for (int lane = 0; lane < 40; lane++) begin
                    scalar_addr = ref_parameter_base + ref_feature * ref_stride + ref_tile * 40 + lane;
                    if (ref_tile * 40 + lane < ref_destinations) begin
                        if (ref_int8) expected[lane*8 +: 8] = 8'(parameters[scalar_addr]);
                        else expected[lane*5 +: 5] = 5'(parameters[scalar_addr]);
                    end
                end
                if (fill_valid != (4'b0001 << region) ||
                    fill_tile != (8'(physical_tile % 4) << (region * 2)) ||
                    fill_addr[region*11 +: 10] != 10'(ref_cache_base + (ref_feature >> 2)) ||
                    {fill_addr[region*11+10], fill_bank[region]} != 2'(ref_feature) ||
                    fill_data[region*200 +: 200] != expected)
                    $fatal(1, "row packing/address mismatch feature=%0d local_tile=%0d physical_tile=%0d",
                        ref_feature, ref_tile, physical_tile);
                fills <= fills + 1;
                if (ref_destination_base == 4) base4_fills <= base4_fills + 1;
                if (physical_tile == 11) endpoint_fills <= endpoint_fills + 1;
                if (!refill_all && ref_destination_base != 0 && ref_tile != 0)
                    local_dirty_checks <= local_dirty_checks + 1;
                if (!refill_all || ref_tile + 1 == ref_tiles) begin
                    ref_feature <= ref_feature + 1;
                    ref_tile <= 0;
                end else ref_tile <= ref_tile + 1;
            end
        end
    end

    task automatic start_refill(input bit int8_format, input int nfeatures,
                                input int ndestinations, input int row_base,
                                input int destination_base = 0, input int tile_count = 1,
                                input int scalar_base = 0, input int stride = -1);
        @(negedge clk);
        if (!ready || busy || error_sticky) $fatal(1, "refill not ready");
        refill_int8 = int8_format;
        features = 12'(nfeatures);
        destinations = 10'(ndestinations);
        cache_base = 10'(row_base);
        destination_tile_base = 4'(destination_base);
        tiles = 5'(tile_count);
        parameter_base = 18'(scalar_base);
        parameter_stride = 18'(stride < 0 ? ndestinations : stride);
        ref_int8 = int8_format;
        ref_destinations = ndestinations;
        ref_feature = refill_all ? 0 : int'(dirty_feature);
        ref_cache_base = row_base;
        ref_destination_base = destination_base;
        ref_tiles = tile_count;
        ref_tile = refill_all ? 0 : int'(dirty_tile);
        ref_parameter_base = scalar_base;
        ref_stride = int'(parameter_stride);
        ref_read_feature = ref_feature;
        ref_read_tile = ref_tile;
        ref_read_lane = 0;
        launch = 1;
        @(negedge clk);
        launch = 0;
        // Live configuration changing after acceptance must not change packing.
        refill_int8 = !int8_format;
        destination_tile_base = ~4'(destination_base);
    endtask

    function automatic int trace_fixture(input int source);
        // Directed UQ6.10 magnitudes, not an accuracy or real-sample trace fixture.
        case (source % 5)
            0: trace_fixture = 0;
            1: trace_fixture = 1;
            2: trace_fixture = 1024;
            3: trace_fixture = 65535;
            default: trace_fixture = (source * 997) % 65536;
        endcase
    endfunction

    task automatic check_real_weight_dot;
        longint signed expected;
        for (int source = 0; source < 160; source += 4) begin
            @(negedge clk);
            store_row = 10'(13 + (source >> 2));
            store_int8 = 1;
            store_read_valid = 1;
            store_result_ready = 0;
            if (!store_read_ready) $fatal(1, "dot-product cache not ready");
            @(negedge clk);
            store_read_valid = 0;
            store_int8 = 0;
            if (!store_result_valid) $fatal(1, "missing dot-product weights");
            for (int bit_index = 0; bit_index < 16; bit_index++) begin
                mac_valid = 1;
                mac_first = source == 0 && bit_index == 0;
                mac_last = source == 156 && bit_index == 15;
                mac_factor = {40{18'(1 << bit_index)}};
                for (int term = 0; term < 4; term++)
                    mac_enable[term] = (trace_fixture(source + term) & (1 << bit_index)) != 0 ?
                        40'h00000003ff : 40'd0;
                if (!mac_ready) $fatal(1, "dot-product accumulator not ready");
                @(negedge clk);
                mac_commands++;
            end
            mac_valid = 0;
            store_result_ready = 1;
            @(negedge clk);
        end
        repeat (3) begin
            if (!mac_result_valid || mac_result_mask != 40'h00000003ff)
                $fatal(1, "missing raw dot result");
            for (int cls = 0; cls < 10; cls++) begin
                expected = 0;
                for (int source = 0; source < 160; source++)
                    expected += longint'(parameters[source*10+cls]) * longint'(trace_fixture(source));
                if ($signed(mac_result[cls*48 +: 48]) != expected)
                    $fatal(1, "raw dot mismatch class=%0d got=%0d expected=%0d",
                        cls, $signed(mac_result[cls*48 +: 48]), expected);
                dot_checks++;
            end
            @(negedge clk);
        end
        mac_result_ready = 1;
        @(negedge clk);
        mac_result_ready = 0;
        $display("FC_W2_RAW_DOT_PASS classes=10 mac_commands=%0d trace=directed_UQ6_10 output_scaling_checked=0", mac_commands);
    endtask

    task automatic expect_success(input int expected_rows);
        while (!done && !error_sticky) @(negedge clk);
        if (error_sticky || rows != 32'(expected_rows) || version != 9)
            $fatal(1, "refill failed rows=%0d expected=%0d", rows, expected_rows);
        refill_cases++;
        // The shared plane releases its retained fill grant one cycle after done.
        @(negedge clk);
        if (busy || !ready) $fatal(1, "shared refill plane did not drain");
    endtask

    task automatic expect_failure(input int old_fills, input int old_reads,
                                  input bit geometry_error);
        while (!error_sticky) @(negedge clk);
        repeat (4) @(negedge clk);
        if (busy || done || fills != old_fills || (geometry_error && reads != old_reads))
            $fatal(1, "failed refill published data, stayed busy, or read illegal geometry");
        clear_error = 1;
        @(negedge clk);
        clear_error = 0;
        @(negedge clk);
        if (!ready || error_sticky) $fatal(1, "clear cannot recover without reset");
        rejected_cases++;
    endtask

    task automatic check_group(input int source_base, input int ndestinations,
                               input int row_base, input bit int8_format);
        int expected;
        @(negedge clk);
        store_row = 10'(row_base + (source_base >> 2));
        store_int8 = int8_format;
        store_read_valid = 1;
        store_result_ready = 0;
        if (!store_read_ready) $fatal(1, "store not ready");
        @(negedge clk);
        store_read_valid = 0;
        store_int8 = !int8_format;
        repeat (3) begin
            if (!store_result_valid) $fatal(1, "lost result under stall");
            for (int term = 0; term < 4; term++) begin
                for (int lane = 0; lane < 40; lane++) begin
                    expected = lane < ndestinations ?
                        int'(parameters[(source_base+term)*ndestinations+lane]) : 0;
                    if (int'($signed(terms[term][lane*25 +: 25])) != expected)
                        $fatal(1, "decode mismatch source=%0d lane=%0d got=%0d expected=%0d",
                            source_base+term, lane, $signed(terms[term][lane*25 +: 25]), expected);
                    checks++;
                end
            end
            @(negedge clk);
        end
        store_result_ready = 1;
        @(negedge clk);
    endtask

    initial begin
        int before_fills, before_reads;
        if (!$value$plusargs("WEIGHTS=%s", fixture_path)) $fatal(1, "missing real W2 fixture");
        $readmemh(fixture_path, parameters, 0, 1599);
        repeat (3) @(negedge clk);
        rst_n = 1;
        start_refill(1, 160, 10, 13);
        expect_success(160);
        for (int source = 0; source < 160; source += 4) check_group(source, 10, 13, 1);
        $display("FC_W2_REPAIR real_w2_codes=1600 cache_rows=160 p4_groups=40 all_four_terms=PASS");
        check_real_weight_dot();

        // Re-publish an updated class coefficient into exactly one dirty row.
        parameters[17*10+4] = -128;
        dirty_feature = 17;
        refill_all = 0;
        start_refill(1, 160, 10, 13);
        expect_success(1);
        check_group(16, 10, 13, 1);
        refill_all = 1;

        // Exercise all 25 INT8 slots, including signed extremes, then INT5 W0.
        for (int i = 0; i < 100; i++) parameters[i] = 16'((i*71)%256 - 128);
        start_refill(1, 4, 25, 13);
        expect_success(4);
        check_group(0, 25, 13, 1);
        for (int i = 0; i < 160; i++) parameters[i] = 16'((i*7)%32 - 16);
        start_refill(0, 4, 40, 13);
        expect_success(4);
        check_group(0, 40, 13, 0);

        before_fills = fills;
        before_reads = reads;
        start_refill(1, 4, 26, 13);
        expect_failure(before_fills, before_reads, 1);
        start_refill(0, 2312, 40, 1);
        expect_failure(before_fills, before_reads, 1);

        // Both formats must reject out-of-range data, not silently truncate it.
        parameters[8] = 128;
        start_refill(1, 4, 10, 13);
        expect_failure(before_fills, before_reads, 0);
        parameters[8] = 16;
        start_refill(0, 4, 40, 13);
        expect_failure(before_fills, before_reads, 0);
        parameters[8] = -129;
        start_refill(1, 4, 10, 13);
        expect_failure(before_fills, before_reads, 0);

        // Abort while responses are in flight, then restart without a reset.
        for (int i = 0; i < 160; i++) parameters[i] = 16'((i*7)%32 - 16);
        before_reads = reads;
        start_refill(0, 4, 40, 13);
        while (reads < before_reads + 3) @(negedge clk);
        abort_refill = 1;
        @(negedge clk);
        abort_refill = 0;
        @(negedge clk);
        if (busy || error_sticky || fills != before_fills) $fatal(1, "abort leaked a row");
        start_refill(0, 4, 40, 13);
        expect_success(4);
        check_group(0, 40, 13, 0);
        if (refill_cases != 5 || rejected_cases != 5 || fills != 173 || checks != 21120)
            $fatal(1, "original repair coverage changed");

        // Route actual W2 rows through the shared plane to tile 4, then back
        // to tile 0. The destination base is deliberately changed after accept.
        $readmemh(fixture_path, parameters, 0, 1599);
        start_refill(1, 4, 10, 13, 4);
        expect_success(4);
        check_group(0, 10, 13, 1);
        parameters[2*10+7] = -128;
        dirty_feature = 2;
        dirty_tile = 0;
        refill_all = 0;
        start_refill(1, 4, 10, 13, 4);
        expect_success(1);
        check_group(0, 10, 13, 1);
        refill_all = 1;
        start_refill(1, 4, 10, 13, 0);
        expect_success(4);
        check_group(0, 10, 13, 1);

        // Cross a region boundary and retain local tile*40 scalar addressing,
        // including a partial final tile and a nonzero parameter base/stride.
        for (int i = 0; i < 4096; i++) parameters[i] = 16'((i*7)%32 - 16);
        start_refill(0, 4, 45, 577, 3, 2, 37, 64);
        expect_success(8);
        parameters[37+2*64+40+2] = -16;
        dirty_feature = 2;
        dirty_tile = 1;
        refill_all = 0;
        start_refill(0, 4, 45, 577, 3, 2, 37, 64);
        expect_success(1);
        refill_all = 1;
        dirty_tile = 0;

        // The final legal tile is 11 for this twelve-tile array.
        start_refill(0, 1, 45, 577, 10, 2, 37, 64);
        expect_success(2);
        $readmemh(fixture_path, parameters, 0, 1599);
        start_refill(1, 4, 10, 13, 11);
        expect_success(4);
        check_group(0, 10, 13, 1);
        before_fills = fills;
        before_reads = reads;
        start_refill(1, 4, 10, 13, 12);
        expect_failure(before_fills, before_reads, 1);
        start_refill(0, 1, 45, 13, 11, 2);
        expect_failure(before_fills, before_reads, 1);
        start_refill(0, 1, 80, 13, 15, 2);
        expect_failure(before_fills, before_reads, 1);
        if (refill_cases != 12 || rejected_cases != 8 || fills != 197 ||
            checks != 23040 || base4_fills != 5 || endpoint_fills != 5 ||
            local_dirty_checks != 1)
            $fatal(1, "destination-base coverage mismatch");
        $display("FC_W2_INT8_CACHE_REPAIR_PASS successes=%0d rejected=%0d checks=%0d dot_checks=%0d mac_commands=%0d reads=%0d responses=%0d fills=%0d base4_fills=%0d endpoint_fills=%0d local_dirty_checks=%0d cycles=%0d final_top_simulated=0",
            refill_cases, rejected_cases, checks, dot_checks, mac_commands, reads, responses, fills,
            base4_fills, endpoint_fills, local_dirty_checks, cycles);
        $finish;
    end
endmodule
`default_nettype wire
