`timescale 1ns/1ps
`default_nettype none

module tb_unified_scnn_selected_layers_parameter_backend_v1;
    localparam int PARAM_DEPTH = 64;
    localparam int PARAM_ADDR_WIDTH = 6;
    localparam int TAG_WIDTH = 4;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic abort = 1'b0;
    logic clear_error = 1'b0;
    logic cache_invalidate = 1'b0;
    logic array_idle = 1'b1;
    always #5 clk = ~clk;

    logic parameter_load_valid;
    logic parameter_load_ready;
    logic [PARAM_ADDR_WIDTH-1:0] parameter_load_addr;
    logic signed [15:0] parameter_load_code;
    logic signed [15:0] parameter_load_residual;
    logic classifier_prime_valid;
    logic classifier_prime_ready;
    logic classifier_prime_done;
    logic classifier_pair = 1'b1;
    logic [PARAM_ADDR_WIDTH-1:0] classifier_parameter_base = 6'd16;
    logic [2:0] classifier_feature_count = 3'd4;
    logic [1:0] classifier_class_count = 2'd2;
    logic commit_publish_valid;
    logic commit_publish_ready;
    logic commit_publish_done;
    logic conv_cache_invalidate;

    logic learning_read_valid;
    logic learning_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] learning_read_addr;
    logic [TAG_WIDTH-1:0] learning_read_tag;
    logic learning_response_valid;
    logic learning_response_ready = 1'b1;
    logic [TAG_WIDTH-1:0] learning_response_tag;
    logic signed [15:0] learning_response_code;
    logic signed [15:0] learning_response_residual;
    logic learning_response_error;
    logic learning_write_valid;
    logic learning_write_ready;
    logic [PARAM_ADDR_WIDTH-1:0] learning_write_addr;
    logic signed [15:0] learning_write_code;
    logic signed [15:0] learning_write_residual;
    logic learning_write_code_enable;
    logic learning_write_residual_enable;

    logic conv_forward_read_valid;
    logic conv_forward_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] conv_forward_read_addr;
    logic [TAG_WIDTH-1:0] conv_forward_read_tag;
    logic conv_forward_response_valid;
    logic conv_forward_response_ready = 1'b1;
    logic [TAG_WIDTH-1:0] conv_forward_response_tag;
    logic signed [15:0] conv_forward_response_code;
    logic conv_forward_response_error;

    logic store_load_valid;
    logic store_load_ready;
    logic [PARAM_ADDR_WIDTH-1:0] store_load_addr;
    logic signed [15:0] store_load_code;
    logic signed [15:0] store_load_residual;
    logic store_learning_read_valid;
    logic store_learning_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] store_learning_read_addr;
    logic [TAG_WIDTH-1:0] store_learning_read_tag;
    logic store_learning_response_valid;
    logic store_learning_response_ready;
    logic [TAG_WIDTH-1:0] store_learning_response_tag;
    logic signed [15:0] store_learning_response_code;
    logic signed [15:0] store_learning_response_residual;
    logic store_learning_response_error;
    logic store_learning_write_valid;
    logic store_learning_write_ready;
    logic [PARAM_ADDR_WIDTH-1:0] store_learning_write_addr;
    logic signed [15:0] store_learning_write_code;
    logic signed [15:0] store_learning_write_residual;
    logic store_learning_write_code_enable;
    logic store_learning_write_residual_enable;
    logic store_forward_read_valid;
    logic store_forward_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] store_forward_read_addr;
    logic [TAG_WIDTH-1:0] store_forward_read_tag;
    logic store_forward_response_valid;
    logic store_forward_response_ready;
    logic [TAG_WIDTH-1:0] store_forward_response_tag;
    logic signed [15:0] store_forward_response_code;
    logic store_forward_response_error;
    logic store_protocol_error;

    logic [3:0] weight_fill_valid;
    logic [3:0] weight_fill_ready = 4'hf;
    logic [7:0] weight_fill_tile_flat;
    logic [3:0] weight_fill_bank;
    logic [43:0] weight_fill_addr_flat;
    logic [799:0] weight_fill_data_flat;
    logic cache_valid;
    logic active_classifier_pair;
    logic [15:0] active_version;
    logic busy;
    logic sticky_error;

    int fill_count;
    bit saw_prime_c0_even;
    bit saw_prime_c0_odd;
    bit saw_prime_c1_even;
    bit saw_prime_c1_odd;
    bit saw_commit_updated;
    bit saw_invalidate;
    bit check_warm_refresh = 1'b0;
    int prime_done_count = 0;
    int commit_done_count = 0;
    logic signed [15:0] saved_code [0:PARAM_DEPTH-1];
    logic signed [15:0] saved_residual [0:PARAM_DEPTH-1];

    unified_scnn_selected_layers_parameter_backend_v1 #(
        .USE_INTERNAL_STORE(1'b0),
        .PARAM_DEPTH(PARAM_DEPTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .MAX_CLASSIFIER_FEATURES(4),
        .MAX_CLASSES(2),
        .ARRAY_WEIGHT_ADDR_WIDTH(11),
        .FEATURE_COUNT_WIDTH(3),
        .CLASS_COUNT_WIDTH(2)
    ) dut (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(abort),
        .i_clear_error(clear_error),
        .i_cache_invalidate(cache_invalidate),
        .i_parameter_load_valid(parameter_load_valid),
        .o_parameter_load_ready(parameter_load_ready),
        .i_parameter_load_addr(parameter_load_addr),
        .i_parameter_load_code(parameter_load_code),
        .i_parameter_load_residual(parameter_load_residual),
        .i_classifier_prime_valid(classifier_prime_valid),
        .o_classifier_prime_ready(classifier_prime_ready),
        .i_classifier_pair(classifier_pair), .i_initial_version(16'd5),
        .i_classifier_parameter_base(classifier_parameter_base),
        .i_classifier_feature_count(classifier_feature_count),
        .i_classifier_class_count(classifier_class_count),
        .o_classifier_prime_done(classifier_prime_done),
        .i_commit_publish_valid(commit_publish_valid),
        .o_commit_publish_ready(commit_publish_ready),
        .o_commit_publish_done(commit_publish_done),
        .o_conv_cache_invalidate(conv_cache_invalidate),
        .i_learning_read_valid(learning_read_valid),
        .o_learning_read_ready(learning_read_ready),
        .i_learning_read_addr(learning_read_addr),
        .i_learning_read_tag(learning_read_tag),
        .o_learning_response_valid(learning_response_valid),
        .i_learning_response_ready(learning_response_ready),
        .o_learning_response_tag(learning_response_tag),
        .o_learning_response_code(learning_response_code),
        .o_learning_response_residual(learning_response_residual),
        .o_learning_response_error(learning_response_error),
        .i_learning_write_valid(learning_write_valid),
        .o_learning_write_ready(learning_write_ready),
        .i_learning_write_addr(learning_write_addr),
        .i_learning_write_code(learning_write_code),
        .i_learning_write_residual(learning_write_residual),
        .i_learning_write_code_enable(learning_write_code_enable),
        .i_learning_write_residual_enable(
            learning_write_residual_enable),
        .i_conv_forward_read_valid(conv_forward_read_valid),
        .o_conv_forward_read_ready(conv_forward_read_ready),
        .i_conv_forward_read_addr(conv_forward_read_addr),
        .i_conv_forward_read_tag(conv_forward_read_tag),
        .o_conv_forward_response_valid(conv_forward_response_valid),
        .i_conv_forward_response_ready(conv_forward_response_ready),
        .o_conv_forward_response_tag(conv_forward_response_tag),
        .o_conv_forward_response_code(conv_forward_response_code),
        .o_conv_forward_response_error(conv_forward_response_error),
        .o_store_load_valid(store_load_valid),
        .i_store_load_ready(store_load_ready),
        .o_store_load_addr(store_load_addr),
        .o_store_load_code(store_load_code),
        .o_store_load_residual(store_load_residual),
        .o_store_learning_read_valid(store_learning_read_valid),
        .i_store_learning_read_ready(store_learning_read_ready),
        .o_store_learning_read_addr(store_learning_read_addr),
        .o_store_learning_read_tag(store_learning_read_tag),
        .i_store_learning_response_valid(store_learning_response_valid),
        .o_store_learning_response_ready(store_learning_response_ready),
        .i_store_learning_response_tag(store_learning_response_tag),
        .i_store_learning_response_code(store_learning_response_code),
        .i_store_learning_response_residual(
            store_learning_response_residual),
        .i_store_learning_response_error(store_learning_response_error),
        .o_store_learning_write_valid(store_learning_write_valid),
        .i_store_learning_write_ready(store_learning_write_ready),
        .o_store_learning_write_addr(store_learning_write_addr),
        .o_store_learning_write_code(store_learning_write_code),
        .o_store_learning_write_residual(store_learning_write_residual),
        .o_store_learning_write_code_enable(
            store_learning_write_code_enable),
        .o_store_learning_write_residual_enable(
            store_learning_write_residual_enable),
        .o_store_forward_read_valid(store_forward_read_valid),
        .i_store_forward_read_ready(store_forward_read_ready),
        .o_store_forward_read_addr(store_forward_read_addr),
        .o_store_forward_read_tag(store_forward_read_tag),
        .i_store_forward_response_valid(store_forward_response_valid),
        .o_store_forward_response_ready(store_forward_response_ready),
        .i_store_forward_response_tag(store_forward_response_tag),
        .i_store_forward_response_code(store_forward_response_code),
        .i_store_forward_response_error(store_forward_response_error),
        .i_store_protocol_error(store_protocol_error),
        .i_array_idle(array_idle),
        .o_weight_fill_valid(weight_fill_valid),
        .i_weight_fill_ready(weight_fill_ready),
        .o_weight_fill_tile_flat(weight_fill_tile_flat),
        .o_weight_fill_bank(weight_fill_bank),
        .o_weight_fill_addr_flat(weight_fill_addr_flat),
        .o_weight_fill_data_flat(weight_fill_data_flat),
        .o_cache_valid(cache_valid),
        .o_active_classifier_pair(active_classifier_pair),
        .o_active_version(active_version), .o_busy(busy),
        .o_sticky_error(sticky_error)
    );

    unified_tp_shared_forward_rmw_store #(
        .DEPTH(PARAM_DEPTH), .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH)
    ) shared_parameter_store (
        .i_clk(clk), .i_rst_n(rst_n), .i_clear_error(clear_error),
        .i_load_valid(store_load_valid), .o_load_ready(store_load_ready),
        .i_load_addr(store_load_addr), .i_load_code(store_load_code),
        .i_load_residual(store_load_residual),
        .i_learning_read_valid(store_learning_read_valid),
        .o_learning_read_ready(store_learning_read_ready),
        .i_learning_read_addr(store_learning_read_addr),
        .i_learning_read_tag(store_learning_read_tag),
        .o_learning_response_valid(store_learning_response_valid),
        .i_learning_response_ready(store_learning_response_ready),
        .o_learning_response_tag(store_learning_response_tag),
        .o_learning_response_code(store_learning_response_code),
        .o_learning_response_residual(store_learning_response_residual),
        .o_learning_response_error(store_learning_response_error),
        .i_learning_write_valid(store_learning_write_valid),
        .o_learning_write_ready(store_learning_write_ready),
        .i_learning_write_addr(store_learning_write_addr),
        .i_learning_write_code(store_learning_write_code),
        .i_learning_write_residual(store_learning_write_residual),
        .i_learning_write_code_enable(store_learning_write_code_enable),
        .i_learning_write_residual_enable(
            store_learning_write_residual_enable),
        .i_forward_read_valid(store_forward_read_valid),
        .o_forward_read_ready(store_forward_read_ready),
        .i_forward_read_addr(store_forward_read_addr),
        .i_forward_read_tag(store_forward_read_tag),
        .o_forward_response_valid(store_forward_response_valid),
        .i_forward_response_ready(store_forward_response_ready),
        .o_forward_response_tag(store_forward_response_tag),
        .o_forward_response_code(store_forward_response_code),
        .o_forward_response_error(store_forward_response_error),
        .o_protocol_error(store_protocol_error)
    );

    task automatic load_parameter(
        input int address,
        input int code,
        input int residual
    );
        begin
            @(negedge clk);
            parameter_load_valid = 1'b1;
            parameter_load_addr = PARAM_ADDR_WIDTH'(address);
            parameter_load_code = 16'(code);
            parameter_load_residual = 16'(residual);
            do @(posedge clk); while (!parameter_load_ready);
            @(negedge clk);
            parameter_load_valid = 1'b0;
        end
    endtask

    task automatic learning_update(
        input int address,
        input int old_code,
        input int old_residual,
        input int new_code,
        input int new_residual
    );
        begin
            @(negedge clk);
            learning_read_valid = 1'b1;
            learning_read_addr = PARAM_ADDR_WIDTH'(address);
            learning_read_tag = TAG_WIDTH'(address);
            do @(posedge clk); while (!learning_read_ready);
            @(negedge clk);
            learning_read_valid = 1'b0;
            do @(posedge clk); while (!learning_response_valid);
            if (learning_response_error ||
                    learning_response_tag != TAG_WIDTH'(address) ||
                    $signed(learning_response_code) != old_code ||
                    $signed(learning_response_residual) != old_residual)
                $fatal(1, "learning read mismatch addr=%0d", address);
            @(negedge clk);
            learning_write_valid = 1'b1;
            learning_write_addr = PARAM_ADDR_WIDTH'(address);
            learning_write_code = 16'(new_code);
            learning_write_residual = 16'(new_residual);
            learning_write_code_enable = 1'b1;
            learning_write_residual_enable = 1'b1;
            do @(posedge clk); while (!learning_write_ready);
            @(negedge clk);
            learning_write_valid = 1'b0;
        end
    endtask

    task automatic check_conv_forward(
        input int address,
        input int expected_code
    );
        begin
            @(negedge clk);
            conv_forward_read_valid = 1'b1;
            conv_forward_read_addr = PARAM_ADDR_WIDTH'(address);
            conv_forward_read_tag = TAG_WIDTH'(address + 1);
            do @(posedge clk); while (!conv_forward_read_ready);
            @(negedge clk);
            conv_forward_read_valid = 1'b0;
            do @(posedge clk); while (!conv_forward_response_valid);
            if (conv_forward_response_error ||
                    conv_forward_response_tag != TAG_WIDTH'(address + 1) ||
                    $signed(conv_forward_response_code) != expected_code)
                $fatal(1, "Conv forward mismatch addr=%0d got=%0d",
                    address, $signed(conv_forward_response_code));
        end
    endtask

    always @(posedge clk) begin
        if (rst_n && conv_cache_invalidate)
            saw_invalidate <= 1'b1;
        if (rst_n && classifier_prime_done)
            prime_done_count <= prime_done_count + 1;
        if (rst_n && commit_publish_done)
            commit_done_count <= commit_done_count + 1;
        if (check_warm_refresh &&
                (active_version != 16'd6 || commit_publish_done ||
                 (store_load_valid && store_load_ready) ||
                 (store_learning_write_valid && store_learning_write_ready)))
            $fatal(1, "warm refresh changed version, commit or parameter store");

        for (int region = 0; region < 4; region++) begin
            if (weight_fill_valid[region] && weight_fill_ready[region]) begin
                int tile;
                int bank;
                logic [199:0] row_data;
                tile = region * 4 +
                    int'(weight_fill_tile_flat[region*2 +: 2]);
                bank = int'({weight_fill_addr_flat[
                    region*11 + 10], weight_fill_bank[region]});
                row_data = weight_fill_data_flat[region*200 +: 200];
                fill_count <= fill_count + 1;
                if (check_warm_refresh) begin
                    logic [199:0] expected_row;
                    expected_row = '0;
                    if (tile > 1 || bank < 2 ||
                            weight_fill_addr_flat[region*11 +: 10] != 0)
                        $fatal(1, "warm refresh used live rather than saved descriptor");
                    for (int word_index = 0; word_index < 2; word_index++)
                        expected_row[word_index*16 +: 16] = saved_code[
                            16 + tile*4 + (bank - 2) + word_index*2];
                    if (row_data !== expected_row)
                        $fatal(1, "warm classifier row mismatch tile=%0d bank=%0d", tile, bank);
                end
                if (tile == 0 && bank == 2 &&
                        $signed(row_data[15:0]) == 10 &&
                        $signed(row_data[31:16]) == 30)
                    saw_prime_c0_even <= 1'b1;
                if (tile == 0 && bank == 3 &&
                        $signed(row_data[15:0]) == 20 &&
                        $signed(row_data[31:16]) == 40)
                    saw_prime_c0_odd <= 1'b1;
                if (tile == 1 && bank == 2 &&
                        $signed(row_data[15:0]) == 50 &&
                        $signed(row_data[31:16]) == 70)
                    saw_prime_c1_even <= 1'b1;
                if (tile == 1 && bank == 3 &&
                        $signed(row_data[15:0]) == 60 &&
                        $signed(row_data[31:16]) == 80)
                    saw_prime_c1_odd <= 1'b1;
                if (tile == 0 && bank == 2 &&
                        $signed(row_data[15:0]) == 15 &&
                        active_version == 16'd5)
                    saw_commit_updated <= 1'b1;
            end
        end
    end

    initial begin
        parameter_load_valid = 1'b0;
        parameter_load_addr = '0;
        parameter_load_code = '0;
        parameter_load_residual = '0;
        classifier_prime_valid = 1'b0;
        commit_publish_valid = 1'b0;
        learning_read_valid = 1'b0;
        learning_read_addr = '0;
        learning_read_tag = '0;
        learning_write_valid = 1'b0;
        learning_write_addr = '0;
        learning_write_code = '0;
        learning_write_residual = '0;
        learning_write_code_enable = 1'b0;
        learning_write_residual_enable = 1'b0;
        conv_forward_read_valid = 1'b0;
        conv_forward_read_addr = '0;
        conv_forward_read_tag = '0;
        fill_count = 0;
        saw_prime_c0_even = 1'b0;
        saw_prime_c0_odd = 1'b0;
        saw_prime_c1_even = 1'b0;
        saw_prime_c1_odd = 1'b0;
        saw_commit_updated = 1'b0;
        saw_invalidate = 1'b0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        load_parameter(0, 111, 1);
        load_parameter(1, 222, 2);
        load_parameter(16, 10, 0);
        load_parameter(17, 20, 0);
        load_parameter(18, 30, 0);
        load_parameter(19, 40, 0);
        load_parameter(20, 50, 0);
        load_parameter(21, 60, 0);
        load_parameter(22, 70, 0);
        load_parameter(23, 80, 0);

        @(negedge clk);
        classifier_prime_valid = 1'b1;
        do @(posedge clk); while (!classifier_prime_ready);
        @(negedge clk);
        classifier_prime_valid = 1'b0;
        do @(posedge clk); while (!classifier_prime_done);
        if (!cache_valid || !active_classifier_pair ||
                active_version != 16'd5)
            $fatal(1, "classifier prime metadata mismatch");
        if (!saw_prime_c0_even || !saw_prime_c0_odd ||
                !saw_prime_c1_even || !saw_prime_c1_odd)
            $fatal(1, "classifier prime packing mismatch");

        check_conv_forward(0, 111);
        learning_update(0, 111, 1, 333, 3);
        learning_update(16, 10, 0, 15, 5);

        @(negedge clk);
        commit_publish_valid = 1'b1;
        do @(posedge clk); while (!commit_publish_ready);
        @(negedge clk);
        commit_publish_valid = 1'b0;
        do @(posedge clk); while (!commit_publish_done);
        @(negedge clk);
        if (active_version != 16'd6 || !active_classifier_pair ||
                !cache_valid || !saw_invalidate || !saw_commit_updated)
            $fatal(1, "atomic publication mismatch version=%0d pair=%0b cache=%0b invalidate=%0b updated=%0b",
                active_version, active_classifier_pair, cache_valid,
                saw_invalidate, saw_commit_updated);

        check_conv_forward(0, 333);

        begin
            int fills_before;
            logic [799:0] stalled_data;
            logic [43:0] stalled_addr;
            logic [7:0] stalled_tile;
            logic [3:0] stalled_bank;
            logic [3:0] stalled_valid;
            @(negedge clk);
            fills_before = fill_count;
            for (int address = 0; address < PARAM_DEPTH; address++) begin
                saved_code[address] = shared_parameter_store.code_mem[address];
                saved_residual[address] = shared_parameter_store.residual_mem[address];
            end
            if (saved_code[0] != 333 || saved_residual[0] != 3 ||
                    saved_code[16] != 15 || saved_residual[16] != 5)
                $fatal(1, "post-commit store fixture mismatch");
            check_warm_refresh = 1'b1;
            classifier_pair = 1'b0;
            classifier_parameter_base = '0;
            classifier_feature_count = 3'd1;
            classifier_class_count = 2'd1;
            cache_invalidate = 1'b1;
            classifier_prime_valid = 1'b1;
            repeat (2) begin
                @(posedge clk);
                #1;
                if (classifier_prime_ready || classifier_prime_done ||
                        cache_valid || busy)
                    $fatal(1, "invalidate admitted a classifier prime");
            end
            @(negedge clk);
            cache_invalidate = 1'b0;
            classifier_prime_valid = 1'b0;
            repeat (3) begin
                @(posedge clk);
                #1;
                if (busy || cache_valid || (|weight_fill_valid) ||
                        parameter_load_ready || fill_count != fills_before)
                    $fatal(1, "eviction refreshed without demand or reopened loading");
            end
            @(negedge clk);
            array_idle = 1'b0;
            classifier_prime_valid = 1'b1;
            repeat (3) begin
                @(posedge clk);
                #1;
                if (classifier_prime_ready || busy || (|weight_fill_valid))
                    $fatal(1, "warm refresh admitted while a foreign client was busy");
            end
            @(negedge clk);
            array_idle = 1'b1;
            weight_fill_ready = '0;
            do @(posedge clk); while (!classifier_prime_ready);
            @(negedge clk);
            classifier_prime_valid = 1'b0;
            while (!(|weight_fill_valid)) @(negedge clk);
            stalled_data = weight_fill_data_flat;
            stalled_addr = weight_fill_addr_flat;
            stalled_tile = weight_fill_tile_flat;
            stalled_bank = weight_fill_bank;
            stalled_valid = weight_fill_valid;
            repeat (3) begin
                @(posedge clk);
                #1;
                if (cache_valid || classifier_prime_done ||
                        weight_fill_valid !== stalled_valid ||
                        weight_fill_data_flat !== stalled_data ||
                        weight_fill_addr_flat !== stalled_addr ||
                        weight_fill_tile_flat !== stalled_tile ||
                        weight_fill_bank !== stalled_bank)
                    $fatal(1, "warm refresh did not hold a stalled fill");
            end
            @(negedge clk);
            weight_fill_ready = '1;
            do @(posedge clk); while (!classifier_prime_done);
            @(negedge clk);
            if (!cache_valid || !active_classifier_pair || active_version != 6 ||
                    fill_count != fills_before + 4 || prime_done_count != 2 ||
                    commit_done_count != 1 || parameter_load_ready || busy)
                $fatal(1, "warm refresh metadata/count mismatch");
            for (int address = 0; address < PARAM_DEPTH; address++) begin
                if (shared_parameter_store.code_mem[address] !== saved_code[address] ||
                        shared_parameter_store.residual_mem[address] !== saved_residual[address])
                    $fatal(1, "warm refresh modified store addr=%0d", address);
            end
            check_warm_refresh = 1'b0;
        end
        if (sticky_error)
            $fatal(1, "unexpected backend sticky error");
        $display("PASS: external shared parameter store backs Conv8/classifier RMW, post-commit demand refresh, collision guard and unchanged code/residual/version");
        $finish;
    end

    initial begin
        repeat (2000) @(posedge clk);
        $fatal(1, "timeout");
    end
endmodule

`default_nettype wire
