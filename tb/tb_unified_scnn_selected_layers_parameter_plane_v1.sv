`timescale 1ns/1ps
`default_nettype none

module tb_unified_scnn_selected_layers_parameter_plane_v1;
    localparam int PARAM_DEPTH = 128;
    localparam int PARAM_ADDR_WIDTH = 7;
    localparam int TAG_WIDTH = 5;
    localparam int CLASSIFIER_BASE = 64;

    logic clk = 1'b0;
    logic rst_n;
    logic abort;
    logic clear_error;
    logic cache_invalidate;
    always #5 clk = ~clk;

    logic parameter_load_valid;
    logic parameter_load_ready;
    logic [PARAM_ADDR_WIDTH-1:0] parameter_load_addr;
    logic signed [15:0] parameter_load_code;
    logic signed [15:0] parameter_load_residual;
    logic classifier_prime_valid;
    logic classifier_prime_ready;
    logic classifier_prime_done;
    logic commit_pending;
    logic commit_publish_ready;
    logic commit_pulse;

    logic learning_read_valid;
    logic learning_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] learning_read_addr;
    logic [TAG_WIDTH-1:0] learning_read_tag;
    logic learning_response_valid;
    logic learning_response_ready;
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

    logic conv_context_valid;
    logic conv_context_ready;
    logic conv_context_load_weights;
    logic [9:0] conv_output_channel_base;
    logic [4:0] conv_output_channel_count;
    logic [1:0] conv_input_channel_count;
    logic conv_context_pair;
    logic [PARAM_ADDR_WIDTH-1:0] conv_parameter_base;
    logic [15:0] conv_context_version;
    logic array_idle;

    logic [3:0] weight_fill_valid;
    logic [3:0] weight_fill_ready;
    logic [7:0] weight_fill_tile_flat;
    logic [3:0] weight_fill_bank;
    logic [43:0] weight_fill_addr_flat;
    logic [799:0] weight_fill_data_flat;
    logic classifier_cache_valid;
    logic active_classifier_pair;
    logic [15:0] active_version;
    logic conv_cache_valid;
    logic [9:0] conv_cached_output_base;
    logic [4:0] conv_cached_output_count;
    logic [15:0] conv_cached_version;
    logic conv_cache_invalidate;
    logic busy;
    logic weight_fill_busy;
    logic sticky_error;

    integer classifier_fill_count;
    integer conv_fill_count;
    logic saw_conv_initial_even;
    logic saw_conv_initial_odd;
    logic saw_conv_updated_even;
    logic saw_classifier_updated;
    logic saw_invalidate;
    logic check_warm_refresh = 1'b0;
    integer prime_done_count = 0;
    logic signed [15:0] saved_code [0:PARAM_DEPTH-1];
    logic signed [15:0] saved_residual [0:PARAM_DEPTH-1];

    unified_scnn_selected_layers_parameter_plane_v1 #(
        .PARAM_DEPTH(PARAM_DEPTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .MAX_INPUT_CHANNELS(2), .MAX_CLASSIFIER_FEATURES(4),
        .MAX_CLASSES(2), .INPUT_CHANNEL_COUNT_WIDTH(2),
        .CLASSIFIER_FEATURE_COUNT_WIDTH(3), .CLASS_COUNT_WIDTH(2)
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
        .i_classifier_pair(1'b1), .i_initial_version(16'd5),
        .i_classifier_parameter_base(PARAM_ADDR_WIDTH'(CLASSIFIER_BASE)),
        .i_classifier_feature_count(3'd4),
        .i_classifier_class_count(2'd2),
        .o_classifier_prime_done(classifier_prime_done),
        .i_commit_pending(commit_pending),
        .o_commit_publish_ready(commit_publish_ready),
        .i_commit_pulse(commit_pulse),
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
        .i_conv_context_valid(conv_context_valid),
        .o_conv_context_ready(conv_context_ready),
        .i_conv_context_load_weights(conv_context_load_weights),
        .i_conv_output_channel_base(conv_output_channel_base),
        .i_conv_output_channel_count(conv_output_channel_count),
        .i_conv_input_channel_count(conv_input_channel_count),
        .i_conv_context_pair(conv_context_pair),
        .i_conv_parameter_base(conv_parameter_base),
        .i_conv_context_version(conv_context_version),
        .i_array_idle(array_idle),
        .o_weight_fill_valid(weight_fill_valid),
        .i_weight_fill_ready(weight_fill_ready),
        .o_weight_fill_tile_flat(weight_fill_tile_flat),
        .o_weight_fill_bank(weight_fill_bank),
        .o_weight_fill_addr_flat(weight_fill_addr_flat),
        .o_weight_fill_data_flat(weight_fill_data_flat),
        .o_weight_fill_busy(weight_fill_busy),
        .o_classifier_cache_valid(classifier_cache_valid),
        .o_active_classifier_pair(active_classifier_pair),
        .o_active_version(active_version),
        .o_conv_cache_valid(conv_cache_valid),
        .o_conv_cached_output_base(conv_cached_output_base),
        .o_conv_cached_output_count(conv_cached_output_count),
        .o_conv_cached_version(conv_cached_version),
        .o_conv_cache_invalidate(conv_cache_invalidate),
        .o_busy(busy), .o_sticky_error(sticky_error)
    );

    task automatic load_parameter(input int address, input int code);
        begin
            @(negedge clk);
            parameter_load_valid = 1'b1;
            parameter_load_addr = PARAM_ADDR_WIDTH'(address);
            parameter_load_code = 16'(code);
            parameter_load_residual = 16'(address);
            do @(posedge clk); while (!parameter_load_ready);
            @(negedge clk);
            parameter_load_valid = 1'b0;
        end
    endtask

    task automatic update_parameter(
        input int address,
        input int old_code,
        input int new_code
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
                    $signed(learning_response_code) != old_code)
                $fatal(1, "learning read mismatch addr=%0d", address);
            @(negedge clk);
            learning_write_valid = 1'b1;
            learning_write_addr = PARAM_ADDR_WIDTH'(address);
            learning_write_code = 16'(new_code);
            learning_write_residual = 16'(address + 1);
            learning_write_code_enable = 1'b1;
            learning_write_residual_enable = 1'b1;
            do @(posedge clk); while (!learning_write_ready);
            @(negedge clk);
            learning_write_valid = 1'b0;
        end
    endtask

    task automatic request_conv_context(
        input bit load_weights,
        input int version
    );
        begin
            @(negedge clk);
            conv_context_load_weights = load_weights;
            conv_context_version = 16'(version);
            conv_context_valid = 1'b1;
            do @(posedge clk); while (!conv_context_ready);
            @(negedge clk);
            conv_context_valid = 1'b0;
        end
    endtask

    always @(posedge clk) begin
        if (rst_n && classifier_prime_done)
            prime_done_count <= prime_done_count + 1;
        if (rst_n && check_warm_refresh &&
                (active_version != 16'd6 || !active_classifier_pair ||
                 commit_publish_ready || commit_pulse ||
                 parameter_load_ready ||
                 (learning_write_valid && learning_write_ready)))
            $fatal(1, "warm refresh changed committed metadata/store access");
        if (rst_n && conv_cache_invalidate)
            saw_invalidate <= 1'b1;
        for (int region = 0; region < 4; region++) begin
            if (weight_fill_valid[region] && weight_fill_ready[region]) begin
                int tile;
                int bank;
                logic [199:0] row_data;
                tile = region * 4 +
                    int'(weight_fill_tile_flat[region*2 +: 2]);
                bank = int'({weight_fill_addr_flat[region*11 + 10],
                    weight_fill_bank[region]});
                row_data = weight_fill_data_flat[region*200 +: 200];
                if (bank >= 2) begin
                    classifier_fill_count <= classifier_fill_count + 1;
                    if (tile == 0 && bank == 2 &&
                            $signed(row_data[15:0]) == 15)
                        saw_classifier_updated <= 1'b1;
                end else begin
                    conv_fill_count <= conv_fill_count + 1;
                    if (tile == 0 && bank == 0 &&
                            $signed(row_data[15:0]) == 100 &&
                            $signed(row_data[31:16]) == 102)
                        saw_conv_initial_even <= 1'b1;
                    if (tile == 0 && bank == 1 &&
                            $signed(row_data[15:0]) == 101 &&
                            $signed(row_data[31:16]) == 103)
                        saw_conv_initial_odd <= 1'b1;
                    if (tile == 0 && bank == 0 &&
                            $signed(row_data[15:0]) == 333)
                        saw_conv_updated_even <= 1'b1;
                end
            end
        end
    end

    initial begin
        rst_n = 1'b0;
        abort = 1'b0;
        clear_error = 1'b0;
        cache_invalidate = 1'b0;
        parameter_load_valid = 1'b0;
        parameter_load_addr = '0;
        parameter_load_code = '0;
        parameter_load_residual = '0;
        classifier_prime_valid = 1'b0;
        commit_pending = 1'b0;
        commit_pulse = 1'b0;
        learning_read_valid = 1'b0;
        learning_read_addr = '0;
        learning_read_tag = '0;
        learning_response_ready = 1'b1;
        learning_write_valid = 1'b0;
        learning_write_addr = '0;
        learning_write_code = '0;
        learning_write_residual = '0;
        learning_write_code_enable = 1'b1;
        learning_write_residual_enable = 1'b1;
        conv_context_valid = 1'b0;
        conv_context_load_weights = 1'b0;
        conv_output_channel_base = 10'd0;
        conv_output_channel_count = 5'd2;
        conv_input_channel_count = 2'd2;
        conv_context_pair = 1'b0;
        conv_parameter_base = '0;
        conv_context_version = 16'd5;
        array_idle = 1'b1;
        weight_fill_ready = 4'hf;
        classifier_fill_count = 0;
        conv_fill_count = 0;
        saw_conv_initial_even = 1'b0;
        saw_conv_initial_odd = 1'b0;
        saw_conv_updated_even = 1'b0;
        saw_classifier_updated = 1'b0;
        saw_invalidate = 1'b0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        for (int address = 0; address < 18; address++)
            load_parameter(address, 100 + address);
        for (int address = 18; address < 36; address++)
            load_parameter(address, 200 + address - 18);
        load_parameter(CLASSIFIER_BASE + 0, 10);
        load_parameter(CLASSIFIER_BASE + 1, 20);
        load_parameter(CLASSIFIER_BASE + 2, 30);
        load_parameter(CLASSIFIER_BASE + 3, 40);
        load_parameter(CLASSIFIER_BASE + 4, 50);
        load_parameter(CLASSIFIER_BASE + 5, 60);
        load_parameter(CLASSIFIER_BASE + 6, 70);
        load_parameter(CLASSIFIER_BASE + 7, 80);

        @(negedge clk);
        classifier_prime_valid = 1'b1;
        do @(posedge clk); while (!classifier_prime_ready);
        @(negedge clk);
        classifier_prime_valid = 1'b0;
        do @(posedge clk); while (!classifier_prime_done);
        if (!classifier_cache_valid || !active_classifier_pair ||
                active_version != 16'd5)
            $fatal(1, "classifier prime state mismatch");

        request_conv_context(1'b1, 5);
        if (!conv_cache_valid || conv_cached_output_base != 0 ||
                conv_cached_output_count != 2 || conv_cached_version != 5 ||
                !saw_conv_initial_even || !saw_conv_initial_odd)
            $fatal(1, "initial Conv residency mismatch");

        begin
            int fills_before_reuse;
            fills_before_reuse = conv_fill_count;
            request_conv_context(1'b0, 5);
            repeat (2) @(posedge clk);
            if (conv_fill_count != fills_before_reuse)
                $fatal(1, "resident spatial wave unexpectedly refilled weights");
        end

        update_parameter(0, 100, 333);
        update_parameter(CLASSIFIER_BASE, 10, 15);

        @(negedge clk);
        commit_pending = 1'b1;
        repeat (3) begin
            @(posedge clk);
            if (commit_publish_ready)
                $fatal(1, "publication completed before classifier refill");
        end
        do @(posedge clk); while (!commit_publish_ready);
        #1;
        if (!saw_invalidate || conv_cache_valid || !saw_classifier_updated ||
                active_version != 16'd6)
            $fatal(1, "publication/invalidation state mismatch invalidate=%0b conv_valid=%0b classifier_updated=%0b version=%0d",
                saw_invalidate, conv_cache_valid, saw_classifier_updated,
                active_version);
        @(negedge clk);
        commit_pulse = 1'b1;
        @(posedge clk);
        @(negedge clk);
        commit_pulse = 1'b0;
        commit_pending = 1'b0;
        @(posedge clk);
        if (commit_publish_ready)
            $fatal(1, "publication grant was not consumed by commit");

        request_conv_context(1'b1, 6);
        if (!conv_cache_valid || !saw_conv_updated_even ||
                conv_cached_version != 16'd6)
            $fatal(1, "updated Conv workset was not refilled");

        begin
            int classifier_fills_before;
            int conv_fills_before;
            @(negedge clk);
            for (int address = 0; address < PARAM_DEPTH; address++) begin
                saved_code[address] =
                    dut.u_backend.g_internal_store.u_parameter_store.code_mem[address];
                saved_residual[address] =
                    dut.u_backend.g_internal_store.u_parameter_store.residual_mem[address];
            end
            if (saved_code[0] != 333 || saved_residual[0] != 1 ||
                    saved_code[CLASSIFIER_BASE] != 15 ||
                    saved_residual[CLASSIFIER_BASE] != CLASSIFIER_BASE + 1)
                $fatal(1, "committed code/residual precondition mismatch");
            classifier_fills_before = classifier_fill_count;
            conv_fills_before = conv_fill_count;
            saw_classifier_updated = 1'b0;
            check_warm_refresh = 1'b1;
            cache_invalidate = 1'b1;
            classifier_prime_valid = 1'b1;
            repeat (2) begin
                #1;
                if (classifier_prime_ready || classifier_prime_done ||
                        classifier_cache_valid || conv_cache_valid ||
                        busy || weight_fill_busy)
                    $fatal(1, "invalidate/prime collision was admitted");
                @(negedge clk);
            end
            cache_invalidate = 1'b0;
            classifier_prime_valid = 1'b0;
            repeat (3) begin
                @(negedge clk);
                if (busy || weight_fill_busy || classifier_cache_valid ||
                        conv_cache_valid || weight_fill_valid != 0 ||
                        classifier_fill_count != classifier_fills_before ||
                        conv_fill_count != conv_fills_before)
                    $fatal(1, "eviction started unsolicited cache refresh");
            end
            array_idle = 1'b0;
            classifier_prime_valid = 1'b1;
            repeat (3) begin
                #1;
                if (classifier_prime_ready || busy || weight_fill_busy)
                    $fatal(1, "warm refresh ignored foreign-client exclusion");
                @(negedge clk);
            end
            array_idle = 1'b1;
            do @(posedge clk); while (!classifier_prime_ready);
            @(negedge clk);
            classifier_prime_valid = 1'b0;
            do @(posedge clk); while (!classifier_prime_done);
            @(negedge clk);
            if (!classifier_cache_valid || conv_cache_valid ||
                    !saw_classifier_updated || active_version != 16'd6 ||
                    classifier_fill_count != classifier_fills_before + 4 ||
                    conv_fill_count != conv_fills_before ||
                    prime_done_count != 2 || busy || weight_fill_busy)
                $fatal(1, "warm prime did not restore classifier-only residency");
            for (int address = 0; address < PARAM_DEPTH; address++) begin
                if (dut.u_backend.g_internal_store.u_parameter_store.code_mem[address]
                            !== saved_code[address] ||
                        dut.u_backend.g_internal_store.u_parameter_store.residual_mem[address]
                            !== saved_residual[address])
                    $fatal(1, "warm prime mutated persistent store addr=%0d", address);
            end
            check_warm_refresh = 1'b0;
            saw_conv_updated_even = 1'b0;
            request_conv_context(1'b1, 6);
            if (!conv_cache_valid || !saw_conv_updated_even ||
                    conv_cached_version != 16'd6 ||
                    conv_fill_count != conv_fills_before + 4)
                $fatal(1, "evicted Conv workset was not restored on demand");
            conv_fills_before = conv_fill_count;
            request_conv_context(1'b0, 6);
            repeat (2) @(negedge clk);
            if (conv_fill_count != conv_fills_before)
                $fatal(1, "restored Conv residency unexpectedly refilled");
        end
        if (sticky_error)
            $fatal(1, "unexpected parameter-plane sticky error");

        $display("PASS: SCNN parameter plane publishes atomically and restores evicted classifier/Conv residency without version or store changes");
        $finish;
    end

    initial begin
        repeat (5000) @(posedge clk);
        $fatal(1, "timeout");
    end
endmodule

`default_nettype wire
