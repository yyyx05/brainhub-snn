`timescale 1ns/1ps
`default_nettype none

module tb_fc_replay_model_logit_scale;
    localparam int MAX_NEURONS = 32;
    localparam int CW = $clog2(MAX_NEURONS + 1);
    // round(W2scale * 64 * 65536), supplied explicitly by this model's caller.
    localparam logic [23:0] MODEL_SCALE_Q16 = 24'd47303;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic abort = 1'b0;
    logic clear_error = 1'b0;
    logic cfg_valid = 1'b0;
    logic cfg_ready;
    logic [1:0] cfg_mode = 2'd0;
    logic [5:0] cfg_sample = '0;
    logic [CW-1:0] cfg_count = '0;
    logic cfg_output = 1'b1;
    logic [3:0] cfg_label = '0;
    logic cfg_scale_enable = 1'b0;
    logic [23:0] cfg_scale = '0;
    logic cfg_accumulate = 1'b0;
    logic cfg_sample_start = 1'b0;
    logic replay_valid = 1'b0;
    logic replay_ready;
    logic [1:0] replay_mode = 2'd0;
    logic [CW-1:0] replay_index = '0;
    logic signed [31:0] replay_pre = '0;
    logic signed [31:0] replay_trace = '0;
    logic replay_last = 1'b0;
    logic student_valid, surrogate_valid, backbone_valid;
    logic row_valid;
    logic row_ready = 1'b0;
    logic [5:0] logit_sample;
    logic [3:0] logit_label;
    logic logit_valid;
    logic logit_ready = 1'b0;
    logic signed [15:0] logit_q6;
    logic logit_last;
    logic done, busy, idle, error;
    logic [3:0] error_code;

    int expected_values[$];
    bit expected_lasts[$];
    bit active_scale_enable;
    logic [23:0] active_scale;
    logic [5:0] active_sample;
    logic [3:0] active_label;
    int active_count;
    bit active_accumulate, active_start;
    int reference_bank [0:63][0:15];
    int row_inputs, row_outputs, row_headers;
    bit bad_item = 1'b0;
    bit stalled = 1'b0;
    logic signed [15:0] held_value;
    logic held_last;
    int cycles = 0;
    int inputs = 0;
    int outputs = 0;
    int dropped = 0;
    int rows = 0;
    int headers = 0;
    int hold_checks = 0;
    int replacements = 0;
    int config_rejects = 0;
    int replay_rejects = 0;
    int abort_checks = 0;
    int clear_checks = 0;
    int cumulative_rows = 0;
    int raw_wraps = 0;
    int zero_steps = 0;

    always #5 clk = ~clk;
    initial begin
        #2000000;
        $fatal(1, "logit-scale watchdog expired");
    end

    unified_fc_srnn_h40_tp_replay_router_v1 #(
        .MAX_NEURONS(MAX_NEURONS), .SAMPLE_ID_WIDTH(6)
    ) u_dut (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(abort),
        .i_clear_error(clear_error),
        .i_cfg_valid(cfg_valid), .o_cfg_ready(cfg_ready),
        .i_cfg_mode(cfg_mode), .i_cfg_sample_id(cfg_sample),
        .i_cfg_neuron_count(cfg_count), .i_cfg_fc_threshold(18'sd1024),
        .i_cfg_fc_output_layer(cfg_output), .i_cfg_fc_label(cfg_label),
        .i_cfg_srnn_path(1'b0),
        .i_cfg_fc_logit_scale_enable(cfg_scale_enable),
        .i_cfg_fc_logit_scale_q16(cfg_scale),
        .i_cfg_fc_logit_accumulate(cfg_accumulate),
        .i_cfg_fc_sample_start(cfg_sample_start),
        .i_replay_valid(replay_valid), .o_replay_ready(replay_ready),
        .i_replay_mode(replay_mode), .i_replay_local_index(replay_index),
        .i_replay_spike(1'b0), .i_replay_pre_state(replay_pre),
        .i_replay_trace_next(replay_trace), .i_replay_last(replay_last),
        .o_fc_student_trace_valid(student_valid),
        .i_fc_student_trace_ready(1'b1), .o_fc_student_trace_sample(),
        .o_fc_student_trace_hidden(), .o_fc_student_trace_q016(),
        .o_fc_surrogate_valid(surrogate_valid),
        .i_fc_surrogate_ready(1'b1), .o_fc_surrogate_sample(),
        .o_fc_surrogate_hidden(), .o_fc_surrogate_q016(),
        .o_fc_logit_row_valid(row_valid), .i_fc_logit_row_ready(row_ready),
        .o_fc_logit_sample(logit_sample), .o_fc_logit_label(logit_label),
        .o_fc_logit_valid(logit_valid), .i_fc_logit_ready(logit_ready),
        .o_fc_logit_q6(logit_q6), .o_fc_logit_last(logit_last),
        .o_srnn_backbone_valid(backbone_valid),
        .i_srnn_backbone_ready(1'b1), .o_srnn_backbone_path(),
        .o_srnn_backbone_sample_id(), .o_srnn_backbone_id(),
        .o_srnn_backbone_q14(), .o_srnn_backbone_last(),
        .o_done(done), .o_busy(busy), .o_idle(idle),
        .o_sticky_error(error), .o_error_code(error_code)
    );

    // Independent signed quotient/remainder oracle, without magnitude shifts.
    function automatic int reference_q6(input int raw, input bit enable,
                                         input logic [23:0] scale);
        longint signed product, quotient, remainder;
        begin
            if (!enable)
                return int'($signed(16'(raw >>> 4)));
            product = 64'(raw) * $signed({40'd0, scale});
            quotient = product / 65536;
            remainder = product % 65536;
            if (remainder > 32768 ||
                (remainder == 32768 && quotient % 2 != 0))
                quotient = quotient + 1;
            else if (remainder < -32768 ||
                     (remainder == -32768 && quotient % 2 != 0))
                quotient = quotient - 1;
            if (quotient > 32767) return 32767;
            if (quotient < -32768) return -32768;
            return int'(quotient);
        end
    endfunction

    function automatic int raw_case(input int index_value);
        case (index_value)
            0: return 0;
            1: return 1;
            2: return -1;
            3: return 3;
            4: return -3;
            5: return 5;
            6: return -5;
            7: return 32767;
            8: return -32767;
            9: return 32768;
            10: return -32768;
            11: return 65535;
            12: return -65535;
            13: return 131071;
            14: return -131072;
            default: return -17;
        endcase
    endfunction

    // All stimulus changes on falling edges; check handshakes before the edge
    // and completion after NBA updates, including the legacy zero-latency path.
    task automatic tick;
        bit take_input, take_output, finish_row;
        int previous_size, raw_value, raw_sum;
        begin
            #1;
            take_input = replay_valid && replay_ready;
            take_output = logit_valid && logit_ready;
            finish_row = take_output && logit_last;
            previous_size = expected_values.size();
            if (student_valid || surrogate_valid || backbone_valid)
                $fatal(1, "output replay leaked onto another route");
            if (abort || clear_error) begin
                if (cfg_ready || replay_ready || row_valid || logit_valid)
                    $fatal(1, "handshake survived abort/clear");
                dropped += expected_values.size();
                expected_values.delete();
                expected_lasts.delete();
                stalled = 1'b0;
            end else begin
                if (stalled) begin
                    if (!logit_valid || logit_q6 !== held_value ||
                        logit_last !== held_last || logit_sample !== active_sample ||
                        logit_label !== active_label)
                        $fatal(1, "stalled logit or metadata changed");
                    hold_checks++;
                end
                if (row_valid) begin
                    if (logit_sample !== active_sample || logit_label !== active_label)
                        $fatal(1, "row metadata did not retain configuration");
                    if (row_ready) begin
                        row_headers++;
                        headers++;
                        if (row_headers != 1) $fatal(1, "duplicate row header");
                    end
                end
                if (take_input && !bad_item) begin
                    if (row_headers != 1 || int'(replay_index) != row_inputs ||
                        replay_last != (row_inputs == active_count - 1))
                        $fatal(1, "unexpected replay acceptance");
                    raw_value = replay_pre;
                    if (active_accumulate) begin
                        raw_sum = (active_start ? 0 :
                            reference_bank[active_sample][row_inputs]) + raw_value;
                        // Independent explicit modulo arithmetic for wrap24.
                        if (raw_sum > 8388607) begin
                            raw_sum -= 16777216;
                            raw_wraps++;
                        end else if (raw_sum < -8388608) begin
                            raw_sum += 16777216;
                            raw_wraps++;
                        end
                        reference_bank[active_sample][row_inputs] = raw_sum;
                        if (raw_value == 0) zero_steps++;
                        raw_value = raw_sum;
                    end
                    expected_values.push_back(reference_q6(raw_value,
                        active_scale_enable, active_scale));
                    expected_lasts.push_back(replay_last);
                    row_inputs++;
                    inputs++;
                end
                if (logit_valid) begin
                    if (!active_scale_enable && !take_input) begin
                        if (int'(logit_q6) != reference_q6(replay_pre, 1'b0, 24'd0) ||
                            logit_last != replay_last)
                            $fatal(1, "legacy combinational stall payload changed");
                    end else if (expected_values.size() == 0 ||
                        int'(logit_q6) != expected_values[0] ||
                        logit_last != expected_lasts[0] ||
                        logit_sample !== active_sample || logit_label !== active_label)
                        $fatal(1, "logit mismatch cycle=%0d raw=%0d scale=%0d got=%0d queue=%p",
                            cycles, replay_pre, active_scale, logit_q6, expected_values);
                    if (active_scale_enable && previous_size == 0)
                        $fatal(1, "scaled output bypassed its register");
                end
                if (take_output) begin
                    void'(expected_values.pop_front());
                    void'(expected_lasts.pop_front());
                    row_outputs++;
                    outputs++;
                    if (active_scale_enable && take_input && !bad_item)
                        replacements++;
                end
                if (bad_item && take_input) begin
                    dropped += expected_values.size();
                    expected_values.delete();
                    expected_lasts.delete();
                    replay_rejects++;
                end
                stalled = logit_valid && !logit_ready && !(bad_item && take_input);
                held_value = logit_q6;
                held_last = logit_last;
            end
            @(posedge clk);
            #1;
            cycles++;
            if (done !== finish_row)
                $fatal(1, "done must coincide with final output consumption");
            if (finish_row) begin
                if (!idle || !cfg_ready || row_inputs != active_count ||
                    row_outputs != active_count || expected_values.size() != 0)
                    $fatal(1, "row completed before exact output drain");
                rows++;
                if (active_accumulate) cumulative_rows++;
            end else if (expected_values.size() != 0 && (!busy || idle || cfg_ready))
                $fatal(1, "pending output released configuration");
            @(negedge clk);
        end
    endtask

    task automatic configure(input bit enable, input logic [23:0] scale,
                             input int count_value,
                             input bit accumulate = 1'b0,
                             input bit sample_start = 1'b0,
                             input int sample_id = -1);
        begin
            if (!idle || error || expected_values.size() != 0)
                $fatal(1, "configuration started before prior row drained");
            cfg_mode = 2'd0;
            cfg_output = 1'b1;
            cfg_count = CW'(count_value);
            cfg_sample = sample_id < 0 ? 6'(headers + 2) : 6'(sample_id);
            cfg_label = 4'(headers + 5);
            cfg_scale_enable = enable;
            cfg_scale = scale;
            cfg_accumulate = accumulate;
            cfg_sample_start = sample_start;
            active_sample = cfg_sample;
            active_label = cfg_label;
            active_scale_enable = enable;
            active_scale = scale;
            active_count = count_value;
            active_accumulate = accumulate;
            active_start = sample_start;
            row_inputs = 0;
            row_outputs = 0;
            row_headers = 0;
            cfg_valid = 1'b1;
            tick();
            cfg_valid = 1'b0;
            if (!busy || error) $fatal(1, "legal config rejected");
            // All later arithmetic and metadata must use the accepted snapshot.
            cfg_scale_enable = !enable;
            cfg_scale = ~scale;
            cfg_accumulate = !accumulate;
            cfg_sample_start = !sample_start;
            cfg_sample = ~active_sample;
            cfg_label = ~active_label;
            cfg_count = '0;
            cfg_mode = 2'd1;
            cfg_output = 1'b0;
        end
    endtask

    task automatic drive_item(input int index_value, input int raw_value);
        begin
            replay_mode = 2'd0;
            replay_index = CW'(index_value);
            replay_pre = raw_value;
            replay_trace = 0;
            replay_last = index_value == active_count - 1;
            replay_valid = 1'b1;
        end
    endtask

    task automatic run_row(input bit enable, input logic [23:0] scale,
                           input int count_value, input int pattern_id,
                           input bit use_stalls,
                           input bit accumulate = 1'b0,
                           input bit sample_start = 1'b0,
                           input int sample_id = -1);
        int started_rows, sent, waited;
        int raw_value;
        bit accepted;
        begin
            started_rows = rows;
            configure(enable, scale, count_value, accumulate, sample_start, sample_id);
            row_ready = 1'b0;
            logit_ready = 1'b0;
            drive_item(0, raw_case(0));
            repeat (3) begin
                tick();
                if (!row_valid || replay_ready || logit_valid)
                    $fatal(1, "row-header backpressure was not retained");
            end
            row_ready = 1'b1;
            for (sent = 0; sent < count_value; sent++) begin
                case (pattern_id)
                    0: raw_value = raw_case(sent);
                    -1: raw_value = 0;
                    -2: raw_value = 1;
                    -3: raw_value = -1;
                    -4: raw_value = 131071;
                    -5: raw_value = -131072;
                    default: raw_value =
                        ((pattern_id * 7919 + sent * 32771) % 262144) - 131072;
                endcase
                drive_item(sent, raw_value);
                waited = 0;
                accepted = 1'b0;
                while (!accepted) begin
                    logit_ready = !use_stalls || cycles % 7 >= 3;
                    #1;
                    accepted = replay_ready;
                    tick();
                    waited++;
                    if (waited > 30) $fatal(1, "replay stalled indefinitely");
                end
            end
            replay_valid = 1'b0;
            if (enable) begin
                logit_ready = 1'b0;
                cfg_valid = 1'b1;
                // A held final input (even invalid next metadata) cannot be
                // accepted again while the final registered output is pending.
                replay_valid = 1'b1;
                replay_index = '1;
                replay_last = 1'b0;
                replay_pre = -99999;
                repeat (5) begin
                    tick();
                    if (done || cfg_ready || replay_ready || !logit_valid || !logit_last)
                        $fatal(1, "final output stall admitted work or finished early");
                end
                replay_valid = 1'b0;
                cfg_valid = 1'b0;
                logit_ready = 1'b1;
                tick();
            end
            repeat (2) tick();
            if (rows != started_rows + 1 || error || !idle)
                $fatal(1, "row did not complete exactly once");
        end
    endtask

    task automatic pending_item(input bit final_item,
                                input bit accumulate = 1'b0,
                                input bit read_pending = 1'b0,
                                input bit sample_start = 1'b1);
        begin
            configure(1'b1, MODEL_SCALE_Q16, final_item ? 1 : 2,
                accumulate, sample_start, 31);
            row_ready = 1'b1;
            logit_ready = 1'b0;
            drive_item(0, -12345);
            tick();
            tick();
            replay_valid = 1'b0;
            if (accumulate && read_pending) begin
                if (logit_valid || !busy || done)
                    $fatal(1, "accumulator read stage bypassed its register");
                return;
            end
            if (accumulate) tick();
            if (!logit_valid || logit_last != final_item || !busy || done)
                $fatal(1, "failed to create pending output");
            repeat (3) tick();
        end
    endtask

    task automatic flush_pending(input bit use_clear, input bit final_item,
                                 input bit accumulate = 1'b0,
                                 input bit read_pending = 1'b0,
                                 input bit sample_start = 1'b1);
        begin
            if (accumulate && !sample_start)
                run_row(1'b1, MODEL_SCALE_Q16, final_item ? 1 : 2,
                    -2, 1'b0, 1'b1, 1'b1, 31);
            pending_item(final_item, accumulate, read_pending, sample_start);
            // Cancel while the sink is ready and another cfg/input is offered.
            logit_ready = 1'b1;
            cfg_valid = 1'b1;
            replay_valid = 1'b1;
            abort = !use_clear;
            clear_error = use_clear;
            tick();
            abort = 1'b0;
            clear_error = 1'b0;
            cfg_valid = 1'b0;
            replay_valid = 1'b0;
            repeat (3) tick();
            if (!idle || error || logit_valid || row_valid)
                $fatal(1, "flush left stale output/configuration");
            if (use_clear) clear_checks++;
            else abort_checks++;
            if (accumulate) begin
                reject_config(2'd0, 1'b1, MODEL_SCALE_Q16,
                    1'b1, 1'b1, 1'b0, 31, final_item ? 1 : 2);
                run_row(1'b1, 24'd65536, 3, 0, 1'b0, 1'b1, 1'b1, 31);
                run_row(1'b1, MODEL_SCALE_Q16, 3, -1, 1'b1, 1'b1, 1'b0, 31);
            end else run_row(1'b1, 24'd65536, 3, 0, 1'b0);
        end
    endtask

    task automatic reject_config(input logic [1:0] mode_value,
                                 input bit output_value,
                                 input logic [23:0] scale,
                                 input bit enable = 1'b1,
                                 input bit accumulate = 1'b0,
                                 input bit sample_start = 1'b0,
                                 input int sample_id = 0,
                                 input int count_value = 2);
        begin
            cfg_mode = mode_value;
            cfg_output = output_value;
            cfg_count = CW'(count_value);
            cfg_sample = 6'(sample_id);
            cfg_scale_enable = enable;
            cfg_scale = scale;
            cfg_accumulate = accumulate;
            cfg_sample_start = sample_start;
            cfg_valid = 1'b1;
            tick();
            cfg_valid = 1'b0;
            if (!error || error_code != 4'h1 || !idle || cfg_ready)
                $fatal(1, "illegal scale config accepted");
            config_rejects++;
            clear_error = 1'b1;
            tick();
            clear_error = 1'b0;
            tick();
            if (error || !cfg_ready) $fatal(1, "config error did not clear");
        end
    endtask

    task automatic reject_replay(input int kind, input bit accumulate = 1'b0);
        begin
            pending_item(1'b0, accumulate);
            drive_item(1, 12);
            case (kind)
                0: replay_index = CW'(0);
                1: replay_last = 1'b0;
                2: replay_pre = 131072;
                3: replay_pre = -131073;
                4: replay_trace = 65536;
                default: replay_mode = 2'd1;
            endcase
            bad_item = 1'b1;
            tick();
            replay_valid = 1'b0;
            bad_item = 1'b0;
            if (!error || !idle || logit_valid ||
                error_code != ((kind >= 2 && kind <= 4) ? 4'h3 : 4'h2))
                $fatal(1, "malformed replay did not flush pending logit");
            clear_error = 1'b1;
            tick();
            clear_error = 1'b0;
            logit_ready = 1'b1;
            repeat (2) tick();
            if (accumulate)
                reject_config(2'd0, 1'b1, MODEL_SCALE_Q16,
                    1'b1, 1'b1, 1'b0, 31, 2);
        end
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        tick();
        if (reference_q6(1, 1'b1, 24'd32768) != 0 ||
            reference_q6(-3, 1'b1, 24'd32768) != -2 ||
            reference_q6(5, 1'b1, 24'd32768) != 2 ||
            reference_q6(-1, 1'b0, 24'd0) != -1)
            $fatal(1, "reference directed anchors failed");

        run_row(1'b1, MODEL_SCALE_Q16, 16, 0, 1'b1);
        run_row(1'b1, 24'd32768, 16, 0, 1'b0);
        run_row(1'b1, 24'd32767, 16, 0, 1'b1);
        run_row(1'b1, 24'd32769, 16, 0, 1'b1);
        run_row(1'b1, 24'd65536, 16, 0, 1'b0);
        run_row(1'b1, 24'd1, 16, 0, 1'b1);
        run_row(1'b1, 24'hffffff, 16, 0, 1'b1);
        run_row(1'b0, 24'd0, 16, 0, 1'b1);
        run_row(1'b0, MODEL_SCALE_Q16, 16, 0, 1'b0);
        run_row(1'b1, MODEL_SCALE_Q16, 1, 0, 1'b1);
        for (int trial = 1; trial <= 64; trial++)
            run_row(1'b1, 24'(trial * 1049), 16, trial, 1'b1);

        reject_config(2'd0, 1'b1, 24'd0);
        reject_config(2'd0, 1'b0, MODEL_SCALE_Q16);
        reject_config(2'd1, 1'b0, MODEL_SCALE_Q16);
        reject_config(2'd1, 1'b1, MODEL_SCALE_Q16);
        reject_config(2'd2, 1'b0, MODEL_SCALE_Q16);
        flush_pending(1'b0, 1'b0);
        flush_pending(1'b0, 1'b1);
        flush_pending(1'b1, 1'b0);
        flush_pending(1'b1, 1'b1);
        for (int kind = 0; kind < 6; kind++) reject_replay(kind);
        run_row(1'b1, MODEL_SCALE_Q16, 16, 0, 1'b1);
        run_row(1'b0, 24'd0, 16, 0, 1'b1);

        // Continuation is legal only after a completely consumed cumulative row.
        reject_config(2'd0, 1'b1, MODEL_SCALE_Q16, 1'b1, 1'b1, 1'b0, 0, 3);
        reject_config(2'd0, 1'b1, MODEL_SCALE_Q16, 1'b0, 1'b1, 1'b1);
        reject_config(2'd0, 1'b0, MODEL_SCALE_Q16, 1'b1, 1'b1, 1'b1);
        reject_config(2'd1, 1'b0, MODEL_SCALE_Q16, 1'b1, 1'b1, 1'b1);
        run_row(1'b1, MODEL_SCALE_Q16, 3, -2, 1'b1, 1'b1, 1'b1, 0);
        run_row(1'b1, MODEL_SCALE_Q16, 3, -3, 1'b1, 1'b1, 1'b1, 63);
        run_row(1'b1, MODEL_SCALE_Q16, 3, -2, 1'b1, 1'b1, 1'b0, 0);
        run_row(1'b1, MODEL_SCALE_Q16, 3, -1, 1'b1, 1'b1, 1'b0, 63);
        run_row(1'b1, MODEL_SCALE_Q16, 3, -1, 1'b1, 1'b1, 1'b0, 0);
        run_row(1'b1, MODEL_SCALE_Q16, 3, -2, 1'b1, 1'b1, 1'b0, 63);
        run_row(1'b1, MODEL_SCALE_Q16, 3, -3, 1'b1, 1'b1, 1'b1, 0);
        run_row(1'b1, MODEL_SCALE_Q16, 3, -3, 1'b1, 1'b1, 1'b0, 63);
        run_row(1'b1, 24'd65536, 3, -1, 1'b1, 1'b1, 1'b0, 0);
        reject_config(2'd0, 1'b1, MODEL_SCALE_Q16, 1'b1, 1'b1, 1'b0, 63, 4);
        run_row(1'b1, MODEL_SCALE_Q16, 16, 0, 1'b1, 1'b1, 1'b1, 40);
        run_row(1'b1, MODEL_SCALE_Q16, 16, 0, 1'b1, 1'b1, 1'b0, 40);

        // Cross each signed24 boundary using only legal signed18 step inputs.
        for (int step_id = 0; step_id < 65; step_id++) begin
            run_row(1'b1, 24'd1, 1, -4, 1'b1, 1'b1, step_id == 0, 20);
            run_row(1'b1, 24'd1, 1, -5, 1'b1, 1'b1, step_id == 0, 21);
        end
        run_row(1'b1, MODEL_SCALE_Q16, 1, -1, 1'b1, 1'b1, 1'b0, 20);
        run_row(1'b1, MODEL_SCALE_Q16, 1, -1, 1'b1, 1'b1, 1'b0, 21);

        flush_pending(1'b0, 1'b0, 1'b1, 1'b0);
        flush_pending(1'b0, 1'b1, 1'b1, 1'b0, 1'b0);
        flush_pending(1'b1, 1'b0, 1'b1, 1'b0, 1'b0);
        flush_pending(1'b1, 1'b1, 1'b1, 1'b0);
        flush_pending(1'b0, 1'b0, 1'b1, 1'b1);
        flush_pending(1'b1, 1'b1, 1'b1, 1'b1, 1'b0);
        for (int kind = 0; kind < 6; kind++) reject_replay(kind, 1'b1);
        run_row(1'b1, MODEL_SCALE_Q16, 3, -1, 1'b1, 1'b1, 1'b0, 0);
        run_row(1'b1, MODEL_SCALE_Q16, 3, -1, 1'b1, 1'b1, 1'b0, 63);
        run_row(1'b1, MODEL_SCALE_Q16, 3, -2, 1'b1, 1'b1, 1'b1, 31);
        run_row(1'b1, MODEL_SCALE_Q16, 3, -2, 1'b1, 1'b1, 1'b0, 31);
        if (inputs != outputs + dropped || expected_values.size() != 0 ||
            replacements == 0 || hold_checks == 0 || rows != 242 ||
            cumulative_rows != 162 || raw_wraps != 2 || zero_steps == 0 ||
            config_rejects != 22 || replay_rejects != 12 ||
            abort_checks != 5 || clear_checks != 5)
            $fatal(1, "coverage/accounting mismatch in=%0d out=%0d drop=%0d rows=%0d",
                inputs, outputs, dropped, rows);
        $display("FC_REPLAY_MODEL_LOGIT_SCALE_PASS inputs=%0d outputs=%0d dropped=%0d rows=%0d headers=%0d hold_checks=%0d replacements=%0d config_rejects=%0d replay_rejects=%0d abort_checks=%0d clear_checks=%0d cumulative_rows=%0d raw_wraps=%0d zero_steps=%0d cycles=%0d",
            inputs, outputs, dropped, rows, headers, hold_checks, replacements,
            config_rejects, replay_rejects, abort_checks, clear_checks,
            cumulative_rows, raw_wraps, zero_steps, cycles);
        $finish;
    end
endmodule

`default_nettype wire
