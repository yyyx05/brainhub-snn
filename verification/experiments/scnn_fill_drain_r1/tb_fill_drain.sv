`timescale 1ns/1ps
`default_nettype none

// Local fill/drain datapath only: actual wrapper/owner/publisher/dispatcher/XPM.
// Scalar source is modeled, NOT selected_layers_parameter_backend_v1 or its
// arbitration/abort routing. No scheduler; final release is a TB promise.
module tb_fill_drain;
    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    integer cycles = 0, cases = 0, checks = 0;
    integer scalar_requests = 0, scalar_responses = 0, cancelled_responses = 0;
    integer row_writes = 0, ram_reads = 0, ram_responses = 0, turnovers = 0;
    integer blocked_reserves = 0, wrong_tags = 0, held_cycles = 0;
    integer recoveries = 0, publications = 0, protected_writes = 0, ram_stalls = 0;
    logic abort_req = 0, clear_error = 0, reserve = 0, activate = 0, release_req = 0;
    logic reserve_ready, reserve_rejected, activate_ready, activate_rejected;
    logic release_ready, release_rejected, action_slot = 0;
    logic [9:0] req_c = 3, req_row = 0;
    logic [23:0] req_parameter = 0;
    logic [1:0] occupied, slot_valid, slot_active;
    logic current_valid, current_slot, current_pair;
    logic [9:0] current_output, current_c, current_row;
    logic [4:0] current_count;
    logic [23:0] current_parameter;
    logic [15:0] current_version;
    logic [10:0] current_rows;
    logic draining, cancel, fence_drained, sticky, fill_drained, inflight, busy, done, discarded;
    logic scalar_valid, scalar_ready, response_valid, response_ready;
    logic [23:0] scalar_addr;
    logic [15:0] scalar_tag, response_tag, backend_tag = 0;
    logic signed [15:0] response_code, backend_code = 0;
    logic backend_pending = 0, response_enable = 1, inject_wrong = 0;
    integer backend_delay = 0;
    logic [3:0] fill_valid, fill_ready, fill_bank;
    logic [7:0] fill_tile;
    logic [43:0] fill_addr;
    logic [799:0] fill_data;
    logic store_fill_ready, ram_valid = 0, ram_ready, ram_result_valid, ram_result_ready = 0;
    logic ram_pending = 0, array_idle, read_drained;
    logic [9:0] ram_row = 0;
    logic [3:0] ram_sub = 0;
    wire [999:0] ram_term [0:3];
    logic [24:0] expected_read [0:3];
    logic [199:0] golden [0:3][0:577];
    bit written [0:3][0:577];
    bit job_live = 0, reserve_expected = 0, protect_a = 0, quarantine = 0;
    integer job_n, job_base, job_parameter, job_rows, job_requests, job_writes;
    integer frozen_writes = 0, frozen_publications = 0, next_tag = 0;
    bit expected_pending = 0;
    logic [15:0] expected_tag, expected_code;

    assign response_valid = inject_wrong || (backend_pending && backend_delay == 0 && response_enable);
    assign response_tag = inject_wrong ? (backend_tag ^ 16'h8000) : backend_tag;
    assign response_code = inject_wrong ? 16'sh55aa : backend_code;
    assign scalar_ready = !backend_pending ||
        (response_valid && response_ready && !inject_wrong);
    assign fill_ready = {3'b000, store_fill_ready};
    assign read_drained = !ram_pending && !ram_result_valid;
    assign array_idle = read_drained && !ram_valid;

    scnn_owned_publisher_v1 dut (
        .i_clk(clk), .i_rst_n(rst_n), .i_abort(abort_req), .i_cache_invalidate(1'b0),
        .i_commit_pending(1'b0), .i_clear_error(clear_error), .i_active_version(16'd0),
        .i_classifier_cache_valid(1'b1), .i_classifier_pair(1'b0),
        .i_array_idle(array_idle), .i_read_compute_drained(read_drained),
        .i_req_output_base(10'd0), .i_req_output_count(5'd1), .i_req_input_count(req_c),
        .i_req_pair(1'b1), .i_req_parameter_base(req_parameter), .i_req_version(16'd0),
        .i_req_row_base(req_row), .i_reserve_valid(reserve), .o_reserve_ready(reserve_ready),
        .o_reserve_rejected(reserve_rejected), .i_activate_valid(activate),
        .i_activate_slot(action_slot), .o_activate_ready(activate_ready),
        .o_activate_rejected(activate_rejected), .i_release_valid(release_req),
        .i_release_slot(action_slot), .i_release_final(1'b1), .o_release_ready(release_ready),
        .o_release_rejected(release_rejected), .o_forward_read_valid(scalar_valid),
        .i_forward_read_ready(scalar_ready), .o_forward_read_addr(scalar_addr),
        .o_forward_read_tag(scalar_tag), .i_forward_response_valid(response_valid),
        .o_forward_response_ready(response_ready), .i_forward_response_tag(response_tag),
        .i_forward_response_code(response_code), .i_forward_response_error(1'b0),
        .o_weight_fill_valid(fill_valid), .i_weight_fill_ready(fill_ready),
        .o_weight_fill_tile_flat(fill_tile), .o_weight_fill_bank(fill_bank),
        .o_weight_fill_addr_flat(fill_addr), .o_weight_fill_data_flat(fill_data),
        .o_slot_occupied(occupied), .o_slot_valid(slot_valid), .o_slot_active(slot_active),
        .o_current_valid(current_valid), .o_current_slot(current_slot), .o_current_pair(current_pair),
        .o_current_output_base(current_output), .o_current_output_count(current_count),
        .o_current_input_count(current_c), .o_current_row_base(current_row),
        .o_current_parameter_base(current_parameter), .o_current_version(current_version),
        .o_current_rows(current_rows), .o_draining(draining), .o_cancel(cancel),
        .o_fence_drained(fence_drained), .o_sticky_error(sticky), .o_fill_drained(fill_drained),
        .o_fill_inflight(inflight), .o_publisher_busy(busy), .o_publication_done(done),
        .o_discarded_response(discarded)
    );
    scnn_held_response_weight_store_v1 #(
        .LANES(40), .BANK_WIDTH(200), .BANK_DEPTH(578), .ADDR_WIDTH(10)
    ) store (
        .i_clk(clk), .i_rst_n(rst_n), .i_fill_valid(fill_valid[0]), .o_fill_ready(store_fill_ready),
        .i_fill_bank({fill_addr[10], fill_bank[0]}), .i_fill_addr(fill_addr[9:0]),
        .i_fill_data(fill_data[199:0]), .i_read_valid(ram_valid), .o_read_ready(ram_ready),
        .i_mode(2'd2), .i_fc_weight_int8(1'b0), .i_scnn_active_pair(1'b1),
        .i_read_addr0(ram_row), .i_read_addr1(ram_row), .i_read_addr2(ram_row), .i_read_addr3(ram_row),
        .i_read_subword0(ram_sub), .i_read_subword1(ram_sub),
        .i_read_subword2(ram_sub), .i_read_subword3(ram_sub),
        .o_read_result_valid(ram_result_valid), .i_read_result_ready(ram_result_ready),
        .o_term0_flat(ram_term[0]), .o_term1_flat(ram_term[1]),
        .o_term2_flat(ram_term[2]), .o_term3_flat(ram_term[3])
    );

    task automatic check(input logic ok, input string message);
        checks = checks + 1;
        if (ok !== 1'b1) $fatal(1, "FAIL_SCNN_FILL_DRAIN case=%0d cycle=%0d %s", cases+1, cycles, message);
    endtask
    task automatic tick;
        @(posedge clk); #1;
    endtask
    function automatic logic [15:0] value_at(input integer address);
        integer mixed;
        mixed = (address*73 + 19) ^ (address >> 8) ^ (address >> 16) ^ (address << 3);
        value_at = 16'((mixed & 32767) | ((address & 1) << 15));
    endfunction
    function automatic logic [199:0] row_at(input integer base, n, row, parity);
        integer f;
        row_at = '0;
        for (integer s = 0; s < 12; s++) begin
            f = row*24 + s*2 + parity;
            if (f < n) row_at[s*16 +: 16] = value_at(base+f);
        end
    endfunction
    task automatic check_current_a;
        check(current_valid && !current_slot && current_pair && current_output == 0 &&
            current_count == 1 && current_c == 3 && current_row == 0 && current_rows == 2 &&
            current_parameter == 24'ha12340 && current_version == 0, "current A changed before release");
    endtask

    // Only reset or a genuine matching response handshake retires source credit.
    // Abort, quarantine, injection and response_enable NEVER clear pending.
    always @(posedge clk) begin
        if (!rst_n) begin backend_pending <= 0; backend_delay <= 0; end
        else begin
            if (backend_pending && backend_delay > 0) backend_delay <= backend_delay - 1;
            if (response_valid && response_ready && !inject_wrong && backend_pending &&
                response_tag == backend_tag) backend_pending <= 0;
            if (scalar_valid && scalar_ready) begin
                backend_pending <= 1; backend_delay <= 2;
                backend_tag <= scalar_tag; backend_code <= value_at(integer'(scalar_addr));
            end
        end
    end
    always @(posedge clk) begin : scoreboard
        integer f, bank, row, even_count;
        logic [15:0] word_value;
        cycles <= cycles + 1;
        if (cycles > 30000) $fatal(1, "FAIL_SCNN_FILL_DRAIN watchdog");
        if (rst_n) begin
            check(inflight === backend_pending && expected_pending == backend_pending, "scalar credit mismatch");
            check(fill_valid[3:1] == 0, "unexpected tile region");
            if (backend_pending) check(!fill_drained && !reserve_ready && !fence_drained, "premature drain/allocation");
            if (fill_drained) check(!backend_pending && !busy && fill_valid == 0, "false fill drained");
            if (cancel) check(!scalar_valid && fill_valid == 0 && !done, "cancel leaked traffic/done");
            if (quarantine) check(row_writes == frozen_writes && publications == frozen_publications &&
                fill_valid == 0 && !done, "cancelled transaction modified RAM/publication");
            if (protect_a) check_current_a();
            if (reserve && !reserve_ready) blocked_reserves++;
            if (reserve && reserve_ready) check(reserve_expected && !backend_pending && !busy, "unplanned reserve");
            if (backend_pending && !response_valid) held_cycles++;
            if (response_valid && response_ready) begin
                if (inject_wrong) begin
                    check(expected_pending && response_tag != expected_tag && discarded && cancel,
                        "wrong tag did not enter sink");
                    wrong_tags++;
                end else begin
                    check(expected_pending && response_tag == expected_tag && response_code === expected_code,
                        "scalar response address/code/tag oracle");
                    check(discarded == cancel, "response sink classification");
                    expected_pending = 0; scalar_responses++;
                    if (cancel) cancelled_responses++;
                    else next_tag++;
                end
            end
            if (scalar_valid && scalar_ready) begin
                check(job_live && !cancel && !expected_pending, "request without available credit/job");
                even_count = (job_n+1)/2;
                f = job_requests < even_count ? job_requests*2 : (job_requests-even_count)*2+1;
                check(job_requests < job_n && scalar_addr == 24'(job_parameter+f), "accepted scalar address");
                check(scalar_tag == 16'(next_tag), "accepted scalar tag sequence");
                expected_code = value_at(job_parameter+f); expected_tag = 16'(next_tag);
                expected_pending = 1; job_requests++; scalar_requests++;
                if (response_valid && response_ready) turnovers++;
            end
            if (fill_valid[0] && fill_ready[0]) begin
                bank = {fill_addr[10], fill_bank[0]}; row = fill_addr[9:0];
                check(job_live && !quarantine && array_idle && fill_tile[1:0] == 0 &&
                    job_writes < job_rows*2 && bank == 2+job_writes/job_rows &&
                    row == job_base+job_writes%job_rows && row < 578, "RAM fill route/order");
                check(fill_data[199:0] === row_at(job_parameter, job_n, row-job_base, bank%2), "RAM fill data/padding");
                golden[bank][row] = row_at(job_parameter, job_n, row-job_base, bank%2);
                written[bank][row] = 1; job_writes++; row_writes++;
                if (protect_a) begin check(row >= 2, "B overwrote A"); protected_writes++; end
            end
            if (done) begin
                check(job_live && job_requests == job_n && job_writes == job_rows*2 && !backend_pending,
                    "publication completion totals");
                publications++;
            end
            if (ram_result_valid) begin
                check(ram_pending, "RAM response without request");
                for (integer t = 0; t < 4; t++)
                    for (integer lane = 0; lane < 40; lane++)
                        check(ram_term[t][lane*25 +: 25] === expected_read[t], "RAM readback/held payload");
                if (!ram_result_ready) ram_stalls++;
                else begin ram_pending <= 0; ram_responses++; end
            end
            if (ram_valid && ram_ready) begin
                check(!ram_pending && !busy, "RAM read before local drain");
                for (integer t = 0; t < 4; t++) begin
                    bank = 2+t%2;
                    check(written[bank][ram_row], "read of uninitialized row");
                    word_value = golden[bank][ram_row][ram_sub*16 +: 16];
                    expected_read[t] = {{9{word_value[15]}}, word_value};
                end
                ram_pending <= 1; ram_reads++;
            end
        end
    end

    task automatic descriptor(input integer c, row, parameter_base);
        req_c = 10'(c); req_row = 10'(row); req_parameter = 24'(parameter_base);
    endtask
    task automatic begin_job(input integer c, row, parameter_base, slot);
        @(negedge clk);
        check(!backend_pending && !busy && fill_drained && !sticky, "job before drain/clear");
        quarantine = 0; descriptor(c, row, parameter_base);
        job_n = c*9; job_base = row; job_parameter = parameter_base; job_rows = (c*9+23)/24;
        job_requests = 0; job_writes = 0; job_live = 1; reserve_expected = 1; reserve = 1;
        #1; check(reserve_ready, "reserve unavailable"); tick();
        check(!reserve_rejected && occupied[slot], "reserve rejected/wrong slot");
        @(negedge clk); reserve = 0; reserve_expected = 0;
        descriptor(511, 1023, 'h123456); // Accepted descriptor must be latched.
    endtask
    task automatic finish_job(input integer slot);
        integer timeout;
        timeout = 0;
        while (!slot_valid[slot] && timeout < 3000) begin tick(); timeout++; end
        check(slot_valid[slot] && !busy && fill_drained && !sticky, "publication timeout/error");
        check(job_requests == job_n && job_writes == job_rows*2, "job totals");
        @(negedge clk); job_live = 0;
    endtask
    task automatic ownership(input bit releasing, input integer slot, c, row, parameter_base);
        @(negedge clk); descriptor(c, row, parameter_base); action_slot = 1'(slot);
        activate = !releasing; release_req = releasing;
        #1; check(releasing ? release_ready : activate_ready, "ownership not ready"); tick();
        check(!activate_rejected && !release_rejected, "ownership descriptor rejected");
        check(releasing ? !current_valid : (current_valid && current_slot == slot &&
            current_c == c && current_row == row && current_parameter == parameter_base), "ownership metadata");
        @(negedge clk); activate = 0; release_req = 0;
    endtask
    task automatic readback(input integer base, rows);
        for (integer r = base; r < base+rows; r++)
            for (integer s = 0; s < 12; s++) begin
                @(negedge clk); ram_row = 10'(r); ram_sub = 4'(s); ram_valid = 1;
                #1; check(ram_ready, "RAM not ready"); tick();
                @(negedge clk); ram_valid = 0;
                check(ram_result_valid && !read_drained, "RAM response/drain missing");
                repeat (2) tick();
                @(negedge clk); ram_result_ready = 1; tick();
                check(!ram_result_valid && !ram_pending, "RAM response not retired");
                @(negedge clk); ram_result_ready = 0;
            end
    endtask
    task automatic wait_empty;
        integer timeout;
        timeout = 0;
        while ((occupied != 0 || cancel || sticky || !fill_drained) && timeout < 64) begin tick(); timeout++; end
        check(occupied == 0 && !cancel && !sticky && fill_drained && reserve_ready, "owner recovery timeout");
    endtask
    task automatic cancel_case(input bit with_wrong_tag);
        integer before_responses, timeout;
        @(negedge clk); response_enable = 0;
        begin_job(3, 576, 'he45670 + cases*256, 0);
        timeout = 0;
        while ((!backend_pending || backend_delay != 0) && timeout < 64) begin tick(); timeout++; end
        check(backend_pending && job_requests == 1, "missing held accepted scalar");
        before_responses = scalar_responses;
        @(negedge clk); abort_req = 1; reserve = 1; job_live = 0; quarantine = 1;
        frozen_writes = row_writes; frozen_publications = publications;
        #1; check(cancel && !reserve_ready && fill_valid == 0, "abort did not quarantine"); tick();
        @(negedge clk); abort_req = 0;
        repeat (3) begin
            tick(); check(backend_pending && inflight && occupied == 1 && !fill_drained && !busy,
                "publisher idle incorrectly retired outstanding scalar");
        end
        if (with_wrong_tag) begin
            @(negedge clk); inject_wrong = 1;
            #1; check(response_valid && response_ready && discarded, "wrong-tag sink handshake missing"); tick();
            @(negedge clk); inject_wrong = 0; clear_error = 1;
            repeat (3) begin
                tick(); check(sticky && backend_pending && inflight && occupied == 1 && !fill_drained &&
                    scalar_responses == before_responses, "wrong tag/early clear released genuine credit");
            end
            @(negedge clk); clear_error = 0;
        end
        @(negedge clk); response_enable = 1;
        #1; check(response_valid && response_ready && discarded && cancel, "late correct response not sunk"); tick();
        check(!backend_pending && !inflight && scalar_responses == before_responses+1, "matching response did not retire");
        @(negedge clk); reserve = 0;
        if (with_wrong_tag) begin
            check(fill_drained && sticky, "malformed sticky lost before post-drain clear");
            clear_error = 1; repeat (2) tick();
            @(negedge clk); clear_error = 0;
        end
        wait_empty();
        check(row_writes == frozen_writes && publications == frozen_publications, "cancel leaked writes/done");
        readback(576, 2); // Previous B/recovery contents must survive the cancelled response.
        begin_job(3, 576, 'hf78900 + cases*256, 0); finish_job(0);
        check(!current_valid, "done implicitly activated recovery");
        ownership(0, 0, 3, 576, 'hf78900 + cases*256); readback(576, 2);
        ownership(1, 0, 3, 576, 'hf78900 + cases*256); recoveries++;
        cases++; $display("CASE_SCNN_FILL_DRAIN id=%0d wrong_tag=%0d", cases, with_wrong_tag);
    endtask

    initial begin
`ifndef UNIFIED_USE_XPM_MEMORY
        $fatal(1, "FAIL_SCNN_FILL_DRAIN requires UNIFIED_USE_XPM_MEMORY and unmodified vendor XPM");
`endif
        for (integer b = 0; b < 4; b++)
            for (integer r = 0; r < 578; r++) begin golden[b][r] = 0; written[b][r] = 0; end
        repeat (4) tick(); @(negedge clk); rst_n = 1; tick();
        begin_job(3, 0, 'ha12340, 0); finish_job(0);
        check(!current_valid, "initial done implicitly activated A");
        ownership(0, 0, 3, 0, 'ha12340); protect_a = 1; readback(0, 2);
        begin_job(9, 574, 'hd23450, 1); finish_job(1);
        check_current_a(); readback(0, 2);
        @(negedge clk); protect_a = 0;
        ownership(1, 0, 3, 0, 'ha12340);
        ownership(0, 1, 9, 574, 'hd23450); readback(574, 4);
        ownership(1, 1, 9, 574, 'hd23450);
        cases++; $display("CASE_SCNN_FILL_DRAIN id=%0d normal_A_B", cases);
        cancel_case(0); cancel_case(1);
        readback(0, 2);
        check(cases == 3 && recoveries == 2 && publications == 4 && protected_writes == 8 &&
            cancelled_responses == 2 && wrong_tags == 1 && turnovers > 0 && blocked_reserves > 0 &&
            held_cycles > 0 && ram_stalls > 0 && row_writes > 0 && ram_reads > 0 &&
            scalar_requests == scalar_responses && ram_reads == ram_responses && !backend_pending &&
            !ram_pending && !ram_result_valid && !busy && fill_drained && occupied == 0 && !sticky,
            "required directed coverage/final drain");
        $display("PASS_SCNN_FILL_DRAIN cases=%0d checks=%0d scalar_requests=%0d scalar_responses=%0d cancelled_responses=%0d row_writes=%0d ram_reads=%0d ram_responses=%0d wrong_tags=%0d turnovers=%0d blocked_reserves=%0d held_cycles=%0d ram_stalls=%0d recoveries=%0d publications=%0d protected_writes=%0d cycles=%0d",
            cases, checks, scalar_requests, scalar_responses, cancelled_responses, row_writes,
            ram_reads, ram_responses, wrong_tags, turnovers, blocked_reserves, held_cycles,
            ram_stalls, recoveries, publications, protected_writes, cycles);
        $finish;
    end
endmodule
`default_nettype wire
