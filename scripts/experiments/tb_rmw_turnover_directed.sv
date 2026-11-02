`timescale 1ns/1ps
`default_nettype none
module tb_rmw_turnover_directed(input logic clk, output logic done);
    logic rst_n=0, cfg_valid=0, release_context=0;
    logic req_valid=0, read_ready=0, rsp_valid=0, rsp_error=0, result_ready=0;
    logic [23:0] local_addr=0;
    logic [15:0] tag=0;
    logic signed [15:0] rsp_code=0, rsp_residual=0;
    wire cfg_ready, cfg_error, release_ready, active, req_ready, req_error;
    wire read_valid, rsp_ready, result_valid, wc, wr, wcf, wrf;
    wire dirty, saturated, config_error, store_error, internal_error, busy;
    wire [23:0] read_addr, result_addr;
    wire [15:0] read_tag, result_tag;
    wire [7:0] read_profile, read_tensor, result_profile, result_tensor;
    wire signed [15:0] result_code, result_residual;
    logic signed [15:0] codes[4], residues[4];
    integer credit=0, read_credit=0, responded=0, owner_addr=0, owner_tag=0;
    integer expected_code=0, expected_residual=0, expected_wc=0;
    integer requests=0, reads=0, responses=0, results=0, turnovers=0;
    integer illegal_requests=0, store_errors=0, reset_drops=0;
    integer code_writes=0, residual_writes=0, cases=0;
    integer read_stalls=0, result_stalls=0, max_credit=0, cycle=0, last_result_cycle=-2;
    logic held=0, read_held=0;
    logic [91:0] held_payload;
    logic [55:0] held_read;
    wire [91:0] payload={result_addr,result_tag,result_code,result_residual,
                        result_profile,result_tensor,wc,wr,dirty,saturated};

    unified_tp_parameter_rmw_service dut (
        .i_clk(clk),.i_rst_n(rst_n),.i_cfg_valid(cfg_valid),.o_cfg_ready(cfg_ready),
        .i_cfg_profile_id(8'd2),.i_cfg_tensor_id(8'd3),
        .i_cfg_parameter_base(24'd4096),.i_cfg_element_count(24'd4),
        .i_cfg_vote_increment(16'd31),.i_cfg_vote_threshold(16'd65),
        .i_cfg_code_min(-16'sd32768),.i_cfg_code_max(16'sd32767),.o_cfg_error(cfg_error),
        .i_release(release_context),.o_release_ready(release_ready),.o_active(active),
        .o_active_profile_id(),.o_active_tensor_id(),
        .i_req_valid(req_valid),.o_req_ready(req_ready),.i_req_local_parameter_addr(local_addr),
        .i_req_tag(tag),.i_req_update_direction(2'sd1),.i_req_magnitude_mode(1'b0),
        .i_req_scaled_residual_delta(18'sd0),.o_req_error(req_error),
        .o_read_valid(read_valid),.i_read_ready(read_ready),.o_read_profile_id(read_profile),
        .o_read_tensor_id(read_tensor),.o_read_parameter_addr(read_addr),.o_read_tag(read_tag),
        .i_read_response_valid(rsp_valid),.o_read_response_ready(rsp_ready),
        .i_read_response_code(rsp_code),.i_read_response_residual(rsp_residual),
        .i_read_response_error(rsp_error),.o_result_valid(result_valid),.i_result_ready(result_ready),
        .o_result_profile_id(result_profile),.o_result_tensor_id(result_tensor),
        .o_result_parameter_addr(result_addr),.o_result_tag(result_tag),
        .o_result_code(result_code),.o_result_residual(result_residual),
        .o_write_code_enable(wc),.o_write_residual_enable(wr),
        .o_write_code_fire(wcf),.o_write_residual_fire(wrf),.o_dirty(dirty),
        .o_saturated(saturated),.o_update_config_error(config_error),
        .o_store_error(store_error),.o_internal_error(internal_error),.o_busy(busy)
    );

    // Retire the old owner before allocating the replacement, just like replay.
    always @(posedge clk) begin
        cycle++;
        if (!rst_n) begin
            credit=0;read_credit=0;responded=0;held=0;read_held=0;
        end else begin
            if (cfg_error || config_error || internal_error) $fatal(1,"directed diagnostic");
            if (held && (!result_valid || payload!==held_payload)) $fatal(1,"held result changed");
            if (read_held && (!read_valid || held_read!=={read_addr,read_tag,read_profile,read_tensor}))
                $fatal(1,"held read changed");
            held=result_valid && !result_ready;held_payload=payload;
            read_held=read_valid && !read_ready;held_read={read_addr,read_tag,read_profile,read_tensor};
            if (held) result_stalls++;
            if (read_held) read_stalls++;
            if ((wcf || wrf) && !(result_valid && result_ready)) $fatal(1,"unqualified write");
            if (read_valid && result_valid) $fatal(1,"read before previous write edge");
            if (req_ready && busy && !(result_valid && result_ready)) $fatal(1,"early request credit");
            if ((cfg_valid || release_context) && req_ready) $fatal(1,"control priority lost");
            if (result_valid && result_ready) begin
                if (credit!=1 || read_credit!=0 || responded!=1) $fatal(1,"result credit");
                if (result_addr!=24'(4096+owner_addr) || result_tag!=16'(owner_tag) ||
                    result_profile!=2 || result_tensor!=3) $fatal(1,"result metadata");
                if (result_code!==16'(expected_code) || result_residual!==16'(expected_residual) ||
                    wc!==1'(expected_wc) || !wr || !dirty || saturated || wcf!==wc || !wrf)
                    $fatal(1,"directed arithmetic/write mismatch");
                if (wcf) begin codes[owner_addr]=result_code;code_writes++;end
                if (wrf) begin residues[owner_addr]=result_residual;residual_writes++;end
                results++;credit=0;responded=0;last_result_cycle=cycle;
                if (req_valid && req_ready) turnovers++;
            end
            if (rsp_valid && rsp_ready) begin
                if (credit!=1 || read_credit!=1) $fatal(1,"response credit");
                responses++;read_credit=0;
                if (rsp_error) begin store_errors++;credit=0;end
                else responded=1;
            end
            if (read_valid && read_ready) begin
                if (credit!=1 || read_credit!=0 || responded!=0 || cycle<=last_result_cycle)
                    $fatal(1,"read credit/order");
                if (read_addr!=24'(4096+owner_addr) || read_tag!=16'(owner_tag) ||
                    read_profile!=2 || read_tensor!=3) $fatal(1,"read metadata");
                rsp_code=codes[owner_addr];rsp_residual=residues[owner_addr];
                expected_code=int'(rsp_code);expected_residual=int'(rsp_residual)+31;
                expected_wc=0;
                if (expected_residual>=65) begin
                    expected_residual-=65;expected_code++;expected_wc=1;
                end
                reads++;read_credit=1;
            end
            if (req_valid && req_ready) begin
                if (!active || local_addr>=4) illegal_requests++;
                else begin
                    if (credit!=0 || read_credit!=0) $fatal(1,"second outstanding owner");
                    requests++;credit=1;owner_addr=int'(local_addr);owner_tag=int'(tag);
                end
            end
            if (credit>max_credit) max_credit=credit;
            if (credit<0 || credit>1 || read_credit<0 || read_credit>1) $fatal(1,"credit bounds");
        end
    end

    task automatic step;
        @(posedge clk);#1;@(negedge clk);#1;
    endtask
    task automatic configure;
        cfg_valid=1;#1;
        if (!cfg_ready || req_ready) $fatal(1,"cfg ready/priority");
        step();cfg_valid=0;#1;
        if (!active || busy) $fatal(1,"configure failed");
    endtask
    task automatic fresh;
        rst_n=0;cfg_valid=0;release_context=0;req_valid=0;read_ready=0;
        rsp_valid=0;rsp_error=0;result_ready=0;
        repeat(2) step();
        for (integer a=0;a<4;a++) begin codes[a]=16'sd10;residues[a]=16'sd60;end
        rst_n=1;step();configure();
    endtask
    task automatic request(input integer address, input integer id);
        local_addr=24'(address);tag=16'(id);req_valid=1;#1;
        if (!req_ready) $fatal(1,"expected request ready");
        step();req_valid=0;
    endtask
    task automatic read_and_respond(input integer stalls, input integer latency, input bit error_response);
        read_ready=0;
        repeat(stalls) step();
        read_ready=1;#1;
        if (!read_valid) $fatal(1,"missing next-cycle read");
        step();read_ready=0;
        repeat(latency) step();
        rsp_valid=1;rsp_error=error_response;#1;
        if (!rsp_ready) $fatal(1,"response not ready");
        step();rsp_valid=0;rsp_error=0;
    endtask
    task automatic wait_result;
        integer timeout_cycles;
        timeout_cycles=0;#1;
        while (!result_valid) begin
            step();timeout_cycles++;
            if (timeout_cycles>20) $fatal(1,"directed result timeout");
        end
    endtask
    task automatic retire;
        result_ready=1;step();result_ready=0;#1;
        if (busy || result_valid || credit!=0) $fatal(1,"retire not drained");
    endtask
    task automatic passed(input string name);
        cases++;$display("RMW_DIRECTED_CASE name=%s",name);
    endtask

    initial begin
        done=0;
        if ($test$plusargs("directed")) begin
            @(negedge clk);#1;
            fresh();request(0,1);read_and_respond(0,0,0);wait_result();
            local_addr=0;tag=2;req_valid=1;result_ready=1;#1;
            if (!req_ready || read_valid) $fatal(1,"turnover boundary");
            step();req_valid=0;result_ready=0;#1;
            if (!read_valid || codes[0]!=11 || residues[0]!=26) $fatal(1,"turnover next read/write");
            read_and_respond(0,0,0);wait_result();retire();
            if (codes[0]!=11 || residues[0]!=57) $fatal(1,"same-address freshness");
            passed("immediate_same_address");

            fresh();request(0,3);read_and_respond(3,4,0);wait_result();
            local_addr=0;tag=4;req_valid=1;
            repeat(5) begin #1;if(req_ready) $fatal(1,"stalled result accepts");step();end
            result_ready=1;#1;if(!req_ready) $fatal(1,"missing stalled turnover");
            step();req_valid=0;result_ready=0;
            read_and_respond(4,5,0);wait_result();retire();
            passed("stalled_result_read_response");

            fresh();request(0,5);read_and_respond(0,0,0);wait_result();
            req_valid=1;local_addr=1;cfg_valid=1;result_ready=1;#1;
            if(req_ready || cfg_ready || release_ready) $fatal(1,"pending cfg retirement priority");
            step();result_ready=0;#1;
            if(busy || req_ready || cfg_ready || !active) $fatal(1,"active cfg must wait for release");
            release_context=1;#1;if(!release_ready) $fatal(1,"release pending cfg");
            step();release_context=0;#1;
            if(active || !cfg_ready || req_ready) $fatal(1,"cfg after release");
            step();cfg_valid=0;req_valid=0;#1;
            if(!active || busy) $fatal(1,"pending reconfigure failed");
            passed("cfg_pending_release_then_cfg");

            fresh();request(0,6);read_and_respond(0,0,0);wait_result();
            req_valid=1;local_addr=1;release_context=1;result_ready=1;#1;
            if(req_ready || release_ready) $fatal(1,"release retirement priority");
            step();result_ready=0;#1;
            if(!release_ready || req_ready || busy) $fatal(1,"release not deferred to idle");
            step();release_context=0;req_valid=0;#1;
            if(active || busy) $fatal(1,"release incomplete");
            passed("release_pending");

            fresh();request(0,7);read_and_respond(0,0,0);wait_result();
            local_addr=4;tag=8;req_valid=1;result_ready=1;#1;
            if(!req_ready) $fatal(1,"illegal turnover not acknowledged");
            step();req_valid=0;result_ready=0;#1;
            if(!req_error || busy || read_valid || result_valid || codes[0]!=11 || residues[0]!=26)
                $fatal(1,"illegal request lost old retirement");
            step();if(req_error) $fatal(1,"request error not a pulse");
            request(0,9);read_and_respond(0,0,0);wait_result();retire();
            passed("illegal_address_turnover_recovery");

            fresh();request(0,10);read_and_respond(0,0,0);wait_result();
            local_addr=0;tag=11;req_valid=1;result_ready=1;step();req_valid=0;result_ready=0;
            read_and_respond(1,2,1);#1;
            if(!store_error || busy || result_valid || codes[0]!=11 || residues[0]!=26)
                $fatal(1,"store error after turnover");
            step();if(store_error) $fatal(1,"store error not a pulse");
            request(0,12);read_and_respond(0,0,0);wait_result();retire();
            passed("store_error_after_turnover_recovery");

            fresh();request(0,13);read_and_respond(0,0,0);wait_result();
            local_addr=0;tag=14;req_valid=1;result_ready=1;step();req_valid=0;result_ready=0;
            rst_n=0;reset_drops++;step();#1;
            if(busy || read_valid || result_valid || wcf || wrf || active || codes[0]!=11 || residues[0]!=26)
                $fatal(1,"reset next request after turnover");
            rst_n=1;step();configure();request(0,15);read_and_respond(0,0,0);wait_result();retire();
            passed("reset_after_turnover_before_read");

            fresh();request(0,16);read_ready=1;step();read_ready=0;
            rst_n=0;reset_drops++;step();rst_n=1;rsp_valid=1;#1;
            if(rsp_ready || result_valid || busy) $fatal(1,"stale response after reset");
            step();rsp_valid=0;configure();request(0,17);read_and_respond(0,0,0);wait_result();retire();
            passed("reset_read_inflight_stale_response");

            fresh();request(0,18);read_and_respond(0,0,0);wait_result();
            req_valid=1;local_addr=0;tag=19;result_ready=0;step();
            rst_n=0;req_valid=0;reset_drops++;step();#1;
            if(result_valid || wcf || wrf || codes[0]!=10 || residues[0]!=60) $fatal(1,"reset stalled result write");
            rst_n=1;step();configure();request(0,20);read_and_respond(0,0,0);wait_result();retire();
            passed("reset_stalled_result_no_write");

            if(cases!=9 || turnovers!=5 || illegal_requests!=1 || store_errors!=1 || reset_drops!=3 ||
               requests!=results+store_errors+reset_drops || credit!=0 || read_credit!=0 || max_credit!=1 ||
               !read_stalls || !result_stalls) $fatal(1,"directed coverage/accounting");
            $display("RMW_DIRECTED_RESULT cases=%0d requests=%0d reads=%0d responses=%0d results=%0d turnovers=%0d illegal_requests=%0d store_errors=%0d reset_drops=%0d code_writes=%0d residual_writes=%0d max_credit=%0d read_stalls=%0d result_stalls=%0d",cases,requests,reads,responses,results,turnovers,illegal_requests,store_errors,reset_drops,code_writes,residual_writes,max_credit,read_stalls,result_stalls);
        end
        done=1;
    end
endmodule
`default_nettype wire
