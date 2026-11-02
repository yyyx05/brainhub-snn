`timescale 1ns/1ps
`default_nettype none
module tb_rmw_turnover_replay;
    wire directed_done;
    tb_rmw_turnover_directed directed_tests (.clk(clk),.done(directed_done));
    localparam int N=588*5120;
    logic clk=0;
    always #5 clk=~clk;
    logic rst_n=0, cfg_valid=0, release_context=0;
    integer policy=0;
    logic req_valid=0, read_ready=0, rsp_valid=0, result_ready=0;
    logic [23:0] local_addr=0;
    logic [15:0] tag=0;
    logic signed [63:0] gradient=0;
    wire signed [1:0] direction=(gradient>0)?-2'sd1:(gradient<0)?2'sd1:2'sd0;
    logic signed [15:0] rsp_code=0;
    logic signed [31:0] rsp_residual=0;
    wire [1:0] cfg_ready, cfg_error, release_ready, active, req_ready, req_error;
    wire [1:0] read_valid, rsp_ready, result_valid, wc, wr, wcf, wrf;
    wire [1:0] dirty, saturated, config_error, store_error, internal_error, busy;
    wire [23:0] read_addr[2], result_addr[2];
    wire [15:0] read_tag[2], result_tag[2];
    wire [7:0] read_profile[2],read_tensor[2],result_profile[2],result_tensor[2];
    wire signed [15:0] result_code[2];
    wire signed [31:0] result_residual[2];
    logic [151:0] vectors [N];
    logic signed [15:0] initial_code[5120], code_store[5120];
    logic signed [31:0] residual_store[5120];

    // Direct-only replay, preserving the accepted input stream and service patterns.
    for (genvar p=0;p<1;p=p+1) begin : arms
        localparam int RW=(p==0)?16:32;
        wire signed [RW-1:0] residual_out;
        assign result_residual[p]={{(32-RW){residual_out[RW-1]}},residual_out};
`define COMMON_PORTS \
            .i_clk(clk),.i_rst_n(rst_n), \
            .i_cfg_valid(cfg_valid && policy==p),.o_cfg_ready(cfg_ready[p]), \
            .i_cfg_profile_id(8'd2),.i_cfg_tensor_id(8'd3), \
            .i_cfg_parameter_base(24'd4096),.i_cfg_element_count(24'd5120), \
            .i_cfg_vote_increment(16'd31),.i_cfg_vote_threshold(16'd65), \
            .i_cfg_code_min(-16'sd32768),.i_cfg_code_max(16'sd32767),.o_cfg_error(cfg_error[p]), \
            .i_release(release_context && policy==p),.o_release_ready(release_ready[p]), \
            .o_active(active[p]),.o_active_profile_id(),.o_active_tensor_id(), \
            .i_req_valid(req_valid && policy==p),.o_req_ready(req_ready[p]), \
            .i_req_local_parameter_addr(local_addr),.i_req_tag(tag), \
            .i_req_update_direction(direction),.i_req_magnitude_mode(1'b0), \
            .i_req_scaled_residual_delta(18'sd0),.o_req_error(req_error[p]), \
            .o_read_valid(read_valid[p]),.i_read_ready(read_ready), \
            .o_read_profile_id(read_profile[p]),.o_read_tensor_id(read_tensor[p]), \
            .o_read_parameter_addr(read_addr[p]),.o_read_tag(read_tag[p]), \
            .i_read_response_valid(rsp_valid && policy==p),.o_read_response_ready(rsp_ready[p]), \
            .i_read_response_code(rsp_code),.i_read_response_residual(rsp_residual[RW-1:0]), \
            .i_read_response_error(1'b0),.o_result_valid(result_valid[p]),.i_result_ready(result_ready), \
            .o_result_profile_id(result_profile[p]),.o_result_tensor_id(result_tensor[p]), \
            .o_result_parameter_addr(result_addr[p]),.o_result_tag(result_tag[p]), \
            .o_result_code(result_code[p]),.o_result_residual(residual_out), \
            .o_write_code_enable(wc[p]),.o_write_residual_enable(wr[p]), \
            .o_write_code_fire(wcf[p]),.o_write_residual_fire(wrf[p]), \
            .o_dirty(dirty[p]),.o_saturated(saturated[p]),.o_update_config_error(config_error[p]), \
            .o_store_error(store_error[p]),.o_internal_error(internal_error[p]),.o_busy(busy[p])
        unified_tp_parameter_rmw_service #(.RESIDUAL_WIDTH(RW)) dut (`COMMON_PORTS);
`undef COMMON_PORTS
    end

    function automatic logic signed [15:0] expected_code(input integer index);
        return $signed(vectors[index][87:72]);
    endfunction
    function automatic logic signed [31:0] expected_residual(input integer index);
        return {{16{vectors[index][71]}},vectors[index][71:56]};
    endfunction
    function automatic logic [3:0] expected_writes(input integer index);
        return vectors[index][55:52];
    endfunction
    task automatic run_case(input integer scenario);
        integer word_index, inflight_index, cycles, due, memory_addr;
        integer requests, reads, responses, results, zeros, code_writes, residual_writes;
        integer turnovers, zero_retirements;
        integer saturation_count, read_stalls, result_stalls, state_checks, dependency_stalls;
        logic outstanding, pending, held, read_held;
        logic [107:0] held_payload;
        logic [39:0] held_read;
        logic [3:0] ew;
        begin
            @(negedge clk);
            rst_n=0;cfg_valid=0;release_context=0;req_valid=0;rsp_valid=0;
            read_ready=0;result_ready=0;
            repeat(4) @(negedge clk);
            for(integer a=0;a<5120;a=a+1) begin
                code_store[a]=initial_code[a];residual_store[a]=0;
            end
            rst_n=1;
            @(negedge clk);
            if (!cfg_ready[policy]) $fatal(1,"configuration not ready");
            cfg_valid=1;
            @(negedge clk);
            cfg_valid=0;
            if (!active[policy]) $fatal(1,"configuration inactive");
            word_index=0;inflight_index=0;cycles=0;due=0;memory_addr=0;
            requests=0;reads=0;responses=0;results=0;zeros=0;code_writes=0;residual_writes=0;
            turnovers=0;zero_retirements=0;
            saturation_count=0;read_stalls=0;result_stalls=0;state_checks=0;dependency_stalls=0;
            outstanding=0;pending=0;held=0;read_held=0;held_payload=0;held_read=0;
            while(word_index<N || outstanding || busy[policy]) begin
                // All drive changes are at the falling edge, not the sample edge.
                gradient=word_index<N?$signed(vectors[word_index][151:88]):64'sd0;
                local_addr=24'(word_index%5120);tag=16'(word_index);
                req_valid=(word_index<N && gradient!=0);
                read_ready=scenario==0 || (cycles%11>=4);
                result_ready=scenario==0 || (cycles%13>=7);
                rsp_valid=pending && cycles>=due;
                @(posedge clk);
                if (cfg_error[policy] || req_error[policy] || config_error[policy] ||
                    store_error[policy] || internal_error[policy]) $fatal(1,"service diagnostic");
                if (held && (!result_valid[policy] || held_payload!==
                    {result_addr[policy],result_tag[policy],result_code[policy],result_residual[policy],
                     wc[policy],wr[policy],dirty[policy],saturated[policy],result_profile[policy],result_tensor[policy]}))
                    $fatal(1,"result changed under backpressure");
                held=result_valid[policy] && !result_ready;
                held_payload={result_addr[policy],result_tag[policy],result_code[policy],result_residual[policy],
                     wc[policy],wr[policy],dirty[policy],saturated[policy],result_profile[policy],result_tensor[policy]};
                if (read_held && (!read_valid[policy] || held_read!=={read_addr[policy],read_tag[policy]}))
                    $fatal(1,"read request changed under backpressure");
                read_held=read_valid[policy] && !read_ready;
                held_read={read_addr[policy],read_tag[policy]};
                if (read_held) read_stalls++;
                if (held) result_stalls++;
                if (word_index<N && !req_ready[policy]) dependency_stalls++;
                if ((wcf[policy] || wrf[policy]) && !(result_valid[policy] && result_ready))
                    $fatal(1,"write without retirement");
                if (read_valid[policy] && result_valid[policy])
                    $fatal(1,"new read overlaps old write");
                if (result_valid[policy] && result_ready && req_ready[policy]) begin
                    if (req_valid) turnovers++;
                    else if (word_index<N && gradient==0) zero_retirements++;
                end
                if (result_valid[policy] && result_ready) begin
                    if (!outstanding || pending || responses!=reads || reads!=requests)
                        $fatal(1,"result credit mismatch");
                    if (result_addr[policy]!=4096+inflight_index%5120 || result_tag[policy]!=16'(inflight_index) ||
                        result_profile[policy]!=2 || result_tensor[policy]!=3) $fatal(1,"result metadata");
                    ew=expected_writes(inflight_index);
                    if (result_code[policy]!==expected_code(inflight_index) ||
                        result_residual[policy]!==expected_residual(inflight_index) ||
                        {wr[policy],wc[policy]}!==ew[1:0] || dirty[policy]!==(|ew[1:0]) ||
                        wcf[policy]!==wc[policy] || wrf[policy]!==wr[policy])
                        $fatal(1,"state/write mismatch policy=%0d index=%0d code=%0d/%0d residual=%0d/%0d",policy,inflight_index,
                            result_code[policy],expected_code(inflight_index),result_residual[policy],expected_residual(inflight_index));
                    memory_addr=inflight_index%5120;
                    if (wcf[policy]) begin code_store[memory_addr]=result_code[policy];code_writes++;end
                    if (wrf[policy]) begin residual_store[memory_addr]=result_residual[policy];residual_writes++;end
                    saturation_count+=int'(saturated[policy]);
                    state_checks++;results++;outstanding=0;
                end
                if (rsp_valid && rsp_ready[policy]) begin
                    if (!pending || !outstanding) $fatal(1,"response credit");
                    pending=0;responses++;
                end
                if (read_valid[policy] && read_ready) begin
                    if (pending || !outstanding || reads!=responses || read_addr[policy]!=4096+inflight_index%5120 ||
                        read_tag[policy]!=16'(inflight_index) || read_profile[policy]!=2 || read_tensor[policy]!=3)
                        $fatal(1,"read credit/metadata");
                    memory_addr=inflight_index%5120;
                    rsp_code=code_store[memory_addr];rsp_residual=residual_store[memory_addr];
                    due=cycles+(scenario==0?1:5);pending=1;reads++;
                end
                if (word_index<N && req_ready[policy]) begin
                    if (gradient==0) begin
                        if (code_store[word_index%5120]!==expected_code(word_index) ||
                            residual_store[word_index%5120]!==expected_residual(word_index)) $fatal(1,"zero skip state");
                        zeros++;state_checks++;
                    end else begin
                        if (outstanding || pending || requests!=results) $fatal(1,"more than one owner");
                        if (gradient>64'sd7328690601984 || gradient< -64'sd7328690601984)
                            $fatal(1,"gradient outside proven bound");
                        inflight_index=word_index;outstanding=1;requests++;
                    end
                    word_index++;
                end
                cycles++;
                if (cycles>150000000) $fatal(1,"watchdog");
                @(negedge clk);
            end
            req_valid=0;rsp_valid=0;
            for(integer a=0;a<5120;a=a+1) begin
                if(code_store[a]!==expected_code(N-5120+a) || residual_store[a]!==expected_residual(N-5120+a))
                    $fatal(1,"final store mismatch");
                state_checks++;
            end
            if(!release_ready[policy]) $fatal(1,"not drained at release");
            release_context=1;
            @(negedge clk);release_context=0;
            if(active[policy] || busy[policy]) $fatal(1,"release incomplete");
            $display("RMW_TURNOVER_RESULT policy=%0d scenario=%0d words=%0d requests=%0d reads=%0d responses=%0d results=%0d zero=%0d code_writes=%0d residual_writes=%0d saturated=%0d state_checks=%0d cycles=%0d read_stalls=%0d result_stalls=%0d dependency_stalls=%0d turnovers=%0d zero_retirements=%0d",policy,scenario,word_index,requests,reads,responses,results,zeros,code_writes,residual_writes,saturation_count,state_checks,cycles,read_stalls,result_stalls,dependency_stalls,turnovers,zero_retirements);
        end
    endtask
    initial begin
        if ($test$plusargs("directed")) begin
            wait(directed_done);
            $display("RMW_TURNOVER_DIRECTED_PASS");
            $finish;
        end else begin
        if (!$test$plusargs("replay")) $fatal(1,"choose +directed or +replay");
        $readmemh("vectors.hex",vectors);
        $readmemh("initial.hex",initial_code);
        for(integer p=0;p<1;p=p+1) begin
            policy=p;
            for(integer s=0;s<2;s=s+1) run_case(s);
        end
        $display("RMW_TURNOVER_REPLAY_PASS");$finish;
        end
    end
endmodule
`default_nettype wire
