`timescale 1ns/1ps
`default_nettype none

// The runner expands only explicit pin connections at the four placeholders.
// All four instantiated modules are real accepted RTL, with a 512-word store.
module tb_m2_refill_turnover;
    logic clk=0;
    always #5 clk=~clk;
    logic rst_n=0, clear_error=0, abort_pub=0, array_idle=1;
    logic cmd_valid=0, cmd_ready, pub_done, pub_busy, pub_error;
    logic [3:0] tile=0, published_tile;
    logic [1:0] bank=0, fmt=0, published_bank;
    logic [9:0] row=0, base=0, stride=1, published_row;
    logic [5:0] count=0;
    logic [15:0] version=0, published_version;
    logic p_req,p_ready,p_rsp,p_take;
    logic [9:0] p_addr;
    logic [7:0] p_tag;
    logic [3:0] fill_valid,fill_ready;
    logic [7:0] fill_tile;
    logic [3:0] fill_bank;
    logic [43:0] fill_addr;
    logic [799:0] fill_data;
    logic cfg_valid=0,cfg_ready,cfg_error,load_valid=0,load_ready;
    logic [9:0] load_addr=0;
    logic signed [15:0] load_code=0;
    logic forward_start=0,forward_ready,forward_end=0,forward_end_ready,forward_done;
    logic forward_error,slice_error,slice_busy,learning_ready;
    logic [15:0] model_version,adapt_version,forward_model_version,forward_adapt_version;
    logic c_req,c_ready,c_rsp,c_take,c_rsp_error;
    logic [9:0] c_addr;
    logic [7:0] c_tag,c_rsp_tag;
    logic signed [15:0] c_rsp_code;
    logic manual=0,m_req=0,m_take=0;
    logic [9:0] m_addr=0;
    logic [7:0] m_tag=0;
    logic s_load,s_req,s_ready,s_rsp,s_take,s_rsp_error;
    logic [9:0] s_load_addr,s_addr;
    logic signed [15:0] s_load_code,s_load_resid,s_rsp_code;
    logic [7:0] s_tag,s_rsp_tag;
    logic unexpected_learning_read,unexpected_learning_write;
    logic owner_active=1;
    logic [1:0] owner=2;
    logic [2:0] r_load_ready,r_ready,r_rsp,r_rsp_error;
    logic [23:0] r_rsp_tag;
    logic [47:0] r_rsp_code;
    logic router_quiet,router_error,store_error;
    logic b_load,b_load_ready,b_req,b_ready,b_rsp,b_take,b_rsp_error;
    logic [9:0] b_load_addr,b_addr,router_addr;
    logic signed [15:0] b_load_code,b_load_resid,b_rsp_code;
    logic [7:0] b_tag,b_rsp_tag;
    logic request_gate,response_gate;
    integer arm,mode=0,fault=0,cycle=0,delay_left=0;
    integer reads=0,responses=0,fills=0,credit=0,max_credit=0;
    integer replacements=0,read_stalls=0,response_stalls=0,fill_stalls=0;
    integer held_request_checks=0,held_response_checks=0,held_fill_checks=0;
    integer expected_count=0,start_cycle=0,limit=0;
    logic checking=0,request_held=0,response_held=0,fill_held=0;
    logic [17:0] saved_request;
    logic [24:0] saved_response;
    logic [859:0] saved_fill;
    logic [199:0] expected_row;

    assign c_req=manual ? m_req : p_req;
    assign c_addr=manual ? m_addr : p_addr;
    assign c_tag=manual ? m_tag : p_tag;
    assign c_take=manual ? m_take : p_take;
    assign p_ready=!manual && c_ready;
    assign p_rsp=!manual && c_rsp;
    // mode 0: L1 replacement, mode 1: deliberately non-turnover service,
    // mode 2: delayed response plus deterministic read/fill backpressure.
    assign request_gate=(mode!=1 || !router.forward_pending_q) &&
                        (mode!=2 || cycle%5>=2);
    assign response_gate=(mode!=2 || delay_left==0);
    assign s_ready=r_ready[1] && request_gate;
    assign s_rsp=r_rsp[1] && response_gate;
    assign s_rsp_tag=r_rsp_tag[8+:8] ^ ((fault==1 && responses==2) ? 8'd1 : 8'd0);
    assign s_rsp_code=(fault==3 && responses==2) ? 16'sd16 :
                      (fault==4 && responses==2) ? 16'sd128 : $signed(r_rsp_code[16+:16]);
    assign s_rsp_error=r_rsp_error[1];
    // Actual shared-store range fault, not a fabricated error-only response.
    assign router_addr=(fault==2 && reads==2) ? 10'd600 : s_addr;
    assign fill_ready=(mode!=2 || cycle%7>=3) ? 4'hf : 4'h0;

    @@PUBLISHER@@
    @@SLICE@@
    @@ROUTER@@
    @@STORE@@

    function automatic logic signed [15:0] value_at(input integer f,input integer addr);
        case(f)
            0: return 16'((addr*7)%32-16);
            1: return 16'((addr*37)%256-128);
            default: return 16'(addr*7919);
        endcase
    endfunction

    task automatic step;
        @(posedge clk); #1; @(negedge clk);
    endtask

    always @(posedge clk) begin
        cycle=cycle+1;
        if (!rst_n) begin
            credit=0; delay_left=0;
            request_held=0; response_held=0; fill_held=0;
        end else begin
            // Check the actual cache and router slots, not merely a TB queue.
            if (slice.cache_active_q !== (credit!=0)) $fatal(1,"cache ownership lost");
            if (router.forward_pending_q !== (credit!=0)) $fatal(1,"router credit lost");
            if (credit && router.forward_owner_q!=2) $fatal(1,"response routed to wrong owner");
            if (r_rsp[0] || r_rsp[2]) $fatal(1,"response escaped SRNN owner");
            if (unexpected_learning_read || unexpected_learning_write) $fatal(1,"unexpected learner access");
            if (model_version || adapt_version || forward_model_version || forward_adapt_version)
                $fatal(1,"refill changed version boundary");
            if (c_req && c_ready && !(b_req && b_ready)) $fatal(1,"unbacked cache acceptance");
            if ((b_req && b_ready) != (c_req && c_ready)) $fatal(1,"backend request accounting");
            if ((b_rsp && b_take) != (c_rsp && c_take)) $fatal(1,"backend response accounting");
            if (request_held && !abort_pub && array_idle && !manual) begin
                if (!p_req || {p_addr,p_tag}!==saved_request) $fatal(1,"read changed under backpressure");
                held_request_checks++;
            end
            request_held=!manual && p_req && !p_ready && !abort_pub && array_idle;
            saved_request={p_addr,p_tag};
            if (response_held) begin
                if (!c_rsp || {c_rsp_tag,c_rsp_code,c_rsp_error}!==saved_response)
                    $fatal(1,"response changed under backpressure");
                held_response_checks++;
            end
            response_held=c_rsp && !c_take;
            saved_response={c_rsp_tag,c_rsp_code,c_rsp_error};
            if (fill_held && !abort_pub && array_idle) begin
                if ({fill_valid,fill_tile,fill_bank,fill_addr,fill_data}!==saved_fill)
                    $fatal(1,"row changed under backpressure");
                held_fill_checks++;
            end
            fill_held=(|(fill_valid & ~fill_ready));
            saved_fill={fill_valid,fill_tile,fill_bank,fill_addr,fill_data};
            if (p_req && !p_ready) read_stalls++;
            if (b_rsp && !b_take) response_stalls++;
            if (|(fill_valid & ~fill_ready)) fill_stalls++;
            if (c_rsp && c_take) begin
                if (credit!=1) $fatal(1,"unsolicited response");
                if (checking && fault==0) begin
                    if (c_rsp_tag!==8'(responses) ||
                        c_rsp_code!==value_at(int'(fmt),int'(base)+responses*int'(stride)) || c_rsp_error)
                        $fatal(1,"scalar response mismatch");
                end
                credit--; responses++;
            end
            if (c_req && c_ready) begin
                if (checking && (c_addr!==10'(int'(base)+reads*int'(stride)) || c_tag!==8'(reads)))
                    $fatal(1,"next address/tag mismatch got=%0d/%0d index=%0d",c_addr,c_tag,reads);
                if (c_rsp && c_take) replacements++;
                credit++; reads++;
                delay_left=(mode==2) ? 3 : 0;
            end else if (delay_left>0) delay_left--;
            if (credit<0 || credit>1) $fatal(1,"more than one credit");
            if (credit>max_credit) max_credit=credit;
            if (fault!=0 && c_rsp && c_take && responses==3 && p_req)
                $fatal(1,"speculative read on fault/abort");
            for (integer region=0; region<4; region++) begin
                if (fill_valid[region] && fill_ready[region]) begin
                    if (fault!=0) $fatal(1,"fault published a row");
                    if (region!=int'(tile[3:2]) || fill_tile[region*2+:2]!==tile[1:0] ||
                        fill_bank[region]!==bank[0] || fill_addr[region*11+:11]!=={bank[1],row} ||
                        fill_data[region*200+:200]!==expected_row || responses!=expected_count)
                        $fatal(1,"atomic row/metadata/packing mismatch");
                    fills++;
                end
            end
        end
    end

    task automatic counters;
        reads=0;responses=0;fills=0;max_credit=0;replacements=0;
        read_stalls=0;response_stalls=0;fill_stalls=0;
    endtask

    task automatic reset_all;
        rst_n=0; checking=0; cmd_valid=0; cfg_valid=0; load_valid=0;
        forward_start=0;forward_end=0;manual=0;m_req=0;m_take=0;
        fault=0;abort_pub=0;array_idle=1;clear_error=0;owner=2;owner_active=1;
        repeat(3) step();
        counters(); rst_n=1; step();
    endtask

    task automatic setup(input integer f,input integer n,input integer service,input integer variant);
        reset_all();
        mode=service; fmt=2'(f); count=6'(n); expected_count=n;
        base=10'(5+variant);stride=10'(1+variant);tile=4'(f*5+variant);
        bank=2'(variant);row=10'(577-variant);version=16'hff00+16'(f*4+variant);
        expected_row='0;
        for(integer j=0;j<n;j++) begin
            load_addr=10'(int'(base)+j*int'(stride));load_code=value_at(f,int'(load_addr));load_valid=1;
            limit=0;
            while(!load_ready) begin step(); if(limit++>100) $fatal(1,"load timeout"); end
            step();load_valid=0;
            case(f)
                0: expected_row[j*5+:5]=load_code[4:0];
                1: expected_row[j*8+:8]=load_code[7:0];
                2: expected_row[j*16+:16]=load_code;
            endcase
        end
        cfg_valid=1;limit=0;
        while(!cfg_ready) begin step();if(limit++>100) $fatal(1,"cfg timeout");end
        step();cfg_valid=0;step();
        if(cfg_error) $fatal(1,"legal config rejected");
        forward_start=1;limit=0;
        while(!forward_ready) begin step();if(limit++>100) $fatal(1,"start timeout");end
        step();forward_start=0;step();
        if(!forward_end_ready) $fatal(1,"random forward boundary not opened");
        counters();checking=1;
    endtask

    task automatic launch_row;
        start_cycle=cycle;cmd_valid=1;limit=0;
        while(!cmd_ready) begin step();if(limit++>100) $fatal(1,"publisher command blocked");end
        step();cmd_valid=0;
    endtask

    task automatic finish_row;
        limit=0;
        while(!pub_done) begin step();if(limit++>1000) $fatal(1,"row timeout reads=%0d rsp=%0d",reads,responses);end
        if(pub_error || slice_error || router_error || store_error || fills!=1 ||
           reads!=expected_count || responses!=expected_count || credit!=0)
            $fatal(1,"row terminal accounting");
        if(published_tile!==tile || published_bank!==bank || published_row!==row || published_version!==version)
            $fatal(1,"publication metadata");
    endtask

    initial begin
        if(!$value$plusargs("arm=%d",arm)) $fatal(1,"arm required");
        @(negedge clk);
        for(integer service=0;service<3;service++) begin
            for(integer f=0;f<3;f++) begin
                for(integer v=0;v<4;v++) begin
                    automatic integer cap=(f==0)?40:((f==1)?25:12);
                    automatic integer n=(v==0)?cap:((v==1)?1:((v==2)?cap-1:7));
                    setup(f,n,service,v);launch_row();finish_row();
                    if(service==0 && replacements!=((arm==1)?n-1:0)) $fatal(1,"L1 replacements");
                    if(service==1 && replacements!=0) $fatal(1,"nonturnover service accepted overlap");
                    $display("M2_REFILL_RESULT arm=%0d mode=%0d format=%0d variant=%0d count=%0d cycles=%0d reads=%0d responses=%0d fills=%0d replacements=%0d max_credit=%0d read_stalls=%0d response_stalls=%0d fill_stalls=%0d",arm,service,f,v,n,cycle-start_cycle,reads,responses,fills,replacements,max_credit,read_stalls,response_stalls,fill_stalls);
                    forward_end=1;step();forward_end=0;repeat(2) step();
                    if(slice_busy || !cfg_ready) $fatal(1,"forward release did not quiesce");
                end
            end
        end
        // Faults retire or explicitly drain the sole pending response before recovery.
        for(integer id=1;id<=8;id++) begin
            setup((id==4)?1:0,7,(id>=7)?2:0,2);
            fault=id;launch_row();limit=0;
            if(id<=4) begin
                while(!pub_error) begin step();if(limit++>200) $fatal(1,"fault not caught");end
                repeat(2) step();
                if(reads!=3 || responses!=3 || fills!=0 || credit!=0 || pub_busy)
                    $fatal(1,"fault retirement did not stop row");
                if(id==2 && (!store_error || !router_error || !forward_error)) $fatal(1,"range fault not propagated");
            end else if(id<=6) begin
                while(!(c_rsp && responses==2)) begin step();if(limit++>200) $fatal(1,"abort target timeout");end
                if(id==5) abort_pub=1; else array_idle=0;
                step();abort_pub=0;array_idle=1;repeat(2) step();
                if(reads!=3 || responses!=3 || credit!=0 || fills || pub_busy) $fatal(1,"response-boundary abort");
                if(id==6 && !pub_error) $fatal(1,"unsafe refill not sticky");
            end else begin
                while(!(reads==3 && credit==1 && !s_rsp)) begin step();if(limit++>300) $fatal(1,"inflight target timeout");end
                if(id==7) begin
                    rst_n=0;step();rst_n=1;step();
                    if(c_rsp || pub_busy || fills || slice.cache_active_q || router.forward_pending_q)
                        $fatal(1,"inflight reset retained transaction");
                    setup(0,7,0,2);
                end else begin
                    abort_pub=1;step();abort_pub=0;step();
                    if(fills || pub_busy || credit!=1 || !slice.cache_active_q)
                        $fatal(1,"early abort must retain backend credit until drain");
                    // Publisher abort is not a store cancel. Explicitly drain the
                    // real slice response before reopening publisher service.
                    manual=1;m_take=1;limit=0;
                    while(credit!=0) begin step();if(limit++>100) $fatal(1,"abort drain timeout");end
                    m_take=0;manual=0;
                end
            end
            fault=0;clear_error=1;repeat(3) step();clear_error=0;repeat(2) step();
            if(pub_error || slice_error || router_error || store_error) $fatal(1,"idle clear failed");
            counters();checking=1;launch_row();finish_row();
            $display("M2_REFILL_FAULT id=%0d pass=1",id);
        end

        // Held cache response, replacement ownership and cfg/release boundaries.
        setup(0,7,2,0);checking=0;manual=1;m_req=1;m_addr=base;m_tag=8'h55;m_take=0;
        limit=0;
        while(!c_ready) begin step();if(limit++>100) $fatal(1,"manual request timeout");end
        step();m_req=0;
        limit=0;
        while(!c_rsp) begin step();if(limit++>100) $fatal(1,"manual response timeout");end
        cfg_valid=1;
        repeat(4) begin
            if(cfg_ready || forward_ready || learning_ready || c_rsp_tag!=8'h55 || credit!=1)
                $fatal(1,"pending cache boundary");
            step();
        end
        cfg_valid=0;
        // An illegal owner switch must be reported without rerouting old data.
        owner=3;step();
        if(!router_error || !c_rsp || c_rsp_tag!=8'h55) $fatal(1,"owner protection");
        owner=2;clear_error=1;repeat(3) step();clear_error=0;step();
        mode=0;m_req=1;m_addr=base+1'b1;m_tag=8'h56;m_take=1;#1;
        if(c_ready !== (arm==1)) $fatal(1,"cache slot replacement eligibility");
        step();
        if(arm==0) begin
            if(!c_ready) $fatal(1,"original normal request not ready");
            step();
        end
        m_req=0;m_take=0;
        repeat(3) begin
            if(!c_rsp || c_rsp_tag!=8'h56 || credit!=1) $fatal(1,"replacement response ownership");
            step();
        end
        m_take=1;step();m_take=0;
        if(credit!=0 || !router_quiet) $fatal(1,"cache did not drain");
        manual=0;forward_end=1;step();forward_end=0;repeat(2) step();
        if(!cfg_ready || slice_busy) $fatal(1,"boundary release failed");
        if(held_request_checks==0 || held_response_checks==0 || held_fill_checks==0)
            $fatal(1,"holding coverage absent");
        $display("M2_REFILL_CACHE held=1 owner=1 boundary=1 reset=1 pass=1");
        $display("M2_REFILL_HOLD request=%0d response=%0d fill=%0d",held_request_checks,held_response_checks,held_fill_checks);
        $display("M2_REFILL_PASS");
        $finish;
    end
    initial begin #10000000; $fatal(1,"global timeout"); end
endmodule
`default_nettype wire
