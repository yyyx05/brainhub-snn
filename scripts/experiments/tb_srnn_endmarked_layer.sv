`timescale 1ns/1ps
`default_nettype none
module tb_srnn_endmarked_layer;
    logic clk=0, rst_n=0;
    always #5 clk=~clk;
    longint cycles=0;
    always @(posedge clk) cycles <= cycles+1;
    logic layer_valid=0, group_valid=0, recurrent=0, ff_end=0, streamed=0;
    logic clear_error=0, abort_layer=0, array_done=0, check_enable=0, negative=0;
    logic owner_ready=0, array_ready=0;
    logic [1:0] mode=1;
    logic [15:0] nf=0, nr=0;
    logic [3:0] mask=0;
    logic [10:0] row=0;
    wire [1:0] layer_ready, group_ready, owner_request, command_valid;
    wire [1:0] layer_done, busy, idle, error;
    wire [1:0][7:0] error_code;
    typedef struct packed {
        logic [1:0] mode;
        logic int8_mode, shift6;
        logic [15:0] neurons;
        logic first, last;
        logic [3:0] mask;
        logic [10:0] a0,a1,a2,a3;
        logic [15:0] leak;
        logic [31:0] threshold;
        logic reset_zero;
        logic [15:0] decay;
        logic [51:0] tag;
        logic [8:0] work_base;
        logic [3:0] tile_base;
    } command_t;
    command_t command[2];
    integer accepted=0, stalls=0, firsts=0, lasts=0;
    always @(negedge clk) begin
        owner_ready=(cycles%7>=2);
        array_ready=(cycles%11>=4);
    end
    always @(posedge clk) if(rst_n && check_enable) begin
        if ({layer_ready[0],group_ready[0],owner_request[0],command_valid[0],layer_done[0],busy[0],idle[0],error[0]} !==
            {layer_ready[1],group_ready[1],owner_request[1],command_valid[1],layer_done[1],busy[1],idle[1],error[1]})
            $fatal(1,"public control mismatch cycle=%0d",cycles);
        if(error!=0) $fatal(1,"unexpected error codes=%h",error_code);
        if(command_valid[0]) begin
            if(command[0] !== command[1]) $fatal(1,"full command payload mismatch cycle=%0d",cycles);
            if(array_ready) begin
                accepted=accepted+1;
                if(command[0].first) firsts=firsts+1;
                if(command[0].last) lasts=lasts+1;
            end else stalls=stalls+1;
        end
    end

`define PORTS \
        .i_clk(clk),.i_rst_n(rst_n),.i_abort(abort_layer),.i_clear_error(clear_error), \
        .i_layer_valid(layer_valid && (!negative || p==1)),.o_layer_ready(layer_ready[p]), \
        .o_owner_request_valid(owner_request[p]),.i_owner_request_ready(owner_ready), \
        .i_mode(mode),.i_fc_weight_int8(1'b0),.i_fc_current_shift6(1'b0), \
        .i_neuron_count(16'd450),.i_feedforward_group_count((p==1 && streamed)?16'd0:nf), \
        .i_recurrent_group_count(nr),.i_recurrent_enable(nr!=0), \
        .i_leak(16'd27853),.i_threshold(32'd32),.i_reset_to_zero(1'b0), \
        .i_trace_decay(16'd27853),.i_transaction_tag(52'h12345678), \
        .i_work_base_addr(9'd256),.i_destination_tile_base(4'd0), \
        .i_group_valid(group_valid && (!negative || p==1)),.o_group_ready(group_ready[p]), \
        .i_group_recurrent(recurrent),.i_group_source_valid(mask), \
        .i_group_weight_addr0(row),.i_group_weight_addr1(row), \
        .i_group_weight_addr2(row+11'd1),.i_group_weight_addr3(row+11'd1), \
        .o_array_command_valid(command_valid[p]),.i_array_command_ready(array_ready), \
        .o_array_command_mode(command[p].mode),.o_array_command_fc_weight_int8(command[p].int8_mode), \
        .o_array_command_fc_current_shift6(command[p].shift6),.o_array_command_neuron_count(command[p].neurons), \
        .o_array_command_first(command[p].first),.o_array_command_last(command[p].last), \
        .o_array_command_source_valid(command[p].mask), \
        .o_array_command_weight_addr0(command[p].a0),.o_array_command_weight_addr1(command[p].a1), \
        .o_array_command_weight_addr2(command[p].a2),.o_array_command_weight_addr3(command[p].a3), \
        .o_array_command_leak(command[p].leak),.o_array_command_threshold(command[p].threshold), \
        .o_array_command_reset_to_zero(command[p].reset_zero),.o_array_command_trace_decay(command[p].decay), \
        .o_array_command_transaction_tag(command[p].tag),.o_array_command_work_base_addr(command[p].work_base), \
        .o_array_command_destination_tile_base(command[p].tile_base), \
        .i_array_transaction_done(array_done && (!negative || p==1)),.i_array_sticky_error(1'b0), \
        .o_layer_done(layer_done[p]),.o_busy(busy[p]),.o_idle(idle[p]), \
        .o_sticky_error(error[p]),.o_error_code(error_code[p])
    for(genvar p=0;p<2;p++) begin:g_clients
        if(p==0) begin:g_counted
            counted_reference original_i(`PORTS);
        end else begin:g_endmarked
            unified_fc_srnn_h40_p4_layer_client_v1 candidate_i(
                .i_streamed_feedforward(streamed),.i_group_feedforward_end(ff_end),`PORTS);
        end
    end
`undef PORTS

    task automatic start_layer;
        @(negedge clk);layer_valid=1;
        do @(posedge clk);while(!layer_ready[1]);
        @(negedge clk);layer_valid=0;
    endtask
    task automatic send_packet(input integer m,r,rec,ending);
        @(negedge clk);
        mask=4'(m);row=11'(r);recurrent=1'(rec);ff_end=1'(ending);group_valid=1;
        do @(posedge clk);while(!group_ready[1]);
        @(negedge clk);group_valid=0;
    endtask
    integer headers, packets, rc, start, nfi,nri,modei,streami;
    integer mi,ri,reci,endi,case_id=0;
    initial begin
        headers=$fopen("headers.txt","r");packets=$fopen("packets.txt","r");
        if(!headers || !packets)$fatal(1,"fixture missing");
        repeat(5)@(negedge clk);rst_n=1;
        rc=$fscanf(headers,"%d %d %d %d %d\n",start,nfi,nri,modei,streami);
        while(rc==5)begin
            nf=16'(nfi);nr=16'(nri);mode=2'(modei);streamed=1'(streami);
            accepted=0;stalls=0;firsts=0;lasts=0;check_enable=1;
            start_layer();
            for(integer k=0;k<nfi+nri;k++)begin
                rc=$fscanf(packets,"%d %d %d %d\n",mi,ri,reci,endi);
                if(rc!=4)$fatal(1,"missing packet");
                send_packet(mi,ri,reci,endi);
            end
            if(accepted!=nfi+nri || firsts!=1 || lasts!=1 || busy!=3)
                $fatal(1,"coverage first/last/count");
            repeat(4)@(negedge clk);
            array_done=1;@(negedge clk);array_done=0;
            if(layer_done!=3)$fatal(1,"drain done");
            @(negedge clk);
            $display("ENDMARKED_CASE case=%0d commands=%0d stalls=%0d",case_id,accepted,stalls);
            check_enable=0;case_id=case_id+1;
            rc=$fscanf(headers,"%d %d %d %d %d\n",start,nfi,nri,modei,streami);
        end
        // Reject a payload-bearing end, an early recurrent packet, an end on
        // the recurrent suffix, and an FC descriptor requesting stream mode.
        negative=1;streamed=1;nf=1;nr=1;mode=1;
        for(integer n=0;n<4;n++)begin
            mode=(n==3)?0:1;
            start_layer();
            if(n==0)send_packet(1,0,0,1);
            if(n==1)send_packet(0,350,1,0);
            if(n==2)begin send_packet(0,0,0,1);send_packet(0,350,1,1);end
            if(!error[1] || error_code[1] != ((n==1)?8'h02:(n==3)?8'h01:8'h06))
                $fatal(1,"negative case=%0d code=%h",n,error_code[1]);
            clear_error=1;@(negedge clk);clear_error=0;
            @(negedge clk);if(error[1])$fatal(1,"clear failed");
        end
        $display("ENDMARKED_PASS negatives=4");$finish;
    end
    initial begin #20000000;$fatal(1,"timeout");end
endmodule
`default_nettype wire
