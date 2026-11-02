    integer streaming=0;
    logic [7:0] raw_counts[0:T*700-1];
    logic feed_active=0, count_valid=0;
    logic [31:0] count_word=0;
    wire count_ready, producer_valid, producer_first, producer_last;
    wire [3:0] producer_mask;
    wire [9:0] producer_row;
    wire [2:0] producer_phase;
    logic manual_group_valid=0, manual_group_recurrent=0;
    logic [3:0] manual_group_source_valid=0;
    logic [10:0] manual_group_weight_addr0=0, manual_group_weight_addr1=0;
    logic [10:0] manual_group_weight_addr2=0, manual_group_weight_addr3=0;
    longint source_words=0, source_groups=0, source_tails=0;
    longint source_count_stalls=0, source_group_stalls=0;
    source_scan_experiment raw_source_i(
        .clk(i_clk),.rst_n(i_rst_n),.sparse(1'b1),.streaming(streaming==1),
        .count_valid(count_valid),.count_ready(count_ready),.counts(count_word),
        .group_valid(producer_valid),.group_ready(feed_active && o_fc_srnn_group_ready),
        .first(producer_first),.last(producer_last),.mask(producer_mask),.row(producer_row),
        .result_fire(o_fc_srnn_layer_done),.phase(producer_phase));
    always_comb begin
        i_fc_srnn_group_valid=feed_active ? producer_valid : manual_group_valid;
        i_fc_srnn_group_recurrent=feed_active ? 1'b0 : manual_group_recurrent;
        i_fc_srnn_group_source_valid=feed_active ? producer_mask : manual_group_source_valid;
        i_fc_srnn_group_weight_addr0=feed_active ? {1'b0,producer_row} : manual_group_weight_addr0;
        i_fc_srnn_group_weight_addr1=feed_active ? {1'b0,producer_row} : manual_group_weight_addr1;
        i_fc_srnn_group_weight_addr2=feed_active ? {1'b0,producer_row}+11'd1 : manual_group_weight_addr2;
        i_fc_srnn_group_weight_addr3=feed_active ? {1'b0,producer_row}+11'd1 : manual_group_weight_addr3;
    end
    always @(posedge i_clk) if(i_rst_n) begin
        if(count_valid && count_ready)source_words<=source_words+1;
        if(count_valid && !count_ready)source_count_stalls<=source_count_stalls+1;
        if(feed_active && producer_valid) begin
            if(o_fc_srnn_group_ready)begin
                source_groups<=source_groups+1;
                if(producer_last)source_tails<=source_tails+1;
            end else source_group_stalls<=source_group_stalls+1;
        end
    end
    task automatic feed_raw_frame;
        longint start_cycle, old_words, old_groups, old_tails;
        start_cycle=cycles;old_words=source_words;old_groups=source_groups;old_tails=source_tails;
        @(negedge i_clk);feed_active=1;count_valid=1;
        for(integer b=0;b<175;b++)begin
            for(integer l=0;l<4;l++)count_word[l*8+:8]=raw_counts[timestep*700+b*4+l];
            do @(posedge i_clk);while(!count_ready);
            @(negedge i_clk);
        end
        count_valid=0;
        while(source_tails==old_tails)@(negedge i_clk);
        feed_active=0;
        $display("SOURCE_CORE_FRAME sequence=%0d timestep=%0d cycles=%0d words=%0d groups=%0d tails=%0d",
            sequence_index,timestep,cycles-start_cycle,source_words-old_words,source_groups-old_groups,source_tails-old_tails);
    endtask
