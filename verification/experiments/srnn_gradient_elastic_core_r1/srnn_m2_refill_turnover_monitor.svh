    longint m2_turn_reads = 0, m2_turn_responses = 0, m2_turn_pairs = 0;
    integer m2_turn_credit = 0, m2_turn_max_credit = 0;
    wire m2_turn_read = u_final.link_system_to_plane_srnn_h40_forward_read_valid &&
        u_final.link_plane_to_system_srnn_h40_forward_read_ready;
    wire m2_turn_response = u_final.link_plane_to_system_srnn_h40_forward_response_valid &&
        u_final.link_system_to_plane_srnn_h40_forward_response_ready;
    always @(posedge i_clk) begin
        if (!i_rst_n) begin
            m2_turn_reads = 0;
            m2_turn_responses = 0;
            m2_turn_pairs = 0;
            m2_turn_credit = 0;
            m2_turn_max_credit = 0;
        end else begin
            if (m2_turn_response && m2_turn_credit != 1)
                $fatal(1,"M2 refill response without prior credit");
            if (m2_turn_read && m2_turn_credit != 0 && !m2_turn_response)
                $fatal(1,"M2 refill second outstanding read");
            if (m2_turn_response) begin
                m2_turn_responses++;
                m2_turn_credit--;
            end
            if (m2_turn_read) begin
                m2_turn_reads++;
                m2_turn_credit++;
            end
            if (m2_turn_read && m2_turn_response) m2_turn_pairs++;
            if (m2_turn_credit > m2_turn_max_credit) m2_turn_max_credit = m2_turn_credit;
            if (m2_turn_credit < 0 || m2_turn_credit > 1)
                $fatal(1,"M2 refill credit range");
        end
    end
