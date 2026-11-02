`timescale 1ns/1ps
`default_nettype none

// Four shallow true-dual-port banks keep the same nominal payload capacity as
// the two deep P2 banks while exposing four topology contributions per cycle.
// FC stripes INT5 H40 or INT8 (up to 25 destinations) rows across banks.
// Precision changes packing only; it does not select another arithmetic mode.
// SRNN stripes source parity
// across bank pairs and destination lanes within each pair. SCNN uses one bank
// pair for four scalar INT16 reads while the other pair remains refillable.
// M2_Q14 combines all four banks at one row into one 40-lane signed Q2.14
// vector. It intentionally consumes one source per cycle so the trainable M2
// head can reuse this store without narrowing its authoritative parameters.
module unified_dual_role_weight_store_p4all_candidate_v1 #(
    parameter int LANES = 40,
    parameter int BANK_WIDTH = 200,
    parameter int BANK_DEPTH = 578,
    parameter int ADDR_WIDTH = $clog2(BANK_DEPTH)
) (
    input  wire logic                    i_clk,
    input  wire logic                    i_rst_n,

    input  wire logic                    i_fill_valid,
    output logic                         o_fill_ready,
    input  wire logic [1:0]              i_fill_bank,
    input  wire logic [ADDR_WIDTH-1:0]   i_fill_addr,
    input  wire logic [BANK_WIDTH-1:0]   i_fill_data,

    input  wire logic                    i_read_valid,
    output logic                         o_read_ready,
    input  wire logic [1:0]              i_mode,
    input  wire logic                    i_fc_weight_int8,
    input  wire logic                    i_scnn_active_pair,
    input  wire logic [ADDR_WIDTH-1:0]   i_read_addr0,
    input  wire logic [ADDR_WIDTH-1:0]   i_read_addr1,
    input  wire logic [ADDR_WIDTH-1:0]   i_read_addr2,
    input  wire logic [ADDR_WIDTH-1:0]   i_read_addr3,
    input  wire logic [3:0]              i_read_subword0,
    input  wire logic [3:0]              i_read_subword1,
    input  wire logic [3:0]              i_read_subword2,
    input  wire logic [3:0]              i_read_subword3,
    output logic                         o_read_result_valid,
    input  wire logic                    i_read_result_ready,
    output logic [LANES*25-1:0]          o_term0_flat,
    output logic [LANES*25-1:0]          o_term1_flat,
    output logic [LANES*25-1:0]          o_term2_flat,
    output logic [LANES*25-1:0]          o_term3_flat
);
    localparam logic [1:0] MODE_FC = 2'd0;
    localparam logic [1:0] MODE_SRNN = 2'd1;
    localparam logic [1:0] MODE_SCNN = 2'd2;
    localparam logic [1:0] MODE_M2_Q14 = 2'd3;

    logic [BANK_WIDTH-1:0] bank_read_data [0:3][0:1];
    logic [1:0] mode_q;
    logic fc_weight_int8_q;
    logic scnn_active_pair_q;
    logic [3:0] read_subword_q [0:3];
    logic read_result_valid_q;
    logic read_fire;
    logic fill_fire;
    logic [3:0] read_bank_enable;
    logic fill_conflict;

    function automatic logic signed [24:0] sign_extend5(
        input logic [4:0] value
    );
        sign_extend5 = {{20{value[4]}}, value};
    endfunction

    function automatic logic signed [24:0] sign_extend8(
        input logic [7:0] value
    );
        sign_extend8 = {{17{value[7]}}, value};
    endfunction

    function automatic logic signed [24:0] scnn_subword(
        input logic [BANK_WIDTH-1:0] word,
        input logic [3:0] index
    );
        logic signed [15:0] selected;
        begin
            selected = '0;
            if (index < 12)
                selected = $signed(word[index*16 +: 16]);
            scnn_subword = {{9{selected[15]}}, selected};
        end
    endfunction

    always_comb begin
        if (i_mode == MODE_SCNN)
            read_bank_enable = i_scnn_active_pair ? 4'b1100 : 4'b0011;
        else
            read_bank_enable = 4'b1111;
    end

    assign fill_conflict = i_read_valid &&
        read_bank_enable[i_fill_bank];
    assign o_read_ready = !read_result_valid_q || i_read_result_ready;
    assign o_fill_ready = !fill_conflict;
    assign read_fire = i_read_valid && o_read_ready;
    assign fill_fire = i_fill_valid && o_fill_ready;
    assign o_read_result_valid = read_result_valid_q;

    generate
        for (genvar bank = 0; bank < 4; bank++) begin : g_bank
            logic port_a_write;
            logic port_a_read;
            logic port_b_read;
            logic [ADDR_WIDTH-1:0] port_a_read_addr;
            logic [ADDR_WIDTH-1:0] port_b_read_addr;

            assign port_a_write = fill_fire && i_fill_bank == bank;
            assign port_a_read = read_fire && read_bank_enable[bank];
            assign port_b_read = read_fire && read_bank_enable[bank] &&
                (i_mode == MODE_SRNN || i_mode == MODE_SCNN);

            always_comb begin
                port_a_read_addr = '0;
                port_b_read_addr = '0;
                if (i_mode == MODE_FC) begin
                    case (bank)
                        0: port_a_read_addr = i_read_addr0;
                        1: port_a_read_addr = i_read_addr1;
                        2: port_a_read_addr = i_read_addr2;
                        default: port_a_read_addr = i_read_addr3;
                    endcase
                end else if (i_mode == MODE_M2_Q14) begin
                    port_a_read_addr = i_read_addr0;
                end else if (i_mode == MODE_SRNN) begin
                    if (bank < 2) begin
                        port_a_read_addr = i_read_addr0;
                        port_b_read_addr = i_read_addr2;
                    end else begin
                        port_a_read_addr = i_read_addr1;
                        port_b_read_addr = i_read_addr3;
                    end
                end else begin
                    if (bank[0] == 1'b0) begin
                        port_a_read_addr = i_read_addr0;
                        port_b_read_addr = i_read_addr2;
                    end else begin
                        port_a_read_addr = i_read_addr1;
                        port_b_read_addr = i_read_addr3;
                    end
                end
            end

`ifdef UNIFIED_USE_XPM_MEMORY
            logic [ADDR_WIDTH-1:0] port_a_addr;

            assign port_a_addr = port_a_write ? i_fill_addr :
                port_a_read_addr;

            xpm_memory_tdpram #(
                .MEMORY_SIZE(BANK_DEPTH * BANK_WIDTH),
                .MEMORY_PRIMITIVE("block"),
                .CLOCKING_MODE("common_clock"),
                .ECC_MODE("no_ecc"),
                .MEMORY_INIT_FILE("none"),
                .MEMORY_INIT_PARAM(""),
                .USE_MEM_INIT(0),
                .USE_MEM_INIT_MMI(0),
                .WAKEUP_TIME("disable_sleep"),
                .AUTO_SLEEP_TIME(0),
                .MESSAGE_CONTROL(0),
                .USE_EMBEDDED_CONSTRAINT(0),
                .MEMORY_OPTIMIZATION("true"),
                .CASCADE_HEIGHT(0),
                .SIM_ASSERT_CHK(0),
                .WRITE_PROTECT(1),
                .IGNORE_INIT_SYNTH(0),
                .WRITE_DATA_WIDTH_A(BANK_WIDTH),
                .READ_DATA_WIDTH_A(BANK_WIDTH),
                .BYTE_WRITE_WIDTH_A(BANK_WIDTH),
                .ADDR_WIDTH_A(ADDR_WIDTH),
                .READ_RESET_VALUE_A("0"),
                .READ_LATENCY_A(1),
                .WRITE_MODE_A("read_first"),
                .RST_MODE_A("SYNC"),
                .WRITE_DATA_WIDTH_B(BANK_WIDTH),
                .READ_DATA_WIDTH_B(BANK_WIDTH),
                .BYTE_WRITE_WIDTH_B(BANK_WIDTH),
                .ADDR_WIDTH_B(ADDR_WIDTH),
                .READ_RESET_VALUE_B("0"),
                .READ_LATENCY_B(1),
                .WRITE_MODE_B("read_first"),
                .RST_MODE_B("SYNC")
            ) u_weight_memory (
                .sleep(1'b0),
                .clka(i_clk), .rsta(1'b0),
                .ena(port_a_write || port_a_read), .regcea(1'b1),
                .wea(port_a_write), .addra(port_a_addr),
                .dina(i_fill_data),
                .injectsbiterra(1'b0), .injectdbiterra(1'b0),
                .douta(bank_read_data[bank][0]), .sbiterra(), .dbiterra(),
                .clkb(i_clk), .rstb(1'b0),
                .enb(port_b_read), .regceb(1'b1), .web(1'b0),
                .addrb(port_b_read_addr), .dinb('0),
                .injectsbiterrb(1'b0), .injectdbiterrb(1'b0),
                .doutb(bank_read_data[bank][1]), .sbiterrb(), .dbiterrb()
            );
`else
            logic [BANK_WIDTH-1:0] memory [0:BANK_DEPTH-1];

            always_ff @(posedge i_clk) begin
                if (port_a_write)
                    memory[i_fill_addr] <= i_fill_data;
                if (port_a_read)
                    bank_read_data[bank][0] <= memory[port_a_read_addr];
                if (port_b_read)
                    bank_read_data[bank][1] <= memory[port_b_read_addr];
            end
`endif
        end
    endgenerate

    always_comb begin
        o_term0_flat = '0;
        o_term1_flat = '0;
        o_term2_flat = '0;
        o_term3_flat = '0;
        for (integer lane = 0; lane < LANES; lane++) begin
            if (mode_q == MODE_FC && fc_weight_int8_q) begin
                if (lane < BANK_WIDTH / 8) begin
                    o_term0_flat[lane*25 +: 25] = sign_extend8(
                        bank_read_data[0][0][lane*8 +: 8]);
                    o_term1_flat[lane*25 +: 25] = sign_extend8(
                        bank_read_data[1][0][lane*8 +: 8]);
                    o_term2_flat[lane*25 +: 25] = sign_extend8(
                        bank_read_data[2][0][lane*8 +: 8]);
                    o_term3_flat[lane*25 +: 25] = sign_extend8(
                        bank_read_data[3][0][lane*8 +: 8]);
                end
            end else if (mode_q == MODE_FC) begin
                o_term0_flat[lane*25 +: 25] = sign_extend5(
                    bank_read_data[0][0][lane*5 +: 5]);
                o_term1_flat[lane*25 +: 25] = sign_extend5(
                    bank_read_data[1][0][lane*5 +: 5]);
                o_term2_flat[lane*25 +: 25] = sign_extend5(
                    bank_read_data[2][0][lane*5 +: 5]);
                o_term3_flat[lane*25 +: 25] = sign_extend5(
                    bank_read_data[3][0][lane*5 +: 5]);
            end else if (mode_q == MODE_SRNN) begin
                if (lane < 20) begin
                    o_term0_flat[lane*25 +: 25] = sign_extend8(
                        bank_read_data[0][0][lane*8 +: 8]);
                    o_term1_flat[lane*25 +: 25] = sign_extend8(
                        bank_read_data[2][0][lane*8 +: 8]);
                    o_term2_flat[lane*25 +: 25] = sign_extend8(
                        bank_read_data[0][1][lane*8 +: 8]);
                    o_term3_flat[lane*25 +: 25] = sign_extend8(
                        bank_read_data[2][1][lane*8 +: 8]);
                end else begin
                    o_term0_flat[lane*25 +: 25] = sign_extend8(
                        bank_read_data[1][0][(lane-20)*8 +: 8]);
                    o_term1_flat[lane*25 +: 25] = sign_extend8(
                        bank_read_data[3][0][(lane-20)*8 +: 8]);
                    o_term2_flat[lane*25 +: 25] = sign_extend8(
                        bank_read_data[1][1][(lane-20)*8 +: 8]);
                    o_term3_flat[lane*25 +: 25] = sign_extend8(
                        bank_read_data[3][1][(lane-20)*8 +: 8]);
                end
            end else if (mode_q == MODE_SCNN) begin
                o_term0_flat[lane*25 +: 25] = scnn_subword(
                    bank_read_data[scnn_active_pair_q*2][0],
                    read_subword_q[0]);
                o_term1_flat[lane*25 +: 25] = scnn_subword(
                    bank_read_data[scnn_active_pair_q*2 + 1][0],
                    read_subword_q[1]);
                o_term2_flat[lane*25 +: 25] = scnn_subword(
                    bank_read_data[scnn_active_pair_q*2][1],
                    read_subword_q[2]);
                o_term3_flat[lane*25 +: 25] = scnn_subword(
                    bank_read_data[scnn_active_pair_q*2 + 1][1],
                    read_subword_q[3]);
            end else if (mode_q == MODE_M2_Q14) begin
                if (lane < 12) begin
                    o_term0_flat[lane*25 +: 25] = scnn_subword(
                        bank_read_data[0][0], 4'(lane));
                end else if (lane < 24) begin
                    o_term0_flat[lane*25 +: 25] = scnn_subword(
                        bank_read_data[1][0], 4'(lane - 12));
                end else if (lane < 36) begin
                    o_term0_flat[lane*25 +: 25] = scnn_subword(
                        bank_read_data[2][0], 4'(lane - 24));
                end else begin
                    o_term0_flat[lane*25 +: 25] = scnn_subword(
                        bank_read_data[3][0], 4'(lane - 36));
                end
            end
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            mode_q <= MODE_FC;
            fc_weight_int8_q <= 1'b0;
            scnn_active_pair_q <= 1'b0;
            for (integer term = 0; term < 4; term++)
                read_subword_q[term] <= '0;
            read_result_valid_q <= 1'b0;
        end else begin
            if (read_result_valid_q && i_read_result_ready)
                read_result_valid_q <= 1'b0;
            if (read_fire) begin
                mode_q <= i_mode;
                fc_weight_int8_q <= i_fc_weight_int8 === 1'b1;
                scnn_active_pair_q <= i_scnn_active_pair;
                read_subword_q[0] <= i_read_subword0;
                read_subword_q[1] <= i_read_subword1;
                read_subword_q[2] <= i_read_subword2;
                read_subword_q[3] <= i_read_subword3;
                read_result_valid_q <= 1'b1;
            end
        end
    end

    initial begin
        if (LANES != 40 || BANK_WIDTH != 200 || BANK_DEPTH != 578)
            $error("P4-all weight store requires fixed 4x200x578 geometry");
    end
endmodule

`default_nettype wire
