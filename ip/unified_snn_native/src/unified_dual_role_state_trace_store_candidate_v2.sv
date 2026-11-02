`timescale 1ns/1ps
`default_nettype none

// Masked-write successor to the shared state/trace working-set RAM. Port A is
// reserved for membrane state and port B for trace. Eight-bit write enables
// let FC/SRNN update selected 32-bit lanes and SCNN update selected 16-bit
// half-row lanes without retaining and merging the untouched 320-bit row in
// flip-flops outside the RAM.
module unified_dual_role_state_trace_store_candidate_v2 #(
    parameter int WORD_WIDTH = 320,
    parameter int DEPTH = 1024,
    parameter int ADDR_WIDTH = $clog2(DEPTH),
    parameter int WRITE_MASK_WIDTH = WORD_WIDTH / 8
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_abort,

    input  wire logic                         i_state_write_valid,
    output logic                              o_state_write_ready,
    input  wire logic [ADDR_WIDTH-1:0]        i_state_write_addr,
    input  wire logic [WORD_WIDTH-1:0]        i_state_write_data,
    input  wire logic [WRITE_MASK_WIDTH-1:0]  i_state_write_mask,
    input  wire logic                         i_state_read_valid,
    output logic                              o_state_read_ready,
    input  wire logic [ADDR_WIDTH-1:0]        i_state_read_addr,
    output logic                              o_state_read_result_valid,
    input  wire logic                         i_state_read_result_ready,
    output logic [ADDR_WIDTH-1:0]             o_state_read_result_addr,
    output logic [WORD_WIDTH-1:0]             o_state_read_data,

    input  wire logic                         i_trace_write_valid,
    output logic                              o_trace_write_ready,
    input  wire logic [ADDR_WIDTH-1:0]        i_trace_write_addr,
    input  wire logic [WORD_WIDTH-1:0]        i_trace_write_data,
    input  wire logic [WRITE_MASK_WIDTH-1:0]  i_trace_write_mask,
    input  wire logic                         i_trace_read_valid,
    output logic                              o_trace_read_ready,
    input  wire logic [ADDR_WIDTH-1:0]        i_trace_read_addr,
    output logic                              o_trace_read_result_valid,
    input  wire logic                         i_trace_read_result_ready,
    output logic [ADDR_WIDTH-1:0]             o_trace_read_result_addr,
    output logic [WORD_WIDTH-1:0]             o_trace_read_data
);
    logic state_read_result_valid_q;
    logic trace_read_result_valid_q;
    logic [ADDR_WIDTH-1:0] state_read_result_addr_q;
    logic [ADDR_WIDTH-1:0] trace_read_result_addr_q;
    logic [WORD_WIDTH-1:0] state_memory_read_data;
    logic [WORD_WIDTH-1:0] trace_memory_read_data;
    logic state_write_fire;
    logic state_read_fire;
    logic trace_write_fire;
    logic trace_read_fire;

    assign o_state_write_ready = !i_abort;
    assign o_state_read_ready = !i_abort && !i_state_write_valid &&
        (!state_read_result_valid_q || i_state_read_result_ready);
    assign o_trace_write_ready = !i_abort;
    assign o_trace_read_ready = !i_abort && !i_trace_write_valid &&
        (!trace_read_result_valid_q || i_trace_read_result_ready);

    assign state_write_fire = i_state_write_valid && o_state_write_ready;
    assign state_read_fire = i_state_read_valid && o_state_read_ready;
    assign trace_write_fire = i_trace_write_valid && o_trace_write_ready;
    assign trace_read_fire = i_trace_read_valid && o_trace_read_ready;
    assign o_state_read_result_valid = !i_abort &&
        state_read_result_valid_q;
    assign o_state_read_result_addr = state_read_result_addr_q;
    assign o_state_read_data = state_memory_read_data;
    assign o_trace_read_result_valid = !i_abort &&
        trace_read_result_valid_q;
    assign o_trace_read_result_addr = trace_read_result_addr_q;
    assign o_trace_read_data = trace_memory_read_data;

`ifdef UNIFIED_USE_XPM_MEMORY
    logic [ADDR_WIDTH-1:0] state_port_addr;
    logic [ADDR_WIDTH-1:0] trace_port_addr;
    logic [WRITE_MASK_WIDTH-1:0] state_write_enable;
    logic [WRITE_MASK_WIDTH-1:0] trace_write_enable;

    assign state_port_addr = state_write_fire ?
        i_state_write_addr : i_state_read_addr;
    assign trace_port_addr = trace_write_fire ?
        i_trace_write_addr : i_trace_read_addr;
    assign state_write_enable = i_state_write_mask &
        {WRITE_MASK_WIDTH{state_write_fire}};
    assign trace_write_enable = i_trace_write_mask &
        {WRITE_MASK_WIDTH{trace_write_fire}};

    xpm_memory_tdpram #(
        .MEMORY_SIZE(DEPTH * WORD_WIDTH),
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
        .WRITE_DATA_WIDTH_A(WORD_WIDTH),
        .READ_DATA_WIDTH_A(WORD_WIDTH),
        .BYTE_WRITE_WIDTH_A(8),
        .ADDR_WIDTH_A(ADDR_WIDTH),
        .READ_RESET_VALUE_A("0"),
        .READ_LATENCY_A(1),
        .WRITE_MODE_A("read_first"),
        .RST_MODE_A("SYNC"),
        .WRITE_DATA_WIDTH_B(WORD_WIDTH),
        .READ_DATA_WIDTH_B(WORD_WIDTH),
        .BYTE_WRITE_WIDTH_B(8),
        .ADDR_WIDTH_B(ADDR_WIDTH),
        .READ_RESET_VALUE_B("0"),
        .READ_LATENCY_B(1),
        .WRITE_MODE_B("read_first"),
        .RST_MODE_B("SYNC")
    ) u_state_trace_memory (
        .sleep(1'b0),
        .clka(i_clk), .rsta(1'b0),
        .ena(state_write_fire || state_read_fire), .regcea(1'b1),
        .wea(state_write_enable), .addra(state_port_addr),
        .dina(i_state_write_data),
        .injectsbiterra(1'b0), .injectdbiterra(1'b0),
        .douta(state_memory_read_data), .sbiterra(), .dbiterra(),
        .clkb(i_clk), .rstb(1'b0),
        .enb(trace_write_fire || trace_read_fire), .regceb(1'b1),
        .web(trace_write_enable), .addrb(trace_port_addr),
        .dinb(i_trace_write_data),
        .injectsbiterrb(1'b0), .injectdbiterrb(1'b0),
        .doutb(trace_memory_read_data), .sbiterrb(), .dbiterrb()
    );
`else
    logic [WORD_WIDTH-1:0] memory [0:DEPTH-1];

    always_ff @(posedge i_clk) begin
        if (state_write_fire) begin
            for (int byte_index = 0;
                 byte_index < WRITE_MASK_WIDTH; byte_index++) begin
                if (i_state_write_mask[byte_index])
                    memory[i_state_write_addr][byte_index*8 +: 8] <=
                        i_state_write_data[byte_index*8 +: 8];
            end
        end
        if (state_read_fire)
            state_memory_read_data <= memory[i_state_read_addr];
        if (trace_write_fire) begin
            for (int byte_index = 0;
                 byte_index < WRITE_MASK_WIDTH; byte_index++) begin
                if (i_trace_write_mask[byte_index])
                    memory[i_trace_write_addr][byte_index*8 +: 8] <=
                        i_trace_write_data[byte_index*8 +: 8];
            end
        end
        if (trace_read_fire)
            trace_memory_read_data <= memory[i_trace_read_addr];
    end
`endif

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n || i_abort) begin
            state_read_result_valid_q <= 1'b0;
            trace_read_result_valid_q <= 1'b0;
            state_read_result_addr_q <= '0;
            trace_read_result_addr_q <= '0;
        end else begin
            if (state_read_result_valid_q && i_state_read_result_ready)
                state_read_result_valid_q <= 1'b0;
            if (state_read_fire) begin
                state_read_result_valid_q <= 1'b1;
                state_read_result_addr_q <= i_state_read_addr;
            end

            if (trace_read_result_valid_q && i_trace_read_result_ready)
                trace_read_result_valid_q <= 1'b0;
            if (trace_read_fire) begin
                trace_read_result_valid_q <= 1'b1;
                trace_read_result_addr_q <= i_trace_read_addr;
            end
        end
    end

    initial begin
        if (WORD_WIDTH != 320 || DEPTH != 1024 || WRITE_MASK_WIDTH != 40)
            $error("masked state/trace store requires 320x1024 with 8-bit masks");
    end
endmodule

`default_nettype wire
