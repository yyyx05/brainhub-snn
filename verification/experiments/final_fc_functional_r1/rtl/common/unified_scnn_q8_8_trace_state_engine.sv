`default_nettype none

module unified_scnn_q8_8_trace_state_engine #(
    parameter int NUM_NEURONS = 480,
    parameter int ADDR_WIDTH = 17,
    parameter int GENERATION_WIDTH = 8
) (
    input  wire logic             i_clk,
    input  wire logic             i_rst_n,
    input  wire logic             i_sample_reset,
    input  wire logic             i_valid,
    output logic                  o_ready,
    input  wire logic [ADDR_WIDTH-1:0] i_addr,
    input  wire logic signed [15:0]    i_spike_q8_8,
    input  wire logic signed [15:0]    i_leak_q8_8,
    output logic                  o_valid,
    input  wire logic             i_ready,
    output logic [ADDR_WIDTH-1:0] o_addr,
    output logic signed [15:0]    o_trace_q8_8,
    output logic                  o_idle,
    output logic                  o_reset_while_busy_error,
    output logic                  o_address_error
);

    localparam int MEMORY_WIDTH = 16 + GENERATION_WIDTH;

    logic [ADDR_WIDTH-1:0] transaction_addr_q;
    logic [GENERATION_WIDTH-1:0] transaction_generation_q;
    logic [GENERATION_WIDTH-1:0] current_generation_q;
    logic signed [15:0] spike_q;
    logic [MEMORY_WIDTH-1:0] trace_read_data_q;
    logic trace_read_pending_q;
    logic signed [15:0] selected_trace;
    logic address_in_range;
    logic transaction_busy_q;
    logic scrub_active_q;
    logic [ADDR_WIDTH-1:0] scrub_addr_q;
    logic multiplier_ready;
    logic multiplier_valid;
    logic multiplier_consume;
    logic signed [31:0] leak_product;
    logic product_stage_valid_q;
    logic signed [31:0] product_stage_q;
    logic signed [31:0] trace_sum_wide;
    logic result_valid_q;
    logic signed [15:0] result_trace_q;
    logic memory_write_enable;
    logic [ADDR_WIDTH-1:0] memory_write_addr;
    logic signed [15:0] memory_write_trace;
    logic [GENERATION_WIDTH-1:0] memory_write_generation;
    logic memory_read_enable;

    assign address_in_range = {1'b0, i_addr} < (ADDR_WIDTH+1)'(NUM_NEURONS);
    assign selected_trace =
        trace_read_data_q[MEMORY_WIDTH-1:16] == transaction_generation_q ?
            $signed(trace_read_data_q[15:0]) : 16'sd0;
    assign o_ready = multiplier_ready && !transaction_busy_q &&
                     !scrub_active_q && !i_sample_reset && address_in_range;
    wire result_stage_ready = !result_valid_q || i_ready;
    wire product_stage_fire = product_stage_valid_q && result_stage_ready;

    assign multiplier_consume = !product_stage_valid_q || result_stage_ready;
    assign trace_sum_wide = (product_stage_q >>> 8) +
        {{16{spike_q[15]}}, spike_q};
    assign memory_write_enable = scrub_active_q ||
        (result_valid_q && i_ready);
    assign memory_write_addr = scrub_active_q ? scrub_addr_q :
        transaction_addr_q;
    assign memory_write_trace = scrub_active_q ? 16'sd0 : result_trace_q;
    assign memory_write_generation = scrub_active_q ?
        GENERATION_WIDTH'(0) : transaction_generation_q;
    assign memory_read_enable = i_valid && o_ready;
    assign o_valid = result_valid_q;
    assign o_addr = transaction_addr_q;
    assign o_trace_q8_8 = result_trace_q;
    assign o_idle = !scrub_active_q && !transaction_busy_q &&
                    !result_valid_q && multiplier_ready;

    unified_q8_8_mul_multipass u_trace_leak_multiply (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_valid(trace_read_pending_q), .o_ready(multiplier_ready),
        .i_a(selected_trace), .i_b(i_leak_q8_8),
        .o_valid(multiplier_valid), .i_ready(multiplier_consume),
        .o_product(leak_product)
    );

    // Payload and generation use separate RAMs so their 16-bit and 8-bit widths
    // map efficiently onto 7-series block-RAM configurations. Sample reset
    // advances the generation in O(1); the rare wrap performs a sequential
    // scrub instead of synthesizing one resettable validity FF per address.
`ifdef UNIFIED_USE_XPM_MEMORY
    logic [15:0] xpm_trace_read_data;
    logic [GENERATION_WIDTH-1:0] xpm_generation_read_data;

    assign trace_read_data_q =
        {xpm_generation_read_data, xpm_trace_read_data};

    xpm_memory_sdpram #(
        .MEMORY_SIZE(NUM_NEURONS * 16),
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
        .WRITE_DATA_WIDTH_A(16),
        .BYTE_WRITE_WIDTH_A(16),
        .ADDR_WIDTH_A(ADDR_WIDTH),
        .RST_MODE_A("SYNC"),
        .READ_DATA_WIDTH_B(16),
        .ADDR_WIDTH_B(ADDR_WIDTH),
        .READ_RESET_VALUE_B("0"),
        .READ_LATENCY_B(1),
        .WRITE_MODE_B("read_first"),
        .RST_MODE_B("SYNC")
    ) u_trace_payload_memory (
        .sleep(1'b0),
        .clka(i_clk),
        .ena(memory_write_enable),
        .wea(memory_write_enable),
        .addra(memory_write_addr),
        .dina(memory_write_trace),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0),
        .clkb(i_clk),
        .rstb(1'b0),
        .enb(memory_read_enable),
        .regceb(1'b1),
        .addrb(i_addr),
        .doutb(xpm_trace_read_data),
        .sbiterrb(),
        .dbiterrb()
    );

    xpm_memory_sdpram #(
        .MEMORY_SIZE(NUM_NEURONS * GENERATION_WIDTH),
        .MEMORY_PRIMITIVE("block"),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("none"),
        .MEMORY_INIT_PARAM("0"),
        .USE_MEM_INIT(1),
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
        .WRITE_DATA_WIDTH_A(GENERATION_WIDTH),
        .BYTE_WRITE_WIDTH_A(GENERATION_WIDTH),
        .ADDR_WIDTH_A(ADDR_WIDTH),
        .RST_MODE_A("SYNC"),
        .READ_DATA_WIDTH_B(GENERATION_WIDTH),
        .ADDR_WIDTH_B(ADDR_WIDTH),
        .READ_RESET_VALUE_B("0"),
        .READ_LATENCY_B(1),
        .WRITE_MODE_B("read_first"),
        .RST_MODE_B("SYNC")
    ) u_trace_generation_memory (
        .sleep(1'b0),
        .clka(i_clk),
        .ena(memory_write_enable),
        .wea(memory_write_enable),
        .addra(memory_write_addr),
        .dina(memory_write_generation),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0),
        .clkb(i_clk),
        .rstb(1'b0),
        .enb(memory_read_enable),
        .regceb(1'b1),
        .addrb(i_addr),
        .doutb(xpm_generation_read_data),
        .sbiterrb(),
        .dbiterrb()
    );
`else
    (* ram_style = "block" *)
    logic [MEMORY_WIDTH-1:0] trace_memory [0:NUM_NEURONS-1];
    integer init_index;

    initial begin
        for (init_index = 0; init_index < NUM_NEURONS;
             init_index = init_index + 1)
            trace_memory[init_index] = '0;
    end

    always_ff @(posedge i_clk) begin
        if (memory_write_enable) begin
            trace_memory[memory_write_addr] <=
                {memory_write_generation, memory_write_trace};
        end
        if (memory_read_enable)
            trace_read_data_q <= trace_memory[i_addr];
    end
`endif

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            transaction_addr_q <= '0;
            transaction_generation_q <= GENERATION_WIDTH'(1);
            current_generation_q <= GENERATION_WIDTH'(1);
            spike_q <= 16'sd0;
            transaction_busy_q <= 1'b0;
            scrub_active_q <= 1'b0;
            scrub_addr_q <= '0;
            trace_read_pending_q <= 1'b0;
            product_stage_valid_q <= 1'b0;
            product_stage_q <= 32'sd0;
            result_valid_q <= 1'b0;
            result_trace_q <= 16'sd0;
            o_reset_while_busy_error <= 1'b0;
            o_address_error <= 1'b0;
        end else begin
            if (i_valid && !address_in_range)
                o_address_error <= 1'b1;

            if (i_valid && o_ready) begin
                transaction_addr_q <= i_addr;
                transaction_generation_q <= current_generation_q;
                spike_q <= i_spike_q8_8;
                transaction_busy_q <= 1'b1;
                trace_read_pending_q <= 1'b1;
            end

            if (trace_read_pending_q && multiplier_ready) begin
                trace_read_pending_q <= 1'b0;
            end

            if (product_stage_fire) begin
                result_trace_q <= trace_sum_wide[15:0];
                result_valid_q <= 1'b1;
                product_stage_valid_q <= 1'b0;
            end

            if (multiplier_valid && multiplier_consume) begin
                product_stage_q <= leak_product;
                product_stage_valid_q <= 1'b1;
            end

            if (result_valid_q && i_ready) begin
                result_valid_q <= 1'b0;
                transaction_busy_q <= 1'b0;
            end

            if (scrub_active_q) begin
                if ({1'b0, scrub_addr_q} ==
                    (ADDR_WIDTH+1)'(NUM_NEURONS - 1)) begin
                    scrub_active_q <= 1'b0;
                    scrub_addr_q <= '0;
                end else begin
                    scrub_addr_q <= scrub_addr_q + ADDR_WIDTH'(1);
                end
            end

            if (i_sample_reset) begin
                if (!o_idle) begin
                    o_reset_while_busy_error <= 1'b1;
                end else if (&current_generation_q) begin
                    current_generation_q <= GENERATION_WIDTH'(1);
                    scrub_active_q <= 1'b1;
                    scrub_addr_q <= '0;
                end else begin
                    current_generation_q <= current_generation_q +
                        GENERATION_WIDTH'(1);
                end
            end
        end
    end

endmodule

`default_nettype wire
