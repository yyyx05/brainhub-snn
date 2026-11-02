`timescale 1ns/1ps
`default_nettype none

// Clears the physical H40 membrane/trace work rows at an SRNN sequence
// boundary.  The caller supplies the logical row ranges produced by the M2
// forward adapter.  One maintenance ownership interval covers the complete
// operation; state and trace writes are issued separately so either memory
// port may apply backpressure without accepting a record twice.
module unified_srnn_m2_h40_state_clear_sequencer_v1 #(
    parameter int ARRAY_TILES = 12,
    parameter int MEMORY_DEPTH = 1024,
    parameter int TILE_INDEX_WIDTH = $clog2(ARRAY_TILES),
    parameter int MEMORY_ADDR_WIDTH = $clog2(MEMORY_DEPTH)
) (
    input  wire logic                               i_clk,
    input  wire logic                               i_rst_n,
    input  wire logic                               i_abort,
    input  wire logic                               i_clear_error,

    input  wire logic                               i_clear_valid,
    output logic                                    o_clear_ready,
    input  wire logic [ARRAY_TILES-1:0]             i_tile_mask,
    input  wire logic [TILE_INDEX_WIDTH-1:0]        i_output_tile,
    input  wire logic [MEMORY_ADDR_WIDTH-2:0]       i_hidden_base,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]       i_hidden_row_count,
    input  wire logic [MEMORY_ADDR_WIDTH-2:0]       i_output_base,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]       i_output_row_count,

    output logic                                    o_maintenance_start_valid,
    input  wire logic                               i_maintenance_start_ready,
    output logic                                    o_maintenance_release_valid,
    input  wire logic                               i_maintenance_release_ready,

    output logic [TILE_INDEX_WIDTH-1:0]             o_maintenance_tile,
    output logic                                    o_state_write_valid,
    input  wire logic                               i_state_write_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]            o_state_write_addr,
    output logic [319:0]                            o_state_write_data,
    output logic [39:0]                             o_state_write_mask,
    output logic                                    o_trace_write_valid,
    input  wire logic                               i_trace_write_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]            o_trace_write_addr,
    output logic [319:0]                            o_trace_write_data,
    output logic [39:0]                             o_trace_write_mask,

    output logic                                    o_done,
    output logic                                    o_busy,
    output logic                                    o_sticky_error
);
    localparam int HALF_DEPTH = MEMORY_DEPTH / 2;

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_ACQUIRE,
        STATE_SELECT_TILE,
        STATE_WRITE_STATE,
        STATE_WRITE_TRACE,
        STATE_RELEASE,
        STATE_COMPLETE
    } state_t;

    state_t state_q;
    logic [ARRAY_TILES-1:0] tile_mask_q;
    logic [TILE_INDEX_WIDTH-1:0] output_tile_q;
    logic [MEMORY_ADDR_WIDTH-2:0] hidden_base_q;
    logic [MEMORY_ADDR_WIDTH-1:0] hidden_rows_q;
    logic [MEMORY_ADDR_WIDTH-2:0] output_base_q;
    logic [MEMORY_ADDR_WIDTH-1:0] output_rows_q;
    logic [TILE_INDEX_WIDTH:0] tile_q;
    logic [MEMORY_ADDR_WIDTH-1:0] row_q;

    logic command_legal;
    logic [MEMORY_ADDR_WIDTH:0] hidden_end;
    logic [MEMORY_ADDR_WIDTH:0] output_end;
    logic selected_tile;
    logic selected_output_tile;
    logic [MEMORY_ADDR_WIDTH-2:0] selected_base;
    logic [MEMORY_ADDR_WIDTH-1:0] selected_rows;
    logic [MEMORY_ADDR_WIDTH-2:0] logical_row;

    always_comb begin
        hidden_end = (MEMORY_ADDR_WIDTH + 1)'(i_hidden_base) +
            (MEMORY_ADDR_WIDTH + 1)'(i_hidden_row_count);
        output_end = (MEMORY_ADDR_WIDTH + 1)'(i_output_base) +
            (MEMORY_ADDR_WIDTH + 1)'(i_output_row_count);
        command_legal = i_tile_mask != '0 &&
            int'(i_output_tile) < ARRAY_TILES &&
            i_tile_mask[i_output_tile] &&
            i_hidden_row_count != '0 && i_output_row_count != '0 &&
            hidden_end <= (MEMORY_ADDR_WIDTH + 1)'(HALF_DEPTH) &&
            output_end <= (MEMORY_ADDR_WIDTH + 1)'(HALF_DEPTH);

        selected_tile = int'(tile_q) < ARRAY_TILES &&
            tile_mask_q[tile_q[TILE_INDEX_WIDTH-1:0]];
        selected_output_tile =
            tile_q[TILE_INDEX_WIDTH-1:0] == output_tile_q;
        selected_base = selected_output_tile ? output_base_q : hidden_base_q;
        selected_rows = selected_output_tile ? output_rows_q : hidden_rows_q;
        logical_row = selected_base +
            (MEMORY_ADDR_WIDTH - 1)'(row_q);

        o_clear_ready = state_q == STATE_IDLE && !o_sticky_error;
        o_maintenance_start_valid = state_q == STATE_ACQUIRE;
        o_maintenance_release_valid = state_q == STATE_RELEASE;
        o_maintenance_tile = tile_q[TILE_INDEX_WIDTH-1:0];

        o_state_write_valid = state_q == STATE_WRITE_STATE;
        o_state_write_addr = {1'b0, logical_row};
        o_state_write_data = '0;
        o_state_write_mask = '1;
        o_trace_write_valid = state_q == STATE_WRITE_TRACE;
        o_trace_write_addr = {1'b1, logical_row};
        o_trace_write_data = '0;
        o_trace_write_mask = '1;
        o_busy = state_q != STATE_IDLE && state_q != STATE_COMPLETE;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            tile_mask_q <= '0;
            output_tile_q <= '0;
            hidden_base_q <= '0;
            hidden_rows_q <= '0;
            output_base_q <= '0;
            output_rows_q <= '0;
            tile_q <= '0;
            row_q <= '0;
            o_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            tile_q <= '0;
            row_q <= '0;
            o_done <= 1'b0;
        end else begin
            o_done <= 1'b0;

            if (i_clear_error && state_q == STATE_IDLE)
                o_sticky_error <= 1'b0;

            unique case (state_q)
                STATE_IDLE: begin
                    if (i_clear_valid && o_clear_ready) begin
                        if (!command_legal) begin
                            o_sticky_error <= 1'b1;
                        end else begin
                            tile_mask_q <= i_tile_mask;
                            output_tile_q <= i_output_tile;
                            hidden_base_q <= i_hidden_base;
                            hidden_rows_q <= i_hidden_row_count;
                            output_base_q <= i_output_base;
                            output_rows_q <= i_output_row_count;
                            tile_q <= '0;
                            row_q <= '0;
                            state_q <= STATE_ACQUIRE;
                        end
                    end
                end

                STATE_ACQUIRE: begin
                    if (i_maintenance_start_ready)
                        state_q <= STATE_SELECT_TILE;
                end

                STATE_SELECT_TILE: begin
                    row_q <= '0;
                    if (int'(tile_q) >= ARRAY_TILES) begin
                        state_q <= STATE_RELEASE;
                    end else if (selected_tile) begin
                        state_q <= STATE_WRITE_STATE;
                    end else begin
                        tile_q <= tile_q + 1'b1;
                    end
                end

                STATE_WRITE_STATE: begin
                    if (i_state_write_ready)
                        state_q <= STATE_WRITE_TRACE;
                end

                STATE_WRITE_TRACE: begin
                    if (i_trace_write_ready) begin
                        if (row_q + 1'b1 >= selected_rows) begin
                            tile_q <= tile_q + 1'b1;
                            row_q <= '0;
                            state_q <= STATE_SELECT_TILE;
                        end else begin
                            row_q <= row_q + 1'b1;
                            state_q <= STATE_WRITE_STATE;
                        end
                    end
                end

                STATE_RELEASE: begin
                    if (i_maintenance_release_ready) begin
                        state_q <= STATE_COMPLETE;
                        o_done <= 1'b1;
                    end
                end

                // The caller presents a level request until it observes done.
                // Keep completion asserted until that request is withdrawn so
                // the same clear cannot be accepted again at the boundary.
                STATE_COMPLETE: begin
                    o_done <= 1'b1;
                    if (!i_clear_valid) begin
                        state_q <= STATE_IDLE;
                        o_done <= 1'b0;
                    end
                end

                default: begin
                    state_q <= STATE_IDLE;
                    o_sticky_error <= 1'b1;
                end
            endcase
        end
    end

    initial begin
        if (ARRAY_TILES < 2 || MEMORY_DEPTH < 4 ||
            (MEMORY_DEPTH % 2) != 0 ||
            TILE_INDEX_WIDTH < $clog2(ARRAY_TILES))
            $error("M2 H40 state clear sequencer has invalid geometry");
    end
endmodule

`default_nettype wire
