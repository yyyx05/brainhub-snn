`timescale 1ns/1ps
`default_nettype none

// Two-entry ready/valid queue with a registered occupancy boundary. Input
// ready never depends on downstream ready, so the queue cuts reverse timing
// paths while retaining one transfer per cycle. Payload is reset-free; count
// and pointers are the sole ownership authority.
module unified_elastic_fifo2 #(
    parameter int DATA_WIDTH = 1
) (
    input  wire logic                  i_clk,
    input  wire logic                  i_rst_n,
    input  wire logic                  i_abort,
    input  wire logic                  i_valid,
    output logic                       o_ready,
    input  wire logic [DATA_WIDTH-1:0] i_data,
    output logic                       o_valid,
    input  wire logic                  i_ready,
    output logic [DATA_WIDTH-1:0]      o_data
);
    // Keep the two payload slots explicit and select the current head with a
    // one-bit read pointer. A shift-style output/skid implementation makes a
    // simultaneous pop/push choose which wide register to write from
    // downstream ready, reconnecting the reverse ready path to every payload
    // clock enable. Ring pointers keep that dependency on two control bits;
    // payload writes depend only on the local enqueue handshake.
    (* shreg_extract = "no" *) logic [DATA_WIDTH-1:0] slot0_q;
    (* shreg_extract = "no" *) logic [DATA_WIDTH-1:0] slot1_q;
    logic [1:0] count_q;
    logic read_slot_q;
    logic write_slot_q;
    logic input_fire;
    logic output_fire;

    assign o_ready = !i_abort && count_q != 2;
    assign o_valid = !i_abort && count_q != 0;
    assign o_data = read_slot_q ? slot1_q : slot0_q;
    assign input_fire = i_valid && o_ready;
    assign output_fire = o_valid && i_ready;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n || i_abort) begin
            count_q <= '0;
            read_slot_q <= 1'b0;
            write_slot_q <= 1'b0;
        end else begin
            case ({input_fire, output_fire})
                2'b10: count_q <= count_q + 1'b1;
                2'b01: count_q <= count_q - 1'b1;
                default: count_q <= count_q;
            endcase

            if (input_fire)
                write_slot_q <= ~write_slot_q;
            if (output_fire)
                read_slot_q <= ~read_slot_q;
        end
    end

    // Payload registers are reset-free; count_q is the sole validity state.
    // At occupancy one, simultaneous pop/push writes the empty slot while the
    // read pointer advances to it, sustaining one record per cycle without a
    // downstream-ready dependency on either wide payload register enable.
    always_ff @(posedge i_clk) begin
        if (input_fire) begin
            if (write_slot_q)
                slot1_q <= i_data;
            else
                slot0_q <= i_data;
        end
    end

    initial begin
        if (DATA_WIDTH < 1)
            $error("elastic fifo requires a positive payload width");
    end
endmodule

`default_nettype wire
