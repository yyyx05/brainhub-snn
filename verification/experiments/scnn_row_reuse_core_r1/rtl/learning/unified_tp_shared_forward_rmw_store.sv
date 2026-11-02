`timescale 1ns/1ps
`default_nettype none

// Shared persistent parameter storage for inference and direct TP learning.
// Inference reads the Q2.14 code field directly; learning reads and updates the
// same code plus its private residual field.  Train/infer serialization is an
// architectural contract, so any concurrent forward and learning access is
// reported instead of relying on device-specific read-during-write behavior.
module unified_tp_shared_forward_rmw_store #(
    parameter int DEPTH = 120320,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16
) (
    input  wire logic                                      i_clk,
    input  wire logic                                      i_rst_n,
    input  wire logic                                      i_clear_error,

    input  wire logic                                      i_load_valid,
    output logic                                      o_load_ready,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]               i_load_addr,
    input  wire logic signed [CODE_WIDTH-1:0]              i_load_code,
    input  wire logic signed [RESIDUAL_WIDTH-1:0]          i_load_residual,

    input  wire logic                                      i_learning_read_valid,
    output logic                                      o_learning_read_ready,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]               i_learning_read_addr,
    input  wire logic [TAG_WIDTH-1:0]                      i_learning_read_tag,
    output logic                                      o_learning_response_valid,
    input  wire logic                                      i_learning_response_ready,
    output logic [TAG_WIDTH-1:0]                      o_learning_response_tag,
    output logic signed [CODE_WIDTH-1:0]              o_learning_response_code,
    output logic signed [RESIDUAL_WIDTH-1:0]          o_learning_response_residual,
    output logic                                      o_learning_response_error,

    input  wire logic                                      i_learning_write_valid,
    output logic                                      o_learning_write_ready,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]               i_learning_write_addr,
    input  wire logic signed [CODE_WIDTH-1:0]              i_learning_write_code,
    input  wire logic signed [RESIDUAL_WIDTH-1:0]          i_learning_write_residual,
    input  wire logic                                      i_learning_write_code_enable,
    input  wire logic                                      i_learning_write_residual_enable,

    input  wire logic                                      i_forward_read_valid,
    output logic                                      o_forward_read_ready,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]               i_forward_read_addr,
    input  wire logic [TAG_WIDTH-1:0]                      i_forward_read_tag,
    output logic                                      o_forward_response_valid,
    input  wire logic                                      i_forward_response_ready,
    output logic [TAG_WIDTH-1:0]                      o_forward_response_tag,
    output logic signed [CODE_WIDTH-1:0]              o_forward_response_code,
    output logic                                      o_forward_response_error,

    output logic                                      o_protocol_error
);

    localparam logic [PARAM_ADDR_WIDTH:0] DEPTH_EXT =
        (PARAM_ADDR_WIDTH + 1)'(DEPTH);
    localparam int STORE_INDEX_WIDTH = $clog2(DEPTH);

    (* ram_style = "block" *) logic signed [CODE_WIDTH-1:0]
        code_mem [0:DEPTH-1];
    (* ram_style = "block" *) logic signed [RESIDUAL_WIDTH-1:0]
        residual_mem [0:DEPTH-1];

    logic response_valid_q;
    logic response_is_learning_q;
    logic [TAG_WIDTH-1:0] response_tag_q;
    logic signed [CODE_WIDTH-1:0] response_code_q;
    logic signed [RESIDUAL_WIDTH-1:0] response_residual_q;
    logic response_error_q;

    logic load_addr_legal;
    logic learning_read_addr_legal;
    logic learning_write_addr_legal;
    logic forward_read_addr_legal;
    logic load_fire;
    logic learning_read_fire;
    logic learning_write_fire;
    logic forward_read_fire;
    logic response_ready;
    logic read_fire;
    logic read_is_learning;
    logic read_addr_legal;
    logic [PARAM_ADDR_WIDTH-1:0] read_addr;
    logic [STORE_INDEX_WIDTH-1:0] read_index;
    logic write_fire;
    logic [PARAM_ADDR_WIDTH-1:0] write_addr;
    logic [STORE_INDEX_WIDTH-1:0] write_index;
    logic signed [CODE_WIDTH-1:0] write_code;
    logic signed [RESIDUAL_WIDTH-1:0] write_residual;
    logic write_code_enable;
    logic write_residual_enable;

    always_comb begin
        load_addr_legal = {1'b0, i_load_addr} < DEPTH_EXT;
        learning_read_addr_legal =
            {1'b0, i_learning_read_addr} < DEPTH_EXT;
        learning_write_addr_legal =
            {1'b0, i_learning_write_addr} < DEPTH_EXT;
        forward_read_addr_legal =
            {1'b0, i_forward_read_addr} < DEPTH_EXT;

        response_ready = response_is_learning_q ?
            i_learning_response_ready : i_forward_response_ready;
        o_learning_read_ready = !response_valid_q || response_ready;
        // Learning and forward accesses share one physical BRAM read port.
        // Learning wins an illegal simultaneous request; the sticky error
        // still records the train/infer contract violation.
        o_forward_read_ready = (!response_valid_q || response_ready) &&
            !i_learning_read_valid;
        o_learning_write_ready = 1'b1;
        o_load_ready = !i_learning_read_valid &&
            !i_learning_write_valid && !i_forward_read_valid &&
            !response_valid_q;

        load_fire = i_load_valid && o_load_ready;
        learning_read_fire = i_learning_read_valid &&
            o_learning_read_ready;
        learning_write_fire = i_learning_write_valid &&
            o_learning_write_ready;
        forward_read_fire = i_forward_read_valid && o_forward_read_ready;

        read_fire = learning_read_fire || forward_read_fire;
        read_is_learning = learning_read_fire;
        read_addr = read_is_learning ? i_learning_read_addr :
            i_forward_read_addr;
        read_addr_legal = read_is_learning ? learning_read_addr_legal :
            forward_read_addr_legal;
        read_index = STORE_INDEX_WIDTH'(read_addr);

        write_fire = load_fire || learning_write_fire;
        if (load_fire) begin
            write_addr = i_load_addr;
            write_code = i_load_code;
            write_residual = i_load_residual;
            write_code_enable = load_addr_legal;
            write_residual_enable = load_addr_legal;
        end else begin
            write_addr = i_learning_write_addr;
            write_code = i_learning_write_code;
            write_residual = i_learning_write_residual;
            write_code_enable = learning_write_addr_legal &&
                i_learning_write_code_enable;
            write_residual_enable = learning_write_addr_legal &&
                i_learning_write_residual_enable;
        end
        write_index = STORE_INDEX_WIDTH'(write_addr);

        o_learning_response_valid = response_valid_q &&
            response_is_learning_q;
        o_learning_response_tag = response_tag_q;
        o_learning_response_code = response_code_q;
        o_learning_response_residual = response_residual_q;
        o_learning_response_error = response_error_q;
        o_forward_response_valid = response_valid_q &&
            !response_is_learning_q;
        o_forward_response_tag = response_tag_q;
        o_forward_response_code = response_code_q;
        o_forward_response_error = response_error_q;
    end

    // A single read and a single write expression per array matches a
    // simple-dual-port block RAM. Payload state intentionally has no reset;
    // validity and range checking below determine whether it is observable.
    always_ff @(posedge i_clk) begin
        if (write_fire) begin
            if (write_code_enable)
                code_mem[write_index] <= write_code;
            if (write_residual_enable)
                residual_mem[write_index] <= write_residual;
        end

        if (read_fire) begin
            if (read_addr_legal) begin
                response_code_q <= code_mem[read_index];
                response_residual_q <= residual_mem[read_index];
            end else begin
                response_code_q <= '0;
                response_residual_q <= '0;
            end
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            response_valid_q <= 1'b0;
            response_is_learning_q <= 1'b0;
            response_tag_q <= '0;
            response_error_q <= 1'b0;
            o_protocol_error <= 1'b0;
        end else begin
            if (i_clear_error)
                o_protocol_error <= 1'b0;

            if (response_valid_q && response_ready)
                response_valid_q <= 1'b0;

            if (load_fire) begin
                if (!load_addr_legal)
                    o_protocol_error <= 1'b1;
            end

            if (read_fire) begin
                response_valid_q <= 1'b1;
                response_is_learning_q <= read_is_learning;
                response_tag_q <= read_is_learning ?
                    i_learning_read_tag : i_forward_read_tag;
                response_error_q <= !read_addr_legal;
                if (!read_addr_legal)
                    o_protocol_error <= 1'b1;
            end

            if (learning_write_fire) begin
                if (!learning_write_addr_legal)
                    o_protocol_error <= 1'b1;
            end

            if ((i_learning_read_valid && i_forward_read_valid &&
                 (!response_valid_q || response_ready)) ||
                (learning_write_fire && read_fire))
                o_protocol_error <= 1'b1;
        end
    end

endmodule

`default_nettype wire
