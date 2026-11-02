`timescale 1ns/1ps
`default_nettype none

// Descriptor-driven FC/SRNN client for the common 16xH40/P4 substrate.
// FC and SRNN share the same command format and physical arithmetic. SRNN
// differs only by permitting a recurrent group suffix after the feedforward
// groups. Source groups are already banked by the ingress/graph scheduler, so
// each packet supplies four validity bits and four cache row addresses.
module unified_fc_srnn_h40_p4_layer_client_v1 #(
    parameter int ARRAY_TILES = 12,
    parameter int WEIGHT_DEPTH = 1156,
    parameter int WEIGHT_ADDR_WIDTH = $clog2(WEIGHT_DEPTH),
    parameter int WEIGHT_ROW_DEPTH = WEIGHT_DEPTH / 2,
    parameter int WORK_ADDR_WIDTH = 9,
    parameter int GROUP_COUNT_WIDTH = 16
) (
    input  wire logic                           i_clk,
    input  wire logic                           i_rst_n,
    input  wire logic                           i_abort,
    input  wire logic                           i_clear_error,

    input  wire logic                           i_layer_valid,
    output logic                                o_layer_ready,
    output logic                                o_owner_request_valid,
    input  wire logic                           i_owner_request_ready,
    input  wire logic [1:0]                     i_mode,
    input  wire logic                           i_fc_weight_int8,
    input  wire logic                           i_fc_current_shift6,
    input  wire logic [15:0]                    i_neuron_count,
    input  wire logic [GROUP_COUNT_WIDTH-1:0]   i_feedforward_group_count,
    input  wire logic [GROUP_COUNT_WIDTH-1:0]   i_recurrent_group_count,
    input  wire logic                           i_recurrent_enable,
    input  wire logic [15:0]                    i_leak,
    input  wire logic [31:0]                    i_threshold,
    input  wire logic                           i_reset_to_zero,
    input  wire logic [15:0]                    i_trace_decay,
    input  wire logic [51:0]                    i_transaction_tag,
    input  wire logic [WORK_ADDR_WIDTH-1:0]     i_work_base_addr,
    input  wire logic [3:0]                     i_destination_tile_base,

    input  wire logic                           i_group_valid,
    output logic                                o_group_ready,
    input  wire logic                           i_group_recurrent,
    input  wire logic [3:0]                     i_group_source_valid,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]   i_group_weight_addr0,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]   i_group_weight_addr1,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]   i_group_weight_addr2,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]   i_group_weight_addr3,

    output logic                                o_array_command_valid,
    input  wire logic                           i_array_command_ready,
    output logic [1:0]                          o_array_command_mode,
    output logic                                o_array_command_fc_weight_int8,
    output logic                                o_array_command_fc_current_shift6,
    output logic [15:0]                         o_array_command_neuron_count,
    output logic                                o_array_command_first,
    output logic                                o_array_command_last,
    output logic [3:0]                          o_array_command_source_valid,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_array_command_weight_addr0,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_array_command_weight_addr1,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_array_command_weight_addr2,
    output logic [WEIGHT_ADDR_WIDTH-1:0]        o_array_command_weight_addr3,
    output logic [15:0]                         o_array_command_leak,
    output logic [31:0]                         o_array_command_threshold,
    output logic                                o_array_command_reset_to_zero,
    output logic [15:0]                         o_array_command_trace_decay,
    output logic [51:0]                         o_array_command_transaction_tag,
    output logic [WORK_ADDR_WIDTH-1:0]          o_array_command_work_base_addr,
    output logic [3:0]                          o_array_command_destination_tile_base,
    input  wire logic                           i_array_transaction_done,
    input  wire logic                           i_array_sticky_error,

    output logic                                o_layer_done,
    output logic                                o_busy,
    output logic                                o_idle,
    output logic                                o_sticky_error,
    output logic [7:0]                          o_error_code
);
    localparam logic [1:0] MODE_FC = 2'd0;
    localparam logic [1:0] MODE_SRNN = 2'd1;
    localparam logic [1:0] MODE_M2_Q14 = 2'd3;
    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_DESCRIPTOR = 8'h01;
    localparam logic [7:0] ERROR_GROUP_KIND = 8'h02;
    localparam logic [7:0] ERROR_WEIGHT_ADDRESS = 8'h03;
    localparam logic [7:0] ERROR_ARRAY = 8'h04;
    localparam logic [7:0] ERROR_EARLY_ARRAY_DONE = 8'h05;

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_STREAM,
        STATE_DRAIN
    } state_t;

    state_t state_q;
    logic [1:0] mode_q;
    logic fc_weight_int8_q;
    logic fc_current_shift6_q;
    logic [15:0] neuron_count_q;
    logic [GROUP_COUNT_WIDTH-1:0] feedforward_group_count_q;
    logic [GROUP_COUNT_WIDTH-1:0] total_group_count_q;
    logic [GROUP_COUNT_WIDTH-1:0] issued_group_count_q;
    logic [15:0] leak_q;
    logic [31:0] threshold_q;
    logic reset_to_zero_q;
    logic [15:0] trace_decay_q;
    logic [51:0] transaction_tag_q;
    logic [WORK_ADDR_WIDTH-1:0] work_base_addr_q;
    logic [3:0] destination_tile_base_q;
    logic sticky_error_q;
    logic [7:0] error_code_q;

    logic descriptor_legal;
    logic fc_weight_int8;
    logic fc_current_shift6;
    logic [GROUP_COUNT_WIDTH:0] descriptor_total_groups;
    logic expected_recurrent;
    logic group_kind_legal;
    logic group_address_legal;
    logic group_legal;
    logic layer_fire;
    logic group_fire;
    logic command_fire;
    logic local_layer_ready;

    always_comb begin
        // Legacy instances may omit these inputs; only an explicit one opts in.
        fc_weight_int8 = i_fc_weight_int8 === 1'b1;
        fc_current_shift6 = i_fc_current_shift6 === 1'b1;
        descriptor_total_groups =
            {1'b0, i_feedforward_group_count} +
            ((i_mode == MODE_SRNN && i_recurrent_enable) ?
                {1'b0, i_recurrent_group_count} : '0);
        descriptor_legal =
            (i_mode == MODE_FC || i_mode == MODE_SRNN ||
             i_mode == MODE_M2_Q14) &&
            i_neuron_count != 0 && i_neuron_count <= 16'd640 &&
            (!fc_current_shift6 || i_mode == MODE_FC) &&
            (!fc_weight_int8 || (i_mode == MODE_FC &&
                i_neuron_count <= 16'd25 &&
                integer'(i_destination_tile_base) < ARRAY_TILES)) &&
            i_feedforward_group_count != 0 &&
            descriptor_total_groups <= {1'b0, {GROUP_COUNT_WIDTH{1'b1}}} &&
            ((i_mode == MODE_FC && !i_recurrent_enable) ||
             (i_mode == MODE_M2_Q14 && !i_recurrent_enable &&
              i_recurrent_group_count == 0) ||
             (i_mode == MODE_SRNN &&
              ((!i_recurrent_enable && i_recurrent_group_count == 0) ||
               (i_recurrent_enable && i_recurrent_group_count != 0))));

        expected_recurrent =
            issued_group_count_q >= feedforward_group_count_q;
        group_kind_legal = i_group_recurrent == expected_recurrent;
        group_address_legal =
            (!i_group_source_valid[0] ||
             integer'(i_group_weight_addr0) < WEIGHT_ROW_DEPTH) &&
            (!i_group_source_valid[1] ||
             integer'(i_group_weight_addr1) < WEIGHT_ROW_DEPTH) &&
            (!i_group_source_valid[2] ||
             integer'(i_group_weight_addr2) < WEIGHT_ROW_DEPTH) &&
            (!i_group_source_valid[3] ||
             integer'(i_group_weight_addr3) < WEIGHT_ROW_DEPTH);
        group_legal = group_kind_legal && group_address_legal;

        local_layer_ready = state_q == STATE_IDLE && !sticky_error_q &&
            !i_array_sticky_error;
        o_owner_request_valid = i_layer_valid && local_layer_ready &&
            descriptor_legal;
        // Malformed descriptors are consumed locally. Legal descriptors do
        // not cross the interface until the physical array is reserved.
        o_layer_ready = local_layer_ready &&
            (!descriptor_legal || i_owner_request_ready);
        layer_fire = i_layer_valid && o_layer_ready;

        o_array_command_valid = state_q == STATE_STREAM &&
            i_group_valid && group_legal;
        o_group_ready = state_q == STATE_STREAM &&
            (group_legal ? i_array_command_ready : 1'b1);
        group_fire = i_group_valid && o_group_ready;
        command_fire = o_array_command_valid && i_array_command_ready;

        o_array_command_mode = mode_q;
        o_array_command_fc_weight_int8 = fc_weight_int8_q;
        o_array_command_fc_current_shift6 = fc_current_shift6_q;
        o_array_command_neuron_count = neuron_count_q;
        o_array_command_first = issued_group_count_q == 0;
        o_array_command_last = issued_group_count_q + 1'b1 ==
            total_group_count_q;
        o_array_command_source_valid = i_group_source_valid;
        o_array_command_weight_addr0 = i_group_weight_addr0;
        o_array_command_weight_addr1 = i_group_weight_addr1;
        o_array_command_weight_addr2 = i_group_weight_addr2;
        o_array_command_weight_addr3 = i_group_weight_addr3;
        o_array_command_leak = leak_q;
        o_array_command_threshold = threshold_q;
        o_array_command_reset_to_zero = reset_to_zero_q;
        o_array_command_trace_decay = trace_decay_q;
        o_array_command_transaction_tag = transaction_tag_q;
        o_array_command_work_base_addr = work_base_addr_q;
        o_array_command_destination_tile_base = destination_tile_base_q;

        o_busy = state_q != STATE_IDLE;
        o_idle = state_q == STATE_IDLE;
        o_sticky_error = sticky_error_q || i_array_sticky_error;
        o_error_code = sticky_error_q ? error_code_q :
            i_array_sticky_error ? ERROR_ARRAY : ERROR_NONE;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            mode_q <= MODE_FC;
            fc_weight_int8_q <= 1'b0;
            fc_current_shift6_q <= 1'b0;
            neuron_count_q <= '0;
            feedforward_group_count_q <= '0;
            total_group_count_q <= '0;
            issued_group_count_q <= '0;
            leak_q <= '0;
            threshold_q <= '0;
            reset_to_zero_q <= 1'b0;
            trace_decay_q <= '0;
            transaction_tag_q <= '0;
            work_base_addr_q <= '0;
            destination_tile_base_q <= '0;
            sticky_error_q <= 1'b0;
            error_code_q <= ERROR_NONE;
            o_layer_done <= 1'b0;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            issued_group_count_q <= '0;
            o_layer_done <= 1'b0;
        end else begin
            o_layer_done <= 1'b0;

            if (i_clear_error && state_q == STATE_IDLE) begin
                sticky_error_q <= 1'b0;
                error_code_q <= ERROR_NONE;
            end

            if (i_array_sticky_error && !sticky_error_q) begin
                sticky_error_q <= 1'b1;
                error_code_q <= ERROR_ARRAY;
                state_q <= STATE_IDLE;
            end else begin
                case (state_q)
                    STATE_IDLE: begin
                        if (layer_fire) begin
                            if (!descriptor_legal) begin
                                sticky_error_q <= 1'b1;
                                error_code_q <= ERROR_DESCRIPTOR;
                            end else begin
                                mode_q <= i_mode;
                                fc_weight_int8_q <= fc_weight_int8;
                                fc_current_shift6_q <= fc_current_shift6;
                                neuron_count_q <= i_neuron_count;
                                feedforward_group_count_q <=
                                    i_feedforward_group_count;
                                total_group_count_q <=
                                    descriptor_total_groups[
                                        GROUP_COUNT_WIDTH-1:0];
                                issued_group_count_q <= '0;
                                leak_q <= i_leak;
                                threshold_q <= i_threshold;
                                reset_to_zero_q <= i_reset_to_zero;
                                trace_decay_q <= i_trace_decay;
                                transaction_tag_q <= i_transaction_tag;
                                work_base_addr_q <= i_work_base_addr;
                                destination_tile_base_q <=
                                    i_destination_tile_base;
                                state_q <= STATE_STREAM;
                            end
                        end
                        if (i_array_transaction_done && !sticky_error_q) begin
                            sticky_error_q <= 1'b1;
                            error_code_q <= ERROR_EARLY_ARRAY_DONE;
                        end
                    end
                    STATE_STREAM: begin
                        if (group_fire && !group_legal) begin
                            sticky_error_q <= 1'b1;
                            error_code_q <= group_kind_legal ?
                                ERROR_WEIGHT_ADDRESS : ERROR_GROUP_KIND;
                            state_q <= STATE_IDLE;
                        end else if (command_fire) begin
                            issued_group_count_q <=
                                issued_group_count_q + 1'b1;
                            if (o_array_command_last)
                                state_q <= STATE_DRAIN;
                        end
                        if (i_array_transaction_done && !sticky_error_q) begin
                            sticky_error_q <= 1'b1;
                            error_code_q <= ERROR_EARLY_ARRAY_DONE;
                            state_q <= STATE_IDLE;
                        end
                    end
                    STATE_DRAIN: begin
                        if (i_array_transaction_done) begin
                            o_layer_done <= 1'b1;
                            state_q <= STATE_IDLE;
                        end
                    end
                    default: state_q <= STATE_IDLE;
                endcase
            end
        end
    end

    initial begin
        if (ARRAY_TILES < 1 || ARRAY_TILES > 16 ||
            WEIGHT_DEPTH < 1 || WEIGHT_ROW_DEPTH < 1 ||
                WEIGHT_ROW_DEPTH > WEIGHT_DEPTH || WORK_ADDR_WIDTH < 1 ||
                GROUP_COUNT_WIDTH < 1)
            $error("FC/SRNN P4 layer client has invalid geometry");
    end
endmodule

`default_nettype wire
