`default_nettype none

// Exact Q8.8 implementation of the type-1 TP surrogate derivative:
//   1 / (1 + (pi * abs(membrane - threshold))^2)
// The retained fixed-point formula truncates to zero above delta code 1301,
// so one shared 1302 x 9 ROM replaces a large run-time divider.
module unified_scnn_tp_surrogate_type1_q8_8 #(
    parameter ROM_FILE =
        "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem"
) (
    input  wire logic               i_clk,
    input  wire logic               i_rst_n,
    input  wire logic               i_abort,
    input  wire logic               i_valid,
    output logic               o_ready,
    input  wire logic signed [15:0] i_membrane_pre_q8_8,
    input  wire logic signed [15:0] i_threshold_q8_8,
    output logic               o_valid,
    input  wire logic               i_ready,
    output logic        [15:0] o_surrogate_q8_8
);
    localparam int ROM_DEPTH = 1302;
    localparam int ROM_ADDR_WIDTH = 11;

    (* rom_style = "block" *) logic [8:0] surrogate_rom [0:ROM_DEPTH-1];
    logic signed [16:0] delta;
    logic [16:0] absolute_delta;
    logic delta_in_rom;
    logic valid_q;
    logic [8:0] result_q;

    initial begin
        $readmemh(ROM_FILE, surrogate_rom);
    end

    always_comb begin
        delta = {i_membrane_pre_q8_8[15], i_membrane_pre_q8_8} -
                {i_threshold_q8_8[15], i_threshold_q8_8};
        if (delta[16])
            absolute_delta = 17'(-delta);
        else
            absolute_delta = 17'(delta);
        delta_in_rom = absolute_delta < 17'(ROM_DEPTH);
    end

    assign o_ready = !valid_q || i_ready;
    assign o_valid = valid_q;
    assign o_surrogate_q8_8 = {7'd0, result_q};

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            valid_q <= 1'b0;
            result_q <= '0;
        end else if (i_abort) begin
            valid_q <= 1'b0;
            result_q <= '0;
        end else if (o_ready) begin
            valid_q <= i_valid;
            if (i_valid) begin
                if (delta_in_rom)
                    result_q <= surrogate_rom[
                        absolute_delta[ROM_ADDR_WIDTH-1:0]];
                else
                    result_q <= '0;
            end
        end
    end
endmodule

`default_nettype wire
