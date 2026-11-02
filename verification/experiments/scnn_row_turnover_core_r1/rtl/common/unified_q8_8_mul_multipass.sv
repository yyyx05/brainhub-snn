module unified_q8_8_mul_multipass (
    input  wire logic               i_clk,
    input  wire logic               i_rst_n,
    input  wire logic               i_valid,
    output logic               o_ready,
    input  wire logic signed [15:0] i_a,
    input  wire logic signed [15:0] i_b,
    output logic               o_valid,
    input  wire logic               i_ready,
    output logic signed [31:0] o_product
);

    logic               busy_q;
    logic [1:0]         phase_q;
    logic signed [7:0] a_hi_q;
    logic signed [7:0] b_hi_q;
    logic signed [7:0] a_lo7_q;
    logic signed [7:0] b_lo7_q;
    logic signed [31:0] correction_q;
    logic signed [31:0] accumulator_q;
    logic signed [31:0] product_q;
    logic               product_valid_q;

    logic signed [7:0]  mul_a;
    logic signed [7:0]  mul_b;
    logic signed [15:0] mul_product;
    logic signed [31:0] mul_extended;
    logic signed [31:0] shifted_term;
    logic signed [31:0] correction_in;
    logic signed [31:0] a_hi_extended;
    logic signed [31:0] b_hi_extended;
    logic signed [31:0] a_lo7_extended;
    logic signed [31:0] b_lo7_extended;

    always_comb begin
        a_hi_extended = {{24{i_a[15]}}, i_a[15:8]};
        b_hi_extended = {{24{i_b[15]}}, i_b[15:8]};
        a_lo7_extended = {25'd0, i_a[6:0]};
        b_lo7_extended = {25'd0, i_b[6:0]};
        correction_in = 32'sd0;
        if (i_b[7]) begin
            correction_in = correction_in +
                (a_hi_extended <<< 15) + (a_lo7_extended <<< 7);
        end
        if (i_a[7]) begin
            correction_in = correction_in +
                (b_hi_extended <<< 15) + (b_lo7_extended <<< 7);
        end
        if (i_a[7] && i_b[7]) begin
            correction_in = correction_in + 32'sd16384;
        end
    end

    always_comb begin
        mul_a = 8'sd0;
        mul_b = 8'sd0;
        shifted_term = 32'sd0;
        case (phase_q)
            2'd0: begin
                mul_a = a_hi_q;
                mul_b = b_hi_q;
            end
            2'd1: begin
                mul_a = a_hi_q;
                mul_b = b_lo7_q;
            end
            2'd2: begin
                mul_a = b_hi_q;
                mul_b = a_lo7_q;
            end
            default: begin
                mul_a = a_lo7_q;
                mul_b = b_lo7_q;
            end
        endcase
        mul_product = mul_a * mul_b;
        mul_extended = {{16{mul_product[15]}}, mul_product};
        case (phase_q)
            2'd0: shifted_term = mul_extended <<< 16;
            2'd1, 2'd2: shifted_term = mul_extended <<< 8;
            default: shifted_term = mul_extended;
        endcase
    end

    assign o_ready = !busy_q && (!product_valid_q || i_ready);
    assign o_valid = product_valid_q;
    assign o_product = product_q;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            busy_q <= 1'b0;
            phase_q <= 2'd0;
            a_hi_q <= 8'sd0;
            b_hi_q <= 8'sd0;
            a_lo7_q <= 8'sd0;
            b_lo7_q <= 8'sd0;
            correction_q <= 32'sd0;
            accumulator_q <= 32'sd0;
            product_q <= 32'sd0;
            product_valid_q <= 1'b0;
        end else begin
            if (product_valid_q && i_ready) begin
                product_valid_q <= 1'b0;
            end

            if (i_valid && o_ready) begin
                a_hi_q <= i_a[15:8];
                b_hi_q <= i_b[15:8];
                a_lo7_q <= {1'b0, i_a[6:0]};
                b_lo7_q <= {1'b0, i_b[6:0]};
                correction_q <= correction_in;
                accumulator_q <= 32'sd0;
                phase_q <= 2'd0;
                busy_q <= 1'b1;
            end else if (busy_q) begin
                if (phase_q == 2'd3) begin
                    product_q <= accumulator_q + shifted_term + correction_q;
                    product_valid_q <= 1'b1;
                    busy_q <= 1'b0;
                    phase_q <= 2'd0;
                end else begin
                    accumulator_q <= accumulator_q + shifted_term;
                    phase_q <= phase_q + 2'd1;
                end
            end
        end
    end

endmodule
