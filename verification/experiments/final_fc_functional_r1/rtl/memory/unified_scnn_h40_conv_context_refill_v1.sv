`timescale 1ns/1ps
`default_nettype none

// Refills one Conv output-channel wave from the shared persistent OIHW image.
// Logical output channels are remapped onto physical tiles 0..15. The selected
// pair must differ from the classifier-resident pair, allowing both layers to
// reuse one four-bank H40 cache without duplicating the compute substrate.
module unified_scnn_h40_conv_context_refill_v1 #(
    parameter int PARAM_DEPTH = 32768,
    parameter int PARAM_ADDR_WIDTH = $clog2(PARAM_DEPTH),
    parameter int TAG_WIDTH = 16,
    parameter int VERSION_WIDTH = 16,
    parameter int MAX_INPUT_CHANNELS = 512,
    parameter int MAX_CONTRIBUTIONS = MAX_INPUT_CHANNELS * 9,
    parameter int ARRAY_WEIGHT_ADDR_WIDTH = 11,
    parameter int INPUT_CHANNEL_COUNT_WIDTH =
        $clog2(MAX_INPUT_CHANNELS + 1),
    parameter int CONTRIBUTION_COUNT_WIDTH =
        $clog2(MAX_CONTRIBUTIONS + 1)
) (
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,
    input  wire logic                              i_abort,
    input  wire logic                              i_clear_error,

    input  wire logic                              i_context_valid,
    output logic                                   o_context_ready,
    input  wire logic                              i_context_load_weights,
    input  wire logic [9:0]                        i_output_channel_base,
    input  wire logic [4:0]                        i_output_channel_count,
    input  wire logic [INPUT_CHANNEL_COUNT_WIDTH-1:0]
                                                    i_input_channel_count,
    input  wire logic                              i_context_pair,
    input  wire logic                              i_reserved_pair_valid,
    input  wire logic                              i_reserved_pair,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_conv_parameter_base,
    input  wire logic [VERSION_WIDTH-1:0]           i_version,
    // True while the graph scheduler is stopped at its context-load barrier.
    input  wire logic                              i_refill_safe,

    output logic                                   o_forward_read_valid,
    input  wire logic                              i_forward_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]             o_forward_read_addr,
    output logic [TAG_WIDTH-1:0]                    o_forward_read_tag,
    input  wire logic                              i_forward_response_valid,
    output logic                                   o_forward_response_ready,
    input  wire logic [TAG_WIDTH-1:0]               i_forward_response_tag,
    input  wire logic signed [15:0]                 i_forward_response_code,
    input  wire logic                              i_forward_response_error,

    output logic [3:0]                             o_weight_fill_valid,
    input  wire logic [3:0]                        i_weight_fill_ready,
    output logic [7:0]                             o_weight_fill_tile_flat,
    output logic [3:0]                             o_weight_fill_bank,
    output logic [4*ARRAY_WEIGHT_ADDR_WIDTH-1:0]    o_weight_fill_addr_flat,
    output logic [799:0]                           o_weight_fill_data_flat,

    output logic                                   o_busy,
    output logic                                   o_refill_done,
    output logic                                   o_sticky_error
);
    typedef enum logic [1:0] {
        REFILL_IDLE,
        REFILL_WAIT,
        REFILL_RESPOND
    } refill_state_t;

    refill_state_t state_q;
    logic publisher_start_valid;
    logic publisher_start_ready;
    logic publisher_done;
    logic publisher_busy;
    logic publisher_error;
    logic [PARAM_ADDR_WIDTH-1:0] wave_parameter_base;
    logic [CONTRIBUTION_COUNT_WIDTH-1:0] contribution_count;
    logic request_legal;
    logic request_fire;
    logic invalid_request_fire;
    logic publisher_start_fire;
    logic [PARAM_ADDR_WIDTH:0] parameter_start_ext;
    logic [PARAM_ADDR_WIDTH:0] parameter_end_ext;
    integer unsigned contribution_count_int;
    integer unsigned wave_parameter_offset_int;

    always_comb begin
        contribution_count_int = integer'(i_input_channel_count) * 9;
        wave_parameter_offset_int = integer'(i_output_channel_base) *
            contribution_count_int;
        parameter_start_ext = {1'b0, i_conv_parameter_base} +
            (PARAM_ADDR_WIDTH + 1)'(wave_parameter_offset_int);
        parameter_end_ext = parameter_start_ext +
            (PARAM_ADDR_WIDTH + 1)'(
                contribution_count_int * integer'(i_output_channel_count));
        contribution_count = CONTRIBUTION_COUNT_WIDTH'(
            contribution_count_int);
        wave_parameter_base = PARAM_ADDR_WIDTH'(parameter_start_ext);
        request_legal = i_input_channel_count != 0 &&
            i_input_channel_count <=
                INPUT_CHANNEL_COUNT_WIDTH'(MAX_INPUT_CHANNELS) &&
            contribution_count_int <= MAX_CONTRIBUTIONS &&
            i_output_channel_count != 0 &&
            i_output_channel_count <= 5'd16 &&
            parameter_start_ext < (PARAM_ADDR_WIDTH + 1)'(PARAM_DEPTH) &&
            parameter_end_ext <= (PARAM_ADDR_WIDTH + 1)'(PARAM_DEPTH) &&
            (!i_reserved_pair_valid ||
             i_context_pair != i_reserved_pair);
    end

    assign publisher_start_valid = state_q == REFILL_IDLE &&
        i_context_valid && i_context_load_weights && request_legal;
    assign publisher_start_fire = publisher_start_valid &&
        publisher_start_ready;
    assign invalid_request_fire = state_q == REFILL_IDLE &&
        i_context_valid && i_refill_safe &&
        i_context_load_weights && !request_legal;
    assign o_context_ready = state_q == REFILL_RESPOND ||
        (state_q == REFILL_IDLE && i_context_valid && i_refill_safe &&
         (!i_context_load_weights || !request_legal));
    assign request_fire = i_context_valid && o_context_ready;
    assign o_busy = state_q != REFILL_IDLE || publisher_busy;
    assign o_refill_done = state_q == REFILL_RESPOND && request_fire;

    unified_scnn_classifier_cache_refill_publisher_v1 #(
        .MAX_FEATURES(MAX_CONTRIBUTIONS), .MAX_CLASSES(16),
        .PARAM_DEPTH(PARAM_DEPTH), .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH), .VERSION_WIDTH(VERSION_WIDTH),
        .ARRAY_WEIGHT_ADDR_WIDTH(ARRAY_WEIGHT_ADDR_WIDTH),
        .FEATURE_COUNT_WIDTH(CONTRIBUTION_COUNT_WIDTH),
        .CLASS_COUNT_WIDTH(5)
    ) u_matrix_publisher (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_start_valid(publisher_start_valid),
        .o_start_ready(publisher_start_ready),
        .i_target_pair(i_context_pair),
        .i_parameter_base(wave_parameter_base),
        .i_feature_count(contribution_count),
        .i_class_count(i_output_channel_count), .i_version(i_version),
        .i_array_idle(i_refill_safe),
        .o_forward_read_valid(o_forward_read_valid),
        .i_forward_read_ready(i_forward_read_ready),
        .o_forward_read_addr(o_forward_read_addr),
        .o_forward_read_tag(o_forward_read_tag),
        .i_forward_response_valid(i_forward_response_valid),
        .o_forward_response_ready(o_forward_response_ready),
        .i_forward_response_tag(i_forward_response_tag),
        .i_forward_response_code(i_forward_response_code),
        .i_forward_response_error(i_forward_response_error),
        .o_weight_fill_valid(o_weight_fill_valid),
        .i_weight_fill_ready(i_weight_fill_ready),
        .o_weight_fill_tile_flat(o_weight_fill_tile_flat),
        .o_weight_fill_bank(o_weight_fill_bank),
        .o_weight_fill_addr_flat(o_weight_fill_addr_flat),
        .o_weight_fill_data_flat(o_weight_fill_data_flat),
        .o_done(publisher_done), .o_published_pair(),
        .o_published_version(), .o_busy(publisher_busy),
        .o_sticky_error(publisher_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= REFILL_IDLE;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= REFILL_IDLE;
        end else begin
            if (i_clear_error && state_q == REFILL_IDLE && !publisher_busy)
                o_sticky_error <= 1'b0;
            if (publisher_error || invalid_request_fire)
                o_sticky_error <= 1'b1;
            if (state_q != REFILL_IDLE && !i_context_valid)
                o_sticky_error <= 1'b1;

            case (state_q)
                REFILL_IDLE: begin
                    if (publisher_start_fire)
                        state_q <= REFILL_WAIT;
                end
                REFILL_WAIT: begin
                    if (publisher_done)
                        state_q <= REFILL_RESPOND;
                end
                REFILL_RESPOND: begin
                    if (request_fire)
                        state_q <= REFILL_IDLE;
                end
                default: begin
                    state_q <= REFILL_IDLE;
                    o_sticky_error <= 1'b1;
                end
            endcase
        end
    end

    initial begin
        if (MAX_INPUT_CHANNELS > 1024 || MAX_CONTRIBUTIONS > 13872 ||
            PARAM_ADDR_WIDTH < 1)
            $error("SCNN Conv context refill geometry is invalid");
    end
endmodule

`default_nettype wire
