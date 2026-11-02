`timescale 1ns/1ps
`default_nettype none

// Batch-level FC W2 scheduler. Hidden traces are retained from inference and
// paired with synchronous output-error reads in an elastic pipeline.
// Zero terms are filtered without changing address-major contribution order.
module unified_fc_tp_w2_contribution_scheduler #(
    parameter int MAX_BATCH = 16,
    parameter int MAX_HIDDEN = 400,
    parameter int MAX_OUTPUTS = 16,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int ERROR_WIDTH = 18,
    parameter bit OUTPUT_MAJOR_ADDRESS = 1'b0,
    parameter int PARAMETER_HIDDEN_STRIDE = 0,
    parameter int PARAMETER_OUTPUT_STRIDE = 0,
    parameter int BATCH_INDEX_WIDTH = (MAX_BATCH <= 2) ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int HIDDEN_INDEX_WIDTH = (MAX_HIDDEN <= 2) ? 1 : $clog2(MAX_HIDDEN),
    parameter int HIDDEN_COUNT_WIDTH = $clog2(MAX_HIDDEN + 1),
    parameter int OUTPUT_INDEX_WIDTH = (MAX_OUTPUTS <= 2) ? 1 : $clog2(MAX_OUTPUTS),
    parameter int OUTPUT_COUNT_WIDTH = $clog2(MAX_OUTPUTS + 1)
) (
    input  wire logic                                  i_clk,
    input  wire logic                                  i_rst_n,
    input  wire logic                                  i_cfg_valid,
    output logic                                  o_cfg_ready,
    input  wire logic [BATCH_COUNT_WIDTH-1:0]           i_cfg_batch_count,
    input  wire logic [HIDDEN_COUNT_WIDTH-1:0]          i_cfg_hidden_count,
    input  wire logic [OUTPUT_COUNT_WIDTH-1:0]          i_cfg_output_count,
    input  wire logic                                  i_cfg_output_major,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]           i_cfg_hidden_stride,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]           i_cfg_output_stride,
    output logic                                  o_cfg_error,
    output logic                                  o_active,
    input  wire logic                                  i_release,
    output logic                                  o_release_ready,
    input  wire logic                                  i_hidden_trace_valid,
    output logic                                  o_hidden_trace_ready,
    input  wire logic [BATCH_INDEX_WIDTH-1:0]           i_hidden_trace_sample_id,
    input  wire logic [HIDDEN_INDEX_WIDTH-1:0]          i_hidden_trace_hidden_index,
    input  wire logic [15:0]                           i_hidden_trace_q016,
    output logic                                  o_hidden_trace_error,
    input  wire logic                                  i_output_error_valid,
    output logic                                  o_output_error_ready,
    input  wire logic [BATCH_INDEX_WIDTH-1:0]           i_output_error_sample_id,
    input  wire logic [OUTPUT_INDEX_WIDTH-1:0]          i_output_error_index,
    input  wire logic signed [ERROR_WIDTH-1:0]          i_output_error_q016,
    output logic                                  o_output_error_error,
    input  wire logic                                  i_observation_done,
    output logic                                  o_observation_done_ready,
    input  wire logic                                  i_emit_start,
    output logic                                  o_emit_ready,
    output logic                                  o_contribution_valid,
    input  wire logic                                  i_contribution_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]            o_contribution_local_addr,
    output logic [15:0]                           o_contribution_source_trace,
    output logic signed [ERROR_WIDTH-1:0]          o_contribution_error_q016,
    output logic                                  o_emit_done,
    output logic                                  o_busy
);

    localparam int TRACE_DEPTH = MAX_BATCH * MAX_HIDDEN;
    localparam int ERROR_DEPTH = MAX_BATCH * MAX_OUTPUTS;
    localparam int TRACE_ADDR_WIDTH = (TRACE_DEPTH <= 2) ? 1 : $clog2(TRACE_DEPTH);
    localparam int ERROR_ADDR_WIDTH = (ERROR_DEPTH <= 2) ? 1 : $clog2(ERROR_DEPTH);

    typedef enum logic [1:0] {
        STATE_IDLE, STATE_OBSERVE, STATE_SEALED, STATE_STREAM
    } state_t;

    state_t state_q;
    logic [15:0] trace_mem [0:TRACE_DEPTH-1];
    logic signed [ERROR_WIDTH-1:0] error_mem [0:ERROR_DEPTH-1];
    logic [BATCH_COUNT_WIDTH-1:0] batch_count_q;
    logic [HIDDEN_COUNT_WIDTH-1:0] hidden_count_q;
    logic [OUTPUT_COUNT_WIDTH-1:0] output_count_q;
    logic [BATCH_INDEX_WIDTH-1:0] hidden_sample_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] hidden_index_q;
    logic [BATCH_INDEX_WIDTH-1:0] error_sample_q;
    logic [OUTPUT_INDEX_WIDTH-1:0] error_index_q;
    logic hidden_complete_q;
    logic error_complete_q;
    logic [BATCH_INDEX_WIDTH-1:0] emit_sample_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] emit_hidden_q;
    logic [OUTPUT_INDEX_WIDTH-1:0] emit_output_q;
    logic [15:0] emit_trace_q;
    logic signed [ERROR_WIDTH-1:0] emit_error_q;
    logic contribution_valid_q;
    logic [PARAM_ADDR_WIDTH-1:0] contribution_local_addr_q;
    logic [PARAM_ADDR_WIDTH-1:0] contribution_local_addr_next;
    logic [15:0] contribution_source_trace_q;
    logic signed [ERROR_WIDTH-1:0] contribution_error_q016_q;

    logic cfg_legal;
    logic cfg_fire;
    logic release_fire;
    logic hidden_fire;
    logic output_error_fire;
    logic observation_done_fire;
    logic emit_fire;
    logic contribution_fire;
    logic emit_advance;
    logic read_valid_q;
    logic scan_exhausted_q;
    logic [PARAM_ADDR_WIDTH-1:0] read_local_addr_q;
    logic read_nonzero;
    logic output_available;
    logic read_available;
    // One bit per observed trace; sample banks have independent write enables.
    wire [MAX_BATCH-1:0] trace_nonzero_read;
    logic [MAX_BATCH-1:0] remaining_samples_q;
    logic mask_valid_q;
    logic sparse_mode;
    logic mask_last;
    logic address_advance;
    logic [BATCH_INDEX_WIDTH-1:0] selected_sample;

    for (genvar sample_bank = 0; sample_bank < MAX_BATCH; sample_bank++) begin : g_trace_mask
        (* ram_style = "distributed" *) logic nonzero_mem [0:MAX_HIDDEN-1];
        assign trace_nonzero_read[sample_bank] = nonzero_mem[emit_hidden_q];
        always_ff @(posedge i_clk) begin
            if (i_rst_n && hidden_fire && hidden_input_matches &&
                (i_hidden_trace_sample_id == BATCH_INDEX_WIDTH'(sample_bank)))
                nonzero_mem[i_hidden_trace_hidden_index] <=
                    (i_hidden_trace_q016 != '0);
        end
    end
    logic hidden_input_matches;
    logic error_input_matches;
    logic hidden_input_last;
    logic error_input_last;
    logic emit_sample_last;
    logic emit_hidden_last;
    logic emit_output_last;
    logic emit_address_last;
    logic [TRACE_ADDR_WIDTH-1:0] hidden_write_addr;
    logic [ERROR_ADDR_WIDTH-1:0] error_write_addr;
    logic [TRACE_ADDR_WIDTH-1:0] emit_trace_addr;
    logic [ERROR_ADDR_WIDTH-1:0] emit_error_addr;
    logic [PARAM_ADDR_WIDTH-1:0] address_stride;
    logic output_major_q;
    logic [PARAM_ADDR_WIDTH-1:0] address_stride_q;

    function automatic logic [TRACE_ADDR_WIDTH-1:0] trace_address(
        input logic [BATCH_INDEX_WIDTH-1:0] sample_index,
        input logic [HIDDEN_INDEX_WIDTH-1:0] hidden_index
    );
        trace_address = TRACE_ADDR_WIDTH'(
            (sample_index * MAX_HIDDEN) + hidden_index
        );
    endfunction

    function automatic logic [ERROR_ADDR_WIDTH-1:0] error_address(
        input logic [BATCH_INDEX_WIDTH-1:0] sample_index,
        input logic [OUTPUT_INDEX_WIDTH-1:0] output_index
    );
        error_address = ERROR_ADDR_WIDTH'(
            (sample_index * MAX_OUTPUTS) + output_index
        );
    endfunction

    always_comb begin
        address_stride = i_cfg_output_major ?
            ((i_cfg_hidden_stride != '0) ? i_cfg_hidden_stride :
             ((PARAMETER_HIDDEN_STRIDE != 0) ?
              PARAM_ADDR_WIDTH'(PARAMETER_HIDDEN_STRIDE) :
              PARAM_ADDR_WIDTH'(i_cfg_hidden_count))) :
            ((i_cfg_output_stride != '0) ? i_cfg_output_stride :
             ((PARAMETER_OUTPUT_STRIDE != 0) ?
              PARAM_ADDR_WIDTH'(PARAMETER_OUTPUT_STRIDE) :
              PARAM_ADDR_WIDTH'(i_cfg_output_count)));
        cfg_legal = (i_cfg_batch_count != '0) &&
            (i_cfg_batch_count <= BATCH_COUNT_WIDTH'(MAX_BATCH)) &&
            (i_cfg_hidden_count != '0) &&
            (i_cfg_hidden_count <= HIDDEN_COUNT_WIDTH'(MAX_HIDDEN)) &&
            (i_cfg_output_count != '0) &&
            (i_cfg_output_count <= OUTPUT_COUNT_WIDTH'(MAX_OUTPUTS)) &&
            (address_stride >= (i_cfg_output_major ?
             PARAM_ADDR_WIDTH'(i_cfg_hidden_count) :
             PARAM_ADDR_WIDTH'(i_cfg_output_count)));
        o_cfg_ready = state_q == STATE_IDLE;
        cfg_fire = i_cfg_valid && o_cfg_ready;
        o_release_ready = (state_q == STATE_OBSERVE) && o_active &&
            (hidden_sample_q == '0) && (hidden_index_q == '0) &&
            (error_sample_q == '0) && (error_index_q == '0) &&
            !hidden_complete_q && !error_complete_q;
        release_fire = i_release && o_release_ready;
        o_hidden_trace_ready = (state_q == STATE_OBSERVE) && o_active &&
            !hidden_complete_q;
        hidden_fire = i_hidden_trace_valid && o_hidden_trace_ready;
        o_output_error_ready = (state_q == STATE_OBSERVE) && o_active &&
            !error_complete_q;
        output_error_fire = i_output_error_valid && o_output_error_ready;
        hidden_input_matches =
            (i_hidden_trace_sample_id == hidden_sample_q) &&
            (i_hidden_trace_hidden_index == hidden_index_q);
        error_input_matches =
            (i_output_error_sample_id == error_sample_q) &&
            (i_output_error_index == error_index_q);
        hidden_input_last =
            (BATCH_COUNT_WIDTH'({1'b0, hidden_sample_q} + 1'b1) >= batch_count_q) &&
            (HIDDEN_COUNT_WIDTH'({1'b0, hidden_index_q} + 1'b1) >= hidden_count_q);
        error_input_last =
            (BATCH_COUNT_WIDTH'({1'b0, error_sample_q} + 1'b1) >= batch_count_q) &&
            (OUTPUT_COUNT_WIDTH'({1'b0, error_index_q} + 1'b1) >= output_count_q);
        o_observation_done_ready = (state_q == STATE_OBSERVE) && o_active &&
            hidden_complete_q && error_complete_q;
        observation_done_fire = i_observation_done && o_observation_done_ready;
        o_emit_ready = state_q == STATE_SEALED;
        emit_fire = i_emit_start && o_emit_ready;
        o_contribution_valid = contribution_valid_q;
        contribution_fire = o_contribution_valid && i_contribution_ready;
        read_nonzero = (emit_trace_q != '0) && (emit_error_q != '0);
        output_available = !contribution_valid_q || i_contribution_ready;
        read_available = !read_valid_q || !read_nonzero || output_available;
        sparse_mode = batch_count_q > BATCH_COUNT_WIDTH'(2);
        selected_sample = emit_sample_q;
        if (sparse_mode) begin
            selected_sample = '0;
            for (int s = MAX_BATCH-1; s >= 0; s--)
                if (remaining_samples_q[s]) selected_sample = BATCH_INDEX_WIDTH'(s);
        end
        mask_last = (remaining_samples_q &
            (remaining_samples_q - MAX_BATCH'(1))) == '0;
        emit_advance = (state_q == STATE_STREAM) &&
            !scan_exhausted_q && read_available &&
            (!sparse_mode || (mask_valid_q && (remaining_samples_q != '0)));
        if (output_major_q) begin
            contribution_local_addr_next = PARAM_ADDR_WIDTH'(
                (emit_output_q * address_stride_q) + emit_hidden_q
            );
        end else begin
            contribution_local_addr_next = PARAM_ADDR_WIDTH'(
                (emit_hidden_q * address_stride_q) + emit_output_q
            );
        end
        o_contribution_local_addr = contribution_local_addr_q;
        o_contribution_source_trace = contribution_source_trace_q;
        o_contribution_error_q016 = contribution_error_q016_q;
        o_busy = (state_q != STATE_IDLE) && (state_q != STATE_OBSERVE) &&
            (state_q != STATE_SEALED);

        emit_sample_last = BATCH_COUNT_WIDTH'(
            {1'b0, emit_sample_q} + 1'b1) >= batch_count_q;
        emit_hidden_last = HIDDEN_COUNT_WIDTH'(
            {1'b0, emit_hidden_q} + 1'b1) >= hidden_count_q;
        emit_output_last = OUTPUT_COUNT_WIDTH'(
            {1'b0, emit_output_q} + 1'b1) >= output_count_q;
        emit_address_last = emit_hidden_last && emit_output_last;
        address_advance = (state_q == STATE_STREAM) && !scan_exhausted_q &&
            (sparse_mode ? (mask_valid_q && ((remaining_samples_q == '0) ||
                (emit_advance && mask_last))) : (emit_advance && emit_sample_last));
        hidden_write_addr = trace_address(
            i_hidden_trace_sample_id, i_hidden_trace_hidden_index);
        error_write_addr = error_address(
            i_output_error_sample_id, i_output_error_index);
        emit_trace_addr = trace_address(selected_sample, emit_hidden_q);
        emit_error_addr = error_address(selected_sample, emit_output_q);
    end

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            batch_count_q <= '0;
            hidden_count_q <= '0;
            output_count_q <= '0;
            hidden_sample_q <= '0;
            hidden_index_q <= '0;
            error_sample_q <= '0;
            error_index_q <= '0;
            hidden_complete_q <= 1'b0;
            error_complete_q <= 1'b0;
            emit_sample_q <= '0;
            emit_hidden_q <= '0;
            emit_output_q <= '0;
            emit_trace_q <= '0;
            emit_error_q <= '0;
            read_valid_q <= 1'b0;
            scan_exhausted_q <= 1'b0;
            read_local_addr_q <= '0;
            remaining_samples_q <= '0;
            mask_valid_q <= 1'b0;
            contribution_valid_q <= 1'b0;
            contribution_local_addr_q <= '0;
            contribution_source_trace_q <= '0;
            contribution_error_q016_q <= '0;
            output_major_q <= OUTPUT_MAJOR_ADDRESS;
            address_stride_q <= '0;
            o_active <= 1'b0;
            o_cfg_error <= 1'b0;
            o_hidden_trace_error <= 1'b0;
            o_output_error_error <= 1'b0;
            o_emit_done <= 1'b0;
        end else begin
            o_cfg_error <= 1'b0;
            o_hidden_trace_error <= 1'b0;
            o_output_error_error <= 1'b0;
            o_emit_done <= 1'b0;

            case (state_q)
                STATE_IDLE: begin
                    if (cfg_fire) begin
                        if (!cfg_legal) begin
                            o_cfg_error <= 1'b1;
                        end else begin
                            batch_count_q <= i_cfg_batch_count;
                            hidden_count_q <= i_cfg_hidden_count;
                            output_count_q <= i_cfg_output_count;
                            output_major_q <= i_cfg_output_major;
                            address_stride_q <= address_stride;
                            hidden_sample_q <= '0;
                            hidden_index_q <= '0;
                            error_sample_q <= '0;
                            error_index_q <= '0;
                            hidden_complete_q <= 1'b0;
                            error_complete_q <= 1'b0;
                            o_active <= 1'b1;
                            state_q <= STATE_OBSERVE;
                        end
                    end
                end

                STATE_OBSERVE: begin
                    if (release_fire) begin
                        o_active <= 1'b0;
                        address_stride_q <= '0;
                        state_q <= STATE_IDLE;
                    end else begin
                        if (hidden_fire) begin
                            if (!hidden_input_matches) begin
                                o_hidden_trace_error <= 1'b1;
                            end else begin
                                trace_mem[hidden_write_addr] <= i_hidden_trace_q016;
                                if (hidden_input_last) begin
                                    hidden_complete_q <= 1'b1;
                                end else if (({1'b0, hidden_index_q} + 1'b1) >=
                                             {1'b0, hidden_count_q}) begin
                                    hidden_sample_q <= hidden_sample_q + 1'b1;
                                    hidden_index_q <= '0;
                                end else begin
                                    hidden_index_q <= hidden_index_q + 1'b1;
                                end
                            end
                        end
                        if (output_error_fire) begin
                            if (!error_input_matches) begin
                                o_output_error_error <= 1'b1;
                            end else begin
                                error_mem[error_write_addr] <= i_output_error_q016;
                                if (error_input_last) begin
                                    error_complete_q <= 1'b1;
                                end else if (OUTPUT_COUNT_WIDTH'(
                                             {1'b0, error_index_q} + 1'b1) >=
                                             output_count_q) begin
                                    error_sample_q <= error_sample_q + 1'b1;
                                    error_index_q <= '0;
                                end else begin
                                    error_index_q <= error_index_q + 1'b1;
                                end
                            end
                        end
                        if (observation_done_fire)
                            state_q <= STATE_SEALED;
                    end
                end

                STATE_SEALED: begin
                    if (emit_fire) begin
                        emit_sample_q <= '0;
                        emit_hidden_q <= '0;
                        emit_output_q <= '0;
                        read_valid_q <= 1'b0;
                        scan_exhausted_q <= 1'b0;
                        mask_valid_q <= 1'b0;
                        remaining_samples_q <= '0;
                        contribution_valid_q <= 1'b0;
                        state_q <= STATE_STREAM;
                    end
                end

                STATE_STREAM: begin
                    // Each stage holds its payload while the downstream stage stalls.
                    if (output_available) begin
                        contribution_valid_q <= read_valid_q && read_nonzero;
                        if (read_valid_q && read_nonzero) begin
                            contribution_local_addr_q <= read_local_addr_q;
                            contribution_source_trace_q <= emit_trace_q;
                            contribution_error_q016_q <= emit_error_q;
                        end
                    end
                    if (read_available) begin
                        read_valid_q <= emit_advance;
                        if (emit_advance) begin
                            emit_trace_q <= trace_mem[emit_trace_addr];
                            emit_error_q <= error_mem[emit_error_addr];
                            read_local_addr_q <= contribution_local_addr_next;
                        end
                    end
                    if (scan_exhausted_q && !read_valid_q && !contribution_valid_q) begin
                        hidden_sample_q <= '0;
                        hidden_index_q <= '0;
                        error_sample_q <= '0;
                        error_index_q <= '0;
                        hidden_complete_q <= 1'b0;
                        error_complete_q <= 1'b0;
                        o_emit_done <= 1'b1;
                        state_q <= STATE_OBSERVE;
                    end
                end

                default: state_q <= STATE_IDLE;
            endcase

            if ((state_q == STATE_STREAM) && sparse_mode && !scan_exhausted_q) begin
                // All active bits are overwritten before seal; inactive banks are masked.
                if (!mask_valid_q) begin
                    for (int s = 0; s < MAX_BATCH; s++)
                        remaining_samples_q[s] <= (s < int'(batch_count_q)) &&
                            trace_nonzero_read[s];
                    mask_valid_q <= 1'b1;
                end else if (address_advance) begin
                    mask_valid_q <= 1'b0;
                end else if (emit_advance) begin
                    remaining_samples_q[selected_sample] <= 1'b0;
                end
            end
            if (address_advance) begin
                emit_sample_q <= '0;
                if (emit_address_last) begin
                    scan_exhausted_q <= 1'b1;
                end else if (output_major_q) begin
                    if (emit_hidden_last) begin
                        emit_hidden_q <= '0;
                        emit_output_q <= emit_output_q + 1'b1;
                    end else begin
                        emit_hidden_q <= emit_hidden_q + 1'b1;
                    end
                end else begin
                    if (emit_output_last) begin
                        emit_output_q <= '0;
                        emit_hidden_q <= emit_hidden_q + 1'b1;
                    end else begin
                        emit_output_q <= emit_output_q + 1'b1;
                    end
                end
            end else if (emit_advance && !sparse_mode) begin
                emit_sample_q <= emit_sample_q + 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
