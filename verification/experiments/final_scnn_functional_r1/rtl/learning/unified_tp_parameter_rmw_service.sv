`timescale 1ns/1ps
`default_nettype none

// Single-outstanding parameter read/modify/write service for the first unified
// learning slice. Topology adapters submit only a tensor-local address and an
// update direction. The backing-store port supplies code/residual state; the
// shared direct-update path returns a writeback-qualified response.
module unified_tp_parameter_rmw_service #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int DELTA_WIDTH = 18
) (
    input  wire logic                               i_clk,
    input  wire logic                               i_rst_n,

    input  wire logic                               i_cfg_valid,
    output logic                               o_cfg_ready,
    input  wire logic [PROFILE_ID_WIDTH-1:0]        i_cfg_profile_id,
    input  wire logic [TENSOR_ID_WIDTH-1:0]         i_cfg_tensor_id,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_cfg_parameter_base,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_cfg_element_count,
    input  wire logic        [VOTE_WIDTH-1:0]       i_cfg_vote_increment,
    input  wire logic        [VOTE_WIDTH-1:0]       i_cfg_vote_threshold,
    input  wire logic signed [CODE_WIDTH-1:0]       i_cfg_code_min,
    input  wire logic signed [CODE_WIDTH-1:0]       i_cfg_code_max,
    output logic                               o_cfg_error,

    input  wire logic                               i_release,
    output logic                               o_release_ready,
    output logic                               o_active,
    output logic [PROFILE_ID_WIDTH-1:0]        o_active_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]         o_active_tensor_id,

    input  wire logic                               i_req_valid,
    output logic                               o_req_ready,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_req_local_parameter_addr,
    input  wire logic [TAG_WIDTH-1:0]               i_req_tag,
    input  wire logic signed [1:0]                  i_req_update_direction,
    input  wire logic                               i_req_magnitude_mode,
    input  wire logic signed [DELTA_WIDTH-1:0]      i_req_scaled_residual_delta,
    output logic                               o_req_error,

    output logic                               o_read_valid,
    input  wire logic                               i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0]        o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]         o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]        o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]               o_read_tag,

    input  wire logic                               i_read_response_valid,
    output logic                               o_read_response_ready,
    input  wire logic signed [CODE_WIDTH-1:0]       i_read_response_code,
    input  wire logic signed [RESIDUAL_WIDTH-1:0]   i_read_response_residual,
    input  wire logic                               i_read_response_error,

    output logic                               o_result_valid,
    input  wire logic                               i_result_ready,
    output logic [PROFILE_ID_WIDTH-1:0]        o_result_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]         o_result_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]        o_result_parameter_addr,
    output logic [TAG_WIDTH-1:0]               o_result_tag,
    output logic signed [CODE_WIDTH-1:0]       o_result_code,
    output logic signed [RESIDUAL_WIDTH-1:0]   o_result_residual,
    output logic                               o_write_code_enable,
    output logic                               o_write_residual_enable,
    output logic                               o_write_code_fire,
    output logic                               o_write_residual_fire,
    output logic                               o_dirty,
    output logic                               o_saturated,
    output logic                               o_update_config_error,
    output logic                               o_store_error,
    output logic                               o_internal_error,
    output logic                               o_busy
);

    typedef enum logic [1:0] {
        RMW_IDLE,
        RMW_READ_REQUEST,
        RMW_READ_RESPONSE,
        RMW_UPDATE_RESPONSE
    } rmw_state_t;

    rmw_state_t state_q;
    logic [PARAM_ADDR_WIDTH-1:0] parameter_base_q;
    logic [PARAM_ADDR_WIDTH-1:0] element_count_q;
    logic [PROFILE_ID_WIDTH-1:0] request_profile_id_q;
    logic [TENSOR_ID_WIDTH-1:0] request_tensor_id_q;
    logic [PARAM_ADDR_WIDTH-1:0] request_local_addr_q;
    logic [PARAM_ADDR_WIDTH-1:0] request_global_addr_q;
    logic [TAG_WIDTH-1:0] request_tag_q;
    logic signed [1:0] request_direction_q;
    logic request_magnitude_mode_q;
    logic signed [DELTA_WIDTH-1:0] request_scaled_residual_delta_q;

    logic path_cfg_valid;
    logic path_cfg_ready;
    logic path_cfg_error;
    logic path_release;
    logic path_release_ready;
    logic path_active;
    logic [PROFILE_ID_WIDTH-1:0] path_active_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] path_active_tensor_id;
    logic path_req_valid;
    logic path_req_ready;
    logic path_req_error;
    logic path_result_valid;
    logic path_result_ready;

    logic request_fire;
    logic read_fire;
    logic read_response_fire;
    logic result_fire;
    logic request_address_legal;

    always_comb begin
        path_cfg_valid = i_cfg_valid && (state_q == RMW_IDLE);
        o_cfg_ready = (state_q == RMW_IDLE) && path_cfg_ready;
        path_release = i_release && (state_q == RMW_IDLE);
        o_release_ready = (state_q == RMW_IDLE) && path_release_ready;

        request_address_legal = path_active &&
            (i_req_local_parameter_addr < element_count_q);

        o_read_valid = state_q == RMW_READ_REQUEST;
        read_fire = o_read_valid && i_read_ready;
        o_read_response_ready = (state_q == RMW_READ_RESPONSE) &&
            (i_read_response_error || path_req_ready);
        read_response_fire = i_read_response_valid && o_read_response_ready;

        path_req_valid = (state_q == RMW_READ_RESPONSE) &&
            i_read_response_valid && !i_read_response_error;
        path_result_ready = (state_q == RMW_UPDATE_RESPONSE) && i_result_ready;
        o_result_valid = (state_q == RMW_UPDATE_RESPONSE) && path_result_valid;
        result_fire = o_result_valid && i_result_ready;

        // Reuse the single request slot only when the old write retires.
        // The replacement read remains in the following cycle.
        o_req_ready = ((state_q == RMW_IDLE) || result_fire) &&
            !i_cfg_valid && !i_release;
        request_fire = i_req_valid && o_req_ready;
    end

    assign o_active = path_active;
    assign o_active_profile_id = path_active_profile_id;
    assign o_active_tensor_id = path_active_tensor_id;
    assign o_cfg_error = path_cfg_error;
    assign o_read_profile_id = request_profile_id_q;
    assign o_read_tensor_id = request_tensor_id_q;
    assign o_read_parameter_addr = request_global_addr_q;
    assign o_read_tag = request_tag_q;
    assign o_busy = state_q != RMW_IDLE;

    unified_tp_active_tensor_update_path #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH),
        .DELTA_WIDTH(DELTA_WIDTH)
    ) update_path (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(path_cfg_valid),
        .o_cfg_ready(path_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id),
        .i_cfg_tensor_id(i_cfg_tensor_id),
        .i_cfg_parameter_base(i_cfg_parameter_base),
        .i_cfg_element_count(i_cfg_element_count),
        .i_cfg_vote_increment(i_cfg_vote_increment),
        .i_cfg_vote_threshold(i_cfg_vote_threshold),
        .i_cfg_code_min(i_cfg_code_min),
        .i_cfg_code_max(i_cfg_code_max),
        .o_cfg_error(path_cfg_error),
        .i_release(path_release),
        .o_release_ready(path_release_ready),
        .o_active(path_active),
        .o_active_profile_id(path_active_profile_id),
        .o_active_tensor_id(path_active_tensor_id),
        .i_req_valid(path_req_valid),
        .o_req_ready(path_req_ready),
        .i_req_local_parameter_addr(request_local_addr_q),
        .i_req_tag(request_tag_q),
        .i_req_code(i_read_response_code),
        .i_req_residual(i_read_response_residual),
        .i_req_update_direction(request_direction_q),
        .i_req_magnitude_mode(request_magnitude_mode_q),
        .i_req_scaled_residual_delta(request_scaled_residual_delta_q),
        .o_req_error(path_req_error),
        .o_valid(path_result_valid),
        .i_ready(path_result_ready),
        .o_profile_id(o_result_profile_id),
        .o_tensor_id(o_result_tensor_id),
        .o_parameter_addr(o_result_parameter_addr),
        .o_tag(o_result_tag),
        .o_code(o_result_code),
        .o_residual(o_result_residual),
        .o_write_code_enable(o_write_code_enable),
        .o_write_residual_enable(o_write_residual_enable),
        .o_write_code_fire(o_write_code_fire),
        .o_write_residual_fire(o_write_residual_fire),
        .o_dirty(o_dirty),
        .o_saturated(o_saturated),
        .o_update_config_error(o_update_config_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= RMW_IDLE;
            parameter_base_q <= '0;
            element_count_q <= '0;
            request_profile_id_q <= '0;
            request_tensor_id_q <= '0;
            request_local_addr_q <= '0;
            request_global_addr_q <= '0;
            request_tag_q <= '0;
            request_direction_q <= '0;
            request_magnitude_mode_q <= 1'b0;
            request_scaled_residual_delta_q <= '0;
            o_req_error <= 1'b0;
            o_store_error <= 1'b0;
            o_internal_error <= 1'b0;
        end else begin
            o_req_error <= 1'b0;
            o_store_error <= 1'b0;
            o_internal_error <= 1'b0;

            if (i_cfg_valid && o_cfg_ready) begin
                parameter_base_q <= i_cfg_parameter_base;
                element_count_q <= i_cfg_element_count;
            end

            if (i_release && o_release_ready) begin
                parameter_base_q <= '0;
                element_count_q <= '0;
            end

            if (request_fire) begin
                if (!request_address_legal) begin
                    o_req_error <= 1'b1;
                end else begin
                    request_profile_id_q <= path_active_profile_id;
                    request_tensor_id_q <= path_active_tensor_id;
                    request_local_addr_q <= i_req_local_parameter_addr;
                    request_global_addr_q <= parameter_base_q +
                        i_req_local_parameter_addr;
                    request_tag_q <= i_req_tag;
                    request_direction_q <= i_req_update_direction;
                    request_magnitude_mode_q <= i_req_magnitude_mode;
                    request_scaled_residual_delta_q <=
                        i_req_scaled_residual_delta;
                    state_q <= RMW_READ_REQUEST;
                end
            end

            if (read_fire)
                state_q <= RMW_READ_RESPONSE;

            if (read_response_fire) begin
                if (i_read_response_error) begin
                    o_store_error <= 1'b1;
                    state_q <= RMW_IDLE;
                end else begin
                    state_q <= RMW_UPDATE_RESPONSE;
                end
            end

            if (path_req_error) begin
                o_internal_error <= 1'b1;
                state_q <= RMW_IDLE;
            end

            // Explicit retirement priority: a legal replacement owns the
            // next cycle; rejection still drains the previous result.
            if (result_fire)
                state_q <= (request_fire && request_address_legal) ?
                    RMW_READ_REQUEST : RMW_IDLE;
        end
    end

endmodule

`default_nettype wire
