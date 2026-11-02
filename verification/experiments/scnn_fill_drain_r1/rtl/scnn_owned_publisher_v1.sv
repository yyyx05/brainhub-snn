`timescale 1ns/1ps
`default_nettype none

// Local integration only. Array-idle remains mandatory; scheduler retirement
// and compute-drained inputs still require a real upstream integration.
module scnn_owned_publisher_v1 (
    input wire logic i_clk, i_rst_n,
    input wire logic i_abort, i_cache_invalidate, i_commit_pending, i_clear_error,
    input wire logic [15:0] i_active_version,
    input wire logic i_classifier_cache_valid, i_classifier_pair,
    input wire logic i_array_idle, i_read_compute_drained,
    input wire logic [9:0] i_req_output_base, i_req_input_count, i_req_row_base,
    input wire logic [4:0] i_req_output_count,
    input wire logic i_req_pair,
    input wire logic [23:0] i_req_parameter_base,
    input wire logic [15:0] i_req_version,
    input wire logic i_reserve_valid,
    output logic o_reserve_ready, o_reserve_rejected,
    input wire logic i_activate_valid, i_activate_slot,
    output logic o_activate_ready, o_activate_rejected,
    input wire logic i_release_valid, i_release_slot, i_release_final,
    output logic o_release_ready, o_release_rejected,

    output logic o_forward_read_valid,
    input wire logic i_forward_read_ready,
    output logic [23:0] o_forward_read_addr,
    output logic [15:0] o_forward_read_tag,
    input wire logic i_forward_response_valid,
    output logic o_forward_response_ready,
    input wire logic [15:0] i_forward_response_tag,
    input wire logic signed [15:0] i_forward_response_code,
    input wire logic i_forward_response_error,
    output logic [3:0] o_weight_fill_valid,
    input wire logic [3:0] i_weight_fill_ready,
    output logic [7:0] o_weight_fill_tile_flat,
    output logic [3:0] o_weight_fill_bank,
    output logic [43:0] o_weight_fill_addr_flat,
    output logic [799:0] o_weight_fill_data_flat,

    output logic [1:0] o_slot_occupied, o_slot_valid, o_slot_active,
    output logic o_current_valid, o_current_slot, o_current_pair,
    output logic [9:0] o_current_output_base, o_current_input_count, o_current_row_base,
    output logic [4:0] o_current_output_count,
    output logic [23:0] o_current_parameter_base,
    output logic [15:0] o_current_version,
    output logic [10:0] o_current_rows,
    output logic o_draining, o_cancel, o_fence_drained, o_sticky_error,
    output logic o_fill_drained, o_fill_inflight, o_publisher_busy,
    output logic o_publication_done, o_discarded_response
);
    logic fill_busy, fill_slot, fill_allowed, owner_error;
    logic [15:0] fill_tag, fill_version;
    logic [9:0] fill_input_count, fill_row_base;
    logic [4:0] fill_output_count;
    logic [23:0] fill_parameter_base;
    logic fill_pair;
    logic started_q, outstanding_q, transport_error_q;
    logic started_slot_q;
    logic [15:0] started_tag_q;
    logic [15:0] outstanding_tag_q;
    logic pub_start_valid, pub_start_ready, pub_done, pub_error;
    logic pub_read_valid, pub_read_ready, pub_response_ready;
    logic [3:0] pub_fill_valid, pub_fill_ready;
    logic [15:0] published_version;
    logic read_fire, response_fire, matching_response_fire, transport_fault, write_allowed;

    // Track accepted traffic across abort. Publisher busy alone loses this state.
    assign transport_fault = i_forward_response_valid &&
        (!outstanding_q || i_forward_response_tag != outstanding_tag_q);
    assign o_forward_read_valid = pub_read_valid && started_q && !o_cancel &&
        (!outstanding_q || matching_response_fire);
    assign pub_read_ready = i_forward_read_ready && started_q && !o_cancel &&
        (!outstanding_q || matching_response_fire);
    assign o_forward_response_ready = outstanding_q ?
        (o_cancel || pub_response_ready) : 1'b1;
    assign read_fire = o_forward_read_valid && i_forward_read_ready;
    assign response_fire = i_forward_response_valid && o_forward_response_ready;
    assign matching_response_fire = response_fire && outstanding_q &&
        i_forward_response_tag == outstanding_tag_q;
    assign o_discarded_response = response_fire && (o_cancel || !outstanding_q);
    assign o_fill_inflight = outstanding_q;
    assign o_fill_drained = !outstanding_q && !o_publisher_busy && !(|pub_fill_valid);
    assign o_sticky_error = owner_error || pub_error || transport_error_q;
    assign pub_start_valid = fill_busy && fill_allowed && !started_q && !o_cancel;
    assign o_publication_done = pub_done && started_q && !o_cancel;
    assign write_allowed = fill_allowed && started_q && !o_cancel;
    assign o_weight_fill_valid = pub_fill_valid & {4{write_allowed}};
    assign pub_fill_ready = i_weight_fill_ready & {4{write_allowed}};

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            started_q <= 1'b0;
            started_slot_q <= 1'b0;
            started_tag_q <= '0;
            outstanding_q <= 1'b0;
            outstanding_tag_q <= '0;
            transport_error_q <= 1'b0;
        end else begin
            // A response may retire in the same cycle as the next request.
            case ({read_fire, matching_response_fire})
                2'b10: outstanding_q <= 1'b1;
                2'b01: outstanding_q <= 1'b0;
                default: begin end
            endcase
            if (read_fire)
                outstanding_tag_q <= o_forward_read_tag;
            if (!fill_busy || (o_cancel && o_fill_drained))
                started_q <= 1'b0;
            else if (pub_start_valid && pub_start_ready) begin
                started_q <= 1'b1;
                started_slot_q <= fill_slot;
                started_tag_q <= fill_tag;
            end
            if (i_clear_error && o_fill_drained && !i_forward_response_valid)
                transport_error_q <= 1'b0;
            if (transport_fault)
                transport_error_q <= 1'b1;
        end
    end

    scnn_window_owner_v1 owner (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_abort(i_abort || pub_error || transport_error_q || transport_fault),
        .i_cache_invalidate(i_cache_invalidate), .i_commit_pending(i_commit_pending),
        .i_clear_error(i_clear_error), .i_active_version(i_active_version),
        .i_classifier_cache_valid(i_classifier_cache_valid), .i_classifier_pair(i_classifier_pair),
        .i_array_idle(i_array_idle), .i_read_compute_drained(i_read_compute_drained),
        .i_fill_drained(o_fill_drained),
        .i_req_output_base(i_req_output_base), .i_req_output_count(i_req_output_count),
        .i_req_input_count(i_req_input_count), .i_req_pair(i_req_pair),
        .i_req_parameter_base(i_req_parameter_base), .i_req_version(i_req_version),
        .i_req_row_base(i_req_row_base),
        .i_reserve_valid(i_reserve_valid), .o_reserve_ready(o_reserve_ready),
        .o_reserve_rejected(o_reserve_rejected),
        .i_activate_valid(i_activate_valid), .i_activate_slot(i_activate_slot),
        .o_activate_ready(o_activate_ready), .o_activate_rejected(o_activate_rejected),
        .i_release_valid(i_release_valid), .i_release_slot(i_release_slot),
        .i_release_final(i_release_final), .o_release_ready(o_release_ready),
        .o_release_rejected(o_release_rejected),
        .i_fill_done(pub_done && started_q), .i_fill_done_slot(started_slot_q),
        .i_fill_done_tag(started_tag_q), .i_fill_done_version(published_version),
        .o_fill_busy(fill_busy), .o_fill_slot(fill_slot), .o_fill_write_allowed(fill_allowed),
        .o_fill_tag(fill_tag), .o_fill_output_base(), .o_fill_output_count(fill_output_count),
        .o_fill_input_count(fill_input_count), .o_fill_pair(fill_pair),
        .o_fill_parameter_base(fill_parameter_base), .o_fill_version(fill_version),
        .o_fill_row_base(fill_row_base), .o_fill_rows(),
        .o_slot_occupied(o_slot_occupied), .o_slot_valid(o_slot_valid), .o_slot_active(o_slot_active),
        .o_current_valid(o_current_valid), .o_current_slot(o_current_slot),
        .o_current_output_base(o_current_output_base), .o_current_output_count(o_current_output_count),
        .o_current_input_count(o_current_input_count), .o_current_pair(o_current_pair),
        .o_current_parameter_base(o_current_parameter_base), .o_current_version(o_current_version),
        .o_current_row_base(o_current_row_base), .o_current_rows(o_current_rows),
        .o_draining(o_draining), .o_cancel(o_cancel), .o_fence_drained(o_fence_drained),
        .o_sticky_error(owner_error)
    );

    scnn_row_window_publisher_v1 #(
        .MAX_FEATURES(4608), .MAX_CLASSES(16), .PARAM_DEPTH(16777216),
        .PARAM_ADDR_WIDTH(24), .TAG_WIDTH(16), .VERSION_WIDTH(16),
        .FEATURE_COUNT_WIDTH(13), .CLASS_COUNT_WIDTH(5)
    ) publisher (
        // Once idle, release abort so its sticky error can be cleared. The owner
        // still blocks all starts, requests and writes throughout quarantine.
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(o_cancel && o_publisher_busy),
        .i_clear_error(i_clear_error),
        .i_start_valid(pub_start_valid), .o_start_ready(pub_start_ready),
        .i_target_pair(fill_pair), .i_target_row_base(fill_row_base),
        .i_parameter_base(fill_parameter_base), .i_feature_count(13'(fill_input_count) * 13'd9),
        .i_class_count(fill_output_count), .i_version(fill_version), .i_array_idle(i_array_idle),
        .o_forward_read_valid(pub_read_valid), .i_forward_read_ready(pub_read_ready),
        .o_forward_read_addr(o_forward_read_addr), .o_forward_read_tag(o_forward_read_tag),
        .i_forward_response_valid(i_forward_response_valid && outstanding_q && !o_cancel),
        .o_forward_response_ready(pub_response_ready), .i_forward_response_tag(i_forward_response_tag),
        .i_forward_response_code(i_forward_response_code), .i_forward_response_error(i_forward_response_error),
        .o_weight_fill_valid(pub_fill_valid), .i_weight_fill_ready(pub_fill_ready),
        .o_weight_fill_tile_flat(o_weight_fill_tile_flat), .o_weight_fill_bank(o_weight_fill_bank),
        .o_weight_fill_addr_flat(o_weight_fill_addr_flat), .o_weight_fill_data_flat(o_weight_fill_data_flat),
        .o_done(pub_done), .o_published_pair(), .o_published_version(published_version),
        .o_busy(o_publisher_busy), .o_sticky_error(pub_error)
    );
endmodule
`default_nettype wire
