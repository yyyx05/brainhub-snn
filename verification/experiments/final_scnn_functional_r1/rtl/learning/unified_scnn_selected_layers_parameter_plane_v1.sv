`timescale 1ns/1ps
`default_nettype none

// Coherent parameter and cache plane for one selected Conv layer plus its
// classifier. Both learners use one persistent RMW store. The classifier stays
// resident in one cache pair; Conv output-channel waves refill the other pair.
module unified_scnn_selected_layers_parameter_plane_v1 #(
    parameter bit USE_INTERNAL_STORE = 1'b1,
    parameter int PARAM_DEPTH = 32768,
    parameter int PARAM_ADDR_WIDTH = $clog2(PARAM_DEPTH),
    parameter int TAG_WIDTH = 16,
    parameter int VERSION_WIDTH = 16,
    parameter int MAX_INPUT_CHANNELS = 512,
    parameter int MAX_CLASSIFIER_FEATURES = 2048,
    parameter int MAX_CLASSES = 11,
    parameter int ARRAY_WEIGHT_ADDR_WIDTH = 11,
    parameter int INPUT_CHANNEL_COUNT_WIDTH =
        $clog2(MAX_INPUT_CHANNELS + 1),
    parameter int CLASSIFIER_FEATURE_COUNT_WIDTH =
        $clog2(MAX_CLASSIFIER_FEATURES + 1),
    parameter int CLASS_COUNT_WIDTH = $clog2(MAX_CLASSES + 1)
) (
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,
    input  wire logic                              i_abort,
    input  wire logic                              i_clear_error,
    input  wire logic                              i_cache_invalidate,

    input  wire logic                              i_parameter_load_valid,
    output logic                                   o_parameter_load_ready,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_parameter_load_addr,
    input  wire logic signed [15:0]                 i_parameter_load_code,
    input  wire logic signed [15:0]                 i_parameter_load_residual,

    output logic                                   o_store_load_valid,
    input  wire logic                              i_store_load_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]            o_store_load_addr,
    output logic signed [15:0]                     o_store_load_code,
    output logic signed [15:0]                     o_store_load_residual,
    output logic                                   o_store_learning_read_valid,
    input  wire logic                              i_store_learning_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]            o_store_learning_read_addr,
    output logic [TAG_WIDTH-1:0]                   o_store_learning_read_tag,
    input  wire logic                              i_store_learning_response_valid,
    output logic                                   o_store_learning_response_ready,
    input  wire logic [TAG_WIDTH-1:0]              i_store_learning_response_tag,
    input  wire logic signed [15:0]                i_store_learning_response_code,
    input  wire logic signed [15:0]                i_store_learning_response_residual,
    input  wire logic                              i_store_learning_response_error,
    output logic                                   o_store_learning_write_valid,
    input  wire logic                              i_store_learning_write_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]            o_store_learning_write_addr,
    output logic signed [15:0]                     o_store_learning_write_code,
    output logic signed [15:0]                     o_store_learning_write_residual,
    output logic                                   o_store_learning_write_code_enable,
    output logic                                   o_store_learning_write_residual_enable,
    output logic                                   o_store_forward_read_valid,
    input  wire logic                              i_store_forward_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]            o_store_forward_read_addr,
    output logic [TAG_WIDTH-1:0]                   o_store_forward_read_tag,
    input  wire logic                              i_store_forward_response_valid,
    output logic                                   o_store_forward_response_ready,
    input  wire logic [TAG_WIDTH-1:0]              i_store_forward_response_tag,
    input  wire logic signed [15:0]                i_store_forward_response_code,
    input  wire logic                              i_store_forward_response_error,
    input  wire logic                              i_store_protocol_error,

    input  wire logic                              i_classifier_prime_valid,
    output logic                                   o_classifier_prime_ready,
    input  wire logic                              i_classifier_pair,
    input  wire logic [VERSION_WIDTH-1:0]           i_initial_version,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_classifier_parameter_base,
    input  wire logic [CLASSIFIER_FEATURE_COUNT_WIDTH-1:0]
                                                    i_classifier_feature_count,
    input  wire logic [CLASS_COUNT_WIDTH-1:0]       i_classifier_class_count,
    output logic                                   o_classifier_prime_done,

    // The learning runtime holds commit_pending until this plane has refreshed
    // the classifier cache and invalidated every Conv residency record. Only
    // then is commit_publish_ready asserted; commit_pulse consumes that grant.
    input  wire logic                              i_commit_pending,
    output logic                                   o_commit_publish_ready,
    input  wire logic                              i_commit_pulse,

    input  wire logic                              i_learning_read_valid,
    output logic                                   o_learning_read_ready,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_learning_read_addr,
    input  wire logic [TAG_WIDTH-1:0]               i_learning_read_tag,
    output logic                                   o_learning_response_valid,
    input  wire logic                              i_learning_response_ready,
    output logic [TAG_WIDTH-1:0]                    o_learning_response_tag,
    output logic signed [15:0]                     o_learning_response_code,
    output logic signed [15:0]                     o_learning_response_residual,
    output logic                                   o_learning_response_error,
    input  wire logic                              i_learning_write_valid,
    output logic                                   o_learning_write_ready,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_learning_write_addr,
    input  wire logic signed [15:0]                 i_learning_write_code,
    input  wire logic signed [15:0]                 i_learning_write_residual,
    input  wire logic                              i_learning_write_code_enable,
    input  wire logic                              i_learning_write_residual_enable,

    input  wire logic                              i_conv_context_valid,
    output logic                                   o_conv_context_ready,
    input  wire logic                              i_conv_context_load_weights,
    input  wire logic [9:0]                        i_conv_output_channel_base,
    input  wire logic [4:0]                        i_conv_output_channel_count,
    input  wire logic [INPUT_CHANNEL_COUNT_WIDTH-1:0]
                                                    i_conv_input_channel_count,
    input  wire logic                              i_conv_context_pair,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_conv_parameter_base,
    input  wire logic [VERSION_WIDTH-1:0]           i_conv_context_version,
    input  wire logic                              i_array_idle,

    output logic [3:0]                             o_weight_fill_valid,
    input  wire logic [3:0]                        i_weight_fill_ready,
    output logic [7:0]                             o_weight_fill_tile_flat,
    output logic [3:0]                             o_weight_fill_bank,
    output logic [4*ARRAY_WEIGHT_ADDR_WIDTH-1:0]    o_weight_fill_addr_flat,
    output logic [799:0]                           o_weight_fill_data_flat,
    output logic                                   o_weight_fill_busy,

    output logic                                   o_classifier_cache_valid,
    output logic                                   o_active_classifier_pair,
    output logic [VERSION_WIDTH-1:0]               o_active_version,
    output logic                                   o_conv_cache_valid,
    output logic [9:0]                             o_conv_cached_output_base,
    output logic [4:0]                             o_conv_cached_output_count,
    output logic [VERSION_WIDTH-1:0]               o_conv_cached_version,
    output logic                                   o_conv_cache_invalidate,
    output logic                                   o_busy,
    output logic                                   o_sticky_error
);
    logic backend_prime_valid;
    logic backend_prime_ready;
    logic backend_prime_done;
    logic backend_commit_valid;
    logic backend_commit_ready;
    logic backend_commit_done;
    logic backend_conv_invalidate;
    logic backend_busy;
    logic backend_error;

    logic backend_learning_read_valid;
    logic backend_learning_read_ready;
    logic backend_learning_write_valid;
    logic backend_learning_write_ready;

    logic refill_context_valid;
    logic refill_context_ready;
    logic refill_forward_read_valid;
    logic refill_forward_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] refill_forward_read_addr;
    logic [TAG_WIDTH-1:0] refill_forward_read_tag;
    logic refill_forward_response_valid;
    logic refill_forward_response_ready;
    logic [TAG_WIDTH-1:0] refill_forward_response_tag;
    logic signed [15:0] refill_forward_response_code;
    logic refill_forward_response_error;
    logic [3:0] refill_fill_valid;
    logic [3:0] refill_fill_ready;
    logic [7:0] refill_fill_tile_flat;
    logic [3:0] refill_fill_bank;
    logic [4*ARRAY_WEIGHT_ADDR_WIDTH-1:0] refill_fill_addr_flat;
    logic [799:0] refill_fill_data_flat;
    logic refill_busy;
    logic refill_done;
    logic refill_error;

    logic [3:0] backend_fill_valid;
    logic [3:0] backend_fill_ready;
    logic [7:0] backend_fill_tile_flat;
    logic [3:0] backend_fill_bank;
    logic [4*ARRAY_WEIGHT_ADDR_WIDTH-1:0] backend_fill_addr_flat;
    logic [799:0] backend_fill_data_flat;

    logic prime_inflight_q;
    logic commit_inflight_q;
    logic publication_ready_q;
    logic conv_cache_valid_q;
    logic [9:0] conv_output_base_q;
    logic [4:0] conv_output_count_q;
    logic [INPUT_CHANNEL_COUNT_WIDTH-1:0] conv_input_count_q;
    logic conv_pair_q;
    logic [PARAM_ADDR_WIDTH-1:0] conv_parameter_base_q;
    logic [VERSION_WIDTH-1:0] conv_version_q;

    logic prime_fire;
    logic backend_commit_fire;
    logic context_start_allowed;
    logic context_common_legal;
    logic resident_match;
    logic no_refill_context_fire;
    logic invalid_context_request;
    logic fill_overlap;
    logic learning_allowed;

    assign learning_allowed = !refill_busy && !prime_inflight_q &&
        !commit_inflight_q && !publication_ready_q;
    assign backend_learning_read_valid = i_learning_read_valid &&
        learning_allowed;
    assign o_learning_read_ready = backend_learning_read_ready &&
        learning_allowed;
    assign backend_learning_write_valid = i_learning_write_valid &&
        learning_allowed;
    assign o_learning_write_ready = backend_learning_write_ready &&
        learning_allowed;

    assign backend_prime_valid = i_classifier_prime_valid && !refill_busy &&
        !commit_inflight_q && !publication_ready_q;
    assign o_classifier_prime_ready = backend_prime_ready && !refill_busy &&
        !commit_inflight_q && !publication_ready_q;
    assign prime_fire = i_classifier_prime_valid &&
        o_classifier_prime_ready;
    assign o_classifier_prime_done = backend_prime_done;

    assign backend_commit_valid = i_commit_pending && !commit_inflight_q &&
        !publication_ready_q && !refill_busy && !prime_inflight_q;
    assign backend_commit_fire = backend_commit_valid && backend_commit_ready;
    assign o_commit_publish_ready = publication_ready_q;

    assign context_start_allowed = !backend_busy && !prime_inflight_q &&
        !commit_inflight_q && !publication_ready_q && !i_commit_pending &&
        !i_cache_invalidate;
    assign context_common_legal = o_classifier_cache_valid &&
        i_conv_context_version == o_active_version &&
        i_conv_context_pair != o_active_classifier_pair;
    assign resident_match = conv_cache_valid_q && context_common_legal &&
        i_conv_output_channel_base == conv_output_base_q &&
        i_conv_output_channel_count == conv_output_count_q &&
        i_conv_input_channel_count == conv_input_count_q &&
        i_conv_context_pair == conv_pair_q &&
        i_conv_parameter_base == conv_parameter_base_q &&
        i_conv_context_version == conv_version_q;

    assign refill_context_valid = i_conv_context_valid &&
        (refill_busy || (context_start_allowed && context_common_legal &&
                         i_conv_context_load_weights));
    assign no_refill_context_fire = i_conv_context_valid &&
        !i_conv_context_load_weights && context_start_allowed &&
        resident_match && i_array_idle;
    assign o_conv_context_ready = i_conv_context_load_weights ?
        refill_context_ready : no_refill_context_fire;
    assign invalid_context_request = i_conv_context_valid &&
        context_start_allowed && i_array_idle &&
        (!context_common_legal ||
         (!i_conv_context_load_weights && !resident_match));

    unified_scnn_selected_layers_parameter_backend_v1 #(
        .USE_INTERNAL_STORE(USE_INTERNAL_STORE),
        .PARAM_DEPTH(PARAM_DEPTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH),
        .MAX_CLASSIFIER_FEATURES(MAX_CLASSIFIER_FEATURES),
        .MAX_CLASSES(MAX_CLASSES),
        .ARRAY_WEIGHT_ADDR_WIDTH(ARRAY_WEIGHT_ADDR_WIDTH),
        .FEATURE_COUNT_WIDTH(CLASSIFIER_FEATURE_COUNT_WIDTH),
        .CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH)
    ) u_backend (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_cache_invalidate(i_cache_invalidate),
        .i_parameter_load_valid(i_parameter_load_valid),
        .o_parameter_load_ready(o_parameter_load_ready),
        .i_parameter_load_addr(i_parameter_load_addr),
        .i_parameter_load_code(i_parameter_load_code),
        .i_parameter_load_residual(i_parameter_load_residual),
        .o_store_load_valid(o_store_load_valid),
        .i_store_load_ready(i_store_load_ready),
        .o_store_load_addr(o_store_load_addr),
        .o_store_load_code(o_store_load_code),
        .o_store_load_residual(o_store_load_residual),
        .o_store_learning_read_valid(o_store_learning_read_valid),
        .i_store_learning_read_ready(i_store_learning_read_ready),
        .o_store_learning_read_addr(o_store_learning_read_addr),
        .o_store_learning_read_tag(o_store_learning_read_tag),
        .i_store_learning_response_valid(i_store_learning_response_valid),
        .o_store_learning_response_ready(o_store_learning_response_ready),
        .i_store_learning_response_tag(i_store_learning_response_tag),
        .i_store_learning_response_code(i_store_learning_response_code),
        .i_store_learning_response_residual(
            i_store_learning_response_residual),
        .i_store_learning_response_error(i_store_learning_response_error),
        .o_store_learning_write_valid(o_store_learning_write_valid),
        .i_store_learning_write_ready(i_store_learning_write_ready),
        .o_store_learning_write_addr(o_store_learning_write_addr),
        .o_store_learning_write_code(o_store_learning_write_code),
        .o_store_learning_write_residual(o_store_learning_write_residual),
        .o_store_learning_write_code_enable(
            o_store_learning_write_code_enable),
        .o_store_learning_write_residual_enable(
            o_store_learning_write_residual_enable),
        .o_store_forward_read_valid(o_store_forward_read_valid),
        .i_store_forward_read_ready(i_store_forward_read_ready),
        .o_store_forward_read_addr(o_store_forward_read_addr),
        .o_store_forward_read_tag(o_store_forward_read_tag),
        .i_store_forward_response_valid(i_store_forward_response_valid),
        .o_store_forward_response_ready(o_store_forward_response_ready),
        .i_store_forward_response_tag(i_store_forward_response_tag),
        .i_store_forward_response_code(i_store_forward_response_code),
        .i_store_forward_response_error(i_store_forward_response_error),
        .i_store_protocol_error(i_store_protocol_error),
        .i_classifier_prime_valid(backend_prime_valid),
        .o_classifier_prime_ready(backend_prime_ready),
        .i_classifier_pair(i_classifier_pair),
        .i_initial_version(i_initial_version),
        .i_classifier_parameter_base(i_classifier_parameter_base),
        .i_classifier_feature_count(i_classifier_feature_count),
        .i_classifier_class_count(i_classifier_class_count),
        .o_classifier_prime_done(backend_prime_done),
        .i_commit_publish_valid(backend_commit_valid),
        .o_commit_publish_ready(backend_commit_ready),
        .o_commit_publish_done(backend_commit_done),
        .o_conv_cache_invalidate(backend_conv_invalidate),
        .i_learning_read_valid(backend_learning_read_valid),
        .o_learning_read_ready(backend_learning_read_ready),
        .i_learning_read_addr(i_learning_read_addr),
        .i_learning_read_tag(i_learning_read_tag),
        .o_learning_response_valid(o_learning_response_valid),
        .i_learning_response_ready(i_learning_response_ready),
        .o_learning_response_tag(o_learning_response_tag),
        .o_learning_response_code(o_learning_response_code),
        .o_learning_response_residual(o_learning_response_residual),
        .o_learning_response_error(o_learning_response_error),
        .i_learning_write_valid(backend_learning_write_valid),
        .o_learning_write_ready(backend_learning_write_ready),
        .i_learning_write_addr(i_learning_write_addr),
        .i_learning_write_code(i_learning_write_code),
        .i_learning_write_residual(i_learning_write_residual),
        .i_learning_write_code_enable(i_learning_write_code_enable),
        .i_learning_write_residual_enable(
            i_learning_write_residual_enable),
        .i_conv_forward_read_valid(refill_forward_read_valid),
        .o_conv_forward_read_ready(refill_forward_read_ready),
        .i_conv_forward_read_addr(refill_forward_read_addr),
        .i_conv_forward_read_tag(refill_forward_read_tag),
        .o_conv_forward_response_valid(refill_forward_response_valid),
        .i_conv_forward_response_ready(refill_forward_response_ready),
        .o_conv_forward_response_tag(refill_forward_response_tag),
        .o_conv_forward_response_code(refill_forward_response_code),
        .o_conv_forward_response_error(refill_forward_response_error),
        .i_array_idle(i_array_idle && !refill_busy),
        .o_weight_fill_valid(backend_fill_valid),
        .i_weight_fill_ready(backend_fill_ready),
        .o_weight_fill_tile_flat(backend_fill_tile_flat),
        .o_weight_fill_bank(backend_fill_bank),
        .o_weight_fill_addr_flat(backend_fill_addr_flat),
        .o_weight_fill_data_flat(backend_fill_data_flat),
        .o_cache_valid(o_classifier_cache_valid),
        .o_active_classifier_pair(o_active_classifier_pair),
        .o_active_version(o_active_version), .o_busy(backend_busy),
        .o_sticky_error(backend_error)
    );

    unified_scnn_h40_conv_context_refill_v1 #(
        .PARAM_DEPTH(PARAM_DEPTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH),
        .MAX_INPUT_CHANNELS(MAX_INPUT_CHANNELS),
        .ARRAY_WEIGHT_ADDR_WIDTH(ARRAY_WEIGHT_ADDR_WIDTH),
        .INPUT_CHANNEL_COUNT_WIDTH(INPUT_CHANNEL_COUNT_WIDTH)
    ) u_conv_refill (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_context_valid(refill_context_valid),
        .o_context_ready(refill_context_ready),
        .i_context_load_weights(i_conv_context_load_weights),
        .i_output_channel_base(i_conv_output_channel_base),
        .i_output_channel_count(i_conv_output_channel_count),
        .i_input_channel_count(i_conv_input_channel_count),
        .i_context_pair(i_conv_context_pair),
        .i_reserved_pair_valid(o_classifier_cache_valid),
        .i_reserved_pair(o_active_classifier_pair),
        .i_conv_parameter_base(i_conv_parameter_base),
        .i_version(i_conv_context_version),
        .i_refill_safe(i_array_idle && !prime_inflight_q &&
            !commit_inflight_q && !publication_ready_q),
        .o_forward_read_valid(refill_forward_read_valid),
        .i_forward_read_ready(refill_forward_read_ready),
        .o_forward_read_addr(refill_forward_read_addr),
        .o_forward_read_tag(refill_forward_read_tag),
        .i_forward_response_valid(refill_forward_response_valid),
        .o_forward_response_ready(refill_forward_response_ready),
        .i_forward_response_tag(refill_forward_response_tag),
        .i_forward_response_code(refill_forward_response_code),
        .i_forward_response_error(refill_forward_response_error),
        .o_weight_fill_valid(refill_fill_valid),
        .i_weight_fill_ready(refill_fill_ready),
        .o_weight_fill_tile_flat(refill_fill_tile_flat),
        .o_weight_fill_bank(refill_fill_bank),
        .o_weight_fill_addr_flat(refill_fill_addr_flat),
        .o_weight_fill_data_flat(refill_fill_data_flat),
        .o_busy(refill_busy), .o_refill_done(refill_done),
        .o_sticky_error(refill_error)
    );

    assign fill_overlap = (|backend_fill_valid) && (|refill_fill_valid);
    always_comb begin
        o_weight_fill_valid = '0;
        o_weight_fill_tile_flat = '0;
        o_weight_fill_bank = '0;
        o_weight_fill_addr_flat = '0;
        o_weight_fill_data_flat = '0;
        backend_fill_ready = '0;
        refill_fill_ready = '0;
        if (|backend_fill_valid) begin
            o_weight_fill_valid = backend_fill_valid;
            o_weight_fill_tile_flat = backend_fill_tile_flat;
            o_weight_fill_bank = backend_fill_bank;
            o_weight_fill_addr_flat = backend_fill_addr_flat;
            o_weight_fill_data_flat = backend_fill_data_flat;
            backend_fill_ready = i_weight_fill_ready;
        end else begin
            o_weight_fill_valid = refill_fill_valid;
            o_weight_fill_tile_flat = refill_fill_tile_flat;
            o_weight_fill_bank = refill_fill_bank;
            o_weight_fill_addr_flat = refill_fill_addr_flat;
            o_weight_fill_data_flat = refill_fill_data_flat;
            refill_fill_ready = i_weight_fill_ready;
        end
    end

    assign o_conv_cache_valid = conv_cache_valid_q && !i_cache_invalidate;
    assign o_conv_cached_output_base = conv_output_base_q;
    assign o_conv_cached_output_count = conv_output_count_q;
    assign o_conv_cached_version = conv_version_q;
    // Only cache publication/refill activity owns the physical fill fabric.
    // Learning-store traffic and the commit publication wait must not block a
    // different topology from refilling the shared array cache.
    assign o_weight_fill_busy = prime_inflight_q || commit_inflight_q ||
        refill_busy;
    assign o_busy = backend_busy || refill_busy || prime_inflight_q ||
        commit_inflight_q || publication_ready_q;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            prime_inflight_q <= 1'b0;
            commit_inflight_q <= 1'b0;
            publication_ready_q <= 1'b0;
            conv_cache_valid_q <= 1'b0;
            conv_output_base_q <= '0;
            conv_output_count_q <= '0;
            conv_input_count_q <= '0;
            conv_pair_q <= 1'b0;
            conv_parameter_base_q <= '0;
            conv_version_q <= '0;
            o_conv_cache_invalidate <= 1'b0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            prime_inflight_q <= 1'b0;
            commit_inflight_q <= 1'b0;
            publication_ready_q <= 1'b0;
            conv_cache_valid_q <= 1'b0;
            o_conv_cache_invalidate <= 1'b0;
        end else begin
            o_conv_cache_invalidate <= 1'b0;
            if (i_clear_error && !o_busy)
                o_sticky_error <= 1'b0;
            if (backend_error || refill_error || fill_overlap ||
                    invalid_context_request ||
                    (i_cache_invalidate && refill_busy))
                o_sticky_error <= 1'b1;

            if (prime_fire)
                prime_inflight_q <= 1'b1;
            if (backend_prime_done)
                prime_inflight_q <= 1'b0;

            if (backend_commit_fire)
                commit_inflight_q <= 1'b1;
            if (backend_commit_done) begin
                commit_inflight_q <= 1'b0;
                publication_ready_q <= 1'b1;
            end
            if (i_commit_pulse) begin
                if (!publication_ready_q)
                    o_sticky_error <= 1'b1;
                publication_ready_q <= 1'b0;
            end
            if (commit_inflight_q && !i_commit_pending)
                o_sticky_error <= 1'b1;

            if (refill_done && !o_sticky_error && !backend_error &&
                    !refill_error && !i_cache_invalidate &&
                    !backend_conv_invalidate) begin
                conv_cache_valid_q <= 1'b1;
                conv_output_base_q <= i_conv_output_channel_base;
                conv_output_count_q <= i_conv_output_channel_count;
                conv_input_count_q <= i_conv_input_channel_count;
                conv_pair_q <= i_conv_context_pair;
                conv_parameter_base_q <= i_conv_parameter_base;
                conv_version_q <= i_conv_context_version;
            end
            if (backend_conv_invalidate || i_cache_invalidate) begin
                conv_cache_valid_q <= 1'b0;
                o_conv_cache_invalidate <= 1'b1;
            end
        end
    end

    initial begin
        if (MAX_CLASSES > 16 || PARAM_ADDR_WIDTH < 1 ||
            ARRAY_WEIGHT_ADDR_WIDTH != 11)
            $error("SCNN selected-layer parameter plane geometry is invalid");
    end
endmodule

`default_nettype wire
