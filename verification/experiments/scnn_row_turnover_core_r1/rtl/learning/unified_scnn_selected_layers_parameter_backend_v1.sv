`timescale 1ns/1ps
`default_nettype none

// One persistent parameter image backs both selected SCNN layers. Learning
// reaches it through one RMW port. The classifier matrix is repacked into its
// resident H40 cache pair, while the Conv workset manager reads updated OIHW
// values through the forward port and refills output-channel waves on demand.
module unified_scnn_selected_layers_parameter_backend_v1 #(
    parameter bit USE_INTERNAL_STORE = 1'b1,
    parameter int PARAM_DEPTH = 32768,
    parameter int PARAM_ADDR_WIDTH = $clog2(PARAM_DEPTH),
    parameter int TAG_WIDTH = 16,
    parameter int VERSION_WIDTH = 16,
    parameter int MAX_CLASSIFIER_FEATURES = 2048,
    parameter int MAX_CLASSES = 11,
    parameter int ARRAY_WEIGHT_ADDR_WIDTH = 11,
    parameter int FEATURE_COUNT_WIDTH =
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

    input  wire logic                              i_classifier_prime_valid,
    output logic                                   o_classifier_prime_ready,
    input  wire logic                              i_classifier_pair,
    input  wire logic [VERSION_WIDTH-1:0]           i_initial_version,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_classifier_parameter_base,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0]     i_classifier_feature_count,
    input  wire logic [CLASS_COUNT_WIDTH-1:0]       i_classifier_class_count,
    output logic                                   o_classifier_prime_done,

    // Accepted only after the selected Conv8 and classifier learners have
    // completed all RMW traffic. The resident classifier pair is refreshed in
    // place while inference is held; version visibility changes afterwards.
    input  wire logic                              i_commit_publish_valid,
    output logic                                   o_commit_publish_ready,
    output logic                                   o_commit_publish_done,
    output logic                                   o_conv_cache_invalidate,

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

    // Conv context refill reads the same persistent code image. Only one read
    // may be outstanding; requests are held while learning or publication owns
    // the store so train/infer serialization is explicit at this boundary.
    input  wire logic                              i_conv_forward_read_valid,
    output logic                                   o_conv_forward_read_ready,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_conv_forward_read_addr,
    input  wire logic [TAG_WIDTH-1:0]               i_conv_forward_read_tag,
    output logic                                   o_conv_forward_response_valid,
    input  wire logic                              i_conv_forward_response_ready,
    output logic [TAG_WIDTH-1:0]                    o_conv_forward_response_tag,
    output logic signed [15:0]                     o_conv_forward_response_code,
    output logic                                   o_conv_forward_response_error,

    // Optional external parameter-store boundary. In external mode this
    // backend remains the SCNN cache/publication owner, but parameter bits
    // live in the system-wide store shared with FC and SRNN clients.
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

    input  wire logic                              i_array_idle,
    output logic [3:0]                             o_weight_fill_valid,
    input  wire logic [3:0]                        i_weight_fill_ready,
    output logic [7:0]                             o_weight_fill_tile_flat,
    output logic [3:0]                             o_weight_fill_bank,
    output logic [4*ARRAY_WEIGHT_ADDR_WIDTH-1:0]    o_weight_fill_addr_flat,
    output logic [799:0]                           o_weight_fill_data_flat,

    output logic                                   o_cache_valid,
    output logic                                   o_active_classifier_pair,
    output logic [VERSION_WIDTH-1:0]               o_active_version,
    output logic                                   o_busy,
    output logic                                   o_sticky_error
);
    typedef enum logic [1:0] {
        BACKEND_IDLE,
        BACKEND_PRIME,
        BACKEND_COMMIT
    } backend_state_t;

    backend_state_t state_q;
    logic configured_q;
    logic cache_valid_q;
    logic classifier_pair_q;
    logic [VERSION_WIDTH-1:0] active_version_q;
    logic [PARAM_ADDR_WIDTH-1:0] classifier_parameter_base_q;
    logic [FEATURE_COUNT_WIDTH-1:0] classifier_feature_count_q;
    logic [CLASS_COUNT_WIDTH-1:0] classifier_class_count_q;

    logic publisher_start_valid;
    logic publisher_start_ready;
    logic publisher_done;
    logic publisher_busy;
    logic publisher_error;
    logic publisher_pair;
    logic [VERSION_WIDTH-1:0] publisher_version;
    logic publisher_forward_read_valid;
    logic publisher_forward_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] publisher_forward_read_addr;
    logic [TAG_WIDTH-1:0] publisher_forward_read_tag;
    logic publisher_forward_response_valid;
    logic publisher_forward_response_ready;

    logic store_load_valid;
    logic store_load_ready;
    logic store_forward_read_valid;
    logic store_forward_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] store_forward_read_addr;
    logic [TAG_WIDTH-1:0] store_forward_read_tag;
    logic store_forward_response_valid;
    logic store_forward_response_ready;
    logic [TAG_WIDTH-1:0] store_forward_response_tag;
    logic signed [15:0] store_forward_response_code;
    logic store_forward_response_error;
    logic store_error;
    logic learning_read_ready;
    logic learning_response_valid;
    logic [TAG_WIDTH-1:0] learning_response_tag;
    logic signed [15:0] learning_response_code;
    logic signed [15:0] learning_response_residual;
    logic learning_response_error;
    logic learning_write_ready;
    logic external_forward_outstanding_q;
    logic learning_transaction_q;

    logic start_quiescent;
    logic prime_fire;
    logic cold_prime_fire;
    logic commit_fire;
    logic publisher_start_fire;
    logic conv_forward_fire;
    logic conv_forward_response_fire;
    logic conv_forward_slot;
    logic learning_read_fire;
    logic learning_write_fire;

    assign start_quiescent = !learning_transaction_q &&
        !external_forward_outstanding_q && !i_learning_read_valid &&
        !i_learning_write_valid && !i_conv_forward_read_valid;
    assign o_classifier_prime_ready = state_q == BACKEND_IDLE &&
        !cache_valid_q && start_quiescent && publisher_start_ready &&
        i_array_idle && !o_sticky_error && !i_cache_invalidate;
    assign o_commit_publish_ready = state_q == BACKEND_IDLE &&
        cache_valid_q && start_quiescent && publisher_start_ready &&
        i_array_idle && !o_sticky_error && !i_classifier_prime_valid &&
        !i_cache_invalidate;
    assign prime_fire = i_classifier_prime_valid &&
        o_classifier_prime_ready;
    assign cold_prime_fire = prime_fire && !configured_q;
    assign commit_fire = i_commit_publish_valid && o_commit_publish_ready;
    assign publisher_start_valid = prime_fire || commit_fire;
    assign publisher_start_fire = publisher_start_valid &&
        publisher_start_ready;

    assign store_load_valid = i_parameter_load_valid &&
        state_q == BACKEND_IDLE && !configured_q &&
        !external_forward_outstanding_q && !learning_transaction_q;
    assign o_parameter_load_ready = store_load_ready &&
        state_q == BACKEND_IDLE && !configured_q &&
        !external_forward_outstanding_q && !learning_transaction_q;

    assign learning_read_fire = i_learning_read_valid &&
        o_learning_read_ready;
    assign learning_write_fire = i_learning_write_valid &&
        o_learning_write_ready;

    assign conv_forward_slot = !external_forward_outstanding_q ||
        conv_forward_response_fire;
    assign store_forward_read_valid = publisher_busy ?
        publisher_forward_read_valid :
        (i_conv_forward_read_valid && state_q == BACKEND_IDLE &&
         !learning_transaction_q && !i_learning_read_valid &&
         !i_learning_write_valid && conv_forward_slot);
    assign store_forward_read_addr = publisher_busy ?
        publisher_forward_read_addr : i_conv_forward_read_addr;
    assign store_forward_read_tag = publisher_busy ?
        publisher_forward_read_tag : i_conv_forward_read_tag;
    assign publisher_forward_read_ready = publisher_busy &&
        store_forward_read_ready;
    assign o_conv_forward_read_ready = !publisher_busy &&
        state_q == BACKEND_IDLE && !learning_transaction_q &&
        !i_learning_read_valid && !i_learning_write_valid &&
        conv_forward_slot && store_forward_read_ready;
    assign conv_forward_fire = i_conv_forward_read_valid &&
        o_conv_forward_read_ready;

    assign publisher_forward_response_valid = publisher_busy &&
        store_forward_response_valid;
    assign o_conv_forward_response_valid = !publisher_busy &&
        external_forward_outstanding_q && store_forward_response_valid;
    assign o_conv_forward_response_tag = store_forward_response_tag;
    assign o_conv_forward_response_code = store_forward_response_code;
    assign o_conv_forward_response_error = store_forward_response_error;
    assign store_forward_response_ready = publisher_busy ?
        publisher_forward_response_ready :
        (external_forward_outstanding_q &&
         i_conv_forward_response_ready);
    assign conv_forward_response_fire = o_conv_forward_response_valid &&
        i_conv_forward_response_ready;

    assign o_cache_valid = cache_valid_q && !i_cache_invalidate;
    assign o_active_classifier_pair = classifier_pair_q;
    assign o_active_version = active_version_q;
    assign o_busy = state_q != BACKEND_IDLE || publisher_busy ||
        external_forward_outstanding_q || learning_transaction_q;

    assign o_learning_read_ready = learning_read_ready;
    assign o_learning_response_valid = learning_response_valid;
    assign o_learning_response_tag = learning_response_tag;
    assign o_learning_response_code = learning_response_code;
    assign o_learning_response_residual = learning_response_residual;
    assign o_learning_response_error = learning_response_error;
    assign o_learning_write_ready = learning_write_ready;

    generate
        if (USE_INTERNAL_STORE) begin : g_internal_store
            unified_tp_shared_forward_rmw_store #(
                .DEPTH(PARAM_DEPTH), .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
                .TAG_WIDTH(TAG_WIDTH)
            ) u_parameter_store (
                .i_clk(i_clk), .i_rst_n(i_rst_n),
                .i_clear_error(i_clear_error),
                .i_load_valid(store_load_valid),
                .o_load_ready(store_load_ready),
                .i_load_addr(i_parameter_load_addr),
                .i_load_code(i_parameter_load_code),
                .i_load_residual(i_parameter_load_residual),
                .i_learning_read_valid(i_learning_read_valid),
                .o_learning_read_ready(learning_read_ready),
                .i_learning_read_addr(i_learning_read_addr),
                .i_learning_read_tag(i_learning_read_tag),
                .o_learning_response_valid(learning_response_valid),
                .i_learning_response_ready(i_learning_response_ready),
                .o_learning_response_tag(learning_response_tag),
                .o_learning_response_code(learning_response_code),
                .o_learning_response_residual(learning_response_residual),
                .o_learning_response_error(learning_response_error),
                .i_learning_write_valid(i_learning_write_valid),
                .o_learning_write_ready(learning_write_ready),
                .i_learning_write_addr(i_learning_write_addr),
                .i_learning_write_code(i_learning_write_code),
                .i_learning_write_residual(i_learning_write_residual),
                .i_learning_write_code_enable(i_learning_write_code_enable),
                .i_learning_write_residual_enable(
                    i_learning_write_residual_enable),
                .i_forward_read_valid(store_forward_read_valid),
                .o_forward_read_ready(store_forward_read_ready),
                .i_forward_read_addr(store_forward_read_addr),
                .i_forward_read_tag(store_forward_read_tag),
                .o_forward_response_valid(store_forward_response_valid),
                .i_forward_response_ready(store_forward_response_ready),
                .o_forward_response_tag(store_forward_response_tag),
                .o_forward_response_code(store_forward_response_code),
                .o_forward_response_error(store_forward_response_error),
                .o_protocol_error(store_error)
            );

            always_comb begin
                o_store_load_valid = 1'b0;
                o_store_load_addr = '0;
                o_store_load_code = '0;
                o_store_load_residual = '0;
                o_store_learning_read_valid = 1'b0;
                o_store_learning_read_addr = '0;
                o_store_learning_read_tag = '0;
                o_store_learning_response_ready = 1'b0;
                o_store_learning_write_valid = 1'b0;
                o_store_learning_write_addr = '0;
                o_store_learning_write_code = '0;
                o_store_learning_write_residual = '0;
                o_store_learning_write_code_enable = 1'b0;
                o_store_learning_write_residual_enable = 1'b0;
                o_store_forward_read_valid = 1'b0;
                o_store_forward_read_addr = '0;
                o_store_forward_read_tag = '0;
                o_store_forward_response_ready = 1'b0;
            end
        end else begin : g_external_store
            always_comb begin
                store_load_ready = i_store_load_ready;
                learning_read_ready = i_store_learning_read_ready;
                learning_response_valid =
                    i_store_learning_response_valid;
                learning_response_tag = i_store_learning_response_tag;
                learning_response_code = i_store_learning_response_code;
                learning_response_residual =
                    i_store_learning_response_residual;
                learning_response_error =
                    i_store_learning_response_error;
                learning_write_ready = i_store_learning_write_ready;
                store_forward_read_ready = i_store_forward_read_ready;
                store_forward_response_valid =
                    i_store_forward_response_valid;
                store_forward_response_tag = i_store_forward_response_tag;
                store_forward_response_code = i_store_forward_response_code;
                store_forward_response_error =
                    i_store_forward_response_error;
                store_error = i_store_protocol_error;

                o_store_load_valid = store_load_valid;
                o_store_load_addr = i_parameter_load_addr;
                o_store_load_code = i_parameter_load_code;
                o_store_load_residual = i_parameter_load_residual;
                o_store_learning_read_valid = i_learning_read_valid;
                o_store_learning_read_addr = i_learning_read_addr;
                o_store_learning_read_tag = i_learning_read_tag;
                o_store_learning_response_ready =
                    i_learning_response_ready;
                o_store_learning_write_valid = i_learning_write_valid;
                o_store_learning_write_addr = i_learning_write_addr;
                o_store_learning_write_code = i_learning_write_code;
                o_store_learning_write_residual =
                    i_learning_write_residual;
                o_store_learning_write_code_enable =
                    i_learning_write_code_enable;
                o_store_learning_write_residual_enable =
                    i_learning_write_residual_enable;
                o_store_forward_read_valid = store_forward_read_valid;
                o_store_forward_read_addr = store_forward_read_addr;
                o_store_forward_read_tag = store_forward_read_tag;
                o_store_forward_response_ready =
                    store_forward_response_ready;
            end
        end
    endgenerate

    unified_scnn_classifier_cache_refill_publisher_v1 #(
        .MAX_FEATURES(MAX_CLASSIFIER_FEATURES),
        .MAX_CLASSES(MAX_CLASSES), .PARAM_DEPTH(PARAM_DEPTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH),
        .ARRAY_WEIGHT_ADDR_WIDTH(ARRAY_WEIGHT_ADDR_WIDTH),
        .FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
        .CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH)
    ) u_classifier_publisher (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_start_valid(publisher_start_valid),
        .o_start_ready(publisher_start_ready),
        .i_target_pair(cold_prime_fire ? i_classifier_pair : classifier_pair_q),
        .i_parameter_base(cold_prime_fire ? i_classifier_parameter_base :
            classifier_parameter_base_q),
        .i_feature_count(cold_prime_fire ? i_classifier_feature_count :
            classifier_feature_count_q),
        .i_class_count(cold_prime_fire ? i_classifier_class_count :
            classifier_class_count_q),
        .i_version(cold_prime_fire ? i_initial_version :
            (prime_fire ? active_version_q : active_version_q + 1'b1)),
        .i_array_idle(i_array_idle),
        .o_forward_read_valid(publisher_forward_read_valid),
        .i_forward_read_ready(publisher_forward_read_ready),
        .o_forward_read_addr(publisher_forward_read_addr),
        .o_forward_read_tag(publisher_forward_read_tag),
        .i_forward_response_valid(publisher_forward_response_valid),
        .o_forward_response_ready(publisher_forward_response_ready),
        .i_forward_response_tag(store_forward_response_tag),
        .i_forward_response_code(store_forward_response_code),
        .i_forward_response_error(store_forward_response_error),
        .o_weight_fill_valid(o_weight_fill_valid),
        .i_weight_fill_ready(i_weight_fill_ready),
        .o_weight_fill_tile_flat(o_weight_fill_tile_flat),
        .o_weight_fill_bank(o_weight_fill_bank),
        .o_weight_fill_addr_flat(o_weight_fill_addr_flat),
        .o_weight_fill_data_flat(o_weight_fill_data_flat),
        .o_done(publisher_done), .o_published_pair(publisher_pair),
        .o_published_version(publisher_version),
        .o_busy(publisher_busy), .o_sticky_error(publisher_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            external_forward_outstanding_q <= 1'b0;
            learning_transaction_q <= 1'b0;
        end else if (i_abort) begin
            external_forward_outstanding_q <= 1'b0;
            learning_transaction_q <= 1'b0;
        end else begin
            if (conv_forward_fire)
                external_forward_outstanding_q <= 1'b1;
            else if (conv_forward_response_fire)
                external_forward_outstanding_q <= 1'b0;

            if (learning_read_fire)
                learning_transaction_q <= 1'b1;
            else if (learning_write_fire)
                learning_transaction_q <= 1'b0;
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= BACKEND_IDLE;
            configured_q <= 1'b0;
            cache_valid_q <= 1'b0;
            classifier_pair_q <= 1'b0;
            active_version_q <= '0;
            classifier_parameter_base_q <= '0;
            classifier_feature_count_q <= '0;
            classifier_class_count_q <= '0;
            o_classifier_prime_done <= 1'b0;
            o_commit_publish_done <= 1'b0;
            o_conv_cache_invalidate <= 1'b0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= BACKEND_IDLE;
            configured_q <= 1'b0;
            cache_valid_q <= 1'b0;
            o_classifier_prime_done <= 1'b0;
            o_commit_publish_done <= 1'b0;
            o_conv_cache_invalidate <= 1'b0;
        end else begin
            o_classifier_prime_done <= 1'b0;
            o_commit_publish_done <= 1'b0;
            o_conv_cache_invalidate <= 1'b0;

            if (i_clear_error && state_q == BACKEND_IDLE &&
                    !publisher_busy)
                o_sticky_error <= 1'b0;
            if (publisher_error || store_error)
                o_sticky_error <= 1'b1;

            case (state_q)
                BACKEND_IDLE: begin
                    if (publisher_start_fire) begin
                        if (prime_fire) begin
                            if (cold_prime_fire) begin
                                classifier_parameter_base_q <=
                                    i_classifier_parameter_base;
                                classifier_feature_count_q <=
                                    i_classifier_feature_count;
                                classifier_class_count_q <=
                                    i_classifier_class_count;
                            end
                            state_q <= BACKEND_PRIME;
                        end else begin
                            state_q <= BACKEND_COMMIT;
                        end
                    end
                end

                BACKEND_PRIME: begin
                    if (publisher_done) begin
                        if (!i_cache_invalidate && !o_sticky_error &&
                                !publisher_error && !store_error) begin
                            configured_q <= 1'b1;
                            cache_valid_q <= 1'b1;
                            classifier_pair_q <= publisher_pair;
                            active_version_q <= publisher_version;
                            o_classifier_prime_done <= 1'b1;
                        end
                        state_q <= BACKEND_IDLE;
                    end
                end

                BACKEND_COMMIT: begin
                    if (publisher_done) begin
                        if (!i_cache_invalidate && !o_sticky_error &&
                                !publisher_error && !store_error) begin
                            classifier_pair_q <= publisher_pair;
                            active_version_q <= publisher_version;
                            o_commit_publish_done <= 1'b1;
                            o_conv_cache_invalidate <= 1'b1;
                        end
                        state_q <= BACKEND_IDLE;
                    end
                end

                default: begin
                    state_q <= BACKEND_IDLE;
                    o_sticky_error <= 1'b1;
                end
            endcase

            // Eviction affects residency, not the configured model or store.
            if (i_cache_invalidate) begin
                cache_valid_q <= 1'b0;
                o_conv_cache_invalidate <= 1'b1;
                if (state_q != BACKEND_IDLE)
                    o_sticky_error <= 1'b1;
            end
        end
    end

    initial begin
        if (MAX_CLASSES > 16 || MAX_CLASSIFIER_FEATURES > 13872 ||
            PARAM_ADDR_WIDTH < 1)
            $error("Selected-layer parameter backend geometry is invalid");
    end
endmodule

`default_nettype wire
