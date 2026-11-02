`timescale 1ns/1ps
`default_nettype none

// Locks one physical array to either a direct FC/SRNN transaction or one
// complete SCNN layer. SCNN ownership is retained after its scheduler finishes
// until the attached observation/learning sink has drained its final record.
module unified_h40_array16_topology_owner_v1 (
    input  wire logic        i_clk,
    input  wire logic        i_rst_n,
    input  wire logic        i_abort,
    input  wire logic        i_clear_error,

    input  wire logic        i_maintenance_start_valid,
    output logic             o_maintenance_start_ready,
    input  wire logic        i_maintenance_release_valid,
    output logic             o_maintenance_release_ready,
    input  wire logic        i_maintenance_idle,

    input  wire logic        i_direct_start_valid,
    output logic             o_direct_start_ready,
    input  wire logic [1:0]  i_direct_start_mode,
    input  wire logic        i_direct_command_valid,
    output logic             o_direct_command_ready,
    input  wire logic [1:0]  i_direct_mode,
    input  wire logic        i_direct_first,
    input  wire logic        i_direct_last,

    input  wire logic        i_scnn_start_valid,
    output logic             o_scnn_start_ready,
    output logic             o_scnn_client_start_valid,
    input  wire logic        i_scnn_client_start_ready,
    input  wire logic        i_scnn_client_done,
    input  wire logic        i_scnn_observation_sink_idle,

    input  wire logic        i_array_command_ready,
    input  wire logic        i_array_command_done,
    input  wire logic        i_array_idle,
    input  wire logic        i_direct_observation_sink_idle,

    output logic             o_select_direct,
    output logic             o_select_scnn,
    output logic             o_select_maintenance,
    output logic             o_direct_done,
    output logic             o_scnn_done,
    output logic             o_maintenance_done,
    output logic [1:0]       o_active_owner,
    output logic             o_busy,
    output logic             o_idle,
    output logic             o_sticky_error
);
    localparam logic [1:0] OWNER_NONE = 2'd0;
    localparam logic [1:0] OWNER_DIRECT = 2'd1;
    localparam logic [1:0] OWNER_SCNN = 2'd2;
    localparam logic [1:0] OWNER_MAINTENANCE = 2'd3;
    localparam logic [1:0] MODE_SCNN = 2'd2;

    logic [1:0] owner_q;
    logic direct_last_inflight_q;
    logic direct_done_seen_q;
    logic scnn_done_seen_q;
    logic direct_legal_at_idle;
    logic direct_fire;
    logic direct_start_fire;
    logic legacy_direct_acquire;
    logic scnn_start_fire;
    logic maintenance_start_fire;
    logic maintenance_release_fire;
    logic acquisition_idle;

    always_comb begin
        // The direct observation path is deliberately armed before its
        // FC/SRNN owner request. Its drain is a release condition below, not
        // an acquisition condition; requiring idle here would deadlock that
        // configure-then-acquire protocol.
        acquisition_idle = owner_q == OWNER_NONE && i_array_idle &&
            i_scnn_observation_sink_idle && !o_sticky_error && !i_abort;
        direct_legal_at_idle = i_direct_first &&
            i_direct_mode != MODE_SCNN;

        // Maintenance commit has boundary priority, followed by SCNN and then
        // FC/SRNN. Once accepted, each owner retains the complete lifecycle.
        // Maintenance requests are allowed to queue while ownership is being
        // acquired. Requiring the request side to be idle here would create a
        // valid/ready deadlock for an automatic writeback publisher. Full
        // maintenance quiescence is required only at release.
        o_maintenance_start_ready = acquisition_idle;
        maintenance_start_fire = i_maintenance_start_valid &&
            o_maintenance_start_ready;

        o_scnn_client_start_valid = acquisition_idle &&
            !i_maintenance_start_valid && i_scnn_start_valid;
        o_scnn_start_ready = acquisition_idle &&
            !i_maintenance_start_valid && i_scnn_client_start_ready;
        scnn_start_fire = o_scnn_client_start_valid &&
            i_scnn_client_start_ready;

        o_direct_start_ready = acquisition_idle &&
            !i_maintenance_start_valid && !i_scnn_start_valid &&
            i_direct_start_mode != MODE_SCNN;
        direct_start_fire = i_direct_start_valid && o_direct_start_ready;
        legacy_direct_acquire = acquisition_idle &&
            !i_maintenance_start_valid && !i_scnn_start_valid &&
            !i_direct_start_valid && i_direct_command_valid &&
            direct_legal_at_idle;

        o_select_scnn = owner_q == OWNER_SCNN;
        o_select_maintenance = owner_q == OWNER_MAINTENANCE;
        o_select_direct = owner_q == OWNER_DIRECT ||
            legacy_direct_acquire;
        o_direct_command_ready = o_select_direct &&
            i_array_command_ready && !direct_done_seen_q;
        direct_fire = i_direct_command_valid &&
            o_direct_command_ready;

        o_maintenance_release_ready =
            owner_q == OWNER_MAINTENANCE && i_maintenance_idle;
        maintenance_release_fire = i_maintenance_release_valid &&
            o_maintenance_release_ready;

        o_active_owner = owner_q;
        o_busy = owner_q != OWNER_NONE;
        o_idle = acquisition_idle;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            owner_q <= OWNER_NONE;
            direct_last_inflight_q <= 1'b0;
            direct_done_seen_q <= 1'b0;
            scnn_done_seen_q <= 1'b0;
            o_direct_done <= 1'b0;
            o_scnn_done <= 1'b0;
            o_maintenance_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            owner_q <= OWNER_NONE;
            direct_last_inflight_q <= 1'b0;
            direct_done_seen_q <= 1'b0;
            scnn_done_seen_q <= 1'b0;
            o_direct_done <= 1'b0;
            o_scnn_done <= 1'b0;
            o_maintenance_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else begin
            o_direct_done <= 1'b0;
            o_scnn_done <= 1'b0;
            o_maintenance_done <= 1'b0;

            if (i_clear_error && owner_q == OWNER_NONE && i_array_idle &&
                i_scnn_observation_sink_idle)
                o_sticky_error <= 1'b0;

            if (owner_q == OWNER_NONE && i_direct_command_valid &&
                !i_maintenance_start_valid && !i_scnn_start_valid &&
                !i_direct_start_valid &&
                !direct_legal_at_idle)
                o_sticky_error <= 1'b1;
            if (owner_q == OWNER_NONE && i_direct_start_valid &&
                !i_scnn_start_valid && i_direct_start_mode == MODE_SCNN)
                o_sticky_error <= 1'b1;

            case (owner_q)
                OWNER_NONE: begin
                    direct_last_inflight_q <= 1'b0;
                    direct_done_seen_q <= 1'b0;
                    scnn_done_seen_q <= 1'b0;
                    if (maintenance_start_fire)
                        owner_q <= OWNER_MAINTENANCE;
                    else if (scnn_start_fire)
                        owner_q <= OWNER_SCNN;
                    else if (direct_start_fire)
                        owner_q <= OWNER_DIRECT;
                    else if (direct_fire) begin
                        owner_q <= OWNER_DIRECT;
                        direct_last_inflight_q <= i_direct_last;
                    end
                end

                OWNER_DIRECT: begin
                    if (direct_fire)
                        direct_last_inflight_q <= i_direct_last;
                    if (i_array_command_done && direct_last_inflight_q)
                        direct_done_seen_q <= 1'b1;
                    if ((direct_done_seen_q ||
                         (i_array_command_done &&
                          direct_last_inflight_q)) &&
                        i_direct_observation_sink_idle) begin
                        owner_q <= OWNER_NONE;
                        direct_last_inflight_q <= 1'b0;
                        direct_done_seen_q <= 1'b0;
                        o_direct_done <= 1'b1;
                    end
                end

                OWNER_SCNN: begin
                    if (i_scnn_client_done)
                        scnn_done_seen_q <= 1'b1;
                    if ((scnn_done_seen_q || i_scnn_client_done) &&
                        i_scnn_observation_sink_idle) begin
                        owner_q <= OWNER_NONE;
                        scnn_done_seen_q <= 1'b0;
                        o_scnn_done <= 1'b1;
                    end
                end

                OWNER_MAINTENANCE: begin
                    if (maintenance_release_fire) begin
                        owner_q <= OWNER_NONE;
                        o_maintenance_done <= 1'b1;
                    end
                end

                default: begin
                    owner_q <= OWNER_NONE;
                    direct_last_inflight_q <= 1'b0;
                    direct_done_seen_q <= 1'b0;
                    scnn_done_seen_q <= 1'b0;
                    o_sticky_error <= 1'b1;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
