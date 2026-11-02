`timescale 1ns/1ps
`default_nettype none

// Passive accounting, excluded from production synthesis. Buckets describe
// priority-ordered observed activity, not mutually exclusive physical causes.
module supplement_h40_activity_monitor (
    input wire logic clk, rst_n, aborting, idle,
    input wire logic accumulator_valid, accumulator_ready,
    input wire logic [39:0] lane_mask, enable0, enable1, enable2, enable3,
    input wire logic [1:0] mode,
    input wire logic issue_valid, issue_ready,
    input wire logic region_valid, region_ready,
    input wire logic observation_valid, observation_ready,
    input wire logic fill_valid, fill_ready
);
    longint unsigned cycles = 0, issue_fires = 0, fill_fires = 0;
    longint unsigned contributions = 0, lane_fires = 0, enabled_terms = 0;
    longint unsigned bucket_abort = 0, bucket_compute = 0, bucket_issue_wait = 0;
    longint unsigned bucket_region_wait = 0, bucket_observation_wait = 0;
    longint unsigned bucket_fill = 0, bucket_internal = 0, bucket_idle = 0;
    longint unsigned mode_fires[4] = '{default:0};
    longint unsigned mode_lanes[4] = '{default:0};
    longint unsigned mode_terms[4] = '{default:0};
    integer lanes, terms;
    string instance_name;
    initial instance_name = $sformatf("%m");

    always @(posedge clk) begin
        if (rst_n) begin
            cycles++;
            if (issue_valid && issue_ready) issue_fires++;
            if (fill_valid && fill_ready) fill_fires++;
            if (accumulator_valid && accumulator_ready) begin
                lanes = $countones(lane_mask);
                terms = $countones(lane_mask & enable0) + $countones(lane_mask & enable1) +
                        $countones(lane_mask & enable2) + $countones(lane_mask & enable3);
                contributions++;
                lane_fires += 64'(lanes);
                enabled_terms += 64'(terms);
                mode_fires[mode]++;
                mode_lanes[mode] += 64'(lanes);
                mode_terms[mode] += 64'(terms);
            end
            if (aborting) bucket_abort++;
            else if (accumulator_valid && accumulator_ready) bucket_compute++;
            else if (issue_valid && !issue_ready) bucket_issue_wait++;
            else if (region_valid && !region_ready) bucket_region_wait++;
            else if (observation_valid && !observation_ready) bucket_observation_wait++;
            else if (fill_valid && fill_ready) bucket_fill++;
            else if (!idle) bucket_internal++;
            else bucket_idle++;
        end
    end

    final begin
        $display("H40_ACTIVITY tile=%s cycles=%0d issue_fires=%0d fill_fires=%0d contributions=%0d lane_fires=%0d enabled_terms=%0d abort=%0d compute=%0d issue_wait=%0d region_wait=%0d observation_wait=%0d fill=%0d internal=%0d idle=%0d",
            instance_name, cycles, issue_fires, fill_fires, contributions, lane_fires, enabled_terms,
            bucket_abort, bucket_compute, bucket_issue_wait, bucket_region_wait,
            bucket_observation_wait, bucket_fill, bucket_internal, bucket_idle);
        for (int m = 0; m < 4; m++)
            $display("H40_ACTIVITY_MODE tile=%s mode=%0d contributions=%0d lane_fires=%0d enabled_terms=%0d",
                instance_name, m, mode_fires[m], mode_lanes[m], mode_terms[m]);
    end
endmodule

`ifndef H40_ACTIVITY_UNIT_TEST
bind unified_dual_role_h40_stationary_tile_candidate_v1
    supplement_h40_activity_monitor u_supplement_activity (
        .clk(i_clk), .rst_n(i_rst_n), .aborting(i_abort), .idle(o_idle),
        .accumulator_valid(accumulator_input_valid), .accumulator_ready(accumulator_ready),
        .lane_mask(lane_mask_q), .enable0(enable_a_q), .enable1(enable_d_q),
        .enable2(enable2_q), .enable3(enable3_q), .mode(issue_mode_q),
        .issue_valid(i_issue_valid), .issue_ready(o_issue_ready),
        .region_valid(o_region_valid), .region_ready(i_region_ready),
        .observation_valid(o_observation_valid), .observation_ready(i_observation_ready),
        .fill_valid(i_weight_fill_valid), .fill_ready(o_weight_fill_ready)
    );
`endif
`default_nettype wire
