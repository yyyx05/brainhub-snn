if {[llength $argv] != 2} {
    error "usage: <package_root> <post_synth_dcp>"
}

set root [file normalize [lindex $argv 0]]
set source_dcp [file normalize [lindex $argv 1]]
set out_dir [file join $root server_results final_composition_impl]
file mkdir $out_dir

if {![file isfile $source_dcp]} {
    error "post-synthesis checkpoint does not exist: $source_dcp"
}

set_param general.maxThreads 8
set expected_part xc7vx690tffg1761-3
set expected_top unified_h40_final_composition_top_v1
set required_top_cells {
    g_srnn_phase_bridge.u_srnn_phase_bridge
    u_system
    u_parameter_plane
}

proc write_timing_reports {tag out_dir} {
    report_timing_summary -delay_type min_max -max_paths 100 \
        -report_unconstrained \
        -file [file join $out_dir timing_summary_$tag.rpt]
    report_timing -delay_type max -max_paths 100 -sort_by group \
        -file [file join $out_dir timing_setup_paths_$tag.rpt]
    report_timing -delay_type min -max_paths 100 -sort_by group \
        -file [file join $out_dir timing_hold_paths_$tag.rpt]
}

proc read_text_file {path} {
    set handle [open $path r]
    set text [read $handle]
    close $handle
    return $text
}

proc route_metric {text label} {
    set pattern [format {# of %s[^:\r\n]*:[[:space:]]*([0-9]+)} $label]
    if {[regexp $pattern $text -> value]} {
        return $value
    }
    return -1
}

proc timing_slack {delay_type} {
    set path [get_timing_paths -delay_type $delay_type -max_paths 1 -nworst 1]
    if {[llength $path] == 0} {
        return no_path
    }
    return [get_property SLACK $path]
}

open_checkpoint $source_dcp

set design_top [get_property NAME [current_design]]
set design_part [get_property PART [current_design]]
if {$design_part ne $expected_part} {
    error "checkpoint part mismatch: expected $expected_part, got $design_part"
}

# Non-project checkpoints may reopen with a checkpoint-derived current-design
# name instead of the original RTL top. Validate the preserved composition
# boundary directly rather than treating that transient object name as TOP.
foreach required_cell $required_top_cells {
    set matches [get_cells -quiet $required_cell]
    if {[llength $matches] != 1} {
        error "checkpoint structure mismatch: expected one $required_cell, got [llength $matches]"
    }
}

set synth_black_boxes [get_cells -quiet -hierarchical \
    -filter {IS_BLACKBOX == 1}]
if {[llength $synth_black_boxes] != 0} {
    error "checkpoint contains black boxes before implementation"
}

set clock_count [llength [get_clocks -quiet]]
if {$clock_count == 0} {
    error "checkpoint contains no timing clocks"
}

# Run71 failed exclusively because region abort drove wide trace-payload reset
# pins across the device.  Verify that synthesis preserved the Run73 split:
# validity may reset, while payload storage has no live reset/set control.
# Restrict the name match to FD primitives so implementation helper cells such
# as CARRY4/DSP48E1 instances suffixed with "_i_*" are not audited as flops.
set trace_payload_cells [get_cells -quiet -hierarchical -filter {
    IS_PRIMITIVE == 1 &&
    REF_NAME =~ FD* &&
    (NAME =~ *trace_product_q_reg* ||
     NAME =~ *trace_old_flat_q_reg* ||
     NAME =~ *o_trace_next_flat_reg*)
}]
set trace_payload_nonconstant_control_count 0
set trace_control_report [open \
    [file join $out_dir trace_payload_control_pre_impl.rpt] w]
puts $trace_control_report "trace_payload_cell_count=[llength $trace_payload_cells]"
foreach payload_cell $trace_payload_cells {
    foreach control_pin_name {R S CLR PRE} {
        set control_pin [get_pins -quiet \
            "[get_property NAME $payload_cell]/$control_pin_name"]
        if {[llength $control_pin] == 0} {
            continue
        }
        set control_net [get_nets -quiet -of_objects $control_pin]
        set driver_pins [get_pins -quiet -leaf -of_objects $control_net \
            -filter {DIRECTION == OUT}]
        set driver_cells [get_cells -quiet -of_objects $driver_pins]
        set driver_refs [lsort -unique [get_property REF_NAME $driver_cells]]
        puts $trace_control_report \
            "cell=[get_property NAME $payload_cell] pin=$control_pin_name net=[get_property NAME $control_net] drivers=$driver_refs"
        if {[lsearch -exact $driver_refs GND] < 0} {
            incr trace_payload_nonconstant_control_count
        }
    }
}
puts $trace_control_report \
    "trace_payload_nonconstant_control_count=$trace_payload_nonconstant_control_count"
close $trace_control_report
if {[llength $trace_payload_cells] == 0} {
    error "no synthesized trace payload registers matched the Run73 audit"
}
if {$trace_payload_nonconstant_control_count != 0} {
    error "trace payload registers retain live reset/set control after synthesis"
}

opt_design
place_design -directive AltSpreadLogic_high
phys_opt_design -directive ExploreWithHoldFix

report_utilization -file [file join $out_dir utilization_post_place.rpt]
report_utilization -hierarchical -hierarchical_depth 12 \
    -file [file join $out_dir utilization_hierarchical_post_place.rpt]
write_timing_reports post_place $out_dir
set post_place_congestion_report \
    [file join $out_dir congestion_post_place.rpt]
report_design_analysis -congestion -file $post_place_congestion_report
report_control_sets -verbose \
    -file [file join $out_dir control_sets_post_place.rpt]
report_high_fanout_nets -timing -max_nets 100 \
    -file [file join $out_dir high_fanout_post_place.rpt]
write_checkpoint -force [file join $out_dir post_place_physopt.dcp]

# RUN115 and its route-only RUN116 continuation both failed at level-6 global
# congestion. Do not spend another full route on a placement with the same
# proven-unroutable severity.
set post_place_congestion_text [read_text_file $post_place_congestion_report]
set level6_pattern {\|[[:space:]]*(North|South|East|West)[[:space:]]*\|[[:space:]]*(Global|Long|Short)[[:space:]]*\|[[:space:]]*6[[:space:]]*\|}
set post_place_level6 [regexp $level6_pattern $post_place_congestion_text]
set placement_gate [open [file join $out_dir placement_gate.txt] w]
puts $placement_gate "place_directive=AltSpreadLogic_high"
puts $placement_gate "phys_opt_directive=ExploreWithHoldFix"
puts $placement_gate "post_place_level6=$post_place_level6"
close $placement_gate
if {$post_place_level6} {
    error "post-place congestion remains level 6 after AltSpreadLogic_high"
}

route_design -directive AggressiveExplore
phys_opt_design -directive Explore

set final_black_boxes [get_cells -quiet -hierarchical \
    -filter {IS_BLACKBOX == 1}]
set final_unplaced [get_cells -quiet -hierarchical -filter {
    IS_PRIMITIVE == 1 && IS_PLACED == 0
}]
set setup_slack [timing_slack max]
set hold_slack [timing_slack min]

report_utilization -file [file join $out_dir utilization_post_route.rpt]
report_utilization -hierarchical -hierarchical_depth 12 \
    -file [file join $out_dir utilization_hierarchical_post_route.rpt]
set route_report [file join $out_dir route_status_post_route.rpt]
report_route_status -file $route_report
report_design_analysis -congestion \
    -file [file join $out_dir congestion_post_route.rpt]
write_timing_reports post_route $out_dir
report_clock_utilization \
    -file [file join $out_dir clock_utilization_post_route.rpt]
report_control_sets -verbose \
    -file [file join $out_dir control_sets_post_route.rpt]
report_high_fanout_nets -timing -max_nets 100 \
    -file [file join $out_dir high_fanout_post_route.rpt]
set methodology_report [file join $out_dir methodology_post_route.rpt]
set drc_report [file join $out_dir drc_post_route.rpt]
report_methodology -file $methodology_report
report_drc -file $drc_report

set drc_error_count 0
set drc_critical_warning_count 0
foreach violation [get_drc_violations -quiet] {
    set severity [get_property SEVERITY $violation]
    if {$severity eq "Error"} {
        incr drc_error_count
    } elseif {$severity eq "Critical Warning"} {
        incr drc_critical_warning_count
    }
}

set qor_status not_available
if {[llength [info commands report_qor_suggestions]] > 0} {
    if {[catch {
        report_qor_suggestions \
            -file [file join $out_dir qor_suggestions_post_route.rpt]
    } qor_error]} {
        set qor_status "failed:$qor_error"
    } else {
        set qor_status generated
    }
}

set power_status not_available
if {[llength [info commands report_power]] > 0} {
    if {[catch {
        report_power \
            -file [file join $out_dir power_post_route_estimated.rpt]
    } power_error]} {
        set power_status "failed:$power_error"
    } else {
        set power_status generated
    }
}

set route_text [read_text_file $route_report]
set routable_nets [route_metric $route_text {routable nets}]
set fully_routed_nets [route_metric $route_text {fully routed nets}]
set routing_error_nets [route_metric $route_text {nets with routing errors}]
set route_complete [expr {$routable_nets >= 0 &&
    $fully_routed_nets == $routable_nets && $routing_error_nets == 0}]

set methodology_text [read_text_file $methodology_report]
set drc_text [read_text_file $drc_report]
set severity_pattern {\|[[:space:]]*(Error|Critical Warning)[[:space:]]*\|}
set methodology_severity_error [regexp $severity_pattern $methodology_text]
set drc_severity_error [regexp $severity_pattern $drc_text]

set timing_closed [expr {$setup_slack ne "no_path" &&
    $hold_slack ne "no_path" && $setup_slack >= 0.0 && $hold_slack >= 0.0}]
set tns_status [expr {$setup_slack eq "no_path" ? "no_path" :
    ($setup_slack >= 0.0 ? 0.0 : "negative")}]
set ths_status [expr {$hold_slack eq "no_path" ? "no_path" :
    ($hold_slack >= 0.0 ? 0.0 : "negative")}]
set acceptance [expr {[llength $final_black_boxes] == 0 &&
    [llength $final_unplaced] == 0 && $route_complete && $timing_closed &&
    !$methodology_severity_error && !$drc_severity_error &&
    $drc_error_count == 0 && $drc_critical_warning_count == 0 ?
    "pass" : "diagnostic_complete"}]

write_checkpoint -force [file join $out_dir post_route_physopt.dcp]

set status_handle [open [file join $out_dir vivado_status.txt] w]
puts $status_handle "status=$acceptance"
puts $status_handle "source_dcp=$source_dcp"
puts $status_handle "top=$expected_top"
puts $status_handle "checkpoint_design_name=$design_top"
puts $status_handle "part=$design_part"
puts $status_handle "clock_count=$clock_count"
puts $status_handle "trace_payload_cell_count=[llength $trace_payload_cells]"
puts $status_handle "trace_payload_nonconstant_control_count=$trace_payload_nonconstant_control_count"
puts $status_handle "black_boxes=[llength $final_black_boxes]"
puts $status_handle "unplaced_primitives=[llength $final_unplaced]"
puts $status_handle "routable_nets=$routable_nets"
puts $status_handle "fully_routed_nets=$fully_routed_nets"
puts $status_handle "routing_error_nets=$routing_error_nets"
puts $status_handle "route_complete=$route_complete"
puts $status_handle "wns_ns=$setup_slack"
puts $status_handle "whs_ns=$hold_slack"
puts $status_handle "tns_ns=$tns_status"
puts $status_handle "ths_ns=$ths_status"
puts $status_handle "drc_errors=$drc_error_count"
puts $status_handle "drc_critical_warnings=$drc_critical_warning_count"
puts $status_handle "methodology_severity_error=$methodology_severity_error"
puts $status_handle "drc_severity_error=$drc_severity_error"
puts $status_handle "qor_suggestions=$qor_status"
puts $status_handle "power_report=$power_status"
puts $status_handle "bitstream=not_generated"
close $status_handle

puts "FINAL_COMPOSITION_IMPL_COMPLETE status=$acceptance black_boxes=[llength $final_black_boxes] unplaced_primitives=[llength $final_unplaced] route_complete=$route_complete wns_ns=$setup_slack whs_ns=$hold_slack drc_errors=$drc_error_count drc_critical_warnings=$drc_critical_warning_count"
exit
