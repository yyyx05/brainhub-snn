# Control-flow unit test only: these stubs do not perform FPGA implementation.
lassign $argv script root dcp recipe stage scenario
set argv [list $root $dcp $recipe $stage]
set routed 0
proc set_param {args} {}
proc open_checkpoint {args} {}
proc current_design {} {return unit_test_design}
proc get_clocks {args} {return core_clk}
proc get_timing_paths {args} {return test_path}
proc get_property {property objects} {
    global scenario routed
    switch -- $property {
        NAME {return $objects}
        PERIOD {return [expr {$scenario eq "wrong_clock" ? 14.0 : 15.0}]}
        PART {return xc7vx690tffg1761-3}
        SLACK {return [expr {$scenario eq "negative_setup" ? -0.1 : 0.1}]}
        DONT_TOUCH {return [expr {!($scenario eq "lost_protection" && $routed)}]}
        REF_NAME {return [expr {$scenario eq "live_trace" && $routed ? "FDRE" : "GND"}]}
        SEVERITY {return Error}
        default {error "unexpected test property: $property"}
    }
}
proc get_cells {args} {
    global scenario
    if {[lsearch -exact $args -of_objects] >= 0} {return driver_cell}
    set filter_index [lsearch -exact $args -filter]
    if {$filter_index >= 0} {
        set filter [lindex $args [expr {$filter_index + 1}]]
        if {[string match *IS_BLACKBOX* $filter]} {return {}}
        if {[string match *IS_PLACED* $filter]} {
            return [expr {$scenario eq "unplaced" ? "unplaced_cell" : ""}]
        }
        if {[string match *REF_NAME* $filter]} {return trace_product_q_reg}
        if {[string match *DONT_TOUCH* $filter]} {return protected_abort_reg}
        error "unexpected cell filter: $filter"
    }
    return [lindex $args end]
}
proc get_pins {args} {
    global scenario routed
    if {$scenario eq "live_trace" && $routed} {
        if {[lsearch -exact $args -leaf] >= 0} {return driver_pin}
        if {[string match */R [lindex $args end]]} {return trace_reset_pin}
    }
    return {}
}
proc get_nets {args} {return trace_reset_net}
proc opt_design {args} {puts MOCK_OPT}
proc place_design {args} {puts "MOCK_PLACE $args"}
proc phys_opt_design {args} {puts "MOCK_PHYS $args"}
proc route_design {args} {global routed; set routed 1; puts MOCK_ROUTE}
proc write_checkpoint {args} {
    set handle [open [lindex $args end] w]
    puts $handle "unit test placeholder, not a checkpoint"
    close $handle
}
proc get_drc_violations {args} {
    global scenario
    return [expr {$scenario eq "drc_error" ? "violation" : ""}]
}
proc report_stub {command args} {
    global scenario
    set index [lsearch -exact $args -file]
    set handle [open [lindex $args [expr {$index + 1}]] w]
    if {$command eq "report_route_status"} {
        puts $handle "# of routable nets : 2"
        puts $handle "# of fully routed nets : [expr {$scenario eq "incomplete_route" ? 1 : 2}]"
        puts $handle "# of nets with routing errors : 0"
    } elseif {$command eq "report_design_analysis" && $scenario eq "level6"} {
        puts $handle "| North | Global | 6 |"
    }
    close $handle
}
foreach command {
    report_clocks report_exceptions report_timing_summary report_timing
    report_utilization report_design_analysis report_control_sets
    report_high_fanout_nets report_clock_utilization report_route_status
    report_methodology report_drc
} {
    interp alias {} $command {} report_stub $command
}
source $script
