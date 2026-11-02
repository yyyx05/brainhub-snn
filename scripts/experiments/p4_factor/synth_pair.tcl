# Server-side Vivado batch entry. Synthesis only, exactly two sequential arms.
set script_dir [file dirname [file normalize [info script]]]
set root [file normalize [file join $script_dir ../../..]]
set experiment [file join $root verification experiments p4_factor]
set production_rel rtl/tiles/unified_dual_role_dsp_lane_p4_cascade_candidate_v1.sv
set production [file join $root $production_rel]
set part xc7vx690tffg1761-3

proc write_record {path values} {
    set f [open $path w]
    foreach {key value} $values { puts $f "$key=$value" }
    close $f
}

proc run_pair {} {
    global argv root experiment production production_rel part script_dir
    if {[llength $argv] != 1 || ![regexp {^[A-Za-z0-9][A-Za-z0-9_.-]*$} [lindex $argv 0]]} {
        error "Usage: vivado -mode batch -source synth_pair.tcl -tclargs NEW_RUN_NAME"
    }
    set out [file join $experiment runs [lindex $argv 0]]
    if {[file exists $out]} { error "Run already exists; choose a new name: $out" }
    set f [open [file join $root rtl final_sources.f] r]
    set sources [split [read $f] \n]
    close $f
    set trimmed {}
    foreach line $sources { lappend trimmed [string trim $line] }
    set sources $trimmed
    if {[lsearch -exact $sources $production_rel] < 0 || ![file readable $production]} {
        error "Current production source reference is missing"
    }
    file mkdir $out
    # Keep Vivado-generated project side files out of the production tree.
    cd $out
    set_param general.maxThreads 1
    foreach {arm control} {factor 0 sop 1} {
        set arm_dir [file join $out $arm]
        file mkdir $arm_dir
        write_record [file join $arm_dir status.txt] {status running}
        puts "P4_ARM_BEGIN $arm"
        if {[catch {
            create_project -in_memory -part $part
            read_verilog -sv [list $production \
                [file join $experiment p4_sum_products.sv] \
                [file join $experiment p4_factor_top.sv]]
            read_xdc [file join $script_dir p4_factor.xdc]
            synth_design -top p4_factor_top -part $part -mode out_of_context \
                -flatten_hierarchy rebuilt -directive Default -generic SOP_CONTROL=$control
            report_utilization -file [file join $arm_dir utilization.rpt]
            report_utilization -hierarchical -file [file join $arm_dir hierarchy.rpt]
            report_timing_summary -delay_type min_max -report_unconstrained \
                -file [file join $arm_dir timing_summary.rpt]
            report_timing -delay_type max -max_paths 10 -file [file join $arm_dir setup_paths.rpt]
            report_timing -delay_type min -max_paths 10 -file [file join $arm_dir hold_paths.rpt]
            report_clocks -file [file join $arm_dir clocks.rpt]
            check_timing -verbose -file [file join $arm_dir check_timing.rpt]
            write_checkpoint [file join $arm_dir post_synth.dcp]
            write_verilog -mode funcsim [file join $arm_dir post_synth.v]
            write_record [file join $arm_dir run_info.txt] [list \
                arm $arm sop_control $control stage post_synthesis \
                top p4_factor_top part $part period_ns 15.000 \
                tool_version [version -short] mode out_of_context \
                flatten_hierarchy rebuilt directive Default max_threads 1 \
                production_source $production_rel \
                blackboxes [llength [get_cells -hier -quiet -filter {IS_BLACKBOX == 1}]]]
            close_project
            write_record [file join $arm_dir status.txt] {status complete}
            puts "P4_ARM_END $arm"
        } message options]} {
            write_record [file join $arm_dir status.txt] [list status failed error $message]
            set f [open [file join $arm_dir failure.txt] w]
            puts $f [dict get $options -errorinfo]
            close $f
            return -options $options $message
        }
    }
    write_record [file join $out flow.txt] {status complete arms factor,sop}
    puts "P4_FACTOR_SYNTH_COMPLETE (post-synthesis reports only; not routed STA)"
}

if {[catch {run_pair} message options]} {
    puts stderr [dict get $options -errorinfo]
    exit 1
}
