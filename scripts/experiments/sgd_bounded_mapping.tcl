set root [lindex $argv 0]
set scratch [lindex $argv 1]
set_param general.maxThreads 2
cd $scratch
foreach top {direct_registered44 sgd_registered44} {
    create_project -in_memory -part xc7vx690tffg1761-3
    read_verilog -sv [list $root/unified_tp_direct_residual_update.sv $root/direct_update_wrapper.sv $root/bounded_sgd_update.sv $root/sgd_registered44.sv]
    read_xdc $root/clock.xdc
    synth_design -top $top -part xc7vx690tffg1761-3 -mode out_of_context
    report_utilization -file $root/${top}_utilization.rpt
    report_utilization -hierarchical -file $root/${top}_hierarchy.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $root/${top}_timing.rpt
    report_timing -delay_type max -max_paths 10 -path_type full_clock_expanded -file $root/${top}_paths.rpt
    foreach kind {gradient code residual} {
        set regs [get_cells -quiet ${kind}_q_reg*]
        set paths [get_timing_paths -from $regs -delay_type max -max_paths 10]
        if {[llength $regs] == 0 || [llength $paths] == 0} {error "Missing input register paths: $top $kind"}
        puts "INPUT_REG_PATH top=$top kind=$kind registers=[llength $regs] paths=[llength $paths] slack=[get_property SLACK [lindex $paths 0]]"
        report_timing -from $regs -delay_type max -max_paths 10 -path_type full_clock_expanded -file $root/${top}_${kind}_paths.rpt
    }
    check_timing -verbose -file $root/${top}_checks.rpt
    close_project
}
puts BOUNDED_MAPPING_COMPLETE
