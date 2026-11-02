set root [lindex $argv 0]
set scratch [lindex $argv 1]
set_param general.maxThreads 2
cd $scratch
foreach top {direct_update_wrapper fixed_sgd_update} {
    create_project -in_memory -part xc7vx690tffg1761-3
    read_verilog -sv [list $root/unified_tp_direct_residual_update.sv $root/direct_update_wrapper.sv $root/fixed_sgd_update.sv]
    read_xdc $root/clock.xdc
    synth_design -top $top -part xc7vx690tffg1761-3 -mode out_of_context
    report_utilization -file $root/${top}_utilization.rpt
    report_utilization -hierarchical -file $root/${top}_hierarchy.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $root/${top}_timing.rpt
    report_timing -delay_type max -max_paths 10 -path_type full_clock_expanded -file $root/${top}_paths.rpt
    check_timing -verbose -file $root/${top}_checks.rpt
    close_project
}
puts SGD_UPDATER_MAPPING_COMPLETE
