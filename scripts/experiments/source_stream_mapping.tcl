set root [lindex $argv 0]
set scratch [lindex $argv 1]
set_param general.maxThreads 2
cd $scratch
foreach policy {0 1 2} {
    create_project -in_memory -part xc7vx690tffg1761-3
    read_verilog -sv [list $root/source_scan.sv $root/source_stream.sv]
    read_xdc $root/clock.xdc
    synth_design -top scan_cost_top -part xc7vx690tffg1761-3 -mode out_of_context -generic POLICY=$policy
    report_utilization -file $root/producer_${policy}_utilization.rpt
    report_utilization -hierarchical -file $root/producer_${policy}_hierarchy.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $root/producer_${policy}_timing.rpt
    check_timing -verbose -file $root/producer_${policy}_checks.rpt
    close_project
}
puts SOURCE_STREAM_MAPPING_COMPLETE
