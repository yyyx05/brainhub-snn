set root [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
set part xc7vx690tffg1761-3
set_param general.maxThreads 2
foreach variant {baseline candidate} opt {0 1} {
    create_project -in_memory -part $part
    foreach source [lsort [glob [file join $root sources_vivado *.sv]]] {
        read_verilog -sv $source
    }
    read_xdc [file join $root clock.xdc]
    synth_design -top turnover_chain -mode out_of_context -part $part \
        -flatten_hierarchy rebuilt -generic OPT=$opt
    report_utilization -file [file join $out ${variant}_utilization.rpt]
    report_utilization -hierarchical -file [file join $out ${variant}_hierarchy.rpt]
    report_timing_summary -delay_type min_max -file [file join $out ${variant}_timing.rpt]
    report_timing -delay_type max -max_paths 10 -file [file join $out ${variant}_setup_paths.rpt]
    report_timing -delay_type min -max_paths 10 -file [file join $out ${variant}_hold_paths.rpt]
    report_clocks -file [file join $out ${variant}_clocks.rpt]
    report_exceptions -file [file join $out ${variant}_exceptions.rpt]
    check_timing -verbose -file [file join $out ${variant}_check_timing.rpt]
    set f [open [file join $out ${variant}_structure.txt] w]
    puts $f "black_boxes=[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]"
    close $f
    write_checkpoint -force [file join $out ${variant}.dcp]
    close_project
}
puts "SCNN_TURNOVER_MAPPING_COMPLETE"
exit
