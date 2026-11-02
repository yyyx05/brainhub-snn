set root [file normalize [lindex $argv 0]]
set part xc7vx690tffg1761-3
set_param general.maxThreads 8
foreach variant {baseline candidate} {
    create_project -in_memory -part $part
    read_verilog -sv [file join $root ${variant}.sv]
    read_xdc [file join $root clock.xdc]
    synth_design -top unified_fc_tp_w2_contribution_scheduler -mode out_of_context \
        -part $part -flatten_hierarchy rebuilt \
        -generic {MAX_BATCH=16 MAX_HIDDEN=2048 MAX_OUTPUTS=16}
    report_utilization -file [file join $root ${variant}_utilization.rpt]
    report_utilization -hierarchical -file [file join $root ${variant}_hierarchy.rpt]
    report_timing_summary -delay_type min_max -file [file join $root ${variant}_timing.rpt]
    report_clocks -file [file join $root ${variant}_clocks.rpt]
    set f [open [file join $root ${variant}_bitmap_cells.rpt] w]
    set counts [dict create]
    foreach cell [get_cells -quiet -hierarchical -filter {IS_PRIMITIVE == 1 && (NAME =~ *nonzero_mem* || NAME =~ *trace_nonzero_mem*)}] {
        dict incr counts [get_property REF_NAME $cell]
    }
    foreach ref [lsort [dict keys $counts]] { puts $f "$ref=[dict get $counts $ref]" }
    puts $f "black_boxes=[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]"
    close $f
    write_checkpoint -force [file join $root ${variant}.dcp]
    close_project
}
if {[llength $argv] == 3} {
    foreach variant {old current} dcp [lrange $argv 1 2] {
        open_checkpoint $dcp
        set f [open [file join $root replay_${variant}_primitives.tsv] w]
        puts $f "name\tref_name"
        foreach cell [get_cells -quiet -hierarchical -filter {IS_PRIMITIVE == 1 && NAME =~ *u_observation_replay/*}] {
            puts $f "[get_property NAME $cell]\t[get_property REF_NAME $cell]"
        }
        close $f
        close_design
    }
}
puts "FC_BITMAP_MAPPING_COMPLETE"
exit
