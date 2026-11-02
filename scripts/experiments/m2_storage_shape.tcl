set root [file normalize [lindex $argv 0]]
set part xc7vx690tffg1761-3
set_param general.maxThreads 8
foreach variant {baseline full578 selected512} index {0 1 2} {
    create_project -in_memory -part $part
    set_property XPM_LIBRARIES {XPM_MEMORY} [current_project]
    foreach source {baseline.sv m2_compact_store.sv m2_storage_shape.sv} {
        read_verilog -sv [file join $root $source]
    }
    read_xdc [file join $root clock.xdc]
    synth_design -top m2_storage_shape -mode out_of_context -part $part \
        -flatten_hierarchy rebuilt -verilog_define {UNIFIED_USE_XPM_MEMORY} \
        -generic VARIANT=$index
    report_utilization -file [file join $root ${variant}_utilization.rpt]
    report_utilization -hierarchical -file [file join $root ${variant}_hierarchy.rpt]
    report_timing_summary -delay_type min_max -file [file join $root ${variant}_timing.rpt]
    report_clocks -file [file join $root ${variant}_clocks.rpt]
    check_timing -verbose -file [file join $root ${variant}_check_timing.rpt]
    set f [open [file join $root ${variant}_memory_primitives.tsv] w]
    puts $f "name\tref\tread_width_a\tread_width_b\twrite_width_a\twrite_width_b\tram_mode"
    foreach cell [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB*}] {
        puts $f "[get_property NAME $cell]\t[get_property REF_NAME $cell]\t[get_property READ_WIDTH_A $cell]\t[get_property READ_WIDTH_B $cell]\t[get_property WRITE_WIDTH_A $cell]\t[get_property WRITE_WIDTH_B $cell]\t[get_property RAM_MODE $cell]"
    }
    close $f
    set f [open [file join $root ${variant}_structure.txt] w]
    puts $f "black_boxes=[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]"
    close $f
    write_checkpoint -force [file join $root ${variant}.dcp]
    close_project
}
puts "M2_STORAGE_MAPPING_COMPLETE"
exit
