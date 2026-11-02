set root [file normalize [lindex $argv 0]]
set enable_online_learning 1
if {[llength $argv] > 1} {
    set enable_online_learning [lindex $argv 1]
}
if {$enable_online_learning ni {0 1}} {
    error "ENABLE_ONLINE_LEARNING must be 0 or 1"
}
set out_dir [file join $root server_results final_composition_synth]
file mkdir $out_dir

set_param general.maxThreads 8
set part xc7vx690tffg1761-3
set top unified_h40_final_composition_top_v1

proc collect_systemverilog_sources {directory} {
    set sources [list]
    foreach entry [glob -nocomplain -directory $directory *] {
        if {[file isdirectory $entry]} {
            set sources [concat $sources \
                [collect_systemverilog_sources $entry]]
        } elseif {[string equal -nocase [file extension $entry] ".sv"]} {
            lappend sources [file normalize $entry]
        }
    }
    return $sources
}

set sources [lsort [collect_systemverilog_sources [file join $root rtl]]]
if {[llength $sources] == 0} {
    error "no SystemVerilog sources found under $root/rtl"
}

set compat_dir [file join $out_dir vivado_2022_2_sources]
file mkdir $compat_dir
set compat_sources [list]
set source_index 0
foreach source $sources {
    set source_handle [open $source r]
    set source_text [read $source_handle]
    close $source_handle
    set compat_text [string map [list \
        {`default_nettype none} {`default_nettype wire}] $source_text]
    set compat_source [file join $compat_dir \
        [format "%03d_%s" $source_index [file tail $source]]]
    set compat_handle [open $compat_source w]
    puts -nonewline $compat_handle $compat_text
    close $compat_handle
    lappend compat_sources $compat_source
    incr source_index
}

foreach source $compat_sources {
    read_verilog -sv $source
}
read_xdc [file join $root vivado unified_h40_final_composition_v1_ooc.xdc]

synth_design -mode out_of_context -flatten_hierarchy rebuilt \
    -part $part -top $top -directive Default \
    -verilog_define {UNIFIED_USE_XPM_MEMORY} \
    -generic [list "ENABLE_ONLINE_LEARNING=$enable_online_learning"]

report_utilization \
    -file [file join $out_dir utilization_post_synth.rpt]
report_utilization -hierarchical -hierarchical_depth 12 \
    -file [file join $out_dir utilization_hierarchical_post_synth.rpt]
report_timing_summary -delay_type min_max -max_paths 100 \
    -report_unconstrained \
    -file [file join $out_dir timing_summary_post_synth.rpt]
report_timing -delay_type max -max_paths 100 -sort_by group \
    -file [file join $out_dir timing_setup_paths_post_synth.rpt]
report_high_fanout_nets -timing -max_nets 100 \
    -file [file join $out_dir high_fanout_post_synth.rpt]
report_methodology \
    -file [file join $out_dir methodology_post_synth.rpt]
write_checkpoint -force [file join $out_dir post_synth.dcp]

set black_boxes [get_cells -quiet -hierarchical \
    -filter {IS_BLACKBOX == 1}]
set setup_path [get_timing_paths -delay_type max -max_paths 1 -nworst 1]
set hold_path [get_timing_paths -delay_type min -max_paths 1 -nworst 1]
set setup_slack no_path
set hold_slack no_path
if {[llength $setup_path] > 0} {
    set setup_slack [get_property SLACK $setup_path]
}
if {[llength $hold_path] > 0} {
    set hold_slack [get_property SLACK $hold_path]
}

set status_handle [open [file join $out_dir vivado_status.txt] w]
puts $status_handle "status=pass"
puts $status_handle "top=$top"
puts $status_handle "part=$part"
puts $status_handle "period_ns=15.000"
puts $status_handle "source_count=[llength $sources]"
puts $status_handle "enable_online_learning=$enable_online_learning"
puts $status_handle "black_boxes=[llength $black_boxes]"
puts $status_handle "wns_ns=$setup_slack"
puts $status_handle "whs_ns=$hold_slack"
puts $status_handle "implementation=not_run"
puts $status_handle "bitstream=not_generated"
close $status_handle

if {[llength $black_boxes] != 0} {
    error "final composition contains black boxes after synthesis"
}

puts "FINAL_COMPOSITION_SYNTH_COMPLETE black_boxes=[llength $black_boxes] wns_ns=$setup_slack whs_ns=$hold_slack"
exit
