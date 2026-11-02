# Read existing DCPs only. No synthesis, placement, routing, or netlist overwrite.
set root [file normalize [file join [file dirname [info script]] ../../..]]
if {[llength $argv] != 2} { error "Usage: export_mapped.tcl SYNTH_RUN NEW_MAPPED_RUN" }
foreach name $argv {
    if {![regexp {^[A-Za-z0-9][A-Za-z0-9_.-]*$} $name]} { error "Invalid run name" }
}
set synth_dir [file join $root verification experiments p4_factor runs [lindex $argv 0]]
set out [file join $synth_dir [lindex $argv 1]]
if {[file exists $out]} { error "Output exists; preserve it and choose a new name" }
file mkdir $out
cd $out
set_param general.maxThreads 1
redirect -variable command_help { help write_verilog }
set f [open write_verilog_help.txt w]
puts $f $command_help
close $f
puts $command_help
if {[string first "-rename_top" $command_help] < 0} {
    error "Installed write_verilog help does not document -rename_top"
}
foreach arm {factor sop} {
    set checkpoint [file join $synth_dir $arm post_synth.dcp]
    puts "P4_READ_CHECKPOINT $checkpoint"
    open_checkpoint $checkpoint
    write_verilog -mode funcsim -rename_top p4_mapped_$arm [file join $out ${arm}_mapped.v]
    redirect -file [file join $out ${arm}_dsp_properties.rpt] {
        foreach dsp [get_cells -hier -filter {REF_NAME == DSP48E1}] {
            puts "DSP $dsp"
            report_property $dsp
        }
    }
    close_design
}
set f [open export_status.txt w]
puts $f "status=complete"
puts $f "source_synth_run=[lindex $argv 0]"
puts $f "mode=read_checkpoint_only"
close $f
