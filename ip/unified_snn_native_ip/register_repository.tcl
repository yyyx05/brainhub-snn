# Source this in an existing Vivado consumer project before creating the IP.
if {[llength $argv] != 1} {
    error "usage: source register_repository.tcl with argv containing <repository_root>"
}
set repository_root [file normalize [lindex $argv 0]]
if {![file isdirectory $repository_root]} {
    error "repository does not exist: $repository_root"
}
set project [current_project]
set repositories [get_property ip_repo_paths $project]
if {[lsearch -exact $repositories $repository_root] < 0} {
    lappend repositories $repository_root
}
set_property ip_repo_paths $repositories $project
set libraries [get_property XPM_LIBRARIES $project]
if {[lsearch -exact $libraries XPM_MEMORY] < 0} {
    lappend libraries XPM_MEMORY
}
set_property XPM_LIBRARIES $libraries $project
foreach fileset [list [current_fileset] [get_filesets sim_1]] {
    set defines [get_property verilog_define $fileset]
    if {[lsearch -exact $defines UNIFIED_USE_XPM_MEMORY] < 0} {
        lappend defines UNIFIED_USE_XPM_MEMORY
    }
    set_property verilog_define $defines $fileset
}
update_ip_catalog
if {[llength [get_ipdefs -all open.local:user:unified_snn_native:1.0]] != 1} {
    error "native IP definition was not found uniquely"
}
set memory_relative rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem
set memory_source [file join $repository_root unified_snn_native $memory_relative]
set memory_target [file join [get_property DIRECTORY $project] $memory_relative]
if {![file isfile $memory_source]} {
    error "repository is missing surrogate ROM: $memory_source"
}
if {[file exists $memory_target]} {
    set handle [open $memory_source rb]
    set original [read $handle]
    close $handle
    set handle [open $memory_target rb]
    set existing [read $handle]
    close $handle
    if {$original ne $existing} {
        error "consumer ROM differs; do not overwrite existing data: $memory_target"
    }
} else {
    file mkdir [file dirname $memory_target]
    file copy $memory_source $memory_target
}
add_files -norecurse $memory_target
set_property FILE_TYPE {Memory Initialization Files} [get_files $memory_target]
puts "Native IP ROM prepared at $memory_target; use the consumer project directory as simulation working directory."
