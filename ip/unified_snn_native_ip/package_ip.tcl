if {$argc != 2} {
    puts stderr "usage: vivado -mode batch -source package_ip.tcl -tclargs <source_package_root> <output_root>"
    exit 2
}

set source_root [file normalize [lindex $argv 0]]
set output_root [file normalize [lindex $argv 1]]
if {[file exists $output_root]} {
    error "output must be a new directory; preserve earlier packaging evidence: $output_root"
}
set filelist_path [file join $source_root src filelist.f]
set xdc_path [file join $source_root constraints unified_h40_final_composition_v1_ooc.xdc]
set memory_relative rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem
set memory_path [file join $source_root $memory_relative]
set adapter_path [file join $source_root src unified_snn_native_wrapper.v]

if {![file exists $filelist_path]} {
    error "missing source filelist: $filelist_path"
}
if {![file exists $xdc_path]} {
    error "missing OOC constraint: $xdc_path"
}
if {![file isfile $memory_path]} {
    error "missing original surrogate ROM: $memory_path"
}
if {![file isfile $adapter_path]} {
    error "missing wire-only Verilog packaging adapter: $adapter_path"
}

file mkdir $output_root
create_project -force unified_snn_native_ip_work [file join $output_root work] -part xc7vx690tffg1761-3

set handle [open $filelist_path r]
set source_files {}
while {[gets $handle line] >= 0} {
    set line [string trim $line]
    if {$line eq "" || [string match "#*" $line]} {
        continue
    }
    set source_path [file normalize [file join $source_root $line]]
    if {![file exists $source_path]} {
        close $handle
        error "filelist entry does not exist: $source_path"
    }
    lappend source_files $source_path
}
close $handle

set_property XPM_LIBRARIES {XPM_MEMORY} [current_project]
set_property verilog_define {UNIFIED_USE_XPM_MEMORY} [current_fileset]
add_files -norecurse $source_files
add_files -norecurse $adapter_path
add_files -fileset constrs_1 -norecurse $xdc_path
set_property USED_IN {out_of_context} [get_files $xdc_path]
set_property top unified_snn_native_wrapper [current_fileset]
update_compile_order -fileset sources_1

set repository_root [file join $output_root repository unified_snn_native]
ipx::package_project -root_dir $repository_root -vendor open.local -library user -taxonomy /UserIP -import_files -set_current true
set core [ipx::current_core]
set_property name unified_snn_native $core
set_property display_name {Unified FC/SRNN/SCNN Accelerator (Native Interface)} $core
set_property description {Profile-driven unified SNN accelerator source IP with native descriptor, memory, state, and learning handshakes.} $core
set_property vendor_display_name {Unified SNN Project} $core
set_property version 1.0 $core
set_property xpm_libraries {XPM_MEMORY} $core

# Preserve the RTL's relative readmemh path without rewriting measured sources.
file mkdir [file dirname [file join $repository_root $memory_relative]]
file copy $memory_path [file join $repository_root $memory_relative]
foreach group [ipx::get_file_groups -of_objects $core] {
    set group_name [get_property NAME $group]
    if {[string match "*synthesis*" $group_name] || [string match "*simulation*" $group_name]} {
        set entry [ipx::add_file $memory_relative $group]
        set_property type memoryInit $entry
    }
}

# Replace only automatically inferred clock/reset bindings for these ports.
foreach bus [ipx::get_bus_interfaces -of_objects $core] {
    foreach mapping [ipx::get_port_maps -of_objects $bus] {
        if {[get_property PHYSICAL_NAME $mapping] in {i_clk i_rst_n}} {
            ipx::remove_bus_interface [get_property NAME $bus] $core
            break
        }
    }
}
set clk_if [ipx::add_bus_interface clk $core]
set_property abstraction_type_vlnv xilinx.com:signal:clock_rtl:1.0 $clk_if
set_property bus_type_vlnv xilinx.com:signal:clock:1.0 $clk_if
set clk_map [ipx::add_port_map CLK $clk_if]
set_property physical_name i_clk $clk_map
set_property interface_mode slave $clk_if
set associated_reset [ipx::add_bus_parameter ASSOCIATED_RESET $clk_if]
set_property value i_rst_n $associated_reset
set nominal_frequency [ipx::add_bus_parameter FREQ_HZ $clk_if]
set_property value 66666667 $nominal_frequency

set rst_if [ipx::add_bus_interface resetn $core]
set_property abstraction_type_vlnv xilinx.com:signal:reset_rtl:1.0 $rst_if
set_property bus_type_vlnv xilinx.com:signal:reset:1.0 $rst_if
set rst_map [ipx::add_port_map RST $rst_if]
set_property physical_name i_rst_n $rst_map
set_property interface_mode slave $rst_if
set rst_parameter [ipx::add_bus_parameter POLARITY $rst_if]
set_property value ACTIVE_LOW $rst_parameter

ipx::save_core $core
set integrity [ipx::check_integrity $core]
puts "IP integrity result: $integrity"

# Keep the file audit independent of the original workstation paths.
set audit [open [file join $output_root packaged_files.rpt] w]
set unique_sv [dict create]
foreach group [ipx::get_file_groups -of_objects $core] {
    puts $audit "file_group=[get_property NAME $group]"
    foreach entry [ipx::get_files -of_objects $group] {
        set name [get_property NAME $entry]
        if {[file pathtype $name] eq "absolute"} {
            error "packaged file has an external absolute path: $name"
        }
        set full_path [file normalize [file join $repository_root $name]]
        if {![string match "${repository_root}/*" $full_path] || ![file isfile $full_path]} {
            error "packaged file missing or outside repository: $name"
        }
        puts $audit "file=$name"
        if {[file extension $name] eq ".sv"} {
            dict set unique_sv $name 1
        }
    }
}
puts $audit "unique_sv_count=[dict size $unique_sv]"
puts $audit "required_verilog_define=UNIFIED_USE_XPM_MEMORY"
puts $audit "core_xpm_libraries=[get_property XPM_LIBRARIES $core]"
puts $audit "memory_file=$memory_relative"
puts $audit "packaging_top=unified_snn_native_wrapper"
puts $audit "production_top=unified_h40_final_composition_top_v1"
close $audit
if {[dict size $unique_sv] != [llength $source_files]} {
    error "packaged SV inventory differs from source closure"
}
close_project

# Catalog registration checks metadata, not implementation or board integration.
create_project native_ip_consumer [file join $output_root consumer] -part xc7vx690tffg1761-3
set original_directory [pwd]
cd [get_property DIRECTORY [current_project]]
set original_argv $argv
set argv [list [file join $output_root repository]]
source [file join $source_root register_repository.tcl]
set argv $original_argv
set definition [get_ipdefs -all open.local:user:unified_snn_native:1.0]
if {[llength $definition] != 1} {
    error "native IP was not uniquely registered in a fresh consumer project"
}
create_ip -vlnv open.local:user:unified_snn_native:1.0 -module_name snn_native_instance
generate_target instantiation_template [get_ips snn_native_instance]
report_ip_status -file [file join $output_root consumer_ip_status.rpt]
set instance [get_ips snn_native_instance]
if {[get_property IS_LOCKED $instance]} {
    error "native IP instance is locked in the fresh consumer project"
}
set status [open [file join $output_root vivado_status.txt] w]
puts $status "status=pass"
puts $status "scope=ip_integrity_and_catalog_registration"
puts $status "synthesis=not_run"
puts $status "implementation=not_run"
puts $status "bitstream=not_run"
close $status
close_project
cd $original_directory
