create_clock -name p4_clk -period 15.000 [get_ports i_clk]
# Matched synthetic zero-delay interfaces, not board timing constraints.
set data_inputs [get_ports -filter {DIRECTION == IN && NAME != i_clk}]
set_input_delay -clock p4_clk -max 0.000 $data_inputs
set_input_delay -clock p4_clk -min 0.000 $data_inputs
set_output_delay -clock p4_clk -max 0.000 [all_outputs]
set_output_delay -clock p4_clk -min 0.000 [all_outputs]
