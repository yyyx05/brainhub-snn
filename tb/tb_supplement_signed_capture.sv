`timescale 1ns/1ps
module tb_supplement_signed_capture;
    logic [127:0] packed_values;
    integer fd;
    string output_path;
    initial begin
        if (!$value$plusargs("OUTPUT=%s", output_path)) $fatal(1, "OUTPUT required");
        fd = $fopen(output_path, "w");
        if (fd == 0) $fatal(1, "output open failed");
        if (!$value$plusargs("PATTERN=%h", packed_values)) $fatal(1, "PATTERN required");
        #1;
        for (int lane = 0; lane < 8; lane++) begin
            $fwrite(fd, "%0d,%0d,%0d\n", lane,
                $signed(packed_values[lane*16 +: 16]),
                int'($signed(packed_values[lane*16 +: 16])));
        end
        $fclose(fd);
        $display("SIGNED_CAPTURE_UNIT_COMPLETED");
        $finish;
    end
endmodule
