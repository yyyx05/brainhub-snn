# The timed stimulus expands into one very large coroutine. Optimizing it
# exhausts host compile resources; keep the cycle-evaluated DUT code optimized.
Vtb_unified_scnn_q015_real_conv1_to_classifier_t2_same_array_v1___024root__0.o: override OPT_FAST = -O0
# GCC cannot reuse the O1 precompiled header for this O0 translation unit.
# The generated .cpp already includes the unsuffixed header explicitly.
Vtb_unified_scnn_q015_real_conv1_to_classifier_t2_same_array_v1___024root__0.o: override VK_PCH_I_FAST =
