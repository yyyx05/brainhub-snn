# The old fullgraph stimulus expands into a very large root coroutine. Keep
# root units at O0/no PCH; reusable cycle-evaluated tile modules retain O1.
Vscgraph___024root%.o: override OPT_FAST = -O0
Vscgraph___024root%.o: override VK_PCH_I_FAST =
