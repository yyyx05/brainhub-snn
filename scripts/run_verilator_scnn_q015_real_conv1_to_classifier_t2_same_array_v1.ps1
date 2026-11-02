param(
  [string]$WorkDirectory,
  [string]$EvidenceDirectory,
  [ValidateRange(1, 4)][int]$Jobs = 1,
  [ValidateSet('0', '1', '2')][string]$HostOptimization = '0',
  [switch]$BuildExisting
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$work = if ($WorkDirectory) { [IO.Path]::GetFullPath($WorkDirectory) } else {
  Join-Path $root 'work\scnn_q015_real_conv1_to_classifier_t2_same_array_v3'
}
$obj = Join-Path $work 'obj_dir'
$verilator = 'C:\msys64\mingw64\bin\verilator_bin.exe'
$evidence = if ($EvidenceDirectory) { [IO.Path]::GetFullPath($EvidenceDirectory) } else {
  Join-Path $root 'results\paper_evidence\rtl\scnn_full_graph_same_array_20260927'
}
$log = Join-Path $evidence `
  'verilator_real_conv1_to_classifier_t2_same_array.log'
$previousVerilatorRoot = $env:VERILATOR_ROOT
$previousPath = $env:PATH
New-Item -ItemType Directory -Force -Path $work | Out-Null
New-Item -ItemType Directory -Force -Path $evidence | Out-Null
$env:VERILATOR_ROOT = 'C:\msys64\mingw64\share\verilator'
$env:PATH = "C:\msys64\mingw64\bin;C:\msys64\usr\bin;$env:PATH"
try {
  $required = @(
    'models\scnn_dvsgesture_vgg9_q0_15\bank_01.hex',
    'models\scnn_dvsgesture_vgg9_q0_15\bank_02.hex',
    'models\scnn_dvsgesture_vgg9_q0_15\bank_03.hex',
    'models\scnn_dvsgesture_vgg9_q0_15\bank_04.hex',
    'models\scnn_dvsgesture_vgg9_q0_15\bank_05.hex',
    'models\scnn_dvsgesture_vgg9_q0_15\bank_06.hex',
    'models\scnn_dvsgesture_vgg9_q0_15\bank_07.hex',
    'models\scnn_dvsgesture_vgg9_q0_15\bank_08.hex',
    'models\scnn_dvsgesture_vgg9_q0_15\bank_12.hex',
    'results\paper_evidence\rtl\scnn_full_graph_same_array_20260927\t2_reference\input_sample0_t1_q8_8.hex',
    'results\paper_evidence\rtl\scnn_full_graph_same_array_20260927\t2_reference\conv7_t1_expected_spike.hex',
    'results\paper_evidence\rtl\scnn_full_graph_same_array_20260927\t2_reference\conv8_t1_expected_pooled_q8_8.hex',
    'results\paper_evidence\rtl\scnn_full_graph_same_array_20260927\t2_reference\classifier_t1_expected_centered_cumulative_q8_8.hex'
  )
  foreach ($file in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $root $file))) {
      throw "Missing fixture: $file"
    }
  }

  $sources = @(
    'rtl\common\unified_elastic_pipeline.sv',
    'rtl\common\unified_elastic_fifo2.sv',
    'rtl\common\unified_q8_8_mul_multipass.sv',
    'rtl\common\unified_scnn_q8_8_trace_state_engine.sv',
    'rtl\memory\unified_dual_role_weight_store_candidate_v1.sv',
    'rtl\memory\unified_dual_role_weight_store_p4fc_candidate_v1.sv',
    'rtl\memory\unified_dual_role_weight_store_p4all_candidate_v1.sv',
    'rtl\memory\unified_dual_role_state_trace_store_candidate_v2.sv',
    'rtl\memory\unified_array16_weight_refill_dispatcher_v1.sv',
    'rtl\tiles\unified_dual_role_dsp_lane_candidate_v3.sv',
    'rtl\tiles\unified_dual_role_h40_tile_candidate_v3.sv',
    'rtl\tiles\unified_dual_role_h40_tile_p4fc_dual_accum_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_h40_tile_p4all_dual_accum_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_dsp_lane_p4_cascade_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_h40_tile_p4all_cascade_accum_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_dsp_add48_lane_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_p4fc_pair_merge10_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_current_normalizer10_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_lif10_arithmetic_candidate_v3.sv',
    'rtl\tiles\unified_dual_role_trace10_arithmetic_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_h40_neuron_region_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_h40_neuron_region_p4fc_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_h40_group_sequencer_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_h40_stationary_tile_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_h40_stationary_region4_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_h40_stationary_array12_candidate_v1.sv',
    'rtl\control\unified_h40_topology_geometry_v1.sv',
    'rtl\control\unified_binary_p4_destination_walker_v1.sv',
    'rtl\control\unified_scnn_h40_spatial_wave_scheduler_v1.sv',
    'rtl\control\unified_scnn_h40_classifier_p4_scheduler_v1.sv',
    'rtl\control\unified_h40_array16_topology_owner_v1.sv',
    'rtl\tiles\unified_binary_p4_h40_array12_path_v1.sv',
    'rtl\learning\unified_scnn_h40_tp_observation_serializer_v1.sv',
    'rtl\learning\unified_scnn_tp_surrogate_type1_q8_8.sv',
    'rtl\learning\unified_scnn_h40_conv8_scalar_observation_fork_v1.sv',
    'rtl\learning\unified_scnn_h40_tp_input_prescan_v1.sv',
    'rtl\learning\unified_scnn_h40_classifier_feature_commit_adapter_v1.sv',
    'rtl\learning\unified_scnn_h40_classifier_logit_wave_adapter_v1.sv',
    'rtl\top\unified_scnn_h40_observation_writeback_v1.sv',
    'rtl\top\unified_scnn_h40_array16_layer_client_v1.sv',
    'rtl\top\unified_scnn_h40_classifier_tail_client_v1.sv',
    'rtl\top\unified_scnn_h40_array16_graph_client_v1.sv',
    'rtl\top\unified_h40_array16_three_topology_top_v1.sv',
    'rtl\learning\unified_scnn_classifier_observation_replay.sv',
    'rtl\learning\unified_tp_row_softmax_q6.sv',
    'rtl\learning\unified_fc_tp_output_error_frontend.sv',
    'rtl\learning\unified_fc_tp_w2_contribution_scheduler.sv',
    'rtl\learning\unified_fc_tp_w2_online_frontend.sv',
    'rtl\learning\unified_tp_direct_residual_update.sv',
    'rtl\learning\unified_tp_q015_magnitude_residual_update.sv',
    'rtl\learning\unified_tp_parameter_update_adapter.sv',
    'rtl\learning\unified_tp_active_tensor_context.sv',
    'rtl\learning\unified_tp_active_tensor_update_path.sv',
    'rtl\learning\unified_tp_parameter_rmw_service.sv',
    'rtl\learning\unified_fc_tp_sign_request_adapter.sv',
    'rtl\learning\unified_tp_q015_magnitude_scaler.sv',
    'rtl\learning\unified_fc_tp_gradient_request_adapter.sv',
    'rtl\learning\unified_fc_tp_gradient_rmw_path.sv',
    'rtl\learning\unified_tp_address_major_sign_reducer.sv',
    'rtl\learning\unified_fc_tp_w2_learning_slice.sv',
    'rtl\learning\unified_fc_tp_w2_online_learning_path.sv',
    'rtl\learning\unified_scnn_tp_classifier_online_learning_path.sv',
    'rtl\learning\unified_scnn_tp_classifier_final_core_learning_bridge.sv',
    'rtl\learning\unified_tp_shared_forward_rmw_store.sv',
    'rtl\learning\unified_scnn_classifier_cache_refill_publisher_v1.sv',
    'rtl\learning\unified_scnn_classifier_online_cache_learning_client_v1.sv',
    'rtl\top\unified_scnn_h40_array16_graph_online_learning_system_v1.sv',
    'tb\tb_unified_scnn_q015_real_conv1_to_classifier_t2_same_array_v1.sv'
  ) | ForEach-Object { Join-Path $root $_ }

  # Host optimization changes simulator cost, not the RTL clock or stimulus.
  $buildWatch = [Diagnostics.Stopwatch]::StartNew()
  $hostMakefile = (Join-Path $PSScriptRoot 'scnn_fullgraph_host_build.mk').Replace('\', '/')
  $makeFlags = "-f $hostMakefile OPT_FAST=-O$HostOptimization OPT_GLOBAL=-O$HostOptimization"
  Write-Output "SUPPLEMENT_BUILD_START host_optimization=$HostOptimization stimulus_optimization=0 jobs=$Jobs existing=$BuildExisting"
  if ($BuildExisting) {
    # Resume only a just-generated, unchanged source build, never stale objects.
    $generatedMakefile = 'Vtb_unified_scnn_q015_real_conv1_to_classifier_t2_same_array_v1.mk'
    if (-not (Test-Path -LiteralPath (Join-Path $obj $generatedMakefile))) {
      throw 'No generated Makefile to resume'
    }
    & 'C:\msys64\usr\bin\make.exe' -C $obj -f $generatedMakefile `
      -f $hostMakefile "OPT_FAST=-O$HostOptimization" "OPT_GLOBAL=-O$HostOptimization" -j $Jobs
  } else {
  & $verilator --binary -j $Jobs --timing --assert -Wall -Wno-fatal `
    --output-groups 4 --output-split 5000 --output-split-cfuncs 2000 `
    -MAKEFLAGS $makeFlags `
    -Wno-DECLFILENAME -Wno-PROCASSINIT -Wno-PINCONNECTEMPTY `
    -Wno-PINMISSING -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND `
    -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-BLKSEQ -Wno-CMPCONST `
    -Wno-SYNCASYNCNET -Wno-TIMESCALEMOD -Wno-WIDTHCONCAT `
    -CFLAGS '-D_GLIBCXX_USE_CXX11_ABI=0' `
    --top-module tb_unified_scnn_q015_real_conv1_to_classifier_t2_same_array_v1 `
    --Mdir $obj $sources
  }
  $buildExit = $LASTEXITCODE
  $buildWatch.Stop()
  Write-Output "SUPPLEMENT_BUILD_END seconds=$($buildWatch.Elapsed.TotalSeconds) exit=$buildExit"
  if ($buildExit -ne 0) { exit $buildExit }

  Push-Location $root
  try {
    $runWatch = [Diagnostics.Stopwatch]::StartNew()
    Write-Output 'SUPPLEMENT_SIM_START'
    & (Join-Path $obj `
      'Vtb_unified_scnn_q015_real_conv1_to_classifier_t2_same_array_v1.exe') `
      2>&1 | Tee-Object -FilePath $log
    $runExit = $LASTEXITCODE
    $runWatch.Stop()
    Write-Output "SUPPLEMENT_SIM_END seconds=$($runWatch.Elapsed.TotalSeconds) exit=$runExit"
    exit $runExit
  } finally {
    Pop-Location
  }
} finally {
  $env:VERILATOR_ROOT = $previousVerilatorRoot
  $env:PATH = $previousPath
}
