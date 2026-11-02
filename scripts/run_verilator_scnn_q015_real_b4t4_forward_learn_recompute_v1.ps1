param(
    [string]$WorkDirectory,
    [string]$EvidenceDirectory,
    [switch]$UseExistingFixture,
    [ValidateRange(1, 8)][int]$Jobs = 4
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Work = if ($WorkDirectory) { [IO.Path]::GetFullPath($WorkDirectory) } else {
    Join-Path $Root 'sim_work\scnn_q015_real_b4t4_forward_learn_recompute_v1'
}
$Obj = Join-Path $Work 'obj_dir'
$Evidence = if ($EvidenceDirectory) { [IO.Path]::GetFullPath($EvidenceDirectory) } else {
    Join-Path $Root 'results\paper_evidence\rtl\scnn_q015_real_b4t4_forward_learn_recompute_20260928'
}
$Verilator = 'C:\msys64\mingw64\bin\verilator_bin.exe'

if (-not (Test-Path -LiteralPath $Verilator)) {
    throw "Verilator not found: $Verilator"
}
New-Item -ItemType Directory -Force -Path $Work, $Evidence | Out-Null

if (-not $UseExistingFixture) {
    & python (Join-Path $Root 'scripts\build_scnn_q015_real_b4t4_forward_learn_recompute_fixture_v1.py')
    if ($LASTEXITCODE -ne 0) {
        throw "Real B4xT4 reference fixture generation failed: $LASTEXITCODE"
    }
}

$PreviousVerilatorRoot = $env:VERILATOR_ROOT
$PreviousPath = $env:PATH
$env:VERILATOR_ROOT = 'C:\msys64\mingw64\share\verilator'
$env:PATH = "C:\msys64\mingw64\bin;C:\msys64\usr\bin;$env:PATH"
try {
    $Sources = @(
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
        'rtl\control\unified_scnn_h40_classifier_p4_scheduler_v1.sv',
        'rtl\tiles\unified_binary_p4_h40_array12_path_v1.sv',
        'rtl\learning\unified_scnn_h40_tp_observation_serializer_v1.sv',
        'rtl\learning\unified_scnn_h40_classifier_logit_wave_adapter_v1.sv',
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
        'tb\tb_unified_scnn_q015_real_b4t4_forward_learn_recompute_v1.sv'
    ) | ForEach-Object { Join-Path $Root $_ }

    $BuildWatch = [Diagnostics.Stopwatch]::StartNew()
    Write-Output 'SUPPLEMENT_BUILD_START'
    $BuildOutput = & $Verilator --binary -j $Jobs --timing --assert -Wall -Wno-fatal `
        -Wno-DECLFILENAME -Wno-PROCASSINIT -Wno-PINCONNECTEMPTY `
        -Wno-PINMISSING -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND `
        -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-BLKSEQ `
        -Wno-CMPCONST -Wno-SYNCASYNCNET -Wno-TIMESCALEMOD `
        -Wno-WIDTHCONCAT -Wno-MULTIDRIVEN `
        -CFLAGS '-D_GLIBCXX_USE_CXX11_ABI=0' `
        --top-module tb_unified_scnn_q015_real_b4t4_forward_learn_recompute_v1 `
        --Mdir $Obj $Sources 2>&1
    $BuildExit = $LASTEXITCODE
    $BuildWatch.Stop()
    $BuildOutput | Set-Content -LiteralPath (Join-Path $Work 'build.log') -Encoding utf8
    $BuildOutput | Select-Object -Last 12 | Write-Output
    Write-Output "SUPPLEMENT_BUILD_END seconds=$($BuildWatch.Elapsed.TotalSeconds) exit=$BuildExit"
    if ($BuildExit -ne 0) {
        throw "Real B4xT4 forward/learn/recompute build failed: $BuildExit"
    }

    Push-Location $Root
    try {
        $RunWatch = [Diagnostics.Stopwatch]::StartNew()
        Write-Output 'SUPPLEMENT_SIM_START'
        $RunOutput = & (Join-Path $Obj 'Vtb_unified_scnn_q015_real_b4t4_forward_learn_recompute_v1.exe') 2>&1
        $RunExit = $LASTEXITCODE
        $RunWatch.Stop()
        Write-Output "SUPPLEMENT_SIM_END seconds=$($RunWatch.Elapsed.TotalSeconds) exit=$RunExit"
    } finally {
        Pop-Location
    }
    $RunOutput | Set-Content -LiteralPath (Join-Path $Work 'run.log') -Encoding utf8
    $RunOutput | Set-Content -LiteralPath (Join-Path $Evidence 'verilator.log') -Encoding utf8
    $RunOutput | Write-Output
    if ($RunExit -ne 0 -or
        ($RunOutput -join "`n") -notmatch 'SCNN_Q015_REAL_B4T4_FORWARD_LEARN_RECOMPUTE_PASS') {
        throw "Real B4xT4 forward/learn/recompute regression failed: $RunExit"
    }
} finally {
    $env:VERILATOR_ROOT = $PreviousVerilatorRoot
    $env:PATH = $PreviousPath
}
