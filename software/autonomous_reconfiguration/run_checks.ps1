[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = $root
$env:PYTHONDONTWRITEBYTECODE = "1"

python -m unittest discover -s (Join-Path $root "tests") -p "test_*.py" -v
if ($LASTEXITCODE -ne 0) {
    throw "Autonomous reconfiguration unit tests failed."
}
python (Join-Path $root "validate_integration.py")
if ($LASTEXITCODE -ne 0) {
    throw "Frontend-to-RTL integration contract validation failed."
}
python -m snn_reconfig.fingerprint_cli `
    (Join-Path $root "examples\normalized_event_samples_smoke.json") `
    --output (Join-Path $root "examples\output_normalized_event_fingerprint.json")
python -m snn_reconfig.dataset_cli shd-h5 `
    (Join-Path $root "..\..\imports\srnn\dataset_cache\SHD\shd_train.h5") `
    --indices 0:32 `
    --output (Join-Path $root "examples\output_shd_project_fingerprint.json")
python -m snn_reconfig.dataset_cli dvsgesture-q8-8 `
    (Join-Path $root "..\..\imports\scnn\datasets\DVSGesture\hardware\test_10x20_inputs_q8_8.hex") `
    --sample-count 10 `
    --output (Join-Path $root "examples\output_dvsgesture_project_fingerprint.json")
python -m snn_reconfig.cli `
    (Join-Path $root "examples\output_shd_project_fingerprint.json") `
    --profiles (Join-Path $root "catalog\hardware_profiles.json") `
    --prototypes (Join-Path $root "catalog\fingerprint_prototypes.json") `
    --learning off `
    --output (Join-Path $root "examples\output_shd_project_decision.json")
python -m snn_reconfig.cli `
    (Join-Path $root "examples\output_dvsgesture_project_fingerprint.json") `
    --profiles (Join-Path $root "catalog\hardware_profiles.json") `
    --prototypes (Join-Path $root "catalog\fingerprint_prototypes.json") `
    --learning off `
    --output (Join-Path $root "examples\output_dvsgesture_project_decision.json")

$shdDecision = Get-Content -Raw (Join-Path $root "examples\output_shd_project_decision.json") | ConvertFrom-Json
if ($shdDecision.status -ne "hardware_profile_selected" -or
    $shdDecision.selected_profile.profile_code -ne "0x10") {
    throw "Project SHD data did not select the SRNN 0x10 profile."
}
$dvsDecision = Get-Content -Raw (Join-Path $root "examples\output_dvsgesture_project_decision.json") | ConvertFrom-Json
if ($dvsDecision.status -ne "hardware_profile_selected" -or
    $dvsDecision.selected_profile.profile_code -ne "0x20") {
    throw "Project DVS-Gesture data did not select the SCNN 0x20 profile."
}
python -m snn_reconfig.cli `
    (Join-Path $root "examples\nmnist_query.json") `
    --profiles (Join-Path $root "catalog\hardware_profiles.json") `
    --prototypes (Join-Path $root "catalog\fingerprint_prototypes.json") `
    --learning off `
    --output (Join-Path $root "examples\output_nmnist.json")
python -m snn_reconfig.cli `
    (Join-Path $root "examples\smnist_unsupported_query.json") `
    --profiles (Join-Path $root "catalog\hardware_profiles.json") `
    --prototypes (Join-Path $root "catalog\fingerprint_prototypes.json") `
    --learning auto `
    --output (Join-Path $root "examples\output_smnist_unsupported.json")

$projectRoot = Resolve-Path (Join-Path $root "..\..")
$projectEntryOutput = Join-Path $root "examples\output_project_entry_nmnist.json"
python -B (Join-Path $projectRoot "scripts\run_autonomous_reconfiguration.py") `
    --fingerprint-json (Join-Path $root "examples\nmnist_query.json") `
    --learning off `
    --output $projectEntryOutput
if ($LASTEXITCODE -ne 0) {
    throw "Project-level autonomous reconfiguration entry failed."
}
$projectEntry = Get-Content -Raw $projectEntryOutput | ConvertFrom-Json
if ($projectEntry.mode -ne "dry_run_contract_trace" -or
    $projectEntry.decision.selected_profile.profile_code -ne "0x01" -or
    $projectEntry.decision.selected_profile.dispatch_contract_id -ne "fc_native_stream_v1" -or
    $projectEntry.dry_run_events[-1].operation -ne "resume_accepting_work") {
    throw "Project-level entry did not emit the qualified FC dispatch sequence."
}

$projectShdOutput = Join-Path $root "examples\output_project_entry_shd.json"
python -B (Join-Path $projectRoot "scripts\run_autonomous_reconfiguration.py") `
    --shd-h5 (Join-Path $projectRoot "imports\srnn\dataset_cache\SHD\shd_train.h5") `
    --shd-indices 0:32 `
    --learning off `
    --output $projectShdOutput
$projectShd = Get-Content -Raw $projectShdOutput | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or
    $projectShd.decision.selected_profile.profile_code -ne "0x10" -or
    $projectShd.decision.selected_profile.dispatch_contract_id -ne "srnn_native_stream_v1") {
    throw "Project-level entry did not emit the qualified SRNN dispatch sequence."
}

$projectDvsOutput = Join-Path $root "examples\output_project_entry_dvsgesture.json"
python -B (Join-Path $projectRoot "scripts\run_autonomous_reconfiguration.py") `
    --dvsgesture-q8-8 (Join-Path $projectRoot "imports\scnn\datasets\DVSGesture\hardware\test_10x20_inputs_q8_8.hex") `
    --dvs-sample-count 10 `
    --learning off `
    --output $projectDvsOutput
$projectDvs = Get-Content -Raw $projectDvsOutput | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or
    $projectDvs.decision.selected_profile.profile_code -ne "0x20" -or
    $projectDvs.decision.selected_profile.dispatch_contract_id -ne "scnn_graph_frame_service_v1") {
    throw "Project-level entry did not emit the qualified SCNN dispatch sequence."
}

$demoOutput = Join-Path $projectRoot "results\paper_evidence\supplement_20261004\software\autonomous_reconfiguration_demo\report.json"
python -B (Join-Path $projectRoot "scripts\run_fixed_profile_sequence_demo.py") `
    --output $demoOutput
if ($LASTEXITCODE -ne 0) {
    throw "Fixed-profile sequence demo failed."
}
$demo = Get-Content -Raw $demoOutput | ConvertFrom-Json
if ($demo.status -ne "PASS" -or
    -not $demo.checks.unsupported_task_rejected -or
    -not $demo.checks.rejected_task_emitted_zero_transport_events -or
    -not $demo.checks.no_global_reset_operation) {
    throw "Fixed-profile sequence demo did not preserve its rejection/reset boundary."
}

Write-Host "Autonomous reconfiguration checks passed."
