[CmdletBinding()]
param(
    [string]$OutputRoot,
    [switch]$RegenerateTop
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
$allowedBase = [System.IO.Path]::GetFullPath((Join-Path $projectRoot "build\ip"))
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $allowedBase "unified_snn_native_ip_source"
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)

if (-not $OutputRoot.StartsWith($allowedBase + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputRoot must stay under $allowedBase"
}

$topSource = Join-Path $projectRoot "rtl\top\unified_h40_final_composition_top_v1.sv"
if ($RegenerateTop) {
    $generator = Join-Path $projectRoot "scripts\generate_unified_h40_final_composition_top_v1.py"
    if (-not (Test-Path -LiteralPath $generator -PathType Leaf)) {
        throw "Final composition generator not found: $generator"
    }
    python $generator
    if ($LASTEXITCODE -ne 0) {
        throw "Final composition generator failed with exit code $LASTEXITCODE"
    }
}
if (-not (Test-Path -LiteralPath $topSource -PathType Leaf)) {
    throw "Final composition source not found: $topSource"
}

if (Test-Path -LiteralPath $OutputRoot) {
    Remove-Item -LiteralPath $OutputRoot -Recurse -Force
}
New-Item -ItemType Directory -Path (Join-Path $OutputRoot "src") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $OutputRoot "constraints") -Force | Out-Null

$closureList = Join-Path $OutputRoot "reachable_sources.f"
$closureReport = Join-Path $OutputRoot "source_closure_report.json"
python (Join-Path $scriptRoot "collect_reachable_sources.py") `
    --project-root $projectRoot `
    --output-list $closureList `
    --report $closureReport
if ($LASTEXITCODE -ne 0) {
    throw "Reachable source collection failed with exit code $LASTEXITCODE"
}

$reachableSources = Get-Content -LiteralPath $closureList | Where-Object { $_.Trim() }
foreach ($relativeSource in $reachableSources) {
    $sourcePath = Join-Path $projectRoot $relativeSource
    $destinationPath = Join-Path (Join-Path $OutputRoot "src") $relativeSource
    New-Item -ItemType Directory -Path (Split-Path -Parent $destinationPath) -Force | Out-Null
    Copy-Item -LiteralPath $sourcePath -Destination $destinationPath -Force
}
$memoryRelative = "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem"
$memoryDestination = Join-Path $OutputRoot $memoryRelative
New-Item -ItemType Directory -Path (Split-Path -Parent $memoryDestination) -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot $memoryRelative) -Destination $memoryDestination
Copy-Item -LiteralPath (Join-Path $projectRoot "vivado\ooc\unified_h40_final_composition_v1_ooc.xdc") `
    -Destination (Join-Path $OutputRoot "constraints\unified_h40_final_composition_v1_ooc.xdc") -Force
Copy-Item -LiteralPath (Join-Path $scriptRoot "package_ip.tcl") -Destination (Join-Path $OutputRoot "package_ip.tcl") -Force
Copy-Item -LiteralPath (Join-Path $scriptRoot "register_repository.tcl") -Destination (Join-Path $OutputRoot "register_repository.tcl") -Force
Copy-Item -LiteralPath (Join-Path $scriptRoot "README_CN.md") -Destination (Join-Path $OutputRoot "README_CN.md") -Force
python (Join-Path $scriptRoot "generate_native_wrapper.py") $topSource `
    (Join-Path $OutputRoot "src\unified_snn_native_wrapper.v")
if ($LASTEXITCODE -ne 0) {
    throw "Native Verilog adapter generation failed with exit code $LASTEXITCODE"
}

$sourceRoot = Join-Path $OutputRoot "src"
$fileList = foreach ($relativeSource in $reachableSources) {
    "src/" + $relativeSource.Replace("\", "/")
}
[System.IO.File]::WriteAllLines((Join-Path $sourceRoot "filelist.f"), $fileList, [System.Text.UTF8Encoding]::new($false))
$closure = Get-Content -Raw -LiteralPath $closureReport | ConvertFrom-Json

$manifest = [ordered]@{
    schema_version = "1.0"
    ip_name = "unified_snn_native"
    ip_version = "1.0"
    top_module = "unified_h40_final_composition_top_v1"
    top_source = "src/rtl/top/unified_h40_final_composition_top_v1.sv"
    packaging_top_module = "unified_snn_native_wrapper"
    packaging_adapter_source = "src/unified_snn_native_wrapper.v"
    part = "xc7vx690tffg1761-3"
    clock_port = "i_clk"
    reset_port = "i_rst_n"
    reset_polarity = "active_low"
    clock_period_ns = 15.0
    interface_kind = "native_profile_descriptor_memory_handshake"
    source_selection = $closure.selection_method
    top_regenerated = [bool]$RegenerateTop
    required_verilog_defines = @("UNIFIED_USE_XPM_MEMORY")
    required_xpm_libraries = @("XPM_MEMORY")
    auxiliary_source_files = @($memoryRelative)
    input_source_count = $closure.input_source_count
    source_count = $reachableSources.Count
    excluded_unreachable_source_count = $closure.excluded_unreachable_source_count
    generated_utc = [DateTime]::UtcNow.ToString("o")
    claim_boundaries = @(
        "source IP package only",
        "not an AXI block-design peripheral",
        "no MMIO address map assigned",
        "no bitstream or board validation",
        "PPA applies only to sources matching the measured current-source snapshot"
    )
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputRoot "ip_manifest.json") -Encoding utf8

python (Join-Path $scriptRoot "validate_source_package.py") $OutputRoot
if ($LASTEXITCODE -ne 0) {
    throw "Source package validation failed with exit code $LASTEXITCODE"
}

$archivePath = $OutputRoot + ".zip"
if (Test-Path -LiteralPath $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}
Compress-Archive -Path (Join-Path $OutputRoot "*") -DestinationPath $archivePath -CompressionLevel Optimal

Write-Host "Source IP package: $OutputRoot"
Write-Host "Archive: $archivePath"
