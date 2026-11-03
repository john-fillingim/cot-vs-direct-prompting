# Run the full pipeline on Windows (same steps as `make all`).
# Usage:  .\run_all.ps1            main run
#         .\run_all.ps1 -Pilot     30-problem pilot
param([switch]$Pilot)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$flag = @()
if ($Pilot) { $flag = @("--pilot") }

function Step([string]$module, [string[]]$extra) {
    Write-Host "==> python -m $module $extra"
    & $py -m $module @extra
    if ($LASTEXITCODE -ne 0) { throw "$module failed with exit code $LASTEXITCODE" }
}

Step "src.sample_problems" @()
Step "src.run_experiment" $flag
Step "src.clean" $flag
if ($Pilot) { Step "src.power_analysis" @("--pilot") }  # power is a pre-study step
Step "src.analyze" $flag
Step "src.figures" $flag
Write-Host "Done."
