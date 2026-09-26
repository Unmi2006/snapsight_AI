<#
.SYNOPSIS
    SnapSight AI -- one-shot setup for the target Windows ARM64 Snapdragon PC.

.DESCRIPTION
    Creates the backend venv, installs the CPU-safe base requirements,
    builds the frontend for production, then runs the deployment
    preflight check so you know exactly what's ready before you demo.

    Deliberately does NOT install requirements-directml.txt or
    requirements-arm64-npu.txt for you -- those each REPLACE a package
    that conflicts with the base 'onnxruntime' package (same import
    name, can't have both), so that's a manual, deliberate step. This
    script prints the exact commands at the end instead of guessing
    which one you want.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File deploy\setup.ps1
#>

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host " SnapSight AI -- setup" -ForegroundColor Cyan
Write-Host "===============================================================" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 0. Sanity-check the Python we're about to build the venv with. An x64
#    Python running under Windows-on-ARM emulation installs and imports
#    fine -- it just can never load QNNExecutionProvider (the NPU path),
#    which is invisible until you specifically check for it. Warn now,
#    loudly, instead of finding out after a demo goes CPU-only.
# ---------------------------------------------------------------------------
$machine = & python -c "import platform; print(platform.machine())"
$arch = $env:PROCESSOR_ARCHITECTURE
Write-Host "`nHost architecture reported by Windows: $arch"
Write-Host "Python interpreter architecture:       $machine"
if ($arch -eq "ARM64" -and $machine -ne "ARM64") {
    Write-Host "WARNING: this looks like an x64 Python running under ARM64 emulation." -ForegroundColor Yellow
    Write-Host "         QNNExecutionProvider (the Hexagon NPU) will NEVER load with this interpreter." -ForegroundColor Yellow
    Write-Host "         Install a native ARM64 Python 3.11.x build and re-run this script." -ForegroundColor Yellow
    Write-Host ""
}

# ---------------------------------------------------------------------------
# 1. Backend venv + base (CPU-safe) requirements
# ---------------------------------------------------------------------------
Push-Location "$RepoRoot\backend"
if (-not (Test-Path ".\venv")) {
    Write-Host "`nCreating venv..." -ForegroundColor Cyan
    python -m venv venv
} else {
    Write-Host "`nvenv already exists, reusing it." -ForegroundColor DarkGray
}

Write-Host "Installing base requirements (CPU-safe -- always works, always the fallback)..." -ForegroundColor Cyan
& .\venv\Scripts\pip.exe install --upgrade pip
& .\venv\Scripts\pip.exe install -r requirements.txt
Pop-Location

# ---------------------------------------------------------------------------
# 2. Frontend production build
# ---------------------------------------------------------------------------
Push-Location "$RepoRoot\frontend"
Write-Host "`nInstalling frontend deps..." -ForegroundColor Cyan
npm install
Write-Host "Building frontend for production (backend will serve this directly)..." -ForegroundColor Cyan
npm run build
Pop-Location

# ---------------------------------------------------------------------------
# 3. Preflight report
# ---------------------------------------------------------------------------
Write-Host "`nRunning deployment preflight check..." -ForegroundColor Cyan
Push-Location "$RepoRoot\backend"
& .\venv\Scripts\python.exe scripts\check_deployment.py
$preflightExit = $LASTEXITCODE
Pop-Location

Write-Host "`n===============================================================" -ForegroundColor Cyan
Write-Host " Base setup done. Optional next steps for real NPU/GPU speed:" -ForegroundColor Cyan
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host "  DirectML GPU (any Windows machine):"
Write-Host "    backend\venv\Scripts\pip.exe uninstall onnxruntime onnxruntime-genai -y"
Write-Host "    backend\venv\Scripts\pip.exe install -r backend\requirements-directml.txt"
Write-Host ""
Write-Host "  QNN NPU (native ARM64 Windows on the Snapdragon PC ONLY):"
Write-Host "    backend\venv\Scripts\pip.exe uninstall onnxruntime onnxruntime-directml -y"
Write-Host "    backend\venv\Scripts\pip.exe install -r backend\requirements-arm64-npu.txt"
Write-Host ""
Write-Host "  Then re-run this script's preflight check (or deploy\run.ps1, which"
Write-Host "  runs on top of whichever variant is currently installed) and check"
Write-Host "  http://localhost:8000/api/hardware/verify once the server is up --"
Write-Host "  that endpoint runs a live micro-benchmark to CONFIRM which execution"
Write-Host "  provider actually executed, not just which one you asked for."
Write-Host ""
Write-Host "  Run the app now: powershell -ExecutionPolicy Bypass -File deploy\run.ps1"

exit $preflightExit
