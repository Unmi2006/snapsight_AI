<#
.SYNOPSIS
    SnapSight AI -- launch the packaged app (one process, one port).

.DESCRIPTION
    Starts the backend, which serves both the API and the built frontend
    from frontend\dist (see app\main.py) -- no separate frontend dev
    server needed. Requires deploy\setup.ps1 (or the equivalent manual
    steps in README.md) to have been run first.

.PARAMETER Port
    Port to listen on. Defaults to 8000.

.PARAMETER SkipBrowser
    Don't auto-open a browser tab.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File deploy\run.ps1
    powershell -ExecutionPolicy Bypass -File deploy\run.ps1 -Port 8080
#>
param(
    [int]$Port = 8000,
    [switch]$SkipBrowser
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

$VenvPython = "$RepoRoot\backend\venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "No venv found at backend\venv -- run deploy\setup.ps1 first." -ForegroundColor Red
    exit 2
}

$Dist = "$RepoRoot\frontend\dist\index.html"
if (-not (Test-Path $Dist)) {
    Write-Host "No frontend production build found ($Dist)." -ForegroundColor Yellow
    Write-Host "The backend will still start and serve the API, but not the UI." -ForegroundColor Yellow
    Write-Host "Run 'npm run build' in frontend\, or deploy\setup.ps1, to fix this." -ForegroundColor Yellow
    Write-Host ""
}

if (-not $SkipBrowser) {
    Start-Job -ScriptBlock {
        param($url)
        Start-Sleep -Seconds 2
        Start-Process $url
    } -ArgumentList "http://localhost:$Port" | Out-Null
}

Write-Host "Starting SnapSight AI on http://localhost:$Port ..." -ForegroundColor Cyan
Push-Location "$RepoRoot\backend"
& $VenvPython -m uvicorn app.main:app --host 0.0.0.0 --port $Port
Pop-Location
