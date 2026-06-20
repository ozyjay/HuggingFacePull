param(
    [switch]$Recreate
)

. (Join-Path $PSScriptRoot "common.ps1")

$root = Get-ProjectRoot
$venv = Join-Path $root ".venv"

Set-Location $root

if ($Recreate -and (Test-Path $venv)) {
    Remove-Item -Recurse -Force $venv
}

if (-not (Test-Path $venv)) {
    Invoke-Checked "python3" "-m" "venv" ".venv"
}

$python = Get-VenvPython
Invoke-Checked $python "-m" "pip" "install" "--upgrade" "pip"
Invoke-Checked $python "-m" "pip" "install" "-e" ".[dev]"

Write-Host "Setup complete. Python: $python"
