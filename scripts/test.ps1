param(
    [switch]$Install,
    [switch]$SkipNode
)

. (Join-Path $PSScriptRoot "common.ps1")

$root = Get-ProjectRoot
$python = Get-VenvPython

Set-Location $root

if ($Install) {
    Invoke-Checked $python "-m" "pip" "install" "-e" ".[dev]"
}

Invoke-Checked $python "-m" "pytest" "-v"

$pythonFiles = @(
    Get-ChildItem -Path (Join-Path $root "src/huggingface_pull") -Filter "*.py" -File
    Get-ChildItem -Path (Join-Path $root "tests") -Filter "*.py" -File
) | ForEach-Object { $_.FullName }

Invoke-Checked $python "-m" "py_compile" @pythonFiles

if (-not $SkipNode) {
    $node = Get-Command node -ErrorAction SilentlyContinue
    if ($node) {
        Invoke-Checked "node" "--check" (Join-Path $root "src/huggingface_pull/web/app.js")
        $desktopMain = Join-Path $root "desktop/main.cjs"
        if (Test-Path $desktopMain) {
            Invoke-Checked "node" "--check" $desktopMain
        }
        $desktopPackage = Join-Path $root "desktop/package.cjs"
        if (Test-Path $desktopPackage) {
            Invoke-Checked "node" "--check" $desktopPackage
        }
    } else {
        Write-Host "Skipping JavaScript syntax check because node is not installed."
    }
}
