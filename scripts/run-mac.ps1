param(
    [string]$BackendURL = "http://127.0.0.1:8019",
    [string]$PythonExecutable = "",
    [ValidateSet("debug", "release")]
    [string]$Configuration = "debug",
    [switch]$NoAutoLaunchBackend
)

. (Join-Path $PSScriptRoot "common.ps1")

$root = Get-ProjectRoot
$macRoot = Join-Path $root "mac/HuggingFacePullMac"

if (-not (Test-Path $macRoot)) {
    throw "Mac app package not found: $macRoot"
}

$appArguments = @("-backendURL", $BackendURL)

if ($PythonExecutable) {
    $appArguments += @("-pythonExecutable", $PythonExecutable)
}

if ($NoAutoLaunchBackend) {
    $appArguments += @("-autoLaunchBackend", "false")
}

$buildArguments = @(
    "build",
    "--package-path", $macRoot,
    "--configuration", $Configuration,
    "--product", "HuggingFacePullMac"
)
Invoke-Checked -FilePath "swift" -ArgumentList $buildArguments

$binPathOutput = & "swift" "build" "--package-path" $macRoot "--configuration" $Configuration "--show-bin-path"
if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code $LASTEXITCODE`: swift build --package-path $macRoot --configuration $Configuration --show-bin-path"
}

$binPath = ($binPathOutput | Select-Object -Last 1).Trim()
$executable = Join-Path $binPath "HuggingFacePullMac"
if (-not (Test-Path $executable)) {
    throw "Mac app executable not found after build: $executable"
}

$launchLog = Join-Path $root ".huggingfacepull-mac.log"
$launchArguments = @(
    "-lc",
    'log="$1"; shift; nohup "$@" > "$log" 2>&1 < /dev/null &',
    "hfp-launch",
    $launchLog,
    $executable
) + $appArguments

Invoke-Checked -FilePath "/bin/bash" -ArgumentList $launchArguments

Write-Host "Launched HuggingFacePullMac. Log: $launchLog"
