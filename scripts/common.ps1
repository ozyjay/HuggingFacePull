Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-ProjectRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Get-VenvPython {
    $root = Get-ProjectRoot
    $candidates = @(
        (Join-Path $root ".venv/bin/python"),
        (Join-Path $root ".venv/Scripts/python.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return "python3"
}

function Get-VenvCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $root = Get-ProjectRoot
    $candidates = @(
        (Join-Path $root ".venv/bin/$Name"),
        (Join-Path $root ".venv/Scripts/$Name.exe"),
        (Join-Path $root ".venv/Scripts/$Name")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $Name
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$ArgumentList
    )

    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($ArgumentList -join ' ')"
    }
}
