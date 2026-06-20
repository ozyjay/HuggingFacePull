param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8019,
    [string]$LibraryDir = ""
)

. (Join-Path $PSScriptRoot "common.ps1")

$root = Get-ProjectRoot
$webCommand = Get-VenvCommand -Name "hfpull-web"

Set-Location $root

$args = @("--host", $HostName, "--port", "$Port")
if ($LibraryDir) {
    $args += @("--library-dir", $LibraryDir)
}

Invoke-Checked $webCommand @args
