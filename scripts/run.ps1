param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8019,
    [string]$LibraryDir = ""
)

. (Join-Path $PSScriptRoot "common.ps1")

$root = Get-ProjectRoot
$webCommand = Get-VenvCommand -Name "hfpull-web"

Set-Location $root

$env:HF_HUB_DISABLE_XET = "1"
Remove-Item Env:HF_XET_HIGH_PERFORMANCE -ErrorAction SilentlyContinue
Remove-Item Env:HF_XET_CHUNK_CACHE_SIZE_BYTES -ErrorAction SilentlyContinue
Remove-Item Env:HF_XET_SHARD_CACHE_SIZE_LIMIT -ErrorAction SilentlyContinue

$args = @("--host", $HostName, "--port", "$Port")
if ($LibraryDir) {
    $args += @("--library-dir", $LibraryDir)
}

Invoke-Checked $webCommand @args
