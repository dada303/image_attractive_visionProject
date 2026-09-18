param([string]$PublicOrigin = "", [int]$Port = 8767)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:HOST = "127.0.0.1"
$env:PORT = "$Port"
if ($PublicOrigin) { $env:PUBLIC_ORIGIN = $PublicOrigin }
if (-not $env:DEFAULT_MODEL) { $env:DEFAULT_MODEL = "efficientnet_b0" }
& "$PSScriptRoot\.venv\Scripts\python.exe" -m backend.production
exit $LASTEXITCODE
