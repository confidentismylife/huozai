$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\save_deepseek_key_secure.ps1"
if ($LASTEXITCODE -ne 0) {
    throw "Saving the encrypted DeepSeek key failed with exit code $LASTEXITCODE"
}
& "$PSScriptRoot\run_deepseek_format_gate_saved.ps1"
if ($LASTEXITCODE -ne 0) {
    throw "DeepSeek format gate failed with exit code $LASTEXITCODE"
}
