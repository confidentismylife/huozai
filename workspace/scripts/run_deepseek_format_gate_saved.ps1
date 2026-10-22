$ErrorActionPreference = 'Stop'
$credentialPath = Join-Path (Join-Path $env:LOCALAPPDATA 'fire_evac') 'deepseek_key.dpapi'
$entropy = [Text.Encoding]::UTF8.GetBytes('fire-evac-deepseek-v1')
if (-not (Test-Path -LiteralPath $credentialPath)) {
    throw "Encrypted DeepSeek key not found. Run save_deepseek_key_secure.ps1 once."
}

$protectedBytes = [Convert]::FromBase64String([IO.File]::ReadAllText($credentialPath, [Text.Encoding]::ASCII))
$plainBytes = [Security.Cryptography.ProtectedData]::Unprotect(
    $protectedBytes,
    $entropy,
    [Security.Cryptography.DataProtectionScope]::CurrentUser
)
try {
    $env:DEEPSEEK_API_KEY = [Text.Encoding]::UTF8.GetString($plainBytes)
    & 'D:\miniconda3\python.exe' "$PSScriptRoot\run_deepseek_format_gate.py" run
    if ($LASTEXITCODE -ne 0) {
        throw "DeepSeek format gate failed with exit code $LASTEXITCODE"
    }
}
finally {
    Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
    [Array]::Clear($plainBytes, 0, $plainBytes.Length)
}
