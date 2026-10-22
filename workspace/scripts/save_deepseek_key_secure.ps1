$ErrorActionPreference = 'Stop'
$credentialDirectory = Join-Path $env:LOCALAPPDATA 'fire_evac'
$credentialPath = Join-Path $credentialDirectory 'deepseek_key.dpapi'
$entropy = [Text.Encoding]::UTF8.GetBytes('fire-evac-deepseek-v1')

$secureKey = Read-Host 'Enter DeepSeek API key once (hidden; saved with Windows DPAPI)' -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
$plainBytes = $null
try {
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    if ([string]::IsNullOrWhiteSpace($plainKey)) {
        throw 'The API key cannot be empty.'
    }
    $plainBytes = [Text.Encoding]::UTF8.GetBytes($plainKey)
    $protectedBytes = [Security.Cryptography.ProtectedData]::Protect(
        $plainBytes,
        $entropy,
        [Security.Cryptography.DataProtectionScope]::CurrentUser
    )
    New-Item -ItemType Directory -Path $credentialDirectory -Force | Out-Null
    [IO.File]::WriteAllText($credentialPath, [Convert]::ToBase64String($protectedBytes), [Text.Encoding]::ASCII)
    Write-Host "DeepSeek key encrypted for the current Windows user at $credentialPath"
}
finally {
    if ($null -ne $plainBytes) {
        [Array]::Clear($plainBytes, 0, $plainBytes.Length)
    }
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    Remove-Variable plainKey -ErrorAction SilentlyContinue
}
