$ErrorActionPreference = 'Stop'
$secureKey = Read-Host 'DeepSeek API key (input is hidden and is not saved)' -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $env:DEEPSEEK_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    & 'D:\miniconda3\python.exe' "$PSScriptRoot\run_deepseek_format_gate.py" run
    if ($LASTEXITCODE -ne 0) {
        throw "DeepSeek format gate failed with exit code $LASTEXITCODE"
    }
}
finally {
    Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
}
