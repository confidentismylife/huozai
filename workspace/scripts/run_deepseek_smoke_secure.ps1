param(
    [int]$Timesteps = 512,
    [int]$MaxApiRequests = 4
)

$ErrorActionPreference = 'Stop'
$secureKey = Read-Host 'DeepSeek API key (input is hidden and is not saved)' -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $env:DEEPSEEK_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    & 'D:\miniconda3\python.exe' "$PSScriptRoot\run_mappo_deepseek_smoke.py" `
        --source deepseek `
        --timesteps $Timesteps `
        --max-api-requests $MaxApiRequests
    if ($LASTEXITCODE -ne 0) {
        throw "DeepSeek smoke test failed with exit code $LASTEXITCODE"
    }
}
finally {
    Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
}
