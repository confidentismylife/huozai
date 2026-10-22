$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$sourcePath = Join-Path $projectRoot 'artifacts\llm_ppo_paper_v4_20261021\固定LLM建议与PPO疏散控制_期刊式改写稿_v4.docx'
$pdfFolder = Join-Path $projectRoot 'output\pdf'
New-Item -ItemType Directory -Path $pdfFolder -Force | Out-Null
$pdfPath = Join-Path $pdfFolder '固定LLM建议与PPO疏散控制_期刊式改写稿_v4.pdf'
$wordApp = $null
$wordDocument = $null
try {
    $wordApp = New-Object -ComObject Word.Application
    $wordApp.Visible = $false
    $wordApp.DisplayAlerts = 0
    $wordApp.AutomationSecurity = 3
    $wordDocument = $wordApp.Documents.Open($sourcePath, $false, $true, $false)
    $wordDocument.Repaginate()
    $wordDocument.ExportAsFixedFormat($pdfPath, 17)
    [pscustomobject]@{pdf=$pdfPath; pages=$wordDocument.ComputeStatistics(2)} | ConvertTo-Json -Compress
} finally {
    if ($null -ne $wordDocument) {
        $wordDocument.Close(0)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordDocument)
    }
    if ($null -ne $wordApp) {
        $wordApp.Quit(0)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordApp)
    }
}
