param(
    [string]$Model = "vosk-model-en-us-0.22-lgraph",
    [string]$OutDir = ".\model",
    [switch]$Force
)

$baseUrl = "https://alphacephei.com/vosk/models"
$zipName = "$Model.zip"
$zipPath = Join-Path (Get-Location) $zipName
$url = "$baseUrl/$zipName"

function Get-VoskModelDir {
    param([string]$Path)

    if (-not (Test-Path $Path)) { return $null }

    $candidate = Get-Item $Path
    if ($candidate.PSIsContainer) {
        $required = @("am", "conf", "graph")
        $hasAll = $true
        foreach ($sub in $required) {
            if (-not (Test-Path (Join-Path $candidate.FullName $sub))) { $hasAll = $false }
        }
        if ($hasAll) { return $candidate.FullName }

        $child = Get-ChildItem $candidate.FullName -Directory | Where-Object {
            (Test-Path (Join-Path $_.FullName "am")) -and
            (Test-Path (Join-Path $_.FullName "conf")) -and
            (Test-Path (Join-Path $_.FullName "graph"))
        } | Select-Object -First 1

        if ($child) { return $child.FullName }
    }

    return $null
}

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}

$dest = Join-Path $OutDir $Model
if ((Test-Path $dest) -and (-not $Force)) {
    $resolved = Get-VoskModelDir -Path $dest
    if (-not $resolved) { $resolved = (Resolve-Path $dest).Path }
    $env:VOSK_MODEL_PATH = $resolved
    Write-Host "Model already exists: $env:VOSK_MODEL_PATH"
    Write-Host "Persist later with: [Environment]::SetEnvironmentVariable('VOSK_MODEL_PATH', '$env:VOSK_MODEL_PATH', 'User')"
    return
}

Write-Host "Downloading $url"
Invoke-WebRequest -Uri $url -OutFile $zipPath
Write-Host "Extracting to $OutDir"
Expand-Archive -Path $zipPath -DestinationPath $OutDir -Force
Remove-Item $zipPath -Force

if (-not (Test-Path $dest)) {
    $candidate = Get-ChildItem $OutDir -Directory | Select-Object -First 1
    if ($candidate) { $dest = $candidate.FullName }
}

$resolved = Get-VoskModelDir -Path $dest
if (-not $resolved) { $resolved = (Resolve-Path $dest).Path }
$env:VOSK_MODEL_PATH = $resolved
Write-Host "VOSK_MODEL_PATH = $env:VOSK_MODEL_PATH"
Write-Host "Persist later with: [Environment]::SetEnvironmentVariable('VOSK_MODEL_PATH', '$env:VOSK_MODEL_PATH', 'User')"
