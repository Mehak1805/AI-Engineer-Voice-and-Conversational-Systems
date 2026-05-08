param(
    [string]$VenvName = ".venv",
    [string]$Model = "vosk-model-en-us-0.22-lgraph",
    [string]$OutDir = ".\model",
    [switch]$Force
)

$root = (Get-Location).Path
$venvPath = Join-Path $root $VenvName
$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$reqPath = Join-Path $root "requirements.txt"
$modelScript = Join-Path $root "scripts\setup_vosk_model.ps1"
$existingRootModel = Join-Path $root $Model

function Get-VoskModelDir {
    param([string]$Path)

    if (-not (Test-Path $Path)) { return $null }

    $item = Get-Item $Path
    if (-not $item.PSIsContainer) { return $null }

    $required = @("am", "conf", "graph")
    $hasAll = $true
    foreach ($sub in $required) {
        if (-not (Test-Path (Join-Path $item.FullName $sub))) { $hasAll = $false }
    }
    if ($hasAll) { return $item.FullName }

    $child = Get-ChildItem $item.FullName -Directory | Where-Object {
        (Test-Path (Join-Path $_.FullName "am")) -and
        (Test-Path (Join-Path $_.FullName "conf")) -and
        (Test-Path (Join-Path $_.FullName "graph"))
    } | Select-Object -First 1

    if ($child) { return $child.FullName }

    return $null
}

if (-not (Test-Path $pythonExe)) {
    python -m venv $venvPath
}

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r $reqPath

if (Test-Path $existingRootModel) {
    $resolvedExisting = Get-VoskModelDir -Path $existingRootModel
    if (-not $resolvedExisting) { $resolvedExisting = (Resolve-Path $existingRootModel).Path }
    $env:VOSK_MODEL_PATH = $resolvedExisting
    Write-Host "Using existing model folder: $env:VOSK_MODEL_PATH"
} else {
    & powershell -ExecutionPolicy Bypass -File $modelScript -Model $Model -OutDir $OutDir -Force:$Force.IsPresent
}

if (-not $env:VOSK_MODEL_PATH -and (Test-Path $OutDir)) {
    $found = Get-ChildItem $OutDir -Directory | Where-Object { $_.Name -like "$Model*" } | Select-Object -First 1
    if ($found) {
        $resolvedFound = Get-VoskModelDir -Path $found.FullName
        if (-not $resolvedFound) { $resolvedFound = (Resolve-Path $found.FullName).Path }
        $env:VOSK_MODEL_PATH = $resolvedFound
        Write-Host "Set VOSK_MODEL_PATH = $env:VOSK_MODEL_PATH"
    }
}

Write-Host "Run: & $pythonExe .\app.py --model $env:VOSK_MODEL_PATH"
