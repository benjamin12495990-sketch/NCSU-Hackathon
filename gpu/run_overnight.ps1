# Unattended training launcher for Windows PowerShell. From the repository root:
#
#   .\gpu\run_overnight.ps1                    # the real run (standard profile), then the report
#   .\gpu\run_overnight.ps1 -Rehearsal         # a few minutes: checks that everything works first
#   .\gpu\run_overnight.ps1 -Cycle2 -Rehearsal # a few minutes: the second cycle in miniature (needs -Rehearsal to have been run first)
#   .\gpu\run_overnight.ps1 -Cycle2            # the second cycle: held-out chemical systems, structure-feature trees, generations of
#                                              # crystal networks; ADDS to the newest real run folder (nothing there is changed); default budget 12 h
#   .\gpu\run_overnight.ps1 -Cycle3 -Rehearsal # a few minutes: the third cycle in miniature (extends the newest rehearsal folder)
#   .\gpu\run_overnight.ps1 -Cycle3            # the third cycle: held-out elements, fine-tuning from formation energy, wider search;
#                                              # ADDS to the newest real run folder; default budget 12 h
#   .\gpu\run_overnight.ps1 -Nightly           # git pull --ff-only, then retrain whatever changed (gpu\schedule_nightly.ps1 runs this every night)
#   .\gpu\run_overnight.ps1 -Resume            # continue the last pipeline (or cycle) after a reboot, power cut or stop
#   .\gpu\run_overnight.ps1 --budget-hours 6   # any other options are passed to python -m gpu.run_pipeline
#
# It runs the pipeline, and if the pipeline program itself crashes (exit code 3) it resumes the same folder, up to 3 times.
# Jobs that fail are retried by the pipeline itself. Nothing is deleted.
[CmdletBinding()]
param(
    [switch]$Resume,
    [switch]$Rehearsal,
    [switch]$Cycle2,
    [switch]$Cycle3,
    [switch]$Nightly,
    [Parameter(ValueFromRemainingArguments = $true)]$Rest
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$py = Join-Path $repo '.venv-gpu\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
$rootDir = Join-Path $repo 'gpu\runs\pipeline'
$pointer = Join-Path $rootDir 'LAST_PIPELINE.txt'
$extra = @(); if ($Rest) { $extra = @($Rest) }

function Get-LastPipeline {
    if (Test-Path $pointer) { return (Get-Content $pointer -Raw).Trim() }
    return $null
}

$started = Get-Date
if ($Resume) {
    $last = Get-LastPipeline
    if (-not $last) { Write-Host 'No earlier pipeline found (gpu\runs\pipeline\LAST_PIPELINE.txt is missing).'; exit 4 }
    Write-Host "Resuming $last"
    & $py -m gpu.run_pipeline --resume $last @extra
}
elseif ($Nightly) {
    if (Get-Command git -ErrorAction SilentlyContinue) {
        git pull --ff-only
        if ($LASTEXITCODE -ne 0) { Write-Host 'git pull did not fast-forward (local changes?); training the code and data as they are.' }
    }
    $prof = 'nightly'; if ($Rehearsal) { $prof = 'nightly-smoke' }
    & $py -m gpu.run_pipeline --profile $prof --device cuda @extra
}
elseif ($Cycle3 -and $Rehearsal) {
    & $py -m gpu.run_pipeline --profile cycle3-smoke --device cuda --extend latest-smoke @extra
}
elseif ($Cycle3) {
    & $py -m gpu.run_pipeline --profile cycle3 --device cuda --extend latest @extra
}
elseif ($Cycle2 -and $Rehearsal) {
    & $py -m gpu.run_pipeline --profile cycle2-smoke --device cuda --extend latest-smoke @extra
}
elseif ($Cycle2) {
    & $py -m gpu.run_pipeline --profile cycle2 --device cuda --extend latest @extra
}
elseif ($Rehearsal) {
    & $py -m gpu.run_pipeline --profile smoke --device cuda --tag rehearsal @extra
}
else {
    & $py -m gpu.run_pipeline --profile standard --device cuda --tag overnight @extra
}
$code = $LASTEXITCODE

$tries = 0
while ($code -eq 3 -and $tries -lt 3) {
    $last = Get-LastPipeline
    # only resume a folder this launcher started or resumed, never an older one
    if (-not $last -or -not (Test-Path $pointer) -or (Get-Item $pointer).LastWriteTime -lt $started) { break }
    $tries++
    Write-Host "The pipeline program crashed; resuming ($tries of 3) ..."
    Start-Sleep -Seconds 20
    & $py -m gpu.run_pipeline --resume $last @extra
    $code = $LASTEXITCODE
}

switch ($code) {
    0 { Write-Host 'All jobs finished. Read REPORT.md in the pipeline folder; send share.zip to whoever needs the results.' }
    1 { Write-Host 'Finished, but some jobs failed. REPORT.md lists them; rerun with -Resume --retry-failed to try them again.' }
    2 { Write-Host 'Stopped early (STOP file, Ctrl+C, or the time budget). Run  .\gpu\run_overnight.ps1 -Resume  to continue.' }
    3 { Write-Host 'The pipeline program kept crashing. Send the last 40 lines of the console and gpu\runs\pipeline\<folder>\pipeline.log.' }
    4 { Write-Host 'Not started: a check failed. The message above says which.' }
    default { Write-Host "Ended with exit code $code." }
}
exit $code
