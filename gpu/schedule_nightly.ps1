# Registers a Windows scheduled task that runs the nightly training (gpu\run_overnight.ps1 -Nightly) every day.
#
#   .\gpu\schedule_nightly.ps1                 # every day at 01:17
#   .\gpu\schedule_nightly.ps1 -Time 23:43     # another time
#   .\gpu\schedule_nightly.ps1 -Remove         # remove the task
#
# The task: runs only while you are signed in, only on mains power (it will not start on battery and stops if the charger is pulled),
# may wake the laptop from sleep (Settings > System > Power: allow wake timers; a closed lid can still stop it), never runs two copies,
# and is stopped after 10 hours. The pipeline itself refuses to start while another pipeline (say, a long cycle) is still running.
# Logs: gpu\runs\pipeline\<date>_nightly\pipeline.log and REPORT.md; the console output goes to gpu\runs\pipeline\nightly_console.log.
[CmdletBinding()]
param([string]$Time = '01:17', [switch]$Remove)
$ErrorActionPreference = 'Stop'
$name = 'NCSU-Hackathon nightly training'
if ($Remove) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false
    Write-Host "Removed the scheduled task '$name'."
    exit 0
}
$repo = Split-Path -Parent $PSScriptRoot
$log = Join-Path $repo 'gpu\runs\pipeline\nightly_console.log'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
$cmd = "-NoProfile -ExecutionPolicy Bypass -Command `"& '$repo\gpu\run_overnight.ps1' -Nightly *>> '$log'`""
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $cmd -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Daily -At $Time
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 10)
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings -Force `
    -Description 'Retrains the NCSU-Hackathon models whose data or recipe changed (gpu\run_overnight.ps1 -Nightly).' | Out-Null
Write-Host "Scheduled '$name' every day at $Time. Check it in Task Scheduler; remove it with: .\gpu\schedule_nightly.ps1 -Remove"
