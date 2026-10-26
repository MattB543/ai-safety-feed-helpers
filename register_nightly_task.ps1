# Registers (or updates) the Windows scheduled task that runs nightly_refresh.py every night at midnight.
#
#   powershell -ExecutionPolicy Bypass -File register_nightly_task.ps1
#
# - Wakes the PC from sleep to run (needs "Allow wake timers" enabled in the power plan; default is on AC only).
# - If a run is missed (PC off, or asleep on battery) it starts as soon as the PC is available again.
# - Tries S4U ("run whether user is logged on or not", no stored password); falls back to running
#   only while you're logged on (a locked screen still counts) if S4U isn't allowed for this account.
# Remove with: Unregister-ScheduledTask -TaskName "AI Safety Feed nightly refresh" -Confirm:$false

$ErrorActionPreference = "Stop"
$TaskName = "AI Safety Feed nightly refresh"
$Here = $PSScriptRoot
$Pythonw = "C:\Users\matth\.pyenv\pyenv-win\versions\3.11.5\pythonw.exe"

$action = New-ScheduledTaskAction -Execute $Pythonw -Argument "`"$Here\nightly_refresh.py`"" -WorkingDirectory $Here
$trigger = New-ScheduledTaskTrigger -Daily -At "00:00"
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 8) -MultipleInstances IgnoreNew
$user = "$env:USERDOMAIN\$env:USERNAME"
$description = "Scrape + enrich the AI Safety Feed, then cleaned titles and images for new rows. Logs: $Here\logs"

try {
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
        -Principal $principal -Description $description -Force | Out-Null
    Write-Output "Registered '$TaskName' (runs whether or not you are logged on)."
} catch {
    Write-Output "S4U registration not allowed ($($_.Exception.Message.Trim())); falling back to logged-on only."
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
        -Principal $principal -Description $description -Force | Out-Null
    Write-Output "Registered '$TaskName' (runs while you are logged on, including when the screen is locked)."
}
