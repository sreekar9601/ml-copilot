# Check ingestion progress
Write-Host "`n===========================================================" -ForegroundColor Cyan
Write-Host "INGESTION PROGRESS MONITOR" -ForegroundColor Cyan
Write-Host "===========================================================`n" -ForegroundColor Cyan

# Check if ingestion is running
$process = Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like "*complete_end_to_end_ingestion*"}

if ($process) {
    Write-Host "✓ Ingestion is RUNNING" -ForegroundColor Green
    Write-Host "  Process ID: $($process.Id)" -ForegroundColor Gray
} else {
    Write-Host "✗ Ingestion is NOT running (may have completed or failed)" -ForegroundColor Yellow
}

# Check latest log entries
Write-Host "`n--- Latest Log Entries (last 20 lines) ---" -ForegroundColor Cyan
if (Test-Path "complete_ingestion.log") {
    Get-Content "complete_ingestion.log" -Tail 20
} else {
    Write-Host "No log file found" -ForegroundColor Red
}

# Check if report exists
Write-Host "`n--- Reports Generated ---" -ForegroundColor Cyan
Get-ChildItem "complete_ingestion_report_*.json" -ErrorAction SilentlyContinue | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 3 |
    ForEach-Object {
        Write-Host "  $($_.Name) - $($_.LastWriteTime)" -ForegroundColor Gray
    }

Write-Host "`n===========================================================`n" -ForegroundColor Cyan
Write-Host "To check again: .\check_ingestion_progress.ps1" -ForegroundColor Gray
Write-Host "To view full logs: Get-Content complete_ingestion.log -Tail 100" -ForegroundColor Gray
Write-Host "===========================================================`n" -ForegroundColor Cyan


