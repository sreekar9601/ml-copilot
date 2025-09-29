# PowerShell script to switch between development and production environments

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("dev", "prod")]
    [string]$Environment
)

Write-Host "Switching to $Environment environment..." -ForegroundColor Green

if ($Environment -eq "dev") {
    # Copy local environment file
    Copy-Item ".env.local" ".env" -Force
    Write-Host "✅ Switched to DEVELOPMENT mode" -ForegroundColor Yellow
    Write-Host "   Frontend will point to: http://localhost:8001" -ForegroundColor Cyan
} elseif ($Environment -eq "prod") {
    # Copy production environment file
    Copy-Item ".env.production" ".env" -Force
    Write-Host "✅ Switched to PRODUCTION mode" -ForegroundColor Yellow
    Write-Host "   Frontend will point to: https://ml-copilot-production.up.railway.app" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "To start the frontend, run:" -ForegroundColor White
Write-Host "  npm run dev" -ForegroundColor Gray
