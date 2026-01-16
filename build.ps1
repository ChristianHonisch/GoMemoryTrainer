# Build executable for Go Memory Trainer
# Run: .\build.ps1

Write-Host "Go Memory Trainer - Build Script" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

# Check if PyInstaller is installed
try {
    python -m PyInstaller --version > $null 2>&1
}
catch {
    Write-Host "Error: PyInstaller not found." -ForegroundColor Red
    Write-Host "Install dependencies with: pip install -r requirements.txt`n" -ForegroundColor Yellow
    exit 1
}

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$trainerFile = Join-Path $projectDir "trainer.py"

if (-not (Test-Path $trainerFile)) {
    Write-Host "Error: $trainerFile not found" -ForegroundColor Red
    exit 1
}

Write-Host "Building executable..." -ForegroundColor Green
Write-Host "Source: $trainerFile`n"

$distPath = Join-Path $projectDir "dist"
$buildPath = Join-Path $projectDir "build"
$specPath = $projectDir

python -m PyInstaller `
    --onefile `
    --windowed `
    --name "GoMemoryTrainer" `
    --distpath $distPath `
    --workpath $buildPath `
    --specpath $specPath `
    $trainerFile

if ($LASTEXITCODE -eq 0) {
    $exePath = Join-Path $distPath "GoMemoryTrainer.exe"
    Write-Host "`nSuccess! Executable created at:" -ForegroundColor Green
    Write-Host "  $exePath" -ForegroundColor Cyan
    Write-Host "`nYou can now distribute this .exe file." -ForegroundColor Green
} else {
    Write-Host "`nError building executable" -ForegroundColor Red
    exit 1
}
