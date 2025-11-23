# PowerShell Setup Script for PySpark Code Analyzer
# Run this script to set up the project on Windows

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "  PySpark Code Analyzer - Windows Setup Script" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Function to print status messages
function Write-Status {
    param($Message, $Type = "Info")
    switch ($Type) {
        "Success" { Write-Host "✓ $Message" -ForegroundColor Green }
        "Error" { Write-Host "✗ $Message" -ForegroundColor Red }
        "Warning" { Write-Host "⚠ $Message" -ForegroundColor Yellow }
        "Info" { Write-Host "→ $Message" -ForegroundColor Cyan }
    }
}

# Check Python installation
Write-Status "Checking Python installation..." "Info"
try {
    $pythonVersion = python --version 2>&1
    Write-Status "Python is installed: $pythonVersion" "Success"
} catch {
    Write-Status "Python is not installed or not in PATH" "Error"
    Write-Host "Please install Python 3.8+ from https://www.python.org/" -ForegroundColor Yellow
    exit 1
}

# Check if virtual environment exists
Write-Status "Checking virtual environment..." "Info"
if (Test-Path "venv") {
    Write-Status "Virtual environment already exists" "Success"
} else {
    Write-Status "Creating virtual environment..." "Info"
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Status "Virtual environment created" "Success"
    } else {
        Write-Status "Failed to create virtual environment" "Error"
        exit 1
    }
}

# Activate virtual environment
Write-Status "Activating virtual environment..." "Info"
& .\venv\Scripts\Activate.ps1
Write-Status "Virtual environment activated" "Success"

# Upgrade pip
Write-Status "Upgrading pip..." "Info"
python -m pip install --upgrade pip --quiet
Write-Status "Pip upgraded" "Success"

# Install dependencies
Write-Status "Installing dependencies (this may take a few minutes)..." "Info"
pip install -r requirements.txt --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Status "Dependencies installed successfully" "Success"
} else {
    Write-Status "Some dependencies may have failed to install" "Warning"
}

# Create .env file
Write-Status "Setting up environment file..." "Info"
if (Test-Path ".env") {
    Write-Status ".env file already exists" "Success"
} else {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Status "Created .env file from template" "Success"
    } else {
        Write-Status ".env.example not found" "Warning"
    }
}

# Create necessary directories
Write-Status "Creating directories..." "Info"
$directories = @("logs", "data/input", "data/output", "data/chromadb")
foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Write-Status "Created $dir/" "Success"
}

# Check Ollama
Write-Status "Checking Ollama installation..." "Info"
try {
    $ollamaCheck = ollama list 2>&1
    Write-Status "Ollama is installed" "Success"
    
    if ($ollamaCheck -match "codellama:7b") {
        Write-Status "CodeLlama:7b model is available" "Success"
    } else {
        Write-Status "CodeLlama:7b model not found" "Warning"
        Write-Host ""
        Write-Host "To install CodeLlama:7b, run:" -ForegroundColor Yellow
        Write-Host "  ollama pull codellama:7b" -ForegroundColor White
        Write-Host ""
    }
} catch {
    Write-Status "Ollama is not installed or not running" "Warning"
    Write-Host ""
    Write-Host "Please install Ollama from: https://ollama.ai" -ForegroundColor Yellow
    Write-Host "Then run: ollama pull codellama:7b" -ForegroundColor Yellow
    Write-Host ""
}

# Print completion message
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Make sure Ollama is running with CodeLlama:7b model" -ForegroundColor White
Write-Host "  2. Run the quick start script:" -ForegroundColor White
Write-Host "     python quickstart.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Or process a file directly:" -ForegroundColor White
Write-Host "     python main.py --mode process --file examples/example_pyspark_etl.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "For more information, see README.md" -ForegroundColor Cyan
Write-Host ""
