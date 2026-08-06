# PowerShell Startup Automation Script for hoboQRAG
# Ensure we run from the project root
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ProjectRoot

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   Starting hoboQRAG Local Pipeline Server Stack   " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Ensure Local Folders Exist
$RequiredFolders = @("LLMModels", "EmbeddingModels", "knowledge_base", "backend/temp_uploads")
foreach ($folder in $RequiredFolders) {
    if (-not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder -Force | Out-Null
        Write-Host "Created folder: $folder" -ForegroundColor DarkGray
    }
}

# 2. Check for Offline LLM Weights
$LLMDir = "LLMModels/Phi-4-mini-instruct"
if (-not (Test-Path $LLMDir) -or -not (Get-ChildItem $LLMDir -Filter *.safetensors -Recurse)) {
    Write-Host "WARNING: LLM model weights not found in '$LLMDir'." -ForegroundColor Yellow
    Write-Host "Please download the Phi-4-mini-instruct repository from Hugging Face" -ForegroundColor Yellow
    Write-Host "and place the weights (including tokenizer and safetensors files) inside it." -ForegroundColor Yellow
    Write-Host "Press Enter to proceed anyway, or CTRL+C to abort..." -ForegroundColor Yellow
    Read-Host
}


# 3. Check if Docker is Running
Write-Host "Checking if Docker Daemon is running..." -ForegroundColor Gray
& docker ps > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker Desktop is not running. Please start Docker Desktop and run this script again." -ForegroundColor Red
    Exit 1
}

# 4. Spin up Docker Stack (Qdrant, vLLM, Prometheus, Grafana)
Write-Host "Booting Docker Compose services (Qdrant, vLLM, Prometheus, Grafana)..." -ForegroundColor Gray
docker compose up -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to spin up Docker Compose services." -ForegroundColor Red
    Exit 1
}
Write-Host "Docker Services are starting in the background." -ForegroundColor Green

# 5. Initialize Python Virtual Environment (using uv)
Write-Host "Configuring Python virtual environment in backend/..." -ForegroundColor Gray
Set-Location "$ProjectRoot/backend"

if (-not (Test-Path ".venv")) {
    Write-Host "Virtual environment not found. Creating .venv via uv..." -ForegroundColor Gray
    # Try using uv (fastest)
    & uv venv .venv --python 3.12 --seed
    if ($LASTEXITCODE -ne 0) {
        Write-Host "uv not found or failed, falling back to standard venv..." -ForegroundColor Yellow
        python -m venv .venv
    }
}

# 6. Activate Environment and Install dependencies
Write-Host "Activating virtual environment and installing python dependencies..." -ForegroundColor Gray
# On Windows PowerShell, we need to bypass execution policy temporarily to activate
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
& .venv/Scripts/Activate.ps1

# Sync dependencies using uv if available, else pip
& uv pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "uv pip install failed or not available, falling back to standard pip..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

Write-Host "Python dependencies synchronized." -ForegroundColor Green

# 7. Launch FastAPI Backend natively on Port 5000
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   FastAPI backend is launching on port 5000...   " -ForegroundColor Green
Write-Host "   - API Documentation: http://localhost:5000/docs" -ForegroundColor Gray
Write-Host "   - Qdrant Dashboard:  http://localhost:6333/dashboard" -ForegroundColor Gray
Write-Host "   - Grafana Metrics:   http://localhost:3000" -ForegroundColor Gray
Write-Host "==================================================" -ForegroundColor Cyan

# Run FastAPI natively
& uvicorn main:app --host 0.0.0.0 --port 5000
