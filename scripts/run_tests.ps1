# run_tests.ps1 - hoboQRAG Test Runner Script for Windows PowerShell
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Resolve-Path "$ScriptDir/.."
$BackendDir = "$ProjectRoot/backend"

Set-Location $BackendDir

if (-not (Test-Path ".venv")) {
    Write-Error "Virtual environment (.venv) not found in $BackendDir."
    Exit 1
}

# Export test mode indicator env var
$env:TESTING = "true"

$Stage = "unit"
if ($args.Count -gt 0) {
    $Stage = $args[0]
}

Write-Host "Test Stage: $Stage" -ForegroundColor Cyan

switch ($Stage) {
    "unit" {
        Write-Host "Running Unit Tests..." -ForegroundColor Green
        & .venv/Scripts/pytest.exe tests/unit/test_document_processing.py tests/unit/test_embedding.py tests/unit/test_qdrant.py tests/unit/test_reranker.py tests/unit/test_prompt_construction.py tests/unit/test_rag_pipeline.py
    }
    "api" {
        Write-Host "Running API Endpoint Tests..." -ForegroundColor Green
        & .venv/Scripts/pytest.exe tests/unit/test_api_endpoints.py
    }
    "integration" {
        Write-Host "Running Integration Tests..." -ForegroundColor Green
        & .venv/Scripts/pytest.exe tests/integration
    }
    "e2e" {
        Write-Host "Running End-to-End Tests..." -ForegroundColor Green
        & .venv/Scripts/pytest.exe tests/e2e
    }
    "all" {
        Write-Host "Running All Tests..." -ForegroundColor Green
        & .venv/Scripts/pytest.exe tests/
    }
    Default {
        Write-Host "Usage: .\run_tests.ps1 {unit|api|integration|e2e|all}" -ForegroundColor Yellow
        Exit 1
    }
}

if ($LASTEXITCODE -ne 0) {
    Write-Error "Tests failed!"
    Exit $LASTEXITCODE
} else {
    Write-Host "Tests completed successfully!" -ForegroundColor Green
}
