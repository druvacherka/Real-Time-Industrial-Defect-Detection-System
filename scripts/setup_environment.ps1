Write-Host "========================================"
Write-Host "Industrial Defect Detection Setup"
Write-Host "========================================"

python -m venv venv

.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt

Write-Host ""
Write-Host "Environment setup completed successfully!"