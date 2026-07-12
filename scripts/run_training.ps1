Write-Host "================================================="
Write-Host "Real-Time Industrial Defect Detection System"
Write-Host "Starting YOLOv8 Model Training Pipeline"
Write-Host "================================================="

# Activate virtual environment
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    . .\venv\Scripts\Activate.ps1
}

# Run training orchestrator
python scripts/execute_model_training.py --epochs 100 --batch_size 16 --img_size 640 --device auto --model yolov8n

Write-Host "Training run command completed."
