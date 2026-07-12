Write-Host "================================================="
Write-Host "Real-Time Industrial Defect Detection System"
Write-Host "Running YOLOv8 Model Evaluation Pipeline"
Write-Host "================================================="

# Activate virtual environment
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    . .\venv\Scripts\Activate.ps1
}

# Run validation split evaluation
Write-Host "Running evaluation on validation split..."
python training/evaluate.py --split val --weights results/train_yolov8n/weights/best.pt --device cpu

# Run comparative performance split checks
Write-Host "Running splits comparative performance evaluation..."
python scripts/compare_splits_performance.py --weights results/train_yolov8n/weights/best.pt --device cpu

Write-Host "Model evaluation pipeline runs completed successfully."
