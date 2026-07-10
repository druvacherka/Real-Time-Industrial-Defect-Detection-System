"""
Model Management API router.
Author: prajwaledu802-coder
"""
from fastapi import APIRouter, HTTPException, Depends
from pathlib import Path
from typing import List

router = APIRouter(prefix="/models", tags=["Model Management"])

@router.get("/list", response_model=List[str])
def list_available_weights():
    weights_dir = Path("weights")
    if not weights_dir.exists():
        return []
    return [p.name for p in weights_dir.glob("*.pt")]

@router.post("/swap/{weight_name}")
def swap_model_weights(weight_name: str):
    weight_path = Path("weights") / weight_name
    if not weight_path.exists():
        raise HTTPException(status_code=404, detail="Weights file not found")
    # Swapping logic here...
    return {"status": "success", "message": f"Successfully loaded weights: {weight_name}"}
