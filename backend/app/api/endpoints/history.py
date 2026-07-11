"""
History Analytics Router.
Author: prajwaledu802-coder
"""
from fastapi import APIRouter
from app.services.history_service import DB_PATH
import sqlite3

router = APIRouter(prefix="/history", tags=["Inference History"])

@router.get("/summary")
def get_inference_summary():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), AVG(detection_count), AVG(processing_time_ms) FROM history")
    row = cursor.fetchone()
    conn.close()
    return {
        "total_requests": row[0] or 0,
        "avg_defect_count": round(row[1] or 0.0, 2),
        "avg_latency_ms": round(row[2] or 0.0, 2)
    }
