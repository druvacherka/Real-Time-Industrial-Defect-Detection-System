"""
History logger sqlite service.
Author: prajwaledu802-coder
"""
import sqlite3
from pathlib import Path
import json

DB_PATH = Path("backend/temp/predictions_history.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

class HistoryService:
    def __init__(self):
        self.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self.create_table()

    def create_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id TEXT PRIMARY KEY,
                    filename TEXT,
                    detection_count INTEGER,
                    processing_time_ms REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def log_prediction(self, request_id: str, filename: str, count: int, duration_ms: float):
        with self.conn:
            self.conn.execute(
                "INSERT INTO history (id, filename, detection_count, processing_time_ms) VALUES (?, ?, ?, ?)",
                (request_id, filename, count, duration_ms)
            )
