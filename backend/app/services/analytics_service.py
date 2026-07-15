"""
Backend Analytics Service
=========================
Tracks API request counts, latencies, and prediction metrics.
Author: prajwaledu802-coder
Date: 2026-07-15
"""

import time
import threading
from typing import Dict, Any, List

class BackendAnalyticsService:
    """
    In-memory analytics recorder tracking endpoint request counts,
    inference latencies, and prediction metrics.
    """
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.request_counts: Dict[str, int] = {}
        self.latency_sums: Dict[str, float] = {}
        self.prediction_counts: Dict[str, int] = {}
        self.start_time = time.time()

    def record_request(self, endpoint: str, latency: float) -> None:
        """
        Record an API request event with its response time.
        """
        with self._lock:
            self.request_counts[endpoint] = self.request_counts.get(endpoint, 0) + 1
            self.latency_sums[endpoint] = self.latency_sums.get(endpoint, 0.0) + latency

    def record_detections(self, detections: List[str]) -> None:
        """
        Record classes detected by inference model.
        """
        with self._lock:
            for cls in detections:
                self.prediction_counts[cls] = self.prediction_counts.get(cls, 0) + 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Compile aggregated statistics report.
        """
        with self._lock:
            uptime = time.time() - self.start_time
            endpoints_info = {}
            total_requests = 0
            
            for endpoint, count in self.request_counts.items():
                total_requests += count
                avg_lat = self.latency_sums[endpoint] / count if count > 0 else 0.0
                endpoints_info[endpoint] = {
                    "request_count": count,
                    "avg_latency_ms": round(avg_lat * 1000, 2)
                }
                
            return {
                "uptime_seconds": round(uptime, 2),
                "total_requests_processed": total_requests,
                "endpoints": endpoints_info,
                "detected_defects_distribution": self.prediction_counts
            }

# Global singleton
analytics_service = BackendAnalyticsService()
