"""
System Service for monitoring resource usage.
Author: prajwaledu802-coder
"""
import psutil
import os
import gc

class SystemService:
    @staticmethod
    def get_resource_stats():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            "memory_rss_mb": round(mem_info.rss / (1024 * 1024), 2),
            "cpu_percent": process.cpu_percent(interval=0.1),
            "thread_count": process.num_threads()
        }

    @staticmethod
    def force_garbage_collection():
        gc.collect()
        return {"status": "success", "message": "Garbage collection completed"}
