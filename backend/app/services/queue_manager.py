"""
Inference Queue Manager
========================
Real-Time Industrial Defect Detection System

Implements an asynchronous FIFO queue for model inference tasks. Secures high 
scalability and prevents model contention under heavy concurrent requests.

Author: prajwaledu802-coder
Date: 2026-07-14
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
import numpy as np

from app.services.model_service import get_model_service

logger = logging.getLogger("defect_detection.queue_manager")


class InferenceJob:
    """Represents a single prediction job in the inference queue."""
    def __init__(self, processed_image: np.ndarray, conf_threshold: Optional[float] = None):
        self.processed_image = processed_image
        self.conf_threshold = conf_threshold
        self.future = asyncio.get_running_loop().create_future()
        self.created_at = time.time()


class InferenceQueueManager:
    """
    Asynchronous Queue Manager that serializes YOLOv8 inference requests.
    Prevents concurrency bottleneck/PyTorch resource contention.
    """
    def __init__(self, max_queue_size: int = 100):
        self.queue: asyncio.Queue[InferenceJob] = asyncio.Queue(maxsize=max_queue_size)
        self.worker_task: Optional[asyncio.Task] = None
        self.total_jobs_processed = 0
        self.total_wait_time = 0.0

    def start(self):
        """Start the background worker task to process the queue."""
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker())
            logger.info("Inference queue worker started successfully.")

    async def stop(self):
        """Gracefully stop the background worker."""
        if self.worker_task and not self.worker_task.done():
            logger.info("Stopping inference queue worker...")
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            logger.info("Inference queue worker stopped.")

    async def submit_job(self, processed_image: np.ndarray, conf_threshold: Optional[float] = None, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Submit a new image inference job to the queue and wait for the results.
        Includes request timeout handling.
        """
        # Ensure worker is running
        self.start()

        job = InferenceJob(processed_image, conf_threshold)
        
        try:
            # Put job in queue (raises QueueFull if full, wait up to timeout)
            await asyncio.wait_for(self.queue.put(job), timeout=2.0)
        except asyncio.TimeoutError:
            logger.error("Inference queue is full. Request rejected.")
            raise RuntimeError("Server is busy. The request queue is full. Please try again later.")

        # Log queue statistics
        qsize = self.queue.qsize()
        logger.info(f"[Queue Stats] New job added. Queue size: {qsize} | Total processed: {self.total_jobs_processed}")

        try:
            # Wait for job completion with timeout
            result = await asyncio.wait_for(job.future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Inference job timed out after {timeout} seconds in queue/processing.")
            # Cancel future
            if not job.future.done():
                job.future.cancel()
            raise asyncio.TimeoutError("The prediction request timed out in the processing queue.")

    async def _worker(self):
        """Worker loop that fetches jobs from the queue and executes them in a thread pool."""
        model_svc = get_model_service()
        
        while True:
            try:
                job = await self.queue.get()
                
                # If job was cancelled (e.g. timeout), skip it
                if job.future.cancelled():
                    self.queue.task_done()
                    continue

                wait_time = time.time() - job.created_at
                self.total_wait_time += wait_time
                self.total_jobs_processed += 1
                
                # Log stats every 10 jobs
                if self.total_jobs_processed % 10 == 0:
                    avg_wait = self.total_wait_time / self.total_jobs_processed
                    logger.info(f"[Queue Stats] Processed {self.total_jobs_processed} jobs | Avg queue wait time: {avg_wait*1000:.1f} ms")

                try:
                    # Execute CPU/GPU heavy model inference in a separate thread to keep loop free
                    prediction_result = await asyncio.to_thread(
                        model_svc.predict_image, 
                        job.processed_image, 
                        job.conf_threshold
                    )
                    # Set result
                    if not job.future.done():
                        job.future.set_result(prediction_result)
                except Exception as exc:
                    logger.error(f"Inference worker error: {exc}")
                    if not job.future.done():
                        job.future.set_exception(exc)
                finally:
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue worker loop: {e}")
                await asyncio.sleep(0.5)


# Global singleton instance
_queue_manager_instance: Optional[InferenceQueueManager] = None


def get_queue_manager() -> InferenceQueueManager:
    """Retrieve or initialize the global queue manager singleton."""
    global _queue_manager_instance
    if _queue_manager_instance is None:
        _queue_manager_instance = InferenceQueueManager()
    return _queue_manager_instance
