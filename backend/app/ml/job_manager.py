import threading
from typing import Optional, Dict, Any

class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_job(self, job_id: str, text: str) -> None:
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "text": text,
                "status": "PENDING",
                "tokens": None,
                "attribution_method": "SHAP_PartitionExplainer",
                "duration_ms": None,
                "error": None
            }

    def complete_job(self, job_id: str, tokens: list, duration_ms: float, attribution_method: str = "SHAP_PartitionExplainer") -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "COMPLETED"
                self._jobs[job_id]["tokens"] = tokens
                self._jobs[job_id]["attribution_method"] = attribution_method
                self._jobs[job_id]["duration_ms"] = duration_ms

    def fail_job(self, job_id: str, error: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "FAILED"
                self._jobs[job_id]["error"] = error

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._jobs.get(job_id)

job_manager = JobManager()
