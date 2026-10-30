"""Fixed GPU admission for the correction learner."""
from __future__ import annotations
import subprocess,time
from typing import Callable

def check_gpu_idle(result: dict, save: Callable[[], None]) -> str:
    sample = {"status": "querying", "gpu_index": 0, "unix_seconds": time.time()}
    result["gpu_check"] = sample
    save()
    try:
        raw = subprocess.check_output(["nvidia-smi", "--id=0",
            "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=5).strip()
        sample["raw"] = raw
        memory, utilization = [int(value.strip()) for value in raw.split(",")]
        if memory < 0 or not 0 <= utilization <= 100:
            raise ValueError("invalid GPU sample")
        busy = memory >= 2048 or utilization > 5
        sample.update(status="busy" if busy else "idle", memory_mib=memory,
                      utilization_percent=utilization)
        save()
    except Exception as exc:
        sample.update(status="query_failed", error=repr(exc))
        save()
        raise
    if busy:
        raise RuntimeError("GPU busy; stop without retry")
    return raw
