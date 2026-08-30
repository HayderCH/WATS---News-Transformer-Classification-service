import time
import threading
from collections import deque
from typing import Dict, Any, Deque
from fastapi import Request


class _MetricsState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {}
        # Store rolling stats per route: count, sum, sumsq,
        # and approximate p50/p95 later
        self._latency: Dict[str, Dict[str, float]] = {}
        self._recent: Dict[str, Deque[float]] = {}

    def incr(self, route: str, status: int) -> None:
        key = f"{route}:{status}"
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + 1

    def observe(self, route: str, ms: float) -> None:
        with self._lock:
            s = self._latency.setdefault(
                route, {"count": 0.0, "sum": 0.0, "sumsq": 0.0}
            )
            s["count"] += 1.0
            s["sum"] += ms
            s["sumsq"] += ms * ms
            recent = self._recent.setdefault(route, deque(maxlen=100))
            recent.append(ms)

    def get_counters(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def get_latencies(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        with self._lock:
            for route, s in self._latency.items():
                cnt = max(1.0, s.get("count", 0.0))
                avg = s.get("sum", 0.0) / cnt
                # Std dev (population)
                mean = avg
                var = max(0.0, s.get("sumsq", 0.0) / cnt - mean * mean)
                std = var**0.5
                # Approximate p50/p95 using normal assumption (quick and light)
                p50 = mean
                p95 = mean + 1.645 * std
                route_stats: Dict[str, Any] = {
                    "count": int(cnt),
                    "avg_ms": avg,
                    "p50_ms": p50,
                    "p95_ms": p95,
                }
                recent_values = list(self._recent.get(route, []))
                if recent_values:
                    recent_sorted = sorted(recent_values)
                    n_recent = len(recent_sorted)
                    p50_idx = int(0.5 * (n_recent - 1))
                    p95_idx = int(0.95 * (n_recent - 1))
                    route_stats["recent"] = {
                        "window": n_recent,
                        "avg_ms": sum(recent_sorted) / n_recent,
                        "p50_ms": recent_sorted[p50_idx],
                        "p95_ms": recent_sorted[p95_idx],
                    }
                out[route] = route_stats
        return out

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._latency.clear()
            self._recent.clear()


METRICS_STATE = _MetricsState()


async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        path = request.url.path
        METRICS_STATE.observe(path, duration_ms)
        METRICS_STATE.incr(path, status)
