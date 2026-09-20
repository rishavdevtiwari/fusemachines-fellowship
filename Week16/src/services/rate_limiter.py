"""
Sliding Window Rate Limiter for ShopAssist AI (Week 16).
Provides in-memory sliding window rate limiting per client IP or identifier.
"""

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Tuple, Dict, Any

from src.config import settings

class SlidingWindowRateLimiter:
    def __init__(self, requests_per_minute: int = 60, window_seconds: int = 60):
        self.rpm = requests_per_minute
        self.window = window_seconds
        self._clients: Dict[str, deque] = defaultdict(deque)
        self._lock = Lock()
        self.total_blocked: int = 0
        self.total_allowed: int = 0

    def is_allowed(self, client_id: str) -> Tuple[bool, int, float]:
        now = time.time()
        cutoff = now - self.window

        with self._lock:
            timestamps = self._clients[client_id]
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) < self.rpm:
                timestamps.append(now)
                self.total_allowed += 1
                remaining = self.rpm - len(timestamps)
                return True, remaining, 0.0

            self.total_blocked += 1
            oldest = timestamps[0]
            retry_after = round(oldest + self.window - now, 2)
            return False, 0, max(0.1, retry_after)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "configured_rpm": self.rpm,
                "window_seconds": self.window,
                "active_clients": len(self._clients),
                "total_allowed": self.total_allowed,
                "total_blocked": self.total_blocked
            }

rate_limiter = SlidingWindowRateLimiter(requests_per_minute=settings.rate_limit_rpm)
