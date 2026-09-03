"""
Sliding Window Rate Limiter for ShopAssist AI.
Prevents API abuse, protects downstream LLM quotas, and enforces traffic policies.
"""

import time
import threading
from typing import Dict, List, Tuple
from collections import defaultdict

from src.config import settings

class SlidingWindowRateLimiter:
    """
    Thread-safe Sliding Window Counter Rate Limiter.
    """
    def __init__(self, requests_per_minute: int = 60, window_seconds: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Telemetry
        self.blocked_requests: int = 0
        self.allowed_requests: int = 0

    def is_allowed(self, client_id: str) -> Tuple[bool, int, int]:
        """
        Determines whether a client request is permitted under rate limit policy.
        
        Returns:
            Tuple of (allowed: bool, remaining_requests: int, retry_after_seconds: int)
        """
        now = time.time()
        cutoff = now - self.window_seconds
        
        with self._lock:
            timestamps = self._history[client_id]
            # Prune timestamps outside current sliding window
            self._history[client_id] = [t for t in timestamps if t > cutoff]
            valid_history = self._history[client_id]
            
            if len(valid_history) >= self.requests_per_minute:
                self.blocked_requests += 1
                oldest_timestamp = valid_history[0]
                retry_after = max(1, int(self.window_seconds - (now - oldest_timestamp)))
                return False, 0, retry_after
                
            valid_history.append(now)
            self.allowed_requests += 1
            remaining = self.requests_per_minute - len(valid_history)
            return True, remaining, 0

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "active_clients": len(self._history),
                "allowed_requests": self.allowed_requests,
                "blocked_requests": self.blocked_requests,
                "rate_limit_rpm": self.requests_per_minute
            }

rate_limiter = SlidingWindowRateLimiter(
    requests_per_minute=settings.rate_limit_requests_per_minute
)
