"""
Reliability, Circuit Breakers, and Exponential Retries for ShopAssist AI (Week 16).
Guarantees resilience against provider failures and upstream timeouts.
"""

import time
import random
from enum import Enum
from typing import Callable, Any, Dict
from threading import Lock

class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout_seconds: float = 30.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self._lock = Lock()

    def can_execute(self) -> bool:
        with self._lock:
            now = time.time()
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                if now - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                return True
            return True

    def record_success(self):
        with self._lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED

    def record_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

CIRCUIT_BREAKERS: Dict[str, CircuitBreaker] = {
    "gemini": CircuitBreaker("gemini"),
    "openai": CircuitBreaker("openai"),
    "claude": CircuitBreaker("claude"),
    "vllm": CircuitBreaker("vllm"),
    "mock": CircuitBreaker("mock")
}

def retry_with_backoff(func: Callable, max_attempts: int = 3, base_delay: float = 0.5):
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts:
                raise e
            sleep_time = (base_delay * (2 ** (attempt - 1))) + random.uniform(0.05, 0.2)
            time.sleep(sleep_time)
