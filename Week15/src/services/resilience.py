"""
Production Resilience Engine for ShopAssist AI.
Implements exponential backoff with jitter, circuit breaker pattern,
and graceful degradation failover across model providers.
"""

import time
import random
import asyncio
from typing import Callable, Any, Optional, Dict, Tuple
from enum import Enum
import threading

from src.config import settings

class CircuitState(str, Enum):
    CLOSED = "CLOSED"        # Normal operations; traffic passes through
    OPEN = "OPEN"            # Fail-fast; remote calls blocked to allow recovery
    HALF_OPEN = "HALF_OPEN"  # Probe traffic to verify provider health

class CircuitBreaker:
    """
    Thread-safe Circuit Breaker preventing cascading infrastructure failure.
    """
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.last_state_change: float = time.time()
        self._lock = threading.Lock()

    def record_success(self):
        with self._lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED

    def record_failure(self):
        with self._lock:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.last_state_change = time.time()
                print(f"[CircuitBreaker:{self.name}] Tripped to OPEN state after {self.failure_count} consecutive failures.")

    def can_execute(self) -> bool:
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            now = time.time()
            if self.state == CircuitState.OPEN:
                if now - self.last_state_change > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.last_state_change = now
                    print(f"[CircuitBreaker:{self.name}] Transitioned to HALF_OPEN probe state.")
                    return True
                return False
            # In HALF_OPEN, allow single test call
            return True

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "consecutive_failures": self.failure_count,
                "failure_threshold": self.failure_threshold
            }

def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    jitter: bool = True
):
    """
    Decorator executing synchronous function with exponential backoff and jitter.
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    if attempt == max_attempts - 1:
                        raise e
                    
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    if jitter:
                        delay += random.uniform(0.05, 0.2)
                    print(f"[Retry] Attempt {attempt+1} failed ({e}). Retrying in {delay:.2f}s...")
                    time.sleep(delay)
            raise last_err
        return wrapper
    return decorator

async def retry_async_with_backoff(
    func: Callable,
    *args,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    **kwargs
) -> Any:
    """
    Executes an asynchronous coroutine with exponential backoff and jitter.
    """
    last_err = None
    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt == max_attempts - 1:
                raise e
            delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0.05, 0.2)
            print(f"[AsyncRetry] Attempt {attempt+1} failed ({e}). Retrying in {delay:.2f}s...")
            await asyncio.sleep(delay)
    raise last_err

# Global Circuit Breakers for registered model providers
CIRCUIT_BREAKERS = {
    "gemini": CircuitBreaker("gemini"),
    "openai": CircuitBreaker("openai"),
    "claude": CircuitBreaker("claude"),
    "vllm": CircuitBreaker("vllm")
}
