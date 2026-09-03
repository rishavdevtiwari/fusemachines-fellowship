"""
Resilience & Reliability Test Suite for ShopAssist AI.
Validates Rate Limiting, Circuit Breakers, Exponential Retries, and Provider Fallbacks.
"""

import os
import sys
import unittest
import time

# Ensure Week 15 is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.rate_limiter import SlidingWindowRateLimiter
from src.services.resilience import CircuitBreaker, CircuitState, retry_with_backoff
from src.core.llm_client import llm_client

class TestShopAssistResilience(unittest.TestCase):

    def test_rate_limiter(self):
        """Tests that sliding window rate limiter blocks traffic above threshold."""
        limiter = SlidingWindowRateLimiter(requests_per_minute=3, window_seconds=2)
        client_id = "test_client_42"
        
        # Requests 1, 2, 3 should be allowed
        self.assertTrue(limiter.is_allowed(client_id)[0])
        self.assertTrue(limiter.is_allowed(client_id)[0])
        self.assertTrue(limiter.is_allowed(client_id)[0])
        
        # 4th request must be blocked
        allowed, remaining, retry_after = limiter.is_allowed(client_id)
        self.assertFalse(allowed)
        self.assertEqual(remaining, 0)
        self.assertGreater(retry_after, 0)
        
        # Wait for window to expire
        time.sleep(2.1)
        allowed_after_wait, _, _ = limiter.is_allowed(client_id)
        self.assertTrue(allowed_after_wait)

    def test_circuit_breaker(self):
        """Tests circuit breaker state machine (CLOSED -> OPEN -> HALF_OPEN)."""
        cb = CircuitBreaker("test_provider", failure_threshold=2, recovery_timeout_seconds=0.5)
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertTrue(cb.can_execute())
        
        # Record 1 failure
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.CLOSED)
        
        # Record 2nd failure -> Trips to OPEN
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)
        self.assertFalse(cb.can_execute())
        
        # Wait for recovery timeout
        time.sleep(0.6)
        # Should transition to HALF_OPEN probe state
        self.assertTrue(cb.can_execute())
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
        
        # On success, reset to CLOSED
        cb.record_success()
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertEqual(cb.failure_count, 0)

    def test_retry_with_backoff(self):
        """Tests exponential backoff retry mechanism."""
        call_count = 0
        
        @retry_with_backoff(max_attempts=3, base_delay=0.05, jitter=False)
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient network glitch")
            return "SUCCESS"
            
        result = flaky_operation()
        self.assertEqual(result, "SUCCESS")
        self.assertEqual(call_count, 3)

    def test_llm_provider_fallback(self):
        """Tests that LLM client gracefully degrades to fallback when primary fails."""
        # Force a call with mock provider or missing key
        result = llm_client.generate(
            user_prompt="I want to track order ORD-1003",
            system_prompt="You are a helpful assistant.",
            preferred_provider="mock"
        )
        self.assertIsNotNone(result)
        self.assertIn("DELIVERY", result.structured_response.intent)
        self.assertGreater(len(result.structured_response.final_response), 0)

if __name__ == "__main__":
    unittest.main()
