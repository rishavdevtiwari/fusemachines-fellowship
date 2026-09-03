"""
Unit and Integration Test Suite for ShopAssist AI Assistant.
Validates ONNX Intent Router, RAG Pipeline, External Tools, Structured JSON Validation, Cache, and Orchestrator.
"""

import os
import sys
import unittest
import json

# Ensure Week 15 is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings
from src.router.onnx_router import onnx_router
from src.core.rag_pipeline import rag_pipeline
from src.core.tools import (
    check_order_status,
    calculate_cancellation_fee,
    check_refund_eligibility,
    escalate_to_human
)
from src.core.structured_outputs import (
    AssistantStructuredResponse,
    validate_or_repair_response,
    CANONICAL_INTENTS
)
from src.services.cache_service import LRUTTLCache
from src.services.assistant_service import assistant_orchestrator, AssistantRequest

class TestShopAssistAssistant(unittest.TestCase):
    
    def test_onnx_intent_router(self):
        """Tests that ONNX router produces valid predictions and covers 10 categories."""
        pred = onnx_router.predict("How do I cancel my order and calculate the penalty fee?")
        self.assertIn(pred.intent, CANONICAL_INTENTS)
        self.assertGreater(pred.confidence, 0.0)
        self.assertLessEqual(pred.confidence, 1.0)
        self.assertGreater(pred.latency_ms, 0.0)
        self.assertEqual(len(pred.probabilities), 10)
        
    def test_rag_pipeline_search(self):
        """Tests that RAG pipeline successfully returns matching policy context."""
        context, sources = rag_pipeline.retrieve_context("What is the return policy window?", top_k=2)
        self.assertGreater(len(context), 0)
        self.assertGreater(len(sources), 0)
        self.assertTrue(any("Return" in s for s in sources))

    def test_tools_order_status(self):
        """Tests order lookup tool for existing and non-existing records."""
        # Existing order ORD-1001
        res = check_order_status("ORD-1001")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["order_id"], "ORD-1001")
        self.assertIn("FedEx", res["courier"])
        
        # Non-existing order
        bad_res = check_order_status("ORD-999999")
        self.assertEqual(bad_res["status"], "not_found")

    def test_tools_cancellation_fee(self):
        """Tests tier-based cancellation fee calculations."""
        # 0.5 hours -> Grace Period ($0 fee)
        res_grace = calculate_cancellation_fee("ORD-1002", hours_elapsed=0.5)
        self.assertEqual(res_grace["applicable_fee"], 0.0)
        self.assertTrue(res_grace["can_cancel"])
        
        # 2.5 hours -> Staging Stage ($5 fee)
        res_stage2 = calculate_cancellation_fee("ORD-1001", hours_elapsed=2.5)
        self.assertEqual(res_stage2["applicable_fee"], 5.0)
        self.assertTrue(res_stage2["can_cancel"])
        
        # 10.0 hours -> Picking Stage ($15 fee)
        res_stage3 = calculate_cancellation_fee("ORD-1001", hours_elapsed=10.0)
        self.assertEqual(res_stage3["applicable_fee"], 15.0)
        self.assertTrue(res_stage3["can_cancel"])
        
        # Shipped order cannot be canceled
        res_shipped = calculate_cancellation_fee("ORD-1004")
        self.assertFalse(res_shipped["can_cancel"])

    def test_tools_refund_eligibility(self):
        """Tests return policy eligibility rules."""
        # Within 30 days (11 days)
        res_valid = check_refund_eligibility("ORD-1003", days_since_delivery=11)
        self.assertTrue(res_valid["eligible"])
        self.assertGreater(res_valid["estimated_refund"], 0)
        
        # Beyond 30 days (45 days)
        res_expired = check_refund_eligibility("ORD-1005", days_since_delivery=45)
        self.assertFalse(res_expired["eligible"])

    def test_structured_json_validation_and_repair(self):
        """Tests Pydantic validation and automatic JSON self-repair."""
        # Valid JSON string
        valid_json = json.dumps({
            "intent": "DELIVERY",
            "confidence": 0.95,
            "thought_process": "Checking courier status.",
            "needs_tool": False,
            "tool_call": None,
            "tool_result": None,
            "rag_sources": ["Shipping Guide"],
            "final_response": "Your package is currently in transit with FedEx.",
            "escalate_to_human": False,
            "action_taken": "Direct Answer"
        })
        resp = validate_or_repair_response(valid_json)
        self.assertIsInstance(resp, AssistantStructuredResponse)
        self.assertEqual(resp.intent, "DELIVERY")
        self.assertEqual(resp.confidence, 0.95)
        
        # Malformed / Plain Text repair
        plain_text = "Here is some raw unstructured text response."
        repaired = validate_or_repair_response(plain_text)
        self.assertIsInstance(repaired, AssistantStructuredResponse)
        self.assertEqual(repaired.final_response, plain_text)

    def test_cache_service(self):
        """Tests LRU caching with capacity eviction and hit/miss telemetry."""
        cache = LRUTTLCache(max_size=3, default_ttl=60)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")
        
        self.assertEqual(cache.get("k1"), "v1")
        # Add 4th item -> k2 should be evicted (since k1 was accessed)
        cache.set("k4", "v4")
        self.assertIsNone(cache.get("k2"))
        self.assertEqual(cache.get("k1"), "v1")
        
        stats = cache.get_stats()
        self.assertEqual(stats["evictions"], 1)
        self.assertGreater(stats["hits"], 0)

    def test_orchestrator_flow(self):
        """Tests end-to-end assistant processing loop."""
        req = AssistantRequest(
            query="Can I cancel order ORD-1001?",
            provider="mock",
            skip_cache=True
        )
        resp = assistant_orchestrator.process_query(req)
        self.assertIsNotNone(resp)
        self.assertEqual(resp.structured_output.intent, "CANCELLATION_FEE")
        self.assertTrue(resp.structured_output.needs_tool)
        self.assertEqual(resp.tool_executed, "calculate_cancellation_fee")
        self.assertGreater(resp.total_latency_ms, 0.0)

if __name__ == "__main__":
    unittest.main()
