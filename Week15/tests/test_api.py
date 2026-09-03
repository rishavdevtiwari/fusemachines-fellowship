"""
FastAPI Endpoints Automated Test Suite.
Validates /health, /metrics, /route, /rag/search, /chat, /batch, and /cache.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from src.api.main import app

class TestFastAPIEndpoints(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/api/v1/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["onnx_router"]["session_loaded"])
        self.assertGreater(data["rag_pipeline"]["indexed_chunks"], 0)

    def test_metrics_endpoint(self):
        resp = self.client.get("/api/v1/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("cache", data)
        self.assertIn("rate_limiter", data)
        self.assertIn("orchestrator", data)

    def test_route_endpoint(self):
        resp = self.client.post("/api/v1/route", json={"text": "Where is my courier tracking for package ORD-1003?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn(data["intent"], ["DELIVERY", "ORDER"])
        self.assertGreater(data["confidence"], 0.0)

    def test_rag_search_endpoint(self):
        resp = self.client.post("/api/v1/rag/search", json={"query": "What items cannot be returned?", "top_k": 2})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data), 2)
        self.assertIn("Return", data[0]["doc_title"])

    def test_chat_interaction_endpoint(self):
        resp = self.client.post("/api/v1/chat", json={
            "query": "Can I cancel order ORD-1001?",
            "customer_id": "api_test_user",
            "temperature": 0.2
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["structured_output"]["intent"], "CANCELLATION_FEE")
        self.assertTrue(data["structured_output"]["needs_tool"])
        self.assertEqual(data["tool_executed"], "calculate_cancellation_fee")

    def test_batch_interaction_endpoint(self):
        queries = [
            "How do I update my shipping address for ORD-1004?",
            "What is your refund policy window?",
            "Send me an invoice receipt"
        ]
        resp = self.client.post("/api/v1/batch", json={
            "queries": queries,
            "temperature": 0.2
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_queries"], 3)
        self.assertEqual(len(data["results"]), 3)
        self.assertGreater(data["total_batch_time_ms"], 0.0)

    def test_clear_cache_endpoint(self):
        resp = self.client.delete("/api/v1/cache")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "success")

if __name__ == "__main__":
    unittest.main()
