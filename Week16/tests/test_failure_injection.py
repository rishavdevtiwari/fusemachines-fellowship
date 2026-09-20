"""
Unit Tests for Failure Injection (Tool Unavailable, Malformed RAG, Timeout).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.failure_injection import run_failure_injection_suite

class TestFailureInjection(unittest.TestCase):

    def test_failure_suite_execution(self):
        report = run_failure_injection_suite()
        self.assertIn("tool_unavailable", report)
        self.assertIn("malformed_rag", report)
        self.assertIn("timeout", report)

        self.assertTrue(report["tool_unavailable"]["recognized_by_agent"])
        self.assertTrue(report["tool_unavailable"]["hallucination_avoided"])

        self.assertTrue(report["malformed_rag"]["system_survived_without_crash"])
        self.assertTrue(report["malformed_rag"]["corruption_filtered_out"])

        self.assertTrue(report["timeout"]["timeout_detected"])
        self.assertTrue(report["timeout"]["degraded_safely"])

if __name__ == "__main__":
    unittest.main()
