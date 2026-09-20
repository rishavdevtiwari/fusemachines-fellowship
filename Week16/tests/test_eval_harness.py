"""
Unit Tests for Custom Evaluation Harness and Metrics Computation (Week 16).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.eval_harness import EvaluationHarness

class TestEvaluationHarness(unittest.TestCase):

    def setUp(self):
        self.harness = EvaluationHarness()

    def test_harness_loads_test_cases(self):
        self.assertGreater(len(self.harness.test_cases), 0)
        self.assertIn("category", self.harness.test_cases[0])

    def test_single_case_evaluation(self):
        case = self.harness.test_cases[0]
        res = self.harness.evaluate_case(case, pattern="multi_agent")
        self.assertTrue(res.completed)
        self.assertTrue(res.tool_correctness)
        self.assertGreater(res.total_tokens, 0)
        self.assertIsNone(res.failure_type)

    def test_comparative_benchmark(self):
        comp = self.harness.compare_patterns()
        self.assertIn("multi_agent", comp)
        self.assertIn("single_agent_baseline", comp)
        self.assertIn("comparison", comp)
        
        m_rate = comp["multi_agent"]["summary"]["completion_rate_pct"]
        s_rate = comp["single_agent_baseline"]["summary"]["completion_rate_pct"]
        self.assertGreaterEqual(m_rate, s_rate)

if __name__ == "__main__":
    unittest.main()
