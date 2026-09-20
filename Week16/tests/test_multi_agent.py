"""
Unit Tests for Multi-Agent Collaboration (Coordinator, Investigator, Verifier).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.coordinator_agent import coordinator_agent
from src.agents.investigator_agent import investigator_agent
from src.agents.verification_agent import verification_agent

class TestMultiAgentSystem(unittest.TestCase):

    def test_investigator_tool_execution(self):
        """Tests that Investigator agent selects and runs tools properly."""
        res = investigator_agent.run_task("check fulfillment for order ORD-1001", order_id="ORD-1001")
        self.assertEqual(res.tool_name, "check_order_status")
        self.assertIn("ORD-1001", res.compacted_observation)
        self.assertGreater(res.total_tokens, 0)

    def test_verification_agent_damage_waiver(self):
        """Tests that Policy Auditor identifies fee waivers and overrides standard deductions."""
        facts = [
            "Order ORD-1003: Status DELIVERED 11 days ago, Total $349.00",
            "Customer reports device arrived cracked and damaged."
        ]
        audit = verification_agent.audit_dispute(facts, "damaged merchandise return fee policy")
        self.assertTrue(audit.fee_waiver_applicable)
        self.assertTrue(audit.policy_compliant)
        self.assertIn("damage", audit.waiver_reason.lower())
        self.assertGreater(len(audit.rag_sources), 0)

    def test_end_to_end_dispute_resolution(self):
        """Tests complete arbitration loop for damaged item dispute (TC-03)."""
        query = "I received order ORD-1003 11 days ago, but the package arrived cracked and damaged. Am I entitled to a full refund or will I be charged a return shipping fee?"
        payload = coordinator_agent.resolve_dispute(query)

        self.assertEqual(payload.scratchpad.status, "COMPLETED")
        self.assertTrue(payload.scratchpad.fee_waiver_applied)
        self.assertIn("100% full refund", payload.final_customer_response)
        self.assertIn("waives", payload.final_customer_response.lower())

if __name__ == "__main__":
    unittest.main()
