"""
Unit Tests for Agentic Loop, Bounded Stopping Conditions, and Context Compaction (Week 16).
"""

import os
import sys
import unittest

# Ensure Week16 is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.coordinator_agent import coordinator_agent
from src.core.context_manager import context_manager
from src.core.structured_outputs import InvestigationScratchpad

class TestAgenticLoop(unittest.TestCase):

    def test_multi_step_trajectory(self):
        """Validates that agent runs for more than 1 iteration when resolving multi-step inquiries."""
        query = "Can I cancel my order ORD-1002 and how much fee will I be charged?"
        payload = coordinator_agent.resolve_dispute(query, max_iterations=4)

        self.assertGreater(payload.iterations_count, 1)
        self.assertLessEqual(payload.iterations_count, 4)
        self.assertFalse(payload.max_iterations_reached)
        self.assertEqual(payload.stopping_condition, "task_completed")
        self.assertGreater(len(payload.trajectory), 1)

    def test_clarification_stopping_condition(self):
        """Validates that loop stops early and asks for clarification when critical details are missing."""
        query = "I want to cancel my recent purchase and get my money back."
        payload = coordinator_agent.resolve_dispute(query, max_iterations=4)

        self.assertEqual(payload.iterations_count, 1)
        self.assertEqual(payload.stopping_condition, "clarification_needed")
        self.assertEqual(payload.scratchpad.status, "NEEDS_CLARIFICATION")
        self.assertTrue(len(payload.scratchpad.pending_questions) > 0)
        self.assertIn("Order ID", payload.final_customer_response)

    def test_escalation_stopping_condition(self):
        """Validates immediate stopping and ticket creation when human escalation is required."""
        query = "This is fraud, I was charged twice. Connect me to a supervisor right now!"
        payload = coordinator_agent.resolve_dispute(query, max_iterations=4)

        self.assertEqual(payload.stopping_condition, "escalation_triggered")
        self.assertEqual(payload.scratchpad.status, "ESCALATED")
        self.assertIn("escalate_to_human", payload.tools_called)

    def test_max_iterations_guardrail(self):
        """Validates that the loop cannot run indefinitely and terminates at max_iterations."""
        query = "Check status for ORD-1001"
        # Force max_iterations = 1
        payload = coordinator_agent.resolve_dispute(query, max_iterations=1)

        self.assertEqual(payload.iterations_count, 1)
        self.assertIsNotNone(payload.final_customer_response)

    def test_context_compaction_and_scratchpad(self):
        """Validates that raw database dumps are compacted into concise factual claims."""
        raw_order = {
            "status": "success",
            "order_id": "ORD-1001",
            "order_status": "PROCESSING",
            "total_amount": 149.99,
            "courier": "FedEx",
            "hours_since_order": 2.5,
            "days_since_delivery": 0,
            "items": [{"name": "Headphones"}]
        }
        compacted = context_manager.compact_tool_output("check_order_status", raw_order)
        self.assertIn("ORD-1001", compacted)
        self.assertIn("Total=$149.99", compacted)
        self.assertLess(len(compacted), 180)

        # Render scratchpad
        sp = InvestigationScratchpad(
            dispute_goal="Cancel order",
            order_id="ORD-1001",
            verified_facts=[compacted]
        )
        rendered = context_manager.render_scratchpad_for_prompt(sp)
        self.assertIn("INVESTIGATION WORKING MEMORY", rendered)
        self.assertIn("ORD-1001", rendered)

if __name__ == "__main__":
    unittest.main()
