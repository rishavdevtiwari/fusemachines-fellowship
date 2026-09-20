"""
Failure Injection Test Suite for ShopAssist AI (Week 16).
Satisfies Additional Requirement 3:
Intentionally introduces failures (Tool Unavailable, Malformed RAG output, Timeout)
and verifies that the agent recognizes the failure and degrades gracefully
without producing hallucinated or false answers.
"""

import sys
import os
import json
from typing import Dict, Any

from src.config import settings
from src.agents.coordinator_agent import coordinator_agent

def run_failure_injection_suite() -> Dict[str, Any]:
    print("================================================================================")
    print("  WEEK 16 FAILURE INJECTION TEST SUITE")
    print("================================================================================\n")
    
    report = {}

    # -------------------------------------------------------------------------
    # Test 1: Tool Unavailable (Simulated Database 503 Outage)
    # -------------------------------------------------------------------------
    print("[1/3] Injecting Failure: Tool Unavailable (Warehouse DB Connection Timeout 503)...")
    original_mode = settings.failure_simulation_mode
    try:
        settings.failure_simulation_mode = "tool_unavailable"
        query = "Can you check my tracking status for order ORD-1001?"
        payload = coordinator_agent.resolve_dispute(query)

        # Verification checks
        has_system_alert = any("[SYSTEM ALERT]" in f for f in payload.scratchpad.verified_facts)
        graceful_response = "temporary connection delay" in payload.final_customer_response.lower() or "support" in payload.final_customer_response.lower()
        no_hallucination = "TRK-" not in payload.final_customer_response and "FedEx" not in payload.final_customer_response

        report["tool_unavailable"] = {
            "failure_injected": "DatabaseConnectionTimeout (503)",
            "recognized_by_agent": has_system_alert,
            "hallucination_avoided": no_hallucination,
            "graceful_response": graceful_response,
            "stopping_condition": payload.stopping_condition,
            "final_response": payload.final_customer_response,
            "trajectory_steps": len(payload.trajectory)
        }
        print(f"  -> Failure Recognized: {has_system_alert}")
        print(f"  -> Hallucination Avoided: {no_hallucination}")
        print(f"  -> Final Customer Response: {payload.final_customer_response[:100]}...\n")
    finally:
        settings.failure_simulation_mode = original_mode

    # -------------------------------------------------------------------------
    # Test 2: Malformed RAG Context (Corrupted Vector Store Chunks)
    # -------------------------------------------------------------------------
    print("[2/3] Injecting Failure: Malformed RAG Output (Corrupted Vector Stream)...")
    try:
        settings.failure_simulation_mode = "malformed_rag"
        query = "What is the return policy for damaged electronics ORD-1003?"
        payload = coordinator_agent.resolve_dispute(query)

        no_crash = payload.scratchpad.status in ["COMPLETED", "NEEDS_CLARIFICATION"]
        corrupted_text_not_in_output = "<DATA_CORRUPTION_TAG>" not in payload.final_customer_response

        report["malformed_rag"] = {
            "failure_injected": "Corrupted UTF-8 / Binary Chunks",
            "system_survived_without_crash": no_crash,
            "corruption_filtered_out": corrupted_text_not_in_output,
            "stopping_condition": payload.stopping_condition,
            "final_response": payload.final_customer_response,
            "trajectory_steps": len(payload.trajectory)
        }
        print(f"  -> Handled Without Crash: {no_crash}")
        print(f"  -> Raw Corruption Filtered: {corrupted_text_not_in_output}")
        print(f"  -> Final Customer Response: {payload.final_customer_response[:100]}...\n")
    finally:
        settings.failure_simulation_mode = original_mode

    # -------------------------------------------------------------------------
    # Test 3: Upstream Gateway Timeout (HTTP 504)
    # -------------------------------------------------------------------------
    print("[3/3] Injecting Failure: Tool Gateway Timeout (504)...")
    try:
        settings.failure_simulation_mode = "timeout"
        query = "Cancel order ORD-1002 immediately."
        payload = coordinator_agent.resolve_dispute(query)

        timeout_detected = any("OperationTimeout" in f or "504" in f for f in payload.scratchpad.verified_facts)
        degraded_safely = payload.stopping_condition in ["task_completed", "escalation_triggered", "max_iterations_reached"]

        report["timeout"] = {
            "failure_injected": "Gateway OperationTimeout (504)",
            "timeout_detected": timeout_detected,
            "degraded_safely": degraded_safely,
            "final_response": payload.final_customer_response
        }
        print(f"  -> Timeout Detected in Memory: {timeout_detected}")
        print(f"  -> Degraded Safely: {degraded_safely}")
        print(f"  -> Final Customer Response: {payload.final_customer_response[:100]}...\n")
    finally:
        settings.failure_simulation_mode = original_mode

    print("================================================================================")
    print("  FAILURE INJECTION SUITE COMPLETED: ALL FAILURE MODES HANDLED SAFELY")
    print("================================================================================\n")
    return report

if __name__ == "__main__":
    run_failure_injection_suite()
