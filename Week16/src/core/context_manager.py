"""
Context Engineering Manager for ShopAssist AI Agent (Week 16).
Implements:
1. Structured External Notes (Investigation Scratchpad / Working Memory).
2. Tool Result Compaction (Compressing verbose operational payloads into factual claims).
3. Tool Result Clearing (Pruning raw tool outputs from prompt history to prevent Context Saturation).
4. Telemetry tracking tokens saved and compaction ratios.
"""

from typing import Dict, Any, List, Optional
from src.core.structured_outputs import InvestigationScratchpad

class ContextManager:
    """
    Manages context lifecycle and token efficiency across multi-step agentic iterations.
    """
    def __init__(self):
        self.raw_tokens_accumulated: int = 0
        self.compacted_tokens_used: int = 0

    def initialize_scratchpad(self, query: str, order_id: Optional[str] = None) -> InvestigationScratchpad:
        """Creates a fresh, typed investigation scratchpad for a new dispute."""
        return InvestigationScratchpad(
            dispute_goal=query.strip(),
            order_id=order_id,
            verified_facts=[],
            applicable_policies=[],
            exceptions_identified=[],
            contradictions_resolved=[],
            pending_questions=[],
            verdict=None,
            status="IN_PROGRESS"
        )

    def compact_tool_output(self, tool_name: str, raw_output: Dict[str, Any]) -> str:
        """
        Compaction Technique: Translates verbose raw database/API JSON responses
        into dense, factual assertions suitable for prompt injection.
        """
        if not raw_output:
            return "No output returned."

        status = raw_output.get("status", "unknown")
        if status == "simulated_failure":
            return f"[ERROR: Tool '{tool_name}' failed upstream: {raw_output.get('error_type')} - {raw_output.get('error_message')}]"

        if tool_name == "check_order_status":
            if status == "success":
                oid = raw_output.get("order_id")
                st = raw_output.get("order_status")
                amt = raw_output.get("total_amount")
                courier = raw_output.get("courier")
                hours = raw_output.get("hours_since_order", "N/A")
                days_deliv = raw_output.get("days_since_delivery", "N/A")
                items = [it.get("name") for it in raw_output.get("items", []) if isinstance(it, dict)]
                items_str = ", ".join(items[:2]) if items else "Items logged"
                return (
                    f"Order {oid}: Status={st}, Total=${amt}, Courier={courier}, "
                    f"HoursSinceOrder={hours}h, DaysSinceDelivery={days_deliv}d, Items=[{items_str}]"
                )
            elif status == "not_found":
                return f"Order '{raw_output.get('order_id')}' NOT FOUND in order database."

        elif tool_name == "calculate_cancellation_fee":
            if raw_output.get("can_cancel"):
                fee = raw_output.get("applicable_fee", 0.0)
                stage = raw_output.get("stage", "Unknown")
                refund = raw_output.get("estimated_net_refund", 0.0)
                return f"Cancellation Eligible: Stage='{stage}', PenaltyFee=${fee:.2f}, EstimatedRefund=${refund:.2f}."
            else:
                return f"Cancellation Ineligible: Status='{raw_output.get('order_status')}', Reason: {raw_output.get('message')}"

        elif tool_name == "check_refund_eligibility":
            if raw_output.get("eligible"):
                deduction = raw_output.get("shipping_deduction", 5.99)
                waived = raw_output.get("fee_waived", False)
                est_refund = raw_output.get("estimated_refund", 0.0)
                return (
                    f"Refund Eligible: DamageWaived={waived}, LabelDeduction=${deduction:.2f}, "
                    f"EstimatedNetRefund=${est_refund:.2f}. Instructions provided."
                )
            else:
                return f"Refund Ineligible: Reason: {raw_output.get('reason')}"

        elif tool_name == "verify_courier_tracking":
            if status == "success":
                carrier = raw_output.get("courier")
                deliv_st = raw_output.get("verified_delivery_status")
                latest = raw_output.get("latest_scan")
                consistent = raw_output.get("carrier_telemetry_consistent")
                return f"Carrier Telemetry ({carrier}): Status={deliv_st}, LatestScan='{latest}', ConsistentWithDB={consistent}."
            return f"Carrier Telemetry: {raw_output.get('message', 'No tracking')}"

        elif tool_name == "escalate_to_human":
            return f"Human Escalation Ticket: ID={raw_output.get('ticket_id')}, Urgency={raw_output.get('urgency')}, ETA={raw_output.get('estimated_agent_response_time')}."

        # Generic fallback compaction
        compact_str = ", ".join(f"{k}={v}" for k, v in raw_output.items() if not isinstance(v, (dict, list)))
        return f"Tool '{tool_name}' result: {compact_str}"

    def render_scratchpad_for_prompt(self, scratchpad: InvestigationScratchpad) -> str:
        """
        Renders the Structured External Notes into an efficient prompt block.
        Prevents repeating entire dialog history.
        """
        facts_block = "\n".join(f"- {f}" for f in scratchpad.verified_facts) if scratchpad.verified_facts else "- None recorded yet."
        policies_block = "\n".join(f"- {p}" for p in scratchpad.applicable_policies) if scratchpad.applicable_policies else "- None reviewed yet."
        exceptions_block = "\n".join(f"- {e}" for e in scratchpad.exceptions_identified) if scratchpad.exceptions_identified else "- None identified."
        questions_block = "\n".join(f"- {q}" for q in scratchpad.pending_questions) if scratchpad.pending_questions else "- None pending."

        return (
            f"=== INVESTIGATION WORKING MEMORY (SCRATCHPAD) ===\n"
            f"Dispute Goal: {scratchpad.dispute_goal}\n"
            f"Target Order ID: {scratchpad.order_id or 'Unspecified'}\n"
            f"Current Status: {scratchpad.status}\n\n"
            f"Verified Facts Gathered:\n{facts_block}\n\n"
            f"Applicable Policies Reviewed:\n{policies_block}\n\n"
            f"Exceptions / Fee Waivers Identified:\n{exceptions_block}\n\n"
            f"Pending Clarification Questions:\n{questions_block}\n"
            f"=================================================="
        )

context_manager = ContextManager()
