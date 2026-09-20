"""
Structured Output Schemas & Pydantic Validation Models for ShopAssist AI Agent (Week 16).
Defines schemas for the Agentic Loop, Investigation Scratchpad (Structured External Notes),
Trajectory Tracking, Multi-Agent Coordination, and JSON Validation/Repair.
"""

import re
import json
from typing import Dict, Any, Optional, List, Union, Literal
from pydantic import BaseModel, Field

CANONICAL_INTENTS = [
    "ACCOUNT",
    "CANCELLATION_FEE",
    "DELIVERY",
    "FEEDBACK",
    "INVOICE",
    "NEWSLETTER",
    "ORDER",
    "PAYMENT",
    "REFUND",
    "SHIPPING_ADDRESS"
]

class ToolCallSpec(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

class ToolExecutionResult(BaseModel):
    tool_name: str
    status: Literal["success", "error", "not_found", "simulated_failure"]
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class InvestigationScratchpad(BaseModel):
    """
    Structured External Notes: Serves as the agent's external working memory,
    tracking verified facts, policy clauses, fee waivers, and arbitration state.
    """
    dispute_goal: str = ""
    order_id: Optional[str] = None
    verified_facts: List[str] = Field(default_factory=list)
    applicable_policies: List[str] = Field(default_factory=list)
    exceptions_identified: List[str] = Field(default_factory=list)
    contradictions_resolved: List[str] = Field(default_factory=list)
    pending_questions: List[str] = Field(default_factory=list)
    verdict: Optional[str] = None
    fee_waiver_applied: bool = False
    calculated_penalty: float = 0.0
    refund_amount: float = 0.0
    status: Literal["IN_PROGRESS", "COMPLETED", "NEEDS_CLARIFICATION", "ESCALATED", "FAILED"] = "IN_PROGRESS"
    stopping_reason: Optional[str] = None

class AgentTrajectoryStep(BaseModel):
    """
    Fine-grained step record within an agentic reasoning loop.
    """
    step_number: int
    agent_role: str = "coordinator"  # "coordinator", "investigator", "verifier", "single_agent"
    thought: str
    action: str  # "tool_call", "rag_search", "policy_audit", "clarification_request", "final_synthesis"
    action_input: Optional[Dict[str, Any]] = None
    observation_raw: Optional[Any] = None
    observation_compacted: Optional[str] = None
    scratchpad_state_snapshot: Optional[Dict[str, Any]] = None
    tokens_used: int = 0
    latency_ms: float = 0.0

class AgenticResolutionPayload(BaseModel):
    """
    Complete response payload returned by the agentic dispute resolution service.
    """
    query: str
    customer_id: str = "guest_user"
    pattern_used: Literal["multi_agent", "single_agent"] = "multi_agent"
    iterations_count: int = 0
    max_iterations_reached: bool = False
    stopping_condition: str = "task_completed"
    scratchpad: InvestigationScratchpad
    trajectory: List[AgentTrajectoryStep] = Field(default_factory=list)
    final_customer_response: str
    tools_called: List[str] = Field(default_factory=list)
    rag_sources: List[str] = Field(default_factory=list)
    total_tokens: int = 0
    coordination_tokens: int = 0
    total_latency_ms: float = 0.0
    error: Optional[str] = None

# Backwards compatibility models for W15 components
class AssistantStructuredResponse(BaseModel):
    intent: str
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    thought_process: str
    needs_tool: bool = False
    tool_call: Optional[ToolCallSpec] = None
    tool_result: Optional[ToolExecutionResult] = None
    rag_sources: List[str] = Field(default_factory=list)
    final_response: str
    escalate_to_human: bool = False
    action_taken: str = "Direct Answer"

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extracts JSON from an LLM response string."""
    if not text or not text.strip():
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try finding markdown code block ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { and last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None

def validate_or_repair_response(raw_text: str) -> AssistantStructuredResponse:
    """Validates raw output against schema or constructs an auto-repaired structured response."""
    data = extract_json_from_text(raw_text)
    if data and isinstance(data, dict):
        try:
            return AssistantStructuredResponse(**data)
        except Exception:
            pass
            
    return AssistantStructuredResponse(
        intent="ORDER",
        confidence=0.75,
        thought_process="Automated repair applied to raw unstructured response.",
        final_response=raw_text.strip() if raw_text else "Thank you for contacting ShopAssist AI. How can I help you?",
        action_taken="Auto-Repaired Fallback"
    )
