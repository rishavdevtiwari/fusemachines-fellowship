"""
Structured Output Schemas & Validation for ShopAssist AI.
Defines strict Pydantic data models for guaranteed JSON responses,
tool calls, intent classification, and automated JSON repair.
"""

import json
import re
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, ValidationError

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

IntentType = Literal[
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
    tool_name: str = Field(..., description="Name of the tool to execute")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Key-value arguments for the tool")

class ToolExecutionResult(BaseModel):
    tool_name: str
    status: Literal["success", "error", "skipped"]
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class AssistantStructuredResponse(BaseModel):
    """
    Standard production response schema guaranteed for all AI Assistant interactions.
    """
    intent: str = Field(
        ...,
        description="Predicted customer support intent category"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    thought_process: str = Field(
        ...,
        description="Chain-of-thought analysis explaining why this intent and action were selected"
    )
    needs_tool: bool = Field(
        default=False,
        description="Whether an external tool execution was required"
    )
    tool_call: Optional[ToolCallSpec] = Field(
        default=None,
        description="Specification of tool to call if needs_tool is True"
    )
    tool_result: Optional[ToolExecutionResult] = Field(
        default=None,
        description="Result of executed tool if applicable"
    )
    rag_sources: List[str] = Field(
        default_factory=list,
        description="List of knowledge base policy documents cited"
    )
    final_response: str = Field(
        ...,
        description="Customer-facing polished answer in markdown format"
    )
    escalate_to_human: bool = Field(
        default=False,
        description="Flag indicating if the issue requires immediate human agent intervention"
    )
    action_taken: str = Field(
        default="Direct Answer",
        description="Brief summary of the action performed (e.g. 'Order Status Retrieved', 'RAG Policy Quoted')"
    )

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extracts JSON from an LLM response string.
    Handles Markdown code fences (```json ... ```), raw objects, and loose formatting.
    """
    if not text or not text.strip():
        return None
        
    text = text.strip()
    
    # 1. Look for ```json ... ``` or ``` ... ``` code blocks
    json_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1))
        except json.JSONDecodeError:
            pass
            
    # 2. Look for outermost curly braces
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
            
    # 3. Direct parse attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def validate_or_repair_response(
    raw_output: Any,
    default_intent: str = "ORDER",
    fallback_message: str = "Thank you for reaching out. How can I assist you today?"
) -> AssistantStructuredResponse:
    """
    Validates model output against AssistantStructuredResponse schema.
    If parsing or schema validation fails, attempts automatic repair
    to ensure the API never returns malformed responses to clients.
    """
    parsed_data = None
    
    if isinstance(raw_output, dict):
        parsed_data = raw_output
    elif isinstance(raw_output, str):
        parsed_data = extract_json_from_text(raw_output)
        
    if parsed_data:
        try:
            # Normalize intent if slightly mismatched
            if "intent" in parsed_data and parsed_data["intent"]:
                intent_upper = str(parsed_data["intent"]).upper().replace(" ", "_")
                for ci in CANONICAL_INTENTS:
                    if ci in intent_upper:
                        parsed_data["intent"] = ci
                        break
            # Clamp confidence
            if "confidence" in parsed_data:
                try:
                    c = float(parsed_data["confidence"])
                    parsed_data["confidence"] = max(0.0, min(1.0, c))
                except (ValueError, TypeError):
                    parsed_data["confidence"] = 0.85
            else:
                parsed_data["confidence"] = 0.85
                
            return AssistantStructuredResponse(**parsed_data)
        except ValidationError:
            # Fallback to repair below
            pass

    # Self-repair: Construct valid structured object from whatever text was returned
    raw_text = str(raw_output) if raw_output else fallback_message
    # Strip markdown fences if present
    clean_text = re.sub(r"```.*?```", "", raw_text, flags=re.DOTALL).strip() or raw_text
    
    return AssistantStructuredResponse(
        intent=default_intent,
        confidence=0.75,
        thought_process="Automated schema repair applied to enforce valid production JSON structure.",
        needs_tool=False,
        tool_call=None,
        tool_result=None,
        rag_sources=[],
        final_response=clean_text,
        escalate_to_human=False,
        action_taken="Direct Answer (Repaired JSON)"
    )
