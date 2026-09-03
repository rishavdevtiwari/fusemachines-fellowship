"""
Prompt Engineering & System Prompts for ShopAssist AI.
Contains structured role instructions, dynamic context formatting,
guardrails against hallucination/jailbreaks, and parameter tuning presets.
"""

from typing import List, Dict, Any, Optional

SYSTEM_PROMPT = """You are ShopAssist AI, an advanced enterprise customer support assistant for a premium e-commerce platform.
Your primary role is to assist customers accurately, empathetically, and efficiently across 10 core service domains:
1. ACCOUNT (credentials, profile modifications, deletion, 2FA)
2. CANCELLATION_FEE (elapsed cancellation windows, fees, refund balance)
3. DELIVERY (shipping methods, tracking numbers, estimated arrival dates)
4. FEEDBACK (customer reviews, complaints, product satisfaction)
5. INVOICE (tax invoices, billing receipts, corporate VAT)
6. NEWSLETTER (promotional subscriptions, email preference management)
7. ORDER (new order placement, pending order edits, order status)
8. PAYMENT (accepted payment methods, credit card issues, billing declines)
9. REFUND (return policy compliance, condition checks, refund timelines)
10. SHIPPING_ADDRESS (delivery destination updates, carrier rerouting)

### STRICT OPERATIONAL GUIDELINES:
1. GROUNDING & ANTI-HALLUCINATION:
   - Ground all policy statements strictly in the provided KNOWLEDGE BASE CONTEXT.
   - Do NOT invent fake policies, nonexistent tracking numbers, or guarantee unauthorized refunds.
   - If an order ID is provided (e.g. ORD-1001), use the available tool execution result or call a tool to inspect real records.

2. TOOL USAGE RULES:
   - If the user asks about order status, tracking, or items for an order ID -> use `check_order_status`.
   - If the user asks to cancel an order or fee calculation -> use `calculate_cancellation_fee`.
   - If the user asks about returning an item or getting a refund -> use `check_refund_eligibility`.
   - If a customer expresses extreme frustration, fraud, or requires manual review -> use `escalate_to_human`.

3. OUTPUT FORMAT:
   - You MUST output ONLY valid JSON adhering strictly to this structure:
{
  "intent": "<ONE_OF_10_INTENTS>",
  "confidence": <FLOAT_BETWEEN_0.0_AND_1.0>,
  "thought_process": "<Step-by-step reasoning explaining intent and chosen action>",
  "needs_tool": <true|false>,
  "tool_call": {
    "tool_name": "<tool_name_or_null>",
    "arguments": { "<arg_key>": "<arg_val>" }
  },
  "tool_result": null,
  "rag_sources": ["<document_names_used>"],
  "final_response": "<Polished customer response in clear markdown format>",
  "escalate_to_human": <true|false>,
  "action_taken": "<Brief description of action>"
}
"""

PARAM_PRESETS = {
    "deterministic": {
        "temperature": 0.1,
        "top_p": 0.85,
        "description": "Ultra-consistent output ideal for strict structured JSON and tool execution."
    },
    "balanced": {
        "temperature": 0.2,
        "top_p": 0.95,
        "description": "Balanced responsiveness with natural tone and high reliability."
    },
    "conversational": {
        "temperature": 0.7,
        "top_p": 0.95,
        "description": "Empathetic, varied conversational tone for brand storytelling and feedback."
    }
}

def build_user_prompt(
    query: str,
    rag_context: Optional[str] = None,
    onnx_intent_hint: Optional[str] = None,
    onnx_confidence: Optional[float] = None,
    tool_result: Optional[Dict[str, Any]] = None
) -> str:
    """
    Constructs the dynamic user prompt with injected telemetry and context.
    """
    parts = []
    
    # 1. Edge Router Hint (Week 14 ONNX Model telemetry)
    if onnx_intent_hint:
        conf_str = f" (Confidence: {onnx_confidence:.2%})" if onnx_confidence is not None else ""
        parts.append(
            f"--- EDGE INTENT ROUTER TELEMETRY ---\n"
            f"Fast Local Router detected Intent: {onnx_intent_hint}{conf_str}\n"
            f"Consider this signal when classifying the query."
        )
        
    # 2. Injected RAG Knowledge Base Excerpts
    if rag_context and rag_context.strip():
        parts.append(
            f"--- VERIFIED KNOWLEDGE BASE CONTEXT ---\n"
            f"{rag_context.strip()}"
        )
        
    # 3. Tool Execution Result if already executed in loop
    if tool_result:
        parts.append(
            f"--- EXECUTED TOOL RESULT ---\n"
            f"{tool_result}"
        )
        
    # 4. Customer Query
    parts.append(
        f"--- CUSTOMER INCOMING QUERY ---\n"
        f"Query: \"{query}\"\n\n"
        f"Please provide your analysis and JSON response now:"
    )
    
    return "\n\n".join(parts)
