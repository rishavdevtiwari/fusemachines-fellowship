"""
Unified Assistant Orchestration Service for ShopAssist AI.
Coordinates Cache -> ONNX Edge Router -> RAG Pipeline -> Tool Execution -> Multi-Provider LLM -> JSON Schema Validation.
"""

import time
import re
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from src.config import settings
from src.core.structured_outputs import AssistantStructuredResponse, ToolCallSpec, ToolExecutionResult
from src.core.tools import execute_tool
from src.core.rag_pipeline import rag_pipeline
from src.core.prompt_templates import SYSTEM_PROMPT, build_user_prompt
from src.core.llm_client import llm_client, LLMGenerationResult
from src.router.onnx_router import onnx_router, IntentPrediction
from src.services.cache_service import cache_service
from src.services.resilience import CIRCUIT_BREAKERS, retry_with_backoff

class AssistantRequest(BaseModel):
    query: str
    customer_id: str = "guest_user"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    provider: Optional[str] = None
    skip_cache: bool = False

class AssistantResponsePayload(BaseModel):
    structured_output: AssistantStructuredResponse
    cached: bool = False
    onnx_intent: str
    onnx_confidence: float
    onnx_latency_ms: float
    rag_sources: List[str]
    tool_executed: Optional[str] = None
    provider_used: str
    model_name: str
    llm_latency_ms: float
    total_latency_ms: float

class AssistantOrchestrator:
    """
    Production assistant coordinator implementing the full agentic loop.
    """
    def __init__(self):
        self.total_queries: int = 0
        self.tool_invocations: int = 0

    def process_query(self, req: AssistantRequest) -> AssistantResponsePayload:
        t_start = time.perf_counter()
        self.total_queries += 1
        
        # 1. Check Prompt / Response Cache
        cache_key = cache_service.generate_key(
            "assistant",
            {
                "query": req.query.strip().lower(),
                "provider": req.provider or settings.primary_provider,
                "temp": req.temperature or settings.default_temperature
            }
        )
        
        if not req.skip_cache:
            cached_item = cache_service.get(cache_key)
            if cached_item:
                total_latency = (time.perf_counter() - t_start) * 1000.0
                cached_payload = AssistantResponsePayload(**cached_item)
                cached_payload.cached = True
                cached_payload.total_latency_ms = round(total_latency, 2)
                return cached_payload

        # 2. Ultra-Fast ONNX Edge Intent Router (<5ms)
        onnx_pred = onnx_router.predict(req.query)
        
        # 3. RAG Knowledge Base Retrieval
        rag_context, rag_sources = rag_pipeline.retrieve_context(req.query, top_k=settings.rag_top_k)
        
        # 4. Tool Calling & Pre-execution Evaluation
        # Inspect for explicit Order IDs (e.g. ORD-1001)
        order_match = re.search(r"\b(ORD-\d{4,6})\b", req.query, re.IGNORECASE)
        found_order_id = order_match.group(1).upper() if order_match else None
        
        tool_name_to_call: Optional[str] = None
        tool_args: Dict[str, Any] = {}
        tool_exec_result: Optional[Dict[str, Any]] = None
        
        q_lower = req.query.lower()
        
        if found_order_id:
            if onnx_pred.intent == "CANCELLATION_FEE" or "cancel" in q_lower or "fee" in q_lower:
                tool_name_to_call = "calculate_cancellation_fee"
                tool_args = {"order_id": found_order_id}
            elif onnx_pred.intent == "REFUND" or "return" in q_lower or "refund" in q_lower:
                tool_name_to_call = "check_refund_eligibility"
                tool_args = {"order_id": found_order_id}
            else:
                tool_name_to_call = "check_order_status"
                tool_args = {"order_id": found_order_id}
        elif any(k in q_lower for k in ["escalate", "human", "supervisor", "fraud", "lawyer", "manager"]):
            tool_name_to_call = "escalate_to_human"
            tool_args = {"issue_summary": req.query, "urgency": "high"}
            
        if tool_name_to_call:
            self.tool_invocations += 1
            tool_exec_result = execute_tool(tool_name_to_call, tool_args)

        # 5. Build Dynamic Engineered Prompt
        user_prompt = build_user_prompt(
            query=req.query,
            rag_context=rag_context,
            onnx_intent_hint=onnx_pred.intent,
            onnx_confidence=onnx_pred.confidence,
            tool_result=tool_exec_result
        )

        # 6. Multi-Provider LLM Generation with Circuit Breakers & Retries
        provider = req.provider or settings.primary_provider
        cb = CIRCUIT_BREAKERS.get(provider.lower())
        
        if cb and not cb.can_execute():
            print(f"[Orchestrator] Circuit for '{provider}' is OPEN. Engaging fallback provider.")
            provider = settings.fallback_provider

        try:
            llm_result = llm_client.generate(
                user_prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=req.temperature,
                top_p=req.top_p,
                preferred_provider=provider
            )
            if cb:
                cb.record_success()
        except Exception as e:
            if cb:
                cb.record_failure()
            # Ultimate safety net: deterministic mock response
            llm_result = llm_client._call_mock_engine(user_prompt, SYSTEM_PROMPT)

        # Attach tool metadata to structured response if tool executed
        structured = llm_result.structured_response
        if tool_name_to_call:
            structured.needs_tool = True
            structured.tool_call = ToolCallSpec(tool_name=tool_name_to_call, arguments=tool_args)
            structured.tool_result = ToolExecutionResult(
                tool_name=tool_name_to_call,
                status="success" if tool_exec_result.get("status") != "error" else "error",
                data=tool_exec_result
            )
            structured.action_taken = f"Executed tool '{tool_name_to_call}'"

        if rag_sources and not structured.rag_sources:
            structured.rag_sources = rag_sources

        total_latency = (time.perf_counter() - t_start) * 1000.0

        payload = AssistantResponsePayload(
            structured_output=structured,
            cached=False,
            onnx_intent=onnx_pred.intent,
            onnx_confidence=onnx_pred.confidence,
            onnx_latency_ms=onnx_pred.latency_ms,
            rag_sources=rag_sources,
            tool_executed=tool_name_to_call,
            provider_used=llm_result.provider_used,
            model_name=llm_result.model_name,
            llm_latency_ms=llm_result.latency_ms,
            total_latency_ms=round(total_latency, 2)
        )

        # 7. Store in Cache
        cache_service.set(cache_key, payload.model_dump())

        return payload

assistant_orchestrator = AssistantOrchestrator()
