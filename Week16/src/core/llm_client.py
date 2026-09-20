"""
Multi-Provider LLM Client for ShopAssist AI Agent (Week 16).
Supports Google Gemini, OpenAI, Anthropic Claude, Local vLLM, and Deterministic Mock Engine.
Provides unified token accounting, error recovery, JSON response parsing, and provider failover.
"""

import os
import time
import json
import re
import httpx
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field

from src.config import settings
from src.core.structured_outputs import (
    AssistantStructuredResponse,
    extract_json_from_text,
    validate_or_repair_response
)

class LLMGenerationResult(BaseModel):
    raw_content: str
    structured_data: Optional[Dict[str, Any]] = None
    structured_response: Optional[AssistantStructuredResponse] = None
    provider_used: str
    model_name: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    fallback_invoked: bool = False

class MultiProviderLLMClient:
    """
    Unified LLM Client orchestrating generative models with token accounting.
    """
    def __init__(self):
        self._init_gemini()

    def _init_gemini(self):
        self.gemini_client = None
        key = settings.gemini_api_key
        if key and key != "your_gemini_api_key_here":
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=key)
            except Exception as e:
                print(f"Warning: Could not initialize google-genai client: {e}")

    def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        preferred_provider: Optional[str] = None
    ) -> LLMGenerationResult:
        """
        Executes generation with automated provider failover and fallback.
        """
        temp = temperature if temperature is not None else settings.default_temperature
        top = top_p if top_p is not None else settings.default_top_p
        tokens = max_tokens if max_tokens is not None else settings.max_tokens
        
        provider = (preferred_provider or settings.primary_provider).lower()
        
        try:
            if provider == "gemini" and self.gemini_client:
                return self._call_gemini(user_prompt, system_prompt, temp, top, tokens)
            elif provider == "openai" and settings.openai_api_key:
                return self._call_openai(user_prompt, system_prompt, temp, top, tokens)
            elif provider == "claude" and settings.anthropic_api_key:
                return self._call_claude(user_prompt, system_prompt, temp, top, tokens)
            elif provider == "vllm":
                return self._call_vllm(user_prompt, system_prompt, temp, top, tokens)
            elif provider == "mock":
                return self._call_mock_engine(user_prompt, system_prompt)
        except Exception as primary_error:
            print(f"Primary provider '{provider}' failed: {primary_error}. Attempting fallback...")

        # Fallback to deterministic mock engine
        res = self._call_mock_engine(user_prompt, system_prompt)
        res.fallback_invoked = True
        return res

    def _estimate_tokens(self, text: str) -> int:
        """Rough 4 chars/token heuristic for mock accounting when API usage is unavailable."""
        return max(1, len(text) // 4)

    def _call_gemini(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int
    ) -> LLMGenerationResult:
        t0 = time.perf_counter()
        from google.genai import types
        
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            response_mime_type="application/json"
        )
        
        response = self.gemini_client.models.generate_content(
            model=settings.gemini_model,
            contents=user_prompt,
            config=config
        )
        latency = (time.perf_counter() - t0) * 1000.0
        
        raw_text = response.text or ""
        structured = validate_or_repair_response(raw_text)
        structured_dict = extract_json_from_text(raw_text)
        
        usage = getattr(response, "usage_metadata", None)
        p_tokens = getattr(usage, "prompt_token_count", self._estimate_tokens(user_prompt + system_prompt)) if usage else self._estimate_tokens(user_prompt)
        c_tokens = getattr(usage, "candidates_token_count", self._estimate_tokens(raw_text)) if usage else self._estimate_tokens(raw_text)
        
        return LLMGenerationResult(
            raw_content=raw_text,
            structured_data=structured_dict,
            structured_response=structured,
            provider_used="gemini",
            model_name=settings.gemini_model,
            latency_ms=round(latency, 2),
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens
        )

    def _call_openai(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int
    ) -> LLMGenerationResult:
        key = settings.openai_api_key
        t0 = time.perf_counter()
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": settings.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"}
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        latency = (time.perf_counter() - t0) * 1000.0
        raw_text = data["choices"][0]["message"]["content"]
        structured_dict = extract_json_from_text(raw_text)
        structured = validate_or_repair_response(raw_text)
        usage = data.get("usage", {})
        p_tokens = usage.get("prompt_tokens", self._estimate_tokens(user_prompt))
        c_tokens = usage.get("completion_tokens", self._estimate_tokens(raw_text))

        return LLMGenerationResult(
            raw_content=raw_text,
            structured_data=structured_dict,
            structured_response=structured,
            provider_used="openai",
            model_name=settings.openai_model,
            latency_ms=round(latency, 2),
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens
        )

    def _call_claude(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int
    ) -> LLMGenerationResult:
        key = settings.anthropic_api_key
        t0 = time.perf_counter()
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload = {
            "model": settings.claude_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}]
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        latency = (time.perf_counter() - t0) * 1000.0
        raw_text = data["content"][0]["text"]
        structured_dict = extract_json_from_text(raw_text)
        structured = validate_or_repair_response(raw_text)
        usage = data.get("usage", {})
        p_tokens = usage.get("input_tokens", self._estimate_tokens(user_prompt))
        c_tokens = usage.get("output_tokens", self._estimate_tokens(raw_text))

        return LLMGenerationResult(
            raw_content=raw_text,
            structured_data=structured_dict,
            structured_response=structured,
            provider_used="claude",
            model_name=settings.claude_model,
            latency_ms=round(latency, 2),
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens
        )

    def _call_vllm(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int
    ) -> LLMGenerationResult:
        t0 = time.perf_counter()
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": settings.vllm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{settings.vllm_base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        latency = (time.perf_counter() - t0) * 1000.0
        raw_text = data["choices"][0]["message"]["content"]
        structured_dict = extract_json_from_text(raw_text)
        structured = validate_or_repair_response(raw_text)
        usage = data.get("usage", {})
        p_tokens = usage.get("prompt_tokens", self._estimate_tokens(user_prompt))
        c_tokens = usage.get("completion_tokens", self._estimate_tokens(raw_text))

        return LLMGenerationResult(
            raw_content=raw_text,
            structured_data=structured_dict,
            structured_response=structured,
            provider_used="vllm",
            model_name=settings.vllm_model,
            latency_ms=round(latency, 2),
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens
        )

    def _call_mock_engine(self, user_prompt: str, system_prompt: str) -> LLMGenerationResult:
        """
        High-fidelity Deterministic Mock Engine.
        Accurately synthesizes multi-step reasoning turns for Coordinator, Investigator,
        Verifier, and baseline agents. Guarantees 100% test coverage without external API keys.
        """
        t0 = time.perf_counter()
        p_tokens = self._estimate_tokens(user_prompt + system_prompt)
        q_lower = user_prompt.lower()

        # Extract actual user query from prompt
        q_match = re.search(r"user query:\s*([^\n]+)", q_lower)
        actual_query = q_match.group(1).strip() if q_match else q_lower

        # Check for Order ID
        m = re.search(r"\b(ORD-\d{4,6})\b", user_prompt, re.IGNORECASE)
        found_order_id = m.group(1).upper() if m else None

        # 1. Coordinator Step Evaluation Mock
        if "Senior Customer Resolution Coordinator" in system_prompt or "evaluate the evidence gathered so far" in q_lower:
            iteration_match = re.search(r"iteration:\s*(\d+)\s*of\s*(\d+)", q_lower)
            cur_iter = int(iteration_match.group(1)) if iteration_match else 1
            max_iter = int(iteration_match.group(2)) if iteration_match else 4

            # Check for system alert (tool failure in scratchpad)
            has_tool_alert = "system alert" in q_lower or "failed upstream" in q_lower

            if has_tool_alert:
                data = {
                    "thought": "Upstream operational tool failed. Graceful degradation triggered; finalizing with policy protection reassurance.",
                    "action": "FINALIZE",
                    "target_tool_or_query": "Graceful failure resolution",
                    "is_finished": True
                }
            elif any(k in actual_query for k in ["escalate", "human", "supervisor", "fraud", "lawyer", "manager"]):
                data = {
                    "thought": "Customer request involves legal threat, fraud, or explicit human supervisor escalation.",
                    "action": "ESCALATE",
                    "target_tool_or_query": f"Order {found_order_id or 'General'}: customer requested supervisor escalation",
                    "is_finished": True
                }
            elif not found_order_id and any(k in actual_query for k in ["my order", "purchase", "package", "cancel", "return", "refund", "track"]):
                data = {
                    "thought": "Customer inquiry concerns order-specific actions (cancel/return/track), but no Order ID was supplied. Stopping condition triggered: ask for clarification.",
                    "action": "ASK_CLARIFICATION",
                    "target_tool_or_query": "Could you please provide your Order ID (e.g. ORD-1001) so I can look into this for you?",
                    "is_finished": True
                }
            elif cur_iter == 1:
                data = {
                    "thought": f"Iteration 1: Need to retrieve base fulfillment and status record for {found_order_id} using Investigator.",
                    "action": "INVESTIGATE",
                    "target_tool_or_query": "check_order_status",
                    "is_finished": False
                }
            elif cur_iter == 2:
                if any(k in actual_query for k in ["cancel", "cancellation"]):
                    data = {
                        "thought": f"Order status gathered. Now delegating to Investigator to calculate cancellation penalty fee for {found_order_id}.",
                        "action": "INVESTIGATE",
                        "target_tool_or_query": "calculate_cancellation_fee",
                        "is_finished": False
                    }
                elif any(k in actual_query for k in ["return", "refund", "damage", "broken", "defective", "cracked"]):
                    data = {
                        "thought": f"Order is delivered. Delegating to Investigator to check refund eligibility with damage waiver for {found_order_id}.",
                        "action": "INVESTIGATE",
                        "target_tool_or_query": "check_refund_eligibility",
                        "is_finished": False
                    }
                elif any(k in actual_query for k in ["courier", "carrier", "live courier", "fedex", "ups"]):
                    data = {
                        "thought": f"Order status retrieved. Now delegating to Investigator for cross-source carrier logistics tracking verification for {found_order_id}.",
                        "action": "INVESTIGATE",
                        "target_tool_or_query": "verify_courier_tracking",
                        "is_finished": False
                    }
                else:
                    data = {
                        "thought": f"All required order tracking facts for {found_order_id} are verified. Formulating final response.",
                        "action": "FINALIZE",
                        "target_tool_or_query": "Delivery update formulation",
                        "is_finished": True
                    }
            elif cur_iter == 3:
                if any(k in actual_query for k in ["damage", "broken", "defective", "cracked", "fee waiver"]):
                    data = {
                        "thought": "Investigator provided refund data. Now delegating to Policy Verifier to audit damage fee waiver exception under Return Policy.",
                        "action": "VERIFY_POLICY",
                        "target_tool_or_query": "Return & Refund Policy Section 4 damaged goods fee waiver",
                        "is_finished": False
                    }
                else:
                    data = {
                        "thought": "All multi-step facts and calculations collected and cross-verified. Ready to finalize.",
                        "action": "FINALIZE",
                        "target_tool_or_query": "Final dispute settlement",
                        "is_finished": True
                    }
            else:
                data = {
                    "thought": "Comprehensive evidence gathered from Investigator and Verifier. Formulating authoritative customer resolution.",
                    "action": "FINALIZE",
                    "target_tool_or_query": "Final resolution formulation",
                    "is_finished": True
                }
            raw_text = json.dumps(data, indent=2)

        # 2. Investigator Tool Selector Mock
        elif "Tactical Operations & Fulfillment Investigator" in system_prompt or "Task from Coordinator" in user_prompt:
            task_match = re.search(r"task from coordinator:\s*([^\n]+)", q_lower)
            task_text = task_match.group(1).strip() if task_match else q_lower

            if any(k in task_text for k in ["cancellation_fee", "cancel", "penalty"]):
                tool_name = "calculate_cancellation_fee"
                args = {"order_id": found_order_id or "ORD-1001"}
            elif any(k in task_text for k in ["refund", "return", "damage", "defective", "cracked"]):
                tool_name = "check_refund_eligibility"
                args = {
                    "order_id": found_order_id or "ORD-1003",
                    "damage_reported": any(k in actual_query for k in ["damage", "defect", "broken", "cracked"]),
                    "item_opened": True
                }
            elif any(k in task_text for k in ["courier", "carrier", "tracking", "fedex", "ups"]):
                tool_name = "verify_courier_tracking"
                args = {"order_id": found_order_id or "ORD-1001"}
            elif "escalate" in task_text:
                tool_name = "escalate_to_human"
                args = {"order_id": found_order_id, "issue_summary": "Escalation requested", "urgency": "high"}
            else:
                tool_name = "check_order_status"
                args = {"order_id": found_order_id or "ORD-1001"}

            data = {
                "thought": f"Selecting {tool_name} with target arguments for order {found_order_id}.",
                "tool_name": tool_name,
                "arguments": args
            }
            raw_text = json.dumps(data, indent=2)

        # 3. Verifier Compliance Auditor Mock
        elif "Corporate Policy Compliance Auditor" in system_prompt or "Investigator Facts Gathered" in user_prompt:
            has_damage = any(k in q_lower for k in ["damage", "defective", "broken"])
            is_cancel = any(k in q_lower for k in ["cancel", "hours"])
            
            if has_damage:
                data = {
                    "thought": "Customer reported damaged product. Under corporate Return & Refund policy Section 4, standard return label deduction ($5.99) is WAIVED for damaged/defective merchandise.",
                    "applicable_policy_names": ["Return & Refund Policy - Section 4: Defective & Damaged Goods"],
                    "fee_waiver_applicable": True,
                    "waiver_reason": "Damaged on arrival exception grants 100% full refund with prepaid return shipping waiver.",
                    "policy_compliant": True,
                    "audit_summary": "Policy verified: Eligible for $0 fee return and full reimbursement."
                }
            elif is_cancel:
                data = {
                    "thought": "Auditing cancellation request. Checking elapsed time against tiered schedule: 0-1h Grace ($0), 1-6h Staging ($5), 6-24h Boxing ($15).",
                    "applicable_policy_names": ["Cancellation Policy - Tiered Penalty Schedule"],
                    "fee_waiver_applicable": False,
                    "waiver_reason": "Standard tier applies based on elapsed hours.",
                    "policy_compliant": True,
                    "audit_summary": "Standard cancellation tiers validated."
                }
            else:
                data = {
                    "thought": "Standard delivery tracking complies with Courier Service Level Agreements.",
                    "applicable_policy_names": ["Shipping & Delivery Guide"],
                    "fee_waiver_applicable": False,
                    "waiver_reason": "N/A",
                    "policy_compliant": True,
                    "audit_summary": "All delivery terms compliant."
                }
            raw_text = json.dumps(data, indent=2)

        # 4. Final Response Synthesis / General Assistant Mock
        else:
            final = (
                f"Thank you for contacting ShopAssist AI. Based on our verification for order **{found_order_id or 'your inquiry'}**, "
                "we have reviewed your request against our corporate service policies. "
                "If you need further assistance or wish to proceed with a return or cancellation, we are here to support you."
            )
            data = {
                "intent": "ORDER",
                "confidence": 0.95,
                "thought_process": "Synthesizing comprehensive response grounded in verified facts.",
                "final_response": final,
                "escalate_to_human": False,
                "action_taken": "Agentic Synthesis"
            }
            raw_text = json.dumps(data, indent=2)

        c_tokens = self._estimate_tokens(raw_text)
        latency = (time.perf_counter() - t0) * 1000.0

        structured_dict = extract_json_from_text(raw_text)
        structured_resp = validate_or_repair_response(raw_text)

        return LLMGenerationResult(
            raw_content=raw_text,
            structured_data=structured_dict,
            structured_response=structured_resp,
            provider_used="mock",
            model_name="deterministic-agentic-engine-v2",
            latency_ms=round(latency, 2),
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens
        )

llm_client = MultiProviderLLMClient()
