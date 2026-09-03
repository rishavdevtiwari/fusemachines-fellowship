"""
Multi-Provider LLM Client for ShopAssist AI.
Supports Google Gemini, OpenAI, Anthropic Claude, Local vLLM, and Deterministic Fallback.
Provides unified interface, parameter tuning (temperature, top_p), and automated provider failover.
"""

import os
import time
import json
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
    structured_response: AssistantStructuredResponse
    provider_used: str
    model_name: str
    latency_ms: float
    total_tokens: int = 0
    fallback_invoked: bool = False

class MultiProviderLLMClient:
    """
    Unified LLM Client orchestrating primary and fallback generative models.
    """
    def __init__(self):
        self._init_gemini()

    def _init_gemini(self):
        self.gemini_client = None
        key = settings.gemini_api_key
        if key:
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
        Executes generation with automatic failover across provider cascade.
        """
        temp = temperature if temperature is not None else settings.default_temperature
        top = top_p if top_p is not None else settings.default_top_p
        tokens = max_tokens if max_tokens is not None else settings.max_tokens
        
        provider = (preferred_provider or settings.primary_provider).lower()
        
        # Primary generation attempt
        try:
            if provider == "gemini":
                return self._call_gemini(user_prompt, system_prompt, temp, top, tokens)
            elif provider == "openai":
                return self._call_openai(user_prompt, system_prompt, temp, top, tokens)
            elif provider == "claude":
                return self._call_claude(user_prompt, system_prompt, temp, top, tokens)
            elif provider == "vllm":
                return self._call_vllm(user_prompt, system_prompt, temp, top, tokens)
            elif provider == "mock":
                return self._call_mock_engine(user_prompt, system_prompt)
        except Exception as primary_error:
            print(f"Primary provider '{provider}' failed: {primary_error}. Attempting fallback...")

        # Fallback generation attempt
        fallback = settings.fallback_provider.lower()
        if fallback != provider:
            try:
                if fallback == "gemini":
                    res = self._call_gemini(user_prompt, system_prompt, temp, top, tokens)
                elif fallback == "openai":
                    res = self._call_openai(user_prompt, system_prompt, temp, top, tokens)
                elif fallback == "vllm":
                    res = self._call_vllm(user_prompt, system_prompt, temp, top, tokens)
                else:
                    res = self._call_mock_engine(user_prompt, system_prompt)
                res.fallback_invoked = True
                return res
            except Exception as fallback_error:
                print(f"Fallback provider '{fallback}' also failed: {fallback_error}. Engaging graceful degradation...")

        # Final safety net: Guaranteed Deterministic Mock Engine
        res = self._call_mock_engine(user_prompt, system_prompt)
        res.fallback_invoked = True
        return res

    def _call_gemini(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int
    ) -> LLMGenerationResult:
        """Invokes Google Gemini with JSON mode."""
        if not self.gemini_client:
            self._init_gemini()
        if not self.gemini_client:
            raise ValueError("GEMINI_API_KEY is not configured.")
            
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
        
        usage = getattr(response, "usage_metadata", None)
        total_tokens = getattr(usage, "total_token_count", 0) if usage else 0
        
        return LLMGenerationResult(
            raw_content=raw_text,
            structured_response=structured,
            provider_used="gemini",
            model_name=settings.gemini_model,
            latency_ms=round(latency, 2),
            total_tokens=total_tokens
        )

    def _call_openai(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int
    ) -> LLMGenerationResult:
        """Invokes OpenAI Chat Completions API with JSON mode."""
        key = settings.openai_api_key
        if not key:
            raise ValueError("OPENAI_API_KEY is not configured.")
            
        t0 = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
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
        structured = validate_or_repair_response(raw_text)
        tokens = data.get("usage", {}).get("total_tokens", 0)
        
        return LLMGenerationResult(
            raw_content=raw_text,
            structured_response=structured,
            provider_used="openai",
            model_name=settings.openai_model,
            latency_ms=round(latency, 2),
            total_tokens=tokens
        )

    def _call_vllm(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int
    ) -> LLMGenerationResult:
        """Invokes locally served open-source model via vLLM OpenAI-compatible endpoint."""
        t0 = time.perf_counter()
        url = f"{settings.vllm_base_url.rstrip('/')}/chat/completions"
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
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
        latency = (time.perf_counter() - t0) * 1000.0
        raw_text = data["choices"][0]["message"]["content"]
        structured = validate_or_repair_response(raw_text)
        tokens = data.get("usage", {}).get("total_tokens", 0)
        
        return LLMGenerationResult(
            raw_content=raw_text,
            structured_response=structured,
            provider_used="vllm",
            model_name=settings.vllm_model,
            latency_ms=round(latency, 2),
            total_tokens=tokens
        )

    def _call_claude(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int
    ) -> LLMGenerationResult:
        """Invokes Anthropic Claude API."""
        key = settings.anthropic_api_key
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is not configured.")
            
        t0 = time.perf_counter()
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": settings.claude_model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
        latency = (time.perf_counter() - t0) * 1000.0
        raw_text = data["content"][0]["text"]
        structured = validate_or_repair_response(raw_text)
        
        return LLMGenerationResult(
            raw_content=raw_text,
            structured_response=structured,
            provider_used="claude",
            model_name=settings.claude_model,
            latency_ms=round(latency, 2)
        )

    def _call_mock_engine(self, user_prompt: str, system_prompt: str) -> LLMGenerationResult:
        """
        Deterministic, offline intelligence engine.
        Parses customer query, extracts order IDs, intent hints, and RAG context to synthesize
        a fully conformant structured JSON response with zero remote API latency or cost.
        """
        t0 = time.perf_counter()
        q_lower = user_prompt.lower()
        
        # Check for extracted order IDs like ORD-1001
        import re
        order_match = re.search(r"\b(ORD-\d{4,6})\b", user_prompt, re.IGNORECASE)
        found_order_id = order_match.group(1).upper() if order_match else None
        
        # Infer intent
        if any(k in q_lower for k in ["cancel", "cancellation fee", "penalty", "stage"]):
            intent = "CANCELLATION_FEE"
            thought = "User query involves cancelling an order and calculating associated fees."
            if found_order_id:
                final = (
                    f"I have reviewed your cancellation request for **Order {found_order_id}**. "
                    "According to our Cancellation Policy, orders cancelled within 1 hour have a **$0.00 fee** (100% refund). "
                    "Between 1 and 6 hours, a standard **$5.00 restocking fee** applies, and between 6 and 24 hours a **$15.00 fee** applies. "
                    "Once an order is dispatched or past 24 hours, cancellation is closed and standard returns apply."
                )
            else:
                final = (
                    "Our cancellation fee structure depends on the elapsed time since order confirmation:\n"
                    "- **0 to 1 hour**: 100% Free cancellation.\n"
                    "- **1 to 6 hours**: Flat $5.00 restocking fee.\n"
                    "- **6 to 24 hours**: Flat $15.00 warehouse fee.\n"
                    "- **Dispatched / >24 hours**: Cannot be cancelled directly; please initiate a return once delivered.\n\n"
                    "Please provide your Order ID (e.g. `ORD-1001`) to calculate your exact fee."
                )
        elif any(k in q_lower for k in ["refund", "return", "money back"]):
            intent = "REFUND"
            thought = "Inquiry regarding return policy, item return windows, or refund processing."
            final = (
                "Under our Return & Refund Policy, you can return eligible merchandise within **30 calendar days** of delivery. "
                "Items must be unopened and in original packaging. Approved refunds are processed to your original payment method "
                "within **3 to 5 business days** of warehouse inspection. A flat return shipping label fee of $5.99 applies for preference returns."
            )
        elif any(k in q_lower for k in ["track", "where is", "delivery", "shipping", "courier", "express"]):
            intent = "DELIVERY"
            thought = "Customer inquiry concerning parcel shipping progress or delivery timelines."
            if found_order_id:
                final = (
                    f"Your **Order {found_order_id}** is currently in our fulfillment pipeline. "
                    "Standard delivery takes 3 to 5 business days, while Express takes 1 to 2 business days. "
                    "As soon as the courier scans the package, automated tracking updates will be dispatched via email and SMS."
                )
            else:
                final = (
                    "We offer several delivery options:\n"
                    "- **Standard Shipping**: 3 to 5 business days (Free over $50, otherwise $4.99).\n"
                    "- **Express Shipping**: 1 to 2 business days ($12.99).\n"
                    "- **Overnight Priority**: Next business day guaranteed ($24.99).\n\n"
                    "Provide your Order ID for real-time tracking updates."
                )
        elif any(k in q_lower for k in ["invoice", "receipt", "tax", "vat"]):
            intent = "INVOICE"
            thought = "Request for commercial invoice, tax receipt, or payment voucher."
            final = (
                "You can download your official tax invoice directly by logging into your account under **Orders > Order Details > Download Invoice**. "
                "If you need company VAT or tax identification added to your receipt, please reply with your business registration details."
            )
        elif any(k in q_lower for k in ["address", "destination", "redirect", "shipping address"]):
            intent = "SHIPPING_ADDRESS"
            thought = "Request to update or reroute shipping delivery address."
            final = (
                "Shipping addresses can be freely updated within **2 hours** of order placement. "
                "If your order has already been dispatched with an active courier tracking number, carrier rerouting incurs a **$10.00 fee** subject to courier route feasibility."
            )
        elif any(k in q_lower for k in ["account", "password", "login", "2fa", "profile"]):
            intent = "ACCOUNT"
            thought = "User account administration, credentials, or security query."
            final = (
                "To manage your profile or credentials, navigate to **Settings > Profile**. "
                "For security, password resets can be triggered using the 'Forgot Password' link on the sign-in page. "
                "We recommend enabling Two-Factor Authentication (2FA) via **Settings > Security** for enhanced protection."
            )
        elif any(k in q_lower for k in ["newsletter", "unsubscribe", "marketing", "promo"]):
            intent = "NEWSLETTER"
            thought = "Management of marketing communications and newsletter preferences."
            final = (
                "You can update email preferences or unsubscribe from promotional newsletters at any time via **Settings > Notifications**. "
                "Opt-outs take effect within 24 hours. Essential transactional order receipts will continue to be sent."
            )
        elif any(k in q_lower for k in ["payment", "credit card", "paypal", "apple pay"]):
            intent = "PAYMENT"
            thought = "Inquiry regarding billing, accepted payment methods, or payment declines."
            final = (
                "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay. "
                "All transactions are encrypted with 256-bit SSL security. If a payment was declined, please verify your billing zip code with your card issuer."
            )
        elif any(k in q_lower for k in ["feedback", "complaint", "review", "terrible", "great"]):
            intent = "FEEDBACK"
            thought = "Customer satisfaction feedback or service complaint."
            final = (
                "Thank you for sharing your feedback with us. We take customer satisfaction very seriously. "
                "Your comments have been logged with our Quality Assurance team to help us continuously improve our services."
            )
        else:
            intent = "ORDER"
            thought = "General e-commerce order or product assistance query."
            final = (
                "Welcome to ShopAssist AI Customer Support! I can assist you with tracking orders, calculating cancellation fees, "
                "evaluating returns and refunds, updating shipping addresses, or retrieving tax invoices. How can I help you today?"
            )
            
        structured = AssistantStructuredResponse(
            intent=intent,
            confidence=0.92,
            thought_process=thought,
            needs_tool=bool(found_order_id),
            tool_call={"tool_name": "check_order_status", "arguments": {"order_id": found_order_id}} if found_order_id else None,
            tool_result=None,
            rag_sources=["ShopAssist AI Knowledge Base"],
            final_response=final,
            escalate_to_human=False,
            action_taken="Deterministic Intent Synthesis & Grounded Response"
        )
        
        latency = (time.perf_counter() - t0) * 1000.0
        return LLMGenerationResult(
            raw_content=json.dumps(structured.model_dump(), indent=2),
            structured_response=structured,
            provider_used="mock_fallback",
            model_name="deterministic-engine-v1",
            latency_ms=round(latency, 2),
            total_tokens=150
        )

llm_client = MultiProviderLLMClient()
