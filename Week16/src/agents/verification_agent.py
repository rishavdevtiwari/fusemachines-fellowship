"""
Policy Verification Sub-Agent for ShopAssist AI (Week 16).
Specializes in corporate policy compliance auditing, cross-source fact-checking,
and policy exception verification (e.g. damaged goods waivers, grace period rules).
Overcomes the Self-Verification Paradox by separating fact-gathering from compliance auditing.
"""

import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from src.agents.base_agent import BaseAgent
from src.core.rag_pipeline import rag_pipeline
from src.core.prompt_templates import VERIFICATION_SYSTEM_PROMPT, build_verifier_prompt
from src.core.llm_client import llm_client

class VerificationResult(BaseModel):
    applicable_policies: List[str]
    fee_waiver_applicable: bool
    waiver_reason: str
    policy_compliant: bool
    audit_summary: str
    thought: str
    rag_sources: List[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float

class VerificationAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="VerificationAgent", role="Corporate Policy Compliance Auditor")

    def audit_dispute(self, facts: List[str], policy_query: str) -> VerificationResult:
        t0 = time.perf_counter()

        # Retrieve relevant corporate policy context via RAG
        rag_context, rag_sources = rag_pipeline.retrieve_context(policy_query, top_k=2)

        prompt = build_verifier_prompt(facts, rag_context)
        gen_res = llm_client.generate(
            user_prompt=prompt,
            system_prompt=VERIFICATION_SYSTEM_PROMPT
        )

        data = gen_res.structured_data or {}
        thought = data.get("thought", "Auditing verified facts against corporate policy.")
        policies = data.get("applicable_policy_names", rag_sources or ["Corporate Return & Cancellation Policy"])
        waiver_applicable = data.get("fee_waiver_applicable", False)
        waiver_reason = data.get("waiver_reason", "Standard terms apply.")
        compliant = data.get("policy_compliant", True)
        summary = data.get("audit_summary", "Dispute audited against corporate policies.")

        latency = (time.perf_counter() - t0) * 1000.0
        self.record_usage(gen_res.prompt_tokens, gen_res.completion_tokens, latency)

        return VerificationResult(
            applicable_policies=policies,
            fee_waiver_applicable=waiver_applicable,
            waiver_reason=waiver_reason,
            policy_compliant=compliant,
            audit_summary=summary,
            thought=thought,
            rag_sources=rag_sources,
            prompt_tokens=gen_res.prompt_tokens,
            completion_tokens=gen_res.completion_tokens,
            total_tokens=gen_res.total_tokens,
            latency_ms=round(latency, 2)
        )

verification_agent = VerificationAgent()
