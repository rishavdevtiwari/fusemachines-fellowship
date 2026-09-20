"""
Investigator Sub-Agent for ShopAssist AI (Week 16).
Specializes in operational data gathering and function execution:
- Order fulfillment lookups
- Cancellation fee calculation
- Refund & return eligibility checks
- Carrier tracking verification
- Human escalation
Applies Context Engineering (Tool Result Compaction & Clearing) to prevent Context Saturation.
"""

import time
from typing import Dict, Any, Optional
from pydantic import BaseModel

from src.agents.base_agent import BaseAgent
from src.core.tools import execute_tool
from src.core.context_manager import context_manager
from src.core.prompt_templates import INVESTIGATOR_SYSTEM_PROMPT, build_investigator_prompt
from src.core.llm_client import llm_client

class InvestigatorResult(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    thought: str
    raw_output: Dict[str, Any]
    compacted_observation: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float

class InvestigatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="InvestigatorAgent", role="Operational Data Gathering Specialist")

    def run_task(self, task_description: str, order_id: Optional[str] = None) -> InvestigatorResult:
        t0 = time.perf_counter()
        clean_oid = (order_id or "ORD-1001").strip().upper()
        
        prompt = build_investigator_prompt(task_description, clean_oid)
        gen_res = llm_client.generate(
            user_prompt=prompt,
            system_prompt=INVESTIGATOR_SYSTEM_PROMPT
        )

        data = gen_res.structured_data or {}
        tool_name = data.get("tool_name", "check_order_status")
        arguments = data.get("arguments", {"order_id": clean_oid})
        thought = data.get("thought", f"Executing {tool_name} for order {clean_oid}")

        # Ensure order_id in arguments if not present
        if "order_id" in arguments and not arguments["order_id"]:
            arguments["order_id"] = clean_oid

        # Execute selected tool
        raw_output = execute_tool(tool_name, arguments)

        # Context Engineering: Compact raw output into high-signal assertion
        compacted = context_manager.compact_tool_output(tool_name, raw_output)

        latency = (time.perf_counter() - t0) * 1000.0
        self.record_usage(gen_res.prompt_tokens, gen_res.completion_tokens, latency)

        return InvestigatorResult(
            tool_name=tool_name,
            arguments=arguments,
            thought=thought,
            raw_output=raw_output,
            compacted_observation=compacted,
            prompt_tokens=gen_res.prompt_tokens,
            completion_tokens=gen_res.completion_tokens,
            total_tokens=gen_res.total_tokens,
            latency_ms=round(latency, 2)
        )

investigator_agent = InvestigatorAgent()
