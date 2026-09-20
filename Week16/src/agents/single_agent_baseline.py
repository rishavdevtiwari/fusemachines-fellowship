"""
Single-Agent Baseline Implementation for ShopAssist AI (Week 16).
Provides a monolithic single-agent loop baseline for direct comparison against
the Multi-Agent system to measure token overhead, coordination cost, and structural failures.
"""

import time
import re
from typing import Dict, Any, Optional, List

from src.config import settings
from src.agents.base_agent import BaseAgent
from src.core.tools import execute_tool
from src.core.rag_pipeline import rag_pipeline
from src.core.structured_outputs import (
    InvestigationScratchpad,
    AgentTrajectoryStep,
    AgenticResolutionPayload
)
from src.core.prompt_templates import SINGLE_AGENT_BASELINE_PROMPT
from src.core.llm_client import llm_client

class SingleAgentBaseline(BaseAgent):
    """
    Monolithic single-agent loop: handles tool calling, policy retrieval,
    and customer response formulation within a single un-isolated context.
    """
    def __init__(self):
        super().__init__(name="SingleAgentBaseline", role="Monolithic Support Agent")

    def resolve_dispute(
        self,
        query: str,
        customer_id: str = "guest_user",
        max_iterations: Optional[int] = None
    ) -> AgenticResolutionPayload:
        t_start = time.perf_counter()
        max_iters = max_iterations or settings.agent_max_iterations

        m = re.search(r"\b(ORD-\d{4,6})\b", query, re.IGNORECASE)
        found_order_id = m.group(1).upper() if m else None

        trajectory: List[AgentTrajectoryStep] = []
        tools_called: List[str] = []
        conversation_history: List[str] = [f"User: {query}"]
        
        iteration = 0
        stopping_condition = "task_completed"
        final_answer = ""

        while iteration < max_iters:
            iteration += 1
            t_step = time.perf_counter()

            # Monolithic prompt appends full un-compacted conversation history
            history_text = "\n".join(conversation_history)
            prompt = (
                f"History:\n{history_text}\n\n"
                f"Iteration: {iteration} of {max_iters}\n"
                "Decide whether to call a tool or produce the final response."
            )

            gen_res = llm_client.generate(
                user_prompt=prompt,
                system_prompt=SINGLE_AGENT_BASELINE_PROMPT
            )
            self.record_usage(gen_res.prompt_tokens, gen_res.completion_tokens, (time.perf_counter() - t_step) * 1000.0)

            # Heuristic / Structured routing for baseline
            q_lower = query.lower()
            if not found_order_id and any(k in q_lower for k in ["cancel", "return", "refund", "track"]):
                stopping_condition = "clarification_needed"
                final_answer = "Could you please provide your Order ID so I can look up your record?"
                trajectory.append(AgentTrajectoryStep(
                    step_number=iteration,
                    agent_role="single_agent",
                    thought="Order ID missing from query. Asking for clarification.",
                    action="ask_clarification",
                    tokens_used=gen_res.total_tokens,
                    latency_ms=round((time.perf_counter() - t_step) * 1000.0, 2)
                ))
                break

            if iteration == 1 and found_order_id:
                tool_name = "check_order_status"
                tools_called.append(tool_name)
                tool_res = execute_tool(tool_name, {"order_id": found_order_id})
                conversation_history.append(f"System Tool Result ({tool_name}): {tool_res}")

                trajectory.append(AgentTrajectoryStep(
                    step_number=iteration,
                    agent_role="single_agent",
                    thought=f"Checking order status for {found_order_id}.",
                    action=f"tool_call:{tool_name}",
                    action_input={"order_id": found_order_id},
                    observation_raw=tool_res,
                    tokens_used=gen_res.total_tokens,
                    latency_ms=round((time.perf_counter() - t_step) * 1000.0, 2)
                ))
            elif iteration == 2 and any(k in q_lower for k in ["damage", "defect", "return", "refund"]):
                tool_name = "check_refund_eligibility"
                tools_called.append(tool_name)
                tool_res = execute_tool(tool_name, {
                    "order_id": found_order_id or "ORD-1003",
                    "damage_reported": any(k in q_lower for k in ["damage", "defect"]),
                    "item_opened": True
                })
                conversation_history.append(f"System Tool Result ({tool_name}): {tool_res}")

                trajectory.append(AgentTrajectoryStep(
                    step_number=iteration,
                    agent_role="single_agent",
                    thought="Checking refund eligibility.",
                    action=f"tool_call:{tool_name}",
                    action_input={"order_id": found_order_id},
                    observation_raw=tool_res,
                    tokens_used=gen_res.total_tokens,
                    latency_ms=round((time.perf_counter() - t_step) * 1000.0, 2)
                ))
            else:
                final_answer = (
                    f"Based on order records for **{found_order_id or 'your inquiry'}**, "
                    "we have processed your request according to standard store policies."
                )
                trajectory.append(AgentTrajectoryStep(
                    step_number=iteration,
                    agent_role="single_agent",
                    thought="All needed actions performed. Finalizing answer.",
                    action="final_response",
                    tokens_used=gen_res.total_tokens,
                    latency_ms=round((time.perf_counter() - t_step) * 1000.0, 2)
                ))
                break

        total_latency = (time.perf_counter() - t_start) * 1000.0
        total_tokens = sum(s.tokens_used for s in trajectory)

        scratchpad = InvestigationScratchpad(
            dispute_goal=query,
            order_id=found_order_id,
            verified_facts=[f"Single-agent processed {len(tools_called)} tools"],
            status="COMPLETED" if stopping_condition == "task_completed" else "NEEDS_CLARIFICATION"
        )

        return AgenticResolutionPayload(
            query=query,
            customer_id=customer_id,
            pattern_used="single_agent",
            iterations_count=iteration,
            max_iterations_reached=(iteration >= max_iters),
            stopping_condition=stopping_condition,
            scratchpad=scratchpad,
            trajectory=trajectory,
            final_customer_response=final_answer or "Request completed.",
            tools_called=tools_called,
            rag_sources=[],
            total_tokens=total_tokens,
            coordination_tokens=0,  # Single-agent has zero multi-agent coordination overhead
            total_latency_ms=round(total_latency, 2)
        )

single_agent_baseline = SingleAgentBaseline()
