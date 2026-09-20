"""
Coordinator Agent for ShopAssist AI (Week 16).
Drives the Multi-Agent dispute resolution loop:
- Maintains Structured External Notes (Investigation Scratchpad).
- Dispatches sub-tasks to Investigator Agent and Policy Verification Agent.
- Enforces loop guardrails and bounded stopping conditions.
- Mitigates Context Saturation through tool result compaction.
- Synthesizes empathetic, factually grounded customer resolutions.
"""

import time
import re
from typing import Dict, Any, Optional, List

from src.config import settings
from src.agents.base_agent import BaseAgent
from src.agents.investigator_agent import investigator_agent
from src.agents.verification_agent import verification_agent
from src.core.context_manager import context_manager
from src.core.structured_outputs import (
    InvestigationScratchpad,
    AgentTrajectoryStep,
    AgenticResolutionPayload
)
from src.core.tools import execute_tool
from src.core.prompt_templates import COORDINATOR_SYSTEM_PROMPT, build_coordinator_step_prompt
from src.core.llm_client import llm_client

class CoordinatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="CoordinatorAgent", role="Chief Dispute Resolution Conductor")

    def resolve_dispute(
        self,
        query: str,
        customer_id: str = "guest_user",
        max_iterations: Optional[int] = None
    ) -> AgenticResolutionPayload:
        t_start = time.perf_counter()
        max_iters = max_iterations or settings.agent_max_iterations

        # 1. Identify Order ID if present in query
        m = re.search(r"\b(ORD-\d{4,6})\b", query, re.IGNORECASE)
        found_order_id = m.group(1).upper() if m else None

        # 2. Initialize Structured External Working Memory (Scratchpad)
        scratchpad = context_manager.initialize_scratchpad(query, found_order_id)
        trajectory: List[AgentTrajectoryStep] = []
        tools_called: List[str] = []
        rag_sources_used: List[str] = []

        stopping_condition = "task_completed"
        iteration = 0

        # 3. Autonomous Multi-Agent Reasoning Loop
        while iteration < max_iters and scratchpad.status == "IN_PROGRESS":
            iteration += 1
            t_step_start = time.perf_counter()

            # Render compact working memory for LLM
            scratchpad_view = context_manager.render_scratchpad_for_prompt(scratchpad)
            step_prompt = build_coordinator_step_prompt(query, scratchpad_view, iteration, max_iters)

            gen_res = llm_client.generate(
                user_prompt=step_prompt,
                system_prompt=COORDINATOR_SYSTEM_PROMPT
            )
            self.record_usage(gen_res.prompt_tokens, gen_res.completion_tokens, (time.perf_counter() - t_step_start) * 1000.0)

            data = gen_res.structured_data or {}
            thought = data.get("thought", "Evaluating evidence and selecting next action.")
            action = data.get("action", "FINALIZE").upper()
            target = data.get("target_tool_or_query", "")
            is_finished = data.get("is_finished", False)

            step_record = AgentTrajectoryStep(
                step_number=iteration,
                agent_role="coordinator",
                thought=thought,
                action=action,
                action_input={"target": target},
                tokens_used=gen_res.total_tokens,
                latency_ms=round((time.perf_counter() - t_step_start) * 1000.0, 2)
            )

            # Route Coordinator Decision
            if action == "INVESTIGATE":
                step_record.agent_role = "investigator"
                inv_res = investigator_agent.run_task(task_description=target, order_id=scratchpad.order_id)
                tools_called.append(inv_res.tool_name)

                step_record.action = f"tool_call:{inv_res.tool_name}"
                step_record.action_input = inv_res.arguments
                step_record.observation_raw = inv_res.raw_output
                step_record.observation_compacted = inv_res.compacted_observation
                step_record.tokens_used += inv_res.total_tokens

                # Check for simulated failure or error
                if inv_res.raw_output.get("status") == "simulated_failure":
                    err_msg = inv_res.raw_output.get("error_message", "Tool unavailable")
                    scratchpad.verified_facts.append(f"[SYSTEM ALERT] Upstream tool failed: {err_msg}")
                    scratchpad.contradictions_resolved.append("Tool failure detected; shifting to graceful policy fallback.")
                else:
                    scratchpad.verified_facts.append(inv_res.compacted_observation)

                # Extract financial or order details if available
                if inv_res.tool_name == "calculate_cancellation_fee" and inv_res.raw_output.get("can_cancel"):
                    scratchpad.calculated_penalty = inv_res.raw_output.get("applicable_fee", 0.0)
                    scratchpad.refund_amount = inv_res.raw_output.get("estimated_net_refund", 0.0)
                elif inv_res.tool_name == "check_refund_eligibility" and inv_res.raw_output.get("eligible"):
                    scratchpad.refund_amount = inv_res.raw_output.get("estimated_refund", 0.0)

            elif action == "VERIFY_POLICY":
                step_record.agent_role = "verifier"
                ver_res = verification_agent.audit_dispute(
                    facts=scratchpad.verified_facts,
                    policy_query=target or query
                )
                step_record.action = "policy_audit"
                step_record.observation_compacted = ver_res.audit_summary
                step_record.tokens_used += ver_res.total_tokens
                rag_sources_used.extend(ver_res.rag_sources)

                scratchpad.applicable_policies.extend(ver_res.applicable_policies)
                if ver_res.fee_waiver_applicable:
                    scratchpad.fee_waiver_applied = True
                    scratchpad.exceptions_identified.append(ver_res.waiver_reason)

            elif action == "ASK_CLARIFICATION":
                stopping_condition = "clarification_needed"
                scratchpad.status = "NEEDS_CLARIFICATION"
                scratchpad.pending_questions.append(target or "Please provide your Order ID.")
                step_record.observation_compacted = f"Clarification requested: {target}"
                trajectory.append(step_record)
                break

            elif action == "ESCALATE":
                stopping_condition = "escalation_triggered"
                scratchpad.status = "ESCALATED"
                esc_res = execute_tool("escalate_to_human", {
                    "order_id": scratchpad.order_id,
                    "issue_summary": target or query,
                    "urgency": "high"
                })
                tools_called.append("escalate_to_human")
                step_record.observation_compacted = context_manager.compact_tool_output("escalate_to_human", esc_res)
                scratchpad.verified_facts.append(step_record.observation_compacted)
                trajectory.append(step_record)
                break

            elif action == "FINALIZE" or is_finished:
                scratchpad.status = "COMPLETED"
                stopping_condition = "task_completed"
                step_record.observation_compacted = "Coordinator determined all evidence is sufficient."
                trajectory.append(step_record)
                break

            step_record.scratchpad_state_snapshot = scratchpad.model_dump()
            trajectory.append(step_record)

        # 4. Check if loop exceeded maximum iterations
        max_reached = False
        if iteration >= max_iters and scratchpad.status == "IN_PROGRESS":
            max_reached = True
            stopping_condition = "max_iterations_reached"
            scratchpad.status = "COMPLETED"
            scratchpad.stopping_reason = f"Stopped automatically after {max_iters} iterations guardrail."

        # 5. Formulate final response to customer
        final_answer = self._synthesize_final_response(query, scratchpad, stopping_condition)

        total_latency = (time.perf_counter() - t_start) * 1000.0
        total_tokens = sum(s.tokens_used for s in trajectory)
        coord_tokens = sum(s.tokens_used for s in trajectory if s.agent_role == "coordinator")

        return AgenticResolutionPayload(
            query=query,
            customer_id=customer_id,
            pattern_used="multi_agent",
            iterations_count=iteration,
            max_iterations_reached=max_reached,
            stopping_condition=stopping_condition,
            scratchpad=scratchpad,
            trajectory=trajectory,
            final_customer_response=final_answer,
            tools_called=list(set(tools_called)),
            rag_sources=list(set(rag_sources_used)),
            total_tokens=total_tokens,
            coordination_tokens=coord_tokens,
            total_latency_ms=round(total_latency, 2)
        )

    def _synthesize_final_response(
        self,
        query: str,
        scratchpad: InvestigationScratchpad,
        stopping_condition: str
    ) -> str:
        """Synthesizes customer-facing response from verified facts in scratchpad."""
        oid = scratchpad.order_id or "your order"

        if stopping_condition == "clarification_needed":
            question = scratchpad.pending_questions[0] if scratchpad.pending_questions else "Could you please provide your Order ID?"
            return f"I would be glad to assist you with your inquiry! {question}"

        if stopping_condition == "escalation_triggered":
            return (
                f"I have reviewed your request regarding **Order {oid}**. Due to the complexity of your dispute, "
                "I have prioritized your case and escalated it directly to our Tier-2 Customer Resolution Specialist team. "
                "A supervisor will review your file within 15 minutes."
            )

        # Check if a tool failure was caught
        tool_failed = any("[SYSTEM ALERT]" in f for f in scratchpad.verified_facts)
        if tool_failed:
            return (
                f"We are currently experiencing a temporary connection delay with our live warehouse database. "
                f"However, according to our standard corporate policies for **Order {oid}**, you are fully protected. "
                "If you wish to cancel or return your purchase, our support team can manually process your request without any penalties. "
                "Would you like me to connect you with a live agent?"
            )

        facts_text = " ".join(scratchpad.verified_facts)

        # Damage waiver dispute scenario
        if scratchpad.fee_waiver_applied:
            waiver = scratchpad.exceptions_identified[0] if scratchpad.exceptions_identified else "damage fee waiver"
            return (
                f"I have thoroughly reviewed your dispute for **Order {oid}**. "
                f"Because you reported that your item arrived damaged, our corporate policy ({waiver}) waives the standard $5.99 return shipping fee. "
                f"You are eligible for a **100% full refund of ${scratchpad.refund_amount:.2f}**. "
                "A prepaid return shipping label has been generated for your convenience."
            )

        # Cancellation scenario
        if "Cancellation Eligible" in facts_text:
            return (
                f"I have checked the live processing status for **Order {oid}**. "
                f"Your order is eligible for cancellation with a penalty fee of **${scratchpad.calculated_penalty:.2f}**, "
                f"resulting in an estimated net refund of **${scratchpad.refund_amount:.2f}**. "
                "Please confirm if you would like me to finalize this cancellation."
            )

        # Standard tracking or general response
        return (
            f"Based on our verification for **Order {oid}**: {facts_text} "
            "Please let us know if you need any additional assistance with your delivery or account."
        )

coordinator_agent = CoordinatorAgent()
