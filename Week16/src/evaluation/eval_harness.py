"""
Custom Evaluation Harness Built from Scratch for ShopAssist AI Agent (Week 16).
Tests actual agentic behavior without external evaluation frameworks.
Measures:
1. Task Completion Rate
2. Tool-Call Correctness
3. Trajectory Length vs Complexity
4. Failure Classification (Hard, Soft, Cascading Soft)
5. Token & Cost Accounting (Multi-Agent vs. Single-Agent Baseline Comparison)
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from src.config import settings
from src.agents.coordinator_agent import coordinator_agent
from src.agents.single_agent_baseline import single_agent_baseline

class EvaluationCaseResult(BaseModel):
    case_id: str
    category: str
    query: str
    pattern: str
    completed: bool
    tool_correctness: bool
    trajectory_length: int
    tools_called: List[str]
    expected_tools: List[str]
    stopping_condition: str
    failure_type: Optional[str] = None  # "hard_failure", "soft_failure", "cascading_soft_failure", None
    failure_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    coordination_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0

class EvaluationSummary(BaseModel):
    total_cases: int
    completed_cases: int
    completion_rate_pct: float
    tool_correctness_rate_pct: float
    avg_trajectory_length: float
    hard_failures: int
    soft_failures: int
    cascading_soft_failures: int
    total_tokens: int
    avg_tokens_per_query: float
    avg_coordination_tokens: float
    total_estimated_cost_usd: float
    avg_latency_ms: float

class EvaluationHarness:
    """
    Independent Evaluation Harness for Agentic Systems.
    """
    def __init__(self, queries_path: Optional[Path] = None):
        self.queries_path = queries_path or (settings.base_dir / "src" / "evaluation" / "test_queries.json")
        self.test_cases = self._load_cases()

    def _load_cases(self) -> List[Dict[str, Any]]:
        if not self.queries_path.exists():
            return []
        try:
            with open(self.queries_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading test queries: {e}")
            return []

    def evaluate_case(self, case: Dict[str, Any], pattern: str = "multi_agent") -> EvaluationCaseResult:
        cid = case.get("id", "TC-00")
        query = case.get("query", "")
        expected_tools = case.get("expected_tools", [])
        expected_waiver = case.get("expected_waiver", False)
        expected_stop = case.get("expected_stopping_condition")
        max_acceptable = case.get("max_acceptable_steps", 4)

        t0 = time.perf_counter()
        failure_type: Optional[str] = None
        failure_reason: Optional[str] = None
        payload = None

        try:
            if pattern == "single_agent":
                payload = single_agent_baseline.resolve_dispute(query)
            else:
                payload = coordinator_agent.resolve_dispute(query)
        except Exception as e:
            failure_type = "hard_failure"
            failure_reason = f"Unhandled Exception / Crash: {str(e)}"

        latency = (time.perf_counter() - t0) * 1000.0

        if not payload:
            return EvaluationCaseResult(
                case_id=cid,
                category=case.get("category", "unknown"),
                query=query,
                pattern=pattern,
                completed=False,
                tool_correctness=False,
                trajectory_length=0,
                tools_called=[],
                expected_tools=expected_tools,
                stopping_condition="failed",
                failure_type="hard_failure",
                failure_reason=failure_reason,
                latency_ms=round(latency, 2)
            )

        # Evaluate Tool Call Correctness
        tools_called = payload.tools_called
        tool_correct = True
        for et in expected_tools:
            if et not in tools_called:
                tool_correct = False
                break

        # Evaluate Task Completion & Stopping Condition
        completed = True
        traj_len = payload.iterations_count

        if expected_stop and payload.stopping_condition != expected_stop:
            completed = False
            failure_type = "soft_failure"
            failure_reason = f"Expected stopping condition '{expected_stop}' but got '{payload.stopping_condition}'"
        elif not tool_correct and expected_tools:
            completed = False
            failure_type = "soft_failure"
            failure_reason = f"Tool selection mismatch: called {tools_called}, expected {expected_tools}"
        elif traj_len > max_acceptable:
            completed = False
            failure_type = "soft_failure"
            failure_reason = f"Trajectory length {traj_len} exceeded max acceptable steps {max_acceptable}"
        elif expected_waiver and not payload.scratchpad.fee_waiver_applied:
            completed = False
            failure_type = "cascading_soft_failure"
            failure_reason = "Policy auditor failed to recognize damage fee waiver exception; penalty miscalculated."

        # Token cost estimation ($0.15/1M tokens blending input/output)
        est_cost = (payload.total_tokens / 1_000_000.0) * 0.25

        return EvaluationCaseResult(
            case_id=cid,
            category=case.get("category", "unknown"),
            query=query,
            pattern=pattern,
            completed=completed,
            tool_correctness=tool_correct,
            trajectory_length=traj_len,
            tools_called=tools_called,
            expected_tools=expected_tools,
            stopping_condition=payload.stopping_condition,
            failure_type=failure_type,
            failure_reason=failure_reason,
            prompt_tokens=payload.total_tokens - (payload.total_tokens // 3),
            completion_tokens=payload.total_tokens // 3,
            total_tokens=payload.total_tokens,
            coordination_tokens=payload.coordination_tokens,
            estimated_cost_usd=round(est_cost, 6),
            latency_ms=round(latency, 2)
        )

    def run_suite(self, pattern: str = "multi_agent") -> Tuple[List[EvaluationCaseResult], EvaluationSummary]:
        results: List[EvaluationCaseResult] = []
        for case in self.test_cases:
            res = self.evaluate_case(case, pattern=pattern)
            results.append(res)

        total = len(results)
        completed = sum(1 for r in results if r.completed)
        tools_ok = sum(1 for r in results if r.tool_correctness)
        hard_fails = sum(1 for r in results if r.failure_type == "hard_failure")
        soft_fails = sum(1 for r in results if r.failure_type == "soft_failure")
        cascading_fails = sum(1 for r in results if r.failure_type == "cascading_soft_failure")

        avg_traj = sum(r.trajectory_length for r in results) / total if total > 0 else 0.0
        tot_toks = sum(r.total_tokens for r in results)
        avg_toks = tot_toks / total if total > 0 else 0.0
        avg_coord = sum(r.coordination_tokens for r in results) / total if total > 0 else 0.0
        tot_cost = sum(r.estimated_cost_usd for r in results)
        avg_lat = sum(r.latency_ms for r in results) / total if total > 0 else 0.0

        summary = EvaluationSummary(
            total_cases=total,
            completed_cases=completed,
            completion_rate_pct=round((completed / total) * 100.0, 2) if total > 0 else 0.0,
            tool_correctness_rate_pct=round((tools_ok / total) * 100.0, 2) if total > 0 else 0.0,
            avg_trajectory_length=round(avg_traj, 2),
            hard_failures=hard_fails,
            soft_failures=soft_fails,
            cascading_soft_failures=cascading_fails,
            total_tokens=tot_toks,
            avg_tokens_per_query=round(avg_toks, 1),
            avg_coordination_tokens=round(avg_coord, 1),
            total_estimated_cost_usd=round(tot_cost, 6),
            avg_latency_ms=round(avg_lat, 2)
        )
        return results, summary

    def compare_patterns(self) -> Dict[str, Any]:
        """Runs evaluation on both Multi-Agent and Single-Agent baseline to measure coordination cost."""
        multi_results, multi_summary = self.run_suite(pattern="multi_agent")
        single_results, single_summary = self.run_suite(pattern="single_agent")

        return {
            "multi_agent": {
                "summary": multi_summary.model_dump(),
                "cases": [r.model_dump() for r in multi_results]
            },
            "single_agent_baseline": {
                "summary": single_summary.model_dump(),
                "cases": [r.model_dump() for r in single_results]
            },
            "comparison": {
                "coordination_token_overhead_pct": round(
                    ((multi_summary.total_tokens - single_summary.total_tokens) / max(1, single_summary.total_tokens)) * 100.0, 2
                ),
                "completion_rate_delta_pct": round(
                    multi_summary.completion_rate_pct - single_summary.completion_rate_pct, 2
                ),
                "tool_correctness_delta_pct": round(
                    multi_summary.tool_correctness_rate_pct - single_summary.tool_correctness_rate_pct, 2
                )
            }
        }
