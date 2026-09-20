"""
Base Agent Abstraction for ShopAssist AI (Week 16).
Provides common interfaces, token accounting, latency metrics, and execution tracing.
"""

import time
from typing import Dict, Any, Optional
from pydantic import BaseModel

class AgentMetrics(BaseModel):
    agent_name: str
    invocations: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0

class BaseAgent:
    """
    Abstract agent foundation tracking token and cost accounting.
    """
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.invocations: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_latency_ms: float = 0.0

    def record_usage(self, p_tokens: int, c_tokens: int, latency_ms: float):
        self.invocations += 1
        self.prompt_tokens += p_tokens
        self.completion_tokens += c_tokens
        self.total_tokens += (p_tokens + c_tokens)
        self.total_latency_ms += latency_ms

    def get_metrics(self) -> AgentMetrics:
        return AgentMetrics(
            agent_name=self.name,
            invocations=self.invocations,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            total_latency_ms=round(self.total_latency_ms, 2)
        )

    def reset_metrics(self):
        self.invocations = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
