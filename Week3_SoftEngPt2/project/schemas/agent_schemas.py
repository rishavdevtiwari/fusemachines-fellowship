"""
Pydantic models for the Task 4 mini SQL agent endpoint.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentSQLRequest(BaseModel):
    question: str = Field(
        ...,
        description="A natural-language question about the ClassicModels database.",
        examples=["How many shipped orders are from USA customers?"],
        min_length=1,
        max_length=500,
    )


class AgentAttempt(BaseModel):
    attempt: int
    sql: str
    ok: bool
    error: Optional[str] = None
    elapsed_ms: float
    row_count: int


class AgentSQLResponse(BaseModel):
    sql: str = Field(..., description="The final SQL query that was executed.")
    result: Any = Field(
        ...,
        description=(
            "The query result. A single scalar for whole-table aggregates, "
            "a single row dict for one-row queries, or a list of row dicts."
        ),
    )
    summary: str = Field(
        ..., description="One-line natural-language summary of the answer."
    )
    status: str = Field(..., description="success | failed", examples=["success"])
    error: Optional[str] = Field(
        None, description="Populated when status == 'failed'."
    )
    decomposition: Dict[str, Any] = Field(
        ...,
        description="The structured Task-2-style decomposition used to generate SQL.",
    )
    retry_count: int = Field(
        ..., description="How many auto-fix retries were performed (max 3)."
    )
    attempts: List[AgentAttempt] = Field(
        default_factory=list,
        description="Step-by-step audit log of every execution attempt.",
    )
    elapsed_ms: float = Field(
        ..., description="Total wall-clock time spent inside the agent."
    )
