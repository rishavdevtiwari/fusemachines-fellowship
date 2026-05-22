"""
Task 4 deliverable: POST /agent/sql

A mini SQL agent that:
  1. understands a natural-language question (Task 2 decomposition)
  2. generates SQL                            (Task 3 generator)
  3. executes it                              (executor)
  4. self-corrects on error                   (validator.attempt_autofix)
  5. retries up to 3 times                    (Task 4 spec)
  6. returns the result + a NL summary        (nl_answer)

Logging requirements (Task 4 spec):
  - decomposition step
  - SQL generation
  - execution time
All three are recorded in text2sql/logs/pipeline.log via the pipeline's
internal logger, plus the per-attempt audit trail is returned in the
response body so consumers can debug agent behaviour without server
access.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from database import SessionLocal
from logger import logger
from schemas.agent_schemas import AgentAttempt, AgentSQLRequest, AgentSQLResponse
from text2sql import Text2SQLPipeline


router = APIRouter(prefix="/agent", tags=["Agent"])

# Single shared pipeline instance. SessionLocal is opened per-execution
# inside the executor, so this is safe to share across requests.
_pipeline = Text2SQLPipeline(SessionLocal, max_retries=3)


def _coerce_result(rows, columns):
    """
    Shape the response 'result' field nicely:
    - one row + one column  -> scalar value (e.g. COUNT(*))
    - one row + many cols   -> single dict
    - else                  -> list of dicts
    """
    if not rows:
        return []
    if len(rows) == 1 and len(columns) == 1:
        return rows[0][columns[0]]
    if len(rows) == 1:
        return rows[0]
    return rows


@router.post("/sql", response_model=AgentSQLResponse)
def agent_sql(request: AgentSQLRequest) -> AgentSQLResponse:
    logger.info("Incoming /agent/sql request: %r", request.question)

    pr = _pipeline.run(request.question)

    # If the pipeline could not generate SQL at all (e.g. empty question
    # slipped past validation), surface a clean 422 so the client knows
    # the request was malformed rather than the DB being broken.
    if not pr.sql:
        raise HTTPException(
            status_code=422,
            detail=pr.error or "Could not generate SQL for the given question.",
        )

    response = AgentSQLResponse(
        sql=pr.sql,
        result=_coerce_result(pr.rows, pr.columns),
        summary=pr.summary or f'Question: "{pr.question}"',
        status=pr.status,
        error=pr.error,
        decomposition=pr.decomposition,
        retry_count=pr.retry_count,
        attempts=[AgentAttempt(**a) for a in pr.attempts],
        elapsed_ms=round(pr.elapsed_ms, 2),
    )
    logger.info(
        "/agent/sql -> status=%s rows=%d retries=%d %.1fms",
        pr.status, pr.row_count, pr.retry_count, pr.elapsed_ms,
    )
    return response
