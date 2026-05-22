"""
End-to-end Text-to-SQL orchestrator.

Flow:
    NL question
       |
       v  decomposer.decompose
    structured decomposition
       |
       v  sql_generator.generate_sql
    SQL string
       |
       v  validator.validate (SELECT-only)
       v  executor.execute
    rows / error
       |
       v  on error: validator.attempt_autofix -> retry  (up to max_retries)
       |
       v  nl_answer.summarize
    final response

This same class is reused by:
- run_benchmark.py        (Task 3 deliverable, max_retries=1)
- routers/agent_router.py (Task 4 deliverable, max_retries=3)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import decomposer, sql_generator, validator, executor, nl_answer

# Per-run logger that ALSO writes to text2sql/logs/pipeline.log so
# graders can inspect every step (Task 3 + Task 4 require step-by-step
# logging of decomposition, SQL generation, and execution time).
_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(_LOG_DIR, "pipeline.log")

logger = logging.getLogger("text2sql.pipeline")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(_LOG_FILE)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    logger.addHandler(fh)
    # Also surface to console for visibility
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("[text2sql] %(message)s"))
    logger.addHandler(sh)
    logger.propagate = False


@dataclass
class PipelineResult:
    question: str
    decomposition: Dict[str, Any]
    sql: str = ""
    rows: List[Dict[str, Any]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    summary: str = ""
    status: str = "pending"        # success | failed
    error: Optional[str] = None
    retry_count: int = 0
    elapsed_ms: float = 0.0
    attempts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        # Truncate row payload in the dict form so it stays log-friendly
        return {
            "question": self.question,
            "sql": self.sql,
            "row_count": self.row_count,
            "result": self.rows[:10],  # preview only
            "summary": self.summary,
            "status": self.status,
            "error": self.error,
            "retry_count": self.retry_count,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "decomposition_source": self.decomposition.get("source"),
            "attempts": self.attempts,
        }


class Text2SQLPipeline:
    """
    Configurable orchestrator. `max_retries` controls how many recovery
    attempts are made *after* the initial execution fails.

    Task 3: max_retries=1
    Task 4: max_retries=3
    """

    def __init__(self, session_factory: Callable, max_retries: int = 1):
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.session_factory = session_factory
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    def run(self, question: str) -> PipelineResult:
        logger.info("Q: %s", question)

        # Step 1: decompose
        decomp = decomposer.decompose(question)
        logger.info("decomposed (source=%s, tables=%s)",
                    decomp.get("source"), decomp.get("tables"))

        result = PipelineResult(question=question, decomposition=decomp)

        # Step 2: generate SQL
        try:
            sql = sql_generator.generate_sql(decomp)
        except Exception as e:  # noqa: BLE001
            result.status = "failed"
            result.error = f"SQL generation failed: {e}"
            logger.error(result.error)
            return result

        result.sql = sql
        logger.info("SQL: %s", sql.replace("\n", " "))

        # Step 3: validate
        v = validator.validate(sql)
        if not v.is_safe:
            result.status = "failed"
            result.error = f"Validator rejected SQL: {v.reason}"
            logger.error(result.error)
            return result

        # Step 4: execute (with up to max_retries auto-fix attempts)
        attempt_no = 0
        current_sql = sql
        while True:
            attempt_no += 1
            ex = executor.execute(current_sql, self.session_factory)
            attempt_record = {
                "attempt": attempt_no,
                "sql": current_sql,
                "ok": ex.success,
                "error": ex.error,
                "elapsed_ms": round(ex.elapsed_ms, 2),
                "row_count": ex.row_count,
            }
            result.attempts.append(attempt_record)
            result.elapsed_ms += ex.elapsed_ms

            if ex.success:
                result.sql = current_sql
                result.rows = ex.rows
                result.columns = ex.columns
                result.row_count = ex.row_count
                result.status = "success"
                result.summary = nl_answer.summarize(
                    question, ex.rows, ex.columns
                )
                logger.info("OK (rows=%d, %.1fms, retries=%d)",
                            ex.row_count, ex.elapsed_ms, result.retry_count)
                return result

            # Failed
            logger.warning("attempt %d failed: %s", attempt_no, ex.error)
            if result.retry_count >= self.max_retries:
                result.status = "failed"
                result.error = ex.error
                # Best-effort summary even on failure
                result.summary = (
                    f"Could not answer \"{question}\" after "
                    f"{attempt_no} attempt(s): {ex.error}"
                )
                return result

            fixed = validator.attempt_autofix(current_sql, ex.error or "")
            if fixed is None:
                # No structural fix available; do not waste a retry
                result.status = "failed"
                result.error = ex.error
                result.summary = (
                    f"Could not answer \"{question}\": {ex.error}"
                )
                return result

            # Re-validate the fixed SQL before re-executing
            v2 = validator.validate(fixed)
            if not v2.is_safe:
                result.status = "failed"
                result.error = f"Auto-fix produced unsafe SQL: {v2.reason}"
                return result

            logger.info("retry %d with auto-fixed SQL", result.retry_count + 1)
            result.retry_count += 1
            current_sql = fixed
