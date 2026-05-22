"""
Execute a validated SELECT statement against PostgreSQL and return
the rows + timing, or a structured error.

Uses the existing project SQLAlchemy engine from `database.SessionLocal`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError


@dataclass
class ExecutionResult:
    success: bool
    rows: List[Dict[str, Any]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    row_count: int = 0


def execute(sql: str, session_factory) -> ExecutionResult:
    """
    Run `sql` using a fresh session from `session_factory`.

    `session_factory` is a callable that returns a SQLAlchemy Session
    (i.e. the project's `SessionLocal`). We open and close it locally
    so the executor is safe to call from any context (FastAPI route,
    benchmark runner, async worker).
    """
    start = time.perf_counter()
    db = session_factory()
    try:
        result = db.execute(text(sql))
        # `result.mappings()` gives dict-like rows preserving column names
        mappings = result.mappings().all()
        rows = [dict(m) for m in mappings]
        cols = list(rows[0].keys()) if rows else list(result.keys())
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return ExecutionResult(
            success=True,
            rows=rows,
            columns=cols,
            elapsed_ms=elapsed_ms,
            row_count=len(rows),
        )
    except SQLAlchemyError as e:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        # Strip the noisy SQLAlchemy wrapper, keep the underlying DB error
        msg = str(getattr(e, "orig", e)) or str(e)
        return ExecutionResult(
            success=False,
            error=msg.strip(),
            elapsed_ms=elapsed_ms,
        )
    except Exception as e:  # noqa: BLE001 - we never want the executor to crash
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return ExecutionResult(
            success=False,
            error=f"{type(e).__name__}: {e}",
            elapsed_ms=elapsed_ms,
        )
    finally:
        db.close()
