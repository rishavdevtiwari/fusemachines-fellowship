"""
Convert a SQL result into a one-line, human-readable summary.

This is rule-based (no LLM) — it inspects the result shape and
chooses an appropriate phrasing. It is good enough to satisfy the
Task 4 deliverable "summary" field for the 50 benchmark questions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _fmt_value(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, float):
        # Trim trailing zeros, keep up to 2 dp for currency-ish numbers
        return f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.2f}"
    return str(v)


def summarize(question: str, rows: List[Dict[str, Any]], columns: List[str]) -> str:
    """Produce a one-line natural-language answer."""
    n = len(rows)

    # Empty result
    if n == 0:
        return f"No rows were returned for: \"{question}\"."

    # Single scalar result (e.g. SELECT COUNT(*) FROM ...)
    if n == 1 and len(columns) == 1:
        col = columns[0]
        val = rows[0][col]
        return f"The answer to \"{question}\" is {_fmt_value(val)} ({col})."

    # Single row, multiple columns -> list the fields
    if n == 1:
        pairs = ", ".join(f"{c}={_fmt_value(rows[0][c])}" for c in columns)
        return f"Found 1 row for \"{question}\": {pairs}."

    # Multiple rows
    preview_n = min(3, n)
    label_col = _pick_label_column(columns)
    if label_col:
        labels = [_fmt_value(rows[i][label_col]) for i in range(preview_n)]
        preview = ", ".join(labels)
        suffix = f", ... ({n - preview_n} more)" if n > preview_n else ""
        return f'Returned {n} rows for "{question}". Top {label_col}: {preview}{suffix}.'

    # Generic multi-column / multi-row fallback
    return f'Returned {n} rows with {len(columns)} columns for "{question}".'


def _pick_label_column(columns: List[str]) -> Optional[str]:
    """Pick a sensible 'label' column for previewing rows."""
    candidates = [
        "customerName", "productName", "city", "country", "status",
        "productLine", "productVendor", "jobTitle", "officeCode",
        "orderNumber", "productCode", "employeeNumber",
    ]
    for c in candidates:
        if c in columns:
            return c
    # Otherwise fall back to the first column
    return columns[0] if columns else None
