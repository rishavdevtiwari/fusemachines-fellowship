"""
Safety validator + light auto-fix for generated SQL.

Rules enforced (per Task 3 / Task 4 specs):
1. Only a single statement is allowed.
2. The statement must be a SELECT (or a CTE that ends in SELECT).
3. Mutating keywords (INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE,
   CREATE, GRANT, REVOKE, MERGE, COPY) are rejected hard.
4. SQL comments (/* ... */ and -- ...) are stripped before parsing
   so a hidden DROP inside a comment can't slip through.

The validator is intentionally conservative: false positives (rejecting
a borderline-safe query) are far better than letting a destructive
statement through.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_MUTATING_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "MERGE", "COPY", "VACUUM",
    "REINDEX", "CLUSTER", "REFRESH",
)


@dataclass
class ValidationResult:
    is_safe: bool
    reason: str = ""


def _strip_comments(sql: str) -> str:
    # Remove block comments
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    # Remove line comments
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql


def validate(sql: str) -> ValidationResult:
    if not sql or not sql.strip():
        return ValidationResult(False, "Empty SQL")

    cleaned = _strip_comments(sql).strip()

    # Reject multiple statements (allow a single trailing ';')
    inner = cleaned.rstrip(";").strip()
    if ";" in inner:
        return ValidationResult(
            False,
            "Multiple statements detected; only one SELECT is allowed",
        )

    upper = inner.upper()
    # Must start with SELECT or WITH
    if not (upper.startswith("SELECT") or upper.startswith("WITH ")):
        return ValidationResult(
            False,
            "Only SELECT statements are allowed (must start with SELECT or WITH)",
        )

    # Block any mutating keywords as whole words
    for kw in _MUTATING_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            return ValidationResult(
                False,
                f"Disallowed keyword detected: {kw}",
            )

    return ValidationResult(True)


def attempt_autofix(sql: str, error_message: str) -> str | None:
    """
    Tiny heuristic fixer used between retry attempts.

    Returns a *new* SQL string if a fix was applied, or None to
    indicate the error is not auto-fixable here. The pipeline will
    then surface the original error.

    Currently handles:
    - PostgreSQL "column ... does not exist" caused by un-quoted
      camelCase identifier: wrap in double quotes.
    - "syntax error at or near \\";\\"" (trailing duplicate semicolons).
    """
    if not error_message:
        return None

    msg = error_message.lower()

    # Trailing extra semicolons
    if "syntax error at or near \";\"" in msg:
        fixed = re.sub(r";+\s*;", ";", sql)
        if fixed != sql:
            return fixed

    # column "x" does not exist  ->  PostgreSQL folded an unquoted
    # identifier to lowercase. Try quoting any case-insensitive match
    # in the original SQL.
    m = re.search(r'column "([^"]+)" does not exist', error_message)
    if m:
        bad = m.group(1)
        from .schema import SCHEMA
        for cols in SCHEMA.values():
            for col in cols:
                if col.lower() == bad.lower():
                    # Case-insensitive match against the original SQL,
                    # but only at standalone identifier positions
                    # (not preceded/followed by another word char or quote).
                    pattern = re.compile(
                        rf'(?<![\w".]){re.escape(col)}(?![\w"])',
                        flags=re.IGNORECASE,
                    )
                    fixed = pattern.sub(f'"{col}"', sql)
                    if fixed != sql:
                        return fixed

    return None
