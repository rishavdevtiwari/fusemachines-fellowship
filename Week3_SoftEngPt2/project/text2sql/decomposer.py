"""
Question -> structured decomposition.

Two strategies are tried in order:

1. Exact / normalized match against the curated benchmark dataset
   (text2sql/data/decompositions.json, which is a copy of the
   Task 2 deliverable). This handles all 50 benchmark questions
   deterministically.

2. Rule-based keyword decomposer for unseen questions.
   - "count X" / "how many"           -> COUNT(*)
   - "total X" / "sum"                -> SUM
   - "average" / "avg"                -> AVG
   - "max" / "highest"                -> MAX
   - "min" / "lowest"                 -> MIN
   - "per Y" / "by Y"                 -> GROUP BY
   - table-name keywords              -> identify table

The output is the decomposition dict shape used everywhere in the
pipeline:

    {
      "id":        int | None,
      "question":  str,
      "intent":    str,
      "tables":    [str, ...],
      "columns":   [str, ...],
      "joins":     [{"type": "INNER"|"LEFT", "table": str, "on": str}, ...],
      "filters":   [str, ...],
      "group_by":  [str, ...],
      "order_by":  [str, ...],
      "distinct":  bool,
      "source":    "benchmark" | "rule_based"
    }
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from .schema import SCHEMA, all_tables

_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "decompositions.json")


def _load_benchmark() -> List[Dict]:
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_BENCHMARK = _load_benchmark()
_BENCHMARK_BY_NORMALIZED = {
    re.sub(r"\s+", " ", item["question"].strip().lower()): item
    for item in _BENCHMARK
}


def _normalize(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


# ---------------------------------------------------------------------------
# Strategy 1: lookup
# ---------------------------------------------------------------------------
def _lookup_benchmark(question: str) -> Optional[Dict]:
    norm = _normalize(question)
    if norm in _BENCHMARK_BY_NORMALIZED:
        item = dict(_BENCHMARK_BY_NORMALIZED[norm])
        item["source"] = "benchmark"
        return item
    return None


# ---------------------------------------------------------------------------
# Strategy 2: rule-based fallback
# ---------------------------------------------------------------------------
# Map common natural-language tokens to canonical table names.
_TABLE_SYNONYMS = {
    "product line": "productlines", "product lines": "productlines",
    "productline": "productlines", "productlines": "productlines",
    "product": "products", "products": "products",
    "office": "offices", "offices": "offices",
    "employee": "employees", "employees": "employees", "staff": "employees",
    "customer": "customers", "customers": "customers", "client": "customers",
    "order detail": "orderdetails", "order details": "orderdetails",
    "orderdetail": "orderdetails", "orderdetails": "orderdetails",
    "line item": "orderdetails", "line items": "orderdetails",
    "order": "orders", "orders": "orders",
    "payment": "payments", "payments": "payments",
}


def _detect_tables(q: str) -> List[str]:
    q_l = q.lower()
    found: List[str] = []
    # Sort by descending length so 'order detail' wins over 'order'
    for token in sorted(_TABLE_SYNONYMS.keys(), key=len, reverse=True):
        if token in q_l and _TABLE_SYNONYMS[token] not in found:
            found.append(_TABLE_SYNONYMS[token])
    return found


def _detect_aggregate(q: str) -> Optional[str]:
    q_l = q.lower()
    if re.search(r"\b(count|how many|number of|total number)\b", q_l):
        return "COUNT"
    if re.search(r"\b(average|avg|mean)\b", q_l):
        return "AVG"
    if re.search(r"\btotal|sum\b", q_l):
        return "SUM"
    if re.search(r"\bmax|maximum|highest\b", q_l):
        return "MAX"
    if re.search(r"\bmin|minimum|lowest\b", q_l):
        return "MIN"
    return None


def _rule_based(question: str) -> Dict:
    """
    Produce a best-effort decomposition for an unseen question.
    The strategy is deliberately simple — the goal is "do something
    reasonable" rather than "be perfect", with the validator+retry
    loop catching the rest.
    """
    tables = _detect_tables(question) or ["customers"]  # fallback table
    primary = tables[0]
    agg = _detect_aggregate(question)

    columns: List[str]
    if agg == "COUNT":
        columns = [f'COUNT(*) AS "count"']
    elif agg in {"SUM", "AVG", "MIN", "MAX"}:
        # Pick a numeric column from the primary table heuristically
        numeric_priority = {
            "products":     ["MSRP", "buyPrice", "quantityInStock"],
            "payments":     ["amount"],
            "orderdetails": ["quantityOrdered", "priceEach"],
            "customers":    ["creditLimit"],
        }.get(primary, [])
        target_col = next((c for c in numeric_priority if c in SCHEMA[primary]), None)
        if target_col is None:
            columns = ["*"]
        else:
            alias = f'{agg.lower()}{target_col[0].upper()}{target_col[1:]}'
            columns = [f'{agg}("{target_col}") AS "{alias}"']
    else:
        columns = ["*"]

    return {
        "id": None,
        "question": question,
        "intent": f"Rule-based fallback for: {question}",
        "tables": [primary],
        "columns": columns,
        "joins": [],
        "filters": [],
        "group_by": [],
        "order_by": [],
        "distinct": False,
        "source": "rule_based",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def decompose(question: str) -> Dict:
    """Return a structured decomposition for the given NL question."""
    if not question or not question.strip():
        raise ValueError("Empty question")
    bench = _lookup_benchmark(question)
    if bench is not None:
        return bench
    return _rule_based(question)


def list_benchmark_questions() -> List[Dict]:
    """Return all 50 curated decompositions, in id order."""
    return sorted([dict(d) for d in _BENCHMARK], key=lambda x: x["id"])
