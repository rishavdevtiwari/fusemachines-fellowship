"""
Decomposition -> SQL string.

A single template handles all 50 benchmark patterns:

    SELECT [DISTINCT] {columns}
    FROM   {tables[0]}
    {joins}
    [WHERE  {filters}]
    [GROUP BY {group_by}]
    [ORDER BY {order_by}]

The decomposition fields hold pre-formatted SQL fragments
(see Task2_Decomposition.json), so generation is essentially
deterministic string composition. This is the "rule-based pipeline"
approach allowed by the Task 3 spec.
"""
from __future__ import annotations

from typing import Dict


def generate_sql(decomp: Dict) -> str:
    if not decomp.get("tables"):
        raise ValueError("Decomposition is missing 'tables'")
    if not decomp.get("columns"):
        raise ValueError("Decomposition is missing 'columns'")

    distinct = " DISTINCT" if decomp.get("distinct") else ""
    cols = ", ".join(decomp["columns"])
    primary = decomp["tables"][0]

    parts = [f"SELECT{distinct} {cols}", f"FROM {primary}"]

    # Joins
    for j in decomp.get("joins", []):
        jt = j.get("type", "INNER").upper()
        if jt not in {"INNER", "LEFT", "RIGHT", "FULL"}:
            raise ValueError(f"Unsupported join type: {jt}")
        parts.append(f'{jt} JOIN {j["table"]} ON {j["on"]}')

    # Filters (WHERE)
    filters = decomp.get("filters", [])
    if filters:
        parts.append("WHERE " + " AND ".join(filters))

    # GROUP BY
    group_by = decomp.get("group_by", [])
    if group_by:
        parts.append("GROUP BY " + ", ".join(group_by))

    # ORDER BY
    order_by = decomp.get("order_by", [])
    if order_by:
        parts.append("ORDER BY " + ", ".join(order_by))

    sql = "\n".join(parts) + ";"
    return sql
