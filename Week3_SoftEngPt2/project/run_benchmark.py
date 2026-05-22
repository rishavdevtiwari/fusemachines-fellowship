"""
Task 3 deliverable: 50-question benchmark runner.

Runs every benchmark question through the Text2SQL pipeline, executes
the corresponding ground-truth SQL directly, compares the two result
sets, and produces:

  text2sql/logs/benchmark_results.json   <- machine-readable per-Q records
  text2sql/logs/benchmark_results.csv    <- spreadsheet-friendly view
  ../deliverables/Task3_Benchmark_Report.md
                                         <- aggregate report scored on the
                                            evaluation framework defined in
                                            Task1_Evaluation_Framework.md

Usage
-----
    # 1. start postgres (docker compose up -d) and seed the DB
    # 2. set DATABASE_URL in your .env
    # 3. run:
    python run_benchmark.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Make sure the project dir is on the path so local imports work
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from text2sql import Text2SQLPipeline
from text2sql.schema import all_tables, all_columns_flat

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL is not set. Put it in .env or export it.")
    sys.exit(1)

LOG_DIR = PROJECT_DIR / "text2sql" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DELIVERABLES_DIR = PROJECT_DIR.parent / "deliverables"
DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)

GROUND_TRUTH_PATH = PROJECT_DIR / "text2sql" / "data" / "ground_truth.json"


# ---------------------------------------------------------------------------
# Result-set comparison (per Task 1 evaluation framework, section 3)
# ---------------------------------------------------------------------------
def _normalize_value(v: Any) -> Any:
    """Coerce numeric-ish values to a canonical Python type so two queries
    that return the same number in different SQL types compare equal."""
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, float):
        # Round to 6 dp to swallow tiny FP noise
        return round(v, 6)
    return v


def _row_signature(row: Dict[str, Any]) -> Tuple:
    """
    Order-stable hashable signature of a row. We rely on dict insertion
    order (Python 3.7+) to align columns: SQLAlchemy mappings preserve
    SELECT-clause order, so equivalent SQL produces aligned signatures.
    """
    return tuple(_normalize_value(v) for v in row.values())


def result_match(rows_a: List[Dict[str, Any]],
                 rows_b: List[Dict[str, Any]],
                 ordered: bool = False) -> bool:
    if rows_a is None or rows_b is None:
        return False
    if len(rows_a) != len(rows_b):
        return False
    if ordered:
        return [_row_signature(r) for r in rows_a] == [_row_signature(r) for r in rows_b]
    return Counter(_row_signature(r) for r in rows_a) == \
           Counter(_row_signature(r) for r in rows_b)


# ---------------------------------------------------------------------------
# Schema-link extraction (lightweight regex; sufficient for our SQL shapes)
# ---------------------------------------------------------------------------
_TABLES = set(all_tables())
_COLS = set(all_columns_flat())


def extract_schema_links(sql: str) -> Tuple[set, set]:
    """Return (tables_used, columns_used) from a SQL string."""
    s = sql
    tables: set = set()
    for t in _TABLES:
        if re.search(rf"\b{t}\b", s, flags=re.IGNORECASE):
            tables.add(t)
    columns: set = set()
    # Columns appear quoted in our SQL: "productName"
    for c in _COLS:
        if f'"{c}"' in s:
            columns.add(c)
    return tables, columns


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
def run() -> None:
    print(f"Connecting to {DATABASE_URL!r} ...")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    pipeline = Text2SQLPipeline(SessionLocal, max_retries=1)

    started_at = datetime.utcnow().isoformat()
    records: List[Dict[str, Any]] = []
    section_for_id = lambda i: ("A" if 1 <= i <= 20 else
                                "B" if 21 <= i <= 30 else
                                "C" if 31 <= i <= 40 else "D")

    print(f"Running {len(ground_truth)} questions...")
    for entry in ground_truth:
        qid = entry["id"]
        question = entry["question"]
        gt_sql = entry["sql"]
        ordered = entry.get("ordered", False)

        # --- pipeline run ---
        result = pipeline.run(question)
        gen_sql = result.sql
        gen_rows = result.rows
        gen_status = result.status
        gen_error = result.error
        retry_count = result.retry_count

        # --- ground-truth execution ---
        gt_rows: List[Dict[str, Any]] = []
        gt_error: str | None = None
        gt_elapsed_ms = 0.0
        t0 = time.perf_counter()
        try:
            db = SessionLocal()
            try:
                gt_rows = [dict(m) for m in db.execute(text(gt_sql)).mappings().all()]
            finally:
                db.close()
        except Exception as e:  # noqa: BLE001
            gt_error = f"{type(e).__name__}: {e}"
        gt_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # --- comparisons ---
        executed_ok = (gen_status == "success")
        rows_match = (executed_ok and gt_error is None and
                      result_match(gen_rows, gt_rows, ordered=ordered))

        gen_t, gen_c = extract_schema_links(gen_sql)
        gt_t, gt_c = extract_schema_links(gt_sql)
        tables_match = (gen_t == gt_t)
        columns_match = (gen_c == gt_c)

        final_status = "Success" if rows_match else "Failed"

        rec = {
            "id": qid,
            "section": section_for_id(qid),
            "question": question,
            "ground_truth_sql": gt_sql,
            "generated_sql": gen_sql,
            "executed_successfully": executed_ok,
            "rows_match": rows_match,
            "retry_needed": retry_count > 0,
            "retry_count": retry_count,
            "tables_match": tables_match,
            "columns_match": columns_match,
            "tables_truth": sorted(gt_t),
            "tables_gen": sorted(gen_t),
            "columns_truth": sorted(gt_c),
            "columns_gen": sorted(gen_c),
            "elapsed_ms": round(result.elapsed_ms, 2),
            "gt_elapsed_ms": round(gt_elapsed_ms, 2),
            "row_count_gen": len(gen_rows),
            "row_count_truth": len(gt_rows),
            "summary": result.summary,
            "error": gen_error,
            "final_status": final_status,
        }
        records.append(rec)

        marker = "PASS" if rows_match else "FAIL"
        print(f"  [{marker}] Q{qid:>2} ({rec['section']}) "
              f"{question[:48]:<48} "
              f"gen_rows={len(gen_rows)}  gt_rows={len(gt_rows)}  "
              f"{result.elapsed_ms:.1f}ms")

    finished_at = datetime.utcnow().isoformat()

    # ---- Persist artifacts ----
    json_path = LOG_DIR / "benchmark_results.json"
    csv_path = LOG_DIR / "benchmark_results.csv"
    md_path = DELIVERABLES_DIR / "Task3_Benchmark_Report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "started_at": started_at,
            "finished_at": finished_at,
            "database_url": DATABASE_URL,
            "records": records,
        }, f, indent=2, default=str)
    print(f"\nWrote {json_path}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "section", "question",
            "executed_successfully", "rows_match", "retry_needed",
            "tables_match", "columns_match",
            "row_count_gen", "row_count_truth",
            "elapsed_ms", "final_status",
        ])
        for r in records:
            writer.writerow([
                r["id"], r["section"], r["question"],
                r["executed_successfully"], r["rows_match"], r["retry_needed"],
                r["tables_match"], r["columns_match"],
                r["row_count_gen"], r["row_count_truth"],
                r["elapsed_ms"], r["final_status"],
            ])
    print(f"Wrote {csv_path}")

    _write_markdown_report(md_path, records, started_at, finished_at)
    print(f"Wrote {md_path}")


def _write_markdown_report(path: Path,
                           records: List[Dict[str, Any]],
                           started_at: str,
                           finished_at: str) -> None:
    n = len(records)
    if n == 0:
        path.write_text("# Task 3 Benchmark Report\n\nNo records.\n")
        return

    c1 = sum(1 for r in records if r["executed_successfully"]) / n
    c2 = sum(1 for r in records if r["rows_match"]) / n
    c3a = sum(1 for r in records if r["tables_match"]) / n
    c3b = sum(1 for r in records if r["columns_match"]) / n
    needed_retry = [r for r in records if r["retry_needed"]]
    succeeded_after_retry = [r for r in needed_retry if r["rows_match"]]
    r1 = (len(succeeded_after_retry) / len(needed_retry)) if needed_retry else 1.0
    r2 = sum(1 for r in records if r["final_status"] == "Success") / n

    lat = sorted(r["elapsed_ms"] for r in records)
    p50 = lat[len(lat) // 2]
    p95 = lat[max(0, int(len(lat) * 0.95) - 1)]

    composite = 0.35 * c2 + 0.20 * c3a + 0.15 * c3b + 0.20 * r2 + 0.10 * c1

    section_pass: Dict[str, Tuple[int, int]] = {}
    for r in records:
        s = r["section"]
        passed, total = section_pass.get(s, (0, 0))
        section_pass[s] = (passed + (1 if r["rows_match"] else 0), total + 1)

    lines: List[str] = []
    lines.append("# Task 3 — Text-to-SQL Pipeline Benchmark Report\n")
    lines.append(f"**Run started:** {started_at} UTC  ")
    lines.append(f"**Run finished:** {finished_at} UTC  ")
    lines.append(f"**Questions evaluated:** {n}\n")

    lines.append("## Aggregate scorecard\n")
    lines.append("| Axis | Symbol | Score | Threshold |")
    lines.append("|------|--------|------:|----------:|")
    lines.append(f"| Execution success rate | C1 | {c1:.2%} | ≥ 95% |")
    lines.append(f"| Result-set equality | C2 | {c2:.2%} | ≥ 85% |")
    lines.append(f"| Schema-link table accuracy | C3a | {c3a:.2%} | ≥ 90% |")
    lines.append(f"| Schema-link column accuracy | C3b | {c3b:.2%} | ≥ 80% |")
    lines.append(f"| Self-correction rate (R1) | R1 | "
                 f"{r1:.2%} ({len(succeeded_after_retry)}/{len(needed_retry) or 0}) | ≥ 50% |")
    lines.append(f"| Final-answer success rate | R2 | {r2:.2%} | ≥ 90% |")
    lines.append(f"| p50 latency (ms) | Q1 | {p50:.1f} | ≤ 500 |")
    lines.append(f"| p95 latency (ms) | Q1 | {p95:.1f} | ≤ 1500 |")
    lines.append("")
    lines.append(f"**Composite score:** **{composite:.2%}** "
                 "(0.35·C2 + 0.20·C3a + 0.15·C3b + 0.20·R2 + 0.10·C1)\n")

    lines.append("## Per-section pass rate\n")
    lines.append("| Section | Description | Pass / Total | Pass rate |")
    lines.append("|---------|-------------|--------------|-----------|")
    section_descs = {"A": "Single-table SELECT (Q1–Q20)",
                     "B": "Joins (Q21–Q30)",
                     "C": "GROUP BY aggregates (Q31–Q40)",
                     "D": "Whole-table aggregates (Q41–Q50)"}
    for s in ["A", "B", "C", "D"]:
        passed, total = section_pass.get(s, (0, 0))
        rate = (passed / total) if total else 0.0
        lines.append(f"| {s} | {section_descs[s]} | {passed} / {total} | {rate:.0%} |")
    lines.append("")

    lines.append("## Per-question detail\n")
    lines.append("| # | Question | Generated SQL OK? | Result match? | Retry? | Final status |")
    lines.append("|---|----------|:-----------------:|:-------------:|:------:|:------------:|")
    for r in records:
        ok = "Yes" if r["executed_successfully"] else "No"
        rm = "Yes" if r["rows_match"] else "No"
        retry = "Yes" if r["retry_needed"] else "No"
        lines.append(
            f"| Q{r['id']} | {r['question']} | {ok} | {rm} | {retry} | {r['final_status']} |"
        )
    lines.append("")

    failures = [r for r in records if r["final_status"] != "Success"]
    if failures:
        lines.append("## Failures\n")
        for r in failures:
            lines.append(f"### Q{r['id']}: {r['question']}")
            lines.append(f"- error: `{r['error']}`")
            lines.append(f"- generated SQL: `{r['generated_sql'].strip()}`")
            lines.append(f"- ground truth: `{r['ground_truth_sql']}`")
            lines.append(f"- gen_rows={r['row_count_gen']}, "
                         f"gt_rows={r['row_count_truth']}\n")
    else:
        lines.append("## Failures\n\n_None — all 50 questions passed._\n")

    lines.append("## Notes\n")
    lines.append("- Result-set comparison is **multiset of value tuples**, "
                 "ignoring column-name differences and minor float noise (1e-6).")
    lines.append("- Questions whose natural-language wording implies a "
                 "specific order are compared **ordered**; the rest are "
                 "compared **unordered** (see `ordered` flag in "
                 "`text2sql/data/ground_truth.json`).")
    lines.append("- Schema-link extraction uses regex matching against the "
                 "static schema in `text2sql/schema.py`. Tables are matched "
                 "case-insensitively; columns are matched only when "
                 "double-quoted.")
    lines.append("- max_retries was set to 1, per the Task 3 spec. "
                 "(Task 4 raises it to 3.)")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
