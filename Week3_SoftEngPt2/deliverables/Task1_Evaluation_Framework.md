# Task 1 — Part 2: Text-to-SQL Agent Evaluation Framework

**Author:** Rishav Dev Tiwari
**Purpose:** Define a rigorous, multi-dimensional framework for measuring whether a Text-to-SQL agent is actually any good. This framework is what Task 3 and Task 4 will be measured against.

---

## 1. Why "did it produce correct SQL?" is not enough

A naive evaluation just checks: *"Does the generated SQL match the ground-truth SQL string?"*

That breaks down for three reasons:

1. **Many SQL queries are equivalent.** `SELECT * FROM customers` and `SELECT "customerNumber", "customerName", ... FROM customers` return identical rows but have different strings.
2. **Two correct queries can return different result orderings** unless `ORDER BY` is present, but the answer is still right.
3. **A query can be syntactically valid yet semantically wrong** (joins the wrong tables, filters the wrong column).

So a real evaluation framework must score the agent on **multiple orthogonal axes** and aggregate them. That is the framework below.

---

## 2. Evaluation axes

I propose **eight axes**, grouped into four families.

### 2.1 Correctness family

| # | Metric | What it measures | How to compute |
|---|--------|------------------|----------------|
| C1 | **Execution success rate** | Does the generated SQL execute without raising an error? | `# queries that executed cleanly / total queries` |
| C2 | **Result-set equality (Exact-Match Accuracy)** | Does the result set match the ground-truth result set? | Run both queries; compare result rows as multisets of tuples (order-insensitive unless the question says "ordered by ..."). |
| C3 | **Schema-link correctness** | Did the agent pick the right tables and columns, even if the surface SQL differs? | Parse generated SQL; extract referenced tables/columns; compare to a reference set per question. |

> **C2 is the gold standard.** Two queries are "equivalent" iff they produce the same result set. This sidesteps the SQL-string-matching trap entirely.

### 2.2 Robustness family

| # | Metric | What it measures | How to compute |
|---|--------|------------------|----------------|
| R1 | **Retry / self-correction rate** | When the first SQL fails, does the agent fix it on retry? | `# queries that succeeded after ≥1 retry / # queries that needed retry` |
| R2 | **Final-answer success rate** | After all retries, what fraction succeeded? | `# final-status = success / total queries` |

### 2.3 Safety family

| # | Metric | What it measures | How to compute |
|---|--------|------------------|----------------|
| S1 | **Read-only enforcement** | Did the agent ever produce or attempt to execute `INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE`? | Static check on every generated SQL. Any non-`SELECT` statement = automatic 0 for that axis. |

### 2.4 Quality family

| # | Metric | What it measures | How to compute |
|---|--------|------------------|----------------|
| Q1 | **Latency** | How long from question → answer? | Wall-clock time per question; report p50 / p95. |
| Q2 | **Natural-language answer quality** | Does the human-readable summary correctly describe the result? | For Task 4 only: spot-check 10% of summaries; mark each as faithful / hallucinated / vague. |

---

## 3. Result-set equality — exact algorithm

This is the most important metric, so its definition needs to be unambiguous.

**Inputs:**

- `R_gen` = result rows from generated SQL
- `R_truth` = result rows from ground-truth SQL

**Algorithm:**

```
def result_match(R_gen, R_truth, ordered=False) -> bool:
    if R_gen is None or R_truth is None:
        return False
    if ordered:
        return list(R_gen) == list(R_truth)
    # Unordered comparison: multiset of tuples
    return Counter(tuple(r) for r in R_gen) == Counter(tuple(r) for r in R_truth)
```

**Special case — column-projection mismatch:**
If the generated query selects extra columns (e.g. `*` instead of `name, city`), but the *requested* columns appear with the same values, we have two options:

- **Strict:** require identical column sets → mismatch fails.
- **Lenient:** project the generated result onto the columns named in the ground truth, then compare.

I recommend **strict by default**, with a lenient mode flag for diagnostics.

**Float / numeric comparisons:** allow `abs(a - b) < 1e-6` for any numeric column.

---

## 4. Per-question report card

For each of the 50 benchmark questions, the framework records:

```json
{
  "id": 31,
  "question": "Count customers per country",
  "ground_truth_sql": "SELECT \"country\", COUNT(*) ...",
  "generated_sql": "SELECT \"country\", COUNT(*) ...",
  "executed_successfully": true,
  "execution_time_ms": 14,
  "retry_count": 0,
  "result_match": true,
  "schema_links": {
    "tables_truth":   ["customers"],
    "tables_gen":     ["customers"],
    "columns_truth":  ["country"],
    "columns_gen":    ["country"],
    "tables_match":   true,
    "columns_match":  true
  },
  "safety_violation": false,
  "final_status": "success"
}
```

These per-question records aggregate into a **single dashboard** (see §5).

---

## 5. Aggregate evaluation report

| Axis | Symbol | Score | Threshold for "good" |
|------|--------|-------|----------------------|
| Execution success rate | C1 | 0..1 | ≥ 0.95 |
| Result-set equality | C2 | 0..1 | ≥ 0.85 |
| Schema-link table accuracy | C3a | 0..1 | ≥ 0.90 |
| Schema-link column accuracy | C3b | 0..1 | ≥ 0.80 |
| Self-correction rate | R1 | 0..1 | ≥ 0.50 |
| Final-answer success rate | R2 | 0..1 | ≥ 0.90 |
| Read-only enforcement | S1 | bool | must be 100 % |
| p50 latency | Q1 | ms | ≤ 500 ms (rule-based) / ≤ 3000 ms (LLM) |
| p95 latency | Q1 | ms | ≤ 1500 ms (rule-based) / ≤ 8000 ms (LLM) |

A single composite score (helpful for tracking progress across versions):

```
Composite = 0.35*C2 + 0.20*C3a + 0.15*C3b + 0.20*R2 + 0.10*C1
```

(The weights reflect: result correctness matters most; schema understanding second; final reliability third; raw execution rate is partially captured by R2 already.)

---

## 6. The benchmark dataset

The benchmark is the 50 questions in `Guidelines/sql_questions_only.csv`, with their ground-truth SQL from **Task 1 Part 1** (`Task1_GroundTruth_SQL.md`) and the structured decompositions from **Task 2** (`Task2_Decomposition.json`).

Difficulty breakdown:

| Section | Count | Skill tested |
|---------|-------|--------------|
| A | 20 | Single-table projection, `DISTINCT`, basic `SELECT` |
| B | 10 | Inner joins, self-joins, left-joins |
| C | 10 | `GROUP BY`, aggregate functions over groups |
| D | 10 | Whole-table aggregates (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`) |

Reporting per-section scores (not just an overall number) reveals **where** the agent fails — usually section B (joins) is the hardest for naive systems.

---

## 7. Failure-mode taxonomy

When a question fails, the framework should classify it into one of these buckets so we can fix the right thing:

| Bucket | Example | Likely fix |
|--------|---------|------------|
| **Schema hallucination** | Agent uses table `customer` (singular) instead of `customers` | Inject schema into the prompt / decomposition |
| **Wrong column** | Agent uses `price` instead of `MSRP` | Better column-name reasoning, synonyms list |
| **Missing JOIN** | Agent SELECTs from one table when two are needed | Decomposition step must enumerate tables |
| **Wrong join condition** | Agent joins `customers.customerNumber = orders.orderNumber` | Enforce FK awareness in prompt |
| **Wrong aggregation** | Agent uses `SUM` where `COUNT` was wanted | Better intent classification |
| **Missing GROUP BY** | Aggregate without grouping → SQL error | Validator catches; retry |
| **Quote/case error** | Agent writes `customername` (Postgres folds to lower) | Auto-fix via case-insensitive column lookup |
| **Safety violation** | Agent emits `DROP TABLE` | Hard reject in validator |
| **Empty / null answer** | Query runs but returns 0 rows when ground truth has rows | Filter / WHERE bug; tighten schema linking |
| **Non-deterministic order** | Ground truth uses `ORDER BY`, agent did not | Lenient unordered compare; warn |

---

## 8. How this connects to Tasks 3 and 4

**Task 3 (Pipeline):**

- Logs each step (decomposition → SQL → execute → retry).
- After running all 50 questions through `pipeline.run_benchmark()`, it emits a CSV + Markdown report scored on every axis above.
- The retry policy is capped at **1 retry** per Task 3 spec.

**Task 4 (Agent endpoint):**

- Same pipeline, but with retry cap **3** per Task 4 spec, plus a final natural-language summary step.
- The endpoint can be evaluated by replaying the 50 benchmark questions against `POST /agent/sql` and feeding the JSON responses back into this framework.

---

## 9. Anti-patterns to avoid in evaluation

1. **String matching the SQL.** Two correct queries can be very different strings.
2. **Counting "agent ran without error" as success.** A query that executes but returns wrong rows is *worse* than one that errors loudly.
3. **Ignoring latency.** A 10-second query that's correct 100 % of the time is unusable in a chat UX.
4. **Single composite score with no per-axis breakdown.** You cannot debug a 0.73 score; you can debug "schema-link column accuracy = 0.42".
5. **Manual eyeballing.** Always automate the full pipeline so the score is reproducible across runs.
6. **Using the same dataset for both prompt-tuning and final evaluation.** Hold out at least 20 % of questions as a test split (in our case the GROUP BY questions Q31–Q40 are a useful held-out slice).

---

## 10. Summary

The framework above gives a **scientific, repeatable** way to answer the assignment's central question:

> "How do we know if the agent is actually good?"

By answering on eight orthogonal axes — and never relying on SQL-string matching — we get a defensible measurement that survives across model upgrades, prompt rewrites, and rule-based vs LLM-based implementations.

This document is the contract that **Task 3** (pipeline + benchmark runner) and **Task 4** (FastAPI agent endpoint) implement against.
