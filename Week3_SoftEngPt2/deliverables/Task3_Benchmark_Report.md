# Task 3 — Text-to-SQL Pipeline Benchmark Report

**Run started:** 2026-05-21T04:05:01.540202 UTC  
**Run finished:** 2026-05-21T04:05:01.714361 UTC  
**Questions evaluated:** 50

## Aggregate scorecard

| Axis | Symbol | Score | Threshold |
|------|--------|------:|----------:|
| Execution success rate | C1 | 100.00% | ≥ 95% |
| Result-set equality | C2 | 100.00% | ≥ 85% |
| Schema-link table accuracy | C3a | 100.00% | ≥ 90% |
| Schema-link column accuracy | C3b | 100.00% | ≥ 80% |
| Self-correction rate (R1) | R1 | 100.00% (0/0) | ≥ 50% |
| Final-answer success rate | R2 | 100.00% | ≥ 90% |
| p50 latency (ms) | Q1 | 0.6 | ≤ 500 |
| p95 latency (ms) | Q1 | 1.6 | ≤ 1500 |

**Composite score:** **100.00%** (0.35·C2 + 0.20·C3a + 0.15·C3b + 0.20·R2 + 0.10·C1)

## Per-section pass rate

| Section | Description | Pass / Total | Pass rate |
|---------|-------------|--------------|-----------|
| A | Single-table SELECT (Q1–Q20) | 20 / 20 | 100% |
| B | Joins (Q21–Q30) | 10 / 10 | 100% |
| C | GROUP BY aggregates (Q31–Q40) | 10 / 10 | 100% |
| D | Whole-table aggregates (Q41–Q50) | 10 / 10 | 100% |

## Per-question detail

| # | Question | Generated SQL OK? | Result match? | Retry? | Final status |
|---|----------|:-----------------:|:-------------:|:------:|:------------:|
| Q1 | List all products | Yes | Yes | No | Success |
| Q2 | Get all customers | Yes | Yes | No | Success |
| Q3 | Show all orders | Yes | Yes | No | Success |
| Q4 | List all employees | Yes | Yes | No | Success |
| Q5 | Get all offices | Yes | Yes | No | Success |
| Q6 | Show all product lines | Yes | Yes | No | Success |
| Q7 | List all payments | Yes | Yes | No | Success |
| Q8 | Get product names and prices | Yes | Yes | No | Success |
| Q9 | Get customer names and cities | Yes | Yes | No | Success |
| Q10 | List employee first and last names | Yes | Yes | No | Success |
| Q11 | Get all order dates | Yes | Yes | No | Success |
| Q12 | Show product vendor list | Yes | Yes | No | Success |
| Q13 | Get all product codes | Yes | Yes | No | Success |
| Q14 | List all countries from offices | Yes | Yes | No | Success |
| Q15 | Show all order statuses | Yes | Yes | No | Success |
| Q16 | Get all payment amounts | Yes | Yes | No | Success |
| Q17 | List all job titles | Yes | Yes | No | Success |
| Q18 | Get customer phone numbers | Yes | Yes | No | Success |
| Q19 | Show product MSRP values | Yes | Yes | No | Success |
| Q20 | List order numbers | Yes | Yes | No | Success |
| Q21 | Get orders with customer names | Yes | Yes | No | Success |
| Q22 | Get employees with office city | Yes | Yes | No | Success |
| Q23 | Get payments with customer names | Yes | Yes | No | Success |
| Q24 | Get order details with product names | Yes | Yes | No | Success |
| Q25 | Get products with product line description | Yes | Yes | No | Success |
| Q26 | Get customers with sales rep names | Yes | Yes | No | Success |
| Q27 | Get orders with customer city | Yes | Yes | No | Success |
| Q28 | Get employees and their manager | Yes | Yes | No | Success |
| Q29 | Get orderdetails with product vendor | Yes | Yes | No | Success |
| Q30 | Get payments with customer country | Yes | Yes | No | Success |
| Q31 | Count customers per country | Yes | Yes | No | Success |
| Q32 | Total payments per customer | Yes | Yes | No | Success |
| Q33 | Number of orders per status | Yes | Yes | No | Success |
| Q34 | Products per product line | Yes | Yes | No | Success |
| Q35 | Employees per office | Yes | Yes | No | Success |
| Q36 | Total stock per product vendor | Yes | Yes | No | Success |
| Q37 | Average buy price per product line | Yes | Yes | No | Success |
| Q38 | Orders per customer | Yes | Yes | No | Success |
| Q39 | Max MSRP per product line | Yes | Yes | No | Success |
| Q40 | Min buy price per vendor | Yes | Yes | No | Success |
| Q41 | Total number of customers | Yes | Yes | No | Success |
| Q42 | Total number of products | Yes | Yes | No | Success |
| Q43 | Total revenue from payments | Yes | Yes | No | Success |
| Q44 | Average product price | Yes | Yes | No | Success |
| Q45 | Max payment amount | Yes | Yes | No | Success |
| Q46 | Min payment amount | Yes | Yes | No | Success |
| Q47 | Count total orders | Yes | Yes | No | Success |
| Q48 | Total quantity in stock | Yes | Yes | No | Success |
| Q49 | Average MSRP | Yes | Yes | No | Success |
| Q50 | Number of employees | Yes | Yes | No | Success |

## Failures

_None — all 50 questions passed._

## Notes

- Result-set comparison is **multiset of value tuples**, ignoring column-name differences and minor float noise (1e-6).
- Questions whose natural-language wording implies a specific order are compared **ordered**; the rest are compared **unordered** (see `ordered` flag in `text2sql/data/ground_truth.json`).
- Schema-link extraction uses regex matching against the static schema in `text2sql/schema.py`. Tables are matched case-insensitively; columns are matched only when double-quoted.
- max_retries was set to 1, per the Task 3 spec. (Task 4 raises it to 3.)