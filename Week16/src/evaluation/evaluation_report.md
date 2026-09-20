# Week 16 Agentic Assistant Evaluation Report

## 1. Executive Summary & Comparison Table

| Metric | Multi-Agent System | Single-Agent Baseline | Delta / Coordination Impact |
| :--- | :--- | :--- | :--- |
| **Task Completion Rate** | **100.0%** (7/7) | 57.14% (4/7) | **+42.86%** |
| **Tool-Call Correctness** | **100.0%** | 71.43% | **+28.57%** |
| **Avg Trajectory Length** | 2.29 steps | 2.0 steps | +0.29 steps |
| **Avg Tokens / Query** | 1968.7 tokens | 671.0 tokens | +193.4% overhead |
| **Avg Coordination Tokens** | 672.4 tokens | 0 tokens | Coordinator orchestrator cost |
| **Total Estimated Cost** | $0.003446 | $0.001174 | +$0.002272 |
| **Avg Latency** | 0.84 ms | 0.36 ms | System decomposition overhead |

---

## 2. Failure Taxonomy Log

Failures categorized according to class framework:

| Failure Category | Multi-Agent Count | Single-Agent Count | Description |
| :--- | :--- | :--- | :--- |
| **Hard Failure** | 0 | 0 | Crashes, unhandled exceptions, infinite loop triggers |
| **Soft Failure** | 0 | 2 | Tool selection errors, parameter mismatch, missed policy clause |
| **Cascading Soft Failure** | 0 | 1 | Early incorrect state propagated down iterations to flawed answer |

---

## 3. Query-by-Query Trajectory & Performance (Multi-Agent System)

| Test ID | Query Category | Trajectory Length | Tools Called | Stopping Condition | Completed | Tokens | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TC-01 | order_tracking | 2 iters | `check_order_status` | `task_completed` | **PASS** | 1569 | 1.21ms |
| TC-02 | cancellation_inquiry | 3 iters | `calculate_cancellation_fee, check_order_status` | `task_completed` | **PASS** | 2575 | 0.96ms |
| TC-03 | damaged_return_dispute | 4 iters | `check_order_status, check_refund_eligibility` | `task_completed` | **PASS** | 4176 | 2.01ms |
| TC-04 | ambiguous_missing_id | 1 iters | `None` | `clarification_needed` | **PASS** | 647 | 0.1ms |
| TC-05 | human_escalation | 1 iters | `escalate_to_human` | `escalation_triggered` | **PASS** | 663 | 0.1ms |
| TC-06 | non_existent_order | 2 iters | `check_order_status` | `task_completed` | **PASS** | 1535 | 0.63ms |
| TC-07 | cross_source_logistics | 3 iters | `verify_courier_tracking, check_order_status` | `task_completed` | **PASS** | 2616 | 0.86ms |

---

## 4. Failure Injection Test Results (Additional Requirement 3)

| Injected Failure Mode | Error Simulated | Recognized & Handled | Hallucination Avoided | Degraded Safely |
| :--- | :--- | :--- | :--- | :--- |
| **Tool Unavailable** | DatabaseConnectionTimeout (503) | True | True | Yes (Reassurance + Escalation) |
| **Malformed RAG Output** | Corrupted UTF-8 / Binary Chunks | True | True | Yes (Sanitized Context) |
| **Gateway Timeout** | Gateway OperationTimeout (504) | True | True | Yes (Timeout Fallback) |

