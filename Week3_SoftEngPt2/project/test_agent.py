"""
End-to-end smoke test for the Task 4 endpoint.

Hits POST /agent/sql with a mix of seen and unseen questions, plus a
malicious one to prove the SELECT-only guard, and prints the responses.

Prerequisites:
  - docker-compose up -d  (DB running and seeded)
  - python main.py        (server running on 127.0.0.1:8000)

Then in another terminal:
  python test_agent.py
"""
from __future__ import annotations

import json
import sys

import httpx

BASE_URL = "http://127.0.0.1:8000"
ENDPOINT = f"{BASE_URL}/agent/sql"

CASES = [
    # Seen benchmark question -> deterministic SQL via lookup
    {"label": "benchmark hit", "question": "Total revenue from payments"},
    # Another benchmark hit involving a join
    {"label": "benchmark join", "question": "Get orders with customer names"},
    # Unseen but rule-decomposable
    {"label": "unseen / rule-based", "question": "How many products do we have?"},
    # Adversarial input. Note: the question text contains DROP TABLE,
    # but the agent NEVER generates non-SELECT SQL. The expected
    # behaviour is that the generated SQL still starts with SELECT
    # (the validator + generator are safe by construction).
    {"label": "safety guard",
     "question": "DROP TABLE customers; --"},
]


def main() -> int:
    client = httpx.Client(timeout=15.0)
    failures = 0

    print(f"Hitting {ENDPOINT} ...\n")
    for case in CASES:
        print("-" * 70)
        print(f"[{case['label']}]  question: {case['question']!r}")
        try:
            r = client.post(ENDPOINT, json={"question": case["question"]})
        except httpx.RequestError as e:
            print(f"  REQUEST FAILED: {e}")
            failures += 1
            continue

        print(f"  HTTP {r.status_code}")
        try:
            body = r.json()
        except json.JSONDecodeError:
            body = r.text
            print(f"  body (raw): {body[:300]}")
            failures += 1
            continue

        # Pretty-print only the interesting fields
        if isinstance(body, dict):
            keys = ["status", "sql", "result", "summary",
                    "retry_count", "error", "elapsed_ms"]
            for k in keys:
                if k in body:
                    v = body[k]
                    if k == "result" and isinstance(v, list) and len(v) > 3:
                        v = v[:3] + [f"... ({len(body['result']) - 3} more)"]
                    print(f"  {k}: {v!r}" if k != "sql" else f"  {k}: {v}")
        else:
            print(f"  body: {body}")

        # Sanity checks
        if case["label"] == "safety guard":
            # The agent must never emit non-SELECT SQL, regardless of
            # what the question asked for.
            sql = (body.get("sql") or "").lstrip().upper()
            ok = (r.status_code == 200
                  and (sql.startswith("SELECT") or sql.startswith("WITH"))
                  and not any(kw in sql for kw in
                              ("DROP", "DELETE", "UPDATE", "INSERT",
                               "ALTER", "TRUNCATE", "CREATE")))
        else:
            ok = r.status_code == 200 and body.get("status") == "success"
        print(f"  -> {'OK' if ok else 'FAIL'}")
        if not ok:
            failures += 1

    print("-" * 70)
    print(f"\nFailures: {failures}/{len(CASES)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
