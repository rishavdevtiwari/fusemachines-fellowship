"""
Optional smoke test that proves the pipeline's self-correction (retry)
behaviour works when the generated SQL is broken.

We intentionally feed broken SQL via a custom decomposition that drops
the double-quotes around a camelCase column. PostgreSQL will fold the
identifier to lowercase, raise 'column "..." does not exist', and the
validator's autofix should re-quote the column and the retry should
succeed.

Run this AFTER the database is up and DATABASE_URL is set.

    python test_retry.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from text2sql import sql_generator, validator, executor
from text2sql.pipeline import Text2SQLPipeline

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
assert DATABASE_URL, "DATABASE_URL is not set"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def main() -> None:
    print("=" * 70)
    print("Retry-mechanism demonstration")
    print("=" * 70)

    # Build a deliberately broken SQL by leaving 'productName' un-quoted.
    # Postgres folds it to lowercase 'productname' which does not exist,
    # so the first execution will fail with:
    #   column "productname" does not exist
    # The autofix in validator.py should re-quote it and retry.
    broken_sql = 'SELECT productName, "MSRP" FROM products LIMIT 3;'

    # Run via raw executor (no pipeline) to show the failure
    print("\n[Direct execute, no retry] broken SQL:")
    print(f"  SQL:   {broken_sql}")
    res = executor.execute(broken_sql, SessionLocal)
    print(f"  ok:    {res.success}")
    print(f"  error: {res.error}")

    # Now manually trigger autofix and re-execute
    fixed = validator.attempt_autofix(broken_sql, res.error or "")
    print(f"\n[After autofix] fixed SQL:")
    print(f"  SQL:   {fixed}")
    res2 = executor.execute(fixed or "", SessionLocal)
    print(f"  ok:    {res2.success}")
    print(f"  rows:  {res2.row_count}")
    if res2.success:
        for r in res2.rows:
            print(f"    {r}")

    # Now show the same scenario via the full pipeline by injecting a
    # broken decomposition. We monkey-patch sql_generator.generate_sql
    # for ONE call to return the broken SQL, then the pipeline's
    # validate->execute->autofix->retry loop should rescue it.
    print("\n[Full pipeline w/ retry on a broken decomposition]")
    pipeline = Text2SQLPipeline(SessionLocal, max_retries=1)

    real_generate = sql_generator.generate_sql
    def broken_generate(decomp):
        return broken_sql
    sql_generator.generate_sql = broken_generate
    try:
        result = pipeline.run("Show 3 product names and MSRPs")
    finally:
        sql_generator.generate_sql = real_generate

    print(f"  status:      {result.status}")
    print(f"  retry_count: {result.retry_count}")
    print(f"  attempts:    {len(result.attempts)}")
    for i, a in enumerate(result.attempts, 1):
        print(f"    attempt {i}: ok={a['ok']}  sql={a['sql']!r}")
    print(f"  final SQL:   {result.sql}")
    print(f"  rows:        {result.row_count}")
    print(f"  summary:     {result.summary}")


if __name__ == "__main__":
    main()
