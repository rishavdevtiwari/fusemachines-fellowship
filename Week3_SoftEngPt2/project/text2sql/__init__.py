"""
Rule-based Text-to-SQL pipeline for the ClassicModels database.

Components
----------
- decomposer:    NL question -> structured decomposition dict
- sql_generator: decomposition -> SQL string
- validator:     SELECT-only safety check + light auto-fix
- executor:      run SQL against PostgreSQL, return rows / error
- pipeline:      orchestrator (decompose -> generate -> validate -> execute -> retry)

This package is consumed by:
- run_benchmark.py         (Task 3 deliverable: 50-question evaluation)
- routers/agent_router.py  (Task 4 deliverable: POST /agent/sql)
"""
from .pipeline import Text2SQLPipeline, PipelineResult

__all__ = ["Text2SQLPipeline", "PipelineResult"]
