"""
FastAPI Async Microservice for ShopAssist AI Agent (Week 16).
Provides high-performance endpoints for:
- Autonomous Multi-Agent Dispute Resolution (/api/v1/agent/resolve)
- Evaluation Harness Benchmarks (/api/v1/agent/evaluate)
- Legacy Single-Pass Assistant (/api/v1/chat)
- Health, Metrics, and Cache Management
"""

import time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import settings
from src.services.agentic_service import (
    agentic_service,
    DisputeResolutionRequest,
    DisputeResolutionResponse
)
from src.services.assistant_service import (
    assistant_orchestrator,
    AssistantRequest,
    AssistantResponsePayload
)
from src.services.rate_limiter import rate_limiter
from src.services.cache_service import cache_service
from src.router.onnx_router import onnx_router
from src.core.rag_pipeline import rag_pipeline
from src.evaluation.eval_harness import EvaluationHarness

app = FastAPI(
    title="ShopAssist AI - Agentic Microservice",
    version="2.0.0",
    description="Production Autonomous Customer Support & Multi-Policy Dispute Resolution Agent."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "127.0.0.1"
    allowed, remaining, retry_after = rate_limiter.is_allowed(client_ip)
    
    if not allowed:
        return Response(
            content=f'{{"detail": "Rate limit exceeded. Retry in {retry_after} seconds."}}',
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            media_type="application/json",
            headers={"Retry-After": str(int(retry_after))}
        )
        
    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    return response

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ShopAssist AI Agentic Gateway",
        "version": "2.0.0",
        "onnx_router": {
            "model_type": onnx_router.model_type,
            "session_loaded": onnx_router.session is not None
        },
        "rag_pipeline": {
            "indexed_chunks": len(rag_pipeline.chunks)
        },
        "agentic_engine": {
            "primary_pattern": settings.agent_pattern,
            "max_iterations": settings.agent_max_iterations,
            "failure_simulation_mode": settings.failure_simulation_mode
        }
    }

@app.get("/api/v1/metrics")
def metrics():
    return {
        "cache": cache_service.get_stats(),
        "rate_limiter": rate_limiter.get_stats(),
        "disputes_handled": agentic_service.total_disputes_handled,
        "queries_handled_w15": assistant_orchestrator.total_queries
    }

@app.post("/api/v1/agent/resolve", response_model=DisputeResolutionResponse)
def resolve_dispute(req: DisputeResolutionRequest):
    """
    Primary Week 16 Endpoint: Resolves complex customer disputes through autonomous
    iterative reasoning, operational tools, RAG policy auditing, and context compaction.
    """
    try:
        return agentic_service.resolve(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/agent/evaluate")
def run_evaluation_benchmark():
    """Executes the custom evaluation harness and returns comparative benchmark metrics."""
    try:
        harness = EvaluationHarness()
        return harness.compare_patterns()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat", response_model=AssistantResponsePayload)
def chat_legacy(req: AssistantRequest):
    """Week 15 Legacy Endpoint: Single-pass assistant pipeline."""
    try:
        return assistant_orchestrator.process_query(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/cache")
def clear_cache():
    cache_service.clear()
    return {"status": "success", "message": "Cache invalidated."}
