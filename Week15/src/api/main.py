"""
FastAPI Production Asynchronous Backend for ShopAssist AI.
Provides high-throughput REST API with concurrent batch processing,
rate limiting, telemetry metrics, and graceful error handling.
"""

import time
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, Response, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.core.structured_outputs import AssistantStructuredResponse
from src.router.onnx_router import onnx_router, IntentPrediction
from src.core.rag_pipeline import rag_pipeline, RAGSearchResult
from src.services.cache_service import cache_service
from src.services.rate_limiter import rate_limiter
from src.services.resilience import CIRCUIT_BREAKERS
from src.services.assistant_service import (
    assistant_orchestrator,
    AssistantRequest,
    AssistantResponsePayload
)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise AI Assistant with ONNX Intent Routing, RAG Pipeline, and Tool Calling."
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Pydantic API Schemas ---
class ChatRequestSchema(BaseModel):
    query: str = Field(..., example="Can I cancel order ORD-1001?")
    customer_id: Optional[str] = Field("guest_user", example="CUST-4421")
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.2)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.95)
    provider: Optional[str] = Field(None, example="gemini")
    skip_cache: Optional[bool] = Field(False)

class BatchChatRequestSchema(BaseModel):
    queries: List[str] = Field(..., min_items=1, max_items=20)
    customer_id: Optional[str] = "batch_user"
    temperature: Optional[float] = 0.2
    provider: Optional[str] = None

class BatchChatResponseSchema(BaseModel):
    total_queries: int
    total_batch_time_ms: float
    average_latency_ms: float
    results: List[AssistantResponsePayload]

class RouteRequestSchema(BaseModel):
    text: str = Field(..., example="Where is my tracking number for ORD-1003?")

class RAGSearchRequestSchema(BaseModel):
    query: str = Field(..., example="What items are non-returnable?")
    top_k: Optional[int] = Field(3, ge=1, le=10)

# --- Rate Limiting Dependency ---
def check_rate_limit(request: Request):
    client_ip = request.client.host if request.client else "unknown_client"
    allowed, remaining, retry_after = rate_limiter.is_allowed(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({settings.rate_limit_requests_per_minute} req/min). Try again in {retry_after}s.",
            headers={"Retry-After": str(retry_after)}
        )

# --- API Endpoints ---

@app.get("/api/v1/health", tags=["Monitoring"])
async def health_check():
    """
    Returns system liveness, active model status, and subsystem health.
    """
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "onnx_router": {
            "model_type": onnx_router.model_type,
            "session_loaded": onnx_router.session is not None,
            "classes_supported": len(onnx_router.labels)
        },
        "rag_pipeline": {
            "indexed_chunks": len(rag_pipeline.vector_db.chunks),
            "status": "ready" if rag_pipeline.initialized else "empty"
        },
        "providers": {
            "primary": settings.primary_provider,
            "fallback": settings.fallback_provider,
            "circuit_breakers": {k: v.get_status() for k, v in CIRCUIT_BREAKERS.items()}
        }
    }

@app.get("/api/v1/metrics", tags=["Monitoring"])
async def system_metrics():
    """
    Returns real-time performance telemetry, cache hit ratios, and rate limiter stats.
    """
    return {
        "cache": cache_service.get_stats(),
        "rate_limiter": rate_limiter.get_stats(),
        "orchestrator": {
            "total_queries_processed": assistant_orchestrator.total_queries,
            "tool_invocations": assistant_orchestrator.tool_invocations
        },
        "circuit_breakers": {k: v.get_status() for k, v in CIRCUIT_BREAKERS.items()}
    }

@app.post("/api/v1/chat", response_model=AssistantResponsePayload, tags=["Assistant"], dependencies=[Depends(check_rate_limit)])
async def chat_interaction(req: ChatRequestSchema):
    """
    Primary chat endpoint: Processes query via Cache -> ONNX Router -> RAG -> Tools -> LLM.
    """
    try:
        assistant_req = AssistantRequest(
            query=req.query,
            customer_id=req.customer_id or "guest_user",
            temperature=req.temperature,
            top_p=req.top_p,
            provider=req.provider,
            skip_cache=req.skip_cache or False
        )
        # Execute asynchronously in thread pool to preserve event loop concurrency
        result = await asyncio.to_thread(assistant_orchestrator.process_query, assistant_req)
        return result
    except Exception as e:
        print(f"Error handling chat request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Assistant processing error: {str(e)}"
        )

@app.post("/api/v1/batch", response_model=BatchChatResponseSchema, tags=["Performance Engineering"], dependencies=[Depends(check_rate_limit)])
async def batch_chat_interaction(req: BatchChatRequestSchema):
    """
    Asynchronous concurrent batch request processor.
    Evaluates up to 20 queries concurrently using async worker pools.
    """
    t_start = time.perf_counter()
    
    async def process_single(q: str) -> AssistantResponsePayload:
        a_req = AssistantRequest(
            query=q,
            customer_id=req.customer_id or "batch_user",
            temperature=req.temperature,
            provider=req.provider
        )
        return await asyncio.to_thread(assistant_orchestrator.process_query, a_req)

    # Concurrently execute all queries in batch
    results = await asyncio.gather(*[process_single(q) for q in req.queries])
    
    total_time = (time.perf_counter() - t_start) * 1000.0
    avg_latency = total_time / len(req.queries) if req.queries else 0.0
    
    return BatchChatResponseSchema(
        total_queries=len(req.queries),
        total_batch_time_ms=round(total_time, 2),
        average_latency_ms=round(avg_latency, 2),
        results=results
    )

@app.post("/api/v1/route", response_model=IntentPrediction, tags=["Intent Routing"])
async def route_intent(req: RouteRequestSchema):
    """
    Standalone sub-5ms ONNX intent router endpoint.
    """
    return onnx_router.predict(req.text)

@app.post("/api/v1/rag/search", response_model=List[RAGSearchResult], tags=["Knowledge Base"])
async def search_knowledge_base(req: RAGSearchRequestSchema):
    """
    Standalone RAG document search endpoint.
    """
    return rag_pipeline.vector_db.search(req.query, top_k=req.top_k or 3)

@app.delete("/api/v1/cache", tags=["Administration"])
async def clear_cache():
    """
    Purges in-memory prompt and response cache.
    """
    cache_service.clear()
    return {"status": "success", "message": "Cache purged successfully."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=settings.api_host, port=settings.api_port, reload=False)
