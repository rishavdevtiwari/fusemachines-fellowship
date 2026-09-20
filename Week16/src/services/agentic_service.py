"""
Unified Agentic Dispute Resolution Service for ShopAssist AI (Week 16).
Coordinates:
- ONNX Edge Intent Router (<5ms fast pre-classification)
- Cache checking & invalidation
- Multi-Agent Orchestration (Coordinator + Investigator + Policy Auditor)
- Single-Agent Baseline execution mode for comparative benchmarking
- Real-time token and latency accounting
"""

import time
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from src.config import settings
from src.router.onnx_router import onnx_router
from src.agents.coordinator_agent import coordinator_agent
from src.agents.single_agent_baseline import single_agent_baseline
from src.core.structured_outputs import AgenticResolutionPayload
from src.services.cache_service import cache_service

class DisputeResolutionRequest(BaseModel):
    query: str
    customer_id: str = "guest_user"
    pattern: Optional[str] = None  # "multi_agent" or "single_agent"
    max_iterations: Optional[int] = None
    skip_cache: bool = False

class DisputeResolutionResponse(BaseModel):
    resolution: AgenticResolutionPayload
    onnx_intent: str
    onnx_confidence: float
    onnx_latency_ms: float
    cached: bool = False
    total_service_time_ms: float

class AgenticService:
    def __init__(self):
        self.total_disputes_handled: int = 0

    def resolve(self, req: DisputeResolutionRequest) -> DisputeResolutionResponse:
        t0 = time.perf_counter()
        self.total_disputes_handled += 1
        pattern = (req.pattern or settings.agent_pattern).lower()

        # 1. Check Cache
        cache_key = cache_service.generate_key(
            "agentic_dispute",
            {"query": req.query.strip().lower(), "pattern": pattern}
        )
        if not req.skip_cache:
            cached_data = cache_service.get(cache_key)
            if cached_data:
                res = DisputeResolutionResponse(**cached_data)
                res.cached = True
                res.total_service_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                return res

        # 2. Fast ONNX Edge Intent Router
        onnx_pred = onnx_router.predict(req.query)

        # 3. Execute Selected Agentic Architecture
        if pattern == "single_agent":
            payload = single_agent_baseline.resolve_dispute(
                query=req.query,
                customer_id=req.customer_id,
                max_iterations=req.max_iterations
            )
        else:
            payload = coordinator_agent.resolve_dispute(
                query=req.query,
                customer_id=req.customer_id,
                max_iterations=req.max_iterations
            )

        total_latency = (time.perf_counter() - t0) * 1000.0

        response = DisputeResolutionResponse(
            resolution=payload,
            onnx_intent=onnx_pred.intent,
            onnx_confidence=onnx_pred.confidence,
            onnx_latency_ms=onnx_pred.latency_ms,
            cached=False,
            total_service_time_ms=round(total_latency, 2)
        )

        # 4. Cache Resolution
        cache_service.set(cache_key, response.model_dump())

        return response

agentic_service = AgenticService()
