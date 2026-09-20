"""
Configuration Manager for ShopAssist AI Agent (Week 16).
Defines paths, environment variables, multi-provider credentials,
agentic loop parameters, context engineering settings, and failure injection toggles.
"""

import os
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseModel):
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", 8000))
    ui_port: int = int(os.getenv("UI_PORT", 8501))

    # Storage Paths
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    kb_dir: Path = BASE_DIR / "data" / "knowledge_base"
    orders_file: Path = BASE_DIR / "data" / "sample_orders.json"
    models_dir: Path = BASE_DIR / "models"
    onnx_fp32_path: Path = BASE_DIR / "models" / "intent_classifier.onnx"
    onnx_int8_path: Path = BASE_DIR / "models" / "intent_classifier_quantized.onnx"
    metadata_path: Path = BASE_DIR / "models" / "model_metadata.json"

    # LLM Providers
    primary_provider: str = os.getenv("PRIMARY_PROVIDER", "gemini")
    fallback_provider: str = os.getenv("FALLBACK_PROVIDER", "mock")

    # API Keys
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

    # Model Names
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
    vllm_model: str = os.getenv("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

    # Inference Hyperparameters
    default_temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))
    default_top_p: float = float(os.getenv("DEFAULT_TOP_P", "0.95"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1024"))

    # RAG Settings
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "3"))
    rag_chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "500"))
    rag_chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))

    # Production Performance & Resilience
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    cache_max_size: int = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    rate_limit_rpm: int = int(os.getenv("RATE_LIMIT_RPM", "60"))
    max_retry_attempts: int = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
    retry_base_delay: float = float(os.getenv("RETRY_BASE_DELAY", "0.5"))
    use_quantized_onnx: bool = os.getenv("USE_QUANTIZED_ONNX", "true").lower() == "true"
    onnx_threads: int = int(os.getenv("ONNX_THREADS", "4"))

    # Week 16: Agentic Loop & Multi-Agent Architecture Settings
    agent_max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "4"))
    agent_pattern: Literal["multi_agent", "single_agent"] = "multi_agent"
    context_compaction_enabled: bool = os.getenv("CONTEXT_COMPACTION_ENABLED", "true").lower() == "true"
    
    # Failure Injection Simulation Hook (for testing and eval harness)
    # Options: "none", "tool_unavailable", "malformed_rag", "timeout"
    failure_simulation_mode: str = "none"

settings = Settings()
