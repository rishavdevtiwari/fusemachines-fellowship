"""
Configuration management for ShopAssist AI Enterprise Assistant.
Loads environment variables with robust defaults for local, docker, and cloud environments.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
KB_DIR = DATA_DIR / "knowledge_base"

class Settings(BaseModel):
    # App Settings
    app_name: str = "ShopAssist AI"
    app_version: str = "1.0.0"
    environment: str = os.getenv("ENVIRONMENT", "development")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", 8000))
    ui_port: int = int(os.getenv("UI_PORT", 8501))
    
    # LLM Provider Configuration
    # Supported: "gemini", "openai", "claude", "vllm", "mock"
    primary_provider: str = os.getenv("PRIMARY_PROVIDER", "gemini")
    fallback_provider: str = os.getenv("FALLBACK_PROVIDER", "mock")
    
    # API Keys
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # Models
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
    
    # Performance & Reliability
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    cache_max_size: int = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    rate_limit_requests_per_minute: int = int(os.getenv("RATE_LIMIT_RPM", "60"))
    max_retry_attempts: int = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
    retry_base_delay_seconds: float = float(os.getenv("RETRY_BASE_DELAY", "0.5"))
    
    # ONNX Router
    use_quantized_onnx: bool = os.getenv("USE_QUANTIZED_ONNX", "true").lower() == "true"
    onnx_threads: int = int(os.getenv("ONNX_THREADS", "4"))
    
    # File Paths
    data_dir: Path = DATA_DIR
    kb_dir: Path = KB_DIR
    models_dir: Path = MODELS_DIR
    orders_file: Path = DATA_DIR / "sample_orders.json"
    onnx_fp32_path: Path = MODELS_DIR / "intent_classifier.onnx"
    onnx_int8_path: Path = MODELS_DIR / "intent_classifier_quantized.onnx"
    metadata_path: Path = MODELS_DIR / "model_metadata.json"

settings = Settings()
