"""
High-Performance LRU + TTL Prompt & Response Cache for ShopAssist AI.
Reduces LLM inference latency, saves compute tokens, and tracks hit/miss telemetry.
"""

import time
import hashlib
import json
from typing import Any, Optional, Dict, Tuple
from collections import OrderedDict
import threading

from src.config import settings

class CacheItem:
    def __init__(self, value: Any, ttl_seconds: int):
        self.value = value
        self.expires_at = time.time() + ttl_seconds

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

class LRUTTLCache:
    """
    Thread-safe Least Recently Used (LRU) Cache with Time-To-Live (TTL).
    """
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheItem] = OrderedDict()
        self._lock = threading.Lock()
        
        # Telemetry metrics
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0

    @staticmethod
    def generate_key(prefix: str, payload: Dict[str, Any]) -> str:
        """Computes deterministic SHA-256 hash for cache key."""
        serialized = json.dumps(payload, sort_keys=True, default=str)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}:{digest}"

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
                
            item = self._cache[key]
            if item.is_expired():
                del self._cache[key]
                self.misses += 1
                return None
                
            # Move to MRU end
            self._cache.move_to_end(key)
            self.hits += 1
            return item.value

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        with self._lock:
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
            
            # Check capacity
            if key not in self._cache and len(self._cache) >= self.max_size:
                # Evict oldest LRU item
                self._cache.popitem(last=False)
                self.evictions += 1
                
            self._cache[key] = CacheItem(value, ttl)
            self._cache.move_to_end(key)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            hit_ratio = (self.hits / total) if total > 0 else 0.0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "total_requests": total,
                "hit_ratio_percent": round(hit_ratio * 100, 2),
                "current_size": len(self._cache),
                "max_size": self.max_size,
                "evictions": self.evictions
            }

cache_service = LRUTTLCache(
    max_size=settings.cache_max_size,
    default_ttl=settings.cache_ttl_seconds
)
