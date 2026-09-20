"""
Thread-Safe LRU + TTL Cache Implementation for ShopAssist AI (Week 16).
Provides sub-millisecond repeated query responses and cache management.
"""

import time
import json
import hashlib
from collections import OrderedDict
from threading import Lock
from typing import Dict, Any, Optional

from src.config import settings

class LRUTTLCache:
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = Lock()
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0

    def generate_key(self, prefix: str, payload: Any) -> str:
        serialized = json.dumps(payload, sort_keys=True, default=str)
        hash_digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"{prefix}:{hash_digest}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            entry = self._cache[key]
            now = time.time()
            if now > entry["expires_at"]:
                del self._cache[key]
                self.misses += 1
                return None

            self._cache.move_to_end(key)
            self.hits += 1
            return entry["data"]

    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        duration = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + duration
        with self._lock:
            if key in self._cache:
                self._cache[key] = {"data": value, "expires_at": expires_at}
                self._cache.move_to_end(key)
                return

            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
                self.evictions += 1

            self._cache[key] = {"data": value, "expires_at": expires_at}

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            hit_ratio = (self.hits / total) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": round(hit_ratio, 4),
                "evictions": self.evictions
            }

cache_service = LRUTTLCache(max_size=settings.cache_max_size, default_ttl=settings.cache_ttl_seconds)
