import asyncio
import os
import json
from typing import Any, Optional

REDIS_URL = os.getenv("REDIS_URL")

try:
    import redis.asyncio as redis_async
    _REDIS_AVAILABLE = True
except Exception:
    redis_async = None
    _REDIS_AVAILABLE = False


class IdempotencyStore:
    """Simple pluggable idempotency store.

    If redis is available and REDIS_URL is provided, it will use Redis.
    Otherwise it falls back to an in-memory store with TTL cleanup. This
    provides basic dedupe semantics for development and can be upgraded to
    a production Redis instance by setting REDIS_URL and installing redis.
    """

    def __init__(self):
        self._use_redis = _REDIS_AVAILABLE and bool(REDIS_URL)
        if self._use_redis:
            self._client = redis_async.from_url(REDIS_URL)
        else:
            # in-memory store: key -> (value_json, expire_at)
            self._store = {}
            self._lock = asyncio.Lock()
            # start background cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def get(self, key: str) -> Optional[Any]:
        if self._use_redis:
            raw = await self._client.get(key)
            if not raw:
                return None
            try:
                return json.loads(raw)
            except Exception:
                return None

        async with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            value_json, expire_at = entry
            if expire_at and expire_at < asyncio.get_event_loop().time():
                del self._store[key]
                return None
            try:
                return json.loads(value_json)
            except Exception:
                return None

    async def set(self, key: str, value: Any, ttl_seconds: int = 24 * 3600):
        raw = json.dumps(value)
        if self._use_redis:
            await self._client.set(key, raw, ex=ttl_seconds)
            return

        async with self._lock:
            expire_at = asyncio.get_event_loop().time() + ttl_seconds if ttl_seconds else None
            self._store[key] = (raw, expire_at)

    async def _cleanup_loop(self):
        try:
            while True:
                await asyncio.sleep(60)
                now = asyncio.get_event_loop().time()
                async with self._lock:
                    keys_to_delete = [k for k, (_, exp) in self._store.items() if exp and exp < now]
                    for k in keys_to_delete:
                        del self._store[k]
        except asyncio.CancelledError:
            return


# Singleton instance
_STORE: Optional[IdempotencyStore] = None

def get_idempotency_store() -> IdempotencyStore:
    global _STORE
    if _STORE is None:
        _STORE = IdempotencyStore()
    return _STORE
