"""
LRU Model Pool — keeps ML models in memory across analysis runs.

All four Stage 2 models total ~1.92 GB, which comfortably fits on any
Apple Silicon Mac.  The pool loads models lazily on first use, then
keeps them resident so subsequent analyses skip the expensive disk-load.

A background daemon evicts models that have been idle for 10 minutes,
and an LRU policy kicks in if the memory ceiling is exceeded.
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

import psutil
import torch


# ─── Model entry ─────────────────────────────────────────────────────────────

class _ModelEntry:
    __slots__ = ("obj", "size_bytes", "last_used", "in_use")

    def __init__(self, obj: Any, size_bytes: int):
        self.obj = obj
        self.size_bytes = size_bytes
        self.last_used = time.time()
        self.in_use = False


# ─── Pool ────────────────────────────────────────────────────────────────────

class ModelPool:
    _instance: Optional["ModelPool"] = None

    # Known model sizes (approximate, for eviction decisions)
    MODEL_SIZES: dict[str, int] = {
        "audio_emotion":  1_200_000_000,   # wav2vec2   ~1.2 GB
        "text_emotion":     400_000_000,   # BERT       ~400 MB
        "sentiment":         67_000_000,   # DistilBERT  ~67 MB
        "zero_shot":        255_000_000,   # DistilBERT ~255 MB
    }
    DEFAULT_SIZE = 500_000_000  # fallback for unknown models

    def __init__(self, max_memory_bytes: Optional[int] = None, idle_timeout: float = 600.0):
        self._models: OrderedDict[str, _ModelEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._idle_timeout = idle_timeout  # seconds (default 10 min)

        if max_memory_bytes is not None:
            self._max_bytes = max_memory_bytes
        else:
            # Reserve 4 GB for system + app, use the rest for models (min 2 GB)
            total_ram = psutil.virtual_memory().total
            self._max_bytes = max(2 * 1024**3, total_ram - 4 * 1024**3)

        # Background eviction daemon
        self._shutdown = threading.Event()
        self._daemon = threading.Thread(target=self._eviction_loop, daemon=True)
        self._daemon.start()

    # ── Singleton ────────────────────────────────────────────────────────────

    @classmethod
    def get(cls) -> "ModelPool":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Public API ───────────────────────────────────────────────────────────

    def acquire(self, name: str, loader: Callable[[], Any]) -> Any:
        """
        Get a model by *name*.  If not cached, call *loader()* to create it.
        Evicts LRU models first if the memory ceiling would be exceeded.
        Returns whatever the loader returned (model, tuple of model+tokenizer, etc).
        """
        with self._lock:
            if name in self._models:
                entry = self._models[name]
                entry.last_used = time.time()
                entry.in_use = True
                self._models.move_to_end(name)   # refresh LRU position
                return entry.obj

        # Load outside the lock (loading is slow, don't block other threads)
        size = self.MODEL_SIZES.get(name, self.DEFAULT_SIZE)

        with self._lock:
            # Double-check: another thread may have loaded it while we waited
            if name in self._models:
                entry = self._models[name]
                entry.last_used = time.time()
                entry.in_use = True
                self._models.move_to_end(name)
                return entry.obj

            self._make_room_locked(size)

        obj = loader()

        with self._lock:
            entry = _ModelEntry(obj, size)
            entry.in_use = True
            self._models[name] = entry
            self._models.move_to_end(name)

        return obj

    def release(self, name: str) -> None:
        """Mark a model as no longer in active use (eligible for eviction)."""
        with self._lock:
            entry = self._models.get(name)
            if entry is not None:
                entry.in_use = False
                entry.last_used = time.time()

    def evict(self, name: str) -> None:
        """Explicitly remove a model from the pool."""
        with self._lock:
            self._evict_locked(name)

    def clear(self) -> None:
        """Remove all models and free memory.  Called on shutdown."""
        with self._lock:
            for name in list(self._models):
                self._evict_locked(name)
        self._flush_gpu_cache()

    def shutdown(self) -> None:
        """Stop the eviction daemon and clear models."""
        self._shutdown.set()
        self.clear()

    @property
    def loaded(self) -> list[str]:
        """Names of currently loaded models (for debugging)."""
        with self._lock:
            return list(self._models.keys())

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return sum(e.size_bytes for e in self._models.values())

    # ── Internal ─────────────────────────────────────────────────────────────

    def _make_room_locked(self, needed: int) -> None:
        """Evict LRU models (oldest first) until *needed* bytes can fit."""
        current = sum(e.size_bytes for e in self._models.values())
        for name in list(self._models):
            if current + needed <= self._max_bytes:
                break
            entry = self._models[name]
            if entry.in_use:
                continue  # never evict a model that's being used
            current -= entry.size_bytes
            self._evict_locked(name)

    def _evict_locked(self, name: str) -> None:
        entry = self._models.pop(name, None)
        if entry is not None:
            del entry.obj

    def _eviction_loop(self) -> None:
        """Background thread: evict models idle for longer than _idle_timeout."""
        while not self._shutdown.is_set():
            self._shutdown.wait(60)  # check every 60 seconds
            if self._shutdown.is_set():
                break
            with self._lock:
                now = time.time()
                to_evict = [
                    name for name, entry in self._models.items()
                    if not entry.in_use and (now - entry.last_used) > self._idle_timeout
                ]
                for name in to_evict:
                    self._evict_locked(name)
            if to_evict:
                self._flush_gpu_cache()

    @staticmethod
    def _flush_gpu_cache() -> None:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
