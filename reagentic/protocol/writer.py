from __future__ import annotations

import asyncio
import logging
import threading
from typing import List, Optional

from .models import ProtocolEntry
from .storage.base import ProtocolStorage

logger = logging.getLogger(__name__)


class ProtocolWriter:
    def __init__(self, storage: ProtocolStorage, buffer_size: int = 100) -> None:
        self._storage = storage
        self._buffer_size = max(0, buffer_size)
        self._buffer: List[ProtocolEntry] = []
        self._lock: Optional[asyncio.Lock] = None
        self._lock_loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock_init = threading.Lock()
        self._sync_lock = threading.Lock()

    def _get_lock(self) -> asyncio.Lock:
        current_loop = asyncio.get_running_loop()
        with self._lock_init:
            if self._lock is None or self._lock_loop is not current_loop:
                self._lock = asyncio.Lock()
                self._lock_loop = current_loop
            return self._lock

    async def write(self, entry: ProtocolEntry) -> None:
        try:
            async with self._get_lock():
                if self._buffer_size == 0:
                    await self._storage.write(entry)
                    return
                self._buffer.append(entry)
                if len(self._buffer) >= self._buffer_size:
                    await self._storage.write_batch(self._buffer)
                    self._buffer.clear()
        except Exception as e:
            logger.error(f"Failed to write protocol entry: {e}")

    async def flush(self) -> None:
        try:
            async with self._get_lock():
                if not self._buffer:
                    return
                await self._storage.write_batch(self._buffer)
                self._buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush protocol buffer: {e}")

    def write_batch_sync(self, entries: List[ProtocolEntry]) -> None:
        """Synchronous batch write, bypasses buffer for sync queue flush."""
        if not entries:
            return
        try:
            asyncio.run(self._storage.write_batch(entries))
        except Exception as e:
            logger.error(f"Failed to write batch sync: {e}")

    def close(self) -> None:
        with self._sync_lock:
            if self._buffer:
                try:
                    loop = asyncio.get_running_loop()
                    future = asyncio.run_coroutine_threadsafe(self.flush(), loop)
                    future.result(timeout=5.0)
                except RuntimeError:
                    try:
                        asyncio.run(self.flush())
                    except Exception as e:
                        logger.error(f"Failed to flush on close: {e}")
                except Exception as e:
                    logger.error(f"Failed to flush on close: {e}")
        self._storage.close()
