from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Dict, List

from ..models import ProtocolEntry
from .base import ProtocolStorage


class JSONLinesProtocolStorage(ProtocolStorage):
    def __init__(self, file_path: str = "protocol.jsonl") -> None:
        self._file_path = Path(file_path)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _serialize_entry(self, entry: ProtocolEntry) -> str:
        data = entry.model_dump()
        return json.dumps(data, default=str, ensure_ascii=False)

    def _sync_write(self, entry: ProtocolEntry) -> None:
        line = self._serialize_entry(entry)
        with self._lock:
            with self._file_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()

    def _sync_write_batch(self, entries: List[ProtocolEntry]) -> None:
        if not entries:
            return
        lines = [self._serialize_entry(entry) for entry in entries]
        with self._lock:
            with self._file_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
                handle.flush()

    def _sync_query(self, filters: Dict) -> List[ProtocolEntry]:
        with self._lock:
            if not self._file_path.exists():
                return []

            results: List[ProtocolEntry] = []
            with self._file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if filters:
                        if not all(data.get(key) == value for key, value in filters.items()):
                            continue
                    results.append(ProtocolEntry.model_validate(data))
        return results

    async def write(self, entry: ProtocolEntry) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync_write, entry)

    async def write_batch(self, entries: List[ProtocolEntry]) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync_write_batch, entries)

    async def read(self, session_id: str) -> List[ProtocolEntry]:
        return await self.query({"session_id": session_id})

    async def query(self, filters: Dict) -> List[ProtocolEntry]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_query, filters)

    def close(self) -> None:
        """No-op: JSONLines uses file handles opened per-write, no cleanup needed."""
        pass
