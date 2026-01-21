from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional

from ..models import ProtocolEntry
from .base import ProtocolStorage


class SQLiteProtocolStorage(ProtocolStorage):
    _KNOWN_COLUMNS = {
        "event_type",
        "session_id",
        "agent_name",
        "agent_id",
        "tool_name",
        "trace_id",
        "span_id",
    }

    def __init__(self, db_path: str = "protocol.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False is safe because all operations are protected
        # by self._lock, ensuring only one thread accesses the connection at a time
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS protocol_entries (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                session_id TEXT,
                agent_name TEXT,
                agent_id TEXT,
                tool_name TEXT,
                trace_id TEXT,
                span_id TEXT,
                entry_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_protocol_session ON protocol_entries(session_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_protocol_event ON protocol_entries(event_type)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_protocol_agent ON protocol_entries(agent_name)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_protocol_tool ON protocol_entries(tool_name)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_protocol_trace ON protocol_entries(trace_id)"
        )
        self._conn.commit()

    def _serialize_entry(self, entry: ProtocolEntry) -> str:
        data = entry.model_dump()
        return json.dumps(data, default=str, ensure_ascii=False)

    def _row_from_entry(self, entry: ProtocolEntry) -> tuple:
        return (
            entry.id,
            entry.timestamp.isoformat(),
            entry.event_type.value,
            entry.session_id,
            entry.agent_name,
            entry.agent_id,
            entry.tool_name,
            entry.trace_id,
            entry.span_id,
            self._serialize_entry(entry),
        )

    def _sync_write(self, entry: ProtocolEntry) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO protocol_entries (
                    id, timestamp, event_type, session_id, agent_name, agent_id, tool_name, trace_id, span_id, entry_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._row_from_entry(entry),
            )
            self._conn.commit()

    def _sync_write_batch(self, entries: List[ProtocolEntry]) -> None:
        if not entries:
            return
        with self._lock:
            rows = [self._row_from_entry(entry) for entry in entries]
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO protocol_entries (
                    id, timestamp, event_type, session_id, agent_name, agent_id, tool_name, trace_id, span_id, entry_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()

    def _sync_read(self, session_id: str) -> List[ProtocolEntry]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT entry_json FROM protocol_entries
                WHERE session_id = ?
                ORDER BY timestamp ASC
                """,
                (session_id,),
            ).fetchall()
        return [ProtocolEntry.model_validate(json.loads(row[0])) for row in rows]

    def _sync_query(self, filters: Dict) -> List[ProtocolEntry]:
        with self._lock:
            if not filters:
                rows = self._conn.execute(
                    "SELECT entry_json FROM protocol_entries ORDER BY timestamp ASC"
                ).fetchall()
                return [ProtocolEntry.model_validate(json.loads(row[0])) for row in rows]

            known_filters = {k: v for k, v in filters.items() if k in self._KNOWN_COLUMNS}
            unknown_filters = {k: v for k, v in filters.items() if k not in self._KNOWN_COLUMNS}

            query = "SELECT entry_json FROM protocol_entries"
            params: List[Optional[str]] = []
            if known_filters:
                clauses = []
                for key, value in known_filters.items():
                    clauses.append(f"{key} = ?")
                    params.append(value)
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY timestamp ASC"
            rows = self._conn.execute(query, params).fetchall()

        entries = [ProtocolEntry.model_validate(json.loads(row[0])) for row in rows]
        if not unknown_filters:
            return entries

        def _match(entry: ProtocolEntry) -> bool:
            data = entry.model_dump()
            return all(data.get(k) == v for k, v in unknown_filters.items())

        return [entry for entry in entries if _match(entry)]

    async def write(self, entry: ProtocolEntry) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync_write, entry)

    async def write_batch(self, entries: List[ProtocolEntry]) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync_write_batch, entries)

    async def read(self, session_id: str) -> List[ProtocolEntry]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_read, session_id)

    async def query(self, filters: Dict) -> List[ProtocolEntry]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_query, filters)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
