"""
API endpoints for Protocol Viewer.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from reagentic.protocol import (
    ProtocolEntry,
    ProtocolEventType,
    SQLiteProtocolStorage,
    JSONLinesProtocolStorage,
)
from reagentic.protocol.storage.base import ProtocolStorage


class SessionInfo(BaseModel):
    """Session summary information."""
    id: str
    agent_name: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[float]
    event_count: int
    has_error: bool
    total_tokens: Optional[int]


class SessionDetail(BaseModel):
    """Session with all its entries."""
    session: SessionInfo
    entries: List[Dict[str, Any]]


class StatsResponse(BaseModel):
    """Overall statistics."""
    total_sessions: int
    total_events: int
    total_tokens: int
    total_duration_ms: float
    event_type_counts: Dict[str, int]


router = APIRouter(prefix="/api")

# Global storage instance (set by server.py)
_storage: Optional[ProtocolStorage] = None


def set_storage(storage: ProtocolStorage) -> None:
    """Set the storage instance for API endpoints."""
    global _storage
    _storage = storage


def get_storage() -> ProtocolStorage:
    """Get the storage instance."""
    if _storage is None:
        raise HTTPException(status_code=500, detail="Storage not initialized")
    return _storage


def _group_entries_into_sessions(entries: List[ProtocolEntry]) -> Dict[str, List[ProtocolEntry]]:
    """
    Group entries into sessions.
    
    If session_id is set, use it. Otherwise, group by agent_id + 5-minute time window.
    """
    sessions: Dict[str, List[ProtocolEntry]] = {}
    
    # First pass: group by explicit session_id
    no_session_entries: List[ProtocolEntry] = []
    for entry in entries:
        if entry.session_id:
            if entry.session_id not in sessions:
                sessions[entry.session_id] = []
            sessions[entry.session_id].append(entry)
        else:
            no_session_entries.append(entry)
    
    # Second pass: group entries without session_id by agent + time window
    if no_session_entries:
        # Sort by timestamp
        no_session_entries.sort(key=lambda e: e.timestamp)
        
        current_session_id = None
        current_agent = None
        current_window_start: Optional[datetime] = None
        session_counter = 0
        
        for entry in no_session_entries:
            agent_key = entry.agent_id or entry.agent_name or "unknown"
            
            # Check if we need a new session
            need_new_session = (
                current_agent != agent_key or
                current_window_start is None or
                (entry.timestamp - current_window_start) > timedelta(minutes=5)
            )
            
            if need_new_session:
                session_counter += 1
                current_session_id = f"auto-{agent_key}-{session_counter}"
                current_agent = agent_key
                current_window_start = entry.timestamp
                sessions[current_session_id] = []
            
            sessions[current_session_id].append(entry)
    
    return sessions


def _build_session_info(session_id: str, entries: List[ProtocolEntry]) -> SessionInfo:
    """Build session info from entries."""
    if not entries:
        return SessionInfo(
            id=session_id,
            agent_name=None,
            start_time=datetime.now(),
            end_time=None,
            duration_ms=None,
            event_count=0,
            has_error=False,
            total_tokens=None,
        )
    
    # Sort by timestamp
    sorted_entries = sorted(entries, key=lambda e: e.timestamp)
    
    # Get agent name from first entry
    agent_name = next(
        (e.agent_name for e in sorted_entries if e.agent_name),
        None
    )
    
    # Calculate duration
    start_time = sorted_entries[0].timestamp
    end_time = sorted_entries[-1].timestamp
    duration_ms = (end_time - start_time).total_seconds() * 1000
    
    # Check for errors
    has_error = any(e.error for e in entries)
    
    # Sum tokens
    total_tokens = sum(e.tokens_used or 0 for e in entries)
    
    return SessionInfo(
        id=session_id,
        agent_name=agent_name,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        event_count=len(entries),
        has_error=has_error,
        total_tokens=total_tokens if total_tokens > 0 else None,
    )


@router.get("/sessions", response_model=List[SessionInfo])
async def get_sessions() -> List[SessionInfo]:
    """Get list of all sessions."""
    storage = get_storage()
    all_entries = await storage.query({})
    
    sessions = _group_entries_into_sessions(all_entries)
    
    session_infos = [
        _build_session_info(sid, entries)
        for sid, entries in sessions.items()
    ]
    
    # Sort by start time descending (newest first)
    session_infos.sort(key=lambda s: s.start_time, reverse=True)
    
    return session_infos


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str) -> SessionDetail:
    """Get all entries for a specific session."""
    storage = get_storage()
    all_entries = await storage.query({})
    
    sessions = _group_entries_into_sessions(all_entries)
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    entries = sessions[session_id]
    session_info = _build_session_info(session_id, entries)
    
    # Sort entries by timestamp
    sorted_entries = sorted(entries, key=lambda e: e.timestamp)
    
    # Convert to dicts for JSON response
    entry_dicts = [entry.model_dump() for entry in sorted_entries]
    
    return SessionDetail(session=session_info, entries=entry_dicts)


@router.get("/entries/{entry_id}")
async def get_entry(entry_id: str) -> Dict[str, Any]:
    """Get a specific entry by ID."""
    storage = get_storage()
    all_entries = await storage.query({})
    
    for entry in all_entries:
        if entry.id == entry_id:
            return entry.model_dump()
    
    raise HTTPException(status_code=404, detail=f"Entry {entry_id} not found")


@router.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """Get overall statistics."""
    storage = get_storage()
    all_entries = await storage.query({})
    
    sessions = _group_entries_into_sessions(all_entries)
    
    # Count event types
    event_type_counts: Dict[str, int] = {}
    total_tokens = 0
    total_duration_ms = 0.0
    
    for entry in all_entries:
        event_type = entry.event_type.value
        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        
        if entry.tokens_used:
            total_tokens += entry.tokens_used
        
        if entry.duration_ms:
            total_duration_ms += entry.duration_ms
    
    return StatsResponse(
        total_sessions=len(sessions),
        total_events=len(all_entries),
        total_tokens=total_tokens,
        total_duration_ms=total_duration_ms,
        event_type_counts=event_type_counts,
    )


@router.get("/graph/{session_id}")
async def get_session_graph(session_id: str) -> Dict[str, Any]:
    """
    Get graph representation of a session for visualization.
    
    Returns nodes and edges for Vue Flow.
    """
    storage = get_storage()
    all_entries = await storage.query({})
    
    sessions = _group_entries_into_sessions(all_entries)
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    entries = sorted(sessions[session_id], key=lambda e: e.timestamp)
    
    nodes = []
    edges = []
    
    # Track agent nodes and their children
    agent_nodes: Dict[str, str] = {}  # agent_name -> node_id
    parent_stack: List[str] = []  # Stack of parent node IDs
    
    for i, entry in enumerate(entries):
        node_id = f"node-{entry.id}"
        
        # Determine node type and label
        event_type = entry.event_type.value
        
        if event_type == "agent_start":
            node_type = "agent"
            label = entry.agent_name or "Agent"
            agent_nodes[entry.agent_name or "unknown"] = node_id
            parent_stack.append(node_id)
        elif event_type == "agent_end":
            node_type = "agent_end"
            label = f"{entry.agent_name or 'Agent'} (end)"
            if parent_stack:
                parent_stack.pop()
        elif event_type in ("llm_start", "llm_end"):
            node_type = "llm"
            label = "LLM Call" if event_type == "llm_start" else "LLM Response"
        elif event_type in ("tool_start", "tool_end"):
            node_type = "tool"
            label = entry.tool_name or "Tool"
        elif event_type == "handoff":
            node_type = "handoff"
            label = "Handoff"
        else:
            node_type = "event"
            label = event_type
        
        node = {
            "id": node_id,
            "type": node_type,
            "data": {
                "label": label,
                "event_type": event_type,
                "entry_id": entry.id,
                "timestamp": entry.timestamp.isoformat(),
                "duration_ms": entry.duration_ms,
                "tokens": entry.tokens_used,
                "has_error": bool(entry.error),
            },
            "position": {"x": 0, "y": 0},  # Will be set by ELK layout
        }
        nodes.append(node)
        
        # Create edge from previous node
        if i > 0:
            prev_node_id = f"node-{entries[i-1].id}"
            edge = {
                "id": f"edge-{prev_node_id}-{node_id}",
                "source": prev_node_id,
                "target": node_id,
                "animated": event_type.endswith("_start"),
            }
            edges.append(edge)
    
    return {
        "nodes": nodes,
        "edges": edges,
    }
