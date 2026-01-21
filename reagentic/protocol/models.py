from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ProtocolEventType(str, Enum):
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    HANDOFF = "handoff"
    TRACE_START = "trace_start"
    TRACE_END = "trace_end"
    SPAN_START = "span_start"
    SPAN_END = "span_end"


class ProtocolDetailLevel(str, Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    FULL = "full"


class ProtocolConfig(BaseModel):
    detail_level: ProtocolDetailLevel = ProtocolDetailLevel.STANDARD
    include_input: bool = True
    include_output: bool = True
    include_prompts: bool = True
    include_intermediate: bool = False
    max_content_length: Optional[int] = None

    def allows_prompts(self) -> bool:
        return self.detail_level in (ProtocolDetailLevel.STANDARD, ProtocolDetailLevel.FULL)

    def allows_intermediate(self) -> bool:
        return self.detail_level == ProtocolDetailLevel.FULL or self.include_intermediate

    def allows_metadata(self) -> bool:
        return self.detail_level == ProtocolDetailLevel.FULL

    def allows_tracing(self) -> bool:
        return self.detail_level == ProtocolDetailLevel.FULL


class ProtocolEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: ProtocolEventType
    session_id: Optional[str] = None

    agent_name: Optional[str] = None
    agent_id: Optional[str] = None
    tool_name: Optional[str] = None

    input_data: Optional[Any] = None
    output_data: Optional[Any] = None
    system_prompt: Optional[str] = None
    tokens_used: Optional[int] = None
    duration_ms: Optional[float] = None

    intermediate_steps: Optional[List[Any]] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
