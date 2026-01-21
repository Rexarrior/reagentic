from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, List, Optional

from .models import ProtocolConfig, ProtocolEntry, ProtocolEventType


class ProtocolExtractor:
    def __init__(self, config: ProtocolConfig) -> None:
        self._config = config

    def _truncate(self, value: Any) -> Any:
        max_len = self._config.max_content_length
        if max_len is None:
            return value
        if isinstance(value, str):
            if len(value) <= max_len:
                return value
            return value[:max_len] + "..."
        return value

    def _serialize(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return self._truncate(value)
        try:
            json.dumps(value)
            return self._truncate(value) if isinstance(value, str) else value
        except TypeError:
            return self._truncate(str(value))

    def _get_agent_id(self, agent: Any) -> Optional[str]:
        if agent is None:
            return None
        if hasattr(agent, "id"):
            return str(agent.id)
        if hasattr(agent, "name"):
            return agent.name
        return None

    def _base_entry(
        self,
        event_type: ProtocolEventType,
        *,
        agent: Any = None,
        tool: Any = None,
        session_id: Optional[str] = None,
    ) -> ProtocolEntry:
        return ProtocolEntry(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            session_id=session_id,
            agent_name=getattr(agent, "name", None) if agent else None,
            agent_id=self._get_agent_id(agent),
            tool_name=getattr(tool, "name", None) if tool else None,
        )

    def _extract_session_id(self, context: Any) -> Optional[str]:
        if context is None:
            return None
        session_id = (
            getattr(context, "run_id", None)
            or getattr(context, "session_id", None)
            or getattr(context, "context_id", None)
        )
        return str(session_id) if session_id else None

    def extract_agent_start(self, agent: Any, context: Any) -> ProtocolEntry:
        session_id = self._extract_session_id(context)
        entry = self._base_entry(
            ProtocolEventType.AGENT_START,
            agent=agent,
            session_id=session_id,
        )
        return entry

    def extract_agent_end(self, agent: Any, output: Any, context: Any = None) -> ProtocolEntry:
        session_id = self._extract_session_id(context)
        entry = self._base_entry(ProtocolEventType.AGENT_END, agent=agent, session_id=session_id)
        if self._config.include_output:
            entry.output_data = self._serialize(output)
        return entry

    def extract_llm_start(self, agent: Any, system_prompt: Optional[str], input_items: List[Any], context: Any = None) -> ProtocolEntry:
        session_id = self._extract_session_id(context)
        entry = self._base_entry(ProtocolEventType.LLM_START, agent=agent, session_id=session_id)
        if self._config.include_input:
            entry.input_data = self._serialize(input_items)
        if self._config.include_prompts and self._config.allows_prompts():
            entry.system_prompt = self._serialize(system_prompt)
        return entry

    def extract_llm_end(self, agent: Any, response: Any, context: Any = None) -> ProtocolEntry:
        session_id = self._extract_session_id(context)
        entry = self._base_entry(ProtocolEventType.LLM_END, agent=agent, session_id=session_id)
        if self._config.include_output:
            entry.output_data = self._serialize(getattr(response, "output", None))
        usage = getattr(response, "usage", None)
        if usage is not None and hasattr(usage, "total_tokens"):
            entry.tokens_used = usage.total_tokens
        return entry

    def extract_tool_start(self, agent: Any, tool: Any, context: Any = None) -> ProtocolEntry:
        session_id = self._extract_session_id(context)
        entry = self._base_entry(ProtocolEventType.TOOL_START, agent=agent, tool=tool, session_id=session_id)
        return entry

    def extract_tool_end(self, agent: Any, tool: Any, result: Any, context: Any = None) -> ProtocolEntry:
        session_id = self._extract_session_id(context)
        entry = self._base_entry(ProtocolEventType.TOOL_END, agent=agent, tool=tool, session_id=session_id)
        if self._config.include_output:
            entry.output_data = self._serialize(result)
        return entry

    def extract_handoff(self, from_agent: Any, to_agent: Any, context: Any = None) -> ProtocolEntry:
        session_id = self._extract_session_id(context)
        entry = self._base_entry(ProtocolEventType.HANDOFF, agent=from_agent, session_id=session_id)
        if self._config.allows_metadata():
            entry.metadata = {
                "from_agent": getattr(from_agent, "name", None),
                "to_agent": getattr(to_agent, "name", None),
            }
        return entry

    def extract_trace(self, trace: Any, is_start: bool) -> ProtocolEntry:
        event_type = ProtocolEventType.TRACE_START if is_start else ProtocolEventType.TRACE_END
        entry = self._base_entry(event_type)
        if self._config.allows_tracing():
            entry.trace_id = getattr(trace, "trace_id", None)
            if self._config.allows_metadata():
                entry.metadata = {
                    "workflow_name": getattr(trace, "workflow_name", None),
                    "group_id": getattr(trace, "group_id", None),
                }
        return entry

    def extract_span(self, span: Any, is_start: bool) -> ProtocolEntry:
        event_type = ProtocolEventType.SPAN_START if is_start else ProtocolEventType.SPAN_END
        entry = self._base_entry(event_type)
        if self._config.allows_tracing():
            entry.span_id = getattr(span, "span_id", None)
            entry.trace_id = getattr(span, "trace_id", None)
        if self._config.allows_metadata():
            span_data = getattr(span, "span_data", None)
            entry.metadata = {
                "span_name": getattr(span, "name", None),
                "span_type": getattr(span_data, "type", None) if span_data else None,
            }
        return entry

    def extract_error(
        self,
        agent: Any,
        error: Exception,
        event_type: ProtocolEventType,
    ) -> ProtocolEntry:
        entry = self._base_entry(event_type, agent=agent)
        entry.error = self._serialize(str(error))
        return entry
