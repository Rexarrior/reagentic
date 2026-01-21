from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from agents.lifecycle import RunHooks
from agents.tracing.processor_interface import TracingProcessor

from .extractor import ProtocolExtractor
from .models import ProtocolEntry
from .writer import ProtocolWriter

logger = logging.getLogger(__name__)

_STALE_THRESHOLD = timedelta(hours=1)


class ProtocolObserver(RunHooks, TracingProcessor):
    def __init__(self, extractor: ProtocolExtractor, writer: ProtocolWriter) -> None:
        self._extractor = extractor
        self._writer = writer
        self._start_times: Dict[str, datetime] = {}
        self._sync_queue: List[ProtocolEntry] = []
        self._sync_lock = threading.Lock()

    def _get_run_id(self, context: Any, agent: Any) -> str:
        """Extract a stable run identifier from context or fall back to agent id."""
        return str(getattr(context, "run_id", None) or id(agent))

    def _find_and_pop_oldest_start(self, prefix: str, end_timestamp: datetime) -> Optional[float]:
        """Find the oldest pending start time matching prefix and calculate duration."""
        matching = [(k, v) for k, v in self._start_times.items() if k.startswith(prefix)]
        if not matching:
            return None
        # Take oldest by timestamp (FIFO matching for concurrent calls)
        key = min(matching, key=lambda x: x[1])[0]
        return self._calculate_duration(key, end_timestamp)

    def _schedule_write(self, entry: ProtocolEntry) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._writer.write(entry))
        except RuntimeError:
            with self._sync_lock:
                self._sync_queue.append(entry)

    def _flush_sync_queue(self) -> None:
        with self._sync_lock:
            if not self._sync_queue:
                return
            entries = self._sync_queue.copy()
            self._sync_queue.clear()
        self._writer.write_batch_sync(entries)

    def _calculate_duration(self, key: str, end_timestamp: datetime) -> float | None:
        start = self._start_times.pop(key, None)

        # Lazy cleanup of stale entries to prevent memory leaks
        stale_cutoff = end_timestamp - _STALE_THRESHOLD
        stale_keys = [k for k, v in self._start_times.items() if v < stale_cutoff]
        for k in stale_keys:
            self._start_times.pop(k, None)

        if start is not None:
            delta = end_timestamp - start
            return delta.total_seconds() * 1000
        return None

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        entry = self._extractor.extract_agent_start(agent, context)
        run_id = self._get_run_id(context, agent)
        self._start_times[f"agent:{run_id}"] = entry.timestamp
        await self._writer.write(entry)

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        entry = self._extractor.extract_agent_end(agent, output, context)
        run_id = self._get_run_id(context, agent)
        entry.duration_ms = self._calculate_duration(f"agent:{run_id}", entry.timestamp)
        await self._writer.write(entry)

    async def on_llm_start(self, context: Any, agent: Any, system_prompt: Any, input_items: Any) -> None:
        entry = self._extractor.extract_llm_start(agent, system_prompt, input_items, context)
        run_id = self._get_run_id(context, agent)
        # Use entry.id in key to support multiple LLM calls per run
        self._start_times[f"llm:{run_id}:{entry.id}"] = entry.timestamp
        await self._writer.write(entry)

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        entry = self._extractor.extract_llm_end(agent, response, context)
        run_id = self._get_run_id(context, agent)
        # Find oldest pending LLM start for this run (FIFO matching)
        entry.duration_ms = self._find_and_pop_oldest_start(f"llm:{run_id}:", entry.timestamp)
        await self._writer.write(entry)

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        entry = self._extractor.extract_tool_start(agent, tool, context)
        run_id = self._get_run_id(context, agent)
        tool_name = getattr(tool, "name", None) or str(id(tool))
        # Use entry.id in key to support multiple calls to same tool per run
        self._start_times[f"tool:{run_id}:{tool_name}:{entry.id}"] = entry.timestamp
        await self._writer.write(entry)

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        entry = self._extractor.extract_tool_end(agent, tool, result, context)
        run_id = self._get_run_id(context, agent)
        tool_name = getattr(tool, "name", None) or str(id(tool))
        # Find oldest pending tool start for this run+tool (FIFO matching)
        entry.duration_ms = self._find_and_pop_oldest_start(f"tool:{run_id}:{tool_name}:", entry.timestamp)
        await self._writer.write(entry)

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        entry = self._extractor.extract_handoff(from_agent, to_agent, context)
        await self._writer.write(entry)

    def on_trace_start(self, trace: Any) -> None:
        entry = self._extractor.extract_trace(trace, is_start=True)
        trace_id = getattr(trace, "trace_id", None)
        if trace_id:
            self._start_times[f"trace:{trace_id}"] = entry.timestamp
        self._schedule_write(entry)

    def on_trace_end(self, trace: Any) -> None:
        entry = self._extractor.extract_trace(trace, is_start=False)
        trace_id = getattr(trace, "trace_id", None)
        if trace_id:
            entry.duration_ms = self._calculate_duration(f"trace:{trace_id}", entry.timestamp)
        self._schedule_write(entry)

    def on_span_start(self, span: Any) -> None:
        entry = self._extractor.extract_span(span, is_start=True)
        span_id = getattr(span, "span_id", None)
        if span_id:
            self._start_times[f"span:{span_id}"] = entry.timestamp
        self._schedule_write(entry)

    def on_span_end(self, span: Any) -> None:
        entry = self._extractor.extract_span(span, is_start=False)
        span_id = getattr(span, "span_id", None)
        if span_id:
            entry.duration_ms = self._calculate_duration(f"span:{span_id}", entry.timestamp)
        self._schedule_write(entry)

    def shutdown(self) -> None:
        self._flush_sync_queue()
        self._writer.close()

    def force_flush(self) -> None:
        self._flush_sync_queue()
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(self._writer.flush(), loop)
            future.result(timeout=5.0)
        except RuntimeError:
            try:
                asyncio.run(self._writer.flush())
            except Exception as e:
                logger.error(f"Failed to force flush protocol: {e}")
        except TimeoutError:
            logger.warning("Protocol flush timed out after 5 seconds, data may be lost")
        except Exception as e:
            logger.error(f"Failed to force flush protocol: {e}")
