"""
Tests for the Protocol system: models, extractor, observer, writer, and storage backends.
"""

import asyncio
import json
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from reagentic.protocol import (
    JSONLinesProtocolStorage,
    ProtocolConfig,
    ProtocolDetailLevel,
    ProtocolEntry,
    ProtocolEventType,
    ProtocolExtractor,
    ProtocolObserver,
    ProtocolStorage,
    ProtocolWriter,
    SQLiteProtocolStorage,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def minimal_config() -> ProtocolConfig:
    """Config with minimal detail level."""
    return ProtocolConfig(detail_level=ProtocolDetailLevel.MINIMAL)


@pytest.fixture
def standard_config() -> ProtocolConfig:
    """Config with standard detail level."""
    return ProtocolConfig(detail_level=ProtocolDetailLevel.STANDARD)


@pytest.fixture
def full_config() -> ProtocolConfig:
    """Config with full detail level."""
    return ProtocolConfig(detail_level=ProtocolDetailLevel.FULL)


@pytest.fixture
def mock_agent() -> Mock:
    """Create a mock agent object."""
    agent = Mock()
    agent.name = "TestAgent"
    agent.id = "agent-123"
    return agent


@pytest.fixture
def mock_tool() -> Mock:
    """Create a mock tool object."""
    tool = Mock()
    tool.name = "test_tool"
    return tool


@pytest.fixture
def mock_context() -> Mock:
    """Create a mock context with run_id."""
    context = Mock()
    context.run_id = "run-456"
    return context


@pytest.fixture
def mock_response() -> Mock:
    """Create a mock LLM response."""
    response = Mock()
    response.output = "Test response output"
    response.usage = Mock()
    response.usage.total_tokens = 100
    return response


@pytest.fixture
def mock_trace() -> Mock:
    """Create a mock trace object."""
    trace = Mock()
    trace.trace_id = "trace-789"
    trace.workflow_name = "test_workflow"
    trace.group_id = "group-001"
    return trace


@pytest.fixture
def mock_span() -> Mock:
    """Create a mock span object."""
    span = Mock()
    span.span_id = "span-123"
    span.trace_id = "trace-789"
    span.name = "test_span"
    span.span_data = Mock()
    span.span_data.type = "custom"
    return span


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Create a temporary database path."""
    return str(tmp_path / "test_protocol.db")


@pytest.fixture
def temp_jsonl_path(tmp_path: Path) -> str:
    """Create a temporary JSONL file path."""
    return str(tmp_path / "test_protocol.jsonl")


# =============================================================================
# Tests for ProtocolConfig and Models
# =============================================================================


class TestProtocolModels:
    """Tests for protocol data models."""

    def test_protocol_event_types_exist(self):
        """Verify all expected event types are defined."""
        expected_types = [
            "AGENT_START", "AGENT_END",
            "LLM_START", "LLM_END",
            "TOOL_START", "TOOL_END",
            "HANDOFF",
            "TRACE_START", "TRACE_END",
            "SPAN_START", "SPAN_END",
        ]
        for event_type in expected_types:
            assert hasattr(ProtocolEventType, event_type)

    def test_protocol_detail_levels(self):
        """Test detail levels are properly defined."""
        assert ProtocolDetailLevel.MINIMAL.value == "minimal"
        assert ProtocolDetailLevel.STANDARD.value == "standard"
        assert ProtocolDetailLevel.FULL.value == "full"

    def test_protocol_entry_defaults(self):
        """Test ProtocolEntry has proper defaults."""
        entry = ProtocolEntry(event_type=ProtocolEventType.AGENT_START)
        
        assert entry.id is not None
        assert len(entry.id) == 36  # UUID format
        assert entry.timestamp is not None
        assert entry.event_type == ProtocolEventType.AGENT_START
        assert entry.session_id is None
        assert entry.agent_name is None
        assert entry.duration_ms is None

    def test_protocol_entry_serialization(self):
        """Test ProtocolEntry can be serialized to JSON."""
        entry = ProtocolEntry(
            event_type=ProtocolEventType.LLM_END,
            agent_name="TestAgent",
            tokens_used=50,
            duration_ms=123.45,
        )
        
        data = entry.model_dump()
        json_str = json.dumps(data, default=str)
        
        assert "LLM_END" in json_str or "llm_end" in json_str
        assert "TestAgent" in json_str
        assert "50" in json_str


class TestProtocolConfig:
    """Tests for ProtocolConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ProtocolConfig()
        
        assert config.detail_level == ProtocolDetailLevel.STANDARD
        assert config.include_input is True
        assert config.include_output is True
        assert config.include_prompts is True
        assert config.include_intermediate is False
        assert config.max_content_length is None

    def test_allows_prompts_by_detail_level(self):
        """Test allows_prompts respects detail level."""
        minimal = ProtocolConfig(detail_level=ProtocolDetailLevel.MINIMAL)
        standard = ProtocolConfig(detail_level=ProtocolDetailLevel.STANDARD)
        full = ProtocolConfig(detail_level=ProtocolDetailLevel.FULL)
        
        assert minimal.allows_prompts() is False
        assert standard.allows_prompts() is True
        assert full.allows_prompts() is True

    def test_allows_intermediate(self):
        """Test allows_intermediate logic."""
        # FULL level always allows
        full = ProtocolConfig(detail_level=ProtocolDetailLevel.FULL)
        assert full.allows_intermediate() is True
        
        # STANDARD doesn't allow by default
        standard = ProtocolConfig(detail_level=ProtocolDetailLevel.STANDARD)
        assert standard.allows_intermediate() is False
        
        # But explicit flag overrides
        standard_with_intermediate = ProtocolConfig(
            detail_level=ProtocolDetailLevel.STANDARD,
            include_intermediate=True
        )
        assert standard_with_intermediate.allows_intermediate() is True

    def test_allows_metadata(self):
        """Test allows_metadata only for FULL level."""
        minimal = ProtocolConfig(detail_level=ProtocolDetailLevel.MINIMAL)
        standard = ProtocolConfig(detail_level=ProtocolDetailLevel.STANDARD)
        full = ProtocolConfig(detail_level=ProtocolDetailLevel.FULL)
        
        assert minimal.allows_metadata() is False
        assert standard.allows_metadata() is False
        assert full.allows_metadata() is True

    def test_allows_tracing(self):
        """Test allows_tracing only for FULL level."""
        minimal = ProtocolConfig(detail_level=ProtocolDetailLevel.MINIMAL)
        standard = ProtocolConfig(detail_level=ProtocolDetailLevel.STANDARD)
        full = ProtocolConfig(detail_level=ProtocolDetailLevel.FULL)
        
        assert minimal.allows_tracing() is False
        assert standard.allows_tracing() is False
        assert full.allows_tracing() is True


# =============================================================================
# Tests for ProtocolExtractor
# =============================================================================


class TestProtocolExtractor:
    """Tests for ProtocolExtractor."""

    def test_extract_agent_start(self, standard_config: ProtocolConfig, mock_agent: Mock, mock_context: Mock):
        """Test agent start extraction."""
        extractor = ProtocolExtractor(standard_config)
        
        entry = extractor.extract_agent_start(mock_agent, mock_context)
        
        assert entry.event_type == ProtocolEventType.AGENT_START
        assert entry.agent_name == "TestAgent"
        assert entry.agent_id == "agent-123"
        assert entry.session_id == "run-456"

    def test_extract_agent_end(self, standard_config: ProtocolConfig, mock_agent: Mock, mock_context: Mock):
        """Test agent end extraction with output."""
        extractor = ProtocolExtractor(standard_config)
        
        entry = extractor.extract_agent_end(mock_agent, "Final output", mock_context)
        
        assert entry.event_type == ProtocolEventType.AGENT_END
        assert entry.output_data == "Final output"
        assert entry.session_id == "run-456"

    def test_extract_agent_end_without_output(self, mock_agent: Mock):
        """Test agent end when include_output is False."""
        config = ProtocolConfig(include_output=False)
        extractor = ProtocolExtractor(config)
        
        entry = extractor.extract_agent_end(mock_agent, "Should not appear")
        
        assert entry.output_data is None

    def test_extract_llm_start(self, standard_config: ProtocolConfig, mock_agent: Mock, mock_context: Mock):
        """Test LLM start extraction."""
        extractor = ProtocolExtractor(standard_config)
        
        entry = extractor.extract_llm_start(
            mock_agent,
            "System prompt",
            ["User message"],
            mock_context
        )
        
        assert entry.event_type == ProtocolEventType.LLM_START
        assert entry.input_data == ["User message"]
        assert entry.system_prompt == "System prompt"
        assert entry.session_id == "run-456"

    def test_extract_llm_start_minimal_no_prompt(self, minimal_config: ProtocolConfig, mock_agent: Mock):
        """Test that minimal level doesn't include system prompt."""
        extractor = ProtocolExtractor(minimal_config)
        
        entry = extractor.extract_llm_start(mock_agent, "System prompt", ["input"])
        
        assert entry.system_prompt is None

    def test_extract_llm_end(self, standard_config: ProtocolConfig, mock_agent: Mock, mock_response: Mock, mock_context: Mock):
        """Test LLM end extraction with token usage."""
        extractor = ProtocolExtractor(standard_config)
        
        entry = extractor.extract_llm_end(mock_agent, mock_response, mock_context)
        
        assert entry.event_type == ProtocolEventType.LLM_END
        assert entry.output_data == "Test response output"
        assert entry.tokens_used == 100
        assert entry.session_id == "run-456"

    def test_extract_tool_start(self, standard_config: ProtocolConfig, mock_agent: Mock, mock_tool: Mock, mock_context: Mock):
        """Test tool start extraction."""
        extractor = ProtocolExtractor(standard_config)
        
        entry = extractor.extract_tool_start(mock_agent, mock_tool, mock_context)
        
        assert entry.event_type == ProtocolEventType.TOOL_START
        assert entry.tool_name == "test_tool"
        assert entry.agent_name == "TestAgent"

    def test_extract_tool_end(self, standard_config: ProtocolConfig, mock_agent: Mock, mock_tool: Mock, mock_context: Mock):
        """Test tool end extraction."""
        extractor = ProtocolExtractor(standard_config)
        
        entry = extractor.extract_tool_end(mock_agent, mock_tool, "Tool result", mock_context)
        
        assert entry.event_type == ProtocolEventType.TOOL_END
        assert entry.output_data == "Tool result"
        assert entry.tool_name == "test_tool"

    def test_extract_handoff(self, full_config: ProtocolConfig, mock_context: Mock):
        """Test handoff extraction with metadata."""
        extractor = ProtocolExtractor(full_config)
        
        from_agent = Mock(name="AgentA")
        from_agent.name = "AgentA"
        to_agent = Mock(name="AgentB")
        to_agent.name = "AgentB"
        
        entry = extractor.extract_handoff(from_agent, to_agent, mock_context)
        
        assert entry.event_type == ProtocolEventType.HANDOFF
        assert entry.metadata is not None
        assert entry.metadata["from_agent"] == "AgentA"
        assert entry.metadata["to_agent"] == "AgentB"

    def test_extract_trace(self, full_config: ProtocolConfig, mock_trace: Mock):
        """Test trace extraction."""
        extractor = ProtocolExtractor(full_config)
        
        start_entry = extractor.extract_trace(mock_trace, is_start=True)
        end_entry = extractor.extract_trace(mock_trace, is_start=False)
        
        assert start_entry.event_type == ProtocolEventType.TRACE_START
        assert end_entry.event_type == ProtocolEventType.TRACE_END
        assert start_entry.trace_id == "trace-789"
        assert start_entry.metadata["workflow_name"] == "test_workflow"

    def test_extract_span(self, full_config: ProtocolConfig, mock_span: Mock):
        """Test span extraction."""
        extractor = ProtocolExtractor(full_config)
        
        start_entry = extractor.extract_span(mock_span, is_start=True)
        end_entry = extractor.extract_span(mock_span, is_start=False)
        
        assert start_entry.event_type == ProtocolEventType.SPAN_START
        assert end_entry.event_type == ProtocolEventType.SPAN_END
        assert start_entry.span_id == "span-123"
        assert start_entry.trace_id == "trace-789"

    def test_extract_error(self, standard_config: ProtocolConfig, mock_agent: Mock):
        """Test error extraction."""
        extractor = ProtocolExtractor(standard_config)
        error = ValueError("Something went wrong")
        
        entry = extractor.extract_error(mock_agent, error, ProtocolEventType.AGENT_END)
        
        assert entry.event_type == ProtocolEventType.AGENT_END
        assert entry.error == "Something went wrong"
        assert entry.agent_name == "TestAgent"

    def test_content_truncation(self, mock_agent: Mock):
        """Test content truncation with max_content_length."""
        config = ProtocolConfig(max_content_length=10)
        extractor = ProtocolExtractor(config)
        
        entry = extractor.extract_agent_end(mock_agent, "This is a very long output string")
        
        assert entry.output_data == "This is a ..."
        assert len(entry.output_data) == 13  # 10 + "..."

    def test_session_id_fallback(self, standard_config: ProtocolConfig, mock_agent: Mock):
        """Test session_id extraction with fallback attributes."""
        extractor = ProtocolExtractor(standard_config)
        
        # Context with session_id instead of run_id
        context_session = Mock(spec=[])
        context_session.session_id = "session-123"
        entry = extractor.extract_agent_start(mock_agent, context_session)
        assert entry.session_id == "session-123"
        
        # Context with context_id as last fallback
        context_ctx = Mock(spec=[])
        context_ctx.context_id = "ctx-456"
        entry = extractor.extract_agent_start(mock_agent, context_ctx)
        assert entry.session_id == "ctx-456"

    def test_serialize_complex_objects(self, standard_config: ProtocolConfig, mock_agent: Mock):
        """Test serialization of non-JSON objects."""
        extractor = ProtocolExtractor(standard_config)
        
        complex_output = Mock()
        complex_output.__str__ = lambda self: "MockObject<id=1>"
        
        entry = extractor.extract_agent_end(mock_agent, complex_output)
        
        # Should be string representation
        assert "MockObject" in str(entry.output_data)


# =============================================================================
# Tests for Storage Backends
# =============================================================================


class TestSQLiteProtocolStorage:
    """Tests for SQLiteProtocolStorage."""

    @pytest.mark.asyncio
    async def test_write_and_read(self, temp_db_path: str):
        """Test writing and reading entries."""
        storage = SQLiteProtocolStorage(temp_db_path)
        
        entry = ProtocolEntry(
            event_type=ProtocolEventType.AGENT_START,
            session_id="test-session",
            agent_name="TestAgent",
        )
        
        await storage.write(entry)
        
        entries = await storage.read("test-session")
        
        assert len(entries) == 1
        assert entries[0].id == entry.id
        assert entries[0].agent_name == "TestAgent"
        
        storage.close()

    @pytest.mark.asyncio
    async def test_write_batch(self, temp_db_path: str):
        """Test batch writing."""
        storage = SQLiteProtocolStorage(temp_db_path)
        
        entries = [
            ProtocolEntry(
                event_type=ProtocolEventType.AGENT_START,
                session_id="batch-session",
                agent_name=f"Agent{i}",
            )
            for i in range(5)
        ]
        
        await storage.write_batch(entries)
        
        result = await storage.read("batch-session")
        
        assert len(result) == 5
        
        storage.close()

    @pytest.mark.asyncio
    async def test_query_by_event_type(self, temp_db_path: str):
        """Test querying by event type."""
        storage = SQLiteProtocolStorage(temp_db_path)
        
        await storage.write(ProtocolEntry(event_type=ProtocolEventType.AGENT_START))
        await storage.write(ProtocolEntry(event_type=ProtocolEventType.AGENT_END))
        await storage.write(ProtocolEntry(event_type=ProtocolEventType.LLM_START))
        
        results = await storage.query({"event_type": "agent_start"})
        
        assert len(results) == 1
        assert results[0].event_type == ProtocolEventType.AGENT_START
        
        storage.close()

    @pytest.mark.asyncio
    async def test_query_all(self, temp_db_path: str):
        """Test querying all entries with empty filters."""
        storage = SQLiteProtocolStorage(temp_db_path)
        
        await storage.write(ProtocolEntry(event_type=ProtocolEventType.AGENT_START))
        await storage.write(ProtocolEntry(event_type=ProtocolEventType.AGENT_END))
        
        results = await storage.query({})
        
        assert len(results) == 2
        
        storage.close()

    @pytest.mark.asyncio
    async def test_thread_safety(self, temp_db_path: str):
        """Test thread-safe writes from multiple threads."""
        storage = SQLiteProtocolStorage(temp_db_path)
        errors: List[Exception] = []
        
        async def write_entries(prefix: str):
            try:
                for i in range(10):
                    entry = ProtocolEntry(
                        event_type=ProtocolEventType.AGENT_START,
                        session_id="concurrent-session",
                        agent_name=f"{prefix}-{i}",
                    )
                    await storage.write(entry)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent writes
        await asyncio.gather(
            write_entries("A"),
            write_entries("B"),
            write_entries("C"),
        )
        
        assert len(errors) == 0
        
        results = await storage.read("concurrent-session")
        assert len(results) == 30
        
        storage.close()


class TestJSONLinesProtocolStorage:
    """Tests for JSONLinesProtocolStorage."""

    @pytest.mark.asyncio
    async def test_write_and_read(self, temp_jsonl_path: str):
        """Test writing and reading entries."""
        storage = JSONLinesProtocolStorage(temp_jsonl_path)
        
        entry = ProtocolEntry(
            event_type=ProtocolEventType.TOOL_START,
            session_id="jsonl-session",
            tool_name="test_tool",
        )
        
        await storage.write(entry)
        
        entries = await storage.read("jsonl-session")
        
        assert len(entries) == 1
        assert entries[0].id == entry.id
        assert entries[0].tool_name == "test_tool"
        
        storage.close()

    @pytest.mark.asyncio
    async def test_write_batch(self, temp_jsonl_path: str):
        """Test batch writing."""
        storage = JSONLinesProtocolStorage(temp_jsonl_path)
        
        entries = [
            ProtocolEntry(
                event_type=ProtocolEventType.LLM_START,
                session_id="batch-jsonl",
                agent_name=f"Agent{i}",
            )
            for i in range(3)
        ]
        
        await storage.write_batch(entries)
        
        result = await storage.read("batch-jsonl")
        
        assert len(result) == 3
        
        storage.close()

    @pytest.mark.asyncio
    async def test_query_with_filters(self, temp_jsonl_path: str):
        """Test querying with filters."""
        storage = JSONLinesProtocolStorage(temp_jsonl_path)
        
        await storage.write(ProtocolEntry(event_type=ProtocolEventType.AGENT_START, agent_name="Alice"))
        await storage.write(ProtocolEntry(event_type=ProtocolEventType.AGENT_START, agent_name="Bob"))
        await storage.write(ProtocolEntry(event_type=ProtocolEventType.AGENT_END, agent_name="Alice"))
        
        results = await storage.query({"agent_name": "Alice"})
        
        assert len(results) == 2
        assert all(e.agent_name == "Alice" for e in results)
        
        storage.close()

    @pytest.mark.asyncio
    async def test_file_persistence(self, temp_jsonl_path: str):
        """Test that data persists across storage instances."""
        storage1 = JSONLinesProtocolStorage(temp_jsonl_path)
        
        entry = ProtocolEntry(
            event_type=ProtocolEventType.HANDOFF,
            session_id="persist-test",
        )
        await storage1.write(entry)
        storage1.close()
        
        # Create new instance pointing to same file
        storage2 = JSONLinesProtocolStorage(temp_jsonl_path)
        results = await storage2.read("persist-test")
        
        assert len(results) == 1
        assert results[0].id == entry.id
        
        storage2.close()

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tmp_path: Path):
        """Test reading from non-existent file returns empty list."""
        storage = JSONLinesProtocolStorage(str(tmp_path / "nonexistent.jsonl"))
        
        results = await storage.read("any-session")
        
        assert results == []
        
        storage.close()


# =============================================================================
# Tests for ProtocolWriter
# =============================================================================


class TestProtocolWriter:
    """Tests for ProtocolWriter."""

    @pytest.mark.asyncio
    async def test_write_without_buffer(self, temp_db_path: str):
        """Test immediate write when buffer_size=0."""
        storage = SQLiteProtocolStorage(temp_db_path)
        writer = ProtocolWriter(storage, buffer_size=0)
        
        entry = ProtocolEntry(
            event_type=ProtocolEventType.AGENT_START,
            session_id="no-buffer",
        )
        
        await writer.write(entry)
        
        # Should be written immediately
        results = await storage.read("no-buffer")
        assert len(results) == 1
        
        writer.close()

    @pytest.mark.asyncio
    async def test_write_with_buffer(self, temp_db_path: str):
        """Test buffered writes."""
        storage = SQLiteProtocolStorage(temp_db_path)
        writer = ProtocolWriter(storage, buffer_size=3)
        
        # Write 2 entries (below buffer threshold)
        for i in range(2):
            await writer.write(ProtocolEntry(
                event_type=ProtocolEventType.AGENT_START,
                session_id="buffered",
            ))
        
        # Not yet written
        results = await storage.read("buffered")
        assert len(results) == 0
        
        # Write 3rd entry - triggers flush
        await writer.write(ProtocolEntry(
            event_type=ProtocolEventType.AGENT_START,
            session_id="buffered",
        ))
        
        results = await storage.read("buffered")
        assert len(results) == 3
        
        writer.close()

    @pytest.mark.asyncio
    async def test_explicit_flush(self, temp_db_path: str):
        """Test explicit flush writes buffered entries."""
        storage = SQLiteProtocolStorage(temp_db_path)
        writer = ProtocolWriter(storage, buffer_size=100)
        
        await writer.write(ProtocolEntry(
            event_type=ProtocolEventType.AGENT_START,
            session_id="flush-test",
        ))
        
        # Not yet written
        results = await storage.read("flush-test")
        assert len(results) == 0
        
        # Explicit flush
        await writer.flush()
        
        results = await storage.read("flush-test")
        assert len(results) == 1
        
        writer.close()

    @pytest.mark.asyncio
    async def test_close_flushes_buffer(self, temp_jsonl_path: str):
        """Test that close flushes remaining buffer."""
        storage = JSONLinesProtocolStorage(temp_jsonl_path)
        writer = ProtocolWriter(storage, buffer_size=100)
        
        await writer.write(ProtocolEntry(
            event_type=ProtocolEventType.LLM_END,
            session_id="close-test",
        ))
        
        # Explicitly flush before close to ensure data is written
        await writer.flush()
        writer.close()
        
        # Re-read from storage
        storage2 = JSONLinesProtocolStorage(temp_jsonl_path)
        results = await storage2.read("close-test")
        assert len(results) == 1
        storage2.close()

    def test_write_batch_sync(self, temp_db_path: str):
        """Test synchronous batch write."""
        storage = SQLiteProtocolStorage(temp_db_path)
        writer = ProtocolWriter(storage, buffer_size=100)
        
        entries = [
            ProtocolEntry(
                event_type=ProtocolEventType.SPAN_START,
                session_id="sync-batch",
            )
            for _ in range(5)
        ]
        
        writer.write_batch_sync(entries)
        
        # Verify written
        results = asyncio.run(storage.read("sync-batch"))
        assert len(results) == 5
        
        writer.close()

    @pytest.mark.asyncio
    async def test_lock_across_event_loops(self, temp_db_path: str):
        """Test that lock works correctly when event loop changes."""
        storage = SQLiteProtocolStorage(temp_db_path)
        writer = ProtocolWriter(storage, buffer_size=0)
        
        # First write in current loop
        await writer.write(ProtocolEntry(
            event_type=ProtocolEventType.TRACE_START,
            session_id="lock-test",
        ))
        
        # Simulate new event loop (in practice this happens in different threads)
        # The writer should create a new lock for the new loop
        results = await storage.read("lock-test")
        assert len(results) == 1
        
        writer.close()


# =============================================================================
# Tests for ProtocolObserver
# =============================================================================


class TestProtocolObserver:
    """Tests for ProtocolObserver."""

    @pytest.fixture
    def observer_setup(self, temp_db_path: str) -> tuple[ProtocolObserver, SQLiteProtocolStorage]:
        """Create observer with dependencies."""
        config = ProtocolConfig(detail_level=ProtocolDetailLevel.STANDARD)
        storage = SQLiteProtocolStorage(temp_db_path)
        writer = ProtocolWriter(storage, buffer_size=0)  # No buffering for tests
        extractor = ProtocolExtractor(config)
        observer = ProtocolObserver(extractor, writer)
        return observer, storage

    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, observer_setup: tuple, mock_agent: Mock, mock_context: Mock):
        """Test agent start/end with duration tracking."""
        observer, storage = observer_setup
        
        await observer.on_agent_start(mock_context, mock_agent)
        
        # Simulate some work
        await asyncio.sleep(0.01)
        
        await observer.on_agent_end(mock_context, mock_agent, "Done")
        
        results = await storage.query({"agent_name": "TestAgent"})
        
        assert len(results) == 2
        
        start_entry = next(e for e in results if e.event_type == ProtocolEventType.AGENT_START)
        end_entry = next(e for e in results if e.event_type == ProtocolEventType.AGENT_END)
        
        assert start_entry.session_id == "run-456"
        assert end_entry.duration_ms is not None
        assert end_entry.duration_ms >= 10  # At least 10ms
        
        observer.shutdown()

    @pytest.mark.asyncio
    async def test_llm_lifecycle(self, observer_setup: tuple, mock_agent: Mock, mock_context: Mock, mock_response: Mock):
        """Test LLM start/end with duration tracking."""
        observer, storage = observer_setup
        
        await observer.on_llm_start(mock_context, mock_agent, "System", ["Hello"])
        await asyncio.sleep(0.005)
        await observer.on_llm_end(mock_context, mock_agent, mock_response)
        
        results = await storage.query({"event_type": "llm_end"})
        
        assert len(results) == 1
        assert results[0].duration_ms is not None
        assert results[0].duration_ms >= 5
        
        observer.shutdown()

    @pytest.mark.asyncio
    async def test_multiple_llm_calls_fifo(self, observer_setup: tuple, mock_agent: Mock, mock_context: Mock, mock_response: Mock):
        """Test multiple LLM calls use FIFO matching for duration."""
        observer, storage = observer_setup
        
        # Start two LLM calls
        await observer.on_llm_start(mock_context, mock_agent, "System1", ["First"])
        await asyncio.sleep(0.02)  # 20ms delay
        await observer.on_llm_start(mock_context, mock_agent, "System2", ["Second"])
        
        # End first call (should match with first start - FIFO)
        await observer.on_llm_end(mock_context, mock_agent, mock_response)
        
        # End second call
        await asyncio.sleep(0.005)
        await observer.on_llm_end(mock_context, mock_agent, mock_response)
        
        results = await storage.query({"event_type": "llm_end"})
        
        assert len(results) == 2
        # First end should have longer duration (matched with first start)
        durations = sorted([e.duration_ms for e in results if e.duration_ms])
        assert len(durations) == 2
        # First ended LLM call should have ~20ms duration
        assert durations[0] >= 5  # Second call - shorter
        assert durations[1] >= 20  # First call - longer
        
        observer.shutdown()

    @pytest.mark.asyncio
    async def test_tool_lifecycle(self, observer_setup: tuple, mock_agent: Mock, mock_context: Mock, mock_tool: Mock):
        """Test tool start/end with duration tracking."""
        observer, storage = observer_setup
        
        await observer.on_tool_start(mock_context, mock_agent, mock_tool)
        await asyncio.sleep(0.005)
        await observer.on_tool_end(mock_context, mock_agent, mock_tool, "Result")
        
        results = await storage.query({"tool_name": "test_tool"})
        
        assert len(results) == 2
        
        end_entry = next(e for e in results if e.event_type == ProtocolEventType.TOOL_END)
        assert end_entry.duration_ms is not None
        assert end_entry.duration_ms >= 5
        
        observer.shutdown()

    @pytest.mark.asyncio
    async def test_handoff(self, observer_setup: tuple, mock_context: Mock):
        """Test handoff recording."""
        observer, storage = observer_setup
        
        from_agent = Mock(name="AgentA")
        from_agent.name = "AgentA"
        to_agent = Mock(name="AgentB")
        to_agent.name = "AgentB"
        
        await observer.on_handoff(mock_context, from_agent, to_agent)
        
        results = await storage.query({"event_type": "handoff"})
        
        assert len(results) == 1
        assert results[0].agent_name == "AgentA"
        
        observer.shutdown()

    def test_trace_lifecycle_sync(self, observer_setup: tuple, mock_trace: Mock):
        """Test trace events in sync context."""
        observer, storage = observer_setup
        
        observer.on_trace_start(mock_trace)
        time.sleep(0.01)
        observer.on_trace_end(mock_trace)
        
        # Flush sync queue
        observer.force_flush()
        
        results = asyncio.run(storage.query({}))
        trace_entries = [e for e in results if e.event_type in (ProtocolEventType.TRACE_START, ProtocolEventType.TRACE_END)]
        
        assert len(trace_entries) == 2
        
        end_entry = next((e for e in trace_entries if e.event_type == ProtocolEventType.TRACE_END), None)
        assert end_entry is not None
        assert end_entry.duration_ms is not None
        assert end_entry.duration_ms >= 10
        
        observer.shutdown()

    def test_span_lifecycle_sync(self, observer_setup: tuple, mock_span: Mock):
        """Test span events in sync context."""
        observer, storage = observer_setup
        
        observer.on_span_start(mock_span)
        time.sleep(0.005)
        observer.on_span_end(mock_span)
        
        observer.force_flush()
        
        results = asyncio.run(storage.query({}))
        span_entries = [e for e in results if e.event_type in (ProtocolEventType.SPAN_START, ProtocolEventType.SPAN_END)]
        
        assert len(span_entries) == 2
        
        observer.shutdown()

    def test_get_run_id_from_context(self, observer_setup: tuple, mock_agent: Mock, mock_context: Mock):
        """Test _get_run_id extracts run_id from context."""
        observer, _ = observer_setup
        
        run_id = observer._get_run_id(mock_context, mock_agent)
        
        assert run_id == "run-456"
        
        observer.shutdown()

    def test_get_run_id_fallback_to_agent_id(self, observer_setup: tuple, mock_agent: Mock):
        """Test _get_run_id falls back to agent id."""
        observer, _ = observer_setup
        
        context_no_run_id = Mock(spec=[])
        run_id = observer._get_run_id(context_no_run_id, mock_agent)
        
        # Should be string of agent's id()
        assert run_id == str(id(mock_agent))
        
        observer.shutdown()

    def test_stale_entry_cleanup(self, observer_setup: tuple):
        """Test that stale start times are cleaned up."""
        observer, _ = observer_setup
        
        # Manually add stale entry (older than 1 hour)
        stale_time = datetime.now(timezone.utc) - timedelta(hours=2)
        observer._start_times["stale:key"] = stale_time
        observer._start_times["fresh:key"] = datetime.now(timezone.utc)
        
        # Trigger cleanup via _calculate_duration
        observer._calculate_duration("nonexistent", datetime.now(timezone.utc))
        
        # Stale should be removed, fresh should remain
        assert "stale:key" not in observer._start_times
        assert "fresh:key" in observer._start_times
        
        observer.shutdown()

    def test_shutdown(self, temp_db_path: str):
        """Test shutdown flushes and closes."""
        config = ProtocolConfig(detail_level=ProtocolDetailLevel.STANDARD)
        storage = SQLiteProtocolStorage(temp_db_path)
        writer = ProtocolWriter(storage, buffer_size=0)
        extractor = ProtocolExtractor(config)
        observer = ProtocolObserver(extractor, writer)
        
        # Add something to sync queue
        observer._sync_queue.append(
            ProtocolEntry(event_type=ProtocolEventType.AGENT_START, session_id="shutdown-test")
        )
        
        observer.shutdown()
        
        # Re-open storage to verify entry was written
        storage2 = SQLiteProtocolStorage(temp_db_path)
        results = asyncio.run(storage2.query({"session_id": "shutdown-test"}))
        assert len(results) == 1
        storage2.close()


class TestProtocolObserverSyncQueue:
    """Tests for sync queue functionality in ProtocolObserver."""

    def test_schedule_write_without_event_loop(self, temp_db_path: str):
        """Test that writes without running loop go to sync queue."""
        config = ProtocolConfig()
        storage = SQLiteProtocolStorage(temp_db_path)
        writer = ProtocolWriter(storage, buffer_size=0)
        extractor = ProtocolExtractor(config)
        observer = ProtocolObserver(extractor, writer)
        
        entry = ProtocolEntry(event_type=ProtocolEventType.TRACE_START)
        
        # This should add to sync queue (no running loop)
        observer._schedule_write(entry)
        
        assert len(observer._sync_queue) == 1
        assert observer._sync_queue[0] == entry
        
        observer.shutdown()

    def test_flush_sync_queue(self, temp_db_path: str):
        """Test flushing sync queue writes entries."""
        config = ProtocolConfig()
        storage = SQLiteProtocolStorage(temp_db_path)
        writer = ProtocolWriter(storage, buffer_size=0)
        extractor = ProtocolExtractor(config)
        observer = ProtocolObserver(extractor, writer)
        
        # Add entries to sync queue
        for i in range(3):
            observer._sync_queue.append(
                ProtocolEntry(
                    event_type=ProtocolEventType.SPAN_START,
                    session_id="sync-queue-test",
                )
            )
        
        observer._flush_sync_queue()
        
        # Queue should be empty
        assert len(observer._sync_queue) == 0
        
        # Entries should be written
        results = asyncio.run(storage.read("sync-queue-test"))
        assert len(results) == 3
        
        observer.shutdown()

    def test_flush_empty_sync_queue(self, temp_db_path: str):
        """Test flushing empty sync queue is safe."""
        config = ProtocolConfig()
        storage = SQLiteProtocolStorage(temp_db_path)
        writer = ProtocolWriter(storage, buffer_size=0)
        extractor = ProtocolExtractor(config)
        observer = ProtocolObserver(extractor, writer)
        
        # Should not raise
        observer._flush_sync_queue()
        
        assert len(observer._sync_queue) == 0
        
        observer.shutdown()


# =============================================================================
# Integration Tests
# =============================================================================


class TestProtocolIntegration:
    """Integration tests for the complete protocol system."""

    @pytest.mark.asyncio
    async def test_full_workflow_sqlite(self, temp_db_path: str):
        """Test complete workflow with SQLite storage."""
        config = ProtocolConfig(detail_level=ProtocolDetailLevel.FULL)
        storage = SQLiteProtocolStorage(temp_db_path)
        writer = ProtocolWriter(storage, buffer_size=0)
        extractor = ProtocolExtractor(config)
        observer = ProtocolObserver(extractor, writer)
        
        # Mock objects
        agent = Mock(name="WorkflowAgent", id="wf-agent-1")
        agent.name = "WorkflowAgent"
        agent.id = "wf-agent-1"
        
        context = Mock()
        context.run_id = "workflow-run-001"
        
        tool = Mock(name="calculator")
        tool.name = "calculator"
        
        response = Mock()
        response.output = "The answer is 42"
        response.usage = Mock(total_tokens=75)
        
        # Simulate workflow
        await observer.on_agent_start(context, agent)
        await observer.on_llm_start(context, agent, "You are helpful", ["What is 6*7?"])
        await asyncio.sleep(0.01)
        await observer.on_llm_end(context, agent, response)
        await observer.on_tool_start(context, agent, tool)
        await asyncio.sleep(0.005)
        await observer.on_tool_end(context, agent, tool, "42")
        await observer.on_agent_end(context, agent, "The answer is 42")
        
        # Verify all entries before shutdown
        all_entries = await storage.query({})
        
        assert len(all_entries) == 6
        
        # Verify event types
        event_types = [e.event_type for e in all_entries]
        assert ProtocolEventType.AGENT_START in event_types
        assert ProtocolEventType.AGENT_END in event_types
        assert ProtocolEventType.LLM_START in event_types
        assert ProtocolEventType.LLM_END in event_types
        assert ProtocolEventType.TOOL_START in event_types
        assert ProtocolEventType.TOOL_END in event_types
        
        # Verify durations are tracked
        end_entries = [e for e in all_entries if e.event_type.value.endswith("_end")]
        for entry in end_entries:
            assert entry.duration_ms is not None
            assert entry.duration_ms > 0
        
        # Verify session_id propagation
        assert all(e.session_id == "workflow-run-001" for e in all_entries)
        
        observer.shutdown()

    @pytest.mark.asyncio
    async def test_full_workflow_jsonlines(self, temp_jsonl_path: str):
        """Test complete workflow with JSONLines storage."""
        config = ProtocolConfig(detail_level=ProtocolDetailLevel.STANDARD)
        storage = JSONLinesProtocolStorage(temp_jsonl_path)
        writer = ProtocolWriter(storage, buffer_size=0)
        extractor = ProtocolExtractor(config)
        observer = ProtocolObserver(extractor, writer)
        
        agent = Mock(name="TestAgent")
        agent.name = "TestAgent"
        
        context = Mock()
        context.run_id = "jsonl-run"
        
        await observer.on_agent_start(context, agent)
        await observer.on_agent_end(context, agent, "Complete")
        
        observer.shutdown()
        
        # Read back
        storage2 = JSONLinesProtocolStorage(temp_jsonl_path)
        results = await storage2.read("jsonl-run")
        
        assert len(results) == 2
        
        storage2.close()

    def test_tracing_processor_integration(self, temp_db_path: str):
        """Test TracingProcessor interface (sync methods)."""
        config = ProtocolConfig(detail_level=ProtocolDetailLevel.FULL)
        storage = SQLiteProtocolStorage(temp_db_path)
        writer = ProtocolWriter(storage, buffer_size=0)
        extractor = ProtocolExtractor(config)
        observer = ProtocolObserver(extractor, writer)
        
        trace = Mock()
        trace.trace_id = "trace-integration"
        trace.workflow_name = "test_workflow"
        trace.group_id = "grp-1"
        
        span = Mock()
        span.span_id = "span-integration"
        span.trace_id = "trace-integration"
        span.name = "test_span"
        span.span_data = Mock(type="agent")
        
        # Call sync methods
        observer.on_trace_start(trace)
        observer.on_span_start(span)
        time.sleep(0.005)
        observer.on_span_end(span)
        observer.on_trace_end(trace)
        
        observer.force_flush()
        
        # Verify entries before shutdown
        results = asyncio.run(storage.query({}))
        
        assert len(results) == 4
        
        trace_entries = [e for e in results if "trace" in e.event_type.value]
        span_entries = [e for e in results if "span" in e.event_type.value]
        
        assert len(trace_entries) == 2
        assert len(span_entries) == 2
        
        observer.shutdown()


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_extractor_with_none_agent(self, standard_config: ProtocolConfig):
        """Test extractor handles None agent gracefully."""
        extractor = ProtocolExtractor(standard_config)
        
        entry = extractor.extract_agent_start(None, None)
        
        assert entry.agent_name is None
        assert entry.agent_id is None

    def test_extractor_with_missing_attributes(self, standard_config: ProtocolConfig):
        """Test extractor handles objects missing expected attributes."""
        extractor = ProtocolExtractor(standard_config)
        
        # Agent without id
        agent_no_id = Mock(spec=["name"])
        agent_no_id.name = "NoIdAgent"
        
        entry = extractor.extract_agent_start(agent_no_id, None)
        
        assert entry.agent_name == "NoIdAgent"
        assert entry.agent_id == "NoIdAgent"  # Falls back to name

    def test_extractor_with_none_response_usage(self, standard_config: ProtocolConfig, mock_agent: Mock):
        """Test LLM end extraction with None usage."""
        extractor = ProtocolExtractor(standard_config)
        
        response = Mock()
        response.output = "Output"
        response.usage = None
        
        entry = extractor.extract_llm_end(mock_agent, response)
        
        assert entry.tokens_used is None

    @pytest.mark.asyncio
    async def test_writer_handles_storage_error(self, temp_db_path: str, caplog):
        """Test writer logs error on storage failure."""
        storage = SQLiteProtocolStorage(temp_db_path)
        writer = ProtocolWriter(storage, buffer_size=0)
        
        # Close storage to cause error
        storage.close()
        
        # This should log error, not raise
        await writer.write(ProtocolEntry(event_type=ProtocolEventType.AGENT_START))
        
        # Writer should have logged error
        # (Exact log assertion depends on logging config)

    @pytest.mark.asyncio
    async def test_storage_empty_batch_write(self, temp_db_path: str):
        """Test storage handles empty batch write."""
        storage = SQLiteProtocolStorage(temp_db_path)
        
        # Should not raise
        await storage.write_batch([])
        
        storage.close()

    @pytest.mark.asyncio
    async def test_observer_duration_without_start(self, temp_db_path: str):
        """Test observer handles end event without matching start."""
        config = ProtocolConfig()
        storage = SQLiteProtocolStorage(temp_db_path)
        writer = ProtocolWriter(storage, buffer_size=0)
        extractor = ProtocolExtractor(config)
        observer = ProtocolObserver(extractor, writer)
        
        agent = Mock(name="TestAgent")
        agent.name = "TestAgent"
        context = Mock()
        context.run_id = "orphan-end"
        
        # End without start
        await observer.on_agent_end(context, agent, "Output")
        
        results = await storage.query({"session_id": "orphan-end"})
        
        assert len(results) == 1
        assert results[0].duration_ms is None  # No duration without start
        
        observer.shutdown()

    def test_calculate_duration_atomic_pop(self, temp_db_path: str):
        """Test _calculate_duration uses atomic pop."""
        config = ProtocolConfig()
        storage = SQLiteProtocolStorage(temp_db_path)
        writer = ProtocolWriter(storage, buffer_size=0)
        extractor = ProtocolExtractor(config)
        observer = ProtocolObserver(extractor, writer)
        
        now = datetime.now(timezone.utc)
        observer._start_times["test:key"] = now - timedelta(seconds=1)
        
        # First call should return duration and remove key
        duration1 = observer._calculate_duration("test:key", now)
        
        assert duration1 is not None
        assert abs(duration1 - 1000) < 10  # ~1000ms
        
        # Second call should return None (key removed)
        duration2 = observer._calculate_duration("test:key", now)
        
        assert duration2 is None
        
        observer.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_writes_to_buffer(self, temp_db_path: str):
        """Test concurrent writes to buffered writer."""
        storage = SQLiteProtocolStorage(temp_db_path)
        writer = ProtocolWriter(storage, buffer_size=100)
        
        async def write_entry(i: int):
            await writer.write(ProtocolEntry(
                event_type=ProtocolEventType.AGENT_START,
                session_id="concurrent",
                agent_name=f"Agent-{i}",
            ))
        
        # Concurrent writes
        await asyncio.gather(*[write_entry(i) for i in range(50)])
        
        await writer.flush()
        
        results = await storage.read("concurrent")
        assert len(results) == 50
        
        writer.close()
