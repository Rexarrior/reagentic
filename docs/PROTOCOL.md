# Reagentic Protocol System

The Reagentic Protocol System provides comprehensive structured logging for AI agent operations. It captures agent lifecycle events, LLM calls, tool executions, handoffs, and traces with automatic duration tracking, configurable detail levels, and multiple storage backends.

## 🚀 Features

- **Event Tracking**: Capture agent, LLM, tool, trace, and span events
- **Duration Tracking**: Automatic duration calculation for all start/end event pairs
- **Multiple Storage Backends**: SQLite for queries, JSONLines for simplicity and streaming
- **Configurable Detail Levels**: MINIMAL, STANDARD, FULL for different use cases
- **Session Tracking**: Group entries by session/run ID
- **Async-First Design**: Non-blocking writes with async/await support
- **Buffer Management**: Configurable buffered writes for performance
- **OpenAI Agents SDK Integration**: Works as `RunHooks` and `TracingProcessor`

## 📋 Quick Start

### Basic Usage

```python
from reagentic.protocol import (
    ProtocolObserver, ProtocolExtractor, ProtocolWriter,
    ProtocolConfig, ProtocolDetailLevel, SQLiteProtocolStorage
)
from agents import Agent, Runner, add_trace_processor

# 1. Configure detail level
config = ProtocolConfig(detail_level=ProtocolDetailLevel.STANDARD)

# 2. Create storage backend
storage = SQLiteProtocolStorage("protocol.db")

# 3. Create writer with optional buffering
writer = ProtocolWriter(storage, buffer_size=100)

# 4. Create extractor and observer
extractor = ProtocolExtractor(config)
observer = ProtocolObserver(extractor, writer)

# 5. Register as trace processor (for spans and traces)
add_trace_processor(observer)

# 6. Create and run agent with observer as hooks
agent = Agent(name="Assistant", instructions="You are helpful")
result = Runner.run_sync(agent, "Hello!", hooks=observer)

# 7. Cleanup
observer.shutdown()
```

### JSONLines Storage Alternative

```python
from reagentic.protocol import JSONLinesProtocolStorage

# Use JSONLines for simple append-only logging
storage = JSONLinesProtocolStorage("protocol.jsonl")
writer = ProtocolWriter(storage, buffer_size=0)  # Immediate writes
```

## 🔧 Configuration

### Detail Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `MINIMAL` | Agent, tool, input/output, time | Production with minimal overhead |
| `STANDARD` | + tokens, prompts, reasoning | Development and debugging |
| `FULL` | + intermediate steps, all metadata, traces | Deep debugging and analysis |

### ProtocolConfig Options

```python
config = ProtocolConfig(
    detail_level=ProtocolDetailLevel.STANDARD,
    include_input=True,       # Include input data in entries
    include_output=True,      # Include output data in entries
    include_prompts=True,     # Include system prompts (STANDARD+)
    include_intermediate=False,  # Include intermediate steps
    max_content_length=1000,  # Truncate long content (None = no limit)
)
```

### Config Helper Methods

```python
config = ProtocolConfig(detail_level=ProtocolDetailLevel.FULL)

config.allows_prompts()      # True for STANDARD and FULL
config.allows_intermediate() # True for FULL or when include_intermediate=True
config.allows_metadata()     # True only for FULL
config.allows_tracing()      # True only for FULL
```

> 📖 **Tests Reference**: See [`tests/test_protocol.py::TestProtocolConfig`](../tests/test_protocol.py) for configuration examples.

## 📊 Event Types

The protocol system captures these event types:

| Event Type | Description | Triggered By |
|------------|-------------|--------------|
| `AGENT_START` | Agent begins processing | `on_agent_start` hook |
| `AGENT_END` | Agent completes processing | `on_agent_end` hook |
| `LLM_START` | LLM call initiated | `on_llm_start` hook |
| `LLM_END` | LLM response received | `on_llm_end` hook |
| `TOOL_START` | Tool execution begins | `on_tool_start` hook |
| `TOOL_END` | Tool execution completes | `on_tool_end` hook |
| `HANDOFF` | Agent handoff occurs | `on_handoff` hook |
| `TRACE_START` | Trace begins | `on_trace_start` processor |
| `TRACE_END` | Trace ends | `on_trace_end` processor |
| `SPAN_START` | Span begins | `on_span_start` processor |
| `SPAN_END` | Span ends | `on_span_end` processor |

## 🏗️ Architecture

```
┌──────────────┐    ┌───────────────────┐    ┌────────────────┐
│    Agent     │───▶│  ProtocolObserver │───▶│ ProtocolWriter │
│   RunHooks   │    │  (RunHooks +      │    │  (buffered)    │
│              │    │   TracingProcessor)│    │                │
└──────────────┘    └───────────────────┘    └───────┬────────┘
                              │                      │
                              ▼                      ▼
                    ┌───────────────────┐    ┌────────────────┐
                    │ ProtocolExtractor │    │ ProtocolStorage│
                    │  (creates entries) │    │ SQLite/JSONLines│
                    └───────────────────┘    └────────────────┘
```

### Components

1. **ProtocolObserver**: Implements `RunHooks` and `TracingProcessor`, tracks start times for duration calculation
2. **ProtocolExtractor**: Creates `ProtocolEntry` objects from agent/tool/trace data
3. **ProtocolWriter**: Buffers entries and writes to storage
4. **ProtocolStorage**: Abstract interface with SQLite and JSONLines implementations

> 📖 **Tests Reference**: See [`tests/test_protocol.py::TestProtocolIntegration`](../tests/test_protocol.py) for full workflow examples.

## 📈 Duration Tracking

The observer automatically tracks duration for all start/end event pairs:

```python
# Duration is calculated automatically
await observer.on_agent_start(context, agent)
# ... agent work happens ...
await observer.on_agent_end(context, agent, output)
# entry.duration_ms contains the elapsed time
```

### FIFO Matching for Multiple Calls

When multiple LLM or tool calls happen concurrently, the observer uses FIFO (First-In-First-Out) matching:

```python
# Two concurrent LLM calls
await observer.on_llm_start(context, agent, "System1", ["First"])
await observer.on_llm_start(context, agent, "System2", ["Second"])

# First on_llm_end matches with first on_llm_start
await observer.on_llm_end(context, agent, response1)
# Second on_llm_end matches with second on_llm_start
await observer.on_llm_end(context, agent, response2)
```

> 📖 **Tests Reference**: See [`tests/test_protocol.py::TestProtocolObserver::test_multiple_llm_calls_fifo`](../tests/test_protocol.py) for FIFO matching examples.

### Stale Entry Cleanup

To prevent memory leaks from orphaned start events (when end is never called), entries older than 1 hour are automatically cleaned up:

```python
# Orphaned start times are removed automatically during duration calculation
# This prevents memory leaks in long-running applications
```

> 📖 **Tests Reference**: See [`tests/test_protocol.py::TestProtocolObserver::test_stale_entry_cleanup`](../tests/test_protocol.py).

## 💾 Storage Backends

### SQLiteProtocolStorage

Best for: Queryable storage, analytics, structured access.

```python
from reagentic.protocol import SQLiteProtocolStorage

storage = SQLiteProtocolStorage("protocol.db")

# Write entries
await storage.write(entry)
await storage.write_batch(entries)

# Query by session
entries = await storage.read("session-123")

# Query with filters
entries = await storage.query({
    "event_type": "agent_end",
    "agent_name": "Assistant"
})

# Get all entries
all_entries = await storage.query({})

storage.close()
```

**Features:**
- WAL mode for concurrent reads
- Indexed columns: session_id, event_type, agent_name, tool_name, trace_id
- Thread-safe with internal locking
- Async methods using `run_in_executor`

> 📖 **Tests Reference**: See [`tests/test_protocol.py::TestSQLiteProtocolStorage`](../tests/test_protocol.py).

### JSONLinesProtocolStorage

Best for: Simple logging, streaming, external tool compatibility.

```python
from reagentic.protocol import JSONLinesProtocolStorage

storage = JSONLinesProtocolStorage("protocol.jsonl")

# Write entries
await storage.write(entry)
await storage.write_batch(entries)

# Query with filters
entries = await storage.query({"session_id": "session-123"})

storage.close()
```

**Features:**
- Append-only format
- Human-readable JSON per line
- Compatible with `jq`, `grep`, and log analysis tools
- Automatic flush after each write for durability

> 📖 **Tests Reference**: See [`tests/test_protocol.py::TestJSONLinesProtocolStorage`](../tests/test_protocol.py).

## ✍️ ProtocolWriter

The writer handles buffering and batching for efficient storage:

```python
from reagentic.protocol import ProtocolWriter

# Buffered writes (default)
writer = ProtocolWriter(storage, buffer_size=100)
await writer.write(entry)  # Buffers until 100 entries
await writer.flush()       # Force write buffer to storage

# Immediate writes (no buffering)
writer = ProtocolWriter(storage, buffer_size=0)
await writer.write(entry)  # Written immediately

# Synchronous batch write (for sync contexts)
writer.write_batch_sync(entries)

# Cleanup (flushes and closes storage)
writer.close()
```

> 📖 **Tests Reference**: See [`tests/test_protocol.py::TestProtocolWriter`](../tests/test_protocol.py).

## 🔍 ProtocolExtractor

The extractor creates protocol entries from various objects:

```python
from reagentic.protocol import ProtocolExtractor, ProtocolConfig

config = ProtocolConfig(detail_level=ProtocolDetailLevel.STANDARD)
extractor = ProtocolExtractor(config)

# Extract agent events
entry = extractor.extract_agent_start(agent, context)
entry = extractor.extract_agent_end(agent, output, context)

# Extract LLM events
entry = extractor.extract_llm_start(agent, system_prompt, input_items, context)
entry = extractor.extract_llm_end(agent, response, context)

# Extract tool events
entry = extractor.extract_tool_start(agent, tool, context)
entry = extractor.extract_tool_end(agent, tool, result, context)

# Extract handoff
entry = extractor.extract_handoff(from_agent, to_agent, context)

# Extract traces and spans (for FULL detail level)
entry = extractor.extract_trace(trace, is_start=True)
entry = extractor.extract_span(span, is_start=False)

# Extract errors
entry = extractor.extract_error(agent, exception, ProtocolEventType.AGENT_END)
```

### Session ID Extraction

The extractor looks for session ID in this order:
1. `context.run_id`
2. `context.session_id`
3. `context.context_id`

> 📖 **Tests Reference**: See [`tests/test_protocol.py::TestProtocolExtractor`](../tests/test_protocol.py).

## 🔄 Sync Context Support

The observer supports both async and sync contexts:

```python
# Async context (normal usage)
async def async_handler():
    await observer.on_agent_start(context, agent)

# Sync context (tracing processor)
def sync_handler():
    observer.on_trace_start(trace)  # Added to sync queue
    observer.on_span_end(span)      # Added to sync queue

# Force flush sync queue
observer.force_flush()

# Shutdown flushes everything
observer.shutdown()
```

> 📖 **Tests Reference**: See [`tests/test_protocol.py::TestProtocolObserverSyncQueue`](../tests/test_protocol.py).

## 📐 ProtocolEntry Schema

```python
class ProtocolEntry(BaseModel):
    id: str                           # UUID, auto-generated
    timestamp: datetime               # UTC timestamp
    event_type: ProtocolEventType     # Event type enum
    session_id: Optional[str]         # Run/session grouping
    
    # Context
    agent_name: Optional[str]         # Agent's name
    agent_id: Optional[str]           # Agent's ID
    tool_name: Optional[str]          # Tool's name (for tool events)
    
    # Data (based on detail level)
    input_data: Optional[Any]         # Input data
    output_data: Optional[Any]        # Output data
    system_prompt: Optional[str]      # System prompt (STANDARD+)
    tokens_used: Optional[int]        # Token count (STANDARD+)
    duration_ms: Optional[float]      # Duration in milliseconds
    
    # Extended data (FULL only)
    intermediate_steps: Optional[List[Any]]
    trace_id: Optional[str]
    span_id: Optional[str]
    metadata: Optional[Dict[str, Any]]
    error: Optional[str]
```

> 📖 **Tests Reference**: See [`tests/test_protocol.py::TestProtocolModels`](../tests/test_protocol.py).

## 🛠️ Complete Example

```python
import asyncio
from reagentic.protocol import (
    ProtocolObserver, ProtocolExtractor, ProtocolWriter,
    ProtocolConfig, ProtocolDetailLevel, SQLiteProtocolStorage
)
from agents import Agent, Runner, add_trace_processor

async def main():
    # Setup protocol system
    config = ProtocolConfig(
        detail_level=ProtocolDetailLevel.FULL,
        max_content_length=500
    )
    storage = SQLiteProtocolStorage("agent_protocol.db")
    writer = ProtocolWriter(storage, buffer_size=0)
    extractor = ProtocolExtractor(config)
    observer = ProtocolObserver(extractor, writer)
    
    # Register trace processor
    add_trace_processor(observer)
    
    try:
        # Create and run agent
        agent = Agent(
            name="Assistant",
            instructions="You are a helpful assistant."
        )
        
        result = await Runner.run(
            agent,
            "What is 2 + 2?",
            hooks=observer
        )
        
        print(f"Result: {result.final_output}")
        
        # Query recorded entries
        entries = await storage.query({"agent_name": "Assistant"})
        print(f"\nRecorded {len(entries)} protocol entries:")
        for entry in entries:
            print(f"  [{entry.event_type.value}] {entry.agent_name} - {entry.duration_ms}ms")
        
    finally:
        observer.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔍 Analyzing Protocol Data

### With SQLite

```bash
# View all entries
sqlite3 protocol.db "SELECT event_type, agent_name, duration_ms FROM protocol_entries"

# Get average LLM call duration
sqlite3 protocol.db "
  SELECT AVG(json_extract(entry_json, '$.duration_ms'))
  FROM protocol_entries
  WHERE event_type = 'llm_end'
"

# Count events by type
sqlite3 protocol.db "
  SELECT event_type, COUNT(*)
  FROM protocol_entries
  GROUP BY event_type
"
```

### With JSONLines

```bash
# View all entries
cat protocol.jsonl | jq '.'

# Filter by event type
cat protocol.jsonl | jq 'select(.event_type == "agent_end")'

# Get durations
cat protocol.jsonl | jq 'select(.duration_ms != null) | {event_type, duration_ms}'
```

## 📚 Tests as Examples

The test file contains comprehensive examples for all functionality:

| Test Class | Purpose |
|------------|---------|
| [`TestProtocolModels`](../tests/test_protocol.py) | Event types, detail levels, entry schema |
| [`TestProtocolConfig`](../tests/test_protocol.py) | Configuration options and helper methods |
| [`TestProtocolExtractor`](../tests/test_protocol.py) | Data extraction from agent/tool objects |
| [`TestSQLiteProtocolStorage`](../tests/test_protocol.py) | SQLite storage operations |
| [`TestJSONLinesProtocolStorage`](../tests/test_protocol.py) | JSONLines storage operations |
| [`TestProtocolWriter`](../tests/test_protocol.py) | Buffering and write operations |
| [`TestProtocolObserver`](../tests/test_protocol.py) | Event handling and duration tracking |
| [`TestProtocolObserverSyncQueue`](../tests/test_protocol.py) | Sync context support |
| [`TestProtocolIntegration`](../tests/test_protocol.py) | Full workflow examples |
| [`TestEdgeCases`](../tests/test_protocol.py) | Error handling and edge cases |

Run tests:

```bash
# Run all protocol tests
pytest tests/test_protocol.py -v

# Run specific test class
pytest tests/test_protocol.py::TestProtocolObserver -v

# Run with output
pytest tests/test_protocol.py -v -s
```

## 🔐 Best Practices

1. **Choose the right detail level**: Use MINIMAL in production, STANDARD for development, FULL for debugging
2. **Buffer appropriately**: Use larger buffer sizes for high-throughput, smaller for real-time visibility
3. **Always call shutdown**: Ensures all buffered data is written
4. **Use session IDs**: Group related entries for easier analysis
5. **Truncate content**: Set `max_content_length` to prevent storage bloat
6. **Index queries**: Use indexed columns (session_id, event_type, agent_name) for fast queries

## 🚨 Troubleshooting

### Entries not appearing

```python
# Ensure you flush before reading
await writer.flush()
# Or use buffer_size=0 for immediate writes
```

### Duration is None

```python
# Duration requires matching start/end events
# Check that on_*_start was called before on_*_end
```

### Memory growing in long-running apps

```python
# Stale entries are cleaned up automatically after 1 hour
# For immediate cleanup, call shutdown() periodically
```

### Storage closed error

```python
# Don't read from storage after observer.shutdown()
# Read data first, then shutdown
```

## 📋 API Reference

### Public Exports

```python
from reagentic.protocol import (
    # Config and models
    ProtocolConfig,
    ProtocolDetailLevel,
    ProtocolEntry,
    ProtocolEventType,
    
    # Core components
    ProtocolExtractor,
    ProtocolObserver,
    ProtocolWriter,
    
    # Storage backends
    ProtocolStorage,      # Abstract base
    SQLiteProtocolStorage,
    JSONLinesProtocolStorage,
)
```
