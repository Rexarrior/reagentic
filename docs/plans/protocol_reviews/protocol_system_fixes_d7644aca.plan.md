---
name: Protocol System Fixes
overview: Fix deprecated asyncio APIs, session_id propagation, duration tracking keys, and resource cleanup in the protocol system.
todos:
  - id: fix-deprecated-asyncio
    content: Replace asyncio.get_event_loop() with get_running_loop() in sqlite.py and jsonlines.py
    status: completed
  - id: fix-lock-binding
    content: Add event loop validation to asyncio.Lock in writer.py
    status: completed
    dependencies:
      - fix-deprecated-asyncio
  - id: fix-duration-keys
    content: Use run_id instead of id(agent) for duration tracking keys in observer.py
    status: completed
    dependencies:
      - fix-lock-binding
  - id: fix-session-propagation
    content: Pass context to extract_*_end methods and propagate session_id
    status: completed
    dependencies:
      - fix-duration-keys
  - id: fix-example-cleanup
    content: Document or fix trace processor cleanup in protocol_example.py
    status: completed
    dependencies:
      - fix-session-propagation
---

# Protocol System Post-Review Fixes

## Files to modify

- [`reagentic/protocol/storage/sqlite.py`](reagentic/protocol/storage/sqlite.py) - deprecated asyncio fix
- [`reagentic/protocol/storage/jsonlines.py`](reagentic/protocol/storage/jsonlines.py) - deprecated asyncio fix
- [`reagentic/protocol/writer.py`](reagentic/protocol/writer.py) - asyncio.Lock loop binding fix
- [`reagentic/protocol/observer.py`](reagentic/protocol/observer.py) - duration tracking keys, session_id propagation
- [`reagentic/protocol/extractor.py`](reagentic/protocol/extractor.py) - session_id in end events
- [`examples/protocol_example.py`](examples/protocol_example.py) - trace processor cleanup

---

## 1. Replace deprecated `asyncio.get_event_loop()`

In both storage implementations, replace:

```python
loop = asyncio.get_event_loop()
await loop.run_in_executor(...)
```

With:

```python
loop = asyncio.get_running_loop()
await loop.run_in_executor(...)
```

Affected lines:

- `sqlite.py`: 158, 162, 166, 170
- `jsonlines.py`: 56, 60, 67

---

## 2. Fix `asyncio.Lock` event loop binding

In `writer.py`, the lock may be created in one event loop but used in another. Add loop validation:

```python
def _get_lock(self) -> asyncio.Lock:
    current_loop = asyncio.get_running_loop()
    with self._lock_init:
        if self._lock is None:
            self._lock = asyncio.Lock()
            self._lock_loop = current_loop
        elif self._lock_loop is not current_loop:
            self._lock = asyncio.Lock()
            self._lock_loop = current_loop
        return self._lock
```

Add `_lock_loop: Optional[asyncio.AbstractEventLoop] = None `to `__init__`.

---

## 3. Improve duration tracking keys

In `observer.py`, replace `id(agent)` with more stable identifiers using `run_id` from context:

```python
async def on_agent_start(self, context, agent) -> None:
    entry = self._extractor.extract_agent_start(agent, context)
    run_id = getattr(context, "run_id", None) or id(agent)
    self._start_times[f"agent:{run_id}"] = entry.timestamp
    await self._writer.write(entry)

async def on_agent_end(self, context, agent, output) -> None:
    entry = self._extractor.extract_agent_end(agent, output, context)
    run_id = getattr(context, "run_id", None) or id(agent)
    entry.duration_ms = self._calculate_duration(f"agent:{run_id}", entry.timestamp)
    await self._writer.write(entry)
```

Apply same pattern for `llm_start/end` and `tool_start/end`.

---

## 4. Propagate session_id to end events

Modify extractor methods to accept context parameter:

```python
def extract_agent_end(self, agent: Any, output: Any, context: Any = None) -> ProtocolEntry:
    session_id = self._extract_session_id(context) if context else None
    entry = self._base_entry(ProtocolEventType.AGENT_END, agent=agent, session_id=session_id)
    if self._config.include_output:
        entry.output_data = self._serialize(output)
    return entry
```

Add helper method:

```python
def _extract_session_id(self, context: Any) -> Optional[str]:
    session_id = (
        getattr(context, "run_id", None)
        or getattr(context, "session_id", None)
        or getattr(context, "context_id", None)
    )
    return str(session_id) if session_id else None
```

Update all `extract_*_end` methods similarly.

---

## 5. Fix trace processor cleanup in example

Since `remove_trace_processor` may not exist in the SDK, use context manager pattern or document the limitation:

```python
def main() -> None:
    # ... setup ...

    add_trace_processor(observer)

    try:
        result = Runner.run_sync(agent, "Write a short greeting.", hooks=observer)
        print(result.final_output)
    finally:
        observer.shutdown()
    
    # Note: trace processor remains registered globally
    # This is acceptable for single-run scripts
```

Alternative: check if `remove_trace_processor` exists and use it conditionally.

---

## Implementation order

```mermaid
flowchart LR
    A[1_Deprecated_asyncio] --> B[2_Lock_binding]
    B --> C[3_Duration_keys]
    C --> D[4_Session_propagation]
    D --> E[5_Example_cleanup]
```

1. Deprecated asyncio fix (isolated, no dependencies)
2. Lock binding fix (isolated, no dependencies)
3. Duration tracking keys (requires context access changes)
4. Session_id propagation (requires extractor signature changes)
5. Example cleanup (final polish)