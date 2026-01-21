---
name: Protocol System Fixes
overview: Fix all identified issues in the protocol implementation including async handling, race conditions, duration tracking, and resource management.
todos:
  - id: fix-writer-lock
    content: Fix race condition in ProtocolWriter._get_lock()
    status: completed
  - id: fix-observer-schedule
    content: Replace expensive event loop creation with sync queue in observer
    status: completed
    dependencies:
      - fix-writer-lock
  - id: add-duration-tracking
    content: Add duration_ms calculation for start/end event pairs
    status: completed
    dependencies:
      - fix-observer-schedule
  - id: fix-force-flush
    content: Make force_flush wait for completion
    status: completed
    dependencies:
      - fix-observer-schedule
  - id: async-sqlite
    content: Make SQLiteProtocolStorage methods truly async with run_in_executor
    status: completed
  - id: async-jsonlines
    content: Make JSONLinesProtocolStorage methods truly async with run_in_executor
    status: completed
  - id: fix-extractor
    content: Fix session_id extraction and add extract_error method
    status: completed
  - id: fix-example
    content: Add resource cleanup to protocol_example.py
    status: completed
    dependencies:
      - fix-force-flush
---

# Protocol System Bug Fixes

## Files to modify

- [`reagentic/protocol/observer.py`](reagentic/protocol/observer.py) - async handling, duration tracking
- [`reagentic/protocol/writer.py`](reagentic/protocol/writer.py) - race condition fix
- [`reagentic/protocol/extractor.py`](reagentic/protocol/extractor.py) - session_id fix, error handling
- [`reagentic/protocol/storage/sqlite.py`](reagentic/protocol/storage/sqlite.py) - true async
- [`reagentic/protocol/storage/jsonlines.py`](reagentic/protocol/storage/jsonlines.py) - true async
- [`examples/protocol_example.py`](examples/protocol_example.py) - resource cleanup

---

## 1. Fix `_schedule_write` in observer.py

**Problem:** Creating `asyncio.new_event_loop()` for each sync call is expensive.

**Solution:** Accumulate entries in a thread-safe queue, flush on shutdown.

```python
class ProtocolObserver(RunHooks, TracingProcessor):
    def __init__(self, extractor: ProtocolExtractor, writer: ProtocolWriter) -> None:
        self._extractor = extractor
        self._writer = writer
        self._start_times: dict[str, datetime] = {}
        self._sync_queue: list[ProtocolEntry] = []
        self._sync_lock = threading.Lock()

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
        try:
            asyncio.run(self._writer._storage.write_batch(entries))
        except Exception as e:
            logger.error(f"Failed to flush sync queue: {e}")
```

---

## 2. Add duration tracking in observer.py

**Problem:** `duration_ms` field is never populated.

**Solution:** Track start times in a dict, calculate duration on `*_end` events.

```python
async def on_agent_start(self, context, agent) -> None:
    entry = self._extractor.extract_agent_start(agent, context)
    self._start_times[f"agent:{id(agent)}"] = entry.timestamp
    await self._writer.write(entry)

async def on_agent_end(self, context, agent, output) -> None:
    entry = self._extractor.extract_agent_end(agent, output)
    key = f"agent:{id(agent)}"
    if key in self._start_times:
        delta = entry.timestamp - self._start_times.pop(key)
        entry.duration_ms = delta.total_seconds() * 1000
    await self._writer.write(entry)
```

Apply same pattern for `llm_start/end` and `tool_start/end`.

---

## 3. Fix race condition in writer.py `_get_lock()`

**Problem:** Multiple coroutines can create multiple locks simultaneously.

**Solution:** Use `threading.Lock` to protect lazy initialization.

```python
def __init__(self, storage: ProtocolStorage, buffer_size: int = 100) -> None:
    self._storage = storage
    self._buffer_size = max(0, buffer_size)
    self._buffer: List[ProtocolEntry] = []
    self._lock: Optional[asyncio.Lock] = None
    self._lock_init = threading.Lock()
    self._sync_lock = threading.Lock()

def _get_lock(self) -> asyncio.Lock:
    with self._lock_init:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock
```

---

## 4. Fix `force_flush` in observer.py

**Problem:** `create_task()` doesn't wait for completion.

**Solution:** Use `run_coroutine_threadsafe` with `result()`.

```python
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
    except Exception as e:
        logger.error(f"Failed to force flush protocol: {e}")
```

---

## 5. Make storage methods truly async

**Problem:** Blocking I/O inside async methods blocks the event loop.

**Solution:** Use `asyncio.get_event_loop().run_in_executor()` for blocking operations.

For [`reagentic/protocol/storage/sqlite.py`](reagentic/protocol/storage/sqlite.py):

```python
async def write(self, entry: ProtocolEntry) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, self._sync_write, entry)

def _sync_write(self, entry: ProtocolEntry) -> None:
    with self._lock:
        self._conn.execute(...)
        self._conn.commit()
```

Apply same pattern to `write_batch`, `read`, `query` for both SQLite and JSONLines storage.

---

## 6. Fix session_id extraction in extractor.py

**Problem:** `getattr(context, "context", None)` looks incorrect.

**Solution:** Try multiple common attribute names.

```python
def extract_agent_start(self, agent: Any, context: Any) -> ProtocolEntry:
    session_id = (
        getattr(context, "run_id", None)
        or getattr(context, "session_id", None)
        or getattr(context, "context_id", None)
    )
    entry = self._base_entry(
        ProtocolEventType.AGENT_START,
        agent=agent,
        session_id=str(session_id) if session_id else None,
    )
    return entry
```

---

## 7. Add error extraction method in extractor.py

**Problem:** No way to populate the `error` field.

**Solution:** Add `extract_error` method.

```python
def extract_error(
    self,
    agent: Any,
    error: Exception,
    event_type: ProtocolEventType,
) -> ProtocolEntry:
    entry = self._base_entry(event_type, agent=agent)
    entry.error = self._serialize(str(error))
    return entry
```

---

## 8. Fix example to close resources

**Problem:** Example doesn't call `shutdown()` or `close()`.

**Solution:** Add try/finally with cleanup.

```python
def main() -> None:
    provider = openrouter.OpenrouterProvider(openrouter.DEEPSEEK_CHAT_V3_0324)
    agent = Agent(...)

    config = ProtocolConfig(detail_level=ProtocolDetailLevel.STANDARD)
    sqlite_storage = SQLiteProtocolStorage("protocol.db")
    writer = ProtocolWriter(sqlite_storage)
    observer = ProtocolObserver(ProtocolExtractor(config), writer)

    add_trace_processor(observer)

    try:
        result = Runner.run_sync(agent, "Write a short greeting.", hooks=observer)
        print(result.final_output)
    finally:
        observer.shutdown()
```

---

## Implementation Order

1. Fix race condition in writer (isolated change)
2. Add sync queue and duration tracking to observer
3. Fix force_flush in observer
4. Make storage methods truly async
5. Fix extractor (session_id + error method)
6. Update example with cleanup