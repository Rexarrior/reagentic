---
name: Protocol System Fixes
overview: Fix thread safety, memory leaks, durability, and improve error handling in the protocol system based on the code review findings.
todos:
  - id: thread-safe-duration
    content: Make _calculate_duration thread-safe with atomic pop
    status: completed
  - id: stale-cleanup
    content: Add lazy cleanup of stale _start_times entries
    status: completed
  - id: timeout-logging
    content: Distinguish TimeoutError in force_flush for better debugging
    status: completed
  - id: jsonlines-flush
    content: Add flush() calls in JSONLines storage for durability
    status: completed
  - id: sqlite-comment
    content: Document check_same_thread=False safety in SQLite storage
    status: completed
---

# Protocol System Minor Fixes

## Files to modify

- [`reagentic/protocol/observer.py`](reagentic/protocol/observer.py) - thread safety, memory leak prevention, timeout handling
- [`reagentic/protocol/storage/jsonlines.py`](reagentic/protocol/storage/jsonlines.py) - durability fix
- [`reagentic/protocol/storage/sqlite.py`](reagentic/protocol/storage/sqlite.py) - documentation comment

---

## 1. Thread-safe duration tracking

**Problem:** `_start_times.pop()` in `_calculate_duration` is not atomic with the `in` check.

**Solution:** Use `dict.pop(key, None)` which is atomic:

```python
def _calculate_duration(self, key: str, end_timestamp: datetime) -> float | None:
    start = self._start_times.pop(key, None)
    if start is not None:
        delta = end_timestamp - start
        return delta.total_seconds() * 1000
    return None
```

---

## 2. Memory leak prevention in `_start_times`

**Problem:** If `*_start` is called without corresponding `*_end` (e.g., on exception), keys accumulate forever.

**Solution:** Add periodic cleanup of stale entries (older than 1 hour) in `_calculate_duration`:

```python
from datetime import timedelta

_STALE_THRESHOLD = timedelta(hours=1)

def _calculate_duration(self, key: str, end_timestamp: datetime) -> float | None:
    start = self._start_times.pop(key, None)
    
    # Cleanup stale entries (lazy, on each call)
    stale_cutoff = end_timestamp - self._STALE_THRESHOLD
    stale_keys = [k for k, v in self._start_times.items() if v < stale_cutoff]
    for k in stale_keys:
        self._start_times.pop(k, None)
    
    if start is not None:
        delta = end_timestamp - start
        return delta.total_seconds() * 1000
    return None
```

---

## 3. Distinguish timeout in `force_flush`

**Problem:** `TimeoutError` is caught as generic `Exception`, making debugging harder.

**Solution:** Catch `TimeoutError` separately with specific warning:

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
    except TimeoutError:
        logger.warning("Protocol flush timed out after 5 seconds, data may be lost")
    except Exception as e:
        logger.error(f"Failed to force flush protocol: {e}")
```

---

## 4. Add flush to JSONLines for durability

**Problem:** Written data may remain in OS buffer, lost on crash.

**Solution:** Call `flush()` after write:

```python
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
```

---

## 5. Document SQLite thread safety

**Problem:** `check_same_thread=False` without explanation looks suspicious.

**Solution:** Add comment explaining why it's safe:

```python
# check_same_thread=False is safe because all operations are protected
# by self._lock, ensuring only one thread accesses the connection at a time
self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
```

---

## Implementation order

```mermaid
flowchart LR
    A[1_Thread_safe_pop] --> B[2_Stale_cleanup]
    B --> C[3_Timeout_logging]
    C --> D[4_JSONLines_flush]
    D --> E[5_SQLite_comment]
```

All fixes are independent and low-risk.