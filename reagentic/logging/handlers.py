"""
Logging Handlers and Formatters

Provides various handlers and formatters for the reagentic logging system.
"""

import json
import logging
import logging.handlers
import sys
import asyncio

try:
    import aiohttp
except Exception:
    pass

import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

from .context import LogContextManager
from .config import LogFormat, LogLevel


class JSONFormatter(logging.Formatter):
    """
    JSON formatter that includes context information and structured data.
    """

    def __init__(self, include_context: bool = True, include_stack_trace: bool = True):
        super().__init__()
        self.include_context = include_context
        self.include_stack_trace = include_stack_trace

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""

        # Base log data
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
        }

        # Add context if enabled
        if self.include_context:
            try:
                context = LogContextManager.get_context()
                log_data['context'] = context.to_dict()
            except Exception:
                # Don't fail logging if context retrieval fails
                pass

        # Add extra fields from record
        if hasattr(record, 'extra'):
            log_data.update(record.extra)

        # Add exception information
        if record.exc_info and self.include_stack_trace:
            # Handle both tuple and boolean exc_info
            if isinstance(record.exc_info, tuple) and len(record.exc_info) >= 3:
                exc_type, exc_value, exc_traceback = record.exc_info
                log_data['exception'] = {
                    'type': exc_type.__name__ if exc_type else None,
                    'message': str(exc_value) if exc_value else None,
                    'traceback': self.formatException(record.exc_info) if record.exc_info else None,
                }
            else:
                # exc_info is True but we don't have the actual exception info
                log_data['exception'] = {
                    'type': 'UnknownException',
                    'message': 'Exception occurred but details not available',
                    'traceback': None,
                }

        return json.dumps(log_data, default=str, ensure_ascii=False)


class ColoredTextFormatter(logging.Formatter):
    """
    Colored text formatter for console output.
    """

    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',  # Reset
    }

    def __init__(self, colored: bool = True, include_context: bool = True):
        super().__init__()
        self.colored = colored and sys.stderr.isatty()  # Only color if terminal supports it
        self.include_context = include_context

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')

        # Format level with color
        level = record.levelname
        if self.colored:
            color = self.COLORS.get(level, '')
            reset = self.COLORS['RESET']
            level = f'{color}{level}{reset}'

        # Format context
        context_str = ''
        if self.include_context:
            try:
                context = LogContextManager.get_context()
                context_parts = []

                if context.request_id:
                    context_parts.append(f'req:{context.request_id[:8]}')
                if context.agent_name:
                    context_parts.append(f'agent:{context.agent_name}')
                if context.provider_name:
                    context_parts.append(f'provider:{context.provider_name}')

                if context_parts:
                    context_str = f' [{",".join(context_parts)}]'
            except Exception:
                pass

        # Format message
        message = record.getMessage()

        # Format exception
        exc_str = ''
        if record.exc_info:
            exc_str = f'\n{self.formatException(record.exc_info)}'

        return f'{timestamp} {level:8} {record.name}{context_str}: {message}{exc_str}'


class SimpleFormatter(logging.Formatter):
    """
    Simple text formatter for basic logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as simple text."""
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        return f'{timestamp} {record.levelname:8} {record.getMessage()}'


class ConsoleHandler(logging.StreamHandler):
    """
    Enhanced console handler with proper formatting.
    """

    def __init__(self, format_type: LogFormat = LogFormat.JSON, colored: bool = True):
        super().__init__(sys.stderr)

        if format_type == LogFormat.JSON:
            self.setFormatter(JSONFormatter())
        elif format_type == LogFormat.TEXT:
            self.setFormatter(ColoredTextFormatter(colored=colored))
        else:  # SIMPLE
            self.setFormatter(SimpleFormatter())


class FileHandler(logging.handlers.RotatingFileHandler):
    """
    Enhanced file handler with rotation and proper formatting.
    """

    def __init__(
        self, filepath: str, format_type: LogFormat = LogFormat.JSON, max_size_mb: int = 100, backup_count: int = 5
    ):
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        max_bytes = max_size_mb * 1024 * 1024
        super().__init__(filepath, maxBytes=max_bytes, backupCount=backup_count)

        if format_type == LogFormat.JSON:
            self.setFormatter(JSONFormatter())
        elif format_type == LogFormat.TEXT:
            self.setFormatter(ColoredTextFormatter(colored=False))
        else:  # SIMPLE
            self.setFormatter(SimpleFormatter())


class AsyncRemoteHandler(logging.Handler):
    """
    Asynchronous remote handler for sending logs to external services.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        batch_size: int = 100,
        flush_interval: int = 5,
    ):
        super().__init__()
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self.setFormatter(JSONFormatter())

        # Buffer for batching logs
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_lock = asyncio.Lock()

        # Start background task
        self._flush_task = None
        self._start_flush_task()

    def _start_flush_task(self):
        """Start the background flush task."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._flush_task = asyncio.create_task(self._flush_loop())
        except RuntimeError:
            # No event loop running, will start when needed
            pass

    async def _flush_loop(self):
        """Background task to flush logs periodically."""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error to stderr to avoid recursion
                print(f'Error in remote log flush: {e}', file=sys.stderr)

    async def _flush_buffer(self):
        """Flush the current buffer to remote endpoint."""
        async with self._buffer_lock:
            if not self._buffer:
                return

            batch = self._buffer.copy()
            self._buffer.clear()

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                headers = {'Content-Type': 'application/json'}
                if self.api_key:
                    headers['Authorization'] = f'Bearer {self.api_key}'

                async with session.post(self.endpoint, json={'logs': batch}, headers=headers) as response:
                    if response.status >= 400:
                        print(f'Remote logging failed: {response.status}', file=sys.stderr)

        except Exception as e:
            print(f'Error sending logs to remote endpoint: {e}', file=sys.stderr)

    def emit(self, record: logging.LogRecord):
        """Emit a log record."""
        try:
            # Format the record
            formatted = self.format(record)
            log_data = json.loads(formatted)

            # Add to buffer
            asyncio.create_task(self._add_to_buffer(log_data))

        except Exception:
            self.handleError(record)

    async def _add_to_buffer(self, log_data: Dict[str, Any]):
        """Add log data to buffer and flush if needed."""
        async with self._buffer_lock:
            self._buffer.append(log_data)

            if len(self._buffer) >= self.batch_size:
                # Flush immediately if buffer is full
                await self._flush_buffer()

    def close(self):
        """Close the handler and flush remaining logs."""
        if self._flush_task:
            self._flush_task.cancel()

        # Attempt to flush remaining logs synchronously
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._flush_buffer())
        except Exception:
            pass

        super().close()


class AsyncQueueHandler(logging.Handler):
    """
    Asynchronous queue handler for non-blocking logging.
    """

    def __init__(self, target_handler: logging.Handler, queue_size: int = 1000):
        super().__init__()
        self.target_handler = target_handler
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.worker_task = None
        self._start_worker()

    def _start_worker(self):
        """Start the background worker task."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self.worker_task = asyncio.create_task(self._worker())
        except RuntimeError:
            pass

    async def _worker(self):
        """Background worker to process queued log records."""
        while True:
            try:
                record = await self.queue.get()
                if record is None:  # Shutdown signal
                    break
                self.target_handler.emit(record)
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f'Error in async log worker: {e}', file=sys.stderr)

    def emit(self, record: logging.LogRecord):
        """Emit a log record to the queue."""
        try:
            # Check if we're in an async context and there's a running loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context with a running loop
                asyncio.create_task(self._async_emit(record))
            except RuntimeError:
                # No running event loop, fall back to synchronous emit
                self.target_handler.emit(record)
        except Exception:
            self.handleError(record)

    async def _async_emit(self, record: logging.LogRecord):
        """Asynchronously emit a record."""
        try:
            await self.queue.put(record)
        except asyncio.QueueFull:
            # Queue is full, drop the log message
            print('Log queue full, dropping message', file=sys.stderr)

    def close(self):
        """Close the handler."""
        if self.worker_task:
            # Send shutdown signal
            asyncio.create_task(self.queue.put(None))
            self.worker_task.cancel()

        self.target_handler.close()
        super().close()
