"""
Main Logger Module

Provides the primary logging interface for the reagentic framework.
"""

import logging
import atexit
from typing import Dict, Optional, Any
from pathlib import Path

from .config import LoggingConfig, LogLevel
from .context import LogContext, LogContextManager
from .handlers import ConsoleHandler, FileHandler, AsyncRemoteHandler, AsyncQueueHandler

# Global configuration
_config: Optional[LoggingConfig] = None
_loggers: Dict[str, logging.Logger] = {}
_setup_done: bool = False


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    Setup the logging system with the given configuration.

    Args:
        config: Logging configuration. If None, loads from environment.
    """
    global _config, _setup_done

    if _setup_done:
        return

    # Use provided config or load from environment
    _config = config or LoggingConfig.from_env()

    # Setup root logger
    root_logger = logging.getLogger('reagentic')
    root_logger.setLevel(getattr(logging, _config.level.value))

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Setup console handler
    if _config.console.enabled:
        console_handler = ConsoleHandler(format_type=_config.console.format, colored=_config.console.colored)
        console_handler.setLevel(getattr(logging, _config.console.level.value))

        if _config.async_logging:
            console_handler = AsyncQueueHandler(console_handler, _config.queue_size)

        root_logger.addHandler(console_handler)

    # Setup file handler
    if _config.file.enabled and _config.file.filepath:
        file_handler = FileHandler(
            filepath=_config.file.filepath,
            format_type=_config.file.format,
            max_size_mb=_config.file.max_size_mb,
            backup_count=_config.file.backup_count,
        )
        file_handler.setLevel(getattr(logging, _config.file.level.value))

        if _config.async_logging:
            file_handler = AsyncQueueHandler(file_handler, _config.queue_size)

        root_logger.addHandler(file_handler)

    # Setup remote handler
    if _config.remote.enabled and _config.remote.endpoint:
        remote_handler = AsyncRemoteHandler(
            endpoint=_config.remote.endpoint,
            api_key=_config.remote.api_key,
            timeout=_config.remote.timeout,
            batch_size=_config.remote.batch_size,
            flush_interval=_config.remote.flush_interval,
        )
        remote_handler.setLevel(getattr(logging, _config.remote.level.value))
        root_logger.addHandler(remote_handler)

    # Prevent propagation to avoid duplicate logs
    root_logger.propagate = False

    # Register cleanup function
    atexit.register(_cleanup_logging)

    _setup_done = True


def get_logger(name: str = None) -> 'ReagenticLogger':
    """
    Get a logger instance.

    Args:
        name: Logger name. If None, uses the calling module name.

    Returns:
        ReagenticLogger instance.
    """
    # Auto-setup if not done
    if not _setup_done:
        setup_logging()

    # Default name based on caller
    if name is None:
        import inspect

        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'reagentic')

    # Ensure name is under reagentic namespace
    if not name.startswith('reagentic'):
        name = f'reagentic.{name}'

    # Return cached logger or create new one
    if name not in _loggers:
        _loggers[name] = ReagenticLogger(logging.getLogger(name))

    return _loggers[name]


def _cleanup_logging():
    """Cleanup logging handlers on shutdown."""
    for logger in _loggers.values():
        logger._cleanup()


class ReagenticLogger:
    """
    Enhanced logger with context support and reagentic-specific features.
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)

    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        kwargs['exc_info'] = True
        self._log(logging.ERROR, message, **kwargs)

    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method."""
        if not self._logger.isEnabledFor(level):
            return

        # Extract extra fields
        extra = kwargs.pop('extra', {})

        # Add remaining kwargs as extra fields
        extra.update(kwargs)

        # Create log record
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn='',
            lno=0,
            msg=message,
            args=(),
            exc_info=extra.pop('exc_info', None),
        )

        # Add extra fields to record
        if extra:
            record.extra = extra

        # Handle the record
        self._logger.handle(record)

    # Reagentic-specific logging methods

    def log_agent_start(self, agent_name: str, **kwargs):
        """Log agent start."""
        if not _config or not _config.log_agent_interactions:
            return
        self.info(f'Agent started: {agent_name}', agent_name=agent_name, event_type='agent_start', **kwargs)

    def log_agent_end(self, agent_name: str, success: bool = True, **kwargs):
        """Log agent end."""
        if not _config or not _config.log_agent_interactions:
            return
        status = 'success' if success else 'failure'
        self.info(
            f'Agent finished: {agent_name} ({status})',
            agent_name=agent_name,
            event_type='agent_end',
            success=success,
            **kwargs,
        )

    def log_model_call(self, provider: str, model: str, **kwargs):
        """Log model API call."""
        if not _config or not _config.log_model_calls:
            return
        self.debug(
            f'Model call: {provider}/{model}',
            provider_name=provider,
            model_name=model,
            event_type='model_call',
            **kwargs,
        )

    def log_model_response(self, provider: str, model: str, tokens_used: Optional[int] = None, **kwargs):
        """Log model API response."""
        if not _config or not _config.log_model_calls:
            return
        self.debug(
            f'Model response: {provider}/{model}',
            provider_name=provider,
            model_name=model,
            tokens_used=tokens_used,
            event_type='model_response',
            **kwargs,
        )

    def log_provider_call(self, provider: str, operation: str, **kwargs):
        """Log provider operation."""
        if not _config or not _config.log_provider_calls:
            return
        self.debug(
            f'Provider operation: {provider}.{operation}',
            provider_name=provider,
            operation=operation,
            event_type='provider_call',
            **kwargs,
        )

    def log_error(self, error: Exception, context: str = '', **kwargs):
        """Log error with context."""
        error_type = type(error).__name__
        self.error(
            f'Error in {context}: {error_type}: {error}',
            error_type=error_type,
            error_message=str(error),
            context=context,
            event_type='error',
            exc_info=True,
            **kwargs,
        )

    def log_performance(self, operation: str, duration_ms: float, **kwargs):
        """Log performance metrics."""
        self.info(
            f'Performance: {operation} took {duration_ms:.2f}ms',
            operation=operation,
            duration_ms=duration_ms,
            event_type='performance',
            **kwargs,
        )

    def log_user_action(self, action: str, user_id: Optional[str] = None, **kwargs):
        """Log user action."""
        self.info(f'User action: {action}', action=action, user_id=user_id, event_type='user_action', **kwargs)

    def _cleanup(self):
        """Cleanup logger handlers."""
        for handler in self._logger.handlers:
            try:
                handler.close()
            except Exception:
                pass


# Convenience functions for quick access
def debug(message: str, **kwargs):
    """Quick debug logging."""
    get_logger().debug(message, **kwargs)


def info(message: str, **kwargs):
    """Quick info logging."""
    get_logger().info(message, **kwargs)


def warning(message: str, **kwargs):
    """Quick warning logging."""
    get_logger().warning(message, **kwargs)


def error(message: str, **kwargs):
    """Quick error logging."""
    get_logger().error(message, **kwargs)


def critical(message: str, **kwargs):
    """Quick critical logging."""
    get_logger().critical(message, **kwargs)


def exception(message: str, **kwargs):
    """Quick exception logging."""
    get_logger().exception(message, **kwargs)
