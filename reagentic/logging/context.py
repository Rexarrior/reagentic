"""
Logging Context Management

Provides context tracking for the reagentic logging system, allowing contextual
information to be automatically included in log messages.
"""

import asyncio
import contextlib
import threading
import uuid
from typing import Dict, Any, Optional, ContextManager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LogContext:
    """
    Context information that gets automatically included in log messages.
    """

    # Request/Session tracking
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    # Agent tracking
    agent_name: Optional[str] = None
    agent_run_id: Optional[str] = None

    # Provider tracking
    provider_name: Optional[str] = None
    model_name: Optional[str] = None

    # Custom fields
    extra: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        result = {
            'request_id': self.request_id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'agent_name': self.agent_name,
            'agent_run_id': self.agent_run_id,
            'provider_name': self.provider_name,
            'model_name': self.model_name,
            'created_at': self.created_at.isoformat(),
        }

        # Add extra fields
        result.update(self.extra)

        # Remove None values
        return {k: v for k, v in result.items() if v is not None}

    def update(self, **kwargs) -> 'LogContext':
        """Create a new context with updated values."""
        new_extra = self.extra.copy()

        # Handle extra fields
        if 'extra' in kwargs:
            new_extra.update(kwargs.pop('extra'))

        return LogContext(
            request_id=kwargs.get('request_id', self.request_id),
            session_id=kwargs.get('session_id', self.session_id),
            user_id=kwargs.get('user_id', self.user_id),
            agent_name=kwargs.get('agent_name', self.agent_name),
            agent_run_id=kwargs.get('agent_run_id', self.agent_run_id),
            provider_name=kwargs.get('provider_name', self.provider_name),
            model_name=kwargs.get('model_name', self.model_name),
            extra=new_extra,
            created_at=self.created_at,
        )


# Context variables for async and threading support
_context_var: ContextVar[LogContext] = ContextVar('log_context', default=LogContext())
_thread_local = threading.local()


class LogContextManager:
    """
    Manages logging context using both contextvars (for async) and thread-local storage.
    """

    @staticmethod
    def get_context() -> LogContext:
        """Get the current logging context."""
        try:
            # Try async context first
            return _context_var.get()
        except LookupError:
            # Fall back to thread-local storage
            return getattr(_thread_local, 'context', LogContext())

    @staticmethod
    def set_context(context: LogContext) -> None:
        """Set the current logging context."""
        # Set in both async and thread-local storage
        _context_var.set(context)
        _thread_local.context = context

    @staticmethod
    def update_context(**kwargs) -> None:
        """Update the current context with new values."""
        current = LogContextManager.get_context()
        updated = current.update(**kwargs)
        LogContextManager.set_context(updated)

    @staticmethod
    def clear_context() -> None:
        """Clear the current context."""
        LogContextManager.set_context(LogContext())

    @staticmethod
    @contextlib.contextmanager
    def context(**kwargs) -> ContextManager[LogContext]:
        """Context manager for temporary context changes."""
        previous = LogContextManager.get_context()
        try:
            new_context = previous.update(**kwargs)
            LogContextManager.set_context(new_context)
            yield new_context
        finally:
            LogContextManager.set_context(previous)


def log_context(**kwargs) -> ContextManager[LogContext]:
    """
    Convenience function for creating a logging context.

    Usage:
        with log_context(user_id="123", agent_name="assistant"):
            logger.info("This will include the context")
    """
    return LogContextManager.context(**kwargs)


def new_request_context(
    user_id: Optional[str] = None, session_id: Optional[str] = None, **kwargs
) -> ContextManager[LogContext]:
    """
    Create a new request context with auto-generated request_id.
    """
    request_id = str(uuid.uuid4())
    return log_context(request_id=request_id, user_id=user_id, session_id=session_id, **kwargs)


def agent_context(agent_name: str, run_id: Optional[str] = None, **kwargs) -> ContextManager[LogContext]:
    """
    Create context for agent operations.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())

    return log_context(agent_name=agent_name, agent_run_id=run_id, **kwargs)


def provider_context(provider_name: str, model_name: Optional[str] = None, **kwargs) -> ContextManager[LogContext]:
    """
    Create context for provider operations.
    """
    return log_context(provider_name=provider_name, model_name=model_name, **kwargs)


# Utility functions for getting specific context values
def get_request_id() -> Optional[str]:
    """Get the current request ID."""
    return LogContextManager.get_context().request_id


def get_session_id() -> Optional[str]:
    """Get the current session ID."""
    return LogContextManager.get_context().session_id


def get_user_id() -> Optional[str]:
    """Get the current user ID."""
    return LogContextManager.get_context().user_id


def get_agent_name() -> Optional[str]:
    """Get the current agent name."""
    return LogContextManager.get_context().agent_name
