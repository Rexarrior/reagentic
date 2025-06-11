"""
Logging Configuration

Handles configuration for the reagentic logging system with environment-based settings.
"""

import os
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path


class LogLevel(Enum):
    """Log levels following Python logging standards."""

    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'


class LogFormat(Enum):
    """Available log formats."""

    JSON = 'json'
    TEXT = 'text'
    SIMPLE = 'simple'


@dataclass
class HandlerConfig:
    """Configuration for individual log handlers."""

    enabled: bool = True
    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.TEXT


@dataclass
class ConsoleConfig(HandlerConfig):
    """Console handler configuration."""

    colored: bool = True


@dataclass
class FileConfig(HandlerConfig):
    """File handler configuration."""

    filepath: Optional[str] = None
    max_size_mb: int = 100
    backup_count: int = 5
    rotation_enabled: bool = True


@dataclass
class RemoteConfig(HandlerConfig):
    """Remote handler configuration (e.g., for centralized logging)."""

    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 30
    batch_size: int = 100
    flush_interval: int = 5


@dataclass
class LoggingConfig:
    """Main logging configuration."""

    # Global settings
    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.TEXT

    # Handler configurations
    console: ConsoleConfig = field(default_factory=ConsoleConfig)
    file: FileConfig = field(default_factory=FileConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)

    # Context settings
    include_context: bool = True
    include_stack_trace: bool = True

    # Performance settings
    async_logging: bool = False  # Disabled by default to avoid event loop issues
    queue_size: int = 1000

    # Framework-specific settings
    log_model_calls: bool = True
    log_agent_interactions: bool = True
    log_provider_calls: bool = True

    @classmethod
    def from_env(cls) -> 'LoggingConfig':
        """Create configuration from environment variables."""

        # Parse log level
        level_str = os.getenv('REAGENTIC_LOG_LEVEL', 'INFO').upper()
        try:
            level = LogLevel(level_str)
        except ValueError:
            level = LogLevel.INFO

        # Parse format
        format_str = os.getenv('REAGENTIC_LOG_FORMAT', 'text').lower()
        try:
            format_type = LogFormat(format_str)
        except ValueError:
            format_type = LogFormat.TEXT

        # Console configuration
        console = ConsoleConfig(
            enabled=os.getenv('REAGENTIC_LOG_CONSOLE_ENABLED', 'true').lower() == 'true',
            level=level,
            format=format_type,
            colored=os.getenv('REAGENTIC_LOG_CONSOLE_COLORED', 'true').lower() == 'true',
        )

        # File configuration
        log_dir = os.getenv('REAGENTIC_LOG_DIR', './logs')
        file_config = FileConfig(
            enabled=os.getenv('REAGENTIC_LOG_FILE_ENABLED', 'true').lower() == 'true',
            level=level,
            format=format_type,
            filepath=os.path.join(log_dir, 'reagentic.log'),
            max_size_mb=int(os.getenv('REAGENTIC_LOG_FILE_MAX_SIZE_MB', '100')),
            backup_count=int(os.getenv('REAGENTIC_LOG_FILE_BACKUP_COUNT', '5')),
        )

        # Remote configuration
        remote = RemoteConfig(
            enabled=os.getenv('REAGENTIC_LOG_REMOTE_ENABLED', 'false').lower() == 'true',
            level=level,
            format=LogFormat.JSON,  # Remote logging always uses JSON
            endpoint=os.getenv('REAGENTIC_LOG_REMOTE_ENDPOINT'),
            api_key=os.getenv('REAGENTIC_LOG_REMOTE_API_KEY'),
            timeout=int(os.getenv('REAGENTIC_LOG_REMOTE_TIMEOUT', '30')),
            batch_size=int(os.getenv('REAGENTIC_LOG_REMOTE_BATCH_SIZE', '100')),
            flush_interval=int(os.getenv('REAGENTIC_LOG_REMOTE_FLUSH_INTERVAL', '5')),
        )

        return cls(
            level=level,
            format=format_type,
            console=console,
            file=file_config,
            remote=remote,
            include_context=os.getenv('REAGENTIC_LOG_INCLUDE_CONTEXT', 'true').lower() == 'true',
            include_stack_trace=os.getenv('REAGENTIC_LOG_INCLUDE_STACK_TRACE', 'true').lower() == 'true',
            async_logging=os.getenv('REAGENTIC_LOG_ASYNC', 'false').lower() == 'true',
            queue_size=int(os.getenv('REAGENTIC_LOG_QUEUE_SIZE', '1000')),
            log_model_calls=os.getenv('REAGENTIC_LOG_MODEL_CALLS', 'true').lower() == 'true',
            log_agent_interactions=os.getenv('REAGENTIC_LOG_AGENT_INTERACTIONS', 'true').lower() == 'true',
            log_provider_calls=os.getenv('REAGENTIC_LOG_PROVIDER_CALLS', 'true').lower() == 'true',
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'level': self.level.value,
            'format': self.format.value,
            'console': {
                'enabled': self.console.enabled,
                'level': self.console.level.value,
                'format': self.console.format.value,
                'colored': self.console.colored,
            },
            'file': {
                'enabled': self.file.enabled,
                'level': self.file.level.value,
                'format': self.file.format.value,
                'filepath': self.file.filepath,
                'max_size_mb': self.file.max_size_mb,
                'backup_count': self.file.backup_count,
            },
            'remote': {
                'enabled': self.remote.enabled,
                'level': self.remote.level.value,
                'endpoint': self.remote.endpoint,
                'timeout': self.remote.timeout,
                'batch_size': self.remote.batch_size,
                'flush_interval': self.remote.flush_interval,
            },
            'include_context': self.include_context,
            'include_stack_trace': self.include_stack_trace,
            'async_logging': self.async_logging,
            'queue_size': self.queue_size,
            'log_model_calls': self.log_model_calls,
            'log_agent_interactions': self.log_agent_interactions,
            'log_provider_calls': self.log_provider_calls,
        }
