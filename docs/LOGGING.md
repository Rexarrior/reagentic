# Reagentic Logging System

The Reagentic framework includes a comprehensive, modern logging system designed for observability, debugging, and monitoring of AI agent applications. The system follows current best practices for structured logging, context tracking, and performance monitoring.

## 🚀 Features

- **Structured Logging**: JSON format for machine readability and analysis
- **Context Tracking**: Automatic context propagation across async operations
- **Multiple Handlers**: Console, file, and remote logging support
- **Performance Monitoring**: Built-in performance tracking and metrics
- **Async Support**: Non-blocking logging with async handlers
- **Agent Integration**: Automatic logging of agent operations and model calls
- **Provider Integration**: Automatic logging of provider operations
- **Configurable**: Environment-based and programmatic configuration
- **Error Handling**: Comprehensive error logging with stack traces

## 📋 Quick Start

### Basic Usage

```python
from reagentic import get_logger, setup_logging

# Setup logging (optional - auto-setup happens on first use)
setup_logging()

# Get a logger
logger = get_logger('myapp')

# Log messages
logger.info("Application started")
logger.error("Something went wrong", error_code=500)
```

### With Context

```python
from reagentic.logging import log_context, new_request_context

# Track requests with auto-generated request ID
with new_request_context(user_id="user_123"):
    logger.info("Processing user request")
    
    # Add operation-specific context
    with log_context(operation="search", query="python"):
        logger.info("Performing search")
```

## 🔧 Configuration

### Environment Variables

The logging system can be configured via environment variables:

```bash
# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
export REAGENTIC_LOG_LEVEL=INFO

# Log format (json, text, simple)
export REAGENTIC_LOG_FORMAT=json

# Console logging
export REAGENTIC_LOG_CONSOLE_ENABLED=true
export REAGENTIC_LOG_CONSOLE_COLORED=true

# File logging
export REAGENTIC_LOG_FILE_ENABLED=true
export REAGENTIC_LOG_DIR=./logs
export REAGENTIC_LOG_FILE_MAX_SIZE_MB=100
export REAGENTIC_LOG_FILE_BACKUP_COUNT=5

# Remote logging
export REAGENTIC_LOG_REMOTE_ENABLED=false
export REAGENTIC_LOG_REMOTE_ENDPOINT=https://logs.example.com
export REAGENTIC_LOG_REMOTE_API_KEY=your_api_key

# Context and performance settings
export REAGENTIC_LOG_INCLUDE_CONTEXT=true
export REAGENTIC_LOG_ASYNC=true
export REAGENTIC_LOG_QUEUE_SIZE=1000

# Framework-specific logging
export REAGENTIC_LOG_MODEL_CALLS=true
export REAGENTIC_LOG_AGENT_INTERACTIONS=true
export REAGENTIC_LOG_PROVIDER_CALLS=true
```

### Programmatic Configuration

```python
from reagentic.logging import LoggingConfig, setup_logging

config = LoggingConfig(
    level=LoggingConfig.LogLevel.DEBUG,
    format=LoggingConfig.LogFormat.JSON,
    console=LoggingConfig.ConsoleConfig(
        enabled=True,
        colored=True
    ),
    file=LoggingConfig.FileConfig(
        enabled=True,
        filepath="./logs/app.log",
        max_size_mb=100,
        backup_count=5
    ),
    remote=LoggingConfig.RemoteConfig(
        enabled=True,
        endpoint="https://logs.example.com",
        api_key="your_api_key"
    )
)

setup_logging(config)
```

## 📊 Context Management

The logging system provides sophisticated context tracking that works across async operations:

### Basic Context

```python
from reagentic.logging import log_context

with log_context(user_id="123", operation="search"):
    logger.info("Starting search")  # Will include user_id and operation
```

### Request Context

```python
from reagentic.logging import new_request_context

with new_request_context(user_id="123", session_id="session_456"):
    logger.info("Processing request")  # Includes auto-generated request_id
```

### Agent Context

```python
from reagentic.logging import agent_context

with agent_context("assistant", run_id="run_123"):
    logger.info("Agent operation")  # Includes agent_name and run_id
```

### Provider Context

```python
from reagentic.logging import provider_context

with provider_context("openrouter", "gpt-4"):
    logger.info("Provider call")  # Includes provider_name and model_name
```

## 🤖 Agent & Provider Integration

The logging system automatically integrates with Reagentic agents and providers:

### Automatic Agent Logging

```python
from reagentic.providers import openrouter
from agents import Agent, Runner

# Providers automatically log operations
provider = openrouter.auto()  # Logged automatically

# Agents log their interactions
agent = Agent(
    name="Assistant",
    instructions="You are helpful",
    model=provider.get_openai_model()  # Model calls logged automatically
)

# Agent runs are tracked
result = Runner.run_sync(agent, "Hello!")  # Execution logged
```

### Manual Agent Logging

```python
logger = get_logger('myapp')

# Log agent lifecycle
logger.log_agent_start("assistant", task="greeting")
# ... agent operations ...
logger.log_agent_end("assistant", success=True)

# Log model calls
logger.log_model_call("openrouter", "gpt-4", prompt_tokens=100)
logger.log_model_response("openrouter", "gpt-4", tokens_used=150)

# Log provider operations
logger.log_provider_call("openrouter", "get_client")
```

## 📈 Performance Monitoring

Built-in performance tracking capabilities:

```python
import time

logger = get_logger('performance')

# Measure operation performance
start_time = time.time()
# ... your operation ...
duration_ms = (time.time() - start_time) * 1000

logger.log_performance("database_query", duration_ms)

# Log with additional context
logger.log_performance("api_call", duration_ms, 
                      endpoint="/users", method="GET")
```

## ❌ Error Handling

Comprehensive error logging with context:

```python
try:
    # Some operation that might fail
    result = risky_operation()
except Exception as e:
    # Log error with full context and stack trace
    logger.log_error(e, "risky_operation")
    
    # Or use standard exception logging
    logger.exception("Operation failed")
```

## 📝 Log Formats

### JSON Format (Default)

```json
{
  "timestamp": "2025-01-01T10:30:00.123456",
  "level": "INFO",
  "logger": "reagentic.agents.model_chooser",
  "message": "Model selection completed",
  "module": "model_chooser",
  "function": "choose_model",
  "line": 45,
  "thread": 140123456789,
  "thread_name": "MainThread",
  "context": {
    "request_id": "req-123e4567-e89b-12d3-a456-426614174000",
    "agent_name": "assistant",
    "provider_name": "openrouter",
    "model_name": "gpt-4"
  },
  "selected_model": "gpt-4-turbo",
  "success": true,
  "event_type": "model_selection"
}
```

### Text Format

```
2025-01-01 10:30:00 INFO     reagentic.agents.model_chooser [req:123e4567,agent:assistant,provider:openrouter]: Model selection completed
```

### Simple Format

```
10:30:00 INFO     Model selection completed
```

## 🔌 Remote Logging

Send logs to external services for centralized monitoring:

```python
from reagentic.logging import LoggingConfig

config = LoggingConfig(
    remote=LoggingConfig.RemoteConfig(
        enabled=True,
        endpoint="https://logs.example.com/api/logs",
        api_key="your_api_key",
        timeout=30,
        batch_size=100,
        flush_interval=5
    )
)
```

The remote handler:
- Batches logs for efficiency
- Sends async HTTP requests
- Handles retries and errors gracefully
- Flushes logs periodically

## 🏗️ Architecture

### Components

1. **Logger**: Main interface (`ReagenticLogger`)
2. **Context Manager**: Tracks contextual information (`LogContextManager`)
3. **Handlers**: Output destinations (Console, File, Remote)
4. **Formatters**: Format log messages (JSON, Text, Simple)
5. **Configuration**: Environment and programmatic config (`LoggingConfig`)

### Handler Types

- **ConsoleHandler**: Colored/plain text output to stderr
- **FileHandler**: Rotating file output with size limits
- **AsyncRemoteHandler**: Batched HTTP logging to remote services
- **AsyncQueueHandler**: Non-blocking wrapper for other handlers

### Integration Points

- **Providers**: Automatic logging of provider operations and model calls
- **Agents**: Automatic logging of agent lifecycle and interactions
- **Context**: Automatic context propagation across async operations

## 📚 Examples

See `examples/logging_example.py` for comprehensive usage examples including:

- Context tracking across operations
- Agent and provider integration
- Performance monitoring
- Error handling
- Different configuration options

Run the example:

```bash
# Default configuration
python examples/logging_example.py

# With custom environment
REAGENTIC_LOG_FORMAT=text REAGENTIC_LOG_LEVEL=debug python examples/logging_example.py
```

## 🛠️ Advanced Usage

### Custom Context Fields

```python
from reagentic.logging import LogContextManager

# Add custom context fields
LogContextManager.update_context(
    custom_field="value",
    experiment_id="exp_123"
)

logger.info("Custom context example")  # Will include custom fields
```

### Logger Hierarchy

```python
# Create loggers for different components
auth_logger = get_logger('myapp.auth')
db_logger = get_logger('myapp.database')
api_logger = get_logger('myapp.api')

# Each has its own namespace in logs
auth_logger.info("User authenticated")
db_logger.info("Query executed")
api_logger.info("Request processed")
```

### Performance Context Manager

```python
import time
from contextlib import contextmanager

@contextmanager
def log_performance(operation_name):
    start_time = time.time()
    try:
        yield
    finally:
        duration = (time.time() - start_time) * 1000
        logger.log_performance(operation_name, duration)

# Usage
with log_performance("complex_operation"):
    # Your operation here
    result = complex_operation()
```

## 🔍 Troubleshooting

### Common Issues

1. **Logs not appearing**: Check log level configuration
2. **Performance issues**: Ensure async logging is enabled
3. **Context not propagating**: Use proper context managers
4. **File rotation not working**: Check file permissions and disk space

### Debug Logging

Enable debug logging to see system internals:

```bash
export REAGENTIC_LOG_LEVEL=debug
```

### Log File Locations

Default log locations:
- Console: stderr
- File: `./logs/reagentic.log`
- Remote: Configured endpoint

## 🔐 Security Considerations

- **Sensitive Data**: Be careful not to log sensitive information like API keys
- **Log Retention**: Configure appropriate log rotation and retention policies
- **Remote Logging**: Use HTTPS and proper authentication for remote endpoints
- **File Permissions**: Ensure log files have appropriate permissions

## 🚀 Best Practices

1. **Use Context**: Always use context managers for tracking operations
2. **Structured Data**: Include relevant structured data in log messages
3. **Error Context**: Provide context when logging errors
4. **Performance**: Use async logging for high-throughput applications
5. **Monitoring**: Set up log analysis and alerting for production systems
6. **Testing**: Use different log levels for development vs production

## 📋 Environment Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `REAGENTIC_LOG_LEVEL` | `INFO` | Minimum log level |
| `REAGENTIC_LOG_FORMAT` | `json` | Log format (json/text/simple) |
| `REAGENTIC_LOG_CONSOLE_ENABLED` | `true` | Enable console logging |
| `REAGENTIC_LOG_CONSOLE_COLORED` | `true` | Enable colored console output |
| `REAGENTIC_LOG_FILE_ENABLED` | `true` | Enable file logging |
| `REAGENTIC_LOG_DIR` | `./logs` | Log file directory |
| `REAGENTIC_LOG_FILE_MAX_SIZE_MB` | `100` | Max log file size |
| `REAGENTIC_LOG_FILE_BACKUP_COUNT` | `5` | Number of backup files |
| `REAGENTIC_LOG_REMOTE_ENABLED` | `false` | Enable remote logging |
| `REAGENTIC_LOG_REMOTE_ENDPOINT` | None | Remote logging endpoint |
| `REAGENTIC_LOG_REMOTE_API_KEY` | None | Remote logging API key |
| `REAGENTIC_LOG_INCLUDE_CONTEXT` | `true` | Include context in logs |
| `REAGENTIC_LOG_ASYNC` | `true` | Enable async logging |
| `REAGENTIC_LOG_QUEUE_SIZE` | `1000` | Async queue size |
| `REAGENTIC_LOG_MODEL_CALLS` | `true` | Log model API calls |
| `REAGENTIC_LOG_AGENT_INTERACTIONS` | `true` | Log agent interactions |
| `REAGENTIC_LOG_PROVIDER_CALLS` | `true` | Log provider operations | 
