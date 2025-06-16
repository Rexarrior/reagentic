# MCP Hub - Model Context Protocol Server Management

The MCP Hub provides centralized lifecycle management for MCP (Model Context Protocol) servers in the reagentic framework. It handles server connections, health monitoring, and provides a clean abstraction for working with multiple MCP servers.

## Features

- **Centralized Management**: Manage multiple MCP servers from a single hub
- **Pydantic Configuration**: Type-safe configuration with validation
- **Async Context Manager**: Clean resource management with automatic cleanup
- **Health Monitoring**: Built-in health checks and status monitoring
- **Error Recovery**: Automatic retry logic and graceful error handling
- **Concurrent Connections**: Efficient parallel server connections
- **Flexible Configuration**: Support for environment variables and dynamic config

## Quick Start

### Basic Usage

```python
import asyncio
from reagentic.mcp import McpHub, create_hub_from_dict

# Create configuration
config = {
    "servers": {
        "tracker": {
            "command": "/path/to/tracker/mcp",
            "env": {"STARTREK_TOKEN": "your_token_here"}
        },
        "search": {
            "command": "/path/to/search/mcp",
            "env": {"SEARCH_API_KEY": "your_key"},
            "args": ["--verbose"]
        }
    },
    "auto_connect": True
}

async def main():
    # Create and use hub
    hub = create_hub_from_dict(config)
    
    async with hub:
        # Get connected servers
        tracker = hub.get_server("tracker")
        search = hub.get_server("search")
        
        # Use servers for MCP operations
        if tracker:
            # Use tracker server...
            pass

asyncio.run(main())
```

### Manual Server Management

```python
async def manual_management():
    hub = create_hub_from_dict({
        "servers": {
            "server1": {"command": "/path/to/server1"},
            "server2": {"command": "/path/to/server2"}
        },
        "auto_connect": False  # Don't auto-connect
    })
    
    try:
        await hub.start()
        
        # Connect servers individually
        success1 = await hub.connect_server("server1")
        success2 = await hub.connect_server("server2")
        
        # Check status
        status = hub.get_connection_status()
        print(f"Connection status: {status}")
        
        # Perform health check
        health = await hub.health_check()
        print(f"Health: {health['overall_status']}")
        
    finally:
        await hub.stop()
```

## Configuration

### Server Configuration

Each MCP server is configured using the `McpServerConfig` model:

```python
from reagentic.mcp import McpServerConfig

config = McpServerConfig(
    command="/path/to/mcp/server",           # Required: Command to execute
    args=["--verbose", "--port", "8080"],    # Optional: Command arguments
    env={"API_KEY": "secret"},               # Optional: Environment variables
    working_directory="/tmp/workspace",      # Optional: Working directory
    cache_tools_list=True,                   # Optional: Cache tools list (default: True)
    timeout=30                               # Optional: Connection timeout (default: 30)
)
```

#### Configuration Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `command` | `str` | Yes | - | Command to execute the MCP server |
| `args` | `List[str]` | No | `[]` | Command line arguments |
| `env` | `Dict[str, str]` | No | `{}` | Environment variables |
| `working_directory` | `str` | No | `None` | Working directory for the server |
| `cache_tools_list` | `bool` | No | `True` | Whether to cache the tools list |
| `timeout` | `int` | No | `30` | Connection timeout in seconds |

### Hub Configuration

The hub itself is configured using the `McpHubConfig` model:

```python
from reagentic.mcp import McpHubConfig, McpServerConfig

server_configs = {
    "server1": McpServerConfig(command="/path/to/server1"),
    "server2": McpServerConfig(command="/path/to/server2")
}

hub_config = McpHubConfig(
    servers=server_configs,                  # Required: Server configurations
    auto_connect=True,                       # Optional: Auto-connect on start
    max_concurrent_connections=10,           # Optional: Max concurrent connections
    connection_retry_attempts=3,             # Optional: Retry attempts
    connection_retry_delay=1.0               # Optional: Delay between retries
)
```

#### Hub Configuration Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `servers` | `Dict[str, McpServerConfig]` | Yes | `{}` | Server configurations |
| `auto_connect` | `bool` | No | `True` | Auto-connect servers on startup |
| `max_concurrent_connections` | `int` | No | `10` | Maximum concurrent connections |
| `connection_retry_attempts` | `int` | No | `3` | Number of retry attempts |
| `connection_retry_delay` | `float` | No | `1.0` | Delay between retries (seconds) |

### Dictionary Configuration

For convenience, you can create configurations from dictionaries:

```python
from reagentic.mcp import create_hub_from_dict

config_dict = {
    "servers": {
        "tracker": {
            "command": "/path/to/tracker/mcp",
            "env": {"STARTREK_TOKEN": "token"},
            "timeout": 30
        },
        "search": {
            "command": "/path/to/search/mcp",
            "args": ["--index", "main"],
            "env": {"API_KEY": "key"}
        }
    },
    "auto_connect": True,
    "connection_retry_attempts": 2
}

hub = create_hub_from_dict(config_dict)
```

## API Reference

### McpHub Class

#### Lifecycle Methods

```python
async def start() -> None
```
Start the hub and optionally auto-connect to servers.

```python
async def stop() -> None
```
Stop the hub and disconnect from all servers.

#### Connection Management

```python
async def connect_server(server_name: str) -> bool
```
Connect to a specific server. Returns `True` if successful.

```python
async def disconnect_server(server_name: str) -> bool
```
Disconnect from a specific server. Returns `True` if successful.

```python
async def connect_all_servers() -> Dict[str, bool]
```
Connect to all configured servers concurrently. Returns connection results.

```python
async def disconnect_all_servers() -> Dict[str, bool]
```
Disconnect from all connected servers. Returns disconnection results.

#### Server Access

```python
def get_server(server_name: str) -> Optional[MCPServerStdio]
```
Get a connected server by name. Returns `None` if not connected.

```python
def get_all_servers() -> Dict[str, MCPServerStdio]
```
Get all connected servers as a dictionary.

```python
def is_server_connected(server_name: str) -> bool
```
Check if a server is connected.

#### Status and Monitoring

```python
def get_connection_status() -> Dict[str, bool]
```
Get connection status for all configured servers.

```python
def get_hub_status() -> Dict[str, Any]
```
Get comprehensive hub status information.

```python
async def health_check() -> Dict[str, Any]
```
Perform health check on all connected servers.

### Context Manager Support

The hub supports async context manager protocol:

```python
async with hub:
    # Hub is started and servers are connected
    server = hub.get_server("my_server")
    # Use server...
# Hub is automatically stopped and servers disconnected
```

## Examples

### Environment Variables

```python
import os
from reagentic.mcp import create_hub_from_dict

# Set environment variables
os.environ["TRACKER_TOKEN"] = "your_tracker_token"
os.environ["SEARCH_KEY"] = "your_search_key"

config = {
    "servers": {
        "tracker": {
            "command": "/path/to/tracker/mcp",
            "env": {
                "STARTREK_TOKEN": os.environ["TRACKER_TOKEN"],
                "DEBUG": "true"
            }
        },
        "search": {
            "command": "/path/to/search/mcp",
            "env": {
                "SEARCH_API_KEY": os.environ["SEARCH_KEY"]
            }
        }
    }
}

hub = create_hub_from_dict(config)
```

### Error Handling

```python
async def robust_connection():
    hub = create_hub_from_dict(config)
    
    try:
        await hub.start()
        
        # Check which servers connected
        status = hub.get_connection_status()
        failed_servers = [name for name, connected in status.items() if not connected]
        
        if failed_servers:
            print(f"Failed to connect to: {failed_servers}")
            
            # Try to reconnect failed servers
            for server_name in failed_servers:
                print(f"Retrying connection to {server_name}...")
                success = await hub.connect_server(server_name)
                if success:
                    print(f"Successfully reconnected to {server_name}")
        
        # Perform health check
        health = await hub.health_check()
        if health['overall_status'] != 'healthy':
            print("Some servers are unhealthy:")
            for name, info in health['server_health'].items():
                if info['status'] != 'healthy':
                    print(f"  {name}: {info['status']} - {info.get('error', 'Unknown error')}")
    
    finally:
        await hub.stop()
```

### Integration with Reagentic Layers

```python
from reagentic.layers import ActionLayer
from reagentic.mcp import create_hub_from_dict

class McpActionLayer(ActionLayer):
    def __init__(self, mcp_hub: McpHub, **kwargs):
        super().__init__(**kwargs)
        self.mcp_hub = mcp_hub
    
    async def execute_action(self, action_data):
        # Get appropriate MCP server based on action
        server_name = action_data.get('mcp_server', 'default')
        server = self.mcp_hub.get_server(server_name)
        
        if not server:
            raise ValueError(f"MCP server '{server_name}' not available")
        
        # Use server to execute action
        # ... implementation details ...

# Usage
async def main():
    # Create MCP hub
    mcp_config = {
        "servers": {
            "tracker": {"command": "/path/to/tracker/mcp"},
            "search": {"command": "/path/to/search/mcp"}
        }
    }
    
    hub = create_hub_from_dict(mcp_config)
    
    async with hub:
        # Create action layer with MCP support
        action_layer = McpActionLayer(mcp_hub=hub)
        
        # Use in your reagentic app...
```

## Best Practices

### 1. Configuration Management

- Use environment variables for sensitive data like API keys
- Validate configurations early in your application startup
- Use meaningful server names that reflect their purpose

### 2. Error Handling

- Always use try/finally or async context managers for cleanup
- Implement retry logic for transient connection failures
- Monitor server health regularly in production

### 3. Resource Management

- Don't keep unused servers connected
- Use connection limits to prevent resource exhaustion
- Implement graceful shutdown procedures

### 4. Monitoring

- Regularly perform health checks
- Log connection status changes
- Monitor server performance metrics

### 5. Testing

- Use mocks for MCP servers in unit tests
- Test connection failure scenarios
- Verify cleanup behavior

## Troubleshooting

### Common Issues

#### "MCP support not available" Error

This error occurs when the `agents.mcp` module is not installed or available.

**Solution**: Ensure the MCP dependencies are installed:
```bash
# Install required dependencies
pip install agents-mcp  # or appropriate package
```

#### Connection Timeouts

Servers may fail to connect due to timeouts.

**Solutions**:
- Increase the `timeout` value in server configuration
- Check that the server command is correct and executable
- Verify environment variables are set correctly
- Check network connectivity

#### Server Command Not Found

The specified command path doesn't exist or isn't executable.

**Solutions**:
- Verify the command path is correct
- Ensure the executable has proper permissions
- Use absolute paths instead of relative paths
- Check that required dependencies are installed

#### Environment Variable Issues

Server fails to start due to missing or invalid environment variables.

**Solutions**:
- Verify all required environment variables are set
- Check variable names for typos
- Ensure variable values are valid for the server
- Use default values where appropriate

### Debugging

Enable debug logging to get more information:

```python
import logging

# Enable debug logging for MCP hub
logging.getLogger('reagentic.mcp.hub').setLevel(logging.DEBUG)

# Or enable for all reagentic modules
logging.getLogger('reagentic').setLevel(logging.DEBUG)
```

### Health Check Interpretation

The health check returns detailed status information:

```python
health = await hub.health_check()

# Overall status: 'healthy', 'degraded', or 'unhealthy'
print(f"Overall: {health['overall_status']}")

# Individual server status
for server_name, server_health in health['server_health'].items():
    status = server_health['status']  # 'healthy', 'unhealthy', 'not_connected'
    error = server_health.get('error')
    
    if status != 'healthy':
        print(f"{server_name}: {status}")
        if error:
            print(f"  Error: {error}")
```

## Migration Guide

### From Direct MCPServerStdio Usage

If you're currently using `MCPServerStdio` directly:

**Before**:
```python
from agents.mcp import MCPServerStdio

server = MCPServerStdio(params={'command': '/path/to/mcp'})
await server.connect()
# Use server...
await server.cleanup()
```

**After**:
```python
from reagentic.mcp import create_hub_from_dict

config = {
    "servers": {
        "my_server": {"command": "/path/to/mcp"}
    }
}

hub = create_hub_from_dict(config)
async with hub:
    server = hub.get_server("my_server")
    # Use server...
```

### Benefits of Migration

- **Centralized Management**: Manage multiple servers from one place
- **Better Error Handling**: Built-in retry logic and health monitoring
- **Resource Management**: Automatic cleanup and connection pooling
- **Configuration Validation**: Type-safe configuration with Pydantic
- **Monitoring**: Built-in health checks and status reporting 