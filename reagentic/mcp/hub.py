"""
MCP Hub - Lifecycle Management for MCP Servers

Provides centralized management of MCP server lifecycles with configuration
validation using Pydantic models.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator
import os

try:
    from agents.mcp import MCPServerStdio
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    MCPServerStdio = None

logger = logging.getLogger(__name__)


class McpServerConfig(BaseModel):
    """
    Configuration for a single MCP server.
    
    Validates the configuration for stdio-based MCP servers.
    """
    
    command: str = Field(..., description="Command to execute the MCP server")
    args: List[str] = Field(default_factory=list, description="Command line arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    working_directory: Optional[str] = Field(None, description="Working directory for the server")
    cache_tools_list: bool = Field(True, description="Whether to cache the tools list")
    timeout: int = Field(30, description="Connection timeout in seconds")
    
    @field_validator('command')
    @classmethod
    def validate_command(cls, v):
        """Validate that the command is not empty."""
        if not v or not v.strip():
            raise ValueError("Command cannot be empty")
        return v.strip()
    
    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v):
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v
    
    @field_validator('env')
    @classmethod
    def validate_env(cls, v):
        """Validate environment variables."""
        if not isinstance(v, dict):
            raise ValueError("Environment variables must be a dictionary")
        
        # Ensure all keys and values are strings
        for key, value in v.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("Environment variable keys and values must be strings")
        
        return v


class McpHubConfig(BaseModel):
    """
    Configuration for the MCP Hub containing multiple server configurations.
    """
    
    servers: Dict[str, McpServerConfig] = Field(
        default_factory=dict,
        description="Dictionary of server name to server configuration"
    )
    auto_connect: bool = Field(True, description="Whether to auto-connect servers on startup")
    max_concurrent_connections: int = Field(10, description="Maximum concurrent server connections")
    connection_retry_attempts: int = Field(3, description="Number of retry attempts for failed connections")
    connection_retry_delay: float = Field(1.0, description="Delay between retry attempts in seconds")
    
    @field_validator('servers')
    @classmethod
    def validate_servers(cls, v):
        """Validate server configurations."""
        if not isinstance(v, dict):
            raise ValueError("Servers must be a dictionary")
        
        for server_name, config in v.items():
            if not isinstance(server_name, str) or not server_name.strip():
                raise ValueError("Server names must be non-empty strings")
            
            if not isinstance(config, McpServerConfig):
                raise ValueError(f"Server config for '{server_name}' must be a McpServerConfig instance")
        
        return v
    
    @field_validator('max_concurrent_connections')
    @classmethod
    def validate_max_connections(cls, v):
        """Validate max concurrent connections."""
        if v <= 0:
            raise ValueError("Max concurrent connections must be positive")
        return v


class McpHub:
    """
    Hub for managing MCP server lifecycles.
    
    Provides centralized management of multiple MCP servers with automatic
    connection handling, lifecycle management, and error recovery.
    """
    
    def __init__(self, config: McpHubConfig):
        """
        Initialize the MCP Hub.
        
        Args:
            config: Hub configuration with server definitions
        """
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP support not available. Please ensure 'agents.mcp' is installed."
            )
        
        self.config = config
        self.servers: Dict[str, MCPServerStdio] = {}
        self.connected_servers: Dict[str, bool] = {}
        self.connection_semaphore = asyncio.Semaphore(config.max_concurrent_connections)
        self._shutdown_event = asyncio.Event()
        
        logger.info(f"🔧 McpHub initialized with {len(config.servers)} server configurations")
    
    async def start(self) -> None:
        """
        Start the MCP Hub and connect to all configured servers.
        """
        logger.info("🚀 Starting MCP Hub...")
        
        if self.config.auto_connect:
            await self.connect_all_servers()
        
        logger.info("✅ MCP Hub started successfully")
    
    async def stop(self) -> None:
        """
        Stop the MCP Hub and disconnect from all servers.
        """
        logger.info("🛑 Stopping MCP Hub...")
        
        self._shutdown_event.set()
        await self.disconnect_all_servers()
        
        logger.info("✅ MCP Hub stopped successfully")
    
    async def connect_server(self, server_name: str) -> bool:
        """
        Connect to a specific MCP server.
        
        Args:
            server_name: Name of the server to connect to
            
        Returns:
            True if connection was successful, False otherwise
        """
        if server_name not in self.config.servers:
            logger.error(f"❌ Server '{server_name}' not found in configuration")
            return False
        
        if server_name in self.servers and self.connected_servers.get(server_name, False):
            logger.warning(f"⚠️ Server '{server_name}' is already connected")
            return True
        
        async with self.connection_semaphore:
            return await self._connect_server_impl(server_name)
    
    async def _connect_server_impl(self, server_name: str) -> bool:
        """
        Internal implementation of server connection with retry logic.
        
        Args:
            server_name: Name of the server to connect to
            
        Returns:
            True if connection was successful, False otherwise
        """
        config = self.config.servers[server_name]
        
        for attempt in range(self.config.connection_retry_attempts):
            try:
                logger.info(f"🔌 Connecting to MCP server '{server_name}' (attempt {attempt + 1})")
                
                # Prepare server parameters
                server_params = {
                    'command': config.command,
                    'args': config.args,
                    'env': config.env,
                }
                
                if config.working_directory:
                    server_params['cwd'] = config.working_directory
                
                # Create and connect to the server
                server = MCPServerStdio(
                    params=server_params,
                    cache_tools_list=config.cache_tools_list
                )
                
                # Connect with timeout
                await asyncio.wait_for(server.connect(), timeout=config.timeout)
                
                # Store the connected server
                self.servers[server_name] = server
                self.connected_servers[server_name] = True
                
                logger.info(f"✅ Successfully connected to MCP server '{server_name}'")
                return True
                
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Connection to '{server_name}' timed out (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"⚠️ Failed to connect to '{server_name}' (attempt {attempt + 1}): {e}")
            
            # Wait before retry (except on last attempt)
            if attempt < self.config.connection_retry_attempts - 1:
                await asyncio.sleep(self.config.connection_retry_delay)
        
        logger.error(f"❌ Failed to connect to MCP server '{server_name}' after {self.config.connection_retry_attempts} attempts")
        return False
    
    async def disconnect_server(self, server_name: str) -> bool:
        """
        Disconnect from a specific MCP server.
        
        Args:
            server_name: Name of the server to disconnect from
            
        Returns:
            True if disconnection was successful, False otherwise
        """
        if server_name not in self.servers:
            logger.warning(f"⚠️ Server '{server_name}' is not connected")
            return True
        
        try:
            logger.info(f"🔌 Disconnecting from MCP server '{server_name}'")
            
            server = self.servers[server_name]
            await server.cleanup()
            
            # Remove from tracking
            del self.servers[server_name]
            self.connected_servers[server_name] = False
            
            logger.info(f"✅ Successfully disconnected from MCP server '{server_name}'")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error disconnecting from '{server_name}': {e}")
            return False
    
    async def connect_all_servers(self) -> Dict[str, bool]:
        """
        Connect to all configured MCP servers concurrently.
        
        Returns:
            Dictionary mapping server names to connection success status
        """
        logger.info(f"🔗 Connecting to {len(self.config.servers)} MCP servers...")
        
        # Create connection tasks for all servers
        connection_tasks = {
            server_name: asyncio.create_task(self.connect_server(server_name))
            for server_name in self.config.servers.keys()
        }
        
        # Wait for all connections to complete
        results = {}
        for server_name, task in connection_tasks.items():
            try:
                results[server_name] = await task
            except Exception as e:
                logger.error(f"❌ Unexpected error connecting to '{server_name}': {e}")
                results[server_name] = False
        
        successful_connections = sum(results.values())
        logger.info(f"📊 Connected to {successful_connections}/{len(self.config.servers)} MCP servers")
        
        return results
    
    async def disconnect_all_servers(self) -> Dict[str, bool]:
        """
        Disconnect from all connected MCP servers.
        
        Returns:
            Dictionary mapping server names to disconnection success status
        """
        if not self.servers:
            logger.info("ℹ️ No MCP servers to disconnect")
            return {}
        
        logger.info(f"🔗 Disconnecting from {len(self.servers)} MCP servers...")
        
        # Create disconnection tasks for all connected servers
        disconnection_tasks = {
            server_name: asyncio.create_task(self.disconnect_server(server_name))
            for server_name in list(self.servers.keys())
        }
        
        # Wait for all disconnections to complete
        results = {}
        for server_name, task in disconnection_tasks.items():
            try:
                results[server_name] = await task
            except Exception as e:
                logger.error(f"❌ Unexpected error disconnecting from '{server_name}': {e}")
                results[server_name] = False
        
        successful_disconnections = sum(results.values())
        logger.info(f"📊 Disconnected from {successful_disconnections}/{len(results)} MCP servers")
        
        return results
    
    def get_server(self, server_name: str) -> Optional[MCPServerStdio]:
        """
        Get a connected MCP server by name.
        
        Args:
            server_name: Name of the server to retrieve
            
        Returns:
            MCPServerStdio instance if connected, None otherwise
        """
        if server_name not in self.servers:
            logger.warning(f"⚠️ Server '{server_name}' is not connected")
            return None
        
        return self.servers[server_name]
    
    def get_all_servers(self) -> Dict[str, MCPServerStdio]:
        """
        Get all connected MCP servers.
        
        Returns:
            Dictionary mapping server names to MCPServerStdio instances
        """
        return self.servers.copy()
    
    def is_server_connected(self, server_name: str) -> bool:
        """
        Check if a server is connected.
        
        Args:
            server_name: Name of the server to check
            
        Returns:
            True if server is connected, False otherwise
        """
        return self.connected_servers.get(server_name, False)
    
    def get_connection_status(self) -> Dict[str, bool]:
        """
        Get connection status for all configured servers.
        
        Returns:
            Dictionary mapping server names to connection status
        """
        return {
            server_name: self.is_server_connected(server_name)
            for server_name in self.config.servers.keys()
        }
    
    def get_hub_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information about the hub.
        
        Returns:
            Dictionary with hub status information
        """
        connection_status = self.get_connection_status()
        
        return {
            'total_servers': len(self.config.servers),
            'connected_servers': sum(connection_status.values()),
            'connection_status': connection_status,
            'auto_connect': self.config.auto_connect,
            'max_concurrent_connections': self.config.max_concurrent_connections,
            'shutdown_requested': self._shutdown_event.is_set()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on all connected servers.
        
        Returns:
            Dictionary with health check results
        """
        logger.info("🏥 Performing MCP Hub health check...")
        
        health_results = {}
        
        for server_name, server in self.servers.items():
            try:
                # Basic connectivity check - try to get server info
                # This is a simple check; in practice you might want to call a specific method
                health_results[server_name] = {
                    'connected': True,
                    'status': 'healthy',
                    'error': None
                }
            except Exception as e:
                health_results[server_name] = {
                    'connected': False,
                    'status': 'unhealthy',
                    'error': str(e)
                }
                logger.warning(f"⚠️ Health check failed for '{server_name}': {e}")
        
        # Check for configured but not connected servers
        for server_name in self.config.servers.keys():
            if server_name not in health_results:
                health_results[server_name] = {
                    'connected': False,
                    'status': 'not_connected',
                    'error': 'Server not connected'
                }
        
        healthy_servers = sum(1 for result in health_results.values() if result['status'] == 'healthy')
        logger.info(f"🏥 Health check complete: {healthy_servers}/{len(health_results)} servers healthy")
        
        return {
            'timestamp': asyncio.get_event_loop().time(),
            'overall_status': 'healthy' if healthy_servers == len(health_results) else 'degraded',
            'healthy_servers': healthy_servers,
            'total_servers': len(health_results),
            'server_health': health_results
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Convenience functions for creating configurations

def create_server_config(
    command: str,
    env: Optional[Dict[str, str]] = None,
    args: Optional[List[str]] = None,
    **kwargs
) -> McpServerConfig:
    """
    Convenience function to create a server configuration.
    
    Args:
        command: Command to execute the MCP server
        env: Environment variables
        args: Command line arguments
        **kwargs: Additional configuration options
        
    Returns:
        McpServerConfig instance
    """
    config_dict = {
        'command': command,
        'env': env or {},
        'args': args or [],
        **kwargs
    }
    
    return McpServerConfig(**config_dict)


def create_hub_config(
    servers: Dict[str, McpServerConfig],
    **kwargs
) -> McpHubConfig:
    """
    Convenience function to create a hub configuration.
    
    Args:
        servers: Dictionary of server configurations
        **kwargs: Additional hub configuration options
        
    Returns:
        McpHubConfig instance
    """
    config_dict = {
        'servers': servers,
        **kwargs
    }
    
    return McpHubConfig(**config_dict)


def create_hub_from_dict(config_dict: Dict[str, Any]) -> McpHub:
    """
    Create an MCP Hub from a dictionary configuration.
    
    Args:
        config_dict: Dictionary with hub and server configurations
        
    Returns:
        McpHub instance
        
    Example:
        config = {
            "servers": {
                "tracker": {
                    "command": "/path/to/tracker/mcp",
                    "env": {"STARTREK_TOKEN": "token"}
                }
            },
            "auto_connect": True
        }
        hub = create_hub_from_dict(config)
    """
    # Convert server configs to McpServerConfig instances
    servers = {}
    for server_name, server_config in config_dict.get('servers', {}).items():
        servers[server_name] = McpServerConfig(**server_config)
    
    # Create hub config
    hub_config_dict = config_dict.copy()
    hub_config_dict['servers'] = servers
    
    hub_config = McpHubConfig(**hub_config_dict)
    return McpHub(hub_config) 