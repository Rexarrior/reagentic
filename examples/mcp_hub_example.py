"""
Example demonstrating the usage of McpHub for managing MCP server lifecycles.

This example shows how to:
1. Create MCP server configurations using Pydantic models
2. Initialize and manage multiple MCP servers
3. Use the hub as an async context manager
4. Handle server connections and health checks
"""

import asyncio
import logging
import os
from typing import Dict, Any

from reagentic.mcp import McpHub, McpServerConfig, McpHubConfig, create_hub_from_dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """
    Basic example of using McpHub with manual configuration.
    """
    logger.info("🚀 Starting basic McpHub example...")
    
    # Create server configurations
    tracker_config = McpServerConfig(
        command="/path/to/tracker/mcp",
        env={"STARTREK_TOKEN": "your_token_here"},
        timeout=30
    )
    
    knowledge_base_config = McpServerConfig(
        command="/path/to/knowledge_base/mcp",
        env={"KB_PATH": "/path/to/knowledge"},
        args=["--verbose"],
        timeout=20
    )
    
    # Create hub configuration
    hub_config = McpHubConfig(
        servers={
            "tracker": tracker_config,
            "knowledge_base": knowledge_base_config
        },
        auto_connect=True,
        max_concurrent_connections=5,
        connection_retry_attempts=2
    )
    
    # Create and use the hub
    hub = McpHub(hub_config)
    
    try:
        # Start the hub (will auto-connect to all servers)
        await hub.start()
        
        # Check connection status
        status = hub.get_connection_status()
        logger.info(f"📊 Connection status: {status}")
        
        # Get a specific server
        tracker_server = hub.get_server("tracker")
        if tracker_server:
            logger.info("✅ Tracker server is available")
        
        # Perform health check
        health = await hub.health_check()
        logger.info(f"🏥 Health check results: {health}")
        
        # Get hub status
        hub_status = hub.get_hub_status()
        logger.info(f"📈 Hub status: {hub_status}")
        
    finally:
        # Stop the hub (will disconnect all servers)
        await hub.stop()


async def example_context_manager():
    """
    Example using McpHub as an async context manager.
    """
    logger.info("🚀 Starting context manager example...")
    
    # Create configuration from dictionary
    config_dict = {
        "servers": {
            "tracker": {
                "command": "/path/to/tracker/mcp",
                "env": {"STARTREK_TOKEN": "your_token_here"}
            },
            "search": {
                "command": "/path/to/search/mcp",
                "env": {"SEARCH_API_KEY": "your_search_key"},
                "args": ["--index", "main"]
            }
        },
        "auto_connect": True,
        "connection_retry_attempts": 3,
        "connection_retry_delay": 2.0
    }
    
    # Use hub as context manager
    hub = create_hub_from_dict(config_dict)
    
    async with hub:
        logger.info("🔗 Hub started and servers connected")
        
        # Work with connected servers
        all_servers = hub.get_all_servers()
        logger.info(f"📋 Available servers: {list(all_servers.keys())}")
        
        # Check individual server connections
        for server_name in ["tracker", "search"]:
            if hub.is_server_connected(server_name):
                logger.info(f"✅ {server_name} is connected")
                server = hub.get_server(server_name)
                # Use the server for MCP operations here
            else:
                logger.warning(f"⚠️ {server_name} is not connected")
    
    logger.info("🛑 Hub stopped and all servers disconnected")


async def example_manual_server_management():
    """
    Example of manually managing individual server connections.
    """
    logger.info("🚀 Starting manual server management example...")
    
    # Create hub with auto_connect=False
    config_dict = {
        "servers": {
            "server1": {
                "command": "/path/to/server1/mcp",
                "env": {"API_KEY": "key1"}
            },
            "server2": {
                "command": "/path/to/server2/mcp", 
                "env": {"API_KEY": "key2"}
            },
            "server3": {
                "command": "/path/to/server3/mcp",
                "env": {"API_KEY": "key3"}
            }
        },
        "auto_connect": False  # Don't auto-connect
    }
    
    hub = create_hub_from_dict(config_dict)
    
    try:
        await hub.start()  # Start hub without connecting servers
        
        # Connect servers individually
        logger.info("🔌 Connecting to server1...")
        success1 = await hub.connect_server("server1")
        logger.info(f"Server1 connection: {'✅ Success' if success1 else '❌ Failed'}")
        
        logger.info("🔌 Connecting to server2...")
        success2 = await hub.connect_server("server2")
        logger.info(f"Server2 connection: {'✅ Success' if success2 else '❌ Failed'}")
        
        # Try to connect to all remaining servers
        logger.info("🔗 Connecting to all remaining servers...")
        results = await hub.connect_all_servers()
        for server_name, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            logger.info(f"  {server_name}: {status}")
        
        # Disconnect a specific server
        logger.info("🔌 Disconnecting server1...")
        disconnect_success = await hub.disconnect_server("server1")
        logger.info(f"Server1 disconnection: {'✅ Success' if disconnect_success else '❌ Failed'}")
        
        # Show final status
        final_status = hub.get_connection_status()
        logger.info(f"📊 Final connection status: {final_status}")
        
    finally:
        await hub.stop()


async def example_error_handling():
    """
    Example demonstrating error handling and recovery.
    """
    logger.info("🚀 Starting error handling example...")
    
    # Create configuration with invalid commands to demonstrate error handling
    config_dict = {
        "servers": {
            "valid_server": {
                "command": "echo",  # Simple command that should work
                "args": ["Hello from MCP server"],
                "env": {}
            },
            "invalid_server": {
                "command": "/nonexistent/path/to/mcp",  # This will fail
                "env": {"SOME_VAR": "value"}
            }
        },
        "auto_connect": True,
        "connection_retry_attempts": 2,
        "connection_retry_delay": 1.0
    }
    
    hub = create_hub_from_dict(config_dict)
    
    try:
        await hub.start()
        
        # Check which servers connected successfully
        connection_status = hub.get_connection_status()
        logger.info(f"📊 Connection results: {connection_status}")
        
        # Perform health check to see detailed status
        health_results = await hub.health_check()
        logger.info("🏥 Health check results:")
        for server_name, health in health_results['server_health'].items():
            status_emoji = "✅" if health['status'] == 'healthy' else "❌"
            logger.info(f"  {status_emoji} {server_name}: {health['status']}")
            if health['error']:
                logger.info(f"    Error: {health['error']}")
        
        # Try to reconnect failed servers
        for server_name, connected in connection_status.items():
            if not connected:
                logger.info(f"🔄 Attempting to reconnect {server_name}...")
                success = await hub.connect_server(server_name)
                logger.info(f"Reconnection result: {'✅ Success' if success else '❌ Failed'}")
        
    finally:
        await hub.stop()


async def example_with_environment_variables():
    """
    Example using environment variables for configuration.
    """
    logger.info("🚀 Starting environment variables example...")
    
    # Set some example environment variables
    os.environ["TRACKER_TOKEN"] = "example_tracker_token"
    os.environ["SEARCH_API_KEY"] = "example_search_key"
    
    config_dict = {
        "servers": {
            "tracker": {
                "command": "/path/to/tracker/mcp",
                "env": {
                    "STARTREK_TOKEN": os.environ.get("TRACKER_TOKEN", ""),
                    "DEBUG": "true"
                }
            },
            "search": {
                "command": "/path/to/search/mcp",
                "env": {
                    "SEARCH_API_KEY": os.environ.get("SEARCH_API_KEY", ""),
                    "SEARCH_INDEX": "main"
                },
                "working_directory": "/tmp/search_workspace"
            }
        },
        "auto_connect": True
    }
    
    hub = create_hub_from_dict(config_dict)
    
    async with hub:
        logger.info("🔗 Hub started with environment-based configuration")
        
        # Show server configurations (without sensitive data)
        for server_name, server_config in hub.config.servers.items():
            logger.info(f"📋 {server_name}:")
            logger.info(f"  Command: {server_config.command}")
            logger.info(f"  Args: {server_config.args}")
            logger.info(f"  Working dir: {server_config.working_directory}")
            logger.info(f"  Environment vars: {list(server_config.env.keys())}")


async def main():
    """
    Run all examples.
    """
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Context Manager", example_context_manager),
        ("Manual Server Management", example_manual_server_management),
        ("Error Handling", example_error_handling),
        ("Environment Variables", example_with_environment_variables),
    ]
    
    for name, example_func in examples:
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Running example: {name}")
        logger.info(f"{'='*60}")
        
        try:
            await example_func()
            logger.info(f"✅ {name} completed successfully")
        except Exception as e:
            logger.error(f"❌ {name} failed: {e}")
        
        logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main()) 