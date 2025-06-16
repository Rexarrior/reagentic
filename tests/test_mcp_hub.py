"""
Tests for the McpHub class and related functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any
from pydantic import ValidationError

from reagentic.mcp import McpHub, McpServerConfig, McpHubConfig, create_hub_from_dict


class TestMcpServerConfig:
    """Test cases for McpServerConfig validation."""
    
    def test_valid_config(self):
        """Test creating a valid server configuration."""
        config = McpServerConfig(
            command="/path/to/mcp",
            env={"TOKEN": "value"},
            args=["--verbose"],
            timeout=30
        )
        
        assert config.command == "/path/to/mcp"
        assert config.env == {"TOKEN": "value"}
        assert config.args == ["--verbose"]
        assert config.timeout == 30
        assert config.cache_tools_list is True
    
    def test_empty_command_validation(self):
        """Test that empty command raises validation error."""
        with pytest.raises(ValueError, match="Command cannot be empty"):
            McpServerConfig(command="")
    
    def test_whitespace_command_validation(self):
        """Test that whitespace-only command raises validation error."""
        with pytest.raises(ValueError, match="Command cannot be empty"):
            McpServerConfig(command="   ")
    
    def test_negative_timeout_validation(self):
        """Test that negative timeout raises validation error."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            McpServerConfig(command="/path/to/mcp", timeout=-1)
    
    def test_zero_timeout_validation(self):
        """Test that zero timeout raises validation error."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            McpServerConfig(command="/path/to/mcp", timeout=0)
    
    def test_invalid_env_validation(self):
        """Test that invalid environment variables raise validation error."""
        with pytest.raises(ValidationError, match="Input should be a valid string"):
            McpServerConfig(command="/path/to/mcp", env={"key": 123})


class TestMcpHubConfig:
    """Test cases for McpHubConfig validation."""
    
    def test_valid_hub_config(self):
        """Test creating a valid hub configuration."""
        server_config = McpServerConfig(command="/path/to/mcp")
        hub_config = McpHubConfig(
            servers={"test_server": server_config},
            auto_connect=False,
            max_concurrent_connections=5
        )
        
        assert len(hub_config.servers) == 1
        assert "test_server" in hub_config.servers
        assert hub_config.auto_connect is False
        assert hub_config.max_concurrent_connections == 5
    
    def test_empty_server_name_validation(self):
        """Test that empty server names raise validation error."""
        server_config = McpServerConfig(command="/path/to/mcp")
        with pytest.raises(ValueError, match="Server names must be non-empty strings"):
            McpHubConfig(servers={"": server_config})
    
    def test_negative_max_connections_validation(self):
        """Test that negative max connections raises validation error."""
        with pytest.raises(ValueError, match="Max concurrent connections must be positive"):
            McpHubConfig(max_concurrent_connections=-1)


class TestMcpHub:
    """Test cases for McpHub functionality."""
    
    @pytest.fixture
    def mock_mcp_server(self):
        """Create a mock MCP server."""
        server = Mock()
        server.connect = AsyncMock()
        server.cleanup = AsyncMock()
        return server
    
    @pytest.fixture
    def hub_config(self):
        """Create a test hub configuration."""
        server_config = McpServerConfig(
            command="/path/to/test/mcp",
            env={"TEST_VAR": "test_value"}
        )
        return McpHubConfig(
            servers={"test_server": server_config},
            auto_connect=False  # Don't auto-connect in tests
        )
    
    @patch('reagentic.mcp.hub.MCP_AVAILABLE', True)
    @patch('reagentic.mcp.hub.MCPServerStdio')
    def test_hub_initialization(self, mock_mcp_class, hub_config):
        """Test hub initialization."""
        hub = McpHub(hub_config)
        
        assert hub.config == hub_config
        assert len(hub.servers) == 0
        assert len(hub.connected_servers) == 0
    
    @patch('reagentic.mcp.hub.MCP_AVAILABLE', False)
    def test_hub_initialization_without_mcp(self, hub_config):
        """Test that hub raises error when MCP is not available."""
        with pytest.raises(ImportError, match="MCP support not available"):
            McpHub(hub_config)
    
    @pytest.mark.asyncio
    @patch('reagentic.mcp.hub.MCP_AVAILABLE', True)
    @patch('reagentic.mcp.hub.MCPServerStdio')
    async def test_start_without_auto_connect(self, mock_mcp_class, hub_config):
        """Test starting hub without auto-connect."""
        hub = McpHub(hub_config)
        
        await hub.start()
        
        # Should not have connected to any servers
        assert len(hub.servers) == 0
        assert len(hub.connected_servers) == 0
    
    @pytest.mark.asyncio
    @patch('reagentic.mcp.hub.MCP_AVAILABLE', True)
    @patch('reagentic.mcp.hub.MCPServerStdio')
    async def test_connect_server_success(self, mock_mcp_class, hub_config, mock_mcp_server):
        """Test successful server connection."""
        mock_mcp_class.return_value = mock_mcp_server
        hub = McpHub(hub_config)
        
        result = await hub.connect_server("test_server")
        
        assert result is True
        assert "test_server" in hub.servers
        assert hub.connected_servers["test_server"] is True
        mock_mcp_server.connect.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('reagentic.mcp.hub.MCP_AVAILABLE', True)
    @patch('reagentic.mcp.hub.MCPServerStdio')
    async def test_connect_nonexistent_server(self, mock_mcp_class, hub_config):
        """Test connecting to a server that doesn't exist in config."""
        hub = McpHub(hub_config)
        
        result = await hub.connect_server("nonexistent_server")
        
        assert result is False
        assert len(hub.servers) == 0
    
    @pytest.mark.asyncio
    @patch('reagentic.mcp.hub.MCP_AVAILABLE', True)
    @patch('reagentic.mcp.hub.MCPServerStdio')
    async def test_disconnect_server_success(self, mock_mcp_class, hub_config, mock_mcp_server):
        """Test successful server disconnection."""
        mock_mcp_class.return_value = mock_mcp_server
        hub = McpHub(hub_config)
        
        # First connect
        await hub.connect_server("test_server")
        assert "test_server" in hub.servers
        
        # Then disconnect
        result = await hub.disconnect_server("test_server")
        
        assert result is True
        assert "test_server" not in hub.servers
        assert hub.connected_servers["test_server"] is False
        mock_mcp_server.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('reagentic.mcp.hub.MCP_AVAILABLE', True)
    @patch('reagentic.mcp.hub.MCPServerStdio')
    async def test_get_server(self, mock_mcp_class, hub_config, mock_mcp_server):
        """Test getting a connected server."""
        mock_mcp_class.return_value = mock_mcp_server
        hub = McpHub(hub_config)
        
        # Server not connected
        server = hub.get_server("test_server")
        assert server is None
        
        # Connect server
        await hub.connect_server("test_server")
        
        # Get connected server
        server = hub.get_server("test_server")
        assert server == mock_mcp_server
    
    @pytest.mark.asyncio
    @patch('reagentic.mcp.hub.MCP_AVAILABLE', True)
    @patch('reagentic.mcp.hub.MCPServerStdio')
    async def test_connection_status(self, mock_mcp_class, hub_config, mock_mcp_server):
        """Test getting connection status."""
        mock_mcp_class.return_value = mock_mcp_server
        hub = McpHub(hub_config)
        
        # Initially not connected
        status = hub.get_connection_status()
        assert status == {"test_server": False}
        
        # After connecting
        await hub.connect_server("test_server")
        status = hub.get_connection_status()
        assert status == {"test_server": True}
    
    @pytest.mark.asyncio
    @patch('reagentic.mcp.hub.MCP_AVAILABLE', True)
    @patch('reagentic.mcp.hub.MCPServerStdio')
    async def test_hub_status(self, mock_mcp_class, hub_config, mock_mcp_server):
        """Test getting hub status."""
        mock_mcp_class.return_value = mock_mcp_server
        hub = McpHub(hub_config)
        
        status = hub.get_hub_status()
        
        assert status['total_servers'] == 1
        assert status['connected_servers'] == 0
        assert status['auto_connect'] is False
        assert 'connection_status' in status
    
    @pytest.mark.asyncio
    @patch('reagentic.mcp.hub.MCP_AVAILABLE', True)
    @patch('reagentic.mcp.hub.MCPServerStdio')
    async def test_health_check(self, mock_mcp_class, hub_config, mock_mcp_server):
        """Test health check functionality."""
        mock_mcp_class.return_value = mock_mcp_server
        hub = McpHub(hub_config)
        
        # Connect server
        await hub.connect_server("test_server")
        
        # Perform health check
        health = await hub.health_check()
        
        assert 'timestamp' in health
        assert 'overall_status' in health
        assert 'server_health' in health
        assert 'test_server' in health['server_health']
    
    @pytest.mark.asyncio
    @patch('reagentic.mcp.hub.MCP_AVAILABLE', True)
    @patch('reagentic.mcp.hub.MCPServerStdio')
    async def test_context_manager(self, mock_mcp_class, hub_config, mock_mcp_server):
        """Test using hub as async context manager."""
        mock_mcp_class.return_value = mock_mcp_server
        hub = McpHub(hub_config)
        
        async with hub:
            # Hub should be started
            pass
        
        # Hub should be stopped after context exit
        # This is a basic test - in practice we'd verify cleanup was called


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_create_hub_from_dict(self):
        """Test creating hub from dictionary configuration."""
        config_dict = {
            "servers": {
                "test_server": {
                    "command": "/path/to/mcp",
                    "env": {"TOKEN": "value"}
                }
            },
            "auto_connect": False
        }
        
        hub = create_hub_from_dict(config_dict)
        
        assert isinstance(hub, McpHub)
        assert len(hub.config.servers) == 1
        assert "test_server" in hub.config.servers
        assert hub.config.auto_connect is False
    
    def test_create_hub_from_dict_with_invalid_server_config(self):
        """Test that invalid server config in dict raises error."""
        config_dict = {
            "servers": {
                "test_server": {
                    "command": "",  # Invalid empty command
                    "env": {"TOKEN": "value"}
                }
            }
        }
        
        with pytest.raises(ValueError, match="Command cannot be empty"):
            create_hub_from_dict(config_dict) 