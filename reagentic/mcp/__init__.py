"""
MCP (Model Context Protocol) Integration for Reagentic Framework

This package provides MCP server lifecycle management and integration
for the reagentic framework.
"""

from .hub import McpHub, McpServerConfig, McpHubConfig, create_hub_from_dict

__all__ = [
    'McpHub',
    'McpServerConfig', 
    'McpHubConfig',
    'create_hub_from_dict'
] 