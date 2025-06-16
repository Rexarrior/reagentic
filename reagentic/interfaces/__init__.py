"""
Team Support Interfaces

Abstract interfaces for external integrations to keep the core logic
independent of specific frameworks and services.
"""

from .messaging import MessagingInterface, Message, MessageType
from .telegram import TelegramMessagingInterface

__all__ = [
    'MessagingInterface',
    'Message', 
    'MessageType',
    'TelegramMessagingInterface'
] 