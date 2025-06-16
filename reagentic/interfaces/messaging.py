"""
Abstract Messaging Interface

Defines the contract for messaging integrations without coupling to specific platforms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Callable, Awaitable
import logging

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages that can be handled."""
    TEXT = "text"
    COMMAND = "command"
    PHOTO = "photo"
    DOCUMENT = "document"
    VOICE = "voice"
    VIDEO = "video"
    STICKER = "sticker"
    LOCATION = "location"
    CONTACT = "contact"
    UNKNOWN = "unknown"


@dataclass
class Message:
    """
    Universal message representation.
    
    This abstraction allows the core logic to work with messages
    regardless of the underlying messaging platform.
    """
    id: str
    text: Optional[str]
    message_type: MessageType
    user_id: str
    user_name: Optional[str]
    chat_id: str
    chat_type: str  # 'private', 'group', 'supergroup', 'channel'
    timestamp: str
    metadata: Dict[str, Any]  # Platform-specific data
    
    def __str__(self) -> str:
        return f"Message({self.message_type.value}, user={self.user_name}, text='{self.text[:50]}...')"


class MessagingInterface(ABC):
    """
    Abstract interface for messaging platforms.
    
    This interface defines the contract that any messaging platform
    implementation must follow to integrate with the team support system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the messaging interface.
        
        Args:
            config: Platform-specific configuration
        """
        self.config = config
        self.message_handler: Optional[Callable[[Message], Awaitable[None]]] = None
        self.is_running = False
        
    @abstractmethod
    async def start(self) -> None:
        """Start the messaging service."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the messaging service."""
        pass
    
    @abstractmethod
    async def send_message(self, chat_id: str, text: str, **kwargs) -> bool:
        """
        Send a text message.
        
        Args:
            chat_id: Target chat/user ID
            text: Message text
            **kwargs: Platform-specific options
            
        Returns:
            True if message was sent successfully
        """
        pass
    
    @abstractmethod
    async def send_photo(self, chat_id: str, photo_path: str, caption: Optional[str] = None) -> bool:
        """
        Send a photo message.
        
        Args:
            chat_id: Target chat/user ID
            photo_path: Path to photo file or URL
            caption: Optional photo caption
            
        Returns:
            True if photo was sent successfully
        """
        pass
    
    @abstractmethod
    async def send_document(self, chat_id: str, document_path: str, caption: Optional[str] = None) -> bool:
        """
        Send a document.
        
        Args:
            chat_id: Target chat/user ID
            document_path: Path to document file
            caption: Optional document caption
            
        Returns:
            True if document was sent successfully
        """
        pass
    
    def set_message_handler(self, handler: Callable[[Message], Awaitable[None]]) -> None:
        """
        Set the message handler function.
        
        Args:
            handler: Async function to handle incoming messages
        """
        self.message_handler = handler
        logger.info("Message handler set")
    
    async def handle_message(self, message: Message) -> None:
        """
        Handle an incoming message by calling the registered handler.
        
        Args:
            message: The incoming message
        """
        if self.message_handler:
            try:
                await self.message_handler(message)
            except Exception as e:
                logger.error(f"Error handling message {message.id}: {e}")
        else:
            logger.warning(f"No message handler set, ignoring message: {message}")
    
    @abstractmethod
    def get_platform_name(self) -> str:
        """Get the name of the messaging platform."""
        pass
    
    @abstractmethod
    def get_bot_info(self) -> Dict[str, Any]:
        """Get information about the bot/account."""
        pass 