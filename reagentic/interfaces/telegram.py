"""
Telegram Messaging Interface Implementation

Implements the MessagingInterface using aiogram for Telegram integration.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from aiogram import Bot, Dispatcher, types
    from aiogram.filters import Command
    from aiogram.types import Message as TelegramMessage
    AIOGRAM_AVAILABLE = True
except ImportError:
    AIOGRAM_AVAILABLE = False
    Bot = None
    Dispatcher = None
    types = None
    Command = None
    TelegramMessage = None

from .messaging import MessagingInterface, Message, MessageType

logger = logging.getLogger(__name__)


class TelegramMessagingInterface(MessagingInterface):
    """
    Telegram implementation of the messaging interface using aiogram.
    
    Requires aiogram to be installed: pip install aiogram
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Telegram bot.
        
        Args:
            config: Configuration dict with 'token' key
        """
        super().__init__(config)
        
        if not AIOGRAM_AVAILABLE:
            raise ImportError(
                "aiogram is required for Telegram integration. "
                "Install it with: pip install aiogram"
            )
        
        self.token = config.get('token')
        if not self.token:
            raise ValueError("Telegram bot token is required in config['token']")
        
        self.bot: Optional[Bot] = None
        self.dispatcher: Optional[Dispatcher] = None
        self._polling_task: Optional[asyncio.Task] = None
        
    async def start(self) -> None:
        """Start the Telegram bot."""
        if self.is_running:
            logger.warning("Telegram bot is already running")
            return
        
        try:
            # Initialize bot and dispatcher
            self.bot = Bot(token=self.token)
            self.dispatcher = Dispatcher()
            
            # Register message handlers
            self.dispatcher.message.register(self._handle_telegram_message)
            
            # Test bot connection
            bot_info = await self.bot.get_me()
            logger.info(f"🤖 Telegram bot started: @{bot_info.username}")
            
            # Start polling in background
            self._polling_task = asyncio.create_task(self._start_polling())
            self.is_running = True
            
        except Exception as e:
            logger.error(f"❌ Failed to start Telegram bot: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if not self.is_running:
            return
        
        logger.info("🛑 Stopping Telegram bot...")
        
        # Stop polling
        if self._polling_task and not self._polling_task.done():
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
        
        # Close bot session
        if self.bot:
            await self.bot.session.close()
        
        self.is_running = False
        logger.info("✅ Telegram bot stopped")
    
    async def _start_polling(self) -> None:
        """Start polling for messages."""
        try:
            await self.dispatcher.start_polling(self.bot)
        except asyncio.CancelledError:
            logger.info("📡 Telegram polling cancelled")
        except Exception as e:
            logger.error(f"❌ Error in Telegram polling: {e}")
    
    async def _handle_telegram_message(self, telegram_message: TelegramMessage) -> None:
        """
        Handle incoming Telegram message and convert to universal format.
        
        Args:
            telegram_message: Raw Telegram message from aiogram
        """
        try:
            # Convert Telegram message to universal Message format
            message = self._convert_telegram_message(telegram_message)
            
            # Pass to the registered message handler
            await self.handle_message(message)
            
        except Exception as e:
            logger.error(f"❌ Error handling Telegram message: {e}")
    
    def _convert_telegram_message(self, tg_msg: TelegramMessage) -> Message:
        """
        Convert Telegram message to universal Message format.
        
        Args:
            tg_msg: Telegram message from aiogram
            
        Returns:
            Universal Message object
        """
        # Determine message type
        message_type = MessageType.TEXT
        text = tg_msg.text
        
        if tg_msg.text and tg_msg.text.startswith('/'):
            message_type = MessageType.COMMAND
        elif tg_msg.photo:
            message_type = MessageType.PHOTO
            text = tg_msg.caption
        elif tg_msg.document:
            message_type = MessageType.DOCUMENT
            text = tg_msg.caption
        elif tg_msg.voice:
            message_type = MessageType.VOICE
            text = tg_msg.caption
        elif tg_msg.video:
            message_type = MessageType.VIDEO
            text = tg_msg.caption
        elif tg_msg.sticker:
            message_type = MessageType.STICKER
            text = tg_msg.sticker.emoji if tg_msg.sticker else None
        elif tg_msg.location:
            message_type = MessageType.LOCATION
            text = f"Location: {tg_msg.location.latitude}, {tg_msg.location.longitude}"
        elif tg_msg.contact:
            message_type = MessageType.CONTACT
            text = f"Contact: {tg_msg.contact.first_name} {tg_msg.contact.phone_number}"
        elif not text:
            message_type = MessageType.UNKNOWN
            text = "[Non-text message]"
        
        # Get user info
        user = tg_msg.from_user
        user_name = None
        if user:
            user_name = user.username or f"{user.first_name or ''} {user.last_name or ''}".strip()
        
        # Get chat info
        chat = tg_msg.chat
        chat_type = chat.type if chat else 'unknown'
        
        # Create metadata with Telegram-specific data
        metadata = {
            'platform': 'telegram',
            'message_id': tg_msg.message_id,
            'chat_title': getattr(chat, 'title', None),
            'user_first_name': getattr(user, 'first_name', None),
            'user_last_name': getattr(user, 'last_name', None),
            'user_username': getattr(user, 'username', None),
            'user_language_code': getattr(user, 'language_code', None),
            'is_bot': getattr(user, 'is_bot', False),
            'raw_message': tg_msg.model_dump() if hasattr(tg_msg, 'model_dump') else str(tg_msg)
        }
        
        return Message(
            id=f"tg_{tg_msg.message_id}_{chat.id}",
            text=text,
            message_type=message_type,
            user_id=str(user.id) if user else 'unknown',
            user_name=user_name,
            chat_id=str(chat.id),
            chat_type=chat_type,
            timestamp=datetime.now().isoformat(),
            metadata=metadata
        )
    
    async def send_message(self, chat_id: str, text: str, **kwargs) -> bool:
        """
        Send a text message via Telegram.
        
        Args:
            chat_id: Telegram chat ID
            text: Message text
            **kwargs: Additional Telegram-specific options
            
        Returns:
            True if message was sent successfully
        """
        if not self.bot:
            logger.error("❌ Telegram bot not initialized")
            return False
        
        try:
            await self.bot.send_message(
                chat_id=int(chat_id),
                text=text,
                **kwargs
            )
            logger.info(f"📤 Sent Telegram message to {chat_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram message to {chat_id}: {e}")
            return False
    
    async def send_photo(self, chat_id: str, photo_path: str, caption: Optional[str] = None) -> bool:
        """
        Send a photo via Telegram.
        
        Args:
            chat_id: Telegram chat ID
            photo_path: Path to photo file or URL
            caption: Optional photo caption
            
        Returns:
            True if photo was sent successfully
        """
        if not self.bot:
            logger.error("❌ Telegram bot not initialized")
            return False
        
        try:
            await self.bot.send_photo(
                chat_id=int(chat_id),
                photo=photo_path,
                caption=caption
            )
            logger.info(f"📸 Sent Telegram photo to {chat_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram photo to {chat_id}: {e}")
            return False
    
    async def send_document(self, chat_id: str, document_path: str, caption: Optional[str] = None) -> bool:
        """
        Send a document via Telegram.
        
        Args:
            chat_id: Telegram chat ID
            document_path: Path to document file
            caption: Optional document caption
            
        Returns:
            True if document was sent successfully
        """
        if not self.bot:
            logger.error("❌ Telegram bot not initialized")
            return False
        
        try:
            await self.bot.send_document(
                chat_id=int(chat_id),
                document=document_path,
                caption=caption
            )
            logger.info(f"📄 Sent Telegram document to {chat_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram document to {chat_id}: {e}")
            return False
    
    def get_platform_name(self) -> str:
        """Get the platform name."""
        return "telegram"
    
    def get_bot_info(self) -> Dict[str, Any]:
        """Get bot information."""
        if not self.bot:
            return {'error': 'Bot not initialized'}
        
        # This would need to be called in an async context
        return {
            'platform': 'telegram',
            'token_configured': bool(self.token),
            'is_running': self.is_running
        }
    
    async def get_bot_info_async(self) -> Dict[str, Any]:
        """Get bot information asynchronously."""
        if not self.bot:
            return {'error': 'Bot not initialized'}
        
        try:
            bot_info = await self.bot.get_me()
            return {
                'platform': 'telegram',
                'id': bot_info.id,
                'username': bot_info.username,
                'first_name': bot_info.first_name,
                'is_bot': bot_info.is_bot,
                'can_join_groups': bot_info.can_join_groups,
                'can_read_all_group_messages': bot_info.can_read_all_group_messages,
                'supports_inline_queries': bot_info.supports_inline_queries,
                'is_running': self.is_running
            }
        except Exception as e:
            return {'error': f'Failed to get bot info: {e}'} 