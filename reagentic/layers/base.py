"""
Base Layer class for the App Architecture.

This module contains the abstract base Layer class that all specific layers inherit from.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..app import App
    from ..events import Event
    from ..subsystems.subsystem_base import SubsystemBase

logger = logging.getLogger(__name__)


class Layer(ABC):
    """
    Base class for all app layers.
    
    Each layer has an event queue and processes events sequentially.
    By default, events are forwarded to the next layer in the chain:
    Monitoring -> Decision -> Action -> Learning
    """
    
    def __init__(self, name: str, next_layer: Optional['Layer'] = None):
        self.name = name
        self.next_layer = next_layer
        self.event_queue: asyncio.Queue['Event'] = asyncio.Queue()
        self.running = False
        self.app: Optional['App'] = None
        
    def set_app(self, app: 'App'):
        """Set reference to the main app for inter-layer communication."""
        self.app = app
        
    async def start(self):
        """Start the layer's event processing loop."""
        self.running = True
        logger.info(f"Starting {self.name} layer")
        
        while self.running:
            try:
                # Wait for events with a timeout to allow periodic checks
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self.process_event(event)
                self.event_queue.task_done()
            except asyncio.TimeoutError:
                # Periodic check for shutdown
                continue
            except Exception as e:
                logger.error(f"Error processing event in {self.name}: {e}")
                
    async def stop(self):
        """Stop the layer's event processing."""
        self.running = False
        logger.info(f"Stopping {self.name} layer")
        
    async def add_event(self, event: 'Event'):
        """Add an event to this layer's processing queue."""
        await self.event_queue.put(event)
        
    async def process_event(self, event: 'Event'):
        """
        Process an event. Override in subclasses for custom behavior.
        Default behavior is to process and forward to next layer.
        """
        logger.debug(f"{self.name} processing event: {event.event_type}")
        
        # Custom processing - override this method
        processed_event = await self.process_event_impl(event)
        
        # Forward to next layer if available
        if processed_event and self.next_layer:
            processed_event.source_layer = self.name
            await self.next_layer.add_event(processed_event)
            
    @abstractmethod
    async def process_event_impl(self, event: 'Event') -> Optional['Event']:
        """
        Implement custom event processing logic in subclasses.
        Return the event to forward to next layer, or None to stop propagation.
        """
        pass
        
    async def send_to_layer(self, event: 'Event', target_layer_name: str):
        """Send an event to a specific layer by name."""
        if self.app:
            target_layer = self.app.get_layer(target_layer_name)
            if target_layer:
                event.source_layer = self.name
                event.target_layer = target_layer_name
                await target_layer.add_event(event)
            else:
                logger.warning(f"Target layer '{target_layer_name}' not found")
        else:
            logger.warning("No app reference set, cannot send to specific layer")
            
    def get_subsystem(self, name: str) -> Optional['SubsystemBase']:
        """Get a subsystem by name from the app."""
        if self.app:
            return self.app.get_subsystem(name)
        else:
            logger.warning("No app reference set, cannot access subsystems")
            return None
            
    @property
    def subsystems(self) -> Dict[str, 'SubsystemBase']:
        """Get all subsystems from the app."""
        if self.app:
            return self.app.subsystems
        else:
            logger.warning("No app reference set, cannot access subsystems")
            return {} 