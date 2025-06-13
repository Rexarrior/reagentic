"""
Monitoring Layer for the App Architecture.

This module contains the MonitoringLayer class responsible for:
- Environmental observation
- Event detection and triggering  
- Scheduled activities
"""

import logging
from typing import Any, Dict, Optional

from .base import Layer
from ..events import Event, EventType

logger = logging.getLogger(__name__)


class MonitoringLayer(Layer):
    """
    Monitoring Layer: Environment sensors, event triggers, time scheduler.
    
    Responsible for:
    - Environmental observation
    - Event detection and triggering
    - Scheduled activities
    """
    
    def __init__(self, next_layer: Optional[Layer] = None):
        super().__init__("Monitoring", next_layer)
        
    async def process_event_impl(self, event: Event) -> Optional[Event]:
        """Process monitoring events and observations."""
        logger.info(f"Monitoring layer processing {event.event_type}")
        
        # Transform observations into decisions for the decision layer
        if event.event_type == EventType.OBSERVATION:
            return Event(
                event_type=EventType.DECISION,
                data={
                    "observation": event.data,
                    "analysis_required": True,
                    "context": "monitoring_input"
                }
            )
        elif event.event_type == EventType.TRIGGER:
            return Event(
                event_type=EventType.DECISION,
                data={
                    "trigger": event.data,
                    "urgent": True,
                    "context": "triggered_event"
                }
            )
        elif event.event_type == EventType.SCHEDULE:
            return Event(
                event_type=EventType.DECISION,
                data={
                    "scheduled_task": event.data,
                    "context": "scheduled_activity"
                }
            )
            
        # Pass through other events
        return event
        
    async def observe(self, observation_data: Dict[str, Any]):
        """Add an observation to the monitoring system."""
        event = Event(EventType.OBSERVATION, observation_data)
        await self.add_event(event)
        
    async def trigger(self, trigger_data: Dict[str, Any]):
        """Add a trigger event to the monitoring system."""
        event = Event(EventType.TRIGGER, trigger_data)
        await self.add_event(event)
        
    async def schedule(self, schedule_data: Dict[str, Any]):
        """Add a scheduled event to the monitoring system."""
        event = Event(EventType.SCHEDULE, schedule_data)
        await self.add_event(event) 