"""
Events module for the App Architecture.

This module contains event types and the Event class used for communication between layers.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class EventType(Enum):
    """Types of events that can flow through the app system."""
    OBSERVATION = "observation"
    DECISION = "decision"
    ACTION = "action"
    FEEDBACK = "feedback"
    TRIGGER = "trigger"
    SCHEDULE = "schedule"
    NOTIFICATION = "notification"
    LEARNING = "learning"


@dataclass
class Event:
    """An event that flows through the app system."""
    event_type: EventType
    data: Dict[str, Any]
    source_layer: Optional[str] = None
    target_layer: Optional[str] = None
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            import time
            self.timestamp = time.time() 