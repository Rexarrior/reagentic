"""
Action Layer for the App Architecture.

This module contains the ActionLayer class responsible for:
- Executing planned actions
- Sending notifications
- Logging activities
"""

import logging
from typing import Any, Dict, Optional

from .base import Layer
from ..events import Event, EventType

logger = logging.getLogger(__name__)


class ActionLayer(Layer):
    """
    Action Layer: Action executor, notification system, activity logger.
    
    Responsible for:
    - Executing planned actions
    - Sending notifications
    - Logging activities
    """
    
    def __init__(self, next_layer: Optional[Layer] = None):
        super().__init__("Action", next_layer)
        
    async def process_event_impl(self, event: Event) -> Optional[Event]:
        """Process action events and execute planned actions."""
        logger.info(f"Action layer processing {event.event_type}")
        
        if event.event_type == EventType.ACTION:
            # Execute the action plan
            execution_result = await self.execute_action(event.data)
            
            # Log the activity
            await self.log_activity(event.data, execution_result)
            
            # Send notifications if needed
            if execution_result.get("notify"):
                await self.send_notification(execution_result)
            
            # Create feedback for learning layer
            return Event(
                event_type=EventType.FEEDBACK,
                data={
                    "execution_result": execution_result,
                    "action_plan": event.data.get("action_plan"),
                    "success": execution_result.get("success", True)
                }
            )
            
        # Pass through other events
        return event
        
    async def execute_action(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the planned action."""
        # Placeholder for action execution logic
        action_plan = action_data.get("action_plan", {})
        logger.info(f"Executing action: {action_plan.get('action_type', 'unknown')}")
        
        return {
            "success": True,
            "action_type": action_plan.get("action_type"),
            "duration": 2.5,
            "output": "Action completed successfully",
            "notify": True
        }
        
    async def log_activity(self, action_data: Dict[str, Any], result: Dict[str, Any]):
        """Log the activity."""
        logger.info(f"Activity logged: {result.get('action_type')} - Success: {result.get('success')}")
        
    async def send_notification(self, result: Dict[str, Any]):
        """Send notifications about action results."""
        logger.info(f"Notification: Action {result.get('action_type')} completed") 