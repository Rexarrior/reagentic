"""
App Architecture

This module implements an app system with layered architecture:
- MonitoringLayer: Environment sensors, event triggers, time scheduler
- DecisionLayer: Context analyzer, trend predictor, action planner  
- ActionLayer: Action executor, notification system, activity logger
- LearningLayer: Experience memory, strategy optimizer, behavior adapter

Each layer processes events sequentially through event queues and can route 
events to any other layer in the system.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .events import Event, EventType
from .layers import Layer, MonitoringLayer, DecisionLayer, ActionLayer, LearningLayer
from .subsystems.subsystem_base import SubsystemBase

logger = logging.getLogger(__name__)


class App:
    """
    Main application class that orchestrates all layers.
    
    Contains monitoring, decision, action, and learning layers.
    Runs async loops for all layer processing.
    """
    
    def __init__(
        self,
        monitoring_layer: Optional[MonitoringLayer] = None,
        decision_layer: Optional[DecisionLayer] = None,
        action_layer: Optional[ActionLayer] = None,
        learning_layer: Optional[LearningLayer] = None,
        subsystems: Optional[Dict[str, SubsystemBase]] = None
    ):
        """
        Initialize the app with optional custom layer instances and subsystems.
        If not provided, default instances will be created.
        
        Args:
            monitoring_layer: Custom monitoring layer instance
            decision_layer: Custom decision layer instance  
            action_layer: Custom action layer instance
            learning_layer: Custom learning layer instance
            subsystems: Dictionary of subsystems (name -> subsystem instance)
        """
        # Create default instances if not provided
        self.learning_layer = learning_layer or LearningLayer()
        self.action_layer = action_layer or ActionLayer(self.learning_layer)
        self.decision_layer = decision_layer or DecisionLayer(self.action_layer)
        self.monitoring_layer = monitoring_layer or MonitoringLayer(self.decision_layer)
        
        # Set up the chain
        self.layers = {
            "monitoring": self.monitoring_layer,
            "decision": self.decision_layer,
            "action": self.action_layer,
            "learning": self.learning_layer
        }
        
        # Set up subsystems
        self.subsystems: Dict[str, SubsystemBase] = subsystems or {}
        
        # Set app reference for all layers
        for layer in self.layers.values():
            layer.set_app(self)
            
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
    def get_layer(self, name: str) -> Optional[Layer]:
        """Get a layer by name."""
        return self.layers.get(name)
        
    def get_subsystem(self, name: str) -> Optional[SubsystemBase]:
        """Get a subsystem by name."""
        return self.subsystems.get(name)
        
    def add_subsystem(self, name: str, subsystem: SubsystemBase):
        """Add a subsystem to the app."""
        self.subsystems[name] = subsystem
        logger.info(f"Added subsystem: {name}")
        
    def remove_subsystem(self, name: str) -> Optional[SubsystemBase]:
        """Remove and return a subsystem by name."""
        subsystem = self.subsystems.pop(name, None)
        if subsystem:
            logger.info(f"Removed subsystem: {name}")
        return subsystem
        
    async def start(self):
        """Start all layers and their event processing loops."""
        logger.info("Starting app")
        self.running = True
        
        # Start all layers
        for layer_name, layer in self.layers.items():
            task = asyncio.create_task(layer.start())
            task.set_name(f"{layer_name}_layer")
            self.tasks.append(task)
            
        logger.info("All layers started successfully")
        
    async def stop(self):
        """Stop all layers and clean up."""
        logger.info("Stopping app")
        self.running = False
        
        # Stop all layers
        for layer in self.layers.values():
            await layer.stop()
            
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        self.tasks.clear()
        logger.info("App stopped")
        
    async def run(self):
        """
        Run the app application.
        This starts all layers and keeps the app running until stopped.
        """
        try:
            await self.start()
            
            # Keep running until stopped
            while self.running:
                await asyncio.sleep(1)
                
                # Check if any critical task failed
                for task in self.tasks:
                    if task.done() and not task.cancelled():
                        exception = task.exception()
                        if exception:
                            logger.error(f"Task {task.get_name()} failed: {exception}")
                            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error in app run loop: {e}")
        finally:
            await self.stop()
            
    # Convenience methods for external interaction
    async def observe(self, observation_data: Dict[str, Any]):
        """Add an observation to the monitoring layer."""
        await self.monitoring_layer.observe(observation_data)
        
    async def trigger(self, trigger_data: Dict[str, Any]):
        """Add a trigger event to the monitoring layer."""
        await self.monitoring_layer.trigger(trigger_data)
        
    async def schedule(self, schedule_data: Dict[str, Any]):
        """Add a scheduled event to the monitoring layer."""
        await self.monitoring_layer.schedule(schedule_data) 