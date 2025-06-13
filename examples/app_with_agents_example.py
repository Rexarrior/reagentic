"""
App Architecture with Agents Example

This example demonstrates:
1. Agents working within each layer (Monitoring, Decision, Action, Learning)
2. Shared memory subsystem between Decision and Learning layers
3. Complex event processing and decision making
4. Feedback loop for learning and improvement
"""

import asyncio
import logging
from typing import Any, Dict, Optional
from agents import Agent, Runner

from reagentic.app import App
from reagentic.events import Event, EventType
from reagentic.layers import Layer, MonitoringLayer, DecisionLayer, ActionLayer, LearningLayer
from reagentic.subsystems.memory import FileBasedMemory
import reagentic.providers.openrouter as openrouter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Provider for agents
provider = openrouter.OpenrouterProvider(openrouter.DEEPSEEK_CHAT_V3_0324)


class AgentMonitoringLayer(MonitoringLayer):
    """Monitoring layer with an agent that summarizes and evaluates events."""
    
    def __init__(self, next_layer: Optional[Layer] = None):
        super().__init__(next_layer)
        self.agent = Agent(
            name="Monitoring Agent",
            instructions="""You are a monitoring agent that analyzes events and creates comprehensive reports.

Your task:
1. Analyze incoming events (observations, triggers, schedules)
2. Summarize key information
3. Evaluate severity, urgency, and potential impact
4. Create a structured report for the decision layer

Output format:
- Summary: Brief description of the event
- Severity: low/medium/high/critical
- Urgency: low/medium/high/immediate
- Impact: Description of potential consequences
- Recommendation: What type of action might be needed

Be concise but thorough. Focus on actionable insights.""",
            model=provider.get_openai_model()
        )
        
    async def process_event_impl(self, event: Event) -> Optional[Event]:
        """Process events using the monitoring agent."""
        logger.info(f"🔍 Monitoring agent analyzing {event.event_type}")
        
        if event.event_type in [EventType.OBSERVATION, EventType.TRIGGER, EventType.SCHEDULE]:
            # Prepare context for the agent
            event_context = f"""
Event Type: {event.event_type.value}
Event Data: {event.data}
Timestamp: {event.timestamp}

Please analyze this event and provide a comprehensive report.
"""
            
            # Get agent analysis
            try:
                result = await Runner.run(self.agent, event_context)
                analysis_report = result.final_output
                
                logger.info(f"📊 Monitoring report generated: {len(analysis_report)} characters")
                
                # Create decision event with the analysis
                return Event(
                    event_type=EventType.DECISION,
                    data={
                        "original_event": event.data,
                        "monitoring_report": analysis_report,
                        "event_type": event.event_type.value,
                        "analysis_required": True,
                        "context": "agent_monitoring"
                    }
                )
                
            except Exception as e:
                logger.error(f"❌ Monitoring agent failed: {e}")
                # Fallback to basic processing
                return await super().process_event_impl(event)
        
        # Pass through other events
        return event


class AgentDecisionLayer(DecisionLayer):
    """Decision layer with an agent that analyzes reports and makes decisions."""
    
    def __init__(self, next_layer: Optional[Layer] = None):
        super().__init__(next_layer)
        self.agent = Agent(
            name="Decision Agent",
            instructions="""You are a decision-making agent that analyzes monitoring reports and decides on actions.

Your task:
1. Review monitoring reports and event data
2. Consider past decisions and outcomes from memory
3. Decide if action is needed and what type
4. Provide clear reasoning for your decision

Decision types:
- "no_action": No action needed, just monitoring
- "investigate": Gather more information
- "respond": Take immediate action
- "escalate": Requires urgent attention
- "maintain": Continue current approach

Output format:
Decision: [decision_type]
Reasoning: [clear explanation]
Priority: [low/medium/high/critical]
Action_details: [specific instructions for action layer]

Be decisive but consider past experiences from memory context.""",
            model=provider.get_openai_model()
        )
        
    async def process_event_impl(self, event: Event) -> Optional[Event]:
        """Process decision events using the decision agent."""
        logger.info(f"🧠 Decision agent analyzing event")
        
        if event.event_type == EventType.DECISION:
            # Get shared memory for context
            memory = self.get_subsystem("shared_memory")
            
            # Prepare context with memory enrichment
            decision_context = f"""
Monitoring Report: {event.data.get('monitoring_report', 'No report available')}
Original Event: {event.data.get('original_event', {})}
Event Type: {event.data.get('event_type', 'unknown')}

Please make a decision based on this information and any relevant past experiences.
"""
            
            # Enrich with memory if available
            if memory:
                decision_context = memory.enrich_full(decision_context)
            
            try:
                result = await Runner.run(self.agent, decision_context)
                decision_output = result.final_output
                
                logger.info(f"⚖️ Decision made: {len(decision_output)} characters")
                
                # Store decision in memory
                if memory:
                    await memory.append_raw(f"Decision made: {decision_output}\n---\n")
                
                # Parse decision (simple parsing for demo)
                decision_type = "respond"  # Default
                if "no_action" in decision_output.lower():
                    decision_type = "no_action"
                elif "investigate" in decision_output.lower():
                    decision_type = "investigate"
                elif "escalate" in decision_output.lower():
                    decision_type = "escalate"
                
                # Only create action event if action is needed
                if decision_type != "no_action":
                    return Event(
                        event_type=EventType.ACTION,
                        data={
                            "decision": decision_output,
                            "decision_type": decision_type,
                            "original_event": event.data.get('original_event'),
                            "monitoring_report": event.data.get('monitoring_report'),
                            "action_required": True
                        }
                    )
                else:
                    logger.info("🚫 No action required, decision complete")
                    return None
                    
            except Exception as e:
                logger.error(f"❌ Decision agent failed: {e}")
                # Fallback to basic processing
                return await super().process_event_impl(event)
        
        return event


class AgentActionLayer(ActionLayer):
    """Action layer with an agent that implements decisions intelligently."""
    
    def __init__(self, next_layer: Optional[Layer] = None):
        super().__init__(next_layer)
        self.agent = Agent(
            name="Action Agent",
            instructions="""You are an action implementation agent that executes decisions intelligently.

Your task:
1. Review the decision and context
2. Plan and implement the appropriate action
3. Provide detailed results of your actions
4. Assess the effectiveness of your implementation

Action types you can handle:
- investigate: Gather and analyze additional information
- respond: Provide appropriate response or solution
- escalate: Prepare escalation with detailed context
- maintain: Continue monitoring with specific parameters

Output format:
Action_taken: [description of what you did]
Results: [detailed results and outcomes]
Effectiveness: [your assessment: excellent/good/fair/poor]
Recommendations: [suggestions for future similar situations]

Be thorough and provide actionable results.""",
            model=provider.get_openai_model()
        )
        
    async def process_event_impl(self, event: Event) -> Optional[Event]:
        """Process action events using the action agent."""
        logger.info(f"🎯 Action agent implementing decision")
        
        if event.event_type == EventType.ACTION:
            # Prepare context for action
            action_context = f"""
Decision: {event.data.get('decision', 'No decision provided')}
Decision Type: {event.data.get('decision_type', 'unknown')}
Original Event: {event.data.get('original_event', {})}
Monitoring Report: {event.data.get('monitoring_report', 'No report')}

Please implement this decision and provide detailed results.
"""
            
            try:
                result = await Runner.run(self.agent, action_context)
                action_results = result.final_output
                
                logger.info(f"✅ Action completed: {len(action_results)} characters")
                
                # Parse effectiveness (simple parsing for demo)
                effectiveness = "good"  # Default
                if "excellent" in action_results.lower():
                    effectiveness = "excellent"
                elif "poor" in action_results.lower():
                    effectiveness = "poor"
                elif "fair" in action_results.lower():
                    effectiveness = "fair"
                
                # Create feedback event for learning
                return Event(
                    event_type=EventType.FEEDBACK,
                    data={
                        "action_results": action_results,
                        "effectiveness": effectiveness,
                        "decision": event.data.get('decision'),
                        "decision_type": event.data.get('decision_type'),
                        "original_event": event.data.get('original_event'),
                        "success": effectiveness in ["excellent", "good"]
                    }
                )
                
            except Exception as e:
                logger.error(f"❌ Action agent failed: {e}")
                # Fallback to basic processing
                return await super().process_event_impl(event)
        
        return event


class AgentLearningLayer(LearningLayer):
    """Learning layer with an agent that processes feedback and evaluates effectiveness."""
    
    def __init__(self, next_layer: Optional[Layer] = None):
        super().__init__(next_layer)
        self.agent = Agent(
            name="Learning Agent",
            instructions="""You are a learning agent that analyzes feedback and improves decision-making.

Your task:
1. Analyze action results and effectiveness
2. Identify patterns in successful and unsuccessful decisions
3. Extract lessons learned for future decisions
4. Update knowledge base with insights

Focus on:
- What worked well and why
- What could be improved
- Patterns in decision effectiveness
- Recommendations for similar future situations

Output format:
Analysis: [detailed analysis of the action effectiveness]
Lessons_learned: [key insights for future decisions]
Pattern_identified: [any patterns in success/failure]
Recommendations: [specific suggestions for improvement]

Be analytical and focus on actionable insights for better future decisions.""",
            model=provider.get_openai_model()
        )
        
    async def process_event_impl(self, event: Event) -> Optional[Event]:
        """Process feedback events using the learning agent."""
        logger.info(f"🧠 Learning agent analyzing feedback")
        
        if event.event_type == EventType.FEEDBACK:
            # Get shared memory for context and storage
            memory = self.get_subsystem("shared_memory")
            
            # Prepare context with memory
            learning_context = f"""
Action Results: {event.data.get('action_results', 'No results')}
Effectiveness: {event.data.get('effectiveness', 'unknown')}
Decision: {event.data.get('decision', 'No decision')}
Decision Type: {event.data.get('decision_type', 'unknown')}
Success: {event.data.get('success', False)}

Please analyze this feedback and extract lessons for future decisions.
"""
            
            # Enrich with memory for pattern recognition
            if memory:
                learning_context = memory.enrich_full(learning_context)
            
            try:
                result = await Runner.run(self.agent, learning_context)
                learning_analysis = result.final_output
                
                logger.info(f"📚 Learning analysis completed: {len(learning_analysis)} characters")
                
                # Store learning in memory
                if memory:
                    await memory.append_raw(f"Learning Analysis: {learning_analysis}\n---\n")
                    
                    # Also store structured insights
                    effectiveness = event.data.get('effectiveness', 'unknown')
                    decision_type = event.data.get('decision_type', 'unknown')
                    success = event.data.get('success', False)
                    
                    await memory.write_by_key(
                        f"pattern_{decision_type}_{effectiveness}",
                        f"Success: {success}, Analysis: {learning_analysis[:200]}..."
                    )
                
                # Store in local experiences
                await super().store_experience(event.data)
                
                logger.info(f"💡 Learning complete. Total experiences: {len(self.experiences)}")
                
                # Learning events don't propagate further
                return None
                
            except Exception as e:
                logger.error(f"❌ Learning agent failed: {e}")
                # Fallback to basic processing
                return await super().process_event_impl(event)
        
        return event


async def simulate_complex_scenarios(app: App):
    """Simulate various scenarios to test the agent-based layers."""
    await asyncio.sleep(1)  # Let the app start
    
    logger.info("🌍 Starting complex scenario simulation...")
    
    scenarios = [
        {
            "name": "High Temperature Alert",
            "type": "observation",
            "data": {
                "sensor_type": "temperature",
                "sensor_value": 0.95,
                "location": "server_room_a",
                "threshold_exceeded": True,
                "critical": True
            }
        },
        {
            "name": "Security Breach Detected",
            "type": "trigger",
            "data": {
                "trigger_type": "security_alert",
                "source": "firewall",
                "severity": "high",
                "details": "Multiple failed login attempts from unknown IP"
            }
        },
        {
            "name": "Routine Maintenance",
            "type": "schedule",
            "data": {
                "task": "system_backup",
                "scheduled_time": "02:00",
                "duration": "30min",
                "priority": "normal"
            }
        },
        {
            "name": "Network Anomaly",
            "type": "observation",
            "data": {
                "sensor_type": "network_traffic",
                "sensor_value": 0.85,
                "location": "main_gateway",
                "anomaly_detected": True,
                "pattern": "unusual_data_transfer"
            }
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        logger.info(f"\n{'='*60}")
        logger.info(f"🎭 Scenario {i+1}: {scenario['name']}")
        logger.info(f"{'='*60}")
        
        if scenario["type"] == "observation":
            await app.observe(scenario["data"])
        elif scenario["type"] == "trigger":
            await app.trigger(scenario["data"])
        elif scenario["type"] == "schedule":
            await app.schedule(scenario["data"])
        
        # Wait for processing
        await asyncio.sleep(5)
    
    logger.info("\n🏁 Complex scenario simulation completed")


async def main():
    """Main example function."""
    logger.info("🚀 Starting App with Agents Example")
    
    # Create shared memory subsystem for decision and learning layers
    shared_memory = FileBasedMemory("agent_decisions_memory.json")
    shared_memory.modify_structure(
        "Shared memory for decision and learning agents. Stores decisions, outcomes, "
        "patterns, and lessons learned to improve future decision-making."
    )
    
    # Create agent-powered layers
    learning_layer = AgentLearningLayer()
    action_layer = AgentActionLayer(learning_layer)
    decision_layer = AgentDecisionLayer(action_layer)
    monitoring_layer = AgentMonitoringLayer(decision_layer)
    
    # Create app with agent layers and shared memory
    app = App(
        monitoring_layer=monitoring_layer,
        decision_layer=decision_layer,
        action_layer=action_layer,
        learning_layer=learning_layer,
        subsystems={
            "shared_memory": shared_memory
        }
    )
    
    logger.info(f"🤖 App created with agent-powered layers")
    logger.info(f"📦 Subsystems: {list(app.subsystems.keys())}")
    
    # Create tasks
    app_task = asyncio.create_task(app.run())
    simulation_task = asyncio.create_task(simulate_complex_scenarios(app))
    
    try:
        # Run simulation
        await asyncio.wait_for(simulation_task, timeout=60)
        
        # Give time for final processing
        await asyncio.sleep(3)
        
        # Show final memory state
        logger.info("\n📊 Final Memory State:")
        memory = app.get_subsystem("shared_memory")
        if memory:
            try:
                keys = await memory.list_keys()
                logger.info(f"Memory keys: {keys}")
            except AttributeError:
                logger.info("Memory keys: Not available for this memory type")
            
            raw_content = memory.read_raw()
            logger.info(f"Raw content length: {len(raw_content)} characters")
            logger.info(f"Learning experiences: {len(learning_layer.experiences)}")
        
    except asyncio.TimeoutError:
        logger.info("⏰ Simulation timeout reached")
    except KeyboardInterrupt:
        logger.info("⌨️ Keyboard interrupt received")
    except Exception as e:
        logger.error(f"❌ Error during simulation: {e}")
    finally:
        # Stop the app
        await app.stop()
        
        # Cancel remaining tasks
        if not app_task.done():
            app_task.cancel()
        
        logger.info("✅ Agent-powered app example completed")


if __name__ == "__main__":
    asyncio.run(main()) 