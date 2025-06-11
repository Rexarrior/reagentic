# Reagentic Subsystem Pattern

The Reagentic framework implements a powerful **Subsystem Pattern** that provides a modular, reusable approach to organizing and sharing functionality across AI agents. This pattern allows you to create self-contained components that can be easily connected to multiple agents, promoting code reuse and clean architecture.

## 🚀 Features

- **Modular Design**: Self-contained subsystems with clear boundaries
- **Tool Organization**: Categorize tools by functionality and purpose
- **Agent Integration**: Easy connection to any agent instance
- **Method Binding**: Proper `self` binding for instance methods
- **Type Safety**: Full typing support with categorized tools
- **Event System**: Built-in event handling for subsystem interactions
- **Flexible Configuration**: Support for both class-level and instance-level tools

## 📋 Quick Start

### Basic Subsystem Creation

```python
from reagentic.subsystems import SubsystemBase
from agents import Agent

class MySubsystem(SubsystemBase):
    def __init__(self, config_value: str):
        super().__init__()
        self.config_value = config_value
    
    @SubsystemBase.subsystem_tool()
    def get_status(self) -> str:
        """Get the current status of this subsystem."""
        return f"Status: {self.config_value}"
    
    @SubsystemBase.subsystem_tool("advanced")
    def advanced_operation(self, data: str) -> str:
        """Perform an advanced operation with the data."""
        return f"Processed {data} with config: {self.config_value}"

# Usage
subsystem = MySubsystem("active")
agent = Agent(name="Assistant", instructions="You are helpful")
subsystem.connect_tools(agent)  # Connect default tools
```

### Tool Categorization

```python
class DataSubsystem(SubsystemBase):
    @SubsystemBase.subsystem_tool("read")
    def read_data(self, key: str) -> str:
        """Read data by key."""
        return f"Data for {key}"
    
    @SubsystemBase.subsystem_tool("write")
    def write_data(self, key: str, value: str) -> str:
        """Write data to storage."""
        return f"Stored {key}={value}"
    
    @SubsystemBase.subsystem_tool(["read", "write"])
    def get_keys(self) -> list[str]:
        """Get all available keys (available in both read and write categories)."""
        return ["key1", "key2", "key3"]

# Connect specific tool categories
subsystem = DataSubsystem()
subsystem.connect_tools(agent, "read")  # Only read tools
subsystem.connect_tools(agent, ["read", "write"])  # Multiple categories
subsystem.connect_all_tools(agent)  # All tools
```

## 🏗️ Architecture

### Subsystem Base Class

The `SubsystemBase` class provides the foundation for all subsystems:

```python
class SubsystemBase:
    def __init__(self):
        # Initialize tool storage
        if not hasattr(self, 'tools'):
            self.tools = {DEFAULT_TOOL_TYPE: []}
    
    @classmethod
    def subsystem_tool(cls, tool_type: Union[str, List[str]] = DEFAULT_TOOL_TYPE):
        """Decorator for registering methods as agent tools."""
        # Implementation handles method binding and tool registration
    
    def get_tools(self, tool_type: Union[str, List[str]] = DEFAULT_TOOL_TYPE) -> List:
        """Get tools for specific categories."""
    
    def connect_tools(self, agent: Agent, tool_type: Union[str, List[str]] = DEFAULT_TOOL_TYPE):
        """Connect tools to an agent."""
    
    def connect_all_tools(self, agent: Agent):
        """Connect all tools from all categories."""
    
    def connect(self, agent: Agent):
        """Connect both tools and hooks to an agent."""
```

### Method Binding System

The subsystem pattern ensures proper method binding:

1. **Decorator Phase**: Methods are registered without immediate tool conversion
2. **Binding Phase**: When `get_tools()` is called, methods are bound to the instance
3. **Tool Creation**: `function_tool()` is applied to bound methods
4. **Agent Connection**: Bound tools are added to the agent

```python
# Internal flow:
def get_tools(self, tool_type):
    for method in class_methods:
        bound_method = python_types.MethodType(method, self)  # Bind to instance
        tool_func = function_tool(bound_method)               # Create tool
        all_tools.append(tool_func)                          # Add to collection
```

## 🔧 Advanced Usage

### Memory Subsystem Example

```python
from reagentic.subsystems.memory import FileBasedMemory

class CustomMemorySubsystem(FileBasedMemory):
    def __init__(self, file_path: str = "memory.json"):
        super().__init__(file_path)
        self.custom_data = {}
    
    @SubsystemBase.subsystem_tool("custom")
    def store_custom(self, category: str, data: str) -> str:
        """Store data in a custom category."""
        if category not in self.custom_data:
            self.custom_data[category] = []
        self.custom_data[category].append(data)
        return f"Stored in {category}: {data}"
    
    @SubsystemBase.subsystem_tool("custom")
    def get_custom(self, category: str) -> str:
        """Retrieve custom data by category."""
        return str(self.custom_data.get(category, []))

# Usage with different tool sets
memory = CustomMemorySubsystem()
agent = Agent(name="Assistant")

# Connect only memory tools (read/write/append)
memory.connect_tools(agent)

# Connect only custom tools
memory.connect_tools(agent, "custom")

# Connect all tools
memory.connect_all_tools(agent)
```

### Multi-Category Tools

```python
class APISubsystem(SubsystemBase):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
    
    @SubsystemBase.subsystem_tool(["get", "api"])
    def get_user(self, user_id: str) -> str:
        """Get user information."""
        return f"GET {self.base_url}/users/{user_id}"
    
    @SubsystemBase.subsystem_tool(["post", "api"])
    def create_user(self, name: str, email: str) -> str:
        """Create a new user."""
        return f"POST {self.base_url}/users - {name}:{email}"
    
    @SubsystemBase.subsystem_tool(["get", "post", "admin"])
    def get_stats(self) -> str:
        """Get API statistics (admin only)."""
        return f"Stats from {self.base_url}/stats"

# Flexible tool connection
api = APISubsystem("https://api.example.com")

# Different agents get different capabilities
readonly_agent = Agent(name="ReadOnly")
api.connect_tools(readonly_agent, "get")

admin_agent = Agent(name="Admin")
api.connect_tools(admin_agent, ["get", "post", "admin"])
```

### Event-Driven Subsystems

```python
from typing import Callable, Dict, Any
from enum import Enum

class EventType(Enum):
    DATA_CHANGED = "data_changed"
    OPERATION_COMPLETED = "operation_completed"

class EventDrivenSubsystem(SubsystemBase):
    def __init__(self):
        super().__init__()
        self.data = {}
        self.event_handlers: Dict[EventType, List[Callable]] = {
            event_type: [] for event_type in EventType
        }
    
    def add_event_handler(self, event_type: EventType, handler: Callable[[Dict[str, Any]], None]):
        """Register an event handler."""
        self.event_handlers[event_type].append(handler)
    
    def _trigger_event(self, event_type: EventType, event_data: Dict[str, Any]):
        """Trigger all handlers for an event type."""
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(event_data)
            except Exception as e:
                print(f"Event handler error: {e}")
    
    @SubsystemBase.subsystem_tool()
    def set_data(self, key: str, value: str) -> str:
        """Set data and trigger events."""
        old_value = self.data.get(key)
        self.data[key] = value
        
        # Trigger event
        self._trigger_event(EventType.DATA_CHANGED, {
            'key': key,
            'old_value': old_value,
            'new_value': value
        })
        
        return f"Set {key}={value}"

# Usage with event handling
def on_data_change(event_data):
    print(f"Data changed: {event_data['key']} = {event_data['new_value']}")

subsystem = EventDrivenSubsystem()
subsystem.add_event_handler(EventType.DATA_CHANGED, on_data_change)
```

## 🔄 Integration Patterns

### Multiple Subsystems

```python
from reagentic.subsystems.memory import FileBasedMemory

# Create multiple subsystems
memory = FileBasedMemory("agent_memory.json")
api = APISubsystem("https://api.example.com")
custom = CustomSubsystem()

# Agent with multiple subsystems
agent = Agent(name="MultiAgent")

# Connect different tool sets
memory.connect_tools(agent)              # Default memory tools
api.connect_tools(agent, "get")          # Read-only API tools
custom.connect_tools(agent, "advanced")  # Advanced custom tools
```

### Conditional Tool Connection

```python
class ConditionalSubsystem(SubsystemBase):
    def __init__(self, admin_mode: bool = False):
        super().__init__()
        self.admin_mode = admin_mode
    
    @SubsystemBase.subsystem_tool()
    def basic_operation(self) -> str:
        """Basic operation available to all users."""
        return "Basic operation completed"
    
    @SubsystemBase.subsystem_tool("admin")
    def admin_operation(self) -> str:
        """Admin-only operation."""
        return "Admin operation completed"
    
    def connect_appropriate_tools(self, agent: Agent):
        """Connect tools based on configuration."""
        self.connect_tools(agent)  # Always connect basic tools
        
        if self.admin_mode:
            self.connect_tools(agent, "admin")  # Add admin tools if enabled

# Usage
regular_subsystem = ConditionalSubsystem(admin_mode=False)
admin_subsystem = ConditionalSubsystem(admin_mode=True)

regular_agent = Agent(name="Regular")
admin_agent = Agent(name="Admin")

regular_subsystem.connect_appropriate_tools(regular_agent)
admin_subsystem.connect_appropriate_tools(admin_agent)
```

## 📚 Built-in Subsystems

### Memory Subsystem

```python
from reagentic.subsystems.memory import FileBasedMemory, MemorySubsystemBase

# File-based memory with persistence
memory = FileBasedMemory("data.json")
agent = Agent(name="Assistant")
memory.connect_tools(agent)

# Available tools:
# - read_structure_t(): Read memory organization
# - modify_structure_t(): Update memory structure  
# - read_keys_t(): List all keys
# - read_by_key_t(): Read specific key
# - write_by_key_t(): Write key-value data
# - read_raw_t(): Read raw text storage
# - write_raw_t(): Replace raw text
# - append_raw_t(): Append to raw text
```

### Tool Categories

The memory subsystem organizes tools into logical categories:

- **Default tools**: Basic read/write operations for raw text
- **`key_based`**: Structured key-value storage operations
- **`dynamic_structure`**: Memory organization metadata tools

```python
# Connect specific categories
memory.connect_tools(agent, "key_based")        # Only key-value tools
memory.connect_tools(agent, "dynamic_structure") # Only structure tools
memory.connect_all_tools(agent)                 # All tools
```

## 🛠️ Best Practices

### 1. Clear Separation of Concerns

```python
# Good: Single responsibility
class DatabaseSubsystem(SubsystemBase):
    """Handles all database operations."""
    
    @SubsystemBase.subsystem_tool("read")
    def query_data(self, query: str) -> str:
        """Execute a read query."""
        pass
    
    @SubsystemBase.subsystem_tool("write")
    def update_data(self, table: str, data: dict) -> str:
        """Update data in a table."""
        pass

# Avoid: Mixed responsibilities
class MixedSubsystem(SubsystemBase):
    """Don't mix database, API, and file operations."""
    pass
```

### 2. Proper Error Handling

```python
@SubsystemBase.subsystem_tool()
def safe_operation(self, data: str) -> str:
    """Operation with proper error handling."""
    try:
        result = self._process_data(data)
        return f"Success: {result}"
    except ValueError as e:
        return f"Error: Invalid data - {str(e)}"
    except Exception as e:
        return f"Error: Unexpected error - {str(e)}"
```

### 3. Tool Documentation

```python
@SubsystemBase.subsystem_tool("data")
def process_data(self, input_data: str, format_type: str = "json") -> str:
    """
    Process input data in the specified format.
    
    Args:
        input_data: The data to process
        format_type: Output format ("json", "xml", "csv")
    
    Returns:
        Processed data in the requested format
        
    Raises:
        ValueError: If format_type is not supported
    """
    # Implementation
```

### 4. State Management

```python
class StatefulSubsystem(SubsystemBase):
    def __init__(self):
        super().__init__()
        self.state = {}
        self._lock = asyncio.Lock()  # For async safety
    
    @SubsystemBase.subsystem_tool()
    async def thread_safe_operation(self, key: str, value: str) -> str:
        """Thread-safe state modification."""
        async with self._lock:
            old_value = self.state.get(key)
            self.state[key] = value
            return f"Updated {key}: {old_value} -> {value}"
```

## 🔍 Troubleshooting

### Common Issues

1. **Tools not working**: Ensure you're calling `connect_tools()` or `connect_all_tools()`
2. **Method binding errors**: Make sure you're inheriting from `SubsystemBase`
3. **Tool not found**: Check that the tool category matches what you're connecting
4. **Type errors**: Ensure proper type hints on tool methods

### Debug Example

```python
# Debug tool registration
subsystem = MySubsystem()
print("Available tool types:", subsystem.get_all_tools().keys())
print("Default tools:", len(subsystem.get_tools()))
print("Custom tools:", len(subsystem.get_tools("custom")))

# Debug agent tools
agent = Agent(name="Debug")
print("Agent tools before:", len(agent.tools))
subsystem.connect_all_tools(agent)
print("Agent tools after:", len(agent.tools))
```

## 🎯 Summary

The Reagentic Subsystem Pattern provides:

- **Modularity**: Reusable components across agents
- **Organization**: Categorized tools for different use cases  
- **Flexibility**: Mix and match functionality as needed
- **Maintainability**: Clear separation of concerns
- **Type Safety**: Full typing support throughout
- **Event System**: Built-in event handling capabilities

This pattern enables building sophisticated AI applications with clean, maintainable, and reusable code architecture. 