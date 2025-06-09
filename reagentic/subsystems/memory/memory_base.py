from ..subsystem_base import SubsystemBase
from agents import function_tool
from pydantic import BaseModel
from typing import Callable, List, Any, Dict
from enum import Enum

class MemoryEventType(Enum):
    """Types of memory change events"""
    STRUCTURE_MODIFIED = "structure_modified"
    KEY_WRITTEN = "key_written"
    RAW_WRITTEN = "raw_written"
    RAW_APPENDED = "raw_appended"

class UsualMemory(BaseModel):
    """
    Multi-layered memory system with three distinct storage types:
    
    1. Structure Definition: Self-describing metadata about how the memory is organized.
       This can be dynamically updated via tools to reflect changing memory usage patterns.
    
    2. Key-Based Storage: Structured dictionary storage for named data items.
       Ideal for storing specific facts, preferences, or categorized information.
    
    3. Raw Text Storage: Unstructured text storage for notes, logs, or free-form content.
       Supports both complete replacement and incremental appending.
    """
    
    structure: str = "This memory system contains three storage types: (1) Structure definition - dynamic metadata describing memory organization, updatable via tools; (2) Key-based storage - structured dictionary for named data items; (3) Raw text storage - unstructured text for notes and free-form content supporting both replacement and appending operations."
    """
    Dynamic structure definition describing how this memory system is organized.
    This metadata can be updated via tools to reflect evolving memory usage patterns
    and organizational strategies. Serves as self-documentation for the memory system.
    """
    
    key_based: dict[str, str] = {}
    """
    Structured key-value storage for named data items. Ideal for storing specific facts,
    user preferences, important identifiers, or any categorized information that needs
    to be quickly retrievable by name.
    """
    
    raw: str = ""
    """
    Unstructured text storage for free-form content such as notes, conversation logs,
    observations, or any text that doesn't fit into the structured key-based system.
    Supports both complete content replacement and incremental appending.
    """
    
class MemorySubsystemBase(SubsystemBase):
    """
    Base memory subsystem providing comprehensive storage capabilities through three
    distinct storage mechanisms:
    
    - Structure Definition: Self-documenting metadata that can be dynamically updated
    - Key-Based Storage: Dictionary-style storage for structured data
    - Raw Text Storage: Free-form text storage with append and replace operations
    
    All storage types are accessible through agent tools, allowing intelligent
    memory organization and retrieval strategies.
    
    Event System:
    The subsystem fires events when memory is modified, allowing custom logic
    to be triggered on memory changes.
    """
    
    def __init__(self):
        super().__init__()
        self.db = UsualMemory()
        self._event_handlers: Dict[MemoryEventType, List[Callable]] = {
            event_type: [] for event_type in MemoryEventType
        }

    def add_event_handler(self, event_type: MemoryEventType, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register an event handler for memory change events.
        
        Args:
            event_type: The type of memory event to listen for
            handler: Callback function that receives event data as a dictionary
            
        Example:
            def on_structure_change(event_data):
                print(f"Structure changed: {event_data['old_value']} -> {event_data['new_value']}")
            
            memory.add_event_handler(MemoryEventType.STRUCTURE_MODIFIED, on_structure_change)
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def remove_event_handler(self, event_type: MemoryEventType, handler: Callable) -> None:
        """Remove an event handler."""
        if event_type in self._event_handlers and handler in self._event_handlers[event_type]:
            self._event_handlers[event_type].remove(handler)

    def _trigger_event(self, event_type: MemoryEventType, event_data: Dict[str, Any]) -> None:
        """
        Trigger all handlers for a specific event type.
        
        Args:
            event_type: The type of event that occurred
            event_data: Dictionary containing event details
        """
        for handler in self._event_handlers.get(event_type, []):
            try:
                handler(event_data)
            except Exception as e:
                # Log error but don't stop execution
                print(f"Error in memory event handler: {e}")

    @SubsystemBase.subsystem_tool
    def read_structure(self) -> str:
        """
        Read the current memory structure definition.
        
        Returns the self-describing metadata about how this memory system is organized,
        including information about all three storage types and their intended usage.
        """
        return self.db.structure

    @SubsystemBase.subsystem_tool  
    def modify_structure(self, new_structure: str) -> str:
        """
        Update the memory structure definition with new organizational metadata.
        
        This tool allows dynamic evolution of the memory system's self-documentation,
        enabling the agent to adapt its memory organization strategy over time based
        on usage patterns and requirements.
        
        Args:
            new_structure: New description of how the memory system should be organized
            
        Returns:
            Confirmation message with the updated structure
        """
        old_structure = self.db.structure
        self.db.structure = new_structure
        
        # Trigger event
        self._trigger_event(MemoryEventType.STRUCTURE_MODIFIED, {
            'old_value': old_structure,
            'new_value': new_structure,
            'timestamp': self._get_timestamp()
        })
        
        return f"Memory structure definition updated to: {new_structure}"

    @SubsystemBase.subsystem_tool
    def read_keys(self) -> list[str]:
        """
        List all available keys in the structured key-based storage.
        
        Returns all key names from the dictionary storage, allowing discovery
        of what structured data is currently stored in memory.
        """
        return list(self.db.key_based.keys())

    @SubsystemBase.subsystem_tool
    def read_by_key(self, key: str) -> str:
        """
        Retrieve value from key-based storage by key name.
        
        Accesses the structured dictionary storage to retrieve specific named data items.
        
        Args:
            key: The name/identifier of the data item to retrieve
            
        Returns:
            The stored value or an error message if key doesn't exist
        """
        if key in self.db.key_based:
            return self.db.key_based[key]
        else:
            return f"Key '{key}' not found in key-based storage"

    @SubsystemBase.subsystem_tool
    def write_by_key(self, key: str, value: str) -> str:
        """
        Store or update data in the structured key-based storage.
        
        Creates new entries or replaces existing ones in the dictionary storage.
        This is ideal for storing facts, preferences, or any categorized information.
        
        Args:
            key: The name/identifier for the data item
            value: The content to store
            
        Returns:
            Confirmation message indicating whether key was created or updated
        """
        old_value = self.db.key_based.get(key)
        action = "updated" if key in self.db.key_based else "created"
        self.db.key_based[key] = value
        
        # Trigger event
        self._trigger_event(MemoryEventType.KEY_WRITTEN, {
            'key': key,
            'old_value': old_value,
            'new_value': value,
            'action': action,
            'timestamp': self._get_timestamp()
        })
        
        return f"Key '{key}' {action} in key-based storage with value: {value}"

    @SubsystemBase.subsystem_tool
    def read_raw(self) -> str:
        """
        Read the complete content of the raw text storage.
        
        Returns all unstructured text content, such as notes, logs, or free-form
        observations that don't fit into the structured key-based system.
        """
        return self.db.raw if self.db.raw else "Raw text storage is empty"

    @SubsystemBase.subsystem_tool
    def write_raw(self, content: str) -> str:
        """
        Replace the entire content of the raw text storage.
        
        Completely overwrites the unstructured text storage with new content.
        Use this for replacing notes entirely or when starting fresh.
        
        Args:
            content: New text content to store
            
        Returns:
            Confirmation with character count of new content
        """
        old_content = self.db.raw
        self.db.raw = content
        
        # Trigger event
        self._trigger_event(MemoryEventType.RAW_WRITTEN, {
            'old_content': old_content,
            'new_content': content,
            'old_length': len(old_content),
            'new_length': len(content),
            'timestamp': self._get_timestamp()
        })
        
        return f"Raw text storage completely replaced. New content length: {len(content)} characters"

    @SubsystemBase.subsystem_tool
    def append_raw(self, content: str) -> str:
        """
        Append new content to the raw text storage.
        
        Adds new text to the end of existing unstructured content, automatically
        handling line breaks. Ideal for adding new notes, observations, or log entries
        without losing existing content.
        
        Args:
            content: Text content to append
            
        Returns:
            Confirmation with total character count after appending
        """
        old_content = self.db.raw
        if self.db.raw and not self.db.raw.endswith('\n'):
            self.db.raw += '\n'
        self.db.raw += content
        
        # Trigger event
        self._trigger_event(MemoryEventType.RAW_APPENDED, {
            'old_content': old_content,
            'appended_content': content,
            'new_content': self.db.raw,
            'old_length': len(old_content),
            'new_length': len(self.db.raw),
            'timestamp': self._get_timestamp()
        })
        
        return f"Content appended to raw text storage. Total length: {len(self.db.raw)} characters"

    def _get_timestamp(self) -> str:
        """Get current timestamp for event data"""
        from datetime import datetime
        return datetime.now().isoformat()

