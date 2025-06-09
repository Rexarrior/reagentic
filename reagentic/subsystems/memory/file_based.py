from .memory_base import MemorySubsystemBase, MemoryEventType, UsualMemory
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any


class FileBasedMemory(MemorySubsystemBase):
    """
    File-based memory subsystem that automatically persists all memory changes to a JSON file.

    Inherits all functionality from MemorySubsystemBase and adds automatic file persistence.
    Every memory modification (structure, key-based, or raw text) is immediately saved to disk.

    Features:
    - Automatic saving on every memory change via event system
    - JSON file format for human-readable storage
    - Automatic loading from file on initialization
    """

    def __init__(self, file_path: str = 'memory.json', auto_load: bool = True):
        """
        Initialize file-based memory with automatic persistence.

        Args:
            file_path: Path to the JSON file for storing memory data
            auto_load: Whether to automatically load existing data from file on initialization
        """
        super().__init__()
        self.file_path = Path(file_path)

        # Register event handlers for automatic saving
        self._register_auto_save_handlers()

        # Load existing data if requested and file exists
        if auto_load and self.file_path.exists():
            self.load_from_file()

    def _register_auto_save_handlers(self) -> None:
        """Register event handlers to automatically save on any memory change."""
        auto_save_handler = self._auto_save_handler

        # Register for all memory change events
        self.add_event_handler(MemoryEventType.STRUCTURE_MODIFIED, auto_save_handler)
        self.add_event_handler(MemoryEventType.KEY_WRITTEN, auto_save_handler)
        self.add_event_handler(MemoryEventType.RAW_WRITTEN, auto_save_handler)
        self.add_event_handler(MemoryEventType.RAW_APPENDED, auto_save_handler)

    def _auto_save_handler(self, event_data: Dict[str, Any]) -> None:
        """
        Event handler that automatically saves memory to file on any change.

        Args:
            event_data: Event information (not used, but required by event system)
        """
        try:
            self.save_to_file()
        except Exception as e:
            print(f'Auto-save failed: {e}')

    def save_to_file(self) -> None:
        """
        Save current memory state to JSON file.

        Creates the directory if it doesn't exist and writes the memory data
        in a human-readable JSON format.

        Raises:
            OSError: If file cannot be written
            json.JSONEncodeError: If memory data cannot be serialized
        """
        try:
            # Create directory if it doesn't exist
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save memory data as formatted JSON
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.db.model_dump(), f, indent=2, ensure_ascii=False)

        except Exception as e:
            raise OSError(f'Failed to save memory to {self.file_path}: {e}')

    def load_from_file(self) -> None:
        """
        Load memory state from JSON file.

        Replaces current memory state with data from the file.

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
            ValueError: If file data is not valid memory format
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate and load the data
            self.db = UsualMemory.model_validate(data)

        except FileNotFoundError:
            raise FileNotFoundError(f'Memory file not found: {self.file_path}')
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f'Invalid JSON in memory file {self.file_path}: {e}')
        except Exception as e:
            raise ValueError(f'Failed to load memory from {self.file_path}: {e}')

    def file_exists(self) -> bool:
        """Check if the memory file exists."""
        return self.file_path.exists()

    def get_file_path(self) -> str:
        """Get the current file path as string."""
        return str(self.file_path)

    def get_file_size(self) -> Optional[int]:
        """
        Get the size of the memory file in bytes.

        Returns:
            File size in bytes, or None if file doesn't exist
        """
        if self.file_exists():
            return self.file_path.stat().st_size
        return None

    def backup_file(self, backup_suffix: str = '.backup') -> str:
        """
        Create a backup copy of the memory file.

        Args:
            backup_suffix: Suffix to add to the backup file name

        Returns:
            Path to the backup file

        Raises:
            FileNotFoundError: If original file doesn't exist
            OSError: If backup cannot be created
        """
        if not self.file_exists():
            raise FileNotFoundError(f'Cannot backup non-existent file: {self.file_path}')

        backup_path = self.file_path.with_suffix(self.file_path.suffix + backup_suffix)

        try:
            # Copy file content
            with open(self.file_path, 'r', encoding='utf-8') as src:
                with open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())

            return str(backup_path)

        except Exception as e:
            raise OSError(f'Failed to create backup: {e}')

    def __repr__(self) -> str:
        """String representation of FileBasedMemory."""
        return f"FileBasedMemory(file_path='{self.file_path}', exists={self.file_exists()})"
