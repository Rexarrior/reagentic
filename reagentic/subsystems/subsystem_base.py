from agents import Agent, function_tool
from typing import Union, List, Dict, Optional
import types as python_types


# Default tool type constant
DEFAULT_TOOL_TYPE = '__default__'


class SubsystemBase:
    def __init__(self):
        if not hasattr(self, 'tools'):
            self.tools = {DEFAULT_TOOL_TYPE: []}

    @classmethod
    def subsystem_tool(cls, tool_type: Union[str, List[str]] = DEFAULT_TOOL_TYPE):
        """
        Class decorator that applies function_tool decorator and adds the result to subsystem tools.
        Inherits from function_tool functionality.

        Args:
            tool_type: Category or categories for the tool. Can be a string or list of strings.
                      Defaults to DEFAULT_TOOL_TYPE.

        Usage:
            @SubsystemBase.subsystem_tool()
            def my_function(self, param: str) -> str:
                return "result"

            @SubsystemBase.subsystem_tool("simple")
            def function_with_simple_logic(self, query: str) -> str:
                return "simple_result"

            @SubsystemBase.subsystem_tool(["simple", "extended"])
            def function_with_complex_logic(self, endpoint: str) -> str:
                return "complex_result"
        """

        def decorator(func):
            # Store the original function instead of applying function_tool immediately
            # We'll apply function_tool when the method is bound to an instance
            
            # Initialize pending methods dict if not exists
            if not hasattr(cls, '_pending_methods'):
                cls._pending_methods = {}

            # Normalize tool_type to list
            if isinstance(tool_type, str):
                types = [tool_type]
            else:
                types = tool_type

            # Add method to each specified type
            for t in types:
                if t not in cls._pending_methods:
                    cls._pending_methods[t] = []
                cls._pending_methods[t].append(func)

            return func

        return decorator

    def __init_subclass__(cls, **kwargs):
        """Initialize subclass and collect tools from decorated methods"""
        super().__init_subclass__(**kwargs)

        # Collect all methods from this class and parent classes
        if not hasattr(cls, '_class_methods'):
            cls._class_methods = {DEFAULT_TOOL_TYPE: []}

        # Add pending methods from this class
        if hasattr(cls, '_pending_methods'):
            for tool_type, methods in cls._pending_methods.items():
                if tool_type not in cls._class_methods:
                    cls._class_methods[tool_type] = []
                cls._class_methods[tool_type].extend(methods)

    def get_tools(self, tool_type: Union[str, List[str]] = DEFAULT_TOOL_TYPE) -> List:
        """
        Return the list of tools for this subsystem filtered by tool type.
        Creates bound method tools from class methods.

        Args:
            tool_type: The type or types of tools to retrieve. Can be a single string or list of strings.
                      Defaults to DEFAULT_TOOL_TYPE.

        Returns:
            List of tools for the specified type(s).
        """
        # Normalize tool_type to list
        if isinstance(tool_type, str):
            types = [tool_type]
        else:
            types = tool_type

        all_tools = []
        class_methods_dict = getattr(self.__class__, '_class_methods', {})

        # Collect tools from all specified types
        for t in types:
            # Get class-level methods for this type and bind them to this instance
            class_methods = class_methods_dict.get(t, [])
            for method in class_methods:
                # Create a bound method
                bound_method = python_types.MethodType(method, self)
                # Apply function_tool to the bound method
                tool_func = function_tool(bound_method)
                all_tools.append(tool_func)

            # Get instance tools for this type
            instance_tools = self.tools.get(t, [])
            for tool in instance_tools:
                if tool not in all_tools:
                    all_tools.append(tool)

        return all_tools

    def get_all_tools(self) -> Dict[str, List]:
        """
        Return all tools organized by type.

        Returns:
            Dictionary mapping tool types to lists of tools.
        """
        all_tools = {}

        # Get class-level methods
        class_methods = getattr(self.__class__, '_class_methods', {})

        # Combine all tool types from class and instance
        all_types = set(class_methods.keys()) | set(self.tools.keys())

        for tool_type in all_types:
            all_tools[tool_type] = self.get_tools(tool_type)

        return all_tools

    def connect(self, agent: Agent):
        self.connect_tools(agent)
        self.connect_hooks(agent)

    def connect_tools(self, agent: Agent, tool_type: Union[str, List[str]] = DEFAULT_TOOL_TYPE):
        """
        Connect tools of specified type(s) to the agent.

        Args:
            agent: The agent to connect tools to.
            tool_type: The type or types of tools to connect. Can be a single string or list of strings.
                      Defaults to DEFAULT_TOOL_TYPE.
        """
        agent.tools.extend(self.get_tools(tool_type))

    def connect_all_tools(self, agent: Agent):
        """
        Connect all tools of all types to the agent.

        Args:
            agent: The agent to connect tools to.
        """
        all_tools = self.get_all_tools()
        for tools_list in all_tools.values():
            agent.tools.extend(tools_list)

    def connect_hooks(self, agent):
        pass
