from agents import Agent, function_tool


class SubsystemBase:
    def __init__(self):
        if not hasattr(self, 'tools'):
            self.tools = []

    def subsystem_tool(self, func):
        """
        Decorator that applies function_tool decorator and adds the result to subsystem tools.
        Inherits from function_tool functionality.

        Usage:
            @subsystem.subsystem_tool
            def my_function(param: str) -> str:
                return "result"
        """
        # Apply the function_tool decorator
        tool_func = function_tool(func)

        # Add the tool to the subsystem's tools list
        self.tools.append(tool_func)

        return tool_func

    def get_tools(self):
        """Return the list of tools for this subsystem"""
        return self.tools

    def connect(self, agent: Agent):
        self.connect_tools(agent)
        self.connect_hooks(agent)

    def connect_tools(self, agent: Agent):
        agent.tools.extend(self.get_tools())

    def connect_hooks(self, agent):
        pass
