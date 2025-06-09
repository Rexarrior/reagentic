from .common import default_free_provider
from agents import Agent, Runner


def build_agent_free():
    agent = Agent(
        name='Hello World',
        instructions='You just say hello world',
        model=default_free_provider.get_openai_model(),
        output_type=str,
    )
    return agent


def main():
    print(Runner.run_sync(build_agent_free(), 'say hello world').final_output)
