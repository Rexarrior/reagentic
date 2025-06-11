from .common import default_free_provider
from agents import Agent, Runner
from ..logging import get_logger, agent_context, log_context

logger = get_logger('agents.hello_world')


def build_agent_free():
    with agent_context('hello_world_agent'):
        logger.log_agent_start('hello_world_agent', task='greeting')

        agent = Agent(
            name='Hello World',
            instructions='You just say hello world',
            model=default_free_provider.get_openai_model(),
            output_type=str,
        )
        return agent


def main():
    with log_context(operation='hello_world_demo'):
        logger.info('Starting hello world demo')

        try:
            agent = build_agent_free()
            result = Runner.run_sync(agent, 'say hello world')

            logger.log_agent_end('hello_world_agent', success=True, response_length=len(result.final_output))

            print(result.final_output)

            logger.info('Hello world demo completed successfully')

        except Exception as e:
            logger.log_agent_end('hello_world_agent', success=False)
            logger.log_error(e, 'hello_world_execution')
            raise
