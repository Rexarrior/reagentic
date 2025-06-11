from agents import Agent, Runner
import reagentic.providers.openrouter as openrouter
from reagentic.subsystems.memory import FileBasedMemory


HAIKU_PROMPT = 'Write a haiku about recursion in programming.'
HAIKU_SYSTEM_PROMPT = 'You are a helpful assistant for write haiku'
HAIKU_AGENT_NAME = 'Haiku Assistant'

provider = openrouter.OpenrouterProvider(openrouter.DEEPSEEK_CHAT_V3_0324)
# provider = openrouter.OpenrouterProvider(openrouter.GPT_4_1)


def run_agent_memory_through_prompt():
    # Simplest way - extend context.
    memory = FileBasedMemory()
    agent = Agent(
        name=HAIKU_AGENT_NAME, instructions=HAIKU_SYSTEM_PROMPT, model=provider.get_openai_model(), memory=memory
    )
    for i in range(3):
        print(f'Iteration {i}')
        result = Runner.run_sync(agent, HAIKU_PROMPT + f'\nYour previous haiku: {memory.read_raw()}')
        result = result.final_output
        memory.append_raw(result)
        print(result)


def run_agent_memory_through_tools():
    memory = FileBasedMemory()
    agent = Agent(
        name=HAIKU_AGENT_NAME,
        instructions=(
            HAIKU_SYSTEM_PROMPT
            + '\nYou must save information about haiku that your wrote to memory using append_raw_t tool'
            + '\nYou must read information about haiku that your wrote earlier from memory using read_raw_t tool'
        ),
        model=provider.get_openai_model(),
    )
    # there we connect default tools to agent - read, write and append  text to raw memory
    memory.connect_tools(agent)
    for i in range(3):
        print(f'Iteration {i}')
        result = Runner.run_sync(agent, HAIKU_PROMPT)
        result = result.final_output
        print(result)
    print(f'Final:')
    result = Runner.run_sync(agent, "Return your memory content")
    result = result.final_output
    print(result)


if __name__ == '__main__':
    print('Simple agent')
    # run_agent_memory_through_prompt()
    print('\n' * 3)
    run_agent_memory_through_tools()
