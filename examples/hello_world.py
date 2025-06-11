from agents import Agent, Runner
import reagentic.providers.openrouter as openrouter

provider = openrouter.OpenrouterProvider(openrouter.DEEPSEEK_CHAT_V3_0324)
agent = Agent(name='Assistant', instructions='You are a helpful assistant', model=provider.get_openai_model())

result = Runner.run_sync(agent, 'Write a haiku about recursion in programming.')
print(result.final_output)
