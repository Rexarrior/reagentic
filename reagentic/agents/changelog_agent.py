from agents import Agent, Runner
from ..providers import openrouter
from agents import Agent
from ..tools import git_tools
from .common import default_provider

SYSTEM_INSTRUCTION = """
'You are a Technical Writer, expert in changelog documentation.
Your goal of life is to write a documentation the best world even meet.
You always write your documentation base on git commitdiff and always very strong\
and clearly formulate your changelog and commit messages
"""


def build_agent():
    agent = Agent(
        name='ChangelogExpert',
        instructions=SYSTEM_INSTRUCTION,
        model=default_provider.get_openai_model(),
        output_type=str,
    )
    return agent


def build_prompt_git_commit():
    git_status = git_tools.call_git_status()
    git_diff = git_tools.call_git_diff_staged()
    return f'Please, write git commit message. \nGit status: {git_status}\nGit diff: {git_diff}\n'


def get_git_commit_message():
    agent = build_agent()
    prompt = build_prompt_git_commit()
    message = Runner.run_sync(agent, prompt)
    return message.final_output


def main():
    message = get_git_commit_message()
    print(message)
