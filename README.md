# Reagentic Framework
This is a home-made openai-agents based framework for simple agent coding. I am writing it for myself, so as they say, no guarantees.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Providers
Now the is only one provider - Openrouter. It has automatic refiller, so, in theory, there will be always actual models.

## Settings
The framework rely on env var for providers key. It can be placed in environment, local .env (in project), .env near .venv

## Usage

Simple hello world

```python
from agents import Agent, Runner
import reagentic.providers.openrouter as openrouter   

provider = openrouter.OpenrouterProvider(openrouter.DEEPSEEK_CHAT_V3_0324)
agent = Agent(name="Assistant", instructions="You are a helpful assistant", model=provider.get_openai_model())

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
```

or with auto choose of model:


```python
from agents import Agent, Runner
from reagentic.providers import openrouter

provider = openrouter.auto()
print(f"Will use {provider.model} model")
agent = Agent(name="Assistant", instructions="You are a helpful assistant", model=provider.get_openai_model())

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

```

## Call some predefined script:

### refill openrouter models

```bash
python3 -m reagentic --script openrouter_refiller
```

### choose some model (minimalistic, but free)

```bash
python3 -m reagentic --script model_chooser --provider openrouter --request "latest deepseek model"
```

Example output:  
`Chosen Model: DEEPSEEK_CHAT`

### choose some model. More smart, but paid

```bash
python -m reagentic --script model_chooser --provider openrouter --request "latest deepseek model" --paid true
```

Example output:
```text
Chosen Model: deepseek/deepseek-chat-v3-0324
Reason: DeepSeek Chat V3 0324 is the latest iteration of DeepSeek models, offering a massive 685B-parameter MoE 
architecture with outstanding instruction following and coding capabilities. This model succeeds the previous DeepSeek V3 and delivers strong performance across a variety of tasks, making it ideal for users seeking cutting-edge open-source DeepSeek models.
```
