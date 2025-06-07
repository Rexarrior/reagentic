import os
import argparse
import importlib
from typing import List
from .common import default_paid_provider, default_free_provider
from ..providers.common import ModelInfo
from agents import Agent, Runner
from pydantic import BaseModel


STRICT_FORMAT = 'YOU MUST TO FOLLOW THE OUTPUT FORMAT VERY STRICTLY!YOU SHOULD PROVIDE ONLY REQUESTED FIELDS! NOT EXTRE FIELDS ALLOWED! ALL REQUESTED FIELDS ARE REQUIRED!'
BASIC_INSTRUCTION = 'You an LLM export. You should learn available models and choose one by users requirements.'
SYSTEM_INSTRUCTION_PAID=f'''
{BASIC_INSTRUCTION}
{STRICT_FORMAT}
'''
SYSTEM_INSTRUCTION_FREE=f'''
{BASIC_INSTRUCTION}
IMPORTANT!!!Your answer should consist of one word - variable name that user should use. No description. No parameters, just variable name.
Ignore description request from user. Just follow user's requirements, peek a model, and return only variable name IMPORTANT!!!
'''
class ModelChoose(BaseModel):
    model_variable_name: str
    'Variable that should be imported from provider.available_models to use choosed model'

    reason: str
    'Why user should use this model'

    model_info: ModelInfo

def read_available_models(provider_name: str):
    fpath = os.path.join(
        os.path.dirname(
            os.path.dirname(__file__)
        ),
        "providers",
        provider_name,
        "available_models.py"
    )
    if not os.path.exists(fpath):
        raise ValueError("Provider not found or don't support available_models list")
    with open(fpath, "rt", encoding="utf-8") as f:
        return f.read()



def build_agent_free():
    agent = Agent(
        name="LLM Model Expert",
        instructions=SYSTEM_INSTRUCTION_FREE,
        model=default_free_provider.get_openai_model(),
        output_type=str
        )
    return agent


def build_agent_paid():
    agent = Agent(
        name="LLM Model Expert",
        instructions=SYSTEM_INSTRUCTION_PAID,
        model=default_paid_provider.get_openai_model(),
        output_type=ModelChoose
        )
    return agent


def choose_model(provider_name: str, request: str, is_paid: bool) -> ModelChoose | str:
    '''
    Choose an model from provider available_models list, base on user request and models properties. 
    If paid, use some power paid model and return ModelChoose, if free, use some free model and just return variable name.
    '''
    models_file_content = read_available_models(provider_name)
    if is_paid:
        agent = build_agent_paid()
    else:
        agent = build_agent_free()
    prompt = f"{request}\n available models: {models_file_content}\n{STRICT_FORMAT}"
    choose = Runner.run_sync(agent, prompt)
    return choose.final_output


def main():

    parser = argparse.ArgumentParser(description='Choose a model based on provider and user request.')
    parser.add_argument('--provider', type=str, required=True, help='Name of the provider')
    parser.add_argument('--request', type=str, required=True, help='User request - requirements to choose model')
    parser.add_argument('--paid', type=str, default=False, required=False, help='Allow to use non-free model to choose')

    args , unknown = parser.parse_known_args()

    chosen_model = choose_model(args.provider, args.request, args.paid)
    if args.paid:
        print(f"Chosen Model: {chosen_model.model_info.str_identifier}")
        print(f"Reason: {chosen_model.reason}")
    else:
        print(f"Chosen Model: {chosen_model}")