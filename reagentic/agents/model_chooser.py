import os
import argparse
import importlib
from typing import List
from .common import default_paid_provider, default_free_provider
from ..providers.common import ModelInfo
from agents import Agent, Runner
from pydantic import BaseModel
from datetime import datetime
from ..logging import get_logger, agent_context, log_context

logger = get_logger('agents.model_chooser')

STRICT_FORMAT = 'YOU MUST TO FOLLOW THE OUTPUT FORMAT VERY STRICTLY!YOU SHOULD PROVIDE ONLY REQUESTED FIELDS! NOT EXTRE FIELDS ALLOWED! ALL REQUESTED FIELDS ARE REQUIRED!'
BASIC_INSTRUCTION = 'You an LLM export. You should learn available models and choose one by users requirements.'
SYSTEM_INSTRUCTION_PAID = f"""
{BASIC_INSTRUCTION}
{STRICT_FORMAT}
"""
SYSTEM_INSTRUCTION_FREE = f"""
{BASIC_INSTRUCTION}
IMPORTANT!!!Your answer should consist of one word - variable name that user should use. No description. No parameters, just variable name IMPORTANT!!!
Ignore description request from user. Just follow user's requirements, peek a model, and return only variable name IMPORTANT!!!
"""


class ModelChoose(BaseModel):
    model_variable_name: str
    'Variable that should be imported from provider.available_models to use choosed model'

    reason: str
    'Why user should use this model'

    model_info: ModelInfo


def read_available_models(provider_name: str):
    fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'providers', provider_name, 'available_models.py')
    if not os.path.exists(fpath):
        raise ValueError("Provider not found or don't support available_models list")
    with open(fpath, 'rt', encoding='utf-8') as f:
        return f.read()


def build_agent_free():
    with agent_context('model_chooser_free'):
        logger.log_agent_start('model_chooser_free', mode='free')

        agent = Agent(
            name='LLM Model Expert',
            instructions=SYSTEM_INSTRUCTION_FREE,
            model=default_free_provider.get_openai_model(),
            output_type=str,
        )
        return agent


def build_agent_paid():
    with agent_context('model_chooser_paid'):
        logger.log_agent_start('model_chooser_paid', mode='paid')

        agent = Agent(
            name='LLM Model Expert',
            instructions=SYSTEM_INSTRUCTION_PAID,
            model=default_paid_provider.get_openai_model(),
            output_type=ModelChoose,
        )
        return agent


def choose_model(provider_name: str, request: str, is_paid: bool) -> ModelChoose | str:
    """
    Choose an model from provider available_models list, base on user request and models properties.
    If paid, use some power paid model and return ModelChoose, if free, use some free model and just return variable name.
    """
    with log_context(provider_name=provider_name, operation='model_selection', is_paid=is_paid):
        logger.info(f'Starting model selection for {provider_name}', request=request, is_paid=is_paid)

        try:
            models_file_content = read_available_models(provider_name)

            if is_paid:
                agent = build_agent_paid()
            else:
                agent = build_agent_free()

            prompt = f'{request}\n available models: {models_file_content}\nNow: {datetime.now()}\n{STRICT_FORMAT}'

            logger.debug('Running model selection agent', prompt_length=len(prompt))

            choose = Runner.run_sync(agent, prompt)

            logger.info(
                'Model selection completed',
                selected_model=choose.final_output
                if not is_paid
                else getattr(choose.final_output, 'model_variable_name', None),
                success=True,
            )

            return choose.final_output

        except Exception as e:
            logger.log_error(e, 'model_selection')
            raise


def main():
    parser = argparse.ArgumentParser(description='Choose a model based on provider and user request.')
    parser.add_argument('--provider', type=str, required=True, help='Name of the provider')
    parser.add_argument('--request', type=str, required=True, help='User request - requirements to choose model')
    parser.add_argument('--paid', type=str, default=False, required=False, help='Allow to use non-free model to choose')

    args, unknown = parser.parse_known_args()

    with log_context(operation='cli_model_selection'):
        logger.log_user_action('model_selection_cli', extra={'provider': args.provider, 'paid': args.paid})

        chosen_model = choose_model(args.provider, args.request, args.paid)
        if args.paid:
            print(f'Chosen Model: {chosen_model.model_info.str_identifier}')
            print(f'Reason: {chosen_model.reason}')
        else:
            print(f'Chosen Model: {chosen_model}')
