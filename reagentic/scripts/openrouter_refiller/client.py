from ...providers.base_provider import BaseProvider
from .models import AllOpenRouterModels
from ...providers.common import ModelInfo

import requests
import tenacity
import logging

logger = logging.getLogger(__name__)

@tenacity.retry(
    wait=tenacity.wait_fixed(2),
    stop=tenacity.stop_after_attempt(5),
    before_sleep=lambda retry_state: logger.info(f"Retrying get_all_models: attempt {retry_state.attempt_number}...")
)
def request_get_all_models() -> dict:
    """
    Fetches all available models from the OpenRouter API with retries.
    """
    logger.info("Fetching all models from OpenRouter API...")
    try:
        response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={}, # Add headers if needed, e.g., Authorization
                timeout=10 # Add a timeout
            )
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        logger.info("Successfully fetched models from OpenRouter API.")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request to OpenRouter API failed: {e}")
        raise # Re-raise the exception to trigger tenacity retry

def convert_to_model_infos(response: dict) -> AllOpenRouterModels:
    """
    Converts the OpenRouter API response to an AllOpenRouterModels object.
    """
    model_list = []
    for model_data in response.get('data', []):
        try:
            model_info = ModelInfo(
                str_identifier=model_data.get('id'),
                price_in=model_data.get('pricing', {}).get('prompt', 0.0),
                price_out=model_data.get('pricing', {}).get('completion', 0.0),
                description=model_data.get('description', '')
            )
            model_list.append(model_info)
        except Exception as e:
            logger.error(f"Failed to parse model data: {model_data}. Error: {e}")
            # Continue processing other models even if one fails

    return AllOpenRouterModels(all_models=model_list)

def get_all_openrouter_models():
    api_response = request_get_all_models()
    all_models = convert_to_model_infos(api_response)
    return all_models