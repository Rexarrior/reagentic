from .models import AllOpenRouterModels
from .client import get_all_openrouter_models
from ...agents.changelog_agent import get_git_commit_message
from ...tools import git_tools
import os

models_relative_path_from_root = os.path.join('reagentic', 'providers', 'openrouter', 'available_models.py')


def identifier_to_var_name(str_identifier: str):
    splitted = str_identifier.split('/')[1]
    formatted = splitted.replace(':', '_').replace('/', '_').replace('-', '_').replace('.', '_').upper()
    if str.isdigit(formatted[0]):
        formatted = f'D{formatted}'
    return formatted


def rewrite_code_available_models(path_to_file: str, models: AllOpenRouterModels):
    # Read the existing file content to keep the import statement
    with open(path_to_file, 'rt', encoding='utf-8') as f:
        existing_content = f.read()

    import_statement = ''
    for line in existing_content.splitlines():
        if line.strip().startswith('from'):
            import_statement = line
            break
    all_models = sorted(models.all_models, key=lambda x: x.str_identifier)
    model_definitions = []
    model_names = []
    free_model_names = []
    all_models_dict = {}
    for model in all_models:
        model_name = identifier_to_var_name(model.str_identifier)
        model_definitions.append(f"""
{model_name} = ModelInfo(
    str_identifier="{model.str_identifier}",
    price_in={model.price_in},
    price_out={model.price_out},
    creator="{model.creator}",
    description='''{model.description},
    created={model.created}'''
)
""")
        model_names.append(model_name)
        all_models_dict[model_name] = model_name
        if model.price_in == 0 and model.price_out == 0:
            free_model_names.append(model_name)

    all_models_list = 'ALL_MODELS = [\n'
    for name in model_names:
        all_models_list += f'    {name},\n'
    all_models_list += ']'

    free_models_list = 'FREE_MODELS = [\n'
    for name in free_model_names:
        free_models_list += f'    {name},\n'
    free_models_list += ']'

    all_models_dict_str = 'ALL_MODELS_DICT = {\n'
    for name in model_names:
        all_models_dict_str += f'    "{name}": {name},\n'
    all_models_dict_str += '}'

    new_content = f'{import_statement}\n\n{"".join(model_definitions)}\n{all_models_list}\n{free_models_list}\n{all_models_dict_str}\n'

    # Write the new content to the file
    with open(path_to_file, 'wt', encoding='utf-8') as f:
        f.write(new_content)


def main():
    models = get_all_openrouter_models()
    rewrite_code_available_models(models_relative_path_from_root, models)
    git_tools.git_add(models_relative_path_from_root)
    commit_message = get_git_commit_message()
    git_tools.git_commit(commit_message)
