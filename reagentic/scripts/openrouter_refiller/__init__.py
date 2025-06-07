
from .models import AllOpenRouterModels
from .client import get_all_openrouter_models
from ..changelog_agent import get_git_commit_message
from ...tools import git_tools
import os

models_relative_path_from_root = os.path.json(
"reagentic", "providers", "openrouter", "available_models.py"
)

def rewrite_code_available_models(path_to_file: str, models=AllOpenRouterModels):
    # Read the existing file content to keep the import statement
    with open(path_to_file, 'r') as f:
        existing_content = f.read()
    
    import_statement = ""
    for line in existing_content.splitlines():
        if line.strip().startswith("from"):
            import_statement = line
            break

    model_definitions = []
    model_names = []
    for model in models.models:
        model_name = model.id.replace('/', '_').replace('-', '_').upper()
        model_definitions.append(f"""
{model_name} = ModelInfo(
    str_identifier="{model.id}",
    price_in={model.pricing.prompt},
    price_out={model.pricing.completion},
    description='''{model.description}'''
)
""")
        model_names.append(model_name)

    all_models_list = "ALL_MODELS = [\n"
    for name in model_names:
        all_models_list += f"    {name},\n"
    all_models_list += "]"

    new_content = f"{import_statement}\n\n{''.join(model_definitions)}\n{all_models_list}\n"

    # Write the new content to the file
    with open(path_to_file, 'w') as f:
        f.write(new_content)

def main():
    models = get_all_openrouter_models()
    rewrite_code_available_models(models_relative_path_from_root, models)
    git_tools.git_add(models_relative_path_from_root)
    commit_message = get_git_commit_message()
    git_tools.git_commit(commit_message)

    