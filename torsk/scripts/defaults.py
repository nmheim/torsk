#coding: future_fstrings
import pathlib
from shutil import copyfile
import click


_module_dir = pathlib.Path(__file__).absolute().parent


@click.command('params',
    short_help='Creates a default params.json for different models')
@click.argument('model_name', type=click.Choice(['mackey', 'kuro']))
@click.option('--outfile', '-o', type=pathlib.Path,
    default=pathlib.Path('params.json'),
    help='Path to save the params.json file')
def cli(model_name, outfile):
    """Get the default params.json for the given MODEL_NAME and save if to PATH"""
    if model_name == 'mackey':
        params_json = _module_dir / 'mackey_default_params.json'
    elif model_name == 'kuro':
        params_json = _module_dir / 'kuro_default_params.json'
    else:
        raise ValueError(f"No default params for model: {model_name}")
    copyfile(params_json, outfile)
