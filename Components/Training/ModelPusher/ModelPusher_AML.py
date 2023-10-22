import ast
from pathlib import Path
from typing import Union, Dict

import click
from click.core import Context, Option, Argument

from ModelPusher import ModelPusher


def validate_dict(ctx: Context, param: Union[Option, Argument], value: str) -> Union[Dict, None]:
    """Callback for click.option to check if the input argument is a dictionary.
    Returns a dictionary or abort the code execution"""
    try:
        value = ast.literal_eval(value)
    except:
        click.echo(f"Cannot evaluate the parameter. Be sure to pass a correct value. "
                   f"The parameter passed was {value}")
    if isinstance(value, dict):
        return value
    else:
        click.echo(f"The parameter in input is not a dictionary but a {type(value).__name__}! "
                   f"The evaluated parameter is {value}")
        ctx.abort()


@click.command()
@click.option("--model_dir", type=Path, help='Path where the model is saved')
@click.option("--model_name", default="model")
@click.option("--model_tags", type=click.UNPROCESSED, default="{}", callback=validate_dict)
@click.option('--input_dataset_name', type=str, help='Name of the registered dataset used to train/test the model')
@click.option('--skip_pushing_model', type=bool, default=False, help='If True, the model is not pushed')
def main(model_dir, model_name, model_tags, input_dataset_name, skip_pushing_model):
    """Register models in Azure

    :param model_dir: directory where the model is stored
    :param model_name: the name that can be used to refer to the model within the storage.
    :param model_tags: dictionary as string with the tags used when registering the model
    :param input_dataset_name: Name of the registered dataset used to train/test the model
    :param skip_pushing_model: if True, the model is not pushed
    """
    print('********** ModelPusher_AML **********')

    ModelPusher.execute(model_dir=model_dir,
                        model_name=model_name,
                        model_tags=model_tags,
                        input_dataset_name=input_dataset_name,
                        skip_pushing_model=skip_pushing_model,
                        )


if __name__ == "__main__":
    main()
