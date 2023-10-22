import ast
import os
import pickle
from pathlib import Path
from typing import Union, Dict

import click
from azureml.core import Run
from click.core import Context, Option, Argument

from DataPusher import DataPusher


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
@click.option("--input_dir", required=True, type=Path)
@click.option("--input_file", default="transformed_data.pickle", type=str)
@click.option("--save_path_datastore", required=True, type=Path)
@click.option("--dataset_name", required=True, type=str, )
@click.option("--dataset_tags", type=click.UNPROCESSED, default="{}", callback=validate_dict)
@click.option("--timestamp_column", type=click.UNPROCESSED, default=" ", callback=strip_white_space)
def main(input_dir, input_file, save_path_datastore, dataset_name, dataset_tags, timestamp_column):
    """
    Persist data in our blob storage in Azure

    :param input_dir directory where the dataset is stored
    :param input_file: name of file with data set
    :param save_path_datastore: location within our blob storage where dataset should be persisted
    :param dataset_name: the name that can be used to refer to the stored dataset within the storage
    :param dataset_tags: dictionary with the tags used when registering the dataset
    :param timestamp_column: optional column name that will be registered as the timestamp column to filter a
                dataset on at a later stage

    :return: None
    """
    print('********** DataPusher_AML **********')
    print('Retrieve the data...')
    df = pickle.load(open(os.path.join(input_dir, input_file), 'rb'))
    print('Data retrieved')

    run = Run.get_context()
    exp = run.experiment
    ws = exp.workspace

    DataPusher.execute(save_path_datastore=save_path_datastore,
                       dataset_name=dataset_name,
                       df=df,
                       dataset_tags=dataset_tags,
                       workspace=ws,
                       timestamp_column=timestamp_column)


if __name__ == "__main__":
    main()
