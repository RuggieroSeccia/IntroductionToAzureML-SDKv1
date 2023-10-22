import ast
import os
import pickle
from pathlib import Path
from typing import Dict, Union

import click
from click.core import Context, Option, Argument

from PipelineTrainer import PipelineTrainer


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
@click.option('--train_data_dir', type=Path, help='Path where the training data are stored')
@click.option('--output_dir', type=Path, help='Path where the encoding_pipeline will be saved')
@click.option('--method_missing_value', type=str, default="mean",
              help="How missing_values are treated. Supported methods are: 'mean', 'median'")
@click.option('--categorical_class_minimum_occurrences', type=float, default=50.,
              help='The minimum amount of occurrences needed for a value \
              to become its own class in each of the categorical columns')
@click.option('--model_params', type=click.UNPROCESSED, default="{}", callback=validate_dict,
              help='Parameter for the regressor model. It is a dictionary passed as a string')
@click.option('--random_state_training', type=int, default=1, help='Random state to initialize the RF model')
def main(train_data_dir,
         output_dir,
         method_missing_value,
         categorical_class_minimum_occurrences,
         model_params,
         random_state_training,
         ):
    """Defines and fits the pipeline. Pipeline is composed of encoding and model training

    :param train_data_dir: path where the training data are stored
    :param output_dir: directory where the output is stored
    :param method_missing_value: how missing_values has to be treated. Supported methods are: 'mean', 'median'.
    :param categorical_class_minimum_occurrences: the minimum amount of occurrences needed for a value to become its own
                            class in each of the categorical columns.
                            - if >=1 only categories with at least <categorical_class_minimum_occurrences> will be encoded
                                     (other values will be encoded as infrequent values)
                            - if <1 the specified percentage of samples in the training set is considered as minimum threshold
    :param model_params: dict of parameters to pass to the model
    :param random_state_training: seed for initializing the RF model
    """
    # On Linux distributions (when running on AzureML), the JOBLIB_TEMP_FOLDER is set to /dev/shm.
    # This results in using shared memory. When not changing this environment variable,
    # it will cause memory issues when doing the hyperparameter tuning
    # See: https://stackoverflow.com/questions/40115043/no-space-left-on-device-error-while-fitting-sklearn-model
    os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

    print('Loading the data...')
    x_train = pickle.load(open(os.path.join(train_data_dir, 'x_train.pickle'), 'rb'))
    y_train = pickle.load(open(os.path.join(train_data_dir, 'y_train.pickle'), 'rb'))
    print('Data loaded')

    PipelineTrainer.execute(X_train=x_train,
                            y_train=y_train,
                            categorical_class_minimum_occurrences=categorical_class_minimum_occurrences,
                            method_missing_value=method_missing_value,
                            model_params=model_params,
                            output_dir=output_dir,
                            random_state_training=random_state_training
                            )


if __name__ == "__main__":
    main()
