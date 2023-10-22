import os
from pathlib import Path
import pickle

import click
import mlflow
import pandas as pd
from azureml.core import Run

from PipelineEvaluator import PipelineEvaluator


@click.command()
@click.option('--test_data_dir', type=Path, help="Path where the test data is stored")
@click.option('--pipeline_dir', type=Path, help="Path where the model_pipeline is stored")
@click.option('--output_dir', type=Path, default='ModelEvaluator_plots', help="Path where the plots will be saved")
def main(test_data_dir,
         pipeline_dir,
         output_dir):
    """Call the ModelEvaluator in AzureML, logs the metrics of the model and save the resulting df

    :param test_data_dir: path where the test data is stored
    :param pipeline_dir: directory where the model_pipeline is stored
    :param output_dir:  directory where output plots will be saved
    :return: None
    """

    print('********** ModelEvaluator_AML **********')

    print('Loading the data and the model...')
    X_test = pd.read_pickle(os.path.join(test_data_dir, 'x_test.pickle'))
    y_test = pd.read_pickle(os.path.join(test_data_dir, 'y_test.pickle'))
    print('Test data loaded')

    pipeline = pickle.load(open(f"{pipeline_dir}\model.pickle", 'rb'))

    # uncomment these lines if you want to save the model with mlflow
    # mlflow_pipeline = mlflow.pyfunc.load_model(str(pipeline_dir))
    # pipeline = mlflow_pipeline._model_impl.python_model.model

    print('Model loaded')

    print('Calling the ModelEvaluator...')

    metrics, df_results_test = PipelineEvaluator.execute(pipeline=pipeline,
                                                         X_test=X_test,
                                                         y_test=y_test,
                                                         output_dir=output_dir,
                                                         )
    print('ModelEvaluator executed')

    print('Saving metrics in the logs...')
    run = Run.get_context()
    parent_run = run.parent if hasattr(run, "parent") else None

    for k, v in metrics.items():
        run.log(k, v)
        if parent_run:
            parent_run.log(k, v)

    print('Metrics saved')

    print('Saving the files in the logs...')
    name_logging_dir = "ModelEvaluator_plots"
    run.upload_folder(name_logging_dir, str(output_dir))
    if parent_run:
        parent_run.upload_folder(name_logging_dir, str(output_dir))
    print('Logging done')


if __name__ == "__main__":
    main()
