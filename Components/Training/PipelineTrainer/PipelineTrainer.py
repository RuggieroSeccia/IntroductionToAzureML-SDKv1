import inspect
import json
import os
import pickle
import sys
from numbers import Number
from pathlib import Path
from time import time
from typing import Dict, Union, Any, List

import mlflow
import numpy as np
import pandas as pd
import typesentry
from mlflow.models.signature import infer_signature
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# sys.path.insert(0, os.path.dirname(__file__))
# from CustomPipelineSteps import OutlierHandler
# from CustomPipelineSteps import MlFlowPythonModelWrapper

typed = typesentry.Config().typed


class PipelineTrainer:

    @staticmethod
    @typed()
    def execute(X_train: pd.DataFrame,
                y_train: pd.Series,
                categorical_class_minimum_occurrences: Number,
                method_missing_value: str,
                model_params: Dict[str, Union[str, Number]],
                output_dir: Path = None,
                random_state_training=1
                ) -> Pipeline:
        # define continuous and categorical columns
        continous_col = ['temp', 'temp_feel', 'humidity', 'wind_speed', 'hour_sin', 'hour_cos']
        categorical_col = ['season', 'year', 'month', 'week_day', 'working_day', 'weather_situation']
        print(f"Total input columns = {len(continous_col) + len(categorical_col)}")

        print('Defining preprocessing pipeline')
        preprocessing_steps = [
            ColumnTransformer(
                transformers=[
                    ('num',
                     Pipeline(steps=[
                         ('imputer', SimpleImputer(strategy=method_missing_value, missing_values=np.nan)),
                         # ('outlier_handler', OutlierHandler(thresh=3, method='mean')),
                         ('scaler', StandardScaler())]),
                     continous_col),
                    ('cat',
                     Pipeline(steps=[
                         ('imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan)),
                         ('ohe', OneHotEncoder(
                             sparse_output=False,
                             handle_unknown='ignore',
                             min_frequency=categorical_class_minimum_occurrences
                         ))]),
                     categorical_col)
                ]
            )
        ]
        model = RandomForestRegressor(**model_params, random_state=random_state_training)

        pipeline = make_pipeline(*preprocessing_steps, model).set_output(transform="pandas")
        print(f"pipeline defined:\n{pipeline}")
        tic = time()
        pipeline.fit(X_train, y_train)
        cpu_time = time() - tic
        print('*******Results from training the pipeline*******')
        print(f'CPU time to fit the pipeline: {cpu_time:.2f} sec')

        if output_dir is not None:
            print(f'Saving pipeline in directory: {output_dir}')
            PipelineTrainer.save_pipeline_as_pickle(pipeline=pipeline,
                                                    output_dir=output_dir)
            # PipelineTrainer.save_pipeline_as_mlflow(pipeline=pipeline,
            #                                         X_train=X_train,
            #                                         output_dir=output_dir)
        return pipeline

    @staticmethod
    def save_pipeline_as_pickle(pipeline: Pipeline,
                                output_dir: Path):
        os.makedirs(output_dir, exist_ok=True)
        pickle.dump(pipeline, open(f"{output_dir}\model.pickle", 'wb'))

    @staticmethod
    def save_pipeline_as_mlflow(pipeline: Pipeline,
                                X_train: pd.DataFrame,
                                output_dir: Path):
        def _get_trained_model_information_dict(model: BaseEstimator, X: pd.DataFrame) -> Dict[str, Any]:
            """Get dictionary describing trained model"""
            return {
                'model_type': model.__class__.__name__,
                'model_params': model.get_params(deep=True),
                'column_names': X.columns.values.tolist()
            }

        def _get_code_file_locations_for_pipeline(pipeline: Union[Pipeline, RegressorMixin]) -> List[str]:
            """
            given the pipeline/model, we cycle over the steps to make sure we create an artifact
            with all the code needed to recreate the model
            """
            code_files_paths = [inspect.getfile(MlFlowPythonModelWrapper)]
            if isinstance(pipeline, Pipeline):
                for step in pipeline:
                    if "sklearn" not in str(step.__class__):
                        code_files_paths.append(inspect.getfile(step.__class__))
                    # if we have a ColumnTransformer, then we need to loop over its steps
                    elif "ColumnTransformer" in str(step.__class__):
                        for _, step_i, _ in step.transformers:
                            if inspect.getfile(step_i.__class__) not in code_files_paths and \
                                    "sklearn" not in str(step_i.__class__):
                                code_files_paths.append(inspect.getfile(step_i.__class__))

                            # if a step is a pipeline, then we need to loop over its steps as well
                            elif "sklearn.pipeline" in str(step_i.__class__):
                                for _, step_ij in step_i.steps:
                                    if inspect.getfile(step_ij.__class__) not in code_files_paths and \
                                            "sklearn" not in str(step_ij.__class__):
                                        code_files_paths.append(inspect.getfile(step_ij.__class__))
            else:
                if "sklearn" not in str(pipeline.__class__):
                    code_files_paths.append(inspect.getfile(pipeline.__class__))

            return [str(fn) for fn in code_files_paths]

        signature = infer_signature(X_train.head(5), pipeline.predict(X_train.head(5)))
        wrapped_pipeline = MlFlowPythonModelWrapper(pipeline)
        mlflow.pyfunc.save_model(path=str(output_dir), python_model=wrapped_pipeline,
                                 code_path=_get_code_file_locations_for_pipeline(pipeline),
                                 signature=signature,
                                 input_example=X_train.head(5),
                                 )
        json_fn = os.path.join(output_dir, "model_info.json")
        with open(json_fn, "w") as f:
            json.dump(_get_trained_model_information_dict(pipeline, X_train), f, indent=4,
                      default=lambda o: type(o).__qualname__)
        print(f'Pipeline saved at location {output_dir}')
