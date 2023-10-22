import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Union, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import typesentry
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.pipeline import Pipeline

typed = typesentry.Config().typed


class PipelineEvaluator:

    @staticmethod
    @typed()
    def execute(pipeline: Pipeline,
                X_test: pd.DataFrame,
                y_test: pd.Series,
                output_dir: Union[Path, str] = None,
                ) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
        """Evaluate the performance of a pipeline over a set of desired values (X_test, y_test).

        :param pipeline: fitted model pipeline with a regressor as last step.
        :param X_test: pandas DataFrame with features. If None then "model" should be None and "predictions" must be passed in input
        :param y_test: pandas Series with response variable for the test set
        :param output_dir: directory where output plots will be saved
        :return:
                - metrics_test: dictionary with various metrics related to the model performance.
                - df_errors_test [Optional]: a dataframe with predictions, actual values and errors for each sample
                                             in the test set
        """
        print('********** ModelEvaluator **********')
        output_dir = PipelineEvaluator.create_output_dir(output_dir)

        print('Computing evaluation metrics for the test set...')
        predictions_test = pipeline.predict(X_test)

        metrics_test = PipelineEvaluator.compute_metrics(y_test, predictions_test)
        print('Evaluation metrics of the test set computed')

        # we plot the distribution of errors
        PipelineEvaluator.error_histogram_distribution(y_test - predictions_test, output_dir)

        # we save the resulting df
        df_results_test = X_test.copy()
        df_results_test['predictions'] = predictions_test
        df_results_test['actuals'] = y_test
        df_results_test['prediction_error'] = df_results_test['predictions'] - df_results_test['actuals']
        path_to_file = os.path.join(output_dir, 'df_results_test.pickle')
        df_results_test.to_pickle(path_to_file)

        return metrics_test, df_results_test

    @staticmethod
    def __median_absolute_percentage_error(pred: pd.Series,
                                           actual: pd.Series) -> float:
        actual, pred = np.array(actual), np.array(pred)
        epsilon = np.finfo(np.float64).eps
        median_ape = np.median(abs((pred - actual) / np.maximum(np.abs(actual), epsilon)))
        return median_ape

    @staticmethod
    def __symmetric_mean_absolute_percentage_error(pred: pd.Series,
                                                   actual: pd.Series) -> float:
        actual, pred = np.array(actual), np.array(pred)
        smape = np.mean(2 * np.abs(pred - actual) / (np.abs(actual) + np.abs(pred))) * 100
        return smape

    @staticmethod
    def __symmetric_mean_percentage_error(pred: pd.Series,
                                          actual: pd.Series) -> float:
        actual, pred = np.array(actual), np.array(pred)
        smpe = np.mean(2 * (pred - actual) / (np.abs(actual) + np.abs(pred))) * 100
        return smpe

    @staticmethod
    def __symmetric_median_absolute_percentage_error(pred: pd.Series,
                                                     actual: pd.Series) -> float:
        actual, pred = np.array(actual), np.array(pred)
        smdape = np.median(2 * np.abs(pred - actual) / (np.abs(actual) + np.abs(pred))) * 100
        return smdape

    @staticmethod
    def __symmetric_median_percentage_error(pred: pd.Series,
                                            actual: pd.Series) -> float:
        actual, pred = np.array(actual), np.array(pred)
        smdpe = np.median(2 * (pred - actual) / (np.abs(actual) + np.abs(pred))) * 100
        return smdpe

    @staticmethod
    def error_histogram_distribution(err, output_dir):
        """Save the histogram distribution of the error"""
        fig = px.histogram(err,
                           title='Error distribution',
                           histnorm='percent',
                           nbins=500)

        fig.update_xaxes(title='Errors')
        fig.update_yaxes(title='Probability (%)')
        fig.update_layout(
            showlegend=False,
            autosize=False,
            width=900,
            height=600,
        )
        path_to_file = os.path.join(output_dir, 'Prediction_error_Histogram_distribution.html')
        print(f'Saving plot at location {path_to_file}')
        fig.write_html(path_to_file)
        fig.show()

    @staticmethod
    def compute_metrics(y_true: pd.Series,
                        predictions: pd.Series):
        return {'mape': mean_absolute_percentage_error(y_true, predictions),
                'median_ape': PipelineEvaluator.__median_absolute_percentage_error(predictions,
                                                                                   y_true),
                'smape': PipelineEvaluator.__symmetric_mean_absolute_percentage_error(predictions,
                                                                                      y_true),
                'smdape': PipelineEvaluator.__symmetric_median_absolute_percentage_error(predictions,
                                                                                         y_true),
                'smpe': PipelineEvaluator.__symmetric_mean_percentage_error(predictions, y_true),
                'smdpe': PipelineEvaluator.__symmetric_median_percentage_error(predictions, y_true),
                'n_samples': len(y_true)
                }

    @staticmethod
    def create_output_dir(output_dir: Path) -> str:
        """Generate and create output directory"""
        if output_dir is None:
            folder_name = datetime.now().strftime("%Y%m%d %H%M%S")
            output_dir = os.path.join(os.getcwd(), rf'Logging/ModelEvaluator_Outputs/{folder_name}')

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        return output_dir
