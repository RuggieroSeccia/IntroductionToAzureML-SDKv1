from typing import Union, List, Dict

import numpy as np
import pandas as pd
import typesentry
from pandas.api.types import is_numeric_dtype
from sklearn.base import TransformerMixin

if __name__ != 'OutliersHandler':
    raise ImportError("You shouldn't import this directly! import via the Encoders Module using:"
                      " 'from Components.Training.ModelComponents.src.DataEncoder.Encoders"
                      " import OutlierHandler' \n"
                      "This ensures that the CustomModels are always loaded from the correct relative path")


typed = typesentry.Config().typed


class OutlierHandler(TransformerMixin):
    SUPPORTED_METHODS = ['mean', 'median']

    @typed()
    def __init__(self,
                 thresh: Union[int, float],
                 method: str,
                 columns_to_exclude: List[str] = []):

        """Handle outliers data in numeric columns.

        :param thresh: the threshold used to find the outliers
        :param method: the method used to treat outliers. Supported methods are "mean" or "median".
        :param columns_to_exclude: columns for which outliers should not be detected
        Returns the new dataframe without missing values
        """
        if thresh < 0:
            raise ValueError(f'"thresh" should be positive.'
                             f'{thresh} was passed instead')
        if method not in OutlierHandler.SUPPORTED_METHODS:
            raise ValueError(f'"method" should be one of {", ".join(OutlierHandler.SUPPORTED_METHODS)}.'
                             f'{method} was passed instead')
        self.thresh = thresh
        self.method = method
        self.mu = None
        self.median = None
        self.sigma = None
        self.columns_to_exclude = columns_to_exclude

    def __outlier_detection(self, series: pd.Series):
        """
        Flags observations as outliers in case they are more than "thresh" standard deviations away from the mean
        """

        average = self.mu[series.name]
        standard_deviation = self.sigma[series.name]

        outlier_indicator = abs(series - average) > self.thresh * standard_deviation
        return outlier_indicator

    def fit(self, df, y=None):
        """
        Computes mean, median and std of numeric columns
        """
        self.columns_ = df.columns

        if self.columns_to_exclude and not set(self.columns_to_exclude).issubset(df.columns):
            raise ValueError(f'the columns to exclude {self.columns_to_exclude} are not in the dataframe')

        self.mu = df.mean(numeric_only=True)
        self.median = df.median(numeric_only=True)
        self.sigma = df.std()

        binary_cols = {col for col in df.select_dtypes(include='number').columns
                       if np.isin(df[col].dropna().unique(), [0, 1]).all()}

        print(f'The following binary columns were detected and will not be considered by the OutliersHandler'
                     f'\n{binary_cols}')
        self.columns_to_exclude = set(self.columns_to_exclude).union(binary_cols)

        if self.columns_to_exclude:
            print(f"The following columns will be excluded from the Outliers Detection:\n{self.columns_to_exclude}")
        return self

    def transform(self,
                  df: pd.DataFrame,
                  y=None):
        """
        Removes outliers from "df" according to "method"
        Supported methods are: "mean", "median"
        """

        print("Checking for outliers in the data...")
        found_outliers = False
        print(f'Outliers method: {self.method}')
        dict_outliers: Dict = {}
        for column, dtype in df.dtypes.to_dict().items():
            if is_numeric_dtype(df[column]) and column not in self.columns_to_exclude:
                outlier_indicator = self.__outlier_detection(df[column])
                if not found_outliers:
                    found_outliers = outlier_indicator.sum() > 0
                if self.method == "mean":
                    df.loc[outlier_indicator, column] = self.mu[column].astype(dtype)
                elif self.method == "median":
                    df.loc[outlier_indicator, column] = self.median[column].astype(dtype)
                else:
                    raise ValueError(f"The parameter \"method\" should be equal to any of "
                                     f"{', '.join(OutlierHandler.SUPPORTED_METHODS)}. "
                                     f"\"{self.method}\" was passed instead.")
                if outlier_indicator.sum() > 0:
                    print(f"Number of outliers in {column} : {outlier_indicator.sum()}")
                    dict_outliers[column] = outlier_indicator.sum()

        print(f'Outliers per column: {dict_outliers}')

        if found_outliers:
            print("Outliers have been replaced")
        else:
            print("No outliers have been found")
        return df

    def get_feature_names_out(self, *args, **params):
        # this method is needed to allow the use of .set_output(transform="pandas")
        return self.columns_