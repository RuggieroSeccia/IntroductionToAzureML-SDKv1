from typing import Tuple, Union

import pandas as pd
import typesentry
from sklearn.model_selection import train_test_split

typed = typesentry.Config().typed


class DataSplitter:

    @staticmethod
    @typed()
    def execute(df: pd.DataFrame,
                target_feat: str,
                test_size: float,
                random_state: Union[float, None] = 1,
                ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Return data split into train and test sets
        :param df: dataframe of potential feature data
        :param target_feat: name of the target column in the dataframe
        :param test_size: size of the test
        :param random_state: fix it for reproducibility purposes [default=1]
        :return: (X_train, X_test, y_train, y_test) containing explanatory and target variables split over
                 train and test set
        """
        print('********** DataSplitter **********')

        print('Splitting in train and test data...')
        df_train, df_test = train_test_split(df,
                                             test_size=test_size,
                                             random_state=random_state, )
        y_train = df_train.pop(target_feat)
        X_train = df_train.copy()
        y_test = df_test.pop(target_feat)
        X_test = df_test.copy()

        print(f"Number of columns in data:  {X_train.shape[1]}")
        print(f"Number of rows in train data:  {X_train.shape[0]}")
        print(f"Number of rows in test data:  {X_test.shape[0]}")

        return X_train, X_test, y_train, y_test
