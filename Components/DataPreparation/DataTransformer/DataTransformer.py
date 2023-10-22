import pandas as pd
import typesentry

import numpy as np

typed = typesentry.Config().typed


class DataTransformer:

    @staticmethod
    @typed()
    def execute(df: pd.DataFrame) -> pd.DataFrame:
        """Perform a variety of data processing steps in order to prepare the data for use in modelling

        :param df: pandas dataframe to be transformed
        :returns DataFrame: A processed pandas dataframe that contain all the KPI's that were calculated
        """
        print('********** DataTransformer **********')

        # we drop the date column and instant since they are not required features.
        df.drop(['dteday'], axis=1, inplace=True)

        # we also encode the hour of the day with a sin-cos encoding
        coef = 2 * np.pi / 23.0
        df['hour_sin'] = np.sin(coef * df['hr'])
        df['hour_cos'] = np.cos(coef * df['hr'])

        # We can also drop holiday because it is the opposite of workingday and one such feature is enough.
        # we also drop registered and casual since their sum gives the final output.
        # Finally, we drop 'hr' since its values have been encoded
        df.drop(['holiday', 'registered', 'casual', 'hr'], axis=1, inplace=True)
        print('Not relevant columns have been removed')


        # Renaming columns for better readability
        df.rename(columns={'yr': 'year', 'mnth': 'month', 'weekday': 'week_day',
                           'workingday': 'working_day', 'weathersit': 'weather_situation', 'atemp': 'temp_feel',
                           'hum': 'humidity', 'windspeed': 'wind_speed', 'cnt': 'count'}, inplace=True)
        print('df columns have been renamed')
        return df
