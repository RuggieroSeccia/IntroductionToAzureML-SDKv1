import os
import shutil
from typing import Dict
from uuid import uuid4

from azureml.core import Workspace
from azureml.core.datastore import Datastore
from azureml.data import DataType
from azureml.data import TabularDataset
from azureml.data.dataset_factory import TabularDatasetFactory
from pandas import DataFrame


class DataPusher:

    @staticmethod
    def execute(save_path_datastore: str,
                dataset_name: str,
                df: DataFrame,
                dataset_tags: Dict,
                timestamp_column: str = None,
                workspace: Workspace = None):
        """ Upload dataframe as csv files to default datastore of workspace and register as dataset

        :param save_path_datastore: folder on the datastore where a sub-folder with a guid is created to save the dataset in
        :param dataset_name: Name the dataset will be saved under. If the name is in use a new version will be registered
        :param df: DataFrame to upload and register
        :param dataset_tags: dictionary with the tags used when registering the dataset
        :param timestamp_column: optional column name that will be registered as the timestamp column to filter a
                dataset on at a later stage
        :param workspace: Azureml workspace where the dataset will be uploaded in the default datastore

        :return: None
        """
        if workspace is None:
            workspace = Workspace.from_config()

        if df.empty:
            raise ValueError('The input DataFrame is empty. Check the df passed in input')

        datastore: Datastore = workspace.get_default_datastore()

        guid = uuid4()
        relative_path_with_guid = f"{save_path_datastore}/{guid}/"
        local_path = f"./{guid}"
        file_name = "part_"
        extension = ".csv"

        print("Writing dataframe locally to csv in chunks.")
        os.makedirs(local_path, exist_ok=True)
        chunk_size = 20000
        splitted_df = [df.iloc[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
        for i, df_chunk in enumerate(splitted_df):
            df_chunk.to_csv(f"{local_path}/{file_name}{i + 1}{extension}", index=False,
                            date_format='%Y-%m-%d %H:%M:%S%z')
        print("Successfully wrote dataframe as csv.")

        print(f"Uploading dataframe as csv files to {relative_path_with_guid} on default datastore")
        datastore.upload(src_dir=f"./{guid}", target_path=relative_path_with_guid)
        shutil.rmtree(local_path)
        print("Successfully uploaded files to datastore.")

        # parse which columns contain datetimes so that these will be registered as datetimes on azureml
        # Note that "datetime" does not include datetime with timezones.
        datetime_columns = df.select_dtypes(include=["datetime"]).columns.values
        datetime_column_dtypes = {col: DataType.to_datetime('%Y-%m-%d %H:%M:%S') for col in datetime_columns}
        print(f"The following columns will be formatted as datetime columns in the dataset: {datetime_columns}")

        #  If we don't specify that datetimetz columns are strings,
        #  then Azure automatically detects them as datetime and report them to UTC.
        #  Thus we prefer to save them as strings
        datetime_columns_with_timezone = df.select_dtypes(include=["datetimetz"]).columns.values
        print(f"The following columns will be formatted as string columns since they are datetimetz: "
              f"{datetime_columns_with_timezone}")

        string_columns = df.select_dtypes(include=["object"]).columns.values
        string_column_dtypes = {col: DataType.to_string() for col in [*string_columns, *datetime_columns_with_timezone]}
        print(f"The following columns will be formatted as string columns in the dataset: {string_columns}")

        print("Creating and registering dataset from uploaded files.")
        dataset: TabularDataset = TabularDatasetFactory.from_delimited_files(
            path=[(datastore, f'{relative_path_with_guid}*{extension}')],
            validate=True,
            set_column_types={
                **datetime_column_dtypes,
                **string_column_dtypes
            }
        )

        if timestamp_column:
            if timestamp_column not in datetime_columns:
                raise KeyError(f"The passed  timestamp_column: {timestamp_column} is not in the datetime columns of the"
                               f" dataframe! The datetime columns are: {datetime_columns}")
            dataset = dataset.with_timestamp_columns(timestamp=timestamp_column)
            print(f"Successfully added '{timestamp_column}' as timestamp column for the dataset")

        dataset.register(workspace, dataset_name, create_new_version=True,
                         tags=dataset_tags)
        print("Successfully created and registered a new dataset.")
