from warnings import warn

import pandas as pd
import typesentry
from azureml.core import Run, Dataset
from azureml.core import Workspace
from azureml.core.run import _OfflineRun

typed = typesentry.Config().typed


class DataFetcher:
    @staticmethod
    @typed()
    def execute(input_dataset_version: str,
                input_dataset_name: str,
                ) -> pd.DataFrame:
        """
        Fetches data <input_dataset_name> version <input_dataset_version> from AML and saves it in <output_dir> as a pickle file
        The name used to save the file is hard-coded (for simplicity) and is "df.pickle"

        :param input_dataset_name: name of the registered dataset to load
        :param input_dataset_version: version of the registered dataset to load
        """
        print(f'Fetching {input_dataset_name} version {input_dataset_version}...')
        run = Run.get_context()
        if isinstance(run, _OfflineRun):
            workspace = Workspace.from_config()
        else:
            workspace = run.experiment.workspace

        if input_dataset_version == "latest":
            warn(
                f'When not specifying the version number of the dataset, AML will try to fetch the latest version.'
                f'However, note that Azure is sometimes using the output of a previous DataFetcher run,'
                f'which might not be the latest version of the data!'
                f'Please specify the version of the dataset to prevent this from happening.')

        #######################
        # YOUR CODE HERE
        input_dataset = None
        df = None
        #######################

        print('Data fetched')
        print(f'Data version: {input_dataset.version}')
        return df
