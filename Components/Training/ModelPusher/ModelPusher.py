from pathlib import Path
from typing import Dict

import typesentry
from azureml.core import Run, Dataset
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.run import _OfflineRun

typed = typesentry.Config().typed


class ModelPusher:
    @staticmethod
    @typed()
    def execute(model_dir: Path,
                model_name: str,
                model_tags: Dict[str, str],
                input_dataset_name: str,
                skip_pushing_model: bool = False, ) -> None:
        """
        :param model_dir: directory where the model is stored
        :param model_name: the name that can be used to refer to the model within the storage.
        :param model_tags: dictionary as string with the tags used when registering the model
        :param input_dataset_name: Name of the registered dataset used to train/test the model
        :param skip_pushing_model: if True, the model is not pushed
        """

        print('********** ModelPusher **********')
        if skip_pushing_model:
            print('Skip pushing the model because "skip_pushing_model" was set to True')
            return

        print('Retrieving the context...')
        run = Run.get_context()
        if isinstance(run, _OfflineRun):
            workspace = Workspace.from_config()
        else:
            workspace = run.experiment.workspace
        input_dataset = Dataset.get_by_name(workspace, name=input_dataset_name)
        print('Context retrieved...')

        print('Registering the model...')
        input_datasets = [('data', input_dataset)]
        Model.register(
            workspace=workspace,
            model_path=str(model_dir),
            model_name=model_name,
            datasets=input_datasets,
            tags=model_tags
        )
        print('Model registered')
