from azureml.pipeline.core import Pipeline
import typesentry
import os
from warnings import warn

from Pipelines.AML_ModelTrainingPipeline import define_AML_ModelTrainingPipeline

from azureml.core.runconfig import RunConfiguration
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azureml.core.workspace import Workspace

typed = typesentry.Config().typed


class AML_PipelinesFactory():

    @staticmethod
    @typed()
    def execute(pipeline_type: str,
                run_config: RunConfiguration,
                datastore: AzureBlobDatastore,
                workspace: Workspace,
                fast_compute_target: str = None,
                allow_reuse_all_components: bool = True
                ) -> Pipeline:
        """ Returns the AML Pipeline to run

        :param pipeline_type: type of the pipeline to run
        :param run_config: configuration of the AML run
        :param datastore: datastore where outputs will be stored
        :param workspace: AML workspace
        :param fast_compute_target: compute target used for computational expensive steps.
        If none, then the same target of the run_config is used
        :param allow_reuse_all_components: if True, allow_reuse for all components.
                                      If False, all components are always rerun, this is used for testing purposes.
        :return the pipeline ready to be submitted
        """
        if 'Components' not in os.listdir():
            warn(
                f'This pipeline factory depends on setting the working folder to the top-level project folder.'
                f' Be sure to be in the correct working directory. '
                f'The current working directory is {os.getcwd()}')

        fast_compute_target = run_config.target if fast_compute_target is None else fast_compute_target

        if pipeline_type == 'ModelTrainingPipeline':
            pipeline = define_AML_ModelTrainingPipeline.execute(run_config=run_config,
                                                                datastore=datastore,
                                                                workspace=workspace,
                                                                fast_compute_target=fast_compute_target,
                                                                allow_reuse_all_components=allow_reuse_all_components)
        else:
            raise ValueError(f"Pipeline {pipeline_type} is not supported")

        return pipeline
