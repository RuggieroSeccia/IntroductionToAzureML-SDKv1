from azureml.pipeline.core import Pipeline
from azureml.pipeline.core import PipelineData, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep

from azureml.core.runconfig import RunConfiguration
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azureml.core import Workspace, Dataset

import typesentry

from Utils.AML_Pipelines.AML_PipelinesFactory_helper_functions import get_git_properties

typed = typesentry.Config().typed


class define_AML_ModelTrainingPipeline:

    @staticmethod
    @typed()
    def execute(run_config: RunConfiguration,
                datastore: AzureBlobDatastore,
                workspace: Workspace,
                fast_compute_target: str,
                allow_reuse_all_components: bool = True) -> Pipeline:
        """Define and returns the ModelTraining Pipeline ready to be submitted on Azure ML

        :param run_config: configuration of the AML run
        :param datastore: datastore where outputs will be stored
        :param workspace: AML workspace
        :param fast_compute_target: compute target used for computational expensive steps,
                                    i.e. ModelTraining and ModelInterpreter
        :param allow_reuse_all_components: if True, allow_reuse for all components.
                                      If False, all components are always rerun, this is used for testing purposes.
        :return the pipeline ready to be submitted
        """

        #################################
        ###### PipelineParameters #######
        #################################
        # DataFetcher params
        input_dataset_name_param = PipelineParameter(name="input_dataset_name", default_value="BikeSharingHours")
        input_dataset_version_param = PipelineParameter(name="input_dataset_version", default_value="latest")

        # DataSplitter params
        target_feat_param = PipelineParameter(name="target_feat", default_value="realised_stop_time_duration")
        test_size_param = PipelineParameter(name="test_size", default_value=0.25)
        random_state_splitting_param = PipelineParameter(name="random_state_splitting", default_value=1)

        # PipelineTrainer params
        method_missing_value_param = PipelineParameter(name="method_missing_value", default_value="mean")
        categorical_class_minimum_occurrences_param = PipelineParameter(name="categorical_class_minimum_occurrences",
                                                                        default_value=500.0)
        model_params_param = PipelineParameter(name="model_params", default_value="{'n_estimators' : 100}")
        random_state_training_param = PipelineParameter(name="random_state_training", default_value=0)

        # ModelPusher params
        model_tags = PipelineParameter(name="model_tags", default_value="{}")
        model_name_param = PipelineParameter(name="model_name", default_value="Model-EF-MLOPS-Demo")
        skip_pushing_model_param = PipelineParameter(name="skip_pushing_model", default_value=False)
        #################################
        ###### Pipeline Definition ######
        #################################
        # Data Fetcher
        fetched_data = PipelineData("data_fetched",
                                    datastore=datastore,
                                    )

        data_fetcher_step = PythonScriptStep(
            name="data_fetcher",
            source_directory="./Components/DataPreparation/DataFetcher",
            script_name="DataFetcher_AML.py",
            arguments=["--input_dataset_name", input_dataset_name_param,
                       "--input_dataset_version", input_dataset_version_param,
                       "--output_dir", fetched_data],
            outputs=[fetched_data],
            runconfig=run_config,
            allow_reuse=allow_reuse_all_components)

        # Data Transformer
        transformed_data = PipelineData("data_transformed",
                                        datastore=datastore,
                                        )

        data_transformer_step = PythonScriptStep(
            name="data_transformer",
            source_directory="./Components/DataPreparation/DataTransformer",
            script_name="DataTransformer_AML.py",
            arguments=["--input_dir", fetched_data,
                       "--output_dir", transformed_data],
            inputs=[fetched_data],
            outputs=[transformed_data],
            runconfig=run_config,
            allow_reuse=allow_reuse_all_components)

        # Data Splitter
        train_data = PipelineData(
            "train_data",
            datastore=datastore)

        test_data = PipelineData(
            "test_data",
            datastore=datastore
        )

        data_splitter_step = PythonScriptStep(
            name="data_splitter",
            source_directory="./Components/Training/DataSplitter",
            script_name="DataSplitter_AML.py",
            arguments=["--input_dir", transformed_data,
                       "--output_train_data_dir", train_data,
                       "--output_test_data_dir", test_data,
                       "--target_feat", target_feat_param,
                       "--test_size", test_size_param,
                       "--random_state", random_state_splitting_param,
                       ],
            inputs=[transformed_data],
            outputs=[train_data, test_data],
            runconfig=run_config,
            allow_reuse=allow_reuse_all_components)

        # PipelineTrainer_AML

        model_pipeline = PipelineData(
            "model_pipeline",
            datastore=datastore
        )
        pipeline_trainer_step = PythonScriptStep(
            name="pipeline_trainer",
            source_directory="./Components/Training/PipelineTrainer",
            script_name="PipelineTrainer_AML.py",
            compute_target=fast_compute_target,
            arguments=["--train_data_dir", train_data,
                       "--output_dir", model_pipeline,
                       "--method_missing_value", method_missing_value_param,
                       "--categorical_class_minimum_occurrences", categorical_class_minimum_occurrences_param,
                       "--model_params", model_params_param,
                       "--random_state_training", random_state_training_param,
                       ],
            inputs=[train_data],
            outputs=[model_pipeline],
            runconfig=run_config,
            allow_reuse=allow_reuse_all_components)

        # PipelineEvaluator_AML
        #######################
        # YOUR CODE HERE - define the AML step for the PipelineEvaluator_AML

        #######################

        # Persisting model
        model_pusher_step = PythonScriptStep(
            name="model_pusher",
            source_directory="./Components/Training/ModelPusher",
            script_name="ModelPusher_AML.py",
            arguments=[
                "--model_dir", model_pipeline,
                "--model_name", model_name_param,
                "--input_dataset_name", input_dataset_name_param,
                "--model_tags", model_tags,
                "--skip_pushing_model", skip_pushing_model_param
            ],
            inputs=[model_pipeline],
            runconfig=run_config,
            allow_reuse=allow_reuse_all_components)

        # Define pipeline
        #######################
        # YOUR CODE HERE - specify the evaluator should run before the model_pusher

        #######################

        pipeline = Pipeline(
            description=get_git_properties(),
            workspace=workspace,
            steps=[
                data_fetcher_step,
                data_transformer_step,
                data_splitter_step,
                pipeline_trainer_step,
                #######################
                # YOUR CODE HERE

                #######################
                model_pusher_step
            ],
        )
        pipeline.validate()
        return pipeline
