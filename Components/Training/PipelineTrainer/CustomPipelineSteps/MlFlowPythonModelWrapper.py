from typing import Union

import mlflow
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline

if __name__ != 'MlFlowPythonModelWrapper':
    raise ImportError("You shouldn't import this directly! import via the CustomModels Module using:"
                      " 'from Components.Training.ModelComponents.src.ModelTrainer.CustomModels"
                      " import MlFlowPythonModelWrapper' \n"
                      "This ensures that the CustomModels are always loaded from the correct relative path")


class MlFlowPythonModelWrapper(mlflow.pyfunc.PythonModel):
    """This class is needed to have custom steps in our pipeline.
    Check here https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#pyfunc-create-custom-workflows
    """
    def __init__(self, model: Union[RegressorMixin, Pipeline]):
        super().__init__()
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)
