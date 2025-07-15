from typing import Optional, List, Union
from pydantic import BaseModel
from label_studio_sdk.label_interface.objects import PredictionValue


# one or multiple predictions per task
SingleTaskPredictions = Union[List[PredictionValue], PredictionValue]


class ModelResponse(BaseModel):
    """Model response with predictions for Label Studio, used in /predict API endpoint"""

    class Config:
        protected_namespaces = ("__.*__", "_.*")  # Excludes 'model_'

    model_version: Optional[str] = None
    predictions: List[SingleTaskPredictions]

    def has_model_version(self) -> bool:
        return bool(self.model_version)

    def update_predictions_version(self) -> None:
        """ """
        for prediction in self.predictions:
            if isinstance(prediction, PredictionValue):
                prediction = [prediction]
            for p in prediction:
                if not p.model_version:
                    p.model_version = self.model_version

    def set_version(self, version: str) -> None:
        """ """
        self.model_version = version
        # Set the version for each prediction
        self.update_predictions_version()
