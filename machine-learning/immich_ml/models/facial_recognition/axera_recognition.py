from typing import Any

from PIL import Image

from immich_ml.models.base import InferenceModel
from immich_ml.schemas import (
    FaceDetectionOutput,
    FacialRecognitionOutput,
    ModelFormat,
    ModelSession,
    ModelTask,
    ModelType,
)

from .axera_insightface import AxeraInsightFace


class AxeraFaceRecognizer(InferenceModel):
    depends = [(ModelType.DETECTION, ModelTask.FACIAL_RECOGNITION)]
    identity = (ModelType.RECOGNITION, ModelTask.FACIAL_RECOGNITION)

    def __init__(self, model_name: str, **model_kwargs: Any) -> None:
        super().__init__(model_name, **model_kwargs, model_format=ModelFormat.AXERA)
        self.axera_model = AxeraInsightFace(self.cache_dir, model_name)

    def _download(self) -> None:
        self.axera_model.download()

    def _load(self) -> ModelSession:
        session = self._make_session(self.axera_model.rec_model_path)
        self.model = self.axera_model.make_recognizer(session)
        return session

    def _predict(self, inputs: Image.Image, faces: FaceDetectionOutput) -> FacialRecognitionOutput:
        return self.axera_model.recognize(self.model, inputs, faces)
