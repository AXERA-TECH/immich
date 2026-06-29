from typing import Any

from PIL import Image

from immich_ml.models.base import InferenceModel
from immich_ml.schemas import FaceDetectionOutput, ModelFormat, ModelSession, ModelTask, ModelType

from .axera_insightface import AxeraInsightFace


class AxeraFaceDetector(InferenceModel):
    depends = []
    identity = (ModelType.DETECTION, ModelTask.FACIAL_RECOGNITION)

    def __init__(self, model_name: str, min_score: float = 0.7, **model_kwargs: Any) -> None:
        super().__init__(model_name, **model_kwargs, model_format=ModelFormat.AXERA)
        self.min_score = model_kwargs.get("minScore", min_score)
        self.axera_model = AxeraInsightFace(self.cache_dir, model_name)

    def _download(self) -> None:
        self.axera_model.download()

    def _load(self) -> ModelSession:
        session = self._make_session(self.axera_model.det_model_path)
        self.model = self.axera_model.make_detector(session, self.min_score)
        return session

    def _predict(self, inputs: Image.Image) -> FaceDetectionOutput:
        return self.axera_model.detect(self.model, inputs)

    def configure(self, **kwargs: Any) -> None:
        self.min_score = kwargs.get("minScore", self.min_score)
        if hasattr(self, "model"):
            self.model.det_thresh = self.min_score
