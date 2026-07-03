from typing import Any

from PIL import Image

from immich_ml.models.base import InferenceModel
from immich_ml.schemas import ModelFormat, ModelSession, ModelTask, ModelType

from .axera_ppocr import AxeraPPOCR
from .schemas import TextDetectionOutput


class AxeraTextDetector(InferenceModel):
    depends = []
    identity = (ModelType.DETECTION, ModelTask.OCR)

    def __init__(self, model_name: str, min_score: float = 0.5, **model_kwargs: Any) -> None:
        super().__init__(model_name, **model_kwargs, model_format=ModelFormat.AXERA)
        self.min_score = model_kwargs.get("minScore", min_score)
        self.max_resolution = model_kwargs.get("maxResolution", 736)
        self.axera_model = AxeraPPOCR(self.cache_dir, model_name)

    def _download(self) -> None:
        self.axera_model.download()

    @property
    def cached(self) -> bool:
        return self.axera_model.is_model_dir_complete(self.axera_model.model_dir)

    def _load(self) -> ModelSession:
        return self._make_session(self.axera_model.det_model_path)

    def _predict(self, inputs: Image.Image) -> TextDetectionOutput:
        return self.axera_model.detect(
            self.session,
            inputs,
            min_score=self.min_score,
            max_resolution=self.max_resolution,
        )

    def configure(self, **kwargs: Any) -> None:
        self.min_score = kwargs.get("minScore", self.min_score)
        self.max_resolution = kwargs.get("maxResolution", self.max_resolution)
