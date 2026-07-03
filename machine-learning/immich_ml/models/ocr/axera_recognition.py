from typing import Any

from PIL import Image

from immich_ml.models.base import InferenceModel
from immich_ml.schemas import ModelFormat, ModelSession, ModelTask, ModelType

from .axera_ppocr import AxeraPPOCR
from .schemas import TextDetectionOutput, TextRecognitionOutput


class AxeraTextRecognizer(InferenceModel):
    depends = [(ModelType.DETECTION, ModelTask.OCR)]
    identity = (ModelType.RECOGNITION, ModelTask.OCR)

    def __init__(self, model_name: str, min_score: float = 0.9, **model_kwargs: Any) -> None:
        super().__init__(model_name, **model_kwargs, model_format=ModelFormat.AXERA)
        self.min_score = model_kwargs.get("minScore", min_score)
        self.axera_model = AxeraPPOCR(self.cache_dir, model_name)

    def _download(self) -> None:
        self.axera_model.download()

    @property
    def cached(self) -> bool:
        return self.axera_model.is_model_dir_complete(self.axera_model.model_dir)

    def _load(self) -> ModelSession:
        self.axera_model.cls_session = self._make_session(self.axera_model.cls_model_path)
        return self._make_session(self.axera_model.rec_model_path)

    def _predict(self, img: Image.Image, texts: TextDetectionOutput) -> TextRecognitionOutput:
        return self.axera_model.recognize(self.session, img, texts, min_score=self.min_score)

    def configure(self, **kwargs: Any) -> None:
        self.min_score = kwargs.get("minScore", self.min_score)
