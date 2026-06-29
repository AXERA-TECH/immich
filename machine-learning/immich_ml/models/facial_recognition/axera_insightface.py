from __future__ import annotations

import importlib.util
import json
import sys
import types
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from immich_ml.models.axera import download_axera_repo, remove_axera_suffix
from immich_ml.models.transforms import pil_to_cv2, serialize_np_array
from immich_ml.schemas import FaceDetectionOutput, FacialRecognitionOutput, ModelSession
from immich_ml.sessions.axengine import InferenceSession as AXSession


AXERA_INSIGHTFACE_REPO = "Insightface"


def get_axera_model_pack_name(model_name: str) -> str:
    return remove_axera_suffix(model_name)


class AxeraInsightFace:
    def __init__(self, cache_dir: Path, model_name: str) -> None:
        self.cache_dir = cache_dir
        self.model_pack_name = get_axera_model_pack_name(model_name)
        self._runtime_package = f"_immich_axera_insightface_{id(self)}"

    def download(self) -> None:
        if self.is_model_pack_complete():
            return
        download_axera_repo(AXERA_INSIGHTFACE_REPO, self.repo_dir)

    def make_detector(self, session: ModelSession, min_score: float) -> Any:
        detector = self.runtime.retinaface.RetinaFace(model_file=self.det_model_path.as_posix(), session=session)
        detector.prepare(ctx_id=0, det_thresh=min_score)
        return detector

    def make_recognizer(self, session: ModelSession) -> Any:
        return self.runtime.arcface.ArcFaceONNX(self.rec_model_path.as_posix(), session=session)

    def detect(self, detector: Any, img: Image.Image) -> FaceDetectionOutput:
        bboxes, landmarks = detector.detect(pil_to_cv2(img))
        if bboxes.shape[0] == 0:
            return self.empty_detection()
        if landmarks is None:
            raise ValueError(f"AXERA InsightFace detection model does not output landmarks: {self.det_model_path}")
        return {
            "boxes": bboxes[:, :4].round(),
            "scores": bboxes[:, 4],
            "landmarks": landmarks,
        }

    def recognize(self, recognizer: Any, img: Image.Image, faces: FaceDetectionOutput) -> FacialRecognitionOutput:
        if faces["boxes"].shape[0] == 0:
            return []

        img_cv = pil_to_cv2(img)
        image_size = recognizer.input_size[0]
        crops = [self.runtime.face_align.norm_crop(img_cv, landmark, image_size=image_size) for landmark in faces["landmarks"]]
        embeddings = self.get_embeddings(recognizer, crops)
        return [
            {
                "boundingBox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "embedding": serialize_np_array(embedding),
                "score": score,
            }
            for (x1, y1, x2, y2), embedding, score in zip(faces["boxes"], embeddings, faces["scores"])
        ]

    def get_embeddings(self, recognizer: Any, crops: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
        batch_size = self.get_static_batch_size(recognizer.session)
        if batch_size is None or batch_size > 1:
            return recognizer.get_feat(crops)

        embeddings = [recognizer.get_feat(crop) for crop in crops]
        return np.concatenate(embeddings, axis=0)

    def get_static_batch_size(self, session: ModelSession) -> int | None:
        batch_size = session.get_inputs()[0].shape[0]
        return batch_size if isinstance(batch_size, int) else None

    def empty_detection(self) -> FaceDetectionOutput:
        return {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "scores": np.empty(0, dtype=np.float32),
            "landmarks": np.empty((0, 5, 2), dtype=np.float32),
        }

    @cached_property
    def det_model_path(self) -> Path:
        return self.get_model_paths()["detection"]

    @cached_property
    def rec_model_path(self) -> Path:
        return self.get_model_paths()["recognition"]

    def get_model_paths(self) -> dict[str, Path]:
        config_paths = self.get_config_model_paths()
        if config_paths is not None:
            return config_paths

        model_paths: dict[str, Path] = {}
        for model_path in sorted(self.model_pack_dir.glob("*.axmodel")):
            model_type = self.get_model_type(model_path)
            if model_type in {"detection", "recognition"} and model_type not in model_paths:
                model_paths[model_type] = model_path

        missing = {"detection", "recognition"} - model_paths.keys()
        if missing:
            raise FileNotFoundError(f"Missing AXERA InsightFace model(s) {sorted(missing)} in {self.model_pack_dir}")
        return model_paths

    def is_model_pack_complete(self) -> bool:
        try:
            self.get_model_paths()
            return True
        except FileNotFoundError:
            return False

    def get_config_model_paths(self) -> dict[str, Path] | None:
        config_path = self.repo_dir / "config.json"
        if not config_path.is_file() or config_path.stat().st_size == 0:
            return None

        config = json.loads(config_path.read_text())
        packs = config.get("model_packs", {})
        pack = packs.get(self.model_pack_name)
        if not isinstance(pack, dict):
            return None

        detection = pack.get("detection")
        recognition = pack.get("recognition")
        if not isinstance(detection, str) or not isinstance(recognition, str):
            return None
        return {
            "detection": self.repo_dir / detection,
            "recognition": self.repo_dir / recognition,
        }

    def get_model_type(self, model_path: Path) -> str | None:
        session = AXSession(model_path.as_posix())
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        input_shape = inputs[0].shape
        output_shapes = [output.shape for output in outputs]

        if len(outputs) in {6, 9, 10, 15}:
            return "detection"
        if len(outputs) == 1 and self.get_spatial_size(input_shape) == (112, 112) and self.has_embedding_output(output_shapes):
            return "recognition"
        return None

    def get_spatial_size(self, shape: list[Any]) -> tuple[int, int] | None:
        if len(shape) != 4:
            return None
        if isinstance(shape[1], int) and isinstance(shape[2], int) and shape[3] in {1, 3}:
            return shape[1], shape[2]
        if isinstance(shape[2], int) and isinstance(shape[3], int) and shape[1] in {1, 3}:
            return shape[2], shape[3]
        return None

    def has_embedding_output(self, shapes: list[list[Any]]) -> bool:
        return any(any(dim == 512 for dim in shape) for shape in shapes)

    @cached_property
    def runtime(self) -> types.SimpleNamespace:
        self.load_runtime_package()
        return types.SimpleNamespace(
            retinaface=sys.modules[f"{self._runtime_package}.model_zoo.retinaface"],
            arcface=sys.modules[f"{self._runtime_package}.model_zoo.arcface_onnx"],
            face_align=sys.modules[f"{self._runtime_package}.utils.face_align"],
        )

    def load_runtime_package(self) -> None:
        package_root = self.repo_dir / "insightface"
        if not package_root.is_dir():
            raise FileNotFoundError(f"AXERA InsightFace runtime package not found: {package_root}")

        self.register_package(self._runtime_package, package_root)
        self.register_package(f"{self._runtime_package}.model_zoo", package_root / "model_zoo")
        self.register_package(f"{self._runtime_package}.utils", package_root / "utils")
        self.load_module(f"{self._runtime_package}.utils.face_align", package_root / "utils" / "face_align.py")
        self.load_module(f"{self._runtime_package}.model_zoo.retinaface", package_root / "model_zoo" / "retinaface.py")
        self.load_module(f"{self._runtime_package}.model_zoo.arcface_onnx", package_root / "model_zoo" / "arcface_onnx.py")

    def register_package(self, module_name: str, path: Path) -> None:
        module = types.ModuleType(module_name)
        module.__path__ = [path.as_posix()]  # type: ignore[attr-defined]
        sys.modules[module_name] = module

    def load_module(self, module_name: str, path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load AXERA InsightFace module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    @property
    def repo_dir(self) -> Path:
        return self.cache_dir

    @property
    def model_pack_dir(self) -> Path:
        return self.repo_dir / "models" / self.model_pack_name
