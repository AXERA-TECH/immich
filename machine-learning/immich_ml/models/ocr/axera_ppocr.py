import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from immich_ml.models.axera import download_axera_repo, remove_axera_suffix
from immich_ml.models.transforms import pil_to_cv2
from immich_ml.schemas import ModelSession

from .schemas import TextDetectionOutput, TextRecognitionOutput


def get_axera_repo_name(model_name: str) -> str:
    # Immich uses the suffix to route to the AXERA implementation, while the
    # Hugging Face repository keeps the upstream model name without the suffix.
    return remove_axera_suffix(model_name)


class AxeraPPOCR:
    def __init__(self, cache_dir: Path, model_name: str) -> None:
        self.cache_dir = cache_dir
        self.repo_name = get_axera_repo_name(model_name)
        self.cls_session = None
        self._demo_module: ModuleType | None = None

    def download(self) -> None:
        if self.is_model_dir_complete(self.model_dir):
            return

        # Download from the upstream repository, but keep the local directory
        # under the configured Immich model name to avoid colliding with built-ins.
        download_axera_repo(self.repo_name, self.model_dir)

    def detect(
        self,
        session: ModelSession,
        img: Image.Image,
        min_score: float,
        max_resolution: int,
    ) -> TextDetectionOutput:
        img_cv = pil_to_cv2(img)
        boxes, scores = self.detect_with_scores(session, img_cv, min_score, max_resolution)
        if boxes is None or len(boxes) == 0:
            return self.empty_detection()

        boxes_np = np.array(boxes, dtype=np.float32)
        scores_np = np.array(scores, dtype=np.float32)
        return {
            "boxes": boxes_np,
            "scores": scores_np,
        }

    def recognize(
        self, session: ModelSession, img: Image.Image, texts: TextDetectionOutput, min_score: float
    ) -> TextRecognitionOutput:
        boxes, box_scores = texts["boxes"].copy(), texts["scores"]
        if boxes.shape[0] == 0:
            return self.empty_recognition()

        img_cv = pil_to_cv2(img)
        img_crop_list = [self.demo.get_rotate_crop_image(img_cv, box.astype(np.float32)) for box in boxes]
        if self.cls_session is not None:
            img_crop_list, _ = self.demo.text_classifier(self.cls_session, img_crop_list, [3, 80, 160])
        rec_res = self.demo.text_recognizer(session, img_crop_list, [3, 48, 320], self.character_dict_path.as_posix())

        text_scores = np.array([score for _, score in rec_res], dtype=np.float32)
        valid_text_score_idx = text_scores > min_score
        if not valid_text_score_idx.any():
            return self.empty_recognition()

        boxes[:, :, 0] /= img.width
        boxes[:, :, 1] /= img.height

        return {
            "box": boxes.reshape(-1, 8)[valid_text_score_idx].reshape(-1),
            "text": [text for i, (text, _) in enumerate(rec_res) if valid_text_score_idx[i]],
            "boxScore": box_scores[valid_text_score_idx],
            "textScore": text_scores[valid_text_score_idx],
        }

    def detect_with_scores(
        self,
        session: ModelSession,
        img: NDArray[np.uint8],
        min_score: float,
        max_resolution: int,
    ) -> tuple[NDArray[np.int32] | None, list[float]]:
        shape = self.get_detection_input_size(session)
        canvas, scale, pad_left, pad_top = self.resize_and_pad(img, shape, max_resolution)
        orig_h, orig_w = img.shape[:2]
        canvas_h, canvas_w = canvas.shape[:2]
        image = canvas.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0).astype(np.float32)

        det_out = session.run(None, input_feed={"x": image})
        pred = det_out[0][:, 0, :, :]
        segmentation = pred > 0.3

        boxes, scores = self.boxes_from_bitmap(pred[0], segmentation[0], canvas_w, canvas_h, min_score)
        boxes, scores = self.filter_detect_results(boxes, scores, canvas.shape)
        boxes = self.map_boxes_to_original(boxes, scale, pad_left, pad_top, orig_w, orig_h)
        boxes, scores = self.filter_detect_results(boxes, scores, img.shape)
        if len(boxes) == 0:
            return None, []

        order = self.sorted_box_indices(boxes)
        return boxes[order], [scores[i] for i in order]

    def resize_and_pad(
        self,
        img: NDArray[np.uint8],
        input_size: tuple[int, int],
        max_resolution: int,
    ) -> tuple[NDArray[np.uint8], float, int, int]:
        input_w, input_h = input_size
        orig_h, orig_w = img.shape[:2]
        # AXERA axmodels require a fixed input tensor shape. Keep the model input
        # size unchanged, and apply Immich's maxResolution to the scaled image
        # content before padding it into the fixed canvas.
        content_limit = max(1, min(int(max_resolution), input_w, input_h))
        scale = min(content_limit / max(orig_w, orig_h), input_w / orig_w, input_h / orig_h)

        resize_w = max(1, int(round(orig_w * scale)))
        resize_h = max(1, int(round(orig_h * scale)))
        resized = cv2.resize(img, (resize_w, resize_h))

        pad_left = (input_w - resize_w) // 2
        pad_top = (input_h - resize_h) // 2
        canvas = np.zeros((input_h, input_w, img.shape[2]), dtype=img.dtype)
        canvas[pad_top : pad_top + resize_h, pad_left : pad_left + resize_w] = resized
        return canvas, scale, pad_left, pad_top

    def map_boxes_to_original(
        self,
        boxes: NDArray[np.int32],
        scale: float,
        pad_left: int,
        pad_top: int,
        orig_w: int,
        orig_h: int,
    ) -> NDArray[np.int32]:
        if len(boxes) == 0:
            return boxes

        mapped = boxes.astype(np.float32)
        mapped[:, :, 0] = (mapped[:, :, 0] - pad_left) / scale
        mapped[:, :, 1] = (mapped[:, :, 1] - pad_top) / scale
        mapped[:, :, 0] = np.clip(np.round(mapped[:, :, 0]), 0, orig_w)
        mapped[:, :, 1] = np.clip(np.round(mapped[:, :, 1]), 0, orig_h)
        return mapped.astype("int32")

    def get_detection_input_size(self, session: ModelSession) -> tuple[int, int]:
        input_shape = session.get_inputs()[0].shape
        if len(input_shape) != 4:
            raise ValueError(f"Unsupported AXERA PPOCR detection input shape: {input_shape}")

        if input_shape[1] in {1, 3}:
            height, width = input_shape[2], input_shape[3]
        elif input_shape[3] in {1, 3}:
            height, width = input_shape[1], input_shape[2]
        else:
            raise ValueError(f"Unsupported AXERA PPOCR detection input shape: {input_shape}")

        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError(f"AXERA PPOCR detection requires a static input shape, got: {input_shape}")

        return width, height

    def boxes_from_bitmap(
        self,
        pred: NDArray[np.float32],
        bitmap: NDArray[np.bool_],
        dest_width: int,
        dest_height: int,
        box_thresh: float,
    ) -> tuple[NDArray[np.int32], list[float]]:
        max_candidates = 1000
        unclip_ratio = 1.5
        min_size = 3
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = outs[1] if len(outs) == 3 else outs[0]
        num_contours = min(len(contours), max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.demo.get_mini_boxes(contour)
            if sside < min_size:
                continue

            points = np.array(points)
            score = self.demo.box_score_fast(pred, points.reshape(-1, 2))
            if score < box_thresh:
                continue

            box = self.demo.unclip(points, unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.demo.get_mini_boxes(box)
            if sside < min_size + 2:
                continue

            box_np = np.array(box)
            box_np[:, 0] = np.clip(np.round(box_np[:, 0] / width * dest_width), 0, dest_width)
            box_np[:, 1] = np.clip(np.round(box_np[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box_np.astype("int32"))
            scores.append(float(score))

        return np.array(boxes, dtype="int32"), scores

    def filter_detect_results(
        self,
        boxes: NDArray[np.int32],
        scores: list[float],
        image_shape: tuple[int, ...],
    ) -> tuple[NDArray[np.int32], list[float]]:
        img_height, img_width = image_shape[0:2]
        filtered_boxes = []
        filtered_scores = []

        for box, score in zip(boxes, scores, strict=False):
            box = self.demo.order_points_clockwise(box)
            box = self.demo.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            filtered_boxes.append(box)
            filtered_scores.append(score)

        return np.array(filtered_boxes, dtype="int32"), filtered_scores

    def sorted_box_indices(self, boxes: NDArray[np.int32]) -> list[int]:
        if len(boxes) == 0:
            return []
        sorted_boxes = self.demo.sorted_boxes(boxes)
        used: set[int] = set()
        order = []
        for sorted_box in sorted_boxes:
            for index, box in enumerate(boxes):
                if index not in used and np.array_equal(sorted_box, box):
                    used.add(index)
                    order.append(index)
                    break
        return order

    @property
    def demo(self) -> ModuleType:
        if self._demo_module is None:
            demo_path = self.model_dir / "infer_axmodel.py"
            spec = importlib.util.spec_from_file_location("axera_ppocr_infer_axmodel", demo_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load AXERA PPOCR demo module from {demo_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._demo_module = module
        return self._demo_module

    def empty_detection(self) -> TextDetectionOutput:
        return {
            "boxes": np.empty(0, dtype=np.float32),
            "scores": np.empty(0, dtype=np.float32),
        }

    def empty_recognition(self) -> TextRecognitionOutput:
        return {
            "box": np.empty(0, dtype=np.float32),
            "boxScore": np.empty(0, dtype=np.float32),
            "text": [],
            "textScore": np.empty(0, dtype=np.float32),
        }

    @property
    def model_dir(self) -> Path:
        return self.cache_dir

    def is_model_dir_complete(self, model_dir: Path) -> bool:
        if not (model_dir / "infer_axmodel.py").is_file() or not (model_dir / "ppocrv5_dict.txt").is_file():
            return False

        model_paths = self.get_platform_model_paths(model_dir)
        return (
            (model_dir / model_paths["det"]).is_file()
            and (model_dir / model_paths["rec"]).is_file()
            and (model_dir / model_paths["cls"]).is_file()
        )

    def get_platform_model_paths(self, model_dir: Path) -> dict[str, str]:
        config_path = model_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"AXERA PPOCR config file not found: {config_path}")

        config = json.loads(config_path.read_text())
        platforms = config.get("supported_platforms", [])
        if len(platforms) == 0:
            raise ValueError(f"No supported_platforms found in {config_path}")

        current_chip = self.detect_chip()
        if current_chip == "":
            supported = ", ".join(platform.get("chip", "") for platform in platforms)
            raise ValueError(
                f"Unable to detect AXERA chip for {config_path}. "
                f"Set AXERA_CHIP to one of: {supported}"
            )

        for platform in platforms:
            if platform.get("chip") == current_chip:
                return platform["models"]

        supported = ", ".join(platform.get("chip", "") for platform in platforms)
        raise ValueError(f"Unsupported AXERA chip '{current_chip}' for {config_path}. Supported chips: {supported}")

    def detect_chip(self) -> str:
        chip = os.environ.get("AXERA_CHIP")
        if chip is None:
            chip_type_path = Path("/proc/ax_proc/chip_type")
            chip = chip_type_path.read_text().strip() if chip_type_path.is_file() else ""

        chip = chip.upper()
        if "AX650" in chip or "MC50" in chip:
            return "AX650"
        if "AX615" in chip:
            return "AX615"
        if "AX630" in chip or "AX620" in chip or "MC20" in chip:
            return "AX630C"
        return chip

    @property
    def det_model_path(self) -> Path:
        return self.model_dir / self.get_platform_model_paths(self.model_dir)["det"]

    @property
    def rec_model_path(self) -> Path:
        return self.model_dir / self.get_platform_model_paths(self.model_dir)["rec"]

    @property
    def cls_model_path(self) -> Path:
        return self.model_dir / self.get_platform_model_paths(self.model_dir)["cls"]

    @property
    def character_dict_path(self) -> Path:
        return self.model_dir / "ppocrv5_dict.txt"
