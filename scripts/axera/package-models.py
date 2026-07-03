from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = [
    {
        "repo": "AXERA-TECH/ViT-L-14-336-CN__axera",
        "local": Path("clip") / "ViT-L-14-336-CN__axera",
    },
    {
        "repo": "AXERA-TECH/PPOCR_v5",
        "local": Path("ocr") / "PPOCR_v5__axera",
    },
    {
        "repo": "AXERA-TECH/Insightface",
        "local": Path("facial-recognition") / "buffalo_l__axera",
    },
]


def remove_dir(path: Path) -> None:
    if not path.is_dir():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    cache_dir = output_dir / "model-cache"
    hf_cache_dir = output_dir / "hf-cache"
    package_path = output_dir / f"immich-axera-models-{args.version}.tar.gz"

    for cleanup_dir in (cache_dir, hf_cache_dir):
        remove_dir(cleanup_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        local_dir = cache_dir / model["local"]
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(model["repo"], cache_dir=hf_cache_dir, local_dir=local_dir)

    for extra_dir in [*cache_dir.rglob(".cache"), *cache_dir.rglob("huggingface")]:
        remove_dir(extra_dir)

    with tarfile.open(package_path, "w:gz") as tar:
        for child in sorted(cache_dir.iterdir()):
            tar.add(child, arcname=child.name)


if __name__ == "__main__":
    main()
