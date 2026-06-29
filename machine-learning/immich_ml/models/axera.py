from pathlib import Path

from huggingface_hub import snapshot_download


AXERA_ORG = "AXERA-TECH"
AXERA_SUFFIX = "__axera"


def remove_axera_suffix(model_name: str) -> str:
    if not model_name.endswith(AXERA_SUFFIX):
        raise ValueError(f"AXERA model name must end with '{AXERA_SUFFIX}': {model_name}")
    return model_name.removesuffix(AXERA_SUFFIX)


def download_axera_repo(repo_name: str, model_dir: Path) -> None:
    snapshot_download(f"{AXERA_ORG}/{repo_name}", cache_dir=model_dir, local_dir=model_dir)
