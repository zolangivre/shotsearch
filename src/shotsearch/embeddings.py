"""Text and image embeddings, computed locally on the Apple GPU when available.

The only network access in shotsearch is `download_models()`. Once the models are
cached, Hugging Face is switched to offline mode before it is even imported.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from shotsearch.config import models_dir

if TYPE_CHECKING:
    from PIL import Image


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    dim: int


TEXT_MODEL = ModelSpec("intfloat/multilingual-e5-small", 384)
IMAGE_MODEL = ModelSpec("google/siglip2-base-patch16-256", 768)
MODELS = (TEXT_MODEL, IMAGE_MODEL)

# Top-level weights, configs and tokenizers only: skip ONNX/OpenVINO copies.
_ALLOW = ["*.json", "*.safetensors", "*.model"]
_IGNORE = ["*/*"]

# SigLIP was trained on text padded to 64 tokens; other paddings hurt accuracy.
_SIGLIP_TEXT_LEN = 64
_E5_MAX_LEN = 512


def _configure_hub(offline: bool) -> None:
    os.environ["HF_HOME"] = str(models_dir())
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)


def missing_models() -> list[str]:
    """Repos not yet in the local cache. Never touches the network."""
    _configure_hub(offline=True)
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    missing = []
    for spec in MODELS:
        try:
            snapshot_download(
                spec.repo, allow_patterns=_ALLOW, ignore_patterns=_IGNORE, local_files_only=True
            )
        except (LocalEntryNotFoundError, FileNotFoundError, OSError):
            missing.append(spec.repo)
    return missing


def download_models(repos: Sequence[str]) -> None:
    """Fetch models into the local cache: the one network access shotsearch makes."""
    _configure_hub(offline=False)
    from huggingface_hub import snapshot_download

    for repo in repos:
        snapshot_download(repo, allow_patterns=_ALLOW, ignore_patterns=_IGNORE)
    _configure_hub(offline=True)


class ModelsNotInstalledError(RuntimeError):
    def __init__(self, missing: list[str]) -> None:
        super().__init__(
            "Models not downloaded yet: " + ", ".join(missing) + ". Run `shotsearch setup`."
        )
        self.missing = missing


def device_name() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def chunk_text(text: str, max_words: int = 120, overlap: int = 20) -> list[str]:
    """Split OCR text into overlapping word windows that fit e5's context."""
    words = text.split()
    if not words:
        return []
    step = max_words - overlap
    return [
        " ".join(words[start : start + max_words])
        for start in range(0, max(len(words) - overlap, 1), step)
    ]


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class Embedder:
    """Loads both models once. Construct lazily: loading takes a few seconds."""

    text_spec = TEXT_MODEL
    image_spec = IMAGE_MODEL

    def __init__(self) -> None:
        missing = missing_models()
        if missing:
            raise ModelsNotInstalledError(missing)
        _configure_hub(offline=True)

        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        self._torch = torch
        self.device = device_name()
        self._text_tok = AutoTokenizer.from_pretrained(TEXT_MODEL.repo)
        self._text_model = AutoModel.from_pretrained(TEXT_MODEL.repo).to(self.device).eval()
        self._image_proc = AutoProcessor.from_pretrained(IMAGE_MODEL.repo)
        self._image_model = AutoModel.from_pretrained(IMAGE_MODEL.repo).to(self.device).eval()

    # -- e5: OCR text -------------------------------------------------------

    def embed_passages(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return self._embed_e5([f"passage: {t}" for t in texts], batch_size)

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed_e5([f"query: {query}"], 1)[0]

    def _embed_e5(self, texts: list[str], batch_size: int) -> np.ndarray:
        torch = self._torch
        out = []
        for start in range(0, len(texts), batch_size):
            batch = self._text_tok(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=_E5_MAX_LEN,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                hidden = self._text_model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.float().cpu().numpy())
        if not out:
            return np.zeros((0, TEXT_MODEL.dim), dtype=np.float32)
        return _l2_normalize(np.concatenate(out)).astype(np.float32)

    # -- SigLIP: image <-> text --------------------------------------------

    def embed_images(self, images: Sequence[Image.Image], batch_size: int = 16) -> np.ndarray:
        torch = self._torch
        out = []
        for start in range(0, len(images), batch_size):
            inputs = self._image_proc(
                images=list(images[start : start + batch_size]), return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                feats = _features(self._image_model.get_image_features(**inputs))
            out.append(feats.float().cpu().numpy())
        if not out:
            return np.zeros((0, IMAGE_MODEL.dim), dtype=np.float32)
        return _l2_normalize(np.concatenate(out)).astype(np.float32)

    def embed_image_query(self, query: str) -> np.ndarray:
        torch = self._torch
        inputs = self._image_proc(
            text=[query.lower()],
            padding="max_length",
            max_length=_SIGLIP_TEXT_LEN,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            feats = _features(self._image_model.get_text_features(**inputs))
        return _l2_normalize(feats.float().cpu().numpy()).astype(np.float32)[0]


def _features(output):
    """transformers may return a tensor or a model output, depending on version."""
    return output if hasattr(output, "shape") else output.pooler_output
