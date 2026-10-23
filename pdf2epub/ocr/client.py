"""OCR client: image preparation, response cache and concurrency around a backend."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import BookConfig
from ..workdir import WorkDir
from .backends import make_backend
from .prompts import prompt_version
from .types import Backend, Completion

log = logging.getLogger(__name__)


def encode_image(image: np.ndarray | Path, max_side: int, fmt: str = "png") -> tuple[bytes, tuple[int, int]]:
    """Image bytes with the longest side at most ``max_side`` (never upscaled), and the size sent."""
    img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED) if isinstance(image, (str, Path)) else image
    if img is None:
        raise FileNotFoundError(f"cannot read image {image}")
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    if fmt == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise ValueError("image encoding failed")
    return buf.tobytes(), (img.shape[1], img.shape[0])


def cache_key(image_bytes: bytes, model: str, prompt_version: str, system: str, user: str, params: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for part in (model, prompt_version, system, user, json.dumps(params, sort_keys=True, ensure_ascii=False)):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    h.update(image_bytes)
    return h.hexdigest()[:40]


class OcrClient:
    """Reads images through a backend, at most ``ocr.concurrency`` at a time, with a disk cache.

    Failed calls are never cached; ``refresh=True`` skips the cache lookup (but
    stores the new answer).
    """

    def __init__(self, cfg: BookConfig, wd: WorkDir | None, backend: Backend | None = None):
        self.cfg = cfg
        self.wd = wd
        self.backend = backend or make_backend(cfg.ocr)
        self._sem = asyncio.Semaphore(max(1, cfg.ocr.concurrency))
        self.calls = 0
        self.cache_hits = 0

    @property
    def model(self) -> str:
        return "fixture" if self.backend.name == "fixture" else self.cfg.ocr.model

    async def read(self, image: np.ndarray | Path, system: str, user: str, *, request_id: str,
                   max_side: int | None = None, refresh: bool = False,
                   extra: dict | None = None) -> tuple[Completion, bool]:
        ocr = self.cfg.ocr
        data, size = encode_image(image, max_side or ocr.max_side, ocr.image_format)
        body_extra = {**ocr.extra_body, **(extra or {})}
        params = {"max_tokens": ocr.max_tokens, "temperature": ocr.temperature, "extra": body_extra,
                  "salt": ocr.cache_salt, "backend": self.backend.name}
        key = cache_key(data, self.model, prompt_version(self.cfg), system, user, params)
        cache: Path | None = self.wd.ocr_cache_dir / f"{key}.json" if self.wd is not None else None
        if cache is not None and cache.exists() and not refresh:
            try:
                saved = json.loads(cache.read_text(encoding="utf-8"))
                self.cache_hits += 1
                return Completion(**saved), True
            except (ValueError, TypeError):
                log.warning("ignoring a corrupt cache entry %s", cache)
        async with self._sem:
            self.calls += 1
            completion = await self.backend.complete(data, system, user, request_id=request_id,
                                                     max_tokens=ocr.max_tokens, temperature=ocr.temperature,
                                                     extra=body_extra)
        completion.usage = {**completion.usage, "image_w": size[0], "image_h": size[1]}
        if cache is not None and completion.finish_reason != "error":
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp")
            tmp.write_text(json.dumps(completion.__dict__, ensure_ascii=False), encoding="utf-8")
            tmp.replace(cache)
        return completion, False

    async def aclose(self) -> None:
        await self.backend.aclose()
