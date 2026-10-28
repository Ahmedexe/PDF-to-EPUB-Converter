"""OCR backends: an OpenAI-compatible endpoint (vLLM), recorded fixtures, Tesseract.

``request_id`` names what is being read: ``NNNN`` for a whole page,
``NNNN.<region>`` for a region, ``NNNN.layout`` for a layout call. The fixture
backend looks up exactly those names.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import random
import re
import time
from pathlib import Path
from typing import Any

import httpx

from ..config import OcrSettings, default_tessdata_dir
from .types import Backend, Completion, OcrError

log = logging.getLogger(__name__)
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}


class OpenAIBackend:
    """``POST {endpoint}/chat/completions`` with the page as a base64 PNG data URL."""

    name = "openai"

    def __init__(self, settings: OcrSettings, *, transport: httpx.AsyncBaseTransport | None = None,
                 sleep: Any = asyncio.sleep):
        from .models import key_for, recorded_endpoint

        endpoint = settings.endpoint or recorded_endpoint(settings.model)
        if not endpoint:
            key = key_for(settings.model)
            hint = f"run  pdf2epub bench deploy {key}  or " if key else ""
            raise OcrError(f"no endpoint for {settings.model}: {hint}pass --endpoint (an OpenAI-compatible URL ending in /v1)")
        self.settings = settings
        self.url = endpoint.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        key = settings.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        # follow_redirects: serverless platforms answer requests that run long (Modal: over 150 s)
        # with a 303 to a URL where the result appears.
        self.client = httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(settings.timeout_s, connect=30.0),
                                        transport=transport, follow_redirects=True)
        self._sleep = sleep

    async def complete(self, image_png: bytes, system: str, user: str, *, request_id: str, max_tokens: int,
                       temperature: float, extra: dict[str, Any]) -> Completion:
        data_url = "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
        if image_png[:3] == b"\xff\xd8\xff":
            data_url = "data:image/jpeg;base64," + base64.b64encode(image_png).decode("ascii")
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": user},
                ]},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        payload.update(extra or {})
        if self.settings.logprobs:
            payload.update({"logprobs": True, "top_logprobs": self.settings.logprobs})
        attempts = max(1, self.settings.retries + 1)
        last_error = ""
        for attempt in range(attempts):
            start = time.monotonic()
            try:
                resp = await self.client.post(self.url, json=payload)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                await self._backoff(attempt, None)
                continue
            if resp.status_code in RETRY_STATUS:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                await self._backoff(attempt, resp.headers.get("retry-after"))
                continue
            if resp.status_code >= 400:
                raise OcrError(f"{request_id}: HTTP {resp.status_code}: {resp.text[:300]}")
            try:
                body = resp.json()
                choice = body["choices"][0]
                content = choice["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError):
                # An empty or cut-off body usually means the server container died mid-request
                # (serverless restarts): worth another try.
                last_error = f"unexpected response (HTTP {resp.status_code}): {resp.text[:200]!r}"
                await self._backoff(attempt, None)
                continue
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            tokens = None
            if self.settings.logprobs and isinstance(choice.get("logprobs"), dict):
                tokens = [{"t": t["token"], "lp": t["logprob"],
                           "alt": [[a["token"], a["logprob"]] for a in t.get("top_logprobs") or []]}
                          for t in choice["logprobs"].get("content") or []]
                while tokens and re.fullmatch(r"<\|[^|]+\|>", tokens[-1]["t"]):  # end-of-answer marker
                    tokens.pop()
            return Completion(text=content or "", finish_reason=choice.get("finish_reason") or "stop",
                              model=body.get("model", self.settings.model), elapsed_s=time.monotonic() - start,
                              usage={k: v for k, v in (body.get("usage") or {}).items() if isinstance(v, int)},
                              tokens=tokens)
        raise OcrError(f"{request_id}: failed after {attempts} attempts ({last_error})")

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        if attempt >= self.settings.retries:
            return
        delay = min(60.0, 2.0 ** attempt + random.random())
        if retry_after:
            try:
                delay = max(delay, min(120.0, float(retry_after)))
            except ValueError:
                pass
        await self._sleep(delay)

    async def aclose(self) -> None:
        await self.client.aclose()


class FixtureBackend:
    """Replays recorded model output: ``<fixtures>/<request_id>.tags`` (or ``.json``/``.txt``)."""

    name = "fixture"

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise OcrError(f"fixture folder not found: {self.folder}")

    async def complete(self, image_png: bytes, system: str, user: str, *, request_id: str, max_tokens: int,
                       temperature: float, extra: dict[str, Any]) -> Completion:
        for suffix in (".tags", ".json", ".txt"):
            path = self.folder / f"{request_id}{suffix}"
            if path.exists():
                return Completion(text=path.read_text(encoding="utf-8"), model="fixture")
        if request_id.endswith(".layout"):
            return Completion(text='{"regions": []}', model="fixture")
        raise FileNotFoundError(f"no fixture for {request_id} in {self.folder}")

    async def aclose(self) -> None:
        return None


class TesseractBackend:
    """Offline dry runs: Tesseract text wrapped as ``<p>`` paragraphs (no structure, no notes)."""

    name = "tesseract"

    def __init__(self, lang: str = "ara", tessdata: str | None = None):
        self.lang = lang
        self.tessdata = tessdata or default_tessdata_dir()

    async def complete(self, image_png: bytes, system: str, user: str, *, request_id: str, max_tokens: int,
                       temperature: float, extra: dict[str, Any]) -> Completion:
        if request_id.endswith(".layout"):
            return Completion(text='{"regions": []}', model="tesseract")
        from xml.sax.saxutils import escape

        import cv2
        import numpy as np
        import pytesseract

        img = cv2.imdecode(np.frombuffer(image_png, np.uint8), cv2.IMREAD_GRAYSCALE)
        config = f'--tessdata-dir "{self.tessdata}"' if self.tessdata else ""
        start = time.monotonic()
        text = await asyncio.to_thread(pytesseract.image_to_string, img, lang=self.lang, config=config)
        paras = [" ".join(chunk.split()) for chunk in text.split("\n\n") if chunk.strip()]
        tags = "\n".join(f"<p>{escape(p)}</p>" for p in paras)
        return Completion(text=tags, model=f"tesseract-{self.lang}", elapsed_s=time.monotonic() - start)

    async def aclose(self) -> None:
        return None


def make_backend(settings: OcrSettings) -> Backend:
    if settings.backend == "fixture":
        if not settings.fixtures:
            raise OcrError("the fixture backend needs --fixtures FOLDER")
        return FixtureBackend(settings.fixtures)
    if settings.backend == "tesseract":
        return TesseractBackend()
    return OpenAIBackend(settings)


__all__ = ["Backend", "Completion", "FixtureBackend", "OcrError", "OpenAIBackend", "TesseractBackend", "make_backend"]
