"""Types shared by OCR backends, the OCR client and the layout step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class OcrError(RuntimeError):
    """A model call failed after all retries."""


@dataclass
class Completion:
    text: str
    finish_reason: str = "stop"  # "stop" | "length" | "error"
    model: str = ""
    elapsed_s: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    # with ocr.logprobs: each output token and its top alternatives, {"t", "lp", "alt": [[t, lp], ...]}
    tokens: list[dict] | None = None


class Backend(Protocol):
    """Something that turns (image, prompts) into text. See ocr/backends.py."""

    name: str

    async def complete(
        self,
        image_png: bytes,
        system: str,
        user: str,
        *,
        request_id: str,
        max_tokens: int,
        temperature: float,
        extra: dict[str, Any],
    ) -> Completion: ...

    async def aclose(self) -> None: ...


class Reader(Protocol):
    """What the layout step needs from ``ocr.client.OcrClient`` (lets tests pass a fake)."""

    async def read(
        self, image: Any, system: str, user: str, *, request_id: str, max_side: int | None = None
    ) -> tuple[Completion, bool]: ...
