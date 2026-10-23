"""Serve an OCR model on Modal with vLLM's OpenAI-compatible server (spec §4, §10).

Pay-per-second on a serverless GPU that scales to zero when no book is being
converted. Weights are cached in a Modal Volume so cold starts do not
re-download them; they are served in 16-bit (bf16): 4-bit quantisation is not
recommended for fine-grained OCR such as diacritics (spec §8 step 6).

Easiest: ``pdf2epub bench deploy qwen3-vl-8b`` (see ``pdf2epub/bench.py``),
which sets the variables below and records the URL. By hand::

    pip install modal && modal setup             # once: log in
    modal secret create pdf2epub-api-key PDF2EPUB_API_KEY=<long random key>
    PDF2EPUB_MODEL=Qwen/Qwen3-VL-8B-Instruct modal deploy pdf2epub/ocr/serve_modal.py

Each model deploys as its own app (``pdf2epub-<model>``), so several can be
compared side by side. Variables read at deploy time: ``PDF2EPUB_MODEL``
(Hugging Face id), ``PDF2EPUB_GPU`` (default L40S), ``PDF2EPUB_MAX_MODEL_LEN``
(16384), ``PDF2EPUB_IDLE_SECONDS`` (scale-down delay, 300).

The same server runs anywhere vLLM does (a rented pod, a Kaggle notebook)::

    vllm serve Qwen/Qwen3-VL-8B-Instruct --dtype bfloat16 --max-model-len 16384 \\
        --limit-mm-per-prompt '{"image": 1, "video": 0}' --api-key $PDF2EPUB_API_KEY
"""

from __future__ import annotations

import os
import re

MODEL = os.environ.get("PDF2EPUB_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
GPU = os.environ.get("PDF2EPUB_GPU", "L40S")
MAX_MODEL_LEN = int(os.environ.get("PDF2EPUB_MAX_MODEL_LEN", "16384"))
IDLE_SECONDS = int(os.environ.get("PDF2EPUB_IDLE_SECONDS", "300"))
VLLM_VERSION = os.environ.get("PDF2EPUB_VLLM_VERSION", "0.30.0")
PORT = 8000


def app_name(model: str = MODEL) -> str:
    """Modal app name for a model: ``pdf2epub-qwen3-vl-8b-instruct``, ``pdf2epub-ft1`` for /models/ft1/merged."""
    parts = model.strip("/").split("/")
    name = parts[1] if parts[0] == "models" and len(parts) > 1 else parts[-1]  # /models/<run>/merged
    short = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"pdf2epub-{short}"[:60]


try:  # Modal is only needed to deploy; importing this module elsewhere must not require it.
    import modal
except ImportError:  # pragma: no cover - exercised only without modal installed
    modal = None


def vllm_command(api_key: str | None) -> list[str]:
    """The ``vllm serve`` command line (also used by tests)."""
    cmd = [
        "vllm", "serve", MODEL,
        "--served-model-name", MODEL,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--dtype", "bfloat16",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--limit-mm-per-prompt", '{"image": 1, "video": 0}',
        "--gpu-memory-utilization", "0.92",
        "--max-num-seqs", "16",
        "--trust-remote-code",
        "--uvicorn-log-level", "warning",
    ]
    if api_key:
        cmd += ["--api-key", api_key]
    return cmd


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/huggingface",
              "PDF2EPUB_MODEL": MODEL, "PDF2EPUB_MAX_MODEL_LEN": str(MAX_MODEL_LEN),
              # FlashInfer's sampler compiles a kernel on first use and the slim image has no
              # nvcc; vLLM's own sampler gives the same results (greedy decoding at temperature 0).
              "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    )
    hf_cache = modal.Volume.from_name("pdf2epub-hf-cache", create_if_missing=True)
    vllm_cache = modal.Volume.from_name("pdf2epub-vllm-cache", create_if_missing=True)
    # Fine-tuned weights (train/finetune_modal.py): PDF2EPUB_MODEL=/models/<run>/merged
    models_vol = modal.Volume.from_name("pdf2epub-models", create_if_missing=True)
    app = modal.App(app_name())

    @app.function(
        image=image,
        gpu=GPU,
        scaledown_window=IDLE_SECONDS,
        timeout=60 * 60,
        volumes={"/cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache, "/models": models_vol},
        secrets=[modal.Secret.from_name("pdf2epub-api-key")],
    )
    @modal.concurrent(max_inputs=32)
    @modal.web_server(port=PORT, startup_timeout=25 * 60)
    def serve() -> None:
        import subprocess

        subprocess.Popen(vllm_command(os.environ.get("PDF2EPUB_API_KEY")))
