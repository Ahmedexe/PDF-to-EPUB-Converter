"""LoRA fine-tuning of Qwen3-VL-8B on page/transcription pairs, on a Modal GPU (spec §8).

    .venv/bin/python -m train.launch --bundle bundle-v2 --run ft2          # deployed: survives the client
    .venv/bin/modal run --detach train/finetune_modal.py --bundle training/bundle-v1 --run ft1
    .venv/bin/modal run train/finetune_modal.py --bundle - --run ft1 --merge-only   # merge the last saved adapter

The bundle (from ``python -m train.bundle``) is uploaded to the ``pdf2epub-train``
volume once. Training is plain transformers + PEFT: LoRA on the language model's
linear layers (``--vision`` adds the vision tower's), bf16 weights, batch 1 with
gradient accumulation, loss on the transcription only. Each epoch saves the
adapter; at the end the adapter is merged into 16-bit weights at
``/models/<run>/merged`` in the ``pdf2epub-models`` volume, which
``serve_modal.py`` can serve directly (``PDF2EPUB_MODEL=/models/<run>/merged``).

Validation loss on the held-out real pages is printed every ``--eval-every``
steps and written to ``/models/<run>/log.jsonl``; the real measure is
``pdf2epub bench report --reference`` on the unseen test set afterwards.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

BASE = "Qwen/Qwen3-VL-8B-Instruct"
GPU = os.environ.get("PDF2EPUB_TRAIN_GPU", "H100")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", "torchvision==0.29.0", "transformers==5.17.0", "peft==0.21.2",
                 "accelerate==1.15.0", "pillow", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/huggingface",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)
hf_cache = modal.Volume.from_name("pdf2epub-hf-cache", create_if_missing=True)
train_vol = modal.Volume.from_name("pdf2epub-train", create_if_missing=True)
models_vol = modal.Volume.from_name("pdf2epub-models", create_if_missing=True)
app = modal.App("pdf2epub-finetune")

LANGUAGE_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
VISION_TARGETS = r".*visual\.blocks\.\d+\.(attn\.qkv|attn\.proj|mlp\.linear_fc1|mlp\.linear_fc2)$"


@app.function(image=image, gpu=GPU, timeout=24 * 60 * 60,
              volumes={"/cache/huggingface": hf_cache, "/data": train_vol, "/models": models_vol})
def train(bundle: str, run: str, epochs: float = 1.0, lr: float = 1e-4, rank: int = 16, accum: int = 8,
          max_length: int = 12288, vision: bool = False, eval_every: int = 200, limit: int = 0) -> dict:
    import math
    import random
    import time

    import torch
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

    root = Path("/data") / bundle
    out = Path("/models") / run
    out.mkdir(parents=True, exist_ok=True)
    log_file = (out / "log.jsonl").open("a", encoding="utf-8")

    def log(**rec):
        rec["time"] = round(time.time())
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        log_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log_file.flush()

    def load(name: str) -> list[dict]:
        return [json.loads(x) for x in (root / name).read_text(encoding="utf-8").splitlines() if x.strip()]

    train_items, val_items = load("train.jsonl"), load("val.jsonl")
    if limit:
        train_items = train_items[:limit]

    processor = AutoProcessor.from_pretrained(BASE)
    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                        device_map={"": 0})
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    targets = LANGUAGE_TARGETS if not vision else f"({LANGUAGE_TARGETS})|({VISION_TARGETS})"
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                                             target_modules=targets, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(event="start", bundle=bundle, train=len(train_items), val=len(val_items), trainable=trainable,
        epochs=epochs, lr=lr, rank=rank, accum=accum, vision=vision)

    def encode(item: dict) -> dict | None:
        img = Image.open(root / item["image"]).convert("RGB")
        prompt = [{"role": "system", "content": [{"type": "text", "text": item["system"]}]},
                  {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": item["user"]}]}]
        full = prompt + [{"role": "assistant", "content": [{"type": "text", "text": item["target"]}]}]
        prompt_text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        full_text = processor.apply_chat_template(full, tokenize=False)
        if not full_text.startswith(prompt_text):
            return None
        batch = processor(text=[full_text], images=[img], return_tensors="pt")
        n_prompt = processor(text=[prompt_text], images=[img], return_tensors="pt")["input_ids"].shape[1]
        ids = batch["input_ids"]
        if ids.shape[1] > max_length:
            return None
        labels = ids.clone()
        labels[:, :n_prompt] = -100
        batch["labels"] = labels
        return {k: v.to(0) for k, v in batch.items()}

    def val_loss() -> float:
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for item in val_items:
                batch = encode(item)
                if batch is None:
                    continue
                total += model(**batch).loss.item()
                n += 1
        model.train()
        return total / max(n, 1)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.999))
    steps_total = math.ceil(len(train_items) * epochs / accum)
    sched = get_cosine_schedule_with_warmup(optim, max(1, round(0.03 * steps_total)), steps_total)
    log(event="val", step=0, loss=round(val_loss(), 4))

    rng = random.Random(0)
    order: list[dict] = []
    while len(order) < len(train_items) * epochs:
        epoch_items = train_items[:]
        rng.shuffle(epoch_items)
        order.extend(epoch_items)
    order = order[: round(len(train_items) * epochs)]

    model.train()
    step, running, seen, skipped = 0, 0.0, 0, 0
    started = time.time()
    for i, item in enumerate(order, 1):
        batch = encode(item)
        if batch is None:
            skipped += 1
        else:
            loss = model(**batch).loss / accum
            loss.backward()
            running += loss.item()
            seen += 1
        if i % accum == 0 or i == len(order):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                rate = (time.time() - started) / i
                log(event="train", step=step, of=steps_total, loss=round(running * accum / max(seen, 1), 4),
                    lr=sched.get_last_lr()[0], s_per_example=round(rate, 2),
                    eta_min=round(rate * (len(order) - i) / 60), skipped=skipped)
                running, seen = 0.0, 0
                if step % 50 == 0:
                    models_vol.commit()  # so the log can be followed with  modal volume get
            if step % eval_every == 0:
                log(event="val", step=step, loss=round(val_loss(), 4))
                model.save_pretrained(out / "adapter")
                models_vol.commit()
    log(event="val", step=step, loss=round(val_loss(), 4))
    model.save_pretrained(out / "adapter")

    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(out / "merged", safe_serialization=True)
    processor.save_pretrained(out / "merged")
    log(event="done", step=step, minutes=round((time.time() - started) / 60), skipped=skipped)
    log_file.close()
    models_vol.commit()
    return {"run": run, "steps": step, "skipped": skipped}


@app.function(image=image, gpu=GPU, timeout=60 * 60,
              volumes={"/cache/huggingface": hf_cache, "/models": models_vol})
def merge(run: str) -> str:
    """Merge a saved adapter (``/models/<run>/adapter``) into 16-bit weights at ``/models/<run>/merged``:
    for a run that stopped early, its last saved adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    out = Path("/models") / run
    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0})
    merged = PeftModel.from_pretrained(model, out / "adapter").merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    AutoProcessor.from_pretrained(BASE).save_pretrained(out / "merged")
    models_vol.commit()
    return str(out / "merged")


@app.local_entrypoint()
def main(bundle: str, run: str, epochs: float = 1.0, lr: float = 1e-4, rank: int = 16, accum: int = 8,
         vision: bool = False, eval_every: int = 200, limit: int = 0, reupload: bool = False,
         merge_only: bool = False):
    if merge_only:
        print(merge.remote(run))
        return
    local = Path(bundle)
    name = local.name
    present = {e.path.strip("/") for e in train_vol.listdir("/")} if not reupload else set()
    if name not in present:
        print(f"uploading {local} to volume pdf2epub-train:/{name}")
        with train_vol.batch_upload(force=True) as up:
            up.put_directory(str(local), f"/{name}")
    print(train.remote(name, run, epochs=epochs, lr=lr, rank=rank, accum=accum, vision=vision,
                       eval_every=eval_every, limit=limit))
