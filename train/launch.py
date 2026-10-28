"""Start a training run on the deployed Modal app, so it does not depend on this machine's session.

    python -m train.launch --run ft2 [--bundle bundle-v2] [--epochs 1] [--lr 1e-4] [--rank 16] [--vision]
    python -m train.launch --log ft2          # print the run's log from the models volume
    python -m train.launch --cancel fc-...    # stop a run (its container too, so Modal does not retry it)

The bundle must already be in the ``pdf2epub-train`` volume (``modal volume put pdf2epub-train
training/bundle-v2 /bundle-v2``). A run started from ``modal run`` stops when the local client's
login expires; a spawned call on a deployed app does not.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--bundle", default="bundle-v2")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--base", default="Qwen/Qwen3-VL-8B-Instruct", help="model to continue from, e.g. /models/ft2/merged")
    ap.add_argument("--log")
    ap.add_argument("--cancel", metavar="CALL_ID")
    args = ap.parse_args()
    modal_bin = str(Path(sys.executable).with_name("modal"))
    env = {**os.environ, "COLUMNS": "500", "TERM": "dumb"}
    if args.cancel:
        import modal

        # Without terminate_containers the running input is only interrupted, and Modal may
        # start it again from the beginning.
        modal.FunctionCall.from_id(args.cancel).cancel(terminate_containers=True)
        print(f"cancelled {args.cancel}")
        return
    if args.log:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([modal_bin, "volume", "get", "--force", "pdf2epub-models", f"{args.log}/log.jsonl", tmp],
                           check=True, env=env, capture_output=True)
            print((Path(tmp) / "log.jsonl").read_text(encoding="utf-8"))
        return
    subprocess.run([modal_bin, "deploy", "train/finetune_modal.py"], check=True, env=env)
    import modal

    train = modal.Function.from_name("pdf2epub-finetune", "train")
    call = train.spawn(args.bundle, args.run, epochs=args.epochs, lr=args.lr, rank=args.rank, accum=8,
                       vision=args.vision, eval_every=args.eval_every, base=args.base)
    print(f"started {args.run}: call {call.object_id}")


if __name__ == "__main__":
    main()
