"""A word n-gram language model of classical Arabic, built on Modal from the OpenITI corpus.

    .venv/bin/modal run --detach train/lm_modal.py::build            # once: corpus + KenLM model
    python -m train.lm_status                                       # (see the log in the volume)

The corpus is every OpenITI text up to ``LAST_PERIOD`` (github.com/OpenITI/<period>AH), one
version per book (the longest), with the test books left out so scores stay honest. Words are
normalised as the scorer does (pdf2epub.arabic.normalize_for_match: no diacritics, alef forms,
ى/ي and ة/ه unified), so the model tells apart exactly what counts as an error. KenLM builds a
pruned ``ORDER``-gram model and a quantised binary. Everything stays in the ``pdf2epub-lm``
volume: ``corpus.txt``, ``vocab.tsv`` (word, count), ``lm.binary``, ``build.log``.

The deployed class ``Scorer`` (``modal deploy train/lm_modal.py``) loads the model once and
scores word windows for tools/rescore.py and the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import modal

KENLM_COMMIT = "4cb443e60b7bf2c0ddf3c745378f76cb59e254e5"  # 2025-03-30
ORDER = 5
MAX_WORDS = 600_000_000
LAST_PERIOD = 1000
# The test books (sample/test, sample/test2): never in the model, or it would know their text.
EXCLUDE = ("ShicrWaShucara", "CuyunAkhbar")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "cmake", "git", "libboost-program-options-dev", "libboost-system-dev",
                 "libboost-thread-dev", "libboost-test-dev", "zlib1g-dev", "libbz2-dev", "liblzma-dev", "unzip")
    .run_commands(
        f"git clone https://github.com/kpu/kenlm /kenlm && cd /kenlm && git checkout {KENLM_COMMIT}",
        "cd /kenlm && mkdir -p build && cd build && cmake .. && make -j8 lmplz build_binary",
        "pip install /kenlm",
    )
    .pip_install("httpx==0.28.1")
    .add_local_python_source("pdf2epub")
)
vol = modal.Volume.from_name("pdf2epub-lm", create_if_missing=True)
app = modal.App("pdf2epub-lm")


def _clean(text: str) -> list[str]:
    """OpenITI mARkdown -> paragraphs of normalised words."""
    import re

    from pdf2epub.arabic import normalize_for_match

    text = text.split("#META#Header#End#", 1)[-1]
    text = re.sub(r"\s*\n~~", " ", text)
    text = re.sub(r"PageV\d+P\d+|ms\d+|%~%|@\w+|#+\s*\|*|\|", " ", text)
    out = []
    for para in re.split(r"\n\s*\n|\n#", text):
        words = [w for w in normalize_for_match(para).split() if re.fullmatch(r"[ء-ي]+", w)]
        if len(words) >= 3:
            out.append(" ".join(words))
    return out


@app.function(image=image, cpu=16, memory=98304, timeout=8 * 60 * 60, volumes={"/lm": vol}, ephemeral_disk=512 * 1024)
def build(last_period: int = LAST_PERIOD, max_words: int = MAX_WORDS, order: int = ORDER) -> dict:
    import collections
    import io
    import subprocess
    import time
    import zipfile

    import httpx

    root = Path("/lm")
    log = (root / "build.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        print(msg, flush=True)
        log.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        log.flush()
        vol.commit()

    periods = [f"{y:04d}AH" for y in range(25, last_period + 1, 25)]
    corpus = Path("/tmp/corpus.txt")
    counts: collections.Counter = collections.Counter()
    total = books = 0
    with corpus.open("w", encoding="utf-8") as out:
        for period in periods:
            if total >= max_words:
                break
            url = f"https://codeload.github.com/OpenITI/{period}/zip/refs/heads/master"
            try:
                data = httpx.get(url, timeout=600, follow_redirects=True).content
            except Exception as exc:  # one missing period does not stop the build
                note(f"{period}: download failed ({exc})")
                continue
            z = zipfile.ZipFile(io.BytesIO(data))
            by_book: dict[str, zipfile.ZipInfo] = {}
            for info in z.infolist():
                parts = info.filename.split("/")
                if len(parts) < 5 or parts[1] != "data" or info.is_dir():
                    continue
                name = parts[-1]
                if name.endswith((".yml", ".md")) or "-ara" not in name or any(x in name for x in EXCLUDE):
                    continue
                book = "/".join(parts[2:4])
                if book not in by_book or info.file_size > by_book[book].file_size:
                    by_book[book] = info
            before = total
            for info in by_book.values():
                for para in _clean(z.read(info).decode("utf-8", "replace")):
                    out.write(para + "\n")
                    words = para.split()
                    counts.update(words)
                    total += len(words)
                books += 1
                if total >= max_words:
                    break
            note(f"{period}: {len(by_book)} books, {total - before:,} words (total {total:,})")
            del z, data
    with (root / "vocab.tsv").open("w", encoding="utf-8") as f:
        for w, n in counts.most_common():
            if n >= 2:
                f.write(f"{w}\t{n}\n")
    note(f"corpus: {books} books, {total:,} words, {len(counts):,} distinct words")
    arpa = Path("/tmp/lm.arpa")
    prune = " ".join(["0", "0"] + ["1"] * (order - 2))
    subprocess.run(f"/kenlm/build/bin/lmplz -o {order} --prune {prune} -S 70% -T /tmp < {corpus} > {arpa}",
                   shell=True, check=True)
    note(f"arpa: {arpa.stat().st_size / 1e9:.1f} GB")
    subprocess.run(["/kenlm/build/bin/build_binary", "-q", "8", "-b", "8", "trie", str(arpa), str(root / "lm.binary")],
                   check=True)
    note(f"lm.binary: {(root / 'lm.binary').stat().st_size / 1e9:.1f} GB")
    log.close()
    vol.commit()
    return {"books": books, "words": total, "vocab": len(counts)}


@app.cls(image=image, cpu=2, memory=16384, volumes={"/lm": vol}, scaledown_window=300, timeout=3600)
class Scorer:
    """The model, loaded once per container: log10 scores of word windows."""

    @modal.enter()
    def load(self) -> None:
        import kenlm

        self.model = kenlm.Model("/lm/lm.binary")

    @modal.method()
    def score(self, windows: list[str]) -> list[float]:
        """log10 probability of each window (normalised words, space separated), with no sentence
        boundary markers: the window is a slice of running text."""
        return [self.model.score(w, bos=False, eos=False) for w in windows]
