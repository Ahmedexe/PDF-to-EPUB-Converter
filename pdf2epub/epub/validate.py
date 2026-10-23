"""EPUBCheck validation (spec §7 rule 8)."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..config import default_epubcheck_jar


@dataclass
class EpubcheckReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fatal: list[str] = field(default_factory=list)
    available: bool = True  # False when Java or the jar is missing
    path: str = ""

    def summary(self) -> str:
        if not self.available:
            return "EPUBCheck not available (needs Java and vendor/epubcheck/epubcheck.jar)"
        state = "passed" if self.ok else "FAILED"
        return f"EPUBCheck {state}: {len(self.fatal)} fatal, {len(self.errors)} errors, {len(self.warnings)} warnings"


def run_epubcheck(epub_path: Path, jar: str | None = None, *, timeout: float = 300) -> EpubcheckReport:
    """Validate one EPUB with EPUBCheck's JSON report."""
    jar = jar or default_epubcheck_jar()
    java = shutil.which("java")
    if not jar or not java or not Path(jar).exists():
        return EpubcheckReport(ok=False, available=False, path=str(epub_path))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.json"
        proc = subprocess.run([java, "-jar", jar, str(epub_path), "--json", str(out)],
                              capture_output=True, text=True, timeout=timeout)
        if not out.exists():
            return EpubcheckReport(ok=False, fatal=[proc.stderr.strip() or proc.stdout.strip() or "EPUBCheck failed"],
                                   path=str(epub_path))
        data = json.loads(out.read_text(encoding="utf-8"))
    report = EpubcheckReport(ok=True, path=str(epub_path))
    for msg in data.get("messages", []):
        sev = msg.get("severity", "")
        locs = msg.get("locations") or [{}]
        where = locs[0].get("path", "")
        line = locs[0].get("line", -1)
        text = f"{msg.get('ID', '')} {where}:{line} {msg.get('message', '')}".strip()
        if sev == "FATAL":
            report.fatal.append(text)
        elif sev == "ERROR":
            report.errors.append(text)
        elif sev == "WARNING":
            report.warnings.append(text)
    report.ok = not report.fatal and not report.errors
    return report
