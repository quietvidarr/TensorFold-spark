"""``TF_CHAT_TEMPLATE=<path to .jinja>``: a chat template that replaces the checkpoint's own at startup.

Everything that reads the template reads the override: the CUDA server's ``ChatTemplate`` (and so its late-system
probe, effort names and message markers), the Mac tokenizer (``families.tokenizer.load_tokenizer``) and the
thinking note. A recipe ships it to carry template fixes (sticky reasoning notes) without touching the weights."""

from __future__ import annotations

import os
from pathlib import Path

ENV = "TF_CHAT_TEMPLATE"


def chat_template_override() -> str | None:
    """The override's source, or None when ``TF_CHAT_TEMPLATE`` is unset or empty; a missing file is an error."""

    path = os.environ.get(ENV, "").strip()
    if not path:
        return None
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"{ENV}={path}: no such file")
    return file.read_text()


def announce(source: str | None) -> None:
    if source is not None:
        print(f"[tensorfold] chat template from {ENV}={os.environ.get(ENV)}", flush=True)
