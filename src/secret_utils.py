"""Secret handling: read tokens from the environment, never log them."""

from __future__ import annotations

import logging
import os


def get_hf_token() -> str | None:
    tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    # An unresolved Runpod secret reference means "not configured".
    if not tok or tok.startswith("{{"):
        return None
    return tok


class RedactFilter(logging.Filter):
    """Replace any secret value that sneaks into a log record."""

    def __init__(self) -> None:
        super().__init__()
        self._secrets = [s for s in (
            get_hf_token(),
            os.environ.get("BUCKET_ACCESS_KEY_ID"),
            os.environ.get("BUCKET_SECRET_ACCESS_KEY"),
        ) if s and len(s) >= 8]

    def filter(self, record: logging.LogRecord) -> bool:
        if self._secrets:
            msg = record.getMessage()
            hit = False
            for s in self._secrets:
                if s in msg:
                    msg = msg.replace(s, "***")
                    hit = True
            if hit:
                record.msg, record.args = msg, ()
        return True


def redact(text: str) -> str:
    for s in RedactFilter()._secrets:
        text = text.replace(s, "***")
    return text
