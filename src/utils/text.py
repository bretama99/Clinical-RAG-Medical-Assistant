# src/utils/text.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def truncate(text: str, max_len: int, suffix: str = "…") -> str:
    text = text or ""
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + suffix


def list_pdfs(dir_path: str) -> List[str]:
    p = Path(dir_path)
    if not p.exists():
        return []
    return [str(f) for f in sorted(p.glob("*.pdf"))]


def safe_json_parse(s: str) -> Dict[str, Any]:
    try:
        return json.loads((s or "").strip())
    except Exception:
        return {}
