# src/rag/verifier.py
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --- Light lexical verifier (fast, no LLM) -----------------------------------

_STOP = {
    "what", "is", "are", "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "from", "as", "by",
    "how", "why", "when", "where", "which", "who", "does", "do", "did", "can", "could", "should", "would",
    "about", "into", "than", "then", "it", "this", "that", "these", "those", "be", "been", "being", "at", "we", "you"
}

# Tuning knobs (keep stable unless you know why you're changing)
_MIN_TOK_LEN = 3
_PDF_MIN_SCORE = 0.12
_WEB_MIN_SCORE = 0.10
_STRONG_SUPPORT = 0.22
_PARTIAL_SUPPORT = 0.12
_OK_SUPPORT_RATIO = 0.60


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _truncate(s: str, n: int = 260) -> str:
    s = s or ""
    return s if len(s) <= n else (s[:n].rstrip() + "…")


def _tokens(text: str) -> List[str]:
    # Keep it simple & fast: alnum tokens, lowercase, remove stopwords and short tokens.
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if len(t) >= _MIN_TOK_LEN and t not in _STOP]


def _overlap_score(qt: Sequence[str], ct: Sequence[str]) -> Tuple[int, float]:
    # Score = |intersection| / |query_unique|  (recall-like; stable for filtering)
    qs, cs = set(qt), set(ct)
    inter = len(qs & cs)
    return inter, inter / max(1, len(qs))


def _rank_by_overlap(question: str, items: Iterable[dict], get_text, min_score: float, max_keep: int) -> List[dict]:
    qt = _tokens(question)
    if not qt:
        return []
    min_overlap = 2 if len(set(qt)) >= 4 else 1

    scored: List[Tuple[float, dict]] = []
    for it in items or []:
        txt = get_text(it) or ""
        ct = _tokens(txt)
        inter, sc = _overlap_score(qt, ct)
        if inter >= min_overlap and sc >= min_score:
            scored.append((sc, it))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored[: max(1, int(max_keep))]]


def filter_pdf_chunks(question: str, chunks: List[dict], max_keep: int = 8) -> List[dict]:
    return _rank_by_overlap(
        question,
        chunks,
        get_text=lambda ch: ch.get("text", "") or "",
        min_score=_PDF_MIN_SCORE,
        max_keep=max_keep,
    )


def filter_web_results(question: str, results: List[dict], max_keep: int = 6) -> List[dict]:
    return _rank_by_overlap(
        question,
        results,
        get_text=lambda r: (r.get("snippet") or r.get("content") or "") or "",
        min_score=_WEB_MIN_SCORE,
        max_keep=max_keep,
    )


def _split_claims(answer: str, max_claims: int = 10) -> List[str]:
    a = (answer or "").strip()
    if not a:
        return []

    # Prefer bullets/numbered lines if present; otherwise sentence-ish splitting.
    lines = [ln.strip() for ln in a.splitlines() if ln.strip()]
    bulletish = [ln.strip("•*- \t").strip() for ln in lines if re.match(r"^(\d+[\).]|[•\-\*])\s*\S", ln)]
    candidates = bulletish if len(bulletish) >= 2 else re.split(r"(?<=[.!?])\s+", a)

    claims = []
    for c in candidates:
        c = c.strip()
        if len(c) >= 25:
            claims.append(c)
        if len(claims) >= max(1, int(max_claims)):
            break
    return claims


def verify_answer(
    question: str,
    answer: str,
    pdf_chunks: Optional[List[dict]] = None,
    web_results: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """
    Fast lexical verification:
      - split answer into claims
      - for each claim, find best matching evidence snippet (pdf/web) by token overlap
      - label supported/partial/unsupported and compute support ratio
    """
    claims = _split_claims(answer, max_claims=10)

    evidence: List[Tuple[str, str, str]] = []

    for ch in pdf_chunks or []:
        src = ch.get("source", "unknown.pdf")
        page = ch.get("page", "?")
        txt = _clean(ch.get("text", "") or "")
        if txt:
            evidence.append(("pdf", f"{src}:p{page}", txt))

    for r in web_results or []:
        url = (r.get("url") or r.get("source") or "WEB").strip()
        snip = _clean((r.get("snippet") or r.get("content") or "") or "")
        if snip:
            evidence.append(("web", url, snip))

    if not evidence:
        return {"ok": False, "support": 0.0, "reason": "No evidence to verify against.", "best_sources": [], "claims": []}

    # Pre-tokenize evidence once
    ev_tok: List[Tuple[str, str, List[str], str]] = [(k, s, _tokens(t), t) for (k, s, t) in evidence]

    rows: List[Dict[str, Any]] = []
    best_sources: List[Dict[str, Any]] = []
    supported_points = 0.0

    for claim in claims:
        ct = _tokens(claim)
        best_sc, best_kind, best_src, best_raw = 0.0, None, None, None

        cset = set(ct)
        denom = max(1, len(cset))
        for kind, src, tok, raw in ev_tok:
            inter = len(cset & set(tok))
            sc = inter / denom
            if sc > best_sc:
                best_sc, best_kind, best_src, best_raw = sc, kind, src, raw

        if best_sc >= _STRONG_SUPPORT:
            label = "supported"
            supported_points += 1.0
        elif best_sc >= _PARTIAL_SUPPORT:
            label = "partial"
            supported_points += 0.5
        else:
            label = "unsupported"

        rows.append({"claim": claim, "support": label, "best": best_src})

        if best_src and best_sc >= _PARTIAL_SUPPORT:
            best_sources.append(
                {"kind": best_kind, "source": best_src, "snippet": _truncate(best_raw or ""), "score": round(best_sc, 3)}
            )

    support = supported_points / max(1, len(claims)) if claims else 0.0
    ok = support >= _OK_SUPPORT_RATIO

    # Deduplicate sources, keep highest score first
    best_sources.sort(key=lambda x: x["score"], reverse=True)
    seen, uniq = set(), []
    for b in best_sources:
        if b["source"] in seen:
            continue
        seen.add(b["source"])
        uniq.append(b)

    reason = (
        "Answer is evidence-supported."
        if ok
        else "Answer is not sufficiently supported by the provided evidence (likely contains external knowledge)."
    )

    return {
        "ok": bool(ok),
        "support": float(round(support, 3)),
        "reason": reason,
        "best_sources": uniq[:8],
        "claims": rows,
    }
