# src/agents/multi_agent.py
"""Expert-level multi-agent agentic RAG system (speed optimized, no functionality loss)."""
from __future__ import annotations

import json
from typing import Dict, Any, List, Optional

from src.utils.config import load_config, Config
from src.llm.ollama_client import ask_ollama
from src.prompting.prompts import (
    NOT_FOUND_MSG,
    DECOMPOSE_SYSTEM, DECOMPOSE_USER,
    RESEARCHER_SYSTEM, RESEARCHER_USER,
    CRITIC_SYSTEM, CRITIC_USER,
    REFLECTION_SYSTEM, REFLECTION_USER,
)
from src.rag.pipeline import hybrid_retrieve, _web_search_whitelisted_resources, _clean
from src.rag.verifier import filter_pdf_chunks, filter_web_results


def _safe_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads((s or "").strip())
    except Exception:
        return {}


def _pfmt(template: str, **kwargs: Any) -> str:
    out = template or ""
    for k, v in kwargs.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _need_decompose(q: str) -> bool:
    # ✅ Speed heuristic: skip decompose unless question looks complex
    ql = (q or "").lower()
    if len(ql.split()) >= 10:
        return True
    if any(x in ql for x in (" and ", " or ", " vs ", "difference", "compare", "causes", "diagnosis", "treatment", "management")):
        return True
    return False


def _pdf_evidence_from_chunks(chunks: List[dict]) -> tuple[list, str]:
    resources, parts = [], []
    for ch in chunks:
        src = ch.get("source", "unknown.pdf")
        page = ch.get("page", "?")
        txt = (ch.get("text", "") or "").strip()
        if not txt:
            continue
        resources.append({"kind": "pdf", "source": src, "page": page, "snippet": (txt[:500] + "…") if len(txt) > 500 else txt})
        parts.append(f"[{src}:p{page}] {txt}")
    return resources, _clean("\n\n".join(parts))[:12000] or "NO_PDF_EVIDENCE"


def _web_evidence_from_results(results: List[dict]) -> tuple[list, str]:
    resources, parts = [], []
    for r in results:
        snip = (r.get("snippet") or r.get("content") or "").strip()
        if not snip:
            continue
        resources.append({"kind": "web", "title": r.get("title", ""), "url": r.get("url", ""), "snippet": (snip[:600] + "…") if len(snip) > 600 else snip})
        parts.append(snip)
    return resources, _clean("\n\n".join(parts))[:12000] or "NO_WEB_EVIDENCE"


def agentic_answer(
    question: str,
    also_search_web: bool = True,
    provided_pdf_chunks: Optional[List[dict]] = None,
    provided_web_results: Optional[List[dict]] = None,
    cfg: Optional[Config] = None,  # ✅ new: avoid reloading config (no behavior change)
) -> Dict[str, Any]:
    cfg = cfg or load_config()
    q = (question or "").strip()
    if not q:
        return {"question": q, "mode": "agentic", "answer": "", "error": "Empty question."}

    # ✅ Speed: skip decompose when not needed
    if _need_decompose(q):
        raw = ask_ollama(
            system=DECOMPOSE_SYSTEM,
            user=_pfmt(DECOMPOSE_USER, question=q),
            model=cfg.app.model,
            temperature=0.0,
            num_predict=160,  # smaller = faster
            raise_on_fail=False,
        )
        obj = _safe_json(raw)
        subqs = obj.get("subquestions") if isinstance(obj.get("subquestions"), list) else []
        subqs = [str(s).strip() for s in subqs if str(s).strip()][:5] or [q]
    else:
        subqs = [q]

    # Evidence (prefer provided to avoid duplicate retrieval)
    if provided_pdf_chunks is None:
        gathered: List[dict] = []
        for sq in subqs:
            gathered.extend(hybrid_retrieve(sq, cfg) or [])
        pdf_chunks = filter_pdf_chunks(q, gathered, max_keep=cfg.retrieval.top_k)
    else:
        pdf_chunks = filter_pdf_chunks(q, provided_pdf_chunks, max_keep=cfg.retrieval.top_k)

    # ✅ IMPORTANT: fetch-only web resources (NO extra LLM calls), identical resources output
    if provided_web_results is None and also_search_web:
        web = _web_search_whitelisted_resources(q, cfg) or {}
        web_results = web.get("resources") or []
    else:
        web_results = provided_web_results or []
    web_results = filter_web_results(q, web_results, max_keep=getattr(cfg.web, "tavily_top_k", 6))

    if not pdf_chunks and not web_results:
        return {
            "question": q,
            "mode": "agentic",
            "subquestions": subqs,
            "answer": NOT_FOUND_MSG,
            "error": "NO_EVIDENCE",
            "pdf_resources": [],
            "web_resources": [],
            "pdf_evidence": "NO_PDF_EVIDENCE",
            "web_evidence": "NO_WEB_EVIDENCE",
        }

    pdf_resources, pdf_evidence = _pdf_evidence_from_chunks(pdf_chunks)
    web_resources, web_evidence = _web_evidence_from_results(web_results)

    draft = ask_ollama(
        system=RESEARCHER_SYSTEM,
        user=_pfmt(RESEARCHER_USER, question=q, pdf_evidence=pdf_evidence, web_evidence=web_evidence),
        model=cfg.app.model,
        temperature=cfg.app.temperature,
        num_predict=cfg.app.max_new_tokens,
        raise_on_fail=False,
    ).strip()

    if not draft:
        return {
            "question": q,
            "mode": "agentic",
            "subquestions": subqs,
            "answer": "",
            "error": "LLM unavailable.",
            "pdf_resources": pdf_resources,
            "web_resources": web_resources,
            "pdf_evidence": pdf_evidence,
            "web_evidence": web_evidence,
        }

    critic = ask_ollama(
        system=CRITIC_SYSTEM,
        user=_pfmt(CRITIC_USER, question=q, draft=draft, evidence=_clean(pdf_evidence + "\n\n" + web_evidence)[:12000]),
        model=cfg.app.model,
        temperature=0.0,
        num_predict=180,  # smaller = faster
        raise_on_fail=False,
    ).strip() or "(Critic unavailable)"

    improved = ask_ollama(
        system=REFLECTION_SYSTEM,
        user=_pfmt(
            REFLECTION_USER,
            question=q,
            draft=draft,
            critique=critic,
            evidence=_clean(pdf_evidence + "\n\n" + web_evidence)[:12000],
        ),
        model=cfg.app.model,
        temperature=cfg.app.temperature,
        num_predict=cfg.app.max_new_tokens,
        raise_on_fail=False,
    ).strip() or draft

    return {
        "question": q,
        "mode": "agentic",
        "subquestions": subqs,
        "draft_answer": draft,
        "critic_review": critic,
        "answer": improved,
        "pdf_resources": pdf_resources,
        "web_resources": web_resources,
        "pdf_evidence": pdf_evidence,
        "web_evidence": web_evidence,
    }
